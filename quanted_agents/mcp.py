"""MCPTool convenience factory for creating MCP server toolsets.

Provides a simplified API for creating pydantic-ai MCP server toolset instances.
Defaults to Streamable HTTP transport (recommended by MCP spec 2025-06-18) and
supports legacy SSE transport for backward compatibility with older MCP servers.
Supports optional transparent retry with async backoff for transient tool call
failures, retrying at the transport level before errors reach the LLM.

Middleware pipeline (when configured) chains stages in fixed order:
intercept -> throttle -> execute (with retry) -> trace.

Example:
    from quanted_agents.mcp import MCPTool

    # Streamable HTTP (default, recommended)
    weather = MCPTool("http://localhost:8001/mcp")

    # Legacy SSE transport
    legacy = MCPTool("http://localhost:8002/sse", transport="sse")

    # With retry and exponential backoff (delays: 1s, 2s, 4s)
    brave = MCPTool(
        "http://localhost:8127/mcp",
        tool_retry_max=3,
        tool_retry_delay=1.0,
        tool_retry_backoff_factor=2.0,
    )

    # With middleware: interceptor + concurrency + tracing
    tools = MCPTool(
        "http://localhost:8001/mcp",
        argument_interceptor=lambda name, args: args,
        max_concurrent_calls=5,
        tool_trace_level="standard",
    )
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP

from quanted_agents.observability import ToolSpan, _truncate_result
from quanted_agents.types import ConcurrencyBackend, InterceptorFn, SemaphoreBackend


async def _call_interceptor(
    interceptor: InterceptorFn,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Call an interceptor function, handling both sync and async callables.

    Uses the same ``asyncio.iscoroutinefunction`` pattern as pipeline.py
    and hierarchical.py for detecting async callables.

    Args:
        interceptor: The interceptor callable (sync or async).
        tool_name: The name of the tool being called.
        args: The tool arguments dict (caller should pass a copy if needed).

    Returns:
        Modified arguments dict, or None to abort the tool call.
    """
    if asyncio.iscoroutinefunction(interceptor):
        return await interceptor(tool_name, args)
    return interceptor(tool_name, args)  # type: ignore[return-value]


async def _retrying_direct_call_tool(
    original_fn: Callable[..., Any],
    name: str,
    args: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    max_retries: int,
    base_delay: float,
    backoff_factor: float,
) -> Any:
    """Retry an MCP tool call with async backoff before propagating errors.

    Calls the original ``direct_call_tool`` method up to ``max_retries + 1``
    times (attempt 0 is the initial call, not a retry). On each failure, waits
    an exponentially increasing delay before the next attempt.

    Args:
        original_fn: The original ``direct_call_tool`` bound method.
        name: The MCP tool name to call.
        args: The tool call arguments dictionary.
        metadata: Optional metadata dictionary for the tool call.
        max_retries: Maximum number of retry attempts after the initial call.
        base_delay: Base delay in seconds between retries.
        backoff_factor: Multiplier applied to delay after each retry.
            1.0 gives constant delay, 2.0 gives exponential (1s, 2s, 4s...).

    Returns:
        The result from a successful ``direct_call_tool`` invocation.

    Raises:
        Exception: The last exception encountered after all retries are
            exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await original_fn(name, args, metadata)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


def _wrap_with_retry(
    server: MCPServerStreamableHTTP | MCPServerSSE,
    max_retries: int,
    base_delay: float,
    backoff_factor: float,
) -> None:
    """Replace ``direct_call_tool`` on server with a retry-wrapped version.

    Saves a reference to the original ``direct_call_tool`` method, then
    replaces it with an async wrapper that delegates to
    ``_retrying_direct_call_tool`` with the configured retry parameters.

    Args:
        server: The pydantic-ai MCP server instance to wrap.
        max_retries: Maximum number of retry attempts per tool call.
        base_delay: Base delay in seconds between retries.
        backoff_factor: Multiplier applied to delay after each retry.
    """
    original_fn = server.direct_call_tool

    @functools.wraps(original_fn)
    async def wrapped(
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await _retrying_direct_call_tool(
            original_fn,
            name,
            args,
            metadata,
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
        )

    server.direct_call_tool = wrapped  # type: ignore[method-assign]


async def _middleware_direct_call_tool(
    original_fn: Callable[..., Any],
    name: str,
    args: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    interceptor: InterceptorFn | None,
    concurrency_backend: ConcurrencyBackend | None,
    trace_collector: list[ToolSpan] | None,
    trace_level: str | None,
    retry_config: dict[str, Any] | None,
) -> Any:
    """Unified middleware wrapper for MCP tool calls.

    Chains stages in fixed order: intercept -> throttle -> execute (with
    optional retry) -> trace. Interceptor runs once before the retry loop.
    Trace is recorded in a finally block to capture both success and error.

    Args:
        original_fn: The original ``direct_call_tool`` bound method.
        name: The MCP tool name to call.
        args: The tool call arguments dictionary.
        metadata: Optional metadata dictionary for the tool call.
        interceptor: Optional interceptor callable for argument modification/abort.
        concurrency_backend: Optional backend for throttling concurrent calls.
        trace_collector: Optional list to append ToolSpan entries to.
        trace_level: Trace verbosity: "minimal", "standard", "verbose", or None.
        retry_config: Optional dict with max_retries, base_delay, backoff_factor.

    Returns:
        The result from a successful tool call.

    Raises:
        ModelRetry: If the interceptor returns None (abort).
        Exception: Any exception from the interceptor (fail-closed) or tool call.
    """
    effective_args = args
    original_args = None
    status = "success"
    result = None
    error_detail = None
    start = time.perf_counter()

    try:
        # Stage 1: Intercept (once, before retry loop)
        if interceptor is not None:
            original_args = dict(args)
            intercepted = await _call_interceptor(interceptor, name, dict(args))
            if intercepted is None:
                status = "aborted"
                raise ModelRetry(f"Tool call '{name}' aborted by interceptor: arguments rejected")
            effective_args = intercepted

        # Stage 2: Throttle + Stage 3: Execute
        async def _execute() -> Any:
            if retry_config is not None:
                return await _retrying_direct_call_tool(
                    original_fn, name, effective_args, metadata,
                    max_retries=retry_config["max_retries"],
                    base_delay=retry_config["base_delay"],
                    backoff_factor=retry_config["backoff_factor"],
                )
            return await original_fn(name, effective_args, metadata)

        if concurrency_backend is not None:
            async with concurrency_backend.acquire(name):
                result = await _execute()
        else:
            result = await _execute()

    except ModelRetry:
        raise
    except Exception as exc:
        status = "error"
        error_detail = str(exc)
        raise
    finally:
        # Stage 4: Trace (always, even on error)
        duration = time.perf_counter() - start
        if trace_collector is not None and trace_level is not None:
            try:
                span = ToolSpan(
                    tool_name=name,
                    status=status,
                    duration_seconds=duration,
                    args=dict(effective_args) if trace_level in ("standard", "verbose") else None,
                    result_preview=_truncate_result(result) if trace_level in ("standard", "verbose") and result is not None else None,
                    error_detail=error_detail if trace_level in ("standard", "verbose") else None,
                    original_args=original_args if trace_level == "verbose" else None,
                    full_result=result if trace_level == "verbose" else None,
                )
                trace_collector.append(span)
            except Exception:
                pass  # Never let trace failures mask tool errors (Pitfall 5)

    return result


def _wrap_with_middleware(
    server: MCPServerStreamableHTTP | MCPServerSSE,
    *,
    interceptor: InterceptorFn | None,
    concurrency_backend: ConcurrencyBackend | None,
    trace_level: str | None,
    retry_config: dict[str, Any] | None,
) -> None:
    """Replace ``direct_call_tool`` with the unified middleware wrapper.

    Creates a closure capturing all middleware configuration and replaces
    the server's ``direct_call_tool`` method. Also sets up trace collection
    storage on the server object.

    Args:
        server: The pydantic-ai MCP server instance to wrap.
        interceptor: Optional interceptor callable.
        concurrency_backend: Optional concurrency backend.
        trace_level: Trace verbosity level or None for no tracing.
        retry_config: Optional retry configuration dict.
    """
    original_fn = server.direct_call_tool
    trace_collector: list[ToolSpan] = []

    server._tool_spans = trace_collector  # type: ignore[attr-defined]
    server._tool_trace_level = trace_level  # type: ignore[attr-defined]

    @functools.wraps(original_fn)
    async def wrapped(
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await _middleware_direct_call_tool(
            original_fn,
            name,
            args,
            metadata,
            interceptor=interceptor,
            concurrency_backend=concurrency_backend,
            trace_collector=trace_collector,
            trace_level=trace_level,
            retry_config=retry_config,
        )

    server.direct_call_tool = wrapped  # type: ignore[method-assign]


def _get_tool_spans(server: MCPServerStreamableHTTP | MCPServerSSE) -> list[ToolSpan]:
    """Retrieve and clear collected tool spans from a server.

    Args:
        server: The MCP server instance with middleware configured.

    Returns:
        A list of ToolSpan entries collected since the last retrieval.
        Returns empty list if no middleware tracing is configured.
    """
    spans: list[ToolSpan] = getattr(server, "_tool_spans", [])
    collected = list(spans)
    spans.clear()
    return collected


def MCPTool(
    url: str,
    *,
    transport: str = "http",
    tool_prefix: str | None = None,
    timeout: float = 5,
    tool_retry_max: int = 0,
    tool_retry_delay: float = 0.0,
    tool_retry_backoff_factor: float = 1.0,
    argument_interceptor: InterceptorFn | None = None,
    max_concurrent_calls: int | None = None,
    concurrency_backend: ConcurrencyBackend | None = None,
    concurrency_timeout: float | None = None,
    tool_trace_level: str | None = None,
    **kwargs: Any,
) -> MCPServerStreamableHTTP | MCPServerSSE:
    """Create an MCP server toolset from a URL.

    Factory function that returns the appropriate pydantic-ai MCPServer instance
    based on the transport parameter. Uses PascalCase naming intentionally as it
    acts as a constructor from the caller's perspective, matching conventions like
    TypeVar, NewType, and namedtuple.

    The returned object is a pydantic-ai AbstractToolset that can be passed
    directly to QuantedAgent's ``toolsets`` parameter or to pydantic-ai's Agent.

    When middleware parameters are set (argument_interceptor, max_concurrent_calls,
    concurrency_backend, or tool_trace_level), a unified middleware pipeline replaces
    direct_call_tool with fixed order: intercept -> throttle -> execute -> trace.
    Retry is absorbed into the middleware pipeline when both are configured.

    Args:
        url: The MCP server endpoint URL (e.g., "http://localhost:8001/mcp").
        transport: Transport protocol to use. "http" for Streamable HTTP
            (recommended, default) or "sse" for legacy SSE transport.
        tool_prefix: Optional prefix for tool names to avoid collisions when
            using multiple MCP servers.
        timeout: Connection initialization timeout in seconds.
        tool_retry_max: Maximum number of silent retries per tool call before
            propagating the error. 0 (default) disables retry, preserving
            current behavior.
        tool_retry_delay: Base delay in seconds between retries.
        tool_retry_backoff_factor: Multiplier applied to the delay after each
            retry attempt. 1.0 (default) gives constant delay, 2.0 gives
            exponential backoff.
        argument_interceptor: Optional per-tool interceptor callable. Receives
            (tool_name, args) and returns modified args or None to abort.
            Supports both sync and async callables.
        max_concurrent_calls: Maximum concurrent tool calls for this server.
            Creates a default SemaphoreBackend. Ignored if concurrency_backend
            is provided.
        concurrency_backend: Custom ConcurrencyBackend implementation. Takes
            priority over max_concurrent_calls.
        concurrency_timeout: Timeout in seconds for the default SemaphoreBackend.
            Fail-open: call proceeds when acquire times out. Only used with
            max_concurrent_calls (ignored if concurrency_backend is provided).
        tool_trace_level: Trace verbosity level: "minimal", "standard", or
            "verbose". None (default) disables tracing.
        **kwargs: Additional keyword arguments passed through to the underlying
            pydantic-ai MCPServer class.

    Returns:
        An MCPServerStreamableHTTP or MCPServerSSE instance ready for use
        as a toolset.

    Raises:
        ValueError: If transport is not "http" or "sse".
    """
    if transport == "http":
        server = MCPServerStreamableHTTP(
            url, tool_prefix=tool_prefix, timeout=timeout, **kwargs
        )
    elif transport == "sse":
        server = MCPServerSSE(
            url, tool_prefix=tool_prefix, timeout=timeout, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown transport {transport!r}. Use 'http' (Streamable HTTP, recommended) or 'sse' (legacy)."
        )

    has_middleware = any([
        argument_interceptor is not None,
        max_concurrent_calls is not None,
        concurrency_backend is not None,
        tool_trace_level is not None,
    ])

    if has_middleware:
        # Resolve concurrency backend
        effective_backend = concurrency_backend
        if effective_backend is None and max_concurrent_calls is not None:
            effective_backend = SemaphoreBackend(max_concurrent_calls, timeout=concurrency_timeout)

        # Build retry config if retry is configured
        retry_config = None
        if tool_retry_max > 0:
            retry_config = {
                "max_retries": tool_retry_max,
                "base_delay": tool_retry_delay,
                "backoff_factor": tool_retry_backoff_factor,
            }

        _wrap_with_middleware(
            server,
            interceptor=argument_interceptor,
            concurrency_backend=effective_backend,
            trace_level=tool_trace_level,
            retry_config=retry_config,
        )
    elif tool_retry_max > 0:
        # Backward compatibility: retry-only wrapping without middleware overhead
        _wrap_with_retry(server, tool_retry_max, tool_retry_delay, tool_retry_backoff_factor)

    return server

"""QuantedAgent: Type-safe agent wrapper enforcing Pydantic BaseModel I/O."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import RunUsage

from quanted_agents._execution import SoftLimitGuard, execute_wrap_up, resolve_timeouts
from quanted_agents._token_counter import count_messages_tokens, count_tokens, truncate_messages
from quanted_agents._usage_limits import build_usage_limits
from quanted_agents.context import ContextManager
from quanted_agents.exceptions import (
    AgentTimeoutError,
    ConfigurationError,
    ContextOverflowError,
    InvalidInputType,
    InvalidOutputType,
    MCPConnectionError,
    RecoveryExhaustedError,
)
from quanted_agents.types import OverflowStrategy, ValidationResult
from quanted_agents.observability import (
    StepTiming,
    TraceEntry,
    extract_model_name,
    extract_tool_calls,
    serialize_messages,
)
from quanted_agents.recovery import RecoveryPipeline
from quanted_agents.restructurer import RestructurerAgent
from quanted_agents.result import QuantedResult
from quanted_agents.trace_writer import TraceWriter, _resolve_trace_writer

logger = logging.getLogger(__name__)

_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_UNSET = object()


class QuantedAgent:
    """Type-safe agent wrapper that enforces Pydantic BaseModel I/O.

    Wraps pydantic-ai's Agent class with construction-time validation of
    input and output types, automatic input serialization to JSON prompts,
    and a rich result object.

    Includes an error recovery pipeline that automatically repairs malformed
    LLM output via json-repair before failing. Optionally supports a two-model
    pattern where a cheap ``restructurer_model`` restructures the heavy model's
    raw output into the target Pydantic schema.

    Supports MCP (Model Context Protocol) integration via the ``toolsets``
    parameter. MCP toolsets are passed through to the inner pydantic-ai Agent,
    enabling dynamic tool discovery from external MCP servers.

    The wrapper is intentionally thin -- pydantic-ai handles LLM communication,
    structured output, tool registration, retries, and provider abstraction.
    QuantedAgent adds: (1) BaseModel I/O enforcement, (2) simplified constructor,
    (3) automatic input serialization, (4) error recovery pipeline,
    (5) two-model restructuring, (6) escape hatch via .inner property,
    (7) MCP toolset integration, (8) on-demand skill/feedback context loading.

    Implements the async context manager protocol (``async with agent:``) for
    pre-initializing MCP connections in multi-run scenarios. For single ``.run()``
    calls, pydantic-ai auto-manages the connection lifecycle.

    Example:
        from pydantic import BaseModel
        from quanted_agents import QuantedAgent
        from quanted_agents.mcp import MCPTool

        class Query(BaseModel):
            question: str

        class Answer(BaseModel):
            response: str
            confidence: float

        agent = QuantedAgent(
            "openai:gpt-4o",
            input_type=Query,
            output_type=Answer,
            system_prompt="Answer questions concisely.",
            toolsets=[MCPTool("http://localhost:8001/mcp")],
        )
        result = await agent.run(Query(question="What is Python?"))
        print(result.data.response)

        # Multi-run with pre-initialized MCP connections:
        async with agent:
            r1 = await agent.run(Query(question="First"))
            r2 = await agent.run(Query(question="Second"))
    """

    def __init__(
        self,
        model: str,
        *,
        input_type: type[BaseModel],
        output_type: type[BaseModel],
        system_prompt: str | list[str] = "",
        instructions: str | None = None,
        tools: list[Any] = [],  # noqa: B006
        toolsets: list[Any] | None = None,
        skills_path: str | Path | None = None,
        feedback_path: str | Path | None = None,
        retries: int = 1,
        deps_type: type[Any] | None = None,
        restructurer_model: str | None = None,
        max_recovery_attempts: int = 3,
        llm_call_limit: int | None = None,
        tool_call_limit: int | None = None,
        total_request_limit: int | None = None,
        soft_limit: bool = False,
        soft_timeout: float | None = None,
        hard_timeout: float | None = None,
        max_context_tokens: int | None = None,
        overflow_strategy: OverflowStrategy | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new QuantedAgent.

        Args:
            model: pydantic-ai model identifier (e.g., "openai:gpt-4o",
                "anthropic:claude-3-5-sonnet").
            input_type: The Pydantic BaseModel subclass for agent input.
                Validated at construction time.
            output_type: The Pydantic BaseModel subclass for agent output.
                Validated at construction time.
            system_prompt: Static system prompt string or list of strings.
            instructions: Per-run instructions that are refreshed on each run
                and not affected by message history context.
            tools: List of tool functions for the agent. Uses pydantic-ai's
                tool registration system.
            toolsets: List of toolset objects (e.g., MCPTool instances) for MCP
                server integration. Passed through to the inner pydantic-ai
                Agent's toolsets parameter. Each toolset is an AbstractToolset
                that provides dynamic tool discovery.
            skills_path: Path to a directory of markdown skill files with YAML
                frontmatter. The agent's LLM will see available skill names in
                its system prompt and can load them on-demand via the internal
                ``_load_context`` tool.
            feedback_path: Path to a directory of markdown feedback files with
                YAML frontmatter. Same loading behavior as skills.
            retries: Number of retries for output validation failures.
            deps_type: Dependency injection type for tools accessing RunContext.
            restructurer_model: If set, enables the two-model pattern where
                a cheap model restructures the heavy model's raw output into
                the target schema (e.g., "openai:gpt-4o-mini").
            max_recovery_attempts: Global retry budget for the recovery pipeline.
                Prevents infinite loops across all recovery stages.
            llm_call_limit: Maximum number of LLM API calls per run. Maps to
                pydantic-ai's UsageLimits.request_limit.
            tool_call_limit: Maximum number of tool invocations per run. Maps
                to pydantic-ai's UsageLimits.tool_calls_limit.
            total_request_limit: Maximum total requests tracked at SDK level.
                No pydantic-ai equivalent; stored for SDK-level tracking.
            soft_limit: If True, usage limit violations trigger a wrap-up
                sequence instead of raising UsageLimitExceeded. The agent
                gets up to 2 additional LLM calls (with tools blocked) to
                produce final output.
            soft_timeout: Seconds before soft wrap-up fires. When only
                soft_timeout is set, an implicit hard backstop is added
                at soft_timeout + 30s. Can be overridden per-run.
            hard_timeout: Seconds before hard kill with AgentTimeoutError.
                Can be overridden per-run.
            max_context_tokens: Maximum estimated input tokens for the
                context window. When set, a pre-flight check runs before
                each agent call to detect overflow. Token counts are
                approximate estimates.
            overflow_strategy: Strategy for handling context overflow.
                When None and max_context_tokens is set, defaults to
                OverflowStrategy.RAISE. Options: RAISE (raise
                ContextOverflowError) or TRUNCATE_OLDEST (drop oldest
                messages while preserving system prompt).
            **kwargs: Additional keyword arguments passed through to the
                underlying pydantic-ai Agent for advanced configuration.
                Accepts api_key (str) for explicit API key configuration.

        Raises:
            InvalidInputType: If input_type is not a Pydantic BaseModel subclass.
            InvalidOutputType: If output_type is not a Pydantic BaseModel subclass.
            FileNotFoundError: If skills_path or feedback_path does not exist.
            ValueError: If a user-defined tool collides with the reserved
                ``_load_context`` tool name, or if hard_timeout <= soft_timeout.
        """
        if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
            raise InvalidInputType(
                f"input_type must be a Pydantic BaseModel subclass, got {input_type}"
            )
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise InvalidOutputType(
                f"output_type must be a Pydantic BaseModel subclass, got {output_type}"
            )

        self.input_type: type[BaseModel] = input_type
        self.output_type: type[BaseModel] = output_type
        self._model: str = model
        self._api_key: str | None = kwargs.pop("api_key", None)
        self._llm_call_limit: int | None = llm_call_limit
        self._tool_call_limit: int | None = tool_call_limit
        self._total_request_limit: int | None = total_request_limit
        self._soft_limit: bool = soft_limit
        self._soft_timeout: float | None = soft_timeout
        self._hard_timeout: float | None = hard_timeout

        # Context overflow configuration
        self._max_context_tokens: int | None = max_context_tokens
        self._overflow_strategy: OverflowStrategy | None = overflow_strategy or (
            OverflowStrategy.RAISE if max_context_tokens is not None else None
        )
        self._provider: str = model.split(":")[0] if isinstance(model, str) and ":" in model else "openai"

        # Validate and compute effective timeout defaults
        self._effective_soft_timeout: float | None
        self._effective_hard_timeout: float | None
        self._effective_soft_timeout, self._effective_hard_timeout = resolve_timeouts(
            soft_timeout, hard_timeout
        )

        # Initialize context manager when skills or feedback paths are set
        self._context_manager: ContextManager | None = None
        if skills_path is not None or feedback_path is not None:
            self._context_manager = ContextManager(
                skills_path=Path(skills_path) if skills_path is not None else None,
                feedback_path=Path(feedback_path) if feedback_path is not None else None,
            )

        # Augment system prompt with context catalog
        if self._context_manager is not None and self._context_manager.has_items:
            catalog = self._context_manager.build_catalog()
            if catalog:
                if isinstance(system_prompt, str):
                    system_prompt = f"{system_prompt}\n\n{catalog}" if system_prompt else catalog
                elif isinstance(system_prompt, list):
                    system_prompt = list(system_prompt) + [catalog]

        # Define and inject _load_context tool when context is configured
        if self._context_manager is not None:
            context_manager = self._context_manager  # capture for closure

            def _load_context(names: list[str]) -> str:
                """Load the full content of one or more skills or feedback by name.

                Call this tool when you need the detailed instructions from a skill
                or feedback item listed in your available context.

                Args:
                    names: List of skill or feedback names to load.

                Returns:
                    The full markdown content of each requested item, or error
                    messages for any names that could not be found.
                """
                return context_manager.load(names)

            # Validate no collision with user-defined tools
            for tool in tools:
                tool_name = tool.__name__ if callable(tool) else getattr(tool, "name", None)
                if tool_name == "_load_context":
                    raise ValueError(
                        "Cannot use tool name '_load_context' -- it is reserved "
                        "for the internal context loading system. Rename your tool "
                        "to avoid this collision."
                    )

            tools = list(tools) + [_load_context]

        agent_kwargs: dict[str, Any] = {
            "output_type": output_type,
            "system_prompt": system_prompt,
            "instructions": instructions,
            "tools": tools,
            "retries": retries,
            "deps_type": deps_type or type(None),
        }
        if toolsets is not None:
            agent_kwargs["toolsets"] = toolsets
        agent_kwargs.update(kwargs)

        self._agent: Agent[Any, Any] = Agent(model, **agent_kwargs)

        self._recovery: RecoveryPipeline = RecoveryPipeline(output_type, max_recovery_attempts)

        if restructurer_model is not None:
            self._restructurer: RestructurerAgent | None = RestructurerAgent(
                restructurer_model, output_type, retries=retries
            )
        else:
            self._restructurer = None

    @property
    def inner(self) -> Agent[Any, Any]:
        """Access the underlying pydantic-ai Agent.

        This is the escape hatch for advanced pydantic-ai usage such as
        dynamic system prompts via @agent.inner.system_prompt, custom
        tool decorators, or direct Agent configuration.

        Returns:
            The wrapped pydantic-ai Agent instance.
        """
        return self._agent

    def add_feedback(self, name: str, content: str, description: str) -> None:
        """Add a feedback file programmatically.

        Creates a new markdown feedback file with YAML frontmatter in the
        agent's feedback directory. The feedback is immediately available
        for loading via the ``_load_context`` tool.

        If a feedback item with the same name already exists, the name is
        auto-suffixed with an incrementing number (e.g., name_1, name_2)
        and a warning is logged.

        Args:
            name: The canonical name for this feedback item.
            content: The markdown body content of the feedback.
            description: A short description shown in the system prompt catalog.

        Raises:
            ValueError: If feedback_path was not configured on this agent.
        """
        if self._context_manager is None:
            raise ValueError(
                "Cannot add feedback: neither skills_path nor feedback_path "
                "was configured on this agent"
            )
        self._context_manager.add_feedback(name, content, description)

    def validate(
        self, *, check_mcp: bool = False, raise_on_error: bool = False
    ) -> ValidationResult:
        """Validate agent configuration synchronously.

        Checks model format, API key availability, and syntactic MCP toolset
        validity. Does NOT make network calls or verify MCP connectivity --
        use avalidate(check_mcp=True) for that.

        Args:
            check_mcp: If True, raises TypeError directing to avalidate().
                Sync validation cannot perform network-based MCP checks.
            raise_on_error: If True, raises ConfigurationError when
                validation fails instead of returning the result.

        Returns:
            A ValidationResult with errors and warnings.

        Raises:
            TypeError: If check_mcp=True (sync cannot check MCP connectivity).
            ConfigurationError: If raise_on_error=True and validation fails.
        """
        if check_mcp:
            raise TypeError(
                "check_mcp=True requires async. Use "
                "await agent.avalidate(check_mcp=True) instead."
            )

        result = ValidationResult()

        # Check model format: "provider:model_name"
        provider = ""
        if ":" not in self._model:
            result.errors.append(
                f"Invalid model format '{self._model}'. Expected "
                f"'provider:model_name' (e.g., 'anthropic:claude-3-5-sonnet')."
            )
        else:
            parts = self._model.split(":", 1)
            provider = parts[0]
            model_name = parts[1]
            if not provider or not model_name:
                result.errors.append(
                    f"Invalid model format '{self._model}'. Expected "
                    f"'provider:model_name' (e.g., 'anthropic:claude-3-5-sonnet')."
                )

        # Check API key availability
        if provider and not self._api_key:
            env_var_name = _PROVIDER_ENV_VARS.get(provider, "")
            if env_var_name:
                env_value = os.getenv(env_var_name, "")
                if not env_value:
                    result.errors.append(
                        f"No API key found for provider '{provider}'. Set "
                        f"{env_var_name} environment variable or pass "
                        f"api_key= to the agent constructor."
                    )
            else:
                result.warnings.append(
                    f"Unknown provider '{provider}'. Cannot validate "
                    f"API key availability."
                )

        # Syntactic MCP validation
        self._validate_mcp_toolsets(result)

        result.valid = len(result.errors) == 0

        if raise_on_error and not result.valid:
            raise ConfigurationError(result.errors)

        return result

    def _validate_mcp_toolsets(self, result: ValidationResult) -> None:
        """Perform syntactic validation on MCP toolsets.

        Checks URL format for HTTP/SSE toolsets and command existence for
        stdio toolsets. Does not make network calls.

        Args:
            result: The ValidationResult to append errors/warnings to.
        """
        for toolset in self._agent.toolsets:
            if isinstance(toolset, (MCPServerStreamableHTTP, MCPServerSSE)):
                url = toolset.url
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    result.errors.append(
                        f"MCP toolset has invalid URL '{url}'. "
                        f"Expected a valid HTTP(S) URL."
                    )
                else:
                    result.warnings.append(
                        f"MCP toolset at '{url}' configured but not "
                        f"connectivity-tested. Use await "
                        f"agent.avalidate(check_mcp=True) to verify."
                    )
            elif isinstance(toolset, MCPServerStdio):
                command = toolset.command
                found = shutil.which(command)
                if found is None and not Path(command).exists():
                    result.errors.append(
                        f"MCP stdio toolset command '{command}' not found "
                        f"on disk. Ensure the executable exists and is on PATH."
                    )
                else:
                    result.warnings.append(
                        f"MCP stdio toolset '{command}' configured but not "
                        f"connectivity-tested. Use await "
                        f"agent.avalidate(check_mcp=True) to verify."
                    )

    async def avalidate(
        self, *, check_mcp: bool = False, raise_on_error: bool = False
    ) -> ValidationResult:
        """Validate agent configuration asynchronously.

        Performs all checks from validate(), plus optional MCP connectivity
        verification when check_mcp=True.

        Args:
            check_mcp: If True, attempts to connect to MCP servers with a
                5-second timeout to verify connectivity.
            raise_on_error: If True, raises ConfigurationError when
                validation fails instead of returning the result.

        Returns:
            A ValidationResult with errors and warnings.

        Raises:
            ConfigurationError: If raise_on_error=True and validation fails.
        """
        result = self.validate(check_mcp=False, raise_on_error=False)

        if check_mcp:
            has_mcp = any(
                isinstance(ts, (MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio))
                for ts in self._agent.toolsets
            )
            if has_mcp:
                try:
                    async with asyncio.timeout(5):
                        async with self._agent:
                            pass
                    # Success: remove MCP connectivity warnings
                    result.warnings = [
                        w for w in result.warnings
                        if "connectivity-tested" not in w
                    ]
                except (TimeoutError, asyncio.TimeoutError):
                    result.errors.append(
                        "MCP connectivity check timed out after 5 seconds."
                    )
                except (ConnectionError, OSError) as exc:
                    result.errors.append(
                        f"MCP connectivity check failed: {exc}"
                    )

        result.valid = len(result.errors) == 0

        if raise_on_error and not result.valid:
            raise ConfigurationError(result.errors)

        return result

    async def __aenter__(self) -> QuantedAgent:
        """Enter async context manager to pre-initialize MCP connections.

        Delegates to the inner pydantic-ai Agent's context manager, which
        initializes all registered MCP toolset connections. This avoids
        per-run connection overhead when running the agent multiple times.

        Returns:
            This QuantedAgent instance.

        Raises:
            MCPConnectionError: If connecting to an MCP server fails.
        """
        try:
            await self._agent.__aenter__()
        except (ConnectionError, OSError) as exc:
            raise MCPConnectionError(
                f"Failed to connect to MCP server: {exc}"
            ) from exc
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager and tear down MCP connections.

        Delegates to the inner pydantic-ai Agent's context manager to
        cleanly close all MCP toolset connections.

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The traceback, if an exception was raised.
        """
        await self._agent.__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        input_data: BaseModel,
        *,
        soft_timeout: Any = _UNSET,
        hard_timeout: Any = _UNSET,
        **kwargs: Any,
    ) -> QuantedResult[Any]:
        """Run the agent with the given input.

        Validates the input type, serializes the BaseModel to JSON, delegates
        to pydantic-ai Agent.run(), and wraps the result in a QuantedResult.

        If the LLM returns malformed output, the recovery pipeline activates:
        1. json-repair attempts to fix the raw JSON
        2. If repair fails and a restructurer_model is configured, the cheap
           model restructures the output into the target schema
        3. A global retry budget prevents infinite recovery loops

        When soft_limit is enabled, UsageLimitExceeded triggers a wrap-up
        sequence instead of propagating. The agent gets up to 2 additional
        LLM calls (with tools blocked) to produce final output.

        Args:
            input_data: A Pydantic BaseModel instance matching the agent's input_type.
            soft_timeout: Per-run override for soft timeout in seconds. When not
                provided, falls back to the constructor default.
            hard_timeout: Per-run override for hard timeout in seconds. When not
                provided, falls back to the constructor default.
            **kwargs: Additional keyword arguments passed to pydantic-ai Agent.run()
                (e.g., message_history, model_settings, usage_limits). Also accepts
                traces_path (str | Path) to write JSONL trace files and _trace_writer
                (TraceWriter) for propagation from parent workflows.

        Returns:
            A QuantedResult wrapping the validated output with access to
            .data, .usage, .messages, and .new_messages.

        Raises:
            InvalidInputType: If input_data is not an instance of the agent's input_type.
            RecoveryExhaustedError: If the recovery budget is exceeded.
            AgentTimeoutError: If hard timeout fires during execution.
            ValueError: If per-run hard_timeout <= soft_timeout.
        """
        if not isinstance(input_data, self.input_type):
            raise InvalidInputType(
                f"Expected {self.input_type.__name__}, got {type(input_data).__name__}"
            )

        # Resolve per-run timeout overrides
        run_soft = self._soft_timeout if soft_timeout is _UNSET else soft_timeout
        run_hard = self._hard_timeout if hard_timeout is _UNSET else hard_timeout
        effective_soft, effective_hard = resolve_timeouts(run_soft, run_hard)

        # Extract session_id BEFORE _resolve_trace_writer pops trace_session from kwargs
        _trace_session = kwargs.get("trace_session")
        session_id: str | None = _trace_session.session_id if _trace_session is not None else None
        trace_writer = _resolve_trace_writer(kwargs)

        usage_limits = build_usage_limits(
            self._llm_call_limit, self._tool_call_limit, self._total_request_limit
        )
        if usage_limits is not None:
            kwargs.setdefault("usage_limits", usage_limits)

        # Pre-flight context overflow check
        overflow_occurred = False
        messages_truncated = 0
        if self._max_context_tokens is not None:
            message_history = kwargs.get("message_history")
            if message_history is not None:
                # pydantic-ai stores system prompts as a tuple in _system_prompts
                prompts = getattr(self._agent, "_system_prompts", ())
                system_prompt_text = " ".join(str(p) for p in prompts) if prompts else ""
                system_prompt_tokens = count_tokens(system_prompt_text, self._provider)
                history_tokens = count_messages_tokens(message_history, self._provider)
                total_tokens = system_prompt_tokens + history_tokens

                if total_tokens > self._max_context_tokens:
                    if self._overflow_strategy == OverflowStrategy.RAISE:
                        raise ContextOverflowError(
                            current_tokens=total_tokens,
                            max_tokens=self._max_context_tokens,
                            store=None,
                            usage=RunUsage(),
                        )
                    elif self._overflow_strategy == OverflowStrategy.TRUNCATE_OLDEST:
                        truncated_msgs, dropped = truncate_messages(
                            message_history,
                            system_prompt_tokens,
                            self._max_context_tokens,
                            self._provider,
                        )
                        kwargs["message_history"] = truncated_msgs
                        overflow_occurred = True
                        messages_truncated = dropped

        prompt = input_data.model_dump_json()
        step_name = f"QuantedAgent({self.output_type.__name__})"
        start = time.perf_counter()
        guard = SoftLimitGuard()

        try:
            async with asyncio.timeout(effective_hard):
                try:
                    async with asyncio.timeout(effective_soft):
                        with capture_run_messages() as messages:
                            try:
                                result = await self._agent.run(prompt, **kwargs)

                                # SDK-level total_request_limit check
                                if self._total_request_limit is not None:
                                    run_usage = result.usage()
                                    total_requests = (run_usage.requests or 0) + (run_usage.tool_calls or 0)
                                    if total_requests >= self._total_request_limit:
                                        if self._soft_limit:
                                            guard.activate("soft_limit")
                                        else:
                                            raise UsageLimitExceeded(
                                                f"Total request limit of "
                                                f"{self._total_request_limit} exceeded "
                                                f"(used {total_requests})"
                                            )

                                if not guard.is_active:
                                    qr = await self._build_quanted_result(
                                        result, input_data, step_name, start,
                                        messages, trace_writer, session_id,
                                    )
                                    qr._context_overflow_occurred = overflow_occurred
                                    qr._messages_truncated = messages_truncated
                                    return qr

                            except UsageLimitExceeded:
                                if not self._soft_limit:
                                    raise
                                guard.activate("soft_limit")

                            except UnexpectedModelBehavior:
                                raw_text = self._extract_raw_text(messages)
                                if not raw_text:
                                    raise
                                return await self._attempt_recovery(
                                    raw_text, start, list(messages), input_data,
                                    step_name, trace_writer, session_id,
                                )

                except TimeoutError:
                    if not guard.is_active:
                        guard.activate("soft_timeout")

                # Wrap-up sequence (soft limit or soft timeout)
                if guard.is_active:
                    wrap_result = await execute_wrap_up(
                        self._agent,
                        list(messages),
                        RunUsage(),
                        guard.reason or "",
                        model_settings=kwargs.get("model_settings"),
                    )
                    if wrap_result is not None:
                        duration = time.perf_counter() - start
                        timing = StepTiming(
                            step_name=step_name,
                            duration_seconds=duration,
                            usage=wrap_result.usage,
                        )
                        wrap_result._trace_entries = [TraceEntry(
                            step_name=step_name,
                            input_data=input_data.model_dump(),
                            output_data=wrap_result.data.model_dump()
                            if hasattr(wrap_result.data, "model_dump") else {},
                            messages=serialize_messages(list(messages)),
                            tool_calls=extract_tool_calls(list(messages)),
                            timing=timing,
                            model_name=extract_model_name(list(messages)),
                            recovery_info=None,
                        )]
                        wrap_result._step_timings = [timing]
                        if trace_writer is not None:
                            await trace_writer.write(wrap_result._trace_entries[0])
                        return wrap_result

                    # No usable output from wrap-up
                    usage_so_far = self._extract_usage_from_messages(list(messages))
                    soft_result = QuantedResult.from_data(
                        self.output_type.model_construct(), usage=usage_so_far
                    )
                    soft_result._termination_reason = guard.reason
                    return soft_result

        except TimeoutError:
            usage_so_far = self._extract_usage_from_messages(list(messages))
            raise AgentTimeoutError(
                f"Agent execution exceeded hard timeout of {effective_hard}s",
                store=None,
                usage=usage_so_far,
                termination_reason="hard_timeout",
            )

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Unexpected execution path in QuantedAgent.run()")  # pragma: no cover

    async def _build_quanted_result(
        self,
        result: Any,
        input_data: BaseModel,
        step_name: str,
        start: float,
        messages: list[Any],
        trace_writer: TraceWriter | None,
        session_id: str | None = None,
    ) -> QuantedResult[Any]:
        """Build a QuantedResult from a successful pydantic-ai run result.

        Extracts timing, trace, and observability data and attaches it to
        the QuantedResult. This centralizes result-building logic used by
        both the normal path and soft-limit paths.

        Args:
            result: The pydantic-ai AgentRunResult from a successful run.
            input_data: The original input BaseModel for trace recording.
            step_name: The step name for trace identification.
            start: The perf_counter timestamp from before execution started.
            messages: The captured pydantic-ai messages.
            trace_writer: Optional TraceWriter for writing trace entries.
            session_id: Optional session ID for trace correlation.

        Returns:
            A fully populated QuantedResult.
        """
        duration = time.perf_counter() - start
        timing = StepTiming(
            step_name=step_name,
            duration_seconds=duration,
            usage=result.usage(),
        )
        trace_entry = TraceEntry(
            step_name=step_name,
            input_data=input_data.model_dump(),
            output_data=result.output.model_dump(),
            messages=serialize_messages(list(messages)),
            tool_calls=extract_tool_calls(list(messages)),
            timing=timing,
            model_name=extract_model_name(list(messages)),
            recovery_info=None,
            session_id=session_id,
        )

        quanted_result: QuantedResult[Any] = QuantedResult(result)
        quanted_result._trace_entries = [trace_entry]
        quanted_result._step_timings = [timing]

        if trace_writer is not None:
            await trace_writer.write(trace_entry)

        return quanted_result

    async def _attempt_recovery(
        self,
        raw_text: str,
        start_time: float,
        messages: list[Any],
        input_data: BaseModel,
        step_name: str,
        trace_writer: TraceWriter | None = None,
        session_id: str | None = None,
    ) -> QuantedResult[Any]:
        """Attempt to recover from malformed LLM output.

        Runs the recovery pipeline: json-repair first, then restructurer
        if configured. Consumes recovery budget on each attempt. Attaches
        trace data with recovery info to the resulting QuantedResult.

        Args:
            raw_text: The raw (potentially malformed) text from the LLM.
            start_time: The perf_counter timestamp from before execution started.
            messages: The captured pydantic-ai messages from the failed run.
            input_data: The original input BaseModel for trace recording.
            step_name: The step name for trace identification.
            trace_writer: Optional TraceWriter for writing trace entries to
                a JSONL file. When provided, recovered trace entries are
                written to the file before returning.
            session_id: Optional session ID for trace correlation.

        Returns:
            A QuantedResult wrapping the recovered output with trace data.

        Raises:
            RecoveryExhaustedError: If the recovery budget is exceeded.
            UnexpectedModelBehavior: If recovery fails and no restructurer
                is configured.
        """
        recovered_usage = self._extract_usage_from_messages(messages)
        repaired = self._recovery.attempt_repair(raw_text)
        if repaired is not None:
            logger.info("Recovery: json-repair succeeded")
            result = QuantedResult.from_data(repaired, usage=recovered_usage)
            result._was_recovered = True
            result._recovery_method = "json_repair"
            duration = time.perf_counter() - start_time
            timing = StepTiming(
                step_name=step_name,
                duration_seconds=duration,
                usage=result.usage,
            )
            attempts_used = self._recovery.budget.max_attempts - self._recovery.budget.remaining
            trace_entry = TraceEntry(
                step_name=step_name,
                input_data=input_data.model_dump(),
                output_data=repaired.model_dump(),
                messages=serialize_messages(messages),
                tool_calls=extract_tool_calls(messages),
                timing=timing,
                model_name=extract_model_name(messages),
                recovery_info={
                    "json_repair_attempted": True,
                    "restructurer_used": False,
                    "attempts_used": attempts_used,
                },
                session_id=session_id,
            )
            result._trace_entries = [trace_entry]
            result._step_timings = [timing]

            if trace_writer is not None:
                await trace_writer.write(trace_entry)

            return result

        if self._restructurer is not None:
            logger.info("Recovery: delegating to restructurer model")
            self._recovery.budget.consume()
            restructured = await self._restructurer.restructure(raw_text)
            result = QuantedResult.from_data(restructured, usage=recovered_usage)
            result._was_recovered = True
            result._recovery_method = "restructurer"
            duration = time.perf_counter() - start_time
            timing = StepTiming(
                step_name=step_name,
                duration_seconds=duration,
                usage=result.usage,
            )
            attempts_used = self._recovery.budget.max_attempts - self._recovery.budget.remaining
            trace_entry = TraceEntry(
                step_name=step_name,
                input_data=input_data.model_dump(),
                output_data=restructured.model_dump(),
                messages=serialize_messages(messages),
                tool_calls=extract_tool_calls(messages),
                timing=timing,
                model_name=extract_model_name(messages),
                recovery_info={
                    "json_repair_attempted": True,
                    "restructurer_used": True,
                    "attempts_used": attempts_used,
                },
                session_id=session_id,
            )
            result._trace_entries = [trace_entry]
            result._step_timings = [timing]

            if trace_writer is not None:
                await trace_writer.write(trace_entry)

            return result

        raise RecoveryExhaustedError(
            f"json-repair failed and no restructurer configured "
            f"(budget remaining: {self._recovery.budget.remaining})"
        )

    @staticmethod
    def _extract_raw_text(messages: list[Any]) -> str:
        """Extract raw text from captured pydantic-ai messages.

        Iterates through captured messages in reverse to find the last
        ModelResponse and extracts text content. Checks TextPart first
        (used by OpenAI and Gemini for structured output), then falls
        back to ToolCallPart (used by Anthropic, which returns structured
        output via tool calls with args as str or dict).

        Args:
            messages: List of captured ModelMessage objects from
                pydantic-ai's capture_run_messages() context manager.

        Returns:
            The extracted raw text, or empty string if no text found.
        """
        for message in reversed(messages):
            if isinstance(message, ModelResponse):
                # Primary path: TextPart (OpenAI, Gemini)
                text_parts = [
                    part.content for part in message.parts if isinstance(part, TextPart)
                ]
                if text_parts:
                    return " ".join(text_parts)

                # Fallback path: ToolCallPart (Anthropic structured output)
                for part in message.parts:
                    if isinstance(part, ToolCallPart) and part.args is not None:
                        return part.args_as_json_str()
        return ""

    @staticmethod
    def _extract_usage_from_messages(messages: list[Any]) -> RunUsage:
        """Sum token usage from captured ModelResponse messages.

        Iterates through all captured messages and aggregates the
        RequestUsage from each ModelResponse into a single RunUsage.
        This preserves token counts from the original LLM call that
        triggered recovery, preventing silent data loss.

        Args:
            messages: List of captured ModelMessage objects from
                pydantic-ai's capture_run_messages() context manager.

        Returns:
            A RunUsage with aggregated token counts and request count.
        """
        usage = RunUsage()
        for msg in messages:
            if isinstance(msg, ModelResponse):
                usage.requests += 1
                usage.incr(msg.usage)
        return usage

    async def run_stream(
        self, input_data: BaseModel, **kwargs: Any
    ) -> AsyncGenerator[Any]:
        """Stream the agent's output as partial results.

        An async generator that validates the input type, serializes the
        BaseModel to JSON, and yields partial output objects as the LLM
        generates them.

        When the stream completes but pydantic-ai raises UnexpectedModelBehavior
        (malformed output), the recovery pipeline activates. If recovery succeeds,
        the final yielded item is a QuantedResult with was_recovered=True. Callers
        that want recovery support can check isinstance(last_item, QuantedResult)
        and inspect was_recovered/recovery_method.

        Args:
            input_data: A Pydantic BaseModel instance matching the agent's input_type.
            **kwargs: Additional keyword arguments passed to pydantic-ai
                Agent.run_stream() (e.g., message_history, model_settings).
                Also accepts traces_path, trace_filename, trace_session for
                trace file configuration.

        Yields:
            Partial output objects as the LLM streams its response. On recovery,
            the final yielded item is a QuantedResult instance.

        Raises:
            InvalidInputType: If input_data is not an instance of the agent's input_type.
        """
        if not isinstance(input_data, self.input_type):
            raise InvalidInputType(
                f"Expected {self.input_type.__name__}, got {type(input_data).__name__}"
            )

        # Extract session_id BEFORE _resolve_trace_writer pops trace_session
        _trace_session = kwargs.get("trace_session")
        session_id: str | None = _trace_session.session_id if _trace_session is not None else None
        trace_writer = _resolve_trace_writer(kwargs)

        usage_limits = build_usage_limits(
            self._llm_call_limit, self._tool_call_limit, self._total_request_limit
        )
        if usage_limits is not None:
            kwargs.setdefault("usage_limits", usage_limits)

        prompt = input_data.model_dump_json()
        step_name = f"QuantedAgent({self.output_type.__name__})"
        start = time.perf_counter()

        with capture_run_messages() as messages:
            try:
                async with self._agent.run_stream(prompt, **kwargs) as stream:
                    async for partial in stream.stream_output():
                        yield partial
                # Stream completed successfully -- no recovery needed
            except UnexpectedModelBehavior:
                raw_text = self._extract_raw_text(messages)
                if not raw_text:
                    raise
                result = await self._attempt_recovery(
                    raw_text, start, list(messages), input_data, step_name,
                    trace_writer, session_id,
                )
                # _attempt_recovery already sets _was_recovered and _recovery_method
                yield result

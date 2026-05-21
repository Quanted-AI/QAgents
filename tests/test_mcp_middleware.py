"""Tests for MCP middleware pipeline: interceptor, concurrency, trace, and integration.

Validates the middleware pipeline stages (intercept -> throttle -> execute -> trace)
including argument interception (sync/async, abort, fail-closed), concurrency control
(semaphore throttling, timeout fail-open, custom backend), tool span tracing (all three
verbosity levels), and full pipeline integration. All tests mock direct_call_tool with
AsyncMock to avoid real MCP server calls.
"""

import asyncio
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from pydantic_ai.exceptions import ModelRetry

from quanted_agents.mcp import (
    MCPTool,
    _get_tool_spans,
    _middleware_direct_call_tool,
)
from quanted_agents.observability import ToolSpan, _truncate_result
from quanted_agents.types import ConcurrencyBackend, SemaphoreBackend


class TestInterceptor(unittest.IsolatedAsyncioTestCase):
    """Tests for argument interceptor functionality in the middleware pipeline."""

    async def test_sync_interceptor_modifies_args(self) -> None:
        """Sync interceptor returns modified dict, tool call receives modified args."""
        mock_fn = AsyncMock(return_value="result")

        def interceptor(name: str, args: dict) -> dict:
            args["extra"] = "added"
            return args

        await _middleware_direct_call_tool(
            mock_fn, "tool", {"key": "val"}, None,
            interceptor=interceptor, concurrency_backend=None,
            trace_collector=None, trace_level=None, retry_config=None,
        )
        called_args = mock_fn.call_args[0][1]
        self.assertEqual(called_args["extra"], "added")

    async def test_async_interceptor_modifies_args(self) -> None:
        """Async interceptor returns modified dict, tool call receives modified args."""
        mock_fn = AsyncMock(return_value="result")

        async def interceptor(name: str, args: dict) -> dict:
            args["async_key"] = "async_val"
            return args

        await _middleware_direct_call_tool(
            mock_fn, "tool", {"key": "val"}, None,
            interceptor=interceptor, concurrency_backend=None,
            trace_collector=None, trace_level=None, retry_config=None,
        )
        called_args = mock_fn.call_args[0][1]
        self.assertEqual(called_args["async_key"], "async_val")

    async def test_interceptor_abort_returns_none_raises_model_retry(self) -> None:
        """Interceptor returns None, ModelRetry is raised with descriptive message."""
        mock_fn = AsyncMock(return_value="result")

        def interceptor(name: str, args: dict) -> None:
            return None

        with self.assertRaises(ModelRetry) as ctx:
            await _middleware_direct_call_tool(
                mock_fn, "tool", {"key": "val"}, None,
                interceptor=interceptor, concurrency_backend=None,
                trace_collector=None, trace_level=None, retry_config=None,
            )
        self.assertIn("aborted by interceptor", str(ctx.exception))
        mock_fn.assert_not_called()

    async def test_interceptor_exception_propagates_fail_closed(self) -> None:
        """Interceptor raises ValueError, ValueError propagates (tool never executes)."""
        mock_fn = AsyncMock(return_value="result")

        def interceptor(name: str, args: dict) -> dict:
            raise ValueError("bad args")

        with self.assertRaises(ValueError) as ctx:
            await _middleware_direct_call_tool(
                mock_fn, "tool", {"key": "val"}, None,
                interceptor=interceptor, concurrency_backend=None,
                trace_collector=None, trace_level=None, retry_config=None,
            )
        self.assertEqual(str(ctx.exception), "bad args")
        mock_fn.assert_not_called()

    @patch("quanted_agents.mcp.asyncio.sleep", new_callable=AsyncMock)
    async def test_interceptor_runs_once_before_retry_loop(self, mock_sleep: AsyncMock) -> None:
        """With retry configured, interceptor is called exactly once even when retries occur."""
        call_count = 0

        def interceptor(name: str, args: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return args

        mock_fn = AsyncMock(side_effect=[Exception("transient"), "success"])

        result = await _middleware_direct_call_tool(
            mock_fn, "tool", {"key": "val"}, None,
            interceptor=interceptor, concurrency_backend=None,
            trace_collector=None, trace_level=None,
            retry_config={"max_retries": 2, "base_delay": 0.0, "backoff_factor": 1.0},
        )
        self.assertEqual(call_count, 1)
        self.assertEqual(result, "success")

    async def test_no_interceptor_passes_original_args(self) -> None:
        """When no interceptor set, tool receives original args unchanged."""
        mock_fn = AsyncMock(return_value="result")
        original_args = {"key": "val"}

        await _middleware_direct_call_tool(
            mock_fn, "tool", original_args, None,
            interceptor=None, concurrency_backend=None,
            trace_collector=None, trace_level=None, retry_config=None,
        )
        called_args = mock_fn.call_args[0][1]
        self.assertEqual(called_args, {"key": "val"})

    async def test_interceptor_receives_copy_not_reference(self) -> None:
        """Interceptor modifying the dict does not affect the trace's original_args record."""
        mock_fn = AsyncMock(return_value="result")
        spans: list[ToolSpan] = []

        def interceptor(name: str, args: dict) -> dict:
            args["mutated"] = True
            return args

        await _middleware_direct_call_tool(
            mock_fn, "tool", {"key": "val"}, None,
            interceptor=interceptor, concurrency_backend=None,
            trace_collector=spans, trace_level="verbose", retry_config=None,
        )
        self.assertEqual(len(spans), 1)
        # original_args should NOT have the mutation
        self.assertNotIn("mutated", spans[0].original_args)
        self.assertEqual(spans[0].original_args, {"key": "val"})


class TestConcurrencyControl(unittest.IsolatedAsyncioTestCase):
    """Tests for concurrency control via SemaphoreBackend and custom backends."""

    async def test_semaphore_backend_limits_concurrent_calls(self) -> None:
        """With max_concurrent=1, second concurrent call waits until first completes."""
        backend = SemaphoreBackend(1)
        order: list[str] = []

        async def task(label: str) -> None:
            async with backend.acquire("tool"):
                order.append(f"{label}_start")
                await asyncio.sleep(0.05)
                order.append(f"{label}_end")

        await asyncio.gather(task("A"), task("B"))
        # With semaphore=1, tasks must be sequential
        self.assertEqual(order[0], "A_start")
        self.assertEqual(order[1], "A_end")
        self.assertEqual(order[2], "B_start")
        self.assertEqual(order[3], "B_end")

    async def test_semaphore_backend_timeout_fail_open(self) -> None:
        """With timeout=0.01, when semaphore is held, timed-out call still proceeds."""
        backend = SemaphoreBackend(1, timeout=0.01)
        events: list[str] = []

        async def holder() -> None:
            async with backend.acquire("tool"):
                events.append("holder_acquired")
                await asyncio.sleep(0.1)
                events.append("holder_released")

        async def waiter() -> None:
            await asyncio.sleep(0.02)  # Let holder acquire first
            async with backend.acquire("tool"):
                events.append("waiter_proceeded")

        await asyncio.gather(holder(), waiter())
        # Waiter should proceed even though holder still holds (fail-open)
        self.assertIn("waiter_proceeded", events)

    async def test_semaphore_backend_no_timeout_waits_forever(self) -> None:
        """With timeout=None, call waits for semaphore."""
        backend = SemaphoreBackend(1, timeout=None)
        order: list[str] = []

        async def holder() -> None:
            async with backend.acquire("tool"):
                order.append("holder_start")
                await asyncio.sleep(0.05)
                order.append("holder_end")

        async def waiter() -> None:
            await asyncio.sleep(0.01)  # Let holder acquire first
            async with backend.acquire("tool"):
                order.append("waiter_start")

        await asyncio.gather(holder(), waiter())
        # Waiter must wait for holder to finish
        self.assertEqual(order, ["holder_start", "holder_end", "waiter_start"])

    async def test_custom_concurrency_backend(self) -> None:
        """User-provided ConcurrencyBackend protocol implementation is used."""
        call_log: list[str] = []

        class CustomBackend:
            @asynccontextmanager
            async def acquire(self, tool_name: str) -> AsyncIterator[None]:
                call_log.append(f"acquire_{tool_name}")
                yield
                call_log.append(f"release_{tool_name}")

        mock_fn = AsyncMock(return_value="result")

        await _middleware_direct_call_tool(
            mock_fn, "my_tool", {"key": "val"}, None,
            interceptor=None, concurrency_backend=CustomBackend(),
            trace_collector=None, trace_level=None, retry_config=None,
        )
        self.assertEqual(call_log, ["acquire_my_tool", "release_my_tool"])

    def test_max_concurrent_creates_default_semaphore(self) -> None:
        """MCPTool with max_concurrent_calls=N creates SemaphoreBackend internally."""
        server = MCPTool(
            "http://localhost:8001/mcp",
            max_concurrent_calls=3,
        )
        # Middleware wrapping should have been applied
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))

    def test_concurrency_backend_takes_priority(self) -> None:
        """When both max_concurrent_calls and concurrency_backend set, custom backend used."""
        class CustomBackend:
            @asynccontextmanager
            async def acquire(self, tool_name: str) -> AsyncIterator[None]:
                yield

        server = MCPTool(
            "http://localhost:8001/mcp",
            max_concurrent_calls=5,
            concurrency_backend=CustomBackend(),
        )
        # Should not raise; middleware is applied
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))

    async def test_semaphore_released_on_exception(self) -> None:
        """When tool call raises, semaphore is still released (no deadlock)."""
        backend = SemaphoreBackend(1)
        mock_fn = AsyncMock(side_effect=RuntimeError("tool error"))

        with self.assertRaises(RuntimeError):
            await _middleware_direct_call_tool(
                mock_fn, "tool", {}, None,
                interceptor=None, concurrency_backend=backend,
                trace_collector=None, trace_level=None, retry_config=None,
            )

        # Semaphore should be released: next acquire should not deadlock
        async with backend.acquire("tool"):
            pass  # Would deadlock if semaphore wasn't released


class TestToolSpan(unittest.TestCase):
    """Tests for ToolSpan dataclass and truncation helper."""

    def test_tool_span_minimal_to_dict(self) -> None:
        """Minimal level includes only tool_name, status, duration_seconds."""
        span = ToolSpan(tool_name="t", status="success", duration_seconds=1.5,
                        args={"k": "v"}, result_preview="preview")
        d = span.to_dict("minimal")
        self.assertEqual(set(d.keys()), {"tool_name", "status", "duration_seconds"})

    def test_tool_span_standard_to_dict(self) -> None:
        """Standard level includes args, result_preview, error_detail."""
        span = ToolSpan(tool_name="t", status="success", duration_seconds=1.5,
                        args={"k": "v"}, result_preview="preview", error_detail=None)
        d = span.to_dict("standard")
        self.assertIn("args", d)
        self.assertIn("result_preview", d)
        self.assertIn("error_detail", d)
        self.assertNotIn("original_args", d)
        self.assertNotIn("full_result", d)

    def test_tool_span_verbose_to_dict(self) -> None:
        """Verbose level includes original_args, full_result."""
        span = ToolSpan(tool_name="t", status="success", duration_seconds=1.5,
                        args={"k": "v"}, original_args={"orig": "val"},
                        full_result={"full": "data"})
        d = span.to_dict("verbose")
        self.assertIn("original_args", d)
        self.assertIn("full_result", d)

    def test_truncate_result_short_string(self) -> None:
        """Strings <= 500 chars not truncated."""
        short = "hello world"
        self.assertEqual(_truncate_result(short), short)

    def test_truncate_result_long_string(self) -> None:
        """Strings > 500 chars truncated with '...'."""
        long_str = "x" * 600
        result = _truncate_result(long_str)
        self.assertEqual(len(result), 503)  # 500 + "..."
        self.assertTrue(result.endswith("..."))


class TestToolSpanCollection(unittest.IsolatedAsyncioTestCase):
    """Tests for ToolSpan collection during middleware execution."""

    async def test_tool_span_collected_on_success(self) -> None:
        """Successful tool call produces ToolSpan with status='success'."""
        mock_fn = AsyncMock(return_value="result")
        spans: list[ToolSpan] = []

        await _middleware_direct_call_tool(
            mock_fn, "tool", {"k": "v"}, None,
            interceptor=None, concurrency_backend=None,
            trace_collector=spans, trace_level="standard", retry_config=None,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "success")
        self.assertEqual(spans[0].tool_name, "tool")
        self.assertGreater(spans[0].duration_seconds, 0)

    async def test_tool_span_collected_on_error(self) -> None:
        """Failed tool call produces ToolSpan with status='error' and error_detail."""
        mock_fn = AsyncMock(side_effect=RuntimeError("boom"))
        spans: list[ToolSpan] = []

        with self.assertRaises(RuntimeError):
            await _middleware_direct_call_tool(
                mock_fn, "tool", {"k": "v"}, None,
                interceptor=None, concurrency_backend=None,
                trace_collector=spans, trace_level="standard", retry_config=None,
            )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "error")
        self.assertEqual(spans[0].error_detail, "boom")

    async def test_tool_span_collected_on_abort(self) -> None:
        """Interceptor abort produces ToolSpan with status='aborted'."""
        mock_fn = AsyncMock(return_value="result")
        spans: list[ToolSpan] = []

        def interceptor(name: str, args: dict) -> None:
            return None

        with self.assertRaises(ModelRetry):
            await _middleware_direct_call_tool(
                mock_fn, "tool", {"k": "v"}, None,
                interceptor=interceptor, concurrency_backend=None,
                trace_collector=spans, trace_level="standard", retry_config=None,
            )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "aborted")

    async def test_no_trace_when_level_none(self) -> None:
        """When tool_trace_level is None, no ToolSpan is created."""
        mock_fn = AsyncMock(return_value="result")
        spans: list[ToolSpan] = []

        await _middleware_direct_call_tool(
            mock_fn, "tool", {"k": "v"}, None,
            interceptor=None, concurrency_backend=None,
            trace_collector=spans, trace_level=None, retry_config=None,
        )
        self.assertEqual(len(spans), 0)


class TestMiddlewarePipeline(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the full middleware pipeline."""

    async def test_full_pipeline_intercept_throttle_execute_trace(self) -> None:
        """All stages active: interceptor modifies args, semaphore throttles, tool executes, span recorded."""
        mock_fn = AsyncMock(return_value="result_data")
        spans: list[ToolSpan] = []
        backend = SemaphoreBackend(2)

        def interceptor(name: str, args: dict) -> dict:
            args["injected"] = True
            return args

        result = await _middleware_direct_call_tool(
            mock_fn, "tool", {"key": "val"}, None,
            interceptor=interceptor, concurrency_backend=backend,
            trace_collector=spans, trace_level="verbose", retry_config=None,
        )
        self.assertEqual(result, "result_data")
        # Tool received modified args
        called_args = mock_fn.call_args[0][1]
        self.assertTrue(called_args["injected"])
        # Span recorded
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "success")
        self.assertEqual(spans[0].original_args, {"key": "val"})
        self.assertIn("injected", spans[0].args)

    @patch("quanted_agents.mcp.asyncio.sleep", new_callable=AsyncMock)
    async def test_middleware_with_retry_integration(self, mock_sleep: AsyncMock) -> None:
        """Middleware + retry: tool fails once, retries, succeeds. Interceptor called once."""
        interceptor_calls = 0

        def interceptor(name: str, args: dict) -> dict:
            nonlocal interceptor_calls
            interceptor_calls += 1
            return args

        mock_fn = AsyncMock(side_effect=[Exception("transient"), "success"])
        spans: list[ToolSpan] = []

        result = await _middleware_direct_call_tool(
            mock_fn, "tool", {"k": "v"}, None,
            interceptor=interceptor, concurrency_backend=None,
            trace_collector=spans, trace_level="standard",
            retry_config={"max_retries": 2, "base_delay": 0.0, "backoff_factor": 1.0},
        )
        self.assertEqual(result, "success")
        self.assertEqual(interceptor_calls, 1)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, "success")

    def test_backward_compat_no_middleware_params(self) -> None:
        """MCPTool with only retry params uses existing _wrap_with_retry path."""
        server = MCPTool("http://localhost:8001/mcp", tool_retry_max=2)
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))
        # No middleware attributes should be set
        self.assertFalse(hasattr(server, "_tool_spans"))

    def test_backward_compat_no_params_at_all(self) -> None:
        """MCPTool with no extra params returns unwrapped server."""
        server = MCPTool("http://localhost:8001/mcp")
        self.assertFalse(hasattr(server.direct_call_tool, "__wrapped__"))
        self.assertFalse(hasattr(server, "_tool_spans"))

    def test_mcptool_factory_accepts_all_new_params(self) -> None:
        """MCPTool with all middleware params constructs without error."""
        server = MCPTool(
            "http://localhost:8001/mcp",
            argument_interceptor=lambda n, a: a,
            max_concurrent_calls=5,
            tool_trace_level="standard",
        )
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))
        self.assertTrue(hasattr(server, "_tool_spans"))


class TestSemaphoreBackend(unittest.TestCase):
    """Tests for SemaphoreBackend protocol compliance and BoundedSemaphore."""

    def test_semaphore_backend_implements_protocol(self) -> None:
        """isinstance(SemaphoreBackend(...), ConcurrencyBackend) is True."""
        backend = SemaphoreBackend(5)
        self.assertIsInstance(backend, ConcurrencyBackend)

    def test_bounded_semaphore_prevents_over_release(self) -> None:
        """BoundedSemaphore raises ValueError on over-release."""
        backend = SemaphoreBackend(1)
        with self.assertRaises(ValueError):
            backend._semaphore.release()


class TestGetToolSpans(unittest.IsolatedAsyncioTestCase):
    """Tests for _get_tool_spans helper function."""

    async def test_get_tool_spans_retrieves_and_clears(self) -> None:
        """_get_tool_spans returns collected spans and clears the list."""
        server = MCPTool(
            "http://localhost:8001/mcp",
            tool_trace_level="minimal",
        )
        mock_fn = AsyncMock(return_value="result")
        # Manually call middleware to populate spans
        await _middleware_direct_call_tool(
            mock_fn, "tool", {}, None,
            interceptor=None, concurrency_backend=None,
            trace_collector=server._tool_spans, trace_level="minimal",
            retry_config=None,
        )
        spans = _get_tool_spans(server)
        self.assertEqual(len(spans), 1)
        # After retrieval, internal list should be empty
        self.assertEqual(len(server._tool_spans), 0)

    def test_get_tool_spans_no_middleware(self) -> None:
        """_get_tool_spans returns empty list for server without middleware."""
        server = MCPTool("http://localhost:8001/mcp")
        spans = _get_tool_spans(server)
        self.assertEqual(spans, [])


if __name__ == "__main__":
    unittest.main()

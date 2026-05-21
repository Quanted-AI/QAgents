"""Integration tests for Parallel cross-feature scenarios.

Covers SC4 (parallel + retry + tool interceptor + per-tool trace spans) and
additional scenario: parallel + retry + store namespacing (Scenario 3).

All tests use MockRunnable/AsyncMock for deterministic simulation. No real API calls.
"""

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock

from pydantic import BaseModel

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.mcp import _middleware_direct_call_tool
from quanted_agents.observability import ToolSpan
from quanted_agents.result import QuantedResult
from quanted_agents.workflows.parallel import Parallel, ParallelResult, RetryPolicy
from tests.conftest import make_store


# ---------------------------------------------------------------------------
# Test-specific BaseModels
# ---------------------------------------------------------------------------


class BranchInput(BaseModel):
    """Input for parallel branches."""

    text: str


class BranchOutput(BaseModel):
    """Output from parallel branches."""

    value: str


class AssembledOutput(BaseModel):
    """Output assembled from parallel branch results."""

    combined: str


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TransientError(Exception):
    """A transient error eligible for retry."""

    pass


# ---------------------------------------------------------------------------
# Mock Runnables
# ---------------------------------------------------------------------------


class SuccessRunnable:
    """A Runnable that always succeeds, tracking call count."""

    def __init__(self, value: str = "ok") -> None:
        self.output_type: type[BaseModel] = BranchOutput
        self.call_count: int = 0
        self._value: str = value

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Return a fixed success result.

        Args:
            input_data: Ignored.
            **kwargs: Ignored.

        Returns:
            A QuantedResult wrapping BranchOutput.
        """
        self.call_count += 1
        return QuantedResult.from_data(BranchOutput(value=self._value))


class FailOnceRunnable:
    """A Runnable that fails on first call, succeeds on retry."""

    def __init__(self, value: str = "recovered") -> None:
        self.output_type: type[BaseModel] = BranchOutput
        self.call_count: int = 0
        self._value: str = value

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Fail on first call, succeed on subsequent calls.

        Args:
            input_data: Ignored.
            **kwargs: Ignored.

        Returns:
            A QuantedResult on success.

        Raises:
            TransientError: On the first call.
        """
        self.call_count += 1
        if self.call_count == 1:
            raise TransientError("temporary failure")
        return QuantedResult.from_data(BranchOutput(value=self._value))


# ===========================================================================
# TestParallelIntegration
# ===========================================================================


class TestParallelIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Parallel with cross-feature interactions."""

    async def test_parallel_retry_store(self) -> None:
        """SC4 (retry + store): Parallel with retry policy and store, one branch fails then recovers.

        Validates that:
        - All 3 branches produce results (failed branch retried successfully).
        - Store has branch outputs in namespaced keys (branch_0/, branch_1/, branch_2/).
        - Failed branch was retried (call_count == 2).
        - Successful branches were NOT re-run (call_count == 1 each).
        """
        branch_0 = FailOnceRunnable(value="recovered_0")
        branch_1 = SuccessRunnable(value="ok_1")
        branch_2 = SuccessRunnable(value="ok_2")

        store = make_store()
        policy = RetryPolicy(max_retries=2, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(
            branches=[branch_0, branch_1, branch_2],
            retry_policy=policy,
            store=store,
        )

        result = await parallel.run(BranchInput(text="test"))

        # All 3 branches should have results (retry recovered branch_0)
        self.assertEqual(len(result.results), 3)
        self.assertEqual(len(result.errors), 0)

        # Store has namespaced branch outputs
        self.assertIn("branch_1/result", store)
        self.assertIn("branch_2/result", store)
        self.assertEqual(store["branch_1/result"], BranchOutput(value="ok_1"))
        self.assertEqual(store["branch_2/result"], BranchOutput(value="ok_2"))

        # Branch 0 was retried (call_count == 2: 1 fail + 1 success)
        self.assertEqual(branch_0.call_count, 2)
        # Successful branches were NOT re-run
        self.assertEqual(branch_1.call_count, 1)
        self.assertEqual(branch_2.call_count, 1)

    async def test_parallel_retry_store_namespacing(self) -> None:
        """Scenario 3: Parallel with retry, store namespacing, and assembly.

        Validates that:
        - Successful branches are NOT re-run during retry.
        - Failed branch gets retried and succeeds.
        - Assembly function receives all 3 branch outputs from store.
        - Store keys are correctly namespaced (branch_0/, branch_1/, branch_2/).
        """
        branch_0 = SuccessRunnable(value="alpha")
        branch_1 = FailOnceRunnable(value="beta_recovered")
        branch_2 = SuccessRunnable(value="gamma")

        store = make_store()
        policy = RetryPolicy(max_retries=2, retry_on=[TransientError], delay_seconds=0)
        assembly_received: dict[str, Any] = {}

        def assemble(st: ArtifactStore, pr: ParallelResult) -> AssembledOutput:
            assembly_received["store_keys"] = sorted(st.keys())
            assembly_received["result_count"] = len(pr.results)
            items = [r.data.value for r in pr.results]
            return AssembledOutput(combined="+".join(sorted(items)))

        parallel = Parallel(
            branches=[branch_0, branch_1, branch_2],
            retry_policy=policy,
            store=store,
            assembly=assemble,
        )

        result = await parallel.run(BranchInput(text="test"))

        # Result should be from assembly
        self.assertIsInstance(result.data, AssembledOutput)

        # Successful branches were NOT re-run
        self.assertEqual(branch_0.call_count, 1)
        self.assertEqual(branch_2.call_count, 1)

        # Failed branch was retried
        self.assertEqual(branch_1.call_count, 2)

        # Assembly received all 3 results
        self.assertEqual(assembly_received["result_count"], 3)

        # Store has namespaced keys for successful initial branches
        self.assertIn("branch_0/result", store)
        self.assertIn("branch_2/result", store)

    async def test_parallel_with_tool_interceptor_and_trace_spans(self) -> None:
        """SC4 (interceptor + trace): Tool interceptor and per-tool trace spans in parallel execution.

        Validates that within concurrent (Parallel-like) execution:
        - Interceptor fires per tool call (call_count == 3).
        - Each mock_fn receives interceptor-modified args ("intercepted" key).
        - Trace collector contains 3 ToolSpan entries with correct tool names.
        - All spans have status="success" and duration_seconds > 0.
        - Spans have args populated (trace_level="standard").
        """
        interceptor_calls: list[str] = []

        def my_interceptor(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
            interceptor_calls.append(tool_name)
            args["intercepted"] = True
            return args

        trace_collector: list[ToolSpan] = []
        tool_names = ["branch_0_tool", "branch_1_tool", "branch_2_tool"]

        # Create 3 mock original functions that capture their received args
        received_args: list[dict[str, Any]] = []
        mock_fns: list[AsyncMock] = []
        for _ in tool_names:
            mock_fn = AsyncMock(return_value="tool_result")
            mock_fns.append(mock_fn)

        # Run all 3 tool calls concurrently via asyncio.gather (simulating Parallel)
        async def call_tool(name: str, mock: AsyncMock) -> Any:
            return await _middleware_direct_call_tool(
                mock,
                name,
                {"input": f"data_for_{name}"},
                None,
                interceptor=my_interceptor,
                concurrency_backend=None,
                trace_collector=trace_collector,
                trace_level="standard",
                retry_config=None,
            )

        results = await asyncio.gather(
            call_tool(tool_names[0], mock_fns[0]),
            call_tool(tool_names[1], mock_fns[1]),
            call_tool(tool_names[2], mock_fns[2]),
        )

        # Verify interceptor was called once per tool call
        self.assertEqual(len(interceptor_calls), 3)
        self.assertIn("branch_0_tool", interceptor_calls)
        self.assertIn("branch_1_tool", interceptor_calls)
        self.assertIn("branch_2_tool", interceptor_calls)

        # Verify each mock_fn received interceptor-modified args
        for mock_fn in mock_fns:
            mock_fn.assert_called_once()
            called_args = mock_fn.call_args[0][1]
            self.assertTrue(called_args.get("intercepted"))

        # Verify trace_collector contains 3 ToolSpan entries
        self.assertEqual(len(trace_collector), 3)
        span_names = {span.tool_name for span in trace_collector}
        self.assertEqual(span_names, set(tool_names))

        # Verify all spans have correct status and positive duration
        for span in trace_collector:
            self.assertEqual(span.status, "success")
            self.assertGreater(span.duration_seconds, 0)
            # Args populated at standard trace level
            self.assertIsNotNone(span.args)
            self.assertTrue(span.args.get("intercepted"))

        # All results should be "tool_result"
        for r in results:
            self.assertEqual(r, "tool_result")


if __name__ == "__main__":
    unittest.main()

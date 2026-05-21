"""Tests for assembly functions on Pipeline, Loop, and Parallel.

Validates store integration, assembly function invocation, error handling,
backward compatibility (no store/no assembly), and trace_artifacts metadata.
"""

import asyncio
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.exceptions import AssemblyError
from quanted_agents.observability import StepTiming
from quanted_agents.result import QuantedResult
from quanted_agents.workflows.loop import Loop
from quanted_agents.workflows.parallel import Parallel, ParallelResult
from quanted_agents.workflows.pipeline import Pipeline


class StepAOutput(BaseModel):
    """Output for step A / input for step B."""

    value: str


class StepBOutput(BaseModel):
    """Output for step B."""

    report: str


class IterOutput(BaseModel):
    """Output for loop iterations."""

    score: float


class Assembled(BaseModel):
    """Assembled output from store artifacts."""

    combined: str


class MockRunnable:
    """A Runnable that returns a fixed QuantedResult.from_data() output.

    Implements the Runnable protocol for testing without real LLM calls.
    """

    def __init__(self, output: BaseModel) -> None:
        self._output: BaseModel = output
        self.output_type: type[BaseModel] = type(output)

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Return a fixed result wrapping the configured output.

        Args:
            input_data: Ignored; the mock always returns the same output.
            **kwargs: Ignored.

        Returns:
            A QuantedResult wrapping the fixed output.
        """
        return QuantedResult.from_data(self._output)


class TestPipelineAssembly(unittest.IsolatedAsyncioTestCase):
    """Tests for Pipeline store and assembly integration."""

    async def test_pipeline_writes_step_data_to_store(self) -> None:
        """Pipeline with store= writes step outputs under step_{i}/{type} keys."""
        step_a = MockRunnable(StepAOutput(value="hello"))
        step_b = MockRunnable(StepBOutput(report="world"))
        store = ArtifactStore()
        pipeline = Pipeline(steps=[step_a, step_b], store=store)
        result = await pipeline.run(StepAOutput(value="input"))

        self.assertIn("step_0/stepaoutput", store)
        self.assertIn("step_1/stepboutput", store)
        self.assertEqual(store["step_0/stepaoutput"], StepAOutput(value="hello"))
        self.assertEqual(store["step_1/stepboutput"], StepBOutput(report="world"))

    async def test_pipeline_assembly_called_with_store_and_result(self) -> None:
        """Pipeline with assembly= calls it with store and last result."""
        step_a = MockRunnable(StepAOutput(value="hello"))
        step_b = MockRunnable(StepBOutput(report="world"))

        def assemble(store: ArtifactStore, last_result: QuantedResult) -> Assembled:
            a_val = store.get("step_0/stepaoutput", StepAOutput).value
            b_val = store.get("step_1/stepboutput", StepBOutput).report
            return Assembled(combined=f"{a_val}-{b_val}")

        pipeline = Pipeline(steps=[step_a, step_b], assembly=assemble)
        result = await pipeline.run(StepAOutput(value="input"))

        self.assertIsInstance(result.data, Assembled)
        self.assertEqual(result.data.combined, "hello-world")

    async def test_pipeline_assembly_error_preserves_state(self) -> None:
        """Assembly that raises is wrapped in AssemblyError with store and result."""
        step_a = MockRunnable(StepAOutput(value="a"))
        step_b = MockRunnable(StepBOutput(report="b"))

        def bad_assembly(store: ArtifactStore, last_result: QuantedResult) -> Assembled:
            raise ValueError("assembly failed")

        pipeline = Pipeline(steps=[step_a, step_b], assembly=bad_assembly)
        with self.assertRaises(AssemblyError) as ctx:
            await pipeline.run(StepAOutput(value="input"))
        err = ctx.exception
        self.assertIsInstance(err.store, ArtifactStore)
        self.assertIsNotNone(err.last_result)
        self.assertIsInstance(err.original_error, ValueError)

    async def test_pipeline_no_store_no_assembly_unchanged(self) -> None:
        """Pipeline without store/assembly works exactly as before."""
        step_a = MockRunnable(StepAOutput(value="hello"))
        step_b = MockRunnable(StepBOutput(report="world"))
        pipeline = Pipeline(steps=[step_a, step_b])
        result = await pipeline.run(StepAOutput(value="input"))

        self.assertEqual(result.data, StepBOutput(report="world"))

    async def test_pipeline_trace_artifacts_writes_metadata(self) -> None:
        """Pipeline with trace_artifacts=True writes _pipeline/step_order."""
        step_a = MockRunnable(StepAOutput(value="a"))
        step_b = MockRunnable(StepBOutput(report="b"))
        store = ArtifactStore()
        pipeline = Pipeline(
            steps=[step_a, step_b], store=store, trace_artifacts=True,
        )
        await pipeline.run(StepAOutput(value="input"))
        self.assertIn("_pipeline/step_order", store)
        order = store["_pipeline/step_order"]
        self.assertEqual(order, ["step_0/stepaoutput", "step_1/stepboutput"])

    async def test_pipeline_result_artifacts_property_returns_store(self) -> None:
        """After Pipeline with store, result.artifacts returns the store."""
        step_a = MockRunnable(StepAOutput(value="a"))
        step_b = MockRunnable(StepBOutput(report="b"))
        store = ArtifactStore()
        pipeline = Pipeline(steps=[step_a, step_b], store=store)
        result = await pipeline.run(StepAOutput(value="input"))
        self.assertIs(result.artifacts, store)


class TestLoopAssembly(unittest.IsolatedAsyncioTestCase):
    """Tests for Loop store and assembly integration."""

    async def test_loop_writes_iteration_result_with_history(self) -> None:
        """Loop with store= writes iteration_result with version history."""
        call_count = 0

        class CountingMock:
            output_type = IterOutput

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                return QuantedResult.from_data(IterOutput(score=call_count * 0.5))

        store = ArtifactStore()
        loop = Loop(
            body=CountingMock(),
            termination_check=lambda d: d.score >= 1.0,
            max_iterations=5,
            store=store,
        )
        result = await loop.run(IterOutput(score=0.0))

        history = store.history("iteration_result")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].score, 0.5)
        self.assertEqual(history[1].score, 1.0)

    async def test_loop_assembly_runs_after_convergence(self) -> None:
        """Loop with assembly= calls it after convergence and returns assembled result."""
        mock = MockRunnable(IterOutput(score=1.0))

        def assemble(store: ArtifactStore, last_result: QuantedResult) -> Assembled:
            return Assembled(combined="assembled")

        loop = Loop(
            body=mock,
            termination_check=lambda d: True,
            max_iterations=3,
            assembly=assemble,
        )
        result = await loop.run(IterOutput(score=0.0))
        self.assertIsInstance(result.data, Assembled)
        self.assertEqual(result.data.combined, "assembled")

    async def test_loop_no_store_unchanged(self) -> None:
        """Loop without store works as before."""
        mock = MockRunnable(IterOutput(score=1.0))
        loop = Loop(
            body=mock,
            termination_check=lambda d: True,
            max_iterations=3,
        )
        result = await loop.run(IterOutput(score=0.0))
        self.assertEqual(result.data, IterOutput(score=1.0))

    async def test_loop_trace_artifacts_writes_metadata(self) -> None:
        """Loop with trace_artifacts=True writes _loop/iteration_count."""
        mock = MockRunnable(IterOutput(score=1.0))
        store = ArtifactStore()
        loop = Loop(
            body=mock,
            termination_check=lambda d: True,
            max_iterations=3,
            store=store,
            trace_artifacts=True,
        )
        await loop.run(IterOutput(score=0.0))
        self.assertEqual(store["_loop/iteration_count"], 1)


class TestParallelAssembly(unittest.IsolatedAsyncioTestCase):
    """Tests for Parallel store and assembly integration."""

    async def test_parallel_namespaced_branches(self) -> None:
        """Parallel with store= writes branch outputs under branch_{i}/ keys."""
        branch_a = MockRunnable(StepAOutput(value="from_a"))
        branch_b = MockRunnable(StepBOutput(report="from_b"))
        store = ArtifactStore()
        parallel = Parallel(branches=[branch_a, branch_b], store=store)
        result = await parallel.run(StepAOutput(value="input"))

        self.assertIn("branch_0/result", store)
        self.assertIn("branch_1/result", store)
        self.assertEqual(store["branch_0/result"], StepAOutput(value="from_a"))
        self.assertEqual(store["branch_1/result"], StepBOutput(report="from_b"))

    async def test_parallel_assembly_receives_parallel_result(self) -> None:
        """Parallel with assembly= receives ParallelResult with .results and .errors."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))
        received_args = {}

        def assemble(store: ArtifactStore, pr: ParallelResult) -> Assembled:
            received_args["store"] = store
            received_args["parallel_result"] = pr
            return Assembled(combined="assembled")

        parallel = Parallel(branches=[branch_a, branch_b], assembly=assemble)
        result = await parallel.run(StepAOutput(value="input"))

        self.assertIsInstance(result.data, Assembled)
        self.assertEqual(result.data.combined, "assembled")
        self.assertIsInstance(received_args["parallel_result"], ParallelResult)
        self.assertEqual(len(received_args["parallel_result"].results), 2)

    async def test_parallel_no_store_unchanged(self) -> None:
        """Parallel without store works as before."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))
        parallel = Parallel(branches=[branch_a, branch_b])
        result = await parallel.run(StepAOutput(value="input"))
        self.assertIsInstance(result, ParallelResult)
        self.assertEqual(len(result.results), 2)

    async def test_parallel_trace_artifacts_writes_metadata(self) -> None:
        """Parallel with trace_artifacts=True writes _parallel/branch_count."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))
        store = ArtifactStore()
        parallel = Parallel(
            branches=[branch_a, branch_b], store=store, trace_artifacts=True,
        )
        await parallel.run(StepAOutput(value="input"))
        self.assertEqual(store["_parallel/branch_count"], 2)

    async def test_parallel_assembly_error_preserves_state(self) -> None:
        """Assembly that raises on Parallel is wrapped in AssemblyError."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))

        def bad_assembly(store: ArtifactStore, pr: ParallelResult) -> Assembled:
            raise RuntimeError("boom")

        parallel = Parallel(branches=[branch_a, branch_b], assembly=bad_assembly)
        with self.assertRaises(AssemblyError) as ctx:
            await parallel.run(StepAOutput(value="input"))
        self.assertIsInstance(ctx.exception.original_error, RuntimeError)

    async def test_async_assembly_function(self) -> None:
        """An async assembly function is correctly awaited."""
        step_a = MockRunnable(StepAOutput(value="a"))
        step_b = MockRunnable(StepBOutput(report="b"))

        async def async_assemble(store: ArtifactStore, last_result: QuantedResult) -> Assembled:
            await asyncio.sleep(0)
            return Assembled(combined="async-assembled")

        pipeline = Pipeline(steps=[step_a, step_b], assembly=async_assemble)
        result = await pipeline.run(StepAOutput(value="input"))
        self.assertEqual(result.data.combined, "async-assembled")

    async def test_parallel_result_has_summary_and_artifacts_attributes(self) -> None:
        """ParallelResult has _summary_extracted, _summary_value, _artifacts."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))
        parallel = Parallel(branches=[branch_a, branch_b])
        result = await parallel.run(StepAOutput(value="input"))
        self.assertTrue(result._summary_extracted)
        self.assertIsNone(result._summary_value)
        self.assertIsNone(result.summary)

    async def test_result_artifacts_property_returns_store(self) -> None:
        """After Parallel with store, result.artifacts returns the store."""
        branch_a = MockRunnable(StepAOutput(value="a"))
        branch_b = MockRunnable(StepBOutput(report="b"))
        store = ArtifactStore()
        parallel = Parallel(branches=[branch_a, branch_b], store=store)
        result = await parallel.run(StepAOutput(value="input"))
        self.assertIs(result.artifacts, store)


if __name__ == "__main__":
    unittest.main()

"""Tests for ParallelResult.trace propagation in nested workflows and standalone mode.

Validates that:
1. When Parallel is the final step of a Pipeline, result.trace includes trace entries
   from all preceding Pipeline steps plus the Parallel branch traces.
2. When Parallel runs standalone (not nested), result.trace returns branch traces
   as before (no regression).
3. No trace entry duplication occurs when Parallel is nested in a Pipeline.
4. Runnable is importable from the top-level quanted_agents package.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent, Runnable
from quanted_agents.workflows import Parallel, Pipeline


class TraceInput(BaseModel):
    """Input model for trace propagation tests."""

    text: str


class TraceMiddle(BaseModel):
    """Intermediate model between pipeline steps."""

    processed: str


class TraceOutputA(BaseModel):
    """Output model for the scoring branch."""

    score: float


class TraceOutputB(BaseModel):
    """Output model for the categorization branch."""

    category: str


def _score_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler returning a TraceOutputA JSON response."""
    output = TraceOutputA(score=0.85)
    return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])


def _category_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler returning a TraceOutputB JSON response."""
    output = TraceOutputB(category="test")
    return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])


class TestParallelTraceInNestedWorkflows(unittest.IsolatedAsyncioTestCase):
    """Tests for ParallelResult.trace in nested workflows and standalone mode."""

    async def test_pipeline_with_parallel_final_step_trace_includes_preceding_steps(self) -> None:
        """Pipeline ending with Parallel includes trace entries from step1 and both branches."""
        step1 = QuantedAgent(
            "test",
            input_type=TraceInput,
            output_type=TraceMiddle,
            system_prompt="Process input",
        )
        branch_a = QuantedAgent(
            "test",
            input_type=TraceMiddle,
            output_type=TraceOutputA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=TraceMiddle,
            output_type=TraceOutputB,
            system_prompt="Categorize",
        )
        parallel = Parallel(branches=[branch_a, branch_b])
        pipeline = Pipeline(steps=[step1, parallel])

        with step1.inner.override(model=TestModel()):
            with branch_a.inner.override(model=FunctionModel(_score_model)):
                with branch_b.inner.override(model=FunctionModel(_category_model)):
                    result = await pipeline.run(TraceInput(text="hello"))

        # At least 3 trace entries: 1 from step1 + 2 from parallel branches
        self.assertGreaterEqual(len(result.trace), 3)

        # Verify step1's trace entry is present (step_name should contain the agent's output type)
        step_names = [entry.step_name for entry in result.trace]
        has_step1_trace = any("TraceMiddle" in name for name in step_names)
        self.assertTrue(
            has_step1_trace,
            f"Expected a trace entry from step1 (TraceMiddle agent), got step_names: {step_names}",
        )

    async def test_standalone_parallel_trace_returns_branch_traces(self) -> None:
        """Standalone Parallel (no outer workflow) returns exactly branch traces."""
        branch_a = QuantedAgent(
            "test",
            input_type=TraceInput,
            output_type=TraceOutputA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=TraceInput,
            output_type=TraceOutputB,
            system_prompt="Categorize",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(_score_model)):
            with branch_b.inner.override(model=FunctionModel(_category_model)):
                result = await parallel.run(TraceInput(text="hello"))

        # Standalone: exactly 2 trace entries (one per branch)
        self.assertEqual(len(result.trace), 2)

    async def test_pipeline_with_parallel_final_step_no_trace_duplication(self) -> None:
        """Pipeline+Parallel final step produces exactly 3 trace entries, not duplicated."""
        step1 = QuantedAgent(
            "test",
            input_type=TraceInput,
            output_type=TraceMiddle,
            system_prompt="Process input",
        )
        branch_a = QuantedAgent(
            "test",
            input_type=TraceMiddle,
            output_type=TraceOutputA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=TraceMiddle,
            output_type=TraceOutputB,
            system_prompt="Categorize",
        )
        parallel = Parallel(branches=[branch_a, branch_b])
        pipeline = Pipeline(steps=[step1, parallel])

        with step1.inner.override(model=TestModel()):
            with branch_a.inner.override(model=FunctionModel(_score_model)):
                with branch_b.inner.override(model=FunctionModel(_category_model)):
                    result = await pipeline.run(TraceInput(text="hello"))

        # Exactly 3 trace entries: 1 from step1 + 2 from branches
        # If duplication occurred, we would see 5 (branch traces appear twice)
        self.assertEqual(len(result.trace), 3)

    def test_runnable_export_from_top_level_package(self) -> None:
        """Runnable is importable from quanted_agents and is the real protocol."""
        self.assertIsNotNone(Runnable)

        # Verify it's actually the protocol by checking a QuantedAgent is an instance
        agent = QuantedAgent(
            "test",
            input_type=TraceInput,
            output_type=TraceOutputA,
            system_prompt="Test",
        )
        self.assertIsInstance(agent, Runnable)


if __name__ == "__main__":
    unittest.main()

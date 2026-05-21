"""Tests for the Parallel workflow composition primitive.

Validates that Parallel runs multiple Runnables concurrently, collects
both successes and errors, provides ParallelResult with structured access,
propagates kwargs, and implements the Runnable protocol.
Uses FunctionModel to simulate different branch behaviors and TestModel
for simple pass-through.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from quanted_agents import QuantedAgent
from quanted_agents.result import QuantedResult
from quanted_agents.types import Runnable
from quanted_agents.workflows import Parallel
from quanted_agents.workflows.parallel import ParallelOutput, ParallelResult


class ParInput(BaseModel):
    """Input model for parallel branch testing."""

    text: str


class AnalysisA(BaseModel):
    """Output model for the first parallel branch (scoring)."""

    score: float


class AnalysisB(BaseModel):
    """Output model for the second parallel branch (topic extraction)."""

    topics: list[str]


def _score_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler returning an AnalysisA JSON response."""
    output = AnalysisA(score=0.85)
    return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])


def _topics_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler returning an AnalysisB JSON response."""
    output = AnalysisB(topics=["python", "testing"])
    return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])


class TestParallel(unittest.IsolatedAsyncioTestCase):
    """Tests for Parallel workflow: concurrency, error collection, protocol."""

    def test_parallel_requires_at_least_two_branches(self) -> None:
        """Constructing a Parallel with fewer than 2 branches raises ValueError."""
        branch = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        with self.assertRaises(ValueError) as ctx:
            Parallel(branches=[branch])
        self.assertIn("at least 2", str(ctx.exception))

    async def test_parallel_runs_branches_concurrently(self) -> None:
        """Parallel executes all branches and collects results from each."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(_score_model)):
            with branch_b.inner.override(model=FunctionModel(_topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        self.assertIsInstance(result, ParallelResult)
        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.errors), 0)

    async def test_parallel_collects_errors(self) -> None:
        """Parallel collects exceptions from failed branches alongside successes."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )

        class FailingRunnable:
            """A Runnable that always raises an exception."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                raise RuntimeError("Branch failed")

        parallel = Parallel(branches=[branch_a, FailingRunnable()])

        with branch_a.inner.override(model=FunctionModel(_score_model)):
            result = await parallel.run(ParInput(text="test input"))

        self.assertEqual(len(result.results), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIsInstance(result.errors[0], RuntimeError)
        self.assertIn("Branch failed", str(result.errors[0]))

    async def test_parallel_result_data_property(self) -> None:
        """ParallelResult.data returns a ParallelOutput with items from each branch."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(_score_model)):
            with branch_b.inner.override(model=FunctionModel(_topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        data = result.data
        self.assertIsInstance(data, ParallelOutput)
        self.assertEqual(len(data.items), 2)

    def test_parallel_implements_runnable(self) -> None:
        """Parallel instances satisfy the Runnable protocol for composability."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])
        self.assertIsInstance(parallel, Runnable)

    async def test_parallel_propagates_kwargs(self) -> None:
        """Kwargs passed to parallel.run() are forwarded to all branches."""
        captured_kwargs: list[dict[str, Any]] = []

        class KwargsCapturingRunnable:
            """A Runnable that captures kwargs for verification."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                captured_kwargs.append(kwargs)
                return QuantedResult.from_data(AnalysisA(score=0.5))

        branch_a = KwargsCapturingRunnable()
        branch_b = KwargsCapturingRunnable()
        parallel = Parallel(branches=[branch_a, branch_b])

        await parallel.run(ParInput(text="test"), custom_param="test_value")

        self.assertEqual(len(captured_kwargs), 2)
        self.assertEqual(captured_kwargs[0]["custom_param"], "test_value")
        self.assertEqual(captured_kwargs[1]["custom_param"], "test_value")


if __name__ == "__main__":
    unittest.main()

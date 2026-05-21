"""Tests for the Pipeline workflow composition primitive.

Validates that Pipeline chains Runnable steps sequentially, enforces
minimum step count, propagates kwargs, and implements the Runnable protocol.
All tests use pydantic-ai's TestModel with agent.inner.override().
"""

import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from quanted_agents import PipelineTypeError, QuantedAgent
from quanted_agents.result import QuantedResult
from quanted_agents.types import Runnable
from quanted_agents.workflows import Pipeline


class StepAInput(BaseModel):
    """Input model for the first pipeline step."""

    text: str


class StepAOutput(BaseModel):
    """Output model for the first pipeline step / input for the second."""

    summary: str


class StepBOutput(BaseModel):
    """Output model for the second pipeline step."""

    report: str


class FinalOutput(BaseModel):
    """Output model for a third pipeline step."""

    conclusion: str


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    """Tests for Pipeline workflow: chaining, validation, kwargs, and protocol."""

    def test_pipeline_requires_at_least_two_steps(self) -> None:
        """Constructing a Pipeline with fewer than 2 steps raises ValueError."""
        step = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Single step",
        )
        with self.assertRaises(ValueError) as ctx:
            Pipeline(steps=[step])
        self.assertIn("at least 2", str(ctx.exception))

    async def test_pipeline_chains_output_to_input(self) -> None:
        """Pipeline passes result.data from step 1 as input_data to step 2."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Summarize",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepAOutput,
            output_type=StepBOutput,
            system_prompt="Report",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(StepAInput(text="hello"))
                self.assertIsInstance(result.data, StepBOutput)

    async def test_pipeline_returns_final_step_result(self) -> None:
        """Pipeline returns the QuantedResult from the last step, not intermediate steps."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Summarize",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepAOutput,
            output_type=StepBOutput,
            system_prompt="Report",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(StepAInput(text="hello"))
                self.assertIsInstance(result, QuantedResult)
                self.assertIsInstance(result.data, StepBOutput)
                self.assertNotIsInstance(result.data, StepAOutput)

    async def test_pipeline_with_three_steps(self) -> None:
        """Pipeline supports chaining three or more steps sequentially."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Summarize",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepAOutput,
            output_type=StepBOutput,
            system_prompt="Report",
        )
        step3 = QuantedAgent(
            "test",
            input_type=StepBOutput,
            output_type=FinalOutput,
            system_prompt="Conclude",
        )
        pipeline = Pipeline(steps=[step1, step2, step3])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                with step3.inner.override(model=TestModel()):
                    result = await pipeline.run(StepAInput(text="hello"))
                    self.assertIsInstance(result.data, FinalOutput)

    async def test_pipeline_propagates_kwargs(self) -> None:
        """Kwargs passed to pipeline.run() are forwarded to each step's run()."""
        captured_kwargs: list[dict[str, Any]] = []

        class KwargsCapturingRunnable:
            """A Runnable that captures kwargs for verification."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                captured_kwargs.append(kwargs)
                return QuantedResult.from_data(StepAOutput(summary="captured"))

        step1 = KwargsCapturingRunnable()
        step2 = KwargsCapturingRunnable()
        pipeline = Pipeline(steps=[step1, step2])

        await pipeline.run(StepAInput(text="hello"), custom_param="test_value")

        self.assertEqual(len(captured_kwargs), 2)
        self.assertEqual(captured_kwargs[0]["custom_param"], "test_value")
        self.assertEqual(captured_kwargs[1]["custom_param"], "test_value")

    def test_pipeline_implements_runnable(self) -> None:
        """Pipeline instances satisfy the Runnable protocol for composability."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Summarize",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepAOutput,
            output_type=StepBOutput,
            system_prompt="Report",
        )
        pipeline = Pipeline(steps=[step1, step2])
        self.assertIsInstance(pipeline, Runnable)

    def test_pipeline_raises_type_error_on_mismatched_steps(self) -> None:
        """Pipeline raises PipelineTypeError when adjacent QuantedAgent step types mismatch."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepBOutput,
            output_type=FinalOutput,
            system_prompt="Step 2",
        )
        with self.assertRaises(PipelineTypeError) as ctx:
            Pipeline(steps=[step1, step2])
        self.assertIn("StepAOutput", str(ctx.exception))
        self.assertIn("StepBOutput", str(ctx.exception))

    def test_pipeline_allows_matching_types(self) -> None:
        """Pipeline construction succeeds when adjacent QuantedAgent step types match."""
        step1 = QuantedAgent(
            "test",
            input_type=StepAInput,
            output_type=StepAOutput,
            system_prompt="Summarize",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepAOutput,
            output_type=StepBOutput,
            system_prompt="Report",
        )
        pipeline = Pipeline(steps=[step1, step2])
        self.assertIsNotNone(pipeline)

    def test_pipeline_skips_type_check_for_non_agent_runnables(self) -> None:
        """Pipeline allows non-QuantedAgent Runnables without type checking."""
        class KwargsCapturingRunnable:
            """A Runnable that captures kwargs for verification."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(StepAOutput(summary="captured"))

        step1 = KwargsCapturingRunnable()
        step2 = KwargsCapturingRunnable()
        pipeline = Pipeline(steps=[step1, step2])
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()

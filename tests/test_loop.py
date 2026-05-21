"""Tests for the Loop workflow composition primitive.

Validates that Loop iterates a body Runnable until a termination check
passes, enforces mandatory max_iterations, returns last result at the
limit, propagates kwargs, and implements the Runnable protocol.
Uses FunctionModel to simulate iterative refinement and TestModel for
simple pass-through.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from quanted_agents import MaxIterationsExceeded, QuantedAgent
from quanted_agents.result import QuantedResult
from quanted_agents.types import Runnable
from quanted_agents.workflows import Loop


class Draft(BaseModel):
    """Test model for iterative refinement loops."""

    content: str
    quality_score: float = 0.0


class TestLoop(unittest.IsolatedAsyncioTestCase):
    """Tests for Loop workflow: iteration, termination, max_iterations, and protocol."""

    def test_loop_requires_max_iterations(self) -> None:
        """Constructing a Loop without max_iterations keyword raises TypeError."""
        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        with self.assertRaises(TypeError):
            Loop(body=body, termination_check=lambda d: True)  # type: ignore[call-arg]

    def test_loop_rejects_zero_max_iterations(self) -> None:
        """Constructing a Loop with max_iterations=0 raises ValueError."""
        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        with self.assertRaises(ValueError) as ctx:
            Loop(body=body, termination_check=lambda d: True, max_iterations=0)
        self.assertIn("max_iterations must be >= 1", str(ctx.exception))

    async def test_loop_terminates_on_check(self) -> None:
        """Loop stops early when the termination check returns True."""
        call_count = [0]

        def _refiner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            score = call_count[0] * 0.3
            draft = Draft(content="refined", quality_score=score)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine the draft",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.8,
            max_iterations=10,
        )

        fm = FunctionModel(_refiner_model)
        with body.inner.override(model=fm):
            result = await loop.run(Draft(content="test", quality_score=0.0))

        self.assertGreaterEqual(result.data.quality_score, 0.8)
        self.assertLess(call_count[0], 10)

    async def test_loop_raises_max_iterations_exceeded(self) -> None:
        """Loop raises MaxIterationsExceeded when termination check never passes."""
        call_count = [0]

        def _low_score_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            draft = Draft(content="low", quality_score=0.1)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.8,
            max_iterations=3,
        )

        fm = FunctionModel(_low_score_model)
        with body.inner.override(model=fm):
            with self.assertRaises(MaxIterationsExceeded) as ctx:
                await loop.run(Draft(content="test", quality_score=0.0))
        self.assertIn("3", str(ctx.exception))
        self.assertEqual(call_count[0], 3)

    async def test_loop_single_iteration_raises_when_check_fails(self) -> None:
        """Loop with max_iterations=1 and failing check raises MaxIterationsExceeded."""
        call_count = [0]

        def _single_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            draft = Draft(content="once", quality_score=0.5)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: False,
            max_iterations=1,
        )

        fm = FunctionModel(_single_model)
        with body.inner.override(model=fm):
            with self.assertRaises(MaxIterationsExceeded):
                await loop.run(Draft(content="test", quality_score=0.0))
        self.assertEqual(call_count[0], 1)

    def test_loop_implements_runnable(self) -> None:
        """Loop instances satisfy the Runnable protocol for composability."""
        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: True,
            max_iterations=5,
        )
        self.assertIsInstance(loop, Runnable)

    async def test_loop_propagates_kwargs(self) -> None:
        """Kwargs passed to loop.run() are forwarded to the body's run()."""
        captured_kwargs: list[dict[str, Any]] = []

        class KwargsCapturingRunnable:
            """A Runnable that captures kwargs for verification."""

            def __init__(self) -> None:
                self.call_count: int = 0

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                captured_kwargs.append(kwargs)
                self.call_count += 1
                return QuantedResult.from_data(Draft(content="captured", quality_score=1.0))

        body = KwargsCapturingRunnable()
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 1.0,
            max_iterations=5,
        )

        await loop.run(Draft(content="test"), custom_param="test_value")

        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(captured_kwargs[0]["custom_param"], "test_value")


if __name__ == "__main__":
    unittest.main()

"""Tests for the Router workflow composition primitive.

Validates that Router dispatches input to the correct specialist based on
the dispatcher's RoutingDecision, raises RoutingError on invalid targets,
passes original input to specialists, and implements the Runnable protocol.
All tests use pydantic-ai's FunctionModel and TestModel with agent.inner.override().
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from quanted_agents.exceptions import RoutingError
from quanted_agents.result import QuantedResult
from quanted_agents.types import Runnable
from quanted_agents.workflows.router import Router, RoutingDecision


class RouterInput(BaseModel):
    """Input model for Router tests."""

    query: str


class SpecialistAOutput(BaseModel):
    """Output model for specialist A."""

    answer_a: str


class SpecialistBOutput(BaseModel):
    """Output model for specialist B."""

    answer_b: str


def _make_dispatcher_function(target: str, reasoning: str = "test routing"):
    """Create a FunctionModel handler that returns a RoutingDecision via output tool.

    Args:
        target: The specialist name to route to.
        reasoning: Explanation for the routing decision.

    Returns:
        A function compatible with FunctionModel that returns the routing decision.
    """

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        """Return a RoutingDecision as a tool call response."""
        decision = {"target": target, "reasoning": reasoning}
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(decision))]
        )

    return handler


class TestRouter(unittest.IsolatedAsyncioTestCase):
    """Tests for Router workflow: dispatch, validation, kwargs, and protocol."""

    def test_router_requires_at_least_one_specialist(self) -> None:
        """Constructing a Router with empty specialists dict raises ValueError."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        with self.assertRaises(ValueError) as ctx:
            Router(dispatcher=dispatcher, specialists={})
        self.assertIn("at least 1", str(ctx.exception))

    async def test_router_routes_to_correct_specialist(self) -> None:
        """Router invokes the specialist named in the dispatcher's RoutingDecision."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist_a = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistAOutput,
            system_prompt="Specialist A",
        )
        specialist_b = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistBOutput,
            system_prompt="Specialist B",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist_a, "specialist_b": specialist_b},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("specialist_a"))
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist_a.inner.override(model=TestModel()):
                with specialist_b.inner.override(model=TestModel()):
                    result = await router.run(RouterInput(query="test"))
                    self.assertIsInstance(result.data, SpecialistAOutput)

    async def test_router_raises_routing_error_on_invalid_target(self) -> None:
        """Router raises RoutingError when dispatcher picks a nonexistent target."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist_a = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistAOutput,
            system_prompt="Specialist A",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist_a},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("nonexistent"))
        with dispatcher.inner.override(model=dispatcher_model):
            with self.assertRaises(RoutingError) as ctx:
                await router.run(RouterInput(query="test"))
            error_message = str(ctx.exception)
            self.assertIn("nonexistent", error_message)
            self.assertIn("specialist_a", error_message)

    async def test_router_passes_original_input_to_specialist(self) -> None:
        """Router passes the original RouterInput to the specialist, not RoutingDecision."""
        captured_prompts: list[str] = []

        def specialist_handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
            """Capture the prompt received by the specialist and return output."""
            for message in messages:
                for part in message.parts:
                    if hasattr(part, "content"):
                        captured_prompts.append(part.content)
            tool = agent_info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool.name, args=json.dumps({"answer_a": "captured"}))]
            )

        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist_a = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistAOutput,
            system_prompt="Specialist A",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist_a},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("specialist_a"))
        specialist_model = FunctionModel(specialist_handler)
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist_a.inner.override(model=specialist_model):
                await router.run(RouterInput(query="important question"))

        # The specialist should receive the original RouterInput JSON, not RoutingDecision
        joined_prompts = " ".join(captured_prompts)
        self.assertIn("important question", joined_prompts)
        self.assertNotIn("specialist_a", joined_prompts)

    def test_router_implements_runnable(self) -> None:
        """Router instances satisfy the Runnable protocol for composability."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist_a = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistAOutput,
            system_prompt="Specialist A",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist_a},
        )
        self.assertIsInstance(router, Runnable)

    async def test_router_propagates_kwargs(self) -> None:
        """Kwargs passed to router.run() are forwarded to both dispatcher and specialist."""
        captured_kwargs: list[dict[str, Any]] = []

        class KwargsCapturingRunnable:
            """A Runnable that captures kwargs for verification."""

            def __init__(self, return_data: BaseModel) -> None:
                self._return_data: BaseModel = return_data

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                """Capture kwargs and return preset data."""
                captured_kwargs.append(kwargs)
                return QuantedResult.from_data(self._return_data)

        dispatcher = KwargsCapturingRunnable(
            return_data=RoutingDecision(target="specialist_a", reasoning="test")
        )
        specialist = KwargsCapturingRunnable(
            return_data=SpecialistAOutput(answer_a="result")
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist},
        )

        await router.run(RouterInput(query="test"), custom_param="test_value")

        self.assertEqual(len(captured_kwargs), 2)
        self.assertEqual(captured_kwargs[0]["custom_param"], "test_value")
        self.assertEqual(captured_kwargs[1]["custom_param"], "test_value")


if __name__ == "__main__":
    unittest.main()

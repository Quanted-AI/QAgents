"""Tests for QuantedAgent run execution.

Validates that agent.run() correctly validates input types, serializes BaseModel
input to JSON prompts, returns QuantedResult with typed data access, and exposes
usage and message metadata. All tests use pydantic-ai's TestModel.
"""

import unittest

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from quanted_agents import InvalidInputType, QuantedAgent, QuantedResult
from tests.conftest import SampleInput, SampleOutput


class TestAgentRun(unittest.IsolatedAsyncioTestCase):
    """Tests for QuantedAgent.run() execution, result access, and input validation."""

    def setUp(self) -> None:
        """Create a standard test agent for reuse across test methods."""
        self.agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )

    async def test_run_returns_quanted_result(self) -> None:
        """Running the agent returns a QuantedResult instance."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)

    async def test_run_result_data_is_output_type(self) -> None:
        """The result.data property returns an instance of the configured output type."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))
            self.assertIsInstance(result.data, SampleOutput)

    async def test_run_result_has_usage(self) -> None:
        """The result.usage property returns a RunUsage with token counts."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))
            self.assertIsNotNone(result.usage)
            self.assertIsInstance(result.usage, RunUsage)
            self.assertGreater(result.usage.requests, 0)

    async def test_run_result_has_messages(self) -> None:
        """The result.messages property returns a non-empty list of all messages."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))
            self.assertIsInstance(result.messages, list)
            self.assertGreater(len(result.messages), 0)

    async def test_run_result_has_new_messages(self) -> None:
        """The result.new_messages property returns a non-empty list of run messages."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))
            self.assertIsInstance(result.new_messages, list)
            self.assertGreater(len(result.new_messages), 0)

    async def test_run_rejects_wrong_input_type(self) -> None:
        """Running with a different BaseModel type raises InvalidInputType."""

        class OtherInput(BaseModel):
            name: str

        with self.agent.inner.override(model=TestModel()):
            with self.assertRaises(InvalidInputType):
                await self.agent.run(OtherInput(name="wrong"))

    async def test_run_rejects_non_basemodel_input(self) -> None:
        """Running with a plain string instead of BaseModel raises InvalidInputType."""
        with self.agent.inner.override(model=TestModel()):
            with self.assertRaises(InvalidInputType):
                await self.agent.run("not a BaseModel")

    async def test_run_serializes_input_to_json(self) -> None:
        """Input BaseModel is serialized to JSON and sent as the user prompt."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                SampleInput(question="test", context="ctx")
            )
            # Find the first ModelRequest with a UserPromptPart
            first_request = result.messages[0]
            self.assertIsInstance(first_request, ModelRequest)
            # The user prompt part should contain the serialized JSON
            user_parts = [
                p for p in first_request.parts
                if hasattr(p, "content") and "question" in str(p.content)
            ]
            self.assertGreater(len(user_parts), 0)
            user_content = user_parts[0].content
            self.assertIn('"question"', user_content)
            self.assertIn('"test"', user_content)
            self.assertIn('"context"', user_content)
            self.assertIn('"ctx"', user_content)

    async def test_run_passes_kwargs_to_inner_agent(self) -> None:
        """Extra kwargs like usage_limits are forwarded to the inner agent's run method."""
        with self.agent.inner.override(model=TestModel()):
            # Passing usage_limits should not raise an error, proving kwargs forwarding
            result = await self.agent.run(
                SampleInput(question="test"),
                usage_limits=UsageLimits(request_limit=10),
            )
            self.assertIsInstance(result, QuantedResult)


if __name__ == "__main__":
    unittest.main()

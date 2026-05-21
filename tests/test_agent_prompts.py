"""Tests for QuantedAgent system prompt and instructions handling.

Validates that static string prompts, list prompts, instructions, and dynamic
system prompts via the inner agent escape hatch all work correctly. All tests
use pydantic-ai's TestModel without real LLM calls.
"""

import unittest

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent, QuantedResult
from tests.conftest import SampleInput, SampleOutput


class TestAgentPrompts(unittest.IsolatedAsyncioTestCase):
    """Tests for system prompt and instructions parameter handling."""

    async def test_static_string_system_prompt(self) -> None:
        """A static string system prompt is included in run messages."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="You are helpful",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            # The system prompt should appear in the first ModelRequest parts
            first_request = result.messages[0]
            self.assertIsInstance(first_request, ModelRequest)
            system_parts = [
                p for p in first_request.parts
                if hasattr(p, "content") and p.content == "You are helpful"
            ]
            self.assertGreater(len(system_parts), 0)

    def test_static_list_system_prompt(self) -> None:
        """A list of system prompt strings is accepted without error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt=["Prompt 1", "Prompt 2"],
        )
        # Verify both prompts are stored in the inner agent
        self.assertIn("Prompt 1", agent.inner._system_prompts)
        self.assertIn("Prompt 2", agent.inner._system_prompts)

    async def test_empty_system_prompt_default(self) -> None:
        """An agent without system_prompt defaults to empty string and runs without error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)

    async def test_instructions_parameter(self) -> None:
        """An agent with instructions runs without error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            instructions="Always respond in JSON",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)

    async def test_both_system_prompt_and_instructions(self) -> None:
        """An agent with both system_prompt and instructions runs without error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="You are a data processor",
            instructions="Always respond in JSON",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)

    async def test_dynamic_system_prompt_via_inner(self) -> None:
        """Dynamic system prompts can be registered via the inner agent escape hatch.

        Tests AGNT-05 (dynamic prompts) by using @agent.inner.system_prompt
        to register a dynamic system prompt function.
        """
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )

        @agent.inner.system_prompt
        def dynamic_prompt(ctx: RunContext) -> str:
            """Generate a dynamic system prompt.

            Args:
                ctx: The run context.

            Returns:
                A dynamic system prompt string.
            """
            return "Dynamic prompt content"

        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)
            # The dynamic prompt should appear in the messages
            first_request = result.messages[0]
            self.assertIsInstance(first_request, ModelRequest)
            prompt_contents = [
                p.content for p in first_request.parts
                if hasattr(p, "content")
            ]
            self.assertIn("Dynamic prompt content", prompt_contents)


if __name__ == "__main__":
    unittest.main()

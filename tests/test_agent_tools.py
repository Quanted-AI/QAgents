"""Tests for QuantedAgent tool registration.

Validates that tools can be passed to QuantedAgent via the constructor and
are correctly registered with the underlying pydantic-ai Agent. All tests
use pydantic-ai's TestModel without real LLM calls.
"""

import unittest

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from tests.conftest import SampleInput, SampleOutput


def search_tool(ctx: RunContext, query: str) -> str:
    """Search for information.

    Args:
        ctx: The run context.
        query: The search query.

    Returns:
        Search results as a string.
    """
    return f"Result for: {query}"


def calculator_tool(ctx: RunContext, expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        ctx: The run context.
        expression: The math expression to evaluate.

    Returns:
        The evaluation result as a string.
    """
    return f"Calculated: {expression}"


class TestAgentTools(unittest.IsolatedAsyncioTestCase):
    """Tests for tool registration via the QuantedAgent constructor."""

    def test_agent_accepts_tools_in_constructor(self) -> None:
        """Creating an agent with a tool list does not raise an error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            tools=[search_tool],
        )
        self.assertIsNotNone(agent)

    async def test_tool_is_callable_by_agent(self) -> None:
        """An agent created with a tool can be run successfully with TestModel."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            tools=[search_tool],
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="find something"))
            self.assertIsInstance(result.data, SampleOutput)

    def test_agent_with_multiple_tools(self) -> None:
        """Creating an agent with multiple tools does not raise an error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            tools=[search_tool, calculator_tool],
        )
        # Verify both tools are registered on the inner agent
        tool_names = list(agent.inner._function_toolset.tools.keys())
        self.assertIn("search_tool", tool_names)
        self.assertIn("calculator_tool", tool_names)

    def test_agent_with_no_tools(self) -> None:
        """Creating an agent with no tools (default) does not raise an error."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()

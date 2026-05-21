"""Tests for QuantedAgent construction and validation.

Validates that QuantedAgent enforces Pydantic BaseModel types at construction
time, correctly wires parameters to the underlying pydantic-ai Agent, and
rejects invalid type arguments with clear exceptions.
"""

import unittest

from pydantic import BaseModel
from pydantic_ai import Agent

from quanted_agents import InvalidInputType, InvalidOutputType, QuantedAgent
from tests.conftest import SampleInput, SampleOutput


class TestAgentCreation(unittest.TestCase):
    """Tests for QuantedAgent constructor validation and parameter wiring."""

    def test_creates_agent_with_valid_basemodel_types(self) -> None:
        """Agent construction succeeds with valid BaseModel input and output types."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        self.assertIs(agent.input_type, SampleInput)
        self.assertIs(agent.output_type, SampleOutput)

    def test_rejects_non_basemodel_input_type(self) -> None:
        """Agent construction raises InvalidInputType when input_type is str."""
        with self.assertRaises(InvalidInputType):
            QuantedAgent(
                "test",
                input_type=str,
                output_type=SampleOutput,
            )

    def test_rejects_non_basemodel_output_type(self) -> None:
        """Agent construction raises InvalidOutputType when output_type is dict."""
        with self.assertRaises(InvalidOutputType):
            QuantedAgent(
                "test",
                input_type=SampleInput,
                output_type=dict,
            )

    def test_rejects_none_input_type(self) -> None:
        """Agent construction raises InvalidInputType when input_type is None."""
        with self.assertRaises(InvalidInputType):
            QuantedAgent(
                "test",
                input_type=None,
                output_type=SampleOutput,
            )

    def test_rejects_none_output_type(self) -> None:
        """Agent construction raises InvalidOutputType when output_type is None."""
        with self.assertRaises(InvalidOutputType):
            QuantedAgent(
                "test",
                input_type=SampleInput,
                output_type=None,
            )

    def test_rejects_basemodel_instance_not_class(self) -> None:
        """Agent construction raises InvalidInputType when given an instance instead of a class."""
        with self.assertRaises(InvalidInputType):
            QuantedAgent(
                "test",
                input_type=SampleInput(question="x"),
                output_type=SampleOutput,
            )

    def test_default_retries_is_one(self) -> None:
        """Agent created without retries param defaults to 1 retry."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        self.assertEqual(agent.inner._max_result_retries, 1)

    def test_custom_retries(self) -> None:
        """Agent created with retries=5 passes the value through to the inner agent."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            retries=5,
        )
        self.assertEqual(agent.inner._max_result_retries, 5)

    def test_inner_property_exposes_pydantic_ai_agent(self) -> None:
        """The inner property returns a pydantic-ai Agent instance."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        self.assertIsInstance(agent.inner, Agent)

    def test_system_prompt_passed_to_inner_agent(self) -> None:
        """System prompt string is forwarded to the inner pydantic-ai Agent."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Custom prompt",
        )
        # pydantic-ai v1 stores static system prompts in _system_prompts tuple
        self.assertIn("Custom prompt", agent.inner._system_prompts)

    def test_instructions_passed_to_inner_agent(self) -> None:
        """Instructions parameter is forwarded to the inner pydantic-ai Agent."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            instructions="Custom instructions",
        )
        # pydantic-ai v1 stores instructions as a list internally
        self.assertIn("Custom instructions", agent.inner._instructions)

    def test_creation_under_ten_lines(self) -> None:
        """QuantedAgent can be fully created in under 10 lines including imports and models.

        This is a documentation test proving the AGNT-01 requirement:
        complete agent creation in under 10 lines of code.

        Lines:
            1. from pydantic import BaseModel
            2. from quanted_agents import QuantedAgent
            3. class Query(BaseModel):
            4.     question: str
            5. class Answer(BaseModel):
            6.     response: str
            7. agent = QuantedAgent("test", input_type=Query, output_type=Answer)
        Total: 7 lines
        """

        class Query(BaseModel):
            question: str

        class Answer(BaseModel):
            response: str

        agent = QuantedAgent("test", input_type=Query, output_type=Answer)
        self.assertIsNotNone(agent)
        self.assertIs(agent.input_type, Query)
        self.assertIs(agent.output_type, Answer)


if __name__ == "__main__":
    unittest.main()

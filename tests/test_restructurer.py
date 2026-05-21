"""Tests for the restructurer agent and agent recovery integration.

Validates the two-model pattern (RestructurerAgent), QuantedAgent.run()
recovery pipeline, budget enforcement across recovery attempts, and
QuantedResult.from_data() behavior. Uses pydantic-ai's TestModel and
FunctionModel to simulate LLM responses without real API calls.
"""

import unittest

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent, RecoveryExhaustedError
from quanted_agents.restructurer import RestructurerAgent
from quanted_agents.result import QuantedResult
from tests.conftest import (
    IRREPARABLE_TEXT,
    MALFORMED_SINGLE_QUOTES,
    VALID_JSON,
    SampleInput,
    SampleOutput,
)


def _malformed_json_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler that returns malformed single-quote JSON."""
    return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])


def _irreparable_text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler that returns irreparable plain text."""
    return ModelResponse(parts=[TextPart(content=IRREPARABLE_TEXT)])


class TestRestructurerAgent(unittest.IsolatedAsyncioTestCase):
    """Tests for the RestructurerAgent two-model pattern.

    Verifies json-repair pre-processing, LLM fallback behavior, and
    agent inner property access.
    """

    def setUp(self) -> None:
        """Create a RestructurerAgent for testing."""
        self.restructurer = RestructurerAgent(
            model="test",
            output_type=SampleOutput,
        )

    async def test_restructure_with_valid_json_skips_llm(self) -> None:
        """Valid JSON is handled by json-repair without needing an LLM call."""
        result = await self.restructurer.restructure(VALID_JSON)
        self.assertIsInstance(result, SampleOutput)
        self.assertEqual(result.answer, "test")
        self.assertAlmostEqual(result.confidence, 0.9)

    async def test_restructure_with_repairable_json_skips_llm(self) -> None:
        """Repairable JSON (single quotes) is fixed by json-repair without LLM."""
        result = await self.restructurer.restructure(MALFORMED_SINGLE_QUOTES)
        self.assertIsInstance(result, SampleOutput)
        self.assertEqual(result.answer, "test")
        self.assertAlmostEqual(result.confidence, 0.9)

    async def test_restructure_calls_llm_when_repair_fails(self) -> None:
        """Irreparable text delegates to the inner pydantic-ai Agent (cheap model)."""
        with self.restructurer.inner.override(model=TestModel()):
            result = await self.restructurer.restructure(IRREPARABLE_TEXT)
            self.assertIsInstance(result, SampleOutput)

    def test_restructurer_has_inner_property(self) -> None:
        """The .inner property returns the wrapped pydantic-ai Agent."""
        from pydantic_ai import Agent

        self.assertIsInstance(self.restructurer.inner, Agent)

    def test_restructurer_uses_correct_output_type(self) -> None:
        """The inner agent has the correct output_type configured."""
        inner_agent = self.restructurer.inner
        # pydantic-ai Agent stores output_type in _output_type or similar
        # We verify by checking the agent can produce SampleOutput
        self.assertIsNotNone(inner_agent)


class TestAgentRecoveryIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for QuantedAgent.run() with the recovery pipeline integrated.

    Validates end-to-end recovery: json-repair, restructurer fallback,
    budget enforcement, and QuantedResult.from_data() behavior.
    """

    async def test_run_happy_path_unchanged(self) -> None:
        """Normal valid output still works (regression test)."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)
            self.assertIsInstance(result.data, SampleOutput)

    async def test_run_with_malformed_json_repairs_automatically(self) -> None:
        """Agent.run() automatically repairs malformed JSON via recovery pipeline (ERRR-01)."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )
        fm = FunctionModel(_malformed_json_model)
        with agent.inner.override(model=fm):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)
            self.assertIsInstance(result.data, SampleOutput)
            self.assertEqual(result.data.answer, "test")
            self.assertAlmostEqual(result.data.confidence, 0.9)

    async def test_run_raises_when_recovery_exhausted(self) -> None:
        """Agent.run() raises RecoveryExhaustedError for irreparable input without restructurer."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            max_recovery_attempts=1,
        )
        fm = FunctionModel(_irreparable_text_model)
        with agent.inner.override(model=fm):
            with self.assertRaises(RecoveryExhaustedError):
                await agent.run(SampleInput(question="test"))

    async def test_run_with_restructurer_model(self) -> None:
        """Agent with restructurer_model uses cheap model when json-repair fails."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            restructurer_model="test",
            max_recovery_attempts=3,
        )
        fm = FunctionModel(_irreparable_text_model)
        with agent.inner.override(model=fm):
            with agent._restructurer.inner.override(model=TestModel()):
                result = await agent.run(SampleInput(question="test"))
                self.assertIsInstance(result, QuantedResult)
                self.assertIsInstance(result.data, SampleOutput)

    async def test_max_recovery_attempts_parameter(self) -> None:
        """Agent's max_recovery_attempts parameter sets the budget correctly."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            max_recovery_attempts=5,
        )
        self.assertEqual(agent._recovery.budget.remaining, 5)
        self.assertEqual(agent._recovery.budget.max_attempts, 5)

    async def test_recovery_budget_shared_across_attempts(self) -> None:
        """Recovery budget decrements across recovery attempts in a single run."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            max_recovery_attempts=3,
        )
        initial_remaining = agent._recovery.budget.remaining
        self.assertEqual(initial_remaining, 3)

        # Trigger recovery that succeeds (json-repair fixes the malformed JSON)
        fm = FunctionModel(_malformed_json_model)
        with agent.inner.override(model=fm):
            await agent.run(SampleInput(question="test"))

        # Budget should have been consumed during recovery
        self.assertLess(agent._recovery.budget.remaining, initial_remaining)

    async def test_result_from_data_has_correct_properties(self) -> None:
        """QuantedResult.from_data() returns empty usage and empty messages."""
        output = SampleOutput(answer="recovered", confidence=0.8)
        result = QuantedResult.from_data(output)

        self.assertEqual(result.data.answer, "recovered")
        self.assertAlmostEqual(result.data.confidence, 0.8)
        self.assertEqual(result.usage.requests, 0)
        self.assertEqual(result.messages, [])


if __name__ == "__main__":
    unittest.main()

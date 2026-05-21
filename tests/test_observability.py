"""Tests for observability data models and trace capture.

Validates StepTiming and TraceEntry dataclasses, helper functions for
extracting tool calls and model names from messages, trace capture in
QuantedAgent.run() on both happy and recovery paths, and QuantedResult
observability properties. All tests use pydantic-ai's TestModel and
FunctionModel without real LLM API calls.
"""

import json
import unittest
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from quanted_agents import QuantedAgent
from quanted_agents.observability import (
    StepTiming,
    TraceEntry,
    extract_model_name,
    extract_tool_calls,
)
from quanted_agents.result import QuantedResult
from tests.conftest import MALFORMED_SINGLE_QUOTES, SampleInput, SampleOutput


class TestStepTiming(unittest.TestCase):
    """Tests for the StepTiming dataclass."""

    def test_step_timing_creation(self) -> None:
        """StepTiming can be created with known values and attributes are correct."""
        usage = RunUsage(requests=2, input_tokens=150, output_tokens=75)
        timing = StepTiming(step_name="test_step", duration_seconds=1.23, usage=usage)

        self.assertEqual(timing.step_name, "test_step")
        self.assertAlmostEqual(timing.duration_seconds, 1.23)
        self.assertEqual(timing.usage.requests, 2)
        self.assertEqual(timing.usage.input_tokens, 150)
        self.assertEqual(timing.usage.output_tokens, 75)

    def test_trace_entry_creation(self) -> None:
        """TraceEntry can be created with all fields and attributes are correct."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        timing = StepTiming(step_name="agent", duration_seconds=0.5, usage=usage)
        entry = TraceEntry(
            step_name="QuantedAgent(SampleOutput)",
            input_data={"question": "test"},
            output_data={"answer": "hello", "confidence": 0.9},
            messages=[{"kind": "request"}],
            tool_calls=[],
            timing=timing,
            model_name="test-model",
            recovery_info=None,
        )

        self.assertEqual(entry.step_name, "QuantedAgent(SampleOutput)")
        self.assertEqual(entry.input_data, {"question": "test"})
        self.assertEqual(entry.output_data["answer"], "hello")
        self.assertEqual(entry.messages, [{"kind": "request"}])
        self.assertEqual(entry.tool_calls, [])
        self.assertEqual(entry.timing.step_name, "agent")
        self.assertEqual(entry.model_name, "test-model")
        self.assertIsNone(entry.recovery_info)

    def test_trace_entry_to_dict_structure(self) -> None:
        """TraceEntry.to_dict() returns a dict with all expected keys."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        timing = StepTiming(step_name="agent", duration_seconds=0.5, usage=usage)
        entry = TraceEntry(
            step_name="test",
            input_data={"q": "hi"},
            output_data={"a": "hello"},
            messages=[],
            tool_calls=[{"tool_name": "search", "args": {}, "tool_call_id": "1"}],
            timing=timing,
            model_name="openai:gpt-4o",
            recovery_info=None,
        )
        d = entry.to_dict()

        expected_keys = {
            "step_name", "input_data", "output_data", "messages",
            "tool_calls", "timing", "model_name", "recovery_info",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertIn("step_name", d["timing"])
        self.assertIn("duration_seconds", d["timing"])
        self.assertIn("usage", d["timing"])
        self.assertIn("input_tokens", d["timing"]["usage"])
        self.assertIn("output_tokens", d["timing"]["usage"])
        self.assertIn("requests", d["timing"]["usage"])

    def test_extract_tool_calls_from_messages(self) -> None:
        """extract_tool_calls finds ToolCallPart instances in ModelResponse messages."""
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_tool",
                        args={"query": "test"},
                        tool_call_id="tc_001",
                    ),
                ]
            ),
        ]
        result = extract_tool_calls(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool_name"], "search_tool")
        self.assertEqual(result[0]["args"], {"query": "test"})
        self.assertEqual(result[0]["tool_call_id"], "tc_001")

    def test_extract_tool_calls_with_string_args(self) -> None:
        """extract_tool_calls parses JSON string args into dicts."""
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="calc",
                        args='{"expression": "2+2"}',
                        tool_call_id="tc_002",
                    ),
                ]
            ),
        ]
        result = extract_tool_calls(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["args"], {"expression": "2+2"})

    def test_extract_model_name_from_messages(self) -> None:
        """extract_model_name returns model_name from the first ModelResponse."""
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[TextPart(content="Hello")],
                model_name="openai:gpt-4o",
            ),
        ]
        name = extract_model_name(messages)

        self.assertEqual(name, "openai:gpt-4o")

    def test_extract_model_name_returns_none_when_no_response(self) -> None:
        """extract_model_name returns None when no ModelResponse exists."""
        messages: list[ModelMessage] = []
        name = extract_model_name(messages)

        self.assertIsNone(name)


class TestTraceCapture(unittest.IsolatedAsyncioTestCase):
    """Tests for trace capture in QuantedAgent.run()."""

    def setUp(self) -> None:
        """Create a standard test agent."""
        self.agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )

    async def test_single_agent_run_has_trace(self) -> None:
        """Running an agent produces a trace with one TraceEntry."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="test"))

        self.assertIsInstance(result.trace, list)
        self.assertEqual(len(result.trace), 1)

        entry = result.trace[0]
        self.assertIsInstance(entry, TraceEntry)
        self.assertEqual(entry.step_name, "QuantedAgent(SampleOutput)")
        self.assertEqual(entry.input_data, {"question": "test", "context": ""})
        self.assertIsInstance(entry.output_data, dict)
        self.assertGreater(entry.timing.duration_seconds, 0)
        self.assertGreater(entry.timing.usage.requests, 0)

    async def test_trace_entry_to_dict_is_json_serializable(self) -> None:
        """TraceEntry.to_dict() from a real run produces JSON-serializable output."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="serialize me"))

        entry = result.trace[0]
        d = entry.to_dict()
        serialized = json.dumps(d)
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 0)

    async def test_trace_captures_model_name(self) -> None:
        """Trace entry captures the model name from the response."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="model name test"))

        entry = result.trace[0]
        # TestModel sets model_name on responses
        self.assertIsNotNone(entry.model_name)

    async def test_trace_tool_calls_captured(self) -> None:
        """Trace captures tool calls when an agent uses tools."""

        def my_tool(ctx: Any, query: str) -> str:
            """Search for information.

            Args:
                ctx: The run context.
                query: The search query.

            Returns:
                Search result string.
            """
            return f"Result: {query}"

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Use the tool.",
            tools=[my_tool],
        )

        # TestModel calls all registered tools by default
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="search test"))

        entry = result.trace[0]
        self.assertIsInstance(entry.tool_calls, list)
        # TestModel calls all tools, so we should have at least one
        self.assertGreater(len(entry.tool_calls), 0)
        tool_names = [tc["tool_name"] for tc in entry.tool_calls]
        self.assertIn("my_tool", tool_names)

    async def test_trace_recovery_info_populated(self) -> None:
        """Trace entry has recovery_info populated when recovery activates."""

        def _malformed_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return malformed JSON to trigger recovery."""
            return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            max_recovery_attempts=3,
        )
        fm = FunctionModel(_malformed_model)
        with agent.inner.override(model=fm):
            result = await agent.run(SampleInput(question="malformed"))

        self.assertEqual(len(result.trace), 1)
        entry = result.trace[0]
        self.assertIsNotNone(entry.recovery_info)
        self.assertTrue(entry.recovery_info["json_repair_attempted"])


class TestResultObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for QuantedResult observability properties."""

    def setUp(self) -> None:
        """Create a standard test agent."""
        self.agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )

    async def test_total_usage_single_agent(self) -> None:
        """total_usage for a single agent run equals usage (fallback)."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="usage test"))

        self.assertEqual(result.total_usage.requests, result.usage.requests)
        self.assertEqual(result.total_usage.input_tokens, result.usage.input_tokens)
        self.assertEqual(result.total_usage.output_tokens, result.usage.output_tokens)

    async def test_step_timings_single_agent(self) -> None:
        """step_timings for a single agent run has one entry with positive duration."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="timing test"))

        self.assertEqual(len(result.step_timings), 1)
        self.assertGreater(result.step_timings[0].duration_seconds, 0)


if __name__ == "__main__":
    unittest.main()

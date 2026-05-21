"""Tests for run_stream() recovery pipeline and was_recovered/recovery_method flags.

Validates that run_stream() activates the recovery pipeline on malformed output,
that QuantedResult includes was_recovered and recovery_method properties, and that
normal runs do not set recovery flags.
"""

import json
import unittest
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent, QuantedResult
from tests.conftest import MALFORMED_SINGLE_QUOTES, SampleInput, SampleOutput


async def _malformed_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    """Stream function that yields malformed text to trigger UnexpectedModelBehavior."""
    yield MALFORMED_SINGLE_QUOTES


async def _irreparable_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    """Stream function that yields irreparable text."""
    yield "totally broken not json at all xyz"


class TestStreamRecovery(unittest.IsolatedAsyncioTestCase):
    """Tests for run_stream() recovery and was_recovered/recovery_method flags."""

    async def test_stream_yields_partials_on_success(self) -> None:
        """run_stream with valid output yields partials, no QuantedResult."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )
        partials: list[Any] = []
        with agent.inner.override(model=TestModel()):
            async for partial in agent.run_stream(SampleInput(question="hello")):
                partials.append(partial)

        self.assertTrue(len(partials) > 0)
        # None of the partials should be a QuantedResult (no recovery)
        for p in partials:
            self.assertNotIsInstance(p, QuantedResult)

    async def test_stream_recovery_json_repair(self) -> None:
        """run_stream with malformed JSON triggers recovery, yields QuantedResult."""

        def _malformed_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return malformed JSON to trigger recovery (non-stream fallback)."""
            return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
            max_recovery_attempts=3,
        )

        items: list[Any] = []
        fm = FunctionModel(_malformed_model, stream_function=_malformed_stream)
        with agent.inner.override(model=fm):
            async for item in agent.run_stream(SampleInput(question="recover")):
                items.append(item)

        # Last item should be a QuantedResult from recovery
        self.assertTrue(len(items) > 0)
        last_item = items[-1]
        self.assertIsInstance(last_item, QuantedResult)
        self.assertTrue(last_item.was_recovered)
        self.assertEqual(last_item.recovery_method, "json_repair")

    async def test_stream_recovery_restructurer(self) -> None:
        """run_stream with irreparable JSON falls back to restructurer."""

        def _irreparable_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return text that json-repair cannot fix."""
            return ModelResponse(
                parts=[TextPart(content="totally broken not json at all xyz")]
            )

        def _restructurer_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Restructurer that returns valid structured output."""
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name=tool.name,
                    args=json.dumps({"answer": "recovered", "confidence": 0.5}),
                )]
            )

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
            max_recovery_attempts=3,
            restructurer_model="test",
        )

        items: list[Any] = []
        fm = FunctionModel(_irreparable_model, stream_function=_irreparable_stream)
        restructurer_fm = FunctionModel(_restructurer_model)
        with agent.inner.override(model=fm), \
                agent._restructurer._agent.override(model=restructurer_fm):
            async for item in agent.run_stream(SampleInput(question="broken")):
                items.append(item)

        last_item = items[-1]
        self.assertIsInstance(last_item, QuantedResult)
        self.assertTrue(last_item.was_recovered)
        self.assertEqual(last_item.recovery_method, "restructurer")
        self.assertEqual(last_item.data.answer, "recovered")

    async def test_stream_no_recovery_no_flag(self) -> None:
        """Normal run() result has was_recovered=False and recovery_method=None."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="hello"))

        self.assertFalse(result.was_recovered)
        self.assertIsNone(result.recovery_method)

    async def test_run_recovery_sets_flags(self) -> None:
        """run() with recovery sets was_recovered=True on result."""

        def _malformed_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return malformed JSON to trigger recovery."""
            return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
            max_recovery_attempts=3,
        )

        fm = FunctionModel(_malformed_model)
        with agent.inner.override(model=fm):
            result = await agent.run(SampleInput(question="recover"))

        self.assertTrue(result.was_recovered)
        self.assertEqual(result.recovery_method, "json_repair")
        self.assertEqual(result.data.answer, "test")

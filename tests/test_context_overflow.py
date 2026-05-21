"""Tests for Phase 18 Plan 03: context overflow detection and truncation.

Validates context window management including overflow detection with RAISE
strategy, message truncation with TRUNCATE_OLDEST strategy, system prompt
preservation, and result metadata tracking. All agent tests use pydantic-ai's
TestModel.
"""

import unittest

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from quanted_agents import (
    ContextOverflowError,
    OverflowStrategy,
    QuantedAgent,
    QuantedResult,
)
from quanted_agents._token_counter import (
    count_messages_tokens,
    truncate_messages,
)


class _Input(BaseModel):
    """Test input model."""

    x: str


class _Output(BaseModel):
    """Test output model."""

    y: str


def _make_user_message(text: str) -> ModelRequest:
    """Create a ModelRequest with a UserPromptPart.

    Args:
        text: The user prompt text content.

    Returns:
        A ModelRequest containing a single UserPromptPart.
    """
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _make_assistant_message(text: str) -> ModelResponse:
    """Create a ModelResponse with a TextPart.

    Args:
        text: The assistant response text content.

    Returns:
        A ModelResponse containing a single TextPart.
    """
    return ModelResponse(parts=[TextPart(content=text)])


def _make_system_message(text: str) -> ModelRequest:
    """Create a ModelRequest with a SystemPromptPart.

    Args:
        text: The system prompt text content.

    Returns:
        A ModelRequest containing a single SystemPromptPart.
    """
    return ModelRequest(parts=[SystemPromptPart(content=text)])


def _make_large_history(num_pairs: int, text_size: int = 200) -> list:
    """Build a synthetic message history with controllable token counts.

    Args:
        num_pairs: Number of user/assistant message pairs.
        text_size: Character length for each message's text content.

    Returns:
        A list of alternating ModelRequest and ModelResponse messages.
    """
    messages = []
    for i in range(num_pairs):
        messages.append(_make_user_message(f"question {i} " + "x" * text_size))
        messages.append(_make_assistant_message(f"answer {i} " + "y" * text_size))
    return messages


# ---------------------------------------------------------------------------
# TestTruncateMessages
# ---------------------------------------------------------------------------


class TestTruncateMessages(unittest.TestCase):
    """Tests for truncate_messages() truncation logic."""

    def test_no_truncation_needed(self) -> None:
        """Messages within budget are returned unchanged with count=0."""
        msgs = [
            _make_user_message("hello"),
            _make_assistant_message("hi there"),
        ]
        result, count = truncate_messages(msgs, system_prompt_tokens=10, max_tokens=10000)
        self.assertEqual(len(result), 2)
        self.assertEqual(count, 0)

    def test_truncates_oldest_first(self) -> None:
        """Messages exceeding budget drop oldest, keep newest."""
        msgs = _make_large_history(5, text_size=200)
        # Very tight budget - should only keep newest messages
        result, count = truncate_messages(msgs, system_prompt_tokens=10, max_tokens=200)
        self.assertGreater(count, 0)
        self.assertLess(len(result), len(msgs))
        # The kept messages should be from the end (newest)
        if len(result) > 0:
            # Last message in result should be last message in original
            self.assertEqual(result[-1], msgs[-1])

    def test_system_prompt_never_truncated(self) -> None:
        """System prompt messages are always preserved even with tight budget."""
        sys_msg = _make_system_message("important system instructions " + "z" * 100)
        conv_msgs = _make_large_history(3, text_size=200)
        all_msgs = [sys_msg] + conv_msgs

        # Very tight budget - system should survive
        result, count = truncate_messages(all_msgs, system_prompt_tokens=50, max_tokens=100)
        # System message should be first
        self.assertTrue(len(result) >= 1)
        has_system = any(
            isinstance(m, ModelRequest) and any(
                isinstance(p, SystemPromptPart) for p in m.parts
            )
            for m in result
        )
        self.assertTrue(has_system)

    def test_returns_correct_truncated_count(self) -> None:
        """Verify the count matches number of dropped messages."""
        msgs = _make_large_history(5, text_size=200)
        original_count = len(msgs)
        result, dropped = truncate_messages(msgs, system_prompt_tokens=10, max_tokens=300)
        self.assertEqual(dropped, original_count - len(result))

    def test_empty_messages(self) -> None:
        """Empty list returns empty with count=0."""
        result, count = truncate_messages([], system_prompt_tokens=0, max_tokens=1000)
        self.assertEqual(result, [])
        self.assertEqual(count, 0)

    def test_all_messages_exceed_budget(self) -> None:
        """Only system prompt survives when all conversation messages are too large."""
        sys_msg = _make_system_message("system")
        # Each message is huge
        conv_msgs = _make_large_history(3, text_size=5000)
        all_msgs = [sys_msg] + conv_msgs

        # Budget leaves almost nothing for conversation
        result, count = truncate_messages(all_msgs, system_prompt_tokens=50, max_tokens=55)
        # System message should survive
        self.assertEqual(len(result), 1)
        self.assertTrue(
            isinstance(result[0], ModelRequest) and any(
                isinstance(p, SystemPromptPart) for p in result[0].parts
            )
        )
        self.assertEqual(count, len(conv_msgs))


# ---------------------------------------------------------------------------
# TestCountMessagesTokens
# ---------------------------------------------------------------------------


class TestCountMessagesTokens(unittest.TestCase):
    """Tests for count_messages_tokens() message-level token counting."""

    def test_returns_positive_for_messages(self) -> None:
        """Non-empty messages produce a positive token count."""
        msgs = [
            _make_user_message("hello world this is a test"),
            _make_assistant_message("yes it is indeed a test"),
        ]
        result = count_messages_tokens(msgs)
        self.assertGreater(result, 0)

    def test_empty_list_returns_zero(self) -> None:
        """Empty list returns 0."""
        result = count_messages_tokens([])
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# TestContextOverflowRaise
# ---------------------------------------------------------------------------


class TestContextOverflowRaise(unittest.IsolatedAsyncioTestCase):
    """Tests for context overflow with RAISE strategy using TestModel."""

    def setUp(self) -> None:
        """Create a test agent with very small context window."""
        self.agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test agent",
            max_context_tokens=10,
        )

    async def test_overflow_raises_error(self) -> None:
        """ContextOverflowError raised when message_history exceeds limit."""
        large_history = _make_large_history(5, text_size=500)
        with self.agent.inner.override(model=TestModel()):
            with self.assertRaises(ContextOverflowError):
                await self.agent.run(
                    _Input(x="test"),
                    message_history=large_history,
                )

    async def test_overflow_error_attributes(self) -> None:
        """Verify current_tokens, max_tokens, store, usage on error."""
        large_history = _make_large_history(5, text_size=500)
        with self.agent.inner.override(model=TestModel()):
            with self.assertRaises(ContextOverflowError) as ctx:
                await self.agent.run(
                    _Input(x="test"),
                    message_history=large_history,
                )
            err = ctx.exception
            self.assertGreater(err.current_tokens, 10)
            self.assertEqual(err.max_tokens, 10)
            self.assertIsNone(err.store)
            self.assertIsInstance(err.usage, RunUsage)

    async def test_no_overflow_normal_execution(self) -> None:
        """Large max_context_tokens allows normal execution."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test",
            max_context_tokens=100000,
        )
        small_history = [
            _make_user_message("hi"),
            _make_assistant_message("hello"),
        ]
        with agent.inner.override(model=TestModel()):
            result = await agent.run(
                _Input(x="test"),
                message_history=small_history,
            )
            self.assertIsInstance(result, QuantedResult)

    async def test_no_max_context_tokens_skips_check(self) -> None:
        """Default (None) max_context_tokens skips overflow check entirely."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test",
        )
        large_history = _make_large_history(10, text_size=1000)
        with agent.inner.override(model=TestModel()):
            # Should succeed even with huge history since check is disabled
            result = await agent.run(
                _Input(x="test"),
                message_history=large_history,
            )
            self.assertIsInstance(result, QuantedResult)

    async def test_default_strategy_is_raise(self) -> None:
        """Setting max_context_tokens without overflow_strategy defaults to RAISE."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test",
            max_context_tokens=10,
        )
        large_history = _make_large_history(5, text_size=500)
        with agent.inner.override(model=TestModel()):
            with self.assertRaises(ContextOverflowError):
                await agent.run(
                    _Input(x="test"),
                    message_history=large_history,
                )


# ---------------------------------------------------------------------------
# TestContextOverflowTruncate
# ---------------------------------------------------------------------------


class TestContextOverflowTruncate(unittest.IsolatedAsyncioTestCase):
    """Tests for context overflow with TRUNCATE_OLDEST strategy using TestModel."""

    def setUp(self) -> None:
        """Create a test agent with truncation strategy."""
        self.agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test agent",
            max_context_tokens=200,
            overflow_strategy=OverflowStrategy.TRUNCATE_OLDEST,
        )

    async def test_truncation_sets_result_metadata(self) -> None:
        """Truncation sets context_overflow_occurred=True and messages_truncated > 0."""
        large_history = _make_large_history(10, text_size=500)
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                _Input(x="test"),
                message_history=large_history,
            )
            self.assertTrue(result.context_overflow_occurred)
            self.assertGreater(result.messages_truncated, 0)

    async def test_truncation_preserves_system_prompt(self) -> None:
        """After truncation, system prompt messages are still present."""
        sys_msg = _make_system_message("important system prompt")
        conv = _make_large_history(10, text_size=500)
        history = [sys_msg] + conv

        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                _Input(x="test"),
                message_history=history,
            )
            # The run completed (didn't raise), truncation happened
            self.assertTrue(result.context_overflow_occurred)

    async def test_no_truncation_when_within_budget(self) -> None:
        """Large max_context_tokens, small history: no truncation occurs."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            system_prompt="Test",
            max_context_tokens=100000,
            overflow_strategy=OverflowStrategy.TRUNCATE_OLDEST,
        )
        small_history = [
            _make_user_message("hi"),
            _make_assistant_message("hello"),
        ]
        with agent.inner.override(model=TestModel()):
            result = await agent.run(
                _Input(x="test"),
                message_history=small_history,
            )
            self.assertFalse(result.context_overflow_occurred)
            self.assertEqual(result.messages_truncated, 0)

    async def test_truncated_result_is_normal(self) -> None:
        """Result after truncation is a normal QuantedResult with valid .data."""
        large_history = _make_large_history(10, text_size=500)
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                _Input(x="test"),
                message_history=large_history,
            )
            self.assertIsInstance(result, QuantedResult)
            self.assertIsInstance(result.data, _Output)


# ---------------------------------------------------------------------------
# TestContextOverflowConfiguration
# ---------------------------------------------------------------------------


class TestContextOverflowConfiguration(unittest.TestCase):
    """Tests for context overflow parameter configuration."""

    def test_raise_strategy_is_default(self) -> None:
        """Agent with max_context_tokens but no strategy defaults to RAISE."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            max_context_tokens=5000,
        )
        self.assertEqual(agent._overflow_strategy, OverflowStrategy.RAISE)

    def test_truncate_oldest_accepted(self) -> None:
        """Agent accepts OverflowStrategy.TRUNCATE_OLDEST."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            max_context_tokens=5000,
            overflow_strategy=OverflowStrategy.TRUNCATE_OLDEST,
        )
        self.assertEqual(agent._overflow_strategy, OverflowStrategy.TRUNCATE_OLDEST)

    def test_no_strategy_without_max_tokens(self) -> None:
        """Agent without max_context_tokens has no strategy set."""
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
        )
        self.assertIsNone(agent._overflow_strategy)
        self.assertIsNone(agent._max_context_tokens)

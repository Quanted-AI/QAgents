"""Tests for Phase 18 Plan 02: soft limits and timeouts."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.usage import RunUsage

from quanted_agents import AgentTimeoutError, QuantedAgent, QuantedResult
from quanted_agents._execution import (
    DEFAULT_GRACE_PERIOD,
    MAX_WRAP_UP_CALLS,
    SoftLimitGuard,
    TOOL_BLOCKED_MESSAGE,
    WRAP_UP_SYSTEM_MESSAGE,
    _strip_pending_tool_calls,
    execute_wrap_up,
    resolve_timeouts,
)


class _Input(BaseModel):
    """Test input model."""

    x: str


class _Output(BaseModel):
    """Test output model."""

    y: str


# ---------------------------------------------------------------------------
# TestResolveTimeouts
# ---------------------------------------------------------------------------


class TestResolveTimeouts(unittest.TestCase):
    """Tests for resolve_timeouts() timeout resolution logic."""

    def test_no_timeouts(self) -> None:
        result = resolve_timeouts(None, None)
        self.assertEqual(result, (None, None))

    def test_soft_only(self) -> None:
        result = resolve_timeouts(10.0, None)
        self.assertEqual(result, (10.0, 10.0 + DEFAULT_GRACE_PERIOD))

    def test_hard_only(self) -> None:
        result = resolve_timeouts(None, 60.0)
        self.assertEqual(result, (None, 60.0))

    def test_both_valid(self) -> None:
        result = resolve_timeouts(10.0, 60.0)
        self.assertEqual(result, (10.0, 60.0))

    def test_hard_lte_soft_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_timeouts(30.0, 30.0)
        self.assertIn("must be greater", str(ctx.exception))

    def test_hard_less_than_soft_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_timeouts(30.0, 10.0)

    def test_soft_only_grace_period_value(self) -> None:
        soft, hard = resolve_timeouts(15.0, None)
        self.assertEqual(soft, 15.0)
        self.assertEqual(hard, 15.0 + 30.0)


# ---------------------------------------------------------------------------
# TestSoftLimitGuard
# ---------------------------------------------------------------------------


class TestSoftLimitGuard(unittest.TestCase):
    """Tests for SoftLimitGuard first-trigger-wins behavior."""

    def test_initial_state(self) -> None:
        guard = SoftLimitGuard()
        self.assertFalse(guard.is_active)
        self.assertIsNone(guard.reason)

    def test_activate_first_wins(self) -> None:
        guard = SoftLimitGuard()
        first = guard.activate("soft_limit")
        second = guard.activate("soft_timeout")
        self.assertTrue(first)
        self.assertFalse(second)

    def test_reason_preserved(self) -> None:
        guard = SoftLimitGuard()
        guard.activate("soft_limit")
        self.assertEqual(guard.reason, "soft_limit")

    def test_reason_not_overwritten(self) -> None:
        guard = SoftLimitGuard()
        guard.activate("soft_limit")
        guard.activate("soft_timeout")
        self.assertEqual(guard.reason, "soft_limit")

    def test_is_active_after_activate(self) -> None:
        guard = SoftLimitGuard()
        guard.activate("soft_timeout")
        self.assertTrue(guard.is_active)


# ---------------------------------------------------------------------------
# TestWrapUpConstants
# ---------------------------------------------------------------------------


class TestWrapUpConstants(unittest.TestCase):
    """Tests for wrap-up module constants."""

    def test_wrap_up_message_content(self) -> None:
        self.assertIn("budget", WRAP_UP_SYSTEM_MESSAGE)
        self.assertIn("final", WRAP_UP_SYSTEM_MESSAGE)

    def test_tool_blocked_message_content(self) -> None:
        self.assertIn("disabled", TOOL_BLOCKED_MESSAGE)

    def test_max_wrap_up_calls(self) -> None:
        self.assertEqual(MAX_WRAP_UP_CALLS, 2)

    def test_default_grace_period(self) -> None:
        self.assertEqual(DEFAULT_GRACE_PERIOD, 30.0)


# ---------------------------------------------------------------------------
# TestStripPendingToolCalls
# ---------------------------------------------------------------------------


class TestStripPendingToolCalls(unittest.TestCase):
    """Tests for the _strip_pending_tool_calls helper function."""

    def test_empty_list_returns_empty(self) -> None:
        messages: list = []
        result = _strip_pending_tool_calls(messages)
        self.assertEqual(result, [])

    def test_no_tool_calls_unchanged(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="world")], model_name="test"),
        ]
        result = _strip_pending_tool_calls(messages)
        self.assertEqual(len(result), 2)

    def test_strips_trailing_model_response_with_tool_calls(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[ToolCallPart(tool_name="my_tool", args="{}")], model_name="test"),
        ]
        result = _strip_pending_tool_calls(messages)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ModelRequest)

    def test_preserves_earlier_messages(self) -> None:
        msg1 = ModelRequest(parts=[UserPromptPart(content="hello")])
        msg2 = ModelResponse(parts=[TextPart(content="thinking")], model_name="test")
        msg3 = ModelRequest(parts=[UserPromptPart(content="continue")])
        msg4 = ModelResponse(
            parts=[ToolCallPart(tool_name="search", args='{"q":"test"}')], model_name="test"
        )
        result = _strip_pending_tool_calls([msg1, msg2, msg3, msg4])
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], msg1)
        self.assertIs(result[1], msg2)
        self.assertIs(result[2], msg3)

    def test_does_not_mutate_original_list(self) -> None:
        original = [
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[ToolCallPart(tool_name="t", args="{}")], model_name="test"),
        ]
        _strip_pending_tool_calls(original)
        self.assertEqual(len(original), 2)

    def test_model_request_last_unchanged(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
        result = _strip_pending_tool_calls(messages)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# TestSoftLimitIntegration (using TestModel)
# ---------------------------------------------------------------------------


class TestSoftLimitIntegration(unittest.TestCase):
    """Integration tests for soft limit behavior with TestModel."""

    def test_normal_run_no_soft_limit(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output
        )
        result = asyncio.run(
            agent.run(_Input(x="hello"))
        )
        self.assertIsNone(result.termination_reason)

    def test_soft_limit_disabled_raises(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            llm_call_limit=1,
            soft_limit=False,
        )
        # TestModel with request_limit=1 should succeed in 1 call
        # To force UsageLimitExceeded, we patch the inner agent.run
        with patch.object(
            agent._agent, "run",
            side_effect=UsageLimitExceeded("limit exceeded"),
        ):
            with self.assertRaises(UsageLimitExceeded):
                asyncio.run(
                    agent.run(_Input(x="hello"))
                )

    def test_soft_limit_catches_usage_exceeded(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_limit=True,
        )
        # Mock inner agent.run to raise UsageLimitExceeded on first call
        # then succeed on wrap-up call
        mock_result = MagicMock()
        mock_result.output = _Output(y="wrapped")
        mock_result.usage.return_value = RunUsage()

        call_count = 0

        async def mock_run(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UsageLimitExceeded("limit exceeded")
            return mock_result

        with patch.object(agent._agent, "run", side_effect=mock_run):
            result = asyncio.run(
                agent.run(_Input(x="hello"))
            )

        self.assertEqual(result.termination_reason, "soft_limit")

    def test_termination_reason_on_result(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_limit=True,
        )
        mock_result = MagicMock()
        mock_result.output = _Output(y="result")
        mock_result.usage.return_value = RunUsage()

        call_count = 0

        async def mock_run(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UsageLimitExceeded("limit exceeded")
            return mock_result

        with patch.object(agent._agent, "run", side_effect=mock_run):
            result = asyncio.run(
                agent.run(_Input(x="test"))
            )

        self.assertIn(result.termination_reason, ("soft_limit", "soft_timeout"))


# ---------------------------------------------------------------------------
# TestTimeoutIntegration
# ---------------------------------------------------------------------------


class TestTimeoutIntegration(unittest.TestCase):
    """Integration tests for timeout behavior."""

    def test_hard_timeout_raises_agent_timeout_error(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            hard_timeout=0.1,
        )

        async def slow_run(prompt, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(agent._agent, "run", side_effect=slow_run):
            with self.assertRaises(AgentTimeoutError) as ctx:
                asyncio.run(
                    agent.run(_Input(x="hello"))
                )
            self.assertEqual(ctx.exception.termination_reason, "hard_timeout")

    def test_hard_timeout_error_attributes(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            hard_timeout=0.1,
        )

        async def slow_run(prompt, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(agent._agent, "run", side_effect=slow_run):
            with self.assertRaises(AgentTimeoutError) as ctx:
                asyncio.run(
                    agent.run(_Input(x="hello"))
                )
            err = ctx.exception
            self.assertIsInstance(err.usage, RunUsage)
            self.assertEqual(err.termination_reason, "hard_timeout")
            self.assertIsInstance(err, TimeoutError)

    def test_no_timeout_normal_execution(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output
        )
        result = asyncio.run(
            agent.run(_Input(x="hello"))
        )
        self.assertIsNone(result.termination_reason)

    def test_soft_timeout_configuration(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_timeout=30.0,
        )
        self.assertEqual(agent._soft_timeout, 30.0)
        self.assertEqual(agent._effective_soft_timeout, 30.0)
        self.assertEqual(agent._effective_hard_timeout, 60.0)

    def test_hard_timeout_lte_soft_raises_at_construction(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            QuantedAgent(
                "test",
                input_type=_Input,
                output_type=_Output,
                soft_timeout=30.0,
                hard_timeout=20.0,
            )
        self.assertIn("must be greater", str(ctx.exception))

    def test_run_timeout_overrides_constructor(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_timeout=30.0,
            hard_timeout=60.0,
        )
        # Per-run override should work without error
        result = asyncio.run(
            agent.run(_Input(x="hello"), soft_timeout=10.0, hard_timeout=45.0)
        )
        self.assertIsNone(result.termination_reason)

    def test_run_timeout_override_validates(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
        )
        with self.assertRaises(ValueError):
            asyncio.run(
                agent.run(
                    _Input(x="hello"),
                    hard_timeout=5.0,
                    soft_timeout=10.0,
                )
            )

    def test_run_timeout_default_uses_constructor(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            hard_timeout=60.0,
        )
        self.assertEqual(agent._effective_hard_timeout, 60.0)
        # Run without per-run overrides should use constructor defaults
        result = asyncio.run(
            agent.run(_Input(x="hello"))
        )
        self.assertIsNone(result.termination_reason)


# ---------------------------------------------------------------------------
# TestTotalRequestLimitEnforcement
# ---------------------------------------------------------------------------


class TestTotalRequestLimitEnforcement(unittest.TestCase):
    """Tests for SDK-level total_request_limit enforcement."""

    def test_total_request_limit_soft_triggers_wrap_up(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            total_request_limit=1,
            soft_limit=True,
        )
        # TestModel produces 1 request with 0 tool calls = 1 total
        # 1 >= 1 triggers soft limit
        mock_result = MagicMock()
        mock_result.output = _Output(y="normal")
        usage = RunUsage()
        usage.requests = 1
        usage.tool_calls = 0
        mock_result.usage.return_value = usage

        # Wrap-up mock
        wrap_result = MagicMock()
        wrap_result.output = _Output(y="wrapped")
        wrap_result.usage.return_value = RunUsage()

        call_count = 0

        async def mock_run(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_result
            return wrap_result

        with patch.object(agent._agent, "run", side_effect=mock_run):
            result = asyncio.run(
                agent.run(_Input(x="test"))
            )

        self.assertEqual(result.termination_reason, "soft_limit")

    def test_total_request_limit_hard_raises(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            total_request_limit=1,
            soft_limit=False,
        )
        mock_result = MagicMock()
        mock_result.output = _Output(y="normal")
        usage = RunUsage()
        usage.requests = 1
        usage.tool_calls = 0
        mock_result.usage.return_value = usage

        async def mock_run(prompt, **kwargs):
            return mock_result

        with patch.object(agent._agent, "run", side_effect=mock_run):
            with self.assertRaises(UsageLimitExceeded) as ctx:
                asyncio.run(
                    agent.run(_Input(x="test"))
                )
            self.assertIn("Total request limit", str(ctx.exception))

    def test_total_request_limit_not_exceeded_normal(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            total_request_limit=100,
        )
        result = asyncio.run(
            agent.run(_Input(x="test"))
        )
        self.assertIsNone(result.termination_reason)

    def test_total_request_limit_none_skips_check(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            total_request_limit=None,
        )
        result = asyncio.run(
            agent.run(_Input(x="test"))
        )
        self.assertIsNone(result.termination_reason)


# ---------------------------------------------------------------------------
# TestExecuteWrapUp
# ---------------------------------------------------------------------------


class TestExecuteWrapUp(unittest.TestCase):
    """Tests for the execute_wrap_up function."""

    def test_wrap_up_success_returns_result(self) -> None:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = _Output(y="wrapped")
        mock_result.usage.return_value = RunUsage()

        async def mock_run(prompt, **kwargs):
            return mock_result

        mock_agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(
            execute_wrap_up(mock_agent, [], RunUsage(), "soft_limit")
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.termination_reason, "soft_limit")

    def test_wrap_up_exhausted_returns_none(self) -> None:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            side_effect=UsageLimitExceeded("exhausted")
        )

        result = asyncio.run(
            execute_wrap_up(mock_agent, [], RunUsage(), "soft_limit")
        )

        self.assertIsNone(result)

    def test_wrap_up_unexpected_behavior_returns_none(self) -> None:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            side_effect=UnexpectedModelBehavior("bad output")
        )

        result = asyncio.run(
            execute_wrap_up(mock_agent, [], RunUsage(), "soft_timeout")
        )

        self.assertIsNone(result)

    def test_wrap_up_sets_termination_reason(self) -> None:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = _Output(y="done")
        mock_result.usage.return_value = RunUsage()
        mock_agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(
            execute_wrap_up(mock_agent, [], RunUsage(), "soft_timeout")
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.termination_reason, "soft_timeout")

    def test_wrap_up_strips_pending_tool_calls_before_run(self) -> None:
        pending_messages = [
            ModelRequest(parts=[UserPromptPart(content="do stuff")]),
            ModelResponse(parts=[ToolCallPart(tool_name="my_tool", args="{}")], model_name="test"),
        ]
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = _Output(y="wrapped")
        mock_result.usage.return_value = RunUsage()
        mock_agent.run = AsyncMock(return_value=mock_result)

        result = asyncio.run(
            execute_wrap_up(mock_agent, pending_messages, RunUsage(), "soft_limit")
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.termination_reason, "soft_limit")
        call_kwargs = mock_agent.run.call_args
        passed_messages = call_kwargs.kwargs.get("message_history") or call_kwargs[1].get("message_history")
        self.assertEqual(len(passed_messages), 1)
        self.assertIsInstance(passed_messages[0], ModelRequest)


# ---------------------------------------------------------------------------
# TestConstructorDefaults
# ---------------------------------------------------------------------------


class TestConstructorDefaults(unittest.TestCase):
    """Tests for constructor default values and attribute storage."""

    def test_default_soft_limit_false(self) -> None:
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output)
        self.assertFalse(agent._soft_limit)

    def test_default_timeouts_none(self) -> None:
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output)
        self.assertIsNone(agent._soft_timeout)
        self.assertIsNone(agent._hard_timeout)
        self.assertIsNone(agent._effective_soft_timeout)
        self.assertIsNone(agent._effective_hard_timeout)

    def test_soft_limit_stored(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output, soft_limit=True
        )
        self.assertTrue(agent._soft_limit)

    def test_effective_timeouts_computed(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_timeout=20.0,
            hard_timeout=50.0,
        )
        self.assertEqual(agent._effective_soft_timeout, 20.0)
        self.assertEqual(agent._effective_hard_timeout, 50.0)

    def test_soft_only_implicit_hard_backstop(self) -> None:
        agent = QuantedAgent(
            "test",
            input_type=_Input,
            output_type=_Output,
            soft_timeout=15.0,
        )
        self.assertEqual(agent._effective_soft_timeout, 15.0)
        self.assertEqual(agent._effective_hard_timeout, 45.0)


if __name__ == "__main__":
    unittest.main()

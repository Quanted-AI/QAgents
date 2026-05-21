"""Tests for Phase 18 execution hardening: foundation types, validation, and token counter."""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.usage import RunUsage, UsageLimits

from quanted_agents import (
    AgentTimeoutError,
    ConfigurationError,
    ContextOverflowError,
    OverflowStrategy,
    QuantedAgent,
    QuantedResult,
    ValidationResult,
)
from quanted_agents._token_counter import count_tokens
from quanted_agents._usage_limits import build_usage_limits


class _Input(BaseModel):
    """Test input model."""

    x: str


class _Output(BaseModel):
    """Test output model."""

    y: str


class TestAgentTimeoutError(unittest.TestCase):
    """Tests for AgentTimeoutError exception class."""

    def test_carries_store_and_usage(self) -> None:
        mock_store = MagicMock()
        usage = RunUsage()
        err = AgentTimeoutError(
            "timed out", store=mock_store, usage=usage, termination_reason="hard_timeout"
        )
        self.assertIs(err.store, mock_store)
        self.assertIs(err.usage, usage)
        self.assertEqual(err.termination_reason, "hard_timeout")

    def test_inherits_timeout_error(self) -> None:
        err = AgentTimeoutError(
            "timed out", store=None, usage=RunUsage(), termination_reason="hard_timeout"
        )
        self.assertIsInstance(err, TimeoutError)

    def test_message_preserved(self) -> None:
        err = AgentTimeoutError(
            "operation timed out after 30s",
            store=None,
            usage=RunUsage(),
            termination_reason="hard_timeout",
        )
        self.assertIn("operation timed out after 30s", str(err))


class TestContextOverflowError(unittest.TestCase):
    """Tests for ContextOverflowError exception class."""

    def test_carries_token_counts(self) -> None:
        err = ContextOverflowError(
            current_tokens=50000, max_tokens=32000, store=None, usage=RunUsage()
        )
        self.assertEqual(err.current_tokens, 50000)
        self.assertEqual(err.max_tokens, 32000)

    def test_auto_message(self) -> None:
        err = ContextOverflowError(
            current_tokens=50000, max_tokens=32000, store=None, usage=RunUsage()
        )
        msg = str(err)
        self.assertIn("50000", msg)
        self.assertIn("32000", msg)

    def test_carries_store_and_usage(self) -> None:
        mock_store = MagicMock()
        usage = RunUsage()
        err = ContextOverflowError(
            current_tokens=100, max_tokens=50, store=mock_store, usage=usage
        )
        self.assertIs(err.store, mock_store)
        self.assertIs(err.usage, usage)


class TestConfigurationError(unittest.TestCase):
    """Tests for ConfigurationError exception class."""

    def test_carries_errors_list(self) -> None:
        errors = ["error1", "error2"]
        err = ConfigurationError(errors)
        self.assertEqual(err.errors, errors)

    def test_inherits_value_error(self) -> None:
        err = ConfigurationError(["some error"])
        self.assertIsInstance(err, ValueError)

    def test_message_joins_errors(self) -> None:
        err = ConfigurationError(["missing key", "bad format"])
        msg = str(err)
        self.assertIn("missing key", msg)
        self.assertIn("bad format", msg)


class TestOverflowStrategy(unittest.TestCase):
    """Tests for OverflowStrategy enum."""

    def test_raise_value(self) -> None:
        self.assertEqual(OverflowStrategy.RAISE, "raise")

    def test_truncate_oldest_value(self) -> None:
        self.assertEqual(OverflowStrategy.TRUNCATE_OLDEST, "truncate_oldest")


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_default_valid(self) -> None:
        result = ValidationResult()
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])

    def test_with_errors(self) -> None:
        result = ValidationResult(valid=False, errors=["err1"], warnings=["warn1"])
        self.assertFalse(result.valid)
        self.assertEqual(result.errors, ["err1"])
        self.assertEqual(result.warnings, ["warn1"])


class TestQuantedResultTerminationFields(unittest.TestCase):
    """Tests for new termination fields on QuantedResult."""

    def test_default_values(self) -> None:
        result = QuantedResult.from_data(None, usage=RunUsage())
        self.assertIsNone(result.termination_reason)
        self.assertFalse(result.context_overflow_occurred)
        self.assertEqual(result.messages_truncated, 0)

    def test_termination_reason_settable(self) -> None:
        result = QuantedResult.from_data(None, usage=RunUsage())
        result._termination_reason = "soft_limit"
        self.assertEqual(result.termination_reason, "soft_limit")

    def test_context_overflow_fields_settable(self) -> None:
        result = QuantedResult.from_data(None, usage=RunUsage())
        result._context_overflow_occurred = True
        result._messages_truncated = 5
        self.assertTrue(result.context_overflow_occurred)
        self.assertEqual(result.messages_truncated, 5)


class TestTokenCounter(unittest.TestCase):
    """Tests for _token_counter module."""

    def test_count_tokens_returns_positive(self) -> None:
        self.assertGreater(count_tokens("hello"), 0)

    def test_empty_string_returns_minimum(self) -> None:
        self.assertGreaterEqual(count_tokens(""), 0)

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("a")
        long = count_tokens("a" * 100)
        self.assertGreater(long, short)

    def test_fallback_when_no_tiktoken(self) -> None:
        import quanted_agents._token_counter as tc
        original_cache = tc._encoder_cache.copy()
        tc._encoder_cache.clear()
        try:
            with patch.dict("sys.modules", {"tiktoken": None}):
                # Force re-evaluation by clearing cache
                tc._encoder_cache.clear()
                result = tc.count_tokens("hello world test", provider="fakeprovider")
                # 16 chars // 4 = 4
                self.assertEqual(result, 4)
        finally:
            tc._encoder_cache.clear()
            tc._encoder_cache.update(original_cache)


class TestBuildUsageLimits(unittest.TestCase):
    """Tests for _usage_limits.build_usage_limits helper."""

    def test_all_none_returns_none(self) -> None:
        self.assertIsNone(build_usage_limits())

    def test_llm_call_limit_maps_to_request_limit(self) -> None:
        limits = build_usage_limits(llm_call_limit=10)
        self.assertIsNotNone(limits)
        self.assertEqual(limits.request_limit, 10)

    def test_tool_call_limit_maps_to_tool_calls_limit(self) -> None:
        limits = build_usage_limits(tool_call_limit=5)
        self.assertIsNotNone(limits)
        self.assertEqual(limits.tool_calls_limit, 5)

    def test_all_three_set(self) -> None:
        limits = build_usage_limits(10, 5, 20)
        self.assertIsNotNone(limits)
        self.assertEqual(limits.request_limit, 10)
        self.assertEqual(limits.tool_calls_limit, 5)

    def test_total_request_limit_only(self) -> None:
        limits = build_usage_limits(total_request_limit=20)
        self.assertIsNotNone(limits)
        self.assertIsInstance(limits, UsageLimits)


class TestQuantedAgentUsageLimits(unittest.TestCase):
    """Tests for usage limit constructor parameters on QuantedAgent."""

    def test_constructor_accepts_limits(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            llm_call_limit=10, tool_call_limit=5, total_request_limit=20,
        )
        self.assertIsNotNone(agent)

    def test_limits_stored_as_attributes(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            llm_call_limit=10, tool_call_limit=5, total_request_limit=20,
        )
        self.assertEqual(agent._llm_call_limit, 10)
        self.assertEqual(agent._tool_call_limit, 5)
        self.assertEqual(agent._total_request_limit, 20)

    def test_default_limits_are_none(self) -> None:
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output)
        self.assertIsNone(agent._llm_call_limit)
        self.assertIsNone(agent._tool_call_limit)
        self.assertIsNone(agent._total_request_limit)


class TestValidate(unittest.TestCase):
    """Tests for QuantedAgent.validate() method."""

    def _make_agent(self, model: str = "test", **kwargs: object) -> QuantedAgent:
        """Create a test agent, then override _model for validation tests."""
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output, **kwargs)
        agent._model = model
        return agent

    def test_valid_config(self) -> None:
        agent = self._make_agent("openai:gpt-4o")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = agent.validate()
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_invalid_model_format_no_colon(self) -> None:
        agent = self._make_agent("nocolon")
        result = agent.validate()
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid model format" in e for e in result.errors))

    def test_invalid_model_format_empty_provider(self) -> None:
        agent = self._make_agent(":model")
        result = agent.validate()
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid model format" in e for e in result.errors))

    def test_invalid_model_format_empty_model(self) -> None:
        agent = self._make_agent("provider:")
        result = agent.validate()
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid model format" in e for e in result.errors))

    def test_missing_api_key(self) -> None:
        agent = self._make_agent("openai:gpt-4o")
        with patch.dict(os.environ, {}, clear=True):
            result = agent.validate()
        self.assertFalse(result.valid)
        self.assertTrue(any("OPENAI_API_KEY" in e for e in result.errors))

    def test_api_key_from_constructor(self) -> None:
        agent = self._make_agent("openai:gpt-4o")
        agent._api_key = "test-key"
        with patch.dict(os.environ, {}, clear=True):
            result = agent.validate()
        self.assertTrue(result.valid)

    def test_unknown_provider_warning(self) -> None:
        agent = self._make_agent("newprovider:model")
        result = agent.validate()
        self.assertTrue(any("Unknown provider" in w for w in result.warnings))

    def test_check_mcp_raises_type_error(self) -> None:
        agent = self._make_agent("test:model")
        with self.assertRaises(TypeError) as ctx:
            agent.validate(check_mcp=True)
        self.assertIn("avalidate", str(ctx.exception))

    def test_raise_on_error(self) -> None:
        agent = self._make_agent("nocolon")
        with self.assertRaises(ConfigurationError) as ctx:
            agent.validate(raise_on_error=True)
        self.assertTrue(len(ctx.exception.errors) > 0)

    def test_mcp_syntactic_valid_url(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerSSE("http://localhost:8001/sse")],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        result = agent.validate()
        self.assertTrue(result.valid)
        # No URL errors
        self.assertFalse(any("invalid URL" in e for e in result.errors))

    def test_mcp_syntactic_invalid_url(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerSSE("not-a-url")],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        result = agent.validate()
        self.assertFalse(result.valid)
        self.assertTrue(any("invalid URL" in e for e in result.errors))

    def test_mcp_unverified_warning_after_syntactic_pass(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerSSE("http://localhost:8001/sse")],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        result = agent.validate()
        self.assertTrue(any("connectivity-tested" in w for w in result.warnings))

    def test_mcp_stdio_valid_command(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerStdio("python", args=[])],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        with patch("quanted_agents.agent.shutil.which", return_value="/usr/bin/python"):
            result = agent.validate()
        self.assertFalse(any("not found" in e for e in result.errors))

    def test_mcp_stdio_invalid_command(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerStdio("nonexistent_binary_xyz", args=[])],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        with patch("quanted_agents.agent.shutil.which", return_value=None):
            with patch("quanted_agents.agent.Path.exists", return_value=False):
                result = agent.validate()
        self.assertTrue(any("not found" in e for e in result.errors))

    def test_mcp_stdio_unverified_warning(self) -> None:
        agent = QuantedAgent(
            "test", input_type=_Input, output_type=_Output,
            toolsets=[MCPServerStdio("python", args=[])],
        )
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        with patch("quanted_agents.agent.shutil.which", return_value="/usr/bin/python"):
            result = agent.validate()
        self.assertTrue(any("connectivity-tested" in w for w in result.warnings))


class TestAvalidate(unittest.TestCase):
    """Tests for QuantedAgent.avalidate() method."""

    def test_avalidate_without_mcp(self) -> None:
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output)
        agent._model = "openai:gpt-4o"
        agent._api_key = "test-key"
        result = asyncio.run(agent.avalidate())
        self.assertTrue(result.valid)

    def test_avalidate_delegates_to_validate(self) -> None:
        agent = QuantedAgent("test", input_type=_Input, output_type=_Output)
        agent._model = "nocolon"
        result = asyncio.run(agent.avalidate())
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid model format" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()

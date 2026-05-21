"""Tests for token usage preservation through the recovery pipeline.

Validates that QuantedResult.from_data() correctly accepts and returns
usage data, that _extract_usage_from_messages() properly sums RequestUsage
from ModelResponse messages, and that the recovery paths in QuantedAgent
pass usage through to the resulting QuantedResult.
"""

import unittest

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage, RunUsage

from quanted_agents.agent import QuantedAgent
from quanted_agents.result import QuantedResult
from tests.conftest import SampleOutput


class TestFromDataUsage(unittest.TestCase):
    """Tests for QuantedResult.from_data() usage parameter."""

    def test_from_data_with_usage_returns_that_usage(self) -> None:
        """from_data(data, usage=some_usage).usage returns the provided usage."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data, usage=usage)
        self.assertEqual(result.usage.requests, 1)
        self.assertEqual(result.usage.input_tokens, 100)
        self.assertEqual(result.usage.output_tokens, 50)

    def test_from_data_without_usage_returns_empty_run_usage(self) -> None:
        """from_data(data) without usage returns RunUsage() with all zeros (backward compat)."""
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data)
        self.assertEqual(result.usage.requests, 0)
        self.assertEqual(result.usage.input_tokens, 0)
        self.assertEqual(result.usage.output_tokens, 0)

    def test_from_data_with_none_usage_returns_empty_run_usage(self) -> None:
        """from_data(data, usage=None) returns RunUsage() with all zeros."""
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data, usage=None)
        self.assertEqual(result.usage.requests, 0)
        self.assertEqual(result.usage.input_tokens, 0)
        self.assertEqual(result.usage.output_tokens, 0)

    def test_from_data_usage_includes_cache_tokens(self) -> None:
        """from_data preserves all token fields including cache tokens."""
        usage = RunUsage(
            requests=2,
            input_tokens=200,
            output_tokens=100,
            cache_read_tokens=50,
            cache_write_tokens=25,
        )
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data, usage=usage)
        self.assertEqual(result.usage.cache_read_tokens, 50)
        self.assertEqual(result.usage.cache_write_tokens, 25)

    def test_from_data_total_usage_matches_usage(self) -> None:
        """total_usage returns the same as usage when from_data is used with usage."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data, usage=usage)
        self.assertEqual(result.total_usage.input_tokens, 100)
        self.assertEqual(result.total_usage.output_tokens, 50)

    def test_from_data_step_timings_use_provided_usage(self) -> None:
        """Default step_timings fallback uses the provided usage, not zeros."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        data = SampleOutput(answer="test", confidence=0.9)
        result = QuantedResult.from_data(data, usage=usage)
        timings = result.step_timings
        self.assertEqual(len(timings), 1)
        self.assertEqual(timings[0].usage.input_tokens, 100)
        self.assertEqual(timings[0].usage.output_tokens, 50)


class TestExtractUsageFromMessages(unittest.TestCase):
    """Tests for QuantedAgent._extract_usage_from_messages()."""

    def test_extracts_usage_from_single_model_response(self) -> None:
        """Extracts usage from a single ModelResponse correctly."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="hello")],
                usage=RequestUsage(input_tokens=100, output_tokens=50),
            ),
        ]
        usage = QuantedAgent._extract_usage_from_messages(messages)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)

    def test_sums_usage_from_multiple_model_responses(self) -> None:
        """Sums usage across multiple ModelResponse messages."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="first")],
                usage=RequestUsage(input_tokens=100, output_tokens=50),
            ),
            ModelResponse(
                parts=[TextPart(content="second")],
                usage=RequestUsage(input_tokens=200, output_tokens=75),
            ),
        ]
        usage = QuantedAgent._extract_usage_from_messages(messages)
        self.assertEqual(usage.requests, 2)
        self.assertEqual(usage.input_tokens, 300)
        self.assertEqual(usage.output_tokens, 125)

    def test_returns_empty_usage_for_empty_messages(self) -> None:
        """Returns RunUsage() with all zeros for an empty message list."""
        usage = QuantedAgent._extract_usage_from_messages([])
        self.assertEqual(usage.requests, 0)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_ignores_non_model_response_messages(self) -> None:
        """Only counts ModelResponse messages, ignores others."""
        messages = [
            "not a ModelResponse",
            42,
            ModelResponse(
                parts=[TextPart(content="real response")],
                usage=RequestUsage(input_tokens=100, output_tokens=50),
            ),
        ]
        usage = QuantedAgent._extract_usage_from_messages(messages)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)

    def test_handles_model_response_with_default_usage(self) -> None:
        """Handles ModelResponse with default (empty) RequestUsage."""
        messages = [
            ModelResponse(parts=[TextPart(content="no usage data")]),
        ]
        usage = QuantedAgent._extract_usage_from_messages(messages)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_preserves_cache_token_fields(self) -> None:
        """Cache token fields from RequestUsage are preserved in the sum."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="cached")],
                usage=RequestUsage(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_tokens=30,
                    cache_write_tokens=20,
                ),
            ),
        ]
        usage = QuantedAgent._extract_usage_from_messages(messages)
        self.assertEqual(usage.cache_read_tokens, 30)
        self.assertEqual(usage.cache_write_tokens, 20)


if __name__ == "__main__":
    unittest.main()

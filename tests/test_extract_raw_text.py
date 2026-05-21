"""Tests for QuantedAgent._extract_raw_text static method.

Validates that raw text extraction works for both TextPart-based responses
(OpenAI, Gemini) and ToolCallPart-based responses (Anthropic). Ensures the
recovery pipeline can receive raw text regardless of model provider.
"""

import json
import unittest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart

from quanted_agents.agent import QuantedAgent


class TestExtractRawTextFromTextPart(unittest.TestCase):
    """Tests for _extract_raw_text with TextPart messages (OpenAI/Gemini path)."""

    def test_extracts_text_from_single_text_part(self) -> None:
        """Extracts content from a single TextPart in a ModelResponse."""
        messages = [
            ModelResponse(parts=[TextPart(content='{"answer": "hello", "confidence": 0.9}')])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, '{"answer": "hello", "confidence": 0.9}')

    def test_joins_multiple_text_parts(self) -> None:
        """Joins content from multiple TextPart instances with spaces."""
        messages = [
            ModelResponse(parts=[
                TextPart(content='{"answer":'),
                TextPart(content='"hello"}'),
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, '{"answer": "hello"}')

    def test_uses_last_model_response(self) -> None:
        """Extracts text from the last ModelResponse when multiple exist."""
        messages = [
            ModelResponse(parts=[TextPart(content="first response")]),
            ModelRequest(parts=[UserPromptPart(content="follow up")]),
            ModelResponse(parts=[TextPart(content="second response")]),
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, "second response")

    def test_returns_empty_for_empty_messages(self) -> None:
        """Returns empty string when message list is empty."""
        result = QuantedAgent._extract_raw_text([])
        self.assertEqual(result, "")

    def test_returns_empty_for_no_model_response(self) -> None:
        """Returns empty string when no ModelResponse exists in messages."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hello")])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, "")


class TestExtractRawTextFromToolCallPart(unittest.TestCase):
    """Tests for _extract_raw_text with ToolCallPart messages (Anthropic path)."""

    def test_extracts_from_tool_call_with_string_args(self) -> None:
        """Extracts raw text when ToolCallPart.args is a JSON string."""
        json_str = '{"answer": "hello", "confidence": 0.9}'
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args=json_str)
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, json_str)

    def test_extracts_from_tool_call_with_dict_args(self) -> None:
        """Extracts raw text when ToolCallPart.args is a dict, serialized to JSON."""
        args_dict = {"answer": "hello", "confidence": 0.9}
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args=args_dict)
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        # Verify the result is valid JSON that matches the original dict
        parsed = json.loads(result)
        self.assertEqual(parsed, args_dict)

    def test_skips_tool_call_with_none_args(self) -> None:
        """Skips ToolCallPart when args is None."""
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args=None)
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, "")

    def test_uses_first_tool_call_with_args(self) -> None:
        """Returns content from the first ToolCallPart that has non-None args."""
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="empty_tool", args=None),
                ToolCallPart(
                    tool_name="final_result",
                    args='{"answer": "found"}',
                ),
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, '{"answer": "found"}')

    def test_uses_last_model_response_for_tool_call(self) -> None:
        """Extracts from last ModelResponse even when using ToolCallPart path."""
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args='{"answer": "first"}')
            ]),
            ModelRequest(parts=[UserPromptPart(content="retry")]),
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args='{"answer": "second"}')
            ]),
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, '{"answer": "second"}')


class TestExtractRawTextPriority(unittest.TestCase):
    """Tests for TextPart vs ToolCallPart priority in _extract_raw_text."""

    def test_text_part_takes_priority_over_tool_call_part(self) -> None:
        """TextPart is preferred when both TextPart and ToolCallPart exist."""
        messages = [
            ModelResponse(parts=[
                TextPart(content="text content"),
                ToolCallPart(tool_name="final_result", args='{"tool": "content"}'),
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, "text content")

    def test_falls_back_to_tool_call_when_no_text_part(self) -> None:
        """Falls back to ToolCallPart when ModelResponse has no TextPart."""
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args='{"answer": "fallback"}')
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, '{"answer": "fallback"}')

    def test_returns_empty_when_neither_part_type_found(self) -> None:
        """Returns empty string when ModelResponse has no TextPart or ToolCallPart."""
        messages = [
            ModelResponse(parts=[])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, "")

    def test_malformed_json_in_tool_call_args_is_returned_as_is(self) -> None:
        """Malformed JSON string in ToolCallPart.args is returned for recovery pipeline."""
        malformed = '{"answer": "test", "confidence":}'
        messages = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="final_result", args=malformed)
            ])
        ]
        result = QuantedAgent._extract_raw_text(messages)
        self.assertEqual(result, malformed)


if __name__ == "__main__":
    unittest.main()

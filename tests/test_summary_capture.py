"""Tests for summary extraction from pydantic-ai message history.

Validates the lazy summary capture on QuantedResult, covering cases where
ModelResponse has both text and tool call parts, only tool calls, only text,
no model response, recovery path, caching, and artifact lazy creation.
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.result import QuantedResult


def _make_result_with_messages(messages: list) -> QuantedResult:
    """Create a QuantedResult with a mocked AgentRunResult returning given messages.

    Args:
        messages: List of pydantic-ai message objects to return from new_messages().

    Returns:
        A QuantedResult wrapping the mock.
    """
    mock_agent_result = MagicMock()
    mock_agent_result.new_messages.return_value = messages
    mock_agent_result.output = MagicMock()
    mock_agent_result.usage.return_value = MagicMock()
    mock_agent_result.all_messages.return_value = messages
    result = QuantedResult.__new__(QuantedResult)
    result._result = mock_agent_result
    result._data = None
    result._trace_entries = []
    result._step_timings = []
    result._total_usage = None
    result._summary_extracted = False
    result._summary_value = None
    result._artifacts = None
    return result


class TestSummaryCapture(unittest.TestCase):
    """Tests for QuantedResult.summary property and _extract_summary."""

    def test_summary_with_text_and_tool_call(self) -> None:
        """ModelResponse with both TextPart and ToolCallPart returns TextPart.content."""
        response = ModelResponse(parts=[
            TextPart(content="Here is my analysis of the data."),
            ToolCallPart(
                tool_name="final_result",
                args='{"answer": "test"}',
                tool_call_id="call_1",
            ),
        ])
        result = _make_result_with_messages([response])
        self.assertEqual(result.summary, "Here is my analysis of the data.")

    def test_summary_tool_call_only(self) -> None:
        """ModelResponse with only ToolCallPart returns None."""
        response = ModelResponse(parts=[
            ToolCallPart(
                tool_name="final_result",
                args='{"answer": "test"}',
                tool_call_id="call_1",
            ),
        ])
        result = _make_result_with_messages([response])
        self.assertIsNone(result.summary)

    def test_summary_text_only(self) -> None:
        """ModelResponse with only TextPart returns None (text-only agent)."""
        response = ModelResponse(parts=[
            TextPart(content="Just some text output"),
        ])
        result = _make_result_with_messages([response])
        self.assertIsNone(result.summary)

    def test_summary_no_model_response(self) -> None:
        """No ModelResponse in messages returns None."""
        request = ModelRequest(parts=[])
        result = _make_result_with_messages([request])
        self.assertIsNone(result.summary)

    def test_summary_empty_messages(self) -> None:
        """Empty message list returns None."""
        result = _make_result_with_messages([])
        self.assertIsNone(result.summary)

    def test_summary_recovery_path(self) -> None:
        """from_data() result has summary=None and _summary_extracted=True."""
        result = QuantedResult.from_data(None)
        self.assertIsNone(result.summary)
        self.assertTrue(result._summary_extracted)

    def test_summary_lazy_caching(self) -> None:
        """Accessing .summary twice calls _extract_summary only once."""
        response = ModelResponse(parts=[
            TextPart(content="Summary text"),
            ToolCallPart(
                tool_name="final_result",
                args='{"answer": "test"}',
                tool_call_id="call_1",
            ),
        ])
        result = _make_result_with_messages([response])

        with patch.object(
            QuantedResult, "_extract_summary", wraps=result._extract_summary
        ) as mock_extract:
            first = result.summary
            second = result.summary
            self.assertEqual(first, "Summary text")
            self.assertEqual(second, "Summary text")
            mock_extract.assert_called_once()

    def test_summary_finds_last_model_response(self) -> None:
        """When multiple ModelResponses exist, uses the last one."""
        early = ModelResponse(parts=[
            TextPart(content="Early summary"),
            ToolCallPart(
                tool_name="final_result",
                args='{"v": 1}',
                tool_call_id="call_1",
            ),
        ])
        request = ModelRequest(parts=[])
        late = ModelResponse(parts=[
            TextPart(content="Late summary"),
            ToolCallPart(
                tool_name="final_result",
                args='{"v": 2}',
                tool_call_id="call_2",
            ),
        ])
        result = _make_result_with_messages([early, request, late])
        self.assertEqual(result.summary, "Late summary")

    def test_artifacts_lazy_creation(self) -> None:
        """Accessing .artifacts on a plain result creates empty ArtifactStore."""
        result = QuantedResult.from_data(None)
        self.assertIsNone(result._artifacts)
        store = result.artifacts
        self.assertIsInstance(store, ArtifactStore)
        self.assertEqual(len(store), 0)

    def test_artifacts_preserves_store(self) -> None:
        """When _artifacts is set, .artifacts returns it directly."""
        result = QuantedResult.from_data(None)
        preset_store = ArtifactStore()
        preset_store["key"] = "value"
        result._artifacts = preset_store
        self.assertIs(result.artifacts, preset_store)
        self.assertEqual(result.artifacts["key"], "value")


if __name__ == "__main__":
    unittest.main()

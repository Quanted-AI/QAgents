"""Observability data models for execution tracing and timing.

Provides StepTiming and TraceEntry dataclasses for capturing rich execution
traces from agent and workflow runs. TraceEntry records what happened at each
step: input/output data, LLM messages, tool calls, timing, token usage,
model identification, and recovery pipeline info.

Helper functions extract tool call and model name metadata from pydantic-ai
message lists for trace construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.usage import RunUsage


@dataclass
class StepTiming:
    """Timing and token usage data for a single execution step.

    Captures wall-clock duration and token usage for one agent or workflow
    step, enabling per-step performance analysis.

    Attributes:
        step_name: Human-readable name identifying the step
            (e.g., "QuantedAgent(Answer)", "Pipeline.step_0").
        duration_seconds: Wall-clock time measured via time.perf_counter().
        usage: Token usage statistics for this step from pydantic-ai.
    """

    step_name: str
    duration_seconds: float
    usage: RunUsage


@dataclass
class TraceEntry:
    """Rich execution trace entry for a single agent or workflow step.

    Records the complete execution context for one step: what went in,
    what came out, what the LLM said, which tools were called, how long
    it took, which model was used, and whether error recovery activated.

    All fields are pre-serialized to JSON-compatible types at construction
    time, so to_dict() produces a fully JSON-serializable dictionary.

    Attributes:
        step_name: Identifies which agent/step produced this entry.
        input_data: Input BaseModel serialized via model_dump().
        output_data: Output BaseModel serialized via model_dump().
        messages: LLM messages serialized via ModelMessagesTypeAdapter.
        tool_calls: Extracted tool call info (tool_name, args, tool_call_id).
        timing: Timing and usage for this step.
        model_name: The model identifier from ModelResponse, or None.
        recovery_info: Recovery pipeline details if activated, or None.
    """

    step_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    timing: StepTiming
    model_name: str | None = None
    recovery_info: dict[str, Any] | None = field(default=None)
    session_id: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Convert this trace entry to a fully JSON-serializable dictionary.

        Returns:
            A dictionary with all trace data suitable for json.dumps().
        """
        d: dict[str, Any] = {
            "step_name": self.step_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "timing": {
                "step_name": self.timing.step_name,
                "duration_seconds": self.timing.duration_seconds,
                "usage": {
                    "input_tokens": self.timing.usage.input_tokens,
                    "output_tokens": self.timing.usage.output_tokens,
                    "requests": self.timing.usage.requests,
                },
            },
            "model_name": self.model_name,
            "recovery_info": self.recovery_info,
        }
        if self.session_id is not None:
            d["session_id"] = self.session_id
        return d


def _truncate_result(result: Any, max_length: int = 500) -> str:
    """Truncate string representation of a result for standard-level trace preview.

    Args:
        result: The tool call result to truncate.
        max_length: Maximum length of the string representation.

    Returns:
        A string representation of the result, truncated with "..." if needed.
    """
    text = str(result)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@dataclass
class ToolSpan:
    """A single MCP tool call span for per-tool observability.

    Captures timing, status, and configurable verbosity data for one tool
    call. Lighter than TraceEntry (which is agent-step-level). Collected
    during agent runs and can be nested inside TraceEntry as child spans.

    Attributes:
        tool_name: The name of the MCP tool that was called.
        status: Outcome status: "success", "error", or "aborted".
        duration_seconds: Wall-clock time for the tool call in seconds.
        args: Tool arguments (included at standard and verbose levels).
        result_preview: Truncated string of the result (standard/verbose).
        error_detail: Error message if the call failed (standard/verbose).
        original_args: Pre-interceptor arguments (verbose only).
        full_result: Complete untruncated result (verbose only).
    """

    tool_name: str
    status: str
    duration_seconds: float
    args: dict[str, Any] | None = None
    result_preview: str | None = None
    error_detail: str | None = None
    original_args: dict[str, Any] | None = None
    full_result: Any | None = None

    def to_dict(self, level: str = "standard") -> dict[str, Any]:
        """Serialize this span based on verbosity level.

        Args:
            level: Verbosity level - "minimal", "standard", or "verbose".

        Returns:
            A dictionary with fields appropriate for the requested level.
        """
        d: dict[str, Any] = {
            "tool_name": self.tool_name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
        }
        if level in ("standard", "verbose"):
            d["args"] = self.args
            d["result_preview"] = self.result_preview
            d["error_detail"] = self.error_detail
        if level == "verbose":
            d["original_args"] = self.original_args
            d["full_result"] = self.full_result
        return d


def extract_tool_calls(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Extract tool call information from pydantic-ai messages.

    Iterates through messages to find ModelResponse parts that are
    ToolCallPart instances and extracts their metadata.

    Args:
        messages: List of ModelMessage objects from pydantic-ai.

    Returns:
        A list of dicts with tool_name, args, and tool_call_id for
        each tool call found in the messages.
    """
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    args = part.args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tool_calls.append({
                        "tool_name": part.tool_name,
                        "args": args,
                        "tool_call_id": part.tool_call_id,
                    })
    return tool_calls


def extract_model_name(messages: list[ModelMessage]) -> str | None:
    """Extract the model name from the first ModelResponse in messages.

    Args:
        messages: List of ModelMessage objects from pydantic-ai.

    Returns:
        The model_name string from the first ModelResponse found,
        or None if no ModelResponse exists in the messages.
    """
    for message in messages:
        if isinstance(message, ModelResponse):
            return message.model_name
    return None


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Serialize pydantic-ai messages to JSON-compatible dicts.

    Uses pydantic-ai's ModelMessagesTypeAdapter for correct serialization
    of all message types including tool calls, text parts, and metadata.

    Args:
        messages: List of ModelMessage objects from pydantic-ai.

    Returns:
        A list of JSON-compatible dictionaries representing the messages.
    """
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")

"""QuantedResult: Rich result wrapper for pydantic-ai AgentRunResult."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel
from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import RunUsage

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.observability import StepTiming, TraceEntry


class QuantedResult[OutputT: BaseModel]:
    """Rich result object wrapping pydantic-ai's AgentRunResult.

    Provides clean, typed access to the agent's output data and metadata
    including token usage and message history.

    The generic type parameter OutputT ensures that .data returns the
    correct BaseModel subclass type.

    Can be created from a normal AgentRunResult (happy path) or from a
    recovered BaseModel instance (recovery path via ``from_data``).

    Example:
        result = await agent.run(MyInput(field="value"))
        print(result.data)          # MyOutput instance
        print(result.usage)         # RunUsage with token counts
        print(result.messages)      # Full message history
    """

    def __init__(self, agent_result: AgentRunResult[OutputT]) -> None:
        """Initialize QuantedResult from a pydantic-ai AgentRunResult.

        Args:
            agent_result: The raw result from pydantic-ai Agent.run().
        """
        self._result: AgentRunResult[OutputT] | None = agent_result
        self._data: OutputT | None = None
        self._trace_entries: list[TraceEntry] = []
        self._step_timings: list[StepTiming] = []
        self._total_usage: RunUsage | None = None
        self._summary_extracted: bool = False
        self._summary_value: str | None = None
        self._artifacts: ArtifactStore | None = None
        self._was_recovered: bool = False
        self._recovery_method: str | None = None
        self._termination_reason: str | None = None
        self._context_overflow_occurred: bool = False
        self._messages_truncated: int = 0

    @classmethod
    def from_data(
        cls, data: Any, usage: RunUsage | None = None
    ) -> QuantedResult[Any]:
        """Create a QuantedResult from a recovered BaseModel instance.

        Used by the recovery pipeline when json-repair or the restructurer
        produces a valid BaseModel without going through pydantic-ai's
        Agent.run(). When ``usage`` is provided (extracted from the captured
        messages of the failed LLM call), it is preserved so that
        ``.usage`` returns accurate token counts instead of zeros.

        Args:
            data: A validated BaseModel instance produced by recovery.
            usage: Optional RunUsage from the original LLM call that
                triggered recovery. When provided, ``.usage`` returns
                this instead of an empty RunUsage.

        Returns:
            A QuantedResult wrapping the recovered data.
        """
        instance: QuantedResult[Any] = cls.__new__(cls)
        instance._result = None
        instance._data = cast(BaseModel, data)
        instance._trace_entries = []
        instance._step_timings = []
        instance._total_usage = usage
        instance._summary_extracted = True
        instance._summary_value = None
        instance._artifacts = None
        instance._was_recovered = False
        instance._recovery_method = None
        instance._termination_reason = None
        instance._context_overflow_occurred = False
        instance._messages_truncated = 0
        return instance

    @property
    def data(self) -> OutputT:
        """The validated output BaseModel instance.

        This is the primary access point for the agent's structured output.
        Returns the pydantic-ai result output on the happy path, or the
        directly stored data on the recovery path.

        Returns:
            The output data typed to the agent's output_type BaseModel.
        """
        if self._result is None:
            return cast(OutputT, self._data)
        return self._result.output

    @property
    def usage(self) -> RunUsage:
        """Token usage statistics for this run.

        Returns:
            A RunUsage object with input_tokens, output_tokens, and request counts.
            On the recovery path (no _result), returns the usage extracted from
            the original LLM call if available, otherwise an empty RunUsage.
            On the happy path (_result exists), delegates to the AgentRunResult.
        """
        if self._result is None:
            if self._total_usage is not None:
                return self._total_usage
            return RunUsage()
        return self._result.usage()

    @property
    def messages(self) -> list[ModelMessage]:
        """Full message history including all prior context.

        Returns:
            A list of all messages from the conversation.
            Returns an empty list on the recovery path.
        """
        if self._result is None:
            return []
        return self._result.all_messages()

    @property
    def new_messages(self) -> list[ModelMessage]:
        """Messages generated during this run only.

        Excludes any message_history that was passed into the run.

        Returns:
            A list of messages from this specific run.
            Returns an empty list on the recovery path.
        """
        if self._result is None:
            return []
        return self._result.new_messages()

    @property
    def trace(self) -> list[TraceEntry]:
        """Execution trace entries for this run.

        Each entry captures step name, input/output data, LLM messages,
        tool calls, timing, token usage, model name, and recovery info.
        Single-agent runs have one entry; workflows have one per step.

        Returns:
            A list of TraceEntry objects recorded during execution.
        """
        return self._trace_entries

    @property
    def step_timings(self) -> list[StepTiming]:
        """Per-step timing and usage data for this run.

        For single-agent runs, returns a single StepTiming with the
        agent's duration and usage. For workflows, returns one timing
        entry per step.

        Returns:
            A list of StepTiming objects. Falls back to a default
            timing entry using self.usage if no timings were recorded.
        """
        if self._step_timings:
            return self._step_timings
        return [StepTiming(step_name="agent", duration_seconds=0.0, usage=self.usage)]

    @property
    def total_usage(self) -> RunUsage:
        """Aggregated token usage across all steps.

        For single-agent runs, returns the same as self.usage. For
        workflows, returns the sum of usage across all steps.

        Returns:
            A RunUsage object with aggregated token counts.
        """
        if self._total_usage is not None:
            return self._total_usage
        return self.usage

    @property
    def summary(self) -> str | None:
        """The model's natural language summary alongside structured output.

        Extracts the text content from the last ModelResponse that also
        contains a ToolCallPart (structured output). Returns None if the
        model did not produce text, if this is a text-only agent, or if
        this result was created via the recovery path.

        Lazy: computed on first access, cached thereafter. Zero overhead
        if never accessed.

        Returns:
            The summary text, or None if unavailable.
        """
        if not self._summary_extracted:
            self._summary_value = self._extract_summary()
            self._summary_extracted = True
        return self._summary_value

    def _extract_summary(self) -> str | None:
        """Extract summary text from pydantic-ai message history.

        Walks backward through new_messages() to find the last ModelResponse.
        If that response contains BOTH a TextPart and a ToolCallPart, the
        TextPart content is the summary. Otherwise returns None.

        Returns:
            The extracted summary text, or None.
        """
        if self._result is None:
            return None

        messages = self._result.new_messages()
        for message in reversed(messages):
            if not isinstance(message, ModelResponse):
                continue
            text_parts = [
                part for part in message.parts if isinstance(part, TextPart)
            ]
            tool_call_parts = [
                part for part in message.parts if isinstance(part, ToolCallPart)
            ]
            if text_parts and tool_call_parts:
                return text_parts[0].content
            return None

        return None

    @property
    def artifacts(self) -> ArtifactStore:
        """Access the ArtifactStore for this result.

        Returns the store used during orchestration, or lazily creates an empty
        one on first access. This ensures zero allocation for code that never
        accesses artifacts.

        Returns:
            The ArtifactStore instance.
        """
        if self._artifacts is None:
            self._artifacts = ArtifactStore()
        return self._artifacts

    @property
    def termination_reason(self) -> str | None:
        """Why the agent run terminated, if abnormally.

        Returns:
            One of "soft_limit", "soft_timeout", "hard_timeout", or None
            for normal completion.
        """
        return self._termination_reason

    @property
    def context_overflow_occurred(self) -> bool:
        """Whether context overflow was detected and handled during this run.

        Returns:
            True if messages were truncated due to context window limits.
        """
        return self._context_overflow_occurred

    @property
    def messages_truncated(self) -> int:
        """Number of messages truncated due to context overflow.

        Returns:
            The count of messages removed to fit within context window limits.
        """
        return self._messages_truncated

    @property
    def was_recovered(self) -> bool:
        """Whether this result went through recovery (json-repair or restructurer).

        Returns:
            True if the result was produced by the recovery pipeline.
        """
        return self._was_recovered

    @property
    def recovery_method(self) -> str | None:
        """How recovery was performed: 'json_repair', 'restructurer', or None.

        Returns:
            The recovery method used, or None if no recovery occurred.
        """
        return self._recovery_method

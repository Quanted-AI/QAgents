"""Soft limit wrap-up and timeout orchestration for QuantedAgent.

Contains the wrap-up sequence logic that fires when a usage limit or
soft timeout is reached. The wrap-up gives the LLM up to 2 additional
calls (with tool calls blocked) to produce final structured output.

Also provides timeout resolution logic that computes effective soft
and hard timeout values from user configuration, and a SoftLimitGuard
that ensures first-trigger-wins semantics when multiple limit sources
can fire concurrently.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from quanted_agents.result import QuantedResult

logger = logging.getLogger(__name__)

WRAP_UP_SYSTEM_MESSAGE = (
    "IMPORTANT: You have reached your execution budget. "
    "You must produce your final structured output immediately. "
    "Do NOT attempt to call any tools -- all tool calls are disabled. "
    "Synthesize the best possible response from the information you have gathered so far."
)

TOOL_BLOCKED_MESSAGE = (
    "Tool calls are disabled -- you have reached your budget. "
    "Produce your final output now."
)

DEFAULT_GRACE_PERIOD = 30.0  # seconds
MAX_WRAP_UP_CALLS = 2


def _strip_pending_tool_calls(messages: list[Any]) -> list[Any]:
    """Remove a trailing ModelResponse with pending tool calls from message history.

    When a soft limit fires mid-tool-call, the last message in the history
    is a ModelResponse containing ToolCallPart objects that were never executed.
    Passing such history to agent.run() with a new user prompt causes pydantic-ai
    to raise UserError. This function strips that trailing message so the
    history ends at the last completed exchange.

    Args:
        messages: The message history from the agent run.

    Returns:
        A new list with the trailing ModelResponse removed if it contained
        tool calls, or the original list unchanged if no stripping was needed.
    """
    if not messages:
        return messages
    last = messages[-1]
    if isinstance(last, ModelResponse) and last.tool_calls:
        return messages[:-1]
    return messages


async def execute_wrap_up(
    agent: Any,
    messages: list[Any],
    usage_so_far: RunUsage,
    termination_reason: str,
    model_settings: ModelSettings | None = None,
) -> QuantedResult[Any] | None:
    """Run the wrap-up sequence after a soft limit or soft timeout fires.

    Gives the LLM up to MAX_WRAP_UP_CALLS additional calls with tool
    calls blocked at the SDK level. Uses the ``instructions`` parameter
    to inject the wrap-up system message (per-run, not affected by
    message history).

    Args:
        agent: The inner pydantic-ai Agent instance.
        messages: The captured message history from the original run.
        usage_so_far: Accumulated RunUsage from the original run.
        termination_reason: The reason string to set on the result
            (e.g., "soft_limit" or "soft_timeout").
        model_settings: Optional ModelSettings to forward to the inner
            wrap-up ``agent.run()`` call. When None, the inner call uses
            pydantic-ai's provider defaults (e.g., max_tokens=4096 for
            Anthropic). Pass the caller's model_settings here so the
            wrap-up honors the same max_tokens, temperature, timeout, etc.
            that the original run was configured with.

    Returns:
        A QuantedResult with termination_reason set, or None if the
        wrap-up sequence failed to produce valid output.
    """
    clean_messages = _strip_pending_tool_calls(messages)

    wrap_up_limits = UsageLimits(
        request_limit=MAX_WRAP_UP_CALLS,
        tool_calls_limit=0,
    )

    try:
        result = await agent.run(
            "Produce your final output now.",
            message_history=clean_messages,
            usage_limits=wrap_up_limits,
            instructions=WRAP_UP_SYSTEM_MESSAGE,
            model_settings=model_settings,
        )
        quanted_result: QuantedResult[Any] = QuantedResult(result)
        quanted_result._termination_reason = termination_reason
        return quanted_result

    except UsageLimitExceeded:
        logger.warning(
            f"Wrap-up sequence exhausted {MAX_WRAP_UP_CALLS} calls "
            f"without producing output (reason: {termination_reason})"
        )
        return None

    except UnexpectedModelBehavior:
        logger.warning(
            f"Wrap-up sequence failed: LLM could not produce valid "
            f"structured output (reason: {termination_reason})"
        )
        return None


def resolve_timeouts(
    soft_timeout: float | None,
    hard_timeout: float | None,
) -> tuple[float | None, float | None]:
    """Compute effective soft and hard timeout values from user configuration.

    Applies the implicit hard backstop rule: when only soft_timeout is
    set, the effective hard timeout is soft_timeout + DEFAULT_GRACE_PERIOD.

    Args:
        soft_timeout: Seconds before soft wrap-up fires, or None.
        hard_timeout: Seconds before hard kill, or None.

    Returns:
        A tuple of (effective_soft_timeout, effective_hard_timeout).
        Either or both may be None.

    Raises:
        ValueError: If hard_timeout is set and is less than or equal
            to soft_timeout.
    """
    if soft_timeout is not None and hard_timeout is not None:
        if hard_timeout <= soft_timeout:
            raise ValueError(
                f"hard_timeout ({hard_timeout}s) must be greater "
                f"than soft_timeout ({soft_timeout}s)"
            )
        return (soft_timeout, hard_timeout)

    if soft_timeout is not None:
        return (soft_timeout, soft_timeout + DEFAULT_GRACE_PERIOD)

    if hard_timeout is not None:
        return (None, hard_timeout)

    return (None, None)


class SoftLimitGuard:
    """First-trigger-wins guard for soft limit activation.

    Prevents duplicate activations when multiple limit sources
    (usage limit, soft timeout) fire concurrently or in sequence.
    Once activated, subsequent activate() calls are no-ops.
    """

    def __init__(self) -> None:
        """Initialize an inactive guard."""
        self._active: bool = False
        self._reason: str | None = None

    def activate(self, reason: str) -> bool:
        """Attempt to activate soft limit mode.

        Args:
            reason: The reason for activation (e.g., "soft_limit",
                "soft_timeout").

        Returns:
            True if this was the first activation, False if already
            active (first trigger wins).
        """
        if self._active:
            return False
        self._active = True
        self._reason = reason
        return True

    @property
    def is_active(self) -> bool:
        """Whether soft limit mode is currently active.

        Returns:
            True if activate() has been called successfully.
        """
        return self._active

    @property
    def reason(self) -> str | None:
        """The reason for soft limit activation.

        Returns:
            The reason string passed to the first successful
            activate() call, or None if not yet activated.
        """
        return self._reason

"""Usage limit translation from SDK names to pydantic-ai UsageLimits.

Maps the QuantedAgent constructor parameter names to pydantic-ai's
internal UsageLimits field names:

- llm_call_limit -> request_limit (number of LLM API calls)
- tool_call_limit -> tool_calls_limit (number of tool invocations)
- total_request_limit -> tracked internally at SDK level (no pydantic-ai equivalent)

This disambiguation avoids confusion between pydantic-ai's "request_limit"
(which counts LLM calls, not HTTP requests) and the more intuitive SDK names.
"""

from __future__ import annotations

from pydantic_ai.usage import UsageLimits


def build_usage_limits(
    llm_call_limit: int | None = None,
    tool_call_limit: int | None = None,
    total_request_limit: int | None = None,
) -> UsageLimits | None:
    """Build a pydantic-ai UsageLimits from SDK parameter names.

    Translates the human-friendly SDK parameter names to pydantic-ai's
    internal field names. Returns None when no limits are configured,
    allowing the caller to skip passing usage_limits entirely.

    Args:
        llm_call_limit: Maximum number of LLM API calls. Maps to
            UsageLimits.request_limit.
        tool_call_limit: Maximum number of tool invocations. Maps to
            UsageLimits.tool_calls_limit (note the plural).
        total_request_limit: Maximum total requests tracked at SDK level.
            No pydantic-ai equivalent; stored for SDK-level tracking.

    Returns:
        A UsageLimits instance if any limit is set, or None if all are None.
    """
    if llm_call_limit is None and tool_call_limit is None and total_request_limit is None:
        return None

    return UsageLimits(
        request_limit=llm_call_limit,
        tool_calls_limit=tool_call_limit,
    )

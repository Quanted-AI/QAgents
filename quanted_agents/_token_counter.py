"""Token counting abstraction for context window management.

Provides token estimation using tiktoken when available, falling back to
a character-based approximation. Used by the context overflow handling
system to estimate message sizes before sending to the LLM.

Includes message-level token counting and truncation logic for context
overflow handling. Truncation drops oldest conversation messages first
while always preserving system prompt messages.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart

logger = logging.getLogger(__name__)

_PROVIDER_ENCODINGS: dict[str, str] = {
    "openai": "cl100k_base",
    "anthropic": "cl100k_base",
}

_CHARS_PER_TOKEN_FALLBACK = 4

_encoder_cache: dict[str, Any] = {}


def _get_encoder(provider: str) -> Any:
    """Get a tiktoken encoder for the given provider.

    Caches encoders at module level to avoid repeated initialization.
    Returns None if tiktoken is not installed.

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic").

    Returns:
        A tiktoken Encoding instance, or None if tiktoken is unavailable.
    """
    if provider in _encoder_cache:
        return _encoder_cache[provider]

    encoding_name = _PROVIDER_ENCODINGS.get(provider, "cl100k_base")
    try:
        import tiktoken
        encoder = tiktoken.get_encoding(encoding_name)
        _encoder_cache[provider] = encoder
        return encoder
    except ImportError:
        logger.debug("tiktoken not installed, using character-based fallback")
        _encoder_cache[provider] = None
        return None


def count_tokens(text: str, provider: str = "openai") -> int:
    """Count tokens in a text string.

    Uses tiktoken when available for accurate counts, otherwise falls
    back to a character-based estimate (len(text) // 4, minimum 1).

    Args:
        text: The text to count tokens for.
        provider: The LLM provider name for encoding selection.

    Returns:
        Estimated token count (always >= 0).
    """
    encoder = _get_encoder(provider)
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK) if text else 0


def count_message_tokens(message: Any, provider: str = "openai") -> int:
    """Count tokens in a pydantic-ai message object.

    Serializes message parts to text and counts tokens. For ModelRequest
    and ModelResponse, iterates parts and sums text content. Non-text
    parts (images, etc.) use a fixed 1000-token estimate.

    Args:
        message: A pydantic-ai ModelRequest or ModelResponse message.
        provider: The LLM provider name for encoding selection.

    Returns:
        Estimated token count for the entire message.
    """
    total = 0
    parts = getattr(message, "parts", [])
    for part in parts:
        content = getattr(part, "content", None)
        if isinstance(content, str):
            total += count_tokens(content, provider)
        else:
            # Non-text parts (images, binary data, etc.)
            total += 1000
    return total


def count_messages_tokens(messages: list[Any], provider: str = "openai") -> int:
    """Count total estimated tokens across a list of pydantic-ai messages.

    Sums the token count for each message by serializing text content
    from message parts.

    Args:
        messages: A list of pydantic-ai ModelRequest or ModelResponse messages.
        provider: The LLM provider name for encoding selection.

    Returns:
        Total estimated token count across all messages.
    """
    total = 0
    for msg in messages:
        total += count_message_tokens(msg, provider)
    return total


def truncate_messages(
    messages: list[Any],
    system_prompt_tokens: int,
    max_tokens: int,
    provider: str = "openai",
) -> tuple[list[Any], int]:
    """Truncate messages to fit within a token budget.

    Separates system prompt messages from conversation messages, then
    walks conversation messages from newest to oldest, keeping those
    that fit within the remaining budget. System prompt messages are
    always preserved. Oldest conversation messages are dropped first.

    Args:
        messages: The full list of pydantic-ai messages.
        system_prompt_tokens: Token count already reserved for the system prompt.
        max_tokens: Maximum total token budget.
        provider: The LLM provider name for encoding selection.

    Returns:
        A tuple of (kept_messages, dropped_count) where kept_messages
        is system messages + newest conversation messages that fit,
        and dropped_count is the number of conversation messages removed.
    """
    if not messages:
        return [], 0

    # Separate system prompt messages from conversation messages
    system_msgs: list[Any] = []
    conversation_msgs: list[Any] = []

    for msg in messages:
        if _is_system_prompt_message(msg):
            system_msgs.append(msg)
        else:
            conversation_msgs.append(msg)

    remaining_budget = max_tokens - system_prompt_tokens

    # Walk from newest to oldest, accumulating token counts
    kept_conversation: list[Any] = []
    accumulated_tokens = 0

    for msg in reversed(conversation_msgs):
        msg_tokens = count_message_tokens(msg, provider)
        if accumulated_tokens + msg_tokens <= remaining_budget:
            kept_conversation.append(msg)
            accumulated_tokens += msg_tokens
        else:
            break

    # Reverse to restore original order
    kept_conversation.reverse()

    dropped_count = len(conversation_msgs) - len(kept_conversation)
    return system_msgs + kept_conversation, dropped_count


def _is_system_prompt_message(message: Any) -> bool:
    """Check if a message is a system prompt message.

    A message is considered a system prompt message if it is a ModelRequest
    containing at least one SystemPromptPart.

    Args:
        message: A pydantic-ai message object.

    Returns:
        True if the message contains a SystemPromptPart.
    """
    if not isinstance(message, ModelRequest):
        return False
    return any(isinstance(part, SystemPromptPart) for part in message.parts)

"""Shared test configuration and fixtures for quanted_agents tests.

Blocks all real LLM API calls via ALLOW_MODEL_REQUESTS = False and provides
reusable sample BaseModels and agent fixtures used across test files.
Includes fixture factory functions (make_agent, make_store, make_budget)
for composable integration test setup.
"""

from typing import Any

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import models

from quanted_agents import ArtifactStore, QuantedAgent, WorkflowBudget

# Load .env so API keys are available for integration tests.
load_dotenv()

# Safety guard: prevent any real LLM API calls during testing.
# If a test accidentally uses a real model, pydantic-ai will raise an error
# instead of making a network request.
models.ALLOW_MODEL_REQUESTS = False

# Malformed JSON samples for error recovery testing
MALFORMED_MISSING_QUOTES = "{answer: 'test', confidence: 0.9}"
MALFORMED_TRAILING_COMMA = '{"answer": "test", "confidence": 0.9,}'
MALFORMED_SINGLE_QUOTES = "{'answer': 'test', 'confidence': 0.9}"
MALFORMED_EXTRA_TEXT = 'Here is the JSON: {"answer": "test", "confidence": 0.9}'
MALFORMED_INCOMPLETE = '{"answer": "test", "confidence":'
VALID_JSON = '{"answer": "test", "confidence": 0.9}'
IRREPARABLE_TEXT = "This is just plain text with no JSON structure whatsoever."


class SampleInput(BaseModel):
    """Sample input model for testing agent construction and execution."""

    question: str
    context: str = ""


class SampleOutput(BaseModel):
    """Sample output model for testing agent construction and execution."""

    answer: str
    confidence: float = 0.0


@pytest.fixture
def sample_agent() -> QuantedAgent:
    """Create a QuantedAgent configured for testing.

    Uses "test" as the model identifier and SampleInput/SampleOutput as
    the BaseModel types. Suitable for use with TestModel override in
    async test methods.

    Returns:
        A QuantedAgent instance ready for testing.
    """
    return QuantedAgent(
        "test",
        input_type=SampleInput,
        output_type=SampleOutput,
        system_prompt="Test agent",
    )


# ---------------------------------------------------------------------------
# Fixture Factories (plain functions, importable by integration test files)
# ---------------------------------------------------------------------------


def make_agent(
    output_type: type[BaseModel] = SampleOutput,
    input_type: type[BaseModel] = SampleInput,
    system_prompt: str = "Test agent",
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> QuantedAgent:
    """Create a QuantedAgent configured for testing.

    Uses ``"test"`` as the model identifier. Callers override the model
    at runtime via ``agent.inner.override(model=...)``.

    Args:
        output_type: Pydantic BaseModel subclass for agent output.
        input_type: Pydantic BaseModel subclass for agent input.
        system_prompt: Static system prompt string.
        tools: Optional list of tool functions or pydantic-ai Tool instances.
        **kwargs: Extra keyword arguments forwarded to QuantedAgent
            (e.g., max_context_tokens, overflow_strategy, soft_limit).

    Returns:
        A QuantedAgent instance ready for testing.
    """
    return QuantedAgent(
        "test",
        input_type=input_type,
        output_type=output_type,
        system_prompt=system_prompt,
        tools=tools or [],
        **kwargs,
    )


def make_store(initial: dict[str, Any] | None = None) -> ArtifactStore:
    """Create an ArtifactStore, optionally pre-populated.

    Args:
        initial: Optional dict of key-value pairs to populate the store
            with. Each pair is set via ``store[key] = value``.

    Returns:
        An ArtifactStore instance, empty or pre-populated.
    """
    store = ArtifactStore()
    for key, value in (initial or {}).items():
        store[key] = value
    return store


def make_budget(
    llm_call_limit: int | None = 20,
    tool_call_limit: int | None = 10,
    total_request_limit: int | None = None,
) -> WorkflowBudget:
    """Create a WorkflowBudget with configurable limits.

    Args:
        llm_call_limit: Maximum LLM calls, or None for unlimited.
        tool_call_limit: Maximum tool calls, or None for unlimited.
        total_request_limit: Maximum total requests, or None for unlimited.

    Returns:
        A WorkflowBudget instance with the given limits.
    """
    return WorkflowBudget(
        llm_call_limit=llm_call_limit,
        tool_call_limit=tool_call_limit,
        total_request_limit=total_request_limit,
    )

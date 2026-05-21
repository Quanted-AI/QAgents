"""Soft Limits Example: Graceful Usage Limits and Timeouts.

Demonstrates QuantedAgent's soft limit and timeout features. Instead of
crashing with UsageLimitExceeded when limits are hit, soft limits trigger
a graceful wrap-up sequence where the agent gets additional LLM calls
(with tools blocked) to produce final output.

Three termination modes are shown:
1. Normal completion (no limits hit)
2. Soft limit (llm_call_limit reached, agent wraps up gracefully)
3. Soft timeout (time limit reached, agent wraps up gracefully)

The result.termination_reason property reports how the run ended:
- None: normal completion
- "soft_limit": usage limit triggered wrap-up
- "soft_timeout": time limit triggered wrap-up

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., OPENAI_API_KEY).
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import os
import sys

from pydantic import BaseModel

from quanted_agents import QuantedAgent


# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResearchTask(BaseModel):
    """Input for the research agent.

    Attributes:
        question: The research question to investigate.
    """

    question: str


class ResearchAnswer(BaseModel):
    """Output from the research agent.

    Attributes:
        answer: The research answer.
        confidence: Confidence level from 0.0 to 1.0.
        sources_consulted: Number of sources the agent considered.
    """

    answer: str
    confidence: float
    sources_consulted: int


# ---------------------------------------------------------------------------
# Pattern 1: Soft Limit on LLM Calls
# ---------------------------------------------------------------------------

async def soft_limit_example() -> None:
    """Demonstrate soft limit triggering graceful wrap-up."""
    print("=" * 60)
    print("Pattern 1: Soft Limit (llm_call_limit)")
    print("=" * 60)

    agent = QuantedAgent(
        MODEL,
        input_type=ResearchTask,
        output_type=ResearchAnswer,
        system_prompt=(
            "You are a thorough researcher. Investigate the question deeply. "
            "Consider multiple angles and perspectives. Be as comprehensive "
            "as possible in your analysis."
        ),
        llm_call_limit=3,    # Low limit to demonstrate soft wrap-up
        soft_limit=True,      # Enable graceful wrap-up instead of crash
    )

    result = await agent.run(
        ResearchTask(question="What are the key challenges in quantum error correction?")
    )

    print(f"Answer: {result.data.answer[:150]}...")
    print(f"Confidence: {result.data.confidence}")
    print(f"Sources consulted: {result.data.sources_consulted}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"  (None = normal, 'soft_limit' = limit hit, 'soft_timeout' = time hit)")
    print(f"LLM requests used: {result.usage.requests}")
    print()


# ---------------------------------------------------------------------------
# Pattern 2: Soft Timeout
# ---------------------------------------------------------------------------

async def soft_timeout_example() -> None:
    """Demonstrate soft timeout triggering graceful wrap-up."""
    print("=" * 60)
    print("Pattern 2: Soft Timeout")
    print("=" * 60)

    agent = QuantedAgent(
        MODEL,
        input_type=ResearchTask,
        output_type=ResearchAnswer,
        system_prompt=(
            "You are a thorough researcher. Take your time to investigate "
            "the question from multiple angles."
        ),
        soft_timeout=15.0,    # 15 seconds before soft wrap-up fires
        hard_timeout=45.0,    # 45 seconds hard kill (safety net)
        soft_limit=True,      # Required to enable soft wrap-up for timeouts
    )

    result = await agent.run(
        ResearchTask(question="Compare approaches to distributed consensus algorithms")
    )

    print(f"Answer: {result.data.answer[:150]}...")
    print(f"Confidence: {result.data.confidence}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"LLM requests used: {result.usage.requests}")
    print()


# ---------------------------------------------------------------------------
# Pattern 3: Combined Limits
# ---------------------------------------------------------------------------

async def combined_limits_example() -> None:
    """Demonstrate combined usage limits and timeouts."""
    print("=" * 60)
    print("Pattern 3: Combined Limits (call limit + timeout)")
    print("=" * 60)

    agent = QuantedAgent(
        MODEL,
        input_type=ResearchTask,
        output_type=ResearchAnswer,
        system_prompt="Research the question thoroughly.",
        llm_call_limit=5,     # Max 5 LLM calls
        tool_call_limit=3,    # Max 3 tool calls
        soft_limit=True,      # Graceful wrap-up
        soft_timeout=20.0,    # 20s soft timeout
        hard_timeout=40.0,    # 40s hard kill
    )

    result = await agent.run(
        ResearchTask(question="What are the tradeoffs of microservices vs monoliths?")
    )

    print(f"Answer: {result.data.answer[:150]}...")
    print(f"Confidence: {result.data.confidence}")
    print(f"Termination reason: {result.termination_reason}")
    print()

    # Show full usage details
    print("Usage details:")
    print(f"  Input tokens: {result.usage.input_tokens}")
    print(f"  Output tokens: {result.usage.output_tokens}")
    print(f"  LLM requests: {result.usage.requests}")
    print(f"  Tool calls: {result.usage.tool_calls}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all three soft limit patterns."""
    await soft_limit_example()
    await soft_timeout_example()
    await combined_limits_example()

    print("All patterns complete.")


if __name__ == "__main__":
    api_key_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ]
    has_key = any(os.environ.get(var) for var in api_key_vars)
    if not has_key:
        print(
            "No API key found. Set one of the following environment variables:\n"
            f"  {', '.join(api_key_vars)}\n"
            "Then run: python examples/10_soft_limits.py"
        )
        sys.exit(1)

    asyncio.run(main())

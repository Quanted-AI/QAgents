"""Parallel Retry Example: Automatic Retry for Failed Branches.

Demonstrates RetryPolicy on Parallel workflows. When a branch fails with
an exception matching the retry_on types, Parallel automatically retries
that branch up to max_retries times with configurable delays.

This is useful for workflows where branches may fail due to transient
issues (network errors, rate limits, timeouts) but should succeed on
retry. Non-retryable failures are preserved in the errors list.

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., OPENAI_API_KEY).
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import os
import sys

from pydantic import BaseModel

from quanted_agents import Parallel, QuantedAgent, RetryPolicy


# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalysisInput(BaseModel):
    """Input for all analysis branches.

    Attributes:
        topic: The topic to analyze.
    """

    topic: str


class SentimentResult(BaseModel):
    """Output from the sentiment analysis branch.

    Attributes:
        sentiment: The detected sentiment (positive, negative, neutral).
        confidence: Confidence score from 0.0 to 1.0.
    """

    sentiment: str
    confidence: float


class TopicResult(BaseModel):
    """Output from the topic extraction branch.

    Attributes:
        topics: List of extracted topics.
        primary_topic: The most prominent topic.
    """

    topics: list[str]
    primary_topic: str


class SummaryResult(BaseModel):
    """Output from the summarization branch.

    Attributes:
        summary: A concise summary of the input.
        word_count: Number of words in the summary.
    """

    summary: str
    word_count: int


# ---------------------------------------------------------------------------
# Branch agents
# ---------------------------------------------------------------------------

sentiment_agent = QuantedAgent(
    MODEL,
    input_type=AnalysisInput,
    output_type=SentimentResult,
    system_prompt="Analyze the sentiment of the given topic. Return positive, negative, or neutral.",
)

topic_agent = QuantedAgent(
    MODEL,
    input_type=AnalysisInput,
    output_type=TopicResult,
    system_prompt="Extract key topics from the given input. Identify the primary topic.",
)

summary_agent = QuantedAgent(
    MODEL,
    input_type=AnalysisInput,
    output_type=SummaryResult,
    system_prompt="Summarize the given topic in 2-3 sentences.",
)


# ---------------------------------------------------------------------------
# Pattern 1: Basic Parallel with Retry
# ---------------------------------------------------------------------------

async def basic_retry_example() -> None:
    """Demonstrate Parallel with RetryPolicy."""
    print("=" * 60)
    print("Pattern 1: Parallel with RetryPolicy")
    print("=" * 60)

    parallel = Parallel(
        branches=[sentiment_agent, topic_agent, summary_agent],
        retry_policy=RetryPolicy(
            max_retries=2,
            retry_on=[ConnectionError, TimeoutError],
            delay_seconds=1.0,
        ),
    )

    result = await parallel.run(
        AnalysisInput(topic="The rise of electric vehicles and their environmental impact")
    )

    # Access individual branch results
    print(f"Successful branches: {len(result.results)}")
    print(f"Failed branches: {len(result.errors)}")

    # ParallelResult.data returns ParallelOutput with items list
    print(f"\nBranch outputs ({len(result.data.items)} items):")
    for i, item in enumerate(result.data.items):
        print(f"  Branch {i}: {type(item).__name__}")
        if isinstance(item, SentimentResult):
            print(f"    Sentiment: {item.sentiment} (confidence: {item.confidence})")
        elif isinstance(item, TopicResult):
            print(f"    Primary topic: {item.primary_topic}")
            print(f"    All topics: {item.topics}")
        elif isinstance(item, SummaryResult):
            print(f"    Summary: {item.summary[:100]}...")
            print(f"    Word count: {item.word_count}")

    # Show any errors
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  {type(error).__name__}: {error}")

    # Usage stats
    print(f"\nTotal input tokens: {result.total_usage.input_tokens}")
    print(f"Total output tokens: {result.total_usage.output_tokens}")
    print(f"Total requests: {result.total_usage.requests}")

    # Per-branch timing
    print("\nPer-branch timing:")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")
    print()


# ---------------------------------------------------------------------------
# Pattern 2: Parallel with Retry Configuration Details
# ---------------------------------------------------------------------------

async def retry_config_example() -> None:
    """Show different RetryPolicy configurations."""
    print("=" * 60)
    print("Pattern 2: RetryPolicy Configurations")
    print("=" * 60)

    # Configuration 1: Aggressive retry for flaky services
    aggressive = RetryPolicy(
        max_retries=3,
        retry_on=[ConnectionError, TimeoutError, OSError],
        delay_seconds=0.5,
    )

    # Configuration 2: Conservative retry for expensive operations
    conservative = RetryPolicy(
        max_retries=1,
        retry_on=[ConnectionError],
        delay_seconds=2.0,
    )

    # Configuration 3: No retry (default behavior)
    no_retry = RetryPolicy(
        max_retries=0,
        retry_on=[],
        delay_seconds=0.0,
    )

    print("Aggressive: max_retries=3, retry_on=[ConnectionError, TimeoutError, OSError]")
    print(f"  delay_seconds=0.5 (first retry attempt has no delay)")
    print("Conservative: max_retries=1, retry_on=[ConnectionError]")
    print(f"  delay_seconds=2.0")
    print("No retry: max_retries=0 (failures go directly to errors list)")
    print()

    # Use aggressive policy for this run
    parallel = Parallel(
        branches=[sentiment_agent, topic_agent],
        retry_policy=aggressive,
    )

    result = await parallel.run(
        AnalysisInput(topic="Advances in renewable energy storage technology")
    )

    print(f"Results: {len(result.results)} successful, {len(result.errors)} failed")
    for i, item in enumerate(result.data.items):
        print(f"  Branch {i}: {type(item).__name__}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all parallel retry patterns."""
    await basic_retry_example()
    await retry_config_example()

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
            "Then run: python examples/13_parallel_retry.py"
        )
        sys.exit(1)

    asyncio.run(main())

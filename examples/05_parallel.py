"""Parallel Example: Multi-Analysis.

Demonstrates a Parallel workflow that runs multiple agents concurrently on the
same input. Three independent analyses (sentiment, topics, summary) execute in
parallel via asyncio.gather, and all results are collected into a ParallelResult.

The Parallel flow is:
1. All branches receive the same input
2. Branches execute concurrently (fan-out)
3. Results are collected into a ParallelResult (fan-in)
4. Access individual branch results via result.results list
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio

from pydantic import BaseModel

from quanted_agents import Parallel, QuantedAgent


class TextInput(BaseModel):
    """Input text for multi-analysis.

    Attributes:
        text: The text to analyze from multiple perspectives.
    """

    text: str


class SentimentResult(BaseModel):
    """Sentiment analysis output.

    Attributes:
        sentiment: The detected sentiment ("positive", "negative", "neutral", "mixed").
        confidence: Confidence score from 0.0 to 1.0.
    """

    sentiment: str
    confidence: float


class TopicsResult(BaseModel):
    """Topic extraction output.

    Attributes:
        topics: List of topics identified in the text.
        relevance_scores: Relevance score for each topic (0.0 to 1.0).
    """

    topics: list[str]
    relevance_scores: list[float]


class SummaryResult(BaseModel):
    """Text summarization output.

    Attributes:
        summary: A concise summary of the input text.
        word_count: Word count of the summary.
    """

    summary: str
    word_count: int


sentiment_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=SentimentResult,
    system_prompt=(
        "You are a sentiment analysis specialist. Analyze the text and determine its "
        "overall sentiment. Return the sentiment label and your confidence score."
    ),
)

topic_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=TopicsResult,
    system_prompt=(
        "You are a topic extraction specialist. Identify the main topics discussed in "
        "the text. Return each topic with a relevance score indicating how central it "
        "is to the text."
    ),
)

summary_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=SummaryResult,
    system_prompt=(
        "You are a text summarization specialist. Create a concise summary of the text "
        "that captures the key points. Keep the summary under 50 words."
    ),
)

parallel = Parallel(branches=[sentiment_agent, topic_agent, summary_agent])


async def main() -> None:
    """Run multi-analysis on a sample text."""
    text = TextInput(
        text=(
            "The global transition to renewable energy has accelerated significantly in "
            "2025, with solar and wind power now accounting for over 40% of electricity "
            "generation in Europe. Major investments from both public and private sectors "
            "have driven costs down to historic lows, making clean energy cheaper than "
            "fossil fuels in most markets. However, challenges remain in energy storage, "
            "grid infrastructure, and ensuring a just transition for workers in traditional "
            "energy sectors. Environmental groups have praised the progress while calling "
            "for even more ambitious targets to meet the Paris Agreement goals."
        )
    )

    result = await parallel.run(text)

    # Access individual branch results
    print("=== Multi-Analysis Results ===")
    print(f"Number of branches: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")

    for i, branch_result in enumerate(result.results):
        print(f"\n--- Branch {i} ---")
        print(f"  Data: {branch_result.data}")

    # Observability: aggregated usage across all branches
    print("\n=== Aggregated Usage ===")
    print(f"Total input tokens: {result.usage.input_tokens}")
    print(f"Total output tokens: {result.usage.output_tokens}")
    print(f"Total requests: {result.usage.requests}")

    # Observability: execution trace from all branches
    print("\n=== Trace ===")
    print(f"Total trace entries: {len(result.trace)}")
    for entry in result.trace:
        print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

"""Single Agent Example: Content Analyzer.

Demonstrates a single QuantedAgent with typed Pydantic I/O, a system prompt,
and observability access. The agent analyzes an article and returns a structured
analysis including summary, sentiment, key topics, and word count.

This is the simplest QuantedAgents pattern -- one agent, one input, one output.
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio

from pydantic import BaseModel

from quanted_agents import QuantedAgent


class ArticleInput(BaseModel):
    """Input for the content analyzer agent.

    Attributes:
        text: The article text to analyze.
        max_words: Maximum word count for the summary.
    """

    text: str
    max_words: int = 100


class AnalysisOutput(BaseModel):
    """Structured output from the content analyzer agent.

    Attributes:
        summary: A concise summary of the article.
        sentiment: The overall sentiment (e.g., "positive", "negative", "neutral").
        key_topics: List of main topics identified in the article.
        word_count: The word count of the original article.
    """

    summary: str
    sentiment: str
    key_topics: list[str]
    word_count: int


agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ArticleInput,
    output_type=AnalysisOutput,
    system_prompt=(
        "You are a content analysis expert. Analyze the provided article and return "
        "a structured analysis. Keep the summary within the requested word limit. "
        "Identify the overall sentiment and extract key topics discussed in the text."
    ),
)


async def main() -> None:
    """Run the content analyzer on a sample article."""
    article = ArticleInput(
        text=(
            "Artificial intelligence is transforming healthcare in remarkable ways. "
            "From early disease detection using machine learning algorithms to robotic "
            "surgery assistants, the integration of AI in medicine has accelerated "
            "dramatically in 2025. Researchers at leading universities have demonstrated "
            "that AI-powered diagnostic tools can identify certain cancers with 95% "
            "accuracy, surpassing many human specialists. However, concerns about data "
            "privacy, algorithmic bias, and the need for human oversight remain important "
            "challenges that the industry must address."
        ),
        max_words=50,
    )

    result = await agent.run(article)

    # Access the typed output
    print("=== Content Analysis ===")
    print(f"Summary: {result.data.summary}")
    print(f"Sentiment: {result.data.sentiment}")
    print(f"Key Topics: {', '.join(result.data.key_topics)}")
    print(f"Word Count: {result.data.word_count}")

    # Observability: token usage
    print("\n=== Usage ===")
    print(f"Input tokens: {result.usage.input_tokens}")
    print(f"Output tokens: {result.usage.output_tokens}")
    print(f"Total requests: {result.usage.requests}")

    # Observability: execution trace
    print("\n=== Trace ===")
    print(f"Trace entries: {len(result.trace)}")
    if result.trace:
        entry = result.trace[0]
        print(f"Step name: {entry.step_name}")
        print(f"Duration: {entry.timing.duration_seconds:.2f}s")
        print(f"Model: {entry.model_name}")

    # Observability: step timings and aggregated usage
    print("\n=== Step Timings ===")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")
    print(f"Total usage (input tokens): {result.total_usage.input_tokens}")


if __name__ == "__main__":
    asyncio.run(main())

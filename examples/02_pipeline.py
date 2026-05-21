"""Pipeline Example: Research Pipeline.

Demonstrates a Pipeline workflow that chains two agents sequentially:
a researcher that gathers findings, and a writer that produces a final report.

Pipeline passes the output of each step as input to the next step. The final
step's result is returned with aggregated observability data across all steps.
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio

from pydantic import BaseModel

from quanted_agents import Pipeline, QuantedAgent


class ResearchQuery(BaseModel):
    """Input for the research pipeline.

    Attributes:
        topic: The research topic to investigate.
        depth: How deep the research should go ("brief", "standard", "thorough").
    """

    topic: str
    depth: str = "standard"


class ResearchNotes(BaseModel):
    """Intermediate output from the researcher agent.

    Attributes:
        topic: The research topic that was investigated.
        findings: List of key findings from the research.
        sources: List of source descriptions or references.
    """

    topic: str
    findings: list[str]
    sources: list[str]


class FinalReport(BaseModel):
    """Final output from the writer agent.

    Attributes:
        title: The report title.
        content: The full report text synthesized from research notes.
        sources: List of sources cited in the report.
    """

    title: str
    content: str
    sources: list[str]


researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchQuery,
    output_type=ResearchNotes,
    system_prompt=(
        "You are a research analyst. Given a topic and depth level, gather key findings "
        "and identify relevant sources. Return structured research notes that a report "
        "writer can use to produce a polished document."
    ),
)

writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchNotes,
    output_type=FinalReport,
    system_prompt=(
        "You are a technical report writer. Given research notes with findings and "
        "sources, synthesize them into a well-structured report with a clear title, "
        "coherent content, and properly cited sources."
    ),
)

pipeline = Pipeline(steps=[researcher, writer])


async def main() -> None:
    """Run the research pipeline on a sample topic."""
    query = ResearchQuery(topic="Quantum Computing Applications in Cryptography", depth="standard")

    result = await pipeline.run(query)

    # Access the final typed output
    print("=== Final Report ===")
    print(f"Title: {result.data.title}")
    print(f"Content (first 200 chars): {result.data.content[:200]}...")
    print(f"Sources: {', '.join(result.data.sources)}")

    # Observability: aggregated usage across all pipeline steps
    print("\n=== Aggregated Usage ===")
    print(f"Total input tokens: {result.total_usage.input_tokens}")
    print(f"Total output tokens: {result.total_usage.output_tokens}")
    print(f"Total requests: {result.total_usage.requests}")

    # Observability: per-step timing
    print("\n=== Step Timings ===")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s ({timing.usage.input_tokens} in / {timing.usage.output_tokens} out)")

    # Observability: execution trace
    print("\n=== Trace ===")
    print(f"Trace entries: {len(result.trace)}")
    for entry in result.trace:
        print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

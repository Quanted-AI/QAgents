"""Dual-Stream Example: Structured Data + Natural Language Summary.

Demonstrates the dual-stream architecture where an agent produces both
structured data (via result.data) and a natural language summary (via
result.summary). Uses a two-stage Pipeline with ArtifactStore and an
assembly function that combines stage outputs into a final deliverable.

The dual-stream pattern is useful when you need both machine-readable
structured output AND a human-readable narrative from the same workflow.

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., OPENAI_API_KEY).
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import os
import sys

from pydantic import BaseModel

from quanted_agents import ArtifactStore, Pipeline, QuantedAgent


# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResearchQuery(BaseModel):
    """Input for the research pipeline.

    Attributes:
        topic: The research topic to investigate.
        depth: How many key points to extract.
    """

    topic: str
    depth: int = 5


class ResearchFindings(BaseModel):
    """Intermediate output from the researcher agent.

    Attributes:
        findings: List of key research findings.
        key_insight: The single most important insight.
    """

    findings: list[str]
    key_insight: str


class FinalReport(BaseModel):
    """Final output from the reporter agent.

    Attributes:
        title: Report title.
        executive_summary: A concise executive summary.
        sections: List of report sections.
    """

    title: str
    executive_summary: str
    sections: list[str]


class AssembledDeliverable(BaseModel):
    """Assembled output combining both pipeline stages.

    Attributes:
        title: The report title from the writer stage.
        executive_summary: The executive summary from the writer stage.
        key_insight: The key insight from the research stage.
        finding_count: Number of research findings produced.
        sections: Report sections from the writer stage.
    """

    title: str
    executive_summary: str
    key_insight: str
    finding_count: int
    sections: list[str]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = QuantedAgent(
    MODEL,
    input_type=ResearchQuery,
    output_type=ResearchFindings,
    system_prompt=(
        "You are a research specialist. Investigate the given topic and "
        "return structured findings with a key insight. Be thorough but concise."
    ),
)

reporter = QuantedAgent(
    MODEL,
    input_type=ResearchFindings,
    output_type=FinalReport,
    system_prompt=(
        "You are a technical writer. Take research findings and produce a "
        "polished report with a clear title, executive summary, and sections."
    ),
)


# ---------------------------------------------------------------------------
# Assembly function
# ---------------------------------------------------------------------------

def assemble_deliverable(
    store: ArtifactStore, result: object
) -> AssembledDeliverable:
    """Combine outputs from both pipeline stages into a single deliverable.

    The store contains outputs keyed by step index and output type name:
    - step_0/researchfindings: the researcher's output
    - step_1/finalreport: the reporter's output

    Args:
        store: The pipeline's ArtifactStore with all step outputs.
        result: The last step's QuantedResult (not used here).

    Returns:
        An AssembledDeliverable combining data from both stages.
    """
    research = store["step_0/researchfindings"]
    report = store["step_1/finalreport"]
    return AssembledDeliverable(
        title=report.title,
        executive_summary=report.executive_summary,
        key_insight=research.key_insight,
        finding_count=len(research.findings),
        sections=report.sections,
    )


# ---------------------------------------------------------------------------
# Pipeline with store and assembly
# ---------------------------------------------------------------------------

store = ArtifactStore()

pipeline = Pipeline(
    steps=[researcher, reporter],
    assembly=assemble_deliverable,
    store=store,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the dual-stream pipeline and inspect results."""
    result = await pipeline.run(
        ResearchQuery(topic="impact of large language models on software engineering", depth=5)
    )

    # Access the assembled output (from assembly function)
    print("=== Assembled Deliverable ===")
    print(f"Title: {result.data.title}")
    print(f"Executive Summary: {result.data.executive_summary[:150]}...")
    print(f"Key Insight: {result.data.key_insight}")
    print(f"Finding Count: {result.data.finding_count}")
    print(f"Sections: {len(result.data.sections)}")

    # Access the natural language summary (provider-dependent, may be None)
    print("\n=== Summary (Dual-Stream) ===")
    print(f"Summary available: {result.summary is not None}")
    if result.summary:
        print(f"Summary: {result.summary[:200]}...")

    # Inspect store contents -- both stage outputs are preserved
    print("\n=== ArtifactStore Contents ===")
    print(f"Store keys: {list(store.keys())}")
    if "step_0/researchfindings" in store:
        findings = store["step_0/researchfindings"]
        print(f"Research findings: {findings.findings[:2]}...")
    if "step_1/finalreport" in store:
        report = store["step_1/finalreport"]
        print(f"Report title: {report.title}")

    # Usage across the entire pipeline
    print("\n=== Pipeline Usage ===")
    print(f"Total input tokens: {result.total_usage.input_tokens}")
    print(f"Total output tokens: {result.total_usage.output_tokens}")
    print(f"Total requests: {result.total_usage.requests}")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")


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
            "Then run: python examples/09_dual_stream.py"
        )
        sys.exit(1)

    asyncio.run(main())

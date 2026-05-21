"""Trace Logging Example: Pipeline with File Traces.

Demonstrates the traces_path parameter on run() for writing real-time JSONL
trace files. A two-step Pipeline (researcher -> writer) runs with trace
logging enabled, and the resulting JSONL file is read back and displayed.

The traces_path kwarg works with all workflow types (Pipeline, Router, Loop,
Parallel) and single agents. When set, it creates a timestamped .jsonl file
in the specified directory with one JSON line per trace entry, written with
flush+fsync for crash safety. The in-memory result.trace is unaffected.
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from quanted_agents import Pipeline, QuantedAgent


class ResearchQuery(BaseModel):
    """Input for the research pipeline.

    Attributes:
        topic: The research topic to investigate.
    """

    topic: str


class ResearchNotes(BaseModel):
    """Intermediate output from the researcher agent.

    Attributes:
        findings: List of key findings from the research.
        key_insight: The single most important insight discovered.
    """

    findings: list[str]
    key_insight: str


class FinalReport(BaseModel):
    """Final output from the writer agent.

    Attributes:
        title: The title of the report.
        content: The full report content.
        sources: List of sources or topics referenced.
    """

    title: str
    content: str
    sources: list[str]


researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchQuery,
    output_type=ResearchNotes,
    system_prompt=(
        "You are a research specialist. Investigate the given topic and "
        "return structured findings with a key insight. Be thorough but concise."
    ),
)

writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchNotes,
    output_type=FinalReport,
    system_prompt=(
        "You are a technical writer. Take research notes and produce a "
        "polished report with a clear title, well-structured content, "
        "and a list of sources referenced."
    ),
)

pipeline = Pipeline(steps=[researcher, writer])


async def main() -> None:
    """Run a research pipeline with trace file logging."""
    with tempfile.TemporaryDirectory() as tmp:
        traces_dir = Path(tmp)

        # Run the pipeline with traces_path to enable JSONL trace logging
        result = await pipeline.run(
            ResearchQuery(topic="quantum computing applications in cryptography"),
            traces_path=traces_dir,
        )

        # Print the structured output
        print("=== Final Report ===")
        print(f"Title: {result.data.title}")
        print(f"Content (first 200 chars): {result.data.content[:200]}...")
        print(f"Sources: {', '.join(result.data.sources)}")

        # Find and read the JSONL trace file
        trace_files = list(traces_dir.glob("*.jsonl"))
        print("\n=== Trace File ===")
        print(f"Trace files found: {len(trace_files)}")

        if trace_files:
            trace_file = trace_files[0]
            print(f"File: {trace_file.name}")

            lines = trace_file.read_text().strip().split("\n")
            print(f"Entries in file: {len(lines)}")

            for i, line in enumerate(lines):
                entry = json.loads(line)
                step_name = entry.get("step_name", "unknown")
                duration = entry.get("timing", {}).get("duration_seconds", 0)
                input_keys = list(entry.get("input_data", {}).keys())
                output_keys = list(entry.get("output_data", {}).keys())
                print(f"\n  Entry {i}:")
                print(f"    Step: {step_name}")
                print(f"    Duration: {duration:.2f}s")
                print(f"    Input keys: {input_keys}")
                print(f"    Output keys: {output_keys}")

        # Show in-memory trace (both are available simultaneously)
        print("\n=== In-Memory Trace ===")
        print(f"Trace entries: {len(result.trace)}")
        for entry in result.trace:
            print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")

        # Total usage across all pipeline steps
        print("\n=== Total Usage ===")
        print(f"Input tokens: {result.total_usage.input_tokens}")
        print(f"Output tokens: {result.total_usage.output_tokens}")
        print(f"Total requests: {result.total_usage.requests}")


if __name__ == "__main__":
    asyncio.run(main())

"""Trace Sessions Example: Correlating Multiple Agent Runs.

Demonstrates TraceSession for consolidating multiple agent runs into a
single trace file with a shared session ID. All runs within the session
write to the same JSONL file, making it easy to correlate related
operations (e.g., a multi-step workflow where each step is a separate
agent.run() call).

TraceSession generates a UUID session_id that is attached to every trace
entry, enabling filtering and grouping in log analysis tools.

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., OPENAI_API_KEY).
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from quanted_agents import QuantedAgent, TraceSession


# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Question(BaseModel):
    """Input for the Q&A agent.

    Attributes:
        question: The question to answer.
    """

    question: str


class Answer(BaseModel):
    """Output from the Q&A agent.

    Attributes:
        answer: The answer text.
        confidence: Confidence level from 0.0 to 1.0.
    """

    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = QuantedAgent(
    MODEL,
    input_type=Question,
    output_type=Answer,
    system_prompt="Answer questions concisely and accurately.",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run multiple agent calls within a TraceSession and inspect the trace file."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_file = Path(tmp) / "session_traces.jsonl"

        # All runs within the session write to the same trace file
        async with TraceSession(trace_file) as session:
            print(f"Session ID: {session.session_id}")
            print(f"Trace file: {session.file_path}")
            print()

            # Run 1: First question
            r1 = await agent.run(
                Question(question="What is the capital of France?"),
                trace_session=session,
            )
            print(f"Q1 Answer: {r1.data.answer}")
            print(f"Q1 Confidence: {r1.data.confidence}")

            # Run 2: Follow-up question
            r2 = await agent.run(
                Question(question="What is the population of that city?"),
                trace_session=session,
            )
            print(f"Q2 Answer: {r2.data.answer}")
            print(f"Q2 Confidence: {r2.data.confidence}")

            # Run 3: Different topic
            r3 = await agent.run(
                Question(question="What programming language was created by Guido van Rossum?"),
                trace_session=session,
            )
            print(f"Q3 Answer: {r3.data.answer}")
            print(f"Q3 Confidence: {r3.data.confidence}")

        # Session complete -- inspect the trace file
        print("\n=== Trace File Contents ===")
        trace_content = trace_file.read_text().strip()
        lines = trace_content.split("\n")
        print(f"Total trace entries: {len(lines)}")

        for i, line in enumerate(lines):
            entry = json.loads(line)
            step_name = entry.get("step_name", "unknown")
            session_id = entry.get("session_id", "none")
            duration = entry.get("timing", {}).get("duration_seconds", 0)
            input_data = entry.get("input_data", {})
            print(f"\n  Entry {i}:")
            print(f"    Step: {step_name}")
            print(f"    Session ID: {session_id}")
            print(f"    Duration: {duration:.2f}s")
            print(f"    Input: {input_data}")

        # Verify all entries share the same session_id
        session_ids = set()
        for line in lines:
            entry = json.loads(line)
            sid = entry.get("session_id")
            if sid:
                session_ids.add(sid)

        print(f"\n=== Session Correlation ===")
        print(f"Unique session IDs: {len(session_ids)}")
        print(f"All entries correlated: {len(session_ids) <= 1}")
        print(f"Session ID: {session.session_id}")

        # Usage summary across all runs
        print("\n=== Usage Summary ===")
        total_input = r1.usage.input_tokens + r2.usage.input_tokens + r3.usage.input_tokens
        total_output = r1.usage.output_tokens + r2.usage.output_tokens + r3.usage.output_tokens
        total_requests = r1.usage.requests + r2.usage.requests + r3.usage.requests
        print(f"Total input tokens: {total_input}")
        print(f"Total output tokens: {total_output}")
        print(f"Total requests: {total_requests}")


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
            "Then run: python examples/12_trace_sessions.py"
        )
        sys.exit(1)

    asyncio.run(main())

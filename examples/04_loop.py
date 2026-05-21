"""Loop Example: Essay Refiner.

Demonstrates a Loop workflow that iteratively refines an essay until a quality
threshold is met or the maximum number of iterations is reached.

The Loop flow is:
1. Body agent receives the current essay and returns a refined version
2. Termination check evaluates if the quality score meets the threshold
3. If not converged, the refined essay is fed back as input for the next iteration
4. Loop raises MaxIterationsExceeded if it does not converge within the iteration budget
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio

from pydantic import BaseModel

from quanted_agents import Loop, MaxIterationsExceeded, QuantedAgent


class Essay(BaseModel):
    """An essay with quality metadata for the refinement loop.

    Attributes:
        content: The essay text.
        quality_score: Self-assessed quality score from 0.0 to 1.0.
        revision_notes: Notes about what was changed in this revision.
    """

    content: str
    quality_score: float = 0.0
    revision_notes: str = ""


refiner = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Essay,
    output_type=Essay,
    system_prompt=(
        "You are an expert essay editor. Given an essay with a quality score and "
        "revision notes, improve the essay and increase the quality score. Focus on:\n"
        "- Clarity and coherence of arguments\n"
        "- Grammar and style improvements\n"
        "- Stronger opening and closing paragraphs\n"
        "- Better transitions between ideas\n\n"
        "Set quality_score to reflect your honest assessment (0.0 to 1.0). "
        "A score of 0.9 or above means the essay is publication-ready. "
        "Include revision_notes explaining what you changed."
    ),
)

loop = Loop(
    body=refiner,
    termination_check=lambda essay: essay.quality_score >= 0.9,
    max_iterations=5,
)


async def main() -> None:
    """Run the essay refiner loop on a rough draft."""
    initial_essay = Essay(
        content=(
            "AI is good for healthcare. It can find diseases early. Robots help with "
            "surgery. Some people worry about privacy. AI is changing how doctors work. "
            "Machine learning is a type of AI that learns from data. Hospitals are using "
            "more AI tools. The future of healthcare will have lots of AI."
        ),
        quality_score=0.3,
        revision_notes="Initial rough draft -- needs significant improvement.",
    )

    try:
        result = await loop.run(initial_essay)
    except MaxIterationsExceeded:
        print("Loop did not converge within 5 iterations")
        return

    # Access the final refined essay
    print("=== Final Essay ===")
    print(f"Quality Score: {result.data.quality_score}")
    print(f"Revision Notes: {result.data.revision_notes}")
    print(f"Content (first 300 chars): {result.data.content[:300]}...")

    # Observability: aggregated usage across all iterations
    print("\n=== Aggregated Usage ===")
    print(f"Total input tokens: {result.total_usage.input_tokens}")
    print(f"Total output tokens: {result.total_usage.output_tokens}")
    print(f"Total requests: {result.total_usage.requests}")

    # Observability: per-iteration timing
    print("\n=== Iteration Timings ===")
    iterations = len(result.step_timings)
    print(f"Number of iterations: {iterations}")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")

    # Observability: full execution trace
    print("\n=== Trace ===")
    for entry in result.trace:
        print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

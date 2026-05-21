"""Skills and Feedback Example: Context-Aware Agent.

Demonstrates a QuantedAgent with skills_path and feedback_path for on-demand
context loading. The agent's LLM sees a catalog of available skills and feedback
in its system prompt and can load their full content via an internal tool.

This example is self-contained using temporary directories. The flow is:
1. Create temp directories with skill and feedback markdown files
2. Create an agent with skills_path and feedback_path
3. Run the agent -- the LLM sees the catalog and can load context
4. Demonstrate add_feedback() for programmatic feedback creation
5. Print observability data (usage, trace)

The equivalent manual directory structure would be:

    project/
        skills/
            error-handling.md    # YAML frontmatter with name + description
        feedback/
            be-concise.md        # YAML frontmatter with name + description
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from quanted_agents import QuantedAgent


class CodingQuestion(BaseModel):
    """Input for the coding assistant agent.

    Attributes:
        question: The coding question to answer.
    """

    question: str


class CodingAnswer(BaseModel):
    """Structured output from the coding assistant agent.

    Attributes:
        answer: The answer to the coding question.
        references: List of skills or feedback items referenced in the answer.
    """

    answer: str
    references: list[str]


async def main() -> None:
    """Run a context-aware coding assistant with skills and feedback."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create a skills directory with one skill file
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "error-handling.md").write_text(
            "---\n"
            "name: error-handling\n"
            "description: Best practices for Python error handling\n"
            "---\n"
            "## Error Handling Guide\n"
            "- Use specific exception types, not bare except\n"
            "- Log errors with context using f-strings\n"
            "- Use try/except/finally for cleanup\n"
        )

        # Create a feedback directory with one feedback file
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()
        (feedback_dir / "be-concise.md").write_text(
            "---\n"
            "name: be-concise\n"
            "description: Keep answers short and actionable\n"
            "---\n"
            "Provide concise answers. Focus on code examples over lengthy explanations.\n"
        )

        # Create the agent with skills and feedback paths
        agent = QuantedAgent(
            "openai:gpt-4o-mini",
            input_type=CodingQuestion,
            output_type=CodingAnswer,
            system_prompt=(
                "You are a Python coding assistant. Use available skills and "
                "feedback context when answering coding questions. Load any "
                "relevant skills or feedback before formulating your response."
            ),
            skills_path=skills_dir,
            feedback_path=feedback_dir,
        )

        # Run the agent with a coding question
        result = await agent.run(
            CodingQuestion(question="How should I handle file I/O errors in Python?")
        )

        # Print the structured output
        print("=== Coding Answer ===")
        print(f"Answer: {result.data.answer}")
        print(f"References: {', '.join(result.data.references)}")

        # Demonstrate add_feedback() for programmatic feedback creation
        agent.add_feedback(
            name="use-type-hints",
            content="Always include type hints in code examples.",
            description="Feedback on code style",
        )
        print("\n=== Feedback Added ===")
        print("Added 'use-type-hints' feedback programmatically")

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
            tool_names = [tc["tool_name"] for tc in entry.tool_calls]
            if tool_names:
                print(f"Tools called: {', '.join(tool_names)}")


if __name__ == "__main__":
    asyncio.run(main())

"""Integration tests for QuantedAgent context loading (skills and feedback).

Validates that QuantedAgent correctly integrates with ContextManager when
skills_path and feedback_path are configured: system prompt augmentation,
_load_context tool injection, tool collision detection, add_feedback(),
zero-overhead when unconfigured, and directory validation.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from tests.conftest import SampleInput, SampleOutput


def create_context_file(
    directory: Path, name: str, description: str, body: str
) -> Path:
    """Write a markdown file with YAML frontmatter to a directory.

    Args:
        directory: The directory to create the file in.
        name: The name field for YAML frontmatter.
        description: The description field for YAML frontmatter.
        body: The markdown body content.

    Returns:
        The path to the created file.
    """
    slug = name.lower().replace(" ", "-")
    filepath = directory / f"{slug}.md"
    content = (
        f"---\nname: {name}\ndescription: {description}\n---\n{body}"
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


class TestAgentContextInit(unittest.TestCase):
    """Tests for QuantedAgent construction with context paths."""

    def setUp(self) -> None:
        """Create temporary directories with skill and feedback files."""
        self._skills_dir = tempfile.TemporaryDirectory()
        self._feedback_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._skills_dir.cleanup)
        self.addCleanup(self._feedback_dir.cleanup)

        self.skills_path: Path = Path(self._skills_dir.name)
        self.feedback_path: Path = Path(self._feedback_dir.name)

        create_context_file(
            self.skills_path, "code-review", "How to review code", "Review code carefully."
        )
        create_context_file(
            self.skills_path, "testing", "How to write tests", "Write thorough tests."
        )
        create_context_file(
            self.feedback_path, "tone-feedback", "Feedback on tone", "Keep tone professional."
        )

    def test_agent_with_skills_path(self) -> None:
        """Agent with skills_path has skill names in system prompt."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
        )
        prompts = agent.inner._system_prompts
        prompt_text = " ".join(str(p) for p in prompts)
        self.assertIn("Available Skills", prompt_text)
        self.assertIn("code-review", prompt_text)
        self.assertIn("testing", prompt_text)

    def test_agent_with_feedback_path(self) -> None:
        """Agent with feedback_path has feedback names in system prompt."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            feedback_path=self.feedback_path,
        )
        prompts = agent.inner._system_prompts
        prompt_text = " ".join(str(p) for p in prompts)
        self.assertIn("Available Feedback", prompt_text)
        self.assertIn("tone-feedback", prompt_text)

    def test_agent_with_both_paths(self) -> None:
        """Agent with both skills_path and feedback_path has both catalog sections."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
            feedback_path=self.feedback_path,
        )
        prompts = agent.inner._system_prompts
        prompt_text = " ".join(str(p) for p in prompts)
        self.assertIn("Available Skills", prompt_text)
        self.assertIn("Available Feedback", prompt_text)

    def test_agent_without_context_paths(self) -> None:
        """Agent without context paths has no catalog or _load_context tool."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Plain prompt",
        )
        prompts = agent.inner._system_prompts
        prompt_text = " ".join(str(p) for p in prompts)
        self.assertNotIn("Available Skills", prompt_text)
        self.assertNotIn("Available Feedback", prompt_text)

        tool_names = list(agent.inner._function_toolset.tools.keys())
        self.assertNotIn("_load_context", tool_names)

    def test_agent_nonexistent_skills_path_raises(self) -> None:
        """Agent with non-existent skills_path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            QuantedAgent(
                "test",
                input_type=SampleInput,
                output_type=SampleOutput,
                skills_path="/tmp/nonexistent_skills_dir_12345",
            )

    def test_agent_nonexistent_feedback_path_raises(self) -> None:
        """Agent with non-existent feedback_path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            QuantedAgent(
                "test",
                input_type=SampleInput,
                output_type=SampleOutput,
                feedback_path="/tmp/nonexistent_feedback_dir_12345",
            )

    def test_agent_tool_name_collision_raises(self) -> None:
        """Agent with user tool named _load_context and skills_path raises ValueError."""
        def _load_context(query: str) -> str:
            """User-defined tool that collides with reserved name.

            Args:
                query: A search query.

            Returns:
                Search result string.
            """
            return f"result: {query}"

        with self.assertRaises(ValueError) as ctx:
            QuantedAgent(
                "test",
                input_type=SampleInput,
                output_type=SampleOutput,
                tools=[_load_context],
                skills_path=self.skills_path,
            )
        self.assertIn("reserved", str(ctx.exception))

    def test_agent_no_collision_without_context(self) -> None:
        """User tool named _load_context is allowed when no context paths are set."""
        def _load_context(query: str) -> str:
            """User-defined tool with reserved name, allowed without context.

            Args:
                query: A search query.

            Returns:
                Search result string.
            """
            return f"result: {query}"

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            tools=[_load_context],
        )
        self.assertIsNotNone(agent)


class TestAgentContextToolInjection(unittest.TestCase):
    """Tests for _load_context tool injection behavior."""

    def setUp(self) -> None:
        """Create temporary directory with skill files."""
        self._skills_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._skills_dir.cleanup)
        self.skills_path: Path = Path(self._skills_dir.name)

        create_context_file(
            self.skills_path, "my-skill", "A test skill", "Skill content here."
        )

    def test_load_context_tool_injected(self) -> None:
        """Agent with skills_path has _load_context in its tool list."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
        )
        tool_names = list(agent.inner._function_toolset.tools.keys())
        self.assertIn("_load_context", tool_names)

    def test_load_context_tool_not_injected_without_paths(self) -> None:
        """Agent without context paths does not have _load_context tool."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        tool_names = list(agent.inner._function_toolset.tools.keys())
        self.assertNotIn("_load_context", tool_names)


class TestAgentContextRun(unittest.IsolatedAsyncioTestCase):
    """Tests for agent execution with context loading enabled."""

    def setUp(self) -> None:
        """Create temporary directory with skill files."""
        self._skills_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._skills_dir.cleanup)
        self.skills_path: Path = Path(self._skills_dir.name)

        create_context_file(
            self.skills_path, "greet-skill", "How to greet users", "Say hello warmly."
        )

    async def test_agent_run_with_skills_path(self) -> None:
        """Agent with skills_path runs successfully with TestModel."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="hello"))
            self.assertIsInstance(result.data, SampleOutput)

    async def test_shared_skills_path(self) -> None:
        """Two agents sharing the same skills_path both see skill names."""
        agent1 = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
        )
        agent2 = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=self.skills_path,
        )

        for agent in [agent1, agent2]:
            prompts = agent.inner._system_prompts
            prompt_text = " ".join(str(p) for p in prompts)
            self.assertIn("greet-skill", prompt_text)


class TestAgentAddFeedback(unittest.TestCase):
    """Tests for the add_feedback() method on QuantedAgent."""

    def setUp(self) -> None:
        """Create temporary feedback directory."""
        self._feedback_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._feedback_dir.cleanup)
        self.feedback_path: Path = Path(self._feedback_dir.name)

    def test_add_feedback_success(self) -> None:
        """add_feedback creates a file with correct frontmatter in the feedback directory."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            feedback_path=self.feedback_path,
        )
        agent.add_feedback("test-feedback", "content here", "A test feedback")

        md_files = list(self.feedback_path.glob("*.md"))
        self.assertEqual(len(md_files), 1)

        file_text = md_files[0].read_text(encoding="utf-8")
        self.assertIn("test-feedback", file_text)
        self.assertIn("A test feedback", file_text)
        self.assertIn("content here", file_text)

    def test_add_feedback_without_feedback_path_raises(self) -> None:
        """add_feedback raises ValueError when only skills_path is configured."""
        skills_dir = tempfile.TemporaryDirectory()
        self.addCleanup(skills_dir.cleanup)
        skills_path = Path(skills_dir.name)
        create_context_file(skills_path, "skill-1", "A skill", "Skill body.")

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            skills_path=skills_path,
        )
        with self.assertRaises(ValueError):
            agent.add_feedback("fb", "content", "desc")

    def test_add_feedback_without_any_context_raises(self) -> None:
        """add_feedback raises ValueError when no context paths are configured."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
        )
        with self.assertRaises(ValueError):
            agent.add_feedback("fb", "content", "desc")

    def test_add_feedback_duplicate_auto_suffix(self) -> None:
        """Adding feedback with the same name twice gives the second a _1 suffix."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            feedback_path=self.feedback_path,
        )
        agent.add_feedback("my-feedback", "first content", "First feedback")
        agent.add_feedback("my-feedback", "second content", "Second feedback")

        md_files = list(self.feedback_path.glob("*.md"))
        self.assertEqual(len(md_files), 2)

        file_texts = [f.read_text(encoding="utf-8") for f in md_files]
        combined = " ".join(file_texts)
        self.assertIn("my-feedback_1", combined)


if __name__ == "__main__":
    unittest.main()

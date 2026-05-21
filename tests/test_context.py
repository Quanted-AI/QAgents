"""Unit tests for the ContextManager context loading system.

Tests cover directory scanning, validation, catalog building, content loading
with fuzzy matching, and atomic feedback file writing.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quanted_agents.context import ContextManager


class ContextTestHelper:
    """Mixin providing helper methods for writing test markdown files."""

    def _write_md(
        self, directory: Path, filename: str, name: str, description: str, body: str
    ) -> Path:
        """Write a markdown file with YAML frontmatter.

        Args:
            directory: Directory to write the file in.
            filename: Name of the file (e.g., "skill.md").
            name: The frontmatter 'name' field.
            description: The frontmatter 'description' field.
            body: The markdown body content.

        Returns:
            The path to the written file.
        """
        filepath = directory / filename
        content = f"---\nname: {name}\ndescription: {description}\n---\n{body}"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def _write_raw(self, directory: Path, filename: str, content: str) -> Path:
        """Write a raw file with arbitrary content.

        Args:
            directory: Directory to write the file in.
            filename: Name of the file.
            content: Raw file content.

        Returns:
            The path to the written file.
        """
        filepath = directory / filename
        filepath.write_text(content, encoding="utf-8")
        return filepath


class TestContextManagerScanning(unittest.TestCase, ContextTestHelper):
    """Tests for directory scanning and markdown file parsing."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self._tmpdir.name) / "skills"
        self.skills_dir.mkdir()
        self.feedback_dir = Path(self._tmpdir.name) / "feedback"
        self.feedback_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_scan_skills_directory(self) -> None:
        """Two valid skill files should be scanned and stored with correct fields."""
        self._write_md(self.skills_dir, "error-handling.md", "error-handling",
                       "How to handle errors", "Use try/except blocks.")
        self._write_md(self.skills_dir, "logging.md", "logging",
                       "Structured logging guide", "Use structured logging.")

        cm = ContextManager(skills_path=self.skills_dir)

        self.assertEqual(len(cm._skills), 2)
        self.assertIn("error-handling", cm._skills)
        self.assertIn("logging", cm._skills)
        self.assertEqual(cm._skills["error-handling"].name, "error-handling")
        self.assertEqual(cm._skills["error-handling"].description, "How to handle errors")
        self.assertEqual(cm._skills["error-handling"].content, "Use try/except blocks.")
        self.assertEqual(cm._skills["error-handling"].source, "skill")

    def test_scan_feedback_directory(self) -> None:
        """Two valid feedback files should be scanned and stored."""
        self._write_md(self.feedback_dir, "review-1.md", "code-review",
                       "Review feedback", "Fix the imports.")
        self._write_md(self.feedback_dir, "api-notes.md", "api-design",
                       "API design notes", "Use REST conventions.")

        cm = ContextManager(feedback_path=self.feedback_dir)

        self.assertEqual(len(cm._feedback), 2)
        self.assertIn("code-review", cm._feedback)
        self.assertIn("api-design", cm._feedback)
        self.assertEqual(cm._feedback["code-review"].source, "feedback")

    def test_scan_ignores_non_md_files(self) -> None:
        """Non-.md files should be silently ignored."""
        self._write_md(self.skills_dir, "valid.md", "valid-skill",
                       "A valid skill", "Content here.")
        self._write_raw(self.skills_dir, "notes.txt", "some text file")
        self._write_raw(self.skills_dir, "data.json", '{"key": "value"}')

        cm = ContextManager(skills_path=self.skills_dir)

        self.assertEqual(len(cm._skills), 1)
        self.assertIn("valid-skill", cm._skills)

    def test_scan_skips_malformed_yaml(self) -> None:
        """Files with invalid YAML frontmatter should be skipped with a warning."""
        self._write_md(self.skills_dir, "good.md", "good-skill",
                       "A good skill", "Good content.")
        self._write_raw(self.skills_dir, "bad.md",
                        "---\nname: foo\nbad: : : invalid\n---\nBody text")

        with patch("quanted_agents.context.logger") as mock_logger:
            cm = ContextManager(skills_path=self.skills_dir)

        self.assertEqual(len(cm._skills), 1)
        self.assertIn("good-skill", cm._skills)
        mock_logger.warning.assert_called()

    def test_scan_skips_missing_name_field(self) -> None:
        """Files missing 'name' in frontmatter should be skipped."""
        self._write_raw(self.skills_dir, "no-name.md",
                        "---\ndescription: Has desc but no name\n---\nBody")
        self._write_md(self.skills_dir, "valid.md", "valid",
                       "Valid skill", "Content.")

        with patch("quanted_agents.context.logger") as mock_logger:
            cm = ContextManager(skills_path=self.skills_dir)

        self.assertEqual(len(cm._skills), 1)
        self.assertIn("valid", cm._skills)
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        found_name_warning = any("missing 'name'" in msg for msg in warning_messages)
        self.assertTrue(found_name_warning, f"Expected warning about missing 'name', got: {warning_messages}")

    def test_scan_skips_missing_description_field(self) -> None:
        """Files missing 'description' in frontmatter should be skipped."""
        self._write_raw(self.skills_dir, "no-desc.md",
                        "---\nname: no-desc-skill\n---\nBody")
        self._write_md(self.skills_dir, "valid.md", "valid",
                       "Valid skill", "Content.")

        with patch("quanted_agents.context.logger") as mock_logger:
            cm = ContextManager(skills_path=self.skills_dir)

        self.assertEqual(len(cm._skills), 1)
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        found_desc_warning = any("missing 'description'" in msg for msg in warning_messages)
        self.assertTrue(found_desc_warning,
                        f"Expected warning about missing 'description', got: {warning_messages}")

    def test_scan_empty_directory(self) -> None:
        """An empty directory should result in zero items without errors."""
        cm = ContextManager(skills_path=self.skills_dir)
        self.assertEqual(len(cm._skills), 0)
        self.assertFalse(cm.has_items)

    def test_case_insensitive_name_storage(self) -> None:
        """Names should be stored with lowercase keys."""
        self._write_md(self.skills_dir, "error-handling.md", "Error-Handling",
                       "Error handling guide", "Content here.")

        cm = ContextManager(skills_path=self.skills_dir)

        self.assertIn("error-handling", cm._skills)
        self.assertEqual(cm._skills["error-handling"].name, "Error-Handling")


class TestContextManagerValidation(unittest.TestCase, ContextTestHelper):
    """Tests for directory validation and name collision detection."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self._tmpdir.name) / "skills"
        self.skills_dir.mkdir()
        self.feedback_dir = Path(self._tmpdir.name) / "feedback"
        self.feedback_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_nonexistent_skills_path_raises(self) -> None:
        """A non-existent skills_path should raise FileNotFoundError."""
        fake_path = Path(self._tmpdir.name) / "nonexistent"
        with self.assertRaises(FileNotFoundError) as ctx:
            ContextManager(skills_path=fake_path)
        self.assertIn("skills_path", str(ctx.exception))

    def test_nonexistent_feedback_path_raises(self) -> None:
        """A non-existent feedback_path should raise FileNotFoundError."""
        fake_path = Path(self._tmpdir.name) / "nonexistent"
        with self.assertRaises(FileNotFoundError) as ctx:
            ContextManager(feedback_path=fake_path)
        self.assertIn("feedback_path", str(ctx.exception))

    def test_name_collision_across_skills_and_feedback(self) -> None:
        """Same name in skills and feedback should raise ValueError."""
        self._write_md(self.skills_dir, "error-handling.md", "error-handling",
                       "Skill version", "Skill content.")
        self._write_md(self.feedback_dir, "error-handling.md", "Error-Handling",
                       "Feedback version", "Feedback content.")

        with self.assertRaises(ValueError) as ctx:
            ContextManager(skills_path=self.skills_dir, feedback_path=self.feedback_dir)
        self.assertIn("collision", str(ctx.exception).lower())


class TestContextManagerCatalog(unittest.TestCase, ContextTestHelper):
    """Tests for catalog building."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self._tmpdir.name) / "skills"
        self.skills_dir.mkdir()
        self.feedback_dir = Path(self._tmpdir.name) / "feedback"
        self.feedback_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_catalog_with_skills_only(self) -> None:
        """Catalog with only skills should contain skills section and no feedback section."""
        self._write_md(self.skills_dir, "skill1.md", "skill-1",
                       "First skill", "Content 1.")

        cm = ContextManager(skills_path=self.skills_dir)
        catalog = cm.build_catalog()

        self.assertIn("## Available Skills", catalog)
        self.assertNotIn("## Available Feedback", catalog)

    def test_catalog_with_feedback_only(self) -> None:
        """Catalog with only feedback should contain feedback section and no skills section."""
        self._write_md(self.feedback_dir, "fb1.md", "feedback-1",
                       "First feedback", "Content 1.")

        cm = ContextManager(feedback_path=self.feedback_dir)
        catalog = cm.build_catalog()

        self.assertIn("## Available Feedback", catalog)
        self.assertNotIn("## Available Skills", catalog)

    def test_catalog_with_both(self) -> None:
        """Catalog with both should have skills section before feedback section."""
        self._write_md(self.skills_dir, "skill1.md", "skill-1",
                       "A skill", "Content.")
        self._write_md(self.feedback_dir, "fb1.md", "feedback-1",
                       "A feedback", "Content.")

        cm = ContextManager(skills_path=self.skills_dir, feedback_path=self.feedback_dir)
        catalog = cm.build_catalog()

        skills_pos = catalog.index("## Available Skills")
        feedback_pos = catalog.index("## Available Feedback")
        self.assertLess(skills_pos, feedback_pos)

    def test_catalog_lists_names_and_descriptions(self) -> None:
        """Each item should appear as '- **name**: description' in the catalog."""
        self._write_md(self.skills_dir, "skill1.md", "My-Skill",
                       "Does something useful", "Content.")

        cm = ContextManager(skills_path=self.skills_dir)
        catalog = cm.build_catalog()

        self.assertIn("- **My-Skill**: Does something useful", catalog)

    def test_catalog_empty_when_no_items(self) -> None:
        """Catalog should return empty string when no items exist."""
        cm = ContextManager(skills_path=self.skills_dir)
        catalog = cm.build_catalog()

        self.assertEqual(catalog, "")


class TestContextManagerLoad(unittest.TestCase, ContextTestHelper):
    """Tests for content loading with fuzzy matching."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self._tmpdir.name) / "skills"
        self.skills_dir.mkdir()
        self.feedback_dir = Path(self._tmpdir.name) / "feedback"
        self.feedback_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_load_single_name(self) -> None:
        """Loading a single valid name should return its body content."""
        self._write_md(self.skills_dir, "error-handling.md", "error-handling",
                       "Error guide", "Use try/except blocks.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["error-handling"])

        self.assertIn("Use try/except blocks.", result)

    def test_load_multiple_names(self) -> None:
        """Loading multiple names should return content separated by '---'."""
        self._write_md(self.skills_dir, "skill1.md", "skill-1",
                       "First", "Content one.")
        self._write_md(self.skills_dir, "skill2.md", "skill-2",
                       "Second", "Content two.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["skill-1", "skill-2"])

        self.assertIn("Content one.", result)
        self.assertIn("Content two.", result)
        self.assertIn("---", result)

    def test_load_case_insensitive(self) -> None:
        """Loading should be case-insensitive."""
        self._write_md(self.skills_dir, "error-handling.md", "error-handling",
                       "Error guide", "Case insensitive content.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["ERROR-HANDLING"])

        self.assertIn("Case insensitive content.", result)

    def test_load_missing_name_with_suggestions(self) -> None:
        """Loading a missing name should return error with fuzzy suggestions."""
        self._write_md(self.skills_dir, "error-handling.md", "error-handling",
                       "Error guide", "Content.")
        self._write_md(self.skills_dir, "error-guide.md", "error-guide",
                       "Another guide", "Content.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["error-handlng"])

        self.assertIn("Couldn't find", result)
        self.assertIn("Did you mean", result)

    def test_load_returns_body_only(self) -> None:
        """Loaded content should not contain frontmatter delimiters or fields."""
        self._write_md(self.skills_dir, "skill.md", "my-skill",
                       "A skill desc", "Body content only here.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["my-skill"])

        # The result should contain the body but not frontmatter fields
        self.assertIn("Body content only here.", result)
        # The result should have the header with the name, then body
        # It should NOT contain frontmatter-style fields
        self.assertNotIn("description: A skill desc", result)

    def test_load_mixed_found_and_missing(self) -> None:
        """Loading with both valid and invalid names should return both results."""
        self._write_md(self.skills_dir, "skill.md", "real-skill",
                       "A real skill", "Real content.")

        cm = ContextManager(skills_path=self.skills_dir)
        result = cm.load(["real-skill", "nonexistent"])

        self.assertIn("Real content.", result)
        self.assertIn("Couldn't find 'nonexistent'", result)


class TestContextManagerAddFeedback(unittest.TestCase, ContextTestHelper):
    """Tests for atomic feedback file writing."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self._tmpdir.name) / "skills"
        self.skills_dir.mkdir()
        self.feedback_dir = Path(self._tmpdir.name) / "feedback"
        self.feedback_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_add_feedback_creates_file(self) -> None:
        """add_feedback should create a file on disk with correct frontmatter and content."""
        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("test-feedback", "Feedback body content.", "A test feedback")

        files = list(self.feedback_dir.glob("*.md"))
        self.assertEqual(len(files), 1)

        file_content = files[0].read_text(encoding="utf-8")
        self.assertIn("name: test-feedback", file_content)
        self.assertIn("description: A test feedback", file_content)
        self.assertIn("Feedback body content.", file_content)

    def test_add_feedback_available_immediately(self) -> None:
        """Added feedback should be loadable immediately after creation."""
        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("new-feedback", "New content.", "New feedback desc")

        result = cm.load(["new-feedback"])
        self.assertIn("New content.", result)

    def test_add_feedback_duplicate_auto_suffix(self) -> None:
        """Adding feedback with an existing name should auto-suffix with _1."""
        self._write_md(self.feedback_dir, "foo.md", "foo",
                       "Original foo", "Original content.")

        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("foo", "Duplicate content.", "Duplicate foo")

        self.assertIn("foo_1", cm._all_names)
        result = cm.load(["foo_1"])
        self.assertIn("Duplicate content.", result)

    def test_add_feedback_multiple_duplicates(self) -> None:
        """Adding the same name three times should produce foo, foo_1, foo_2."""
        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("foo", "First.", "First foo")
        cm.add_feedback("foo", "Second.", "Second foo")
        cm.add_feedback("foo", "Third.", "Third foo")

        self.assertIn("foo", cm._all_names)
        self.assertIn("foo_1", cm._all_names)
        self.assertIn("foo_2", cm._all_names)

    def test_add_feedback_without_feedback_path_raises(self) -> None:
        """add_feedback without feedback_path configured should raise ValueError."""
        cm = ContextManager(skills_path=self.skills_dir)

        with self.assertRaises(ValueError) as ctx:
            cm.add_feedback("test", "Content.", "Description")
        self.assertIn("feedback_path was not configured", str(ctx.exception))

    def test_add_feedback_slugified_filename(self) -> None:
        """Feedback filename should be slugified from the name."""
        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("My Error Handling", "Content.", "A description")

        expected_file = self.feedback_dir / "my-error-handling.md"
        self.assertTrue(expected_file.exists(),
                        f"Expected file {expected_file} to exist, "
                        f"but found: {list(self.feedback_dir.glob('*.md'))}")

    def test_add_feedback_atomic_write(self) -> None:
        """The written file should exist with complete and correct content."""
        cm = ContextManager(feedback_path=self.feedback_dir)
        cm.add_feedback("atomic-test", "Complete content here.", "Atomic test desc")

        target = self.feedback_dir / "atomic-test.md"
        self.assertTrue(target.exists())

        content = target.read_text(encoding="utf-8")
        self.assertIn("name: atomic-test", content)
        self.assertIn("Complete content here.", content)

        # Verify no temp files remain
        tmp_files = list(self.feedback_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, f"Temp files should be cleaned up: {tmp_files}")


if __name__ == "__main__":
    unittest.main()

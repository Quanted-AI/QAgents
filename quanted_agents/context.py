"""ContextManager: on-demand skill and feedback loading from markdown files.

Provides directory scanning, YAML frontmatter parsing, catalog building,
content retrieval with fuzzy matching, and atomic feedback file writing.
This is an internal module -- ContextManager is not exported in __init__.py.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import frontmatter  # type: ignore[import-untyped]
import yaml
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """A parsed skill or feedback item from a markdown file.

    Attributes:
        name: Original case name from frontmatter.
        description: Description from frontmatter.
        content: Body markdown with frontmatter stripped.
        filepath: Source file path for debugging/logging.
        source: Category of the item ("skill" or "feedback").
    """

    name: str
    description: str
    content: str
    filepath: Path
    source: str


class ContextManager:
    """Manages skill and feedback context files for on-demand loading.

    Scans configured directories for markdown files with YAML frontmatter,
    builds a catalog of available items, and provides content retrieval
    with fuzzy matching on missing names. Supports atomic feedback file
    writing with automatic duplicate name suffixing.

    Args:
        skills_path: Path to directory containing skill markdown files.
        feedback_path: Path to directory containing feedback markdown files.

    Raises:
        FileNotFoundError: If a configured directory does not exist.
        ValueError: If duplicate names are found across skills and feedback.
    """

    def __init__(
        self,
        skills_path: Path | None = None,
        feedback_path: Path | None = None,
    ) -> None:
        self._skills_path: Path | None = skills_path
        self._feedback_path: Path | None = feedback_path
        self._skills: dict[str, ContextItem] = {}
        self._feedback: dict[str, ContextItem] = {}
        self._all_names: dict[str, ContextItem] = {}

        if skills_path is not None:
            self._validate_directory(skills_path, "skills_path")
            self._skills = self._scan_directory(skills_path, "skill")

        if feedback_path is not None:
            self._validate_directory(feedback_path, "feedback_path")
            self._feedback = self._scan_directory(feedback_path, "feedback")

        self._all_names = {**self._skills, **self._feedback}
        self._check_collisions()

    def _validate_directory(self, path: Path, param_name: str) -> None:
        """Validate that a path is an existing directory.

        Args:
            path: The path to validate.
            param_name: The parameter name for error messages.

        Raises:
            FileNotFoundError: If the path does not exist or is not a directory.
        """
        if not path.is_dir():
            raise FileNotFoundError(
                f"{param_name} directory does not exist: {path}"
            )

    def _scan_directory(self, directory: Path, source: str) -> dict[str, ContextItem]:
        """Scan a directory for markdown files with valid YAML frontmatter.

        Iterates over all .md files in the directory, parses their frontmatter,
        and creates ContextItem instances for files that have both 'name' and
        'description' fields. Malformed or incomplete files are skipped with
        a warning.

        Args:
            directory: The directory to scan for .md files.
            source: The source category ("skill" or "feedback").

        Returns:
            A dict mapping lowercase names to ContextItem instances.
        """
        items: dict[str, ContextItem] = {}

        for filepath in directory.glob("*.md"):
            try:
                post = frontmatter.load(filepath)
            except yaml.YAMLError:
                logger.warning(f"Skipping {filepath}: malformed YAML frontmatter")
                continue
            except Exception:
                logger.warning(f"Skipping {filepath}: failed to parse file")
                continue

            name = post.metadata.get("name")
            description = post.metadata.get("description")

            if not name:
                logger.warning(f"Skipping {filepath}: missing 'name' field in frontmatter")
                continue
            if not description:
                logger.warning(f"Skipping {filepath}: missing 'description' field in frontmatter")
                continue

            item = ContextItem(
                name=name,
                description=description,
                content=post.content,
                filepath=filepath,
                source=source,
            )
            items[name.lower()] = item

        return items

    def _check_collisions(self) -> None:
        """Check for name collisions between skills and feedback.

        Raises:
            ValueError: If any name exists in both skills and feedback dicts.
        """
        skill_names = set(self._skills.keys())
        feedback_names = set(self._feedback.keys())
        collisions = skill_names & feedback_names

        for name in sorted(collisions):
            raise ValueError(
                f"Name collision between skills and feedback: '{name}'. "
                f"Names must be globally unique across skills and feedback."
            )

    def build_catalog(self) -> str:
        """Build a formatted catalog string with available skills and feedback.

        Creates separate sections for skills and feedback, each with a header
        and list of items showing names and descriptions. Only includes sections
        that have items.

        Returns:
            The formatted catalog string, or empty string if no items exist.
        """
        sections: list[str] = []

        if self._skills:
            lines = [
                "## Available Skills\n",
                "The following skills are available. Use the `_load_context` "
                "tool to load the full content of any skill when you need it.\n",
            ]
            for item in self._skills.values():
                lines.append(f"- **{item.name}**: {item.description}")
            sections.append("\n".join(lines))

        if self._feedback:
            lines = [
                "## Available Feedback\n",
                "The following feedback items are available. Use the `_load_context` "
                "tool to load feedback that is relevant to your current task.\n",
            ]
            for item in self._feedback.values():
                lines.append(f"- **{item.name}**: {item.description}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def load(self, names: list[str]) -> str:
        """Load content for one or more context items by name.

        Performs case-insensitive lookup for each name. Found items return
        their body content. Missing items return an error message with
        fuzzy-matched suggestions.

        Args:
            names: List of context item names to load.

        Returns:
            The content for each requested name, separated by '---' dividers.
        """
        results: list[str] = []

        for name in names:
            key = name.lower()
            item = self._all_names.get(key)

            if item is not None:
                results.append(f"## {item.name}\n\n{item.content}")
            else:
                suggestions = self._suggest_alternatives(name)
                if suggestions:
                    results.append(
                        f"Couldn't find '{name}'. Did you mean: {', '.join(suggestions)}?"
                    )
                else:
                    results.append(
                        f"Couldn't find '{name}'. No context items are available."
                    )

        return "\n\n---\n\n".join(results)

    def _suggest_alternatives(self, name: str) -> list[str]:
        """Find the closest matching names using fuzzy string matching.

        Args:
            name: The name that was not found.

        Returns:
            List of up to 5 closest matching names (lowercase keys).
        """
        all_keys = list(self._all_names.keys())
        if not all_keys:
            return []

        results = process.extract(
            name.lower(),
            all_keys,
            scorer=fuzz.WRatio,
            limit=5,
        )
        return [match[0] for match in results]

    def add_feedback(self, name: str, content: str, description: str) -> None:
        """Write a new feedback file atomically and register it in memory.

        If a name collision exists, auto-suffixes with incrementing numbers
        (_1, _2, etc.) until a unique name is found. The file is written
        atomically using a temp file and os.replace().

        Args:
            name: The canonical name for this feedback item.
            content: The markdown body content.
            description: A short description for the catalog.

        Raises:
            ValueError: If feedback_path was not configured.
        """
        if self._feedback_path is None:
            raise ValueError("Cannot add feedback: feedback_path was not configured")

        final_name = name
        key = name.lower()

        if key in self._all_names:
            suffix = 1
            while True:
                suffixed_name = f"{name}_{suffix}"
                if suffixed_name.lower() not in self._all_names:
                    logger.warning(f"Name '{name}' already exists, using '{suffixed_name}' instead")
                    final_name = suffixed_name
                    break
                suffix += 1

        post = frontmatter.Post(content=content, name=final_name, description=description)
        file_content = frontmatter.dumps(post)

        filename = self._slugify(final_name) + ".md"
        target_path = self._feedback_path / filename

        fd, tmp_path = tempfile.mkstemp(dir=str(self._feedback_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(file_content)
            os.replace(tmp_path, str(target_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        item = ContextItem(
            name=final_name,
            description=description,
            content=content,
            filepath=target_path,
            source="feedback",
        )
        self._feedback[final_name.lower()] = item
        self._all_names[final_name.lower()] = item

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a name to a URL-friendly slug for use as a filename.

        Args:
            name: The name to slugify.

        Returns:
            A lowercase, hyphen-separated string suitable for filenames.
        """
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    @property
    def has_items(self) -> bool:
        """Check whether any context items are loaded.

        Returns:
            True if at least one skill or feedback item exists.
        """
        return bool(self._all_names)

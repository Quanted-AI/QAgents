"""Tests for TraceSession, trace_filename, and session_id trace correlation.

Validates TraceSession context manager behavior, session_id injection into
TraceEntry, trace_filename custom naming, precedence rules, and exception
handling (partial trace preservation).
"""

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent, TraceSession
from tests.conftest import SampleInput, SampleOutput


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dicts, one per JSONL line.
    """
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line]


class TestTraceSession(unittest.IsolatedAsyncioTestCase):
    """Tests for TraceSession context manager and session_id injection."""

    def setUp(self) -> None:
        """Create temp directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Remove temp directory."""
        shutil.rmtree(self.tmp_dir)

    async def test_trace_session_creates_single_file(self) -> None:
        """Multiple agent.run() calls within TraceSession produce entries in one file."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        async with TraceSession(session_path) as session:
            with agent.inner.override(model=TestModel()):
                await agent.run(SampleInput(question="first"), trace_session=session)
                await agent.run(SampleInput(question="second"), trace_session=session)

        self.assertTrue(session_path.exists())
        entries = _read_jsonl(session_path)
        self.assertEqual(len(entries), 2)

    async def test_trace_session_generates_session_id(self) -> None:
        """TraceSession generates a valid UUID session_id."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        async with TraceSession(session_path) as session:
            sid = session.session_id
            # Validate it's a valid UUID
            parsed = uuid.UUID(sid)
            self.assertEqual(str(parsed), sid)

    async def test_session_id_in_trace_entries(self) -> None:
        """TraceEntry objects written within session include session_id field."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        async with TraceSession(session_path) as session:
            with agent.inner.override(model=TestModel()):
                await agent.run(SampleInput(question="test"), trace_session=session)

        entries = _read_jsonl(session_path)
        self.assertEqual(len(entries), 1)
        self.assertIn("session_id", entries[0])
        self.assertEqual(entries[0]["session_id"], session.session_id)

    async def test_trace_filename_custom_name(self) -> None:
        """run() with trace_filename and traces_path creates file with custom name."""
        traces_dir = Path(self.tmp_dir) / "traces"
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        with agent.inner.override(model=TestModel()):
            await agent.run(
                SampleInput(question="test"),
                traces_path=str(traces_dir),
                trace_filename="custom.jsonl",
            )

        expected_file = traces_dir / "custom.jsonl"
        self.assertTrue(expected_file.exists())
        entries = _read_jsonl(expected_file)
        self.assertEqual(len(entries), 1)

    async def test_trace_filename_without_traces_path_ignored(self) -> None:
        """trace_filename without traces_path does nothing (no file created)."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        with agent.inner.override(model=TestModel()):
            result = await agent.run(
                SampleInput(question="test"),
                trace_filename="custom.jsonl",
            )

        # No file should be created anywhere in tmp_dir
        jsonl_files = list(Path(self.tmp_dir).rglob("*.jsonl"))
        self.assertEqual(len(jsonl_files), 0)
        # But the run should still succeed
        self.assertIsNotNone(result.data)

    async def test_trace_session_precedence_over_traces_path(self) -> None:
        """trace_session takes priority over traces_path."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        traces_dir = Path(self.tmp_dir) / "traces"

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        async with TraceSession(session_path) as session:
            with agent.inner.override(model=TestModel()):
                await agent.run(
                    SampleInput(question="test"),
                    trace_session=session,
                    traces_path=str(traces_dir),
                )

        # Session file should have the entry
        self.assertTrue(session_path.exists())
        entries = _read_jsonl(session_path)
        self.assertEqual(len(entries), 1)

        # traces_path directory should NOT be created (session takes precedence)
        self.assertFalse(traces_dir.exists())

    async def test_trace_session_flushes_on_exception(self) -> None:
        """Entries written before exception are preserved in trace file."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
        )

        with self.assertRaises(RuntimeError):
            async with TraceSession(session_path) as session:
                with agent.inner.override(model=TestModel()):
                    await agent.run(
                        SampleInput(question="before-error"),
                        trace_session=session,
                    )
                raise RuntimeError("Simulated error")

        # Entry written before the exception should be preserved
        self.assertTrue(session_path.exists())
        entries = _read_jsonl(session_path)
        self.assertEqual(len(entries), 1)

    async def test_trace_session_as_context_manager(self) -> None:
        """async with TraceSession() works correctly, returns self."""
        session_path = Path(self.tmp_dir) / "session.jsonl"
        async with TraceSession(session_path) as session:
            self.assertIsInstance(session, TraceSession)
            self.assertEqual(session.file_path, session_path)
            self.assertIsNotNone(session.writer)
            self.assertIsNotNone(session.session_id)

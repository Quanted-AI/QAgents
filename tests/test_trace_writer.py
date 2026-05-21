"""Tests for TraceWriter and trace file integration in QuantedAgent.

Validates TraceWriter JSONL output, crash-safe flush+fsync behavior,
_resolve_trace_writer kwargs handling, and QuantedAgent.run() integration
with traces_path for file-based trace logging. All tests use pydantic-ai's
TestModel and FunctionModel without real LLM API calls.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from quanted_agents import QuantedAgent
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.trace_writer import TraceWriter, _resolve_trace_writer
from tests.conftest import MALFORMED_SINGLE_QUOTES, SampleInput, SampleOutput


def _make_trace_entry(step_name: str = "test_step") -> TraceEntry:
    """Create a minimal TraceEntry for testing.

    Args:
        step_name: The step name for the trace entry.

    Returns:
        A TraceEntry with valid test data.
    """
    usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
    timing = StepTiming(step_name=step_name, duration_seconds=0.5, usage=usage)
    return TraceEntry(
        step_name=step_name,
        input_data={"question": "test"},
        output_data={"answer": "hello", "confidence": 0.9},
        messages=[{"kind": "request"}],
        tool_calls=[],
        timing=timing,
        model_name="test-model",
        recovery_info=None,
    )


class TestTraceWriter(unittest.IsolatedAsyncioTestCase):
    """Tests for the TraceWriter class."""

    def setUp(self) -> None:
        """Create a temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_trace_writer_creates_jsonl_file(self) -> None:
        """Writing a TraceEntry creates a JSONL file with valid JSON content."""
        file_path = Path(self.tmp_dir) / "test.jsonl"
        writer = TraceWriter(file_path)
        entry = _make_trace_entry()

        await writer.write(entry)

        self.assertTrue(file_path.exists())
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()
        data = json.loads(line)
        self.assertEqual(data["step_name"], "test_step")
        self.assertEqual(data["input_data"]["question"], "test")
        self.assertEqual(data["output_data"]["answer"], "hello")

    async def test_trace_writer_appends_multiple_entries(self) -> None:
        """Writing multiple entries appends them as separate JSONL lines."""
        file_path = Path(self.tmp_dir) / "multi.jsonl"
        writer = TraceWriter(file_path)

        for i in range(3):
            entry = _make_trace_entry(step_name=f"step_{i}")
            await writer.write(entry)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)
        for i, line in enumerate(lines):
            data = json.loads(line)
            self.assertEqual(data["step_name"], f"step_{i}")

    async def test_trace_writer_crash_safe_flush(self) -> None:
        """Written content is immediately readable without explicit close."""
        file_path = Path(self.tmp_dir) / "flush.jsonl"
        writer = TraceWriter(file_path)
        entry = _make_trace_entry()

        await writer.write(entry)

        # Content should be on disk immediately after write returns
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 0)
        data = json.loads(content.strip())
        self.assertEqual(data["step_name"], "test_step")

    def test_file_path_property(self) -> None:
        """The file_path property returns the path set at construction."""
        file_path = Path(self.tmp_dir) / "prop.jsonl"
        writer = TraceWriter(file_path)
        self.assertEqual(writer.file_path, file_path)


class TestResolveTraceWriter(unittest.TestCase):
    """Tests for the _resolve_trace_writer helper function."""

    def setUp(self) -> None:
        """Create a temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_from_traces_path(self) -> None:
        """_resolve_trace_writer creates a TraceWriter from traces_path kwarg."""
        traces_dir = os.path.join(self.tmp_dir, "traces")
        kwargs: dict[str, Any] = {"traces_path": traces_dir}

        writer = _resolve_trace_writer(kwargs)

        self.assertIsNotNone(writer)
        self.assertIsInstance(writer, TraceWriter)
        self.assertTrue(Path(traces_dir).exists())
        self.assertTrue(str(writer.file_path).startswith(traces_dir))
        self.assertTrue(str(writer.file_path).endswith(".jsonl"))
        self.assertEqual(kwargs, {})

    def test_passes_through_existing_writer(self) -> None:
        """_resolve_trace_writer returns an existing _trace_writer as-is."""
        existing = TraceWriter(Path(self.tmp_dir) / "existing.jsonl")
        kwargs: dict[str, Any] = {"_trace_writer": existing}

        writer = _resolve_trace_writer(kwargs)

        self.assertIs(writer, existing)
        self.assertEqual(kwargs, {})

    def test_returns_none_when_no_kwargs(self) -> None:
        """_resolve_trace_writer returns None when no trace kwargs are set."""
        kwargs: dict[str, Any] = {}

        writer = _resolve_trace_writer(kwargs)

        self.assertIsNone(writer)

    def test_pops_kwargs(self) -> None:
        """_resolve_trace_writer removes traces_path but preserves other kwargs."""
        traces_dir = os.path.join(self.tmp_dir, "traces")
        kwargs: dict[str, Any] = {
            "traces_path": traces_dir,
            "model_settings": {"temperature": 0.5},
        }

        _resolve_trace_writer(kwargs)

        self.assertNotIn("traces_path", kwargs)
        self.assertNotIn("_trace_writer", kwargs)
        self.assertIn("model_settings", kwargs)
        self.assertEqual(kwargs["model_settings"]["temperature"], 0.5)

    def test_trace_writer_kwarg_takes_precedence(self) -> None:
        """When both traces_path and _trace_writer are set, _trace_writer wins."""
        traces_dir = os.path.join(self.tmp_dir, "traces")
        existing = TraceWriter(Path(self.tmp_dir) / "existing.jsonl")
        kwargs: dict[str, Any] = {
            "traces_path": traces_dir,
            "_trace_writer": existing,
        }

        writer = _resolve_trace_writer(kwargs)

        self.assertIs(writer, existing)


class TestAgentTraceFileIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for QuantedAgent.run() trace file writing integration."""

    def setUp(self) -> None:
        """Create a temporary directory and test agent."""
        self.tmp_dir = tempfile.mkdtemp()
        self.agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_agent_run_writes_trace_file(self) -> None:
        """Running with traces_path creates a JSONL file with one trace entry."""
        traces_dir = os.path.join(self.tmp_dir, "traces")

        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                SampleInput(question="trace test"),
                traces_path=traces_dir,
            )

        # Verify directory and file created
        self.assertTrue(Path(traces_dir).exists())
        jsonl_files = list(Path(traces_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # Verify file content
        with open(jsonl_files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["step_name"], "QuantedAgent(SampleOutput)")
        self.assertEqual(data["input_data"]["question"], "trace test")
        self.assertIsNotNone(data["output_data"])

    async def test_agent_run_no_file_when_traces_path_none(self) -> None:
        """Running without traces_path creates no files, but result still has traces."""
        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(SampleInput(question="no file"))

        # No JSONL files should exist in the temp dir
        jsonl_files = list(Path(self.tmp_dir).glob("**/*.jsonl"))
        self.assertEqual(len(jsonl_files), 0)

        # In-memory trace still present
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].step_name, "QuantedAgent(SampleOutput)")

    async def test_agent_run_trace_file_and_memory_traces_both_present(self) -> None:
        """Running with traces_path produces both file and in-memory traces."""
        traces_dir = os.path.join(self.tmp_dir, "both")

        with self.agent.inner.override(model=TestModel()):
            result = await self.agent.run(
                SampleInput(question="both traces"),
                traces_path=traces_dir,
            )

        # In-memory trace
        self.assertEqual(len(result.trace), 1)
        memory_entry = result.trace[0]

        # File trace
        jsonl_files = list(Path(traces_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)
        with open(jsonl_files[0], "r", encoding="utf-8") as f:
            file_data = json.loads(f.readline())

        # Both should have the same core data
        self.assertEqual(memory_entry.step_name, file_data["step_name"])
        self.assertEqual(memory_entry.input_data, file_data["input_data"])
        self.assertEqual(memory_entry.output_data, file_data["output_data"])

    async def test_agent_run_recovery_writes_trace_file(self) -> None:
        """Recovery path also writes trace entry to the trace file."""

        def _malformed_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return malformed JSON to trigger recovery."""
            return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            max_recovery_attempts=3,
        )
        traces_dir = os.path.join(self.tmp_dir, "recovery")

        fm = FunctionModel(_malformed_model)
        with agent.inner.override(model=fm):
            result = await agent.run(
                SampleInput(question="recover"),
                traces_path=traces_dir,
            )

        # Verify trace file written on recovery path
        jsonl_files = list(Path(traces_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)
        with open(jsonl_files[0], "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        self.assertIsNotNone(data["recovery_info"])
        self.assertTrue(data["recovery_info"]["json_repair_attempted"])

        # In-memory trace also present
        self.assertEqual(len(result.trace), 1)
        self.assertIsNotNone(result.trace[0].recovery_info)


if __name__ == "__main__":
    unittest.main()

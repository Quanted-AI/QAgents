"""Integration tests for trace file writing across all workflow types.

Validates that Pipeline, Router, Loop, and Parallel produce correct single
JSONL trace files when traces_path is provided. Tests cover single-file
creation, nested workflow propagation, concurrent write safety, directory
auto-creation, and zero overhead when traces_path is None.
All tests use pydantic-ai's TestModel and FunctionModel.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from quanted_agents.workflows import Loop, Parallel, Pipeline, Router
from quanted_agents.workflows.router import RoutingDecision


class StepInput(BaseModel):
    """Input model for pipeline and general workflow tests."""

    text: str


class StepOutput(BaseModel):
    """Output model for pipeline steps and general tests."""

    summary: str


class Draft(BaseModel):
    """Model for loop iteration tests."""

    content: str
    quality_score: float = 0.0


class BranchOutput(BaseModel):
    """Output model for parallel branches."""

    value: str


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dicts, one per JSONL line.
    """
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line]


def _make_dispatcher_function(target: str):
    """Create a FunctionModel handler that returns a RoutingDecision.

    Args:
        target: The specialist name to route to.

    Returns:
        A function compatible with FunctionModel.
    """

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        """Return a RoutingDecision as a tool call response."""
        decision = {"target": target, "reasoning": "test routing"}
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(decision))]
        )

    return handler


class TestPipelineTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests for Pipeline workflow trace file writing."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_pipeline_traces_path_creates_single_file(self) -> None:
        """Pipeline with traces_path creates one JSONL file with entries from all steps."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepOutput,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepOutput,
            output_type=StepOutput,
            system_prompt="Step 2",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(
                    StepInput(text="hello"),
                    traces_path=self.tmp_dir,
                )

        # Exactly 1 JSONL file
        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # File has exactly 2 lines (one per agent step)
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 2)

        # Each entry has required fields
        for entry in entries:
            self.assertIn("step_name", entry)
            self.assertIn("input_data", entry)
            self.assertIn("output_data", entry)

        # In-memory trace also has 2 entries (TRACE-08)
        self.assertEqual(len(result.trace), 2)


class TestRouterTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests for Router workflow trace file writing."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_router_traces_path_creates_single_file(self) -> None:
        """Router with traces_path creates one JSONL file with dispatcher + specialist entries."""
        dispatcher = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=RoutingDecision,
            system_prompt="Classify",
        )
        specialist = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepOutput,
            system_prompt="Handle",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"handler": specialist},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("handler"))
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist.inner.override(model=TestModel()):
                result = await router.run(
                    StepInput(text="route me"),
                    traces_path=self.tmp_dir,
                )

        # Exactly 1 JSONL file
        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # File has 2 lines (dispatcher + specialist)
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 2)

        # In-memory trace also has 2 entries
        self.assertEqual(len(result.trace), 2)


class TestLoopTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests for Loop workflow trace file writing."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_loop_traces_path_creates_single_file(self) -> None:
        """Loop with traces_path creates one JSONL file with entries from all iterations."""
        call_count = [0]

        def _refiner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            """Return a Draft with increasing quality score."""
            call_count[0] += 1
            score = call_count[0] * 0.5
            draft = Draft(content="refined", quality_score=score)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.9,
            max_iterations=5,
        )

        fm = FunctionModel(_refiner_model)
        with body.inner.override(model=fm):
            result = await loop.run(
                Draft(content="initial", quality_score=0.0),
                traces_path=self.tmp_dir,
            )

        # Should have terminated after 2 iterations (0.5, 1.0)
        self.assertEqual(call_count[0], 2)

        # Exactly 1 JSONL file
        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # File has 2 lines (one per iteration)
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 2)

        # In-memory trace also has 2 entries
        self.assertEqual(len(result.trace), 2)


class TestParallelTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests for Parallel workflow trace file writing."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_parallel_traces_path_creates_single_file(self) -> None:
        """Parallel with traces_path creates one JSONL file with entries from all branches."""
        branch_a = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepOutput,
            system_prompt="Branch A",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=BranchOutput,
            system_prompt="Branch B",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=TestModel()):
            with branch_b.inner.override(model=TestModel()):
                result = await parallel.run(
                    StepInput(text="fan out"),
                    traces_path=self.tmp_dir,
                )

        # Exactly 1 JSONL file
        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # File has 2 lines (one per branch)
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 2)

        # Each line is valid JSON (no interleaving/corruption)
        for entry in entries:
            self.assertIn("step_name", entry)
            self.assertIn("input_data", entry)
            self.assertIn("output_data", entry)

    async def test_parallel_concurrent_writes_no_corruption(self) -> None:
        """Parallel with 3+ branches produces valid, non-corrupted JSONL lines."""
        branches = []
        overrides = []
        for i in range(4):
            branch = QuantedAgent(
                "test",
                input_type=StepInput,
                output_type=StepOutput,
                system_prompt=f"Branch {i}",
            )
            branches.append(branch)
            overrides.append(branch.inner.override(model=TestModel()))

        parallel = Parallel(branches=branches)

        # Nested context managers for all 4 branches
        with overrides[0], overrides[1], overrides[2], overrides[3]:
            await parallel.run(
                StepInput(text="concurrent"),
                traces_path=self.tmp_dir,
            )

        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # Every line is valid JSON and total count matches branches
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 4)
        for entry in entries:
            self.assertIsInstance(entry, dict)
            self.assertIn("step_name", entry)


class TestNestedWorkflowTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests for nested workflow trace file propagation."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_nested_workflow_single_trace_file(self) -> None:
        """Nested Pipeline (Pipeline containing Pipeline) produces a single trace file."""
        step_a = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepOutput,
            system_prompt="Step A",
        )
        step_b = QuantedAgent(
            "test",
            input_type=StepOutput,
            output_type=StepOutput,
            system_prompt="Step B",
        )
        inner_pipeline = Pipeline(steps=[step_a, step_b])

        step_c = QuantedAgent(
            "test",
            input_type=StepOutput,
            output_type=StepOutput,
            system_prompt="Step C",
        )
        outer_pipeline = Pipeline(steps=[inner_pipeline, step_c])

        with step_a.inner.override(model=TestModel()):
            with step_b.inner.override(model=TestModel()):
                with step_c.inner.override(model=TestModel()):
                    result = await outer_pipeline.run(
                        StepInput(text="nested"),
                        traces_path=self.tmp_dir,
                    )

        # Still exactly 1 file in the directory
        jsonl_files = list(Path(self.tmp_dir).glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)

        # All 3 agent-level trace entries present
        entries = _read_jsonl(jsonl_files[0])
        self.assertEqual(len(entries), 3)

        # In-memory trace also has all 3 entries
        self.assertEqual(len(result.trace), 3)


class TestNoTraceFile(unittest.IsolatedAsyncioTestCase):
    """Tests verifying zero overhead when traces_path is None."""

    def setUp(self) -> None:
        """Create temporary directory to verify no files are created."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_no_trace_file_when_traces_path_none(self) -> None:
        """Running workflows without traces_path creates no files."""
        # Pipeline
        step1 = QuantedAgent("test", input_type=StepInput, output_type=StepOutput, system_prompt="S1")
        step2 = QuantedAgent("test", input_type=StepOutput, output_type=StepOutput, system_prompt="S2")
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                pipeline_result = await pipeline.run(StepInput(text="no trace"))

        # No files created in tmp dir
        jsonl_files = list(Path(self.tmp_dir).glob("**/*.jsonl"))
        self.assertEqual(len(jsonl_files), 0)

        # In-memory trace still present
        self.assertGreater(len(pipeline_result.trace), 0)

        # Parallel
        branch_a = QuantedAgent("test", input_type=StepInput, output_type=StepOutput, system_prompt="A")
        branch_b = QuantedAgent("test", input_type=StepInput, output_type=StepOutput, system_prompt="B")
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=TestModel()):
            with branch_b.inner.override(model=TestModel()):
                parallel_result = await parallel.run(StepInput(text="no trace"))

        jsonl_files = list(Path(self.tmp_dir).glob("**/*.jsonl"))
        self.assertEqual(len(jsonl_files), 0)
        self.assertGreater(len(parallel_result.trace), 0)


class TestTracePathAutoCreation(unittest.IsolatedAsyncioTestCase):
    """Tests for automatic directory creation."""

    def setUp(self) -> None:
        """Create temporary directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_traces_path_directory_auto_created(self) -> None:
        """Non-existent nested directory is auto-created when traces_path is set."""
        nested_path = Path(self.tmp_dir) / "nested" / "deep"
        self.assertFalse(nested_path.exists())

        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepOutput,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepOutput,
            output_type=StepOutput,
            system_prompt="Step 2",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                await pipeline.run(
                    StepInput(text="auto-create"),
                    traces_path=str(nested_path),
                )

        # Directory was created
        self.assertTrue(nested_path.exists())

        # Trace file exists inside it
        jsonl_files = list(nested_path.glob("trace_*.jsonl"))
        self.assertEqual(len(jsonl_files), 1)


if __name__ == "__main__":
    unittest.main()

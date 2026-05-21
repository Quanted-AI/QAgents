"""Integration tests for Pipeline cross-feature scenarios.

Covers SC2 (pipeline + input_transform + soft limit + trace session + dual-stream)
and additional scenarios: pipeline + dual-stream + assembly (Scenario 2) and
stream recovery + dual-stream (Scenario 5).

All tests use TestModel/FunctionModel for deterministic LLM simulation. No real API calls.
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

from quanted_agents import ArtifactStore, QuantedAgent, QuantedResult, TraceSession
from quanted_agents.workflows import Pipeline
from tests.conftest import (
    MALFORMED_SINGLE_QUOTES,
    SampleInput,
    SampleOutput,
    make_agent,
    make_store,
)


# ---------------------------------------------------------------------------
# Test-specific BaseModels
# ---------------------------------------------------------------------------


class StepAInput(BaseModel):
    """Input for pipeline step A."""

    text: str


class StepAOutput(BaseModel):
    """Output from pipeline step A."""

    summary: str


class StepBInput(BaseModel):
    """Input for pipeline step B (different type from StepAOutput)."""

    query: str


class StepBOutput(BaseModel):
    """Output from pipeline step B."""

    report: str


class StepCOutput(BaseModel):
    """Output from pipeline step C."""

    conclusion: str


class FinalOutput(BaseModel):
    """Assembled output combining multiple pipeline stages."""

    combined: str


# ---------------------------------------------------------------------------
# Helper: read JSONL trace file
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dicts, one per JSONL line.
    """
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line]


# ===========================================================================
# TestPipelineIntegration
# ===========================================================================


class TestPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Pipeline with cross-feature interactions."""

    def setUp(self) -> None:
        """Create temp directory for trace files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Remove temp directory."""
        shutil.rmtree(self.tmp_dir)

    async def test_pipeline_with_transforms_softlimits_trace_summary(self) -> None:
        """SC2: Pipeline with input_transform, soft_limit, trace session, and store.

        Validates that a 2-stage pipeline with:
        - input_transforms bridging incompatible types
        - soft_limit=True on at least one agent
        - TraceSession wrapping pipeline.run() with trace_session kwarg
        - ArtifactStore recording step outputs
        runs end-to-end without errors and produces correct results.
        """
        # Stage 1: agent with soft_limit enabled
        step1 = make_agent(
            input_type=StepAInput,
            output_type=StepAOutput,
            soft_limit=True,
            llm_call_limit=10,
        )
        # Stage 2: normal agent
        step2 = make_agent(
            input_type=StepBInput,
            output_type=StepBOutput,
        )

        # Bridge function to transform StepAOutput -> StepBInput
        def bridge(result: QuantedResult, store: ArtifactStore, index: int) -> StepBInput:
            return StepBInput(query=result.data.summary)

        store = make_store()
        pipeline = Pipeline(
            steps=[step1, step2],
            input_transforms={1: bridge},
            store=store,
        )

        # Run within a TraceSession
        session_path = Path(self.tmp_dir) / "pipeline_trace.jsonl"
        async with TraceSession(session_path) as session:
            with step1.inner.override(model=TestModel()):
                with step2.inner.override(model=TestModel()):
                    result = await pipeline.run(
                        StepAInput(text="hello world"),
                        trace_session=session,
                    )

        # Verify pipeline completed with correct output type
        self.assertIsInstance(result.data, StepBOutput)

        # Verify store contains step artifacts
        self.assertIn("step_0/stepaoutput", store)
        self.assertIn("step_1/stepboutput", store)
        self.assertIsInstance(store["step_0/stepaoutput"], StepAOutput)
        self.assertIsInstance(store["step_1/stepboutput"], StepBOutput)

        # Verify trace session file exists and contains entries
        self.assertTrue(session_path.exists())
        entries = _read_jsonl(session_path)
        self.assertGreaterEqual(len(entries), 1)

    async def test_pipeline_dualstream_assembly(self) -> None:
        """Scenario 2: 3-stage pipeline with store and assembly combining intermediates.

        Validates that a pipeline with assembly function reads from all store
        namespaces (step_0/, step_1/, step_2/) and builds a FinalOutput that
        replaces the last step's output as the pipeline result.
        """
        step1 = make_agent(input_type=StepAInput, output_type=StepAOutput)
        step2 = make_agent(input_type=StepAOutput, output_type=StepBOutput)
        step3 = make_agent(input_type=StepBOutput, output_type=StepCOutput)

        store = make_store()

        def assemble(st: ArtifactStore, last_result: QuantedResult) -> FinalOutput:
            a_val = st.get("step_0/stepaoutput", StepAOutput).summary
            b_val = st.get("step_1/stepboutput", StepBOutput).report
            c_val = st.get("step_2/stepcoutput", StepCOutput).conclusion
            return FinalOutput(combined=f"{a_val}|{b_val}|{c_val}")

        pipeline = Pipeline(
            steps=[step1, step2, step3],
            assembly=assemble,
            store=store,
        )

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                with step3.inner.override(model=TestModel()):
                    result = await pipeline.run(StepAInput(text="start"))

        # Verify result.data is FinalOutput from assembly (not StepCOutput)
        self.assertIsInstance(result.data, FinalOutput)
        # Assembly combined all three store values
        self.assertIn("|", result.data.combined)

        # Verify store has all 3 step outputs
        self.assertIn("step_0/stepaoutput", store)
        self.assertIn("step_1/stepboutput", store)
        self.assertIn("step_2/stepcoutput", store)

    async def test_stream_recovery_dualstream(self) -> None:
        """Scenario 5: Stream recovery produces QuantedResult with recovery flags.

        Validates that when run_stream() encounters malformed JSON output,
        recovery pipeline activates and produces a QuantedResult with
        was_recovered=True and recovery_method set.
        """

        def _malformed_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            """Return malformed JSON to trigger recovery."""
            return ModelResponse(parts=[TextPart(content=MALFORMED_SINGLE_QUOTES)])

        async def _malformed_stream(
            messages: list[ModelMessage], info: AgentInfo
        ):
            """Stream function yielding malformed text."""
            yield MALFORMED_SINGLE_QUOTES

        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test recovery",
            max_recovery_attempts=3,
        )

        items: list[Any] = []
        fm = FunctionModel(_malformed_model, stream_function=_malformed_stream)
        with agent.inner.override(model=fm):
            async for item in agent.run_stream(SampleInput(question="recover")):
                items.append(item)

        # Last item should be a QuantedResult from recovery
        self.assertTrue(len(items) > 0)
        last_item = items[-1]
        self.assertIsInstance(last_item, QuantedResult)
        self.assertTrue(last_item.was_recovered)
        self.assertEqual(last_item.recovery_method, "json_repair")

        # Verify recovered data is valid
        self.assertIsInstance(last_item.data, SampleOutput)
        self.assertIsNotNone(last_item.data.answer)


if __name__ == "__main__":
    unittest.main()

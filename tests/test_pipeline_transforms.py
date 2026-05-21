"""Tests for Pipeline input_transforms feature (ORCH-01).

Validates that Pipeline supports input_transforms to bridge type gaps between
stages, handles sync and async transforms, validates stage 0 rejection,
and provides proper error messages for type mismatches.
"""

import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from quanted_agents import PipelineTypeError, QuantedAgent
from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.result import QuantedResult
from quanted_agents.workflows import Pipeline


class InputA(BaseModel):
    """Input for the first stage."""

    text: str


class OutputA(BaseModel):
    """Output from the first stage (incompatible with InputB)."""

    summary: str


class InputB(BaseModel):
    """Input for the second stage (different type from OutputA)."""

    query: str


class OutputB(BaseModel):
    """Output from the second stage."""

    answer: str


class TestPipelineTransforms(unittest.IsolatedAsyncioTestCase):
    """Tests for Pipeline input_transforms parameter."""

    async def test_transform_bridges_type_gap(self) -> None:
        """Pipeline with 2 stages of different types succeeds when transform bridges them."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Answer"
        )

        def bridge(result: QuantedResult, store: ArtifactStore, index: int) -> InputB:
            return InputB(query=result.data.summary)

        pipeline = Pipeline(steps=[step1, step2], input_transforms={1: bridge})

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(InputA(text="hello"))
                self.assertIsInstance(result.data, OutputB)

    async def test_transform_receives_full_result_and_store(self) -> None:
        """Transform receives QuantedResult (not just .data), store with prior artifacts, and index."""
        captured: dict[str, Any] = {}

        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Answer"
        )

        def inspecting_bridge(
            result: QuantedResult, store: ArtifactStore, index: int
        ) -> InputB:
            captured["result_type"] = type(result).__name__
            captured["has_data"] = hasattr(result, "data")
            captured["store_keys"] = list(store.keys())
            captured["index"] = index
            return InputB(query=result.data.summary)

        pipeline = Pipeline(
            steps=[step1, step2],
            input_transforms={1: inspecting_bridge},
            store=ArtifactStore(),
        )

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                await pipeline.run(InputA(text="hello"))

        self.assertEqual(captured["result_type"], "QuantedResult")
        self.assertTrue(captured["has_data"])
        self.assertEqual(captured["index"], 1)
        self.assertTrue(len(captured["store_keys"]) > 0)

    async def test_async_transform(self) -> None:
        """Async transform function works correctly."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Answer"
        )

        async def async_bridge(
            result: QuantedResult, store: ArtifactStore, index: int
        ) -> InputB:
            return InputB(query=f"async:{result.data.summary}")

        pipeline = Pipeline(steps=[step1, step2], input_transforms={1: async_bridge})

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(InputA(text="hello"))
                self.assertIsInstance(result.data, OutputB)

    async def test_no_transform_passes_data_directly(self) -> None:
        """Existing behavior preserved: same types, no transform needed."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=OutputA, output_type=OutputB, system_prompt="Report"
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(InputA(text="hello"))
                self.assertIsInstance(result.data, OutputB)

    def test_stage_0_transform_raises_valueerror(self) -> None:
        """input_transforms={0: fn} raises ValueError at construction."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=OutputA, output_type=OutputB, system_prompt="Report"
        )

        def dummy(result: QuantedResult, store: ArtifactStore, index: int) -> OutputA:
            return OutputA(summary="dummy")

        with self.assertRaises(ValueError) as ctx:
            Pipeline(steps=[step1, step2], input_transforms={0: dummy})
        self.assertIn("Stage 0", str(ctx.exception))

    def test_type_mismatch_without_transform_raises(self) -> None:
        """PipelineTypeError with updated message including 'Provide an input_transform' hint."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Step 1"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Step 2"
        )

        with self.assertRaises(PipelineTypeError) as ctx:
            Pipeline(steps=[step1, step2])
        msg = str(ctx.exception)
        self.assertIn("OutputA", msg)
        self.assertIn("InputB", msg)
        self.assertIn("Provide an input_transform", msg)

    def test_type_mismatch_with_transform_skips_check(self) -> None:
        """Different types with transform provided causes no error at construction."""
        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Step 1"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Step 2"
        )

        def bridge(result: QuantedResult, store: ArtifactStore, index: int) -> InputB:
            return InputB(query=result.data.summary)

        pipeline = Pipeline(steps=[step1, step2], input_transforms={1: bridge})
        self.assertIsNotNone(pipeline)

    async def test_transform_store_empty_when_no_store_configured(self) -> None:
        """Transform receives an ArtifactStore even when Pipeline has no explicit store."""
        captured_store: list[Any] = []

        step1 = QuantedAgent(
            "test", input_type=InputA, output_type=OutputA, system_prompt="Summarize"
        )
        step2 = QuantedAgent(
            "test", input_type=InputB, output_type=OutputB, system_prompt="Answer"
        )

        def store_capturing_bridge(
            result: QuantedResult, store: ArtifactStore, index: int
        ) -> InputB:
            captured_store.append(store)
            return InputB(query=result.data.summary)

        pipeline = Pipeline(steps=[step1, step2], input_transforms={1: store_capturing_bridge})

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                await pipeline.run(InputA(text="hello"))

        self.assertEqual(len(captured_store), 1)
        self.assertIsInstance(captured_store[0], ArtifactStore)


if __name__ == "__main__":
    unittest.main()

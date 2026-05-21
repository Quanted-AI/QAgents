"""Pipeline: Sequential workflow that chains Runnable steps."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.exceptions import AssemblyError, PipelineTypeError
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.result import QuantedResult
from quanted_agents.trace_writer import _resolve_trace_writer
from quanted_agents.types import AssemblyFn, PipelineTransformFn, Runnable


class Pipeline:
    """Sequential workflow that chains the output of each step as input to the next.

    Pipeline implements the Runnable protocol, enabling it to be nested inside
    other workflows (e.g., as a step in another Pipeline, or as the body of a
    Loop). Each step must be a Runnable whose ``run()`` method accepts a
    BaseModel and returns a QuantedResult.

    The pipeline passes ``result.data`` from step N as ``input_data`` to step
    N+1. The final step's QuantedResult is returned as the pipeline's result,
    enriched with aggregated usage, per-step timing, and trace data.

    Example:
        summarizer = QuantedAgent("openai:gpt-4o", input_type=RawText, output_type=Summary, ...)
        reporter = QuantedAgent("openai:gpt-4o", input_type=Summary, output_type=Report, ...)
        pipeline = Pipeline(steps=[summarizer, reporter])
        result = await pipeline.run(RawText(text="Long document..."))
        print(result.data.title)  # Report.title
    """

    def __init__(
        self,
        steps: list[Runnable],
        *,
        input_transforms: dict[int, PipelineTransformFn] | None = None,
        assembly: AssemblyFn | None = None,
        store: ArtifactStore | None = None,
        trace_artifacts: bool = False,
    ) -> None:
        """Create a new Pipeline with the given steps.

        Args:
            steps: Ordered list of Runnable instances to execute sequentially.
                Must contain at least 2 steps.
            input_transforms: Optional mapping of stage index to transform
                function. When provided for stage N, the transform receives the
                previous stage's QuantedResult, the Pipeline's ArtifactStore,
                and the stage index, returning a BaseModel for stage N's input.
                Stage 0 cannot have a transform (it receives pipeline input
                directly). When a transform is provided for a stage boundary,
                the type mismatch check is skipped for that boundary.
            assembly: Optional assembly function to transform accumulated
                store artifacts into a final output after the last step.
            store: Optional ArtifactStore for recording step outputs. If not
                provided but assembly is set, a store is created automatically.
            trace_artifacts: Whether to write SDK metadata to the store under
                reserved '_' prefix keys. Defaults to False.

        Raises:
            ValueError: If fewer than 2 steps are provided or stage 0 has
                an input_transform.
            PipelineTypeError: If adjacent QuantedAgent steps have mismatched
                output_type/input_type and no input_transform bridges them.
        """
        if len(steps) < 2:
            raise ValueError("Pipeline requires at least 2 steps")
        transforms = input_transforms or {}
        if 0 in transforms:
            raise ValueError(
                "Stage 0 cannot have an input_transform (it receives pipeline input directly)"
            )
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            if (i + 1) in transforms:
                continue
            if hasattr(current_step, "output_type") and hasattr(next_step, "input_type"):
                if current_step.output_type is not next_step.input_type:
                    raise PipelineTypeError(
                        f"Stage {i} output type {current_step.output_type.__name__} doesn't "
                        f"match stage {i + 1} input type {next_step.input_type.__name__}. "
                        f"Provide an input_transform for stage {i + 1}."
                    )
        self._steps: list[Runnable] = steps
        self._transforms: dict[int, PipelineTransformFn] = transforms
        self._assembly: AssemblyFn | None = assembly
        self._store: ArtifactStore | None = store
        self._trace_artifacts: bool = trace_artifacts

    def _step_key(self, i: int, step: Runnable) -> str:
        """Derive a unique store key for a pipeline step.

        Args:
            i: The step index.
            step: The Runnable step.

        Returns:
            A key string in the format "step_{i}/{type_name}".
        """
        type_name = getattr(step, "output_type", None)
        type_name = type_name.__name__.lower() if type_name else "step"
        return f"step_{i}/{type_name}"

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Run the pipeline by chaining all steps sequentially.

        Executes each step in order, passing the output data of step N as
        the input to step N+1. Returns the final step's QuantedResult
        enriched with aggregated usage, per-step timing, and trace entries
        collected from all steps.

        When a store or assembly function is configured, step outputs are
        written to the store and the assembly function transforms accumulated
        artifacts into the final result.

        Args:
            input_data: A Pydantic BaseModel instance to feed into the first step.
            **kwargs: Additional keyword arguments forwarded to each step's
                ``run()`` method (e.g., usage_limits, model_settings).

        Returns:
            The QuantedResult from the final step (or assembly output),
            enriched with workflow-level observability data.
        """
        trace_writer = _resolve_trace_writer(kwargs)
        store = self._store or (ArtifactStore() if self._assembly is not None else None)
        transform_store = store if store is not None else ArtifactStore()

        total_usage = RunUsage()
        step_timings: list[StepTiming] = []
        trace_entries: list[TraceEntry] = []

        current: BaseModel = input_data
        result: QuantedResult[Any] | None = None
        for i, step in enumerate(self._steps):
            if i > 0 and result is not None:
                if i in self._transforms:
                    transform = self._transforms[i]
                    if asyncio.iscoroutinefunction(transform):
                        current = await transform(result, transform_store, i)
                    else:
                        current = transform(result, transform_store, i)
                else:
                    current = result.data

            start = time.perf_counter()
            result = await step.run(current, _trace_writer=trace_writer, **kwargs)
            duration = time.perf_counter() - start

            step_key = self._step_key(i, step)

            if store is not None:
                store[step_key] = result.data
                if result.summary is not None:
                    store[f"{step_key}/summary"] = result.summary

            step_timings.append(StepTiming(
                step_name=f"Pipeline.step_{i}",
                duration_seconds=duration,
                usage=result.total_usage,
            ))
            if any(not t.step_name.startswith("QuantedAgent(") for t in result._step_timings):
                step_timings.extend(result._step_timings)
            total_usage = total_usage + result.total_usage
            trace_entries.extend(result.trace)

        if store is not None and self._trace_artifacts:
            store._sdk_set("_pipeline/step_order", [
                self._step_key(i, s) for i, s in enumerate(self._steps)
            ])

        if self._assembly is not None:
            try:
                if asyncio.iscoroutinefunction(self._assembly):
                    assembled = await self._assembly(store, result)
                else:
                    assembled = self._assembly(store, result)
            except Exception as exc:
                raise AssemblyError(
                    f"Assembly function failed: {exc}",
                    store=store,
                    last_result=result,
                    original_error=exc,
                ) from exc
            result = QuantedResult.from_data(assembled, usage=total_usage)

        if store is not None:
            result._artifacts = store  # type: ignore[union-attr]
        result._trace_entries = trace_entries  # type: ignore[union-attr]
        result._step_timings = step_timings  # type: ignore[union-attr]
        result._total_usage = total_usage  # type: ignore[union-attr]
        return result  # type: ignore[return-value]

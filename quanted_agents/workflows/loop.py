"""Loop: Iterative workflow that runs a Runnable body until convergence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.exceptions import AssemblyError, MaxIterationsExceeded
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.result import QuantedResult
from quanted_agents.trace_writer import _resolve_trace_writer
from quanted_agents.types import AssemblyFn, Runnable


class Loop:
    """Iterative workflow that runs a body Runnable until a termination check passes.

    Loop implements the Runnable protocol, enabling it to be nested inside
    other workflows (e.g., as a step in a Pipeline, or as a branch in Parallel).
    The body Runnable is executed repeatedly, passing the previous output as
    the next input, until the termination check returns True or the maximum
    iteration count is reached.

    The ``max_iterations`` parameter is keyword-only with no default, forcing
    the developer to explicitly choose an iteration budget.

    When ``max_iterations`` is reached without the termination check passing,
    ``MaxIterationsExceeded`` is raised to signal non-convergence.

    Example:
        refiner = QuantedAgent("openai:gpt-4o", input_type=Draft, output_type=Draft, ...)
        loop = Loop(
            body=refiner,
            termination_check=lambda d: d.quality_score >= 0.9,
            max_iterations=5,
        )
        result = await loop.run(Draft(content="rough draft", quality_score=0.0))
        print(result.data.quality_score)  # >= 0.9 (converged)
    """

    def __init__(
        self,
        body: Runnable,
        termination_check: Callable[[BaseModel], bool],
        *,
        max_iterations: int,
        assembly: AssemblyFn | None = None,
        store: ArtifactStore | None = None,
        trace_artifacts: bool = False,
    ) -> None:
        """Create a new Loop with the given body, check, and iteration limit.

        Args:
            body: A Runnable to execute on each iteration. Its output is
                fed back as input on the next iteration.
            termination_check: A callable that receives the body's output
                data (a BaseModel) and returns True to stop iterating.
            max_iterations: Maximum number of iterations before raising
                MaxIterationsExceeded. Must be >= 1. Keyword-only with no default.
            assembly: Optional assembly function to transform accumulated
                store artifacts into a final output after convergence.
            store: Optional ArtifactStore for recording iteration outputs.
            trace_artifacts: Whether to write SDK metadata to the store.

        Raises:
            ValueError: If max_iterations is less than 1.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._body: Runnable = body
        self._check: Callable[[BaseModel], bool] = termination_check
        self._max_iterations: int = max_iterations
        self._assembly: AssemblyFn | None = assembly
        self._store: ArtifactStore | None = store
        self._trace_artifacts: bool = trace_artifacts

    @property
    def max_iterations(self) -> int:
        """The maximum number of iterations allowed.

        Returns:
            The configured max_iterations value.
        """
        return self._max_iterations

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Run the loop until termination check passes or max_iterations reached.

        Executes the body Runnable repeatedly. After each iteration, the
        termination check is called with the body's output data. If the check
        returns True, the loop exits and returns the result (optionally
        transformed by the assembly function). If all iterations are exhausted
        without the termination check passing, raises MaxIterationsExceeded.

        Args:
            input_data: A Pydantic BaseModel instance to feed into the first
                iteration of the body.
            **kwargs: Additional keyword arguments forwarded to the body's
                ``run()`` method on each iteration.

        Returns:
            The QuantedResult from the converged iteration (or assembly output),
            enriched with workflow-level observability data.

        Raises:
            MaxIterationsExceeded: If all iterations are exhausted without
                the termination check passing.
        """
        trace_writer = _resolve_trace_writer(kwargs)
        store = self._store or (ArtifactStore() if self._assembly is not None else None)

        total_usage = RunUsage()
        step_timings: list[StepTiming] = []
        trace_entries: list[TraceEntry] = []

        current: BaseModel = input_data
        result: QuantedResult[Any] | None = None
        iteration_count = 0

        for i in range(self._max_iterations):
            start = time.perf_counter()
            result = await self._body.run(current, _trace_writer=trace_writer, **kwargs)
            duration = time.perf_counter() - start
            iteration_count = i + 1

            if store is not None:
                store["iteration_result"] = result.data
                if result.summary is not None:
                    store["iteration_summary"] = result.summary

            step_timings.append(StepTiming(
                step_name=f"Loop.iteration_{i}",
                duration_seconds=duration,
                usage=result.total_usage,
            ))
            if any(not t.step_name.startswith("QuantedAgent(") for t in result._step_timings):
                step_timings.extend(result._step_timings)
            total_usage = total_usage + result.total_usage
            trace_entries.extend(result.trace)

            if self._check(result.data):
                break
            current = result.data
        else:
            raise MaxIterationsExceeded(
                f"Loop did not converge after {self._max_iterations} iterations"
            )

        if store is not None and self._trace_artifacts:
            store._sdk_set("_loop/iteration_count", iteration_count)

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

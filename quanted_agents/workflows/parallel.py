"""Parallel: Concurrent fan-out/fan-in workflow running multiple Runnables."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.usage import RunUsage

from quanted_agents.artifact_store import ArtifactStore, _NamespacedStore
from quanted_agents.exceptions import AssemblyError
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.result import QuantedResult
from quanted_agents.trace_writer import TraceWriter, _resolve_trace_writer
from quanted_agents.types import ParallelAssemblyFn, Runnable


@dataclass
class RetryPolicy:
    """Configuration for retrying failed Parallel branches.

    Args:
        max_retries: Maximum number of retry attempts for each failed branch.
        retry_on: List of exception types eligible for retry. Only branches
            that failed with one of these types will be retried.
        delay_seconds: Seconds to wait between retry attempts.
    """

    max_retries: int = 0
    retry_on: list[type[Exception]] = field(default_factory=list)
    delay_seconds: float = 1.0


class ParallelOutput(BaseModel):
    """Container for aggregated parallel branch outputs.

    Stores the individual data items from each successful branch execution.

    Attributes:
        items: List of output values from each successful branch.
    """

    items: list[Any]


class ParallelResult(QuantedResult[Any]):
    """Result type for Parallel workflow runs.

    Extends QuantedResult to store both successful results and errors
    from concurrent branch execution. Provides access to individual
    branch outcomes via the ``results`` and ``errors`` properties.

    The ``data`` property returns a ParallelOutput containing the
    individual data values from successful branches.
    """

    def __init__(
        self,
        results: list[QuantedResult[Any]],
        errors: list[Exception],
        total_usage: RunUsage | None = None,
        step_timings: list[StepTiming] | None = None,
    ) -> None:
        """Create a ParallelResult from branch outcomes.

        Bypasses the parent ``__init__`` since there is no single
        AgentRunResult to wrap. Instead, stores the collection of
        individual branch results and any errors.

        Args:
            results: List of successful QuantedResult instances from branches.
            errors: List of exceptions raised by failed branches.
            total_usage: Pre-computed aggregated usage, or None to compute on access.
            step_timings: Per-branch timing data, or None to build from branch results.
        """
        self._result = None
        self._data = None
        self._parallel_results: list[QuantedResult[Any]] = results
        self._errors: list[Exception] = errors
        self._total_usage: RunUsage | None = total_usage
        self._step_timings: list[StepTiming] = step_timings or []
        self._trace_entries: list[TraceEntry] = []
        self._summary_extracted: bool = True
        self._summary_value: str | None = None
        self._artifacts: ArtifactStore | None = None

    @property
    def results(self) -> list[QuantedResult[Any]]:
        """The successful branch results.

        Returns:
            List of QuantedResult instances from branches that completed
            without errors.
        """
        return self._parallel_results

    @property
    def errors(self) -> list[Exception]:
        """The exceptions from failed branches.

        Returns:
            List of exceptions raised by branches that failed during execution.
        """
        return self._errors

    @property
    def data(self) -> ParallelOutput:
        """Aggregated output data from all successful branches.

        Returns:
            A ParallelOutput containing the individual data values from
            each successful branch result.
        """
        items = [r.data for r in self._parallel_results]
        return ParallelOutput(items=items)

    @property
    def usage(self) -> RunUsage:
        """Aggregated usage statistics from all successful branches.

        Returns:
            A RunUsage instance with token counts summed across all
            successful branch results.
        """
        total = RunUsage()
        for r in self._parallel_results:
            total.incr(r.usage)
        return total

    @property
    def trace(self) -> list[TraceEntry]:
        """Aggregated trace entries from all successful branches.

        When populated by an outer workflow (e.g., Pipeline sets _trace_entries
        to include preceding step traces plus branch traces), returns those
        directly. When empty (standalone Parallel), falls back to collecting
        trace entries from individual branch results.

        Returns:
            A flat list of TraceEntry objects. Includes outer workflow context
            when available, otherwise collected from branch results.
        """
        if self._trace_entries:
            return self._trace_entries
        entries: list[TraceEntry] = []
        for r in self._parallel_results:
            entries.extend(r.trace)
        return entries

    @property
    def step_timings(self) -> list[StepTiming]:
        """Per-branch timing and usage data.

        Returns the step timings set by Parallel.run() if available,
        otherwise builds timing data from individual branch results.

        Returns:
            A list of StepTiming objects, one per branch.
        """
        if self._step_timings:
            return self._step_timings
        timings: list[StepTiming] = []
        for i, r in enumerate(self._parallel_results):
            timings.append(StepTiming(
                step_name=f"Parallel.branch_{i}",
                duration_seconds=0.0,
                usage=r.usage,
            ))
        return timings

    @property
    def total_usage(self) -> RunUsage:
        """Aggregated token usage across all branches.

        Returns:
            A RunUsage with total_usage if set, otherwise computed from usage.
        """
        if self._total_usage is not None:
            return self._total_usage
        return self.usage

    @property
    def messages(self) -> list[ModelMessage]:
        """Message history (not applicable for parallel execution).

        Returns:
            An empty list since parallel execution has no single message history.
        """
        return []

    @property
    def new_messages(self) -> list[ModelMessage]:
        """New messages (not applicable for parallel execution).

        Returns:
            An empty list since parallel execution has no single message history.
        """
        return []


class Parallel:
    """Concurrent fan-out/fan-in workflow running multiple Runnables.

    Parallel implements the Runnable protocol, enabling it to be nested
    inside other workflows. All branches receive the same input and execute
    concurrently via ``asyncio.gather``. Both successes and errors are
    collected into a ParallelResult.

    Requires at least 2 branches -- a single branch should use the
    Runnable directly instead.

    Example:
        sentiment = QuantedAgent("openai:gpt-4o", input_type=Text, output_type=Sentiment, ...)
        topics = QuantedAgent("openai:gpt-4o", input_type=Text, output_type=Topics, ...)
        parallel = Parallel(branches=[sentiment, topics])
        result = await parallel.run(Text(content="Great product!"))
        print(result.results)  # [QuantedResult[Sentiment], QuantedResult[Topics]]
        print(result.errors)   # [] (empty if all succeeded)
    """

    def __init__(
        self,
        branches: list[Runnable],
        *,
        assembly: ParallelAssemblyFn | None = None,
        store: ArtifactStore | None = None,
        trace_artifacts: bool = False,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Create a new Parallel with the given branches.

        Args:
            branches: List of Runnable instances to execute concurrently.
                Must contain at least 2 branches.
            assembly: Optional assembly function to transform accumulated
                store artifacts into a final output after all branches complete.
            store: Optional ArtifactStore for recording branch outputs.
            trace_artifacts: Whether to write SDK metadata to the store.
            retry_policy: Optional retry configuration for failed branches.
                When provided, branches that fail with exceptions matching
                retry_on types are retried up to max_retries times.

        Raises:
            ValueError: If fewer than 2 branches are provided.
        """
        if len(branches) < 2:
            raise ValueError("Parallel requires at least 2 branches")
        self._branches: list[Runnable] = branches
        self._assembly: ParallelAssemblyFn | None = assembly
        self._store: ArtifactStore | None = store
        self._trace_artifacts: bool = trace_artifacts
        self._retry_policy: RetryPolicy | None = retry_policy

    async def run(self, input_data: BaseModel, **kwargs: Any) -> ParallelResult | QuantedResult[Any]:
        """Run all branches concurrently and collect results.

        Executes all branches in parallel using ``asyncio.gather``.
        Separates outcomes into successful QuantedResult instances and
        Exception instances. When a store is configured, each branch gets
        a namespaced view. When an assembly function is set, it transforms
        accumulated artifacts into the final result.

        Args:
            input_data: A Pydantic BaseModel instance to feed into every branch.
            **kwargs: Additional keyword arguments forwarded to each branch's
                ``run()`` method.

        Returns:
            A ParallelResult (no assembly) or QuantedResult (with assembly)
            containing results and workflow-level observability data.
        """
        trace_writer = _resolve_trace_writer(kwargs)
        store = self._store or (ArtifactStore() if self._assembly is not None else None)

        async def _run_branch(
            branch: Runnable, index: int
        ) -> tuple[int, float, QuantedResult[Any] | Exception]:
            """Run a single branch with timing and optional store writes."""
            branch_store = (
                _NamespacedStore(store, f"branch_{index}") if store is not None else None
            )
            start = time.perf_counter()
            try:
                result = await branch.run(input_data, _trace_writer=trace_writer, **kwargs)
                duration = time.perf_counter() - start
                if branch_store is not None:
                    branch_store["result"] = result.data
                    if result.summary is not None:
                        branch_store["summary"] = result.summary
                return (index, duration, result)
            except Exception as exc:
                duration = time.perf_counter() - start
                return (index, duration, exc)

        coros = [
            _run_branch(branch, i)
            for i, branch in enumerate(self._branches)
        ]
        timed_results = await asyncio.gather(*coros)

        results: list[QuantedResult[Any]] = []
        failed_items: list[tuple[int, Exception]] = []
        step_timings: list[StepTiming] = []
        total_usage = RunUsage()

        for index, duration, outcome in timed_results:
            if isinstance(outcome, Exception):
                failed_items.append((index, outcome))
            else:
                results.append(outcome)
                step_timings.append(StepTiming(
                    step_name=f"Parallel.branch_{index}",
                    duration_seconds=duration,
                    usage=outcome.total_usage,
                ))
                if any(not t.step_name.startswith("QuantedAgent(") for t in outcome._step_timings):
                    step_timings.extend(outcome._step_timings)
                total_usage.incr(outcome.total_usage)

        # Retry logic for failed branches
        policy = self._retry_policy
        if policy is not None and policy.max_retries > 0 and failed_items:
            for attempt in range(policy.max_retries):
                retryable = [
                    (idx, exc) for idx, exc in failed_items
                    if any(isinstance(exc, t) for t in policy.retry_on)
                ]
                if not retryable:
                    break
                if attempt > 0:
                    await asyncio.sleep(policy.delay_seconds)

                still_failed: list[tuple[int, Exception]] = []
                for idx, prev_error in retryable:
                    error_context = (
                        f"Previous attempt failed: "
                        f"{type(prev_error).__name__}: {prev_error}. Try again."
                    )
                    retry_history = [
                        ModelRequest(parts=[UserPromptPart(content=error_context)])
                    ]
                    retry_kwargs = {
                        **kwargs,
                        "_trace_writer": trace_writer,
                        "message_history": retry_history,
                    }
                    start = time.perf_counter()
                    try:
                        retry_result = await self._branches[idx].run(
                            input_data, **retry_kwargs
                        )
                        duration = time.perf_counter() - start
                        results.append(retry_result)
                        step_timings.append(StepTiming(
                            step_name=f"Parallel.branch_{idx}(retry)",
                            duration_seconds=duration,
                            usage=retry_result.total_usage,
                        ))
                        total_usage.incr(retry_result.total_usage)
                    except Exception as exc:
                        still_failed.append((idx, exc))

                # Update failed_items: keep non-retryable + newly failed
                non_retryable = [
                    (idx, exc) for idx, exc in failed_items
                    if not any(isinstance(exc, t) for t in policy.retry_on)
                ]
                failed_items = non_retryable + still_failed

        errors = [exc for _, exc in failed_items]

        if store is not None and self._trace_artifacts:
            store._sdk_set("_parallel/branch_count", len(self._branches))

        if self._assembly is not None:
            parallel_result = ParallelResult(
                results=results, errors=errors,
                total_usage=total_usage, step_timings=step_timings,
            )
            try:
                if asyncio.iscoroutinefunction(self._assembly):
                    assembled = await self._assembly(store, parallel_result)
                else:
                    assembled = self._assembly(store, parallel_result)
            except Exception as exc:
                raise AssemblyError(
                    f"Assembly function failed: {exc}",
                    store=store, last_result=parallel_result, original_error=exc,
                ) from exc
            result = QuantedResult.from_data(assembled, usage=total_usage)
            result._artifacts = store
            result._trace_entries = []
            result._step_timings = step_timings
            result._total_usage = total_usage
            return result

        parallel_result = ParallelResult(
            results=results, errors=errors,
            total_usage=total_usage, step_timings=step_timings,
        )
        if store is not None:
            parallel_result._artifacts = store
        return parallel_result

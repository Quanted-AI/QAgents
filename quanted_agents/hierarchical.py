"""Hierarchical agent primitives: RunnableTool, WorkflowBudget, EscalationPolicy.

Provides the core building blocks for parent-child agent hierarchies with
budget propagation, error control, and artifact store integration.

RunnableTool wraps any Runnable as a pydantic-ai Tool. WorkflowBudget tracks
workflow-wide consumption counters. EscalationPolicy controls which child
exceptions propagate vs. return as error text.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic_ai import Tool
from pydantic_ai._agent_graph import _RunMessages, _messages_ctx_var
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from quanted_agents.artifact_store import _NamespacedStore
from quanted_agents.types import InputTransformFn, Runnable

if TYPE_CHECKING:
    from quanted_agents.artifact_store import ArtifactStore


class EscalationPolicy:
    """Configures which child exceptions propagate vs. return as text.

    Default behavior: ALL exceptions except those in always_escalate are
    caught by RunnableTool and returned as error text to the parent LLM.
    The parent then decides whether to retry, skip, or fail.

    The default always_escalate set contains:
    - UsageLimitExceeded: shared budget pool is exhausted, parent cannot recover
    - KeyboardInterrupt: user requested termination
    - SystemExit: process termination

    Users can customize the set to add domain-specific exceptions that
    should always propagate or remove defaults to catch everything.

    Args:
        always_escalate: Set of exception types that should always
            propagate to the parent. When None, uses the default set.
    """

    DEFAULT: ClassVar[EscalationPolicy]

    def __init__(
        self,
        always_escalate: set[type[Exception]] | None = None,
    ) -> None:
        """Create an EscalationPolicy with the given escalation set.

        Args:
            always_escalate: Set of exception types to always escalate.
                Defaults to {UsageLimitExceeded, KeyboardInterrupt, SystemExit}.
        """
        self._always_escalate: set[type[Exception]] = always_escalate or {
            UsageLimitExceeded,
            KeyboardInterrupt,
            SystemExit,
        }

    def should_escalate(self, exc: Exception) -> bool:
        """Determine whether an exception should propagate to the parent.

        Args:
            exc: The exception raised by the child Runnable.

        Returns:
            True if the exception type is in the always_escalate set,
            False if it should be caught and returned as error text.
        """
        return isinstance(exc, tuple(self._always_escalate))


EscalationPolicy.DEFAULT = EscalationPolicy()


class WorkflowBudget:
    """Tracks workflow-wide budget counters with deduction semantics.

    Wraps multiple named counters (llm_call_limit, tool_call_limit,
    total_request_limit) in a shared pool. Parent agent and all child
    agents draw from the same budget via the deduction model.

    The budget bridges to pydantic-ai's UsageLimits via to_usage_limits(),
    which maps SDK counter names to pydantic-ai's field names. Internal
    counters are stored in a generic dict to accommodate future counter
    types (Phase 18 / EXEC-04).

    Args:
        llm_call_limit: Maximum number of LLM calls (parent + all children).
            Maps to pydantic-ai's request_limit. None means unlimited.
        tool_call_limit: Maximum number of tool executions across the
            hierarchy. Maps to pydantic-ai's tool_calls_limit. None means
            unlimited.
        total_request_limit: Maximum total requests (LLM + tool + any future
            request types). No pydantic-ai equivalent -- tracked internally.
            None means unlimited.
    """

    def __init__(
        self,
        llm_call_limit: int | None = None,
        tool_call_limit: int | None = None,
        total_request_limit: int | None = None,
    ) -> None:
        """Create a WorkflowBudget with the given counter limits.

        Args:
            llm_call_limit: Maximum LLM calls, or None for unlimited.
            tool_call_limit: Maximum tool calls, or None for unlimited.
            total_request_limit: Maximum total requests, or None for unlimited.
        """
        self._counters: dict[str, int | None] = {
            "llm_call_limit": llm_call_limit,
            "tool_call_limit": tool_call_limit,
            "total_request_limit": total_request_limit,
        }

    @property
    def llm_call_limit(self) -> int | None:
        """Remaining LLM call budget, or None if unlimited."""
        return self._counters["llm_call_limit"]

    @property
    def tool_call_limit(self) -> int | None:
        """Remaining tool call budget, or None if unlimited."""
        return self._counters["tool_call_limit"]

    @property
    def total_request_limit(self) -> int | None:
        """Remaining total request budget, or None if unlimited."""
        return self._counters["total_request_limit"]

    def remaining(self, counter: str) -> int | None:
        """Get the remaining count for a named counter.

        Args:
            counter: The counter name (e.g., "llm_call_limit").

        Returns:
            The remaining count, or None if the counter is unlimited
            or does not exist.
        """
        return self._counters.get(counter)

    def to_usage_limits(self) -> UsageLimits:
        """Bridge to pydantic-ai's UsageLimits.

        Maps SDK counter names to pydantic-ai fields:
        - llm_call_limit -> request_limit
        - tool_call_limit -> tool_calls_limit
        - total_request_limit -> no pydantic-ai equivalent (tracked internally)

        Returns:
            A UsageLimits instance for passing to agent.run(usage_limits=...).
        """
        return UsageLimits(
            request_limit=self._counters.get("llm_call_limit"),
            tool_calls_limit=self._counters.get("tool_call_limit"),
        )

    def deduct(self, usage: RunUsage) -> None:
        """Subtract consumed resources from remaining counters.

        Called automatically by RunnableTool's closure after each child run
        completes. Users should NOT call this directly -- RunnableTool
        handles it.

        Only deducts from counters that have limits set (not None).
        Floors at zero to prevent negative values.

        Args:
            usage: The RunUsage from the completed child run.
        """
        if self._counters["llm_call_limit"] is not None:
            self._counters["llm_call_limit"] = max(
                0, self._counters["llm_call_limit"] - usage.requests
            )
        if self._counters["tool_call_limit"] is not None:
            self._counters["tool_call_limit"] = max(
                0, self._counters["tool_call_limit"] - usage.tool_calls
            )
        if self._counters["total_request_limit"] is not None:
            total_consumed = usage.requests + usage.tool_calls
            self._counters["total_request_limit"] = max(
                0, self._counters["total_request_limit"] - total_consumed
            )


class RunnableTool:
    """Wraps a Runnable as a pydantic-ai Tool for hierarchical agent dispatch.

    The parent agent's LLM sees this as a tool with a single `instruction: str`
    parameter. When the LLM calls the tool, RunnableTool:
    1. Optionally transforms the instruction via input_transform
    2. Runs the child Runnable
    3. Writes result.data and result.summary to a namespaced store
    4. Returns result.summary (or str(result.data)) as text to the parent LLM
    5. On error, consults the EscalationPolicy to decide: re-raise or return error text

    Args:
        runnable: The child Runnable to wrap (QuantedAgent, Pipeline, etc.).
        name: Tool name visible to the parent LLM. Must be unique among siblings.
        description: Tool description sent to the parent LLM. Critical for
            routing quality.
        input_transform: Optional closure that converts (store, instruction) into
            the child Runnable's input. Required when child input_type is not str.
        escalation_policy: Error handling policy. When None, uses
            EscalationPolicy.DEFAULT.
    """

    def __init__(
        self,
        runnable: Runnable,
        *,
        name: str,
        description: str,
        input_transform: InputTransformFn | None = None,
        escalation_policy: EscalationPolicy | None = None,
    ) -> None:
        """Create a RunnableTool wrapping a child Runnable.

        Args:
            runnable: The child Runnable to wrap.
            name: Tool name visible to the parent LLM.
            description: Tool description for the parent LLM.
            input_transform: Optional input transformation closure.
            escalation_policy: Error handling policy, defaults to EscalationPolicy.DEFAULT.
        """
        self.runnable: Runnable = runnable
        self.name: str = name
        self.description: str = description
        self.input_transform: InputTransformFn | None = input_transform
        self.escalation_policy: EscalationPolicy = (
            escalation_policy or EscalationPolicy.DEFAULT
        )

    def as_tool(
        self,
        store: ArtifactStore | None = None,
        budget: WorkflowBudget | None = None,
    ) -> Tool:
        """Create a pydantic-ai Tool instance from this RunnableTool.

        Each call creates a new Tool with independent closure bindings. The same
        RunnableTool can be bound to multiple stores/budgets for use across
        different parent agents.

        Args:
            store: Optional ArtifactStore for writing child results. When
                provided, child results are written to a namespaced sub-store.
            budget: Optional WorkflowBudget for tracking consumption. When
                provided, budget.deduct() is called automatically after each
                child run.

        Returns:
            A pydantic-ai Tool ready for registration on a parent agent.

        Raises:
            TypeError: If the child Runnable's input_type is not str and
                no input_transform was provided.
        """
        if self.input_transform is None:
            input_type = getattr(self.runnable, "input_type", str)
            if input_type is not str:
                raise TypeError(
                    f"RunnableTool '{self.name}' has input_type={input_type.__name__} "
                    f"but no input_transform. Either set input_type=str on the child, "
                    f"or provide an input_transform that builds a {input_type.__name__} "
                    f"from (store, instruction)."
                )

        runnable = self.runnable
        tool_name = self.name
        transform = self.input_transform
        policy = self.escalation_policy

        async def _tool_fn(ctx: RunContext[Any], instruction: str) -> str:
            """Dispatch to child Runnable and return text result."""
            # 2. Run child with isolated context and fresh usage.
            # contextvars.copy_context() copies all vars so model overrides,
            # capability settings, and other per-run context inherited by the
            # child agent remain intact. _messages_ctx_var is also copied but
            # _run_child resets it immediately so capture_run_messages() inside
            # the child creates its own independent messages list.
            child_usage = RunUsage()

            async def _run_child() -> Any:
                # Unconditionally set a fresh _RunMessages so any call to
                # capture_run_messages() (or get_captured_run_messages()) inside
                # the child sees an independent list, not the parent's reference.
                # This shadows the copied value within this task's context.
                _messages_ctx_var.set(_RunMessages([]))
                return await runnable.run(input_data, usage=child_usage)

            try:
                # 1. Transform input (inside try so exceptions go through EscalationPolicy)
                if transform is not None:
                    if asyncio.iscoroutinefunction(transform):
                        input_data = await transform(store, instruction)
                    else:
                        input_data = transform(store, instruction)
                else:
                    input_data = instruction

                loop = asyncio.get_running_loop()
                result = await loop.create_task(
                    _run_child(),
                    context=contextvars.copy_context(),
                )
            except UsageLimitExceeded as exc:
                # Budget exhaustion: construct partial message, then check policy
                partial_msg = (
                    f"Child '{tool_name}' exceeded budget. "
                    f"Partial results may be available in the store "
                    f"under '{tool_name}/' namespace."
                )
                if policy.should_escalate(exc):
                    raise
                return partial_msg
            except Exception as exc:
                if policy.should_escalate(exc):
                    raise
                msg = str(exc)
                error_detail = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
                return f"Error running {tool_name}: {error_detail}"

            # 2b. Aggregate child usage into parent (tokens + requests only, NOT tool_calls)
            # Use result.usage since pydantic-ai populates it with the child's final usage.
            child_result_usage = result.usage
            ctx.usage.requests = (ctx.usage.requests or 0) + (child_result_usage.requests or 0)
            ctx.usage.input_tokens = (ctx.usage.input_tokens or 0) + (child_result_usage.input_tokens or 0)
            ctx.usage.output_tokens = (ctx.usage.output_tokens or 0) + (child_result_usage.output_tokens or 0)
            ctx.usage.cache_read_tokens = (
                (ctx.usage.cache_read_tokens or 0) + (child_result_usage.cache_read_tokens or 0)
            )
            ctx.usage.cache_write_tokens = (
                (ctx.usage.cache_write_tokens or 0) + (child_result_usage.cache_write_tokens or 0)
            )
            # Deliberately NOT aggregating tool_calls -- child tool calls must not
            # consume parent's tool_call_limit budget

            # 3. Deduct from WorkflowBudget (if provided)
            if budget is not None:
                budget.deduct(result.usage)

            # 4. Write to namespaced store
            if store is not None:
                child_store = _NamespacedStore(store, tool_name)
                child_store["result"] = result.data
                if result.summary is not None:
                    child_store["summary"] = result.summary

            # 5. Return text to parent LLM
            return result.summary or str(result.data)

        return Tool(
            function=_tool_fn,
            takes_ctx=True,
            name=tool_name,
            description=self.description,
            sequential=True,  # Default sequential=True for concurrent safety per design
        )

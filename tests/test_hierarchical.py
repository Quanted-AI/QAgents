"""Tests for hierarchical agent primitives: RunnableTool, WorkflowBudget, EscalationPolicy.

Validates escalation classification, budget deduction arithmetic, registration-time
validation, error text formatting, and end-to-end parent-child dispatch.
Unit tests use mocked Runnables; integration tests use real LLM calls.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from pydantic import BaseModel
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.hierarchical import EscalationPolicy, RunnableTool, WorkflowBudget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ChildInput(BaseModel):
    """Non-str input type for registration validation tests."""
    query: str


class _ChildOutput(BaseModel):
    """Simple output type for mock results."""
    answer: str


def _make_mock_runnable(
    *,
    input_type: type = str,
    return_data: BaseModel | None = None,
    return_summary: str | None = None,
    raise_exc: Exception | None = None,
    usage: RunUsage | None = None,
) -> MagicMock:
    """Create a mock Runnable with configurable behavior.

    Args:
        input_type: The input_type attribute on the mock.
        return_data: Data to put on the mock result.
        return_summary: Summary to put on the mock result.
        raise_exc: Exception to raise from run().
        usage: RunUsage to attach to the result.

    Returns:
        A MagicMock implementing the Runnable protocol.
    """
    mock = MagicMock()
    mock.input_type = input_type

    result_mock = MagicMock()
    result_mock.data = return_data or _ChildOutput(answer="test")
    result_mock.summary = return_summary
    type(result_mock).usage = PropertyMock(return_value=usage or RunUsage())

    if raise_exc is not None:
        mock.run = AsyncMock(side_effect=raise_exc)
    else:
        mock.run = AsyncMock(return_value=result_mock)

    return mock


def _make_mock_ctx() -> MagicMock:
    """Create a mock RunContext with usage tracking."""
    ctx = MagicMock(spec=RunContext)
    ctx.usage = RunUsage()
    return ctx


# ===========================================================================
# EscalationPolicy Tests
# ===========================================================================

class TestEscalationPolicy(unittest.TestCase):
    """Tests for EscalationPolicy escalation classification."""

    def test_default_escalates_usage_limit_exceeded(self) -> None:
        """Default policy escalates UsageLimitExceeded."""
        self.assertTrue(EscalationPolicy.DEFAULT.should_escalate(UsageLimitExceeded("limit hit")))

    def test_default_escalates_keyboard_interrupt(self) -> None:
        """Default policy escalates KeyboardInterrupt."""
        self.assertTrue(EscalationPolicy.DEFAULT.should_escalate(KeyboardInterrupt()))

    def test_default_escalates_system_exit(self) -> None:
        """Default policy escalates SystemExit."""
        self.assertTrue(EscalationPolicy.DEFAULT.should_escalate(SystemExit()))

    def test_default_catches_value_error(self) -> None:
        """Default policy catches ValueError (returns as text)."""
        self.assertFalse(EscalationPolicy.DEFAULT.should_escalate(ValueError("bad")))

    def test_default_catches_runtime_error(self) -> None:
        """Default policy catches RuntimeError (returns as text)."""
        self.assertFalse(EscalationPolicy.DEFAULT.should_escalate(RuntimeError("oops")))

    def test_custom_policy_escalates_custom_exception(self) -> None:
        """Custom policy escalates a user-defined exception type."""

        class DomainError(Exception):
            pass

        policy = EscalationPolicy(
            always_escalate={DomainError, KeyboardInterrupt, SystemExit}
        )
        self.assertTrue(policy.should_escalate(DomainError("domain issue")))

    def test_permissive_policy_catches_usage_limit(self) -> None:
        """Permissive policy (only KB+SE) catches UsageLimitExceeded."""
        policy = EscalationPolicy(always_escalate={KeyboardInterrupt, SystemExit})
        self.assertFalse(policy.should_escalate(UsageLimitExceeded("limit hit")))

    def test_strict_policy_escalates_everything(self) -> None:
        """Strict policy (Exception) escalates any exception."""
        policy = EscalationPolicy(always_escalate={Exception})
        self.assertTrue(policy.should_escalate(ValueError("anything")))


# ===========================================================================
# WorkflowBudget Tests
# ===========================================================================

class TestWorkflowBudget(unittest.TestCase):
    """Tests for WorkflowBudget counter tracking and deduction."""

    def test_initial_remaining(self) -> None:
        """All counters match constructor args."""
        budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5, total_request_limit=20)
        self.assertEqual(budget.remaining("llm_call_limit"), 10)
        self.assertEqual(budget.remaining("tool_call_limit"), 5)
        self.assertEqual(budget.remaining("total_request_limit"), 20)

    def test_unlimited_counter_returns_none(self) -> None:
        """Default WorkflowBudget() has all None counters."""
        budget = WorkflowBudget()
        self.assertIsNone(budget.remaining("llm_call_limit"))
        self.assertIsNone(budget.remaining("tool_call_limit"))
        self.assertIsNone(budget.remaining("total_request_limit"))

    def test_deduct_subtracts_requests(self) -> None:
        """Deducting usage.requests from llm_call_limit works correctly."""
        budget = WorkflowBudget(llm_call_limit=10)
        usage = RunUsage()
        usage.requests = 3
        budget.deduct(usage)
        self.assertEqual(budget.remaining("llm_call_limit"), 7)

    def test_deduct_subtracts_tool_calls(self) -> None:
        """Deducting usage.tool_calls from tool_call_limit works correctly."""
        budget = WorkflowBudget(tool_call_limit=8)
        usage = RunUsage()
        usage.tool_calls = 2
        budget.deduct(usage)
        self.assertEqual(budget.remaining("tool_call_limit"), 6)

    def test_deduct_subtracts_total(self) -> None:
        """total_request_limit deducts requests + tool_calls combined."""
        budget = WorkflowBudget(total_request_limit=15)
        usage = RunUsage()
        usage.requests = 3
        usage.tool_calls = 2
        budget.deduct(usage)
        self.assertEqual(budget.remaining("total_request_limit"), 10)

    def test_deduct_floors_at_zero(self) -> None:
        """Deducting more than available floors at 0, not negative."""
        budget = WorkflowBudget(llm_call_limit=2)
        usage = RunUsage()
        usage.requests = 10
        budget.deduct(usage)
        self.assertEqual(budget.remaining("llm_call_limit"), 0)

    def test_deduct_skips_none_counters(self) -> None:
        """None counters stay None after deduct."""
        budget = WorkflowBudget(llm_call_limit=10)
        usage = RunUsage()
        usage.requests = 3
        usage.tool_calls = 2
        budget.deduct(usage)
        self.assertIsNone(budget.remaining("tool_call_limit"))
        self.assertIsNone(budget.remaining("total_request_limit"))
        self.assertEqual(budget.remaining("llm_call_limit"), 7)

    def test_to_usage_limits_maps_correctly(self) -> None:
        """to_usage_limits() maps to pydantic-ai field names."""
        budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5)
        limits = budget.to_usage_limits()
        self.assertIsInstance(limits, UsageLimits)
        self.assertEqual(limits.request_limit, 10)
        self.assertEqual(limits.tool_calls_limit, 5)

    def test_properties_match_remaining(self) -> None:
        """Property accessors match remaining() calls."""
        budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5, total_request_limit=20)
        self.assertEqual(budget.llm_call_limit, budget.remaining("llm_call_limit"))
        self.assertEqual(budget.tool_call_limit, budget.remaining("tool_call_limit"))
        self.assertEqual(budget.total_request_limit, budget.remaining("total_request_limit"))


# ===========================================================================
# RunnableTool Registration Tests
# ===========================================================================

class TestRunnableToolRegistration(unittest.TestCase):
    """Tests for RunnableTool registration-time validation."""

    def test_raises_type_error_without_input_transform(self) -> None:
        """as_tool() raises TypeError when input_type is not str and no transform."""
        runnable = _make_mock_runnable(input_type=_ChildInput)
        tool = RunnableTool(
            runnable, name="search", description="Search things"
        )
        with self.assertRaises(TypeError) as ctx:
            tool.as_tool()
        self.assertIn("input_type=_ChildInput", str(ctx.exception))
        self.assertIn("no input_transform", str(ctx.exception))

    def test_no_error_with_input_transform(self) -> None:
        """as_tool() succeeds when input_transform is provided."""
        runnable = _make_mock_runnable(input_type=_ChildInput)

        def transform(store: ArtifactStore, instruction: str) -> _ChildInput:
            return _ChildInput(query=instruction)

        tool = RunnableTool(
            runnable, name="search", description="Search things",
            input_transform=transform,
        )
        result = tool.as_tool()
        self.assertIsNotNone(result)

    def test_no_error_when_input_type_is_str(self) -> None:
        """as_tool() succeeds when input_type is str (no transform needed)."""
        runnable = _make_mock_runnable(input_type=str)
        tool = RunnableTool(
            runnable, name="echo", description="Echo things"
        )
        result = tool.as_tool()
        self.assertIsNotNone(result)

    def test_no_error_when_input_type_attr_missing(self) -> None:
        """as_tool() succeeds when runnable has no input_type attr (defaults to str)."""
        runnable = MagicMock()
        del runnable.input_type
        runnable.run = AsyncMock()
        tool = RunnableTool(
            runnable, name="generic", description="Generic thing"
        )
        result = tool.as_tool()
        self.assertIsNotNone(result)


# ===========================================================================
# RunnableTool Error Formatting Tests
# ===========================================================================

class TestRunnableToolErrorFormatting(unittest.IsolatedAsyncioTestCase):
    """Tests for error text format returned by the tool function."""

    async def test_error_with_message(self) -> None:
        """Error text includes exception type and message."""
        runnable = _make_mock_runnable(raise_exc=ValueError("bad input"))
        rt = RunnableTool(
            runnable, name="broken", description="A broken tool",
            escalation_policy=EscalationPolicy(always_escalate={KeyboardInterrupt}),
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "do something")
        self.assertEqual(result, "Error running broken: ValueError: bad input")

    async def test_error_without_message(self) -> None:
        """Error text uses just exception type when message is empty."""
        runnable = _make_mock_runnable(raise_exc=ValueError())
        rt = RunnableTool(
            runnable, name="broken", description="A broken tool",
            escalation_policy=EscalationPolicy(always_escalate={KeyboardInterrupt}),
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "do something")
        # ValueError() has empty str representation
        self.assertEqual(result, "Error running broken: ValueError")

    async def test_budget_exhaustion_message(self) -> None:
        """Budget exhaustion returns partial result hint when policy allows."""
        runnable = _make_mock_runnable(raise_exc=UsageLimitExceeded("limit hit"))
        permissive = EscalationPolicy(always_escalate={KeyboardInterrupt, SystemExit})
        rt = RunnableTool(
            runnable, name="expensive", description="Expensive tool",
            escalation_policy=permissive,
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "do expensive thing")
        self.assertIn("exceeded budget", result)
        self.assertIn("expensive", result)

    async def test_budget_exhaustion_escalates_by_default(self) -> None:
        """Default policy re-raises UsageLimitExceeded."""
        runnable = _make_mock_runnable(raise_exc=UsageLimitExceeded("limit hit"))
        rt = RunnableTool(
            runnable, name="expensive", description="Expensive tool",
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        with self.assertRaises(UsageLimitExceeded):
            await tool.function(ctx, "do expensive thing")


# ===========================================================================
# RunnableTool input_transform Exception Handling Tests
# ===========================================================================

class TestInputTransformExceptionHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for input_transform exception handling through EscalationPolicy."""

    async def test_input_transform_sync_exception_returns_error_text(self) -> None:
        """Sync input_transform raising ValueError with default policy returns error text."""
        runnable = _make_mock_runnable(input_type=_ChildInput)

        def transform(s: ArtifactStore, instr: str) -> _ChildInput:
            raise ValueError("zero datasets")

        rt = RunnableTool(
            runnable,
            name="insights_cluster",
            description="Insights tool",
            input_transform=transform,
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "test")
        self.assertEqual(result, "Error running insights_cluster: ValueError: zero datasets")

    async def test_input_transform_async_exception_returns_error_text(self) -> None:
        """Async input_transform raising KeyError with default policy returns error text."""
        runnable = _make_mock_runnable(input_type=_ChildInput)

        async def transform(s: ArtifactStore, instr: str) -> _ChildInput:
            raise KeyError("missing_key")

        rt = RunnableTool(
            runnable,
            name="lookup_cluster",
            description="Lookup tool",
            input_transform=transform,
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "test")
        self.assertEqual(result, "Error running lookup_cluster: KeyError: 'missing_key'")

    async def test_input_transform_exception_escalates_with_strict_policy(self) -> None:
        """Sync input_transform raising ValueError with strict policy re-raises ValueError."""
        runnable = _make_mock_runnable(input_type=_ChildInput)

        def transform(s: ArtifactStore, instr: str) -> _ChildInput:
            raise ValueError("should escalate")

        strict_policy = EscalationPolicy(always_escalate={Exception})
        rt = RunnableTool(
            runnable,
            name="strict_cluster",
            description="Strict tool",
            input_transform=transform,
            escalation_policy=strict_policy,
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        with self.assertRaises(ValueError):
            await tool.function(ctx, "test")


# ===========================================================================
# RunnableTool Dispatch Flow Tests
# ===========================================================================

class TestRunnableToolDispatch(unittest.IsolatedAsyncioTestCase):
    """Tests for the full dispatch flow through _tool_fn."""

    async def test_returns_summary_when_available(self) -> None:
        """Tool returns result.summary when it is not None."""
        runnable = _make_mock_runnable(return_summary="Here is the summary")
        rt = RunnableTool(runnable, name="summarizer", description="Summarize")
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "summarize this")
        self.assertEqual(result, "Here is the summary")

    async def test_returns_str_data_when_no_summary(self) -> None:
        """Tool returns str(result.data) when summary is None."""
        data = _ChildOutput(answer="the answer")
        runnable = _make_mock_runnable(return_data=data, return_summary=None)
        rt = RunnableTool(runnable, name="answerer", description="Answer")
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "answer this")
        self.assertIn("the answer", result)

    async def test_writes_to_namespaced_store(self) -> None:
        """Tool writes result to namespaced store when store provided."""
        data = _ChildOutput(answer="stored")
        runnable = _make_mock_runnable(return_data=data, return_summary="summary")
        store = ArtifactStore()
        rt = RunnableTool(runnable, name="writer", description="Write")
        tool = rt.as_tool(store=store)
        ctx = _make_mock_ctx()
        await tool.function(ctx, "write this")
        self.assertEqual(store["writer/result"], data)
        self.assertEqual(store["writer/summary"], "summary")

    async def test_deducts_budget_after_run(self) -> None:
        """Tool deducts from budget after successful child run."""
        usage = RunUsage()
        usage.requests = 3
        usage.tool_calls = 1
        runnable = _make_mock_runnable(usage=usage)
        budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5)
        rt = RunnableTool(runnable, name="budgeted", description="Budgeted")
        tool = rt.as_tool(budget=budget)
        ctx = _make_mock_ctx()
        await tool.function(ctx, "do it")
        self.assertEqual(budget.remaining("llm_call_limit"), 7)
        self.assertEqual(budget.remaining("tool_call_limit"), 4)

    async def test_input_transform_called(self) -> None:
        """Input transform is called with store and instruction."""
        runnable = _make_mock_runnable(input_type=_ChildInput)
        store = ArtifactStore()

        def transform(s: ArtifactStore, instruction: str) -> _ChildInput:
            return _ChildInput(query=instruction)

        rt = RunnableTool(
            runnable, name="transformed", description="Transformed",
            input_transform=transform,
        )
        tool = rt.as_tool(store=store)
        ctx = _make_mock_ctx()
        await tool.function(ctx, "search for X")
        call_args = runnable.run.call_args
        input_data = call_args[0][0]
        self.assertIsInstance(input_data, _ChildInput)
        self.assertEqual(input_data.query, "search for X")

    async def test_async_input_transform(self) -> None:
        """Async input transform is awaited correctly."""
        runnable = _make_mock_runnable(input_type=_ChildInput)

        async def async_transform(s: ArtifactStore, instruction: str) -> _ChildInput:
            return _ChildInput(query=f"async: {instruction}")

        rt = RunnableTool(
            runnable, name="async_t", description="Async transform",
            input_transform=async_transform,
        )
        tool = rt.as_tool()
        ctx = _make_mock_ctx()
        await tool.function(ctx, "test input")
        call_args = runnable.run.call_args
        input_data = call_args[0][0]
        self.assertEqual(input_data.query, "async: test input")

    async def test_no_store_write_when_store_is_none(self) -> None:
        """No store writes occur when store is None."""
        runnable = _make_mock_runnable(return_summary="result")
        rt = RunnableTool(runnable, name="no_store", description="No store")
        tool = rt.as_tool(store=None)
        ctx = _make_mock_ctx()
        result = await tool.function(ctx, "do it")
        self.assertEqual(result, "result")

    async def test_no_budget_deduction_when_budget_is_none(self) -> None:
        """No deduction occurs when budget is None."""
        runnable = _make_mock_runnable()
        rt = RunnableTool(runnable, name="no_budget", description="No budget")
        tool = rt.as_tool(budget=None)
        ctx = _make_mock_ctx()
        await tool.function(ctx, "do it")
        # No exception means success -- budget=None is handled


# ===========================================================================
# Integration Tests (real LLM)
# ===========================================================================

class TestHierarchicalIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real LLM calls using anthropic:claude-haiku-4-5."""

    def setUp(self) -> None:
        """Enable real model requests for integration tests."""
        from pydantic_ai import models
        self._original_allow = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

    def tearDown(self) -> None:
        """Restore model request guard."""
        from pydantic_ai import models
        models.ALLOW_MODEL_REQUESTS = self._original_allow

    async def test_parent_dispatches_to_child(self) -> None:
        """Parent agent dispatches to a child agent via RunnableTool."""
        from pydantic_ai.usage import UsageLimits as PaiUsageLimits

        from quanted_agents import QuantedAgent

        class ChildInput(BaseModel):
            question: str

        class ChildOutput(BaseModel):
            answer: str

        class ParentInput(BaseModel):
            task: str

        class ParentOutput(BaseModel):
            result: str

        child = QuantedAgent(
            "anthropic:claude-haiku-4-5",
            input_type=ChildInput,
            output_type=ChildOutput,
            system_prompt="Answer the question in one sentence.",
        )

        def transform(store: ArtifactStore, instruction: str) -> ChildInput:
            return ChildInput(question=instruction)

        child_tool = RunnableTool(
            child, name="ask_child", description="Ask a question to the child agent.",
            input_transform=transform,
        )

        store = ArtifactStore()
        parent = QuantedAgent(
            "anthropic:claude-haiku-4-5",
            input_type=ParentInput,
            output_type=ParentOutput,
            system_prompt="You are a parent agent. Use the ask_child tool to answer the task.",
            tools=[child_tool.as_tool(store=store)],
        )

        result = await parent.run(
            ParentInput(task="What is the capital of France?"),
            usage_limits=PaiUsageLimits(request_limit=5),
        )
        self.assertIsNotNone(result.data)
        self.assertIsInstance(result.data, ParentOutput)
        self.assertTrue(len(result.data.result) > 0)

    async def test_budget_propagation(self) -> None:
        """WorkflowBudget counters decrease after child dispatch."""
        from pydantic_ai.usage import UsageLimits as PaiUsageLimits

        from quanted_agents import QuantedAgent

        class ChildInput(BaseModel):
            question: str

        class ChildOutput(BaseModel):
            answer: str

        class ParentInput(BaseModel):
            task: str

        class ParentOutput(BaseModel):
            result: str

        child = QuantedAgent(
            "anthropic:claude-haiku-4-5",
            input_type=ChildInput,
            output_type=ChildOutput,
            system_prompt="Answer briefly.",
        )

        def transform(store: ArtifactStore, instruction: str) -> ChildInput:
            return ChildInput(question=instruction)

        budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5)

        child_tool = RunnableTool(
            child, name="ask_child", description="Ask a question.",
            input_transform=transform,
        )

        store = ArtifactStore()
        parent = QuantedAgent(
            "anthropic:claude-haiku-4-5",
            input_type=ParentInput,
            output_type=ParentOutput,
            system_prompt="Use ask_child to answer the task.",
            tools=[child_tool.as_tool(store=store, budget=budget)],
        )

        initial_llm = budget.llm_call_limit
        initial_tool = budget.tool_call_limit
        await parent.run(
            ParentInput(task="What is 2+2?"),
            usage_limits=PaiUsageLimits(request_limit=5),
        )
        # Budget should have decreased (child made at least 1 LLM call)
        self.assertLess(budget.remaining("llm_call_limit"), initial_llm)


# ===========================================================================
# RunnableTool Isolation Tests
# ===========================================================================

class TestRunnableToolIsolation(unittest.IsolatedAsyncioTestCase):
    """Tests for ContextVar isolation and RunUsage isolation in _tool_fn.

    Verifies that child agents receive an isolated message context (ContextVar)
    and an isolated RunUsage so that child tool_calls do not consume the parent's
    tool_call_limit budget, while child tokens and requests do aggregate back.
    """

    async def test_child_gets_fresh_context_var(self) -> None:
        """Child agent runs with a fresh _messages_ctx_var, not inheriting parent's reference.

        The parent sets _messages_ctx_var to a sentinel _RunMessages. After _tool_fn
        dispatches the child, the child's run() must observe a DIFFERENT _RunMessages
        object (a fresh one), never the parent's list reference. This ensures
        capture_run_messages() in the child creates an independent messages list.
        """
        from pydantic_ai._agent_graph import _RunMessages, _messages_ctx_var

        sentinel_messages = _RunMessages([])
        observed_in_child: list[object] = []

        async def _fake_run(input_data: str, usage: RunUsage | None = None) -> MagicMock:
            """Capture what _messages_ctx_var looks like inside the child's task."""
            try:
                val = _messages_ctx_var.get()
                observed_in_child.append(val)
            except LookupError:
                observed_in_child.append("MISSING")
            result_mock = MagicMock()
            result_mock.data = _ChildOutput(answer="test")
            result_mock.summary = "done"
            type(result_mock).usage = PropertyMock(return_value=RunUsage())
            return result_mock

        runnable = MagicMock()
        runnable.input_type = str
        runnable.run = _fake_run

        rt = RunnableTool(runnable, name="child", description="Child tool")
        tool = rt.as_tool()
        ctx = _make_mock_ctx()

        # Set parent's context var to sentinel_messages before calling tool
        token = _messages_ctx_var.set(sentinel_messages)
        try:
            await tool.function(ctx, "hello")
        finally:
            _messages_ctx_var.reset(token)

        # Child must have observed a DIFFERENT _RunMessages, not the parent's sentinel
        self.assertEqual(len(observed_in_child), 1)
        child_observed = observed_in_child[0]
        self.assertIsInstance(child_observed, _RunMessages,
                              "Child must see a _RunMessages instance (not LookupError)")
        self.assertIsNot(child_observed, sentinel_messages,
                         "Child must see a fresh _RunMessages, not the parent's reference")

    async def test_child_usage_does_not_consume_parent_tool_calls(self) -> None:
        """Child tool_calls must NOT accumulate into parent's ctx.usage.tool_calls.

        After the child completes with tool_calls=5, the parent's ctx.usage.tool_calls
        must remain 0. Child tool_calls must not consume the parent's tool_call_limit.
        """
        child_usage = RunUsage()
        child_usage.tool_calls = 5
        child_usage.requests = 2
        child_usage.input_tokens = 100

        runnable = _make_mock_runnable(usage=child_usage, return_summary="done")
        rt = RunnableTool(runnable, name="child", description="Child tool")
        tool = rt.as_tool()
        ctx = _make_mock_ctx()

        await tool.function(ctx, "do it")

        self.assertEqual(ctx.usage.tool_calls, 0,
                         "Child tool_calls must not be aggregated into parent ctx.usage")

    async def test_child_tokens_aggregate_to_parent_usage(self) -> None:
        """Child input_tokens, output_tokens, and requests aggregate back to parent ctx.usage.

        Only tool_calls are excluded from aggregation. All other usage fields
        (tokens, requests) must accumulate into the parent's ctx.usage.
        """
        child_usage = RunUsage()
        child_usage.input_tokens = 100
        child_usage.output_tokens = 50
        child_usage.requests = 2
        child_usage.tool_calls = 5

        runnable = _make_mock_runnable(usage=child_usage, return_summary="done")
        rt = RunnableTool(runnable, name="child", description="Child tool")
        tool = rt.as_tool()
        ctx = _make_mock_ctx()

        await tool.function(ctx, "do it")

        self.assertEqual(ctx.usage.input_tokens, 100,
                         "Child input_tokens must aggregate to parent ctx.usage")
        self.assertEqual(ctx.usage.output_tokens, 50,
                         "Child output_tokens must aggregate to parent ctx.usage")
        self.assertEqual(ctx.usage.requests, 2,
                         "Child requests must aggregate to parent ctx.usage")

    async def test_budget_deduct_receives_full_child_usage(self) -> None:
        """WorkflowBudget.deduct() still receives full child usage including tool_calls.

        Even though child tool_calls are excluded from ctx.usage aggregation,
        budget.deduct() must still receive the child's complete RunUsage
        (including tool_calls) for accurate workflow-level budget tracking.
        """
        child_usage = RunUsage()
        child_usage.tool_calls = 5
        child_usage.requests = 2

        runnable = _make_mock_runnable(usage=child_usage, return_summary="done")
        budget = MagicMock(spec=WorkflowBudget)
        rt = RunnableTool(runnable, name="child", description="Child tool")
        tool = rt.as_tool(budget=budget)
        ctx = _make_mock_ctx()

        await tool.function(ctx, "do it")

        budget.deduct.assert_called_once()
        deducted_usage = budget.deduct.call_args[0][0]
        self.assertEqual(deducted_usage.tool_calls, 5,
                         "budget.deduct must receive full child usage including tool_calls")
        self.assertEqual(deducted_usage.requests, 2,
                         "budget.deduct must receive full child usage including requests")


if __name__ == "__main__":
    unittest.main()

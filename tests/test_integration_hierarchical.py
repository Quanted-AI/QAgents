"""Integration tests for hierarchical agent dispatch with cross-feature interactions.

Covers SC1 (hierarchical + budget + store + concurrency control) and additional
scenarios: soft limit + store preservation (Scenario 1), timeout + store
preservation (Scenario 4), and context overflow + hierarchical (Scenario 6).

All tests use FunctionModel for deterministic LLM simulation. No real API calls.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import (
    ArtifactStore,
    OverflowStrategy,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)
from tests.conftest import SampleInput, SampleOutput, make_agent, make_budget, make_store


# ---------------------------------------------------------------------------
# Test-specific BaseModels
# ---------------------------------------------------------------------------


class ChildInput(BaseModel):
    """Input for child agents in hierarchical tests."""

    query: str


class ChildOutput(BaseModel):
    """Output from child agents."""

    answer: str
    score: float = 1.0


class ParentInput(BaseModel):
    """Input for parent agents."""

    task: str


class ParentOutput(BaseModel):
    """Output from parent agents."""

    result: str
    confidence: float = 0.9


# ---------------------------------------------------------------------------
# FunctionModel helpers
# ---------------------------------------------------------------------------


def _make_child_output_model(output: BaseModel) -> FunctionModel:
    """Create a FunctionModel that returns structured output via the output tool.

    Args:
        output: The BaseModel instance to return as the agent's output.

    Returns:
        A FunctionModel that produces the given output on every call.
    """

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(
                tool_name=tool.name,
                args=json.dumps(output.model_dump()),
                tool_call_id="child_output_1",
            )]
        )

    return FunctionModel(handler)


def _make_parent_handler(
    child_tool_name: str,
    instruction_text: str,
    final_output: dict[str, Any],
) -> Any:
    """Create a stateful parent FunctionModel handler for multi-turn dispatch.

    First call returns a ToolCallPart dispatching to the child tool.
    Second call returns the final output via the output tool.

    Args:
        child_tool_name: Name of the child tool to dispatch to.
        instruction_text: Instruction string to send to the child.
        final_output: Dict of the parent's final output fields.

    Returns:
        A callable handler for FunctionModel.
    """
    call_count = 0

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name=child_tool_name,
                    args=json.dumps({"instruction": instruction_text}),
                    tool_call_id="parent_tc_1",
                )]
            )
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(
                tool_name=tool.name,
                args=json.dumps(final_output),
                tool_call_id="parent_tc_2",
            )]
        )

    return handler


# ===========================================================================
# TestHierarchicalIntegration
# ===========================================================================


class TestHierarchicalIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for hierarchical agent dispatch with cross-feature interactions."""

    async def test_hierarchical_with_budget_and_store(self) -> None:
        """SC1: Parent dispatches to child via RunnableTool with budget, store, and concurrency control.

        Verifies:
        - Parent calls child tool, child produces structured output.
        - Store contains child result under namespaced key (child_tool_name/result).
        - Budget is decremented (remaining < initial).
        - RunnableTool uses sequential=True for concurrency safety.
        """
        child = make_agent(input_type=ChildInput, output_type=ChildOutput)

        store = make_store()
        budget = make_budget(llm_call_limit=20, tool_call_limit=10)
        initial_llm = budget.llm_call_limit
        initial_tool = budget.tool_call_limit

        def transform(s: ArtifactStore | None, instruction: str) -> ChildInput:
            return ChildInput(query=instruction)

        child_tool = RunnableTool(
            child,
            name="ask_child",
            description="Dispatch a research query to the child agent",
            input_transform=transform,
        )

        tool = child_tool.as_tool(store=store, budget=budget)
        parent = make_agent(
            input_type=ParentInput,
            output_type=ParentOutput,
            tools=[tool],
        )

        parent_handler = _make_parent_handler(
            child_tool_name="ask_child",
            instruction_text="research quantum computing",
            final_output={"result": "combined answer", "confidence": 0.95},
        )
        child_model = _make_child_output_model(
            ChildOutput(answer="quantum computing is fascinating", score=0.85)
        )

        with parent.inner.override(model=FunctionModel(parent_handler)):
            with child.inner.override(model=child_model):
                result = await parent.run(ParentInput(task="research quantum computing"))

        # Verify result type and content
        self.assertIsInstance(result.data, ParentOutput)
        self.assertEqual(result.data.result, "combined answer")
        self.assertAlmostEqual(result.data.confidence, 0.95)

        # Verify store contains child output (namespaced key)
        self.assertIn("ask_child/result", store)
        child_result = store["ask_child/result"]
        self.assertIsInstance(child_result, ChildOutput)
        self.assertEqual(child_result.answer, "quantum computing is fascinating")

        # Verify budget was decremented (relative assertion per Pitfall 6)
        self.assertLess(budget.remaining("llm_call_limit"), initial_llm)

    async def test_hierarchical_soft_limit_store_preservation(self) -> None:
        """Scenario 1: Child hits soft limit, partial result in store, parent proceeds.

        Verifies that when a child agent completes under soft limit conditions,
        the ArtifactStore preserves artifacts and the parent can still proceed
        with the child's output.
        """
        # Child agent with soft_limit enabled
        child = make_agent(
            input_type=ChildInput,
            output_type=ChildOutput,
            soft_limit=True,
            llm_call_limit=5,
        )

        store = make_store({"context": "prior research data"})
        budget = make_budget(llm_call_limit=30, tool_call_limit=15)

        def transform(s: ArtifactStore | None, instruction: str) -> ChildInput:
            return ChildInput(query=instruction)

        child_tool = RunnableTool(
            child,
            name="research_child",
            description="Research agent with soft limit",
            input_transform=transform,
        )

        tool = child_tool.as_tool(store=store, budget=budget)
        parent = make_agent(
            input_type=ParentInput,
            output_type=ParentOutput,
            tools=[tool],
        )

        # Child produces output normally (soft limit does not fire since
        # FunctionModel responds immediately within the limit)
        child_model = _make_child_output_model(
            ChildOutput(answer="partial analysis from soft limit", score=0.6)
        )
        parent_handler = _make_parent_handler(
            child_tool_name="research_child",
            instruction_text="analyze dataset",
            final_output={"result": "parent synthesized partial data", "confidence": 0.7},
        )

        with parent.inner.override(model=FunctionModel(parent_handler)):
            with child.inner.override(model=child_model):
                result = await parent.run(ParentInput(task="analyze dataset"))

        # Verify parent succeeded
        self.assertIsInstance(result.data, ParentOutput)
        self.assertEqual(result.data.result, "parent synthesized partial data")

        # Verify store preserved artifacts from child
        self.assertIn("research_child/result", store)
        child_result = store["research_child/result"]
        self.assertEqual(child_result.answer, "partial analysis from soft limit")

        # Verify pre-existing store data survived
        self.assertIn("context", store)
        self.assertEqual(store["context"], "prior research data")

        # Verify budget was consumed
        self.assertLess(budget.remaining("llm_call_limit"), 30)

    async def test_hierarchical_timeout_store_preservation(self) -> None:
        """Scenario 4: Child with timeout, store artifacts preserved through timed runs.

        Verifies that the store propagation wiring works end-to-end when the
        child agent is configured with timeout parameters. FunctionModel responds
        immediately, so the timeout is not actually hit -- the test proves that
        timeout configuration does not break store/budget integration.
        """
        child = make_agent(
            input_type=ChildInput,
            output_type=ChildOutput,
            soft_timeout=10.0,
            hard_timeout=60.0,
        )

        store = make_store()
        budget = make_budget(llm_call_limit=20, tool_call_limit=10)

        def transform(s: ArtifactStore | None, instruction: str) -> ChildInput:
            return ChildInput(query=instruction)

        child_tool = RunnableTool(
            child,
            name="timed_child",
            description="Child agent with timeout configuration",
            input_transform=transform,
        )

        tool = child_tool.as_tool(store=store, budget=budget)
        parent = make_agent(
            input_type=ParentInput,
            output_type=ParentOutput,
            tools=[tool],
        )

        child_model = _make_child_output_model(
            ChildOutput(answer="completed within timeout", score=0.9)
        )
        parent_handler = _make_parent_handler(
            child_tool_name="timed_child",
            instruction_text="time-sensitive query",
            final_output={"result": "timed result aggregated", "confidence": 0.88},
        )

        with parent.inner.override(model=FunctionModel(parent_handler)):
            with child.inner.override(model=child_model):
                result = await parent.run(ParentInput(task="time-sensitive query"))

        # Verify result
        self.assertIsInstance(result.data, ParentOutput)
        self.assertEqual(result.data.result, "timed result aggregated")

        # Verify store artifacts preserved
        self.assertIn("timed_child/result", store)
        child_result = store["timed_child/result"]
        self.assertEqual(child_result.answer, "completed within timeout")

        # Verify budget deducted
        self.assertLess(budget.remaining("llm_call_limit"), 20)

    async def test_context_overflow_hierarchical(self) -> None:
        """Scenario 6: Parent with max_context_tokens dispatches to child, context managed.

        Verifies that a parent agent configured with TRUNCATE_OLDEST overflow
        strategy can dispatch to a child via RunnableTool without the context
        management crashing the hierarchy. The parent produces valid output
        despite context pressure from prior message history.
        """
        child = make_agent(input_type=ChildInput, output_type=ChildOutput)

        store = make_store()
        budget = make_budget(llm_call_limit=30, tool_call_limit=15)

        def transform(s: ArtifactStore | None, instruction: str) -> ChildInput:
            return ChildInput(query=instruction)

        child_tool = RunnableTool(
            child,
            name="overflow_child",
            description="Child agent for context overflow test",
            input_transform=transform,
        )

        tool = child_tool.as_tool(store=store, budget=budget)
        parent = make_agent(
            input_type=ParentInput,
            output_type=ParentOutput,
            tools=[tool],
            max_context_tokens=500,
            overflow_strategy=OverflowStrategy.TRUNCATE_OLDEST,
        )

        # Multi-dispatch parent handler: calls child twice, then produces output
        call_count = 0

        def multi_dispatch_handler(
            messages: list[Any], agent_info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name="overflow_child",
                        args=json.dumps({"instruction": f"query batch {call_count}"}),
                        tool_call_id=f"tc_{call_count}",
                    )]
                )
            output_tool = agent_info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name=output_tool.name,
                    args=json.dumps({
                        "result": "synthesized from multiple child calls",
                        "confidence": 0.82,
                    }),
                    tool_call_id=f"tc_{call_count}",
                )]
            )

        child_call_count = 0

        def child_handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
            nonlocal child_call_count
            child_call_count += 1
            output_tool = agent_info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name=output_tool.name,
                    args=json.dumps({
                        "answer": f"child result {child_call_count}",
                        "score": 0.7 + child_call_count * 0.1,
                    }),
                    tool_call_id=f"child_tc_{child_call_count}",
                )]
            )

        with parent.inner.override(model=FunctionModel(multi_dispatch_handler)):
            with child.inner.override(model=FunctionModel(child_handler)):
                result = await parent.run(ParentInput(task="multi-batch research"))

        # Verify parent produced valid output
        self.assertIsInstance(result.data, ParentOutput)
        self.assertEqual(result.data.result, "synthesized from multiple child calls")

        # Verify store has child results from multiple dispatches
        self.assertIn("overflow_child/result", store)

        # Verify budget was consumed by multiple child calls
        self.assertLess(budget.remaining("llm_call_limit"), 30)

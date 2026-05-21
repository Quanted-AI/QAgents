"""Tests for nested workflow observability: usage aggregation and step_timings flattening.

Validates that when workflows are nested (Pipeline contains Router, Pipeline
contains Pipeline, Loop contains Pipeline), total_usage correctly aggregates
all inner agent invocations and step_timings includes inner workflow breakdowns
alongside outer summary entries.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from quanted_agents.workflows import Loop, Pipeline, Router
from quanted_agents.workflows.router import RoutingDecision


class StepInput(BaseModel):
    """Input model for nested observability tests."""

    text: str


class StepMiddle(BaseModel):
    """Intermediate model between pipeline steps."""

    processed: str


class StepOutput(BaseModel):
    """Output model for nested observability tests."""

    result: str


def _make_dispatcher_function(target: str, reasoning: str = "test routing"):
    """Create a FunctionModel handler that returns a RoutingDecision via output tool.

    Args:
        target: The specialist name to route to.
        reasoning: Explanation for the routing decision.

    Returns:
        A function compatible with FunctionModel that returns the routing decision.
    """

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        """Return a RoutingDecision as a tool call response."""
        decision = {"target": target, "reasoning": reasoning}
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(decision))]
        )

    return handler


class TestNestedWorkflowObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for nested workflow observability: Pipeline+Router, Pipeline+Pipeline, Loop+Pipeline."""

    async def test_pipeline_with_router_total_usage_includes_all_agents(self) -> None:
        """Pipeline containing Router reports total_usage that includes all 3 agent invocations."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Step 1",
        )
        dispatcher = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=RoutingDecision,
            system_prompt="Classify",
        )
        specialist = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Handle",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"handle": specialist},
        )
        pipeline = Pipeline(steps=[step1, router])

        dispatcher_model = FunctionModel(_make_dispatcher_function("handle"))
        with step1.inner.override(model=TestModel()):
            with dispatcher.inner.override(model=dispatcher_model):
                with specialist.inner.override(model=TestModel()):
                    result = await pipeline.run(StepInput(text="test"))

        self.assertGreaterEqual(result.total_usage.requests, 3)
        self.assertGreater(
            result.total_usage.input_tokens,
            result.usage.input_tokens,
        )

    async def test_pipeline_with_router_step_timings_includes_inner_breakdown(self) -> None:
        """Pipeline containing Router has step_timings with outer and inner Router entries."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Step 1",
        )
        dispatcher = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=RoutingDecision,
            system_prompt="Classify",
        )
        specialist = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Handle",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"handle": specialist},
        )
        pipeline = Pipeline(steps=[step1, router])

        dispatcher_model = FunctionModel(_make_dispatcher_function("handle"))
        with step1.inner.override(model=TestModel()):
            with dispatcher.inner.override(model=dispatcher_model):
                with specialist.inner.override(model=TestModel()):
                    result = await pipeline.run(StepInput(text="test"))

        step_names = [t.step_name for t in result.step_timings]
        self.assertIn("Pipeline.step_0", step_names)
        self.assertIn("Pipeline.step_1", step_names)
        self.assertIn("Router.dispatcher", step_names)
        specialist_entries = [n for n in step_names if n.startswith("Router.specialist_")]
        self.assertGreater(len(specialist_entries), 0)
        self.assertGreaterEqual(len(result.step_timings), 4)

    async def test_pipeline_containing_pipeline_total_usage(self) -> None:
        """Pipeline containing inner Pipeline reports total_usage from all 3 agents."""
        outer_step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Outer step 1",
        )
        inner_step1 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepMiddle,
            system_prompt="Inner step 1",
        )
        inner_step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Inner step 2",
        )
        inner_pipeline = Pipeline(steps=[inner_step1, inner_step2])
        outer_pipeline = Pipeline(steps=[outer_step1, inner_pipeline])

        with outer_step1.inner.override(model=TestModel()):
            with inner_step1.inner.override(model=TestModel()):
                with inner_step2.inner.override(model=TestModel()):
                    result = await outer_pipeline.run(StepInput(text="test"))

        self.assertGreaterEqual(result.total_usage.requests, 3)

    async def test_pipeline_containing_pipeline_step_timings_flattened(self) -> None:
        """Pipeline containing inner Pipeline has flattened step_timings from both levels."""
        outer_step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Outer step 1",
        )
        inner_step1 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepMiddle,
            system_prompt="Inner step 1",
        )
        inner_step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Inner step 2",
        )
        inner_pipeline = Pipeline(steps=[inner_step1, inner_step2])
        outer_pipeline = Pipeline(steps=[outer_step1, inner_pipeline])

        with outer_step1.inner.override(model=TestModel()):
            with inner_step1.inner.override(model=TestModel()):
                with inner_step2.inner.override(model=TestModel()):
                    result = await outer_pipeline.run(StepInput(text="test"))

        step_names = [t.step_name for t in result.step_timings]
        pipeline_step_0_count = step_names.count("Pipeline.step_0")
        pipeline_step_1_count = step_names.count("Pipeline.step_1")
        self.assertGreaterEqual(pipeline_step_0_count, 2)
        self.assertGreaterEqual(pipeline_step_1_count, 1)
        self.assertGreaterEqual(len(result.step_timings), 4)

    async def test_loop_with_nested_pipeline_total_usage(self) -> None:
        """Loop with Pipeline body aggregates total_usage across iterations and inner steps."""

        class LoopState(BaseModel):
            """State model for loop iteration tracking."""

            content: str
            iteration: int = 0

        iteration_count = [0]

        def _loop_step2_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            """Return LoopState with incremented iteration count."""
            iteration_count[0] += 1
            state = LoopState(content="refined", iteration=iteration_count[0])
            return ModelResponse(parts=[TextPart(content=json.dumps(state.model_dump()))])

        pipe_step1 = QuantedAgent(
            "test",
            input_type=LoopState,
            output_type=StepMiddle,
            system_prompt="Process",
        )
        pipe_step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=LoopState,
            system_prompt="Convert back",
        )
        pipeline_body = Pipeline(steps=[pipe_step1, pipe_step2])

        loop = Loop(
            body=pipeline_body,
            termination_check=lambda d: d.iteration >= 2,
            max_iterations=5,
        )

        with pipe_step1.inner.override(model=TestModel()):
            with pipe_step2.inner.override(model=FunctionModel(_loop_step2_model)):
                result = await loop.run(LoopState(content="start", iteration=0))

        self.assertGreaterEqual(result.total_usage.requests, 4)

        step_names = [t.step_name for t in result.step_timings]
        self.assertIn("Loop.iteration_0", step_names)
        self.assertIn("Loop.iteration_1", step_names)
        pipeline_steps = [n for n in step_names if n.startswith("Pipeline.step_")]
        self.assertGreater(len(pipeline_steps), 0)


if __name__ == "__main__":
    unittest.main()

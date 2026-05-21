"""Tests for workflow nesting and composability (WKFL-07, WKFL-08).

Validates that workflow primitives compose recursively: Pipeline steps can be
Routers, Loop bodies can be Pipelines, Pipeline steps can be Parallels, and
deep nesting (3+ levels) works correctly. Also verifies manual output capture
via QuantedResult.data for custom routing logic (WKFL-07).
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from quanted_agents import QuantedAgent
from quanted_agents.types import Runnable
from quanted_agents.workflows import Loop, Parallel, Pipeline, Router
from quanted_agents.workflows.parallel import ParallelOutput
from quanted_agents.workflows.router import RoutingDecision


class TextInput(BaseModel):
    """Input model for nesting tests."""

    text: str


class ProcessedText(BaseModel):
    """Intermediate model after first processing step."""

    processed: str


class FinalOutput(BaseModel):
    """Final output model for pipeline and nesting tests."""

    result: str


def _make_dispatcher_function(target: str):
    """Create a FunctionModel handler that returns a RoutingDecision via output tool.

    Args:
        target: The specialist name to route to.

    Returns:
        A function compatible with FunctionModel that returns the routing decision.
    """

    def handler(messages: list[Any], agent_info: AgentInfo) -> ModelResponse:
        """Return a RoutingDecision as a tool call response."""
        decision = {"target": target, "reasoning": "test routing"}
        tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(decision))]
        )

    return handler


class TestWorkflowNesting(unittest.IsolatedAsyncioTestCase):
    """Tests for workflow nesting: Pipeline+Router, Loop+Pipeline, Pipeline+Parallel, deep nesting."""

    async def test_pipeline_with_router_step(self) -> None:
        """WKFL-08: A Pipeline step can be a Router (nested composition)."""
        # Step 1: QuantedAgent TextInput -> ProcessedText
        step1 = QuantedAgent(
            "test",
            input_type=TextInput,
            output_type=ProcessedText,
            system_prompt="Process text",
        )

        # Step 2: Router with dispatcher and specialist
        dispatcher = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=RoutingDecision,
            system_prompt="Classify processed text",
        )
        specialist_a = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=FinalOutput,
            system_prompt="Specialist A",
        )
        specialist_b = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=FinalOutput,
            system_prompt="Specialist B",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"specialist_a": specialist_a, "specialist_b": specialist_b},
        )

        pipeline = Pipeline(steps=[step1, router])

        dispatcher_model = FunctionModel(_make_dispatcher_function("specialist_a"))
        with step1.inner.override(model=TestModel()):
            with dispatcher.inner.override(model=dispatcher_model):
                with specialist_a.inner.override(model=TestModel()):
                    with specialist_b.inner.override(model=TestModel()):
                        result = await pipeline.run(TextInput(text="hello"))

        self.assertIsInstance(result.data, FinalOutput)

    async def test_loop_with_pipeline_body(self) -> None:
        """WKFL-08: A Loop body can be a Pipeline (nested composition)."""
        iteration_count = [0]

        # Pipeline body: TextInput -> ProcessedText -> TextInput
        # (Loop requires same input/output type, so we use adapters)
        class IterationState(BaseModel):
            """State model that tracks iteration progress."""

            content: str
            iteration: int = 0

        step1 = QuantedAgent(
            "test",
            input_type=IterationState,
            output_type=ProcessedText,
            system_prompt="Process",
        )

        # Step 2 needs to produce IterationState back, using a FunctionModel
        # to increment iteration count
        step2_agent = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=IterationState,
            system_prompt="Convert back to state",
        )

        pipeline_body = Pipeline(steps=[step1, step2_agent])

        def termination_check(data: BaseModel) -> bool:
            """Terminate after 2 iterations."""
            iteration_count[0] += 1
            return iteration_count[0] >= 2

        loop = Loop(
            body=pipeline_body,
            termination_check=termination_check,
            max_iterations=5,
        )

        with step1.inner.override(model=TestModel()):
            with step2_agent.inner.override(model=TestModel()):
                result = await loop.run(IterationState(content="start", iteration=0))

        self.assertIsInstance(result.data, IterationState)
        self.assertEqual(iteration_count[0], 2)

    async def test_pipeline_with_parallel_step(self) -> None:
        """WKFL-08: A Pipeline step can be a Parallel (nested composition)."""
        # Step 1: TextInput -> ProcessedText
        step1 = QuantedAgent(
            "test",
            input_type=TextInput,
            output_type=ProcessedText,
            system_prompt="Process",
        )

        # Step 2: Parallel with 2 branches (both take ProcessedText)
        class BranchAOutput(BaseModel):
            """Output for branch A."""

            score: float

        class BranchBOutput(BaseModel):
            """Output for branch B."""

            category: str

        branch_a = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=BranchAOutput,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=BranchBOutput,
            system_prompt="Categorize",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        # Step 3: Takes ParallelOutput -> FinalOutput
        step3_agent = QuantedAgent(
            "test",
            input_type=ParallelOutput,
            output_type=FinalOutput,
            system_prompt="Summarize parallel results",
        )

        pipeline = Pipeline(steps=[step1, parallel, step3_agent])

        with step1.inner.override(model=TestModel()):
            with branch_a.inner.override(model=TestModel()):
                with branch_b.inner.override(model=TestModel()):
                    with step3_agent.inner.override(model=TestModel()):
                        result = await pipeline.run(TextInput(text="hello"))

        self.assertIsInstance(result.data, FinalOutput)

    async def test_manual_output_capture(self) -> None:
        """WKFL-07: Developer can capture agent output as a variable for custom routing."""
        agent_a = QuantedAgent(
            "test",
            input_type=TextInput,
            output_type=ProcessedText,
            system_prompt="Process text",
        )
        agent_positive = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=FinalOutput,
            system_prompt="Handle positive",
        )
        agent_negative = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=FinalOutput,
            system_prompt="Handle negative",
        )

        # Run first agent and capture result.data
        with agent_a.inner.override(model=TestModel()):
            result_a = await agent_a.run(TextInput(text="test input"))

        # Use captured data as a variable for custom if/else routing
        captured_data = result_a.data
        self.assertIsInstance(captured_data, ProcessedText)

        # Branch A: condition is True (simulating positive path)
        if isinstance(captured_data, ProcessedText):
            with agent_positive.inner.override(model=TestModel()):
                result_positive = await agent_positive.run(captured_data)
            self.assertIsInstance(result_positive.data, FinalOutput)
        else:
            self.fail("captured_data should be ProcessedText")

        # Branch B: demonstrate the else path also works
        with agent_negative.inner.override(model=TestModel()):
            result_negative = await agent_negative.run(captured_data)
        self.assertIsInstance(result_negative.data, FinalOutput)

    async def test_deeply_nested_pipeline_in_router_in_pipeline(self) -> None:
        """WKFL-08: Three levels of nesting -- Pipeline > Router > Pipeline."""
        class InnerIntermediate(BaseModel):
            """Intermediate model for inner pipeline."""

            value: str

        # We need a 2-step inner pipeline, so add another type
        inner_agent_1 = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=InnerIntermediate,
            system_prompt="Inner pipeline step 1",
        )
        inner_agent_2 = QuantedAgent(
            "test",
            input_type=InnerIntermediate,
            output_type=FinalOutput,
            system_prompt="Inner pipeline step 2",
        )
        inner_pipeline = Pipeline(steps=[inner_agent_1, inner_agent_2])

        # Router (level 2): dispatcher + specialist that is the inner pipeline
        dispatcher = QuantedAgent(
            "test",
            input_type=ProcessedText,
            output_type=RoutingDecision,
            system_prompt="Classify",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"deep_specialist": inner_pipeline},
        )

        # Outer pipeline (level 1): TextInput -> ProcessedText -> Router
        outer_step1 = QuantedAgent(
            "test",
            input_type=TextInput,
            output_type=ProcessedText,
            system_prompt="Outer step 1",
        )
        outer_pipeline = Pipeline(steps=[outer_step1, router])

        dispatcher_model = FunctionModel(_make_dispatcher_function("deep_specialist"))
        with outer_step1.inner.override(model=TestModel()):
            with dispatcher.inner.override(model=dispatcher_model):
                with inner_agent_1.inner.override(model=TestModel()):
                    with inner_agent_2.inner.override(model=TestModel()):
                        result = await outer_pipeline.run(TextInput(text="deep test"))

        self.assertIsInstance(result.data, FinalOutput)

    def test_all_workflows_are_runnable(self) -> None:
        """All workflow types (Pipeline, Router, Loop, Parallel) implement Runnable."""
        # Pipeline
        step1 = QuantedAgent(
            "test", input_type=TextInput, output_type=ProcessedText, system_prompt="S1"
        )
        step2 = QuantedAgent(
            "test", input_type=ProcessedText, output_type=FinalOutput, system_prompt="S2"
        )
        pipeline = Pipeline(steps=[step1, step2])
        self.assertIsInstance(pipeline, Runnable)

        # Router
        dispatcher = QuantedAgent(
            "test", input_type=TextInput, output_type=RoutingDecision, system_prompt="D"
        )
        specialist = QuantedAgent(
            "test", input_type=TextInput, output_type=FinalOutput, system_prompt="Sp"
        )
        router = Router(dispatcher=dispatcher, specialists={"sp": specialist})
        self.assertIsInstance(router, Runnable)

        # Loop
        body = QuantedAgent(
            "test", input_type=TextInput, output_type=TextInput, system_prompt="L"
        )
        loop = Loop(body=body, termination_check=lambda d: True, max_iterations=5)
        self.assertIsInstance(loop, Runnable)

        # Parallel
        branch_a = QuantedAgent(
            "test", input_type=TextInput, output_type=ProcessedText, system_prompt="A"
        )
        branch_b = QuantedAgent(
            "test", input_type=TextInput, output_type=FinalOutput, system_prompt="B"
        )
        parallel = Parallel(branches=[branch_a, branch_b])
        self.assertIsInstance(parallel, Runnable)


if __name__ == "__main__":
    unittest.main()

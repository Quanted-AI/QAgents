"""Tests for workflow-level observability across all four workflow patterns.

Validates that Pipeline, Router, Loop, and Parallel correctly aggregate
token usage, track per-step timing, and collect trace entries across their
sub-runs. Uses pydantic-ai's TestModel and FunctionModel for all tests.
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from quanted_agents import MaxIterationsExceeded, QuantedAgent
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.result import QuantedResult
from quanted_agents.workflows import Loop, Parallel, Pipeline, Router
from quanted_agents.workflows.parallel import ParallelResult
from quanted_agents.workflows.router import RoutingDecision


class StepInput(BaseModel):
    """Input model for pipeline observability tests."""

    text: str


class StepMiddle(BaseModel):
    """Intermediate model between pipeline steps."""

    processed: str


class StepOutput(BaseModel):
    """Output model for pipeline observability tests."""

    result: str


class RouterInput(BaseModel):
    """Input model for router observability tests."""

    query: str


class SpecialistOutput(BaseModel):
    """Output model for router specialist."""

    answer: str


class Draft(BaseModel):
    """Model for loop iteration tests."""

    content: str
    quality_score: float = 0.0


class AnalysisA(BaseModel):
    """Output model for parallel branch A."""

    score: float


class AnalysisB(BaseModel):
    """Output model for parallel branch B."""

    topics: list[str]


class ParInput(BaseModel):
    """Input model for parallel observability tests."""

    text: str


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


class TestPipelineObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for Pipeline workflow observability: usage aggregation, step timing, trace."""

    async def test_pipeline_total_usage_aggregates_all_steps(self) -> None:
        """Pipeline result.total_usage aggregates token usage from all steps."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Step 2",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(StepInput(text="hello"))

        self.assertGreaterEqual(result.total_usage.requests, 2)
        self.assertGreater(result.total_usage.input_tokens, 0)

    async def test_pipeline_step_timings_per_step(self) -> None:
        """Pipeline result.step_timings has one entry per step with correct names."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Step 2",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(StepInput(text="hello"))

        self.assertEqual(len(result.step_timings), 2)
        self.assertEqual(result.step_timings[0].step_name, "Pipeline.step_0")
        self.assertEqual(result.step_timings[1].step_name, "Pipeline.step_1")
        for timing in result.step_timings:
            self.assertGreater(timing.duration_seconds, 0)
            self.assertIsInstance(timing, StepTiming)

    async def test_pipeline_trace_flat_list(self) -> None:
        """Pipeline result.trace is a flat list with entries from all steps."""
        step1 = QuantedAgent(
            "test",
            input_type=StepInput,
            output_type=StepMiddle,
            system_prompt="Step 1",
        )
        step2 = QuantedAgent(
            "test",
            input_type=StepMiddle,
            output_type=StepOutput,
            system_prompt="Step 2",
        )
        pipeline = Pipeline(steps=[step1, step2])

        with step1.inner.override(model=TestModel()):
            with step2.inner.override(model=TestModel()):
                result = await pipeline.run(StepInput(text="hello"))

        self.assertIsInstance(result.trace, list)
        self.assertGreaterEqual(len(result.trace), 2)
        for entry in result.trace:
            self.assertIsInstance(entry, TraceEntry)


class TestRouterObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for Router workflow observability: usage aggregation, step timing, trace."""

    async def test_router_total_usage_aggregates_dispatcher_and_specialist(self) -> None:
        """Router result.total_usage aggregates usage from dispatcher and specialist."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistOutput,
            system_prompt="Answer",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"answer": specialist},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("answer"))
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist.inner.override(model=TestModel()):
                result = await router.run(RouterInput(query="test"))

        self.assertGreaterEqual(result.total_usage.requests, 2)

    async def test_router_step_timings_has_dispatcher_and_specialist(self) -> None:
        """Router result.step_timings has entries for dispatcher and specialist."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistOutput,
            system_prompt="Answer",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"answer": specialist},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("answer"))
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist.inner.override(model=TestModel()):
                result = await router.run(RouterInput(query="test"))

        self.assertEqual(len(result.step_timings), 2)
        self.assertEqual(result.step_timings[0].step_name, "Router.dispatcher")
        self.assertEqual(result.step_timings[1].step_name, "Router.specialist_answer")
        for timing in result.step_timings:
            self.assertGreater(timing.duration_seconds, 0)

    async def test_router_trace_includes_both_agents(self) -> None:
        """Router result.trace has entries from both dispatcher and specialist."""
        dispatcher = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=RoutingDecision,
            system_prompt="Classify input",
        )
        specialist = QuantedAgent(
            "test",
            input_type=RouterInput,
            output_type=SpecialistOutput,
            system_prompt="Answer",
        )
        router = Router(
            dispatcher=dispatcher,
            specialists={"answer": specialist},
        )

        dispatcher_model = FunctionModel(_make_dispatcher_function("answer"))
        with dispatcher.inner.override(model=dispatcher_model):
            with specialist.inner.override(model=TestModel()):
                result = await router.run(RouterInput(query="test"))

        self.assertIsInstance(result.trace, list)
        self.assertGreaterEqual(len(result.trace), 2)
        for entry in result.trace:
            self.assertIsInstance(entry, TraceEntry)


class TestLoopObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for Loop workflow observability: usage aggregation, iteration timing, trace."""

    async def test_loop_total_usage_aggregates_iterations(self) -> None:
        """Loop result.total_usage aggregates usage from all iterations."""
        call_count = [0]

        def _refiner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            score = call_count[0] * 0.4
            draft = Draft(content="refined", quality_score=score)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine the draft",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.9,
            max_iterations=5,
        )

        fm = FunctionModel(_refiner_model)
        with body.inner.override(model=fm):
            result = await loop.run(Draft(content="test", quality_score=0.0))

        self.assertGreaterEqual(result.total_usage.requests, 3)

    async def test_loop_step_timings_per_iteration(self) -> None:
        """Loop with convergence has step_timings with one entry per iteration."""
        call_count = [0]

        def _converging_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            score = call_count[0] * 0.3
            draft = Draft(content="improving", quality_score=score)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.8,
            max_iterations=5,
        )

        fm = FunctionModel(_converging_model)
        with body.inner.override(model=fm):
            result = await loop.run(Draft(content="test", quality_score=0.0))

        self.assertEqual(len(result.step_timings), 3)
        self.assertEqual(result.step_timings[0].step_name, "Loop.iteration_0")
        self.assertEqual(result.step_timings[1].step_name, "Loop.iteration_1")
        self.assertEqual(result.step_timings[2].step_name, "Loop.iteration_2")
        for timing in result.step_timings:
            self.assertGreater(timing.duration_seconds, 0)

    async def test_loop_trace_collects_all_iterations(self) -> None:
        """Loop result.trace has entries from each iteration when converged."""
        call_count = [0]

        def _converging_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_count[0] += 1
            score = call_count[0] * 0.3
            draft = Draft(content="improving", quality_score=score)
            return ModelResponse(parts=[TextPart(content=json.dumps(draft.model_dump()))])

        body = QuantedAgent(
            "test",
            input_type=Draft,
            output_type=Draft,
            system_prompt="Refine",
        )
        loop = Loop(
            body=body,
            termination_check=lambda d: d.quality_score >= 0.8,
            max_iterations=5,
        )

        fm = FunctionModel(_converging_model)
        with body.inner.override(model=fm):
            result = await loop.run(Draft(content="test", quality_score=0.0))

        self.assertIsInstance(result.trace, list)
        self.assertGreaterEqual(len(result.trace), 3)
        for entry in result.trace:
            self.assertIsInstance(entry, TraceEntry)


class TestParallelObservability(unittest.IsolatedAsyncioTestCase):
    """Tests for Parallel workflow observability: usage aggregation, trace, total_usage."""

    def _score_model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """FunctionModel handler returning an AnalysisA JSON response."""
        output = AnalysisA(score=0.85)
        return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])

    def _topics_model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """FunctionModel handler returning an AnalysisB JSON response."""
        output = AnalysisB(topics=["python", "testing"])
        return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])

    async def test_parallel_usage_aggregates_branches(self) -> None:
        """Parallel result.usage aggregates usage from all successful branches."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(self._score_model)):
            with branch_b.inner.override(model=FunctionModel(self._topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        self.assertIsInstance(result, ParallelResult)
        self.assertGreaterEqual(result.usage.requests, 2)
        self.assertGreater(result.usage.input_tokens, 0)

    async def test_parallel_trace_collects_all_branches(self) -> None:
        """Parallel result.trace has entries from all branches."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(self._score_model)):
            with branch_b.inner.override(model=FunctionModel(self._topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        self.assertIsInstance(result.trace, list)
        self.assertGreaterEqual(len(result.trace), 2)
        for entry in result.trace:
            self.assertIsInstance(entry, TraceEntry)

    async def test_parallel_total_usage_matches_usage(self) -> None:
        """Parallel result.total_usage equals result.usage."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(self._score_model)):
            with branch_b.inner.override(model=FunctionModel(self._topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        self.assertEqual(result.total_usage.requests, result.usage.requests)
        self.assertEqual(result.total_usage.input_tokens, result.usage.input_tokens)
        self.assertEqual(result.total_usage.output_tokens, result.usage.output_tokens)

    async def test_parallel_step_timings_per_branch(self) -> None:
        """Parallel result.step_timings has entries for each successful branch."""
        branch_a = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisA,
            system_prompt="Score",
        )
        branch_b = QuantedAgent(
            "test",
            input_type=ParInput,
            output_type=AnalysisB,
            system_prompt="Extract topics",
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(self._score_model)):
            with branch_b.inner.override(model=FunctionModel(self._topics_model)):
                result = await parallel.run(ParInput(text="test input"))

        self.assertEqual(len(result.step_timings), 2)
        step_names = [t.step_name for t in result.step_timings]
        self.assertIn("Parallel.branch_0", step_names)
        self.assertIn("Parallel.branch_1", step_names)
        for timing in result.step_timings:
            self.assertGreater(timing.duration_seconds, 0)


if __name__ == "__main__":
    unittest.main()

"""Router: Dispatcher-based workflow that routes input to a specialist."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from quanted_agents.exceptions import RoutingError
from quanted_agents.observability import StepTiming, TraceEntry
from quanted_agents.result import QuantedResult
from quanted_agents.trace_writer import _resolve_trace_writer
from quanted_agents.types import Runnable


class RoutingDecision(BaseModel):
    """Structured output type for the dispatcher agent in a Router workflow.

    The dispatcher agent returns a RoutingDecision indicating which specialist
    should handle the input. The Router uses the ``target`` field to look up
    the specialist in its registry and delegates execution.

    Attributes:
        target: The name of the specialist to invoke. Must match a key in the
            Router's specialists dictionary.
        reasoning: Optional explanation of why this specialist was chosen.
            Useful for debugging and observability.
    """

    target: str
    reasoning: str = ""


class Router:
    """Dispatcher-based workflow that routes input to a specialist agent.

    Router implements the Runnable protocol, enabling it to be nested inside
    other workflows (e.g., as a step in a Pipeline, or as the body of a Loop).
    It uses a dispatcher Runnable (typically a QuantedAgent with
    RoutingDecision as output_type) to classify input and select which
    specialist Runnable should handle it.

    The routing flow is:
    1. Dispatcher receives the original input and returns a RoutingDecision
    2. Router validates the target exists in the specialists dictionary
    3. Selected specialist receives the original input and returns the result

    Example:
        classifier = QuantedAgent(
            "openai:gpt-4o",
            input_type=Query,
            output_type=RoutingDecision,
            system_prompt="Classify the query as 'math' or 'history'.",
        )
        math_agent = QuantedAgent("openai:gpt-4o", input_type=Query, output_type=Answer, ...)
        history_agent = QuantedAgent("openai:gpt-4o", input_type=Query, output_type=Answer, ...)

        router = Router(
            dispatcher=classifier,
            specialists={"math": math_agent, "history": history_agent},
        )
        result = await router.run(Query(question="What is 2+2?"))
        print(result.data.response)  # Answer from math_agent
    """

    def __init__(self, dispatcher: Runnable, specialists: dict[str, Runnable]) -> None:
        """Create a new Router with a dispatcher and specialist registry.

        Args:
            dispatcher: A Runnable that classifies input and returns a
                RoutingDecision indicating which specialist to invoke.
            specialists: Dictionary mapping specialist names to Runnable
                instances. Must contain at least 1 specialist.

        Raises:
            ValueError: If fewer than 1 specialist is provided.
        """
        if len(specialists) < 1:
            raise ValueError("Router requires at least 1 specialist")
        self._dispatcher: Runnable = dispatcher
        self._specialists: dict[str, Runnable] = specialists

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Run the router by dispatching input to the appropriate specialist.

        Executes the dispatcher to obtain a RoutingDecision, validates the
        target exists in the specialists dictionary, then runs the selected
        specialist with the original input. Returns the specialist's result
        enriched with aggregated usage, timing, and trace data.

        Args:
            input_data: A Pydantic BaseModel instance to classify and route.
            **kwargs: Additional keyword arguments forwarded to both the
                dispatcher's and specialist's ``run()`` methods.

        Returns:
            The QuantedResult from the selected specialist, enriched with
            workflow-level observability data (total_usage, step_timings, trace).

        Raises:
            RoutingError: If the dispatcher selects a target that does not
                exist in the specialists dictionary.
        """
        trace_writer = _resolve_trace_writer(kwargs)

        trace_entries: list[TraceEntry] = []

        start = time.perf_counter()
        decision_result = await self._dispatcher.run(input_data, _trace_writer=trace_writer, **kwargs)
        dispatcher_duration = time.perf_counter() - start

        dispatcher_timing = StepTiming(
            step_name="Router.dispatcher",
            duration_seconds=dispatcher_duration,
            usage=decision_result.total_usage,
        )
        total_usage = decision_result.total_usage
        trace_entries.extend(decision_result.trace)

        decision = decision_result.data
        target = decision.target

        if target not in self._specialists:
            raise RoutingError(
                f"Dispatcher selected '{target}', "
                f"available specialists: {list(self._specialists.keys())}"
            )

        start = time.perf_counter()
        specialist_result = await self._specialists[target].run(input_data, _trace_writer=trace_writer, **kwargs)
        specialist_duration = time.perf_counter() - start

        specialist_timing = StepTiming(
            step_name=f"Router.specialist_{target}",
            duration_seconds=specialist_duration,
            usage=specialist_result.total_usage,
        )
        total_usage = total_usage + specialist_result.total_usage
        trace_entries.extend(specialist_result.trace)

        step_timings = [dispatcher_timing, specialist_timing]
        if any(not t.step_name.startswith("QuantedAgent(") for t in specialist_result._step_timings):
            step_timings.extend(specialist_result._step_timings)

        specialist_result._trace_entries = trace_entries
        specialist_result._step_timings = step_timings
        specialist_result._total_usage = total_usage
        return specialist_result

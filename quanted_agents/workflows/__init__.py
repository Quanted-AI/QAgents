"""Workflow composition primitives for quanted_agents.

Provides reusable workflow patterns that compose QuantedAgents (and other
Runnable implementations) into higher-order execution patterns. Each
workflow implements the Runnable protocol, enabling recursive nesting.
"""

from quanted_agents.workflows.loop import Loop
from quanted_agents.workflows.parallel import Parallel, ParallelOutput, ParallelResult, RetryPolicy
from quanted_agents.workflows.pipeline import Pipeline
from quanted_agents.workflows.router import Router, RoutingDecision

__all__ = ["Loop", "Parallel", "ParallelOutput", "ParallelResult", "Pipeline", "RetryPolicy", "Router", "RoutingDecision"]

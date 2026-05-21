"""QuantedAgents: Type-safe agent SDK wrapping pydantic-ai with Pydantic BaseModel I/O.

Provides a simplified, type-safe interface for creating agents with Pydantic
BaseModel inputs and outputs. Wraps pydantic-ai with construction-time type
validation and automatic input serialization. Includes composable workflow
primitives (Pipeline, Router, Loop, Parallel) that nest recursively for
building complex agentic workflows. Supports MCP (Model Context Protocol)
integration for connecting agents to external tool servers.

Example:
    from pydantic import BaseModel
    from quanted_agents import QuantedAgent, MCPTool

    class Query(BaseModel):
        question: str

    class Answer(BaseModel):
        response: str

    agent = QuantedAgent(
        "openai:gpt-4o",
        input_type=Query,
        output_type=Answer,
        system_prompt="Answer concisely.",
        toolsets=[MCPTool("http://localhost:8001/mcp")],
    )
    result = await agent.run(Query(question="What is Python?"))
    print(result.data.response)
"""

from quanted_agents.agent import QuantedAgent
from quanted_agents.artifact_store import ArtifactStore
from quanted_agents.observability import StepTiming, ToolSpan, TraceEntry
from quanted_agents.trace_writer import TraceSession, TraceWriter
from quanted_agents.exceptions import (
    AgentTimeoutError,
    AssemblyError,
    ConfigurationError,
    ContextOverflowError,
    InvalidInputType,
    InvalidOutputType,
    MaxIterationsExceeded,
    MCPConnectionError,
    PipelineTypeError,
    RecoveryExhaustedError,
    RoutingError,
)
from quanted_agents.mcp import MCPTool
from quanted_agents.result import QuantedResult
from quanted_agents.hierarchical import EscalationPolicy, RunnableTool, WorkflowBudget
from quanted_agents.types import AssemblyFn, ConcurrencyBackend, InputTransformFn, OverflowStrategy, ParallelAssemblyFn, PipelineTransformFn, Runnable, SemaphoreBackend, ValidationResult
from quanted_agents.workflows import Loop, Parallel, ParallelOutput, ParallelResult, Pipeline, RetryPolicy, Router, RoutingDecision

__version__ = "2.0.0"

__all__ = [
    "AgentTimeoutError",
    "ArtifactStore",
    "AssemblyError",
    "AssemblyFn",
    "ConfigurationError",
    "ContextOverflowError",
    "ConcurrencyBackend",
    "EscalationPolicy",
    "InputTransformFn",
    "ParallelAssemblyFn",
    "QuantedAgent",
    "QuantedResult",
    "Runnable",
    "StepTiming",
    "TraceEntry",
    "InvalidInputType",
    "InvalidOutputType",
    "RecoveryExhaustedError",
    "PipelineTypeError",
    "RoutingError",
    "MaxIterationsExceeded",
    "MCPConnectionError",
    "MCPTool",
    "OverflowStrategy",
    "Pipeline",
    "Router",
    "RoutingDecision",
    "Loop",
    "Parallel",
    "ParallelOutput",
    "ParallelResult",
    "PipelineTransformFn",
    "RetryPolicy",
    "RunnableTool",
    "SemaphoreBackend",
    "ToolSpan",
    "TraceSession",
    "TraceWriter",
    "ValidationResult",
    "WorkflowBudget",
]

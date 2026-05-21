"""Hierarchical Agents Example: Parent-Child Agent Delegation.

Demonstrates hierarchical agent patterns using RunnableTool, WorkflowBudget,
EscalationPolicy, and ArtifactStore. A parent agent delegates subtasks to
specialized child agents via LLM tool calling.

Three patterns are shown:
1. Basic dispatch -- parent with two child agents, no store, no budget
2. Store + Budget -- cross-child artifact flow and budget propagation
3. Custom error handling -- permissive EscalationPolicy for graceful degradation

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""

import asyncio
import os
import sys

from pydantic import BaseModel

from quanted_agents import (
    ArtifactStore,
    EscalationPolicy,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)

# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "anthropic:claude-haiku-4-5")


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class TaskInput(BaseModel):
    """Input for the orchestrator agent.

    Attributes:
        task: A natural language description of the task to perform.
    """

    task: str


class TaskOutput(BaseModel):
    """Output from the orchestrator agent.

    Attributes:
        summary: The synthesized result from delegated subtasks.
    """

    summary: str


class SearchResult(BaseModel):
    """Output from the search child agent.

    Attributes:
        answer: The search answer text.
        sources: List of sources for the answer.
    """

    answer: str
    sources: list[str]


class AnalysisResult(BaseModel):
    """Output from the analysis child agent.

    Attributes:
        insight: The main analytical insight.
        confidence: Confidence score between 0.0 and 1.0.
    """

    insight: str
    confidence: float


# ---------------------------------------------------------------------------
# Pattern 1: Basic Hierarchical Dispatch
# ---------------------------------------------------------------------------

async def basic_dispatch() -> None:
    """Demonstrate basic parent-child dispatch with no store or budget.

    The simplest hierarchical pattern: child agents use input_type=str
    so they accept the raw instruction from the parent LLM without
    needing an input_transform.
    """
    print("=" * 60)
    print("Pattern 1: Basic Hierarchical Dispatch")
    print("=" * 60)

    # Child agents with input_type=str (no input_transform needed)
    searcher = QuantedAgent(
        MODEL,
        input_type=str,
        output_type=SearchResult,
        system_prompt="Search for information and return an answer with sources.",
    )

    analyzer = QuantedAgent(
        MODEL,
        input_type=str,
        output_type=AnalysisResult,
        system_prompt="Analyze the given topic. Return an insight with confidence.",
    )

    # Wrap as RunnableTools
    search_tool = RunnableTool(
        runnable=searcher,
        name="search",
        description="Search for factual information. Returns an answer with sources.",
    )

    analysis_tool = RunnableTool(
        runnable=analyzer,
        name="analyze",
        description="Analyze a topic in depth. Returns an insight with confidence.",
    )

    # Parent orchestrator sees children as tools
    orchestrator = QuantedAgent(
        MODEL,
        input_type=TaskInput,
        output_type=TaskOutput,
        system_prompt=(
            "You orchestrate research tasks. Use search for facts, "
            "analyze for insights. Combine the results into a summary."
        ),
        tools=[search_tool.as_tool(), analysis_tool.as_tool()],
    )

    result = await orchestrator.run(TaskInput(task="Research renewable energy trends"))
    print(f"Summary: {result.data.summary}")
    print(f"Total tokens: {result.total_usage.input_tokens + result.total_usage.output_tokens}")
    print()


# ---------------------------------------------------------------------------
# Pattern 2: Store + Budget
# ---------------------------------------------------------------------------

class DetailedSearchInput(BaseModel):
    """Typed input for the detailed search agent.

    Attributes:
        query: The search query string.
        max_results: Maximum number of results to return.
    """

    query: str
    max_results: int = 5


class DetailedSearchOutput(BaseModel):
    """Typed output from the detailed search agent.

    Attributes:
        findings: List of search findings.
    """

    findings: list[str]


class SynthesisInput(BaseModel):
    """Typed input for the synthesis agent.

    Attributes:
        findings: Findings from a prior search to synthesize.
        focus: What aspect to focus the synthesis on.
    """

    findings: list[str]
    focus: str


class SynthesisOutput(BaseModel):
    """Typed output from the synthesis agent.

    Attributes:
        synthesis: The synthesized analysis text.
    """

    synthesis: str


async def store_and_budget() -> None:
    """Demonstrate store-based artifact flow and budget propagation.

    Uses ArtifactStore for cross-child data flow and WorkflowBudget
    for hierarchy-wide budget tracking. Input transforms build typed
    inputs from (store, instruction).
    """
    print("=" * 60)
    print("Pattern 2: Store + Budget")
    print("=" * 60)

    # Child agents with typed inputs (not str) -- requires input_transform
    search_agent = QuantedAgent(
        MODEL,
        input_type=DetailedSearchInput,
        output_type=DetailedSearchOutput,
        system_prompt="Search for detailed information on the given query.",
    )

    synthesis_agent = QuantedAgent(
        MODEL,
        input_type=SynthesisInput,
        output_type=SynthesisOutput,
        system_prompt="Synthesize the provided findings into a coherent analysis.",
    )

    # Input transforms: build typed inputs from (store, instruction)
    def build_search(store: ArtifactStore, instruction: str) -> DetailedSearchInput:
        """Build DetailedSearchInput from parent instruction."""
        return DetailedSearchInput(query=instruction, max_results=5)

    def build_synthesis(store: ArtifactStore, instruction: str) -> SynthesisInput:
        """Read search results from store and build SynthesisInput."""
        search_output = store["detailed_search/result"]
        return SynthesisInput(findings=search_output.findings, focus=instruction)

    # Budget: 15 LLM calls across the entire hierarchy
    budget = WorkflowBudget(llm_call_limit=15, tool_call_limit=8)

    # Shared store for cross-child artifact flow
    store = ArtifactStore()

    # RunnableTools with input transforms
    search_tool = RunnableTool(
        runnable=search_agent,
        name="detailed_search",
        description="Search for detailed findings on a topic.",
        input_transform=build_search,
    )

    synthesis_tool = RunnableTool(
        runnable=synthesis_agent,
        name="synthesize",
        description="Synthesize search findings into a coherent analysis. Run after search.",
        input_transform=build_synthesis,
    )

    # Parent orchestrator with store and budget
    orchestrator = QuantedAgent(
        MODEL,
        input_type=TaskInput,
        output_type=TaskOutput,
        system_prompt=(
            "Orchestrate research. First use detailed_search to find information, "
            "then use synthesize to create an analysis. Always search before synthesizing."
        ),
        tools=[
            search_tool.as_tool(store=store, budget=budget),
            synthesis_tool.as_tool(store=store, budget=budget),
        ],
    )

    result = await orchestrator.run(
        TaskInput(task="Research quantum computing applications"),
        usage_limits=budget.to_usage_limits(),
    )
    print(f"Summary: {result.data.summary}")

    # Check store artifacts written by children
    if "detailed_search/result" in store:
        print(f"Search findings in store: {store['detailed_search/result'].findings[:2]}...")
    if "synthesize/result" in store:
        print(f"Synthesis in store: {store['synthesize/result'].synthesis[:80]}...")

    # Check remaining budget
    print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")
    print(f"Tool calls remaining: {budget.remaining('tool_call_limit')}")
    print()


# ---------------------------------------------------------------------------
# Pattern 3: Custom Error Handling
# ---------------------------------------------------------------------------

async def custom_error_handling() -> None:
    """Demonstrate custom EscalationPolicy for graceful error handling.

    Uses a permissive policy where budget exhaustion returns as text
    instead of crashing the parent, allowing graceful degradation.
    """
    print("=" * 60)
    print("Pattern 3: Custom Error Handling")
    print("=" * 60)

    # Child agent
    worker = QuantedAgent(
        MODEL,
        input_type=str,
        output_type=SearchResult,
        system_prompt="Search for information and return results.",
    )

    # Permissive policy: only system-level exceptions escalate.
    # Budget exhaustion (UsageLimitExceeded) returns as text to parent.
    permissive_policy = EscalationPolicy(
        always_escalate={KeyboardInterrupt, SystemExit}
    )

    worker_tool = RunnableTool(
        runnable=worker,
        name="worker",
        description="Search for information. May fail gracefully on budget exhaustion.",
        escalation_policy=permissive_policy,
    )

    # Tight budget to demonstrate graceful degradation
    budget = WorkflowBudget(llm_call_limit=10, tool_call_limit=5)

    orchestrator = QuantedAgent(
        MODEL,
        input_type=TaskInput,
        output_type=TaskOutput,
        system_prompt=(
            "Use the worker tool to search for information. "
            "If the worker reports an error, summarize what you know so far."
        ),
        tools=[worker_tool.as_tool(budget=budget)],
    )

    result = await orchestrator.run(
        TaskInput(task="Research climate change mitigation strategies"),
        usage_limits=budget.to_usage_limits(),
    )
    print(f"Summary: {result.data.summary}")
    print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")

    # Show the different policy configurations
    print("\nEscalation policy configurations:")
    print(f"  Default: escalates UsageLimitExceeded, KeyboardInterrupt, SystemExit")
    print(f"  Permissive: only escalates KeyboardInterrupt, SystemExit")
    print(f"  Strict: escalates all exceptions (always_escalate={{Exception}})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all three hierarchical agent patterns."""
    await basic_dispatch()
    await store_and_budget()
    await custom_error_handling()

    print("All patterns complete.")


if __name__ == "__main__":
    # Check for API key
    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ]
    has_key = any(os.environ.get(var) for var in api_key_vars)
    if not has_key:
        print(
            "No API key found. Set one of the following environment variables:\n"
            f"  {', '.join(api_key_vars)}\n"
            "Then run: python examples/08_hierarchical_agents.py"
        )
        sys.exit(1)

    asyncio.run(main())

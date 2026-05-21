# Hierarchical Agents Guide

This guide covers building parent-child agent hierarchies using QuantedAgents. A parent agent delegates subtasks to specialized child agents via LLM tool calling. The SDK handles dispatch, budget tracking, error control, and artifact storage automatically.

**When to use hierarchical agents:** Complex tasks that benefit from specialization -- a "project manager" agent that delegates research, writing, and review to focused sub-agents rather than doing everything in one prompt.

For full parameter details, see [API Reference](api-reference.md#runnabletool).

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [RunnableTool](#runnabletool)
3. [Closure-as-Tool Dispatch](#closure-as-tool-dispatch)
4. [Budget Propagation](#budget-propagation)
5. [Error Escalation](#error-escalation)
6. [ArtifactStore Patterns](#artifactstore-patterns)
7. [Complete Example](#complete-example)

---

## Core Concepts

Hierarchical agents use three building blocks:

- **RunnableTool**: Wraps any `Runnable` (QuantedAgent, Pipeline, Parallel, Loop) as a tool that a parent agent can call. The parent LLM decides which child to invoke by calling the tool with a natural language instruction.
- **WorkflowBudget**: Shared pool of counters (LLM calls, tool calls) that the parent and all children draw from. Prevents runaway hierarchies.
- **EscalationPolicy**: Controls which child exceptions propagate to the parent vs. get returned as error text for the parent LLM to handle.

### Import

```python
from quanted_agents import (
    ArtifactStore,
    EscalationPolicy,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)
```

All classes are exported at the top level. No submodule imports needed.

---

## RunnableTool

`RunnableTool` wraps any `Runnable` as a pydantic-ai Tool. The parent agent's LLM sees it as a tool with a single `instruction: str` parameter. When the LLM calls the tool:

1. The instruction is optionally transformed via `input_transform`
2. The child Runnable executes
3. Results are written to a namespaced store (if configured)
4. The child's summary (or string representation of data) is returned to the parent LLM

### Constructor

```python
RunnableTool(
    runnable: Runnable,
    *,
    name: str,
    description: str,
    input_transform: InputTransformFn | None = None,
    escalation_policy: EscalationPolicy | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runnable` | `Runnable` | (required) | The child to wrap (QuantedAgent, Pipeline, etc.). |
| `name` | `str` | (required) | Tool name visible to the parent LLM. Must be unique among siblings. |
| `description` | `str` | (required) | Tool description sent to the parent LLM. Critical for routing quality. |
| `input_transform` | `InputTransformFn \| None` | `None` | Converts `(store, instruction)` into the child's input. Required when child `input_type` is not `str`. |
| `escalation_policy` | `EscalationPolicy \| None` | `None` | Error handling policy. Defaults to `EscalationPolicy.DEFAULT`. |

### Basic Example: Child with `input_type=str`

When a child agent accepts `str` input, no `input_transform` is needed. The parent LLM's instruction string passes directly to the child:

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import QuantedAgent, RunnableTool


class TaskOutput(BaseModel):
    """Output from the orchestrator."""

    summary: str


class SearchResult(BaseModel):
    """Output from the search agent."""

    answer: str
    sources: list[str]


# Child agent with input_type=str -- no transform needed
searcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=SearchResult,
    system_prompt="Search for information and return an answer with sources.",
)

# Wrap as a RunnableTool
search_tool = RunnableTool(
    runnable=searcher,
    name="search",
    description="Search for factual information on any topic.",
)

# Parent sees 'search' as a callable tool
orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=TaskOutput,
    system_prompt="Use the search tool to find information, then summarize.",
    tools=[search_tool.as_tool()],
)


async def main() -> None:
    result = await orchestrator.run("What are the benefits of solar energy?")
    print(f"Summary: {result.data.summary}")
    print(f"Total requests: {result.total_usage.requests}")


asyncio.run(main())
```

### Example with `input_transform`

When a child agent has a typed `input_type` (not `str`), you must provide an `input_transform` that builds the input from the parent's instruction:

```python
from quanted_agents import ArtifactStore, InputTransformFn


class ResearchQuery(BaseModel):
    """Typed input for the research agent."""

    topic: str
    max_results: int = 5


class ResearchFindings(BaseModel):
    """Output from the research agent."""

    findings: list[str]
    key_insight: str


researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchQuery,
    output_type=ResearchFindings,
    system_prompt="Research the given topic thoroughly.",
)


def build_research_input(store: ArtifactStore, instruction: str) -> ResearchQuery:
    """Build typed input from the parent LLM's instruction string."""
    return ResearchQuery(topic=instruction, max_results=5)


research_tool = RunnableTool(
    runnable=researcher,
    name="research",
    description="Research a topic and return structured findings.",
    input_transform=build_research_input,
)
```

The `input_transform` receives the `ArtifactStore` (which may be `None` if no store is configured) and the parent LLM's instruction string. It can be sync or async.

---

## Closure-as-Tool Dispatch

The core hierarchical pattern: a parent agent has multiple children registered as tools. The parent LLM decides which child to call based on the tool descriptions. The SDK handles the dispatch automatically.

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import QuantedAgent, RunnableTool


class ProjectSummary(BaseModel):
    """Final output from the project manager."""

    report: str


class DraftText(BaseModel):
    """Output from the writer agent."""

    text: str
    word_count: int


class ReviewFeedback(BaseModel):
    """Output from the reviewer agent."""

    feedback: str
    approved: bool


# Specialist children
writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=DraftText,
    system_prompt="Write clear, concise content based on the given instructions.",
)

reviewer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=ReviewFeedback,
    system_prompt="Review the given content. Provide feedback and approval status.",
)

# Wrap as tools -- descriptions drive routing quality
write_tool = RunnableTool(
    runnable=writer,
    name="write_content",
    description="Write content such as articles, reports, or documentation.",
)

review_tool = RunnableTool(
    runnable=reviewer,
    name="review_content",
    description="Review written content for quality, accuracy, and completeness.",
)

# Parent orchestrator sees children as tools
manager = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=ProjectSummary,
    system_prompt=(
        "You are a project manager. Use write_content to create drafts "
        "and review_content to check quality. Produce a final report."
    ),
    tools=[write_tool.as_tool(), review_tool.as_tool()],
)


async def main() -> None:
    result = await manager.run("Write and review a brief summary of cloud computing trends.")
    print(f"Report: {result.data.report}")
    print(f"Total LLM requests: {result.total_usage.requests}")


asyncio.run(main())
```

**How it works:** The parent LLM receives the tool descriptions in its system context. When it decides to delegate, it calls the tool with an instruction string. RunnableTool runs the child and returns the child's summary (or `str(result.data)`) back to the parent LLM as the tool result. The parent then uses that information to continue its reasoning.

**Pitfall:** Tool descriptions are critical for routing. Vague descriptions like "do stuff" lead to poor delegation decisions. Be specific about what each child does and when to use it.

---

## Budget Propagation

`WorkflowBudget` tracks workflow-wide consumption counters. The parent and all children draw from the same shared pool, preventing runaway hierarchies where children make unlimited LLM calls.

### Constructor

```python
WorkflowBudget(
    llm_call_limit: int | None = None,
    tool_call_limit: int | None = None,
    total_request_limit: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_call_limit` | `int \| None` | `None` | Maximum LLM calls across the hierarchy. Maps to pydantic-ai's `request_limit`. |
| `tool_call_limit` | `int \| None` | `None` | Maximum tool executions across the hierarchy. Maps to pydantic-ai's `tool_calls_limit`. |
| `total_request_limit` | `int \| None` | `None` | Maximum total requests (LLM + tool). SDK-level tracking only. |

### How Budget Works

1. Create a `WorkflowBudget` with limits
2. Pass it to `RunnableTool.as_tool(budget=budget)` -- children automatically deduct from it
3. Pass `budget.to_usage_limits()` to the parent's `run()` call so the parent also respects the budget
4. After execution, inspect remaining budget via `budget.remaining("llm_call_limit")`

### Complete Example

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import QuantedAgent, RunnableTool, WorkflowBudget


class TaskInput(BaseModel):
    """Input for the orchestrator."""

    task: str


class TaskOutput(BaseModel):
    """Output from the orchestrator."""

    summary: str


class FactResult(BaseModel):
    """Output from the fact-checker."""

    fact: str
    verified: bool


# Child agents
fact_checker = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=FactResult,
    system_prompt="Verify the given claim. Return the fact and whether it's verified.",
)

summarizer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=TaskOutput,
    system_prompt="Summarize the given information concisely.",
)

# Budget: 20 LLM calls shared across parent + all children
budget = WorkflowBudget(llm_call_limit=20, tool_call_limit=10)

# Wrap children as tools, passing the shared budget
check_tool = RunnableTool(
    runnable=fact_checker,
    name="check_fact",
    description="Verify a factual claim. Returns whether the claim is verified.",
)

summarize_tool = RunnableTool(
    runnable=summarizer,
    name="summarize",
    description="Summarize information into a concise output.",
)

orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt=(
        "Check facts using check_fact, then summarize findings using summarize."
    ),
    tools=[
        check_tool.as_tool(budget=budget),
        summarize_tool.as_tool(budget=budget),
    ],
)


async def main() -> None:
    # Parent run also uses the budget via to_usage_limits()
    result = await orchestrator.run(
        TaskInput(task="Verify and summarize: Python was created in 1991"),
        usage_limits=budget.to_usage_limits(),
    )
    print(f"Summary: {result.data.summary}")

    # Inspect remaining budget
    print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")
    print(f"Tool calls remaining: {budget.remaining('tool_call_limit')}")
    print(f"Total requests used: {20 - (budget.remaining('llm_call_limit') or 0)}")


asyncio.run(main())
```

**How deduction works:** When a child completes, `RunnableTool` automatically calls `budget.deduct(result.usage)`, subtracting the child's consumed LLM calls and tool calls from the shared pool. The parent sees the reduced `usage_limits` on its next LLM call via `budget.to_usage_limits()`.

See [API Reference](api-reference.md#workflowbudget) for full method details.

---

## Error Escalation

`EscalationPolicy` controls which child exceptions propagate to the parent vs. get caught and returned as error text. By default, most exceptions are caught -- the parent LLM receives an error message and can decide to retry, skip, or fail gracefully.

### Default Behavior

The default policy (`EscalationPolicy.DEFAULT`) escalates only:

- `UsageLimitExceeded` -- shared budget is exhausted, parent cannot recover
- `KeyboardInterrupt` -- user requested termination
- `SystemExit` -- process termination

All other exceptions are caught and returned as text: `"Error running {tool_name}: {ExceptionType}: {message}"`. The parent LLM then decides what to do.

### Constructor

```python
EscalationPolicy(
    always_escalate: set[type[Exception]] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_escalate` | `set[type[Exception]] \| None` | `None` | Exception types that always propagate. Defaults to `{UsageLimitExceeded, KeyboardInterrupt, SystemExit}`. |

### Permissive Policy (Catch Everything)

For graceful degradation where even budget exhaustion returns as text:

```python
from quanted_agents import EscalationPolicy

# Only system-level exceptions escalate
permissive = EscalationPolicy(
    always_escalate={KeyboardInterrupt, SystemExit}
)
```

### Strict Policy (Escalate Everything)

For fail-fast behavior where any child error crashes the parent:

```python
# All exceptions propagate to the parent
strict = EscalationPolicy(always_escalate={Exception})
```

### Complete Example

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import (
    EscalationPolicy,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)


class TaskInput(BaseModel):
    """Input for the orchestrator."""

    task: str


class TaskOutput(BaseModel):
    """Output from the orchestrator."""

    summary: str


class AnalysisResult(BaseModel):
    """Output from the analysis agent."""

    insight: str


# Child that might fail
analyzer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=AnalysisResult,
    system_prompt="Analyze the given topic deeply.",
)

# Permissive: budget exhaustion returns as text, not exception
permissive_policy = EscalationPolicy(
    always_escalate={KeyboardInterrupt, SystemExit}
)

analysis_tool = RunnableTool(
    runnable=analyzer,
    name="analyze",
    description="Analyze a topic. May fail gracefully on budget exhaustion.",
    escalation_policy=permissive_policy,
)

# Tight budget to demonstrate graceful degradation
budget = WorkflowBudget(llm_call_limit=8, tool_call_limit=4)

orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt=(
        "Use analyze to research the topic. If the tool reports an error, "
        "summarize what you know without it."
    ),
    tools=[analysis_tool.as_tool(budget=budget)],
)


async def main() -> None:
    result = await orchestrator.run(
        TaskInput(task="Analyze trends in renewable energy"),
        usage_limits=budget.to_usage_limits(),
    )
    print(f"Summary: {result.data.summary}")
    print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")


asyncio.run(main())
```

**Pitfall:** The default policy escalates `UsageLimitExceeded`. If you want the parent to handle budget exhaustion gracefully (e.g., summarize partial results), use a permissive policy that removes `UsageLimitExceeded` from the escalation set.

See [API Reference](api-reference.md#escalationpolicy) for the `should_escalate()` method.

---

## ArtifactStore Patterns

When hierarchical agents need to share data, `ArtifactStore` provides a typed key-value store. Each child tool writes its results to a namespaced section of the store, and other children (or the parent) can read from it.

### How Store Integration Works

1. Create an `ArtifactStore`
2. Pass it to `RunnableTool.as_tool(store=store)`
3. When a child completes, RunnableTool writes `result.data` to `"{tool_name}/result"` and `result.summary` to `"{tool_name}/summary"` in the store
4. Other children's `input_transform` functions can read from the store to build their input

### Namespace Isolation

Each child tool gets a namespaced view. A tool named `"research"` writes to `"research/result"` and `"research/summary"`. This prevents key collisions between children.

### Cross-Child Data Flow

The power of the store is enabling data flow between children. Child A's output is stored automatically; Child B's `input_transform` reads it:

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import (
    ArtifactStore,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)


class TaskInput(BaseModel):
    """Input for the orchestrator."""

    task: str


class TaskOutput(BaseModel):
    """Output from the orchestrator."""

    summary: str


class ResearchInput(BaseModel):
    """Typed input for the research agent."""

    query: str
    depth: int = 3


class ResearchOutput(BaseModel):
    """Output from the research agent."""

    findings: list[str]


class SynthesisInput(BaseModel):
    """Typed input for the synthesis agent."""

    findings: list[str]
    focus: str


class SynthesisOutput(BaseModel):
    """Output from the synthesis agent."""

    synthesis: str


# Child agents with typed inputs
research_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchInput,
    output_type=ResearchOutput,
    system_prompt="Research the given query and return findings.",
)

synthesis_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SynthesisInput,
    output_type=SynthesisOutput,
    system_prompt="Synthesize the provided findings into a coherent analysis.",
)

# Shared store for cross-child data flow
store = ArtifactStore()


# Input transforms: build typed inputs from (store, instruction)
def build_research(store: ArtifactStore, instruction: str) -> ResearchInput:
    """Build ResearchInput from the parent LLM's instruction."""
    return ResearchInput(query=instruction, depth=3)


def build_synthesis(store: ArtifactStore, instruction: str) -> SynthesisInput:
    """Read research results from store and build SynthesisInput."""
    research_output = store["research/result"]
    return SynthesisInput(findings=research_output.findings, focus=instruction)


research_tool = RunnableTool(
    runnable=research_agent,
    name="research",
    description="Research a topic. Run this first.",
    input_transform=build_research,
)

synthesis_tool = RunnableTool(
    runnable=synthesis_agent,
    name="synthesize",
    description="Synthesize research findings. Run after research.",
    input_transform=build_synthesis,
)

budget = WorkflowBudget(llm_call_limit=15)

orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt=(
        "First use research to gather information, then use synthesize "
        "to create a coherent analysis. Always research before synthesizing."
    ),
    tools=[
        research_tool.as_tool(store=store, budget=budget),
        synthesis_tool.as_tool(store=store, budget=budget),
    ],
)


async def main() -> None:
    result = await orchestrator.run(
        TaskInput(task="Research quantum computing applications"),
        usage_limits=budget.to_usage_limits(),
    )
    print(f"Summary: {result.data.summary}")

    # Inspect store contents written by children
    if "research/result" in store:
        print(f"Research findings: {store['research/result'].findings[:2]}...")
    if "synthesize/result" in store:
        print(f"Synthesis: {store['synthesize/result'].synthesis[:80]}...")

    # Store keys show namespace isolation
    print(f"Store keys: {list(store.keys())}")


asyncio.run(main())
```

**Key point:** The `input_transform` for `synthesize` reads `store["research/result"]` to access the research agent's output. This works because RunnableTool automatically writes child results to the store under the tool's name namespace.

See [API Reference](api-reference.md#artifactstore) for store methods.

---

## Complete Example

A realistic hierarchical workflow: a project manager agent with three specialist children (researcher, writer, reviewer). Uses `RunnableTool` for dispatch, `WorkflowBudget` for limits, `ArtifactStore` for shared state, and a custom `EscalationPolicy` for graceful error handling.

```python
import asyncio

from pydantic import BaseModel

from quanted_agents import (
    ArtifactStore,
    EscalationPolicy,
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProjectRequest(BaseModel):
    """Input for the project manager."""

    topic: str
    requirements: str


class ProjectDeliverable(BaseModel):
    """Final output from the project manager."""

    title: str
    content: str
    review_status: str


class ResearchBrief(BaseModel):
    """Output from the researcher."""

    key_findings: list[str]
    summary: str


class ArticleDraft(BaseModel):
    """Output from the writer."""

    title: str
    body: str
    word_count: int


class ReviewReport(BaseModel):
    """Output from the reviewer."""

    feedback: list[str]
    approved: bool
    quality_score: float


# Input models for typed children
class ResearchInput(BaseModel):
    """Typed input for the researcher."""

    topic: str


class WriterInput(BaseModel):
    """Typed input for the writer."""

    findings: list[str]
    requirements: str


class ReviewInput(BaseModel):
    """Typed input for the reviewer."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# Child agents
# ---------------------------------------------------------------------------

researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchInput,
    output_type=ResearchBrief,
    system_prompt=(
        "You are a research specialist. Investigate the topic and return "
        "key findings with a brief summary."
    ),
)

writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=WriterInput,
    output_type=ArticleDraft,
    system_prompt=(
        "You are a technical writer. Use the provided findings to write "
        "a well-structured article meeting the given requirements."
    ),
)

reviewer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ReviewInput,
    output_type=ReviewReport,
    system_prompt=(
        "You are an editor. Review the article for quality, accuracy, "
        "and completeness. Provide specific feedback."
    ),
)

# ---------------------------------------------------------------------------
# Store and budget
# ---------------------------------------------------------------------------

store = ArtifactStore()
budget = WorkflowBudget(llm_call_limit=25, tool_call_limit=15)

# Permissive policy: budget exhaustion returns as text
policy = EscalationPolicy(always_escalate={KeyboardInterrupt, SystemExit})

# ---------------------------------------------------------------------------
# Input transforms
# ---------------------------------------------------------------------------


def build_research(store: ArtifactStore, instruction: str) -> ResearchInput:
    """Build research input from instruction."""
    return ResearchInput(topic=instruction)


def build_writer(store: ArtifactStore, instruction: str) -> WriterInput:
    """Build writer input from store (research findings) + instruction."""
    research = store["research/result"]
    return WriterInput(findings=research.key_findings, requirements=instruction)


def build_review(store: ArtifactStore, instruction: str) -> ReviewInput:
    """Build review input from store (writer draft)."""
    draft = store["write/result"]
    return ReviewInput(title=draft.title, body=draft.body)


# ---------------------------------------------------------------------------
# RunnableTools
# ---------------------------------------------------------------------------

research_tool = RunnableTool(
    runnable=researcher,
    name="research",
    description="Research a topic. Returns key findings and a summary. Run first.",
    input_transform=build_research,
    escalation_policy=policy,
)

write_tool = RunnableTool(
    runnable=writer,
    name="write",
    description="Write an article using research findings. Run after research.",
    input_transform=build_writer,
    escalation_policy=policy,
)

review_tool = RunnableTool(
    runnable=reviewer,
    name="review",
    description="Review a written article for quality. Run after writing.",
    input_transform=build_review,
    escalation_policy=policy,
)

# ---------------------------------------------------------------------------
# Parent orchestrator
# ---------------------------------------------------------------------------

project_manager = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ProjectRequest,
    output_type=ProjectDeliverable,
    system_prompt=(
        "You manage content projects. Follow this workflow:\n"
        "1. Use 'research' to investigate the topic\n"
        "2. Use 'write' to create an article (pass requirements as instruction)\n"
        "3. Use 'review' to check quality\n"
        "Produce a final deliverable with the title, content, and review status."
    ),
    tools=[
        research_tool.as_tool(store=store, budget=budget),
        write_tool.as_tool(store=store, budget=budget),
        review_tool.as_tool(store=store, budget=budget),
    ],
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    result = await project_manager.run(
        ProjectRequest(
            topic="Benefits of microservices architecture",
            requirements="Technical audience, 300 words, include tradeoffs",
        ),
        usage_limits=budget.to_usage_limits(),
    )

    # Final output
    print(f"Title: {result.data.title}")
    print(f"Content: {result.data.content[:200]}...")
    print(f"Review Status: {result.data.review_status}")

    # Budget consumption
    print(f"\nLLM calls remaining: {budget.remaining('llm_call_limit')}")
    print(f"Tool calls remaining: {budget.remaining('tool_call_limit')}")

    # Store inspection -- all child artifacts accessible
    print(f"\nStore keys: {list(store.keys())}")
    if "research/result" in store:
        brief = store["research/result"]
        print(f"Research findings: {len(brief.key_findings)} items")
    if "write/result" in store:
        draft = store["write/result"]
        print(f"Article: {draft.title} ({draft.word_count} words)")
    if "review/result" in store:
        review = store["review/result"]
        print(f"Review: approved={review.approved}, score={review.quality_score}")


asyncio.run(main())
```

**What this demonstrates:**

- Three specialist children with typed inputs (not `str`), requiring `input_transform` functions
- `ArtifactStore` enabling cross-child data flow: writer reads researcher's findings, reviewer reads writer's draft
- `WorkflowBudget` shared across the entire hierarchy (parent + 3 children)
- Permissive `EscalationPolicy` for graceful degradation
- Result inspection showing both the final output and store contents

**Pitfall:** The `input_transform` for the writer reads `store["research/result"]`. If the parent LLM calls `write` before `research`, this raises a `KeyError`. Use clear tool descriptions and system prompt instructions to guide the parent's execution order.

---

## Summary

| Concept | Class | Purpose |
|---------|-------|---------|
| Wrap child as tool | `RunnableTool` | Any Runnable becomes a tool the parent LLM can call |
| Shared resource limits | `WorkflowBudget` | Prevents runaway hierarchies with shared counters |
| Error control | `EscalationPolicy` | Choose which child errors crash vs. degrade gracefully |
| Cross-child data | `ArtifactStore` | Namespaced key-value store for sharing artifacts |
| Type bridging | `input_transform` | Converts parent's instruction string to child's typed input |

For complete parameter tables and method signatures, see [API Reference](api-reference.md).

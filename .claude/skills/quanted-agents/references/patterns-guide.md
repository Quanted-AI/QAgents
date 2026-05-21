# QuantedAgents Patterns Guide

Practical guide to composing workflows with the QuantedAgents SDK. Each pattern includes a complete, runnable example.

## Core Concept: Runnables

Everything in QuantedAgents is a **Runnable** -- an object with an async `run()` method:

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]
```

All of these are Runnables:
- `QuantedAgent` -- single LLM agent
- `Pipeline` -- sequential chain
- `Router` -- conditional dispatch
- `Loop` -- iterative refinement
- `Parallel` -- concurrent fan-out/fan-in

Because they share the same interface, **any Runnable can be a step in any workflow**. A Pipeline step can be a Router. A Loop body can be a Pipeline. Nest as deep as you need.

---

## Pattern 1: Single Agent

**When to use:** Standalone task with typed input/output.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent

class ArticleInput(BaseModel):
    text: str
    max_words: int = 100

class AnalysisOutput(BaseModel):
    summary: str
    sentiment: str
    key_topics: list[str]
    word_count: int

agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ArticleInput,
    output_type=AnalysisOutput,
    system_prompt="Analyze articles. Return summary, sentiment, topics, and word count.",
)

async def main() -> None:
    result = await agent.run(ArticleInput(text="Python 3.13 introduces..."))

    # Structured output
    print(result.data.summary)
    print(result.data.sentiment)
    print(result.data.key_topics)

    # Token usage
    print(f"Input tokens: {result.usage.input_tokens}")
    print(f"Output tokens: {result.usage.output_tokens}")

    # Execution trace
    entry = result.trace[0]
    print(f"Step: {entry.step_name}")
    print(f"Duration: {entry.timing.duration_seconds:.2f}s")
    print(f"Model: {entry.model_name}")

asyncio.run(main())
```

**Key points:**
- `input_type` and `output_type` must be Pydantic BaseModel subclasses
- `result.data` is typed to your output model
- `result.trace` has one entry for single-agent runs

---

## Pattern 2: Pipeline (Sequential)

**When to use:** Step-by-step processing where each step transforms the data.

**Type rule:** Output type of step N must match input type of step N+1.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Pipeline

class ResearchQuery(BaseModel):
    topic: str
    depth: str = "standard"

class ResearchNotes(BaseModel):
    topic: str
    findings: list[str]
    sources: list[str]

class FinalReport(BaseModel):
    title: str
    content: str
    sources: list[str]

researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchQuery,
    output_type=ResearchNotes,
    system_prompt="Research the topic and produce structured findings.",
)

writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchNotes,
    output_type=FinalReport,
    system_prompt="Write a polished report from research notes.",
)

pipeline = Pipeline(steps=[researcher, writer])

async def main() -> None:
    result = await pipeline.run(ResearchQuery(topic="quantum computing"))

    # Final output (from last step)
    print(result.data.title)
    print(result.data.content[:200])

    # Aggregated usage across all steps
    print(f"Total tokens: {result.total_usage.input_tokens + result.total_usage.output_tokens}")

    # Per-step timing
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")
        # Pipeline.step_0, Pipeline.step_1, ...

    # Flat trace across all agent invocations
    print(f"Trace entries: {len(result.trace)}")

asyncio.run(main())
```

**Key points:**
- Minimum 2 steps
- `result.data` is the final step's output
- `result.total_usage` aggregates token usage from all steps
- `result.step_timings` has one entry per step (nested workflow steps also include inner breakdown entries)

---

## Pattern 3: Router (Conditional)

**When to use:** Input needs classification followed by specialized handling.

The dispatcher agent must return a `RoutingDecision` with a `target` field matching one of the specialist keys.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Router, RoutingDecision

class SupportTicket(BaseModel):
    subject: str
    body: str
    priority: str = "normal"

class TicketResponse(BaseModel):
    response: str
    category: str
    estimated_resolution: str

dispatcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=RoutingDecision,
    system_prompt=(
        "Classify the support ticket. Available specialists: "
        "'billing', 'technical', 'general'. "
        "Return the specialist name as target with reasoning."
    ),
)

billing_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt="Handle billing support tickets.",
)

technical_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt="Handle technical support tickets.",
)

general_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt="Handle general support tickets.",
)

router = Router(
    dispatcher=dispatcher,
    specialists={
        "billing": billing_agent,
        "technical": technical_agent,
        "general": general_agent,
    },
)

async def main() -> None:
    ticket = SupportTicket(
        subject="Charge on my account",
        body="I see an unexpected charge of $50.",
    )
    result = await router.run(ticket)

    print(result.data.response)
    print(result.data.category)

    # See dispatcher vs specialist timing
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")
        # Router.dispatcher, Router.specialist_billing

    print(f"Total tokens: {result.total_usage.input_tokens}")

asyncio.run(main())
```

**Key points:**
- Dispatcher output type must be `RoutingDecision`
- All specialists receive the original input (not the dispatcher's output)
- `RoutingError` raised if dispatcher returns an invalid target

---

## Pattern 4: Loop (Iterative)

**When to use:** Refinement, convergence, iterative improvement.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Loop, MaxIterationsExceeded

class Essay(BaseModel):
    content: str
    quality_score: float = 0.0
    revision_notes: str = ""

refiner = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Essay,
    output_type=Essay,
    system_prompt=(
        "Improve the essay. Fix grammar, clarity, and structure. "
        "Set quality_score between 0.0 and 1.0 based on quality. "
        "Add revision_notes describing what you changed."
    ),
)

loop = Loop(
    body=refiner,
    termination_check=lambda essay: essay.quality_score >= 0.9,
    max_iterations=5,
)

async def main() -> None:
    initial = Essay(content="Python is good. It does stuff.", quality_score=0.3)
    try:
        result = await loop.run(initial)
    except MaxIterationsExceeded:
        print("Did not converge within max_iterations")
        return

    print(f"Final quality: {result.data.quality_score}")
    print(f"Iterations: {len(result.step_timings)}")
    print(result.data.content[:200])

    # Per-iteration timing
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")
        # Loop.iteration_0, Loop.iteration_1, ...

    # Total token spend across all iterations
    print(f"Total tokens: {result.total_usage.input_tokens + result.total_usage.output_tokens}")

asyncio.run(main())
```

**Key points:**
- `max_iterations` is required (keyword-only, no default)
- `termination_check` receives the body's output `.data` and returns `True` to stop
- If max iterations reached without convergence, raises `MaxIterationsExceeded`
- Body input and output types must match (output feeds back as input)

---

## Pattern 5: Parallel (Concurrent)

**When to use:** Independent analyses on the same input.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Parallel

class TextInput(BaseModel):
    text: str

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float

class TopicsResult(BaseModel):
    topics: list[str]

class SummaryResult(BaseModel):
    summary: str
    word_count: int

sentiment_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=SentimentResult,
    system_prompt="Analyze sentiment. Return sentiment and confidence.",
)

topics_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=TopicsResult,
    system_prompt="Extract key topics from the text.",
)

summary_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TextInput,
    output_type=SummaryResult,
    system_prompt="Summarize the text concisely.",
)

parallel = Parallel(branches=[sentiment_agent, topics_agent, summary_agent])

async def main() -> None:
    result = await parallel.run(TextInput(text="Python 3.13 introduces exciting new features..."))

    # Access individual branch results
    print(f"Successes: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")

    for r in result.results:
        print(f"  {type(r.data).__name__}: {r.data}")

    # Aggregated usage across all branches
    print(f"Total tokens: {result.usage.input_tokens}")

    # Trace entries from all branches
    for entry in result.trace:
        print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")

asyncio.run(main())
```

**Key points:**
- Minimum 2 branches
- All branches receive the same input
- Branches can have different output types
- `result.results` contains successful results; `result.errors` contains exceptions
- `result.usage` aggregates usage from all successful branches
- Branches run concurrently via `asyncio.gather`

---

## Pattern 6: Nesting Workflows

Any Runnable can be a step in any workflow. Here are common nesting patterns.

### Pipeline containing a Router

```python
from quanted_agents import Pipeline, Router, QuantedAgent

# Step 1: Classify and handle
router = Router(dispatcher=classifier, specialists={"a": agent_a, "b": agent_b})

# Step 2: Format the output
formatter = QuantedAgent("openai:gpt-4o-mini", input_type=..., output_type=...)

# Compose: classify -> handle -> format
pipeline = Pipeline(steps=[router, formatter])
result = await pipeline.run(input_data)
```

### Loop containing a Pipeline

```python
from quanted_agents import Loop, Pipeline, QuantedAgent

# Each iteration: research -> write
research_write = Pipeline(steps=[researcher, writer])

# Iterate until quality threshold
loop = Loop(
    body=research_write,
    termination_check=lambda report: report.quality >= 0.9,
    max_iterations=3,
)
result = await loop.run(initial_query)
```

### Three-level nesting: Pipeline > Router > Pipeline

```python
from quanted_agents import Pipeline, Router

# Inner pipeline for complex specialist work
specialist_pipeline = Pipeline(steps=[analyzer, formatter])

# Router with pipeline as a specialist
router = Router(
    dispatcher=classifier,
    specialists={"complex": specialist_pipeline, "simple": simple_agent},
)

# Outer pipeline: preprocess -> route -> postprocess
full_pipeline = Pipeline(steps=[preprocessor, router, postprocessor])
result = await full_pipeline.run(raw_input)
```

---

## Pattern 7: Context Loading (Skills and Feedback)

**When to use:** Your agent needs domain-specific knowledge (coding standards, API conventions, architectural patterns), self-healing through feedback (corrections from past mistakes, quality guidelines), or shared expertise across multiple agents (common skills loaded by different agents on demand).

### How it works

1. **Construction:** Pass `skills_path` and/or `feedback_path` directories to the QuantedAgent constructor
2. **Directory scan:** Each directory is scanned for `.md` files with YAML frontmatter (`name` and `description` fields)
3. **Catalog in system prompt:** Only the names and descriptions are added to the system prompt -- full content is NOT eagerly loaded
4. **On-demand loading:** The LLM sees the catalog and can call the internal `_load_context` tool to load full content for any items it needs
5. **Batch support:** `_load_context` accepts a list of names, so the LLM can load multiple items in a single tool call

This design keeps the system prompt small while giving the LLM access to a large library of skills and feedback.

### Complete example

```python
import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from quanted_agents import QuantedAgent


class CodeRequest(BaseModel):
    task: str
    language: str = "python"


class CodeResponse(BaseModel):
    code: str
    explanation: str


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create skill and feedback directories
        skills_dir = Path(tmpdir) / "skills"
        skills_dir.mkdir()
        feedback_dir = Path(tmpdir) / "feedback"
        feedback_dir.mkdir()

        # Write a skill file (markdown with YAML frontmatter)
        (skills_dir / "error-handling.md").write_text(
            "---\n"
            "name: error-handling\n"
            "description: Best practices for error handling in production code\n"
            "---\n"
            "## Error Handling Guide\n\n"
            "Always use try/except with specific exception types.\n"
            "Log errors with context before re-raising.\n"
            "Use custom exception classes for domain-specific errors.\n"
        )

        # Write a feedback file (identical format, different directory)
        (feedback_dir / "be-concise.md").write_text(
            "---\n"
            "name: be-concise\n"
            "description: Keep code examples short and focused\n"
            "---\n"
            "When writing code examples, avoid boilerplate.\n"
            "Focus on the core pattern being demonstrated.\n"
            "Use comments sparingly -- let the code speak.\n"
        )

        # Create agent with skills and feedback
        agent = QuantedAgent(
            "openai:gpt-4o-mini",
            input_type=CodeRequest,
            output_type=CodeResponse,
            system_prompt="Write clean, production-ready code.",
            skills_path=skills_dir,
            feedback_path=feedback_dir,
        )

        # Run the agent -- it sees the catalog and can load context on demand
        result = await agent.run(
            CodeRequest(task="Write a function to parse CSV files with error handling")
        )

        print(result.data.code)
        print(result.data.explanation)

        # Add feedback programmatically (writes a new .md file to feedback_dir)
        agent.add_feedback(
            name="use-type-hints",
            content="Always include type hints on function signatures and return types.",
            description="Reminder to use type hints consistently",
        )

        # The new feedback is immediately available for loading in subsequent runs
        result2 = await agent.run(
            CodeRequest(task="Write a function to validate email addresses")
        )

        print(result2.data.code)

        # Observability
        entry = result.trace[0]
        print(f"Tool calls: {len(entry.tool_calls)}")
        print(f"Duration: {entry.timing.duration_seconds:.2f}s")


asyncio.run(main())
```

### Feedback File Format

Skills and feedback files use markdown with YAML frontmatter. Both use the same format -- they are stored in separate directories for conceptual clarity.

**Required format:**

```markdown
---
name: my-feedback-name
description: Short description shown in the agent's catalog
---
The full markdown content of the feedback.
This is what gets returned when the LLM loads this item.
```

**Required frontmatter fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `name` | string | The loading key -- the LLM uses this name to request the content |
| `description` | string | Shown in the system prompt catalog so the LLM knows what is available |

**What happens when fields are missing:** Files with a missing `name` or `description` field are silently skipped with a warning log. This is skip-and-warn behavior -- the agent still starts successfully, but the incomplete file is not included in the catalog.

**Directory structure convention:**

```
skills/
  error-handling.md
  code-review.md
  api-design.md
feedback/
  be-concise.md
  use-type-hints.md
  prefer-composition.md
```

### Key points

- `skills_path` and `feedback_path` are optional -- omit both for zero overhead
- Only names and descriptions appear in the system prompt (not full content)
- The LLM decides what to load via the internal `_load_context` tool
- `add_feedback()` requires `feedback_path` to be configured (raises `ValueError` otherwise)
- `_load_context` is a reserved tool name when context paths are set -- do not name your own tools `_load_context`
- Files missing `name` or `description` frontmatter are silently skipped with a warning

---

## Pattern 8: Trace File Logging

**When to use:** Debugging production workflows, building crash-safe audit trails, or analyzing agent behavior post-run.

### How it works

1. **Pass `traces_path`** to any `run()` method (QuantedAgent, Pipeline, Router, Loop, Parallel)
2. **Timestamped JSONL file** is created in that directory (e.g., `trace_20260220T143052_123456.jsonl`)
3. **One JSON line per trace entry** is written in real-time as each agent step completes
4. **Crash-safe:** Each entry is flushed and fsynced immediately -- entries written before a crash are preserved
5. **In-memory traces** (`result.trace`) are always available regardless of `traces_path`

### Complete example

```python
import asyncio
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from quanted_agents import QuantedAgent, Pipeline


class ResearchQuery(BaseModel):
    topic: str


class ResearchNotes(BaseModel):
    findings: list[str]
    sources: list[str]


class FinalReport(BaseModel):
    title: str
    content: str


researcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchQuery,
    output_type=ResearchNotes,
    system_prompt="Research the topic and produce structured findings.",
)

writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchNotes,
    output_type=FinalReport,
    system_prompt="Write a polished report from research notes.",
)

pipeline = Pipeline(steps=[researcher, writer])


async def main() -> None:
    with tempfile.TemporaryDirectory() as traces_dir:
        # Run pipeline with trace file logging
        result = await pipeline.run(
            ResearchQuery(topic="quantum computing"),
            traces_path=traces_dir,
        )

        print(f"Report: {result.data.title}")

        # Find and read the JSONL trace file
        trace_files = list(Path(traces_dir).glob("trace_*.jsonl"))
        print(f"Trace files created: {len(trace_files)}")

        for trace_file in trace_files:
            print(f"\n--- {trace_file.name} ---")
            for line in trace_file.read_text().strip().split("\n"):
                entry = json.loads(line)
                print(f"  Step: {entry['step_name']}")
                print(f"  Model: {entry['model_name']}")
                print(f"  Duration: {entry['timing']['duration_seconds']:.2f}s")

        # In-memory trace is always available too
        print(f"\nIn-memory trace entries: {len(result.trace)}")
        for entry in result.trace:
            print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s")


asyncio.run(main())
```

### Key points

- Works with all `run()` methods: QuantedAgent, Pipeline, Router, Loop, Parallel
- `traces_path` accepts `str` or `Path`
- File is named `trace_YYYYMMDDTHHMMSS_ffffff.jsonl` (microsecond precision to prevent collisions)
- Each line is a complete JSON object (JSONL format) -- one per agent step
- Crash-safe: flush+fsync per entry means no data loss on process crash
- In-memory `result.trace` is always available regardless of `traces_path`
- For advanced usage, `TraceWriter` can be imported directly from `quanted_agents`

---

## Pattern 9: Hierarchical Agents

**When to use:** A parent agent needs to delegate subtasks to specialized child agents. The parent LLM decides which children to invoke via tool calling, and children can share artifacts through a store.

Hierarchical agents use three primitives:
- **RunnableTool** -- wraps any Runnable as a tool the parent LLM can call
- **WorkflowBudget** -- tracks LLM/tool call limits across the entire hierarchy
- **EscalationPolicy** -- controls which child errors propagate vs. return as text

### Basic Hierarchical Dispatch

The simplest path: a parent agent with two child agents as tools. No store, no budget, no input_transform. Children use `input_type=str` so they accept the raw instruction from the parent LLM.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, RunnableTool


class TaskInput(BaseModel):
    task: str


class TaskOutput(BaseModel):
    summary: str


# Child agents with input_type=str -- no input_transform needed
searcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=BaseModel,  # Any output type works
    system_prompt="Search for information and return structured results.",
)

analyzer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=BaseModel,
    system_prompt="Analyze the given topic and provide insights.",
)

# Wrap as RunnableTools
search_tool = RunnableTool(
    runnable=searcher,
    name="search",
    description="Search for factual information. Returns answer with sources.",
)

analysis_tool = RunnableTool(
    runnable=analyzer,
    name="analyze",
    description="Analyze a topic in depth. Returns insight with confidence score.",
)

# Parent orchestrator with children as tools
orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt="Orchestrate research. Use search for facts, analyze for insights.",
    tools=[search_tool.as_tool(), analysis_tool.as_tool()],
)


async def main() -> None:
    result = await orchestrator.run(TaskInput(task="Research AI trends in 2026"))
    print(result.data.summary)


asyncio.run(main())
```

**Key points:**
- Child agents must have `input_type=str` when no `input_transform` is provided
- `as_tool()` returns a pydantic-ai `Tool` -- pass it in the parent's `tools=` list
- The parent LLM sees each child as a tool with a single `instruction: str` parameter
- The parent LLM decides which children to invoke based on their descriptions

### Full-Featured Orchestration

Store for cross-child artifact flow, `input_transform` closures for typed child inputs, custom `EscalationPolicy`, and `WorkflowBudget` for budget propagation. This is the production pattern.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import (
    QuantedAgent,
    RunnableTool,
    WorkflowBudget,
    EscalationPolicy,
    ArtifactStore,
)


class SearchInput(BaseModel):
    query: str
    max_results: int = 10


class SearchOutput(BaseModel):
    datasets: list[dict]


class InsightInput(BaseModel):
    datasets: list[dict]
    analysis_instruction: str


class InsightOutput(BaseModel):
    insights: list[str]
    recommendation: str


class TaskInput(BaseModel):
    task: str


class TaskOutput(BaseModel):
    final_report: str


# Child agents with typed inputs (not str)
searching = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SearchInput,
    output_type=SearchOutput,
    system_prompt="Search financial datasets.",
)

insights = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=InsightInput,
    output_type=InsightOutput,
    system_prompt="Analyze datasets and produce investment insights.",
)


# Input transforms: build typed inputs from (store, instruction)
def build_search_input(store: ArtifactStore, instruction: str) -> SearchInput:
    """Build SearchInput from the parent's instruction."""
    return SearchInput(query=instruction, max_results=10)


def build_insight_input(store: ArtifactStore, instruction: str) -> InsightInput:
    """Read search results from store and build InsightInput."""
    search_data = store["run_searching/result"]
    return InsightInput(
        datasets=search_data.datasets,
        analysis_instruction=instruction,
    )


# Custom escalation: budget exhaustion returns as text instead of crashing
permissive_policy = EscalationPolicy(
    always_escalate={KeyboardInterrupt, SystemExit}
)

# Budget: 20 LLM calls, 10 tool calls across the entire hierarchy
budget = WorkflowBudget(llm_call_limit=20, tool_call_limit=10)

# Shared store for cross-child artifact flow
store = ArtifactStore()

# RunnableTools with full configuration
search_tool = RunnableTool(
    runnable=searching,
    name="run_searching",
    description="Search financial datasets for companies matching the query.",
    input_transform=build_search_input,
    escalation_policy=permissive_policy,
)

insight_tool = RunnableTool(
    runnable=insights,
    name="run_insights",
    description="Analyze datasets and generate investment insights.",
    input_transform=build_insight_input,
    escalation_policy=permissive_policy,
)

# Parent orchestrator with store and budget
orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt=(
        "Orchestrate financial research. First search for relevant datasets, "
        "then analyze them for insights. Always search before analyzing."
    ),
    tools=[
        search_tool.as_tool(store=store, budget=budget),
        insight_tool.as_tool(store=store, budget=budget),
    ],
)


async def main() -> None:
    # Run with budget enforcement
    result = await orchestrator.run(
        TaskInput(task="Find and analyze tech sector companies"),
        usage_limits=budget.to_usage_limits(),
    )
    print(result.data.final_report)

    # Inspect store artifacts written by children
    print(store["run_searching/result"])   # SearchOutput
    print(store["run_insights/result"])    # InsightOutput

    # Check remaining budget
    print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")


asyncio.run(main())
```

**Key points:**
- `input_transform` closures receive `(store, instruction)` and return the child's typed input
- The store enables data flow between siblings (e.g., `run_insights` reads `run_searching/result`)
- `WorkflowBudget.to_usage_limits()` bridges to pydantic-ai's `UsageLimits` for enforcement
- `budget.deduct()` is called automatically -- never call it manually
- Custom `EscalationPolicy` lets budget exhaustion return as text for graceful degradation

### Pipeline + Hierarchical Agents

A Pipeline where one step is an orchestrator with RunnableTools. Shows store flow through both layers -- Pipeline step-level and RunnableTool child-level.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import (
    QuantedAgent,
    Pipeline,
    RunnableTool,
    WorkflowBudget,
    ArtifactStore,
)


class ResearchInput(BaseModel):
    topic: str


class ResearchOutput(BaseModel):
    findings: list[str]
    raw_data: dict


class FinalReport(BaseModel):
    title: str
    body: str
    citations: list[str]


# Child agent for the orchestrator step
data_gatherer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=ResearchOutput,
    system_prompt="Gather research data on the given topic.",
)

gather_tool = RunnableTool(
    runnable=data_gatherer,
    name="gather_data",
    description="Gather research data and findings on a topic.",
)

# Step 1: Orchestrator that uses child tools
store = ArtifactStore()
orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchInput,
    output_type=ResearchOutput,
    system_prompt="Orchestrate research by gathering data, then synthesize findings.",
    tools=[gather_tool.as_tool(store=store)],
)

# Step 2: Report writer (regular agent, no children)
report_writer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=ResearchOutput,
    output_type=FinalReport,
    system_prompt="Write a polished research report from the provided findings.",
)

# Pipeline: orchestrator (with children) -> report writer
pipeline = Pipeline(steps=[orchestrator, report_writer])

budget = WorkflowBudget(llm_call_limit=30, tool_call_limit=15)


async def main() -> None:
    result = await pipeline.run(
        ResearchInput(topic="Quantum computing market 2026"),
        usage_limits=budget.to_usage_limits(),
    )
    print(result.data.title)
    print(result.data.body)

    # Store contains child-level artifacts
    print(store["gather_data/result"])  # ResearchOutput from child


asyncio.run(main())
```

**Key points:**
- Pipeline steps can be orchestrator agents with their own child tools
- The store captures child-level artifacts alongside pipeline-level data flow
- Budget enforcement works across both Pipeline and hierarchical layers
- Step type matching still applies: orchestrator output type must match report writer input type

---

## Error Recovery

QuantedAgents automatically handles malformed LLM output through a cascading recovery pipeline.

### How it works

1. **json-repair** -- fixes common JSON syntax errors (missing quotes, trailing commas, etc.)
2. **Restructurer** -- if configured and repair wasn't enough, a cheap model restructures the raw output

Note: pydantic-ai's built-in `retries` parameter handles re-prompting the LLM with validation errors independently of this recovery pipeline. The QuantedAgents recovery pipeline activates *after* pydantic-ai's retries are exhausted.

### Two-model pattern

Use an expensive model for reasoning and a cheap model for output structuring:

```python
agent = QuantedAgent(
    "openai:gpt-4o",              # Heavy model for reasoning
    input_type=Query,
    output_type=DetailedAnalysis,
    system_prompt="Analyze thoroughly.",
    restructurer_model="openai:gpt-4o-mini",  # Cheap model for output formatting
    max_recovery_attempts=3,                   # Global retry budget
)
```

### When recovery activates

- LLM returns syntactically invalid JSON -> json-repair tries to fix it
- Repaired JSON fails Pydantic validation and `restructurer_model` is set -> cheap model restructures
- Budget exceeded -> raises `RecoveryExhaustedError`

### Checking if recovery was used

```python
result = await agent.run(input_data)
entry = result.trace[0]
if entry.recovery_info:
    print(f"Recovery activated: {entry.recovery_info}")
    # {"json_repair_attempted": True, "restructurer_used": False, "attempts_used": 1}
```

---

## MCP Integration

Connect agents to external tool servers using the Model Context Protocol.

### Basic usage

```python
from quanted_agents import QuantedAgent, MCPTool

agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Use available tools to answer questions.",
    toolsets=[MCPTool("http://localhost:8001/mcp")],
)

result = await agent.run(Query(question="What's the weather?"))
```

### Multiple MCP servers

```python
agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Use weather and search tools.",
    toolsets=[
        MCPTool("http://localhost:8001/mcp", tool_prefix="weather"),
        MCPTool("http://localhost:8002/mcp", tool_prefix="search"),
    ],
)
```

### Multi-run with pre-initialized connections

```python
async with agent:
    r1 = await agent.run(Query(question="First question"))
    r2 = await agent.run(Query(question="Second question"))
```

### Transport options

- `"http"` (default) -- Streamable HTTP, recommended by MCP spec
- `"sse"` -- Legacy Server-Sent Events for older MCP servers

```python
legacy_tool = MCPTool("http://localhost:8002/sse", transport="sse")
```

---

## Observability

Every agent run and workflow execution captures observability data automatically.

### What's available

| Property | Available on | Description |
|----------|-------------|-------------|
| `result.trace` | All results | List of `TraceEntry` per agent invocation |
| `result.total_usage` | All results | Aggregated `RunUsage` across all steps |
| `result.step_timings` | All results | List of `StepTiming` per workflow step |
| `result.usage` | All results | Token usage (same as `total_usage` for single agents) |

### Inspecting a trace

```python
import json

result = await pipeline.run(input_data)

# Per-step timing
for timing in result.step_timings:
    print(f"{timing.step_name}: {timing.duration_seconds:.2f}s, "
          f"{timing.usage.input_tokens} in / {timing.usage.output_tokens} out")

# Full trace
for entry in result.trace:
    print(f"\n--- {entry.step_name} ---")
    print(f"  Model: {entry.model_name}")
    print(f"  Duration: {entry.timing.duration_seconds:.2f}s")
    print(f"  Tool calls: {len(entry.tool_calls)}")
    if entry.recovery_info:
        print(f"  Recovery: {entry.recovery_info}")

# Export to JSON
trace_data = [entry.to_dict() for entry in result.trace]
print(json.dumps(trace_data, indent=2))
```

### Step timing names by workflow type

| Workflow | Naming pattern |
|----------|---------------|
| Single agent | `QuantedAgent({OutputType})` |
| Pipeline | `Pipeline.step_0`, `Pipeline.step_1`, ... |
| Router | `Router.dispatcher`, `Router.specialist_{target}` |
| Loop | `Loop.iteration_0`, `Loop.iteration_1`, ... |
| Parallel | `Parallel.branch_0`, `Parallel.branch_1`, ... |

### Nested workflow observability

When workflows are nested (e.g., a Pipeline step is a Router), observability data propagates through all nesting levels:

- **`total_usage`** aggregates token usage from all agent invocations at all nesting levels.
- **`step_timings`** includes the outer workflow's summary entries followed by the inner workflow's detailed breakdown entries.
- **`trace`** is a flat list of `TraceEntry` objects from every agent invocation across all nesting levels.

Example `step_timings` for a Pipeline containing a Router (3 agents total):

| step_timings entry | Description |
|---|---|
| `Pipeline.step_0` | First Pipeline step (single agent) |
| `Pipeline.step_1` | Second Pipeline step (Router -- summary with aggregated usage) |
| `Router.dispatcher` | Inner Router's dispatcher agent |
| `Router.specialist_billing` | Inner Router's selected specialist |

Use `result.total_usage` for the authoritative token total. Do not sum individual `StepTiming.usage` values -- summary entries (like `Pipeline.step_1` when it wraps a Router) have `usage` equal to the inner workflow's `total_usage`, which overlaps with the inner breakdown entries.

---

## Pattern 10: Soft Limits

**When to use:** Agent nears usage limits and you want a graceful wrap-up instead of a crash.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent


class ResearchQuery(BaseModel):
    topic: str


class ResearchReport(BaseModel):
    findings: list[str]
    conclusion: str


agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=ResearchQuery,
    output_type=ResearchReport,
    system_prompt="Research the topic thoroughly using available tools.",
    llm_call_limit=10,
    soft_limit=True,
)


async def main() -> None:
    result = await agent.run(ResearchQuery(topic="quantum computing advances"))

    print(f"Findings: {result.data.findings}")
    print(f"Conclusion: {result.data.conclusion}")
    print(f"Termination reason: {result.termination_reason}")
    # None = normal completion, "soft_limit" = wrapped up due to limit

    if result.termination_reason == "soft_limit":
        print("Agent hit the limit and produced a wrap-up response")


asyncio.run(main())
```

> **Pitfall:** Setting `soft_limit=True` without any limit (`llm_call_limit`, `tool_call_limit`, or `total_request_limit`) does nothing -- there is no limit to trigger the soft wrap-up.

**Key points:**
- With `soft_limit=True`, the agent gets up to 2 additional LLM calls (tools blocked) to produce final output when a limit is hit
- Without `soft_limit`, hitting a limit raises `UsageLimitExceeded` immediately
- Check `result.termination_reason` to distinguish normal completion from soft-limit wrap-up

---

## Pattern 11: Timeouts

**When to use:** Enforce time bounds on agent execution to prevent runaway runs.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, AgentTimeoutError


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    response: str


agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer the question thoroughly.",
    soft_timeout=30.0,
    hard_timeout=60.0,
)


async def main() -> None:
    try:
        result = await agent.run(Query(question="Explain general relativity"))
    except AgentTimeoutError as e:
        print(f"Hard timeout fired: {e}")
        print(f"Usage before timeout: {e.usage.input_tokens} input tokens")
        if e.store:
            print(f"Store keys saved: {list(e.store.keys())}")
        return

    print(f"Response: {result.data.response}")
    print(f"Termination reason: {result.termination_reason}")
    # None = normal, "soft_timeout" = wrapped up, "hard_timeout" = should not reach here

    if result.termination_reason == "soft_timeout":
        print("Agent wrapped up due to soft timeout")


asyncio.run(main())
```

> **Pitfall:** Setting `soft_timeout` without `hard_timeout` adds an implicit hard backstop at `soft_timeout + 30s`. If the wrap-up hangs, the implicit backstop fires. Set an explicit `hard_timeout` for precise control.

**Key points:**
- `soft_timeout` triggers a graceful wrap-up sequence (same as soft limits -- tools blocked, up to 2 LLM calls)
- `hard_timeout` raises `AgentTimeoutError` immediately, carrying `store` and `usage` for inspection
- Both can be overridden per-run via `kwargs`
- `hard_timeout` must be greater than `soft_timeout` (raises `ValueError` otherwise)

---

## Pattern 12: Tool Interceptor

**When to use:** Modify, validate, or inject arguments into MCP tool calls before execution.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, MCPTool


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    response: str


TENANT_ID = "tenant_abc123"


def inject_tenant(tool_name: str, args: dict) -> dict | None:
    """Inject tenant_id into every tool call for multi-tenant isolation."""
    args["tenant_id"] = TENANT_ID
    return args


tools = MCPTool(
    "http://localhost:8001/mcp",
    argument_interceptor=inject_tenant,
)

agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Use tools to answer questions.",
    toolsets=[tools],
)


async def main() -> None:
    result = await agent.run(Query(question="Look up order #12345"))
    print(f"Response: {result.data.response}")


asyncio.run(main())
```

> **Pitfall:** The interceptor runs synchronously in the middleware pipeline. Do not perform I/O (network calls, file reads) inside it -- use an async interceptor if I/O is needed. Returning `None` from the interceptor aborts the tool call with a `ModelRetry`.

**Key points:**
- Interceptor receives `(tool_name, args_dict)` and returns modified args or `None` to abort
- Supports both sync and async callables
- Runs once before the retry loop (intercepted args are used for all retry attempts)
- Aborting (returning `None`) surfaces as a `ModelRetry` to the LLM

---

## Pattern 13: MCP Concurrency

**When to use:** Limit concurrent MCP tool calls to avoid overwhelming an external API.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, MCPTool


class AnalysisRequest(BaseModel):
    urls: list[str]


class AnalysisResult(BaseModel):
    results: list[dict]


tools = MCPTool(
    "http://localhost:8001/mcp",
    max_concurrent_calls=5,
)

agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=AnalysisRequest,
    output_type=AnalysisResult,
    system_prompt="Analyze each URL using available tools.",
    toolsets=[tools],
)


async def main() -> None:
    result = await agent.run(
        AnalysisRequest(urls=["https://example.com/a", "https://example.com/b"])
    )
    print(f"Results: {result.data.results}")


asyncio.run(main())
```

> **Pitfall:** Setting `max_concurrent_calls` too low (e.g., 1) can cause the agent to wait excessively between tool calls, dramatically increasing total run time. Start with 5-10 and tune based on the external API's rate limits.

**Key points:**
- Uses `SemaphoreBackend` by default (in-process `asyncio.BoundedSemaphore`)
- Provide a custom `ConcurrencyBackend` for distributed throttling (e.g., Redis-backed)
- `concurrency_timeout` enables fail-open: if the semaphore cannot be acquired in time, the call proceeds anyway

---

## Pattern 14: Trace Sessions

**When to use:** Consolidate multiple agent runs into a single trace file for correlated analysis.

```python
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel
from quanted_agents import QuantedAgent, TraceSession


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    response: str


agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer concisely.",
)


async def main() -> None:
    async with TraceSession("traces/multi_run.jsonl") as session:
        print(f"Session ID: {session.session_id}")

        r1 = await agent.run(
            Query(question="What is Python?"),
            trace_session=session,
        )
        print(f"Answer 1: {r1.data.response}")

        r2 = await agent.run(
            Query(question="What is Rust?"),
            trace_session=session,
        )
        print(f"Answer 2: {r2.data.response}")

    # Read consolidated trace
    trace_file = Path("traces/multi_run.jsonl")
    for line in trace_file.read_text().strip().split("\n"):
        entry = json.loads(line)
        print(f"  Step: {entry['step_name']}, Session: {entry.get('session_id', 'N/A')}")


asyncio.run(main())
```

> **Pitfall:** Forgetting `async with` means the session context manager is never entered. While writes are flushed per-entry (so data is not lost), the structured cleanup pattern is not followed. Always use `async with TraceSession(...)`.

**Key points:**
- All runs within the session write to the same JSONL file
- `session.session_id` (UUID) is attached to every trace entry for correlation
- Partial traces are preserved on exception -- `TraceWriter` flushes per-entry
- Works with any `run()` call: QuantedAgent, Pipeline, Router, Loop, Parallel

---

## Pattern 15: Parallel Retry

**When to use:** Automatically retry failed Parallel branches for transient errors.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Parallel, RetryPolicy


class DataRequest(BaseModel):
    source: str


class DataResult(BaseModel):
    records: list[dict]
    source: str


api_agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=DataRequest,
    output_type=DataResult,
    system_prompt="Fetch data from the specified source.",
)

db_agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=DataRequest,
    output_type=DataResult,
    system_prompt="Query the database for the specified source.",
)

parallel = Parallel(
    branches=[api_agent, db_agent],
    retry_policy=RetryPolicy(
        max_retries=2,
        retry_on=[ConnectionError, TimeoutError],
        delay_seconds=1.0,
    ),
)


async def main() -> None:
    result = await parallel.run(DataRequest(source="users"))

    print(f"Successes: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")

    for r in result.results:
        print(f"  Source: {r.data.source}, Records: {len(r.data.records)}")

    for err in result.errors:
        print(f"  Permanently failed: {err}")


asyncio.run(main())
```

> **Pitfall:** Using broad exception types like `retry_on=[Exception]` masks bugs. Be specific about which transient errors deserve retries -- `ConnectionError`, `TimeoutError`, etc.

**Key points:**
- Only branches that fail with exception types in `retry_on` are retried
- Non-retryable failures are preserved unchanged
- First retry attempt does not sleep; `delay_seconds` applies from attempt 1 onward
- Error context is injected into retried runs via `message_history` for informed retries

---

## Pattern 16: Dual-Stream Architecture

**When to use:** Get both structured data and a natural language summary from a single agent run.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent


class FinancialQuery(BaseModel):
    company: str
    quarter: str


class FinancialAnalysis(BaseModel):
    revenue: float
    profit_margin: float
    risk_factors: list[str]
    recommendation: str


agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=FinancialQuery,
    output_type=FinancialAnalysis,
    system_prompt=(
        "Analyze the company's financial performance. "
        "Provide both structured data AND a natural language summary "
        "explaining your analysis."
    ),
)


async def main() -> None:
    result = await agent.run(
        FinancialQuery(company="Acme Corp", quarter="Q4 2025")
    )

    # Structured output (always available)
    print(f"Revenue: ${result.data.revenue:,.2f}")
    print(f"Profit margin: {result.data.profit_margin:.1%}")
    print(f"Risk factors: {result.data.risk_factors}")
    print(f"Recommendation: {result.data.recommendation}")

    # Natural language summary (provider-dependent)
    if result.summary:
        print(f"\nSummary: {result.summary}")
    else:
        print("\nNo summary available (provider did not produce text alongside structured output)")

    # Access artifacts from workflow runs
    print(f"Artifacts: {list(result.artifacts.keys())}")


asyncio.run(main())
```

> **Pitfall:** The `summary` property is provider-dependent. Not all LLM providers produce text alongside structured output (tool calls). It may return `None` even when you request a summary in the system prompt. Always check `if result.summary` before using it.

**Key points:**
- `result.data` contains the structured `BaseModel` output (always available)
- `result.summary` extracts text from the last `ModelResponse` that also contains a `ToolCallPart`
- Summary is lazy: computed on first access, zero overhead if never accessed
- `result.artifacts` provides access to the `ArtifactStore` used during orchestration

---

## Pattern 17: Pipeline Transforms

**When to use:** Bridge type gaps between Pipeline stages when adjacent steps have incompatible input/output types.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Pipeline, ArtifactStore
from quanted_agents.result import QuantedResult


class RawText(BaseModel):
    content: str


class ExtractedEntities(BaseModel):
    entities: list[str]
    raw_text: str


class ClassificationInput(BaseModel):
    items: list[str]


class ClassificationResult(BaseModel):
    categories: dict[str, str]


extractor = QuantedAgent(
    "openai:gpt-4o",
    input_type=RawText,
    output_type=ExtractedEntities,
    system_prompt="Extract named entities from the text.",
)

classifier = QuantedAgent(
    "openai:gpt-4o",
    input_type=ClassificationInput,
    output_type=ClassificationResult,
    system_prompt="Classify each entity into categories.",
)


def transform_for_classifier(
    result: QuantedResult, store: ArtifactStore, stage: int
) -> ClassificationInput:
    """Convert ExtractedEntities output to ClassificationInput."""
    return ClassificationInput(items=result.data.entities)


pipeline = Pipeline(
    steps=[extractor, classifier],
    input_transforms={1: transform_for_classifier},
)


async def main() -> None:
    result = await pipeline.run(
        RawText(content="Apple announced a partnership with Google in San Francisco.")
    )

    print(f"Categories: {result.data.categories}")
    for entity, category in result.data.categories.items():
        print(f"  {entity}: {category}")


asyncio.run(main())
```

> **Pitfall:** Stage 0 cannot have an `input_transform` -- it receives the pipeline input directly. Only stages 1+ can have transforms. When a transform is provided for a stage boundary, the type mismatch check is skipped for that boundary.

**Key points:**
- `input_transforms` maps stage index to a transform function
- Transform receives `(QuantedResult, ArtifactStore, stage_index)` and returns a `BaseModel`
- Supports both sync and async transforms
- Without a transform, Pipeline enforces output_type/input_type matching at construction time

---

## Pattern 18: Assembly Functions

**When to use:** Custom output assembly for workflows that need to combine intermediate results from multiple stages.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Pipeline, ArtifactStore
from quanted_agents.result import QuantedResult


class ResearchQuery(BaseModel):
    topic: str


class ResearchFindings(BaseModel):
    findings: list[str]
    sources: list[str]


class FinalReport(BaseModel):
    title: str
    executive_summary: str
    detailed_findings: list[str]
    all_sources: list[str]


researcher = QuantedAgent(
    "openai:gpt-4o",
    input_type=ResearchQuery,
    output_type=ResearchFindings,
    system_prompt="Research the topic and return findings with sources.",
)

deep_researcher = QuantedAgent(
    "openai:gpt-4o",
    input_type=ResearchFindings,
    output_type=ResearchFindings,
    system_prompt="Expand on these findings with deeper research.",
)

store = ArtifactStore()


def assemble_report(store: ArtifactStore, last_result: QuantedResult) -> FinalReport:
    """Combine both research stages into a final report."""
    initial = store["Pipeline.step_0/result"]
    expanded = last_result.data

    all_sources = list(set(initial.sources + expanded.sources))

    return FinalReport(
        title=f"Research Report: {initial.findings[0][:50]}",
        executive_summary=initial.findings[0] if initial.findings else "No summary",
        detailed_findings=expanded.findings,
        all_sources=all_sources,
    )


pipeline = Pipeline(
    steps=[researcher, deep_researcher],
    assembly=assemble_report,
    store=store,
)


async def main() -> None:
    result = await pipeline.run(ResearchQuery(topic="quantum computing"))

    print(f"Title: {result.data.title}")
    print(f"Summary: {result.data.executive_summary}")
    print(f"Findings: {len(result.data.detailed_findings)}")
    print(f"Sources: {result.data.all_sources}")

    # Store still accessible after assembly
    print(f"Store keys: {list(store.keys())}")
    print(f"Step 0 result: {store['Pipeline.step_0/result']}")


asyncio.run(main())
```

> **Pitfall:** Assembly replaces the default `result.data`. If you forget to pass `store=` to the Pipeline, the assembly function cannot access intermediate results via store keys. Always pass both `assembly` and `store` together.

**Key points:**
- Assembly function receives `(ArtifactStore, QuantedResult)` and returns the final output
- The return value becomes `result.data` (replaces the last step's output)
- Store keys follow `{WorkflowType}.step_{N}/result` convention
- Assembly supports both sync and async functions
- If assembly raises, `AssemblyError` is thrown with `store`, `last_result`, and `original_error` attached

---

## Pattern 19: ArtifactStore in Workflows

**When to use:** Share state across workflow stages and inspect intermediate results after a run.

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Pipeline, ArtifactStore


class CustomerQuery(BaseModel):
    customer_id: str
    question: str


class CustomerProfile(BaseModel):
    name: str
    tier: str
    history: list[str]


class PersonalizedResponse(BaseModel):
    greeting: str
    answer: str
    upsell_suggestion: str


profiler = QuantedAgent(
    "openai:gpt-4o",
    input_type=CustomerQuery,
    output_type=CustomerProfile,
    system_prompt="Look up customer profile based on their ID.",
)

responder = QuantedAgent(
    "openai:gpt-4o",
    input_type=CustomerProfile,
    output_type=PersonalizedResponse,
    system_prompt="Write a personalized response based on the customer profile.",
)

store = ArtifactStore()
pipeline = Pipeline(steps=[profiler, responder], store=store)


async def main() -> None:
    result = await pipeline.run(
        CustomerQuery(customer_id="C-1234", question="What is my account balance?")
    )

    # Final output
    print(f"Greeting: {result.data.greeting}")
    print(f"Answer: {result.data.answer}")
    print(f"Upsell: {result.data.upsell_suggestion}")

    # Inspect intermediate results via store
    print(f"\nStore keys: {list(store.keys())}")

    profile = store["Pipeline.step_0/result"]
    print(f"Customer name: {profile.name}")
    print(f"Customer tier: {profile.tier}")

    response = store["Pipeline.step_1/result"]
    print(f"Final response type: {type(response).__name__}")

    # Check if a key exists before accessing
    if "Pipeline.step_0/result" in store:
        print("Profile was stored successfully")

    # Version history (if a key was written multiple times)
    history = store.history("Pipeline.step_0/result")
    print(f"Step 0 versions: {len(history)}")


asyncio.run(main())
```

> **Pitfall:** Store keys follow the `{WorkflowType}.{step_name}/result` convention. If you are unsure of the exact key names, call `store.keys()` after a run to discover them. Parallel uses `Parallel.branch_{N}/result`, Loop uses `Loop.iteration_{N}/result`.

**Key points:**
- Pass `store=ArtifactStore()` to any workflow to enable artifact recording
- Each step's output is automatically written to the store under a conventional key
- Store is accessible both during assembly functions and after the run completes
- `store.history(key)` returns all values ever written to a key (useful for Loop iterations)
- Parallel uses `_NamespacedStore` internally -- branch keys are prefixed (e.g., `Parallel.branch_0/result`)

---

## Tips

- **Use `openai:gpt-4o-mini` for development** -- cheap, fast, widely available. Switch to larger models for production.
- **Use the two-model pattern for complex structured output** -- `restructurer_model="openai:gpt-4o-mini"` catches formatting errors from the heavy model.
- **Always set `max_iterations` on Loop** -- it's required and has no default. This prevents runaway execution.
- **Use `result.trace` for debugging** -- inspect what the LLM said, what tools were called, and whether recovery activated.
- **Nest workflows freely** -- Pipeline, Router, Loop, and Parallel all implement Runnable. Compose as needed.
- **Check `result.errors` on Parallel** -- branches can fail independently. Always check for errors alongside results.
- **Use `traces_path` for production debugging** -- pass it to any `run()` call to get a crash-safe JSONL trace file. Read it later with `json.loads()` per line.

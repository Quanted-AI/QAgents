# QuantedAgents API Reference

Complete API reference for the QuantedAgents SDK -- a type-safe agent framework wrapping pydantic-ai with Pydantic BaseModel I/O, composable workflow primitives, and built-in observability.

## Installation

```bash
pip install quanted-agents
```

## Environment Variables

QuantedAgents delegates all LLM communication to [pydantic-ai](https://ai.pydantic.dev/). No SDK-specific environment variables are required -- you only need to set the standard API key environment variable for whichever model provider you select.

| Provider Prefix | Environment Variable | Example Model String | Where to Get Key |
|-----------------|---------------------|----------------------|------------------|
| `openai:` | `OPENAI_API_KEY` | `"openai:gpt-4o"` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `anthropic:` | `ANTHROPIC_API_KEY` | `"anthropic:claude-sonnet-4-20250514"` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `google-gla:` | `GEMINI_API_KEY` | `"google-gla:gemini-2.0-flash"` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `groq:` | `GROQ_API_KEY` | `"groq:llama-3.3-70b-versatile"` | [console.groq.com](https://console.groq.com/keys) |
| `mistral:` | `MISTRAL_API_KEY` | `"mistral:mistral-large-latest"` | [console.mistral.ai](https://console.mistral.ai/api-keys) |
| `cohere:` | `CO_API_KEY` | `"cohere:command-r-plus"` | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| `deepseek:` | `DEEPSEEK_API_KEY` | `"deepseek:deepseek-chat"` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |

```bash
export OPENAI_API_KEY="sk-..."
```

pydantic-ai supports additional providers beyond those listed above (e.g., `bedrock:`, `google-vertex:`, `xai:`, `grok:`). See the [pydantic-ai model documentation](https://ai.pydantic.dev/models/) for the complete list of supported providers and their authentication requirements.

> **Note:** If no API key is set for the chosen provider, pydantic-ai raises a `UserError` at runtime when the agent attempts to call the LLM -- not at agent construction time.

## Quick Start

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    response: str
    confidence: float

agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer questions concisely.",
)

async def main() -> None:
    result = await agent.run(Query(question="What is Python?"))
    print(result.data.response)      # str
    print(result.data.confidence)    # float
    print(result.usage.input_tokens) # int

asyncio.run(main())
```

## Package Exports

Everything below is importable from the top-level package:

```python
from quanted_agents import (
    # Core classes
    QuantedAgent,           # Core agent wrapper with typed I/O
    QuantedResult,          # Rich result object from agent/workflow runs
    MCPTool,                # MCP server toolset factory
    ArtifactStore,          # Typed key-value store for workflow artifacts

    # Workflow classes
    Pipeline,               # Sequential workflow
    Router,                 # Dispatcher-based conditional workflow
    RoutingDecision,        # Structured output type for Router dispatcher
    Loop,                   # Iterative refinement workflow
    Parallel,               # Concurrent fan-out/fan-in workflow
    ParallelResult,         # Result type for Parallel runs
    ParallelOutput,         # Data model wrapping Parallel branch results
    RetryPolicy,            # Retry configuration for Parallel branches

    # Hierarchical agents
    RunnableTool,           # Wraps a Runnable as a pydantic-ai Tool for hierarchical dispatch
    WorkflowBudget,         # Tracks workflow-wide budget counters
    EscalationPolicy,       # Configures child exception escalation behavior

    # Observability
    StepTiming,             # Per-step timing and usage data
    TraceEntry,             # Rich execution trace entry
    TraceWriter,            # Crash-safe JSONL trace file writer
    TraceSession,           # Consolidates multiple runs into a single trace file
    ToolSpan,               # Per-tool call trace data

    # Type aliases and protocols
    Runnable,               # Protocol for custom workflow implementations
    AssemblyFn,             # Assembly function type for Pipeline/Loop
    ParallelAssemblyFn,     # Assembly function type for Parallel
    PipelineTransformFn,    # Transform function for Pipeline stage boundaries
    InputTransformFn,       # Input transform for RunnableTool closures
    OverflowStrategy,       # Enum for context overflow behavior (RAISE, TRUNCATE_OLDEST)
    ValidationResult,       # Dataclass returned by agent.validate()
    ConcurrencyBackend,     # Protocol for custom concurrency control
    SemaphoreBackend,       # Default asyncio.Semaphore-based concurrency backend

    # Exceptions
    InvalidInputType,       # input_type or input_data is not a BaseModel subclass/instance
    InvalidOutputType,      # output_type is not a BaseModel subclass
    RecoveryExhaustedError, # Recovery budget exceeded
    PipelineTypeError,      # Pipeline step type mismatch
    RoutingError,           # Router dispatcher selected invalid target
    MaxIterationsExceeded,  # Loop hit max_iterations without convergence
    MCPConnectionError,     # MCP server connection failure
    AgentTimeoutError,      # Hard timeout fired during agent execution
    AssemblyError,          # Assembly function failed
    ConfigurationError,     # Agent configuration validation failed
    ContextOverflowError,   # Context window token count exceeded max_context_tokens
)
```

---

## QuantedAgent

### Import

```python
from quanted_agents import QuantedAgent
```

### Constructor

```python
QuantedAgent(
    model: str,
    *,
    input_type: type[BaseModel],
    output_type: type[BaseModel],
    system_prompt: str | list[str] = "",
    instructions: str | None = None,
    tools: list[Any] = [],
    toolsets: list[Any] | None = None,
    skills_path: str | Path | None = None,
    feedback_path: str | Path | None = None,
    retries: int = 1,
    deps_type: type[Any] | None = None,
    restructurer_model: str | None = None,
    max_recovery_attempts: int = 3,
    llm_call_limit: int | None = None,
    tool_call_limit: int | None = None,
    total_request_limit: int | None = None,
    soft_limit: bool = False,
    soft_timeout: float | None = None,
    hard_timeout: float | None = None,
    max_context_tokens: int | None = None,
    overflow_strategy: OverflowStrategy | None = None,
    **kwargs: Any,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | (required) | pydantic-ai model identifier (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`) |
| `input_type` | `type[BaseModel]` | (required) | Pydantic BaseModel subclass for agent input. Validated at construction time. |
| `output_type` | `type[BaseModel]` | (required) | Pydantic BaseModel subclass for agent output. Validated at construction time. |
| `system_prompt` | `str \| list[str]` | `""` | Static system prompt string or list of strings. |
| `instructions` | `str \| None` | `None` | Per-run instructions refreshed on each run, not affected by message history context. |
| `tools` | `list[Any]` | `[]` | List of tool functions using pydantic-ai's tool registration. |
| `toolsets` | `list[Any] \| None` | `None` | List of toolset objects (e.g., `MCPTool` instances) for MCP integration. |
| `skills_path` | `str \| Path \| None` | `None` | Path to a directory of markdown skill files with YAML frontmatter. The agent's LLM sees available skill names in its system prompt and can load them on-demand via an internal tool. |
| `feedback_path` | `str \| Path \| None` | `None` | Path to a directory of markdown feedback files with YAML frontmatter. Same on-demand loading behavior as skills. |
| `retries` | `int` | `1` | Number of retries for output validation failures. |
| `deps_type` | `type[Any] \| None` | `None` | Dependency injection type for tools accessing `RunContext`. |
| `restructurer_model` | `str \| None` | `None` | Cheap model for two-model restructuring pattern (e.g., `"openai:gpt-4o-mini"`). |
| `max_recovery_attempts` | `int` | `3` | Global retry budget across all recovery stages. |
| `llm_call_limit` | `int \| None` | `None` | Maximum number of LLM API calls per run. Maps to pydantic-ai's `UsageLimits.request_limit`. |
| `tool_call_limit` | `int \| None` | `None` | Maximum number of tool invocations per run. Maps to pydantic-ai's `UsageLimits.tool_calls_limit`. |
| `total_request_limit` | `int \| None` | `None` | Maximum total requests tracked at SDK level. No pydantic-ai equivalent; stored for SDK-level tracking. |
| `soft_limit` | `bool` | `False` | If `True`, usage limit violations trigger a wrap-up sequence instead of raising `UsageLimitExceeded`. The agent gets up to 2 additional LLM calls (with tools blocked) to produce final output. |
| `soft_timeout` | `float \| None` | `None` | Seconds before soft wrap-up fires. When only `soft_timeout` is set, an implicit hard backstop is added at `soft_timeout + 30s`. Can be overridden per-run. |
| `hard_timeout` | `float \| None` | `None` | Seconds before hard kill with `AgentTimeoutError`. Can be overridden per-run. |
| `max_context_tokens` | `int \| None` | `None` | Maximum estimated input tokens for the context window. When set, a pre-flight check runs before each agent call to detect overflow. Token counts are approximate estimates. |
| `overflow_strategy` | `OverflowStrategy \| None` | `None` | Strategy for handling context overflow. When `None` and `max_context_tokens` is set, defaults to `OverflowStrategy.RAISE`. Options: `RAISE` (raise `ContextOverflowError`) or `TRUNCATE_OLDEST` (drop oldest messages while preserving system prompt). |
| `**kwargs` | `Any` | | Additional kwargs passed to the underlying pydantic-ai `Agent`. Accepts `api_key` (`str`) for explicit API key configuration. |

**Raises:**
- `InvalidInputType` -- if `input_type` is not a Pydantic BaseModel subclass
- `InvalidOutputType` -- if `output_type` is not a Pydantic BaseModel subclass
- `FileNotFoundError` -- if `skills_path` or `feedback_path` directory does not exist
- `ValueError` -- if a user-defined tool collides with the reserved `_load_context` internal tool name, or if `hard_timeout <= soft_timeout`

### Methods

#### `run(input_data, **kwargs) -> QuantedResult`

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]
```

Run the agent with typed input. Validates the input type, serializes to JSON, calls the LLM, and returns a `QuantedResult` with the validated output and observability data.

If the LLM returns malformed output, the recovery pipeline activates:
1. `json-repair` attempts to fix the raw JSON
2. If repair fails and `restructurer_model` is configured, the cheap model restructures the output
3. A global retry budget prevents infinite recovery loops

**Parameters:**
- `input_data` -- a Pydantic BaseModel instance matching the agent's `input_type`
- `**kwargs` -- passed to pydantic-ai `Agent.run()` (e.g., `message_history`, `model_settings`, `usage_limits`). Also accepts `traces_path` (`str | Path`) to write real-time JSONL trace files to a directory. When set, creates a timestamped `.jsonl` file (e.g., `trace_20260220T143052_123456.jsonl`) with one JSON line per trace entry, written with flush+fsync for crash safety. The in-memory `result.trace` is unaffected.

**Returns:** `QuantedResult[Any]` with `.data`, `.usage`, `.messages`, `.new_messages`, `.trace`, `.step_timings`, `.total_usage`

**Raises:**
- `InvalidInputType` -- if `input_data` is not an instance of the agent's `input_type`
- `RecoveryExhaustedError` -- if recovery budget is exceeded

#### `run_stream(input_data, **kwargs) -> AsyncGenerator`

```python
async def run_stream(self, input_data: BaseModel, **kwargs: Any) -> AsyncGenerator[Any]
```

Stream the agent's output as partial results. Validates input, serializes to JSON, and yields partial output objects as the LLM generates them.

**Note:** Streaming does not use the recovery pipeline.

**Parameters:**
- `input_data` -- a Pydantic BaseModel instance matching the agent's `input_type`
- `**kwargs` -- passed to pydantic-ai `Agent.run_stream()`

**Yields:** Partial output objects as the LLM streams its response.

**Raises:**
- `InvalidInputType` -- if `input_data` is not an instance of the agent's `input_type`

#### `add_feedback(name, content, description) -> None`

```python
def add_feedback(self, name: str, content: str, description: str) -> None
```

Create a feedback file programmatically. Writes a markdown file with YAML frontmatter to the agent's feedback directory. The file is immediately available for loading via the internal `_load_context` tool.

**Parameters:**
- `name` -- identifier for the feedback entry (used as filename and loading key)
- `content` -- the full markdown content of the feedback
- `description` -- short description shown in the agent's system prompt catalog

**Raises:**
- `ValueError` -- if `feedback_path` was not configured on this agent

**Example:**
```python
agent.add_feedback(
    name="tone-correction",
    content="Use a more formal tone when addressing technical topics.",
    description="Feedback on communication tone",
)
```

### Properties

#### `inner -> Agent`

```python
@property
def inner(self) -> Agent[Any, Any]
```

Access the underlying pydantic-ai `Agent` for advanced usage:
- Dynamic system prompts via `@agent.inner.system_prompt`
- Custom tool decorators
- Direct pydantic-ai Agent configuration

### Async Context Manager

```python
async with agent:
    r1 = await agent.run(Query(question="First"))
    r2 = await agent.run(Query(question="Second"))
```

Pre-initializes MCP connections for multi-run scenarios. For single `.run()` calls, pydantic-ai auto-manages the connection lifecycle.

---

## QuantedResult

### Import

```python
from quanted_agents import QuantedResult
```

### Generic Type

```python
class QuantedResult[OutputT: BaseModel]:
```

Uses PEP 695 type parameter syntax (Python 3.13+). `OutputT` is the agent's `output_type` BaseModel subclass.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `OutputT` | The validated output BaseModel instance. Primary access point for structured output. |
| `usage` | `RunUsage` | Token usage statistics (`input_tokens`, `output_tokens`, `requests`). Empty `RunUsage` on recovery path. |
| `messages` | `list[ModelMessage]` | Full message history including prior context. Empty list on recovery path. |
| `new_messages` | `list[ModelMessage]` | Messages from this run only (excludes `message_history`). Empty list on recovery path. |
| `trace` | `list[TraceEntry]` | Execution trace entries. One entry for single-agent runs; one per step for workflows. |
| `step_timings` | `list[StepTiming]` | Per-step timing and usage. Falls back to a default timing entry if none recorded. |
| `total_usage` | `RunUsage` | Aggregated token usage across all steps. Same as `.usage` for single-agent runs. |
| `summary` | `str \| None` | Natural language summary alongside structured output. Extracted from the last `ModelResponse` that contains both a `TextPart` and a `ToolCallPart`. Returns `None` if unavailable (text-only agent, recovery path, or provider did not produce text). Lazy: computed on first access. |
| `artifacts` | `ArtifactStore` | The `ArtifactStore` for this result. Returns the store used during orchestration, or lazily creates an empty one on first access. |
| `termination_reason` | `str \| None` | Why the agent run terminated abnormally. One of `"soft_limit"`, `"soft_timeout"`, `"hard_timeout"`, or `None` for normal completion. |
| `was_recovered` | `bool` | Whether this result went through the recovery pipeline (json-repair or restructurer). |
| `recovery_method` | `str \| None` | How recovery was performed: `"json_repair"`, `"restructurer"`, or `None`. |
| `context_overflow_occurred` | `bool` | Whether context overflow was detected and handled during this run. |
| `messages_truncated` | `int` | Number of messages truncated due to context overflow. |

### Class Methods

#### `from_data(data) -> QuantedResult`

```python
@classmethod
def from_data(cls, data: Any) -> QuantedResult[Any]
```

Create a `QuantedResult` from a recovered BaseModel instance (used by the recovery pipeline). Usage, messages, and new_messages return sensible defaults.

---

## Observability

### StepTiming

```python
from quanted_agents import StepTiming
```

A dataclass capturing timing and token usage for a single execution step.

```python
@dataclass
class StepTiming:
    step_name: str        # e.g., "QuantedAgent(Answer)", "Pipeline.step_0"
    duration_seconds: float  # Wall-clock time via time.perf_counter()
    usage: RunUsage       # Token usage for this step
```

### TraceEntry

```python
from quanted_agents import TraceEntry
```

A dataclass capturing the complete execution context for one step.

```python
@dataclass
class TraceEntry:
    step_name: str                       # Identifies which agent/step produced this entry
    input_data: dict[str, Any]           # Input BaseModel serialized via model_dump()
    output_data: dict[str, Any]          # Output BaseModel serialized via model_dump()
    messages: list[dict[str, Any]]       # LLM messages serialized via ModelMessagesTypeAdapter
    tool_calls: list[dict[str, Any]]     # Extracted tool call info (tool_name, args, tool_call_id)
    timing: StepTiming                   # Timing and usage for this step
    model_name: str | None = None        # Model identifier from ModelResponse
    recovery_info: dict[str, Any] | None = None  # Recovery details if activated
```

**`recovery_info` dict (when present):**

| Key | Type | Description |
|-----|------|-------------|
| `json_repair_attempted` | `bool` | Whether json-repair was tried |
| `restructurer_used` | `bool` | Whether the restructurer model was used |
| `attempts_used` | `int` | Number of recovery attempts consumed |

#### `to_dict() -> dict[str, Any]`

Converts the trace entry to a fully JSON-serializable dictionary. Suitable for `json.dumps()`, logging, or export to external analysis tools.

```python
import json

entry = result.trace[0]
print(json.dumps(entry.to_dict(), indent=2))
```

**Output structure:**

```python
{
    "step_name": "QuantedAgent(Answer)",
    "input_data": {"question": "What is Python?"},
    "output_data": {"response": "...", "confidence": 0.95},
    "messages": [...],
    "tool_calls": [...],
    "timing": {
        "step_name": "QuantedAgent(Answer)",
        "duration_seconds": 1.23,
        "usage": {
            "input_tokens": 150,
            "output_tokens": 50,
            "requests": 1,
        },
    },
    "model_name": "openai:gpt-4o",
    "recovery_info": None,
}
```

---

## TraceWriter

### Import

```python
from quanted_agents import TraceWriter
```

Crash-safe, async-safe writer for JSONL trace files. Used internally when `traces_path` is passed to `run()`, but also exported for advanced custom usage.

### Constructor

```python
TraceWriter(file_path: Path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `Path` | Path to the JSONL file. Parent directory must exist. |

### Methods

#### `write(entry) -> None`

```python
async def write(self, entry: TraceEntry) -> None
```

Write a single TraceEntry as a JSON line. Uses `asyncio.Lock` for concurrency safety and `flush+fsync` for crash safety.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `file_path` | `Path` | The path to the trace file being written. |

---

## Workflow Classes

All workflow classes implement the `Runnable` protocol and can be nested inside each other.

### Pipeline

```python
from quanted_agents import Pipeline
```

Sequential workflow that chains the output of each step as input to the next.

#### Constructor

```python
Pipeline(
    steps: list[Runnable],
    *,
    input_transforms: dict[int, PipelineTransformFn] | None = None,
    assembly: AssemblyFn | None = None,
    store: ArtifactStore | None = None,
    trace_artifacts: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `list[Runnable]` | (required) | Ordered list of Runnable instances. Must contain at least 2 steps. |
| `input_transforms` | `dict[int, PipelineTransformFn] \| None` | `None` | Mapping of stage index to transform function. When provided for stage N, the transform receives the previous stage's `QuantedResult`, the Pipeline's `ArtifactStore`, and the stage index, returning a `BaseModel` for stage N's input. Stage 0 cannot have a transform. When a transform is provided, the type mismatch check is skipped for that boundary. |
| `assembly` | `AssemblyFn \| None` | `None` | Assembly function to transform accumulated store artifacts into a final output after the last step. |
| `store` | `ArtifactStore \| None` | `None` | `ArtifactStore` for recording step outputs. If not provided but `assembly` is set, a store is created automatically. |
| `trace_artifacts` | `bool` | `False` | Whether to write SDK metadata to the store under reserved `_` prefix keys. |

**Raises:** `ValueError` if fewer than 2 steps or if stage 0 has an `input_transform`. `PipelineTypeError` if adjacent steps have mismatched types and no `input_transform` bridges them.

#### `run(input_data, **kwargs) -> QuantedResult`

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]
```

Runs each step sequentially. `result.data` from step N becomes `input_data` for step N+1. Returns the final step's `QuantedResult` enriched with aggregated observability.

**Step timing names:** `Pipeline.step_0`, `Pipeline.step_1`, ...

#### Example

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Pipeline

class RawText(BaseModel):
    text: str

class Summary(BaseModel):
    summary: str

class Report(BaseModel):
    title: str
    content: str

summarizer = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=RawText,
    output_type=Summary,
    system_prompt="Summarize the text.",
)
reporter = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Summary,
    output_type=Report,
    system_prompt="Write a report from the summary.",
)

pipeline = Pipeline(steps=[summarizer, reporter])

async def main() -> None:
    result = await pipeline.run(RawText(text="Long document..."))
    print(result.data.title)             # Report.title
    print(result.total_usage.input_tokens)  # aggregated across steps
    for timing in result.step_timings:
        print(f"{timing.step_name}: {timing.duration_seconds:.2f}s")

asyncio.run(main())
```

---

### Router

```python
from quanted_agents import Router, RoutingDecision
```

Dispatcher-based workflow that classifies input and routes to a specialist.

#### RoutingDecision

```python
class RoutingDecision(BaseModel):
    target: str      # Name of the specialist to invoke
    reasoning: str = ""  # Why this specialist was chosen
```

#### Constructor

```python
Router(dispatcher: Runnable, specialists: dict[str, Runnable])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dispatcher` | `Runnable` | Agent that returns a `RoutingDecision`. |
| `specialists` | `dict[str, Runnable]` | Map of specialist names to Runnable instances. Must have at least 1. |

**Raises:** `ValueError` if fewer than 1 specialist.

#### `run(input_data, **kwargs) -> QuantedResult`

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]
```

1. Dispatcher receives `input_data`, returns `RoutingDecision`
2. Router validates `decision.target` exists in specialists
3. Selected specialist receives `input_data`, returns result

**Step timing names:** `Router.dispatcher`, `Router.specialist_{target}`

**Raises:** `RoutingError` if dispatcher selects an invalid target.

#### Example

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Router, RoutingDecision

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    response: str

classifier = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Query,
    output_type=RoutingDecision,
    system_prompt=(
        "Classify the query as 'math' or 'history'. "
        "Return the specialist name as target."
    ),
)

math_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer math questions.",
)
history_agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer history questions.",
)

router = Router(
    dispatcher=classifier,
    specialists={"math": math_agent, "history": history_agent},
)

async def main() -> None:
    result = await router.run(Query(question="What is 2+2?"))
    print(result.data.response)
    for timing in result.step_timings:
        print(f"{timing.step_name}: {timing.duration_seconds:.2f}s")

asyncio.run(main())
```

---

### Loop

```python
from quanted_agents import Loop
```

Iterative workflow that runs a body Runnable until a termination check passes or max iterations is reached.

#### Constructor

```python
Loop(
    body: Runnable,
    termination_check: Callable[[BaseModel], bool],
    *,
    max_iterations: int,
    assembly: AssemblyFn | None = None,
    store: ArtifactStore | None = None,
    trace_artifacts: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `body` | `Runnable` | (required) | Runnable to execute on each iteration. Output feeds back as input. |
| `termination_check` | `Callable[[BaseModel], bool]` | (required) | Receives the body's output data; returns `True` to stop. |
| `max_iterations` | `int` | (required) | Maximum iterations. Keyword-only, no default. Must be >= 1. |
| `assembly` | `AssemblyFn \| None` | `None` | Assembly function to transform accumulated store artifacts into a final output after convergence. |
| `store` | `ArtifactStore \| None` | `None` | `ArtifactStore` for recording iteration outputs. |
| `trace_artifacts` | `bool` | `False` | Whether to write SDK metadata to the store. |

**Raises:** `ValueError` if `max_iterations < 1`.

#### `run(input_data, **kwargs) -> QuantedResult`

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]
```

Runs body repeatedly. After each iteration, calls `termination_check(result.data)`. If it returns `True`, returns immediately. If max iterations reached without convergence, raises `MaxIterationsExceeded`.

**Raises:** `MaxIterationsExceeded` if all iterations exhaust without the termination check returning `True`.

**Step timing names:** `Loop.iteration_0`, `Loop.iteration_1`, ...

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `max_iterations` | `int` | The configured maximum iteration count. |

#### Example

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Loop, MaxIterationsExceeded

class Draft(BaseModel):
    content: str
    quality_score: float = 0.0

refiner = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Draft,
    output_type=Draft,
    system_prompt="Improve the essay. Set quality_score 0.0-1.0.",
)

loop = Loop(
    body=refiner,
    termination_check=lambda d: d.quality_score >= 0.9,
    max_iterations=5,
)

async def main() -> None:
    try:
        result = await loop.run(Draft(content="rough draft", quality_score=0.3))
    except MaxIterationsExceeded:
        print("Did not converge")
        return
    print(result.data.quality_score)
    print(f"Iterations: {len(result.step_timings)}")

asyncio.run(main())
```

---

### Parallel

```python
from quanted_agents import Parallel, ParallelResult
```

Concurrent fan-out/fan-in workflow. All branches receive the same input and execute concurrently via `asyncio.gather`.

#### Constructor

```python
Parallel(
    branches: list[Runnable],
    *,
    assembly: ParallelAssemblyFn | None = None,
    store: ArtifactStore | None = None,
    trace_artifacts: bool = False,
    retry_policy: RetryPolicy | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branches` | `list[Runnable]` | (required) | Runnable instances to execute concurrently. Must have at least 2. |
| `assembly` | `ParallelAssemblyFn \| None` | `None` | Assembly function to transform accumulated store artifacts into a final output after all branches complete. |
| `store` | `ArtifactStore \| None` | `None` | `ArtifactStore` for recording branch outputs. |
| `trace_artifacts` | `bool` | `False` | Whether to write SDK metadata to the store. |
| `retry_policy` | `RetryPolicy \| None` | `None` | Retry configuration for failed branches. When provided, branches that fail with exceptions matching `retry_on` types are retried up to `max_retries` times. |

**Raises:** `ValueError` if fewer than 2 branches.

#### `run(input_data, **kwargs) -> ParallelResult`

```python
async def run(self, input_data: BaseModel, **kwargs: Any) -> ParallelResult
```

Runs all branches concurrently. Returns a `ParallelResult` with both successes and errors.

**Step timing names:** `Parallel.branch_0`, `Parallel.branch_1`, ...

#### ParallelResult

`ParallelResult` extends `QuantedResult` with parallel-specific properties:

| Property | Type | Description |
|----------|------|-------------|
| `results` | `list[QuantedResult[Any]]` | Successful branch results. |
| `errors` | `list[Exception]` | Exceptions from failed branches. |
| `data` | `ParallelOutput` | Aggregated output with `.items` list containing each branch's data. |
| `usage` | `RunUsage` | Aggregated usage across all successful branches. |
| `trace` | `list[TraceEntry]` | Flat list of trace entries from all branches. |
| `step_timings` | `list[StepTiming]` | Per-branch timing data. |
| `total_usage` | `RunUsage` | Same as `usage` (aggregated across branches). |
| `messages` | `list[ModelMessage]` | Always empty (no single message history for parallel). |
| `new_messages` | `list[ModelMessage]` | Always empty. |

**ParallelOutput:**

```python
class ParallelOutput(BaseModel):
    items: list[Any]  # Individual data values from each successful branch
```

#### Example

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, Parallel

class Text(BaseModel):
    content: str

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

class Topics(BaseModel):
    topics: list[str]

sentiment = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Text,
    output_type=Sentiment,
    system_prompt="Analyze sentiment.",
)
topics = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Text,
    output_type=Topics,
    system_prompt="Extract topics.",
)

parallel = Parallel(branches=[sentiment, topics])

async def main() -> None:
    result = await parallel.run(Text(content="Great product!"))
    print(f"Successes: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")
    for r in result.results:
        print(r.data)
    print(f"Total tokens: {result.usage.input_tokens}")

asyncio.run(main())
```

---

## MCP Integration

### MCPTool

```python
from quanted_agents import MCPTool
```

Factory function that creates pydantic-ai MCP server toolset instances.

```python
def MCPTool(
    url: str,
    *,
    transport: str = "http",
    tool_prefix: str | None = None,
    timeout: float = 5,
    tool_retry_max: int = 0,
    tool_retry_delay: float = 0.0,
    tool_retry_backoff_factor: float = 1.0,
    argument_interceptor: InterceptorFn | None = None,
    max_concurrent_calls: int | None = None,
    concurrency_backend: ConcurrencyBackend | None = None,
    concurrency_timeout: float | None = None,
    tool_trace_level: str | None = None,
    **kwargs: Any,
) -> MCPServerStreamableHTTP | MCPServerSSE
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | (required) | MCP server endpoint (e.g., `"http://localhost:8001/mcp"`) |
| `transport` | `str` | `"http"` | `"http"` for Streamable HTTP (recommended) or `"sse"` for legacy SSE |
| `tool_prefix` | `str \| None` | `None` | Optional prefix for tool names to avoid collisions |
| `timeout` | `float` | `5` | Connection initialization timeout in seconds |
| `tool_retry_max` | `int` | `0` | Maximum silent retries per tool call before propagating error. `0` disables retry. |
| `tool_retry_delay` | `float` | `0.0` | Base delay in seconds between retries. |
| `tool_retry_backoff_factor` | `float` | `1.0` | Multiplier applied to delay after each retry. `1.0` = constant delay, `2.0` = exponential backoff. |
| `argument_interceptor` | `InterceptorFn \| None` | `None` | Per-tool interceptor callable. Receives `(tool_name, args)` and returns modified args or `None` to abort. Supports sync and async. |
| `max_concurrent_calls` | `int \| None` | `None` | Maximum concurrent tool calls for this server. Creates a default `SemaphoreBackend`. Ignored if `concurrency_backend` is provided. |
| `concurrency_backend` | `ConcurrencyBackend \| None` | `None` | Custom `ConcurrencyBackend` implementation. Takes priority over `max_concurrent_calls`. |
| `concurrency_timeout` | `float \| None` | `None` | Timeout in seconds for the default `SemaphoreBackend`. Fail-open: call proceeds on timeout. Only used with `max_concurrent_calls`. |
| `tool_trace_level` | `str \| None` | `None` | Trace verbosity: `"minimal"`, `"standard"`, or `"verbose"`. `None` disables tracing. |
| `**kwargs` | `Any` | | Passed to underlying pydantic-ai MCPServer class |

When middleware parameters are set (`argument_interceptor`, `max_concurrent_calls`, `concurrency_backend`, or `tool_trace_level`), a unified middleware pipeline replaces `direct_call_tool` with fixed order: **intercept -> throttle -> execute (with retry) -> trace**. Retry is absorbed into the middleware pipeline when both are configured.

**Raises:** `ValueError` if transport is not `"http"` or `"sse"`.

**Returns:** An `MCPServerStreamableHTTP` or `MCPServerSSE` instance.

#### Usage with QuantedAgent

```python
from quanted_agents import QuantedAgent, MCPTool

# Single MCP server
agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Use available tools.",
    toolsets=[MCPTool("http://localhost:8001/mcp")],
)

# Multiple MCP servers with prefixes
agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Use available tools.",
    toolsets=[
        MCPTool("http://localhost:8001/mcp", tool_prefix="weather"),
        MCPTool("http://localhost:8002/mcp", tool_prefix="search"),
    ],
)

# Multi-run with pre-initialized MCP connections
async with agent:
    r1 = await agent.run(Query(question="First"))
    r2 = await agent.run(Query(question="Second"))
```

---

## Exceptions

All exceptions are importable from `quanted_agents`.

| Exception | Parent | When Raised |
|-----------|--------|-------------|
| `InvalidInputType` | `TypeError` | `input_type` or `input_data` is not a Pydantic BaseModel subclass/instance |
| `InvalidOutputType` | `TypeError` | `output_type` is not a Pydantic BaseModel subclass |
| `RecoveryExhaustedError` | `RuntimeError` | Recovery budget exceeded (json-repair failed, no restructurer or restructurer failed) |
| `PipelineTypeError` | `TypeError` | Pipeline step output type does not match next step input type |
| `RoutingError` | `ValueError` | Router dispatcher selected a target not in the specialists dictionary |
| `MaxIterationsExceeded` | `RuntimeError` | Loop hits max_iterations without termination check passing |
| `MCPConnectionError` | `ConnectionError` | Connection to MCP server failed |
| `AgentTimeoutError` | `TimeoutError` | Agent's hard timeout fires. Carries `store`, `usage`, and `termination_reason` attributes for inspecting intermediate state. |
| `AssemblyError` | `Exception` | Assembly function raised an exception. Carries `store`, `last_result`, and `original_error` attributes for debugging. |
| `ConfigurationError` | `ValueError` | Agent configuration validation failed (via `agent.validate()`). Carries `errors` list attribute with specific validation error descriptions. |
| `ContextOverflowError` | `Exception` | Context window token count exceeds `max_context_tokens` (when `overflow_strategy=RAISE`). Carries `current_tokens`, `max_tokens`, `store`, and `usage` attributes. |

---

## Hierarchical Agents

Hierarchical agents enable parent-child delegation where a parent agent dispatches subtasks to child agents via LLM tool calling. The parent LLM sees each child as a tool with a single `instruction: str` parameter and decides which children to invoke based on their descriptions.

### RunnableTool

```python
from quanted_agents import RunnableTool
```

Wraps any `Runnable` (QuantedAgent, Pipeline, Loop, Parallel) as a pydantic-ai `Tool` for hierarchical dispatch. The parent LLM sees a simple `instruction: str` parameter; complex input construction happens in the optional `input_transform` closure.

#### Constructor

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
| `runnable` | `Runnable` | (required) | The child Runnable to wrap (QuantedAgent, Pipeline, etc.). |
| `name` | `str` | (required) | Tool name visible to the parent LLM. Must be unique among siblings. |
| `description` | `str` | (required) | Tool description sent to the parent LLM. Critical for routing quality -- the LLM picks which child to invoke based on this text. |
| `input_transform` | `InputTransformFn \| None` | `None` | Optional closure that converts `(store, instruction)` into the child Runnable's input. Required when child `input_type` is not `str`. |
| `escalation_policy` | `EscalationPolicy \| None` | `None` | Error handling policy. When `None`, uses `EscalationPolicy.DEFAULT`. |

**Closure semantics:** RunnableTool does not execute child agents directly. Instead, `as_tool()` returns a pydantic-ai `Tool` whose inner closure captures the store, budget, and escalation policy. The closure executes the full dispatch flow when the parent LLM calls the tool. This means the same RunnableTool instance can be bound to different stores/budgets for use across different parent agents.

#### `as_tool(store, budget) -> Tool`

```python
def as_tool(
    self,
    store: ArtifactStore | None = None,
    budget: WorkflowBudget | None = None,
) -> Tool
```

Create a pydantic-ai `Tool` instance from this RunnableTool. The returned Tool is passed to the parent agent's `tools=` parameter.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `ArtifactStore \| None` | `None` | Optional store for writing child results to a namespaced sub-store (e.g., `"search/result"`). |
| `budget` | `WorkflowBudget \| None` | `None` | Optional budget for tracking consumption. `budget.deduct()` is called automatically after each child run. |

**Returns:** A pydantic-ai `Tool` ready for registration on a parent agent.

**Raises:** `TypeError` if the child Runnable's `input_type` is not `str` and no `input_transform` was provided.

**Dispatch flow:**

1. Parent LLM calls the tool with `instruction: str`
2. If `input_transform` is set, transforms `(store, instruction)` into the child's input
3. Runs the child Runnable with `usage=ctx.usage` (shared budget reference)
4. Deducts from `WorkflowBudget` (if provided)
5. Writes `result.data` and `result.summary` to the namespaced store (if provided)
6. Returns `result.summary` (or `str(result.data)`) as text to the parent LLM
7. On error, consults `EscalationPolicy` to decide: re-raise or return error text

#### Example

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent, RunnableTool

class SearchResult(BaseModel):
    answer: str

class TaskInput(BaseModel):
    task: str

class TaskOutput(BaseModel):
    summary: str

# Child agent with input_type=str (no input_transform needed)
searcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=str,
    output_type=SearchResult,
    system_prompt="Search for information.",
)

# Wrap as tool
search_tool = RunnableTool(
    runnable=searcher,
    name="search",
    description="Search for factual information.",
)

# Parent agent uses the tool
orchestrator = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=TaskInput,
    output_type=TaskOutput,
    system_prompt="Use search to find facts, then summarize.",
    tools=[search_tool.as_tool()],
)

async def main() -> None:
    result = await orchestrator.run(TaskInput(task="Research AI trends"))
    print(result.data.summary)

asyncio.run(main())
```

---

### WorkflowBudget

```python
from quanted_agents import WorkflowBudget
```

Tracks workflow-wide budget counters with deduction semantics. Parent and all child agents draw from the same shared pool.

#### Constructor

```python
WorkflowBudget(
    llm_call_limit: int | None = None,
    tool_call_limit: int | None = None,
    total_request_limit: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_call_limit` | `int \| None` | `None` | Maximum LLM calls (parent + all children). Maps to pydantic-ai's `request_limit`. `None` means unlimited. |
| `tool_call_limit` | `int \| None` | `None` | Maximum tool executions across the hierarchy. Maps to pydantic-ai's `tool_calls_limit`. `None` means unlimited. |
| `total_request_limit` | `int \| None` | `None` | Maximum total requests (LLM + tool). No pydantic-ai equivalent -- tracked internally. `None` means unlimited. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `llm_call_limit` | `int \| None` | Remaining LLM call budget, or `None` if unlimited. |
| `tool_call_limit` | `int \| None` | Remaining tool call budget, or `None` if unlimited. |
| `total_request_limit` | `int \| None` | Remaining total request budget, or `None` if unlimited. |

#### Methods

##### `remaining(counter) -> int | None`

```python
def remaining(self, counter: str) -> int | None
```

Get the remaining count for a named counter.

**Parameters:**
- `counter` -- The counter name (e.g., `"llm_call_limit"`, `"tool_call_limit"`, `"total_request_limit"`).

**Returns:** The remaining count, or `None` if the counter is unlimited or does not exist.

##### `to_usage_limits() -> UsageLimits`

```python
def to_usage_limits(self) -> UsageLimits
```

Bridge to pydantic-ai's `UsageLimits`. Maps SDK counter names to pydantic-ai fields:
- `llm_call_limit` -> `request_limit`
- `tool_call_limit` -> `tool_calls_limit`
- `total_request_limit` -> no pydantic-ai equivalent (tracked internally only)

**Returns:** A `UsageLimits` instance for passing to `agent.run(usage_limits=...)`.

##### `deduct(usage) -> None`

```python
def deduct(self, usage: RunUsage) -> None
```

Subtract consumed resources from remaining counters. **Called automatically by RunnableTool** after each child run completes -- users should NOT call this directly.

Only deducts from counters that have limits set (not `None`). Floors at zero to prevent negative values.

**Parameters:**
- `usage` -- The `RunUsage` from the completed child run.

#### Example

```python
from quanted_agents import WorkflowBudget

# Create a budget: 20 LLM calls, 10 tool calls
budget = WorkflowBudget(llm_call_limit=20, tool_call_limit=10)

# Pass to parent agent as usage_limits
result = await orchestrator.run(
    input_data,
    usage_limits=budget.to_usage_limits(),
)

# Check remaining budget after the run
print(f"LLM calls remaining: {budget.remaining('llm_call_limit')}")
print(f"Tool calls remaining: {budget.remaining('tool_call_limit')}")
```

---

### EscalationPolicy

```python
from quanted_agents import EscalationPolicy
```

Configures which child exceptions propagate to the parent (crashing the tool call) versus being caught and returned as error text (letting the parent LLM decide how to proceed).

#### Constructor

```python
EscalationPolicy(
    always_escalate: set[type[Exception]] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_escalate` | `set[type[Exception]] \| None` | `None` | Exception types that should always propagate. When `None`, uses the default set. |

**Default escalation set:** `{UsageLimitExceeded, KeyboardInterrupt, SystemExit}`

- `UsageLimitExceeded` -- shared budget pool is exhausted, parent cannot recover
- `KeyboardInterrupt` -- user requested termination
- `SystemExit` -- process termination

All other exceptions are caught and returned as error text to the parent LLM.

#### Class Variable

| Variable | Type | Description |
|----------|------|-------------|
| `DEFAULT` | `EscalationPolicy` | Pre-built instance with the default escalation set. Used when no policy is specified. |

#### Methods

##### `should_escalate(exc) -> bool`

```python
def should_escalate(self, exc: Exception) -> bool
```

Determine whether an exception should propagate to the parent.

**Parameters:**
- `exc` -- The exception raised by the child Runnable.

**Returns:** `True` if the exception type is in the `always_escalate` set, `False` if it should be caught and returned as error text.

#### Common Configurations

```python
from quanted_agents import EscalationPolicy
from pydantic_ai.exceptions import UsageLimitExceeded

# Default: UsageLimitExceeded, KeyboardInterrupt, SystemExit escalate
default_policy = EscalationPolicy.DEFAULT

# Permissive: only system-level exceptions escalate.
# Budget exhaustion returns as text -- parent can wrap up gracefully.
permissive = EscalationPolicy(
    always_escalate={KeyboardInterrupt, SystemExit}
)

# Strict: everything escalates -- any child error crashes the parent
strict = EscalationPolicy(
    always_escalate={Exception}
)

# Custom: add domain-specific exceptions to the escalation set
from quanted_agents import MCPConnectionError
custom = EscalationPolicy(
    always_escalate={UsageLimitExceeded, MCPConnectionError, KeyboardInterrupt, SystemExit}
)
```

---

### InputTransformFn

```python
from quanted_agents import InputTransformFn
```

Type alias for functions that transform the parent LLM's instruction into the child Runnable's input.

#### Signature

```python
InputTransformFn = Union[
    Callable[[ArtifactStore, str], Any],
    Callable[[ArtifactStore, str], Awaitable[Any]],
]
```

**Parameters received:**
- `store` (`ArtifactStore`) -- The artifact store, provided at invocation time (not captured at registration). Gives access to artifacts written by prior child runs.
- `instruction` (`str`) -- The raw string the parent LLM passed as the tool argument.

**Returns:** The input for the child Runnable -- typically a Pydantic BaseModel instance.

**When needed:** When the child Runnable's `input_type` is not `str`. If `input_type` is `str`, the instruction is passed directly and no transform is required. If `input_type` is not `str` and no `input_transform` is provided, `as_tool()` raises `TypeError` at registration time.

**Sync/async support:** May be sync or async. When async, RunnableTool awaits it before calling the child Runnable.

#### Example

```python
from pydantic import BaseModel
from quanted_agents import ArtifactStore

class SearchInput(BaseModel):
    query: str
    max_results: int = 10

# Sync transform: builds typed input from instruction
def build_search_input(store: ArtifactStore, instruction: str) -> SearchInput:
    return SearchInput(query=instruction, max_results=10)

# Async transform: reads from store to build input
async def build_analysis_input(store: ArtifactStore, instruction: str) -> AnalysisInput:
    search_data = store.get("search/result")
    return AnalysisInput(data=search_data, focus=instruction)
```

---

## Types

### Runnable Protocol

```python
from quanted_agents import Runnable
# or
from quanted_agents.types import Runnable
```

The protocol that all agents and workflows implement. Use this for custom implementations.

```python
@runtime_checkable
class Runnable(Protocol):
    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        ...
```

Any object implementing `async def run(input_data: BaseModel, **kwargs) -> QuantedResult` can be used as a Pipeline step, Router specialist, Loop body, or Parallel branch.

---

### TraceSession

```python
from quanted_agents import TraceSession
```

Context manager that consolidates multiple agent runs into a single JSONL trace file. All `agent.run()` calls within the session block that pass `trace_session=session` write their trace entries to the same file. Generates a `session_id` (UUID) attached to every trace entry for correlation.

#### Constructor

```python
TraceSession(file_path: str | Path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str \| Path` | Path to the JSONL file for consolidated traces. Parent directory is created automatically. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session_id` | `str` | Unique session identifier (UUID). Attached to every trace entry for correlation. |
| `writer` | `TraceWriter` | The `TraceWriter` used for this session. |
| `file_path` | `Path` | The path to the session trace file. |

#### Usage

```python
import asyncio
from quanted_agents import QuantedAgent, TraceSession

async def main() -> None:
    async with TraceSession("traces/session.jsonl") as session:
        print(f"Session ID: {session.session_id}")
        r1 = await agent.run(input1, trace_session=session)
        r2 = await agent.run(input2, trace_session=session)
    # Both runs written to the same file with the same session_id

asyncio.run(main())
```

---

### RetryPolicy

```python
from quanted_agents import RetryPolicy
```

Dataclass for configuring retry behavior on failed `Parallel` branches.

#### Constructor

```python
@dataclass
class RetryPolicy:
    max_retries: int = 0
    retry_on: list[type[Exception]] = field(default_factory=list)
    delay_seconds: float = 1.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `0` | Maximum number of retry attempts for each failed branch. |
| `retry_on` | `list[type[Exception]]` | `[]` | Exception types eligible for retry. Only branches that failed with one of these types will be retried. |
| `delay_seconds` | `float` | `1.0` | Seconds to wait between retry attempts. First retry attempt does not sleep; delay applies from attempt 1 onward. |

#### Example

```python
from quanted_agents import Parallel, RetryPolicy

parallel = Parallel(
    branches=[agent_a, agent_b],
    retry_policy=RetryPolicy(
        max_retries=2,
        retry_on=[ConnectionError, TimeoutError],
        delay_seconds=1.0,
    ),
)
```

---

### AssemblyFn

```python
from quanted_agents import AssemblyFn
```

Type alias for assembly functions used by `Pipeline` and `Loop`. An assembly function receives the `ArtifactStore` and the last `QuantedResult`, and returns a transformed output. May be sync or async.

#### Signature

```python
AssemblyFn = Union[
    Callable[[ArtifactStore, QuantedResult], T],
    Callable[[ArtifactStore, QuantedResult], Awaitable[T]],
]
```

#### Example

```python
from quanted_agents import ArtifactStore, Pipeline
from quanted_agents.result import QuantedResult

def assemble_report(store: ArtifactStore, last_result: QuantedResult) -> FinalReport:
    step_0_data = store["Pipeline.step_0/result"]
    step_1_data = last_result.data
    return FinalReport(summary=step_0_data.summary, details=step_1_data.content)

pipeline = Pipeline(steps=[summarizer, detailer], assembly=assemble_report, store=ArtifactStore())
```

---

### ParallelAssemblyFn

```python
from quanted_agents import ParallelAssemblyFn
```

Type alias for assembly functions used by `Parallel`. Same as `AssemblyFn` but the second argument is `ParallelResult`, giving access to `.results` and `.errors` properties.

#### Signature

```python
ParallelAssemblyFn = Union[
    Callable[[ArtifactStore, ParallelResult], T],
    Callable[[ArtifactStore, ParallelResult], Awaitable[T]],
]
```

---

### PipelineTransformFn

```python
from quanted_agents import PipelineTransformFn
```

Transform function for bridging type gaps between `Pipeline` stages. Receives the previous stage's full `QuantedResult`, the Pipeline's `ArtifactStore`, and the current stage index. Returns a `BaseModel` instance suitable as input for the current stage.

#### Signature

```python
PipelineTransformFn = Union[
    Callable[[QuantedResult, ArtifactStore, int], BaseModel],
    Callable[[QuantedResult, ArtifactStore, int], Awaitable[BaseModel]],
]
```

#### Example

```python
from quanted_agents import ArtifactStore, Pipeline
from quanted_agents.result import QuantedResult

def transform_stage_1(result: QuantedResult, store: ArtifactStore, stage: int) -> StageOneInput:
    return StageOneInput(data=result.data.raw_text, metadata={"source": "stage_0"})

pipeline = Pipeline(
    steps=[extractor, analyzer],
    input_transforms={1: transform_stage_1},
)
```

---

### OverflowStrategy

```python
from quanted_agents import OverflowStrategy
```

Enum for context window overflow behavior. Used with `QuantedAgent`'s `overflow_strategy` parameter.

#### Values

| Value | Description |
|-------|-------------|
| `OverflowStrategy.RAISE` | Raise `ContextOverflowError` when estimated token count exceeds `max_context_tokens`. This is the default when `max_context_tokens` is set and `overflow_strategy` is `None`. |
| `OverflowStrategy.TRUNCATE_OLDEST` | Drop oldest messages (preserving system prompt) to fit within the token budget. |

#### Example

```python
from quanted_agents import QuantedAgent, OverflowStrategy

agent = QuantedAgent(
    "openai:gpt-4o",
    input_type=Query,
    output_type=Answer,
    system_prompt="Answer questions.",
    max_context_tokens=8000,
    overflow_strategy=OverflowStrategy.TRUNCATE_OLDEST,
)
```

---

### ArtifactStore

```python
from quanted_agents import ArtifactStore
```

Typed key-value store with version history for workflow artifacts. Provides dict-like access for the latest value per key and append-only version history. Keys starting with `_` are reserved for SDK internal use.

#### Constructor

```python
ArtifactStore()
```

No parameters. Creates an empty store.

#### Methods and Operators

| Method/Operator | Signature | Description |
|----------------|-----------|-------------|
| `store[key] = value` | `__setitem__(key: str, value: Any) -> None` | Store a value, appending to version history. Raises `KeyError` if key starts with `_`. |
| `store[key]` | `__getitem__(key: str) -> Any` | Get the latest value for a key. Raises `KeyError` if not found. |
| `store.get(key, type_)` | `get(key: str, type_: type[T]) -> T` | Get latest value with runtime type checking. Raises `TypeError` if type mismatch. |
| `store.history(key)` | `history(key: str) -> list[Any]` | Get full version history for a key (returns a copy). |
| `key in store` | `__contains__(key: str) -> bool` | Check if a key exists. |
| `store.keys()` | `keys() -> KeysView[str]` | Return all keys in the store. |
| `len(store)` | `__len__() -> int` | Return the number of keys. |
| `bool(store)` | `__bool__() -> bool` | Always returns `True` (even when empty) to prevent falsy-empty-store gotchas. |

#### Store Key Conventions

Workflow patterns write step results using these key patterns:

| Workflow | Key Pattern | Example |
|----------|------------|---------|
| Pipeline | `Pipeline.step_{N}/result` | `store["Pipeline.step_0/result"]` |
| Loop | `Loop.iteration_{N}/result` | `store["Loop.iteration_0/result"]` |
| Parallel | `Parallel.branch_{N}/result` | `store["Parallel.branch_0/result"]` |
| RunnableTool | `{tool_name}/result` | `store["search/result"]` |

#### Example

```python
from quanted_agents import ArtifactStore

store = ArtifactStore()
store["analysis"] = AnalysisResult(score=0.8)
store["analysis"] = RefinedAnalysis(score=0.95)  # overwrites latest, appends to history

latest = store["analysis"]                        # RefinedAnalysis
typed = store.get("analysis", RefinedAnalysis)    # typed access with runtime check
all_versions = store.history("analysis")          # [AnalysisResult, RefinedAnalysis]
print(list(store.keys()))                         # ["analysis"]
```

---

### ToolSpan

```python
from quanted_agents import ToolSpan
```

Dataclass capturing per-tool call trace data for MCP tool observability. Lighter than `TraceEntry` (which is agent-step-level). Collected during agent runs when `tool_trace_level` is set on an `MCPTool`.

#### Fields

```python
@dataclass
class ToolSpan:
    tool_name: str                          # Name of the MCP tool called
    status: str                             # "success", "error", or "aborted"
    duration_seconds: float                 # Wall-clock time in seconds
    args: dict[str, Any] | None = None      # Tool arguments (standard/verbose)
    result_preview: str | None = None       # Truncated result string (standard/verbose)
    error_detail: str | None = None         # Error message if failed (standard/verbose)
    original_args: dict[str, Any] | None = None  # Pre-interceptor args (verbose only)
    full_result: Any | None = None          # Complete untruncated result (verbose only)
```

#### Methods

##### `to_dict(level) -> dict[str, Any]`

Serialize the span based on verbosity level (`"minimal"`, `"standard"`, or `"verbose"`). Minimal includes only `tool_name`, `status`, `duration_seconds`. Standard adds `args`, `result_preview`, `error_detail`. Verbose adds `original_args`, `full_result`.

---

### ValidationResult

```python
from quanted_agents import ValidationResult
```

Dataclass returned by `agent.validate()` and `agent.avalidate()`. Contains validation outcome with categorized issues.

#### Fields

```python
@dataclass
class ValidationResult:
    valid: bool = True                    # True if no errors were found
    errors: list[str] = field(default_factory=list)    # Validation error descriptions
    warnings: list[str] = field(default_factory=list)  # Validation warning descriptions
```

#### Example

```python
from quanted_agents import QuantedAgent

agent = QuantedAgent("openai:gpt-4o", input_type=Query, output_type=Answer)
result = agent.validate()
if not result.valid:
    print(f"Errors: {result.errors}")
if result.warnings:
    print(f"Warnings: {result.warnings}")
```

---

### ConcurrencyBackend

```python
from quanted_agents import ConcurrencyBackend
```

Protocol for pluggable concurrency control backends. Implement this protocol for custom concurrency control (e.g., Redis-backed distributed semaphore).

#### Required Methods

```python
@runtime_checkable
class ConcurrencyBackend(Protocol):
    def acquire(self, tool_name: str) -> Any:
        """Return an async context manager that acquires and releases a concurrency slot."""
        ...
```

The `acquire` method must return an async context manager. Usage:

```python
async with backend.acquire(tool_name):
    result = await call_tool(tool_name, args)
```

---

### SemaphoreBackend

```python
from quanted_agents import SemaphoreBackend
```

Default in-process concurrency backend using `asyncio.BoundedSemaphore`. Created automatically when `max_concurrent_calls` is passed to `MCPTool`, but can also be instantiated directly.

#### Constructor

```python
SemaphoreBackend(max_concurrent: int, timeout: float | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent` | `int` | (required) | Maximum number of concurrent tool calls allowed. |
| `timeout` | `float \| None` | `None` | Timeout in seconds for semaphore acquisition. `None` means wait forever. On timeout, call proceeds (fail-open semantics). |

#### Example

```python
from quanted_agents import MCPTool, SemaphoreBackend

# Auto-created via MCPTool
tools = MCPTool("http://localhost:8001/mcp", max_concurrent_calls=5)

# Or create directly for custom configuration
backend = SemaphoreBackend(max_concurrent=3, timeout=10.0)
tools = MCPTool("http://localhost:8001/mcp", concurrency_backend=backend)
```

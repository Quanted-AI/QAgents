# QuantedAgents

[![PyPI version](https://img.shields.io/pypi/v/quanted-agents.svg)](https://pypi.org/project/quanted-agents/)
[![Python versions](https://img.shields.io/pypi/pyversions/quanted-agents.svg)](https://pypi.org/project/quanted-agents/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Quanted-AI/QAgents/blob/main/LICENSE)
[![CI](https://github.com/Quanted-AI/QAgents/actions/workflows/ci.yml/badge.svg)](https://github.com/Quanted-AI/QAgents/actions/workflows/ci.yml)

> Type-safe agent SDK wrapping [pydantic-ai](https://ai.pydantic.dev/) with Pydantic
> `BaseModel` I/O enforcement, composable workflow primitives
> (Pipeline, Router, Loop, Parallel), hierarchical agent orchestration, and
> built-in observability.

---

## Installation

```bash
pip install quanted-agents
```

Requires Python 3.13+.

Set the API key for whichever LLM provider you use:

```bash
export OPENAI_API_KEY="sk-..."
# or ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ...
```

## Quickstart

```python
import asyncio
from pydantic import BaseModel
from quanted_agents import QuantedAgent

class Article(BaseModel):
    text: str

class Analysis(BaseModel):
    summary: str
    sentiment: str
    topics: list[str]

agent = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=Article,
    output_type=Analysis,
    system_prompt="Analyze the article: short summary, sentiment, key topics.",
)

async def main() -> None:
    result = await agent.run(Article(text="AI is reshaping healthcare ..."))
    print(result.data.summary)
    print(result.data.sentiment, result.data.topics)
    print(f"Tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out")

asyncio.run(main())
```

The input and output are real Pydantic models — validated, typed, and IDE-completable.

## Why QuantedAgents

- **Type-safe I/O** — every agent declares an input and output `BaseModel`. Inputs are serialized for the LLM automatically; outputs are validated. No string-juggling.
- **Composable workflows** — `Pipeline`, `Router`, `Loop`, and `Parallel` are first-class primitives that nest recursively. Build complex agentic flows without leaving the type system.
- **Hierarchical agents** — parent agents can delegate to child agents as tools, with shared `WorkflowBudget` enforcement and configurable escalation policies.
- **Observability built in** — every `run()` returns usage, step timings, and a trace you can serialize to JSONL via `TraceWriter` / `TraceSession`.
- **MCP support** — connect to Model Context Protocol servers via `MCPTool`.
- **Recovery primitives** — JSON repair, structured re-prompting via a restructurer model, configurable retry policies.
- **Claude Code skill included** — bundled `.claude/skills/quanted-agents/` makes Claude Code reach for this SDK by default when building agent workflows.

## Workflow primitives at a glance

```python
from quanted_agents import Pipeline, Router, Loop, Parallel

# Pipeline: run agents in sequence, piping output -> input
flow = Pipeline([extract_agent, classify_agent, summarize_agent])

# Router: pick a downstream agent based on a routing decision
router = Router(decider, branches={"refund": refund_agent, "support": support_agent})

# Loop: re-run until a condition is met (or a max iteration count)
loop = Loop(critic_agent, until=lambda r: r.data.is_acceptable, max_iterations=5)

# Parallel: fan out to N agents, gather typed results
fanout = Parallel([researcher_a, researcher_b, researcher_c])
```

All four nest inside each other and inside `QuantedAgent` tool definitions.

## Documentation

- [API Reference](https://github.com/Quanted-AI/QAgents/blob/main/docs/api-reference.md) — every class, method, and parameter.
- [Patterns Guide](https://github.com/Quanted-AI/QAgents/blob/main/docs/patterns-guide.md) — 19 production patterns with runnable examples.
- [Hierarchical Agents Guide](https://github.com/Quanted-AI/QAgents/blob/main/docs/hierarchical-agents-guide.md) — building parent/child agent systems.

## Examples

Thirteen runnable examples in [`examples/`](https://github.com/Quanted-AI/QAgents/tree/main/examples), numbered by complexity:

| #  | Example                  | Concept                                 |
|----|--------------------------|-----------------------------------------|
| 01 | Single agent             | Typed I/O, basic observability          |
| 02 | Pipeline                 | Sequential composition                  |
| 03 | Router                   | Branching by routing decision           |
| 04 | Loop                     | Iterate until a predicate is satisfied  |
| 05 | Parallel                 | Fan-out / fan-in                        |
| 06 | Skills + feedback        | On-demand context loading               |
| 07 | Trace logging            | JSONL traces via `TraceWriter`          |
| 08 | Hierarchical agents      | Parent/child orchestration              |
| 09 | Dual stream              | Streaming + structured output           |
| 10 | Soft limits              | `WorkflowBudget` with soft caps         |
| 11 | Tool middleware          | Wrapping tool calls                     |
| 12 | Trace sessions           | Long-running session tracing            |
| 13 | Parallel retry           | Per-branch retry policies               |

## Claude Code skill

The repository ships a [Claude Code](https://docs.anthropic.com/claude-code) skill under `.claude/skills/quanted-agents/`. Drop it into any project and Claude Code will:

- Use `quanted_agents` as its default agent framework.
- Read the bundled SDK reference before writing agent code.
- Apply production defaults (restructurer model, recovery retries, soft budgets, trace logging).
- Prefer focused child agents over monolithic single agents.

To install the skill into a project:

```bash
mkdir -p /path/to/your/project/.claude/skills
cp -r .claude/skills/quanted-agents /path/to/your/project/.claude/skills/
```

Or globally for all your projects:

```bash
mkdir -p ~/.claude/skills
cp -r .claude/skills/quanted-agents ~/.claude/skills/
```

## Contributing

Issues and pull requests welcome. See [CONTRIBUTING.md](https://github.com/Quanted-AI/QAgents/blob/main/CONTRIBUTING.md) for the development setup, code style, and PR process. Bug reports and feature requests use the issue templates.

By contributing you agree your contributions are licensed under Apache 2.0 (the project's license).

## Security

For security vulnerabilities, please follow the disclosure process in [SECURITY.md](https://github.com/Quanted-AI/QAgents/blob/main/SECURITY.md) — **do not** open a public issue.

## License

Licensed under the [Apache License, Version 2.0](https://github.com/Quanted-AI/QAgents/blob/main/LICENSE). See [NOTICE](https://github.com/Quanted-AI/QAgents/blob/main/NOTICE) for attribution.

Copyright (c) 2026 Quanted.

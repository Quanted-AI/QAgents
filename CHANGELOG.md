# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-05-21

### Added

- Initial public release as open source under Apache 2.0.
- Type-safe `QuantedAgent` with Pydantic `BaseModel` inputs and outputs.
- Composable workflow primitives: `Pipeline`, `Router`, `Loop`, `Parallel`.
- Hierarchical agent orchestration with `WorkflowBudget`, `EscalationPolicy`, and `RunnableTool`.
- Built-in trace logging via `TraceWriter` and `TraceSession`.
- MCP (Model Context Protocol) integration via `MCPTool`.
- Artifact store for sharing typed objects across agents (`ArtifactStore`).
- Recovery primitives: JSON repair, restructurer model, configurable retry policies.
- Claude Code skill (`.claude/skills/quanted-agents/`) bundling SDK docs and conventions.
- 13 runnable examples covering single agents through hierarchical workflows.
- Documentation: API reference, patterns guide, hierarchical agents guide.

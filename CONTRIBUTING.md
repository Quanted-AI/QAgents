# Contributing to QuantedAgents

Thanks for your interest in contributing. This project is maintained by [Quanted](https://quanted.com) and welcomes external contributions under the Apache 2.0 license.

## Quick start

```bash
git clone https://github.com/Quanted-AI/QAgents.git
cd QAgents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the test suite

```bash
pytest -v
```

Some tests exercise real model providers and require API keys via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). The bulk of the suite runs offline.

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) — config in `pyproject.toml`. Run `ruff check .` and `ruff format .` before committing.
- **Type checking**: [mypy](https://mypy.readthedocs.io/) in strict mode. Run `mypy quanted_agents`.
- **Line length**: 110 characters.
- **Python**: 3.13+.
- **Type hints**: required on every public function/method signature and `__init__` attribute. See [api-reference.md](docs/api-reference.md) for examples.
- **Docstrings**: Google style for every public module, class, and function.
- **Imports**: top of file only, grouped (stdlib / third-party / local), no wildcards.
- **No `.format()` / `%` formatting**: use f-strings.
- **No walrus operator (`:=`)**.

## Commit messages

Conventional, short subject (≤72 chars), imperative mood. Examples:

```
fix: handle empty tool result in restructurer
feat: add soft_limit to WorkflowBudget
docs: clarify Router decision contract
```

## Pull request process

1. Open an issue first for non-trivial changes (new features, breaking changes) so the design can be agreed before code is written.
2. Branch from `main`: `git switch -c your-feature-name`.
3. Add or update tests. PRs without tests covering the changed behavior will not be merged.
4. Run `ruff check . && mypy quanted_agents && pytest` locally — CI runs the same checks.
5. Update `CHANGELOG.md` under the `## [Unreleased]` section (add the section if it doesn't exist yet).
6. Open a PR against `main`. Fill out the PR template.

## Reporting bugs

Use the [bug report issue template](https://github.com/Quanted-AI/QAgents/issues/new?template=bug_report.yml). Include:

- A minimal reproducible example.
- The expected vs. actual behavior.
- Your Python version, OS, and `quanted-agents` version (`pip show quanted-agents`).
- Full traceback if applicable.

## Reporting security vulnerabilities

**Do not open public issues for security vulnerabilities.** See [SECURITY.md](SECURITY.md) for the private disclosure channel.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating you agree to abide by it.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 (the same license as the project).

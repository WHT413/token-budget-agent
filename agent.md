# token-budget-agent Source Standard

This file defines the coding standard for this repository so future vibecoding stays consistent, testable, and production-oriented.

## Architecture

- Keep source code under `src/`.
- Use package boundaries by domain:
  - `src/config.py`: validated runtime configuration only.
  - `src/agent/`: LangGraph state, graph wiring, and node implementations.
  - `src/agent/nodes/`: single-purpose graph nodes.
  - `src/api/`: FastAPI app/routes.
  - `src/fe/`: reserved for future frontend technology decisions.
  - `scripts/`: operational scripts, never business logic.
- Do not create nested project folders inside the repository root.

## Configuration Pattern

- Environment variables are the single source of truth.
- Use Pydantic Settings for loading, validation, and type control.
- Do not hardcode runtime values as defaults in code.
- Keep all required keys documented in `.env.example`.
- Load settings through a single cached provider:

```python
settings = get_settings()
```

- Import `settings` in application code instead of repeatedly reading `os.environ`.
- Secrets must use `SecretStr` or equivalent secret-safe types.
- Derived mappings such as model aliases may be computed from validated settings.

## Import Conventions

- Inside a package, prefer relative imports:

```python
from .state import AgentState
from ..state import AgentState
```

- Scripts may add the repository root to `sys.path` only at the script boundary.
- Do not mix `src.agent...` imports with `agent...` imports inside package modules.

## LangGraph Conventions

- `src/agent/state.py` owns `AgentState`.
- Each node module exposes one primary node function.
- Node functions accept `AgentState` and return `AgentState` unless they are explicit routing functions.
- `src/agent/graph.py` owns graph construction/compilation.
- Keep graph wiring declarative and simple; node internals should not wire graph edges.

## API Conventions

- `src/api/routes.py` owns the FastAPI `app` until the API grows enough to split routers.
- Route handlers should be thin and delegate agent/business behavior to `src/agent/` modules.
- Keep `/health` lightweight and dependency-free.

## Script Conventions

- Scripts must be runnable with `uv run python scripts/<name>.py`.
- Scripts should load settings via `src.config.settings`.
- Scripts should return meaningful process exit codes.
- Operational checks must print actionable `PASS`/`FAIL` output.

## Dependency Management

- Use `uv add` / `uv add --dev` for dependency changes.
- Keep `pyproject.toml` and `uv.lock` in sync.
- Do not add unused framework dependencies.

## Validation Workflow

Before committing, run:

```bash
uv run python scripts/check_connection.py
```

When graph code changes, also validate imports/compilation:

```bash
uv run python -c "from src.agent.graph import compile_graph; compile_graph(); print('IMPORTS_OK')"
```

## Git Workflow

- Work from the repository root.
- Use branch `dev` for setup/development commits.
- Do not commit `.env`, virtual environments, caches, or generated log JSON files.

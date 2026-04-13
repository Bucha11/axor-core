# Contributing to axor-core

## Architecture principles

Before contributing, read the design invariants in README.md.

The most important ones:

- **Core never imports providers.** No anthropic, openai, or other SDK imports in `axor_core/`.
- **Policy meaning belongs to core.** Adapters translate envelopes, never define governance.
- **Executors never self-assign capabilities.** Always derived from policy.
- **Waste elimination always runs.** Mode controls aggressiveness, not whether it happens.

## Setup

```bash
git clone https://github.com/your-org/axor-core
cd axor-core
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
pytest tests/integration/   # integration only
pytest tests/ -k "cancel"   # filter by name
```

## Adding a subsystem

1. Add contracts to `axor_core/contracts/` first — no imports from other axor_core modules
2. Implement in the appropriate subsystem package
3. Export from the package `__init__.py`
4. Add tests in `tests/<subsystem>/`
5. Update `axor_core/__init__.py` if adding public API

## Adding an adapter

Adapters live in separate repositories (`axor-claude`, `axor-openai`, etc.).
They implement:
- `Invokable` — the executor contract
- `ToolHandler` — one per tool
- `ExtensionLoader` — optional, for provider-specific extensions (CLAUDE.md etc.)

See `README.md` → "Implementing an Adapter" for details.

## Pull request checklist

- [ ] No provider imports in `axor_core/`
- [ ] New contracts added to `contracts/__init__.py`
- [ ] Tests added for new functionality  
- [ ] `PYTHONPATH=. python -c "from axor_core import GovernedSession"` passes

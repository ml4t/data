# `ml4t.data` source guide

This directory implements the `ml4t.data` package. Follow the repository-level `AGENTS.md` for
quality, compatibility, and release requirements. This guide only routes source changes.

## Change locations

| Area | Responsibility |
|---|---|
| `providers/` | Provider contracts, registry metadata, synchronous and asynchronous adapters |
| `storage/` | Storage protocols, path and key handling, Parquet backends, metadata, and migrations |
| `futures/` | Futures symbology, downloads, continuous contracts, rolls, and adjustments |
| `data_manager.py`, `update_manager.py`, `provider_updater.py`, `managers/` | Acquisition and update orchestration |
| `core/`, `config/` | Shared models, configuration, and exception types |
| `assets/`, `calendar/`, `sessions/` | Asset contracts and trading-session behavior |
| `validation/`, `anomaly/`, `security/` | Data validation, anomaly checks, and input safety |
| `synthetic/` | Deterministic synthetic data models and registry integration |
| `cli/`, `cli_interface.py` | Command-line behavior exposed by the `ml4t-data` entry point |

Providers, storage, and futures contain directory-specific `AGENTS.md` files. Use those guides when
editing within those subsystems. Public API and usage details belong in `docs/api/`,
`docs/user-guide/`, and `docs/providers/` rather than here.

## Package boundaries

- Keep `src/ml4t/` as a namespace package without an `__init__.py` file.
- Re-export only supported public names from package `__init__.py` files. Test imports from an
  installed wheel when changing exports.
- Keep optional-provider imports isolated so the base package works without every provider extra.
- Route data paths through `core.config`; shared callers must not invent another environment-variable
  or default-path convention.
- Extend shared protocols and models deliberately. A contract change requires tests for adapters,
  orchestration, and storage consumers that rely on it.
- Add tests in the matching `tests/` area and use the repository's established markers for any
  non-default lane.

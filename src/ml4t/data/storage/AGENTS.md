# Storage subsystem guide

This directory owns the storage protocols and local persistence implementations used by data
managers and update workflows. General data-root selection lives in `ml4t.data.core.config`.

## Change locations

| File | Responsibility |
|---|---|
| `protocols.py`, `backend.py`, `async_base.py` | Storage interfaces and shared behavior |
| `hive.py`, `flat.py`, `chunked.py` | Concrete Parquet-backed layouts |
| `config.py` | Storage-specific configuration |
| `keys.py` | Dataset-key normalization and validation |
| `metadata_tracker.py` | Dataset metadata persistence |
| `migration.py`, `legacy_migration.py` | Supported layout and compatibility migrations |
| `data_profile.py` | Dataset and column profiling |

## Invariants

- Resolve defaults through the shared package data-root helpers. Explicit caller paths take
  precedence, and source code must not contain hard-coded home directories.
- Normalize and validate dataset keys through `keys.py`; storage backends must agree on key meaning.
- Preserve protocol behavior across concrete backends, including empty datasets, overwrite and
  append behavior, metadata updates, and error types.
- Keep migrations explicit and tested. Do not silently reinterpret an existing layout or remove a
  compatibility path without the repository's deprecation process.
- Keep storage tests local and deterministic. Network acquisition belongs to providers, not storage.
- Test resource cleanup for asynchronous or file-backed changes; the repository's ResourceWarning
  lane is required.

Run the matching tests in `tests/storage/` and the root-level `tests/test_*storage*.py` modules, then
run the root quality gates. User workflows and configuration examples belong in
`docs/user-guide/storage.md` and `docs/user-guide/configuration.md`.

# ml4t-data contributor guide

`ml4t-data` provides market-data acquisition, validation, storage, and update workflows for
machine-learning-for-trading applications. The public Python package is `ml4t.data`; package
metadata and supported Python versions are authoritative in `pyproject.toml`.

## Repository map

- `src/ml4t/data/` contains the package. Its `AGENTS.md` routes work within the source tree.
- `tests/` contains deterministic unit and contract tests. Tests needing credentials or live
  services belong in an explicitly marked integration lane.
- `docs/` contains the MkDocs site. Start with `docs/index.md`; use
  `docs/book-guide/index.md` for book chapter-to-API mappings.
- `examples/` contains maintained runnable examples and configuration samples.
- `scripts/` contains release and verification programs used by GitHub Actions.
- `.github/workflows/` defines pull-request, compatibility, documentation, security, and release
  automation.

The principal source areas are provider adapters and routing in `providers/`, persistence in
`storage/`, futures acquisition and contract construction in `futures/`, orchestration in
`data_manager.py`, `update_manager.py`, and `managers/`, and shared contracts in `core/`, `assets/`,
`config/`, and `validation/`. Providers, storage, and futures have more specific guides in their
directories.

## Public surface

Prefer supported imports from `ml4t.data` or documented subpackages. The synthetic provider is the
offline reference workflow:

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider(seed=42)
data = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
```

Provider-specific dependencies are optional. Keep imports usable without unrelated extras, and do
not require credentials or network access at import time. User-facing behavior belongs in the
documentation, not in agent guides.

## Change rules

- Preserve the PEP 420 namespace layout: do not add `src/ml4t/__init__.py`.
- Keep default tests deterministic and offline. Mark live, paid-tier, credentialed, and slow tests
  with the existing pytest markers.
- Resolve storage roots through the shared configuration helpers. Do not introduce hard-coded home
  directories.
- Use shared exceptions, retry, rate-limit, and provider contracts rather than adapter-specific
  variants of the same behavior.
- Treat `CHANGELOG.md` and generated version files as generated artifacts; do not edit them by hand.
- Keep release artifacts bound to one commit. Publishing is performed by the release workflow after
  qualification and documentation deployment, never by a local publish command.

## Verification

Install the complete development environment with `uv sync --locked --all-extras --all-groups`.
Run focused tests while editing, then run the repository gates:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check
uv run pytest tests -q -ra
uv run pytest tests -q -ra -W error::ResourceWarning
uv run mkdocs build --strict
uv build
actionlint .github/workflows/*.yml
pre-commit run --all-files
```

For packaging or release changes, also run the verification scripts used by
`.github/workflows/release.yml`. Never bypass hooks or weaken provider-specific checks to make a
shared gate pass.

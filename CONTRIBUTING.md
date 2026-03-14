# Contributing to ml4t-data

Thank you for your interest in contributing to ml4t-data. This guide covers everything you need to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ml4t/data.git ml4t-data
cd ml4t-data

# Install dependencies (requires uv)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Verify your setup
uv run pytest tests/ -q
```

**Requirements**: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Quality Gates

All changes must pass these checks before merging:

```bash
# Lint and format
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Type check
uv run ty check

# Tests (core suite, excludes integration/slow/paid)
uv run pytest tests/ -q

# Resource leak check
uv run pytest tests/ -q -W error::ResourceWarning

# Run everything at once
uv run pre-commit run --all-files
```

## Code Style

- **Formatter/linter**: [ruff](https://docs.astral.sh/ruff/) (100 char line length)
- **Type checker**: [ty](https://docs.astral.sh/ty/) (Astral) -- not mypy
- **Type hints**: required on all public functions
- **Docstrings**: Google style
- **Imports**: sorted by ruff (stdlib, third-party, local)
- **DataFrames**: Polars as primary, Pandas for compatibility

All style rules are configured in `pyproject.toml` under `[tool.ruff]` and `[tool.ty]`. Ruff auto-fixes most issues via pre-commit.

## Testing

### Running Tests

```bash
# Default: core tests only (fast, no API keys needed)
uv run pytest tests/ -q

# Verbose with full output
uv run pytest tests/ -v

# Run a specific file or test
uv run pytest tests/test_core_models.py -q
uv run pytest tests/test_core_models.py::test_function_name

# Run tests matching a pattern
uv run pytest -k "yahoo" -q

# Stop on first failure
uv run pytest tests/ -x
```

### Test Markers

Tests are organized by marker. The default `pytest` invocation skips slow, integration, and paid-tier tests.

| Marker | Description | Default |
|--------|-------------|---------|
| `slow` | Tests taking >10 seconds | Skipped |
| `integration` | Real API calls to external services | Skipped |
| `requires_api_key` | Needs a specific API key in env | Skipped |
| `paid_tier` | Requires paid API subscription | Skipped |
| `optional_dependency` | Requires optional heavy dependency | Skipped |
| `expensive` | High API cost | Skipped |

To run a specific marker lane:

```bash
# Integration tests (requires API keys in environment)
uv run pytest -m integration -q

# All tests, ignoring marker filters
uv run pytest -m "" -q
```

### Writing Tests

- Place tests in `tests/` mirroring the source layout
- Use `pytest` fixtures in `conftest.py` for shared setup
- Mark integration tests with `@pytest.mark.integration`
- Ensure OHLCV invariants hold: `high >= low`, `high >= open`, `high >= close`

## Pull Request Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** with tests. Ensure all quality gates pass:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/ -q
   ```

3. **Push and open a PR**:
   ```bash
   git push -u origin feat/my-feature
   gh pr create --title "feat: description" --body "..."
   ```

4. **CI runs automatically** on the PR. All checks must pass before merge:
   - Lint (ruff check + format)
   - Type Check (ty)
   - Test Core (Python 3.12, 3.13, 3.14)
   - Package Build

5. **Address review feedback**, then the PR is squash-merged into `main`.

## Commit Message Format

Use conventional commit prefixes:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance, dependencies, CI |

Examples:
```
feat: add Stooq provider for international equities
fix: handle empty response in Yahoo provider
test: add integration tests for Binance websocket
refactor: extract common HTTP retry logic to base class
```

## Adding a New Provider

Providers inherit from `BaseProvider` and implement three methods:

1. `name()` -- lowercase provider identifier
2. `_fetch_raw_data()` -- API call, returns raw response
3. `_transform_data()` -- converts raw data to standard Polars DataFrame

See `docs/contributing/creating-a-provider.md` for a full walkthrough and `src/ml4t/data/providers/` for existing implementations.

## Project Structure

```
src/ml4t/data/
├── providers/      # 20+ data source implementations
├── storage/        # Hive-partitioned Parquet backends
├── futures/        # CME/CBOE/ICE futures pipelines
├── validation/     # OHLCV schema and quality checks
├── config/         # Pydantic configuration models
├── cli/            # Click CLI interface
└── utils/          # Rate limiting, retry, formatting
```

## Getting Help

- [GitHub Issues](https://github.com/ml4t/data/issues) -- bug reports and feature requests
- [GitHub Discussions](https://github.com/ml4t/data/discussions) -- questions and ideas

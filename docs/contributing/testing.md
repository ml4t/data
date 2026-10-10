# Testing

The default pytest configuration runs the deterministic offline lane. Live API, paid-tier,
credentialed, expensive, and slow tests require an explicit marker or a dedicated workflow.

## Install the test environment

Use the lock file and install all provider extras and development groups:

```bash
uv sync --locked --all-extras --all-groups
```

## Run the default lane

```bash
uv run pytest tests -q -ra --cov=ml4t.data --cov-report=term-missing
```

`pyproject.toml` excludes `slow`, `paid_tier`, `integration`, and `requires_api_key` tests from this
command. It also enables strict marker validation and fails if package coverage falls below 85%.

Run the separate resource-leak lane before submitting a change that opens files, HTTP clients, or
other managed resources:

```bash
uv run pytest tests -q -ra -W error::ResourceWarning
```

## Run focused tests

Use a file, node ID, or expression while developing:

```bash
uv run pytest tests/test_storage_paths.py -q -ra
uv run pytest tests/test_storage_paths.py::test_legacy_env_warns -q -ra
uv run pytest tests -q -ra -k yahoo
uv run pytest tests -q -ra -x
```

The suite runs sequentially by default. Add `-n auto` only when the focused tests are known to be
safe under parallel execution.

## Test categories

| Marker | Contract | Default lane |
|---|---|---:|
| `integration` | Contacts an external service or exercises a cross-system boundary | Excluded |
| `requires_api_key` | Requires one or more provider credentials | Excluded |
| `paid_tier` | Can consume a metered or paid provider allowance | Excluded |
| `slow` | Unsuitable for the routine offline lane | Excluded |
| `expensive` | Has a material resource or provider cost | Not excluded automatically |
| `network_guard_probe` | Verifies that the offline network guard blocks network access | Included |

Mark a test according to what it does, not according to the directory containing it. A provider
test using `httpx.MockTransport`, a fixture, or a local file belongs in the default lane.

## Run live provider contracts

The `Provider Contract` workflow runs one selected provider with only that provider's credentials.
Use it for repository-level evidence. For local diagnosis, configure the named environment variable
and run only the intended node:

```bash
uv run pytest tests/integration/test_coingecko.py::TestCoinGeckoProvider::test_fetch_ohlcv_btc \
  -m integration -q -ra
```

Do not run every integration test against a live account. Several providers impose quotas, require
subscriptions, or charge for requests.

## Optional dependencies

The complete development environment includes every provider extra. To reproduce a minimal optional
dependency boundary, create an isolated environment with the relevant extra, for example:

```bash
uv sync --locked --extra databento --group test
```

Import-time behavior must not require unrelated extras or credentials.

## Coverage and failure diagnosis

The default lane reports missing lines for the `ml4t.data` package and enforces the same 85%
threshold as CI and the release workflow.

When a test fails only in the full suite, rerun it sequentially and preserve the order-dependent
reproduction. Do not remove the resource-warning lane, weaken markers, or replace a live contract
with a mock merely to obtain a green result.

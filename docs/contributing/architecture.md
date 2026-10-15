# Architecture

`ml4t-data` separates external data access from validation, persistence, and update orchestration.
The separation lets provider adapters vary without changing the canonical data passed to downstream
ML4T libraries.

## Package boundaries

| Area | Responsibility |
|---|---|
| `providers/` | External-service adapters, provider protocols, registry metadata, retry, rate-limit, and HTTP behavior |
| `validation/` and `anomaly/` | Schema, invariant, quality, and anomaly checks |
| `storage/` | Storage protocols, Parquet layouts, metadata, atomic publication, and migrations |
| `futures/` | Futures acquisition, calendars, rolls, and continuous-contract construction |
| `assets/`, `calendar/`, and `sessions/` | Asset identity and trading-session semantics |
| `core/` and `config/` | Shared exceptions, data models, and configuration resolution |
| `data_manager.py`, `update_manager.py`, and `managers/` | Coordination of provider reads, storage writes, and updates |
| `cli/` | Command-line entry points over the same public services |

The package uses a PEP 420 namespace. `src/ml4t/` intentionally has no `__init__.py`.

## Provider contracts

`providers/protocols.py` defines structural interfaces for OHLCV, factor, event, and asynchronous
providers. Code that only needs an interface should depend on the protocol rather than a concrete
adapter.

Most OHLCV adapters subclass `BaseProvider`. Its `fetch_ohlcv()` template owns the shared request
lifecycle:

1. validate the symbol and date inputs;
2. acquire the provider rate limit;
3. call the adapter's fetch and transformation method;
4. execute the call through the circuit breaker; and
5. normalize and validate the returned frame.

An adapter implements either `_fetch_and_transform_data()` or the pair `_fetch_raw_data()` and
`_transform_data()`. It does not override `fetch_ohlcv()` merely to duplicate rate limiting,
retries, logging, or schema validation.

The synchronous request helpers in `providers/base.py` map transport failures and HTTP status codes
to the shared exception hierarchy. Native asynchronous providers use the async session mixin and
must retain the same public error and cleanup behavior.

## Canonical OHLCV data

The validation mixin requires these columns in order:

| Column | Normalized type | Constraint |
|---|---|---|
| `timestamp` | timezone-aware `Datetime` in UTC | No duplicate timestamp and symbol pair |
| `symbol` | `String` | One normalized symbol per single-symbol request |
| `open`, `high`, `low`, `close` | `Float64` | Finite values and valid OHLC relationships |
| `volume` | `Float64` | Finite and non-negative |

Provider-specific columns may follow the canonical columns. A zero-column empty frame is normalized
to the canonical empty schema. Other malformed responses raise `DataValidationError` rather than
passing incomplete data downstream.

## Provider discovery

`providers/registry.py` is the static source for provider names, import targets, capabilities,
credentials, configuration requirements, and optional extras. It supports discovery without
constructing providers or contacting external services.

The registry and public imports serve different purposes:

- registry metadata answers whether an adapter is advertised and configured;
- `providers/__init__.py` exposes supported provider classes; and
- provider pages document data scope, credentials, optional dependencies, and service limits.

A new adapter normally updates all three surfaces.

## Storage and publication

Storage implementations satisfy the protocols in `storage/protocols.py`. Layout and metadata code
define how logical dataset keys map to files and records. Publication must keep data and metadata
consistent under concurrent writes and must reject unsafe paths and symlink escapes.

Callers should use the storage protocols or manager layer instead of depending on a backend's
private files. Migrations belong in `storage/migration.py` and require recovery tests for partially
completed work.

## Configuration and errors

Configuration helpers resolve public settings such as `ML4T_DATA_PATH`. Provider credentials and
service-specific settings remain scoped to the adapter that consumes them. Imports must stay usable
without credentials and without unrelated optional dependencies.

Public failures use the exception hierarchy in `core/exceptions.py`. Adapters preserve useful
provider context and exception chaining while classifying authentication, rate-limit, missing-data,
validation, and transient network failures consistently.

## Extension points

- Add an OHLCV source by following [Creating a provider](creating-a-provider.md).
- Implement a protocol directly when `BaseProvider` supplies behavior that does not apply to the
  data source.
- Add a storage backend behind the storage protocols and contract tests.
- Add validation without embedding provider-specific acquisition logic in the validation layer.

The generated [API reference](../api/index.md) is authoritative for signatures. Source-level
orientation and required verification commands are in the repository's nested `AGENTS.md` files.

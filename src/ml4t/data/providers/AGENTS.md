# Provider subsystem guide

This directory contains provider contracts, registry metadata, and adapters that translate external
data sources into the library's supported data models. Provider availability, credentials, and
service limitations are documented in `docs/providers/`.

## Shared components

| Path | Responsibility |
|---|---|
| `base.py`, `async_base.py` | Synchronous and asynchronous provider base behavior |
| `protocols.py` | Structural contracts used by callers and type checking |
| `registry.py` | Provider discovery, capabilities, aliases, and factory metadata |
| `mixins/` | Shared HTTP-session and rate-limit behavior |
| `synthetic.py`, `mock.py` | Deterministic offline and testing providers |
| Other top-level modules | One external or packaged data source per adapter |

## Adapter rules

- Importing `ml4t.data.providers` must not perform network calls or require credentials.
- Read credentials through the adapter's documented configuration. Never log secrets or place them
  in URLs when the service supports authenticated headers.
- Reuse shared sessions, rate limiting, retries, validation, and exception types. Do not add sleeps,
  retry loops, or error hierarchies local to one adapter without a contract reason.
- Normalize remote responses at the adapter boundary and test missing, empty, malformed, and
  rate-limited responses as well as the successful path.
- Keep optional dependencies isolated and fail with an actionable installation message when the
  selected adapter needs an unavailable extra.
- Register supported adapters through `registry.py` and update registry contract tests when
  capabilities or aliases change.
- Default tests mock external services. Live, credentialed, paid-tier, and slow checks use the
  existing pytest markers and must not enter the offline lane.

Start with the adapter's focused test module plus `tests/test_provider_registry_contract.py` and
`tests/test_provider_consistency.py`. For a new adapter, follow
`docs/contributing/creating-a-provider.md`, add its provider documentation, and then run the root
quality gates.

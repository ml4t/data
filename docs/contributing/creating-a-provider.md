# Create an OHLCV provider

This guide adds a synchronous OHLCV adapter that uses the shared provider lifecycle. Read
[Architecture](architecture.md) first if the source does not return symbol-based OHLCV data. Factor,
event, file-backed, and native asynchronous sources may need a different protocol or an existing
adapter as their model.

## Prerequisites

Before writing code, record the source's:

- authentication and optional-dependency requirements;
- supported instruments, frequencies, and history;
- start and end date semantics;
- request quotas and paid-tier boundaries;
- pagination and response schema;
- timestamp timezone; and
- status codes or response fields for authentication, rate limits, missing symbols, and transient
  failures.

Do not infer these contracts from an SDK method name. Link the provider documentation from the new
provider page and test the observed boundary behavior.

## Implement the adapter

Create `src/ml4t/data/providers/<name>.py`. For a JSON service, the two-step form keeps acquisition
and transformation separate:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import polars as pl

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.protocols import ProviderCapabilities


class ExampleProvider(BaseProvider):
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (30, 60.0)

    def __init__(self, api_key: str, **session_config: Any) -> None:
        self.api_key = api_key
        self.base_url = "https://api.example.test/v1"
        super().__init__(session_config=session_config)

    @property
    def name(self) -> str:
        return "example"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(requires_api_key=True, rate_limit=self.DEFAULT_RATE_LIMIT)

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str,
    ) -> list[dict[str, object]]:
        payload = self._request_json(
            f"{self.base_url}/bars",
            resource=symbol,
            params={
                "symbol": symbol,
                "start": start,
                "end": end,
                "frequency": frequency,
                "api_key": self.api_key,
            },
        )
        records = payload.get("data")
        if not isinstance(records, list):
            raise DataValidationError(self.name, "Response does not contain a data list")
        return records

    def _transform_data(
        self,
        raw_data: list[dict[str, object]],
        symbol: str,
    ) -> pl.DataFrame:
        if not raw_data:
            return self._create_empty_dataframe()
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime.fromtimestamp(float(row["time"]), tz=UTC) for row in raw_data
                ],
                "symbol": [symbol] * len(raw_data),
                "open": [row["open"] for row in raw_data],
                "high": [row["high"] for row in raw_data],
                "low": [row["low"] for row in raw_data],
                "close": [row["close"] for row in raw_data],
                "volume": [row["volume"] for row in raw_data],
            }
        )
```

Replace the placeholder endpoint and fields with the source's documented contract. Raise shared
exceptions from `ml4t.data.core.exceptions` for provider failures. Use `_request()` or
`_request_json()` so retryable transport, rate-limit, 404, and server failures retain the shared
classification. Override `_classify_error_response()` only when the service expresses those states
differently.

The returned frame must satisfy the canonical schema described in [Architecture](architecture.md).
Do not silently invent volume, timezone, symbol, or date-boundary semantics.

Always close providers in application code or use their context-manager support:

```python
with ExampleProvider(api_key="...") as provider:
    frame = provider.fetch_ohlcv("ABC", "2026-01-01", "2026-01-31", "daily")
```

## Register the provider

Add one `_spec()` entry to `src/ml4t/data/providers/registry.py`. Record:

- the stable provider name, module, and class;
- factual capabilities;
- required or optional credential environment variables;
- required configuration fields;
- the optional dependency extra, if one exists; and
- whether the manager-compatible OHLCV path applies.

Expose the supported class from `src/ml4t/data/providers/__init__.py`. Keep optional integrations
importable without installing unrelated extras.

## Write deterministic contract tests

Unit tests must exercise the public `fetch_ohlcv()` path with a transport fixture or local response
fixture. A useful provider test proves that:

- request parameters preserve inclusive or exclusive date semantics;
- timestamps are interpreted in the source timezone and normalized to UTC;
- pagination cannot silently truncate the requested range;
- malformed responses raise the expected shared exception;
- HTTP authentication, missing-symbol, rate-limit, and transient errors are classified correctly;
- the frame satisfies canonical schema, ordering, symbol, and OHLC invariants; and
- the HTTP client is closed.

`httpx.MockTransport` can drive the real request and transformation path without network access:

```python
import httpx


def test_example_provider_preserves_end_date() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["end"] == "2026-01-31"
        return httpx.Response(200, json={"data": []})

    with ExampleProvider(
        api_key="test",
        transport=httpx.MockTransport(respond),
    ) as provider:
        result = provider.fetch_ohlcv("ABC", "2026-01-01", "2026-01-31", "daily")

    assert result.is_empty()
```

Adapt the fixture to the provider's real response shape. Tests that contact the live service use the
`integration` marker and, when applicable, `requires_api_key`, `paid_tier`, `slow`, or `expensive`.
Add a narrowly bounded job to `.github/workflows/provider-contracts.yml` when ongoing live evidence
is necessary.

Run the focused offline tests first:

```bash
uv run pytest tests/test_<name>_provider.py -q -ra
uv run ruff check src/ml4t/data/providers/<name>.py tests/test_<name>_provider.py
uv run ty check
```

## Document the service boundary

Add `docs/providers/<name>.md` and include it in `mkdocs.yml`. State the source, supported data,
credentials, optional extra, quota or paid-service boundary, symbol and date semantics, and one
supported example. Update the appropriate asset-class source page without duplicating the provider
reference.

Do not add a provider to the README unless it changes the package-level installation or integration
boundary. The provider index and registry are the complete provider inventory.

## Verify the contribution

Before opening the pull request, run the full gates in the root `AGENTS.md`. The strict MkDocs build
must resolve the new page and navigation entry, and the package build must keep the base import
usable without the provider's optional dependency or credentials.

The pull request should link its owning issue and identify any compatibility or release-note impact.

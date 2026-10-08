"""Offline tests for the Polymarket US public-gateway provider.

Fixtures copy the layout of ``gateway.polymarket.us`` responses probed on 2026-10-08. One quirk
is load-bearing: ``outcomes`` labels are not always in ``outcomePrices`` order (the FOMC
"maintains" market lists ``["No","Yes"]`` with long-first prices ``["1","0"]``), so the
outcome must come from the long market side.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import DataValidationError, RateLimitError
from ml4t.data.providers.polymarket_us import (
    MARKET_SCHEMA,
    PRICE_HISTORY_SCHEMA,
    PolymarketUSProvider,
)


def _market(
    slug: str,
    long_price: str,
    *,
    outcomes: list[str] | None = None,
    outcome_prices: list[str] | None = None,
    status: str = "MARKET_STATUS_RESOLVED",
    long_label: str = "Yes",
    short_label: str = "No",
    **overrides: Any,
) -> dict[str, Any]:
    short_price = "" if long_price == "" else str(round(1 - float(long_price), 4)).rstrip("0")
    market = {
        "id": "11307",
        "question": "CPI year-over-year in April",
        "slug": slug,
        "title": "Exactly 3.0",
        "category": "macro",
        "marketType": "value",
        "startDate": "2026-04-07T14:10:44Z",
        "endDate": "2026-08-14T03:59:00Z",
        "closed": status == "MARKET_STATUS_RESOLVED",
        "status": status,
        "marketSides": [
            {"description": long_label, "price": long_price, "long": True},
            {"description": short_label, "price": short_price, "long": False},
        ],
        "outcomes": json.dumps(outcomes or [long_label, short_label]),
        "outcomePrices": json.dumps(outcome_prices or [long_price, short_price]),
        "feeCoefficient": 0.0695,
    }
    market.update(overrides)
    return market


class GatewayFake:
    """In-process gateway: market pages by offset, settlements and price history."""

    def __init__(self, markets: list[dict[str, Any]], page_cap: int = 500) -> None:
        self.markets = markets
        self.page_cap = page_cap
        self.settlements: dict[str, float] = {}
        self.history: dict[str, Any] = {"history": []}
        self.overrides: dict[str, list[httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        queued = self.overrides.get(path)
        if queued:
            return queued.pop(0)
        if path == "/v1/markets":
            offset = int(request.url.params.get("offset", "0"))
            limit = min(int(request.url.params.get("limit", "100")), self.page_cap)
            return httpx.Response(200, json={"markets": self.markets[offset : offset + limit]})
        if path.endswith("/settlement"):
            slug = path.split("/")[3]
            if slug not in self.settlements:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"slug": slug, "settlement": self.settlements[slug]})
        if path == "/v1/price-history":
            return httpx.Response(200, json=self.history)
        return httpx.Response(500, json={"error": path})


RESOLVED = [
    # Reversed labels: zipping outcomes with outcomePrices would say "No" won.
    _market(
        "rdc-usfed-fomc-2026-04-29-maintains",
        "1",
        outcomes=["No", "Yes"],
        outcome_prices=["1", "0"],
    ),
    _market(
        "cpic-uscpi-apr2026yoy-2026-05-12-2pt9pct",
        "0",
        outcomes=["No", "Yes"],
        outcome_prices=["0", "1"],
    ),
    _market("cpic-uscpi-july-yoy-2026-08-12-2pt7pct", "0.5"),
    _market(
        "aec-nfl-lac-ten-2025-11-02",
        "1",
        long_label="Chargers",
        short_label="Titans",
        category="sports",
    ),
    _market("usfed-fomc-2026-12-09-hike50", "0.04", status="MARKET_STATUS_OPEN"),
]


@pytest.fixture
def provider():
    provider = PolymarketUSProvider(rate_limit=(10_000, 1.0))
    yield provider
    provider.close()


@pytest.fixture
def gateway(provider, use_transport):
    fake = GatewayFake(RESOLVED)
    fake.requests = use_transport(provider, fake)
    return fake


class TestResolutionOutcome:
    def test_outcome_comes_from_the_long_market_side(self, provider, gateway):
        markets = provider.fetch_markets(closed=True)

        assert markets.schema == pl.Schema(MARKET_SCHEMA)
        rows = {row["ticker"]: row for row in markets.iter_rows(named=True)}
        maintains = rows["rdc-usfed-fomc-2026-04-29-maintains"]
        assert maintains["result"] == "yes"
        assert maintains["settlement_value"] == 1.0
        assert maintains["winning_outcome"] == "Yes"
        cpi = rows["cpic-uscpi-apr2026yoy-2026-05-12-2pt9pct"]
        assert cpi["result"] == "no"
        assert cpi["settlement_value"] == 0.0
        assert cpi["winning_outcome"] == "No"
        split = rows["cpic-uscpi-july-yoy-2026-08-12-2pt7pct"]
        assert split["result"] == "other"
        assert split["settlement_value"] == 0.5
        assert split["winning_outcome"] is None
        game = rows["aec-nfl-lac-ten-2025-11-02"]
        assert (game["result"], game["yes_outcome"], game["winning_outcome"]) == (
            "yes",
            "Chargers",
            "Chargers",
        )
        assert game["settlement_ts"] is None
        open_market = rows["usfed-fomc-2026-12-09-hike50"]
        assert open_market["result"] is None
        assert open_market["settlement_value"] is None

    def test_parsed_fields(self, provider, gateway):
        row = provider.fetch_markets().row(0, named=True)

        assert row["market_id"] == "11307"
        assert row["outcomes"] == ["No", "Yes"]
        assert row["outcome_prices"] == [1.0, 0.0]
        assert row["start_date"] == datetime(2026, 4, 7, 14, 10, 44, tzinfo=UTC)
        assert row["fee_coefficient"] == 0.0695

    def test_resolved_market_without_sides_falls_back_to_first_price(self, provider, gateway):
        bare = _market("bare-market", "1")
        bare.pop("marketSides")
        gateway.markets = [bare]

        row = provider.fetch_markets().row(0, named=True)

        assert row["result"] == "yes"
        assert row["yes_outcome"] == "Yes"

    def test_settlement_endpoint(self, provider, gateway):
        gateway.settlements["rdc-usfed-fomc-2026-04-29-maintains"] = 1

        assert provider.get_settlement("rdc-usfed-fomc-2026-04-29-maintains") == 1.0
        assert provider.get_settlement("usfed-fomc-2026-12-09-hike50") is None


class TestMarketPagination:
    def test_offsets_follow_returned_rows_until_an_empty_page(self, provider, use_transport):
        fake = GatewayFake(RESOLVED, page_cap=2)
        fake.requests = use_transport(provider, fake)

        markets = provider.fetch_markets(closed=True, categories=["macro", "sports"])

        assert len(markets) == len(RESOLVED)
        offsets = [r.url.params["offset"] for r in fake.requests]
        assert offsets == ["0", "2", "4", "5"]
        first = fake.requests[0].url.params
        assert first.get_list("categories") == ["macro", "sports"]
        assert first["closed"] == "true"
        assert first["limit"] == "500"

    def test_filters_are_encoded(self, provider, gateway):
        provider.fetch_markets(
            active=False,
            slugs=["a", "b"],
            end_date_min="2026-09-01",
            end_date_max=datetime(2026, 9, 2, tzinfo=UTC),
            page_size=50,
            max_pages=1,
        )

        params = gateway.requests[0].url.params
        assert params["active"] == "false"
        assert params.get_list("slug") == ["a", "b"]
        assert params["endDateMin"] == "2026-09-01T00:00:00Z"
        assert params["endDateMax"] == "2026-09-02T00:00:00Z"
        assert params["limit"] == "50"
        assert len(gateway.requests) == 1

    def test_empty_listing_returns_typed_empty_frame(self, provider, use_transport):
        use_transport(provider, GatewayFake([]))

        markets = provider.fetch_markets(closed=True)

        assert markets.is_empty()
        assert markets.schema == pl.Schema(MARKET_SCHEMA)


class TestMalformedAndFailing:
    def test_missing_markets_list_is_rejected(self, provider, gateway):
        gateway.overrides["/v1/markets"] = [httpx.Response(200, json={"items": []})]

        with pytest.raises(DataValidationError, match="no 'markets' list"):
            provider.fetch_markets()

    def test_bad_outcome_prices_are_rejected(self, provider, gateway):
        gateway.markets = [_market("broken", "1", outcomePrices="[1, oops")]

        with pytest.raises(DataValidationError, match="Malformed market record for broken"):
            provider.fetch_markets()

    def test_resolved_market_without_any_price_is_rejected(self, provider, gateway):
        bare = _market("no-price", "1", outcomePrices="[]")
        bare.pop("marketSides")
        gateway.markets = [bare]

        with pytest.raises(DataValidationError, match="without final prices"):
            provider.fetch_markets()

    def test_rate_limit_is_retried(self, provider, gateway, retry_sleeps):
        gateway.overrides["/v1/markets"] = [httpx.Response(429, headers={"Retry-After": "9"})]

        markets = provider.fetch_markets()

        assert len(markets) == len(RESOLVED)
        retry_sleeps.assert_called_once_with(9.0)

    def test_persistent_rate_limit_raises(self, provider, gateway):
        gateway.overrides["/v1/markets"] = [httpx.Response(429) for _ in range(3)]

        with pytest.raises(RateLimitError):
            provider.fetch_markets()

    def test_non_numeric_settlement_is_rejected(self, provider, gateway):
        gateway.overrides["/v1/markets/x/settlement"] = [
            httpx.Response(200, json={"slug": "x", "settlement": "n/a"})
        ]

        with pytest.raises(DataValidationError, match="Non-numeric settlement"):
            provider.get_settlement("x")


class TestPriceHistory:
    def test_full_history_uses_fixed_interval(self, provider, gateway):
        gateway.history = {
            "history": [
                {"timestamp": 1775606400, "longPrice": 0.15, "shortPrice": 0.99},
                {"timestamp": 1775520000, "longPrice": 0.5, "shortPrice": 0.99},
            ]
        }

        history = provider.fetch_price_history("cpic-uscpi-apr2026yoy-2026-05-12-3pt0pct")

        assert history.schema == pl.Schema(PRICE_HISTORY_SCHEMA)
        assert history["timestamp"].to_list() == [
            datetime(2026, 4, 7, tzinfo=UTC),
            datetime(2026, 4, 8, tzinfo=UTC),
        ]
        assert history["long_price"].to_list() == [0.5, 0.15]
        params = gateway.requests[0].url.params
        assert params["fixedInterval"] == "INTERVAL_ALL"
        assert params["fidelity"] == "180"

    def test_custom_range_is_split_into_24_hour_windows(self, provider, gateway):
        gateway.history = {"history": [{"timestamp": 1778457600, "longPrice": 0.01}]}

        history = provider.fetch_price_history("slug", start="2026-05-11", end="2026-05-12T06:00")

        windows = [
            (r.url.params["timestamp.startTimestamp"], r.url.params["timestamp.endTimestamp"])
            for r in gateway.requests
        ]
        assert windows == [("1778457600", "1778544000"), ("1778544000", "1778565600")]
        assert all(r.url.params["fidelity"] == "1" for r in gateway.requests)
        # The same point returned by both windows appears once; a missing side is null.
        assert len(history) == 1
        assert history["short_price"].to_list() == [None]

    def test_empty_history_returns_typed_empty_frame(self, provider, gateway):
        history = provider.fetch_price_history("slug")

        assert history.is_empty()
        assert history.schema == pl.Schema(PRICE_HISTORY_SCHEMA)

    def test_half_open_range_is_rejected(self, provider, gateway):
        with pytest.raises(DataValidationError, match="both start and end"):
            provider.fetch_price_history("slug", start="2026-05-11")

    def test_malformed_point_is_rejected(self, provider, gateway):
        gateway.history = {"history": [{"longPrice": 0.5}]}

        with pytest.raises(DataValidationError, match="Malformed price-history point"):
            provider.fetch_price_history("slug")

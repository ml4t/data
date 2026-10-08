"""Offline tests for Kalshi market enumeration, resolution outcomes and trade history.

The fixtures copy the field layout of live responses probed on 2026-10-08: live ``/markets``
and ``/historical/markets`` records use ``*_dollars``/``*_fp`` fields, and the archive tier
holds markets settled before ``/historical/cutoff``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    DataValidationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.kalshi import (
    CANDLE_SCHEMA,
    KALSHI_TRADE_SCHEMA,
    MARKET_SCHEMA,
    KalshiProvider,
)

CUTOFF = {
    "market_settled_ts": "2026-08-09T00:00:00Z",
    "trades_created_ts": "2026-08-09T00:00:00Z",
    "orders_updated_ts": "2026-09-24T00:00:00Z",
}
CUTOFF_SECONDS = int(datetime(2026, 8, 9, tzinfo=UTC).timestamp())


def _market(ticker: str, **overrides: Any) -> dict[str, Any]:
    market = {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "title": f"Market {ticker}",
        "market_type": "binary",
        "status": "finalized",
        "open_time": "2025-09-29T14:00:00Z",
        "close_time": "2026-09-16T17:59:00Z",
        "result": "yes",
        "settlement_value_dollars": "1.0000",
        "settlement_ts": "2026-09-16T18:08:08.647338Z",
        "expiration_value": "Hike 25bps",
        "last_price_dollars": "0.8800",
        "volume_fp": "18770408.08",
        "open_interest_fp": "10984000.31",
    }
    market.update(overrides)
    return market


def _trade(trade_id: str, created: str, **overrides: Any) -> dict[str, Any]:
    trade = {
        "trade_id": trade_id,
        "ticker": "KXFEDDECISION-26SEP-H25",
        "created_time": created,
        "count_fp": "450.75",
        "yes_price_dollars": "0.8800",
        "no_price_dollars": "0.1200",
        "taker_side": "yes",
        "taker_outcome_side": "yes",
        "taker_book_side": "bid",
        "is_block_trade": False,
    }
    trade.update(overrides)
    return trade


def _json(payload: Any, status: int = 200, headers: dict[str, str] | None = None):
    return httpx.Response(status, json=payload, headers=headers)


class KalshiFake:
    """In-process Kalshi API keyed by path; each path serves pages by cursor."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, dict[str, Any]]] = {}
        self.cutoff: dict[str, Any] | None = CUTOFF
        self.overrides: dict[str, list[httpx.Response]] = {}
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, dict[str, Any]] = {}

    def add_pages(self, path: str, key: str, pages: list[list[dict[str, Any]]]) -> None:
        served: dict[str, dict[str, Any]] = {}
        for index, items in enumerate(pages):
            cursor_in = "" if index == 0 else f"{path}-c{index}"
            cursor_out = f"{path}-c{index + 1}" if index + 1 < len(pages) else ""
            served[cursor_in] = {key: items, "cursor": cursor_out}
        self.pages[path] = served

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/trade-api/v2")
        queued = self.overrides.get(path)
        if queued:
            return queued.pop(0)
        if path == "/historical/cutoff":
            if self.cutoff is None:
                return _json({"error": "not found"}, status=404)
            return _json(self.cutoff)
        if path in self.routes:
            return _json(self.routes[path])
        if path not in self.pages:
            if path.startswith(("/series/", "/historical/markets/", "/markets/", "/events/")):
                return _json({"error": {"code": "not_found"}}, status=404)
            return _json({"error": f"unexpected path {path}"}, status=500)
        cursor = request.url.params.get("cursor", "")
        return _json(self.pages[path][cursor])


@pytest.fixture
def provider():
    provider = KalshiProvider(rate_limit=(10_000, 1.0))
    yield provider
    provider.close()


@pytest.fixture
def fake(provider, use_transport):
    venue = KalshiFake()
    venue.requests = use_transport(provider, venue)
    return venue


def _paths(fake: KalshiFake) -> list[str]:
    return [request.url.path.removeprefix("/trade-api/v2") for request in fake.requests]


class TestResolutionOutcome:
    """The normalized outcome is the reason the market listing exists."""

    def test_settled_markets_carry_normalized_outcome_from_both_tiers(self, provider, fake):
        fake.add_pages(
            "/markets",
            "markets",
            [
                [
                    _market("KXFEDDECISION-26SEP-H25"),
                    _market(
                        "KXFEDDECISION-26SEP-H0",
                        result="no",
                        settlement_value_dollars="0.0000",
                    ),
                ]
            ],
        )
        fake.add_pages(
            "/historical/markets",
            "markets",
            [
                [
                    _market(
                        "FED-23DEC-T5.25",
                        close_time="2023-12-13T18:55:00Z",
                        settlement_ts="2023-12-13T22:09:00.327987Z",
                    ),
                    _market(
                        "KXHIGHNY-24JAN01-T50",
                        result="scalar",
                        settlement_value_dollars="0.4000",
                    ),
                ]
            ],
        )

        markets = provider.fetch_markets(status="settled")

        assert markets.schema == pl.Schema(MARKET_SCHEMA)
        by_ticker = {row["ticker"]: row for row in markets.iter_rows(named=True)}
        assert by_ticker["KXFEDDECISION-26SEP-H25"]["result"] == "yes"
        assert by_ticker["KXFEDDECISION-26SEP-H25"]["settlement_value"] == 1.0
        assert by_ticker["KXFEDDECISION-26SEP-H25"]["settlement_ts"] == datetime(
            2026, 9, 16, 18, 8, 8, 647338, tzinfo=UTC
        )
        assert by_ticker["KXFEDDECISION-26SEP-H0"]["result"] == "no"
        assert by_ticker["KXFEDDECISION-26SEP-H0"]["settlement_value"] == 0.0
        assert by_ticker["FED-23DEC-T5.25"]["result"] == "yes"
        assert by_ticker["FED-23DEC-T5.25"]["source"] == "historical"
        assert by_ticker["KXHIGHNY-24JAN01-T50"]["result"] == "other"
        assert by_ticker["KXHIGHNY-24JAN01-T50"]["result_raw"] == "scalar"
        assert by_ticker["KXHIGHNY-24JAN01-T50"]["settlement_value"] == 0.4

    def test_unresolved_market_has_null_outcome(self, provider, fake):
        fake.add_pages(
            "/markets",
            "markets",
            [
                [
                    _market(
                        "KXFEDDECISION-26DEC-H0",
                        status="active",
                        result="",
                        settlement_value_dollars=None,
                        settlement_ts=None,
                    )
                ]
            ],
        )

        markets = provider.fetch_markets(status="open")

        row = markets.row(0, named=True)
        assert row["result"] is None
        assert row["settlement_value"] is None
        assert row["settlement_ts"] is None

    def test_legacy_unsuffixed_fields_are_normalized(self, provider, fake):
        """Integer cents (legacy API) and unsuffixed dollar strings both map to dollars."""
        legacy = _market("FED-21SEP-T0.25", result="no", settlement_value=0, last_price=3)
        for key in ("settlement_value_dollars", "last_price_dollars", "volume_fp"):
            legacy.pop(key)
        legacy["volume"] = 120
        archive_strings = _market("FED-22DEC-T4.00", settlement_value="1.0000")
        archive_strings.pop("settlement_value_dollars")
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages("/historical/markets", "markets", [[legacy, archive_strings]])

        markets = provider.fetch_markets()

        rows = {row["ticker"]: row for row in markets.iter_rows(named=True)}
        assert rows["FED-21SEP-T0.25"]["settlement_value"] == 0.0
        assert rows["FED-21SEP-T0.25"]["last_price"] == pytest.approx(0.03)
        assert rows["FED-21SEP-T0.25"]["volume"] == 120.0
        assert rows["FED-22DEC-T4.00"]["settlement_value"] == 1.0


class TestMarketPagination:
    def test_pages_live_then_archive_and_drops_archive_duplicates(self, provider, fake):
        fake.add_pages(
            "/markets",
            "markets",
            [[_market("KXA-1")], [_market("KXA-2")]],
        )
        fake.add_pages(
            "/historical/markets",
            "markets",
            [[_market("KXA-2", result="no")], [_market("OLD-1")]],
        )

        markets = provider.fetch_markets(status="settled", series_ticker="kxa")

        assert markets["ticker"].to_list() == ["KXA-1", "KXA-2", "OLD-1"]
        assert markets["source"].to_list() == ["live", "live", "historical"]
        # The live copy wins over the archive duplicate.
        assert markets.filter(pl.col("ticker") == "KXA-2")["result"].item() == "yes"
        live = [r for r in fake.requests if r.url.path.endswith("/trade-api/v2/markets")]
        assert [dict(r.url.params).get("cursor") for r in live] == [None, "/markets-c1"]
        assert all(r.url.params["status"] == "settled" for r in live)
        assert all(r.url.params["series_ticker"] == "KXA" for r in live)
        archive = [r for r in fake.requests if r.url.path.endswith("/historical/markets")]
        assert len(archive) == 2
        assert "status" not in archive[0].url.params
        assert archive[0].url.params["series_ticker"] == "KXA"

    def test_open_status_never_queries_archive(self, provider, fake):
        fake.add_pages("/markets", "markets", [[_market("KXA-1", status="active", result="")]])

        provider.fetch_markets(status="open")

        assert _paths(fake) == ["/markets"]

    def test_min_close_after_cutoff_skips_archive(self, provider, fake):
        fake.add_pages("/markets", "markets", [[_market("KXA-1")]])

        provider.fetch_markets(status="settled", min_close_ts="2026-09-01")

        assert _paths(fake) == ["/markets", "/historical/cutoff"]

    def test_close_window_is_enforced_on_both_tiers(self, provider, fake):
        fake.add_pages(
            "/markets",
            "markets",
            [
                [
                    _market("KXA-IN", close_time="2026-09-16T17:59:00Z"),
                    _market("KXA-LATE", close_time="2026-10-01T00:00:00Z"),
                ]
            ],
        )
        fake.add_pages(
            "/historical/markets",
            "markets",
            [
                [
                    _market("OLD-IN", close_time="2026-08-01T00:00:00Z"),
                    _market("OLD-EARLY", close_time="2023-12-13T18:55:00Z"),
                ]
            ],
        )

        markets = provider.fetch_markets(
            status="settled",
            min_close_ts=datetime(2026, 7, 1, tzinfo=UTC),
            max_close_ts="2026-09-30",
        )

        assert sorted(markets["ticker"].to_list()) == ["KXA-IN", "OLD-IN"]
        live_params = fake.requests[0].url.params
        # Kalshi rejects close-time filters combined with status=settled.
        assert "min_close_ts" not in live_params
        assert "max_close_ts" not in live_params

    def test_close_window_is_sent_to_live_tier_when_compatible(self, provider, fake):
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages("/historical/markets", "markets", [[]])

        provider.fetch_markets(min_close_ts=1_700_000_000, max_close_ts=1_800_000_000)

        live_params = fake.requests[0].url.params
        assert live_params["min_close_ts"] == "1700000000"
        assert live_params["max_close_ts"] == "1800000000"

    def test_archive_receives_one_filter_and_the_rest_is_client_side(self, provider, fake):
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages(
            "/historical/markets",
            "markets",
            [
                [
                    _market("FED-23DEC-T5.25", event_ticker="FED-23DEC"),
                    _market(
                        "KXMVECROSSCATEGORY-S1-X",
                        event_ticker="FED-23DEC",
                        mve_collection_ticker="KXMVECROSSCATEGORY",
                    ),
                ]
            ],
        )

        markets = provider.fetch_markets(
            series_ticker="KXFED", event_ticker="FED-23DEC", mve_filter="exclude"
        )

        assert markets["ticker"].to_list() == ["FED-23DEC-T5.25"]
        archive = next(r for r in fake.requests if r.url.path.endswith("/historical/markets"))
        assert dict(archive.url.params) == {"limit": "1000", "event_ticker": "FED-23DEC"}
        live = fake.requests[0].url.params
        assert live["mve_filter"] == "exclude"
        assert live["event_ticker"] == "FED-23DEC"

    def test_mve_filter_only_selects_combo_markets_on_both_tiers(self, provider, fake):
        combo = {"mve_collection_ticker": "KXMVECROSSCATEGORY"}
        fake.add_pages(
            "/markets",
            "markets",
            [[_market("KXMVECROSSCATEGORY-S2026B4-1", **combo), _market("KXA-1")]],
        )
        fake.add_pages(
            "/historical/markets",
            "markets",
            [[_market("KXMVESPORTS-S2026-2"), _market("OLD-1")]],
        )

        markets = provider.fetch_markets(status="settled", mve_filter="only")

        assert markets["ticker"].to_list() == [
            "KXMVECROSSCATEGORY-S2026B4-1",
            "KXMVESPORTS-S2026-2",
        ]
        assert fake.requests[0].url.params["mve_filter"] == "only"
        archive = next(r for r in fake.requests if r.url.path.endswith("/historical/markets"))
        # The archive rejects mve_filter=only (HTTP 400), so it is applied on the client.
        assert "mve_filter" not in archive.url.params

    def test_mve_filter_exclude_is_sent_to_archive_without_ticker_filter(self, provider, fake):
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages("/historical/markets", "markets", [[_market("OLD-1")]])

        provider.fetch_markets(mve_filter="exclude")

        archive = next(r for r in fake.requests if r.url.path.endswith("/historical/markets"))
        assert archive.url.params["mve_filter"] == "exclude"

    def test_invalid_mve_filter_is_rejected(self, provider, fake):
        with pytest.raises(DataValidationError, match="mve_filter"):
            provider.fetch_markets(mve_filter="include")
        assert fake.requests == []

    def test_series_previous_price_and_volume_columns(self, provider, fake):
        fake.add_pages(
            "/markets",
            "markets",
            [[_market("KXFEDDECISION-26SEP-H25", previous_price_dollars="0.8700")]],
        )
        fake.add_pages(
            "/historical/markets",
            "markets",
            [[_market("FED-23DEC-T5.25", event_ticker="FED-23DEC")]],
        )

        derived = provider.fetch_markets(status="settled")
        filtered = provider.fetch_markets(status="settled", series_ticker="kxfed")

        rows = {row["ticker"]: row for row in derived.iter_rows(named=True)}
        assert rows["KXFEDDECISION-26SEP-H25"]["series"] == "KXFEDDECISION"
        assert rows["KXFEDDECISION-26SEP-H25"]["previous_price"] == 0.87
        assert rows["KXFEDDECISION-26SEP-H25"]["volume"] == 18770408.08
        assert rows["FED-23DEC-T5.25"]["series"] == "FED"
        assert set(filtered["series"].to_list()) == {"KXFED"}

    def test_missing_cutoff_endpoint_queries_both_tiers(self, provider, fake):
        fake.cutoff = None
        fake.add_pages("/markets", "markets", [[_market("KXA-1")]])
        fake.add_pages(
            "/historical/markets",
            "markets",
            [[_market("OLD-1", close_time="2023-12-13T18:55:00Z")]],
        )

        markets = provider.fetch_markets(min_close_ts="2026-09-01", status="settled")

        assert provider.get_historical_cutoff() is None
        assert "/historical/markets" in _paths(fake)
        assert markets["ticker"].to_list() == ["KXA-1"]

    def test_empty_tiers_return_typed_empty_frame(self, provider, fake):
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages("/historical/markets", "markets", [[]])

        markets = provider.fetch_markets()

        assert markets.is_empty()
        assert markets.schema == pl.Schema(MARKET_SCHEMA)

    def test_max_pages_limits_each_tier(self, provider, fake):
        fake.add_pages("/markets", "markets", [[_market("KXA-1")], [_market("KXA-2")]])
        fake.add_pages("/historical/markets", "markets", [[_market("O-1")], [_market("O-2")]])

        markets = provider.fetch_markets(max_pages=1)

        assert markets["ticker"].to_list() == ["KXA-1", "O-1"]


class TestMalformedAndFailingResponses:
    def test_missing_markets_list_is_rejected(self, provider, fake):
        fake.overrides["/markets"] = [_json({"cursor": ""})]

        with pytest.raises(DataValidationError, match="no 'markets' list"):
            provider.fetch_markets(status="open")

    def test_non_json_body_is_rejected(self, provider, fake):
        fake.overrides["/markets"] = [httpx.Response(200, text="<html>maintenance</html>")]

        with pytest.raises(DataValidationError, match="Malformed JSON"):
            provider.fetch_markets(status="open")

    def test_bad_timestamp_is_rejected(self, provider, fake):
        fake.add_pages("/markets", "markets", [[_market("KXA-1", settlement_ts="yesterday")]])

        with pytest.raises(DataValidationError, match="Malformed market record for KXA-1"):
            provider.fetch_markets(status="open")

    def test_repeated_cursor_aborts(self, provider, fake):
        page = _json({"markets": [_market("KXA-1")], "cursor": "same"})
        fake.overrides["/markets"] = [page, _json({"markets": [], "cursor": "same"})]

        with pytest.raises(DataValidationError, match="cursor repeated"):
            provider.fetch_markets(status="open")

    def test_rate_limit_is_retried_with_server_delay(self, provider, fake, retry_sleeps):
        fake.overrides["/markets"] = [
            _json({"error": "too many"}, status=429, headers={"Retry-After": "7"})
        ]
        fake.add_pages("/markets", "markets", [[_market("KXA-1")]])

        markets = provider.fetch_markets(status="open")

        assert markets["ticker"].to_list() == ["KXA-1"]
        retry_sleeps.assert_called_once_with(7.0)

    def test_persistent_rate_limit_raises(self, provider, fake):
        fake.overrides["/markets"] = [_json({}, status=429) for _ in range(3)]

        with pytest.raises(RateLimitError):
            provider.fetch_markets(status="open")
        assert len(fake.requests) == 3

    def test_html_edge_403_is_retried(self, provider, fake):
        fake.overrides["/markets"] = [
            httpx.Response(403, text="<!DOCTYPE html><html>blocked</html>")
        ]
        fake.add_pages("/markets", "markets", [[_market("KXA-1")]])

        markets = provider.fetch_markets(status="open")

        assert len(markets) == 1

    def test_transport_failure_is_retried_then_raised(self, provider, use_transport):
        attempts: list[httpx.Request] = []

        def refuse(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            raise httpx.ConnectError("connection refused", request=request)

        use_transport(provider, refuse)

        with pytest.raises(NetworkError, match="connection refused"):
            provider.fetch_markets(status="open")
        assert len(attempts) == 3

    def test_client_error_is_not_retried(self, provider, fake):
        fake.overrides["/markets"] = [_json({"error": "bad request"}, status=400)]

        with pytest.raises(NetworkError, match="HTTP 400"):
            provider.fetch_markets(status="open")
        assert len(fake.requests) == 1


class TestTradeHistory:
    def test_trades_span_live_and_archive_with_normalized_columns(self, provider, fake):
        fake.add_pages(
            "/markets/trades",
            "trades",
            [
                [_trade("t3", "2026-09-16T17:58:57.123098Z")],
                [
                    _trade(
                        "t2",
                        "2026-08-10T12:00:00Z",
                        taker_side=None,
                        taker_outcome_side="no",
                        count_fp="62.92",
                        yes_price_dollars="0.8700",
                        is_block_trade=True,
                    )
                ],
            ],
        )
        legacy = {
            "trade_id": "t1",
            "ticker": "KXFEDDECISION-26SEP-H25",
            "created_time": "2026-08-01T09:30:00Z",
            "count": 5,
            "yes_price": 41,
            "taker_side": "no",
        }
        fake.add_pages(
            "/historical/trades",
            "trades",
            [[legacy, _trade("t2", "2026-08-10T12:00:00Z")]],
        )

        trades = provider.fetch_trades("kxfeddecision-26sep-h25")

        assert trades.schema == pl.Schema(KALSHI_TRADE_SCHEMA)
        assert trades["trade_id"].to_list() == ["t1", "t2", "t3"]
        first, second, third = trades.iter_rows(named=True)
        assert first["price"] == pytest.approx(0.41)
        assert first["count"] == 5.0
        assert first["source"] == "historical"
        assert first["timestamp"] == datetime(2026, 8, 1, 9, 30, tzinfo=UTC)
        assert second["taker_side"] == "no"
        assert second["taker_outcome_side"] == "no"
        assert second["taker_book_side"] == "bid"
        assert second["is_block_trade"] is True
        assert first["taker_book_side"] is None
        assert second["source"] == "live"
        assert third["price"] == 0.88
        assert third["count"] == 450.75
        assert all(r.url.params["ticker"] == "KXFEDDECISION-26SEP-H25" for r in fake.requests[1:])

    def test_min_ts_after_cutoff_skips_archive(self, provider, fake):
        fake.add_pages("/markets/trades", "trades", [[_trade("t1", "2026-09-16T17:00:00Z")]])

        trades = provider.fetch_trades("KXA-1", min_ts=CUTOFF_SECONDS)

        assert len(trades) == 1
        assert "/historical/trades" not in _paths(fake)
        live = next(r for r in fake.requests if r.url.path.endswith("/markets/trades"))
        assert live.url.params["min_ts"] == str(CUTOFF_SECONDS)

    def test_max_ts_before_cutoff_skips_live(self, provider, fake):
        fake.add_pages("/historical/trades", "trades", [[_trade("t1", "2022-12-12T20:57:45Z")]])

        trades = provider.fetch_trades("FED-22DEC-T4.00", max_ts="2023-01-01")

        assert len(trades) == 1
        assert "/markets/trades" not in _paths(fake)

    def test_no_trades_returns_typed_empty_frame(self, provider, fake):
        fake.add_pages("/markets/trades", "trades", [[]])
        fake.add_pages("/historical/trades", "trades", [[]])

        trades = provider.fetch_trades("KXA-1")

        assert trades.is_empty()
        assert trades.schema == pl.Schema(KALSHI_TRADE_SCHEMA)

    def test_trade_without_price_is_rejected(self, provider, fake):
        broken = _trade("t1", "2026-09-16T17:00:00Z")
        broken.pop("yes_price_dollars")
        fake.add_pages("/markets/trades", "trades", [[broken]])

        with pytest.raises(DataValidationError, match="missing"):
            provider.fetch_trades("KXA-1", include_historical=False)

    def test_trades_rate_limit_is_retried(self, provider, fake, retry_sleeps):
        fake.overrides["/markets/trades"] = [_json({}, status=429)]
        fake.add_pages("/markets/trades", "trades", [[_trade("t1", "2026-09-16T17:00:00Z")]])

        trades = provider.fetch_trades("KXA-1", include_historical=False)

        assert len(trades) == 1
        assert retry_sleeps.call_count == 1


def test_existing_list_markets_contract_is_unchanged(provider, fake):
    """list_markets keeps returning one page of raw dictionaries."""
    fake.add_pages("/markets", "markets", [[_market("KXA-1")], [_market("KXA-2")]])

    markets = provider.list_markets(status="settled", limit=5)

    assert [m["ticker"] for m in markets] == ["KXA-1"]
    assert json.loads(json.dumps(markets))[0]["result"] == "yes"


# Archive candle as served by /historical/markets/{ticker}/candlesticks (probed 2026-10-08):
# unsuffixed dollar strings, null trade prices when nothing traded.
ARCHIVE_CANDLE = {
    "end_period_ts": 1702098000,
    "open_interest": "7758.00",
    "price": {
        "close": "0.9900",
        "high": "0.9900",
        "low": "0.9800",
        "mean": "0.9850",
        "open": "0.9800",
        "previous": "0.9800",
    },
    "volume": "114.00",
    "yes_ask": {"close": "1.0000", "high": "1.0000", "low": "0.9900", "open": "1.0000"},
    "yes_bid": {"close": "0.9900", "high": "0.9900", "low": "0.9800", "open": "0.9900"},
}
QUIET_ARCHIVE_CANDLE = {
    **ARCHIVE_CANDLE,
    "end_period_ts": 1702184400,
    "price": {
        "close": None,
        "high": None,
        "low": None,
        "mean": None,
        "open": None,
        "previous": "0.9900",
    },
    "volume": "0.00",
}
# Live candle as served by /series/{series}/markets/{ticker}/candlesticks.
LIVE_CANDLE = {
    "end_period_ts": 1789581600,
    "open_interest_fp": "10984000.31",
    "price": {
        "close_dollars": "0.8800",
        "high_dollars": "0.8800",
        "low_dollars": "0.8400",
        "mean_dollars": "0.8610",
        "open_dollars": "0.8800",
        "previous_dollars": "0.8700",
    },
    "volume_fp": "577503.04",
    "yes_ask": {"close_dollars": "1.0000", "open_dollars": "0.8800"},
    "yes_bid": {"close_dollars": "0.0000", "open_dollars": "0.8700"},
}
ARCHIVE_PATH = "/historical/markets/FED-23DEC-T5.25/candlesticks"


class TestCandles:
    def test_archive_candles_are_dollars_not_cents(self, provider, fake):
        fake.routes[ARCHIVE_PATH] = {"candlesticks": [ARCHIVE_CANDLE, QUIET_ARCHIVE_CANDLE]}

        candles = provider.fetch_candles(
            "FED-23DEC-T5.25", "2023-12-08", "2023-12-11", period="1h", tier="historical"
        )

        assert candles.schema == pl.Schema(CANDLE_SCHEMA)
        first, quiet = candles.iter_rows(named=True)
        assert (first["open"], first["close"], first["mean"]) == (0.98, 0.99, 0.985)
        assert (first["yes_bid_close"], first["yes_ask_close"]) == (0.99, 1.0)
        assert (first["volume"], first["open_interest"]) == (114.0, 7758.0)
        assert first["timestamp"] == datetime(2023, 12, 9, 5, tzinfo=UTC)
        assert quiet["close"] is None
        assert quiet["previous"] == 0.99

    def test_fetch_ohlcv_keeps_archive_candles_in_dollars(self, provider, fake):
        fake.routes[ARCHIVE_PATH] = {"candlesticks": [ARCHIVE_CANDLE]}

        bars = provider.fetch_ohlcv("FED-23DEC-T5.25", "2023-12-08", "2023-12-10", "hourly")

        assert bars["close"].to_list() == [0.99]

    def test_archived_market_is_served_by_the_archive_endpoint(self, provider, fake):
        fake.routes[ARCHIVE_PATH] = {"candlesticks": [ARCHIVE_CANDLE]}

        candles = provider.fetch_candles("FED-23DEC-T5.25", 1702000000, 1702600000)

        assert candles["source"].to_list() == ["historical"]
        assert [r.url.path.removeprefix("/trade-api/v2") for r in fake.requests] == [
            "/series/FED/markets/FED-23DEC-T5.25/candlesticks",
            ARCHIVE_PATH,
        ]

    def test_live_market_with_series_unlike_its_prefix_is_resolved(self, provider, fake):
        live_path = "/series/KXFED/markets/FED-26DEC-T4.00/candlesticks"
        fake.routes[live_path] = {"candlesticks": [LIVE_CANDLE]}
        fake.routes["/markets/FED-26DEC-T4.00"] = {"market": {"event_ticker": "FED-26DEC"}}
        fake.routes["/events/FED-26DEC"] = {"event": {"series_ticker": "KXFED"}}

        candles = provider.fetch_candles("FED-26DEC-T4.00", 1789500000, 1789600000)

        assert candles["source"].to_list() == ["live"]
        assert candles["close"].to_list() == [0.88]
        assert fake.requests[-1].url.path.endswith(live_path)

    def test_explicit_series_goes_straight_to_the_live_endpoint(self, provider, fake):
        live_path = "/series/KXFED/markets/FED-26DEC-T4.00/candlesticks"
        fake.routes[live_path] = {"candlesticks": [LIVE_CANDLE]}

        candles = provider.fetch_candles(
            "FED-26DEC-T4.00", 1789500000, 1789600000, series_ticker="kxfed"
        )

        assert len(fake.requests) == 1
        row = candles.row(0, named=True)
        assert (row["close"], row["previous"], row["yes_bid_close"]) == (0.88, 0.87, 0.0)
        assert (row["volume"], row["open_interest"]) == (577503.04, 10984000.31)

    def test_legacy_integer_cents_are_scaled(self, provider, fake):
        live_path = "/series/KXINFL/markets/KXINFL-25JAN/candlesticks"
        fake.routes[live_path] = {
            "candlesticks": [
                {
                    "end_period_ts": 1736000000,
                    "price": {"open": 45, "high": 48, "low": 44, "close": 47},
                    "volume": 10,
                }
            ]
        }

        candles = provider.fetch_candles("KXINFL-25JAN", 1735900000, 1736100000, period="1d")
        bars = provider.fetch_ohlcv("KXINFL-25JAN", "2025-01-03", "2025-01-05", "daily")

        assert candles["close"].to_list() == [0.47]
        assert bars["close"].to_list() == [0.47]

    def test_wrong_tier_is_not_masked(self, provider, fake):
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_candles("FED-23DEC-T5.25", 1702000000, 1702600000, tier="live")
        assert len(fake.requests) == 1

    def test_unknown_market_raises(self, provider, fake):
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_candles("NOPE-26DEC-X", 1702000000, 1702600000)

    def test_long_ranges_are_split_under_the_candle_limit(self, provider, fake):
        fake.routes[ARCHIVE_PATH] = {"candlesticks": [ARCHIVE_CANDLE]}
        start = 1_700_000_000
        end = start + 6000 * 3600

        candles = provider.fetch_candles("FED-23DEC-T5.25", start, end, tier="historical")

        windows = [
            (int(r.url.params["start_ts"]), int(r.url.params["end_ts"])) for r in fake.requests
        ]
        assert windows == [(start, start + 4999 * 3600), (start + 4999 * 3600, end)]
        assert all(r.url.params["period_interval"] == "60" for r in fake.requests)
        assert len(candles) == 1

    def test_empty_and_malformed_candles(self, provider, fake):
        fake.routes[ARCHIVE_PATH] = {"candlesticks": []}
        empty = provider.fetch_candles("FED-23DEC-T5.25", 1, 2, tier="historical")
        assert empty.is_empty()
        assert empty.schema == pl.Schema(CANDLE_SCHEMA)

        fake.routes[ARCHIVE_PATH] = {"candlesticks": [{"price": {"close": "0.5"}}]}
        with pytest.raises(DataValidationError, match="Malformed candlestick"):
            provider.fetch_candles("FED-23DEC-T5.25", 1, 2, tier="historical")

    def test_invalid_period_and_tier(self, provider, fake):
        with pytest.raises(DataValidationError, match="Unsupported frequency"):
            provider.fetch_candles("FED-23DEC-T5.25", 1, 2, period=5)
        with pytest.raises(DataValidationError, match="tier"):
            provider.fetch_candles("FED-23DEC-T5.25", 1, 2, tier="archive")

    def test_rate_limited_candles_are_retried(self, provider, fake, retry_sleeps):
        fake.overrides[ARCHIVE_PATH] = [_json({}, status=429, headers={"Retry-After": "5"})]
        fake.routes[ARCHIVE_PATH] = {"candlesticks": [ARCHIVE_CANDLE]}

        candles = provider.fetch_candles("FED-23DEC-T5.25", 1, 2, tier="historical")

        assert len(candles) == 1
        retry_sleeps.assert_called_once_with(5.0)


class TestMarketFrames:
    def test_frames_stream_in_chunks(self, provider, fake):
        fake.add_pages("/markets", "markets", [[_market("KXA-1"), _market("KXA-2")]])
        fake.add_pages("/historical/markets", "markets", [[_market("OLD-1")]])

        frames = list(provider.iter_market_frames(chunk_size=2))

        assert [len(frame) for frame in frames] == [2, 1]
        assert all(frame.schema == pl.Schema(MARKET_SCHEMA) for frame in frames)
        assert pl.concat(frames)["ticker"].to_list() == ["KXA-1", "KXA-2", "OLD-1"]

    def test_no_matches_yield_nothing(self, provider, fake):
        fake.add_pages("/markets", "markets", [[]])
        fake.add_pages("/historical/markets", "markets", [[]])

        assert list(provider.iter_market_frames()) == []

    def test_chunk_size_must_be_positive(self, provider, fake):
        with pytest.raises(DataValidationError, match="chunk_size"):
            list(provider.iter_market_frames(chunk_size=0))

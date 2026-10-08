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

from ml4t.data.core.exceptions import DataValidationError, NetworkError, RateLimitError
from ml4t.data.providers.kalshi import KALSHI_TRADE_SCHEMA, MARKET_SCHEMA, KalshiProvider

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
        if path not in self.pages:
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
            series_ticker="KXFED", event_ticker="FED-23DEC", exclude_multivariate=True
        )

        assert markets["ticker"].to_list() == ["FED-23DEC-T5.25"]
        archive = next(r for r in fake.requests if r.url.path.endswith("/historical/markets"))
        assert dict(archive.url.params) == {"limit": "1000", "event_ticker": "FED-23DEC"}
        live = fake.requests[0].url.params
        assert live["mve_filter"] == "exclude"
        assert live["event_ticker"] == "FED-23DEC"

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
        assert second["is_block_trade"] is True
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

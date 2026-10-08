"""Offline tests for Polymarket resolved-market listing, price history and trades.

The fixtures under ``fixtures/polymarket/`` are responses recorded on 2026-10-08 from the
Gamma keyset listing, the CLOB price history and the data-API trades endpoint, with long text
fields trimmed. The fake servers below implement the behaviour verified against the live
endpoints that day: ``/markets/keyset`` pages by ``after_cursor`` (any other name returns page 1
again), ``/prices-history`` rejects spans over 15 days with HTTP 400 and returns an empty
history for ``interval`` ranges of resolved markets, and ``/trades`` returns trades newest first
with inclusive ``start``/``end`` bounds and rejects ``offset`` above its cap.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import DataValidationError, NetworkError
from ml4t.data.providers.polymarket import (
    CANDLE_SCHEMA,
    MARKET_SCHEMA,
    POLYMARKET_TRADE_SCHEMA,
    PolymarketProvider,
)

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"
KEYSET = json.loads((FIXTURES / "gamma_keyset_closed.json").read_text())["pages"]
KEYSET_ACTIVE = json.loads((FIXTURES / "gamma_keyset_active.json").read_text())["pages"]
TRADES = json.loads((FIXTURES / "data_api_trades.json").read_text())["trades"]
PRICES = json.loads((FIXTURES / "clob_prices_history.json").read_text())
CURSOR = KEYSET[0]["next_cursor"]
CONDITION_ID = "0x8ee2f1640386310eb5e7ffa596ba9335f2d324e303d21b0dfea6998874445791"
TOKEN_ID = json.loads(KEYSET[1]["markets"][0]["clobTokenIds"])[0]  # YES token of 516719


class CallLimitExceeded(AssertionError):
    """Raised by a fake server when a client keeps requesting past any sane page count."""


def _keyset_server(pages: list[dict[str, Any]], max_calls: int = 20):
    """Serve ``pages`` by ``after_cursor``; unknown or missing cursors get page 1, like Gamma."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] > max_calls:
            raise CallLimitExceeded(f"more than {max_calls} keyset requests")
        assert request.url.path == "/markets/keyset"
        cursor = request.url.params.get("after_cursor")
        page = pages[1] if cursor == CURSOR and len(pages) > 1 else pages[0]
        return httpx.Response(200, json=page)

    return handler


@pytest.fixture
def provider():
    provider = PolymarketProvider(rate_limit=(10_000, 1.0))
    yield provider
    provider.close()


def _by_id(frame: pl.DataFrame, market_id: str) -> dict[str, Any]:
    return frame.filter(pl.col("market_id") == market_id).row(0, named=True)


class TestMarketListing:
    def test_keyset_pagination_passes_cursor_as_after_cursor(self, provider, use_transport):
        """Prevents: sending the cursor under another name, which Gamma ignores (page 1 again)."""
        requests = use_transport(provider, _keyset_server(KEYSET))

        markets = provider.fetch_markets()

        assert markets.height == 7
        assert markets["market_id"].n_unique() == 7
        assert requests[0].url.params.get("after_cursor") is None
        assert requests[1].url.params["after_cursor"] == CURSOR
        assert len(requests) == 2

    def test_repeated_cursor_raises_instead_of_looping(self, provider, use_transport):
        """Prevents: an endless loop when the server keeps answering page 1 and its cursor."""
        stuck = [KEYSET[0]]  # the server ignores the cursor and always returns page 1
        use_transport(provider, _keyset_server(stuck, max_calls=5))

        with pytest.raises(DataValidationError, match="cursor repeated"):
            list(provider.iter_markets())

    def test_filters_are_sent_and_reapplied_on_the_client(self, provider, use_transport):
        """Prevents: rows outside the volume or end-date window leaking if Gamma ignores them."""
        requests = use_transport(provider, _keyset_server(KEYSET))

        markets = provider.fetch_markets(
            min_volume=1_000_000, end_date_min="2025-01-01", end_date_max="2025-12-31T23:59:59Z"
        )

        params = requests[0].url.params
        assert params["closed"] == "true"
        assert params["include_tag"] == "true"
        assert float(params["volume_num_min"]) == 1_000_000
        assert params["end_date_min"] == "2025-01-01T00:00:00Z"
        assert params["end_date_max"] == "2025-12-31T23:59:59Z"
        # The fake server ignores the filters; the client must drop the 2024 and low-volume rows.
        assert sorted(markets["market_id"].to_list()) == ["516719", "516723", "516926"]
        assert (markets["volume"] >= 1_000_000).all()

    def test_outcomes_map_by_position_not_label(self, provider, use_transport):
        """Prevents: team-name markets losing their result, and 50/50 resolutions not void."""
        use_transport(provider, _keyset_server(KEYSET))

        markets = provider.fetch_markets()

        party = _by_id(markets, "240613")  # ["Democratic", "Republican"], Republican won
        assert (party["yes_outcome"], party["no_outcome"]) == ("Democratic", "Republican")
        assert party["result"] == "no"
        assert party["settlement_value"] == 0.0
        baby = _by_id(markets, "501517")  # ["Boy", "Girl"], Boy won
        assert (baby["yes_outcome"], baby["result"], baby["settlement_value"]) == (
            "Boy",
            "yes",
            1.0,
        )
        boden = _by_id(markets, "500853")  # Trump vs. Boden, resolved 50/50
        assert boden["result"] == "void"
        assert boden["settlement_value"] == 0.5
        assert boden["outcome_prices"] == [0.5, 0.5]
        ceasefire = _by_id(markets, "516719")
        tokens = json.loads(KEYSET[1]["markets"][0]["clobTokenIds"])
        assert (ceasefire["yes_token_id"], ceasefire["no_token_id"]) == tuple(tokens)
        assert ceasefire["result"] == "no"

    def test_settlement_time_is_closed_time(self, provider, use_transport):
        """Prevents: settlement time read from the scheduled endDate instead of closedTime."""
        use_transport(provider, _keyset_server(KEYSET))

        markets = provider.fetch_markets()

        nomination = _by_id(markets, "240379")  # closed in July, scheduled end in September
        assert nomination["settlement_ts"] == datetime(2024, 7, 18, 0, 10, 37, 491895, tzinfo=UTC)
        assert nomination["close_time"] == datetime(2024, 9, 10, tzinfo=UTC)
        assert markets["settlement_ts"].null_count() == 0
        assert markets.schema["settlement_ts"] == MARKET_SCHEMA["settlement_ts"]

    def test_open_market_is_unresolved(self, provider, use_transport):
        """Prevents: last-trade prices of a market still trading being read as a resolution."""
        requests = use_transport(provider, _keyset_server(KEYSET_ACTIVE))

        markets = provider.fetch_markets(closed=False)

        assert requests[0].url.params["closed"] == "false"
        row = markets.row(0, named=True)
        assert row["closed"] is False
        assert row["outcome_prices"] == [0.0265, 0.9735]
        assert (row["result"], row["settlement_value"], row["settlement_ts"]) == (None, None, None)

    def test_tags_fees_and_neg_risk_columns(self, provider, use_transport):
        """Prevents: topic tags or fee terms going missing (category is null on recent markets)."""
        use_transport(provider, _keyset_server(KEYSET))

        markets = provider.fetch_markets()

        fees = _by_id(markets, "516926")
        assert fees["fees_enabled"] is True
        assert fees["fee_type"] == "finance_prices_fees"
        assert (fees["maker_base_fee"], fees["taker_base_fee"]) == (1000.0, 1000.0)
        assert json.loads(fees["fee_schedule"])["rate"] == 0.04
        assert "Crypto" in fees["tags"]
        assert fees["category"] is None
        fed = _by_id(markets, "516723")
        assert fed["neg_risk"] is True
        assert fed["neg_risk_market_id"].startswith("0x")
        assert fed["event_slug"] == "how-many-fed-rate-cuts-in-2025"
        assert _by_id(markets, "240379")["category"] == "US-current-affairs"

    def test_market_frames_keep_one_schema_across_chunks(self, provider, use_transport):
        """Prevents: per-chunk schema inference (all-null columns as Null) breaking concat."""
        use_transport(provider, _keyset_server(KEYSET))

        frames = list(provider.iter_market_frames(chunk_size=3))

        assert [frame.height for frame in frames] == [3, 3, 1]
        assert all(frame.schema == pl.Schema(MARKET_SCHEMA) for frame in frames)
        assert pl.concat(frames).height == 7

    def test_empty_listing_returns_typed_empty_frame(self, provider, use_transport):
        """Prevents: an untyped empty frame when no market matches the filters."""
        empty = [{"$schema": KEYSET[0]["$schema"], "markets": []}]
        use_transport(provider, _keyset_server(empty))

        markets = provider.fetch_markets(min_volume=1e12)

        assert markets.is_empty()
        assert markets.schema == pl.Schema(MARKET_SCHEMA)


def _prices_server(history: list[dict[str, Any]], max_window: int = 15 * 86_400):
    """Serve the recorded history filtered to the window, enforcing the 15-day span limit."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "interval" in params:
            return httpx.Response(200, json={"history": []})  # resolved market, interval range
        start, end = int(params["startTs"]), int(params["endTs"])
        if end - start > max_window:
            return httpx.Response(400, json=PRICES["too_long_window_response"])
        points = [point for point in history if start <= point["t"] <= end]
        return httpx.Response(200, json={"history": points})

    return handler


class TestCandles:
    def test_long_ranges_are_split_into_15_day_windows(self, provider, use_transport):
        """Prevents: one request over 15 days (HTTP 400) or interval=max (empty when resolved)."""
        history = PRICES["response"]["history"]
        requests = use_transport(provider, _prices_server(history))
        start, end = history[0]["t"] - 20 * 86_400, history[-1]["t"] + 20 * 86_400

        candles = provider.fetch_candles(TOKEN_ID, start, end, fidelity_minutes=60)

        assert candles.height == len(history)
        windows = [(int(r.url.params["startTs"]), int(r.url.params["endTs"])) for r in requests]
        assert len(windows) == 3  # 41 days
        assert windows[0][0] == start and windows[-1][1] == end
        assert all(b - a <= 15 * 86_400 for a, b in windows)
        assert all(nxt[0] == prev[1] + 1 for prev, nxt in zip(windows, windows[1:]))
        assert all(r.url.params["fidelity"] == "60" for r in requests)
        assert all("interval" not in r.url.params for r in requests)

    def test_recorded_history_parses_to_sorted_utc_prices(self, provider, use_transport):
        """Prevents: seconds read as milliseconds, naive timestamps or points outside the range."""
        history = PRICES["response"]["history"]
        use_transport(provider, _prices_server(history))
        start, end = history[2]["t"], history[5]["t"]

        candles = provider.fetch_candles(TOKEN_ID, start, end)

        assert candles.schema == pl.Schema(CANDLE_SCHEMA)
        assert candles["timestamp"].to_list() == [
            datetime.fromtimestamp(point["t"], UTC) for point in history[2:6]
        ]
        assert candles["price"].to_list() == [point["p"] for point in history[2:6]]
        assert set(candles["token_id"].to_list()) == {TOKEN_ID}

    def test_window_too_long_surfaces_as_error(self, provider, use_transport, monkeypatch):
        """Prevents: a widened window constant silently passing; the server's 400 must surface."""
        use_transport(provider, _prices_server(PRICES["response"]["history"]))
        monkeypatch.setattr(provider, "PRICE_HISTORY_MAX_WINDOW_SECONDS", 16 * 86_400)

        with pytest.raises(NetworkError, match="interval is too long"):
            provider.fetch_candles(TOKEN_ID, 0, 30 * 86_400)

    def test_no_history_returns_typed_empty_frame(self, provider, use_transport):
        """Prevents: an untyped frame for unknown tokens (the CLOB answers 200 and [])."""
        use_transport(provider, _prices_server([]))

        candles = provider.fetch_candles(TOKEN_ID, "2025-06-01", "2025-06-02")

        assert candles.is_empty()
        assert candles.schema == pl.Schema(CANDLE_SCHEMA)

    def test_malformed_point_raises(self, provider, use_transport):
        """Prevents: a point without a price being stored as a valid sample."""
        use_transport(provider, lambda _: httpx.Response(200, json={"history": [{"t": 1}]}))

        with pytest.raises(DataValidationError, match="Malformed price point"):
            provider.fetch_candles(TOKEN_ID, 0, 10)


def _trades_server(trades: list[dict[str, Any]], max_offset: int, ignore_offset: bool = False):
    """Serve trades newest first with inclusive start/end and an offset cap, like the data API."""
    ordered = sorted(trades, key=lambda trade: -trade["timestamp"])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] > 50:
            raise CallLimitExceeded("more than 50 trade requests")
        params = request.url.params
        assert params["market"] == CONDITION_ID
        limit, offset = int(params["limit"]), int(params.get("offset", 0))
        if offset > max_offset:
            return httpx.Response(
                400, json={"error": f"max historical trades offset of {max_offset} exceeded"}
            )
        rows = [
            trade
            for trade in ordered
            if ("start" not in params or trade["timestamp"] >= int(params["start"]))
            and ("end" not in params or trade["timestamp"] <= int(params["end"]))
        ]
        first = 0 if ignore_offset else offset
        return httpx.Response(200, json=rows[first : first + limit])

    return handler


class TestTrades:
    def test_yes_price_and_taker_side_from_outcome_and_side(self, provider, use_transport):
        """Prevents: a NO-token price reported as the YES price, or the taker side inverted."""
        use_transport(provider, _trades_server(TRADES, max_offset=10_000))

        trades = provider.fetch_trades(CONDITION_ID)

        assert trades.schema == pl.Schema(POLYMARKET_TRADE_SCHEMA)
        assert trades.height == len(TRADES)
        sell_no = trades.filter((pl.col("outcome_index") == 1) & (pl.col("side") == "SELL"))
        assert not sell_no.is_empty()
        row = sell_no.row(0, named=True)
        assert row["price"] == pytest.approx(1.0 - row["outcome_price"])
        assert row["taker_side"] == "yes"
        buy_yes = trades.filter((pl.col("outcome_index") == 0) & (pl.col("side") == "BUY"))
        assert (buy_yes["price"] == buy_yes["outcome_price"]).all()
        assert (buy_yes["taker_side"] == "yes").all()
        assert trades["timestamp"].is_sorted()
        assert trades["ticker"].unique().to_list() == [CONDITION_ID]

    def test_paging_past_the_offset_cap_moves_the_end_bound(self, provider, use_transport):
        """Prevents: HTTP 400 past the offset cap, or trades lost/duplicated at the boundary."""
        requests = use_transport(provider, _trades_server(TRADES, max_offset=8))
        provider.TRADES_MAX_OFFSET = 8

        trades = provider.fetch_trades(CONDITION_ID, page_size=4)

        expected = {provider._trade_key(trade) for trade in TRADES}
        assert trades.height == len(TRADES)
        assert set(trades["trade_id"].to_list()) == expected
        assert any("end" in request.url.params for request in requests)
        assert max(int(request.url.params["offset"]) for request in requests) <= 8

    def test_start_and_end_are_sent_and_enforced(self, provider, use_transport):
        """Prevents: trades outside the requested window, or bounds not passed to the API."""
        requests = use_transport(provider, _trades_server(TRADES, max_offset=10_000))
        start, end = 1767245799, 1767245803

        trades = provider.fetch_trades(CONDITION_ID, start=start, end=end)

        assert requests[0].url.params["start"] == str(start)
        assert requests[0].url.params["end"] == str(end)
        inside = [trade for trade in TRADES if start <= trade["timestamp"] <= end]
        assert trades.height == len(inside) == 18

    def test_repeated_page_raises_instead_of_looping(self, provider, use_transport):
        """Prevents: an endless loop when the server ignores offset and repeats one page."""
        use_transport(provider, _trades_server(TRADES, max_offset=10_000, ignore_offset=True))

        with pytest.raises(DataValidationError, match="ignored offset"):
            provider.fetch_trades(CONDITION_ID, page_size=4)

    def test_one_second_with_more_trades_than_a_window_raises(self, provider, use_transport):
        """Prevents: an endless loop when one second holds more trades than offsets can reach."""
        same_second = []
        for index in range(12):
            trade = copy.deepcopy(TRADES[6])
            trade["transactionHash"] = f"0x{index:064x}"
            same_second.append(trade)
        use_transport(provider, _trades_server(same_second, max_offset=4))
        provider.TRADES_MAX_OFFSET = 4

        with pytest.raises(DataValidationError, match="cannot page past"):
            provider.fetch_trades(CONDITION_ID, page_size=2)

    def test_malformed_trade_raises(self, provider, use_transport):
        """Prevents: a trade with an unknown outcome index being priced as YES or NO."""
        bad = dict(TRADES[0], outcomeIndex=2)
        use_transport(provider, _trades_server([bad], max_offset=10_000))

        with pytest.raises(DataValidationError, match="outcomeIndex"):
            provider.fetch_trades(CONDITION_ID)

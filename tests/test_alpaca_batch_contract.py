"""Public batch requests preserve pagination and stock identity contracts."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.alpaca import AlpacaDataProvider


def response(bars, token=None):
    result = MagicMock(status_code=200)
    result.json.return_value = {"bars": bars, "next_page_token": token}
    return result


def bar(timestamp, close):
    return {"t": timestamp, "o": 100, "h": 110, "l": 90, "c": close, "v": 1000}


@pytest.mark.parametrize("frequency,timeframe", [("weekly", "1Week"), ("monthly", "1Month")])
def test_batch_pages_preserve_query_bounds_asof_and_missing_symbols(frequency, timeframe):
    first = "2026-01-01T05:00:00Z"
    last = "2026-02-02T05:00:00Z"
    with (
        AlpacaDataProvider(api_key="k", api_secret="s", asof="-") as provider,
        patch.object(
            provider.session,
            "get",
            side_effect=[
                response({"SPY": [bar(first, 101)]}, "p2"),
                response({"SPY": [bar(last, 102)]}),
                response({"QQQ": [bar(first, 103)]}),
            ],
        ) as get,
    ):
        result = provider.fetch_batch_ohlcv(
            ["spy", "MISSING", "SPY", "qqq"], first, last, frequency, chunk_size=2
        )
        params = [call.kwargs["params"] for call in get.call_args_list]
        assert [query["symbols"] for query in params] == ["SPY,MISSING", "SPY,MISSING", "QQQ"]
        assert "page_token" not in params[0]
        assert params[1]["page_token"] == "p2"
        assert "page_token" not in params[2]
        for query in params:
            assert (query["start"], query["end"], query["asof"], query["timeframe"]) == (
                first,
                last,
                "-",
                timeframe,
            )
        assert result.schema == provider.CANONICAL_SCHEMA
        assert result.select("symbol", "close").rows() == [
            ("QQQ", 103.0),
            ("SPY", 101.0),
            ("SPY", 102.0),
        ]
        assert result["timestamp"].to_list() == [
            datetime(2026, 1, 1, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 5, tzinfo=UTC),
            datetime(2026, 2, 2, 5, tzinfo=UTC),
        ]


def test_batch_clipped_empty_sip_interval_does_not_request():
    with (
        AlpacaDataProvider(
            api_key="k", api_secret="s", feed="sip", sip_delay_minutes=15
        ) as provider,
        patch("ml4t.data.providers.alpaca.datetime", wraps=datetime) as clock,
        patch.object(provider.session, "get") as get,
    ):
        clock.now.return_value = datetime(2026, 11, 23, 16, tzinfo=UTC)
        result = provider.fetch_batch_ohlcv(
            ["SPY", "QQQ"], "2026-11-23T15:50:00Z", "2026-11-23T15:59:00Z", "minute"
        )
        assert result.is_empty()
        assert result.schema == provider.CANONICAL_SCHEMA
        get.assert_not_called()


@pytest.mark.parametrize(
    "bars",
    [[bar("2026-01-01T05:00:00Z", 100)], {"UNREQUESTED": [bar("2026-01-01T05:00:00Z", 100)]}],
)
def test_batch_rejects_malformed_or_unrequested_symbol_response(bars):
    with (
        AlpacaDataProvider(api_key="k", api_secret="s") as provider,
        patch.object(provider.session, "get", return_value=response(bars)),
        pytest.raises(DataValidationError),
    ):
        provider.fetch_batch_ohlcv(["SPY"], "2026-01-01", "2026-01-02")


def test_batch_repeated_cursor_raises_instead_of_returning_partial_history():
    with (
        AlpacaDataProvider(api_key="k", api_secret="s") as provider,
        patch.object(
            provider.session,
            "get",
            side_effect=[
                response({"SPY": [bar("2026-01-01T05:00:00Z", 100)]}, "again"),
                response({"SPY": [bar("2026-01-02T05:00:00Z", 101)]}, "again"),
            ],
        ) as get,
        pytest.raises(DataValidationError, match="repeated"),
    ):
        provider.fetch_batch_ohlcv(["SPY"], "2026-01-01", "2026-01-02")
    assert get.call_count == 2


def test_all_missing_batch_returns_typed_empty_frame():
    with (
        AlpacaDataProvider(api_key="k", api_secret="s") as provider,
        patch.object(provider.session, "get", return_value=response({})),
    ):
        result = provider.fetch_batch_ohlcv(["MISSING"], "2026-01-01", "2026-01-02")
        assert result.is_empty()
        assert result.schema == provider.CANONICAL_SCHEMA

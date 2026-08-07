"""Reliability tests for the OKX provider."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.okx import OKXProvider


def _candle(timestamp: int) -> list[str]:
    return [str(timestamp), "100", "105", "99", "104", "10", "0", "0", "1"]


def _response(data: list[object]) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {"code": "0", "data": data}
    return response


@pytest.fixture
def provider():
    """Create an OKX provider and close its sync client after each test."""
    instance = OKXProvider()
    yield instance
    instance.close()


def test_repeated_candle_cursor_terminates_public_fetch(provider) -> None:
    """The sync public path rejects a timestamp cursor that stops moving backward."""
    repeated = _response([_candle(1_704_153_600_000)])
    calls = 0

    def repeat_with_probe_limit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 3:
            raise AssertionError("pagination did not terminate")
        return repeated

    with (
        patch.object(provider.client, "get", side_effect=repeat_with_probe_limit) as mock_get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        with pytest.raises(DataValidationError, match="pagination cursor did not move backward"):
            provider.fetch_ohlcv("BTC-USDT-SWAP", "2024-01-01", "2024-01-03", "daily")

    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_repeated_candle_cursor_terminates_public_fetch_async(provider) -> None:
    """The async public path applies the same timestamp-progress contract."""
    repeated = _response([_candle(1_704_153_600_000)])
    calls = 0

    def repeat_with_probe_limit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 3:
            raise AssertionError("pagination did not terminate")
        return repeated

    with (
        patch.object(
            provider, "_aget", new=AsyncMock(side_effect=repeat_with_probe_limit)
        ) as mock_get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        with pytest.raises(DataValidationError, match="pagination cursor did not move backward"):
            await provider.fetch_ohlcv_async("BTC-USDT-SWAP", "2024-01-01", "2024-01-03", "daily")

    assert mock_get.await_count == 2


def test_repeated_funding_cursor_terminates_fetch(provider) -> None:
    """Funding-rate pagination rejects a repeated timestamp cursor."""
    repeated = _response(
        [
            {
                "fundingTime": "1704153600000",
                "fundingRate": "0.0001",
                "realizedRate": "0.0001",
            }
        ]
    )
    calls = 0

    def repeat_with_probe_limit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 3:
            raise AssertionError("pagination did not terminate")
        return repeated

    with (
        patch.object(provider.client, "get", side_effect=repeat_with_probe_limit) as mock_get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        with pytest.raises(DataValidationError, match="pagination cursor did not move backward"):
            provider.fetch_funding_rates("BTC-USDT-SWAP", "2024-01-01", "2024-01-03")

    assert mock_get.call_count == 2


def test_candle_page_limit_terminates_public_fetch(provider) -> None:
    """A unique but endless cursor sequence stops at the declared page limit."""
    provider.MAX_PAGES = 2
    timestamps = iter([1_704_758_400_000, 1_704_672_000_000])

    def next_page(*args, **kwargs):
        return _response([_candle(next(timestamps))])

    with (
        patch.object(provider.client, "get", side_effect=next_page) as mock_get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        with pytest.raises(DataValidationError, match="pagination page limit"):
            provider.fetch_ohlcv("BTC-USDT-SWAP", "2024-01-01", "2024-01-10", "daily")

    assert mock_get.call_count == provider.MAX_PAGES


def test_candle_pagination_uses_history_endpoint_and_after_cursor(provider) -> None:
    newest = int(datetime(2024, 1, 9, tzinfo=UTC).timestamp() * 1000)
    oldest = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    end = int(datetime(2024, 1, 10, tzinfo=UTC).timestamp() * 1000)

    with (
        patch.object(
            provider.client,
            "get",
            side_effect=[_response([_candle(newest)]), _response([_candle(oldest)])],
        ) as get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        provider.fetch_ohlcv("BTC-USDT-SWAP", "2024-01-01", "2024-01-10", "daily")

    assert [call.args[0] for call in get.call_args_list] == [
        f"{provider.BASE_URL}/market/history-candles",
        f"{provider.BASE_URL}/market/history-candles",
    ]
    assert [call.kwargs["params"]["after"] for call in get.call_args_list] == [
        str(end + 1),
        str(newest),
    ]
    assert all(call.kwargs["params"]["bar"] == "1Dutc" for call in get.call_args_list)


@pytest.mark.asyncio
async def test_async_candle_pagination_uses_history_endpoint_and_after_cursor(provider) -> None:
    newest = int(datetime(2024, 1, 9, tzinfo=UTC).timestamp() * 1000)
    oldest = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    end = int(datetime(2024, 1, 10, tzinfo=UTC).timestamp() * 1000)

    with (
        patch.object(
            provider,
            "_aget",
            new=AsyncMock(side_effect=[_response([_candle(newest)]), _response([_candle(oldest)])]),
        ) as get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        await provider.fetch_ohlcv_async("BTC-USDT-SWAP", "2024-01-01", "2024-01-10", "daily")

    assert [call.args[0] for call in get.call_args_list] == [
        f"{provider.BASE_URL}/market/history-candles",
        f"{provider.BASE_URL}/market/history-candles",
    ]
    assert [call.kwargs["params"]["after"] for call in get.call_args_list] == [
        str(end + 1),
        str(newest),
    ]


def test_funding_pagination_uses_after_cursor(provider) -> None:
    newest = int(datetime(2024, 1, 2, 16, tzinfo=UTC).timestamp() * 1000)
    oldest = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    end = int(datetime(2024, 1, 3, 23, 59, 59, tzinfo=UTC).timestamp() * 1000)

    def rate(timestamp: int) -> dict[str, str]:
        return {
            "fundingTime": str(timestamp),
            "fundingRate": "0.0001",
            "realizedRate": "0.0001",
        }

    with (
        patch.object(
            provider.client,
            "get",
            side_effect=[_response([rate(newest)]), _response([rate(oldest)])],
        ) as get,
        patch.object(provider, "_acquire_rate_limit"),
    ):
        provider.fetch_funding_rates("BTC-USDT-SWAP", "2024-01-01", "2024-01-03")

    assert [call.kwargs["params"]["after"] for call in get.call_args_list] == [
        str(end + 1),
        str(newest),
    ]

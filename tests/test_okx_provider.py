"""Reliability tests for the OKX provider."""

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

"""SIP delays are explicit and never create an invalid request window."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml4t.data.core.exceptions import AuthenticationError, DataValidationError
from ml4t.data.providers.alpaca import AlpacaDataProvider


def test_paid_sip_default_preserves_requested_recent_end():
    end = "2026-11-23T15:59:00Z"
    with (
        AlpacaDataProvider(api_key="k", api_secret="s", feed="sip") as provider,
        patch("ml4t.data.providers.alpaca.datetime", wraps=datetime) as clock,
        patch.object(provider.session, "get") as get,
    ):
        clock.now.return_value = datetime(2026, 11, 23, 16, tzinfo=UTC)
        get.return_value = MagicMock(status_code=200)
        get.return_value.json.return_value = {"bars": [], "next_page_token": None}
        provider.fetch_ohlcv("SPY", "2026-11-23T15:58:00Z", end, "minute")
        assert get.call_args.kwargs["params"]["end"] == end


def test_datetime_start_with_inclusive_date_end_is_not_an_empty_window():
    with (
        AlpacaDataProvider(api_key="k", api_secret="s", feed="sip") as provider,
        patch.object(provider.session, "get") as get,
    ):
        get.return_value = MagicMock(status_code=200)
        get.return_value.json.return_value = {"bars": [], "next_page_token": None}
        provider.fetch_ohlcv("SPY", "2026-11-20T16:00:00Z", "2026-11-20", "minute")
        assert get.call_args.kwargs["params"]["end"] == "2026-11-20"


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-11-23T15:50:00Z", "2026-11-23T15:59:00Z"),
        ("2026-11-24", "2026-11-24"),
    ],
)
def test_clipped_sip_window_without_available_bars_does_not_request(start, end):
    with (
        AlpacaDataProvider(
            api_key="k", api_secret="s", feed="sip", sip_delay_minutes=15
        ) as provider,
        patch("ml4t.data.providers.alpaca.datetime", wraps=datetime) as clock,
        patch.object(provider.session, "get") as get,
    ):
        clock.now.return_value = datetime(2026, 11, 23, 16, tzinfo=UTC)
        result = provider.fetch_ohlcv("SPY", start, end, "minute")
        assert result.is_empty()
        assert result.columns == list(provider.CANONICAL_COLUMNS)
        get.assert_not_called()


@pytest.mark.asyncio
async def test_async_clipped_sip_window_does_not_request():
    with (
        AlpacaDataProvider(
            api_key="k", api_secret="s", feed="sip", sip_delay_minutes=15
        ) as provider,
        patch("ml4t.data.providers.alpaca.datetime", wraps=datetime) as clock,
        patch.object(provider, "_aget", new_callable=AsyncMock) as get,
    ):
        clock.now.return_value = datetime(2026, 11, 23, 0, 5, tzinfo=UTC)
        result = await provider.fetch_ohlcv_async("SPY", "2026-11-23", "2026-11-23")
        assert result.is_empty()
        get.assert_not_called()


def test_sip_subscription_denial_does_not_claim_invalid_credentials():
    response = MagicMock(status_code=403)
    response.text = '{"message":"subscription does not permit querying recent SIP data"}'
    with (
        AlpacaDataProvider(api_key="k", api_secret="s", feed="sip") as provider,
        patch.object(provider.session, "get", return_value=response) as get,
    ):
        with pytest.raises(AuthenticationError, match="subscription does not permit") as error:
            provider.fetch_ohlcv("SPY", "2026-11-19", "2026-11-20")
        assert "Invalid API key" not in str(error.value)
        get.assert_called_once()


@pytest.mark.parametrize("delay", [-1, True, 1.5, "15", None])
def test_invalid_sip_delay_is_a_configuration_error(delay):
    with pytest.raises(DataValidationError, match="sip_delay_minutes"):
        AlpacaDataProvider(api_key="k", api_secret="s", sip_delay_minutes=delay)

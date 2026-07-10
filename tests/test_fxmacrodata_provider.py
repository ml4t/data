"""Tests for FXMacroData provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from ml4t.data.providers.fxmacrodata import FXMacroDataProvider

ANNOUNCEMENT_PAYLOAD = {
    "currency": "USD",
    "indicator": "inflation",
    "name": "Inflation (CPI)",
    "source": "BLS",
    "source_series_id": "BLS:CUUR0000SA0",
    "requested_start_date": "2025-07-10",
    "requested_end_date": "2026-07-10",
    "data_quality": {
        "is_official": True,
        "is_proxy": False,
        "has_announcement_datetime": True,
        "point_in_time_safe": True,
        "row_count": 1,
    },
    "pagination": {
        "limit": 1,
        "offset": 0,
        "returned_count": 1,
    },
    "data": [
        {
            "announcement_id": "usd_inflation_2026-05-31",
            "date": "2026-05-31",
            "val": 4.2,
            "announcement_datetime": 1781094600,
            "announcement_datetime_local": "2026-06-10T08:30:00-04:00",
        }
    ],
}


def _response(status_code: int = 200, payload=None, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.json.return_value = payload if payload is not None else {}
    return response


class TestFXMacroDataProviderInit:
    def test_init_without_api_key_allowed(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = FXMacroDataProvider()
            try:
                assert provider.name == "fxmacrodata"
                assert provider.api_key is None
                assert provider.base_url == "https://api.fxmacrodata.com/v1"
            finally:
                provider.close()

    def test_init_with_env_api_key(self):
        with patch.dict("os.environ", {"FXMACRODATA_API_KEY": "env_key"}, clear=True):
            provider = FXMacroDataProvider()
            try:
                assert provider.api_key == "env_key"
            finally:
                provider.close()

    def test_init_with_legacy_env_api_key(self):
        with patch.dict("os.environ", {"FXMD_API_KEY": "legacy_key"}, clear=True):
            provider = FXMacroDataProvider()
            try:
                assert provider.api_key == "legacy_key"
            finally:
                provider.close()


class TestFXMacroDataRequests:
    @pytest.fixture
    def provider(self):
        provider = FXMacroDataProvider(api_key="test_key", rate_limit=(1000, 1.0))
        yield provider
        provider.close()

    def test_fetch_announcements_preserves_key_fields_and_metadata(self, provider):
        response = _response(payload=ANNOUNCEMENT_PAYLOAD)

        with patch.object(provider.session, "get", return_value=response) as get:
            frame, metadata = provider.fetch_announcements(
                "USD",
                "inflation",
                start_date="2025-01-01",
                limit=1,
                include_metadata=True,
            )

        assert isinstance(frame, pl.DataFrame)
        assert len(frame) == 1
        assert frame["announcement_id"][0] == "usd_inflation_2026-05-31"
        assert frame["announcement_datetime"][0] == 1781094600
        assert frame["currency"][0] == "USD"
        assert frame["indicator"][0] == "inflation"
        assert frame["data_quality_point_in_time_safe"][0] is True
        assert metadata["pagination"]["returned_count"] == 1
        assert metadata["data_quality"]["has_announcement_datetime"] is True

        url = get.call_args.args[0]
        params = get.call_args.kwargs["params"]
        assert url == "https://api.fxmacrodata.com/v1/announcements/usd/inflation"
        assert params["api_key"] == "test_key"
        assert params["start_date"] == "2025-01-01"
        assert params["limit"] == 1
        assert "end_date" not in params

    def test_fetch_catalogue_reshapes_indicator_mapping(self, provider):
        payload = {
            "gdp": {
                "name": "GDP",
                "unit": "USD bn",
                "frequency": "Quarterly",
                "coverage": {"available": True},
            },
            "inflation": {
                "name": "Inflation",
                "unit": "%",
                "frequency": "Monthly",
                "coverage": {"available": True},
            },
        }
        response = _response(payload=payload)

        with patch.object(provider.session, "get", return_value=response):
            frame = provider.fetch_catalogue("usd")

        assert set(frame["indicator"].to_list()) == {"gdp", "inflation"}
        assert "coverage" in frame.columns

    def test_fetch_endpoint_routes_to_forex(self, provider):
        response = _response(payload={"data": [{"date": "2026-01-01", "rate": 1.1}]})

        with patch.object(provider.session, "get", return_value=response) as get:
            frame = provider.fetch_endpoint(
                "forex",
                currency="eur",
                quote="usd",
                start_date="2026-01-01",
                limit=10,
            )

        assert len(frame) == 1
        assert frame["rate"][0] == 1.1
        assert get.call_args.args[0] == "https://api.fxmacrodata.com/v1/forex/eur/usd"
        assert get.call_args.kwargs["params"]["limit"] == 10

    def test_health_returns_payload(self, provider):
        response = _response(payload={"status": "ok", "service": "fxmacrodata-api"})

        with patch.object(provider.session, "get", return_value=response):
            payload = provider.health()

        assert payload == {"status": "ok", "service": "fxmacrodata-api"}

    def test_fetch_market_sessions_handles_list_payload(self, provider):
        response = _response(payload=[{"session": "London", "status": "open"}])

        with patch.object(provider.session, "get", return_value=response):
            frame = provider.fetch_market_sessions()

        assert frame.shape == (1, 2)
        assert frame["session"][0] == "London"

    def test_rate_limit_error(self, provider):
        response = _response(status_code=429)
        response.headers = {"Retry-After": "30"}

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(RateLimitError) as exc_info:
                provider.fetch_risk_sentiment()

        assert exc_info.value.retry_after == 30.0

    def test_authentication_error(self, provider):
        response = _response(status_code=403, text="API key required")

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(AuthenticationError, match="API key required"):
                provider.fetch_cot("eur")

    def test_not_found_error(self, provider):
        response = _response(status_code=404)

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(DataNotAvailableError):
                provider.fetch_announcements("usd", "missing")

    def test_http_error(self, provider):
        response = _response(status_code=500, text="server error")

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(ProviderError, match="HTTP 500"):
                provider.fetch_risk_sentiment()

    def test_json_parse_error(self, provider):
        response = _response()
        response.json.side_effect = ValueError("invalid json")

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(NetworkError, match="Failed to parse JSON"):
                provider.fetch_risk_sentiment()

    def test_api_error_payload(self, provider):
        response = _response(payload={"error": "bad request"})

        with patch.object(provider.session, "get", return_value=response):
            with pytest.raises(ProviderError, match="bad request"):
                provider.fetch_risk_sentiment()

    def test_request_error(self, provider):
        with patch.object(provider.session, "get", side_effect=httpx.RequestError("boom")):
            with pytest.raises(NetworkError, match="Request failed"):
                provider.fetch_risk_sentiment()

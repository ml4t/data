"""FXMacroData provider for FX macro, calendar, positioning, and sentiment data.

FXMacroData provides release-aware macroeconomic data built for FX research:
announcement histories with known-at timestamps, release calendars, FX spot
history, COT positioning, commodities, market sessions, and risk sentiment.

API Documentation: https://fxmacrodata.com/documentation
"""

from __future__ import annotations

import os
from typing import Any, Literal, overload

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from ml4t.data.providers.mixins.rate_limit import RateLimitMixin
from ml4t.data.providers.mixins.session import SessionMixin

logger = structlog.get_logger()

FXMacroEndpoint = Literal[
    "announcements",
    "predictions",
    "calendar",
    "catalogue",
    "forex",
    "cot",
    "commodity",
    "commodities_latest",
    "market_sessions",
    "risk_sentiment",
    "news",
    "press_releases",
]

PayloadResult = tuple[pl.DataFrame, dict[str, Any]]


class FXMacroDataProvider(RateLimitMixin, SessionMixin):
    """Provider for FXMacroData macro and FX research endpoints.

    API keys are optional at construction time because several discovery and USD
    endpoints are public. Protected endpoints raise AuthenticationError when the
    service rejects unauthenticated requests.
    """

    DEFAULT_RATE_LIMIT = (60, 60.0)
    BASE_URL = "https://api.fxmacrodata.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limit: tuple[int, float] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize FXMacroData provider.

        Args:
            api_key: Optional API key, or set FXMACRODATA_API_KEY / FXMD_API_KEY.
            base_url: Optional API base URL override.
            rate_limit: Optional (calls, period_seconds) tuple.
            timeout: Optional request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("FXMACRODATA_API_KEY") or os.getenv("FXMD_API_KEY")
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.init_rate_limit(self.name, rate_limit or self.DEFAULT_RATE_LIMIT)
        self.init_session(timeout=timeout, headers={"Accept": "application/json"})
        self.logger = structlog.get_logger(name=self.__class__.__name__)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "fxmacrodata"

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.close_session()

    def __del__(self) -> None:
        """Best-effort session cleanup for leaked provider instances."""
        try:
            self.close()
        except Exception:
            pass

    def health(self) -> dict[str, Any]:
        """Return service health payload."""
        payload = self._request_json("health", {})
        if not isinstance(payload, dict):
            raise ProviderError(provider=self.name, message="Unexpected health response")
        return payload

    @overload
    def fetch_announcements(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        seasonality: str | None = None,
        frequency: str | None = None,
        basis: str | None = None,
        revisions: str | None = None,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_announcements(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        seasonality: str | None = None,
        frequency: str | None = None,
        basis: str | None = None,
        revisions: str | None = None,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_announcements(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        seasonality: str | None = None,
        frequency: str | None = None,
        basis: str | None = None,
        revisions: str | None = None,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch macro announcement history for a currency and indicator."""
        payload = self._request_json(
            f"announcements/{self._slug(currency)}/{self._slug(indicator)}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
                "seasonality": seasonality,
                "frequency": frequency,
                "basis": basis,
                "revisions": revisions,
            },
        )
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_predictions(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_predictions(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_predictions(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch forecast and consensus rows joined to macro announcements."""
        payload = self._request_json(
            f"predictions/{self._slug(currency)}/{self._slug(indicator)}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
            },
        )
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_calendar(
        self,
        currency: str,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_calendar(
        self,
        currency: str,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_calendar(
        self,
        currency: str,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch upcoming macro release calendar for a currency."""
        payload = self._request_json(f"calendar/{self._slug(currency)}", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_catalogue(
        self,
        currency: str,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_catalogue(
        self,
        currency: str,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_catalogue(
        self,
        currency: str,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch available macro indicators for a currency."""
        payload = self._request_json(f"data_catalogue/{self._slug(currency)}", {})
        rows = self._catalogue_rows(payload)
        metadata = {"currency": currency.upper()}
        frame = self._rows_to_frame(rows)
        return (frame, metadata) if include_metadata else frame

    @overload
    def fetch_forex(
        self,
        base: str,
        quote: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        indicators: str | None = None,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_forex(
        self,
        base: str,
        quote: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        indicators: str | None = None,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_forex(
        self,
        base: str,
        quote: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        indicators: str | None = None,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch FX spot history and optional technical indicator columns."""
        payload = self._request_json(
            f"forex/{self._slug(base)}/{self._slug(quote)}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
                "indicators": indicators,
            },
        )
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_cot(
        self,
        currency: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_cot(
        self,
        currency: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_cot(
        self,
        currency: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch CFTC Commitment of Traders positioning data."""
        payload = self._request_json(
            f"cot/{self._slug(currency)}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
            },
        )
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_commodities(
        self,
        commodity: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_commodities(
        self,
        commodity: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_commodities(
        self,
        commodity: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 500,
        offset: int | None = None,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch commodity price history."""
        payload = self._request_json(
            f"commodities/{self._slug(commodity)}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
            },
        )
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_latest_commodities(
        self,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_latest_commodities(
        self,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_latest_commodities(
        self,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch latest commodity prices."""
        payload = self._request_json("commodities/latest", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_market_sessions(
        self,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_market_sessions(
        self,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_market_sessions(
        self,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch FX market session timetable."""
        payload = self._request_json("market_sessions", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_risk_sentiment(
        self,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_risk_sentiment(
        self,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_risk_sentiment(
        self,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch global risk-on/risk-off sentiment indicator."""
        payload = self._request_json("risk_sentiment", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_news(
        self,
        currency: str,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_news(
        self,
        currency: str,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_news(
        self,
        currency: str,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch recent central bank news for a currency."""
        payload = self._request_json(f"news/{self._slug(currency)}", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    @overload
    def fetch_press_releases(
        self,
        currency: str,
        *,
        include_metadata: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def fetch_press_releases(
        self,
        currency: str,
        *,
        include_metadata: Literal[True],
    ) -> PayloadResult: ...

    def fetch_press_releases(
        self,
        currency: str,
        *,
        include_metadata: bool = False,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch recent central bank press releases for a currency."""
        payload = self._request_json(f"press-releases/{self._slug(currency)}", {})
        return self._format_payload(payload, include_metadata=include_metadata)

    def fetch_endpoint(
        self,
        endpoint: FXMacroEndpoint,
        *,
        currency: str = "usd",
        indicator: str = "policy_rate",
        quote: str = "usd",
        commodity: str = "gold",
        include_metadata: bool = False,
        **params: Any,
    ) -> pl.DataFrame | PayloadResult:
        """Fetch any supported endpoint by name."""
        endpoints_without_params = {
            "calendar",
            "catalogue",
            "commodities_latest",
            "market_sessions",
            "risk_sentiment",
            "news",
            "press_releases",
        }
        if endpoint in endpoints_without_params and params:
            unsupported = ", ".join(sorted(params))
            raise ValueError(f"Endpoint '{endpoint}' does not accept parameters: {unsupported}")

        if endpoint == "announcements":
            return self.fetch_announcements(
                currency, indicator, include_metadata=include_metadata, **params
            )
        if endpoint == "predictions":
            return self.fetch_predictions(
                currency, indicator, include_metadata=include_metadata, **params
            )
        if endpoint == "calendar":
            return self.fetch_calendar(currency, include_metadata=include_metadata)
        if endpoint == "catalogue":
            return self.fetch_catalogue(currency, include_metadata=include_metadata)
        if endpoint == "forex":
            return self.fetch_forex(currency, quote, include_metadata=include_metadata, **params)
        if endpoint == "cot":
            return self.fetch_cot(currency, include_metadata=include_metadata, **params)
        if endpoint == "commodity":
            return self.fetch_commodities(commodity, include_metadata=include_metadata, **params)
        if endpoint == "commodities_latest":
            return self.fetch_latest_commodities(include_metadata=include_metadata)
        if endpoint == "market_sessions":
            return self.fetch_market_sessions(include_metadata=include_metadata)
        if endpoint == "risk_sentiment":
            return self.fetch_risk_sentiment(include_metadata=include_metadata)
        if endpoint == "news":
            return self.fetch_news(currency, include_metadata=include_metadata)
        if endpoint == "press_releases":
            return self.fetch_press_releases(currency, include_metadata=include_metadata)
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    def _request_json(self, path: str, params: dict[str, Any]) -> Any:
        clean_params = {key: value for key, value in params.items() if value is not None}
        if self.api_key:
            clean_params["api_key"] = self.api_key

        url = f"{self.base_url}/{path.lstrip('/')}"
        self._acquire_rate_limit()

        try:
            response = self.session.get(url, params=clean_params)
        except httpx.RequestError as err:
            raise NetworkError(provider=self.name, message=f"Request failed: {err}") from err

        if response.status_code == 429:
            retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
            raise RateLimitError(provider=self.name, retry_after=retry_after)
        if response.status_code in {401, 403}:
            raise AuthenticationError(provider=self.name, message=response.text[:500])
        if response.status_code == 404:
            raise DataNotAvailableError(provider=self.name, symbol=path)
        if response.status_code >= 400:
            raise ProviderError(
                provider=self.name,
                message=f"HTTP {response.status_code}: {response.text[:500]}",
            )

        try:
            payload = response.json()
        except ValueError as err:
            raise NetworkError(provider=self.name, message="Failed to parse JSON response") from err

        if isinstance(payload, dict):
            error = payload.get("error") or payload.get("detail")
            if not error and payload.get("success") is False:
                error = payload.get("message") or "API request failed"
            if error:
                raise ProviderError(provider=self.name, message=str(error))
        return payload

    def _format_payload(
        self, payload: Any, *, include_metadata: bool
    ) -> pl.DataFrame | PayloadResult:
        rows = self._rows_from_payload(payload)
        metadata = self._metadata_from_payload(payload)
        frame = self._rows_to_frame(rows)
        frame = self._add_metadata_columns(frame, metadata)
        return (frame, metadata) if include_metadata else frame

    @staticmethod
    def _slug(value: str) -> str:
        return value.strip().lower().replace(" ", "-")

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _contains_payload_rows(payload: dict[str, Any]) -> bool:
        return any(
            isinstance(payload.get(key), list) for key in ("data", "rows", "results", "items")
        )

    @staticmethod
    def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("data", "rows", "results", "items"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [row if isinstance(row, dict) else {"value": row} for row in rows]
            return [payload]
        if isinstance(payload, list):
            return [row if isinstance(row, dict) else {"value": row} for row in payload]
        return [{"value": payload}]

    @staticmethod
    def _metadata_from_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        return {
            key: value
            for key, value in payload.items()
            if key not in {"data", "rows", "results", "items"}
        }

    @staticmethod
    def _catalogue_rows(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return FXMacroDataProvider._rows_from_payload(payload)
        if any(key in payload for key in ("data", "rows", "results", "items")):
            return FXMacroDataProvider._rows_from_payload(payload)
        return [
            {"indicator": indicator, **details}
            if isinstance(details, dict)
            else {"indicator": indicator, "value": details}
            for indicator, details in payload.items()
        ]

    @staticmethod
    def _rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, infer_schema_length=None)

    @staticmethod
    def _add_metadata_columns(frame: pl.DataFrame, metadata: dict[str, Any]) -> pl.DataFrame:
        if frame.is_empty():
            return frame

        scalar_keys = [
            "currency",
            "indicator",
            "name",
            "source",
            "source_series_id",
            "source_series_name",
            "seasonal_adjustment",
            "requested_start_date",
            "requested_end_date",
            "start_date",
            "end_date",
            "earliest_available_date",
            "latest_available_date",
        ]
        expressions = []
        for key in scalar_keys:
            if key not in metadata:
                continue
            value = metadata.get(key)
            if (
                value is not None
                and FXMacroDataProvider._is_scalar(value)
                and key not in frame.columns
            ):
                expressions.append(pl.lit(value).alias(key))

        data_quality = metadata.get("data_quality")
        if isinstance(data_quality, dict):
            for key, value in data_quality.items():
                column = f"data_quality_{key}"
                if (
                    value is not None
                    and FXMacroDataProvider._is_scalar(value)
                    and column not in frame.columns
                ):
                    expressions.append(pl.lit(value).alias(column))

        return frame.with_columns(expressions) if expressions else frame

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, str | int | float | bool)

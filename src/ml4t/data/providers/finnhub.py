"""Finnhub data provider.

Finnhub provides comprehensive financial market data with global coverage.

API Documentation: https://finnhub.io/docs/api
Pricing: https://finnhub.io/pricing

Tier limitations:

FREE TIER (60 API calls/minute):
- Real-time US equity quotes
- Selected US company data

PAID TIER:
- Historical OHLCV
- Global market data and extended history

Example:
    >>> from ml4t.data.providers.finnhub import FinnhubProvider
    >>> provider = FinnhubProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> provider.close()
"""

import os
from datetime import datetime
from typing import Any, ClassVar

import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.fundamentals import (
    PeriodType,
    StatementType,
    normalize_period_type,
    normalize_statement_type,
    numeric_mapping_to_metric_rows,
    records_to_financials_rows,
    rows_to_company_metrics_frame,
    rows_to_financials_frame,
)

logger = structlog.get_logger()


class FinnhubProvider(BaseProvider):
    """Finnhub data provider.

    Supports stocks, ETFs, forex, and crypto with multiple resolutions.

    Rate Limits:
    - Free tier: 60 requests/minute for supported endpoints such as US quotes
    - Paid tier: Historical OHLC and higher limits depend on the current plan
    """

    # Free tier: 60 requests/min = 1 per second
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1, 1.0)

    # Map frequency to Finnhub resolution codes
    RESOLUTION_MAP: ClassVar[dict[str, str]] = {
        "1min": "1",
        "1m": "1",
        "5min": "5",
        "5m": "5",
        "15min": "15",
        "15m": "15",
        "30min": "30",
        "30m": "30",
        "60min": "60",
        "1h": "60",
        "hour": "60",
        "hourly": "60",
        "daily": "D",
        "1d": "D",
        "day": "D",
        "D": "D",
        "weekly": "W",
        "1w": "W",
        "week": "W",
        "W": "W",
        "monthly": "M",
        "1M": "M",
        "month": "M",
        "M": "M",
    }

    FINANCIAL_STATEMENT_MAP: ClassVar[dict[StatementType, str]] = {
        "income": "ic",
        "balance": "bs",
        "cashflow": "cf",
    }

    FINANCIAL_PERIOD_MAP: ClassVar[dict[PeriodType, str]] = {
        "annual": "annual",
        "quarterly": "quarterly",
    }

    def __init__(self, api_key: str | None = None, rate_limit: tuple[int, float] | None = None):
        """Initialize Finnhub provider.

        Args:
            api_key: Finnhub API key (or set FINNHUB_API_KEY env var)
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="finnhub",
                message="Finnhub API key required. Set FINNHUB_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get free key at: https://finnhub.io/register",
            )

        self.base_url = "https://finnhub.io/api/v1"

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info(
            "Initialized Finnhub provider", rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "finnhub"

    def fetch_quote(self, symbol: str) -> dict[str, str | int | float]:
        """Fetch a real-time US equity quote available on Finnhub's free tier.

        Args:
            symbol: US equity symbol, such as ``AAPL``.

        Returns:
            Normalized quote values with a Unix timestamp in seconds.

        Raises:
            SymbolNotFoundError: If Finnhub returns no quote for the symbol.
            DataValidationError: If the response is missing required quote values.
        """
        normalized_symbol = symbol.upper()
        data = self._request_json(
            "/quote",
            params={"symbol": normalized_symbol, "token": self.api_key},
            symbol=normalized_symbol,
        )
        if not data.get("t") or not data.get("c"):
            raise SymbolNotFoundError(provider=self.name, symbol=normalized_symbol)

        response_keys = {
            "timestamp": "t",
            "current": "c",
            "high": "h",
            "low": "l",
            "open": "o",
            "previous_close": "pc",
        }
        missing = [name for name, key in response_keys.items() if data.get(key) is None]
        if missing:
            raise DataValidationError(
                provider=self.name,
                message=f"Finnhub quote is missing required values: {', '.join(missing)}",
            )

        try:
            current = float(data["c"])
            previous_close = float(data["pc"])
            change = float(data["d"]) if data.get("d") is not None else current - previous_close
            percent_change = (
                float(data["dp"])
                if data.get("dp") is not None
                else (change / previous_close * 100 if previous_close else 0.0)
            )
            return {
                "symbol": normalized_symbol,
                "timestamp": int(data["t"]),
                "current": current,
                "change": change,
                "percent_change": percent_change,
                "high": float(data["h"]),
                "low": float(data["l"]),
                "open": float(data["o"]),
                "previous_close": previous_close,
            }
        except (TypeError, ValueError) as err:
            raise DataValidationError(
                provider=self.name,
                message="Finnhub quote contains invalid numeric values",
            ) from err

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> dict[str, Any]:
        """Fetch raw data from Finnhub API."""
        # Map frequency to Finnhub resolution
        finnhub_resolution = self.RESOLUTION_MAP.get(frequency.lower(), frequency)
        if finnhub_resolution not in ["1", "5", "15", "30", "60", "D", "W", "M"]:
            raise DataValidationError(
                provider="finnhub",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.RESOLUTION_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Convert dates to unix timestamps
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
        except ValueError as err:
            raise DataValidationError(
                provider="finnhub",
                message=f"Invalid date format. Use YYYY-MM-DD. Error: {err}",
                field="start/end",
                value=f"{start}/{end}",
            ) from err

        # Build request
        endpoint = f"{self.base_url}/stock/candle"
        params = {
            "symbol": symbol.upper(),
            "resolution": finnhub_resolution,
            "from": start_ts,
            "to": end_ts,
            "token": self.api_key,
        }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="finnhub", retry_after=60.0)
            if response.status_code in [401, 403]:
                raise AuthenticationError(provider="finnhub", message="Invalid API key")
            if response.status_code == 404:
                raise DataNotAvailableError(provider="finnhub", symbol=symbol)
            if response.status_code != 200:
                raise NetworkError(
                    provider="finnhub", message=f"HTTP {response.status_code}: {response.text}"
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(provider="finnhub", message="Failed to parse JSON") from err

            # Check status field
            if data.get("s") == "no_data":
                raise SymbolNotFoundError(provider="finnhub", symbol=symbol)
            if data.get("s") == "error":
                raise ProviderError(provider="finnhub", message=f"API error: {data.get('error')}")

            # Verify we have data
            if not data.get("c") or not data.get("t"):
                raise SymbolNotFoundError(provider="finnhub", symbol=symbol)

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
            DataValidationError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="finnhub", message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: dict[str, Any], symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        try:
            # Finnhub returns arrays: c, h, l, o, t, v, s
            df = pl.DataFrame(
                {
                    "timestamp": pl.Series(raw_data["t"]).cast(pl.Int64),
                    "open": raw_data["o"],
                    "high": raw_data["h"],
                    "low": raw_data["l"],
                    "close": raw_data["c"],
                    "volume": raw_data["v"],
                }
            )

            # Convert unix timestamp to datetime
            df = df.with_columns(
                pl.from_epoch("timestamp", time_unit="s")
                .dt.replace_time_zone("UTC")
                .alias("timestamp")
            )

            # Ensure numeric columns are float
            df = df.with_columns(
                [
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )

            # Sort and add symbol
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # Reorder columns
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as err:
            raise DataValidationError(
                provider="finnhub", message=f"Failed to transform data for {symbol}"
            ) from err

    def fetch_financials(
        self,
        symbol: str,
        statement: str = "income",
        period: str = "annual",
    ) -> pl.DataFrame:
        """Fetch standardized financial statements from Finnhub."""
        try:
            statement_type = normalize_statement_type(statement)
            period_type = normalize_period_type(period)
        except ValueError as err:
            raise DataValidationError("finnhub", str(err)) from err

        if period_type == "ttm":
            raise DataValidationError(
                "finnhub",
                "Finnhub financial statements support annual and quarterly periods",
                field="period",
                value=period,
            )

        data = self._request_json(
            "/stock/financials",
            params={
                "symbol": symbol.upper(),
                "statement": self.FINANCIAL_STATEMENT_MAP[statement_type],
                "freq": self.FINANCIAL_PERIOD_MAP[period_type],
                "token": self.api_key,
            },
            symbol=symbol,
        )
        rows = records_to_financials_rows(
            data.get("financials", []),
            symbol=symbol,
            provider=self.name,
            statement_type=statement_type,
            period_type=period_type,
            source="stock/financials",
        )
        return rows_to_financials_frame(rows)

    def fetch_company_metrics(
        self,
        symbol: str,
        *,
        metric_group: str = "all",
        metrics: list[str] | None = None,
    ) -> pl.DataFrame:
        """Fetch numeric company metrics from Finnhub."""
        data = self._request_json(
            "/stock/metric",
            params={"symbol": symbol.upper(), "metric": metric_group, "token": self.api_key},
            symbol=symbol,
        )
        rows = numeric_mapping_to_metric_rows(
            data.get("metric", {}),
            symbol=symbol,
            provider=self.name,
            source="stock/metric",
            metrics=metrics,
        )
        return rows_to_company_metrics_frame(rows)

    def _request_json(
        self,
        path: str,
        params: dict[str, Any],
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Fetch JSON from a Finnhub endpoint."""
        endpoint = f"{self.base_url}{path}"
        try:
            self.rate_limiter.acquire(blocking=True)
            response = self.session.get(endpoint, params=params)
            if response.status_code == 429:
                raise RateLimitError(provider="finnhub", retry_after=60.0)
            if response.status_code in [401, 403]:
                raise AuthenticationError(provider="finnhub", message="Invalid API key")
            if response.status_code == 404:
                raise DataNotAvailableError(provider="finnhub", symbol=symbol or path)
            if response.status_code != 200:
                raise NetworkError(
                    provider="finnhub", message=f"HTTP {response.status_code}: {response.text}"
                )
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(provider="finnhub", message="Failed to parse JSON") from err
            if isinstance(data, dict) and data.get("s") == "error":
                raise ProviderError(provider="finnhub", message=f"API error: {data.get('error')}")
            if isinstance(data, dict) and data.get("error"):
                raise ProviderError(provider="finnhub", message=f"API error: {data['error']}")
            if not isinstance(data, dict):
                raise DataValidationError("finnhub", "Expected JSON object response")
            return data
        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
            DataValidationError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="finnhub", message=f"Request failed: {endpoint}") from err

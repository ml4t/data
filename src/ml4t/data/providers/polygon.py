"""Massive data provider with Polygon compatibility.

Massive.com, formerly Polygon.io, offers comprehensive financial data across
multiple asset classes:
- Stocks (equities)
- Options
- Futures
- Cryptocurrencies
- Forex

Rate Limits:
    - Free tier: 5 requests/minute
    - Paid tiers: higher or unlimited usage depending on product

Authentication:
    Requires API key from https://massive.com/
    Set MASSIVE_API_KEY or POLYGON_API_KEY, or pass api_key explicitly.

Example:
    >>> from ml4t.data.providers.polygon import MassiveProvider
    >>> provider = MassiveProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> futures = provider.fetch_ohlcv("ESM6", "2024-01-01", "2024-01-31", asset_class="futures")
    >>> provider.close()
"""

import os
import warnings
from typing import Any, ClassVar, Literal

import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ml4t.data.core.exceptions import (
    AuthenticationError,
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
    nested_mapping_to_metric_rows,
    normalize_period_type,
    normalize_statement_type,
    records_to_financials_rows,
    rows_to_company_metrics_frame,
    rows_to_financials_frame,
)

MassiveAssetClass = Literal["stocks", "options", "futures", "crypto", "forex"]
SUPPORTED_MASSIVE_ASSET_CLASSES = {"stocks", "options", "futures", "crypto", "forex"}


class MassiveProvider(BaseProvider):
    """Massive data provider.

    Supports OHLCV bars for stocks, options, futures, cryptocurrencies, and forex.

    ``POLYGON_API_KEY`` remains supported because existing Polygon.io keys and
    accounts continue to work after the Massive.com rebrand.
    """

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.massive.com"
    LEGACY_BASE_URL: ClassVar[str] = "https://api.polygon.io"
    API_KEY_ENV_VARS: ClassVar[tuple[str, str]] = ("MASSIVE_API_KEY", "POLYGON_API_KEY")
    BASE_URL_ENV_VARS: ClassVar[tuple[str, str]] = ("MASSIVE_BASE_URL", "POLYGON_BASE_URL")

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (5, 60.0)  # Basic free tier

    # Map frequencies to Massive/Polygon timespans.
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "day": "day",
        "daily": "day",
        "1d": "day",
        "1day": "day",
        "week": "week",
        "weekly": "week",
        "1w": "week",
        "1week": "week",
        "month": "month",
        "monthly": "month",
        "1M": "month",
        "1month": "month",
        "hour": "hour",
        "hourly": "hour",
        "1h": "hour",
        "1hour": "hour",
        "minute": "minute",
        "1m": "minute",
        "1minute": "minute",
    }

    FINANCIAL_STATEMENT_SECTION_MAP: ClassVar[dict[StatementType, tuple[str, ...]]] = {
        "income": ("income_statement", "income"),
        "balance": ("balance_sheet", "balance"),
        "cashflow": ("cash_flow_statement", "cash_flow", "cashflow"),
    }

    FINANCIAL_PERIOD_MAP: ClassVar[dict[PeriodType, str]] = {
        "annual": "annual",
        "quarterly": "quarterly",
    }

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: tuple[int, float] | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Massive provider.

        Args:
            api_key: Massive/Polygon API key. If None, reads MASSIVE_API_KEY first,
                then POLYGON_API_KEY for backward compatibility.
            rate_limit: Optional custom rate limit (calls, period_seconds)
            base_url: Optional API base URL. Defaults to https://api.massive.com.

        Raises:
            AuthenticationError: If API key is not provided
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise AuthenticationError(
                provider=self.name,
                message=(
                    "API key required. Set MASSIVE_API_KEY or POLYGON_API_KEY, "
                    "or pass api_key. Get your key at: https://massive.com/"
                ),
            )

        self.base_url = self._resolve_base_url(base_url)

        self.logger.info(
            "Initialized Massive provider",
            base_url=self.base_url,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "massive"

    @classmethod
    def _resolve_api_key(cls) -> str | None:
        """Resolve API key from Massive first, then legacy Polygon env vars."""
        for env_var in cls.API_KEY_ENV_VARS:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key
        return None

    @classmethod
    def _resolve_base_url(cls, base_url: str | None = None) -> str:
        """Resolve API base URL, preserving a legacy override path."""
        resolved = base_url
        if resolved is None:
            for env_var in cls.BASE_URL_ENV_VARS:
                env_value = os.getenv(env_var)
                if env_value:
                    resolved = env_value
                    break
        return (resolved or cls.DEFAULT_BASE_URL).rstrip("/")

    @staticmethod
    def _infer_asset_class(symbol: str, asset_class: MassiveAssetClass | None = None) -> str:
        """Infer Massive asset class from symbol prefix when not explicit."""
        if asset_class:
            if asset_class not in SUPPORTED_MASSIVE_ASSET_CLASSES:
                supported = ", ".join(sorted(SUPPORTED_MASSIVE_ASSET_CLASSES))
                raise DataValidationError(
                    provider="massive",
                    message=f"Unsupported asset_class '{asset_class}'. Supported: {supported}",
                    field="asset_class",
                    value=asset_class,
                )
            return asset_class

        upper = symbol.upper()
        if upper.startswith("X:"):
            return "crypto"
        if upper.startswith("C:"):
            return "forex"
        if upper.startswith("O:"):
            return "options"
        if upper.startswith("F:") or upper.startswith("FUT:"):
            return "futures"
        return "stocks"

    @staticmethod
    def _api_symbol(symbol: str, asset_class: str) -> str:
        """Normalize user-facing symbol prefixes for Massive endpoints."""
        upper = symbol.upper()
        if asset_class == "futures":
            if upper.startswith("FUT:"):
                return upper[4:]
            if upper.startswith("F:"):
                return upper[2:]
        return upper

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError)),
        reraise=True,
    )
    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: MassiveAssetClass | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars across Massive-supported asset classes.

        Args:
            symbol: Massive ticker. Prefixes are supported for disambiguation:
                ``X:BTCUSD`` for crypto, ``C:EURUSD`` for forex, ``O:...`` for
                options, and ``F:ESM6`` or ``asset_class="futures"`` for futures.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            frequency: Bar frequency, such as ``daily``, ``1h``, or ``1m``.
            asset_class: Optional explicit asset class. Required for ambiguous
                futures tickers without an ``F:`` prefix.

        Returns:
            DataFrame with canonical OHLCV columns.
        """
        self._validate_inputs(symbol, start, end, frequency)
        self._acquire_rate_limit()

        def _fetch_and_process() -> pl.DataFrame:
            raw_data = self._fetch_raw_data(symbol, start, end, frequency, asset_class=asset_class)
            data = self._transform_data(raw_data, symbol)
            return self._validate_ohlcv(data, self.name)

        return self._with_circuit_breaker(_fetch_and_process)

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: MassiveAssetClass | None = None,
    ) -> pl.DataFrame:
        """Async wrapper around fetch_ohlcv that preserves Massive options."""
        import asyncio

        if asset_class is None:
            return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency)
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency, asset_class)

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "day",
        asset_class: MassiveAssetClass | None = None,
    ) -> dict[str, Any]:
        """Fetch raw data from Massive API."""
        timespan = self.FREQUENCY_MAP.get(frequency.lower())
        if not timespan:
            raise DataValidationError(
                provider=self.name,
                message=f"Unsupported frequency '{frequency}'. Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        resolved_asset_class = self._infer_asset_class(symbol, asset_class)
        api_symbol = self._api_symbol(symbol, resolved_asset_class)
        if resolved_asset_class == "futures":
            endpoint = f"{self.base_url}/futures/v1/aggs/{api_symbol}"
            params = {
                "multiplier": 1,
                "timespan": timespan,
                "from": start,
                "to": end,
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key,
            }
        else:
            endpoint = (
                f"{self.base_url}/v2/aggs/ticker/{api_symbol}/range/1/{timespan}/{start}/{end}"
            )
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key,
            }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 401:
                raise AuthenticationError(
                    provider=self.name,
                    message="Invalid API key. Get your key at: https://massive.com/",
                )
            elif response.status_code == 429:
                raise RateLimitError(provider=self.name, retry_after=60.0)
            elif response.status_code != 200:
                raise NetworkError(
                    provider=self.name,
                    message=f"API error (HTTP {response.status_code}): {response.text}",
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider=self.name, message="Failed to parse JSON response"
                ) from err

            # Check for API-level errors
            if data.get("status") == "ERROR":
                error_msg = data.get("error", "Unknown error")
                # Check if it's a symbol not found error
                if "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
                    raise SymbolNotFoundError(
                        provider=self.name, symbol=symbol, details={"error": error_msg}
                    )
                raise ProviderError(provider=self.name, message=f"API error: {error_msg}")

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider=self.name, message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: dict[str, Any], symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        if not raw_data.get("results"):
            self.logger.warning(f"No data returned for {symbol}")
            raise SymbolNotFoundError(
                provider=self.name,
                symbol=symbol,
                details={"message": "No results returned from API"},
            )

        try:
            df = pl.DataFrame(raw_data["results"])

            # Rename columns
            df = df.rename(
                {
                    "t": "timestamp_ms",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )

            # Convert timestamp from milliseconds to datetime
            df = df.with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp"))
            df = df.drop("timestamp_ms")

            # Convert OHLCV columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort, add symbol, and select in standard order
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as err:
            raise DataValidationError(
                provider=self.name, message=f"Failed to transform data for {symbol}"
            ) from err

    def fetch_financials(
        self,
        symbol: str,
        statement: str = "income",
        period: str = "annual",
        limit: int = 100,
    ) -> pl.DataFrame:
        """Fetch Massive stock financial statements in canonical long form."""
        try:
            statement_type = normalize_statement_type(statement)
            period_type = normalize_period_type(period)
        except ValueError as err:
            raise DataValidationError(self.name, str(err)) from err

        if period_type == "ttm":
            raise DataValidationError(
                self.name,
                "Massive financial statements support annual and quarterly periods",
                field="period",
                value=period,
            )

        data = self._request_json(
            "/stocks/financials/v1/financials",
            {
                "ticker": symbol.upper(),
                "timeframe": self.FINANCIAL_PERIOD_MAP[period_type],
                "limit": limit,
            },
        )
        rows: list[dict[str, Any]] = []
        for record in data.get("results", []):
            if not isinstance(record, dict):
                continue
            financials = record.get("financials", {})
            if not isinstance(financials, dict):
                continue
            section = self._pick_statement_section(financials, statement_type)
            if not section:
                continue
            combined = {
                **section,
                "end_date": record.get("end_date"),
                "filing_date": record.get("filing_date"),
                "fiscal_period": record.get("fiscal_period"),
                "fiscal_year": record.get("fiscal_year"),
            }
            rows.extend(
                records_to_financials_rows(
                    [combined],
                    symbol=symbol,
                    provider=self.name,
                    statement_type=statement_type,
                    period_type=period_type,
                    source="stocks/financials/v1/financials",
                )
            )
        return rows_to_financials_frame(rows)

    def fetch_company_metrics(
        self,
        symbol: str,
        *,
        limit: int = 100,
        metrics: list[str] | None = None,
    ) -> pl.DataFrame:
        """Fetch Massive stock fundamental ratios in canonical long form."""
        data = self._request_json(
            "/stocks/financials/v1/ratios",
            {"ticker": symbol.upper(), "limit": limit},
        )
        rows: list[dict[str, Any]] = []
        for record in data.get("results", []):
            if not isinstance(record, dict):
                continue
            rows.extend(
                nested_mapping_to_metric_rows(
                    record,
                    symbol=symbol,
                    provider=self.name,
                    period=record.get("fiscal_period"),
                    as_of=record.get("end_date"),
                    source="stocks/financials/v1/ratios",
                    metrics=metrics,
                )
            )
        return rows_to_company_metrics_frame(rows)

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch JSON from a Massive endpoint."""
        endpoint = f"{self.base_url}{path}"
        request_params = {**params, "apiKey": self.api_key}
        try:
            self.rate_limiter.acquire(blocking=True)
            response = self.session.get(endpoint, params=request_params)
            if response.status_code == 401:
                raise AuthenticationError(
                    provider=self.name,
                    message="Invalid API key. Get your key at: https://massive.com/",
                )
            if response.status_code == 429:
                raise RateLimitError(provider=self.name, retry_after=60.0)
            if response.status_code != 200:
                raise NetworkError(
                    provider=self.name,
                    message=f"API error (HTTP {response.status_code}): {response.text}",
                )
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider=self.name, message="Failed to parse JSON response"
                ) from err
            if isinstance(data, dict) and data.get("status") == "ERROR":
                raise ProviderError(provider=self.name, message=f"API error: {data.get('error')}")
            if not isinstance(data, dict):
                raise DataValidationError(self.name, "Expected JSON object response")
            return data
        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataValidationError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider=self.name, message=f"Request failed: {endpoint}") from err

    @classmethod
    def _pick_statement_section(
        cls,
        financials: dict[str, Any],
        statement_type: StatementType,
    ) -> dict[str, Any]:
        for key in cls.FINANCIAL_STATEMENT_SECTION_MAP[statement_type]:
            section = financials.get(key)
            if isinstance(section, dict):
                return section
        return {}


class PolygonProvider(MassiveProvider):
    """Deprecated compatibility alias for Polygon.io integrations.

    Use ``MassiveProvider`` for new code.
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: tuple[int, float] | None = None,
        base_url: str | None = None,
    ) -> None:
        warnings.warn(
            "PolygonProvider is deprecated; use MassiveProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(api_key=api_key, rate_limit=rate_limit, base_url=base_url)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "polygon"

"""FRED (Federal Reserve Economic Data) provider.

FRED provides free access to over 800,000 economic time series from
national, international, public, and private sources.

API Documentation: https://fred.stlouisfed.org/docs/api/fred/

Rate Limits:
- 120 requests per minute (we use 100 to be safe)
- No daily limit

Key Series Examples:
- VIXCLS: VIX volatility index (daily)
- UNRATE: Unemployment rate (monthly)
- GDP: Gross Domestic Product (quarterly)
- DGS10: 10-Year Treasury yield (daily)
- CPIAUCSL: Consumer Price Index (monthly)
- SP500: S&P 500 index (daily)

Point-in-Time Support:
- FRED data often gets revised after initial release
- Use vintage_date parameter to get data as it was known at a specific time
- Essential for backtesting macro-based strategies

Example:
    >>> from ml4t.data.providers.fred import FREDProvider
    >>> provider = FREDProvider(api_key="your_key")  # or set FRED_API_KEY env var
    >>> data = provider.fetch_ohlcv("VIXCLS", "2024-01-01", "2024-01-31")
    >>> # Point-in-time: What was GDP known to be on 2024-01-15?
    >>> gdp = provider.fetch_ohlcv("GDP", "2023-01-01", "2023-12-31",
    ...                            vintage_date="2024-01-15")
    >>> provider.close()
"""

import os
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
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class FREDProvider(BaseProvider):
    """FRED economic data provider.

    Provides access to Federal Reserve Economic Data (FRED) with support for:
    - 800,000+ economic time series
    - Point-in-time data via vintage dates
    - Multiple frequencies (daily, weekly, monthly, quarterly, annual)
    - Automatic handling of missing values

    The OHLCV schema is used for compatibility with other providers:
    - open = high = low = close = economic value
    - volume = 1.0 (placeholder, not meaningful for economic data)

    Rate Limits:
    - 120 requests per minute (provider uses 100 to be safe)
    """

    # Conservative rate limit: 100 req/min (FRED allows 120)
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (100, 60.0)

    # FRED API base URL
    BASE_URL: ClassVar[str] = "https://api.stlouisfed.org/fred"

    # Map common frequency names to FRED frequency codes
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "d",
        "1d": "d",
        "day": "d",
        "weekly": "w",
        "1w": "w",
        "week": "w",
        "monthly": "m",
        "1M": "m",
        "month": "m",
        "quarterly": "q",
        "1Q": "q",
        "quarter": "q",
        "annual": "a",
        "1Y": "a",
        "yearly": "a",
        "year": "a",
    }

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: tuple[int, float] | None = None,
    ):
        """Initialize FRED provider.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var).
                     Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="fred",
                message="FRED API key required. Set FRED_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html",
            )

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info("Initialized FRED provider")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "fred"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        vintage_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw data from FRED API.

        Args:
            symbol: FRED series ID (e.g., "VIXCLS", "UNRATE", "GDP")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly, quarterly, annual)
            vintage_date: Optional point-in-time date. If provided, returns data
                         as it was known on this date (handles revisions).

        Returns:
            List of observation dictionaries from FRED API
        """
        series_id = symbol.upper()

        # Map frequency if provided
        freq_code = self.FREQUENCY_MAP.get(frequency.lower())
        if not freq_code:
            raise DataValidationError(
                provider="fred",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Build request parameters
        endpoint = f"{self.BASE_URL}/series/observations"
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "frequency": freq_code,
        }

        # Point-in-time support via realtime_start/end
        if vintage_date:
            params["realtime_start"] = vintage_date
            params["realtime_end"] = vintage_date

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="fred", retry_after=60.0)
            if response.status_code in [400, 401, 403]:
                # Parse error message from FRED
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error_message", "Authentication failed")
                except Exception:
                    error_msg = f"HTTP {response.status_code}"

                if "api_key" in error_msg.lower():
                    raise AuthenticationError(provider="fred", message=error_msg)
                raise ProviderError(provider="fred", message=error_msg)
            if response.status_code == 404:
                raise DataNotAvailableError(
                    provider="fred",
                    symbol=series_id,
                    start=start,
                    end=end,
                )
            if response.status_code != 200:
                raise NetworkError(
                    provider="fred",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            # Parse JSON response
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider="fred",
                    message="Failed to parse JSON response",
                ) from err

            # Check for API-level errors
            if "error_code" in data or "error_message" in data:
                error_msg = data.get("error_message", "Unknown FRED API error")
                if "Bad Request" in error_msg and "series_id" in error_msg.lower():
                    raise DataNotAvailableError(
                        provider="fred",
                        symbol=series_id,
                        details={"error": error_msg},
                    )
                raise ProviderError(provider="fred", message=error_msg)

            # Extract observations
            observations = data.get("observations", [])

            return observations

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
            raise NetworkError(
                provider="fred",
                message=f"Request failed for series {series_id}",
            ) from err

    def _transform_data(self, raw_data: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
        """Transform raw FRED API response to Polars DataFrame.

        Maps economic value to OHLCV schema:
        - open = high = low = close = value
        - volume = 1.0 (placeholder)

        Handles FRED's missing value indicator "." by converting to null.

        Args:
            raw_data: List of observation dictionaries from FRED API
            symbol: Series ID for labeling

        Returns:
            Polars DataFrame with OHLCV schema
        """
        if not raw_data:
            return self._create_empty_dataframe()

        try:
            # Convert to DataFrame
            df = pl.DataFrame(raw_data)

            # FRED returns "date" and "value" columns
            # Convert date to datetime
            df = df.with_columns(pl.col("date").str.to_date().cast(pl.Datetime).alias("timestamp"))

            # Handle missing values: FRED uses "." for missing data
            # Replace "." with null first, then cast to float
            df = df.with_columns(pl.col("value").replace(".", None).cast(pl.Float64).alias("value"))

            # Map to OHLCV schema (all prices equal for economic data)
            df = df.with_columns(
                [
                    pl.col("value").alias("open"),
                    pl.col("value").alias("high"),
                    pl.col("value").alias("low"),
                    pl.col("value").alias("close"),
                    pl.lit(1.0).alias("volume"),
                    pl.lit(symbol.upper()).alias("symbol"),
                ]
            )

            # Drop intermediate columns and select final schema
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            # Sort by timestamp and remove any duplicates
            df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

            return df

        except Exception as err:
            raise DataValidationError(
                provider="fred",
                message=f"Failed to transform data for {symbol}",
            ) from err

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    def _validate_response(self, df: pl.DataFrame) -> pl.DataFrame:
        """Override base validation to skip OHLC invariant checks.

        Economic data has identical OHLC values, so the standard
        high >= low, high >= open, etc. checks would be meaningless.
        We still validate required columns and handle duplicates.

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame
        """
        # Handle empty responses
        if df.is_empty():
            self.logger.info(
                "Provider returned empty DataFrame - no data available for requested range"
            )
            return df

        # Check required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise DataValidationError(self.name, f"Missing required column: {col}")

        # Skip OHLC invariant validation (values are identical for economic data)

        # Sort and deduplicate
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        vintage_date: str | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a FRED series.

        The OHLCV schema maps economic values as:
        - open = high = low = close = economic value
        - volume = 1.0 (placeholder)

        Args:
            symbol: FRED series ID (e.g., "VIXCLS", "UNRATE", "GDP")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly, quarterly, annual)
            vintage_date: Optional point-in-time date. If provided, returns data
                         as it was known on this date (handles data revisions).

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> provider = FREDProvider()
            >>> # Daily VIX data
            >>> vix = provider.fetch_ohlcv("VIXCLS", "2024-01-01", "2024-06-30")
            >>>
            >>> # Monthly unemployment with point-in-time
            >>> unrate = provider.fetch_ohlcv(
            ...     "UNRATE", "2020-01-01", "2024-01-01",
            ...     frequency="monthly",
            ...     vintage_date="2024-03-15"
            ... )
        """
        series_id = symbol.upper()

        self.logger.info(
            f"Fetching {frequency} OHLCV",
            symbol=series_id,
            start=start,
            end=end,
            vintage_date=vintage_date,
        )

        # Validate inputs
        self._validate_inputs(symbol, start, end, frequency)

        # Acquire rate limit
        self._acquire_rate_limit()

        # Fetch and transform
        raw_data = self._fetch_raw_data(symbol, start, end, frequency, vintage_date=vintage_date)
        df = self._transform_data(raw_data, symbol)

        # Validate (skips OHLC invariant checks)
        df = self._validate_response(df)

        self.logger.info(f"Fetched {len(df)} records", symbol=series_id)

        return df

    def fetch_series_metadata(self, series_id: str) -> dict[str, Any]:
        """Fetch metadata for a FRED series.

        Args:
            series_id: FRED series ID (e.g., "VIXCLS", "GDP")

        Returns:
            Dictionary with series metadata including:
            - id: Series ID
            - title: Human-readable title
            - frequency: Data frequency (Daily, Monthly, Quarterly, etc.)
            - units: Unit of measurement
            - seasonal_adjustment: Seasonal adjustment type
            - last_updated: Last update timestamp
            - observation_start: First available date
            - observation_end: Last available date

        Example:
            >>> provider = FREDProvider()
            >>> meta = provider.fetch_series_metadata("VIXCLS")
            >>> print(meta['title'])
            'CBOE Volatility Index: VIX'
            >>> print(meta['frequency'])
            'Daily, Close'
        """
        series_id = series_id.upper()

        endpoint = f"{self.BASE_URL}/series"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        self._acquire_rate_limit()

        try:
            response = self.session.get(endpoint, params=params)

            if response.status_code != 200:
                if response.status_code == 404:
                    raise DataNotAvailableError(
                        provider="fred",
                        symbol=series_id,
                        details={"error": "Series not found"},
                    )
                raise NetworkError(
                    provider="fred",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()

            # Check for errors
            if "error_message" in data:
                raise ProviderError(
                    provider="fred",
                    message=data["error_message"],
                )

            # Extract series info
            series_list = data.get("seriess", [])
            if not series_list:
                raise DataNotAvailableError(
                    provider="fred",
                    symbol=series_id,
                    details={"error": "No series data returned"},
                )

            return series_list[0]

        except (DataNotAvailableError, NetworkError, ProviderError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="fred",
                message=f"Failed to fetch metadata for {series_id}",
            ) from err

    def fetch_multiple(
        self,
        series_ids: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        forward_fill: bool = True,
    ) -> pl.DataFrame:
        """Fetch multiple FRED series and align to common frequency.

        Useful for macro factor analysis where you need to combine
        daily, monthly, and quarterly series.

        Args:
            series_ids: List of FRED series IDs
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Target frequency for all series (daily, weekly, monthly).
                       All series are fetched at this frequency and aligned.
            forward_fill: Whether to forward-fill lower frequency data

        Returns:
            Wide-format DataFrame with columns:
            - timestamp
            - {series_id}_close for each series

        Example:
            >>> provider = FREDProvider()
            >>> # Combine daily VIX with monthly unemployment
            >>> df = provider.fetch_multiple(
            ...     ["VIXCLS", "UNRATE"],
            ...     "2024-01-01", "2024-06-30",
            ...     frequency="daily",
            ...     forward_fill=True
            ... )
            >>> print(df.columns)
            ['timestamp', 'VIXCLS_close', 'UNRATE_close']
        """
        if not series_ids:
            raise DataValidationError(
                provider="fred",
                message="series_ids cannot be empty",
                field="series_ids",
            )

        self.logger.info(
            "Fetching multiple series",
            series_ids=series_ids,
            start=start,
            end=end,
            frequency=frequency,
        )

        # Fetch each series
        dataframes: list[pl.DataFrame] = []
        for series_id in series_ids:
            try:
                df = self.fetch_ohlcv(series_id, start, end, frequency=frequency)
                # Rename close column to series-specific name
                df = df.select(
                    [
                        "timestamp",
                        pl.col("close").alias(f"{series_id.upper()}_close"),
                    ]
                )
                dataframes.append(df)
            except DataNotAvailableError:
                self.logger.warning(f"No data available for series {series_id}")
                continue

        if not dataframes:
            raise DataNotAvailableError(
                provider="fred",
                symbol=",".join(series_ids),
                start=start,
                end=end,
                details={"error": "No data available for any requested series"},
            )

        # Join all series on timestamp
        result = dataframes[0]
        for df in dataframes[1:]:
            result = result.join(df, on="timestamp", how="outer_coalesce")

        # Sort by timestamp
        result = result.sort("timestamp")

        # Forward fill if requested
        if forward_fill:
            # Forward fill all columns except timestamp
            value_cols = [c for c in result.columns if c != "timestamp"]
            result = result.with_columns([pl.col(c).forward_fill() for c in value_cols])

        self.logger.info(f"Fetched {len(dataframes)} series with {len(result)} aligned rows")

        return result

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Closed FRED API client")

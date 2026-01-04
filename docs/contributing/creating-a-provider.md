# Creating a New Provider

This guide walks you through creating a new data provider for ML4T Data. We'll use a fictional "Stooq" provider as an example.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Guide](#step-by-step-guide)
- [Testing Your Provider](#testing-your-provider)
- [Documentation](#documentation)
- [Checklist](#checklist)

## Overview

A ML4T Data provider is a class that inherits from `BaseProvider` and implements methods to:
1. Fetch raw data from an API
2. Transform it into standardized Polars DataFrames
3. Handle errors, rate limiting, and retries

**Time estimate:** 2-4 hours for a basic provider

## Prerequisites

- Familiarity with Python async/await
- Understanding of the target API
- API key from the data provider (if required)
- Development environment set up (see [CONTRIBUTING.md](../CONTRIBUTING.md))

## Step-by-Step Guide

### Step 1: Research the API

**Before writing code**, understand the API:

```python
# Questions to answer:
# 1. Authentication: API key? OAuth? Bearer token?
# 2. Base URL: https://api.example.com
# 3. Endpoints: /v1/ohlcv, /v1/quote, etc.
# 4. Rate limits: requests per minute/day
# 5. Data format: JSON? CSV? XML?
# 6. Date format: YYYY-MM-DD? Unix timestamp?
# 7. Response structure: nested? flat?
# 8. Error codes: 429 for rate limit? 401 for auth?
```

**Example** - Stooq API research:
```python
# Stooq API (fictional example):
# Base URL: https://stooq.com/q/d/l/
# Authentication: None (public API)
# Format: CSV
# Parameters:
#   s = symbol (e.g., AAPL.US)
#   d1 = start date (YYYYMMDD)
#   d2 = end date (YYYYMMDD)
#   i = frequency (d=daily, w=weekly, m=monthly)
# Rate limit: 10 requests/minute
# Response: CSV with Date,Open,High,Low,Close,Volume
```

### Step 2: Create Provider File

Create `src/ml4t-data/providers/stooq.py`:

```python
"""Stooq data provider.

Stooq provides free market data for international exchanges.

API Documentation: https://stooq.com/
Features:
- No API key required
- 50+ international exchanges
- Daily/weekly/monthly OHLCV data
- CSV format

Rate Limits:
- 10 requests per minute
- No daily limit

Example:
    >>> from ml4t.data.providers.stooq import StooqProvider
    >>> provider = StooqProvider()
    >>> data = provider.fetch_ohlcv("AAPL.US", "2024-01-01", "2024-01-31")
    >>> provider.close()
"""

import os
from datetime import datetime
from typing import Any, ClassVar, Optional

import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class StooqProvider(BaseProvider):
    """Stooq data provider for international equities.

    Attributes:
        base_url: API base URL
    """

    # Rate limit: 10 requests/minute
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (10, 60.0)

    CIRCUIT_BREAKER_CONFIG: ClassVar[dict[str, Any]] = {
        "failure_threshold": 3,
        "reset_timeout": 300.0,  # 5 minutes
    }

    # Map frequency strings to Stooq codes
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
    }

    def __init__(
        self,
        rate_limit: Optional[tuple[int, float]] = None,
        session_config: Optional[dict[str, Any]] = None,
        circuit_breaker_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize Stooq provider.

        Args:
            rate_limit: Custom rate limiting override
            session_config: HTTP session configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.base_url = "https://stooq.com/q/d/l/"

        super().__init__(
            rate_limit=rate_limit,
            session_config=session_config,
            circuit_breaker_config=circuit_breaker_config,
        )

        self.logger.info("Initialized Stooq provider")

    def name(self) -> str:
        """Return provider name."""
        return "stooq"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> str:
        """Fetch raw CSV data from Stooq API.

        Args:
            symbol: Stock symbol with exchange (e.g., "AAPL.US")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly)

        Returns:
            Raw CSV string

        Raises:
            DataValidationError: If frequency is invalid
            NetworkError: If request fails
        """
        # 1. Validate frequency
        freq_code = self.FREQUENCY_MAP.get(frequency.lower())
        if not freq_code:
            raise DataValidationError(
                provider="stooq",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # 2. Convert dates to Stooq format (YYYYMMDD)
        start_formatted = start.replace("-", "")
        end_formatted = end.replace("-", "")

        # 3. Build URL with parameters
        params = {
            "s": symbol.upper(),
            "d1": start_formatted,
            "d2": end_formatted,
            "i": freq_code,
        }

        try:
            # 4. Apply rate limiting
            self.rate_limiter.acquire(blocking=True)

            # 5. Make HTTP request
            response = self.session.get(self.base_url, params=params)

            # 6. Check for errors
            if response.status_code != 200:
                raise NetworkError(
                    provider="stooq",
                    message=f"HTTP {response.status_code}: {response.text}",
                )

            # 7. Return raw data
            csv_data = response.text

            # 8. Basic validation
            if not csv_data or len(csv_data) < 50:
                raise DataNotAvailableError(
                    provider="stooq",
                    symbol=symbol,
                    start=start,
                    end=end,
                    frequency=frequency,
                )

            return csv_data

        except (NetworkError, DataNotAvailableError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="stooq",
                message=f"Request failed: {self.base_url}",
            ) from err

    def _transform_data(
        self,
        raw_data: str,
        symbol: str,
    ) -> pl.DataFrame:
        """Transform CSV data to Polars DataFrame.

        Args:
            raw_data: Raw CSV string
            symbol: Symbol for logging

        Returns:
            Polars DataFrame with OHLCV data

        Raises:
            DataValidationError: If transformation fails
        """
        try:
            # 1. Parse CSV into Polars DataFrame
            df = pl.read_csv(raw_data.encode())

            # 2. Rename columns to standard format
            # Stooq CSV: Date,Open,High,Low,Close,Volume
            df = df.rename({
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # 3. Convert date string to datetime
            df = df.with_columns(
                pl.col("date").str.to_date("%Y%m%d").cast(pl.Datetime).alias("timestamp")
            )

            # 4. Drop original date column
            df = df.drop("date")

            # 5. Convert numeric columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # 6. Add symbol column
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # 7. Sort by timestamp
            df = df.sort("timestamp")

            # 8. Select final columns in standard order
            df = df.select([
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ])

            return df

        except Exception as err:
            raise DataValidationError(
                provider="stooq",
                message=f"Failed to transform data for {symbol}",
            ) from err

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Closed Stooq API client")
```

**Key points:**
1. **Module docstring** - Explain what the provider does
2. **Inherit from BaseProvider** - Gets rate limiting, circuit breaker, etc.
3. **Class variables** - `DEFAULT_RATE_LIMIT`, `CIRCUIT_BREAKER_CONFIG`, mappings
4. **`__init__`** - Set up provider-specific config, call `super().__init__()`
5. **`name()`** - Return lowercase provider name
6. **`_fetch_raw_data()`** - Get data from API, minimal processing
7. **`_transform_data()`** - Convert to standard DataFrame format
8. **`close()`** - Clean up resources

### Step 3: Handle Edge Cases

Common edge cases to handle:

```python
def _fetch_raw_data(self, symbol, start, end, frequency="daily"):
    # ... existing code ...

    # Edge case: Empty response
    if not csv_data or csv_data.strip() == "":
        raise DataNotAvailableError(
            provider="stooq",
            symbol=symbol,
            message="API returned empty response"
        )

    # Edge case: Error message in response
    if "error" in csv_data.lower():
        raise ProviderError(
            provider="stooq",
            message=f"API error: {csv_data[:100]}"
        )

    # Edge case: Invalid symbol (returns HTML instead of CSV)
    if csv_data.startswith("<!DOCTYPE") or csv_data.startswith("<html"):
        raise DataNotAvailableError(
            provider="stooq",
            symbol=symbol,
            message="Invalid symbol or no data available"
        )

    return csv_data
```

### Step 4: Add Updater Class (Optional but Recommended)

```python
class StooqUpdater:
    """Incremental updater for Stooq provider."""

    def __init__(self, provider: StooqProvider, storage):
        """Initialize updater.

        Args:
            provider: StooqProvider instance
            storage: Storage backend instance
        """
        self.provider = provider
        self.storage = storage
        self.provider_name = "stooq"
        self.logger = provider.logger.bind(updater=True)

    def update_symbol(
        self,
        symbol: str,
        start_time: str | None = None,
        end_time: str | None = None,
        frequency: str = "daily",
        incremental: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """Update symbol with new data.

        Args:
            symbol: Symbol to update
            start_time: Start date (YYYY-MM-DD), defaults to 90 days ago
            end_time: End date (YYYY-MM-DD), defaults to today
            frequency: Data frequency
            incremental: If True, only fetch new data since last update
            dry_run: If True, don't actually store data

        Returns:
            Result dictionary with status and metrics
        """
        from datetime import timedelta

        # Default date range
        if not end_time:
            end_time = datetime.now().strftime("%Y-%m-%d")
        if not start_time:
            start_time = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        try:
            # Fetch data
            df = self.provider.fetch_ohlcv(symbol, start_time, end_time, frequency)

            if df.is_empty():
                return {
                    "success": True,
                    "symbol": symbol,
                    "records_fetched": 0,
                    "message": "No data available",
                }

            # Store data if not dry run
            if not dry_run:
                self.storage.write(df, symbol, self.provider_name)

            return {
                "success": True,
                "symbol": symbol,
                "records_fetched": len(df),
                "start_date": start_time,
                "end_date": end_time,
            }

        except Exception as err:
            self.logger.error(f"Failed to update {symbol}: {err}")
            return {
                "success": False,
                "symbol": symbol,
                "error": str(err),
            }
```

### Step 5: Register Provider

Add to `src/ml4t-data/providers/__init__.py`:

```python
# In the imports section
try:
    from ml4t.data.providers.stooq import StooqProvider, StooqUpdater
except ImportError:
    StooqProvider = None  # type: ignore
    StooqUpdater = None  # type: ignore

# In __all__ list
__all__ = [
    # ... existing providers ...
    "StooqProvider",
    "StooqUpdater",
]

# In docstring
"""
Available Providers:
    - StooqProvider: International equities (free, no API key)
    - ...
"""
```

## Testing Your Provider

### Step 6: Create Integration Tests

Create `tests/integration/test_stooq.py`:

```python
"""Integration tests for Stooq provider (real API calls).

These tests verify the Stooq provider works correctly with actual API calls.

Requirements:
    - No API key needed (public API)
    - Rate limit: 10 requests/minute

Test Coverage:
    - Stock daily OHLCV data (AAPL.US)
    - Multiple frequencies (daily, weekly, monthly)
    - International exchanges
    - Error handling
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.stooq import StooqProvider


@pytest.fixture
def provider():
    """Create Stooq provider."""
    provider = StooqProvider()
    yield provider
    provider.close()


class TestStooqProvider:
    """Test Stooq provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized."""
        provider = StooqProvider()
        assert provider.name() == "stooq"
        provider.close()

    def test_fetch_ohlcv_daily(self, provider):
        """Test fetching daily stock data with real API call."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="AAPL.US",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["symbol"].dtype == pl.String
        assert df["open"].dtype == pl.Float64

        # Verify OHLCV relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()

        print(f"✅ Fetched {len(df)} rows of AAPL.US daily data")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv(
                symbol="INVALID_XYZ.US",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid symbol correctly raises DataNotAvailableError")
```

### Step 7: Run Tests

```bash
# Run integration tests
pytest tests/integration/test_stooq.py -v -s

# Check coverage
pytest tests/integration/test_stooq.py --cov=src/ml4t-data/providers/stooq
```

## Documentation

### Step 8: Create Example Script

Create `examples/stooq_example.py`:

```python
"""Example usage of Stooq provider for international equities."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import StooqProvider, StooqUpdater
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


def example_basic_usage():
    """Example 1: Basic OHLCV fetching."""
    print("\n" + "=" * 60)
    print("  Example 1: Basic OHLCV Data Fetching")
    print("=" * 60 + "\n")

    provider = StooqProvider()

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch AAPL data
        print(f"Fetching AAPL.US data from {start} to {end}...\n")
        df = provider.fetch_ohlcv("AAPL.US", start, end, frequency="daily")

        print(f"Fetched {len(df)} records:")
        print(df.head())
        print(f"\nPrice range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    finally:
        provider.close()


def example_international_exchanges():
    """Example 2: Fetching from multiple international exchanges."""
    print("\n" + "=" * 60)
    print("  Example 2: International Exchanges")
    print("=" * 60 + "\n")

    provider = StooqProvider()

    stocks = [
        ("AAPL.US", "Apple (US)"),
        ("VOD.UK", "Vodafone (London)"),
        ("BMW.DE", "BMW (Germany)"),
    ]

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        for symbol, name in stocks:
            print(f"\n📊 {name} ({symbol}):")
            df = provider.fetch_ohlcv(symbol, start, end)
            print(f"  Records: {len(df)}")
            print(f"  Last close: {df['close'][-1]:.2f}")

    finally:
        provider.close()


if __name__ == "__main__":
    example_basic_usage()
    example_international_exchanges()
```

### Step 9: Update README

Add provider to README.md:

```markdown
## Phase X Providers

| Provider | Asset Classes | API Key | Rate Limit | Best For |
|----------|--------------|---------|------------|----------|
| **Stooq** | International Equities | No | 10/min | Free international data |

### Stooq

```bash
# No API key needed!
```

```python
from ml4t.data.providers import StooqProvider

provider = StooqProvider()
data = provider.fetch_ohlcv("AAPL.US", "2024-01-01", "2024-01-31")
```

**Why Stooq:**
- Free international equities
- 50+ exchanges worldwide
- No registration required
- Simple CSV format
```

## Checklist

Before submitting your provider, verify:

### Code Quality
- [ ] Inherits from `BaseProvider`
- [ ] Implements `name()`, `_fetch_raw_data()`, `_transform_data()`
- [ ] Has rate limiting configured
- [ ] Has circuit breaker configured
- [ ] Uses ml4t-data exception classes
- [ ] Has type hints on all public methods
- [ ] Has Google-style docstrings
- [ ] Passes `ruff` linting
- [ ] Passes `mypy` type checking

### Testing
- [ ] Integration tests created
- [ ] Tests cover happy path
- [ ] Tests cover error cases
- [ ] Tests respect rate limits
- [ ] 80%+ code coverage
- [ ] All tests passing

### Documentation
- [ ] Module docstring complete
- [ ] Class docstring complete
- [ ] Method docstrings complete
- [ ] Example script created
- [ ] README updated
- [ ] Provider registered in `__init__.py`

### Optional but Recommended
- [ ] Updater class implemented
- [ ] Updater tests created
- [ ] Multiple frequencies supported
- [ ] Custom parameters documented

## Common Pitfalls

❌ **Forgetting to call `super().__init__()`**
```python
def __init__(self):
    # Missing super().__init__() - rate limiting won't work!
    pass
```

✅ **Always call super()**
```python
def __init__(self):
    super().__init__(rate_limit=..., ...)
```

❌ **Not handling rate limits**
```python
def _fetch_raw_data(self, ...):
    # Makes request without rate limiting - will get banned!
    response = self.session.get(url)
```

✅ **Use rate limiter**
```python
def _fetch_raw_data(self, ...):
    self.rate_limiter.acquire(blocking=True)
    response = self.session.get(url)
```

❌ **Generic exceptions**
```python
if error:
    raise Exception("Something went wrong")
```

✅ **Specific exceptions**
```python
if response.status_code == 429:
    raise RateLimitError(provider="stooq", retry_after=60.0)
```

## Getting Help

- Check existing providers for patterns: `src/ml4t-data/providers/`
- Use the template: `provider_template/`
- Read [extending_ml4t-data.md](extending_ml4t-data.md) for architecture details
- Ask in GitHub Discussions

---

**Ready to contribute?** Follow this guide, and submit a PR! 🚀

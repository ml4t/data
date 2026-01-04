# Extending ML4T Data - Architecture Guide

This document explains ML4T Data's architecture, design patterns, and extension points for contributors.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Abstractions](#core-abstractions)
- [Design Patterns](#design-patterns)
- [Extension Points](#extension-points)
- [Best Practices](#best-practices)

## Architecture Overview

### High-Level Structure

```
ML4T Data Architecture
├── Providers (Data Acquisition)
│   ├── BaseProvider (Template Method)
│   ├── Rate Limiting (Global)
│   ├── Circuit Breakers
│   └── Retry Logic
│
├── Storage (Data Persistence)
│   ├── Hive-Partitioned Parquet
│   ├── Metadata Tracking
│   └── Incremental Updates
│
├── Validation (Data Quality)
│   ├── Schema Validation
│   ├── OHLCV Invariants
│   └── Anomaly Detection
│
└── API (Access Layer)
    ├── REST API (FastAPI)
    ├── WebSocket (Real-time)
    └── CLI Interface
```

### Design Philosophy

1. **Separation of Concerns** - Each layer has single responsibility
2. **Template Method Pattern** - Common workflow, provider-specific implementation
3. **Type Safety** - Full type hints, mypy strict mode
4. **Performance First** - Polars, not pandas
5. **Production Ready** - Error handling, retry logic, monitoring

## Core Abstractions

### 1. BaseProvider

**Location**: `src/ml4t-data/providers/base.py`

The `BaseProvider` class is the foundation of all data providers. It implements the **Template Method** pattern.

#### Key Features

```python
class BaseProvider(ABC):
    """Base class for all data providers.

    Provides:
    - Rate limiting (global, per-provider)
    - Circuit breaker pattern
    - Retry logic with exponential backoff
    - HTTP session management
    - Structured logging
    """

    # Class variables (override in subclass)
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (60, 60.0)
    CIRCUIT_BREAKER_CONFIG: ClassVar[dict[str, Any]] = {...}

    # Abstract methods (must implement)
    @abstractmethod
    def name(self) -> str:
        """Return provider name (lowercase)."""

    @abstractmethod
    def _fetch_raw_data(self, symbol, start, end, frequency) -> Any:
        """Fetch raw data from API."""

    @abstractmethod
    def _transform_data(self, raw_data, symbol) -> pl.DataFrame:
        """Transform to standard DataFrame."""
```

#### Template Method

The `fetch_ohlcv()` method defines the workflow:

```python
@circuit_breaker(...)
@retry(...)
def fetch_ohlcv(self, symbol, start, end, frequency="daily"):
    """Template method - don't override!

    Workflow:
    1. Log request
    2. Call _fetch_raw_data() (provider-specific)
    3. Call _transform_data() (provider-specific)
    4. Validate output
    5. Return standardized DataFrame
    """
    raw_data = self._fetch_raw_data(symbol, start, end, frequency)
    df = self._transform_data(raw_data, symbol)
    return df
```

**Key Insight**: Subclasses implement `_fetch_raw_data()` and `_transform_data()`, but never override `fetch_ohlcv()`. This ensures consistent error handling, logging, and retries across all providers.

### 2. Rate Limiting

**Location**: `src/ml4t-data/utils/global_rate_limit.py`

Rate limiting is **global** (not per-instance) to prevent parallel requests from violating API limits.

```python
from ml4t.data.utils.global_rate_limit import global_rate_limit_manager

# Get rate limiter for provider (shared across instances)
rate_limiter = global_rate_limit_manager.get_rate_limiter(
    provider_name="tiingo",
    max_calls=1000,
    period=86400.0,  # 1 day in seconds
)

# In _fetch_raw_data():
self.rate_limiter.acquire(blocking=True)  # Blocks until token available
response = self.session.get(url)
```

**Why Global?**
- Multiple provider instances share the same API quota
- Prevents race conditions in parallel execution
- Respects daily/monthly limits across application

### 3. Circuit Breaker

**Location**: `src/ml4t-data/providers/base.py:CircuitBreaker`

Implements the Circuit Breaker pattern to prevent cascading failures.

**States:**
- **CLOSED** - Normal operation
- **OPEN** - Too many failures, block all requests
- **HALF_OPEN** - Test if service recovered

```python
class CircuitBreaker:
    """Circuit breaker with exponential backoff.

    Configuration:
    - failure_threshold: Number of failures before opening (default: 5)
    - reset_timeout: Seconds before attempting reset (default: 300)
    - expected_exception: Exceptions that trigger circuit breaker
    """
```

**Usage:**
```python
@circuit_breaker(
    failure_threshold=3,
    reset_timeout=600.0,
    expected_exception=NetworkError
)
def fetch_ohlcv(...):
    ...
```

### 4. Exception Hierarchy

**Location**: `src/ml4t-data/core/exceptions.py`

All exceptions inherit from `ProviderError`:

```
ProviderError (base)
├── AuthenticationError (401, invalid API key)
├── RateLimitError (429, too many requests)
├── DataNotAvailableError (404, no data for symbol)
├── DataValidationError (invalid parameters or data)
├── NetworkError (connection issues, timeouts)
└── CircuitBreakerOpenError (circuit breaker tripped)
```

**Best Practice:**
```python
# Good - specific exception
if response.status_code == 429:
    raise RateLimitError(
        provider="tiingo",
        retry_after=60.0,
        message="Rate limit exceeded"
    )

# Bad - generic exception
if error:
    raise Exception("Error occurred")  # ❌ Don't do this
```

### 5. Standard DataFrame Schema

All providers must return DataFrames with this schema:

```python
{
    "timestamp": pl.Datetime,      # UTC datetime
    "symbol": pl.String,           # Uppercase symbol
    "open": pl.Float64,            # Opening price
    "high": pl.Float64,            # High price
    "low": pl.Float64,             # Low price
    "close": pl.Float64,           # Closing price (adjusted)
    "volume": pl.Float64,          # Trading volume
}
```

**Optional columns:**
- `adj_open`, `adj_high`, `adj_low`, `adj_close` - Unadjusted prices
- `dividend` - Dividend amount
- `split_factor` - Stock split factor

## Design Patterns

### 1. Template Method Pattern

**Used in**: `BaseProvider.fetch_ohlcv()`

**Purpose**: Define skeleton algorithm, let subclasses fill in steps

**Example:**
```python
# BaseProvider (template)
def fetch_ohlcv(self, ...):  # DON'T override
    raw_data = self._fetch_raw_data(...)  # Override this
    df = self._transform_data(...)         # Override this
    return df

# TiingoProvider (concrete)
def _fetch_raw_data(self, ...):
    # Tiingo-specific fetching
    return tiingo_data

def _transform_data(self, ...):
    # Tiingo-specific transformation
    return dataframe
```

### 2. Strategy Pattern

**Used in**: Provider selection

**Purpose**: Swap providers at runtime

**Example:**
```python
# Different strategies for different use cases
providers = {
    "crypto": CoinGeckoProvider(),
    "stocks": TiingoProvider(api_key="..."),
    "futures": DataBentoProvider(api_key="..."),
}

# Select strategy at runtime
provider = providers[asset_class]
data = provider.fetch_ohlcv(symbol, start, end)
```

### 3. Factory Pattern

**Used in**: Storage backend selection

**Purpose**: Create objects without specifying exact class

**Example:**
```python
def create_storage(storage_type: str, config: StorageConfig):
    if storage_type == "hive":
        return HiveStorage(config)
    elif storage_type == "flat":
        return FlatStorage(config)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
```

### 4. Decorator Pattern

**Used in**: `@circuit_breaker`, `@retry`

**Purpose**: Add behavior to methods without modifying them

**Example:**
```python
@circuit_breaker(failure_threshold=5, reset_timeout=300.0)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
def fetch_ohlcv(self, ...):
    # Method gets automatic retry and circuit breaker
    ...
```

## Extension Points

### 1. Adding a New Provider

**See**: [Creating a Provider](creating_a_provider.md)

**Steps:**
1. Inherit from `BaseProvider`
2. Implement `name()`, `_fetch_raw_data()`, `_transform_data()`
3. Configure rate limiting and circuit breaker
4. Add integration tests
5. Register in `__init__.py`

**Example:**
```python
class NewProvider(BaseProvider):
    DEFAULT_RATE_LIMIT = (10, 60.0)  # 10/min

    def name(self) -> str:
        return "newprovider"

    def _fetch_raw_data(self, symbol, start, end, frequency):
        # Your fetching logic
        return raw_data

    def _transform_data(self, raw_data, symbol):
        # Your transformation logic
        return polars_dataframe
```

### 2. Adding a New Storage Backend

**Location**: `src/ml4t-data/storage/`

**Interface**: `StorageProtocol` (in `protocols.py`)

**Required methods:**
- `write(df, symbol, provider)` - Store data
- `read(symbol, start, end, provider)` - Retrieve data
- `list_symbols(provider)` - List available symbols
- `get_metadata(symbol, provider)` - Get metadata

**Example:**
```python
class S3Storage:
    """Store data in AWS S3."""

    def write(self, df, symbol, provider):
        # Upload to S3
        ...

    def read(self, symbol, start, end, provider):
        # Download from S3
        ...
```

### 3. Adding a New Validator

**Location**: `src/ml4t-data/validation/`

**Interface**: `BaseValidator`

**Example:**
```python
class CustomValidator(BaseValidator):
    """Custom validation rules."""

    def validate(self, df: pl.DataFrame) -> ValidationResult:
        errors = []

        # Check custom rules
        if (df["close"] < 0).any():
            errors.append("Negative prices found")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )
```

### 4. Adding a New CLI Command

**Location**: `src/ml4t-data/cli.py`

**Framework**: Click

**Example:**
```python
@cli.command()
@click.argument("symbol")
@click.option("--provider", default="tiingo")
def analyze(symbol, provider):
    """Analyze symbol data quality."""
    # Your command logic
    ...
```

## Best Practices

### 1. Error Handling

**Always use ml4t-data exceptions:**
```python
from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError,
)

# Good
if response.status_code == 404:
    raise DataNotAvailableError(
        provider="myProvider",
        symbol=symbol,
        start=start,
        end=end,
    )

# Bad
if response.status_code == 404:
    raise Exception("Not found")  # ❌
```

### 2. Logging

**Use structured logging:**
```python
self.logger.info(
    "Fetching data",
    symbol=symbol,
    start=start,
    end=end,
    provider=self.name(),
)

# Not this:
print(f"Fetching {symbol} from {start} to {end}")  # ❌
```

### 3. Type Hints

**Use type hints everywhere:**
```python
def _transform_data(
    self,
    raw_data: list[dict[str, Any]],  # ✅
    symbol: str,
) -> pl.DataFrame:  # ✅
    ...

# Not this:
def _transform_data(self, raw_data, symbol):  # ❌
    ...
```

### 4. Testing

**Test real API calls:**
```python
@pytest.mark.integration
def test_fetch_real_data(provider):
    """Test with actual API call."""
    df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    assert len(df) > 0
    assert all(df["high"] >= df["low"])
```

### 5. Documentation

**Document everything:**
```python
def _fetch_raw_data(self, symbol, start, end, frequency):
    """Fetch raw data from API.

    This method makes HTTP requests to the provider's API
    and returns the raw response with minimal processing.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        frequency: Data frequency (daily, weekly, monthly)

    Returns:
        Raw API response (dict or str depending on format)

    Raises:
        RateLimitError: If rate limit exceeded
        DataNotAvailableError: If no data found
        NetworkError: If request fails
    """
```

## Architecture Decisions

### Why Polars Instead of Pandas?

**Performance**: 10-100x faster for common operations
**Memory**: More efficient memory usage
**Type Safety**: Better type system
**Future**: Better maintained, more modern

### Why Global Rate Limiting?

**Problem**: Multiple instances of same provider share API quota
**Solution**: Global rate limiter ensures compliance
**Trade-off**: Slightly more complex, but much safer

### Why Template Method Pattern?

**Problem**: Ensure consistent error handling across providers
**Solution**: Template method in base class
**Benefit**: Add circuit breaker/retry once, applies everywhere

### Why Circuit Breaker?

**Problem**: Cascading failures when API goes down
**Solution**: Fail fast, recover gracefully
**Benefit**: Better user experience, prevents wasted API calls

## Contributing Guidelines

1. **Read first**: [CONTRIBUTING.md](../CONTRIBUTING.md)
2. **Use templates**: `provider_template/` directory
3. **Follow patterns**: Study existing providers
4. **Test thoroughly**: Integration tests required
5. **Document well**: Code should explain itself

---

**Questions?** Open a GitHub Discussion or issue!

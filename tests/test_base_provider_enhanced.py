"""Test enhanced BaseProvider architecture with Template Method pattern."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import polars as pl
import pytest
from tenacity import wait_none

from ml4t.data.core.exceptions import (
    AuthenticationError,
    CircuitBreakerOpenError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider, CircuitBreaker


class MockProvider(BaseProvider):
    """Test implementation of BaseProvider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._should_fail = False
        self._fail_count = 0
        self._raw_data = None

    @property
    def name(self) -> str:
        return "mock"

    def _fetch_raw_data(self, symbol: str, start: str, end: str, frequency: str):
        """Mock raw data fetch."""
        if self._should_fail:
            self._fail_count += 1
            raise NetworkError("mock", "Simulated network error")

        # Return mock raw data
        return self._raw_data or {
            "data": [
                {"time": 1640995200, "o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0, "v": 1000},
                {"time": 1641081600, "o": 103.0, "h": 108.0, "l": 101.0, "c": 106.0, "v": 1500},
            ]
        }

    def _transform_data(self, raw_data, symbol: str) -> pl.DataFrame:
        """Transform mock data to standard schema."""
        records = []
        for item in raw_data["data"]:
            records.append(
                {
                    "timestamp": datetime.fromtimestamp(item["time"], tz=UTC),
                    "open": item["o"],
                    "high": item["h"],
                    "low": item["l"],
                    "close": item["c"],
                    "volume": item["v"],
                }
            )
        return pl.DataFrame(records)


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_success_path(self):
        """Test successful calls don't affect circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3)

        def mock_func():
            return "success"

        result = breaker.call(mock_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker counts failures."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=Exception)

        def failing_func():
            raise Exception("Mock failure")

        # First failure
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 1
        assert breaker.state == "CLOSED"

        # Second failure
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 2
        assert breaker.state == "CLOSED"

        # Third failure - should open circuit
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 3
        assert breaker.state == "OPEN"

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker prevents calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=Exception)

        def failing_func():
            raise Exception("Mock failure")

        # Cause circuit to open
        with pytest.raises(Exception):
            breaker.call(failing_func)
        with pytest.raises(Exception):
            breaker.call(failing_func)

        assert breaker.state == "OPEN"

        # Now calls should be prevented
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(failing_func)

    def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker recovery on success."""
        now = [100.0]
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=10.0,
            expected_exception=Exception,
            clock=lambda: now[0],
        )

        def failing_func():
            raise Exception("Mock failure")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(failing_func)
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.state == "OPEN"

        now[0] += 10.0

        # Successful call should reset circuit
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_call_async_success_and_failure_accounting(self):
        """call_async mirrors call: successes pass through, failures count."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=Exception)

        async def success_func():
            return "success"

        async def failing_func():
            raise Exception("Mock failure")

        assert await breaker.call_async(success_func) == "success"
        assert breaker.state == "CLOSED"

        with pytest.raises(Exception, match="Mock failure"):
            await breaker.call_async(failing_func)
        assert breaker.failure_count == 1

        with pytest.raises(Exception, match="Mock failure"):
            await breaker.call_async(failing_func)
        assert breaker.state == "OPEN"

        # While open, calls are refused before executing the function.
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call_async(success_func)

    @pytest.mark.asyncio
    async def test_call_async_half_open_recovery(self):
        """After the reset timeout, call_async probes HALF_OPEN and recovers."""
        now = [100.0]
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=10.0,
            expected_exception=Exception,
            clock=lambda: now[0],
        )

        async def failing_func():
            raise Exception("Mock failure")

        async def success_func():
            return "success"

        with pytest.raises(Exception, match="Mock failure"):
            await breaker.call_async(failing_func)
        assert breaker.state == "OPEN"

        now[0] += 10.0

        assert await breaker.call_async(success_func) == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    @pytest.mark.parametrize(
        "error",
        [
            SymbolNotFoundError("mock", "BAD"),
            AuthenticationError("mock"),
            DataNotAvailableError("mock", "BAD"),
            DataValidationError("mock", "invalid response"),
            RateLimitError("mock", retry_after=1.0),
            ValueError("invalid request"),
        ],
    )
    def test_default_circuit_ignores_permanent_and_client_errors(self, error):
        """Only transient service failures affect the default circuit."""
        breaker = CircuitBreaker(failure_threshold=1)

        def fail():
            raise error

        with pytest.raises(type(error)):
            breaker.call(fail)

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
        assert breaker.ignored_failure_count == 1

    def test_default_circuit_counts_transient_network_errors(self):
        """A classified transport failure opens the default circuit."""
        breaker = CircuitBreaker(failure_threshold=1)

        def fail():
            raise NetworkError("mock", "connection reset")

        with pytest.raises(NetworkError):
            breaker.call(fail)

        assert breaker.state == "OPEN"
        assert breaker.failure_count == 1


class TestBaseProvider:
    """Test enhanced BaseProvider functionality."""

    def test_provider_initialization(self):
        """Test provider initialization with default settings."""
        provider = MockProvider()
        assert provider.name == "mock"
        assert hasattr(provider, "rate_limiter")
        assert hasattr(provider, "session")
        assert hasattr(provider, "circuit_breaker")

    def test_provider_custom_rate_limit(self):
        """Test provider with custom rate limiting."""
        provider = MockProvider(rate_limit=(30, 60.0))
        assert provider.rate_limiter is not None

    def test_context_manager(self):
        """Test provider as context manager."""
        with MockProvider() as provider:
            assert provider.name == "mock"
        # Session should be closed after exit

    def test_fetch_ohlcv_success(self):
        """Test successful OHLCV data fetch."""
        provider = MockProvider()

        df = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03", "daily")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert df.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.schema == {
            "timestamp": pl.Datetime("us", "UTC"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
        assert df["symbol"].to_list() == ["AAPL", "AAPL"]

        # Check data is sorted by timestamp
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_input_validation(self):
        """Test input parameter validation."""
        provider = MockProvider()

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            provider.fetch_ohlcv("", "2022-01-01", "2022-01-03")

        # Test invalid date format
        with pytest.raises(ValueError, match="Invalid date format"):
            provider.fetch_ohlcv("AAPL", "invalid-date", "2022-01-03")

        # Test start >= end
        with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
            provider.fetch_ohlcv("AAPL", "2022-01-03", "2022-01-01")

    def test_data_validation_empty_dataframe(self):
        """A provider's untyped empty sentinel becomes a typed canonical response."""
        provider = MockProvider()
        provider._raw_data = {"data": []}

        result = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")
        assert result.is_empty()
        assert result.schema == {
            "timestamp": pl.Datetime("us", "UTC"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }

    def test_data_validation_missing_columns(self):
        """Test validation with missing required columns."""
        provider = MockProvider()
        provider._raw_data = {
            "data": [{"time": 1640995200, "o": 100.0}]  # Missing h, l, c, v
        }

        def bad_transform(raw_data, symbol):
            return pl.DataFrame([{"timestamp": datetime.now(), "open": 100.0}])

        provider._transform_data = bad_transform

        with pytest.raises(DataValidationError, match="Missing required column"):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

    def test_data_validation_ohlc_invariants_strict_by_default(self):
        """Invalid OHLC data fails instead of becoming a successful empty response."""
        provider = MockProvider()

        def bad_transform(raw_data, symbol):
            return pl.DataFrame(
                [
                    {
                        "timestamp": datetime.now(UTC),
                        "open": 100.0,
                        "high": 90.0,  # High < open (invalid)
                        "low": 85.0,
                        "close": 95.0,
                        "volume": 1000,
                    }
                ]
            )

        provider._transform_data = bad_transform

        with pytest.raises(DataValidationError, match="invalid OHLC relationships"):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

    def test_data_validation_ohlc_invariants_drop_mode_is_explicit(self):
        """Callers can explicitly opt into dropping invalid OHLC rows."""
        provider = MockProvider()
        provider.ohlc_mode = "drop"

        def bad_transform(raw_data, symbol):
            return pl.DataFrame(
                [
                    {
                        "timestamp": datetime.now(UTC),
                        "open": 100.0,
                        "high": 90.0,
                        "low": 85.0,
                        "close": 95.0,
                        "volume": 1000,
                    }
                ]
            )

        provider._transform_data = bad_transform

        df = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")
        assert df.is_empty()
        assert df.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

    def test_data_validation_canonicalizes_safe_types_and_column_order(self):
        """Successful responses have one schema even when an adapter uses safe source types."""
        provider = MockProvider()

        def reordered_transform(raw_data, symbol):
            return pl.DataFrame(
                {
                    "source_id": [7],
                    "volume": [1000],
                    "close": [103],
                    "low": [98],
                    "high": [105],
                    "open": [100],
                    "timestamp": [datetime(2022, 1, 1, tzinfo=UTC)],
                }
            )

        provider._transform_data = reordered_transform
        df = provider.fetch_ohlcv("aapl", "2022-01-01", "2022-01-03")

        assert df.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_id",
        ]
        assert df["symbol"].to_list() == ["AAPL"]
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        for column in ["open", "high", "low", "close", "volume"]:
            assert df[column].dtype == pl.Float64

    @pytest.mark.parametrize(
        ("mutator", "message"),
        [
            (
                lambda frame: frame.with_columns(pl.col("timestamp").dt.replace_time_zone(None)),
                "timezone-aware",
            ),
            (lambda frame: frame.with_columns(pl.col("open").cast(pl.String)), "numeric"),
            (lambda frame: frame.with_columns(pl.lit(float("nan")).alias("close")), "finite"),
            (lambda frame: frame.with_columns(pl.lit(None).alias("volume")), "null"),
            (lambda frame: frame.with_columns(pl.lit(-1.0).alias("volume")), "negative"),
            (lambda frame: frame.with_columns(pl.lit("MSFT").alias("symbol")), "requested symbol"),
        ],
    )
    def test_data_validation_rejects_malformed_successes(self, mutator, message):
        """Malformed source data cannot be logged and returned as a successful fetch."""
        provider = MockProvider()
        valid = pl.DataFrame(
            {
                "timestamp": [datetime(2022, 1, 1, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [103.0],
                "volume": [1000.0],
            }
        )
        invalid = mutator(valid)
        provider._transform_data = lambda raw_data, symbol: invalid

        with pytest.raises(DataValidationError, match=message):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

    def test_circuit_breaker_integration(self, monkeypatch):
        """Repeated transient failures open the provider's service circuit."""
        provider = MockProvider(circuit_breaker_config={"failure_threshold": 3})
        monkeypatch.setattr(provider.fetch_ohlcv.retry, "wait", wait_none())
        provider._should_fail = True

        with pytest.raises(NetworkError):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

        assert provider.circuit_breaker.state == "OPEN"
        assert provider.circuit_breaker.metrics["counted_failures"] == 3
        assert provider.circuit_breaker.metrics["opened"] == 1

        provider._should_fail = False
        with pytest.raises(CircuitBreakerOpenError):
            provider.fetch_ohlcv("GOOD", "2022-01-01", "2022-01-03")

    def test_retry_mechanism(self):
        """Test retry mechanism for transient failures."""
        provider = MockProvider()

        call_count = 0
        original_fetch = provider._fetch_raw_data

        def failing_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("mock", "Transient error")
            return original_fetch(*args, **kwargs)

        provider._fetch_raw_data = failing_fetch

        # Should succeed after 2 retries
        df = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")
        assert isinstance(df, pl.DataFrame)
        assert call_count == 3

    @pytest.mark.parametrize(
        "error",
        [
            SymbolNotFoundError("mock", "BAD"),
            AuthenticationError("mock"),
            DataNotAvailableError("mock", "BAD"),
            DataValidationError("mock", "invalid response"),
        ],
    )
    def test_permanent_error_never_blocks_later_valid_symbol(self, error):
        """One invalid request cannot make a healthy provider unavailable."""
        provider = MockProvider(circuit_breaker_config={"failure_threshold": 2})
        original_fetch = provider._fetch_raw_data

        def fail_bad_symbol(symbol, start, end, frequency):
            raise error

        provider._fetch_raw_data = fail_bad_symbol
        for _ in range(2):
            with pytest.raises(type(error)):
                provider.fetch_ohlcv("BAD", "2022-01-01", "2022-01-03")

        assert provider.circuit_breaker.state == "CLOSED"
        assert provider.circuit_breaker.failure_count == 0

        provider._fetch_raw_data = original_fetch
        result = provider.fetch_ohlcv("GOOD", "2022-01-01", "2022-01-03")
        assert result.height == 2
        assert result["symbol"].unique().to_list() == ["GOOD"]

    @pytest.mark.skip(
        reason="Limiter class doesn't exist - test needs rewrite for global_rate_limit_manager"
    )
    @patch("ml4t.data.providers.base.Limiter")
    def test_rate_limiting(self, mock_limiter_class):
        """Test rate limiting integration."""
        mock_limiter = Mock()
        mock_limiter_class.return_value = mock_limiter

        provider = MockProvider()
        provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

        # Verify rate limiter was called
        mock_limiter.try_acquire.assert_called_with("api_call")

    def test_duplicate_timestamp_rejected(self):
        """Conflicting duplicate bars are rejected instead of silently discarded."""
        provider = MockProvider()
        provider._raw_data = {
            "data": [
                {"time": 1640995200, "o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0, "v": 1000},
                {
                    "time": 1640995200,
                    "o": 101.0,
                    "h": 106.0,
                    "l": 99.0,
                    "c": 104.0,
                    "v": 1100,
                },  # Duplicate timestamp
                {"time": 1641081600, "o": 103.0, "h": 108.0, "l": 101.0, "c": 106.0, "v": 1500},
            ]
        }

        with pytest.raises(DataValidationError, match="duplicate"):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

    def test_backward_compatibility(self):
        """Test backward compatibility with Provider alias."""
        from ml4t.data.providers.base import Provider

        # Should be able to use Provider class
        assert Provider is not None
        assert issubclass(Provider, BaseProvider)


class TestProviderLogging:
    """Test provider logging functionality."""

    @pytest.mark.skip(reason="Structlog logger initialized at import time - mock timing issue")
    def test_structured_logging(self):
        """Test that providers use structured logging."""
        provider = MockProvider()

        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

            # Verify info logs were called
            assert mock_logger.info.call_count >= 2

            # Check log structure
            start_call = mock_logger.info.call_args_list[0]
            assert "Fetching OHLCV data" in start_call[0]
            assert "symbol" in start_call[1]
            assert "provider" in start_call[1]


if __name__ == "__main__":
    pytest.main([__file__])

"""Test enhanced BaseProvider architecture with Template Method pattern."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    CircuitBreakerOpenError,
    DataValidationError,
    NetworkError,
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
        breaker = CircuitBreaker(failure_threshold=3)

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
        breaker = CircuitBreaker(failure_threshold=2)

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
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

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

        # Wait for reset timeout
        import time

        time.sleep(0.2)

        # Successful call should reset circuit
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


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
        assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]

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
        """Test that empty DataFrame is valid when no data is available."""
        provider = MockProvider()
        provider._raw_data = {"data": []}

        # Empty DataFrame is now valid - provider logs info and returns empty
        result = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")
        assert result.is_empty()

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

    def test_data_validation_ohlc_invariants(self):
        """Test OHLC invariant validation."""
        provider = MockProvider()

        def bad_transform(raw_data, symbol):
            return pl.DataFrame(
                [
                    {
                        "timestamp": datetime.now(),
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

    @pytest.mark.skip(reason="Circuit breaker state pollution in parallel execution")
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with provider."""
        provider = MockProvider(circuit_breaker_config={"failure_threshold": 2})
        provider._should_fail = True

        # First failure
        with pytest.raises(NetworkError):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

        # Second failure should open circuit
        with pytest.raises(NetworkError):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

        # Third call should be prevented by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

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

    def test_duplicate_timestamp_removal(self):
        """Test that duplicate timestamps are removed."""
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

        df = provider.fetch_ohlcv("AAPL", "2022-01-01", "2022-01-03")

        # Should have only unique timestamps
        assert len(df) == 2
        timestamps = df["timestamp"].to_list()
        assert len(set(timestamps)) == len(timestamps)

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

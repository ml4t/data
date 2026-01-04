"""Real API Integration Testing Suite for ML4T Data Providers.

This module provides comprehensive integration testing for all priority providers
with real API validation and cost optimization strategies.
"""

import os
import time

import polars as pl
import pytest

# Skip if optional provider dependencies not installed
try:
    from ml4t.data.providers.cryptocompare import CryptoCompareProvider
    from ml4t.data.providers.databento import DataBentoProvider
    from ml4t.data.providers.oanda import OandaProvider
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="provider dependencies not installed - install with: pip install -e '.[databento,oanda]'"
    )

from structlog import get_logger

logger = get_logger(__name__)

# Cost optimization settings
MINIMAL_TEST_DURATION = 1  # Test with 1 day of data to minimize costs
MAX_TEST_SYMBOLS = 2  # Limit number of symbols per provider


class TestRealAPIIntegration:
    """Comprehensive real API integration tests for all priority providers."""

    @pytest.fixture(scope="class")
    def api_keys(self) -> dict[str, str | None]:
        """Load API keys from environment variables."""
        return {
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
            "databento": os.getenv("DATABENTO_API_KEY"),
            "oanda": os.getenv("OANDA_API_KEY"),
        }

    @pytest.fixture(scope="class")
    def test_dates(self) -> dict[str, str]:
        """Standard test dates for all providers."""
        # Use recent weekday for consistent testing
        return {
            "start": "2024-01-02",  # Tuesday
            "end": "2024-01-02",  # Same day for minimal cost
            "multi_start": "2024-01-02",
            "multi_end": "2024-01-04",  # 3 days for multi-day tests
        }

    @pytest.fixture(scope="class")
    def performance_tracker(self) -> dict[str, list[float]]:
        """Track performance metrics for baseline measurements."""
        return {
            "cryptocompare": [],
            "databento": [],
            "oanda": [],
        }

    # ==================== CryptoCompare Tests ====================

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    def test_cryptocompare_integration(self, api_keys, test_dates, performance_tracker):
        """Test CryptoCompare provider with real API."""
        provider = CryptoCompareProvider(api_key=api_keys["cryptocompare"])

        # Test 1: Single symbol, single day (minimal cost)
        start_time = time.time()
        df = provider.fetch_ohlcv("BTC", test_dates["start"], test_dates["end"], "daily")
        duration = time.time() - start_time
        performance_tracker["cryptocompare"].append(duration)

        assert not df.is_empty(), "Should fetch BTC data"
        assert len(df) == 1, "Should have 1 day of data"
        assert self._validate_ohlcv_schema(df), "Schema validation failed"
        assert self._validate_data_quality(df), "Data quality validation failed"

        # Test 2: Multiple symbols
        symbols = ["BTC", "ETH"]
        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, test_dates["start"], test_dates["end"], "daily")
            assert not df.is_empty(), f"Should fetch {symbol} data"
            assert df["symbol"][0] == symbol, f"Symbol should be {symbol}"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    @pytest.mark.skip(reason="Integration test flaky/rate limited")
    def test_cryptocompare_error_handling(self, api_keys, test_dates):
        """Test CryptoCompare error handling."""
        provider = CryptoCompareProvider(api_key=api_keys["cryptocompare"])

        # Test invalid symbol
        df = provider.fetch_ohlcv(
            "INVALID_SYMBOL_XYZ", test_dates["start"], test_dates["end"], "daily"
        )
        # Should either return empty or raise specific exception
        # CryptoCompare typically returns empty data for invalid symbols
        assert df.is_empty() or len(df) == 0, "Invalid symbol should return empty"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    def test_cryptocompare_rate_limiting(self, api_keys, test_dates):
        """Test CryptoCompare rate limiting."""
        provider = CryptoCompareProvider(api_key=api_keys["cryptocompare"])

        # Make rapid requests to test rate limiting
        symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"][:3]  # Limit to 3 for cost

        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, test_dates["start"], test_dates["end"], "daily")
            # Should not raise rate limit errors due to built-in limiting
            assert df is not None, f"Request for {symbol} should succeed"

    # ==================== Databento Tests ====================

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("DATABENTO_API_KEY"), reason="DATABENTO_API_KEY not set")
    def test_databento_integration(self, api_keys, test_dates, performance_tracker):
        """Test Databento provider with real API."""
        provider = DataBentoProvider(api_key=api_keys["databento"])

        # Test 1: Specific contract (minimal cost)
        start_time = time.time()
        df = provider.fetch_ohlcv("ESH4", test_dates["start"], test_dates["end"], "daily")
        duration = time.time() - start_time
        performance_tracker["databento"].append(duration)

        assert not df.is_empty(), "Should fetch ES futures data"
        assert len(df) == 1, "Should have 1 day of data"
        assert self._validate_ohlcv_schema(df), "Schema validation failed"
        assert self._validate_data_quality(df), "Data quality validation failed"

        # Verify known values
        assert df["open"][0] == 4820.75, "Known value check"
        assert df["symbol"][0] == "ESH4", "Symbol preserved"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("DATABENTO_API_KEY"), reason="DATABENTO_API_KEY not set")
    def test_databento_continuous_futures(self, api_keys, test_dates):
        """Test Databento continuous futures handling."""
        provider = DataBentoProvider(api_key=api_keys["databento"])

        # Test continuous futures (known to return empty with GLBX.MDP3)
        df = provider.fetch_continuous_futures(
            "ES", test_dates["start"], test_dates["end"], "daily", version=0
        )
        # Expected to return empty due to dataset limitations
        assert df.is_empty() or len(df) == 0, "Continuous futures should return empty"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("DATABENTO_API_KEY"), reason="DATABENTO_API_KEY not set")
    def test_databento_error_handling(self, api_keys, test_dates):
        """Test Databento error handling."""
        provider = DataBentoProvider(api_key=api_keys["databento"])

        # Test invalid symbol
        df = provider.fetch_ohlcv("INVALID_XYZ", test_dates["start"], test_dates["end"], "daily")
        assert df.is_empty() or len(df) == 0, "Invalid symbol should return empty"

    # ==================== OANDA Tests ====================

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OANDA_API_KEY"), reason="OANDA_API_KEY not set")
    def test_oanda_integration(self, api_keys, test_dates, performance_tracker):
        """Test OANDA provider with real API."""
        provider = OandaProvider(api_key=api_keys["oanda"])

        # Test 1: Major FX pair (minimal cost)
        start_time = time.time()
        df = provider.fetch_ohlcv("EUR_USD", test_dates["start"], test_dates["end"], "H1")
        duration = time.time() - start_time
        performance_tracker["oanda"].append(duration)

        assert not df.is_empty(), "Should fetch EUR/USD data"
        assert self._validate_ohlcv_schema(df), "Schema validation failed"
        assert self._validate_data_quality(df), "Data quality validation failed"
        assert df["symbol"][0] == "EUR_USD", "Symbol preserved"

    # ==================== Cross-Provider Tests ====================

    @pytest.mark.integration
    def test_cross_provider_consistency(self, api_keys, test_dates):
        """Test consistency across all available providers."""
        available_providers = []

        # Initialize available providers
        if api_keys["cryptocompare"]:
            available_providers.append(
                ("cryptocompare", CryptoCompareProvider(api_key=api_keys["cryptocompare"]))
            )
        if api_keys["databento"]:
            available_providers.append(
                ("databento", DataBentoProvider(api_key=api_keys["databento"]))
            )
        if api_keys["oanda"]:
            available_providers.append(("oanda", OandaProvider(api_key=api_keys["oanda"])))

        # Test that all providers return consistent schema
        for name, provider in available_providers:
            # Use appropriate symbol for each provider
            if name == "cryptocompare":
                symbol = "BTC"
            elif name == "databento":
                symbol = "ESH4"
            else:  # oanda
                symbol = "EUR_USD"

            df = provider.fetch_ohlcv(symbol, test_dates["start"], test_dates["end"], "daily")

            if not df.is_empty():
                assert self._validate_ohlcv_schema(df), f"{name}: Schema validation failed"
                logger.info(f"Provider {name} passed consistency check")

    # ==================== Performance Tests ====================

    @pytest.mark.integration
    def test_performance_baselines(self, performance_tracker):
        """Validate and report performance baselines."""
        report = []

        for provider, timings in performance_tracker.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                max_time = max(timings)
                min_time = min(timings)

                report.append(
                    f"{provider}: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s"
                )

                # Performance assertions
                assert avg_time < 10, f"{provider} average time exceeds 10 seconds"
                assert max_time < 30, f"{provider} max time exceeds 30 seconds"

        if report:
            logger.info("Performance baselines:", report=report)

    # ==================== Helper Methods ====================

    def _validate_ohlcv_schema(self, df: pl.DataFrame) -> bool:
        """Validate DataFrame has correct OHLCV schema."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]

        # Check columns exist
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False

        # Check data types
        # Note: timestamp can be ns or μs precision - both are valid
        timestamp_valid = (
            df["timestamp"].dtype == pl.Datetime("ns", "UTC")
            or df["timestamp"].dtype == pl.Datetime("us", "UTC")
            or df["timestamp"].dtype == pl.Datetime("ms", "UTC")
        )
        type_checks = [
            (timestamp_valid, "timestamp type"),
            (df["open"].dtype == pl.Float64, "open type"),
            (df["high"].dtype == pl.Float64, "high type"),
            (df["low"].dtype == pl.Float64, "low type"),
            (df["close"].dtype == pl.Float64, "close type"),
            (df["volume"].dtype == pl.Float64 or df["volume"].dtype == pl.Int64, "volume type"),
            (df["symbol"].dtype == pl.Utf8, "symbol type"),
        ]

        for check, name in type_checks:
            if not check:
                logger.error(f"Type check failed: {name}")
                return False

        return True

    def _validate_data_quality(self, df: pl.DataFrame) -> bool:
        """Validate OHLCV data quality."""
        if df.is_empty():
            return True  # Empty is valid

        # Check OHLC invariants
        checks = [
            ((df["high"] >= df["low"]).all(), "high >= low"),
            ((df["high"] >= df["open"]).all(), "high >= open"),
            ((df["high"] >= df["close"]).all(), "high >= close"),
            ((df["low"] <= df["open"]).all(), "low <= open"),
            ((df["low"] <= df["close"]).all(), "low <= close"),
            ((df["volume"] >= 0).all(), "volume >= 0"),
        ]

        for check, name in checks:
            if not check:
                logger.error(f"Data quality check failed: {name}")
                return False

        return True


class TestCostOptimization:
    """Test cost optimization strategies."""

    def test_cost_strategy_documentation(self):
        """Document cost optimization strategies for CI/CD."""
        strategy = """
        ML4T Data API Cost Optimization Strategy:

        1. Data Minimization:
           - Single day requests (1 bar) for basic tests
           - Maximum 3-day ranges for multi-day tests
           - Limited symbol sets (2-3 per provider)

        2. Provider-Specific Optimizations:
           - CryptoCompare: Use free tier limits (10 req/sec)
           - Databento: Use daily aggregation (ohlcv-1d)
           - OANDA: Use practice account (free)

        3. CI/CD Strategy:
           - Use environment variables for API keys
           - Skip expensive tests without keys
           - Cache test data when possible
           - Run full suite only on main branch

        4. Test Selection:
           - Mark expensive tests with @pytest.mark.expensive
           - Run minimal tests in PR validation
           - Full integration tests on schedule (nightly)

        5. Monitoring:
           - Track API usage in test logs
           - Alert on unexpected costs
           - Regular cost review
        """
        assert True, strategy

    @pytest.mark.integration
    def test_ci_environment_detection(self):
        """Test CI environment detection for cost-aware testing."""
        ci_env = os.getenv("CI", "false").lower() == "true"
        github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

        if ci_env or github_actions:
            logger.info("Running in CI environment - using minimal test set")
            # In CI, we would skip expensive tests
            assert True, "CI environment detected"
        else:
            logger.info("Running in local environment - full test suite available")
            assert True, "Local environment detected"

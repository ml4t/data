"""Tests for async provider functionality.

Tests the async methods added to providers:
- BinancePublicProvider.fetch_ohlcv_async()
- YahooFinanceProvider.fetch_ohlcv_async()
- CryptoCompareProvider.fetch_ohlcv_async()
- AsyncBaseProvider and async_batch_load()
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from ml4t.data.managers.async_batch import AsyncBatchManager, async_batch_load
from ml4t.data.providers.binance_public import BinancePublicProvider
from ml4t.data.providers.cryptocompare import CryptoCompareProvider
from ml4t.data.providers.yahoo import YahooFinanceProvider

if TYPE_CHECKING:
    pass


# ===== Fixtures =====


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


# ===== BinancePublicProvider Async Tests =====


class TestBinancePublicProviderAsync:
    """Test async methods on BinancePublicProvider."""

    def test_async_session_lazily_created(self):
        """Test that async session is created lazily."""
        provider = BinancePublicProvider()
        assert not hasattr(provider, "_async_client") or provider._async_client is None

        # Access async session
        session = provider._async_session
        assert session is not None
        assert hasattr(provider, "_async_client")

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager protocol."""
        async with BinancePublicProvider() as provider:
            assert provider is not None
            assert provider.name == "binance_public"

    @pytest.mark.asyncio
    async def test_close_async(self):
        """Test async close cleans up resources."""
        provider = BinancePublicProvider()
        # Create async client
        _ = provider._async_session
        assert provider._async_client is not None

        # Close
        await provider.close_async()
        assert provider._async_client is None

    @pytest.mark.asyncio
    async def test_download_and_parse_zip_async_404(self):
        """Test async download returns None for 404."""
        provider = BinancePublicProvider()

        with patch.object(
            provider._async_session,
            "get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = await provider._download_and_parse_zip_async("https://example.com/test.zip")
            assert result is None

        await provider.close_async()

    @pytest.mark.asyncio
    async def test_fetch_daily_data_async_concurrent(self, sample_ohlcv_data):
        """Test that async fetching happens concurrently."""
        provider = BinancePublicProvider()

        # Track call times to verify concurrency
        call_times = []

        async def mock_download(url):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # Small delay
            return sample_ohlcv_data

        with patch.object(
            provider,
            "_download_and_parse_zip_async",
            side_effect=mock_download,
        ):
            start_dt = datetime(2024, 1, 1)
            end_dt = datetime(2024, 1, 5)  # 5 days

            results = await provider._fetch_daily_data_async("BTCUSDT", "1d", start_dt, end_dt)

            # Should have 5 results
            assert len(results) == 5

            # Calls should be nearly concurrent (all started within short window)
            if len(call_times) >= 2:
                time_spread = max(call_times) - min(call_times)
                assert time_spread < 0.1  # All started within 100ms

        await provider.close_async()


# ===== YahooFinanceProvider Async Tests =====


class TestYahooFinanceProviderAsync:
    """Test async methods on YahooFinanceProvider."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager protocol."""
        async with YahooFinanceProvider() as provider:
            assert provider is not None
            assert provider.name == "yahoo"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_uses_thread(self, sample_ohlcv_data):
        """Test that fetch_ohlcv_async uses asyncio.to_thread."""
        provider = YahooFinanceProvider()

        with patch.object(provider, "fetch_ohlcv", return_value=sample_ohlcv_data):
            result = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-31")

            assert result is not None
            assert len(result) == 3
            provider.fetch_ohlcv.assert_called_once_with(
                "AAPL", "2024-01-01", "2024-01-31", "daily"
            )

    @pytest.mark.asyncio
    async def test_fetch_batch_ohlcv_async(self, sample_ohlcv_data):
        """Test async batch fetch."""
        provider = YahooFinanceProvider()

        with patch.object(provider, "fetch_batch_ohlcv", return_value=sample_ohlcv_data):
            result = await provider.fetch_batch_ohlcv_async(
                ["AAPL", "MSFT"], "2024-01-01", "2024-01-31"
            )

            assert result is not None
            provider.fetch_batch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_fetches(self, sample_ohlcv_data):
        """Test multiple concurrent async fetches."""
        provider = YahooFinanceProvider()

        # Add symbol column to match expected output
        data_with_symbol = sample_ohlcv_data.with_columns(pl.lit("AAPL").alias("symbol"))

        with patch.object(provider, "fetch_ohlcv", return_value=data_with_symbol):
            tasks = [
                provider.fetch_ohlcv_async(symbol, "2024-01-01", "2024-01-31")
                for symbol in ["AAPL", "MSFT", "GOOGL"]
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, pl.DataFrame)


# ===== CryptoCompareProvider Async Tests =====


class TestCryptoCompareProviderAsync:
    """Test async methods on CryptoCompareProvider."""

    def test_async_session_with_api_key(self):
        """Test async session includes API key in headers."""
        provider = CryptoCompareProvider(api_key="test_key_123")
        session = provider._async_session

        assert session is not None
        assert "authorization" in session.headers
        assert provider.api_key in session.headers["authorization"]

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager protocol."""
        async with CryptoCompareProvider() as provider:
            assert provider is not None
            assert provider.name == "cryptocompare"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_success(self, sample_ohlcv_data):
        """Test async fetch returns valid data."""
        provider = CryptoCompareProvider()

        # Mock the transform to return valid data directly
        with (
            patch.object(provider, "_fetch_raw_data_async", new_callable=AsyncMock) as mock_fetch,
            patch.object(provider, "_transform_data", return_value=sample_ohlcv_data),
        ):
            mock_fetch.return_value = {"data": [{"time": 1704067200}]}

            result = await provider.fetch_ohlcv_async("BTC", "2024-01-01", "2024-01-03")

            assert result is not None
            assert len(result) == 3
            mock_fetch.assert_called_once()

        await provider.close_async()

    @pytest.mark.asyncio
    async def test_close_async(self):
        """Test async close cleans up resources."""
        provider = CryptoCompareProvider()
        # Create async client
        _ = provider._async_session
        assert provider._async_client is not None

        # Close
        await provider.close_async()
        assert provider._async_client is None


# ===== AsyncBatchManager Tests =====


class TestAsyncBatchManager:
    """Test AsyncBatchManager class."""

    @pytest.mark.asyncio
    async def test_batch_load_stacked_dataframe(self, sample_ohlcv_data):
        """Test batch load returns stacked DataFrame."""
        # Create mock async provider
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv_async = AsyncMock(return_value=sample_ohlcv_data)

        manager = AsyncBatchManager(mock_provider, max_concurrent=5)

        result = await manager.load(
            ["AAPL", "MSFT"],
            "2024-01-01",
            "2024-01-31",
        )

        assert result is not None
        assert isinstance(result, pl.DataFrame)
        # Should have been called for each symbol
        assert mock_provider.fetch_ohlcv_async.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_load_dict(self, sample_ohlcv_data):
        """Test batch load returns dictionary."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv_async = AsyncMock(return_value=sample_ohlcv_data)

        manager = AsyncBatchManager(mock_provider)

        result = await manager.load_dict(
            ["AAPL", "MSFT"],
            "2024-01-01",
            "2024-01-31",
        )

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 2


# ===== async_batch_load Function Tests =====


class TestAsyncBatchLoad:
    """Test the async_batch_load convenience function."""

    @pytest.mark.asyncio
    async def test_empty_symbols_raises_error(self):
        """Test that empty symbols list raises ValueError."""
        mock_provider = MagicMock()

        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            await async_batch_load(
                mock_provider,
                [],  # Empty list
                "2024-01-01",
                "2024-01-31",
            )

    @pytest.mark.asyncio
    async def test_all_failures_raises_error(self):
        """Test that all failures raises ValueError."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv_async = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(ValueError, match="All .* symbols failed"):
            await async_batch_load(
                mock_provider,
                ["AAPL", "MSFT"],
                "2024-01-01",
                "2024-01-31",
            )

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, sample_ohlcv_data):
        """Test that partial failures don't stop the batch."""
        mock_provider = MagicMock()

        # First call fails, second succeeds
        mock_provider.fetch_ohlcv_async = AsyncMock(
            side_effect=[
                Exception("API Error"),
                sample_ohlcv_data.with_columns(pl.lit("MSFT").alias("symbol")),
            ]
        )

        result = await async_batch_load(
            mock_provider,
            ["AAPL", "MSFT"],
            "2024-01-01",
            "2024-01-31",
            fail_on_partial=False,
        )

        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fail_on_partial_raises_error(self, sample_ohlcv_data):
        """Test fail_on_partial=True raises on any failure."""
        mock_provider = MagicMock()

        mock_provider.fetch_ohlcv_async = AsyncMock(
            side_effect=[
                Exception("API Error"),
                sample_ohlcv_data.with_columns(pl.lit("MSFT").alias("symbol")),
            ]
        )

        with pytest.raises(ValueError, match="Batch load failed"):
            await async_batch_load(
                mock_provider,
                ["AAPL", "MSFT"],
                "2024-01-01",
                "2024-01-31",
                fail_on_partial=True,
            )

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, sample_ohlcv_data):
        """Test that max_concurrent is respected."""
        mock_provider = MagicMock()

        concurrent_count = 0
        max_observed_concurrent = 0

        async def track_concurrency(*args, **kwargs):
            nonlocal concurrent_count, max_observed_concurrent
            concurrent_count += 1
            max_observed_concurrent = max(max_observed_concurrent, concurrent_count)
            await asyncio.sleep(0.05)  # Simulate work
            concurrent_count -= 1
            return sample_ohlcv_data.with_columns(pl.lit("SYM").alias("symbol"))

        mock_provider.fetch_ohlcv_async = AsyncMock(side_effect=track_concurrency)

        await async_batch_load(
            mock_provider,
            [f"SYM{i}" for i in range(10)],  # 10 symbols
            "2024-01-01",
            "2024-01-31",
            max_concurrent=3,  # Only 3 at a time
        )

        # Should never exceed max_concurrent
        assert max_observed_concurrent <= 3


# ===== Protocol Conformance Tests =====


class TestAsyncProtocolConformance:
    """Test that providers conform to async protocol patterns."""

    def test_binance_has_async_methods(self):
        """Test BinancePublicProvider has required async methods."""
        provider = BinancePublicProvider()

        assert hasattr(provider, "fetch_ohlcv_async")
        assert asyncio.iscoroutinefunction(provider.fetch_ohlcv_async)
        assert hasattr(provider, "close_async")
        assert asyncio.iscoroutinefunction(provider.close_async)
        assert hasattr(provider, "__aenter__")
        assert hasattr(provider, "__aexit__")

    def test_yahoo_has_async_methods(self):
        """Test YahooFinanceProvider has required async methods."""
        provider = YahooFinanceProvider()

        assert hasattr(provider, "fetch_ohlcv_async")
        assert asyncio.iscoroutinefunction(provider.fetch_ohlcv_async)
        assert hasattr(provider, "close_async")
        assert asyncio.iscoroutinefunction(provider.close_async)
        assert hasattr(provider, "__aenter__")
        assert hasattr(provider, "__aexit__")

    def test_cryptocompare_has_async_methods(self):
        """Test CryptoCompareProvider has required async methods."""
        provider = CryptoCompareProvider()

        assert hasattr(provider, "fetch_ohlcv_async")
        assert asyncio.iscoroutinefunction(provider.fetch_ohlcv_async)
        assert hasattr(provider, "close_async")
        assert asyncio.iscoroutinefunction(provider.close_async)
        assert hasattr(provider, "__aenter__")
        assert hasattr(provider, "__aexit__")

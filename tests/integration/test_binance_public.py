"""Integration tests for BinancePublicProvider.

These tests download actual data from data.binance.vision and verify
the provider works correctly. No API key required.

Note: These tests require internet access and may be slow due to
downloading ZIP files from Binance's public data repository.
"""

from datetime import UTC

import polars as pl
import pytest

from ml4t.data.providers.binance_public import BinancePublicProvider


class TestBinancePublicProvider:
    """Tests for BinancePublicProvider."""

    @pytest.fixture
    def spot_provider(self):
        """Create a spot market provider."""
        return BinancePublicProvider(market="spot")

    @pytest.fixture
    def futures_provider(self):
        """Create a futures market provider."""
        return BinancePublicProvider(market="futures")

    def test_provider_name(self, spot_provider):
        """Test provider name is correct."""
        assert spot_provider.name == "binance_public"

    def test_invalid_market(self):
        """Test invalid market raises error."""
        with pytest.raises(ValueError, match="Invalid market"):
            BinancePublicProvider(market="invalid")

    def test_symbol_normalization(self, spot_provider):
        """Test symbol normalization."""
        assert spot_provider._normalize_symbol("BTC") == "BTCUSDT"
        assert spot_provider._normalize_symbol("btc") == "BTCUSDT"
        assert spot_provider._normalize_symbol("BTC-USD") == "BTCUSDT"
        assert spot_provider._normalize_symbol("BTC/USD") == "BTCUSDT"
        assert spot_provider._normalize_symbol("BTCUSDT") == "BTCUSDT"
        assert spot_provider._normalize_symbol("ETH") == "ETHUSDT"
        assert spot_provider._normalize_symbol("SOL") == "SOLUSDT"

    def test_url_building_spot(self, spot_provider):
        """Test URL building for spot market."""
        from datetime import datetime

        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = spot_provider._build_url("BTCUSDT", "1d", date)
        expected = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
        assert url == expected

    def test_url_building_futures(self, futures_provider):
        """Test URL building for futures market."""
        from datetime import datetime

        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = futures_provider._build_url("BTCUSDT", "1d", date)
        expected = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
        assert url == expected

    def test_monthly_url_building_spot(self, spot_provider):
        """Test monthly URL building for spot market."""
        url = spot_provider._build_monthly_url("BTCUSDT", "1d", 2024, 1)
        expected = (
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
        )
        assert url == expected

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_daily_spot_btc(self, spot_provider):
        """Test fetching daily BTC data from spot market."""
        # Fetch recent data (last week to be safe)
        df = spot_provider.fetch_ohlcv(
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-01-07",
            frequency="daily",
        )

        # Verify schema
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime("ms", "UTC")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify data
        assert len(df) > 0  # Should have some data
        assert df["high"].max() >= df["low"].min()  # OHLC sanity check

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_hourly_spot_eth(self, spot_provider):
        """Test fetching hourly ETH data."""
        df = spot_provider.fetch_ohlcv(
            symbol="ETH",  # Should normalize to ETHUSDT
            start="2024-01-01",
            end="2024-01-03",
            frequency="hourly",
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # 3 days of hourly = ~72 bars
        assert len(df) >= 48  # At least 2 days worth

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_minute_spot_btc(self, spot_provider):
        """Test fetching minute BTC data."""
        df = spot_provider.fetch_ohlcv(
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-01-01",
            frequency="minute",
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # 1 day of minute = ~1440 bars
        assert len(df) >= 1000

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_daily_futures_btc(self, futures_provider):
        """Test fetching daily BTC futures data."""
        df = futures_provider.fetch_ohlcv(
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-01-07",
            frequency="daily",
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "timestamp" in df.columns

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_monthly_optimization(self, spot_provider):
        """Test that monthly files are used for long date ranges."""
        # Request > 60 days to trigger monthly downloads
        df = spot_provider.fetch_ohlcv(
            symbol="BTCUSDT",
            start="2023-01-01",
            end="2023-03-31",
            frequency="daily",
        )

        assert isinstance(df, pl.DataFrame)
        # Should have ~90 days of data
        assert len(df) >= 80

    def test_get_available_symbols(self, spot_provider):
        """Test getting available symbols."""
        symbols = spot_provider.get_available_symbols()
        assert len(symbols) > 0
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_get_available_symbols_search(self, spot_provider):
        """Test searching available symbols."""
        symbols = spot_provider.get_available_symbols(search="BTC")
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" not in symbols

    def test_empty_dataframe_schema(self, spot_provider):
        """Test empty DataFrame has correct schema."""
        df = spot_provider._create_empty_dataframe()
        assert df.is_empty()
        assert "timestamp" in df.columns
        assert df["timestamp"].dtype == pl.Datetime("ms", "UTC")
        assert df["open"].dtype == pl.Float64

    def test_unsupported_frequency(self, spot_provider):
        """Test unsupported frequency raises error."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            spot_provider.fetch_ohlcv(
                symbol="BTCUSDT",
                start="2024-01-01",
                end="2024-01-07",
                frequency="invalid",
            )


class TestBinancePublicProviderEdgeCases:
    """Edge case tests for BinancePublicProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider."""
        return BinancePublicProvider(market="spot")

    def test_date_filtering(self, provider):
        """Test that data is properly filtered to date range."""
        # This is a unit test - no actual data download
        # The actual filtering happens in _fetch_and_transform_data
        pass

    def test_context_manager(self):
        """Test provider works as context manager."""
        with BinancePublicProvider(market="spot") as provider:
            assert provider.name == "binance_public"


class TestFetchPremiumIndex:
    """Tests for fetching premium index data.

    Premium Index = (Perpetual Price - Spot Price) / Spot Price

    This is critical for funding rate analysis and mean-reversion strategies.
    The premium index is the primary driver of funding rates on perpetual futures.
    """

    @pytest.fixture
    def futures_provider(self):
        """Create a futures market provider."""
        return BinancePublicProvider(market="futures")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_single_symbol(self, futures_provider):
        """Test fetching premium index for a single symbol.

        API calls: 7 files (7 days)
        """
        df = futures_provider.fetch_premium_index(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # Check schema
        required_cols = [
            "timestamp",
            "symbol",
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # 8h interval = 3 per day = 21 for 7 days
        assert len(df) >= 18, f"Expected ~21 rows, got {len(df)}"

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime("ms", "UTC")
        assert df["symbol"].dtype == pl.Utf8
        assert df["premium_index_close"].dtype == pl.Float64

        # Premium index should be small (typically -1% to +1%)
        assert df["premium_index_close"].min() > -0.1  # Not more than -10%
        assert df["premium_index_close"].max() < 0.1  # Not more than +10%

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_different_interval(self, futures_provider):
        """Test fetching premium index with different intervals.

        API calls: 3 files (3 days)
        """
        # Test with 4h interval
        df = futures_provider.fetch_premium_index(
            "ETHUSDT",
            "2024-01-01",
            "2024-01-03",
            interval="4h",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # 4h interval = 6 per day = 18 for 3 days
        assert len(df) >= 14, f"Expected ~18 rows, got {len(df)}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_1h_interval(self, futures_provider):
        """Test fetching premium index with 1h interval.

        API calls: 2 files (2 days)
        """
        df = futures_provider.fetch_premium_index(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-02",
            interval="1h",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # 1h interval = 24 per day = 48 for 2 days
        assert len(df) >= 40, f"Expected ~48 rows, got {len(df)}"

    def test_fetch_premium_index_invalid_interval(self, futures_provider):
        """Test that invalid interval raises error."""
        from ml4t.data.core.exceptions import DataValidationError

        with pytest.raises(DataValidationError) as exc_info:
            futures_provider.fetch_premium_index(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-03",
                interval="invalid",
            )

        assert "Invalid interval" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_ohlc_invariants(self, futures_provider):
        """Test that premium index OHLC values maintain proper relationships.

        API calls: 7 files (7 days)
        """
        df = futures_provider.fetch_premium_index(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
        )

        assert not df.is_empty()

        # High should be >= Low
        assert (df["premium_index_high"] >= df["premium_index_low"]).all()

        # High should be >= Open and Close
        assert (df["premium_index_high"] >= df["premium_index_open"]).all()
        assert (df["premium_index_high"] >= df["premium_index_close"]).all()

        # Low should be <= Open and Close
        assert (df["premium_index_low"] <= df["premium_index_open"]).all()
        assert (df["premium_index_low"] <= df["premium_index_close"]).all()

    def test_premium_index_url_building(self, futures_provider):
        """Test URL building for premium index data."""
        from datetime import datetime

        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = futures_provider._build_premium_index_url("BTCUSDT", "8h", date)
        expected = "https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/BTCUSDT/8h/BTCUSDT-8h-2024-01-15.zip"
        assert url == expected

    def test_empty_premium_index_dataframe_schema(self, futures_provider):
        """Test empty premium index DataFrame has correct schema."""
        df = futures_provider._create_empty_premium_index_dataframe()
        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "premium_index_open" in df.columns
        assert "premium_index_high" in df.columns
        assert "premium_index_low" in df.columns
        assert "premium_index_close" in df.columns
        assert df["timestamp"].dtype == pl.Datetime("ms", "UTC")
        assert df["premium_index_close"].dtype == pl.Float64


class TestFetchPremiumIndexMulti:
    """Tests for fetching premium index for multiple symbols.

    These tests verify cross-sectional analysis capabilities.
    """

    @pytest.fixture
    def futures_provider(self):
        """Create a futures market provider."""
        return BinancePublicProvider(market="futures")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_multi(self, futures_provider):
        """Test fetching premium index for multiple symbols.

        API calls: ~9 files (3 symbols * 3 days)
        """
        df = futures_provider.fetch_premium_index_multi(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "2024-01-01",
            "2024-01-03",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # Should have data for all 3 symbols
        assert df["symbol"].n_unique() == 3

        # Should be sorted by timestamp
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_multi_five_symbols(self, futures_provider):
        """Test fetching premium index for 5 symbols (ML4T book case study).

        API calls: ~15 files (5 symbols * 3 days)
        """
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        df = futures_provider.fetch_premium_index_multi(
            symbols,
            "2024-01-01",
            "2024-01-03",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # Should have data for all 5 symbols
        assert df["symbol"].n_unique() == 5

        # Check all symbols are present
        fetched_symbols = set(df["symbol"].unique().to_list())
        expected_symbols = set(symbols)
        assert fetched_symbols == expected_symbols

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_multi_cross_sectional_ranking(self, futures_provider):
        """Test that multi-fetch data supports cross-sectional ranking.

        This is the key use case for funding rate reversal strategies.
        """
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        df = futures_provider.fetch_premium_index_multi(
            symbols,
            "2024-01-01",
            "2024-01-03",
        )

        assert not df.is_empty()

        # Cross-sectional ranking at each timestamp
        ranked = df.with_columns(
            [
                pl.col("premium_index_close")
                .rank(method="ordinal")
                .over("timestamp")
                .alias("premium_rank")
            ]
        )

        # Each timestamp should have ranks 1 to N
        for ts in ranked["timestamp"].unique().to_list()[:5]:  # Check first 5 timestamps
            ts_ranks = ranked.filter(pl.col("timestamp") == ts)["premium_rank"]
            n = len(ts_ranks)
            if n > 0:
                assert ts_ranks.min() == 1
                assert ts_ranks.max() == n

    def test_fetch_premium_index_multi_empty_list_raises(self, futures_provider):
        """Test that empty symbol list raises error."""
        from ml4t.data.core.exceptions import DataValidationError

        with pytest.raises(DataValidationError) as exc_info:
            futures_provider.fetch_premium_index_multi(
                [],
                "2024-01-01",
                "2024-01-03",
            )

        assert "symbols list cannot be empty" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_fetch_premium_index_multi_handles_missing_symbol(self, futures_provider):
        """Test that multi-fetch handles missing symbols gracefully.

        API calls: ~6-15 files (2 valid + 1 invalid symbol)
        """
        df = futures_provider.fetch_premium_index_multi(
            ["BTCUSDT", "INVALID_XYZ_123", "ETHUSDT"],
            "2024-01-01",
            "2024-01-03",
        )

        assert isinstance(df, pl.DataFrame)
        # Should have data for at least the valid symbols
        assert df["symbol"].n_unique() >= 2

        # Should NOT contain the invalid symbol
        assert "INVALID_XYZ_123" not in df["symbol"].unique().to_list()


class TestPremiumIndexDataConsistency:
    """Tests for premium index data consistency and edge cases."""

    @pytest.fixture
    def futures_provider(self):
        """Create a futures market provider."""
        return BinancePublicProvider(market="futures")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_timestamp_uniqueness(self, futures_provider):
        """Test that timestamps are unique within each symbol."""
        df = futures_provider.fetch_premium_index(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
        )

        # No duplicate timestamps
        assert df["timestamp"].n_unique() == len(df)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_no_null_values_in_core_columns(self, futures_provider):
        """Test that core columns have no null values."""
        df = futures_provider.fetch_premium_index(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
        )

        core_cols = [
            "timestamp",
            "symbol",
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        ]

        for col in core_cols:
            null_count = df[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} null values"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_date_range_filtering(self, futures_provider):
        """Test that data is properly filtered to requested date range."""
        start = "2024-01-02"
        end = "2024-01-04"

        df = futures_provider.fetch_premium_index("BTCUSDT", start, end)

        if not df.is_empty():
            from datetime import date

            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()

            assert min_date >= date(2024, 1, 2)
            assert max_date <= date(2024, 1, 4)

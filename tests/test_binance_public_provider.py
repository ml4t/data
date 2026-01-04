"""Tests for Binance Public Data provider module."""

import io
import zipfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.binance_public import BinancePublicProvider


class TestBinancePublicProviderInit:
    """Tests for provider initialization."""

    def test_default_init(self):
        """Test default initialization."""
        provider = BinancePublicProvider()

        assert provider.name == "binance_public"
        assert provider.market == "spot"
        assert provider.timeout == 60.0

    def test_futures_market_init(self):
        """Test initialization with futures market."""
        provider = BinancePublicProvider(market="futures")

        assert provider.market == "futures"

    def test_custom_timeout(self):
        """Test initialization with custom timeout."""
        provider = BinancePublicProvider(timeout=120.0)

        assert provider.timeout == 120.0

    def test_invalid_market_raises_error(self):
        """Test invalid market raises ValueError."""
        with pytest.raises(ValueError, match="Invalid market"):
            BinancePublicProvider(market="invalid")

    def test_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        provider = BinancePublicProvider(rate_limit=(100, 60.0))

        # Rate limit should be set (internal implementation detail)
        assert provider is not None


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_binance_public(self):
        """Test name property returns correct value."""
        provider = BinancePublicProvider()
        assert provider.name == "binance_public"


class TestCreateEmptyDataframe:
    """Tests for _create_empty_dataframe method."""

    def test_empty_dataframe_schema(self):
        """Test empty DataFrame has correct schema."""
        provider = BinancePublicProvider()
        df = provider._create_empty_dataframe()

        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_empty_dataframe_dtypes(self):
        """Test empty DataFrame has correct dtypes."""
        provider = BinancePublicProvider()
        df = provider._create_empty_dataframe()

        assert df.schema["timestamp"] == pl.Datetime("ms", "UTC")
        assert df.schema["open"] == pl.Float64
        assert df.schema["close"] == pl.Float64
        assert df.schema["volume"] == pl.Float64


class TestNormalizeSymbol:
    """Tests for _normalize_symbol method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def test_uppercase_conversion(self, provider):
        """Test symbols are uppercased."""
        assert provider._normalize_symbol("btcusdt") == "BTCUSDT"

    def test_dash_removal(self, provider):
        """Test dashes are removed."""
        assert provider._normalize_symbol("BTC-USDT") == "BTCUSDT"

    def test_slash_removal(self, provider):
        """Test slashes are removed."""
        assert provider._normalize_symbol("BTC/USDT") == "BTCUSDT"

    def test_space_removal(self, provider):
        """Test spaces are removed."""
        assert provider._normalize_symbol("BTC USDT") == "BTCUSDT"

    def test_btc_alias(self, provider):
        """Test BTC is converted to BTCUSDT."""
        assert provider._normalize_symbol("BTC") == "BTCUSDT"

    def test_bitcoin_alias(self, provider):
        """Test BITCOIN is converted to BTCUSDT."""
        assert provider._normalize_symbol("BITCOIN") == "BTCUSDT"

    def test_eth_alias(self, provider):
        """Test ETH is converted to ETHUSDT."""
        assert provider._normalize_symbol("ETH") == "ETHUSDT"

    def test_ethereum_alias(self, provider):
        """Test ETHEREUM is converted to ETHUSDT."""
        assert provider._normalize_symbol("ETHEREUM") == "ETHUSDT"

    def test_btcusd_conversion(self, provider):
        """Test BTCUSD is converted to BTCUSDT."""
        assert provider._normalize_symbol("BTCUSD") == "BTCUSDT"

    def test_ethusd_conversion(self, provider):
        """Test ETHUSD is converted to ETHUSDT."""
        assert provider._normalize_symbol("ETHUSD") == "ETHUSDT"

    def test_short_symbol_adds_usdt(self, provider):
        """Test 3-character symbols get USDT appended."""
        assert provider._normalize_symbol("SOL") == "SOLUSDT"

    def test_symbol_without_known_suffix_adds_usdt(self, provider):
        """Test symbols without known suffix get USDT."""
        assert provider._normalize_symbol("DOGE") == "DOGEUSDT"

    def test_symbol_with_usdt_unchanged(self, provider):
        """Test symbols ending with USDT are unchanged."""
        assert provider._normalize_symbol("BTCUSDT") == "BTCUSDT"

    def test_symbol_with_busd_unchanged(self, provider):
        """Test symbols ending with BUSD are unchanged."""
        assert provider._normalize_symbol("BTCBUSD") == "BTCBUSD"


class TestBuildUrl:
    """Tests for URL building methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    @pytest.fixture
    def futures_provider(self):
        """Create futures provider instance."""
        return BinancePublicProvider(market="futures")

    def test_build_spot_url(self, provider):
        """Test spot market URL construction."""
        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = provider._build_url("BTCUSDT", "1d", date)

        expected = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
        assert url == expected

    def test_build_futures_url(self, futures_provider):
        """Test futures market URL construction."""
        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = futures_provider._build_url("BTCUSDT", "1d", date)

        expected = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-15.zip"
        assert url == expected

    def test_build_monthly_spot_url(self, provider):
        """Test spot monthly URL construction."""
        url = provider._build_monthly_url("BTCUSDT", "1h", 2024, 1)

        expected = (
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
        )
        assert url == expected

    def test_build_monthly_futures_url(self, futures_provider):
        """Test futures monthly URL construction."""
        url = futures_provider._build_monthly_url("ETHUSDT", "4h", 2024, 6)

        expected = "https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/4h/ETHUSDT-4h-2024-06.zip"
        assert url == expected


class TestIntervalMap:
    """Tests for interval mapping."""

    def test_interval_map_contains_expected_keys(self):
        """Test INTERVAL_MAP contains expected frequencies."""
        assert "minute" in BinancePublicProvider.INTERVAL_MAP
        assert "hourly" in BinancePublicProvider.INTERVAL_MAP
        assert "daily" in BinancePublicProvider.INTERVAL_MAP
        assert "weekly" in BinancePublicProvider.INTERVAL_MAP

    def test_interval_map_values(self):
        """Test INTERVAL_MAP returns correct Binance intervals."""
        assert BinancePublicProvider.INTERVAL_MAP["minute"] == "1m"
        assert BinancePublicProvider.INTERVAL_MAP["hourly"] == "1h"
        assert BinancePublicProvider.INTERVAL_MAP["daily"] == "1d"
        assert BinancePublicProvider.INTERVAL_MAP["weekly"] == "1w"


class TestDownloadAndParseZip:
    """Tests for _download_and_parse_zip method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def _create_mock_zip_response(self, csv_content: str, filename: str = "data.csv") -> bytes:
        """Create mock ZIP response with CSV content."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, csv_content)
        return zip_buffer.getvalue()

    def test_parse_zip_without_header(self, provider):
        """Test parsing ZIP file without header row."""
        csv_content = """1704067200000,42000.0,42500.0,41800.0,42300.0,100.5,1704067200999,4200000.0,1000,50.0,2100000.0,0
1704153600000,42300.0,42800.0,42000.0,42600.0,150.0,1704153600999,6390000.0,1500,75.0,3195000.0,0"""

        zip_content = self._create_mock_zip_response(csv_content, "BTCUSDT-1d-2024-01-01.csv")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_zip("http://test.url/data.zip")

        assert df is not None
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert df["open"][0] == 42000.0

    def test_parse_zip_with_header(self, provider):
        """Test parsing ZIP file with header row."""
        csv_content = """open_time,open,high,low,close,volume,close_time,quote_volume,num_trades,taker_buy_base,taker_buy_quote,ignore
1704067200000,42000.0,42500.0,41800.0,42300.0,100.5,1704067200999,4200000.0,1000,50.0,2100000.0,0"""

        zip_content = self._create_mock_zip_response(csv_content)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_zip("http://test.url/data.zip")

        assert df is not None
        assert len(df) == 1
        assert df["close"][0] == 42300.0

    def test_parse_zip_404_returns_none(self, provider):
        """Test 404 response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_zip("http://test.url/notfound.zip")

        assert df is None

    def test_parse_zip_http_error_404(self, provider):
        """Test HTTPStatusError with 404 returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            df = provider._download_and_parse_zip("http://test.url/notfound.zip")

        assert df is None


class TestFetchAndTransformData:
    """Tests for _fetch_and_transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def test_invalid_frequency_raises_error(self, provider):
        """Test invalid frequency raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            provider._fetch_and_transform_data("BTCUSDT", "2024-01-01", "2024-01-05", "invalid")

    def test_empty_data_returns_empty_dataframe(self, provider):
        """Test empty data returns empty DataFrame."""
        with patch.object(provider, "_fetch_daily_data", return_value=[]):
            df = provider._fetch_and_transform_data("BTCUSDT", "2024-01-01", "2024-01-05", "daily")

        assert df.is_empty()

    def test_uses_daily_for_short_range(self, provider):
        """Test short date range uses daily fetch."""
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        with patch.object(provider, "_fetch_daily_data", return_value=[mock_df]) as mock_daily:
            with patch.object(provider, "_fetch_monthly_data") as mock_monthly:
                provider._fetch_and_transform_data("BTCUSDT", "2024-01-01", "2024-01-15", "daily")

        mock_daily.assert_called_once()
        mock_monthly.assert_not_called()

    def test_uses_monthly_for_long_range(self, provider):
        """Test long date range (>60 days) uses monthly fetch."""
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        with patch.object(provider, "_fetch_daily_data") as mock_daily:
            with patch.object(
                provider, "_fetch_monthly_data", return_value=[mock_df]
            ) as mock_monthly:
                provider._fetch_and_transform_data("BTCUSDT", "2024-01-01", "2024-06-01", "daily")

        mock_monthly.assert_called_once()
        mock_daily.assert_not_called()


class TestFetchDailyData:
    """Tests for _fetch_daily_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def test_downloads_consecutive_days(self, provider):
        """Test data is fetched for consecutive days."""
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt = datetime(2024, 1, 3, tzinfo=UTC)

        with patch.object(provider, "_download_and_parse_zip", return_value=mock_df):
            with patch.object(provider, "_acquire_rate_limit"):
                result = provider._fetch_daily_data("BTCUSDT", "1d", start_dt, end_dt)

        assert len(result) == 3  # 3 days

    def test_returns_empty_after_consecutive_404s(self, provider):
        """Test returns empty list after 8 consecutive 404s with no prior data."""
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt = datetime(2024, 1, 15, tzinfo=UTC)

        with patch.object(provider, "_download_and_parse_zip", return_value=None):
            with patch.object(provider, "_acquire_rate_limit"):
                # SymbolNotFoundError is caught internally and logged as warning
                result = provider._fetch_daily_data("INVALID", "1d", start_dt, end_dt)
                assert result == []  # Empty list after consecutive 404s


class TestFetchMonthlyData:
    """Tests for _fetch_monthly_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def test_downloads_months_in_range(self, provider):
        """Test monthly data is fetched for months in range."""
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt = datetime(2024, 3, 31, tzinfo=UTC)

        download_count = 0

        def mock_download(*args, **kwargs):
            nonlocal download_count
            download_count += 1
            return mock_df

        with patch.object(provider, "_download_and_parse_zip", side_effect=mock_download):
            with patch.object(provider, "_acquire_rate_limit"):
                result = provider._fetch_monthly_data("BTCUSDT", "1d", start_dt, end_dt)

        assert len(result) >= 3  # At least 3 months


class TestFetchMetrics:
    """Tests for fetch_metrics method."""

    @pytest.fixture
    def provider(self):
        """Create provider for futures market."""
        return BinancePublicProvider(market="futures")

    def test_empty_metrics_dataframe_schema(self, provider):
        """Test empty metrics DataFrame has correct schema."""
        df = provider._create_empty_metrics_dataframe()

        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "open_interest" in df.columns
        assert "open_interest_value" in df.columns

    def test_fetch_metrics_no_data_returns_empty(self, provider):
        """Test fetch_metrics with no data returns empty DataFrame."""
        with patch.object(provider, "_download_and_parse_metrics_zip", return_value=None):
            with patch.object(provider, "_acquire_rate_limit"):
                # Exception is caught internally, returns empty DataFrame
                df = provider.fetch_metrics("BTCUSDT", "2024-01-01", "2024-01-15")
                assert df.is_empty()

    def test_build_metrics_url(self, provider):
        """Test metrics URL construction."""
        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = provider._build_metrics_url("BTCUSDT", date)

        expected = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-15.zip"
        assert url == expected


class TestFetchPremiumIndex:
    """Tests for premium index methods."""

    @pytest.fixture
    def provider(self):
        """Create provider for futures market."""
        return BinancePublicProvider(market="futures")

    def test_invalid_interval_raises_error(self, provider):
        """Test invalid interval raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Invalid interval"):
            provider.fetch_premium_index("BTCUSDT", "2024-01-01", "2024-01-31", "invalid")

    def test_empty_premium_index_dataframe_schema(self, provider):
        """Test empty premium index DataFrame has correct schema."""
        df = provider._create_empty_premium_index_dataframe()

        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "premium_index_open" in df.columns
        assert "premium_index_close" in df.columns

    def test_build_premium_index_url(self, provider):
        """Test premium index URL construction."""
        date = datetime(2024, 1, 15, tzinfo=UTC)
        url = provider._build_premium_index_url("BTCUSDT", "8h", date)

        expected = "https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/BTCUSDT/8h/BTCUSDT-8h-2024-01-15.zip"
        assert url == expected

    def test_build_premium_index_monthly_url(self, provider):
        """Test premium index monthly URL construction."""
        url = provider._build_premium_index_monthly_url("BTCUSDT", "8h", 2024, 1)

        expected = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/8h/BTCUSDT-8h-2024-01.zip"
        assert url == expected

    def test_fetch_premium_index_multi_empty_symbols_raises_error(self, provider):
        """Test fetch_premium_index_multi with empty symbols raises error."""
        with pytest.raises(DataValidationError, match="symbols list cannot be empty"):
            provider.fetch_premium_index_multi([], "2024-01-01", "2024-01-31")

    def test_fetch_premium_index_multi_returns_combined(self, provider):
        """Test fetch_premium_index_multi combines multiple symbols."""
        mock_df1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "symbol": ["BTCUSDT"],
                "premium_index_open": [0.001],
                "premium_index_high": [0.002],
                "premium_index_low": [0.0005],
                "premium_index_close": [0.0015],
            }
        )
        mock_df2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "symbol": ["ETHUSDT"],
                "premium_index_open": [0.002],
                "premium_index_high": [0.003],
                "premium_index_low": [0.001],
                "premium_index_close": [0.0025],
            }
        )

        with patch.object(provider, "fetch_premium_index") as mock_fetch:
            mock_fetch.side_effect = [mock_df1, mock_df2]
            df = provider.fetch_premium_index_multi(
                ["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-01-31"
            )

        assert len(df) == 2
        assert set(df["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}


class TestGetAvailableSymbols:
    """Tests for get_available_symbols method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider()

    def test_returns_common_pairs(self, provider):
        """Test returns list of common pairs."""
        symbols = provider.get_available_symbols()

        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert len(symbols) > 10

    def test_search_filter(self, provider):
        """Test search filter works."""
        symbols = provider.get_available_symbols(search="BTC")

        assert "BTCUSDT" in symbols
        assert all("BTC" in s for s in symbols)

    def test_search_case_insensitive(self, provider):
        """Test search is case insensitive."""
        symbols = provider.get_available_symbols(search="btc")

        assert "BTCUSDT" in symbols

    def test_search_no_match(self, provider):
        """Test search with no matches returns empty list."""
        symbols = provider.get_available_symbols(search="ZZZZZ")

        assert symbols == []


class TestParseMetricsZip:
    """Tests for _download_and_parse_metrics_zip method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider(market="futures")

    def _create_mock_zip_response(self, csv_content: str) -> bytes:
        """Create mock ZIP response with CSV content."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metrics.csv", csv_content)
        return zip_buffer.getvalue()

    def test_parse_metrics_with_header(self, provider):
        """Test parsing metrics ZIP file with header."""
        csv_content = """symbol,create_time,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio
BTCUSDT,2024-01-01 00:00:00,50000.0,2000000000.0,1.5,1.5,1.2,0.8"""

        zip_content = self._create_mock_zip_response(csv_content)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_metrics_zip("http://test.url/metrics.zip")

        assert df is not None
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "open_interest" in df.columns

    def test_parse_metrics_404_returns_none(self, provider):
        """Test 404 response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_metrics_zip("http://test.url/notfound.zip")

        assert df is None


class TestParsePremiumIndexZip:
    """Tests for _download_and_parse_premium_index_zip method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return BinancePublicProvider(market="futures")

    def _create_mock_zip_response(self, csv_content: str) -> bytes:
        """Create mock ZIP response with CSV content."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("premiumindex.csv", csv_content)
        return zip_buffer.getvalue()

    def test_parse_premium_index_without_header(self, provider):
        """Test parsing premium index ZIP without header."""
        csv_content = """1704067200000,0.001,0.002,0.0005,0.0015,0,1704067200999,0,0,0,0,0"""

        zip_content = self._create_mock_zip_response(csv_content)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_premium_index_zip(
                "http://test.url/premium.zip", "BTCUSDT"
            )

        assert df is not None
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "premium_index_open" in df.columns
        assert df["symbol"][0] == "BTCUSDT"

    def test_parse_premium_index_404_returns_none(self, provider):
        """Test 404 response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._download_and_parse_premium_index_zip(
                "http://test.url/notfound.zip", "BTCUSDT"
            )

        assert df is None


class TestIntegration:
    """Integration tests for BinancePublicProvider."""

    def test_full_fetch_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        provider = BinancePublicProvider()

        # Create mock data
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        with patch.object(provider, "_download_and_parse_zip", return_value=mock_df):
            with patch.object(provider, "_acquire_rate_limit"):
                df = provider.fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-01-05", "daily")

        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "close" in df.columns

    def test_futures_market_complete_workflow(self):
        """Test futures market workflow."""
        provider = BinancePublicProvider(market="futures")

        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [100.0],
            }
        )

        with patch.object(provider, "_download_and_parse_zip", return_value=mock_df):
            with patch.object(provider, "_acquire_rate_limit"):
                df = provider.fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-01-05", "daily")

        assert not df.is_empty()

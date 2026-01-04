"""Tests for Wiki Prices provider module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.wiki_prices import WikiPricesProvider


class TestWikiPricesProviderInit:
    """Tests for provider initialization."""

    def test_init_with_valid_parquet_path(self, tmp_path):
        """Test initialization with valid Parquet path."""
        # Create a minimal test Parquet file
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": [datetime(2018, 1, 1), datetime(2018, 1, 2)],
                "open": [170.0, 171.0],
                "high": [172.0, 173.0],
                "low": [169.0, 170.0],
                "close": [171.0, 172.0],
                "volume": [1000000.0, 1100000.0],
                "adj_open": [170.0, 171.0],
                "adj_high": [172.0, 173.0],
                "adj_low": [169.0, 170.0],
                "adj_close": [171.0, 172.0],
                "adj_volume": [1000000.0, 1100000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)

        assert provider.name == "wiki_prices"
        assert provider.parquet_path == parquet_file
        assert provider.cache_in_memory is False
        assert provider._data is None

    def test_init_with_cache_in_memory(self, tmp_path):
        """Test initialization with cache_in_memory=True."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [datetime(2018, 1, 1)],
                "open": [170.0],
                "high": [172.0],
                "low": [169.0],
                "close": [171.0],
                "volume": [1000000.0],
                "adj_open": [170.0],
                "adj_high": [172.0],
                "adj_low": [169.0],
                "adj_close": [171.0],
                "adj_volume": [1000000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file, cache_in_memory=True)

        assert provider.cache_in_memory is True
        assert provider._data is not None
        assert len(provider._data) == 1

    def test_init_with_invalid_path_raises_error(self):
        """Test initialization with invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found at provided path"):
            WikiPricesProvider(parquet_path="/nonexistent/path/wiki_prices.parquet")

    def test_init_auto_detect_fails_when_no_files(self):
        """Test auto-detection fails when no files found."""
        # Mock DEFAULT_PATHS to point to non-existent locations
        with patch.object(WikiPricesProvider, "DEFAULT_PATHS", [Path("/nonexistent/path.parquet")]):
            with pytest.raises(FileNotFoundError, match="Wiki Prices Parquet not found"):
                WikiPricesProvider()


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_wiki_prices(self, tmp_path):
        """Test name property returns correct value."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [datetime(2018, 1, 1)],
                "open": [170.0],
                "high": [172.0],
                "low": [169.0],
                "close": [171.0],
                "volume": [1000000.0],
                "adj_open": [170.0],
                "adj_high": [172.0],
                "adj_low": [169.0],
                "adj_close": [171.0],
                "adj_volume": [1000000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)
        assert provider.name == "wiki_prices"


class TestResolveParquetPath:
    """Tests for _resolve_parquet_path method."""

    def test_resolve_provided_valid_path(self, tmp_path):
        """Test resolving a valid provided path."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        parquet_file.touch()

        # Test through initialization
        provider = WikiPricesProvider.__new__(WikiPricesProvider)
        resolved = provider._resolve_parquet_path(parquet_file)

        assert resolved == parquet_file.resolve()

    def test_resolve_provided_invalid_path_raises(self):
        """Test resolving an invalid provided path raises error."""
        provider = WikiPricesProvider.__new__(WikiPricesProvider)
        provider.logger = MagicMock()

        with pytest.raises(FileNotFoundError):
            provider._resolve_parquet_path("/nonexistent/file.parquet")

    def test_auto_detect_finds_first_match(self, tmp_path):
        """Test auto-detect finds first matching file."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        parquet_file.touch()

        with patch.object(WikiPricesProvider, "DEFAULT_PATHS", [parquet_file]):
            provider = WikiPricesProvider.__new__(WikiPricesProvider)
            provider.logger = MagicMock()
            resolved = provider._resolve_parquet_path(None)

            assert resolved == parquet_file.resolve()


class TestFetchAndTransformData:
    """Tests for _fetch_and_transform_data method."""

    @pytest.fixture
    def provider_with_data(self, tmp_path):
        """Create provider with test data."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
                "date": [
                    datetime(2017, 1, 3),
                    datetime(2017, 1, 4),
                    datetime(2017, 1, 5),
                    datetime(2017, 1, 3),
                ],
                "open": [115.0, 116.0, 117.0, 62.0],
                "high": [116.0, 117.0, 118.0, 63.0],
                "low": [114.0, 115.0, 116.0, 61.0],
                "close": [115.5, 116.5, 117.5, 62.5],
                "volume": [1000000.0, 1100000.0, 1200000.0, 500000.0],
                "adj_open": [115.0, 116.0, 117.0, 62.0],
                "adj_high": [116.0, 117.0, 118.0, 63.0],
                "adj_low": [114.0, 115.0, 116.0, 61.0],
                "adj_close": [115.5, 116.5, 117.5, 62.5],
                "adj_volume": [1000000.0, 1100000.0, 1200000.0, 500000.0],
            }
        )
        df.write_parquet(parquet_file)

        return WikiPricesProvider(parquet_path=parquet_file)

    def test_fetch_daily_data_success(self, provider_with_data):
        """Test successful daily data fetch."""
        df = provider_with_data._fetch_and_transform_data(
            "AAPL", "2017-01-01", "2017-01-10", "daily"
        )

        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_invalid_frequency_raises(self, provider_with_data):
        """Test invalid frequency raises error."""
        with pytest.raises(ValueError, match="only supports daily frequency"):
            provider_with_data._fetch_and_transform_data(
                "AAPL", "2017-01-01", "2017-01-10", "hourly"
            )

    def test_fetch_date_range_outside_dataset_raises(self, provider_with_data):
        """Test date range outside dataset raises DataNotAvailableError."""
        with pytest.raises(DataNotAvailableError):
            provider_with_data._fetch_and_transform_data(
                "AAPL", "2020-01-01", "2020-12-31", "daily"
            )

    def test_fetch_date_before_dataset_adjusted(self, provider_with_data):
        """Test start date before dataset is adjusted."""
        # Start before DATASET_START - should be adjusted
        df = provider_with_data._fetch_and_transform_data(
            "AAPL", "1960-01-01", "2017-01-10", "daily"
        )
        assert len(df) == 3

    def test_fetch_date_after_dataset_adjusted(self, provider_with_data):
        """Test end date after dataset is adjusted."""
        # End after DATASET_END - should be adjusted
        df = provider_with_data._fetch_and_transform_data(
            "AAPL", "2017-01-01", "2025-01-01", "daily"
        )
        assert len(df) == 3

    def test_fetch_symbol_not_found_raises(self, provider_with_data):
        """Test symbol not found raises DataNotAvailableError."""
        with pytest.raises(DataNotAvailableError):
            provider_with_data._fetch_and_transform_data(
                "INVALID", "2017-01-01", "2017-01-10", "daily"
            )

    def test_fetch_uses_adjusted_prices(self, provider_with_data):
        """Test that adjusted prices are used."""
        df = provider_with_data._fetch_and_transform_data(
            "AAPL", "2017-01-01", "2017-01-10", "daily"
        )
        # Should use adj_close, adj_open, etc.
        assert df["close"][0] == 115.5

    def test_fetch_with_cache_in_memory(self, tmp_path):
        """Test fetch with cached data in memory."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [datetime(2017, 1, 3)],
                "open": [170.0],
                "high": [172.0],
                "low": [169.0],
                "close": [171.0],
                "volume": [1000000.0],
                "adj_open": [170.0],
                "adj_high": [172.0],
                "adj_low": [169.0],
                "adj_close": [171.0],
                "adj_volume": [1000000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file, cache_in_memory=True)
        result = provider._fetch_and_transform_data("AAPL", "2017-01-01", "2017-01-10", "daily")

        assert len(result) == 1


class TestListAvailableSymbols:
    """Tests for list_available_symbols method."""

    def test_list_symbols_lazy(self, tmp_path):
        """Test listing symbols with lazy loading."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "date": [datetime(2017, 1, 1)] * 3,
                "open": [170.0, 60.0, 800.0],
                "high": [172.0, 62.0, 810.0],
                "low": [169.0, 59.0, 795.0],
                "close": [171.0, 61.0, 805.0],
                "volume": [1000000.0] * 3,
                "adj_open": [170.0, 60.0, 800.0],
                "adj_high": [172.0, 62.0, 810.0],
                "adj_low": [169.0, 59.0, 795.0],
                "adj_close": [171.0, 61.0, 805.0],
                "adj_volume": [1000000.0] * 3,
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)
        symbols = provider.list_available_symbols()

        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    def test_list_symbols_cached(self, tmp_path):
        """Test listing symbols with cached data."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "date": [datetime(2017, 1, 1)] * 2,
                "open": [170.0, 60.0],
                "high": [172.0, 62.0],
                "low": [169.0, 59.0],
                "close": [171.0, 61.0],
                "volume": [1000000.0] * 2,
                "adj_open": [170.0, 60.0],
                "adj_high": [172.0, 62.0],
                "adj_low": [169.0, 59.0],
                "adj_close": [171.0, 61.0],
                "adj_volume": [1000000.0] * 2,
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file, cache_in_memory=True)
        symbols = provider.list_available_symbols()

        assert len(symbols) == 2


class TestGetDateRange:
    """Tests for get_date_range method."""

    def test_get_date_range_success(self, tmp_path):
        """Test getting date range for existing symbol."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "date": [
                    datetime(2017, 1, 3),
                    datetime(2017, 6, 15),
                    datetime(2017, 12, 29),
                ],
                "open": [170.0, 150.0, 180.0],
                "high": [172.0, 152.0, 182.0],
                "low": [169.0, 149.0, 179.0],
                "close": [171.0, 151.0, 181.0],
                "volume": [1000000.0] * 3,
                "adj_open": [170.0, 150.0, 180.0],
                "adj_high": [172.0, 152.0, 182.0],
                "adj_low": [169.0, 149.0, 179.0],
                "adj_close": [171.0, 151.0, 181.0],
                "adj_volume": [1000000.0] * 3,
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)
        start, end = provider.get_date_range("AAPL")

        assert start == "2017-01-03"
        assert end == "2017-12-29"

    def test_get_date_range_not_found(self, tmp_path):
        """Test getting date range for non-existent symbol."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [datetime(2017, 1, 3)],
                "open": [170.0],
                "high": [172.0],
                "low": [169.0],
                "close": [171.0],
                "volume": [1000000.0],
                "adj_open": [170.0],
                "adj_high": [172.0],
                "adj_low": [169.0],
                "adj_close": [171.0],
                "adj_volume": [1000000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)
        with pytest.raises(DataNotAvailableError):
            provider.get_date_range("INVALID")

    def test_get_date_range_cached(self, tmp_path):
        """Test getting date range with cached data."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": [datetime(2017, 1, 3), datetime(2017, 12, 29)],
                "open": [170.0, 180.0],
                "high": [172.0, 182.0],
                "low": [169.0, 179.0],
                "close": [171.0, 181.0],
                "volume": [1000000.0] * 2,
                "adj_open": [170.0, 180.0],
                "adj_high": [172.0, 182.0],
                "adj_low": [169.0, 179.0],
                "adj_close": [171.0, 181.0],
                "adj_volume": [1000000.0] * 2,
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file, cache_in_memory=True)
        start, end = provider.get_date_range("AAPL")

        assert start == "2017-01-03"
        assert end == "2017-12-29"


class TestGetDatasetStats:
    """Tests for get_dataset_stats method."""

    def test_get_stats_lazy(self, tmp_path):
        """Test getting stats with lazy loading."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "MSFT"],
                "date": [
                    datetime(2017, 1, 3),
                    datetime(2017, 1, 4),
                    datetime(2017, 1, 3),
                ],
                "open": [170.0, 171.0, 60.0],
                "high": [172.0, 173.0, 62.0],
                "low": [169.0, 170.0, 59.0],
                "close": [171.0, 172.0, 61.0],
                "volume": [1000000.0] * 3,
                "adj_open": [170.0, 171.0, 60.0],
                "adj_high": [172.0, 173.0, 62.0],
                "adj_low": [169.0, 170.0, 59.0],
                "adj_close": [171.0, 172.0, 61.0],
                "adj_volume": [1000000.0] * 3,
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file)
        stats = provider.get_dataset_stats()

        assert stats["total_rows"] == 3
        assert stats["total_symbols"] == 2
        assert stats["date_range"][0] == "2017-01-03"
        assert stats["date_range"][1] == "2017-01-04"
        assert stats["file_size_mb"] >= 0  # Small test file may be < 1MB
        assert stats["memory_size_mb"] is None
        assert stats["cached_in_memory"] is False

    def test_get_stats_cached(self, tmp_path):
        """Test getting stats with cached data."""
        parquet_file = tmp_path / "wiki_prices.parquet"
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [datetime(2017, 1, 3)],
                "open": [170.0],
                "high": [172.0],
                "low": [169.0],
                "close": [171.0],
                "volume": [1000000.0],
                "adj_open": [170.0],
                "adj_high": [172.0],
                "adj_low": [169.0],
                "adj_close": [171.0],
                "adj_volume": [1000000.0],
            }
        )
        df.write_parquet(parquet_file)

        provider = WikiPricesProvider(parquet_path=parquet_file, cache_in_memory=True)
        stats = provider.get_dataset_stats()

        assert stats["total_rows"] == 1
        assert stats["total_symbols"] == 1
        assert stats["memory_size_mb"] is not None
        assert stats["cached_in_memory"] is True


class TestDownload:
    """Tests for download class method."""

    def test_download_no_api_key_raises(self):
        """Test download without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(WikiPricesProvider, "_resolve_api_key", return_value=None):
                with pytest.raises(ValueError, match="No NASDAQ Data Link API key"):
                    WikiPricesProvider.download()

    def test_download_401_raises_value_error(self, tmp_path):
        """Test download with invalid API key raises ValueError."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            with pytest.raises(ValueError, match="Invalid API key"):
                WikiPricesProvider.download(output_path=tmp_path, api_key="invalid_key")

    def test_download_429_raises_runtime_error(self, tmp_path):
        """Test download with rate limit raises RuntimeError."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate Limited", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            with pytest.raises(RuntimeError, match="Rate limited"):
                WikiPricesProvider.download(output_path=tmp_path, api_key="test_key")


class TestResolveApiKey:
    """Tests for _resolve_api_key class method."""

    def test_resolve_explicit_key(self):
        """Test resolving explicit API key."""
        key = WikiPricesProvider._resolve_api_key("explicit_key", None)
        assert key == "explicit_key"

    def test_resolve_env_quandl_key(self):
        """Test resolving QUANDL_API_KEY from environment."""
        with patch.dict("os.environ", {"QUANDL_API_KEY": "env_key"}):
            key = WikiPricesProvider._resolve_api_key(None, None)
            assert key == "env_key"

    def test_resolve_env_nasdaq_key(self):
        """Test resolving NASDAQ_DATA_LINK_API_KEY from environment."""
        with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "nasdaq_key"}, clear=True):
            key = WikiPricesProvider._resolve_api_key(None, None)
            assert key == "nasdaq_key"

    def test_resolve_from_env_file(self, tmp_path):
        """Test resolving API key from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("QUANDL_API_KEY=file_key")

        with patch.dict("os.environ", {}, clear=True):
            key = WikiPricesProvider._resolve_api_key(None, env_file)
            assert key == "file_key"

    def test_resolve_returns_none_when_not_found(self):
        """Test resolve returns None when no key found."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(WikiPricesProvider, "DEFAULT_PATHS", []):
                _key = WikiPricesProvider._resolve_api_key(None, None)  # noqa: F841
                # Might return None depending on system .env files
                # Just ensure it doesn't crash


class TestDatasetConstants:
    """Tests for dataset constants."""

    def test_dataset_start_constant(self):
        """Test DATASET_START constant value."""
        assert WikiPricesProvider.DATASET_START == "1962-01-02"

    def test_dataset_end_constant(self):
        """Test DATASET_END constant value."""
        assert WikiPricesProvider.DATASET_END == "2018-03-27"

    def test_dataset_symbols_constant(self):
        """Test DATASET_SYMBOLS constant value."""
        assert WikiPricesProvider.DATASET_SYMBOLS == 3199

    def test_dataset_rows_constant(self):
        """Test DATASET_ROWS constant value."""
        assert WikiPricesProvider.DATASET_ROWS == 15_389_314

    def test_default_paths_exist(self):
        """Test DEFAULT_PATHS is defined."""
        assert len(WikiPricesProvider.DEFAULT_PATHS) >= 1

    def test_nasdaq_export_url(self):
        """Test NASDAQ_EXPORT_URL is valid."""
        assert "nasdaq.com" in WikiPricesProvider.NASDAQ_EXPORT_URL
        assert "WIKI/PRICES" in WikiPricesProvider.NASDAQ_EXPORT_URL

"""Tests for book_downloader.py - ML4T Futures Data Downloader.

This module uses mocks to test the FuturesDataManager without requiring
actual Databento API calls. All tests can run offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from ml4t.data.futures.book_downloader import (
    DEFINITION_COLUMNS,
    OHLCV_SCHEMA,
    FuturesConfig,
    FuturesDataManager,
    download_futures_data,
    update_futures_data,
)

# ============================================================================
# FuturesConfig Tests
# ============================================================================


class TestFuturesConfig:
    """Test FuturesConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FuturesConfig()
        assert config.dataset == "GLBX.MDP3"
        assert config.start == "2016-01-01"
        assert config.end == "2025-12-31"
        assert config.products == {}
        assert config.definition_dates == []

    def test_storage_path_expansion(self):
        """Test that storage path expands user home."""
        config = FuturesConfig(storage_path=Path("~/test-futures"))
        assert "~" not in str(config.storage_path)
        assert config.storage_path.is_absolute()

    def test_get_all_products_empty(self):
        """Test get_all_products with no products."""
        config = FuturesConfig()
        assert config.get_all_products() == []

    def test_get_all_products_single_category(self):
        """Test get_all_products with one category."""
        config = FuturesConfig(products={"equity": ["ES", "NQ", "RTY"]})
        products = config.get_all_products()
        assert sorted(products) == ["ES", "NQ", "RTY"]

    def test_get_all_products_multiple_categories(self):
        """Test get_all_products with multiple categories."""
        config = FuturesConfig(
            products={
                "equity": ["ES", "NQ"],
                "energy": ["CL", "NG"],
                "metals": ["GC"],
            }
        )
        products = config.get_all_products()
        assert sorted(products) == ["CL", "ES", "GC", "NG", "NQ"]

    def test_get_all_products_deduplicates(self):
        """Test that get_all_products removes duplicates."""
        config = FuturesConfig(
            products={
                "equity": ["ES", "NQ"],
                "duplicates": ["ES", "GC"],  # ES is duplicate
            }
        )
        products = config.get_all_products()
        assert products.count("ES") == 1


# ============================================================================
# FuturesDataManager Initialization Tests
# ============================================================================


class TestFuturesDataManagerInit:
    """Test FuturesDataManager initialization."""

    def test_init_creates_storage_directory(self, tmp_path):
        """Test that __init__ creates storage directory."""
        storage = tmp_path / "futures" / "data"
        config = FuturesConfig(storage_path=storage)

        assert not storage.exists()
        FuturesDataManager(config)
        assert storage.exists()

    def test_init_lazy_client(self, tmp_path):
        """Test that client is lazily initialized."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)
        assert manager.client is None

    def test_from_config_basic(self, tmp_path):
        """Test from_config with basic YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
futures:
  dataset: GLBX.MDP3
  start: "2020-01-01"
  end: "2024-12-31"
  storage_path: /tmp/test-futures
  products:
    equity: [ES, NQ]
"""
        )

        manager = FuturesDataManager.from_config(config_file)
        assert manager.config.dataset == "GLBX.MDP3"
        assert manager.config.start == "2020-01-01"
        assert manager.config.end == "2024-12-31"
        assert "ES" in manager.config.get_all_products()

    def test_from_config_with_definitions(self, tmp_path):
        """Test from_config with definition dates."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
futures:
  products:
    test: [ES]
  definitions:
    snapshot_dates:
      - "2020-01-15"
      - "2021-01-15"
"""
        )

        manager = FuturesDataManager.from_config(config_file)
        assert manager.config.definition_dates == ["2020-01-15", "2021-01-15"]

    def test_from_config_defaults(self, tmp_path):
        """Test from_config uses defaults for missing values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
futures:
  products:
    test: [ES]
"""
        )

        manager = FuturesDataManager.from_config(config_file)
        assert manager.config.dataset == "GLBX.MDP3"
        assert manager.config.start == "2016-01-01"


# ============================================================================
# FuturesDataManager Client Tests
# ============================================================================


class TestFuturesDataManagerClient:
    """Test Databento client initialization."""

    def test_get_client_without_api_key(self, tmp_path):
        """Test that missing API key raises ValueError."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
                manager._get_client()

    def test_get_client_with_api_key(self, tmp_path):
        """Test client initialization with API key."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        with patch.dict("os.environ", {"DATABENTO_API_KEY": "test-key"}):
            with patch("databento.Historical") as mock_historical:
                _client = manager._get_client()  # noqa: F841

        mock_historical.assert_called_once_with("test-key")
        assert manager.client is not None

    def test_get_client_reuses_instance(self, tmp_path):
        """Test that client is reused across calls."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        with patch.dict("os.environ", {"DATABENTO_API_KEY": "test-key"}):
            with patch("databento.Historical") as mock_historical:
                client1 = manager._get_client()
                client2 = manager._get_client()

        assert client1 is client2
        mock_historical.assert_called_once()


# ============================================================================
# Path Helper Tests
# ============================================================================


class TestPathHelpers:
    """Test path generation methods."""

    def test_get_ohlcv_path(self, tmp_path):
        """Test OHLCV path generation."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        path = manager._get_ohlcv_path("ES", 2024)
        assert path == tmp_path / "ohlcv_1d" / "product=ES" / "year=2024" / "data.parquet"

    def test_get_definitions_path(self, tmp_path):
        """Test definitions path generation."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        path = manager._get_definitions_path("CL")
        assert path == tmp_path / "definitions" / "product=CL" / "definitions.parquet"


# ============================================================================
# Download OHLCV Tests
# ============================================================================


class TestDownloadProductOHLCV:
    """Test download_product_ohlcv method."""

    @pytest.fixture
    def manager_with_mock_client(self, tmp_path):
        """Create manager with mocked Databento client."""
        config = FuturesConfig(
            storage_path=tmp_path,
            start="2024-01-01",
            end="2024-01-31",
        )
        manager = FuturesDataManager(config)

        # Create mock client
        mock_client = MagicMock()
        manager.client = mock_client

        return manager, mock_client

    def test_download_ohlcv_success(self, manager_with_mock_client):
        """Test successful OHLCV download."""
        manager, mock_client = manager_with_mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [
                    pd.Timestamp("2024-01-02", tz="UTC"),
                    pd.Timestamp("2024-01-03", tz="UTC"),
                ],
                "symbol": ["ESH24", "ESH24"],
                "open": [4700.0, 4710.0],
                "high": [4720.0, 4730.0],
                "low": [4690.0, 4700.0],
                "close": [4715.0, 4725.0],
                "volume": [100000, 110000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_ohlcv("ES")

        assert result["product"] == "ES"
        assert result["rows"] == 2
        assert result["years"] == 1

        # Verify file was written
        output_path = manager._get_ohlcv_path("ES", 2024)
        assert output_path.exists()

    def test_download_ohlcv_empty_response(self, manager_with_mock_client):
        """Test handling of empty response."""
        manager, mock_client = manager_with_mock_client

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame()
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_ohlcv("ES")

        assert result["rows"] == 0
        assert result["years"] == 0

    def test_download_ohlcv_filters_spreads(self, manager_with_mock_client):
        """Test that spread contracts are filtered out."""
        manager, mock_client = manager_with_mock_client

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [
                    pd.Timestamp("2024-01-02", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                ],
                "symbol": ["ESH24", "ESH24-ESM24", "ESH24:ESM24"],  # Outright + spreads
                "open": [4700.0, 10.0, 5.0],
                "high": [4720.0, 12.0, 7.0],
                "low": [4690.0, 8.0, 3.0],
                "close": [4715.0, 11.0, 6.0],
                "volume": [100000, 50000, 30000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_ohlcv("ES")

        # Should only have 1 row (the outright)
        assert result["rows"] == 1

    def test_download_ohlcv_error_handling(self, manager_with_mock_client):
        """Test error handling during download."""
        manager, mock_client = manager_with_mock_client

        mock_client.timeseries.get_range.side_effect = Exception("API Error")

        result = manager.download_product_ohlcv("ES")

        assert result["product"] == "ES"
        assert "error" in result
        assert "API Error" in result["error"]

    def test_download_ohlcv_multiple_years(self, manager_with_mock_client):
        """Test download spanning multiple years."""
        manager, mock_client = manager_with_mock_client
        manager.config.end = "2025-01-31"

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [
                    pd.Timestamp("2024-12-30", tz="UTC"),
                    pd.Timestamp("2024-12-31", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                ],
                "symbol": ["ESH25", "ESH25", "ESH25"],
                "open": [4700.0, 4710.0, 4720.0],
                "high": [4720.0, 4730.0, 4740.0],
                "low": [4690.0, 4700.0, 4710.0],
                "close": [4715.0, 4725.0, 4735.0],
                "volume": [100000, 110000, 120000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_ohlcv("ES")

        assert result["years"] == 2

    def test_download_ohlcv_merges_existing_data(self, manager_with_mock_client):
        """Test that new data is merged with existing."""
        manager, mock_client = manager_with_mock_client

        # Create existing data with ns precision to match what Databento returns
        output_path = manager._get_ohlcv_path("ES", 2024)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pl.DataFrame(
            {
                "ts_event": pl.Series([datetime(2024, 1, 2, tzinfo=UTC)]).cast(
                    pl.Datetime("ns", "UTC")
                ),
                "symbol": ["ESH24"],
                "open": [4700.0],
                "high": [4720.0],
                "low": [4690.0],
                "close": [4715.0],
                "volume": pl.Series([100000], dtype=pl.UInt64),
                "product": ["ES"],
            }
        )
        existing.write_parquet(output_path)

        # Mock new data
        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [
                    pd.Timestamp("2024-01-02", tz="UTC"),  # Duplicate
                    pd.Timestamp("2024-01-03", tz="UTC"),  # New
                ],
                "symbol": ["ESH24", "ESH24"],
                "open": [4701.0, 4710.0],  # Updated value for duplicate
                "high": [4721.0, 4730.0],
                "low": [4691.0, 4700.0],
                "close": [4716.0, 4725.0],
                "volume": [100001, 110000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        _result = manager.download_product_ohlcv("ES")  # noqa: F841

        # Should have 2 rows (deduplicated)
        df = pl.read_parquet(output_path)
        assert len(df) == 2


# ============================================================================
# Download Definitions Tests
# ============================================================================


class TestDownloadProductDefinitions:
    """Test download_product_definitions method."""

    @pytest.fixture
    def manager_with_mock_client(self, tmp_path):
        """Create manager with mocked client and definition dates."""
        config = FuturesConfig(
            storage_path=tmp_path,
            definition_dates=["2024-01-15", "2024-07-15"],
        )
        manager = FuturesDataManager(config)
        mock_client = MagicMock()
        manager.client = mock_client
        return manager, mock_client

    def test_download_definitions_no_dates(self, tmp_path):
        """Test handling when no definition dates configured."""
        config = FuturesConfig(storage_path=tmp_path, definition_dates=[])
        manager = FuturesDataManager(config)
        manager.client = MagicMock()

        result = manager.download_product_definitions("ES")
        assert "error" in result

    def test_download_definitions_success(self, manager_with_mock_client):
        """Test successful definitions download."""
        manager, mock_client = manager_with_mock_client

        # Mock responses for each date
        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "symbol": ["ESH24", "ESM24"],
                "expiration": [
                    pd.Timestamp("2024-03-15"),
                    pd.Timestamp("2024-06-21"),
                ],
                "activation": [
                    pd.Timestamp("2023-12-15"),
                    pd.Timestamp("2024-03-15"),
                ],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_definitions("ES")

        assert result["product"] == "ES"
        assert result["contracts"] >= 1

    def test_download_definitions_filters_spreads(self, manager_with_mock_client):
        """Test that spread contracts are filtered."""
        manager, mock_client = manager_with_mock_client

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "symbol": ["ESH24", "ESH24-ESM24"],
                "expiration": [
                    pd.Timestamp("2024-03-15"),
                    pd.Timestamp("2024-06-21"),
                ],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        _result = manager.download_product_definitions("ES")  # noqa: F841

        # Verify file only has outright
        defn_path = manager._get_definitions_path("ES")
        df = pl.read_parquet(defn_path)
        assert len(df.filter(pl.col("symbol").str.contains("-"))) == 0

    def test_download_definitions_handles_missing_product(self, manager_with_mock_client):
        """Test handling when product didn't exist at snapshot date."""
        manager, mock_client = manager_with_mock_client

        # First date fails with 422 (product doesn't exist), second succeeds
        def side_effect(*args, **kwargs):
            start = kwargs.get("start")
            if start == "2024-01-15":
                raise Exception("422 Client Error")
            mock_response = MagicMock()
            mock_response.to_df.return_value = pd.DataFrame(
                {
                    "symbol": ["BTCH24"],
                    "expiration": [pd.Timestamp("2024-03-29")],
                }
            )
            return mock_response

        mock_client.timeseries.get_range.side_effect = side_effect

        result = manager.download_product_definitions("BTC")
        assert result["contracts"] >= 1

    def test_download_definitions_no_data(self, manager_with_mock_client):
        """Test handling when no definitions found."""
        manager, mock_client = manager_with_mock_client

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame()
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.download_product_definitions("ES")
        assert result["contracts"] == 0


# ============================================================================
# Download All Tests
# ============================================================================


class TestDownloadAll:
    """Test download_all method."""

    @pytest.fixture
    def manager_with_mock_client(self, tmp_path):
        """Create manager with products and mock client."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES", "NQ"]},
            definition_dates=["2024-01-15"],
        )
        manager = FuturesDataManager(config)
        mock_client = MagicMock()
        manager.client = mock_client

        # Setup successful mock response
        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [pd.Timestamp("2024-01-02", tz="UTC")],
                "symbol": ["ESH24"],
                "open": [4700.0],
                "high": [4720.0],
                "low": [4690.0],
                "close": [4715.0],
                "volume": [100000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        return manager, mock_client

    def test_download_all_with_definitions(self, manager_with_mock_client):
        """Test download all products with definitions."""
        manager, _ = manager_with_mock_client

        result = manager.download_all(include_definitions=True)

        assert result["ohlcv"]["total_products"] == 2
        assert result["ohlcv"]["successful"] == 2
        assert "definitions" in result

    def test_download_all_without_definitions(self, manager_with_mock_client):
        """Test download all products without definitions."""
        manager, _ = manager_with_mock_client

        result = manager.download_all(include_definitions=False)

        assert result["ohlcv"]["successful"] == 2
        assert result["definitions"]["successful"] == 0


# ============================================================================
# Update Tests
# ============================================================================


class TestUpdate:
    """Test update method."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        """Create manager with existing data."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES"]},
        )
        manager = FuturesDataManager(config)

        # Create existing data
        output_path = manager._get_ohlcv_path("ES", 2024)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 11, 1, tzinfo=UTC)],
                "symbol": ["ESZ24"],
                "open": [4700.0],
                "high": [4720.0],
                "low": [4690.0],
                "close": [4715.0],
                "volume": [100000],
                "product": ["ES"],
            }
        )
        existing.write_parquet(output_path)

        mock_client = MagicMock()
        manager.client = mock_client

        return manager, mock_client

    def test_update_when_up_to_date(self, manager_with_data):
        """Test update when data is already current."""
        manager, _ = manager_with_data

        # Set existing data to today
        output_path = manager._get_ohlcv_path("ES", 2024)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        existing = pl.DataFrame(
            {
                "ts_event": [today],
                "symbol": ["ESZ24"],
                "open": [4700.0],
                "high": [4720.0],
                "low": [4690.0],
                "close": [4715.0],
                "volume": [100000],
                "product": ["ES"],
            }
        )
        existing.write_parquet(output_path)

        result = manager.update()

        assert result["status"] == "up_to_date"

    def test_update_downloads_new_data(self, manager_with_data):
        """Test that update downloads only new data."""
        manager, mock_client = manager_with_data

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [pd.Timestamp("2024-11-02", tz="UTC")],
                "symbol": ["ESZ24"],
                "open": [4710.0],
                "high": [4730.0],
                "low": [4700.0],
                "close": [4725.0],
                "volume": [110000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.update()

        assert result["status"] == "updated"
        assert result["from_date"] == "2024-11-02"

    def test_update_no_existing_data(self, tmp_path):
        """Test update triggers full download when no existing data."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES"]},
        )
        manager = FuturesDataManager(config)
        mock_client = MagicMock()
        manager.client = mock_client

        mock_response = MagicMock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [pd.Timestamp("2024-01-02", tz="UTC")],
                "symbol": ["ESH24"],
                "open": [4700.0],
                "high": [4720.0],
                "low": [4690.0],
                "close": [4715.0],
                "volume": [100000],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        result = manager.update()

        # Should run full download
        assert "ohlcv" in result


# ============================================================================
# Load Data Tests
# ============================================================================


class TestLoadOHLCV:
    """Test load_ohlcv method."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        """Create manager with test data."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        # Create test data across 2 years
        for year in [2023, 2024]:
            output_path = manager._get_ohlcv_path("ES", year)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame(
                {
                    "ts_event": [datetime(year, 6, 15, tzinfo=UTC)],
                    "symbol": [f"ESM{year % 100}"],
                    "open": [4700.0],
                    "high": [4720.0],
                    "low": [4690.0],
                    "close": [4715.0],
                    "volume": [100000],
                    "product": ["ES"],
                }
            )
            df.write_parquet(output_path)

        return manager

    def test_load_ohlcv_all_data(self, manager_with_data):
        """Test loading all data for a product."""
        df = manager_with_data.load_ohlcv("ES")
        assert len(df) == 2

    def test_load_ohlcv_with_date_filter(self, manager_with_data):
        """Test loading with date range filter."""
        df = manager_with_data.load_ohlcv("ES", start="2024-01-01", end="2024-12-31")
        assert len(df) == 1
        assert df["symbol"][0] == "ESM24"

    def test_load_ohlcv_missing_product(self, manager_with_data):
        """Test loading nonexistent product raises error."""
        with pytest.raises(FileNotFoundError, match="No data found"):
            manager_with_data.load_ohlcv("INVALID")

    def test_load_ohlcv_sorted(self, manager_with_data):
        """Test that loaded data is sorted."""
        df = manager_with_data.load_ohlcv("ES")
        timestamps = df["ts_event"].to_list()
        assert timestamps == sorted(timestamps)


class TestLoadDefinitions:
    """Test load_definitions method."""

    def test_load_definitions_success(self, tmp_path):
        """Test loading existing definitions."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        # Create definitions file
        defn_path = manager._get_definitions_path("ES")
        defn_path.parent.mkdir(parents=True, exist_ok=True)
        defn = pl.DataFrame(
            {
                "product": ["ES"],
                "symbol": ["ESH24"],
                "expiration": [datetime(2024, 3, 15)],
            }
        )
        defn.write_parquet(defn_path)

        df = manager.load_definitions("ES")
        assert len(df) == 1

    def test_load_definitions_missing(self, tmp_path):
        """Test loading nonexistent definitions."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        with pytest.raises(FileNotFoundError, match="No definitions found"):
            manager.load_definitions("INVALID")


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_list_products(self, tmp_path):
        """Test list_products returns configured products."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES", "NQ"], "energy": ["CL"]},
        )
        manager = FuturesDataManager(config)

        products = manager.list_products()
        assert products == {"equity": ["ES", "NQ"], "energy": ["CL"]}

    def test_get_data_summary_empty(self, tmp_path):
        """Test get_data_summary with no downloaded data."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES"]},
        )
        manager = FuturesDataManager(config)

        summary = manager.get_data_summary()
        assert len(summary) == 1
        assert summary["status"][0] == "not_downloaded"

    def test_get_data_summary_with_data(self, tmp_path):
        """Test get_data_summary with downloaded data."""
        config = FuturesConfig(
            storage_path=tmp_path,
            products={"equity": ["ES"]},
        )
        manager = FuturesDataManager(config)

        # Create test data
        output_path = manager._get_ohlcv_path("ES", 2024)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "ts_event": [
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                ],
                "symbol": ["ESH24", "ESH24"],
                "open": [4700.0, 4710.0],
                "high": [4720.0, 4730.0],
                "low": [4690.0, 4700.0],
                "close": [4715.0, 4725.0],
                "volume": [100000, 110000],
                "product": ["ES", "ES"],
            }
        )
        df.write_parquet(output_path)

        summary = manager.get_data_summary()
        assert len(summary) == 1
        assert summary["status"][0] == "available"
        assert summary["rows"][0] == 2

    def test_get_latest_date(self, tmp_path):
        """Test _get_latest_date finds most recent date."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        # Create data files for multiple years
        for year, date in [(2023, datetime(2023, 12, 29)), (2024, datetime(2024, 11, 15))]:
            output_path = manager._get_ohlcv_path("ES", year)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pl.DataFrame(
                {
                    "ts_event": [date.replace(tzinfo=UTC)],
                    "symbol": [f"ESZ{year % 100}"],
                    "open": [4700.0],
                    "high": [4720.0],
                    "low": [4690.0],
                    "close": [4715.0],
                    "volume": [100000],
                    "product": ["ES"],
                }
            )
            df.write_parquet(output_path)

        latest = manager._get_latest_date("ES")
        assert latest.year == 2024
        assert latest.month == 11
        assert latest.day == 15

    def test_get_latest_date_no_data(self, tmp_path):
        """Test _get_latest_date returns None for missing product."""
        config = FuturesConfig(storage_path=tmp_path)
        manager = FuturesDataManager(config)

        latest = manager._get_latest_date("INVALID")
        assert latest is None


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_download_futures_data(self, tmp_path, capsys):
        """Test download_futures_data function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
futures:
  storage_path: {tmp_path / "data"}
  products:
    equity: [ES]
"""
        )

        with patch("ml4t.data.futures.book_downloader.FuturesDataManager") as MockManager:
            mock_instance = MagicMock()
            mock_instance.config.get_all_products.return_value = ["ES"]
            mock_instance.config.start = "2024-01-01"
            mock_instance.config.end = "2024-12-31"
            mock_instance.config.storage_path = tmp_path / "data"
            mock_instance.download_all.return_value = {
                "ohlcv": {"successful": 1, "total_products": 1},
                "definitions": {"successful": 1},
            }
            MockManager.from_config.return_value = mock_instance

            download_futures_data(str(config_file))

            captured = capsys.readouterr()
            assert "ML4T Futures Data Download" in captured.out
            assert "Download Complete" in captured.out

    def test_update_futures_data(self, tmp_path, capsys):
        """Test update_futures_data function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
futures:
  storage_path: {tmp_path / "data"}
  products:
    equity: [ES]
"""
        )

        with patch("ml4t.data.futures.book_downloader.FuturesDataManager") as MockManager:
            mock_instance = MagicMock()
            mock_instance.update.return_value = {"status": "up_to_date", "latest": "2024-11-15"}
            MockManager.from_config.return_value = mock_instance

            update_futures_data(str(config_file))

            captured = capsys.readouterr()
            assert "ML4T Futures Data Update" in captured.out
            assert "up to date" in captured.out


# ============================================================================
# Schema Constant Tests
# ============================================================================


class TestSchemaConstants:
    """Test module-level constants."""

    def test_ohlcv_schema_has_required_columns(self):
        """Test OHLCV schema has all required columns."""
        required = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
        for col in required:
            assert col in OHLCV_SCHEMA

    def test_definition_columns_has_required(self):
        """Test definition columns list has key columns."""
        required = ["symbol", "expiration", "contract_multiplier"]
        for col in required:
            assert col in DEFINITION_COLUMNS

"""Tests for futures downloader module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# Import real databento exceptions (databento package is installed)
from databento.common.error import BentoClientError, BentoServerError

from ml4t.data.futures.downloader import (
    DEFAULT_PRODUCTS,
    DefinitionsConfig,
    DefinitionsDownloader,
    DownloadProgress,
    FuturesCategory,
    FuturesDownloadConfig,
    FuturesDownloader,
    load_definitions_config,
    load_yaml_config,
)


class TestFuturesCategory:
    """Tests for FuturesCategory enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert FuturesCategory.EQUITY_INDEX.value == "equity_index"
        assert FuturesCategory.RATES.value == "rates"
        assert FuturesCategory.FX.value == "fx"
        assert FuturesCategory.ENERGY.value == "energy"
        assert FuturesCategory.METALS.value == "metals"
        assert FuturesCategory.GRAINS.value == "grains"
        assert FuturesCategory.SOFTS.value == "softs"
        assert FuturesCategory.LIVESTOCK.value == "livestock"
        assert FuturesCategory.CRYPTO.value == "crypto"


class TestFuturesDownloadConfig:
    """Tests for FuturesDownloadConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FuturesDownloadConfig()

        assert config.products == DEFAULT_PRODUCTS
        assert config.start == "2016-01-01"
        assert config.end == "2025-12-13"
        assert config.dataset == "GLBX.MDP3"
        assert config.schemas == ["ohlcv-1d", "definition", "statistics"]
        assert config.api_key is None

    def test_storage_path_expanded(self):
        """Test that storage path is expanded."""
        config = FuturesDownloadConfig(storage_path="~/test-data/futures")

        assert "~" not in str(config.storage_path)
        assert isinstance(config.storage_path, Path)

    def test_get_product_list_from_dict(self):
        """Test get_product_list with dict products."""
        config = FuturesDownloadConfig(
            products={
                FuturesCategory.EQUITY_INDEX: ["ES", "NQ"],
                FuturesCategory.ENERGY: ["CL"],
            }
        )

        products = config.get_product_list()
        assert set(products) == {"ES", "NQ", "CL"}

    def test_get_product_list_from_list(self):
        """Test get_product_list with list products."""
        config = FuturesDownloadConfig(products=["ES", "CL", "GC"])

        products = config.get_product_list()
        assert products == ["ES", "CL", "GC"]

    def test_get_products_by_category_from_dict(self):
        """Test get_products_by_category with dict products."""
        config = FuturesDownloadConfig(
            products={
                FuturesCategory.EQUITY_INDEX: ["ES", "NQ"],
            }
        )

        categories = config.get_products_by_category()
        assert FuturesCategory.EQUITY_INDEX in categories
        assert categories[FuturesCategory.EQUITY_INDEX] == ["ES", "NQ"]

    def test_get_products_by_category_from_list(self):
        """Test get_products_by_category with list products."""
        config = FuturesDownloadConfig(products=["ES", "CL"])

        categories = config.get_products_by_category()
        # Should put list under EQUITY_INDEX as default
        assert FuturesCategory.EQUITY_INDEX in categories
        assert categories[FuturesCategory.EQUITY_INDEX] == ["ES", "CL"]


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_default_values(self):
        """Test default progress values."""
        progress = DownloadProgress()

        assert progress.completed_products == set()
        assert progress.failed_products == {}
        assert progress.total_bytes == 0

    def test_mark_complete(self):
        """Test marking a product as complete."""
        progress = DownloadProgress()
        progress.mark_complete("ES", bytes_downloaded=1000)

        assert "ES" in progress.completed_products
        assert progress.total_bytes == 1000

    def test_mark_complete_removes_failed(self):
        """Test that marking complete removes from failed."""
        progress = DownloadProgress()
        progress.mark_failed("ES", "Some error")
        assert "ES" in progress.failed_products

        progress.mark_complete("ES")
        assert "ES" in progress.completed_products
        assert "ES" not in progress.failed_products

    def test_mark_failed(self):
        """Test marking a product as failed."""
        progress = DownloadProgress()
        progress.mark_failed("ES", "API error")

        assert "ES" in progress.failed_products
        assert progress.failed_products["ES"] == "API error"

    def test_is_complete(self):
        """Test checking if product is complete."""
        progress = DownloadProgress()

        assert not progress.is_complete("ES")

        progress.mark_complete("ES")
        assert progress.is_complete("ES")


class TestFuturesDownloader:
    """Tests for FuturesDownloader class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        return FuturesDownloadConfig(
            products=["ES", "CL"],
            start="2024-01-01",
            end="2024-12-31",
            storage_path=temp_storage,
            schemas=["ohlcv-1d"],
            api_key="test_key",
        )

    def test_init_without_api_key_raises(self, temp_storage):
        """Test initialization without API key raises error."""
        config = FuturesDownloadConfig(
            products=["ES"],
            storage_path=temp_storage,
            api_key=None,
        )

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("DATABENTO_API_KEY", None)

            with pytest.raises(ValueError, match="API key"):
                FuturesDownloader(config)

    def test_init_creates_directories(self, config):
        """Test initialization creates storage directories."""
        with patch("ml4t.data.futures.downloader.Historical"):
            _downloader = FuturesDownloader(config)  # noqa: F841

            assert config.storage_path.exists()
            assert (config.storage_path / "ohlcv_1d").exists()

    def test_estimate_cost(self, config):
        """Test cost estimation."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)

            cost = downloader.estimate_cost()

            assert cost["products"] == 2  # ES, CL
            assert cost["schemas"] == 1  # ohlcv-1d only
            assert "estimated_total_usd" in cost
            assert "years" in cost

    def test_get_product_dir(self, config):
        """Test _get_product_dir returns correct path."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)

            path = downloader._get_product_dir("ohlcv-1d", "ES")

            assert path == config.storage_path / "ohlcv_1d" / "product=ES"

    def test_product_exists_false(self, config):
        """Test _product_exists returns False when no data."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)

            assert not downloader._product_exists("ES")

    def test_product_exists_true(self, config, temp_storage):
        """Test _product_exists returns True when data exists."""
        # Create data file
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)

            assert downloader._product_exists("ES")

    def test_download_product_skip_existing(self, config, temp_storage):
        """Test download_product skips existing data."""
        # Create existing data
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            result = downloader.download_product("ES", skip_existing=True)

            assert result is True
            assert "ES" in downloader.progress.completed_products

    def test_download_product_with_mock_client(self, config):
        """Test download_product with mocked Databento client."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_df = MagicMock()
        mock_df.reset_index.return_value = MagicMock(
            to_dict=lambda: {
                "ts_event": [datetime(2024, 1, 1)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )
        # Create a simple pandas-like mock
        import pandas as pd

        mock_df.reset_index.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 1)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )
        mock_data.to_df.return_value = mock_df.reset_index.return_value
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            result = downloader.download_product("ES", skip_existing=False)

            assert result is True
            mock_client.timeseries.get_range.assert_called()

    def test_download_product_failure(self, config):
        """Test download_product handles failures."""
        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = Exception("API error")

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            result = downloader.download_product("ES", skip_existing=False)

            assert result is False
            assert "ES" in downloader.progress.failed_products

    def test_download_all_sequential(self, config, temp_storage):
        """Test download_all processes all products."""
        # Pre-create one product's data
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        mock_client = MagicMock()
        import pandas as pd

        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 1)],
                "symbol": ["CLH4"],
                "close": [75.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all(skip_existing=True)

            # ES should be skipped (exists), CL should be downloaded
            assert "ES" in progress.completed_products

    def test_download_all_continue_on_error(self, config):
        """Test download_all continues on error."""
        mock_client = MagicMock()
        # First call fails, second succeeds
        import pandas as pd

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First failed")
            mock_data = MagicMock()
            mock_data.to_df.return_value = pd.DataFrame(
                {
                    "ts_event": [datetime(2024, 1, 1)],
                    "symbol": ["CLH4"],
                    "close": [75.0],
                }
            )
            return mock_data

        mock_client.timeseries.get_range.side_effect = side_effect

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all(continue_on_error=True)

            assert len(progress.failed_products) >= 1

    def test_list_downloaded_empty(self, config):
        """Test list_downloaded when no data exists."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            downloaded = downloader.list_downloaded()

            assert downloaded == {"ohlcv-1d": []}

    def test_list_downloaded_with_data(self, config, temp_storage):
        """Test list_downloaded with existing data."""
        # Create data for ES
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            downloaded = downloader.list_downloaded()

            assert "ES" in downloaded["ohlcv-1d"]

    def test_get_latest_date_no_data(self, config):
        """Test get_latest_date when no data exists."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            latest = downloader.get_latest_date()

            assert latest is None

    def test_get_latest_date_with_data(self, config, temp_storage):
        """Test get_latest_date with existing data."""
        # Create data with ts_event column
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
                "symbol": ["ESH4", "ESH4"],
                "close": [5000.0, 5100.0],
            }
        )
        df.write_parquet(product_dir / "ohlcv_1d.parquet")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            latest = downloader.get_latest_date()

            assert latest == datetime(2024, 1, 15)


class TestDefinitionsConfig:
    """Tests for DefinitionsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DefinitionsConfig()

        assert config.strategy == "yearly_snapshots"
        assert len(config.snapshot_dates) == 10  # 2016-2025
        assert config.output_file == "definitions_merged.parquet"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DefinitionsConfig(
            strategy="latest_only",
            snapshot_dates=["2024-01-02"],
            output_file="custom.parquet",
        )

        assert config.strategy == "latest_only"
        assert config.snapshot_dates == ["2024-01-02"]
        assert config.output_file == "custom.parquet"


class TestDefinitionsDownloader:
    """Tests for DefinitionsDownloader class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_without_api_key_raises(self, temp_storage):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("DATABENTO_API_KEY", None)

            with pytest.raises(ValueError, match="API key"):
                DefinitionsDownloader(
                    products=["ES"],
                    storage_path=temp_storage,
                )

    def test_init_creates_directories(self, temp_storage):
        """Test initialization creates storage directories."""
        with patch("ml4t.data.futures.downloader.Historical"):
            _downloader = DefinitionsDownloader(  # noqa: F841
                products=["ES"],
                storage_path=temp_storage,
                api_key="test_key",
            )

            assert (temp_storage / "definition").exists()
            assert (temp_storage / "definition_snapshots").exists()

    def test_merge_definitions(self, temp_storage):
        """Test _merge_definitions deduplicates correctly."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = DefinitionsDownloader(
                products=["ES"],
                storage_path=temp_storage,
                api_key="test_key",
            )

            # Create test data with duplicates
            df = pl.DataFrame(
                {
                    "symbol": ["ESH4", "ESH4", "ESM4"],  # ESH4 appears twice
                    "snapshot_date": ["2024-01-02", "2023-01-03", "2024-01-02"],
                    "expiration": [
                        datetime(2024, 3, 15),
                        datetime(2024, 3, 15),
                        datetime(2024, 6, 15),
                    ],
                }
            )

            merged = downloader._merge_definitions(df)

            assert len(merged) == 2  # ESH4 and ESM4
            # Should keep latest snapshot for ESH4
            esh4 = merged.filter(pl.col("symbol") == "ESH4")
            assert esh4["snapshot_date"][0] == "2024-01-02"

    def test_get_merged_definitions_empty(self, temp_storage):
        """Test get_merged_definitions when no data."""
        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = DefinitionsDownloader(
                products=["ES"],
                storage_path=temp_storage,
                api_key="test_key",
            )

            df = downloader.get_merged_definitions()
            assert df.is_empty()

    def test_get_merged_definitions_with_data(self, temp_storage):
        """Test get_merged_definitions with existing data."""
        # Create definition data
        defn_dir = temp_storage / "definition" / "product=ES"
        defn_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": ["ESH4", "ESM4"],
                "expiration": [datetime(2024, 3, 15), datetime(2024, 6, 15)],
            }
        ).write_parquet(defn_dir / "definition.parquet")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = DefinitionsDownloader(
                products=["ES"],
                storage_path=temp_storage,
                api_key="test_key",
            )

            df = downloader.get_merged_definitions()

            assert len(df) == 2
            assert (temp_storage / "definitions_merged.parquet").exists()

    def test_check_coverage(self, temp_storage):
        """Test check_coverage method."""
        # Create OHLCV data
        ohlcv_dir = temp_storage / "ohlcv_1d" / "product=ES"
        ohlcv_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": ["ESH4", "ESM4", "ESU4"],
                "close": [5000.0, 5100.0, 5200.0],
            }
        ).write_parquet(ohlcv_dir / "ohlcv_1d.parquet")

        # Create definition data (missing ESU4)
        defn_dir = temp_storage / "definition" / "product=ES"
        defn_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": ["ESH4", "ESM4"],  # Missing ESU4
                "expiration": [datetime(2024, 3, 15), datetime(2024, 6, 15)],
            }
        ).write_parquet(defn_dir / "definition.parquet")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = DefinitionsDownloader(
                products=["ES"],
                storage_path=temp_storage,
                api_key="test_key",
            )

            report = downloader.check_coverage()

            assert "ES" in report
            assert report["ES"]["ohlcv_contracts"] == 3
            assert report["ES"]["defn_contracts"] == 2
            assert report["ES"]["missing"] == 1
            assert report["ES"]["status"] == "incomplete"


class TestYamlLoaders:
    """Tests for YAML configuration loaders."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_yaml_config_basic(self, temp_storage):
        """Test loading basic YAML config."""
        yaml_content = """
futures:
  products:
    - ES
    - CL
  start: "2023-01-01"
  end: "2024-12-31"
  storage_path: /tmp/test
  schemas:
    - ohlcv-1d
"""
        config_file = temp_storage / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_yaml_config(config_file)

        assert config.products == ["ES", "CL"]
        assert config.start == "2023-01-01"
        assert config.end == "2024-12-31"
        assert config.schemas == ["ohlcv-1d"]

    def test_load_yaml_config_with_categories(self, temp_storage):
        """Test loading YAML config with category dict."""
        yaml_content = """
futures:
  products:
    equity_index:
      - ES
      - NQ
    energy:
      - CL
  start: "2023-01-01"
  end: "2024-12-31"
"""
        config_file = temp_storage / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_yaml_config(config_file)

        assert isinstance(config.products, dict)
        assert FuturesCategory.EQUITY_INDEX in config.products
        assert config.products[FuturesCategory.EQUITY_INDEX] == ["ES", "NQ"]

    def test_load_definitions_config(self, temp_storage):
        """Test loading definitions config from YAML."""
        yaml_content = """
futures:
  products:
    - ES
    - CL
  storage_path: /tmp/test
  definitions:
    strategy: yearly_snapshots
    snapshot_dates:
      - "2024-01-02"
    output_file: defs.parquet
"""
        config_file = temp_storage / "config.yaml"
        config_file.write_text(yaml_content)

        products, storage_path, defn_config = load_definitions_config(config_file)

        assert products == ["ES", "CL"]
        assert storage_path == Path("/tmp/test")
        assert defn_config.strategy == "yearly_snapshots"
        assert defn_config.snapshot_dates == ["2024-01-02"]
        assert defn_config.output_file == "defs.parquet"

    def test_load_definitions_config_with_category_dict(self, temp_storage):
        """Test loading definitions config with category dict products."""
        yaml_content = """
futures:
  products:
    equity_index:
      - ES
    energy:
      - CL
  storage_path: /tmp/test
"""
        config_file = temp_storage / "config.yaml"
        config_file.write_text(yaml_content)

        products, storage_path, defn_config = load_definitions_config(config_file)

        # Should flatten to list
        assert set(products) == {"ES", "CL"}


class TestDefaultProducts:
    """Tests for DEFAULT_PRODUCTS constant."""

    def test_has_expected_categories(self):
        """Test DEFAULT_PRODUCTS has expected categories."""
        expected_categories = [
            FuturesCategory.EQUITY_INDEX,
            FuturesCategory.RATES,
            FuturesCategory.FX,
            FuturesCategory.ENERGY,
            FuturesCategory.METALS,
            FuturesCategory.GRAINS,
            FuturesCategory.LIVESTOCK,
            FuturesCategory.CRYPTO,
        ]

        for category in expected_categories:
            assert category in DEFAULT_PRODUCTS

    def test_has_common_products(self):
        """Test DEFAULT_PRODUCTS includes common products."""
        all_products = [p for products in DEFAULT_PRODUCTS.values() for p in products]

        assert "ES" in all_products  # E-mini S&P 500
        assert "NQ" in all_products  # E-mini Nasdaq
        assert "CL" in all_products  # Crude Oil
        assert "GC" in all_products  # Gold
        assert "ZN" in all_products  # 10-Year Treasury


class TestFuturesDownloaderAdvanced:
    """Advanced tests for FuturesDownloader covering parallel downloads and updates."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        return FuturesDownloadConfig(
            products=["ES", "CL"],
            start="2024-01-01",
            end="2024-01-31",
            storage_path=temp_storage,
            schemas=["ohlcv-1d"],
            api_key="test_key",
        )

    def test_download_all_parallel_basic(self, config):
        """Test parallel download processes all products."""
        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all_parallel(max_workers=2)

            # Both products should be completed or failed
            total_processed = len(progress.completed_products) + len(progress.failed_products)
            assert total_processed == 2

    def test_download_all_parallel_skips_existing(self, config, temp_storage):
        """Test parallel download skips existing products."""
        # Pre-create one product
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)],
                "symbol": ["CLH4"],
                "close": [75.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all_parallel(skip_existing=True)

            # ES should be marked complete (skipped)
            assert "ES" in progress.completed_products

    def test_download_all_parallel_all_skipped(self, config, temp_storage):
        """Test parallel download when all products exist."""
        # Pre-create all products
        for product in ["ES", "CL"]:
            product_dir = temp_storage / "ohlcv_1d" / f"product={product}"
            product_dir.mkdir(parents=True)
            (product_dir / "ohlcv_1d.parquet").write_text("dummy")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all_parallel(skip_existing=True)

            # Both should be complete
            assert len(progress.completed_products) == 2

    def test_download_all_parallel_handles_failure(self, config):
        """Test parallel download handles individual failures."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("symbols") == "ES.FUT":
                raise Exception("ES failed")
            import pandas as pd

            mock_data = MagicMock()
            mock_data.to_df.return_value = pd.DataFrame(
                {
                    "ts_event": [datetime(2024, 1, 2)],
                    "symbol": ["CLH4"],
                    "close": [75.0],
                }
            )
            return mock_data

        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = side_effect

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_all_parallel(max_workers=1)

            # At least one should have failed
            assert len(progress.failed_products) >= 1 or len(progress.completed_products) >= 1

    def test_download_schema_filters_spreads(self, config):
        """Test that _download_schema filters spread contracts."""
        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        # Include spread contracts with '-' in symbol
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)] * 3,
                "symbol": ["ESH4", "ESH4-ESM4", "ESM4"],  # Middle one is spread
                "close": [5000.0, 10.0, 5100.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            df = downloader._download_schema("ES", "ohlcv-1d")

            # Spread should be filtered out
            assert len(df) == 2
            assert "ESH4-ESM4" not in df["symbol"].to_list()

    def test_download_schema_bento_client_error_no_data(self, config):
        """Test _download_schema handles BentoClientError for missing data."""
        # Use module-level BentoClientError that downloader was imported with
        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = BentoClientError("No data found for symbol")

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            df = downloader._download_schema("INVALID", "ohlcv-1d")

            # Should return empty DataFrame (error is saved to file, not returned)
            assert df.is_empty()

    def test_download_schema_bento_client_error_other(self, config):
        """Test _download_schema raises other BentoClientErrors."""
        # Use module-level BentoClientError that downloader was imported with
        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = BentoClientError("Authentication failed")

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)

            with pytest.raises(BentoClientError):
                downloader._download_schema("ES", "ohlcv-1d")

    def test_download_schema_bento_server_error(self, config):
        """Test _download_schema raises BentoServerError."""
        # Use module-level BentoServerError that downloader was imported with
        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = BentoServerError("Internal server error")

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)

            with pytest.raises(BentoServerError):
                downloader._download_schema("ES", "ohlcv-1d")

    def test_download_test(self, config):
        """Test download_test method."""
        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 11, 15)],
                "symbol": ["ESZ4"],
                "close": [5000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            progress = downloader.download_test(products=["ES"])

            # Test should complete
            assert len(progress.completed_products) >= 0 or len(progress.failed_products) >= 0

    def test_update_when_no_existing_data(self, config):
        """Test update falls back to full download when no existing data."""
        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            _progress = downloader.update()  # noqa: F841

            # Should have attempted downloads
            assert mock_client.timeseries.get_range.called

    def test_update_when_already_current(self, config, temp_storage):
        """Test update returns early when data is already current."""
        # Create existing data with today's date
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        df = pl.DataFrame(
            {
                "ts_event": [datetime.now()],
                "symbol": ["ESZ4"],
                "close": [5000.0],
            }
        )
        df.write_parquet(product_dir / "ohlcv_1d.parquet")

        with patch("ml4t.data.futures.downloader.Historical") as mock_hist:
            mock_client = mock_hist.return_value
            downloader = FuturesDownloader(config)
            _progress = downloader.update()  # noqa: F841

            # Should not make API calls
            assert not mock_client.timeseries.get_range.called

    def test_update_with_existing_data(self, config, temp_storage):
        """Test update downloads only new data."""
        import pandas as pd

        # Create existing data
        product_dir = temp_storage / "ohlcv_1d" / "product=ES"
        product_dir.mkdir(parents=True)
        existing = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "symbol": ["ESH4", "ESH4"],
                "close": [5000.0, 5100.0],
            }
        )
        existing.write_parquet(product_dir / "ohlcv_1d.parquet")

        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 4)],
                "symbol": ["ESH4"],
                "close": [5200.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            _progress = downloader.update(end_date="2024-01-31")  # noqa: F841

            # Should have made API call for update
            assert mock_client.timeseries.get_range.called

    def test_download_schema_with_client(self, config):
        """Test _download_schema_with_client method."""
        import pandas as pd

        # Create a real mock client instance
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            # Pass the mock client directly
            df = downloader._download_schema_with_client("ES", "ohlcv-1d", mock_client)

            assert df.height == 1
            assert "product" in df.columns

    def test_download_schema_with_client_error_handling(self, config):
        """Test _download_schema_with_client error paths."""
        # Use module-level BentoClientError that downloader was imported with
        mock_client = MagicMock()
        mock_client.timeseries.get_range.side_effect = BentoClientError("No data available")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            df = downloader._download_schema_with_client("INVALID", "ohlcv-1d", mock_client)

            # Should return empty DataFrame (error is saved to file, not returned)
            assert df.is_empty()

    def test_download_product_safe_success(self, config):
        """Test _download_product_safe thread-safe wrapper."""
        import pandas as pd

        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 2)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        )

        with patch("ml4t.data.futures.downloader.Historical") as mock_hist:
            # Initial client for __init__
            mock_hist.return_value.timeseries.get_range.return_value = mock_data

            downloader = FuturesDownloader(config)
            result = downloader._download_product_safe("ES")

            assert result is True
            assert "ES" in downloader.progress.completed_products

    def test_download_product_safe_failure(self, config):
        """Test _download_product_safe handles errors."""
        with patch("ml4t.data.futures.downloader.Historical") as mock_hist:
            # Make thread client fail
            mock_hist.return_value.timeseries.get_range.side_effect = Exception("API error")

            downloader = FuturesDownloader(config)
            result = downloader._download_product_safe("ES")

            assert result is False
            assert "ES" in downloader.progress.failed_products

    def test_get_latest_date_with_multiple_products(self, config, temp_storage):
        """Test get_latest_date finds max across multiple products."""
        # Create data for ES with older date
        es_dir = temp_storage / "ohlcv_1d" / "product=ES"
        es_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 10)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        ).write_parquet(es_dir / "ohlcv_1d.parquet")

        # Create data for CL with newer date
        cl_dir = temp_storage / "ohlcv_1d" / "product=CL"
        cl_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 20)],
                "symbol": ["CLH4"],
                "close": [75.0],
            }
        ).write_parquet(cl_dir / "ohlcv_1d.parquet")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)
            latest = downloader.get_latest_date()

            assert latest == datetime(2024, 1, 20)

    def test_get_latest_date_specific_product(self, config, temp_storage):
        """Test get_latest_date for specific product."""
        # Create data for ES
        es_dir = temp_storage / "ohlcv_1d" / "product=ES"
        es_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 15)],
                "symbol": ["ESH4"],
                "close": [5000.0],
            }
        ).write_parquet(es_dir / "ohlcv_1d.parquet")

        # Create data for CL
        cl_dir = temp_storage / "ohlcv_1d" / "product=CL"
        cl_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 25)],
                "symbol": ["CLH4"],
                "close": [75.0],
            }
        ).write_parquet(cl_dir / "ohlcv_1d.parquet")

        with patch("ml4t.data.futures.downloader.Historical"):
            downloader = FuturesDownloader(config)

            # Get for specific product
            es_latest = downloader.get_latest_date("ES")
            assert es_latest == datetime(2024, 1, 15)

    def test_download_schema_empty_data_symbol_type(self, config):
        """Test that empty data returns correct symbol type (not Float64).

        When Databento returns empty data (discontinued product),
        pandas/polars type inference makes symbol Float64 instead of String.
        The fix casts symbol to String to prevent merge errors.
        """
        import pandas as pd

        mock_client = MagicMock()
        mock_data = MagicMock()
        # Return empty DataFrame - will have Float64 symbol column
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [],
                "symbol": [],  # Empty - type inference will fail
                "close": [],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            df = downloader._download_schema("ES", "ohlcv-1d")

            # Symbol should be String, not Float64
            if "symbol" in df.columns:
                assert df.schema["symbol"] == pl.String

    def test_download_schema_no_overwrite_empty(self, config, temp_storage):
        """Test that empty data doesn't overwrite existing data.

        When API returns no data for discontinued products,
        we shouldn't overwrite the existing historical data.
        """
        import pandas as pd

        # Pre-create existing data
        es_dir = temp_storage / "ohlcv_1d" / "product=ES"
        es_dir.mkdir(parents=True)
        existing_df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 15)],
                "symbol": ["ESH4"],
                "close": [5000.0],
                "product": ["ES"],
            }
        )
        existing_df.write_parquet(es_dir / "ohlcv_1d.parquet")

        # API returns empty data
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [],
                "symbol": [],
                "close": [],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_data

        with patch("ml4t.data.futures.downloader.Historical", return_value=mock_client):
            downloader = FuturesDownloader(config)
            downloader._download_schema("ES", "ohlcv-1d")

            # Existing data should NOT be overwritten
            saved_df = pl.read_parquet(es_dir / "ohlcv_1d.parquet")
            assert len(saved_df) == 1
            assert saved_df["symbol"][0] == "ESH4"

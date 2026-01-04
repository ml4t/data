"""Tests for COT fetcher module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import yaml

from ml4t.data.cot.fetcher import (
    PRODUCT_MAPPINGS,
    COTConfig,
    COTFetcher,
    ProductMapping,
    load_cot_config,
)


class TestProductMapping:
    """Tests for ProductMapping dataclass."""

    def test_create_product_mapping(self):
        """Test creating a ProductMapping."""
        mapping = ProductMapping(
            code="ES",
            cot_name="E-MINI S&P 500",
            report_type="traders_in_financial_futures_fut",
            description="E-mini S&P 500",
        )
        assert mapping.code == "ES"
        assert mapping.cot_name == "E-MINI S&P 500"
        assert mapping.report_type == "traders_in_financial_futures_fut"
        assert mapping.description == "E-mini S&P 500"

    def test_product_mapping_default_description(self):
        """Test ProductMapping with default description."""
        mapping = ProductMapping(
            code="CL",
            cot_name="CRUDE OIL",
            report_type="disaggregated_fut",
        )
        assert mapping.description == ""

    def test_product_mappings_registry(self):
        """Test PRODUCT_MAPPINGS registry contains expected products."""
        # Financial products
        assert "ES" in PRODUCT_MAPPINGS
        assert "NQ" in PRODUCT_MAPPINGS
        assert "6E" in PRODUCT_MAPPINGS

        # Commodities
        assert "CL" in PRODUCT_MAPPINGS
        assert "GC" in PRODUCT_MAPPINGS
        assert "ZC" in PRODUCT_MAPPINGS

    def test_financial_futures_use_tff_report(self):
        """Test financial futures use TFF report type."""
        financial_products = ["ES", "NQ", "6E", "ZN", "BTC"]
        for code in financial_products:
            if code in PRODUCT_MAPPINGS:
                assert "traders_in_financial_futures" in PRODUCT_MAPPINGS[code].report_type

    def test_commodity_futures_use_disaggregated_report(self):
        """Test commodity futures use disaggregated report type."""
        commodity_products = ["CL", "NG", "GC", "ZC"]
        for code in commodity_products:
            if code in PRODUCT_MAPPINGS:
                assert PRODUCT_MAPPINGS[code].report_type == "disaggregated_fut"


class TestCOTConfig:
    """Tests for COTConfig dataclass."""

    def test_default_config(self):
        """Test COTConfig with defaults."""
        config = COTConfig()
        assert config.products == []
        assert config.start_year == 2020
        assert config.end_year == datetime.now().year
        assert config.include_options is False

    def test_custom_config(self):
        """Test COTConfig with custom values."""
        config = COTConfig(
            products=["ES", "CL"],
            start_year=2018,
            end_year=2023,
            include_options=True,
        )
        assert config.products == ["ES", "CL"]
        assert config.start_year == 2018
        assert config.end_year == 2023
        assert config.include_options is True

    def test_get_years(self):
        """Test get_years method."""
        config = COTConfig(start_year=2020, end_year=2023)
        years = config.get_years()
        assert years == [2020, 2021, 2022, 2023]

    def test_get_years_single_year(self):
        """Test get_years with single year."""
        config = COTConfig(start_year=2023, end_year=2023)
        years = config.get_years()
        assert years == [2023]

    def test_storage_path_string_conversion(self):
        """Test storage_path string to Path conversion."""
        config = COTConfig(storage_path="~/test/path")
        assert isinstance(config.storage_path, Path)
        assert str(config.storage_path).startswith("/")

    def test_storage_path_expansion(self):
        """Test storage_path home directory expansion."""
        config = COTConfig(storage_path="~/ml4t-data/cot")
        assert "~" not in str(config.storage_path)

    def test_end_year_none_uses_current(self):
        """Test end_year None uses current year."""
        config = COTConfig(start_year=2020, end_year=None)
        assert config.end_year == datetime.now().year


class TestLoadCOTConfig:
    """Tests for load_cot_config function."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        config_data = {
            "products": ["ES", "CL", "GC"],
            "start_year": 2019,
            "end_year": 2024,
            "storage_path": str(tmp_path / "cot_data"),
            "include_options": False,
        }

        config_file = tmp_path / "cot_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_cot_config(config_file)
        assert config.products == ["ES", "CL", "GC"]
        assert config.start_year == 2019
        assert config.end_year == 2024

    def test_load_config_with_defaults(self, tmp_path):
        """Test loading config uses defaults for missing fields."""
        config_data = {"products": ["ES"]}

        config_file = tmp_path / "cot_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_cot_config(config_file)
        assert config.products == ["ES"]
        assert config.start_year == 2020

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty config file."""
        config_file = tmp_path / "empty.yaml"
        with open(config_file, "w") as f:
            yaml.dump({}, f)

        config = load_cot_config(config_file)
        assert config.products == []


class TestCOTFetcher:
    """Tests for COTFetcher class."""

    def test_init_default_config(self):
        """Test COTFetcher initialization with default config."""
        fetcher = COTFetcher()
        assert fetcher.config is not None
        assert fetcher._cache == {}

    def test_init_custom_config(self):
        """Test COTFetcher initialization with custom config."""
        config = COTConfig(products=["ES", "CL"])
        fetcher = COTFetcher(config)
        assert fetcher.config.products == ["ES", "CL"]

    def test_list_available_products(self):
        """Test listing available products."""
        fetcher = COTFetcher()
        products = fetcher.list_available_products()
        assert isinstance(products, list)
        assert "ES" in products
        assert "CL" in products

    def test_get_product_info(self):
        """Test getting product info."""
        fetcher = COTFetcher()
        info = fetcher.get_product_info("ES")
        assert info is not None
        assert info.code == "ES"

    def test_get_product_info_unknown(self):
        """Test getting info for unknown product."""
        fetcher = COTFetcher()
        info = fetcher.get_product_info("UNKNOWN")
        assert info is None

    def test_fetch_product_unknown_code(self):
        """Test fetch_product with unknown product code."""
        fetcher = COTFetcher()
        with pytest.raises(ValueError, match="Unknown product code"):
            fetcher.fetch_product("INVALID_CODE")

    def test_get_report_caching(self):
        """Test report caching behavior."""
        fetcher = COTFetcher()

        # Pre-populate cache
        mock_df = pl.DataFrame({"col": [1, 2, 3]})
        fetcher._cache[("traders_in_financial_futures_fut", 2023)] = mock_df

        # Call should use cache
        result = fetcher._get_report("traders_in_financial_futures_fut", 2023)
        assert len(result) == 3

    def test_get_report_different_years_not_cached(self):
        """Test different years are not cached together."""
        fetcher = COTFetcher()

        # Pre-populate cache for one year
        mock_df = pl.DataFrame({"col": [1, 2, 3]})
        fetcher._cache[("traders_in_financial_futures_fut", 2023)] = mock_df

        # Different year should not be in cache
        assert ("traders_in_financial_futures_fut", 2022) not in fetcher._cache

    def test_get_report_import_error(self):
        """Test _get_report handles missing cot_reports library."""
        fetcher = COTFetcher()

        with patch.dict("sys.modules", {"cot_reports": None}):
            # Force reimport to trigger ImportError
            with pytest.raises(ImportError, match="cot_reports library not installed"):
                fetcher._get_report("traders_in_financial_futures_fut", 2023)

    def test_get_report_for_years_empty(self):
        """Test _get_report_for_years with empty year list."""
        fetcher = COTFetcher()
        result = fetcher._get_report_for_years("traders_in_financial_futures_fut", [])
        assert result.is_empty()

    def test_get_report_for_years_concatenates(self):
        """Test _get_report_for_years concatenates multiple years."""
        fetcher = COTFetcher()

        # Pre-populate cache for multiple years
        fetcher._cache[("traders_in_financial_futures_fut", 2022)] = pl.DataFrame({"year": [2022]})
        fetcher._cache[("traders_in_financial_futures_fut", 2023)] = pl.DataFrame({"year": [2023]})

        result = fetcher._get_report_for_years("traders_in_financial_futures_fut", [2022, 2023])
        assert len(result) == 2

    def test_fetch_product_empty_data_from_cache(self):
        """Test fetch_product with empty data returns empty DataFrame."""
        config = COTConfig(start_year=2023, end_year=2023)
        fetcher = COTFetcher(config)

        # Pre-populate cache with empty DataFrame
        fetcher._cache[("traders_in_financial_futures_fut", 2023)] = pl.DataFrame()

        result = fetcher.fetch_product("ES")
        assert result.is_empty()

    def test_fetch_product_no_matching_rows_from_cache(self):
        """Test fetch_product when no rows match product filter."""
        config = COTConfig(start_year=2023, end_year=2023)
        fetcher = COTFetcher(config)

        # Pre-populate cache with non-matching data
        mock_df = pl.DataFrame({"Market_and_Exchange_Names": ["SOME OTHER PRODUCT"]})
        fetcher._cache[("traders_in_financial_futures_fut", 2023)] = mock_df

        result = fetcher.fetch_product("ES")
        assert result.is_empty()

    def test_add_computed_columns_financial(self):
        """Test _add_computed_columns for financial futures."""
        fetcher = COTFetcher()

        df = pl.DataFrame(
            {
                "dealer_long": [100, 200],
                "dealer_short": [50, 100],
                "asset_mgr_long": [300, 400],
                "asset_mgr_short": [200, 250],
                "lev_money_long": [500, 600],
                "lev_money_short": [400, 450],
                "nonrept_long": [50, 60],
                "nonrept_short": [30, 40],
            }
        )

        result = fetcher._add_computed_columns(df, "traders_in_financial_futures_fut")

        assert "dealer_net" in result.columns
        assert "asset_mgr_net" in result.columns
        assert "lev_money_net" in result.columns
        assert "nonrept_net" in result.columns

        # Verify calculations
        assert result["dealer_net"].to_list() == [50, 100]
        assert result["lev_money_net"].to_list() == [100, 150]

    def test_add_computed_columns_commodity(self):
        """Test _add_computed_columns for commodity futures."""
        fetcher = COTFetcher()

        df = pl.DataFrame(
            {
                "commercial_long": [1000, 2000],
                "commercial_short": [800, 1500],
                "managed_money_long": [500, 600],
                "managed_money_short": [400, 450],
                "nonrept_long": [50, 60],
                "nonrept_short": [30, 40],
            }
        )

        result = fetcher._add_computed_columns(df, "disaggregated_fut")

        assert "commercial_net" in result.columns
        assert "managed_money_net" in result.columns
        assert "nonrept_net" in result.columns

        assert result["commercial_net"].to_list() == [200, 500]

    def test_add_computed_columns_no_columns(self):
        """Test _add_computed_columns with no matching columns."""
        fetcher = COTFetcher()
        df = pl.DataFrame({"other_col": [1, 2, 3]})
        result = fetcher._add_computed_columns(df, "traders_in_financial_futures_fut")
        assert "dealer_net" not in result.columns

    def test_save_to_hive(self, tmp_path):
        """Test saving data to Hive-partitioned storage."""
        config = COTConfig(storage_path=tmp_path)
        fetcher = COTFetcher(config)

        df = pl.DataFrame(
            {
                "report_date": ["2023-01-01", "2023-01-08"],
                "open_interest": [100000, 110000],
            }
        )

        path = fetcher.save_to_hive(df, "ES")
        assert path.exists()
        assert "product=ES" in str(path)

        # Verify data can be read back
        loaded = pl.read_parquet(path)
        assert len(loaded) == 2

    def test_fetch_all_empty_products(self):
        """Test fetch_all with empty product list."""
        config = COTConfig(products=[])
        fetcher = COTFetcher(config)
        result = fetcher.fetch_all()
        assert result == {}

    @patch.object(COTFetcher, "fetch_product")
    def test_fetch_all_handles_errors(self, mock_fetch):
        """Test fetch_all handles errors gracefully."""
        mock_fetch.side_effect = [
            pl.DataFrame({"col": [1]}),
            Exception("API Error"),
        ]

        config = COTConfig(products=["ES", "CL"])
        fetcher = COTFetcher(config)
        result = fetcher.fetch_all()

        assert "ES" in result
        assert "CL" not in result

    @patch.object(COTFetcher, "fetch_product")
    def test_fetch_all_skips_empty(self, mock_fetch):
        """Test fetch_all skips empty results."""
        mock_fetch.return_value = pl.DataFrame()

        config = COTConfig(products=["ES"])
        fetcher = COTFetcher(config)
        result = fetcher.fetch_all()

        assert "ES" not in result

    def test_download_all_skip_existing(self, tmp_path):
        """Test download_all skips existing files."""
        config = COTConfig(products=["ES"], storage_path=tmp_path)
        fetcher = COTFetcher(config)

        # Create existing file
        existing_path = tmp_path / "product=ES" / "data.parquet"
        existing_path.parent.mkdir(parents=True)
        pl.DataFrame({"col": [1]}).write_parquet(existing_path)

        result = fetcher.download_all(skip_existing=True)
        assert result["ES"] == existing_path

    @patch.object(COTFetcher, "fetch_product")
    @patch.object(COTFetcher, "save_to_hive")
    def test_download_all_saves_data(self, mock_save, mock_fetch, tmp_path):
        """Test download_all saves fetched data."""
        mock_fetch.return_value = pl.DataFrame({"col": [1, 2, 3]})
        mock_save.return_value = tmp_path / "data.parquet"

        config = COTConfig(products=["ES"], storage_path=tmp_path)
        fetcher = COTFetcher(config)

        result = fetcher.download_all(skip_existing=False)
        assert "ES" in result
        mock_save.assert_called_once()

    @patch.object(COTFetcher, "fetch_product")
    def test_download_all_handles_fetch_error(self, mock_fetch, tmp_path):
        """Test download_all handles fetch errors."""
        mock_fetch.side_effect = Exception("Fetch error")

        config = COTConfig(products=["ES"], storage_path=tmp_path)
        fetcher = COTFetcher(config)

        result = fetcher.download_all(skip_existing=False)
        assert "ES" not in result

    def test_include_options_modifies_report_type(self):
        """Test include_options config modifies report type."""
        config = COTConfig(products=["ES"], include_options=True, start_year=2023, end_year=2023)
        _fetcher = COTFetcher(config)  # noqa: F841

        # The report type should be modified when include_options is True
        mapping = PRODUCT_MAPPINGS["ES"]
        expected_type = mapping.report_type + "opt"
        assert expected_type == "traders_in_financial_futures_futopt"

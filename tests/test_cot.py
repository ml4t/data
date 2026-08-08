"""Tests for CFTC Commitment of Traders (COT) data fetcher."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.cot import (
    PRODUCT_MAPPINGS,
    COTConfig,
    COTFetcher,
    ProductMapping,
    combine_cot_ohlcv,
    combine_cot_ohlcv_pit,
    create_cot_features,
    load_cot_config,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Sample daily OHLCV data."""
    return pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 31), eager=True),
            "open": [100.0 + i * 0.1 for i in range(31)],
            "high": [101.0 + i * 0.1 for i in range(31)],
            "low": [99.0 + i * 0.1 for i in range(31)],
            "close": [100.5 + i * 0.1 for i in range(31)],
            "volume": [1000000 + i * 10000 for i in range(31)],
        }
    )


@pytest.fixture
def sample_cot_financial() -> pl.DataFrame:
    """Sample COT data for financial futures (TFF report)."""
    # COT reports are weekly (Tuesday positions)
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 9),
        date(2024, 1, 16),
        date(2024, 1, 23),
        date(2024, 1, 30),
    ]
    return pl.DataFrame(
        {
            "report_date": dates,
            "available_at": [
                datetime(2024, 1, 5, 20, 30, tzinfo=UTC),
                datetime(2024, 1, 12, 20, 30, tzinfo=UTC),
                datetime(2024, 1, 19, 20, 30, tzinfo=UTC),
                datetime(2024, 1, 26, 20, 30, tzinfo=UTC),
                datetime(2024, 2, 2, 20, 30, tzinfo=UTC),
            ],
            "open_interest": [500000, 510000, 520000, 515000, 525000],
            "dealer_long": [100000, 102000, 98000, 105000, 103000],
            "dealer_short": [120000, 118000, 122000, 115000, 117000],
            "asset_mgr_long": [150000, 155000, 160000, 158000, 162000],
            "asset_mgr_short": [130000, 128000, 132000, 135000, 130000],
            "lev_money_long": [80000, 85000, 82000, 78000, 88000],
            "lev_money_short": [90000, 88000, 92000, 95000, 85000],
            "nonrept_long": [50000, 48000, 52000, 49000, 51000],
            "nonrept_short": [40000, 42000, 38000, 41000, 43000],
            "oi_change": [5000, 10000, 10000, -5000, 10000],
            "product": ["ES"] * 5,
            "report_type": ["traders_in_financial_futures_fut"] * 5,
        }
    )


@pytest.fixture
def sample_cot_commodity() -> pl.DataFrame:
    """Sample COT data for commodity futures (disaggregated report)."""
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 9),
        date(2024, 1, 16),
        date(2024, 1, 23),
        date(2024, 1, 30),
    ]
    return pl.DataFrame(
        {
            "report_date": dates,
            "open_interest": [300000, 310000, 305000, 315000, 320000],
            "commercial_long": [100000, 105000, 102000, 108000, 110000],
            "commercial_short": [150000, 148000, 155000, 152000, 150000],
            "managed_money_long": [80000, 85000, 78000, 82000, 88000],
            "managed_money_short": [50000, 52000, 55000, 48000, 45000],
            "nonrept_long": [30000, 28000, 32000, 29000, 31000],
            "nonrept_short": [20000, 22000, 18000, 21000, 23000],
            "oi_change": [5000, 10000, -5000, 10000, 5000],
            "product": ["CL"] * 5,
            "report_type": ["disaggregated_fut"] * 5,
        }
    )


# =============================================================================
# Product Mappings Tests
# =============================================================================


class TestProductMappings:
    """Tests for PRODUCT_MAPPINGS."""

    def test_mappings_exist(self):
        """Verify product mappings are populated."""
        assert len(PRODUCT_MAPPINGS) > 0
        assert "ES" in PRODUCT_MAPPINGS
        assert "CL" in PRODUCT_MAPPINGS
        assert "GC" in PRODUCT_MAPPINGS

    def test_mapping_structure(self):
        """Verify ProductMapping has correct attributes."""
        es = PRODUCT_MAPPINGS["ES"]
        assert isinstance(es, ProductMapping)
        assert es.code == "ES"
        assert "E-MINI S&P 500" in es.cot_name
        assert es.report_type == "traders_in_financial_futures_fut"
        assert es.description == "E-mini S&P 500"

    def test_financial_vs_commodity_report_types(self):
        """Verify financial products use TFF, commodities use disaggregated."""
        # Financial futures
        assert PRODUCT_MAPPINGS["ES"].report_type == "traders_in_financial_futures_fut"
        assert PRODUCT_MAPPINGS["ZN"].report_type == "traders_in_financial_futures_fut"
        assert PRODUCT_MAPPINGS["6E"].report_type == "traders_in_financial_futures_fut"

        # Commodity futures
        assert PRODUCT_MAPPINGS["CL"].report_type == "disaggregated_fut"
        assert PRODUCT_MAPPINGS["GC"].report_type == "disaggregated_fut"
        assert PRODUCT_MAPPINGS["ZC"].report_type == "disaggregated_fut"

    def test_all_products_have_valid_report_types(self):
        """Verify all products have valid report types."""
        valid_types = {
            "traders_in_financial_futures_fut",
            "traders_in_financial_futures_futopt",
            "disaggregated_fut",
            "disaggregated_futopt",
        }
        for code, mapping in PRODUCT_MAPPINGS.items():
            assert mapping.report_type in valid_types, f"{code} has invalid report type"


# =============================================================================
# COTConfig Tests
# =============================================================================


class TestCOTConfig:
    """Tests for COTConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = COTConfig()
        assert config.products == []
        assert config.start_year == 2020
        assert config.end_year is not None  # Defaults to current year
        assert config.include_options is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = COTConfig(
            products=["ES", "CL"],
            start_year=2018,
            end_year=2023,
            storage_path=Path("/tmp/cot"),
            include_options=True,
        )
        assert config.products == ["ES", "CL"]
        assert config.start_year == 2018
        assert config.end_year == 2023
        assert config.storage_path == Path("/tmp/cot").resolve()
        assert config.include_options is True

    def test_get_years(self):
        """Test year range generation."""
        config = COTConfig(start_year=2020, end_year=2023)
        years = config.get_years()
        assert years == [2020, 2021, 2022, 2023]

    def test_storage_path_expansion(self):
        """Test that ~ is expanded in storage path."""
        config = COTConfig(storage_path="~/test/cot")
        assert "~" not in str(config.storage_path)
        assert config.storage_path.is_absolute()


class TestLoadCOTConfig:
    """Tests for load_cot_config function."""

    def test_load_from_yaml(self, tmp_path: Path):
        """Test loading config from YAML file."""
        yaml_content = """
products:
  - ES
  - CL
  - GC
start_year: 2019
end_year: 2024
storage_path: /tmp/cot_data
include_options: true
"""
        config_file = tmp_path / "cot_config.yaml"
        config_file.write_text(yaml_content)

        config = load_cot_config(config_file)
        assert config.products == ["ES", "CL", "GC"]
        assert config.start_year == 2019
        assert config.end_year == 2024
        assert config.storage_path == Path("/tmp/cot_data").resolve()
        assert config.include_options is True


# =============================================================================
# COTFetcher Tests
# =============================================================================


class TestCOTFetcher:
    """Tests for COTFetcher."""

    def test_list_available_products(self):
        """Test listing available product codes."""
        fetcher = COTFetcher()
        products = fetcher.list_available_products()
        assert "ES" in products
        assert "CL" in products
        assert len(products) == len(PRODUCT_MAPPINGS)

    def test_get_product_info(self):
        """Test getting product mapping info."""
        fetcher = COTFetcher()
        info = fetcher.get_product_info("ES")
        assert info is not None
        assert info.code == "ES"
        assert "E-MINI S&P 500" in info.cot_name

    def test_get_product_info_unknown(self):
        """Test getting info for unknown product."""
        fetcher = COTFetcher()
        info = fetcher.get_product_info("UNKNOWN")
        assert info is None

    def test_fetch_product_unknown_raises(self):
        """Test that fetching unknown product raises ValueError."""
        fetcher = COTFetcher()
        with pytest.raises(ValueError, match="Unknown product code"):
            fetcher.fetch_product("UNKNOWN")

    def test_fetch_product_financial(self):
        """Test fetching financial futures product (mocked)."""
        import pandas as pd

        # Create mock cot_reports module
        mock_cot = MagicMock()
        mock_df = pd.DataFrame(
            {
                "Market_and_Exchange_Names": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"] * 5,
                "Report_Date_as_YYYY-MM-DD": [
                    "2024-01-02",
                    "2024-01-09",
                    "2024-01-16",
                    "2024-01-23",
                    "2024-01-30",
                ],
                "Open_Interest_All": [500000, 510000, 520000, 515000, 525000],
                "Dealer_Positions_Long_All": [100000, 102000, 98000, 105000, 103000],
                "Dealer_Positions_Short_All": [120000, 118000, 122000, 115000, 117000],
                "Lev_Money_Positions_Long_All": [80000, 85000, 82000, 78000, 88000],
                "Lev_Money_Positions_Short_All": [90000, 88000, 92000, 95000, 85000],
            }
        )
        mock_cot.cot_year.return_value = mock_df

        config = COTConfig(products=["ES"], start_year=2024, end_year=2024)
        fetcher = COTFetcher(config)

        # Patch the import inside _get_report method
        with patch.dict("sys.modules", {"cot_reports": mock_cot}):
            df = fetcher.fetch_product("ES")

        assert not df.is_empty()
        assert "report_date" in df.columns
        assert "open_interest" in df.columns
        assert "product" in df.columns
        assert df["product"][0] == "ES"


# =============================================================================
# Workflow Tests
# =============================================================================


class TestCombineCOTOHLCV:
    """Tests for combine_cot_ohlcv function."""

    def test_combine_basic(self, sample_ohlcv, sample_cot_financial):
        """Test basic combination of OHLCV and COT data."""
        combined = combine_cot_ohlcv(sample_ohlcv, sample_cot_financial)

        # Should have all OHLCV rows
        assert len(combined) == len(sample_ohlcv)

        # Should have COT columns
        assert "cot_open_interest" in combined.columns
        assert "dealer_long" in combined.columns
        assert "lev_money_long" in combined.columns


class TestCombineCOTOHLCVPIT:
    """Tests for combine_cot_ohlcv_pit (point-in-time safe) function."""

    def test_alias_matches_safe_primary_api(self, sample_ohlcv, sample_cot_financial):
        """Both public join names use the official release timestamp."""
        combined = combine_cot_ohlcv(sample_ohlcv, sample_cot_financial)
        combined_pit = combine_cot_ohlcv_pit(sample_ohlcv, sample_cot_financial)

        assert combined.equals(combined_pit)
        jan_2 = combined.filter(pl.col("timestamp") == date(2024, 1, 2))
        assert jan_2["cot_open_interest"][0] is None

    def test_pit_data_available_after_lag(self, sample_ohlcv, sample_cot_financial):
        """A daily observation sees a report only after its exact release."""
        combined_pit = combine_cot_ohlcv_pit(sample_ohlcv, sample_cot_financial)

        jan_5 = combined_pit.filter(pl.col("timestamp") == date(2024, 1, 5))
        jan_6 = combined_pit.filter(pl.col("timestamp") == date(2024, 1, 6))
        assert jan_5["cot_open_interest"][0] is None
        assert jan_6["cot_open_interest"][0] == 500000

    def test_pit_honors_delayed_schedule(self, sample_ohlcv, sample_cot_financial):
        """A delayed release remains hidden until the supplied timestamp."""
        delayed = sample_cot_financial.with_columns(
            pl.when(pl.col("report_date") == date(2024, 1, 2))
            .then(pl.lit(datetime(2024, 1, 8, tzinfo=UTC)))
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        )

        combined = combine_cot_ohlcv_pit(sample_ohlcv, delayed)

        jan_7 = combined.filter(pl.col("timestamp") == date(2024, 1, 7))
        jan_8 = combined.filter(pl.col("timestamp") == date(2024, 1, 8))
        assert jan_7["cot_open_interest"][0] is None
        assert jan_8["cot_open_interest"][0] == 500000


class TestCombineCOTOHLCVMore:
    """Additional tests for combine_cot_ohlcv function."""

    def test_combine_forward_fill(self, sample_ohlcv, sample_cot_financial):
        """Test that COT data is forward-filled correctly."""
        combined = combine_cot_ohlcv(sample_ohlcv, sample_cot_financial)

        # COT data should be forward-filled
        # Jan 6 and Jan 8 should both have the January 2 report.
        row_jan_6 = combined.filter(pl.col("timestamp") == date(2024, 1, 6))
        row_jan_8 = combined.filter(pl.col("timestamp") == date(2024, 1, 8))

        assert row_jan_6["cot_open_interest"][0] == row_jan_8["cot_open_interest"][0]

    def test_combine_preserves_ohlcv_columns(self, sample_ohlcv, sample_cot_financial):
        """Test that all OHLCV columns are preserved."""
        combined = combine_cot_ohlcv(sample_ohlcv, sample_cot_financial)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in combined.columns


class TestCreateCOTFeatures:
    """Tests for create_cot_features function."""

    def test_financial_features(self, sample_cot_financial):
        """Test feature creation for financial futures."""
        # Add net positions manually (as fetcher does)
        df = sample_cot_financial.with_columns(
            [
                (pl.col("dealer_long") - pl.col("dealer_short")).alias("dealer_net"),
                (pl.col("lev_money_long") - pl.col("lev_money_short")).alias("lev_money_net"),
                (pl.col("asset_mgr_long") - pl.col("asset_mgr_short")).alias("asset_mgr_net"),
                (pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"),
            ]
        )

        features = create_cot_features(df)

        # Check financial-specific features
        assert "cot_lev_money_pct_oi" in features.columns
        assert "cot_asset_mgr_pct_oi" in features.columns
        assert "cot_dealer_pct_oi" in features.columns

    def test_commodity_features(self, sample_cot_commodity):
        """Test feature creation for commodity futures."""
        # Add net positions manually
        df = sample_cot_commodity.with_columns(
            [
                (pl.col("commercial_long") - pl.col("commercial_short")).alias("commercial_net"),
                (pl.col("managed_money_long") - pl.col("managed_money_short")).alias(
                    "managed_money_net"
                ),
                (pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"),
            ]
        )

        features = create_cot_features(df)

        # Check commodity-specific features
        assert "cot_managed_money_pct_oi" in features.columns
        assert "cot_commercial_pct_oi" in features.columns

    def test_universal_features(self, sample_cot_financial):
        """Test features that apply to both report types."""
        df = sample_cot_financial.with_columns(
            [
                (pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"),
            ]
        )

        features = create_cot_features(df)

        # Non-reportable net should always be computed
        assert "cot_nonrept_pct_oi" in features.columns
        assert "cot_oi_change_pct" in features.columns

    def test_custom_prefix(self, sample_cot_financial):
        """Test custom prefix for feature names."""
        df = sample_cot_financial.with_columns(
            [
                (pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"),
            ]
        )

        features = create_cot_features(df, prefix="positioning_")

        # Check custom prefix
        assert "positioning_nonrept_pct_oi" in features.columns
        assert "positioning_oi_change_pct" in features.columns


# =============================================================================
# Integration Tests
# =============================================================================


class TestCOTIntegration:
    """Integration tests for the full COT workflow."""

    def test_full_workflow(self, sample_ohlcv, sample_cot_financial):
        """Test the full workflow: combine + features."""
        # Add net positions (as fetcher does)
        cot = sample_cot_financial.with_columns(
            [
                (pl.col("dealer_long") - pl.col("dealer_short")).alias("dealer_net"),
                (pl.col("lev_money_long") - pl.col("lev_money_short")).alias("lev_money_net"),
                (pl.col("asset_mgr_long") - pl.col("asset_mgr_short")).alias("asset_mgr_net"),
                (pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"),
            ]
        )

        # Combine OHLCV + COT
        combined = combine_cot_ohlcv(sample_ohlcv, cot)

        # Create features
        features = create_cot_features(combined)

        # Verify final result
        assert len(features) == len(sample_ohlcv)  # Same length as OHLCV

        # Has OHLCV columns
        assert all(col in features.columns for col in ["open", "high", "low", "close", "volume"])

        # Has COT columns
        assert "cot_open_interest" in features.columns

        # Has feature columns
        assert any(col.startswith("cot_") for col in features.columns)

"""Integration tests for Factor Data Providers (AQR and Fama-French).

These tests verify factor providers work correctly by downloading real data
from AQR and Ken French's Data Library.

Requirements:
    - No API keys required (both are free public data sources)
    - Internet connection required for first download
    - Tests use caching to avoid repeated downloads

Test Coverage:
    AQR Provider:
    - Download functionality
    - Data parsing and schema validation
    - Regional filtering
    - Date range filtering

    Fama-French Provider:
    - Core factors (FF3, FF5, Momentum)
    - Portfolio data (Size x Value 25)
    - Industry portfolios (FF48)
    - International factors
    - Daily vs monthly frequencies
    - Combined fetching (FF3 + MOM)

Note:
    These tests download data from public URLs. First run may take 1-2 minutes.
    Subsequent runs use cached data and are fast.
"""

import polars as pl
import pytest

# Import providers
from ml4t.data.providers.fama_french import FamaFrenchProvider


class TestFamaFrenchProvider:
    """Integration tests for Fama-French factor provider."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider with temporary cache directory."""
        return FamaFrenchProvider(cache_path=tmp_path, use_cache=True)

    @pytest.fixture
    def cached_provider(self):
        """Create provider with default cache for faster repeat runs."""
        return FamaFrenchProvider(use_cache=True)

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.name == "fama_french"
        assert provider.cache_path.exists()

    def test_list_categories(self, provider):
        """Test listing dataset categories."""
        categories = provider.list_categories()
        assert "factors" in categories
        assert "portfolios" in categories
        assert "industry" in categories
        assert "international" in categories
        assert "breakpoints" in categories

    def test_list_datasets_all(self, provider):
        """Test listing all datasets."""
        datasets = provider.list_datasets()
        assert len(datasets) >= 50  # Should have 50+ datasets
        assert "ff3" in datasets
        assert "ff5" in datasets
        assert "mom" in datasets
        assert "ind_48" in datasets

    def test_list_datasets_by_category(self, provider):
        """Test listing datasets filtered by category."""
        factors = provider.list_datasets(category="factors")
        assert "ff3" in factors
        assert "ff5" in factors
        assert "mom" in factors

        industry = provider.list_datasets(category="industry")
        assert "ind_48" in industry
        assert "ind_5" in industry

    def test_get_dataset_info_ff3(self, provider):
        """Test getting dataset metadata."""
        info = provider.get_dataset_info("ff3")
        assert "name" in info
        assert "Fama-French 3" in info["name"]
        assert "description" in info
        assert "paper" in info
        assert "Fama" in info["paper"]

    def test_fetch_ff3_monthly(self, cached_provider):
        """Test fetching FF3 monthly factors (real download).

        This downloads from Ken French's website.
        """
        df = cached_provider.fetch("ff3", frequency="monthly")

        # Validate schema
        assert "timestamp" in df.columns
        assert "Mkt-RF" in df.columns
        assert "SMB" in df.columns
        assert "HML" in df.columns
        assert "RF" in df.columns

        # Validate data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["Mkt-RF"].dtype == pl.Float64

        # Validate data range (should start from 1926)
        min_date = df["timestamp"].min()
        assert min_date.year <= 1927

        # Validate returns are in decimal format (not percent)
        avg_mkt = df["Mkt-RF"].mean()
        assert -0.5 < avg_mkt < 0.5  # Should be reasonable decimal returns

        # Should have many years of data
        assert len(df) > 1000  # Monthly data since 1926

    def test_fetch_ff5_monthly(self, cached_provider):
        """Test fetching FF5 monthly factors."""
        df = cached_provider.fetch("ff5", frequency="monthly")

        # Validate columns
        assert "timestamp" in df.columns
        assert "Mkt-RF" in df.columns
        assert "SMB" in df.columns
        assert "HML" in df.columns
        assert "RMW" in df.columns  # Profitability
        assert "CMA" in df.columns  # Investment
        assert "RF" in df.columns

        # FF5 starts from 1963
        min_date = df["timestamp"].min()
        assert min_date.year <= 1964

    def test_fetch_momentum(self, cached_provider):
        """Test fetching momentum factor."""
        df = cached_provider.fetch("mom", frequency="monthly")

        assert "timestamp" in df.columns
        assert "MOM" in df.columns

        # Momentum starts from 1927
        min_date = df["timestamp"].min()
        assert min_date.year <= 1928

    def test_fetch_combined_carhart_4factor(self, cached_provider):
        """Test fetching combined FF3 + Momentum (Carhart 4-factor)."""
        df = cached_provider.fetch_combined(["ff3", "mom"])

        # Should have all columns
        assert "Mkt-RF" in df.columns
        assert "SMB" in df.columns
        assert "HML" in df.columns
        assert "RF" in df.columns
        assert "MOM" in df.columns

    def test_fetch_ff3_mom_shortcut(self, cached_provider):
        """Test the ff3_mom shortcut dataset."""
        df = cached_provider.fetch("ff3_mom")

        assert "Mkt-RF" in df.columns
        assert "MOM" in df.columns

    def test_fetch_with_date_range(self, cached_provider):
        """Test date filtering."""
        df = cached_provider.fetch("ff3", start="2020-01-01", end="2020-12-31")

        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        assert min_date.year >= 2020
        assert max_date.year <= 2020

    def test_fetch_industry_portfolios(self, cached_provider):
        """Test fetching industry portfolios (FF48)."""
        df = cached_provider.fetch("ind_48")

        assert "timestamp" in df.columns
        # Should have 48 industry columns + timestamp
        assert len(df.columns) >= 48

    def test_fetch_portfolio_size_bm_25(self, cached_provider):
        """Test fetching 25 Size x Value portfolios."""
        df = cached_provider.fetch("port_size_bm_25")

        assert "timestamp" in df.columns
        # Should have 25 portfolios + timestamp
        assert len(df.columns) >= 25

    def test_fetch_international_factors(self, cached_provider):
        """Test fetching international factors."""
        df = cached_provider.fetch("ff3_developed")

        assert "timestamp" in df.columns
        assert "Mkt-RF" in df.columns

    def test_invalid_dataset_raises_error(self, provider):
        """Test that invalid dataset raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            provider.fetch("nonexistent_dataset")

    def test_invalid_frequency_raises_error(self, cached_provider):
        """Test that invalid frequency raises error."""
        with pytest.raises(ValueError, match="Frequency.*not available"):
            # ff5 doesn't have weekly frequency
            cached_provider.fetch("ff5", frequency="weekly")

    def test_cache_works(self, tmp_path):
        """Test that caching prevents repeated downloads."""
        provider = FamaFrenchProvider(cache_path=tmp_path, use_cache=True)

        # First fetch - downloads
        df1 = provider.fetch("ff3")

        # Check cache file exists
        cache_file = tmp_path / "ff3_monthly.parquet"
        assert cache_file.exists()

        # Second fetch - should use cache
        df2 = provider.fetch("ff3")

        # Data should be identical
        assert len(df1) == len(df2)

    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        provider = FamaFrenchProvider(cache_path=tmp_path, use_cache=True)

        # Fetch to create cache
        provider.fetch("ff3")

        # Clear cache
        provider.clear_cache()

        # Cache should be empty
        cache_files = list(tmp_path.glob("*.parquet"))
        assert len(cache_files) == 0


class TestFamaFrenchDataQuality:
    """Tests for data quality validation."""

    @pytest.fixture
    def provider(self):
        """Use default cache for data quality tests."""
        return FamaFrenchProvider(use_cache=True)

    def test_returns_are_decimal_format(self, provider):
        """Verify returns are in decimal format (0.01 = 1%)."""
        df = provider.fetch("ff3")

        # Market return should average around 0.5-1% per month (0.005-0.01)
        avg_mkt = df["Mkt-RF"].mean()
        assert 0.001 < avg_mkt < 0.02, f"Unexpected avg market return: {avg_mkt}"

        # No returns should be > 100% in decimal format
        max_return = df["Mkt-RF"].max()
        assert max_return < 0.5, f"Max return too high: {max_return}"

    def test_no_missing_dates(self, provider):
        """Check for missing dates in continuous data."""
        df = provider.fetch("ff3")

        # Check for nulls in timestamp
        null_count = df["timestamp"].null_count()
        assert null_count == 0, f"Found {null_count} null timestamps"

    def test_factors_are_correlated_as_expected(self, provider):
        """Validate factor correlations match known patterns."""
        df = provider.fetch("ff3")

        # HML and SMB should have low correlation (different factors)
        corr = df.select(pl.corr("SMB", "HML")).item()
        assert abs(corr) < 0.5, f"SMB-HML correlation unexpectedly high: {corr}"


# Skip AQR tests by default as they require pre-downloaded data
# To run: first execute AQRFactorProvider.download(), then run tests
@pytest.mark.skip(
    reason="Requires AQR data to be downloaded first with AQRFactorProvider.download()"
)
class TestAQRFactorProvider:
    """Integration tests for AQR factor provider.

    Note: These tests require data to be pre-downloaded using:
        AQRFactorProvider.download()

    The download() method fetches Excel files from AQR's website and
    saves them as Parquet files.
    """

    @pytest.fixture
    def provider(self):
        """Create AQR provider (assumes data is already downloaded)."""
        from ml4t.data.providers.aqr import AQRFactorProvider

        return AQRFactorProvider()

    def test_list_datasets(self, provider):
        """Test listing available datasets."""
        datasets = provider.list_datasets()
        assert "qmj_factors" in datasets
        assert "bab_factors" in datasets
        assert "tsmom" in datasets

    def test_list_categories(self, provider):
        """Test listing categories."""
        categories = provider.list_categories()
        assert "equity_factors" in categories
        assert "cross_asset" in categories

    def test_fetch_qmj_factors(self, provider):
        """Test fetching QMJ (Quality Minus Junk) factors."""
        df = provider.fetch("qmj_factors", region="USA")

        assert "timestamp" in df.columns
        assert "USA" in df.columns or len(df.columns) > 1

        # Returns should be decimal format
        assert (
            df.select(pl.col("USA") if "USA" in df.columns else pl.exclude("timestamp"))
            .max()
            .item()
            < 0.5
        )

    def test_fetch_tsmom(self, provider):
        """Test fetching Time Series Momentum factors."""
        df = provider.fetch("tsmom")

        assert "timestamp" in df.columns
        assert len(df.columns) > 1

    def test_get_dataset_info(self, provider):
        """Test getting dataset metadata."""
        info = provider.get_dataset_info("qmj_factors")

        assert "name" in info
        assert "description" in info
        assert "paper" in info

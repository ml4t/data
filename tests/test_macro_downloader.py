"""Tests for Macro downloader module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.macro.downloader import MacroConfig, MacroDataManager


class TestMacroConfig:
    """Tests for MacroConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MacroConfig()

        assert config.provider == "fred"
        assert config.start == "2000-01-01"
        assert config.end == "2025-12-31"
        assert config.series == {}

    def test_storage_path_expanded(self):
        """Test that storage path is expanded."""
        config = MacroConfig(storage_path=Path("~/test-data/macro"))

        assert "~" not in str(config.storage_path)
        assert config.storage_path.is_absolute()

    def test_get_treasury_symbols_default(self):
        """Test get_treasury_symbols returns defaults."""
        config = MacroConfig()

        symbols = config.get_treasury_symbols()
        assert symbols == ["DGS2", "DGS5", "DGS10", "DGS30"]

    def test_get_treasury_symbols_from_config(self):
        """Test get_treasury_symbols with configured values."""
        config = MacroConfig(
            series={
                "treasury_yields": {"symbols": ["DGS5", "DGS10"]},
            }
        )

        symbols = config.get_treasury_symbols()
        assert symbols == ["DGS5", "DGS10"]

    def test_get_derived_series_empty(self):
        """Test get_derived_series when empty."""
        config = MacroConfig()
        assert config.get_derived_series() == []

    def test_get_derived_series_with_data(self):
        """Test get_derived_series with configured formulas."""
        config = MacroConfig(
            series={
                "derived": [
                    {"name": "SLOPE", "formula": "DGS10 - DGS2"},
                ]
            }
        )

        derived = config.get_derived_series()
        assert len(derived) == 1
        assert derived[0]["name"] == "SLOPE"


class TestMacroDataManager:
    """Tests for MacroDataManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        return MacroConfig(
            provider="fred",
            start="2024-01-01",
            end="2024-12-31",
            storage_path=temp_storage,
            series={
                "treasury_yields": {"symbols": ["DGS2", "DGS10"]},
                "derived": [
                    {"name": "SLOPE", "formula": "DGS10 - DGS2"},
                ],
            },
        )

    @pytest.fixture
    def manager(self, config):
        """Create MacroDataManager instance."""
        return MacroDataManager(config)

    def test_init(self, manager, temp_storage):
        """Test initialization."""
        assert manager.config.storage_path == temp_storage
        assert manager._provider is None
        assert temp_storage.exists()

    def test_from_config_yaml(self, temp_storage):
        """Test creating manager from YAML config."""
        yaml_content = f"""
macro:
  provider: fred
  start: "2020-01-01"
  end: "2024-12-31"
  storage_path: {temp_storage}
  series:
    treasury_yields:
      symbols: ["DGS10"]
"""
        config_file = temp_storage / "test_config.yaml"
        config_file.write_text(yaml_content)

        manager = MacroDataManager.from_config(config_file)

        assert manager.config.provider == "fred"
        assert manager.config.start == "2020-01-01"

    def test_get_provider_without_api_key(self, manager):
        """Test _get_provider returns None without API key."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove FRED_API_KEY if it exists
            import os

            os.environ.pop("FRED_API_KEY", None)

            provider = manager._get_provider()
            assert provider is None

    def test_get_provider_with_api_key(self, manager):
        """Test _get_provider creates FREDProvider with key."""
        with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            with patch("ml4t.data.providers.fred.FREDProvider") as mock_fred:
                mock_fred.return_value = MagicMock()
                _provider = manager._get_provider()  # noqa: F841

                mock_fred.assert_called_once_with(api_key="test_key")

    def test_download_treasury_yields_empty_result(self, manager):
        """Test download when no data returned."""
        with patch.object(manager, "_get_provider", return_value=None):
            with patch.object(manager, "_download_from_yfinance", return_value=pl.DataFrame()):
                df = manager.download_treasury_yields()

                assert df.is_empty()

    def test_download_from_fred(self, manager):
        """Test _download_from_fred method."""
        mock_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "DGS10_close": [4.5],
                "DGS2_close": [4.0],
            }
        )

        mock_provider = MagicMock()
        mock_provider.fetch_multiple.return_value = mock_data

        df = manager._download_from_fred(mock_provider, ["DGS10", "DGS2"])

        assert "DGS10" in df.columns
        assert "DGS2" in df.columns

    def test_download_from_fred_error(self, manager):
        """Test _download_from_fred handles errors."""
        mock_provider = MagicMock()
        mock_provider.fetch_multiple.side_effect = Exception("API error")

        df = manager._download_from_fred(mock_provider, ["DGS10"])

        assert df.is_empty()

    def test_download_from_yfinance(self, manager):
        """Test _download_from_yfinance method."""
        import pandas as pd

        # Create mock yfinance response
        mock_df = pd.DataFrame(
            {
                "Close": [4.5, 4.6],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date"),
        )
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", "")])

        with patch("yfinance.download", return_value=mock_df):
            df = manager._download_from_yfinance(["DGS10"])

            assert not df.is_empty()
            assert "DGS10" in df.columns

    def test_compute_derived_series(self, manager):
        """Test _compute_derived_series method."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "DGS10": [4.5, 4.6],
                "DGS2": [4.0, 4.1],
            }
        )

        result = manager._compute_derived_series(df)

        assert "SLOPE" in result.columns
        assert "YIELD_CURVE_SLOPE" in result.columns
        # SLOPE = DGS10 - DGS2 = 4.5 - 4.0 = 0.5
        assert result["SLOPE"][0] == 0.5

    def test_compute_derived_series_missing_columns(self, manager):
        """Test _compute_derived_series handles missing columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "DGS10": [4.5],
                # Missing DGS2
            }
        )

        result = manager._compute_derived_series(df)

        # Should not crash, just skip computation
        assert "SLOPE" not in result.columns

    def test_save_and_load_treasury_yields(self, manager, temp_storage):
        """Test saving and loading Treasury yields."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "DGS10": [4.5, 4.6],
                "DGS2": [4.0, 4.1],
            }
        )

        manager._save_treasury_yields(df)

        loaded = manager.load_treasury_yields()
        assert len(loaded) == 2
        assert "DGS10" in loaded.columns

    def test_load_treasury_yields_no_data(self, manager):
        """Test load_treasury_yields when no data exists."""
        df = manager.load_treasury_yields()
        assert df.is_empty()

    def test_load_treasury_yields_handles_date_column(self, manager, temp_storage):
        """Test load handles date vs timestamp column."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)],  # Using 'date' instead of 'timestamp'
                "DGS10": [4.5],
            }
        )
        df.write_parquet(temp_storage / "treasury_yields.parquet")

        loaded = manager.load_treasury_yields()
        assert "timestamp" in loaded.columns
        assert "date" not in loaded.columns

    def test_get_yield_curve_slope(self, manager, temp_storage):
        """Test get_yield_curve_slope method."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "DGS10": [4.5, 4.6],
                "DGS2": [4.0, 4.1],
                "YIELD_CURVE_SLOPE": [0.5, 0.5],
            }
        )
        df.write_parquet(temp_storage / "treasury_yields.parquet")

        slope_df = manager.get_yield_curve_slope()

        assert len(slope_df) == 2
        assert list(slope_df.columns) == ["timestamp", "YIELD_CURVE_SLOPE"]

    def test_get_yield_curve_slope_no_data(self, manager):
        """Test get_yield_curve_slope when no data."""
        df = manager.get_yield_curve_slope()
        assert df.is_empty()

    def test_get_regime_default_threshold(self, manager, temp_storage):
        """Test get_regime with default threshold."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "YIELD_CURVE_SLOPE": [1.0, 0.3, -0.2],  # Above, below, negative
            }
        )
        df.write_parquet(temp_storage / "treasury_yields.parquet")

        regime_df = manager.get_regime()

        assert len(regime_df) == 3
        assert "regime" in regime_df.columns
        assert regime_df["regime"][0] == "risk_on"
        assert regime_df["regime"][1] == "risk_off"
        assert regime_df["regime"][2] == "risk_off"

    def test_get_regime_custom_threshold(self, manager, temp_storage):
        """Test get_regime with custom threshold."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "YIELD_CURVE_SLOPE": [0.8, 0.3],
            }
        )
        df.write_parquet(temp_storage / "treasury_yields.parquet")

        # With threshold 0.5, both should be different
        regime_df = manager.get_regime(threshold=0.5)
        assert regime_df["regime"][0] == "risk_on"
        assert regime_df["regime"][1] == "risk_off"

        # With threshold 1.0, both should be risk_off
        regime_df = manager.get_regime(threshold=1.0)
        assert regime_df["regime"][0] == "risk_off"
        assert regime_df["regime"][1] == "risk_off"

    def test_get_regime_no_data(self, manager):
        """Test get_regime when no data exists."""
        df = manager.get_regime()
        assert df.is_empty()


class TestMacroDataManagerIntegration:
    """Integration tests for MacroDataManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow(self, temp_storage):
        """Test complete download and analysis workflow."""
        config = MacroConfig(
            storage_path=temp_storage,
            series={
                "treasury_yields": {"symbols": ["DGS10", "DGS2"]},
            },
        )
        manager = MacroDataManager(config)

        # Create mock data directly
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "DGS10": [4.5, 4.6],
                "DGS2": [4.0, 4.1],
                "YIELD_CURVE_SLOPE": [0.5, 0.5],
            }
        )
        manager._save_treasury_yields(df)

        # Test loading
        loaded = manager.load_treasury_yields()
        assert len(loaded) == 2

        # Test regime classification
        regime = manager.get_regime()
        assert len(regime) == 2
        assert all(r == "risk_off" for r in regime["regime"].to_list())  # 0.5 = threshold

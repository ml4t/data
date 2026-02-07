"""Tests for data_profile module."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.storage.data_profile import (
    ColumnProfile,
    DatasetProfile,
    generate_profile,
    get_profile_path,
    load_profile,
    save_profile,
)


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    return pl.DataFrame({
        "timestamp": [datetime(2025, 1, i) for i in range(1, 6)],
        "symbol": ["SPY"] * 5,
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000, 2000, 3000, 4000, 5000],
    })


@pytest.fixture
def sample_df_with_nulls():
    """Create a sample DataFrame with null values."""
    return pl.DataFrame({
        "timestamp": [datetime(2025, 1, i) for i in range(1, 6)],
        "symbol": ["SPY", "SPY", None, "QQQ", "QQQ"],
        "value": [100.0, None, 102.0, None, 104.0],
    })


class TestColumnProfile:
    """Tests for ColumnProfile dataclass."""

    def test_to_dict(self):
        profile = ColumnProfile(
            name="close",
            dtype="Float64",
            total_count=100,
            null_count=5,
            unique_count=95,
            min=50.0,
            max=150.0,
            mean=100.0,
            std=25.0,
        )
        d = profile.to_dict()
        assert d["name"] == "close"
        assert d["dtype"] == "Float64"
        assert d["null_count"] == 5
        assert d["mean"] == 100.0

    def test_from_dict(self):
        data = {
            "name": "volume",
            "dtype": "Int64",
            "total_count": 100,
            "null_count": 0,
            "unique_count": 80,
            "min": 1000,
            "max": 10000,
            "mean": 5000.0,
            "std": 2500.0,
        }
        profile = ColumnProfile.from_dict(data)
        assert profile.name == "volume"
        assert profile.unique_count == 80


class TestDatasetProfile:
    """Tests for DatasetProfile dataclass."""

    def test_to_dict(self):
        col_profile = ColumnProfile(
            name="test",
            dtype="Float64",
            total_count=10,
            null_count=0,
            unique_count=10,
        )
        profile = DatasetProfile(
            total_rows=10,
            total_columns=1,
            columns=[col_profile],
            generated_at="2025-01-01T00:00:00",
            source="TestSource",
            symbols=["SPY", "QQQ"],
        )
        d = profile.to_dict()
        assert d["total_rows"] == 10
        assert d["source"] == "TestSource"
        assert len(d["columns"]) == 1
        assert d["symbols"] == ["SPY", "QQQ"]

    def test_from_dict(self):
        data = {
            "total_rows": 100,
            "total_columns": 5,
            "columns": [
                {
                    "name": "col1",
                    "dtype": "Float64",
                    "total_count": 100,
                    "null_count": 0,
                    "unique_count": 50,
                }
            ],
            "generated_at": "2025-01-01T00:00:00",
            "source": "TestManager",
            "date_range_start": "2025-01-01",
            "date_range_end": "2025-12-31",
            "symbols": ["SPY"],
        }
        profile = DatasetProfile.from_dict(data)
        assert profile.total_rows == 100
        assert len(profile.columns) == 1
        assert profile.symbols == ["SPY"]

    def test_summary(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df, source="Test")
        summary = profile.summary()
        assert "Dataset Profile" in summary
        assert "Rows: 5" in summary
        assert "Columns: 7" in summary
        assert "SPY" in summary or "1" in summary  # symbols count


class TestGenerateProfile:
    """Tests for generate_profile function."""

    def test_basic_profile(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df)
        assert profile.total_rows == 5
        assert profile.total_columns == 7
        assert len(profile.columns) == 7

    def test_profile_with_source(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df, source="ETFDataManager")
        assert profile.source == "ETFDataManager"

    def test_date_range_extraction(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df, timestamp_col="timestamp")
        assert profile.date_range_start is not None
        assert profile.date_range_end is not None
        assert "2025-01-01" in profile.date_range_start
        assert "2025-01-05" in profile.date_range_end

    def test_symbol_extraction(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df, symbol_col="symbol")
        assert "SPY" in profile.symbols

    def test_numeric_statistics(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df)
        close_col = next(c for c in profile.columns if c.name == "close")
        assert close_col.mean is not None
        assert close_col.std is not None
        assert close_col.min == 100.5
        assert close_col.max == 104.5

    def test_null_handling(self, sample_df_with_nulls):
        profile = generate_profile(sample_df_with_nulls, symbol_col=None)
        value_col = next(c for c in profile.columns if c.name == "value")
        assert value_col.null_count == 2
        assert value_col.total_count == 5

    def test_datetime_min_max(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df)
        ts_col = next(c for c in profile.columns if c.name == "timestamp")
        assert ts_col.min is not None
        assert ts_col.max is not None
        assert "2025-01-01" in str(ts_col.min)

    def test_skip_symbol_extraction(self, sample_ohlcv_df):
        profile = generate_profile(sample_ohlcv_df, symbol_col=None)
        assert profile.symbols == []


class TestSaveLoadProfile:
    """Tests for save_profile and load_profile functions."""

    def test_save_and_load(self, sample_ohlcv_df, tmp_path):
        profile = generate_profile(sample_ohlcv_df, source="Test")
        profile_path = tmp_path / "test_profile.json"

        save_profile(profile, profile_path)
        assert profile_path.exists()

        loaded = load_profile(profile_path)
        assert loaded is not None
        assert loaded.total_rows == profile.total_rows
        assert loaded.source == profile.source
        assert len(loaded.columns) == len(profile.columns)

    def test_load_nonexistent(self, tmp_path):
        result = load_profile(tmp_path / "nonexistent.json")
        assert result is None


class TestGetProfilePath:
    """Tests for get_profile_path function."""

    def test_file_path(self, tmp_path):
        data_path = tmp_path / "data.parquet"
        profile_path = get_profile_path(data_path)
        assert profile_path.name == "data_profile.json"
        assert profile_path.parent == tmp_path

    def test_directory_path(self, tmp_path):
        data_dir = tmp_path / "data_dir"
        data_dir.mkdir()
        profile_path = get_profile_path(data_dir)
        assert profile_path.name == "_profile.json"
        assert profile_path.parent == data_dir

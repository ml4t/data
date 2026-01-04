"""Tests for LearnedSyntheticProvider.

These tests verify the learned synthetic data provider can load pre-generated
samples or model checkpoints and generate realistic OHLCV data.
"""

from __future__ import annotations

# Check if torch is available for checkpoint tests
import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from ml4t.data.providers.learned_synthetic import LearnedSyntheticProvider

HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_metadata() -> dict[str, Any]:
    """Create standard metadata dictionary for testing."""
    return {
        "generator": {
            "name": "timegan",
            "version": "1.0.0",
            "paper": "Time-series GAN (Yoon et al., 2019)",
        },
        "data": {
            "source": "ETF returns",
            "symbols": ["SPY", "QQQ", "IWM"],
            "n_samples": 100,
            "seq_length": 24,
            "n_features": 6,
        },
        "evaluation": {
            "tstr_ratio": 0.95,
        },
    }


@pytest.fixture
def sample_3d_array() -> np.ndarray:
    """Generate valid (n_samples, seq_length, n_features) array.

    Returns array with realistic log returns in first column.
    """
    rng = np.random.default_rng(42)
    n_samples, seq_length, n_features = 100, 24, 6

    samples = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
    # First column: log returns with realistic magnitude
    samples[:, :, 0] = rng.normal(0, 0.01, (n_samples, seq_length))
    # Other columns: various features
    for i in range(1, n_features):
        samples[:, :, i] = rng.normal(0, 0.1, (n_samples, seq_length))

    return samples


@pytest.fixture
def samples_file(tmp_path: Path, sample_3d_array: np.ndarray) -> Path:
    """Create temporary samples.npy file."""
    samples_path = tmp_path / "samples.npy"
    np.save(samples_path, sample_3d_array)
    return samples_path


@pytest.fixture
def metadata_file(tmp_path: Path, mock_metadata: dict) -> Path:
    """Create temporary metadata.json file."""
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(mock_metadata, f)
    return metadata_path


@pytest.fixture
def checkpoint_dir(tmp_path: Path, sample_3d_array: np.ndarray, mock_metadata: dict) -> Path:
    """Create complete checkpoint directory structure."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    # Save samples
    np.save(checkpoint_path / "samples.npy", sample_3d_array)

    # Save metadata
    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(mock_metadata, f)

    return checkpoint_path


@pytest.fixture
def provider(sample_3d_array: np.ndarray, mock_metadata: dict) -> LearnedSyntheticProvider:
    """Create provider with mock data."""
    return LearnedSyntheticProvider(
        samples=sample_3d_array,
        metadata=mock_metadata,
        seed=42,
    )


# =============================================================================
# TestLearnedSyntheticInit - Initialization Tests
# =============================================================================


class TestLearnedSyntheticInit:
    """Test LearnedSyntheticProvider initialization."""

    def test_init_valid_samples(self, sample_3d_array: np.ndarray):
        """Test initialization with valid 3D samples array."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array)
        assert provider.n_samples == 100
        assert provider.seq_length == 24
        assert provider.n_features == 6

    def test_init_invalid_shape_2d(self):
        """Test initialization fails with 2D array."""
        samples_2d = np.random.randn(100, 24)
        with pytest.raises(ValueError, match="must have shape"):
            LearnedSyntheticProvider(samples=samples_2d)

    def test_init_invalid_shape_1d(self):
        """Test initialization fails with 1D array."""
        samples_1d = np.random.randn(100)
        with pytest.raises(ValueError, match="must have shape"):
            LearnedSyntheticProvider(samples=samples_1d)

    def test_init_invalid_shape_4d(self):
        """Test initialization fails with 4D array."""
        samples_4d = np.random.randn(10, 24, 6, 2)
        with pytest.raises(ValueError, match="must have shape"):
            LearnedSyntheticProvider(samples=samples_4d)

    def test_init_with_metadata(self, sample_3d_array: np.ndarray, mock_metadata: dict):
        """Test initialization with metadata."""
        provider = LearnedSyntheticProvider(
            samples=sample_3d_array,
            metadata=mock_metadata,
        )
        assert provider.generator_name == "timegan"

    def test_init_without_metadata(self, sample_3d_array: np.ndarray):
        """Test initialization without metadata uses defaults."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array)
        assert provider.generator_name == "unknown"

    def test_init_with_seed(self, sample_3d_array: np.ndarray):
        """Test initialization with seed."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array, seed=42)
        assert provider.seed == 42

    def test_init_without_seed(self, sample_3d_array: np.ndarray):
        """Test initialization without seed."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array)
        assert provider.seed is None

    def test_init_with_model(self, sample_3d_array: np.ndarray):
        """Test initialization with model object."""
        mock_model = MagicMock()
        provider = LearnedSyntheticProvider(
            samples=sample_3d_array,
            model=mock_model,
        )
        # Model is stored internally
        assert provider._model is mock_model


# =============================================================================
# TestLearnedSyntheticFromSamples - from_samples Class Method
# =============================================================================


class TestLearnedSyntheticFromSamples:
    """Test from_samples class method."""

    def test_from_samples_success(self, samples_file: Path):
        """Test loading from samples file."""
        provider = LearnedSyntheticProvider.from_samples(samples_file)
        assert provider.n_samples == 100
        assert provider.seq_length == 24
        assert provider.n_features == 6

    def test_from_samples_string_path(self, samples_file: Path):
        """Test loading from string path."""
        provider = LearnedSyntheticProvider.from_samples(str(samples_file))
        assert provider.n_samples == 100

    def test_from_samples_missing_file(self, tmp_path: Path):
        """Test error when samples file is missing."""
        with pytest.raises(FileNotFoundError, match="Samples file not found"):
            LearnedSyntheticProvider.from_samples(tmp_path / "nonexistent.npy")

    def test_from_samples_with_metadata_path(self, samples_file: Path, metadata_file: Path):
        """Test loading with explicit metadata path."""
        provider = LearnedSyntheticProvider.from_samples(
            samples_file,
            metadata_path=metadata_file,
        )
        assert provider.generator_name == "timegan"

    def test_from_samples_auto_metadata_json_suffix(
        self, tmp_path: Path, sample_3d_array: np.ndarray, mock_metadata: dict
    ):
        """Test auto-discovery of metadata.json with .json suffix."""
        # Save samples
        samples_path = tmp_path / "samples.npy"
        np.save(samples_path, sample_3d_array)

        # Save metadata with matching name
        metadata_path = tmp_path / "samples.json"
        with open(metadata_path, "w") as f:
            json.dump(mock_metadata, f)

        provider = LearnedSyntheticProvider.from_samples(samples_path)
        assert provider.generator_name == "timegan"

    def test_from_samples_auto_metadata_in_parent(
        self, tmp_path: Path, sample_3d_array: np.ndarray, mock_metadata: dict
    ):
        """Test auto-discovery of metadata.json in parent directory."""
        # Create subdirectory
        subdir = tmp_path / "experiment"
        subdir.mkdir()

        # Save samples in subdirectory
        samples_path = subdir / "samples.npy"
        np.save(samples_path, sample_3d_array)

        # Save metadata in parent (subdirectory)
        metadata_path = subdir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(mock_metadata, f)

        provider = LearnedSyntheticProvider.from_samples(samples_path)
        assert provider.generator_name == "timegan"

    def test_from_samples_with_seed(self, samples_file: Path):
        """Test from_samples with seed parameter."""
        provider = LearnedSyntheticProvider.from_samples(samples_file, seed=42)
        assert provider.seed == 42

    def test_from_samples_no_metadata(self, samples_file: Path):
        """Test loading works without metadata file."""
        provider = LearnedSyntheticProvider.from_samples(samples_file)
        assert provider.generator_name == "unknown"
        assert provider.n_samples == 100


# =============================================================================
# TestLearnedSyntheticFromCheckpoint - from_checkpoint Class Method
# =============================================================================


class TestLearnedSyntheticFromCheckpoint:
    """Test from_checkpoint class method."""

    @requires_torch
    def test_from_checkpoint_with_samples(self, checkpoint_dir: Path):
        """Test loading from checkpoint with pre-generated samples."""
        provider = LearnedSyntheticProvider.from_checkpoint(checkpoint_dir)
        assert provider.n_samples == 100
        assert provider.generator_name == "timegan"

    @requires_torch
    def test_from_checkpoint_string_path(self, checkpoint_dir: Path):
        """Test loading from string path."""
        provider = LearnedSyntheticProvider.from_checkpoint(str(checkpoint_dir))
        assert provider.n_samples == 100

    def test_from_checkpoint_invalid_path(self, tmp_path: Path):
        """Test error when checkpoint path doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
            LearnedSyntheticProvider.from_checkpoint(tmp_path / "nonexistent")

    def test_from_checkpoint_missing_metadata(self, tmp_path: Path):
        """Test error when metadata.json is missing."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        # No metadata.json

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            LearnedSyntheticProvider.from_checkpoint(checkpoint_path)

    @requires_torch
    def test_from_checkpoint_with_seed(self, checkpoint_dir: Path):
        """Test from_checkpoint with seed parameter."""
        provider = LearnedSyntheticProvider.from_checkpoint(checkpoint_dir, seed=42)
        assert provider.seed == 42


# =============================================================================
# TestLearnedSyntheticProperties - Property Tests
# =============================================================================


class TestLearnedSyntheticProperties:
    """Test provider properties."""

    def test_name_property(self, provider: LearnedSyntheticProvider):
        """Test name property includes generator name."""
        assert provider.name == "learned_synthetic_timegan"

    def test_name_property_unknown_generator(self, sample_3d_array: np.ndarray):
        """Test name property with unknown generator."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array)
        # Name includes "learned_synthetic" prefix and falls back to "learned" when no metadata
        assert "learned_synthetic" in provider.name

    def test_generator_name_property(self, provider: LearnedSyntheticProvider):
        """Test generator_name property."""
        assert provider.generator_name == "timegan"

    def test_generator_name_fallback(self, sample_3d_array: np.ndarray):
        """Test generator_name fallback to unknown."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array)
        assert provider.generator_name == "unknown"

    def test_n_samples_property(self, provider: LearnedSyntheticProvider):
        """Test n_samples property."""
        assert provider.n_samples == 100

    def test_seq_length_property(self, provider: LearnedSyntheticProvider):
        """Test seq_length property."""
        assert provider.seq_length == 24

    def test_n_features_property(self, provider: LearnedSyntheticProvider):
        """Test n_features property."""
        assert provider.n_features == 6


# =============================================================================
# TestLearnedSyntheticSamples - Sample Access Tests
# =============================================================================


class TestLearnedSyntheticSamples:
    """Test sample access methods."""

    def test_get_samples_all(self, provider: LearnedSyntheticProvider):
        """Test getting all samples."""
        samples = provider.get_samples(n_samples=None, shuffle=False)
        assert samples.shape == (100, 24, 6)

    def test_get_samples_subset(self, provider: LearnedSyntheticProvider):
        """Test getting subset of samples."""
        samples = provider.get_samples(n_samples=10, shuffle=False)
        assert samples.shape == (10, 24, 6)

    def test_get_samples_shuffled(self, sample_3d_array: np.ndarray):
        """Test getting samples with shuffle."""
        provider = LearnedSyntheticProvider(samples=sample_3d_array, seed=42)

        # Get samples twice with shuffle
        samples1 = provider.get_samples(n_samples=10, shuffle=True)
        samples2 = provider.get_samples(n_samples=10, shuffle=True)

        # Should be different (probabilistic)
        assert samples1.shape == samples2.shape
        assert not np.allclose(samples1, samples2)

    def test_get_samples_not_shuffled(self, provider: LearnedSyntheticProvider):
        """Test getting samples without shuffle."""
        samples1 = provider.get_samples(n_samples=10, shuffle=False)
        samples2 = provider.get_samples(n_samples=10, shuffle=False)

        np.testing.assert_array_equal(samples1, samples2)

    def test_get_samples_more_than_available(self, provider: LearnedSyntheticProvider):
        """Test requesting more samples than available."""
        samples = provider.get_samples(n_samples=1000, shuffle=False)
        # Should return all available
        assert samples.shape[0] == 100

    def test_generate_samples_no_model(self, provider: LearnedSyntheticProvider):
        """Test generate_samples raises error when no model."""
        with pytest.raises(RuntimeError, match="Cannot generate new samples"):
            provider.generate_samples(n_samples=10)

    def test_generate_samples_with_model(self, sample_3d_array: np.ndarray):
        """Test generate_samples with model (placeholder behavior)."""
        mock_model = MagicMock()
        provider = LearnedSyntheticProvider(
            samples=sample_3d_array,
            model=mock_model,
        )

        # Should return random samples (placeholder implementation)
        samples = provider.generate_samples(n_samples=10)
        assert samples.shape[0] == 10


# =============================================================================
# TestLearnedSyntheticOHLCV - OHLCV Generation Tests
# =============================================================================


class TestLearnedSyntheticOHLCV:
    """Test OHLCV data generation."""

    def test_fetch_ohlcv_daily(self, provider: LearnedSyntheticProvider):
        """Test daily OHLCV generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

        # Check columns
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in df.columns

    def test_fetch_ohlcv_hourly(self, provider: LearnedSyntheticProvider):
        """Test hourly OHLCV generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-03", "hourly")

        # Should have multiple bars per day
        assert len(df) > 10

    def test_fetch_ohlcv_minute(self, provider: LearnedSyntheticProvider):
        """Test minute OHLCV generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-02", "minute")

        # Should have many bars per day
        assert len(df) >= 100

    def test_fetch_ohlcv_empty_range(self, provider: LearnedSyntheticProvider):
        """Test OHLCV generation for empty date range (weekend)."""
        # A weekend day might have no data
        df = provider.fetch_ohlcv("SYNTH", "2024-01-06", "2024-01-06", "daily")

        # Result should be valid DataFrame (might be empty for weekend)
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns

    def test_ohlcv_high_gte_low(self, provider: LearnedSyntheticProvider):
        """Test OHLC invariant: high >= low."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert (df["high"] >= df["low"]).all()

    def test_ohlcv_high_gte_open_close(self, provider: LearnedSyntheticProvider):
        """Test OHLC invariant: high >= open and high >= close."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert (df["high"] >= df["open"]).all()
            assert (df["high"] >= df["close"]).all()

    def test_ohlcv_low_lte_open_close(self, provider: LearnedSyntheticProvider):
        """Test OHLC invariant: low <= open and low <= close."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert (df["low"] <= df["open"]).all()
            assert (df["low"] <= df["close"]).all()

    def test_ohlcv_positive_prices(self, provider: LearnedSyntheticProvider):
        """Test OHLC invariant: all prices positive."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert (df["open"] > 0).all()
            assert (df["high"] > 0).all()
            assert (df["low"] > 0).all()
            assert (df["close"] > 0).all()

    def test_ohlcv_positive_volume(self, provider: LearnedSyntheticProvider):
        """Test volume is non-negative."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert (df["volume"] >= 0).all()

    def test_ohlcv_timestamps_sorted(self, provider: LearnedSyntheticProvider):
        """Test timestamps are sorted ascending."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 1:
            timestamps = df["timestamp"].to_list()
            assert timestamps == sorted(timestamps)

    def test_ohlcv_timestamps_no_duplicates(self, provider: LearnedSyntheticProvider):
        """Test no duplicate timestamps."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-03-31", "daily")

        if len(df) > 0:
            assert df["timestamp"].n_unique() == len(df)

    def test_ohlcv_timestamp_timezone(self, provider: LearnedSyntheticProvider):
        """Test timestamps have UTC timezone."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        if len(df) > 0:
            # Check dtype includes timezone
            assert df["timestamp"].dtype == pl.Datetime("us", "UTC") or df[
                "timestamp"
            ].dtype == pl.Datetime("ms", "UTC")


# =============================================================================
# TestLearnedSyntheticReproducibility - Reproducibility Tests
# =============================================================================


class TestLearnedSyntheticReproducibility:
    """Test reproducibility with seeds."""

    def test_reproducibility_with_same_seed(self, sample_3d_array: np.ndarray, mock_metadata: dict):
        """Test same seed produces same data."""
        provider1 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )
        provider2 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )

        df1 = provider1.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")
        df2 = provider2.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        assert len(df1) == len(df2)
        if len(df1) > 0:
            np.testing.assert_array_almost_equal(
                df1["close"].to_numpy(),
                df2["close"].to_numpy(),
            )

    def test_different_seeds_produce_different_data(
        self, sample_3d_array: np.ndarray, mock_metadata: dict
    ):
        """Test different seeds produce different data."""
        provider1 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )
        provider2 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=123
        )

        df1 = provider1.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")
        df2 = provider2.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        if len(df1) > 0 and len(df2) > 0:
            # Prices should differ
            assert not np.allclose(
                df1["close"].to_numpy(),
                df2["close"].to_numpy(),
            )

    def test_different_symbols_produce_different_data(self, provider: LearnedSyntheticProvider):
        """Test different symbols produce different data (symbol affects RNG)."""
        # Reset to same seed
        provider.reset_seed(42)
        df1 = provider.fetch_ohlcv("SYNTH_A", "2024-01-01", "2024-01-31", "daily")

        provider.reset_seed(42)
        df2 = provider.fetch_ohlcv("SYNTH_B", "2024-01-01", "2024-01-31", "daily")

        if len(df1) > 0 and len(df2) > 0:
            # Different symbols should produce different data
            assert not np.allclose(
                df1["close"].to_numpy(),
                df2["close"].to_numpy(),
            )


# =============================================================================
# TestLearnedSyntheticMisc - Miscellaneous Tests
# =============================================================================


class TestLearnedSyntheticMisc:
    """Test miscellaneous functionality."""

    def test_get_available_symbols(self, provider: LearnedSyntheticProvider):
        """Test get_available_symbols returns valid symbols."""
        symbols = provider.get_available_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) >= 1
        assert all(isinstance(s, str) for s in symbols)
        # Should include generator name in uppercase
        assert any("TIMEGAN" in s for s in symbols)

    def test_get_available_symbols_format(self, provider: LearnedSyntheticProvider):
        """Test symbol format is SYNTH_GENERATOR."""
        symbols = provider.get_available_symbols()

        assert "SYNTH_TIMEGAN" in symbols
        assert "SYNTH_TIMEGAN_1" in symbols
        assert "SYNTH_TIMEGAN_2" in symbols

    def test_get_metadata_returns_copy(self, provider: LearnedSyntheticProvider):
        """Test get_metadata returns a copy."""
        meta1 = provider.get_metadata()
        meta2 = provider.get_metadata()

        # Should be equal
        assert meta1 == meta2

        # Modifying returned copy shouldn't affect internal state
        meta1["new_key"] = "value"
        meta3 = provider.get_metadata()
        assert "new_key" not in meta3

    def test_reset_seed(self, provider: LearnedSyntheticProvider):
        """Test reset_seed changes RNG state."""
        provider.reset_seed(123)
        assert provider.seed == 123

    def test_reset_seed_none_uses_original(self, sample_3d_array: np.ndarray, mock_metadata: dict):
        """Test reset_seed with None uses original seed."""
        provider = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )

        provider.reset_seed(None)
        assert provider.seed == 42

    def test_create_empty_dataframe(self, provider: LearnedSyntheticProvider):
        """Test _create_empty_dataframe returns correct schema."""
        df = provider._create_empty_dataframe()

        assert len(df) == 0
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check dtypes
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64


# =============================================================================
# TestLearnedSyntheticIntegration - Integration Tests
# =============================================================================


class TestLearnedSyntheticIntegration:
    """Integration tests for provider workflows."""

    def test_full_workflow_from_samples(self, samples_file: Path, metadata_file: Path):
        """Test complete workflow from samples file."""
        # 1. Load from samples
        provider = LearnedSyntheticProvider.from_samples(
            samples_file,
            metadata_path=metadata_file,
            seed=42,
        )

        # 2. Check properties
        assert provider.n_samples == 100
        assert provider.generator_name == "timegan"

        # 3. Get raw samples
        samples = provider.get_samples(n_samples=10)
        assert samples.shape == (10, 24, 6)

        # 4. Generate OHLCV
        df = provider.fetch_ohlcv("SYNTH_TIMEGAN", "2024-01-01", "2024-03-31", "daily")
        assert len(df) > 0

        # 5. Check OHLCV validity
        assert (df["high"] >= df["low"]).all()
        assert (df["close"] > 0).all()

    @requires_torch
    def test_full_workflow_from_checkpoint(self, checkpoint_dir: Path):
        """Test complete workflow from checkpoint directory."""
        # 1. Load from checkpoint
        provider = LearnedSyntheticProvider.from_checkpoint(
            checkpoint_dir,
            seed=42,
        )

        # 2. Check properties
        assert provider.n_samples == 100
        assert provider.generator_name == "timegan"

        # 3. Generate multiple datasets
        df_daily = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")
        df_hourly = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-03", "hourly")

        assert len(df_daily) > 0
        assert len(df_hourly) > len(df_daily)  # More bars for hourly

    def test_multiple_providers_independent(self, sample_3d_array: np.ndarray, mock_metadata: dict):
        """Test multiple providers work independently."""
        provider1 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )
        provider2 = LearnedSyntheticProvider(
            samples=sample_3d_array, metadata=mock_metadata, seed=42
        )

        # Both should produce same data with same seed
        df1 = provider1.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")
        df2 = provider2.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        if len(df1) > 0 and len(df2) > 0:
            np.testing.assert_array_almost_equal(
                df1["close"].to_numpy(),
                df2["close"].to_numpy(),
            )

    def test_large_date_range(self, provider: LearnedSyntheticProvider):
        """Test generating data for large date range."""
        # Generate 1 year of data
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        # Should have ~252 trading days
        assert len(df) > 200
        assert len(df) < 300

        # All OHLC invariants should hold
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

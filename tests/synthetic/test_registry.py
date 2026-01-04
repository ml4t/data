"""Tests for SyntheticRegistry.

These tests verify the synthetic data generator registry can discover,
load, and provide access to trained generative model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml4t.data.synthetic.registry import SyntheticRegistry

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
            "n_samples": 1000,
            "seq_length": 24,
            "n_features": 6,
        },
        "evaluation": {
            "tstr_ratio": 0.95,
            "discriminative_score": 0.12,
            "predictive_score": 0.08,
        },
    }


@pytest.fixture
def mock_samples() -> np.ndarray:
    """Create mock samples array."""
    return np.random.randn(100, 24, 6).astype(np.float32)


@pytest.fixture
def mock_checkpoint_dir(tmp_path: Path, mock_metadata: dict, mock_samples: np.ndarray) -> Path:
    """Create mock checkpoint directory structure.

    Structure:
        checkpoints/
        ├── timegan/
        │   └── etf_returns/
        │       ├── metadata.json
        │       └── samples.npy
        └── tailgan/
            └── tail_risk/
                ├── metadata.json
                └── samples.npy
    """
    checkpoints_dir = tmp_path / "synthetic" / "checkpoints"

    # Create timegan checkpoint
    timegan_exp = checkpoints_dir / "timegan" / "etf_returns"
    timegan_exp.mkdir(parents=True)
    with open(timegan_exp / "metadata.json", "w") as f:
        json.dump(mock_metadata, f)
    np.save(timegan_exp / "samples.npy", mock_samples)

    # Create tailgan checkpoint with different metadata
    tailgan_meta = mock_metadata.copy()
    tailgan_meta["generator"] = {"name": "tailgan", "version": "1.0.0", "paper": "Tail-GAN"}
    tailgan_exp = checkpoints_dir / "tailgan" / "tail_risk"
    tailgan_exp.mkdir(parents=True)
    with open(tailgan_exp / "metadata.json", "w") as f:
        json.dump(tailgan_meta, f)
    np.save(tailgan_exp / "samples.npy", mock_samples)

    return tmp_path


@pytest.fixture
def registry(mock_checkpoint_dir: Path) -> SyntheticRegistry:
    """Create registry with mock checkpoint directory."""
    return SyntheticRegistry(
        data_dir=mock_checkpoint_dir,
        checkpoints_subdir="synthetic/checkpoints",
    )


@pytest.fixture
def empty_registry(tmp_path: Path) -> SyntheticRegistry:
    """Create registry with empty checkpoint directory."""
    checkpoints_dir = tmp_path / "synthetic" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    return SyntheticRegistry(
        data_dir=tmp_path,
        checkpoints_subdir="synthetic/checkpoints",
    )


@pytest.fixture
def nonexistent_registry(tmp_path: Path) -> SyntheticRegistry:
    """Create registry with non-existent checkpoint directory."""
    return SyntheticRegistry(
        data_dir=tmp_path,
        checkpoints_subdir="nonexistent/path",
    )


# =============================================================================
# TestSyntheticRegistry - Basic Initialization
# =============================================================================


class TestSyntheticRegistryInit:
    """Test SyntheticRegistry initialization."""

    def test_init_with_custom_data_dir(self, tmp_path: Path):
        """Test initialization with custom data directory."""
        registry = SyntheticRegistry(
            data_dir=tmp_path,
            checkpoints_subdir="custom/checkpoints",
        )
        assert registry.checkpoints_dir == tmp_path / "custom" / "checkpoints"

    def test_init_with_string_path(self, tmp_path: Path):
        """Test initialization with string path."""
        registry = SyntheticRegistry(
            data_dir=str(tmp_path),
            checkpoints_subdir="synthetic/checkpoints",
        )
        assert isinstance(registry.checkpoints_dir, Path)
        assert registry.checkpoints_dir == tmp_path / "synthetic" / "checkpoints"

    def test_checkpoints_dir_property(self, registry: SyntheticRegistry, mock_checkpoint_dir: Path):
        """Test checkpoints_dir property returns correct path."""
        expected = mock_checkpoint_dir / "synthetic" / "checkpoints"
        assert registry.checkpoints_dir == expected
        assert isinstance(registry.checkpoints_dir, Path)


# =============================================================================
# TestRegistryDiscovery - Generator Discovery
# =============================================================================


class TestRegistryDiscovery:
    """Test generator discovery functionality."""

    def test_list_generators_with_checkpoints(self, registry: SyntheticRegistry):
        """Test listing generators when checkpoints exist."""
        generators = registry.list_generators()
        assert isinstance(generators, list)
        assert len(generators) == 2
        assert "timegan" in generators
        assert "tailgan" in generators
        # Should be sorted
        assert generators == sorted(generators)

    def test_list_generators_empty_dir(self, empty_registry: SyntheticRegistry):
        """Test listing generators from empty directory."""
        generators = empty_registry.list_generators()
        assert generators == []

    def test_list_generators_nonexistent_dir(self, nonexistent_registry: SyntheticRegistry):
        """Test listing generators from non-existent directory."""
        generators = nonexistent_registry.list_generators()
        assert generators == []

    def test_list_experiments(self, registry: SyntheticRegistry):
        """Test listing experiments for a generator."""
        experiments = registry.list_experiments("timegan")
        assert isinstance(experiments, list)
        assert len(experiments) == 1
        assert "etf_returns" in experiments

    def test_list_experiments_invalid_generator(self, registry: SyntheticRegistry):
        """Test listing experiments for non-existent generator."""
        experiments = registry.list_experiments("nonexistent")
        assert experiments == []

    def test_list_all(self, registry: SyntheticRegistry):
        """Test listing all generators and experiments."""
        all_generators = registry.list_all()
        assert isinstance(all_generators, dict)
        assert len(all_generators) == 2
        assert "timegan" in all_generators
        assert "tailgan" in all_generators
        assert all_generators["timegan"] == ["etf_returns"]
        assert all_generators["tailgan"] == ["tail_risk"]

    def test_list_all_empty(self, empty_registry: SyntheticRegistry):
        """Test listing all from empty registry."""
        all_generators = empty_registry.list_all()
        assert all_generators == {}

    def test_discover_skips_non_directories(self, mock_checkpoint_dir: Path):
        """Test that discovery skips non-directory files."""
        # Create a file in the checkpoints directory
        checkpoints_dir = mock_checkpoint_dir / "synthetic" / "checkpoints"
        (checkpoints_dir / "README.md").write_text("# Checkpoints")

        registry = SyntheticRegistry(
            data_dir=mock_checkpoint_dir,
            checkpoints_subdir="synthetic/checkpoints",
        )
        generators = registry.list_generators()
        # Should only include directories, not files
        assert "README.md" not in generators
        assert "README" not in generators

    def test_discover_skips_dirs_without_metadata(self, mock_checkpoint_dir: Path):
        """Test that discovery skips directories without metadata.json."""
        # Create a generator directory without metadata
        checkpoints_dir = mock_checkpoint_dir / "synthetic" / "checkpoints"
        incomplete_dir = checkpoints_dir / "incomplete" / "experiment"
        incomplete_dir.mkdir(parents=True)
        # No metadata.json

        registry = SyntheticRegistry(
            data_dir=mock_checkpoint_dir,
            checkpoints_subdir="synthetic/checkpoints",
        )
        # The generator exists but has no valid experiments
        generators = registry.list_generators()
        assert "incomplete" in generators  # Generator dir exists
        experiments = registry.list_experiments("incomplete")
        assert experiments == []  # But no valid experiments


# =============================================================================
# TestRegistryPathResolution - Checkpoint Path Resolution
# =============================================================================


class TestRegistryPathResolution:
    """Test checkpoint path resolution."""

    def test_get_checkpoint_path_valid(
        self, registry: SyntheticRegistry, mock_checkpoint_dir: Path
    ):
        """Test getting checkpoint path for valid generator/experiment."""
        path = registry._get_checkpoint_path("timegan", "etf_returns")
        expected = mock_checkpoint_dir / "synthetic" / "checkpoints" / "timegan" / "etf_returns"
        assert path == expected

    def test_get_checkpoint_path_default_experiment(
        self, registry: SyntheticRegistry, mock_checkpoint_dir: Path
    ):
        """Test getting checkpoint path with default experiment."""
        path = registry._get_checkpoint_path("timegan", experiment=None)
        expected = mock_checkpoint_dir / "synthetic" / "checkpoints" / "timegan" / "etf_returns"
        assert path == expected

    def test_get_checkpoint_path_invalid_generator(self, registry: SyntheticRegistry):
        """Test error when generator doesn't exist."""
        with pytest.raises(ValueError, match="Generator 'nonexistent' not found"):
            registry._get_checkpoint_path("nonexistent", "experiment")

    def test_get_checkpoint_path_invalid_experiment(self, registry: SyntheticRegistry):
        """Test error when experiment doesn't exist."""
        with pytest.raises(ValueError, match="Experiment 'nonexistent' not found"):
            registry._get_checkpoint_path("timegan", "nonexistent")

    def test_get_checkpoint_path_no_experiments(self, mock_checkpoint_dir: Path):
        """Test error when generator has no experiments."""
        # Create generator dir without experiments
        checkpoints_dir = mock_checkpoint_dir / "synthetic" / "checkpoints"
        (checkpoints_dir / "empty_gen").mkdir(parents=True)

        registry = SyntheticRegistry(
            data_dir=mock_checkpoint_dir,
            checkpoints_subdir="synthetic/checkpoints",
        )
        with pytest.raises(ValueError, match="No experiments found"):
            registry._get_checkpoint_path("empty_gen")

    def test_error_message_includes_available_options(self, registry: SyntheticRegistry):
        """Test that error messages include available options."""
        with pytest.raises(ValueError) as exc_info:
            registry._get_checkpoint_path("nonexistent")

        error_msg = str(exc_info.value)
        assert "timegan" in error_msg or "tailgan" in error_msg


# =============================================================================
# TestRegistryMetadata - Metadata Loading
# =============================================================================


class TestRegistryMetadata:
    """Test metadata loading functionality."""

    def test_get_metadata_success(self, registry: SyntheticRegistry, mock_metadata: dict):
        """Test loading metadata successfully."""
        metadata = registry.get_metadata("timegan", "etf_returns")
        assert isinstance(metadata, dict)
        assert metadata["generator"]["name"] == "timegan"
        assert metadata["data"]["n_samples"] == 1000

    def test_get_metadata_default_experiment(self, registry: SyntheticRegistry):
        """Test loading metadata with default experiment."""
        metadata = registry.get_metadata("timegan", experiment=None)
        assert metadata["generator"]["name"] == "timegan"

    def test_get_metadata_cached(self, registry: SyntheticRegistry):
        """Test that metadata is cached."""
        # First call
        metadata1 = registry.get_metadata("timegan", "etf_returns")
        # Second call should return cached version
        metadata2 = registry.get_metadata("timegan", "etf_returns")

        assert metadata1 == metadata2
        # Check cache was used (same object)
        assert len(registry._cache) == 1

    def test_get_metadata_different_experiments_cached_separately(
        self, registry: SyntheticRegistry
    ):
        """Test that different experiments are cached separately."""
        meta1 = registry.get_metadata("timegan", "etf_returns")
        meta2 = registry.get_metadata("tailgan", "tail_risk")

        assert meta1["generator"]["name"] == "timegan"
        assert meta2["generator"]["name"] == "tailgan"
        assert len(registry._cache) == 2


# =============================================================================
# TestRegistrySamples - Sample Loading
# =============================================================================


class TestRegistrySamples:
    """Test sample loading functionality."""

    def test_load_samples_all(self, registry: SyntheticRegistry, mock_samples: np.ndarray):
        """Test loading all samples."""
        samples = registry.load_samples("timegan", "etf_returns")
        assert isinstance(samples, np.ndarray)
        assert samples.shape == mock_samples.shape

    def test_load_samples_subset(self, registry: SyntheticRegistry):
        """Test loading subset of samples."""
        samples = registry.load_samples("timegan", "etf_returns", n_samples=10)
        assert samples.shape[0] == 10

    def test_load_samples_more_than_available(self, registry: SyntheticRegistry):
        """Test loading more samples than available returns all."""
        samples = registry.load_samples("timegan", "etf_returns", n_samples=1000)
        # Should return all available (100 from mock)
        assert samples.shape[0] == 100

    def test_load_samples_shuffled(self, registry: SyntheticRegistry):
        """Test loading samples with shuffle."""
        # Load twice with shuffle
        samples1 = registry.load_samples("timegan", "etf_returns", shuffle=True)
        samples2 = registry.load_samples("timegan", "etf_returns", shuffle=True)

        # Arrays should likely be different (very unlikely to be same)
        # Note: This is probabilistic, could theoretically fail
        assert samples1.shape == samples2.shape
        # At least one element should differ
        assert not np.allclose(samples1, samples2)

    def test_load_samples_not_shuffled(self, registry: SyntheticRegistry):
        """Test loading samples without shuffle."""
        samples1 = registry.load_samples("timegan", "etf_returns", shuffle=False)
        samples2 = registry.load_samples("timegan", "etf_returns", shuffle=False)

        # Should be identical
        np.testing.assert_array_equal(samples1, samples2)

    def test_load_samples_default_experiment(self, registry: SyntheticRegistry):
        """Test loading samples with default experiment."""
        samples = registry.load_samples("timegan", experiment=None)
        assert samples.shape == (100, 24, 6)

    def test_load_samples_missing_file(self, mock_checkpoint_dir: Path):
        """Test error when samples file is missing."""
        # Create checkpoint without samples.npy
        checkpoints_dir = mock_checkpoint_dir / "synthetic" / "checkpoints"
        no_samples_dir = checkpoints_dir / "no_samples" / "experiment"
        no_samples_dir.mkdir(parents=True)
        with open(no_samples_dir / "metadata.json", "w") as f:
            json.dump({"generator": {"name": "no_samples"}}, f)

        registry = SyntheticRegistry(
            data_dir=mock_checkpoint_dir,
            checkpoints_subdir="synthetic/checkpoints",
        )
        with pytest.raises(FileNotFoundError, match="Samples file not found"):
            registry.load_samples("no_samples", "experiment")


# =============================================================================
# TestRegistryProvider - Provider Creation
# =============================================================================


class TestRegistryProvider:
    """Test provider creation functionality."""

    def test_get_provider_from_samples(self, registry: SyntheticRegistry):
        """Test getting provider when samples.npy exists."""
        from ml4t.data.providers.learned_synthetic import LearnedSyntheticProvider

        provider = registry.get_provider("timegan", "etf_returns")
        assert isinstance(provider, LearnedSyntheticProvider)
        assert provider.n_samples == 100
        assert provider.seq_length == 24
        assert provider.n_features == 6

    def test_get_provider_with_seed(self, registry: SyntheticRegistry):
        """Test getting provider with specific seed."""
        provider = registry.get_provider("timegan", "etf_returns", seed=42)
        assert provider.seed == 42

    def test_get_provider_default_experiment(self, registry: SyntheticRegistry):
        """Test getting provider with default experiment."""
        provider = registry.get_provider("timegan", experiment=None)
        assert provider.n_samples == 100

    def test_get_provider_generates_ohlcv(self, registry: SyntheticRegistry):
        """Test that provider can generate OHLCV data."""
        provider = registry.get_provider("timegan", "etf_returns", seed=42)
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns


# =============================================================================
# TestRegistryInfo - Info and Summary
# =============================================================================


class TestRegistryInfo:
    """Test info and summary functionality."""

    def test_get_info_format(self, registry: SyntheticRegistry):
        """Test get_info returns formatted string."""
        info = registry.get_info("timegan", "etf_returns")
        assert isinstance(info, str)

        # Check expected content
        assert "timegan" in info.lower()
        assert "1.0.0" in info  # Version
        assert "Time-series GAN" in info  # Paper
        assert "ETF returns" in info  # Source
        assert "1000" in info  # n_samples
        assert "24" in info  # seq_length

    def test_get_info_includes_evaluation_metrics(self, registry: SyntheticRegistry):
        """Test get_info includes evaluation metrics."""
        info = registry.get_info("timegan", "etf_returns")

        # Should include evaluation section
        assert "Evaluation" in info
        assert "tstr_ratio" in info

    def test_get_info_includes_path(self, registry: SyntheticRegistry):
        """Test get_info includes checkpoint path."""
        info = registry.get_info("timegan", "etf_returns")
        assert "Path:" in info

    def test_summary_with_generators(self, registry: SyntheticRegistry):
        """Test summary with available generators."""
        summary = registry.summary()
        assert isinstance(summary, str)

        # Should include both generators
        assert "timegan" in summary
        assert "tailgan" in summary
        assert "etf_returns" in summary
        assert "tail_risk" in summary

        # Should include sample counts
        assert "100" in summary  # n_samples from mock

    def test_summary_empty(self, empty_registry: SyntheticRegistry):
        """Test summary when no generators found."""
        summary = empty_registry.summary()
        assert "No generators found" in summary

    def test_summary_handles_metadata_errors(self, mock_checkpoint_dir: Path):
        """Test summary handles metadata loading errors gracefully."""
        # Create checkpoint with invalid metadata
        checkpoints_dir = mock_checkpoint_dir / "synthetic" / "checkpoints"
        broken_dir = checkpoints_dir / "broken" / "experiment"
        broken_dir.mkdir(parents=True)
        (broken_dir / "metadata.json").write_text("invalid json{")

        registry = SyntheticRegistry(
            data_dir=mock_checkpoint_dir,
            checkpoints_subdir="synthetic/checkpoints",
        )
        summary = registry.summary()

        # Should still include valid generators
        assert "timegan" in summary
        # Should mention error for broken generator
        assert "broken" in summary
        assert "Error" in summary


# =============================================================================
# TestRegistryIntegration - Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Integration tests for registry workflow."""

    def test_full_workflow(self, registry: SyntheticRegistry):
        """Test complete workflow: discover -> metadata -> samples -> provider."""
        # 1. Discover generators
        generators = registry.list_generators()
        assert len(generators) > 0

        # 2. List experiments
        gen = generators[0]
        experiments = registry.list_experiments(gen)
        assert len(experiments) > 0

        # 3. Get metadata
        exp = experiments[0]
        metadata = registry.get_metadata(gen, exp)
        assert "generator" in metadata

        # 4. Load samples
        samples = registry.load_samples(gen, exp, n_samples=10)
        assert samples.shape[0] == 10

        # 5. Get provider
        provider = registry.get_provider(gen, exp)
        assert provider.n_samples > 0

        # 6. Generate OHLCV
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
        assert len(df) > 0

    def test_multiple_generators_independent(self, registry: SyntheticRegistry):
        """Test that multiple generators work independently."""
        provider1 = registry.get_provider("timegan", seed=42)
        provider2 = registry.get_provider("tailgan", seed=42)

        # Different generator names
        assert provider1.generator_name == "timegan"
        assert provider2.generator_name == "tailgan"

        # Can generate data independently
        df1 = provider1.fetch_ohlcv("SYNTH1", "2024-01-01", "2024-01-10", "daily")
        df2 = provider2.fetch_ohlcv("SYNTH2", "2024-01-01", "2024-01-10", "daily")

        assert len(df1) > 0
        assert len(df2) > 0

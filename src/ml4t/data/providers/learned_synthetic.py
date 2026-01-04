"""Provider for learned generative models (TimeGAN, Sig-CWGAN, etc.).

This provider loads trained generative models or pre-generated samples
from Chapter 6 notebooks and provides a consistent API for generating
synthetic OHLCV data.

Key differences from SyntheticProvider:
- SyntheticProvider: Parameterize stochastic model → Generate on the fly
- LearnedSyntheticProvider: Load checkpoint → Generate from trained model

Usage examples:

    # From pre-generated samples (faster, no model needed)
    provider = LearnedSyntheticProvider.from_samples(
        DATA_DIR / "synthetic/timegan_sequences.npy"
    )
    df = provider.fetch_ohlcv("SYNTH_TIMEGAN", "2024-01-01", "2024-12-31", "daily")

    # From checkpoint (can generate new samples)
    provider = LearnedSyntheticProvider.from_checkpoint(
        DATA_DIR / "synthetic/checkpoints/timegan/etf_2010_2024"
    )
    df = provider.fetch_ohlcv("SYNTH_TIMEGAN", "2024-01-01", "2024-12-31", "daily")
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import structlog

from ml4t.data.providers.base import BaseProvider
from ml4t.data.synthetic import (
    generate_ohlc_from_close,
    generate_timestamps,
    generate_volume,
    get_bars_per_day,
    returns_to_prices,
)

logger = structlog.get_logger()


class LearnedSyntheticProvider(BaseProvider):
    """Provider for learned generative models.

    This provider wraps trained generative models (TimeGAN, Sig-CWGAN,
    Tail-GAN, TransFusion, GT-GAN, etc.) from Chapter 6 and provides
    a consistent API for generating synthetic OHLCV data.

    There are two modes of operation:
    1. Sample-based: Load pre-generated samples from .npy file
    2. Checkpoint-based: Load trained model and generate new samples

    For ML training workflows (TSTR), use `get_samples()` to access
    raw return sequences directly.

    Parameters
    ----------
    samples : np.ndarray
        Pre-loaded samples of shape (n_samples, seq_length, n_features)
    metadata : dict
        Metadata about the generator and training
    model : Any, optional
        Loaded model for generating new samples (checkpoint mode only)
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> # Load from samples
    >>> provider = LearnedSyntheticProvider.from_samples("timegan_sequences.npy")
    >>> df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

    >>> # Get raw samples for ML training
    >>> X_synth = provider.get_samples()  # shape: (n_samples, seq_length, n_features)
    """

    # No rate limiting needed for synthetic data
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1000, 1.0)

    # Trading days per year
    TRADING_DAYS = 252

    def __init__(
        self,
        samples: np.ndarray,
        metadata: dict[str, Any] | None = None,
        model: Any = None,
        seed: int | None = None,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize provider with samples or model.

        Note: Prefer using class methods from_samples() or from_checkpoint()
        instead of calling __init__ directly.
        """
        self._samples = samples
        self._metadata = metadata or {}
        self._model = model
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Validate samples shape
        if samples.ndim != 3:
            raise ValueError(
                f"Samples must have shape (n_samples, seq_length, n_features), got {samples.shape}"
            )

        self._n_samples, self._seq_length, self._n_features = samples.shape

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        logger.info(
            "Initialized LearnedSyntheticProvider",
            n_samples=self._n_samples,
            seq_length=self._seq_length,
            n_features=self._n_features,
            generator=self._metadata.get("generator", {}).get("name", "unknown"),
        )

    @classmethod
    def from_samples(
        cls,
        samples_path: str | Path,
        metadata_path: str | Path | None = None,
        seed: int | None = None,
    ) -> LearnedSyntheticProvider:
        """Create provider from pre-generated samples.

        This is the faster option when you don't need to generate new samples.

        Parameters
        ----------
        samples_path : str or Path
            Path to .npy file containing samples
            Shape: (n_samples, seq_length, n_features)
        metadata_path : str or Path, optional
            Path to metadata.json file. If None, tries to find it
            next to the samples file.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        LearnedSyntheticProvider
            Configured provider instance
        """
        samples_path = Path(samples_path)

        # Load samples
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")

        samples = np.load(samples_path)
        logger.info(f"Loaded samples from {samples_path}", shape=samples.shape)

        # Try to find metadata
        metadata = {}
        if metadata_path is None:
            # Look for metadata.json in same directory
            potential_paths = [
                samples_path.with_suffix(".json"),
                samples_path.parent / "metadata.json",
            ]
            for path in potential_paths:
                if path.exists():
                    metadata_path = path
                    break

        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")

        return cls(samples=samples, metadata=metadata, model=None, seed=seed)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        seed: int | None = None,
    ) -> LearnedSyntheticProvider:
        """Create provider from a trained model checkpoint.

        This allows generating new samples on the fly using the trained model.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint directory containing:
            - checkpoint.pt: Model weights
            - metadata.json: Training config and sample data
            - samples.npy (optional): Pre-generated samples
        device : str, default="cpu"
            Device to load model on ("cpu" or "cuda")
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        LearnedSyntheticProvider
            Configured provider instance
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        generator_name = metadata.get("generator", {}).get("name", "unknown")
        logger.info(f"Loading checkpoint for {generator_name}", path=checkpoint_path)

        # Load model based on generator type
        model = cls._load_model(checkpoint_path, generator_name, device)

        # Load or generate initial samples
        samples_file = checkpoint_path / "samples.npy"
        if samples_file.exists():
            samples = np.load(samples_file)
            logger.info("Loaded pre-generated samples", shape=samples.shape)
        else:
            # Generate initial batch of samples from model
            n_initial = metadata.get("n_initial_samples", 1000)
            seq_length = metadata.get("data", {}).get("seq_length", 24)
            n_features = metadata.get("data", {}).get("n_features", 6)
            samples = cls._generate_from_model(model, n_initial, seq_length, n_features)
            logger.info("Generated initial samples from model", shape=samples.shape)

        return cls(samples=samples, metadata=metadata, model=model, seed=seed)

    @staticmethod
    def _load_model(
        checkpoint_path: Path,
        generator_name: str,
        device: str,
    ) -> Any:
        """Load the trained model from checkpoint.

        This is a dispatch function that calls the appropriate loader
        based on the generator type.
        """
        # Import torch lazily to avoid dependency if not using checkpoints
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load model checkpoints. Install it with: pip install torch"
            )

        model_file = checkpoint_path / "checkpoint.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)

        # Different generators have different model structures
        # For now, we just return the checkpoint dict
        # Full model loading would require importing generator-specific code
        logger.warning(
            "Full model loading not yet implemented for all generators. "
            "Using pre-generated samples mode.",
            generator=generator_name,
        )

        return checkpoint

    @staticmethod
    def _generate_from_model(
        model: Any,  # noqa: ARG004 - used when model generation is implemented
        n_samples: int,
        seq_length: int,
        n_features: int,
    ) -> np.ndarray:
        """Generate samples from the loaded model.

        This is a placeholder that returns random data.
        Full implementation would use the actual model.
        """
        # Placeholder: return random samples
        # In a full implementation, this would call the model's generate method
        logger.warning("Model generation not implemented, returning random placeholder samples")
        return np.random.randn(n_samples, seq_length, n_features) * 0.01

    @property
    def name(self) -> str:
        """Return the provider name."""
        generator = self._metadata.get("generator", {}).get("name", "learned")
        return f"learned_synthetic_{generator}"

    @property
    def generator_name(self) -> str:
        """Return the name of the underlying generator."""
        return self._metadata.get("generator", {}).get("name", "unknown")

    @property
    def n_samples(self) -> int:
        """Return the number of available samples."""
        return self._n_samples

    @property
    def seq_length(self) -> int:
        """Return the sequence length per sample."""
        return self._seq_length

    @property
    def n_features(self) -> int:
        """Return the number of features per timestep."""
        return self._n_features

    def get_samples(
        self,
        n_samples: int | None = None,
        shuffle: bool = True,
    ) -> np.ndarray:
        """Get raw samples for ML training.

        This is useful for Train on Synthetic, Test on Real (TSTR)
        evaluation workflows.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to return. If None, return all.
        shuffle : bool, default=True
            Whether to shuffle the samples

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples, seq_length, n_features)
        """
        if n_samples is None:
            n_samples = self._n_samples

        if n_samples > self._n_samples:
            logger.warning(
                f"Requested {n_samples} samples but only {self._n_samples} available. "
                "Returning all available samples."
            )
            n_samples = self._n_samples

        if shuffle:
            indices = self._rng.choice(self._n_samples, size=n_samples, replace=False)
            return self._samples[indices]
        else:
            return self._samples[:n_samples]

    def generate_samples(
        self,
        n_samples: int,
        seq_length: int | None = None,
    ) -> np.ndarray:
        """Generate new samples using the loaded model.

        This only works if the provider was created from a checkpoint.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        seq_length : int, optional
            Sequence length. If None, uses the default from training.

        Returns
        -------
        np.ndarray
            Generated samples

        Raises
        ------
        RuntimeError
            If no model is loaded (sample-only mode)
        """
        if self._model is None:
            raise RuntimeError(
                "Cannot generate new samples without a loaded model. "
                "Use from_checkpoint() to load a model, or use get_samples() "
                "to access pre-generated samples."
            )

        if seq_length is None:
            seq_length = self._seq_length

        return self._generate_from_model(self._model, n_samples, seq_length, self._n_features)

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Generate OHLCV data from learned samples.

        This method:
        1. Generates timestamps for the requested date range
        2. Samples return sequences and concatenates them
        3. Converts returns to prices using shared utilities
        4. Generates realistic OHLC and volume

        Parameters
        ----------
        symbol : str
            Symbol name (used as seed modifier)
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        frequency : str
            Data frequency

        Returns
        -------
        pl.DataFrame
            Synthetic OHLCV data
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Generate timestamps
        timestamps = generate_timestamps(start_dt, end_dt, frequency)
        n_steps = len(timestamps)

        if n_steps == 0:
            return self._create_empty_dataframe()

        logger.info(
            f"Generating {n_steps} bars from learned samples",
            symbol=symbol,
            generator=self.generator_name,
            frequency=frequency,
        )

        # Modify RNG state based on symbol for reproducibility
        if self.seed is not None:
            symbol_hash = hash(symbol) % (2**31)
            self._rng = np.random.default_rng(self.seed + symbol_hash)

        # Calculate how many sequences we need
        n_sequences_needed = (n_steps // self._seq_length) + 1

        # Sample and concatenate sequences
        # For simplicity, we use the first feature column as log returns
        sampled = self.get_samples(n_samples=min(n_sequences_needed, self._n_samples))
        returns_all = sampled[:, :, 0].flatten()  # Use first feature as returns

        # Truncate to exact length needed
        returns = returns_all[:n_steps]

        # If we don't have enough data, extend with more samples
        while len(returns) < n_steps:
            more_samples = self.get_samples(n_samples=1)
            returns = np.concatenate([returns, more_samples[0, :, 0]])
        returns = returns[:n_steps]

        # Convert returns to prices
        closes = returns_to_prices(returns, base_price=100.0, log_returns=True)

        # Calculate daily volatility from returns for OHLC generation
        bars_per_day = get_bars_per_day(frequency)
        realized_vol = np.std(returns) * np.sqrt(self.TRADING_DAYS * bars_per_day)
        daily_vol = realized_vol / np.sqrt(self.TRADING_DAYS * bars_per_day)

        # Generate OHLC using shared utility
        opens, highs, lows = generate_ohlc_from_close(closes, daily_vol, rng=self._rng)

        # Generate volume using shared utility
        volume = generate_volume(returns, base_volume=1_000_000, rng=self._rng)

        # Create DataFrame
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            }
        )

        # Ensure correct types
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

        return df

    def get_available_symbols(self) -> list[str]:
        """Return suggested synthetic symbol names."""
        generator = self.generator_name.upper()
        return [
            f"SYNTH_{generator}",
            f"SYNTH_{generator}_1",
            f"SYNTH_{generator}_2",
        ]

    def get_metadata(self) -> dict[str, Any]:
        """Return the metadata dictionary."""
        return self._metadata.copy()

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the random number generator.

        Parameters
        ----------
        seed : int, optional
            New seed value. If None, uses original seed.
        """
        self.seed = seed if seed is not None else self.seed
        self._rng = np.random.default_rng(self.seed)

"""Provider for learned generative models (TimeGAN, Sig-CWGAN, etc.).

This provider loads pre-generated samples from trained generative models
from Chapter 6 notebooks and provides a consistent API for generating
synthetic OHLCV data.

Key differences from SyntheticProvider:
- SyntheticProvider: Parameterize stochastic model → Generate on the fly
- LearnedSyntheticProvider: Load pre-generated samples from a training artifact

Usage examples:

    # From pre-generated samples (faster, no model needed)
    provider = LearnedSyntheticProvider.from_samples(
        DATA_DIR / "synthetic/timegan_sequences.npy"
    )
    df = provider.fetch_ohlcv("SYNTH_TIMEGAN", "2024-01-01", "2024-12-31", "daily")

    # From a training artifact containing pre-generated samples
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
    CalendarMode,
    create_rng,
    derive_symbol_seed,
    generate_ohlc_from_close,
    generate_timestamps,
    generate_volume,
    returns_to_prices,
)

logger = structlog.get_logger()


class LearnedSyntheticProvider(BaseProvider):
    """Provider for learned generative models.

    This provider wraps samples produced by trained generative models
    (TimeGAN, Sig-CWGAN, Tail-GAN, TransFusion, GT-GAN, etc.) from Chapter 6
    and provides a consistent API for generating synthetic OHLCV data.

    Samples can be loaded directly from an ``.npy`` file or from a training
    artifact directory containing ``samples.npy`` and ``metadata.json``.

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
    calendar_mode : {"equity", "continuous"}, default="equity"
        Calendar used to generate output timestamps.

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
    MAX_SAMPLE_FILE_BYTES: ClassVar[int] = 4 * 1024**3
    MAX_METADATA_FILE_BYTES: ClassVar[int] = 1024**2
    SUPPORTED_SAMPLE_DTYPES: ClassVar[frozenset[str]] = frozenset({"float32", "float64"})

    @classmethod
    def _validate_samples(cls, samples: np.ndarray) -> None:
        """Validate the non-executable sample tensor contract."""
        if samples.ndim != 3:
            raise ValueError(
                f"Samples must have shape (n_samples, seq_length, n_features), got {samples.shape}"
            )
        if any(dimension <= 0 for dimension in samples.shape):
            raise ValueError(f"Sample dimensions must be positive, got {samples.shape}")
        if samples.dtype.name not in cls.SUPPORTED_SAMPLE_DTYPES:
            raise ValueError(
                f"Unsupported sample dtype {samples.dtype}; expected float32 or float64"
            )
        if samples.nbytes > cls.MAX_SAMPLE_FILE_BYTES:
            raise ValueError(
                f"Sample tensor exceeds size limit of {cls.MAX_SAMPLE_FILE_BYTES} bytes"
            )

    @classmethod
    def _load_samples_file(cls, samples_path: Path) -> np.ndarray:
        """Load a bounded NumPy array without enabling pickle deserialization."""
        if not samples_path.is_file():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
        if samples_path.stat().st_size > cls.MAX_SAMPLE_FILE_BYTES:
            raise ValueError(f"Sample file exceeds size limit of {cls.MAX_SAMPLE_FILE_BYTES} bytes")
        try:
            samples = np.load(samples_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError(f"Failed to load safe NumPy sample array: {samples_path}") from error
        if not isinstance(samples, np.ndarray):
            close = getattr(samples, "close", None)
            if callable(close):
                close()
            raise ValueError("Sample artifact must contain one NumPy array, not an archive")
        cls._validate_samples(samples)
        return samples

    @classmethod
    def _load_metadata_file(cls, metadata_path: Path) -> dict[str, Any]:
        """Load a bounded JSON object used only as descriptive metadata."""
        if metadata_path.stat().st_size > cls.MAX_METADATA_FILE_BYTES:
            raise ValueError(
                f"Metadata file exceeds size limit of {cls.MAX_METADATA_FILE_BYTES} bytes"
            )
        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Failed to load metadata JSON: {metadata_path}") from error
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        return metadata

    def __init__(
        self,
        samples: np.ndarray,
        metadata: dict[str, Any] | None = None,
        model: Any = None,
        seed: int | None = None,
        rate_limit: tuple[int, float] | None = None,
        calendar_mode: CalendarMode = "equity",
    ) -> None:
        """Initialize provider with samples or model.

        Note: Prefer using class methods from_samples() or from_checkpoint()
        instead of calling __init__ directly.
        """
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")
        self._validate_samples(samples)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")

        self._samples = samples
        self._metadata = metadata or {}
        self._model = model
        self.seed = seed
        if calendar_mode not in {"equity", "continuous"}:
            raise ValueError("calendar_mode must be 'equity' or 'continuous'")
        self.calendar_mode = calendar_mode
        self._rng = create_rng(seed)

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
        calendar_mode: CalendarMode = "equity",
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
        calendar_mode : {"equity", "continuous"}, default="equity"
            Calendar used to generate output timestamps.

        Returns
        -------
        LearnedSyntheticProvider
            Configured provider instance
        """
        samples_path = Path(samples_path)

        samples = cls._load_samples_file(samples_path)
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
            metadata = cls._load_metadata_file(Path(metadata_path))
            logger.info(f"Loaded metadata from {metadata_path}")

        return cls(
            samples=samples,
            metadata=metadata,
            model=None,
            seed=seed,
            calendar_mode=calendar_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",  # noqa: ARG003 - retained for beta API compatibility
        seed: int | None = None,
        calendar_mode: CalendarMode = "equity",
    ) -> LearnedSyntheticProvider:
        """Create a provider from a safe training-artifact directory.

        Model checkpoints are not deserialized. The directory must contain
        pre-generated samples in NumPy's non-pickle format.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to an artifact directory containing ``metadata.json`` and
            ``samples.npy``. A ``checkpoint.pt`` file may be present but is
            ignored because PyTorch checkpoints can execute pickle payloads.
        device : str, default="cpu"
            Retained for compatibility and ignored.
        seed : int, optional
            Random seed for reproducibility
        calendar_mode : {"equity", "continuous"}, default="equity"
            Calendar used to generate output timestamps.

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

        metadata = cls._load_metadata_file(metadata_file)

        samples_file = checkpoint_path / "samples.npy"
        if not samples_file.is_file():
            raise FileNotFoundError(
                f"Pre-generated samples file not found: {samples_file}. "
                "Executable model checkpoints are not supported."
            )

        samples = cls._load_samples_file(samples_file)
        logger.info("Loaded pre-generated samples", path=samples_file, shape=samples.shape)
        return cls(
            samples=samples,
            metadata=metadata,
            model=None,
            seed=seed,
            calendar_mode=calendar_mode,
        )

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
        timestamps = generate_timestamps(start_dt, end_dt, frequency, self.calendar_mode)
        n_steps = len(timestamps)

        if n_steps == 0:
            return self._create_empty_dataframe()

        logger.info(
            f"Generating {n_steps} bars from learned samples",
            symbol=symbol,
            generator=self.generator_name,
            frequency=frequency,
        )

        # Modify RNG state based on symbol for reproducibility.
        if self.seed is not None:
            self._rng = create_rng(derive_symbol_seed(self.seed, symbol))

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

        bar_volatility = float(np.std(returns))

        # Generate OHLC using shared utility
        opens, highs, lows = generate_ohlc_from_close(closes, bar_volatility, rng=self._rng)

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
        self._rng = create_rng(self.seed)

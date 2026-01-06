"""Registry for discovering and loading trained synthetic data generators.

This module provides a unified interface to discover and load all available
synthetic data generators from the checkpoints directory.

Usage:
    from ml4t.data.synthetic import SyntheticRegistry

    registry = SyntheticRegistry()
    print(registry.list_generators())  # ['timegan', 'tailgan', 'sigcwgan', ...]

    # Load samples directly
    samples = registry.load_samples("timegan")

    # Get a provider for OHLCV generation
    provider = registry.get_provider("timegan")
    df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

    # Get metadata
    meta = registry.get_metadata("timegan")
    print(meta["generator"]["paper"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from ml4t.data.providers.learned_synthetic import LearnedSyntheticProvider

logger = structlog.get_logger()


class SyntheticRegistry:
    """Registry for discovering and loading trained generators.

    The registry scans the checkpoints directory structure:
    ```
    DATA_DIR/synthetic/checkpoints/
    ├── timegan/
    │   └── etf_returns/
    │       ├── checkpoint.pt
    │       ├── metadata.json
    │       └── samples.npy
    ├── tailgan/
    │   └── etf_tail_risk/
    │       └── ...
    └── ...
    ```

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory. If None, uses DATA_DIR from ml4t_code.
    checkpoints_subdir : str, default="synthetic/checkpoints"
        Subdirectory containing checkpoints relative to data_dir.

    Examples
    --------
    >>> registry = SyntheticRegistry()
    >>> registry.list_generators()
    ['timegan', 'tailgan', 'sigcwgan', 'transfusion', 'gtgan']

    >>> samples = registry.load_samples("timegan")
    >>> samples.shape
    (1000, 24, 6)
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        checkpoints_subdir: str = "synthetic/checkpoints",
    ) -> None:
        """Initialize the registry."""
        if data_dir is None:
            # Import here to avoid circular imports
            from ml4t_code import DATA_DIR

            data_dir = DATA_DIR

        self._data_dir = Path(data_dir)
        self._checkpoints_dir = self._data_dir / checkpoints_subdir
        self._cache: dict[str, dict[str, Any]] = {}

        logger.info(
            "Initialized SyntheticRegistry",
            checkpoints_dir=str(self._checkpoints_dir),
        )

    @property
    def checkpoints_dir(self) -> Path:
        """Return the checkpoints directory path."""
        return self._checkpoints_dir

    def _discover_generators(self) -> dict[str, dict[str, Path]]:
        """Discover all available generators and their experiments.

        Returns
        -------
        dict
            Mapping of generator_name -> {experiment_name -> checkpoint_path}
        """
        generators: dict[str, dict[str, Path]] = {}

        if not self._checkpoints_dir.exists():
            logger.warning(
                "Checkpoints directory does not exist",
                path=str(self._checkpoints_dir),
            )
            return generators

        for generator_dir in self._checkpoints_dir.iterdir():
            if not generator_dir.is_dir():
                continue

            generator_name = generator_dir.name
            generators[generator_name] = {}

            for experiment_dir in generator_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue

                # Check if this is a valid checkpoint (has metadata.json)
                if (experiment_dir / "metadata.json").exists():
                    generators[generator_name][experiment_dir.name] = experiment_dir

        return generators

    def list_generators(self) -> list[str]:
        """List all available generator types.

        Returns
        -------
        list[str]
            Names of available generators (e.g., ['timegan', 'tailgan', ...])
        """
        generators = self._discover_generators()
        return sorted(generators.keys())

    def list_experiments(self, generator: str) -> list[str]:
        """List all experiments for a specific generator.

        Parameters
        ----------
        generator : str
            Generator name (e.g., 'timegan')

        Returns
        -------
        list[str]
            Names of available experiments
        """
        generators = self._discover_generators()
        if generator not in generators:
            return []
        return sorted(generators[generator].keys())

    def list_all(self) -> dict[str, list[str]]:
        """List all generators and their experiments.

        Returns
        -------
        dict
            Mapping of generator_name -> list of experiment names
        """
        generators = self._discover_generators()
        return {name: sorted(exps.keys()) for name, exps in generators.items()}

    def _get_checkpoint_path(
        self,
        generator: str,
        experiment: str | None = None,
    ) -> Path:
        """Get the path to a specific checkpoint.

        If experiment is None, returns the first available experiment.
        """
        generators = self._discover_generators()

        if generator not in generators:
            available = list(generators.keys())
            raise ValueError(f"Generator '{generator}' not found. Available: {available}")

        experiments = generators[generator]
        if not experiments:
            raise ValueError(f"No experiments found for generator '{generator}'")

        if experiment is None:
            # Use first available experiment
            experiment = sorted(experiments.keys())[0]
            logger.info(
                f"Using default experiment for {generator}",
                experiment=experiment,
            )

        if experiment not in experiments:
            available = list(experiments.keys())
            raise ValueError(
                f"Experiment '{experiment}' not found for '{generator}'. Available: {available}"
            )

        return experiments[experiment]

    def get_metadata(
        self,
        generator: str,
        experiment: str | None = None,
    ) -> dict[str, Any]:
        """Get metadata for a generator/experiment.

        Parameters
        ----------
        generator : str
            Generator name (e.g., 'timegan')
        experiment : str, optional
            Experiment name. If None, uses the first available.

        Returns
        -------
        dict
            Metadata including generator info, training config, and eval metrics
        """
        checkpoint_path = self._get_checkpoint_path(generator, experiment)

        # Use cache
        cache_key = str(checkpoint_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        self._cache[cache_key] = metadata
        return metadata

    def load_samples(
        self,
        generator: str,
        experiment: str | None = None,
        n_samples: int | None = None,
        shuffle: bool = False,
    ) -> np.ndarray:
        """Load pre-generated samples for a generator.

        Parameters
        ----------
        generator : str
            Generator name (e.g., 'timegan')
        experiment : str, optional
            Experiment name. If None, uses the first available.
        n_samples : int, optional
            Number of samples to return. If None, returns all.
        shuffle : bool, default=False
            Whether to shuffle samples before returning.

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples, seq_length, n_features)
        """
        checkpoint_path = self._get_checkpoint_path(generator, experiment)
        samples_file = checkpoint_path / "samples.npy"

        if not samples_file.exists():
            raise FileNotFoundError(
                f"Samples file not found: {samples_file}. "
                "The checkpoint may not have pre-generated samples."
            )

        samples = np.load(samples_file)
        logger.info(f"Loaded samples for {generator}", shape=samples.shape)

        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(samples)

        if n_samples is not None and n_samples < len(samples):
            samples = samples[:n_samples]

        return samples

    def get_provider(
        self,
        generator: str,
        experiment: str | None = None,
        seed: int | None = None,
    ) -> LearnedSyntheticProvider:
        """Get a LearnedSyntheticProvider for OHLCV generation.

        Parameters
        ----------
        generator : str
            Generator name (e.g., 'timegan')
        experiment : str, optional
            Experiment name. If None, uses the first available.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        LearnedSyntheticProvider
            Provider configured with the loaded samples/checkpoint
        """
        # Import here to avoid circular imports
        from ml4t.data.providers.learned_synthetic import LearnedSyntheticProvider

        checkpoint_path = self._get_checkpoint_path(generator, experiment)

        # Prefer loading from samples (faster)
        samples_file = checkpoint_path / "samples.npy"
        if samples_file.exists():
            return LearnedSyntheticProvider.from_samples(
                samples_path=samples_file,
                metadata_path=checkpoint_path / "metadata.json",
                seed=seed,
            )
        else:
            # Fall back to checkpoint loading
            return LearnedSyntheticProvider.from_checkpoint(
                checkpoint_path=checkpoint_path,
                seed=seed,
            )

    def get_info(self, generator: str, experiment: str | None = None) -> str:
        """Get a human-readable summary of a generator.

        Parameters
        ----------
        generator : str
            Generator name
        experiment : str, optional
            Experiment name

        Returns
        -------
        str
            Formatted summary string
        """
        metadata = self.get_metadata(generator, experiment)
        checkpoint_path = self._get_checkpoint_path(generator, experiment)

        gen_info = metadata.get("generator", {})
        data_info = metadata.get("data", {})
        eval_info = metadata.get("evaluation", {})

        lines = [
            f"Generator: {gen_info.get('name', generator)}",
            f"Version: {gen_info.get('version', 'unknown')}",
            f"Paper: {gen_info.get('paper', 'N/A')}",
            "",
            "Data:",
            f"  Source: {data_info.get('source', 'unknown')}",
            f"  Symbols: {data_info.get('symbols', [])}",
            f"  Samples: {data_info.get('n_samples', 'unknown')}",
            f"  Sequence Length: {data_info.get('seq_length', 'unknown')}",
            f"  Features: {data_info.get('n_features', 'unknown')}",
            "",
            "Evaluation:",
        ]

        for key, value in eval_info.items():
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.extend(["", f"Path: {checkpoint_path}"])

        return "\n".join(lines)

    def summary(self) -> str:
        """Get a summary of all available generators.

        Returns
        -------
        str
            Formatted summary table
        """
        generators = self._discover_generators()

        if not generators:
            return "No generators found. Run Ch6 notebooks to generate checkpoints."

        lines = ["Available Synthetic Data Generators", "=" * 40, ""]

        for gen_name in sorted(generators.keys()):
            experiments = generators[gen_name]
            for exp_name, _exp_path in sorted(experiments.items()):
                try:
                    metadata = self.get_metadata(gen_name, exp_name)
                    n_samples = metadata.get("data", {}).get("n_samples", "?")
                    eval_info = metadata.get("evaluation", {})

                    # Get a representative metric
                    metric_str = ""
                    if "tstr_ratio" in eval_info:
                        metric_str = f"TSTR={eval_info['tstr_ratio']:.2f}"
                    elif "tstr_mse_ratio" in eval_info:
                        metric_str = f"TSTR={eval_info['tstr_mse_ratio']:.2f}"

                    lines.append(f"{gen_name}/{exp_name}: {n_samples} samples {metric_str}")
                except Exception as e:
                    lines.append(f"{gen_name}/{exp_name}: Error loading metadata ({e})")

        return "\n".join(lines)

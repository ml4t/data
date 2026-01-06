"""Synthetic data generation module.

This module provides utilities for generating synthetic financial data,
including both classical stochastic models and learned generative models.

Components:
- ohlcv_utils: Shared utilities for generating realistic OHLCV data
- registry: Discovery and loading of trained generators

Usage:
    from ml4t.data.synthetic import SyntheticRegistry

    registry = SyntheticRegistry()
    print(registry.list_generators())  # ['timegan', 'tailgan', ...]

    # Load samples for ML training
    samples = registry.load_samples("timegan")

    # Get provider for OHLCV generation
    provider = registry.get_provider("timegan")
"""

from ml4t.data.synthetic.ohlcv_utils import (
    generate_ohlc_from_close,
    generate_timestamps,
    generate_volume,
    get_bars_per_day,
    returns_to_prices,
)
from ml4t.data.synthetic.registry import SyntheticRegistry

__all__ = [
    # OHLCV utilities
    "generate_ohlc_from_close",
    "generate_timestamps",
    "generate_volume",
    "get_bars_per_day",
    "returns_to_prices",
    # Registry
    "SyntheticRegistry",
]

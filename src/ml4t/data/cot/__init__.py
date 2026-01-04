"""CFTC Commitment of Traders (COT) data fetcher.

This module provides functionality for fetching COT reports from the CFTC,
which provide weekly positioning data broken down by trader type (commercials,
speculators, etc.).

COT data is free and provides valuable sentiment/positioning signals for ML features:
- Net positioning by trader type (hedge funds vs commercials)
- Week-over-week positioning changes
- Extreme positioning detection (z-scores)
- Commercial vs speculative divergence

Example:
    from ml4t.data.cot import COTFetcher

    fetcher = COTFetcher()
    df = fetcher.fetch_product('ES', start_year=2020, end_year=2024)
"""

from ml4t.data.cot.fetcher import (
    PRODUCT_MAPPINGS,
    COTConfig,
    COTFetcher,
    ProductMapping,
    load_cot_config,
)
from ml4t.data.cot.workflow import (
    combine_cot_ohlcv,
    combine_cot_ohlcv_pit,
    create_cot_features,
    load_combined_futures_data,
)

__all__ = [
    "COTFetcher",
    "COTConfig",
    "load_cot_config",
    "PRODUCT_MAPPINGS",
    "ProductMapping",
    "combine_cot_ohlcv",
    "combine_cot_ohlcv_pit",
    "create_cot_features",
    "load_combined_futures_data",
]

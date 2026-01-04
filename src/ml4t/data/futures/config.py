"""Configuration classes and utilities for futures data downloading.

This module provides configuration dataclasses, enums, and YAML loading
utilities for the futures downloader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FuturesCategory(str, Enum):
    """Futures product categories."""

    EQUITY_INDEX = "equity_index"
    RATES = "rates"
    FX = "fx"
    ENERGY = "energy"
    METALS = "metals"
    GRAINS = "grains"
    FERTILIZER = "fertilizer"
    SOFTS = "softs"
    LIVESTOCK = "livestock"
    DAIRY = "dairy"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"


# Default product universe for ML4T book (25 products)
DEFAULT_PRODUCTS: dict[FuturesCategory, list[str]] = {
    FuturesCategory.EQUITY_INDEX: ["ES", "NQ", "RTY", "YM"],
    FuturesCategory.RATES: ["ZT", "ZF", "ZN", "ZB", "SR3"],
    FuturesCategory.FX: ["6E", "6J", "6B", "6C", "6A"],
    FuturesCategory.ENERGY: ["CL", "NG", "RB"],
    FuturesCategory.METALS: ["GC", "SI", "HG"],
    FuturesCategory.GRAINS: ["ZC", "ZW", "ZS"],
    FuturesCategory.LIVESTOCK: ["LE"],
    FuturesCategory.CRYPTO: ["BTC"],
}


@dataclass
class FuturesDownloadConfig:
    """Configuration for futures data download.

    Attributes:
        products: List of product symbols to download, or dict by category.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        storage_path: Base path for storing downloaded data.
        dataset: Databento dataset (default: GLBX.MDP3 for CME).
        schemas: List of schemas to download.
        api_key: Databento API key (defaults to DATABENTO_API_KEY env var).
    """

    products: list[str] | dict[FuturesCategory, list[str]] = field(
        default_factory=lambda: DEFAULT_PRODUCTS
    )
    start: str = "2016-01-01"
    end: str = "2025-12-13"
    storage_path: str | Path = "~/ml4t-data/futures"
    dataset: str = "GLBX.MDP3"
    schemas: list[str] = field(default_factory=lambda: ["ohlcv-1d", "definition", "statistics"])
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.storage_path = Path(self.storage_path).expanduser()

    def get_product_list(self) -> list[str]:
        """Get flat list of all products."""
        if isinstance(self.products, dict):
            return [p for products in self.products.values() for p in products]
        return self.products

    def get_products_by_category(self) -> dict[FuturesCategory, list[str]]:
        """Get products organized by category."""
        if isinstance(self.products, dict):
            return self.products
        # If flat list, return as unknown category
        return {FuturesCategory.EQUITY_INDEX: self.products}


@dataclass
class DownloadProgress:
    """Track download progress for resume capability."""

    completed_products: set[str] = field(default_factory=set)
    failed_products: dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0

    def mark_complete(self, product: str, bytes_downloaded: int = 0) -> None:
        """Mark a product as successfully downloaded."""
        self.completed_products.add(product)
        self.total_bytes += bytes_downloaded
        if product in self.failed_products:
            del self.failed_products[product]

    def mark_failed(self, product: str, error: str) -> None:
        """Mark a product as failed."""
        self.failed_products[product] = error

    def is_complete(self, product: str) -> bool:
        """Check if product was already downloaded."""
        return product in self.completed_products


@dataclass
class DefinitionsConfig:
    """Configuration for definition snapshot downloads.

    Attributes:
        strategy: Download strategy ("yearly_snapshots", "latest_only", "full").
        snapshot_dates: List of dates to download for yearly_snapshots strategy.
        output_file: Filename for merged definitions output.
    """

    strategy: str = "yearly_snapshots"
    snapshot_dates: list[str] = field(
        default_factory=lambda: [
            "2016-01-04",
            "2017-01-03",
            "2018-01-02",
            "2019-01-02",
            "2020-01-02",
            "2021-01-04",
            "2022-01-03",
            "2023-01-03",
            "2024-01-02",
            "2025-01-02",
        ]
    )
    output_file: str = "definitions_merged.parquet"


def load_yaml_config(yaml_path: str | Path) -> FuturesDownloadConfig:
    """Load download configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        FuturesDownloadConfig instance.
    """
    import yaml

    yaml_path = Path(yaml_path).expanduser()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    futures_config = data.get("futures", data)

    # Handle products (can be list or dict by category)
    products = futures_config.get("products", DEFAULT_PRODUCTS)
    if isinstance(products, dict):
        # Convert string keys to FuturesCategory enum
        products = {FuturesCategory(k) if isinstance(k, str) else k: v for k, v in products.items()}

    return FuturesDownloadConfig(
        products=products,
        start=futures_config.get("start", "2016-01-01"),
        end=futures_config.get("end", "2025-12-13"),
        storage_path=futures_config.get("storage_path", "~/ml4t-data/futures"),
        dataset=futures_config.get("dataset", "GLBX.MDP3"),
        schemas=futures_config.get("schemas", ["ohlcv-1d", "definition", "statistics"]),
        api_key=futures_config.get("api_key"),
    )


def load_definitions_config(yaml_path: str | Path) -> tuple[list[str], Path, DefinitionsConfig]:
    """Load definitions configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Tuple of (products list, storage path, DefinitionsConfig).
    """
    import yaml

    yaml_path = Path(yaml_path).expanduser()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    futures_config = data.get("futures", data)

    # Get products as flat list
    products = futures_config.get("products", DEFAULT_PRODUCTS)
    if isinstance(products, dict):
        products = [p for cat_products in products.values() for p in cat_products]

    storage_path = Path(futures_config.get("storage_path", "~/ml4t-data/futures")).expanduser()

    # Get definitions config
    defn_config = futures_config.get("definitions", {})
    default_dates = [
        "2016-01-04",
        "2017-01-03",
        "2018-01-02",
        "2019-01-02",
        "2020-01-02",
        "2021-01-04",
        "2022-01-03",
        "2023-01-03",
        "2024-01-02",
        "2025-01-02",
    ]
    definitions_config = DefinitionsConfig(
        strategy=defn_config.get("strategy", "yearly_snapshots"),
        snapshot_dates=defn_config.get("snapshot_dates", default_dates),
        output_file=defn_config.get("output_file", "definitions_merged.parquet"),
    )

    return products, storage_path, definitions_config

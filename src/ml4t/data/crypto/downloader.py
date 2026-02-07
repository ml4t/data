"""ML4T Crypto Data Downloader for Book Readers.

This module provides a unified interface for downloading and managing
cryptocurrency data from Binance Public Data for the funding rate
arbitrage case study.

Features:
- Download premium index data for perpetual futures
- No API key required (public S3 bucket)
- No geographic restrictions
- Config-driven approach for reproducibility

Usage:
    from ml4t.data.crypto.downloader import CryptoDataManager

    # Initialize with config
    manager = CryptoDataManager.from_config("configs/ml4t_etfs.yaml")

    # Download premium index
    manager.download_premium_index()

    # Load data
    premium = manager.load_premium_index()

References:
    - Binance Public Data: https://data.binance.vision
    - GitHub: https://github.com/binance/binance-public-data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

from ml4t.data.storage.data_profile import generate_profile, get_profile_path, save_profile

logger = structlog.get_logger(__name__)


@dataclass
class CryptoConfig:
    """Configuration for crypto data download."""

    provider: str = "binance_public"
    market: str = "futures"
    start: str = "2021-01-01"
    end: str = "2025-12-31"
    interval: str = "8h"  # Funding rate settlement interval
    storage_path: Path = field(default_factory=lambda: Path.home() / "ml4t-data" / "crypto")
    symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    generate_profile: bool = True  # Generate column-level statistics on save

    def __post_init__(self):
        self.storage_path = Path(self.storage_path).expanduser()

    def get_all_symbols(self) -> list[str]:
        """Get flat list of all symbols across categories."""
        symbols = []
        for category_data in self.symbols.values():
            if isinstance(category_data, dict) and "symbols" in category_data:
                symbols.extend(category_data["symbols"])
        return sorted(set(symbols))

    def get_categories(self) -> dict[str, list[str]]:
        """Get symbols organized by category."""
        categories = {}
        for category, category_data in self.symbols.items():
            if isinstance(category_data, dict) and "symbols" in category_data:
                categories[category] = category_data["symbols"]
        return categories


class CryptoDataManager:
    """Manages crypto data download and storage for ML4T book.

    This class provides a simple interface for book readers to:
    1. Download premium index data from Binance
    2. Load data for analysis

    Data is stored as:
        {storage_path}/premium_index.parquet
        {storage_path}/premium_index/symbol={SYMBOL}/data.parquet
    """

    def __init__(self, config: CryptoConfig):
        """Initialize the crypto data manager.

        Args:
            config: Configuration object with symbols and storage path
        """
        self.config = config
        self._provider = None

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> CryptoDataManager:
        """Create manager from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized CryptoDataManager
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        crypto_config = raw_config.get("crypto", {})

        config = CryptoConfig(
            provider=crypto_config.get("provider", "binance_public"),
            market=crypto_config.get("market", "futures"),
            start=crypto_config.get("start", "2021-01-01"),
            end=crypto_config.get("end", "2025-12-31"),
            interval=crypto_config.get("interval", "8h"),
            storage_path=Path(crypto_config.get("storage_path", "~/ml4t-data/crypto")).expanduser(),
            symbols=crypto_config.get("symbols", {}),
        )

        return cls(config)

    @property
    def provider(self):
        """Lazily initialize Binance Public provider."""
        if self._provider is None:
            from ml4t.data.providers.binance_public import BinancePublicProvider

            self._provider = BinancePublicProvider(market=self.config.market)

        return self._provider

    def download_premium_index(self, symbols: list[str] | None = None) -> pl.DataFrame:
        """Download premium index data for perpetual futures.

        The premium index measures the basis between perpetual and spot prices,
        and is the primary driver of funding rates.

        Premium Index = (Perpetual Price - Spot Price) / Spot Price
        - High premium → Crowded longs → Expected underperformance
        - Low/negative premium → Crowded shorts → Expected outperformance

        Args:
            symbols: List of symbols to download (default: all from config)

        Returns:
            DataFrame with premium index data
        """
        if symbols is None:
            symbols = self.config.get_all_symbols()

        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT"]  # Default if no config
            logger.warning("No symbols in config, using defaults: BTCUSDT, ETHUSDT")

        logger.info(
            "Downloading premium index data",
            symbols=len(symbols),
            start=self.config.start,
            end=self.config.end,
            interval=self.config.interval,
        )

        # Use parallel multi-symbol download (3-10x faster)
        df = self.provider.fetch_premium_index_multi_parallel(
            symbols=symbols,
            start=self.config.start,
            end=self.config.end,
            interval=self.config.interval,
        )

        if df.is_empty():
            logger.warning("No premium index data downloaded")
            return df

        # Save combined file
        self._save_premium_index(df)

        # Save by symbol (partitioned)
        self._save_by_symbol(df)

        logger.info(
            "Download complete",
            symbols=df["symbol"].n_unique(),
            rows=len(df),
        )

        return df

    def _save_premium_index(self, df: pl.DataFrame) -> None:
        """Save combined premium index data."""
        output_file = self.config.storage_path / "premium_index.parquet"
        df.write_parquet(output_file)
        logger.info(f"Saved: {output_file}")

        # Generate profile if enabled
        if self.config.generate_profile:
            profile = generate_profile(df, source="CryptoDataManager")
            profile_path = get_profile_path(output_file)
            save_profile(profile, profile_path)
            logger.info(f"Saved data profile: {profile_path}")

    def _save_by_symbol(self, df: pl.DataFrame) -> None:
        """Save premium index data partitioned by symbol."""
        partition_dir = self.config.storage_path / "premium_index"
        partition_dir.mkdir(exist_ok=True)

        for symbol in df["symbol"].unique().to_list():
            symbol_df = df.filter(pl.col("symbol") == symbol)
            symbol_dir = partition_dir / f"symbol={symbol}"
            symbol_dir.mkdir(exist_ok=True)
            output_file = symbol_dir / "data.parquet"
            symbol_df.write_parquet(output_file)

    def load_premium_index(self, symbols: list[str] | None = None) -> pl.DataFrame:
        """Load premium index data.

        Args:
            symbols: List of symbols to load (default: all available)

        Returns:
            DataFrame with premium index data
        """
        # Try combined file first
        combined_file = self.config.storage_path / "premium_index.parquet"

        if combined_file.exists():
            df = pl.read_parquet(combined_file)

            if symbols:
                df = df.filter(pl.col("symbol").is_in(symbols))

            return df.sort(["symbol", "timestamp"])

        # Fall back to partitioned files
        partition_dir = self.config.storage_path / "premium_index"

        if not partition_dir.exists():
            logger.warning("No premium index data found. Run download_premium_index() first.")
            return pl.DataFrame()

        dfs = []
        for symbol_dir in partition_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                symbol = symbol_dir.name.replace("symbol=", "")

                if symbols and symbol not in symbols:
                    continue

                data_file = symbol_dir / "data.parquet"
                if data_file.exists():
                    dfs.append(pl.read_parquet(data_file))

        if not dfs:
            return pl.DataFrame()

        return pl.concat(dfs).sort(["symbol", "timestamp"])

    def load_symbol(self, symbol: str) -> pl.DataFrame:
        """Load premium index data for a single symbol.

        Args:
            symbol: Symbol to load (e.g., "BTCUSDT")

        Returns:
            DataFrame with premium index data
        """
        return self.load_premium_index(symbols=[symbol])

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with downloaded data.

        Returns:
            List of symbols with data files
        """
        partition_dir = self.config.storage_path / "premium_index"

        if not partition_dir.exists():
            return []

        symbols = []
        for path in partition_dir.iterdir():
            if path.is_dir() and path.name.startswith("symbol="):
                symbol = path.name.replace("symbol=", "")
                symbols.append(symbol)

        return sorted(symbols)

    def get_data_summary(self) -> pl.DataFrame:
        """Get summary of available data.

        Returns:
            DataFrame with symbol, start_date, end_date, row_count
        """
        df = self.load_premium_index()

        if df.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "start_date": pl.Datetime,
                    "end_date": pl.Datetime,
                    "row_count": pl.UInt64,
                }
            )

        return (
            df.group_by("symbol")
            .agg(
                pl.col("timestamp").min().alias("start_date"),
                pl.col("timestamp").max().alias("end_date"),
                pl.len().alias("row_count"),
            )
            .sort("symbol")
        )


def main():
    """CLI entry point for crypto data download."""
    import argparse

    parser = argparse.ArgumentParser(description="ML4T Crypto Data Downloader")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ml4t_etfs.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to download (default: all from config)",
    )

    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = package_dir / args.config

    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        return 1

    # Initialize manager
    manager = CryptoDataManager.from_config(config_path)

    # Download premium index
    df = manager.download_premium_index(symbols=args.symbols)
    print(f"Downloaded premium index: {len(df)} rows for {df['symbol'].n_unique()} symbols")

    # Show summary
    print("\nData summary:")
    print(manager.get_data_summary())

    return 0


if __name__ == "__main__":
    exit(main())

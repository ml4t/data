"""ML4T ETF Data Downloader for Book Readers.

This module provides a unified interface for downloading and managing ETF data
from Yahoo Finance. It is designed to be simple for book readers to use while
providing robust data management features.

Features:
- Download OHLCV daily data for 50 diversified ETFs
- Hive-partitioned storage for efficient updates
- Incremental updates to extend existing data
- Config-driven approach for reproducibility

Usage:
    from ml4t.data.etfs.downloader import ETFDataManager

    # Initialize with config
    manager = ETFDataManager.from_config("configs/ml4t_etfs.yaml")

    # Download all data (initial or update)
    manager.download_all()

    # Load data for analysis
    spy_data = manager.load_ohlcv("SPY")
    all_data = manager.load_all()

References:
    - Yahoo Finance: https://finance.yahoo.com
    - yfinance library: https://github.com/ranaroussi/yfinance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

logger = structlog.get_logger(__name__)

# Standard OHLCV schema for consistency
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass
class ETFConfig:
    """Configuration for ETF data download."""

    provider: str = "yahoo"
    start: str = "2005-01-01"
    end: str = "2025-12-31"
    frequency: str = "daily"
    storage_path: Path = field(default_factory=lambda: Path.home() / "ml4t-data" / "etfs")
    tickers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.storage_path = Path(self.storage_path).expanduser()

    def get_all_symbols(self) -> list[str]:
        """Get flat list of all symbols across categories."""
        symbols = []
        for category_data in self.tickers.values():
            if isinstance(category_data, dict) and "symbols" in category_data:
                symbols.extend(category_data["symbols"])
        return sorted(set(symbols))

    def get_categories(self) -> dict[str, list[str]]:
        """Get symbols organized by category."""
        categories = {}
        for category, category_data in self.tickers.items():
            if isinstance(category_data, dict) and "symbols" in category_data:
                categories[category] = category_data["symbols"]
        return categories


class ETFDataManager:
    """Manages ETF data download and storage for ML4T book.

    This class provides a simple interface for book readers to:
    1. Download initial historical data
    2. Update data incrementally
    3. Load data for analysis

    Data is stored in Hive-partitioned format:
        {storage_path}/ohlcv_1d/ticker={SYMBOL}/data.parquet
    """

    def __init__(self, config: ETFConfig):
        """Initialize the ETF data manager.

        Args:
            config: Configuration object with tickers, dates, and storage path
        """
        self.config = config
        self._provider = None  # Lazy initialization

        # Ensure storage directories exist
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> ETFDataManager:
        """Create manager from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized ETFDataManager
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        etf_config = raw_config.get("etfs", {})

        config = ETFConfig(
            provider=etf_config.get("provider", "yahoo"),
            start=etf_config.get("start", "2005-01-01"),
            end=etf_config.get("end", "2025-12-31"),
            frequency=etf_config.get("frequency", "daily"),
            storage_path=Path(etf_config.get("storage_path", "~/ml4t-data/etfs")).expanduser(),
            tickers=etf_config.get("tickers", {}),
        )

        return cls(config)

    @property
    def provider(self):
        """Lazily initialize Yahoo Finance provider."""
        if self._provider is None:
            from ml4t.data.providers.yahoo import YahooFinanceProvider

            self._provider = YahooFinanceProvider(enable_progress=True)
        return self._provider

    def download_all(self, force: bool = False) -> dict[str, int]:  # noqa: ARG002
        """Download all ETF data.

        Args:
            force: If True, re-download even if data exists

        Returns:
            Dictionary of symbol -> row count
        """
        symbols = self.config.get_all_symbols()

        logger.info(
            "Starting ETF data download",
            symbols=len(symbols),
            start=self.config.start,
            end=self.config.end,
        )

        # Use batch download for efficiency
        df = self.provider.fetch_batch_ohlcv(
            symbols=symbols,
            start=self.config.start,
            end=self.config.end,
            frequency=self.config.frequency,
            chunk_size=50,
            delay_seconds=1.0,
        )

        if df.is_empty():
            logger.warning("No data downloaded")
            return {}

        # Save by ticker (Hive-partitioned)
        stats = self._save_by_ticker(df)

        # Also save combined file for convenience
        self._save_combined(df)

        # Save metadata
        self._save_metadata()

        logger.info(
            "Download complete",
            symbols_downloaded=len(stats),
            total_rows=sum(stats.values()),
        )

        return stats

    def update(self) -> dict[str, int]:
        """Update existing data with latest available.

        Detects the last date in existing data and downloads
        from there to the configured end date.

        Returns:
            Dictionary of symbol -> new rows added
        """
        symbols = self.config.get_all_symbols()
        stats = {}

        for symbol in symbols:
            try:
                # Load existing data to find last date
                existing = self.load_ohlcv(symbol)

                if existing.is_empty():
                    # No existing data, do full download for this symbol
                    start_date = self.config.start
                else:
                    # Start from day after last existing date
                    last_date = existing["timestamp"].max()
                    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

                # Check if we need to update
                end_date = datetime.strptime(self.config.end, "%Y-%m-%d")
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")

                if start_dt > end_date:
                    logger.debug(f"{symbol} is up to date")
                    stats[symbol] = 0
                    continue

                # Download new data
                from ml4t.data.providers.yahoo import YahooFinanceProvider

                provider = YahooFinanceProvider()
                new_data = provider.fetch_ohlcv(
                    symbol=symbol,
                    start=start_date,
                    end=self.config.end,
                    frequency=self.config.frequency,
                )

                if new_data.is_empty():
                    stats[symbol] = 0
                    continue

                # Append to existing file
                if not existing.is_empty():
                    # Ensure datetime precision matches (existing may be ns, new data is μs)
                    existing = existing.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
                    new_data = new_data.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
                    combined = pl.concat([existing, new_data])
                    combined = combined.unique(subset=["timestamp"], maintain_order=True)
                    combined = combined.sort("timestamp")
                else:
                    combined = new_data

                # Save
                self._save_ticker(combined, symbol)
                stats[symbol] = len(new_data)

                logger.info(f"Updated {symbol}: +{len(new_data)} rows")

            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")
                stats[symbol] = 0

        # Regenerate combined file
        self._regenerate_combined()

        return stats

    def _save_by_ticker(self, df: pl.DataFrame) -> dict[str, int]:
        """Save data partitioned by ticker.

        Args:
            df: DataFrame with symbol column

        Returns:
            Dictionary of symbol -> row count
        """
        ohlcv_dir = self.config.storage_path / "ohlcv_1d"
        ohlcv_dir.mkdir(parents=True, exist_ok=True)

        stats = {}
        for symbol in df["symbol"].unique().to_list():
            ticker_df = df.filter(pl.col("symbol") == symbol)

            # Save to Hive partition
            partition_dir = ohlcv_dir / f"ticker={symbol}"
            partition_dir.mkdir(exist_ok=True)
            output_file = partition_dir / "data.parquet"

            # Remove symbol column for storage (it's in the partition name)
            ticker_df = ticker_df.drop("symbol")
            ticker_df.write_parquet(output_file)

            stats[symbol] = len(ticker_df)
            logger.debug(f"Saved {symbol}: {len(ticker_df)} rows")

        return stats

    def _save_ticker(self, df: pl.DataFrame, symbol: str) -> None:
        """Save data for a single ticker.

        Args:
            df: DataFrame for the ticker
            symbol: Ticker symbol
        """
        ohlcv_dir = self.config.storage_path / "ohlcv_1d"
        partition_dir = ohlcv_dir / f"ticker={symbol}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_file = partition_dir / "data.parquet"

        # Remove symbol column if present
        if "symbol" in df.columns:
            df = df.drop("symbol")

        df.write_parquet(output_file)

    def _save_combined(self, df: pl.DataFrame) -> None:
        """Save combined file for convenience."""
        output_file = self.config.storage_path / "etf_universe.parquet"
        df.write_parquet(output_file)
        logger.info(f"Saved combined file: {output_file} ({len(df):,} rows)")

    def _regenerate_combined(self) -> None:
        """Regenerate combined file from individual ticker files."""
        all_data = self.load_all()
        if not all_data.is_empty():
            self._save_combined(all_data)

    def _save_metadata(self) -> None:
        """Save metadata file describing the universe."""
        import json

        metadata = {
            "name": "ML4T 50-ETF Universe",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": "Curated ETF universe for ML4T 3rd Edition case study",
            "config": {
                "start": self.config.start,
                "end": self.config.end,
                "frequency": self.config.frequency,
            },
            "categories": self.config.get_categories(),
            "total_tickers": len(self.config.get_all_symbols()),
        }

        metadata_file = self.config.storage_path / "etf_universe_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")

    def load_ohlcv(self, symbol: str) -> pl.DataFrame:
        """Load OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., "SPY")

        Returns:
            DataFrame with OHLCV data (columns: timestamp, symbol, open, high, low, close, volume)
        """
        symbol = symbol.upper()
        data_file = self.config.storage_path / "ohlcv_1d" / f"ticker={symbol}" / "data.parquet"

        if not data_file.exists():
            logger.warning(f"No data found for {symbol}")
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        df = pl.read_parquet(data_file)

        # Handle different column naming conventions
        # (date vs timestamp, ticker vs symbol)
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename({"date": "timestamp"})

        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename({"ticker": "symbol"})

        # Add symbol column if not present (from partition name)
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        return df.sort("timestamp")

    def load_symbols(self, symbols: list[str]) -> pl.DataFrame:
        """Load OHLCV data for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Combined DataFrame with symbol column
        """
        dfs = []
        for symbol in symbols:
            df = self.load_ohlcv(symbol)
            if not df.is_empty():
                dfs.append(df)

        if not dfs:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        return pl.concat(dfs).sort(["symbol", "timestamp"])

    def load_all(self) -> pl.DataFrame:
        """Load all ETF data.

        Returns:
            Combined DataFrame with all tickers (columns: timestamp, symbol, open, high, low, close, volume)
        """
        # Try combined file first
        combined_file = self.config.storage_path / "etf_universe.parquet"
        if combined_file.exists():
            df = pl.read_parquet(combined_file)

            # Handle different column naming conventions
            if "date" in df.columns and "timestamp" not in df.columns:
                df = df.rename({"date": "timestamp"})
            if "ticker" in df.columns and "symbol" not in df.columns:
                df = df.rename({"ticker": "symbol"})

            return df.sort(["symbol", "timestamp"])

        # Fall back to loading individual files
        return self.load_symbols(self.config.get_all_symbols())

    def load_category(self, category: str) -> pl.DataFrame:
        """Load OHLCV data for a category.

        Args:
            category: Category name (e.g., "us_equity_broad", "fixed_income")

        Returns:
            DataFrame with tickers from that category
        """
        categories = self.config.get_categories()
        if category not in categories:
            available = list(categories.keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available}")

        symbols = categories[category]
        return self.load_symbols(symbols)

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with downloaded data.

        Returns:
            List of ticker symbols that have data files
        """
        ohlcv_dir = self.config.storage_path / "ohlcv_1d"
        if not ohlcv_dir.exists():
            return []

        symbols = []
        for path in ohlcv_dir.iterdir():
            if path.is_dir() and path.name.startswith("ticker="):
                symbol = path.name.replace("ticker=", "")
                symbols.append(symbol)

        return sorted(symbols)

    def get_data_summary(self) -> pl.DataFrame:
        """Get summary of available data.

        Returns:
            DataFrame with symbol, start_date, end_date, row_count
        """
        symbols = self.get_available_symbols()

        summaries = []
        for symbol in symbols:
            df = self.load_ohlcv(symbol)
            if not df.is_empty():
                summaries.append(
                    {
                        "symbol": symbol,
                        "start_date": df["timestamp"].min(),
                        "end_date": df["timestamp"].max(),
                        "row_count": len(df),
                    }
                )

        if not summaries:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "start_date": pl.Datetime,
                    "end_date": pl.Datetime,
                    "row_count": pl.UInt64,
                }
            )

        return pl.DataFrame(summaries).sort("symbol")


def main():
    """CLI entry point for ETF data download."""
    import argparse

    parser = argparse.ArgumentParser(description="ML4T ETF Data Downloader")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ml4t_etfs.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing data instead of full download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )

    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to package
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = package_dir / args.config

    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        return 1

    # Initialize manager
    manager = ETFDataManager.from_config(config_path)

    # Download or update
    if args.update:
        stats = manager.update()
        print(f"Updated {len(stats)} symbols")
    else:
        stats = manager.download_all(force=args.force)
        print(f"Downloaded {len(stats)} symbols, {sum(stats.values()):,} total rows")

    return 0


if __name__ == "__main__":
    exit(main())

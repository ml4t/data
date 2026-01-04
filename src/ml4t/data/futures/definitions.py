"""Futures contract definitions downloader.

This module provides the DefinitionsDownloader for downloading futures contract
definitions using yearly snapshots from Databento.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog
from databento import Historical
from databento.common.error import BentoClientError, BentoServerError

from .config import DefinitionsConfig

logger = structlog.get_logger(__name__)


class DefinitionsDownloader:
    """Download futures contract definitions using yearly snapshots.

    Instead of downloading full daily history (millions of rows), this
    downloads one snapshot per year and merges them to get expiration
    dates for all contracts.

    Example:
        >>> downloader = DefinitionsDownloader(
        ...     products=["ES", "CL", "GC"],
        ...     storage_path="~/ml4t-data/futures",
        ... )
        >>> downloader.download_snapshots()
        >>> df = downloader.get_merged_definitions()
    """

    def __init__(
        self,
        products: list[str],
        storage_path: str | Path,
        config: DefinitionsConfig | None = None,
        dataset: str = "GLBX.MDP3",
        api_key: str | None = None,
    ) -> None:
        """Initialize definitions downloader.

        Args:
            products: List of product symbols.
            storage_path: Base path for storage.
            config: Definitions configuration.
            dataset: Databento dataset.
            api_key: Databento API key.
        """
        self.products = products
        self.storage_path = Path(storage_path).expanduser()
        self.config = config or DefinitionsConfig()
        self.dataset = dataset
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")

        if not self.api_key:
            raise ValueError("Databento API key required")

        self.client = Historical(self.api_key)
        self._ensure_dirs()

        logger.info(
            "Initialized DefinitionsDownloader",
            products=len(products),
            strategy=self.config.strategy,
            snapshot_dates=len(self.config.snapshot_dates),
        )

    def _ensure_dirs(self) -> None:
        """Create storage directories."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "definition").mkdir(exist_ok=True)
        (self.storage_path / "definition_snapshots").mkdir(exist_ok=True)

    def download_snapshot(
        self, date: str, product: str, max_retries: int = 3
    ) -> pl.DataFrame | None:
        """Download definitions for a single product on a single date.

        Args:
            date: Date string (YYYY-MM-DD).
            product: Product symbol.
            max_retries: Maximum retry attempts on server errors.

        Returns:
            DataFrame with definitions, or None if no data.
        """
        import time
        from datetime import timedelta

        parent_symbol = f"{product}.FUT"

        # Calculate next day for end date (Databento requires end > start)
        start_dt = datetime.strptime(date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # Download one day only with retry logic
        for attempt in range(max_retries):
            try:
                data = self.client.timeseries.get_range(
                    dataset=self.dataset,
                    symbols=parent_symbol,
                    stype_in="parent",
                    schema="definition",
                    start=date,
                    end=end_date,
                )

                df_pandas = data.to_df().reset_index()
                df = pl.from_pandas(df_pandas)

                # Add product and snapshot date
                df = df.with_columns(
                    [
                        pl.lit(product).alias("product"),
                        pl.lit(date).alias("snapshot_date"),
                    ]
                )

                # Filter out spreads
                if "symbol" in df.columns:
                    df = df.filter(~pl.col("symbol").str.contains("-"))

                logger.debug(
                    "Downloaded definitions snapshot",
                    product=product,
                    date=date,
                    rows=df.height,
                )

                return df

            except BentoClientError as e:
                if "not found" in str(e).lower() or "no data" in str(e).lower():
                    logger.debug("No definitions for date", product=product, date=date)
                    return None
                raise

            except BentoServerError as e:
                # Retry on server errors (502, 503, etc.)
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.warning(
                        "Server error, retrying",
                        product=product,
                        date=date,
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                        error=str(e),
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Server error after retries",
                        product=product,
                        date=date,
                        error=str(e),
                    )
                    raise

        return None  # Should not reach here

    def download_snapshots(
        self,
        products: list[str] | None = None,
        dates: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Download definition snapshots for all products and dates.

        Args:
            products: Products to download (default: all configured).
            dates: Dates to download (default: all snapshot_dates).

        Returns:
            Dict mapping product to merged definitions DataFrame.
        """
        products = products or self.products
        dates = dates or self.config.snapshot_dates

        logger.info(
            "Downloading definition snapshots",
            products=len(products),
            dates=len(dates),
            total_requests=len(products) * len(dates),
        )

        results: dict[str, list[pl.DataFrame]] = {p: [] for p in products}

        for product in products:
            for date in dates:
                df = self.download_snapshot(date, product)
                if df is not None and df.height > 0:
                    results[product].append(df)

            # Save raw snapshots
            if results[product]:
                merged = pl.concat(results[product])
                snapshot_file = (
                    self.storage_path / "definition_snapshots" / f"{product}_snapshots.parquet"
                )
                merged.write_parquet(snapshot_file)
                logger.info(
                    "Saved snapshots",
                    product=product,
                    total_rows=merged.height,
                    file=str(snapshot_file),
                )

        # Merge and save per-product definitions
        merged_results = {}
        for product, dfs in results.items():
            if dfs:
                merged = self._merge_definitions(pl.concat(dfs))
                merged_results[product] = merged

                # Save to definition directory
                product_dir = self.storage_path / "definition" / f"product={product}"
                product_dir.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(product_dir / "definition.parquet")

        logger.info(
            "Completed definition downloads",
            products_with_data=len(merged_results),
        )

        return merged_results

    def _merge_definitions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merge and deduplicate definitions.

        Keeps one row per symbol with the latest snapshot data.

        Args:
            df: Raw definitions from multiple snapshots.

        Returns:
            Deduplicated definitions with one row per symbol.
        """
        if df.height == 0:
            return df

        # Sort by snapshot date descending (latest first)
        if "snapshot_date" in df.columns:
            df = df.sort("snapshot_date", descending=True)

        # Keep first occurrence of each symbol (latest snapshot)
        if "symbol" in df.columns:
            df = df.unique(subset=["symbol"], keep="first")

        # Sort by expiration for readability
        if "expiration" in df.columns:
            df = df.sort("expiration")

        return df

    def get_merged_definitions(self) -> pl.DataFrame:
        """Load and merge all downloaded definitions.

        Returns:
            Single DataFrame with all contract definitions.
        """
        definition_dir = self.storage_path / "definition"
        if not definition_dir.exists():
            return pl.DataFrame()

        dfs = []
        for product_dir in definition_dir.iterdir():
            if product_dir.is_dir() and product_dir.name.startswith("product="):
                parquet_file = product_dir / "definition.parquet"
                if parquet_file.exists():
                    df = pl.read_parquet(parquet_file)
                    dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        merged = pl.concat(dfs)

        # Save merged output
        output_file = self.storage_path / self.config.output_file
        merged.write_parquet(output_file)
        logger.info("Saved merged definitions", file=str(output_file), rows=merged.height)

        return merged

    def check_coverage(self) -> dict[str, dict]:
        """Check definition coverage against OHLCV data.

        Returns:
            Coverage report by product.
        """
        ohlcv_dir = self.storage_path / "ohlcv_1d"
        defn_dir = self.storage_path / "definition"

        report = {}

        for product in self.products:
            ohlcv_path = ohlcv_dir / f"product={product}"
            defn_path = defn_dir / f"product={product}" / "definition.parquet"

            ohlcv_symbols = set()
            defn_symbols = set()

            if (ohlcv_path / "ohlcv_1d.parquet").exists():
                ohlcv = pl.read_parquet(ohlcv_path / "ohlcv_1d.parquet")
                ohlcv_symbols = set(ohlcv["symbol"].unique().to_list())

            if defn_path.exists():
                defn = pl.read_parquet(defn_path)
                if "symbol" in defn.columns:
                    defn_symbols = set(defn["symbol"].unique().to_list())

            missing = ohlcv_symbols - defn_symbols
            coverage = (
                len(ohlcv_symbols & defn_symbols) / len(ohlcv_symbols) * 100 if ohlcv_symbols else 0
            )

            report[product] = {
                "ohlcv_contracts": len(ohlcv_symbols),
                "defn_contracts": len(defn_symbols),
                "missing": len(missing),
                "coverage_pct": round(coverage, 1),
                "status": "complete" if coverage == 100 else "incomplete",
            }

        return report

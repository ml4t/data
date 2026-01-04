"""ML4T Futures Data Downloader for Book Readers.

This module provides a unified interface for downloading and managing CME futures data
from Databento. It is designed to be simple for book readers to use while providing
robust data management features.

Features:
- Download OHLCV daily data for 70 diversified CME futures products
- Yearly Hive-partitioned storage for efficient updates
- Contract definitions with expiration dates
- Incremental updates to extend existing data

Usage:
    from ml4t.data.futures.book_downloader import FuturesDataManager

    # Initialize with config
    manager = FuturesDataManager.from_config("configs/ml4t_futures.yaml")

    # Download all data (initial or update)
    manager.download_all()

    # Load data for analysis
    es_data = manager.load_ohlcv("ES")
    es_definitions = manager.load_definitions("ES")

References:
    - CME Globex Products: https://www.cmegroup.com/markets/products.html
    - Databento Documentation: https://databento.com/docs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

logger = structlog.get_logger(__name__)

# Standard OHLCV schema for consistency across downloads and updates
OHLCV_SCHEMA = {
    "ts_event": pl.Datetime(time_unit="ns", time_zone="UTC"),
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.UInt64,
    "product": pl.Utf8,
}

# Key definition columns for contract information
DEFINITION_COLUMNS = [
    "symbol",
    "expiration",
    "activation",
    "min_price_increment",
    "contract_multiplier",
    "currency",
    "exchange",
    "asset",
    "instrument_class",
    "unit_of_measure",
    "unit_of_measure_qty",
]


@dataclass
class FuturesConfig:
    """Configuration for futures data download."""

    dataset: str = "GLBX.MDP3"
    start: str = "2016-01-01"
    end: str = "2025-12-31"
    storage_path: Path = field(default_factory=lambda: Path.home() / "ml4t-data" / "futures")
    products: dict[str, list[str]] = field(default_factory=dict)
    definition_dates: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.storage_path = Path(self.storage_path).expanduser()

    def get_all_products(self) -> list[str]:
        """Get flat list of all products across categories."""
        products = []
        for category_products in self.products.values():
            products.extend(category_products)
        return sorted(set(products))


class FuturesDataManager:
    """Manages CME futures data download and storage for ML4T book.

    This class provides a simple interface for book readers to:
    1. Download initial historical data
    2. Update data incrementally
    3. Load data for analysis

    Data is stored in Hive-partitioned format:
        {storage_path}/ohlcv_1d/product={PRODUCT}/year={YYYY}/data.parquet
        {storage_path}/definitions/product={PRODUCT}/definitions.parquet
    """

    def __init__(self, config: FuturesConfig):
        """Initialize the futures data manager.

        Args:
            config: Configuration object with products, dates, and storage path
        """
        self.config = config
        self.client = None  # Lazy initialization

        # Ensure storage directories exist
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> FuturesDataManager:
        """Create manager from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized FuturesDataManager
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        futures_config = raw_config.get("futures", {})

        config = FuturesConfig(
            dataset=futures_config.get("dataset", "GLBX.MDP3"),
            start=futures_config.get("start", "2016-01-01"),
            end=futures_config.get("end", "2025-12-31"),
            storage_path=Path(
                futures_config.get("storage_path", "~/ml4t-data/futures")
            ).expanduser(),
            products=futures_config.get("products", {}),
            definition_dates=futures_config.get("definitions", {}).get("snapshot_dates", []),
        )

        return cls(config)

    def _get_client(self):
        """Lazily initialize Databento client."""
        if self.client is None:
            import databento as db

            api_key = os.environ.get("DATABENTO_API_KEY")
            if not api_key:
                raise ValueError(
                    "DATABENTO_API_KEY environment variable not set.\n"
                    "Get your free API key at https://databento.com/"
                )
            self.client = db.Historical(api_key)
        return self.client

    def _get_ohlcv_path(self, product: str, year: int) -> Path:
        """Get path for OHLCV data file."""
        return (
            self.config.storage_path
            / "ohlcv_1d"
            / f"product={product}"
            / f"year={year}"
            / "data.parquet"
        )

    def _get_definitions_path(self, product: str) -> Path:
        """Get path for definitions file."""
        return (
            self.config.storage_path / "definitions" / f"product={product}" / "definitions.parquet"
        )

    def download_product_ohlcv(
        self,
        product: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Download OHLCV data for a single product.

        Args:
            product: CME product symbol (e.g., "ES", "CL")
            start_date: Start date (YYYY-MM-DD), defaults to config start
            end_date: End date (YYYY-MM-DD), defaults to config end

        Returns:
            Dict with download statistics
        """
        client = self._get_client()

        start = start_date or self.config.start
        end = end_date or self.config.end

        logger.info("Downloading OHLCV", product=product, start=start, end=end)

        try:
            data = client.timeseries.get_range(
                dataset=self.config.dataset,
                symbols=f"{product}.FUT",
                stype_in="parent",
                schema="ohlcv-1d",
                start=start,
                end=end,
            )

            df = data.to_df()

            if len(df) == 0:
                logger.warning("No data returned", product=product)
                return {"product": product, "rows": 0, "years": 0}

            # Convert to polars
            df_pl = pl.from_pandas(df.reset_index())

            # Add product column
            df_pl = df_pl.with_columns(pl.lit(product).alias("product"))

            # Filter out spread contracts (contain - or :)
            df_pl = df_pl.filter(
                ~pl.col("symbol").str.contains("-") & ~pl.col("symbol").str.contains(":")
            )

            # Select and rename columns to standard schema
            df_pl = df_pl.select(
                [
                    pl.col("ts_event"),
                    pl.col("symbol"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.UInt64),
                    pl.col("product"),
                ]
            )

            # Partition by year and save
            df_pl = df_pl.with_columns(pl.col("ts_event").dt.year().alias("_year"))

            years_written = 0
            total_rows = 0

            for year in df_pl["_year"].unique().sort().to_list():
                year_df = df_pl.filter(pl.col("_year") == year).drop("_year")

                output_path = self._get_ohlcv_path(product, year)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file exists and merge
                if output_path.exists():
                    existing = pl.read_parquet(output_path)
                    year_df = pl.concat([existing, year_df])
                    year_df = year_df.unique(subset=["ts_event", "symbol"], keep="last")

                year_df = year_df.sort(["ts_event", "symbol"])
                year_df.write_parquet(output_path)

                years_written += 1
                total_rows += len(year_df)

            logger.info("Saved OHLCV", product=product, rows=total_rows, years=years_written)

            return {"product": product, "rows": total_rows, "years": years_written}

        except Exception as e:
            logger.error("Failed to download", product=product, error=str(e))
            return {"product": product, "error": str(e)}

    def download_product_definitions(self, product: str) -> dict[str, Any]:
        """Download definition snapshots for a single product.

        Uses yearly snapshots to efficiently capture contract definitions
        without downloading the full daily history.

        Args:
            product: CME product symbol

        Returns:
            Dict with download statistics
        """
        client = self._get_client()

        if not self.config.definition_dates:
            logger.warning("No definition dates configured")
            return {"product": product, "error": "No definition dates"}

        logger.info(
            "Downloading definitions", product=product, snapshots=len(self.config.definition_dates)
        )

        all_defs = []

        for date in self.config.definition_dates:
            try:
                start_dt = datetime.strptime(date, "%Y-%m-%d")
                end_dt = start_dt + timedelta(days=1)
                end_date = end_dt.strftime("%Y-%m-%d")

                data = client.timeseries.get_range(
                    dataset=self.config.dataset,
                    symbols=f"{product}.FUT",
                    stype_in="parent",
                    schema="definition",
                    start=date,
                    end=end_date,
                )

                df = data.to_df()

                if len(df) > 0:
                    df_pl = pl.from_pandas(df.reset_index())
                    df_pl = df_pl.with_columns(
                        [
                            pl.lit(product).alias("product"),
                            pl.lit(date).alias("snapshot_date"),
                        ]
                    )

                    # Filter spreads
                    if "symbol" in df_pl.columns:
                        df_pl = df_pl.filter(
                            ~pl.col("symbol").str.contains("-")
                            & ~pl.col("symbol").str.contains(":")
                        )

                    all_defs.append(df_pl)

            except Exception as e:
                # Skip dates where product didn't exist (e.g., BTC before 2017)
                if "422" not in str(e):
                    logger.debug("Snapshot failed", product=product, date=date, error=str(e)[:50])

        if not all_defs:
            logger.warning("No definitions found", product=product)
            return {"product": product, "contracts": 0}

        # Combine and deduplicate
        combined = pl.concat(all_defs, how="diagonal")

        # Select key columns (handle missing columns gracefully)
        available_cols = [c for c in DEFINITION_COLUMNS if c in combined.columns]
        available_cols = ["product", "snapshot_date"] + available_cols

        combined = combined.select(available_cols)

        # Deduplicate by symbol, keeping latest snapshot
        combined = combined.sort("snapshot_date", descending=True).unique("symbol")
        combined = combined.sort("expiration" if "expiration" in combined.columns else "symbol")

        # Save
        output_path = self._get_definitions_path(product)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)

        logger.info("Saved definitions", product=product, contracts=len(combined))

        return {"product": product, "contracts": len(combined)}

    def download_all(self, include_definitions: bool = True, parallel: int = 1) -> dict[str, Any]:
        """Download all configured products.

        Args:
            include_definitions: Whether to also download definitions
            parallel: Number of parallel downloads (1 = sequential)

        Returns:
            Summary statistics
        """
        products = self.config.get_all_products()
        logger.info("Starting download", products=len(products), parallel=parallel)

        ohlcv_results = []
        defn_results = []

        for i, product in enumerate(products, 1):
            logger.info(f"[{i}/{len(products)}] {product}")

            # Download OHLCV
            result = self.download_product_ohlcv(product)
            ohlcv_results.append(result)

            # Download definitions
            if include_definitions:
                defn_result = self.download_product_definitions(product)
                defn_results.append(defn_result)

        # Summary
        successful_ohlcv = sum(1 for r in ohlcv_results if "error" not in r)
        successful_defn = sum(1 for r in defn_results if "error" not in r)

        return {
            "ohlcv": {
                "successful": successful_ohlcv,
                "failed": len(ohlcv_results) - successful_ohlcv,
                "total_products": len(products),
            },
            "definitions": {
                "successful": successful_defn,
                "failed": len(defn_results) - successful_defn,
            },
        }

    def update(self, end_date: str | None = None) -> dict[str, Any]:
        """Update existing data to latest available date.

        Finds the latest date in existing data and downloads only new data.

        Args:
            end_date: End date for update (default: today)

        Returns:
            Update statistics
        """
        products = self.config.get_all_products()

        # Find latest date across all products
        latest_dates = {}
        for product in products:
            latest = self._get_latest_date(product)
            if latest:
                latest_dates[product] = latest

        if not latest_dates:
            logger.info("No existing data found, running full download")
            return self.download_all()

        # Get overall latest
        overall_latest = max(latest_dates.values())

        # Calculate update range
        start_date = (overall_latest + timedelta(days=1)).strftime("%Y-%m-%d")
        end = end_date or datetime.now(UTC).strftime("%Y-%m-%d")

        if start_date >= end:
            logger.info("Already up to date", latest=overall_latest.strftime("%Y-%m-%d"))
            return {"status": "up_to_date", "latest": overall_latest.strftime("%Y-%m-%d")}

        logger.info("Updating", from_date=start_date, to_date=end)

        results = []
        for product in products:
            result = self.download_product_ohlcv(product, start_date=start_date, end_date=end)
            results.append(result)

        successful = sum(1 for r in results if "error" not in r and r.get("rows", 0) > 0)

        return {
            "status": "updated",
            "from_date": start_date,
            "to_date": end,
            "products_updated": successful,
        }

    def _get_latest_date(self, product: str) -> datetime | None:
        """Get the latest date in existing data for a product."""
        ohlcv_dir = self.config.storage_path / "ohlcv_1d" / f"product={product}"

        if not ohlcv_dir.exists():
            return None

        # Find latest year directory
        year_dirs = sorted(ohlcv_dir.glob("year=*"), reverse=True)

        for year_dir in year_dirs:
            data_file = year_dir / "data.parquet"
            if data_file.exists():
                df = pl.read_parquet(data_file)
                if len(df) > 0:
                    latest = df["ts_event"].max()
                    if latest:
                        return latest.replace(tzinfo=UTC)

        return None

    def load_ohlcv(
        self, product: str, start: str | None = None, end: str | None = None
    ) -> pl.DataFrame:
        """Load OHLCV data for a product.

        Args:
            product: CME product symbol
            start: Optional start date filter (YYYY-MM-DD)
            end: Optional end date filter (YYYY-MM-DD)

        Returns:
            Polars DataFrame with OHLCV data
        """
        ohlcv_dir = self.config.storage_path / "ohlcv_1d" / f"product={product}"

        if not ohlcv_dir.exists():
            raise FileNotFoundError(f"No data found for {product}")

        # Read all year files
        files = list(ohlcv_dir.glob("year=*/data.parquet"))

        if not files:
            raise FileNotFoundError(f"No data files found for {product}")

        df = pl.read_parquet(files)

        # Apply date filters
        if start:
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
            df = df.filter(pl.col("ts_event") >= start_dt)

        if end:
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
            df = df.filter(pl.col("ts_event") <= end_dt)

        return df.sort(["ts_event", "symbol"])

    def load_definitions(self, product: str) -> pl.DataFrame:
        """Load contract definitions for a product.

        Args:
            product: CME product symbol

        Returns:
            Polars DataFrame with contract definitions
        """
        defn_path = self._get_definitions_path(product)

        if not defn_path.exists():
            raise FileNotFoundError(f"No definitions found for {product}")

        return pl.read_parquet(defn_path)

    def list_products(self) -> dict[str, list[str]]:
        """List all configured products by category."""
        return self.config.products

    def get_data_summary(self) -> pl.DataFrame:
        """Get summary of downloaded data.

        Returns:
            DataFrame with product, date range, row count, etc.
        """
        summaries = []

        for product in self.config.get_all_products():
            ohlcv_dir = self.config.storage_path / "ohlcv_1d" / f"product={product}"

            if not ohlcv_dir.exists():
                summaries.append(
                    {
                        "product": product,
                        "status": "not_downloaded",
                        "rows": 0,
                        "contracts": 0,
                        "start_date": None,
                        "end_date": None,
                    }
                )
                continue

            try:
                df = self.load_ohlcv(product)
                defn_df = (
                    self.load_definitions(product)
                    if self._get_definitions_path(product).exists()
                    else None
                )

                summaries.append(
                    {
                        "product": product,
                        "status": "available",
                        "rows": len(df),
                        "contracts": df["symbol"].n_unique(),
                        "definitions": len(defn_df) if defn_df is not None else 0,
                        "start_date": df["ts_event"].min(),
                        "end_date": df["ts_event"].max(),
                    }
                )
            except Exception as e:
                summaries.append(
                    {
                        "product": product,
                        "status": "error",
                        "error": str(e),
                    }
                )

        return pl.DataFrame(summaries)


def download_futures_data(config_path: str = "configs/ml4t_futures.yaml"):
    """Convenience function to download all futures data.

    This is the main entry point for book readers.

    Args:
        config_path: Path to configuration file
    """
    manager = FuturesDataManager.from_config(config_path)

    print("ML4T Futures Data Download")
    print("=" * 50)
    print(f"Products: {len(manager.config.get_all_products())}")
    print(f"Date range: {manager.config.start} to {manager.config.end}")
    print(f"Storage: {manager.config.storage_path}")
    print()

    result = manager.download_all()

    print()
    print("Download Complete!")
    print(f"  OHLCV: {result['ohlcv']['successful']}/{result['ohlcv']['total_products']} products")
    print(f"  Definitions: {result['definitions']['successful']} products")


def update_futures_data(config_path: str = "configs/ml4t_futures.yaml"):
    """Convenience function to update existing futures data.

    Args:
        config_path: Path to configuration file
    """
    manager = FuturesDataManager.from_config(config_path)

    print("ML4T Futures Data Update")
    print("=" * 50)

    result = manager.update()

    if result.get("status") == "up_to_date":
        print(f"Data already up to date (latest: {result['latest']})")
    else:
        print(f"Updated from {result['from_date']} to {result['to_date']}")
        print(f"Products updated: {result['products_updated']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "update":
        update_futures_data()
    else:
        download_futures_data()

"""Download continuous futures contracts from Databento.

This module downloads pre-rolled continuous contracts using DataBento's
volume-based (.v.) or calendar-based (.c.) symbology, storing in Hive
partitioned format for efficient incremental updates.

Unlike FuturesDownloader which downloads individual contracts for local
roll construction, this downloads ready-to-use continuous contracts.

Key features:
- Downloads continuous contracts directly (.v.0, .v.1, .v.2)
- Hive partitioning by product and year
- Year-by-year downloads for rate limit friendliness
- Retry logic with exponential backoff
- Parallel download support
- Resume capability (skips existing years)

Usage:
    from ml4t.data.futures.continuous_downloader import (
        ContinuousDownloader,
        ContinuousDownloadConfig,
    )

    config = ContinuousDownloadConfig(
        products=["ES", "CL", "GC"],
        start="2011-01-01",
        end="2025-12-31",
        storage_path="./data/futures/continuous",
        tenors=[0, 1, 2],
        schema="ohlcv-1h",
    )
    downloader = ContinuousDownloader(config)
    downloader.download_all()
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import polars as pl
import structlog
import yaml
from databento import Historical
from databento.common.error import BentoClientError, BentoServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ml4t.data.paths import default_ml4t_data_path, resolve_ml4t_data_path

logger = structlog.get_logger(__name__)


@dataclass
class ContinuousDownloadConfig:
    """Configuration for continuous contract downloads.

    Attributes:
        products: List of product symbols to download.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        storage_path: Base path for storing downloaded data.
        dataset: Databento dataset (default: GLBX.MDP3 for CME).
        schema: OHLCV schema (ohlcv-1h, ohlcv-1d, ohlcv-1m).
        roll_type: Roll methodology ("v" for volume, "c" for calendar).
        tenors: List of tenor positions (0=front, 1=second, 2=third).
        api_key: Databento API key (defaults to DATABENTO_API_KEY env var).
    """

    products: list[str] = field(default_factory=list)
    start: str = "2011-01-01"
    end: str = "2025-12-31"
    storage_path: str | Path = field(
        default_factory=lambda: resolve_ml4t_data_path(
            "futures/continuous",
            default_path=default_ml4t_data_path("futures/continuous"),
        )
    )
    dataset: str = "GLBX.MDP3"
    schema: str = "ohlcv-1h"
    roll_type: str = "v"  # "v" = volume-based, "c" = calendar-based
    tenors: list[int] = field(default_factory=lambda: [0, 1, 2])
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.storage_path = Path(self.storage_path).expanduser()


@dataclass
class ContinuousDownloadProgress:
    """Track download progress for resume capability."""

    completed_years: dict[str, set[int]] = field(default_factory=dict)
    failed_years: dict[str, dict[int, str]] = field(default_factory=dict)

    def mark_complete(self, product: str, year: int) -> None:
        """Mark a product-year as successfully downloaded."""
        if product not in self.completed_years:
            self.completed_years[product] = set()
        self.completed_years[product].add(year)

        # Remove from failed if present
        if product in self.failed_years and year in self.failed_years[product]:
            del self.failed_years[product][year]

    def mark_failed(self, product: str, year: int, error: str) -> None:
        """Mark a product-year as failed."""
        if product not in self.failed_years:
            self.failed_years[product] = {}
        self.failed_years[product][year] = error


def load_continuous_config(yaml_path: str | Path) -> ContinuousDownloadConfig:
    """Load download configuration from YAML file.

    Expected YAML structure:
        dataset: "GLBX.MDP3"
        schema: "ohlcv-1h"
        roll_type: "v"
        tenors: [0, 1, 2]
        default_start: "2011-01-01"
        default_end: "2025-12-31"
        products:
          ES:
            name: "E-mini S&P 500"
            start: "2011-01-01"
          CL:
            name: "WTI Crude Oil"
            start: "2011-01-01"

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        ContinuousDownloadConfig instance.
    """
    yaml_path = Path(yaml_path).expanduser()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Extract products as list
    products_data = data.get("products", {})
    products = list(products_data.keys())

    return ContinuousDownloadConfig(
        products=products,
        start=data.get("default_start", "2011-01-01"),
        end=data.get("default_end", "2025-12-31"),
        storage_path=resolve_ml4t_data_path(
            "futures/continuous",
            default_path=default_ml4t_data_path("futures/continuous"),
            configured_path=data.get("storage", {}).get("path"),
            config=data,
            config_dir=yaml_path.parent,
        ),
        dataset=data.get("dataset", "GLBX.MDP3"),
        schema=data.get("schema", "ohlcv-1h"),
        roll_type=data.get("roll_type", "v"),
        tenors=data.get("tenors", [0, 1, 2]),
    )


class ContinuousDownloader:
    """Download continuous futures contracts from Databento.

    Downloads pre-rolled continuous contracts using DataBento's symbology:
    - Volume-based roll: {PRODUCT}.v.{TENOR} (e.g., ES.v.0)
    - Calendar-based roll: {PRODUCT}.c.{TENOR} (e.g., ES.c.0)

    Storage format (Hive partitioned):
        {storage_path}/product={PRODUCT}/year={YEAR}/data.parquet

    Example:
        >>> config = ContinuousDownloadConfig(
        ...     products=["ES", "CL"],
        ...     start="2020-01-01",
        ...     end="2024-12-31",
        ... )
        >>> downloader = ContinuousDownloader(config)
        >>> downloader.download_all()
    """

    # Approximate cost per product-year (3 tenors, hourly)
    COST_PER_PRODUCT_YEAR: ClassVar[float] = 0.06

    def __init__(self, config: ContinuousDownloadConfig) -> None:
        """Initialize downloader with configuration.

        Args:
            config: Download configuration.

        Raises:
            ValueError: If API key is not available.
        """
        self.config = config
        self.api_key = config.api_key or os.getenv("DATABENTO_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Databento API key not provided. "
                "Set DATABENTO_API_KEY environment variable or pass api_key in config."
            )

        self.client = Historical(self.api_key)
        self.progress = ContinuousDownloadProgress()

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized ContinuousDownloader",
            products=len(config.products),
            start=config.start,
            end=config.end,
            schema=config.schema,
            tenors=config.tenors,
        )

    def _get_output_path(self, product: str, year: int) -> Path:
        """Get output path for a product-year partition."""
        return self.config.storage_path / f"product={product}" / f"year={year}" / "data.parquet"

    def _year_exists(self, product: str, year: int) -> bool:
        """Check if product-year data already exists and has rows."""
        path = self._get_output_path(product, year)
        if not path.exists():
            return False
        try:
            df = pl.read_parquet(path)
            return len(df) > 0
        except Exception:
            return False

    def _get_years_range(self) -> list[int]:
        """Get list of years from config date range."""
        start_year = int(self.config.start[:4])
        end_year = int(self.config.end[:4])
        return list(range(start_year, end_year + 1))

    def _get_missing_years(self, product: str) -> list[int]:
        """Get years that need to be downloaded for a product."""
        all_years = self._get_years_range()
        return [y for y in all_years if not self._year_exists(product, y)]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((BentoServerError, ConnectionError, OSError)),
    )
    def _download_product_full(self, product: str) -> pl.DataFrame:
        """Download full date range for a product in ONE API call.

        Args:
            product: Product symbol (e.g., "ES", "CL").

        Returns:
            DataFrame with OHLCV data for all tenors, full date range.

        Raises:
            BentoClientError: On client errors (bad request, auth).
            BentoServerError: On server errors (will retry).
        """
        symbols = [f"{product}.{self.config.roll_type}.{t}" for t in self.config.tenors]

        data = self.client.timeseries.get_range(
            dataset=self.config.dataset,
            symbols=symbols,
            schema=self.config.schema,
            stype_in="continuous",
            start=self.config.start,
            end=self.config.end,
        )

        df = pl.from_pandas(data.to_df().reset_index())
        return df

    def _partition_to_hive(self, df: pl.DataFrame, product: str) -> int:
        """Partition DataFrame by year and save to Hive structure.

        Args:
            df: DataFrame with ts_event column.
            product: Product symbol.

        Returns:
            Number of years written.
        """
        if df.height == 0:
            return 0

        # Extract year from timestamp
        df = df.with_columns(pl.col("ts_event").dt.year().alias("year"))

        years_written = 0
        for year in df["year"].unique().to_list():
            year_df = df.filter(pl.col("year") == year).drop("year")
            output_path = self._get_output_path(product, year)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            year_df.write_parquet(output_path)
            years_written += 1

        return years_written

    def download_product(
        self,
        product: str,
        skip_existing: bool = True,
    ) -> tuple[int, int]:
        """Download full date range for a product in ONE API call, partition locally.

        Args:
            product: Product symbol.
            skip_existing: Skip if product is already complete.

        Returns:
            Tuple of (years_written, 0 on success or 0, 1 on failure).
        """
        missing = self._get_missing_years(product)

        if not missing and skip_existing:
            logger.info("Product already complete", product=product)
            return 0, 0

        logger.info(
            "Downloading product (full range)",
            product=product,
            start=self.config.start,
            end=self.config.end,
        )

        try:
            # ONE API call for full date range
            df = self._download_product_full(product)

            if df.height == 0:
                logger.warning("No data returned", product=product)
                return 0, 0

            # Partition locally to Hive format
            years_written = self._partition_to_hive(df, product)

            logger.info(
                "Downloaded and partitioned",
                product=product,
                rows=df.height,
                years=years_written,
            )

            # Mark all years complete
            for year in self._get_years_range():
                if self._year_exists(product, year):
                    self.progress.mark_complete(product, year)

            return years_written, 0

        except BentoClientError as e:
            error_msg = str(e)
            logger.warning("Client error", product=product, error=error_msg)
            return 0, 1

        except Exception as e:
            error_msg = str(e)
            logger.error("Failed to download", product=product, error=error_msg)
            return 0, 1

    def download_all(
        self,
        skip_existing: bool = True,
        delay_between_products: float = 1.0,
    ) -> ContinuousDownloadProgress:
        """Download all products - ONE API call per product.

        Each product downloads its full date range in a single API call,
        then partitions locally to Hive format by year.

        Args:
            skip_existing: Skip products that are already complete.
            delay_between_products: Delay in seconds between products.

        Returns:
            Download progress with completed and failed products.
        """
        products = self.config.products
        total = len(products)

        # Filter to products needing work
        products_needing_work = [p for p in products if self._get_missing_years(p)]

        if not products_needing_work:
            logger.info("All products complete - nothing to download")
            return self.progress

        logger.info(
            "Starting continuous download",
            products_to_download=len(products_needing_work),
            total_products=total,
        )

        total_years = 0
        total_fail = 0

        for i, product in enumerate(products_needing_work, 1):
            logger.info(
                "Downloading",
                product=product,
                progress=f"{i}/{len(products_needing_work)}",
            )

            years, fail = self.download_product(product, skip_existing=skip_existing)
            total_years += years
            total_fail += fail

            if i < len(products_needing_work) and delay_between_products > 0:
                time.sleep(delay_between_products)

        logger.info(
            "Download complete",
            products_downloaded=len(products_needing_work) - total_fail,
            years_written=total_years,
            failed=total_fail,
        )

        return self.progress

    def download_all_parallel(
        self,
        max_workers: int = 4,
        skip_existing: bool = True,
    ) -> ContinuousDownloadProgress:
        """Download all products in parallel - ONE API call per product.

        Uses ThreadPoolExecutor for concurrent downloads with separate
        client instances per thread. Each product downloads full range
        in one API call, then partitions locally.

        Args:
            max_workers: Maximum concurrent downloads.
            skip_existing: Skip products that are already complete.

        Returns:
            Download progress with completed and failed products.
        """
        products = self.config.products

        # Filter to products needing work
        products_needing_work = [p for p in products if self._get_missing_years(p)]

        if not products_needing_work:
            logger.info("All products complete - nothing to download")
            return self.progress

        logger.info(
            "Starting parallel download",
            products=len(products_needing_work),
            max_workers=max_workers,
        )

        def download_product_safe(product: str) -> tuple[str, int, bool]:
            """Thread-safe download with separate client - ONE API call."""
            thread_client = Historical(self.api_key)

            try:
                symbols = [f"{product}.{self.config.roll_type}.{t}" for t in self.config.tenors]

                data = thread_client.timeseries.get_range(
                    dataset=self.config.dataset,
                    symbols=symbols,
                    schema=self.config.schema,
                    stype_in="continuous",
                    start=self.config.start,
                    end=self.config.end,
                )

                df = pl.from_pandas(data.to_df().reset_index())

                if df.height == 0:
                    return product, 0, True

                # Partition to Hive
                df = df.with_columns(pl.col("ts_event").dt.year().alias("year"))
                years_written = 0

                for year in df["year"].unique().to_list():
                    year_df = df.filter(pl.col("year") == year).drop("year")
                    output_path = self._get_output_path(product, year)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    year_df.write_parquet(output_path)
                    self.progress.mark_complete(product, year)
                    years_written += 1

                return product, years_written, True

            except Exception as e:
                logger.error("Failed", product=product, error=str(e))
                return product, 0, False

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_product_safe, p): p for p in products_needing_work}

            completed = 0
            for future in as_completed(futures):
                product = futures[future]
                try:
                    _, years, success = future.result()
                    completed += 1
                    logger.info(
                        "Completed",
                        product=product,
                        years=years,
                        progress=f"{completed}/{len(products_needing_work)}",
                    )
                except Exception as e:
                    logger.error("Product failed", product=product, error=str(e))

        return self.progress

    def estimate_cost(self) -> dict[str, float]:
        """Estimate download cost before fetching.

        Returns:
            Dictionary with cost breakdown and total.
        """
        products = self.config.products
        total_years = sum(len(self._get_missing_years(p)) for p in products)

        return {
            "products": len(products),
            "years_needed": total_years,
            "tenors": len(self.config.tenors),
            "cost_per_product_year": self.COST_PER_PRODUCT_YEAR,
            "estimated_total_usd": round(total_years * self.COST_PER_PRODUCT_YEAR, 2),
        }

    def list_status(self) -> dict[str, dict]:
        """List download status for all products.

        Returns:
            Dictionary mapping product to status info.
        """
        status = {}
        all_years = set(self._get_years_range())

        for product in self.config.products:
            downloaded = set()
            for year in all_years:
                if self._year_exists(product, year):
                    downloaded.add(year)

            missing = sorted(all_years - downloaded)
            status[product] = {
                "downloaded": len(downloaded),
                "missing": len(missing),
                "missing_years": missing[:5] if len(missing) > 5 else missing,  # Show first 5
                "complete": len(missing) == 0,
            }

        return status

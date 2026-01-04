"""Databento futures data downloader for batch downloading historical futures data.

This module provides a bulk downloader for futures data from Databento,
designed for the ML4T book's futures chapter. It downloads:
- OHLCV daily bars for all listed contracts
- Instrument definitions (expiration, tick size, multiplier)
- Statistics (settlement prices, open interest)

Key features:
- Parent symbology ({PRODUCT}.FUT) to get all contracts per product
- Hive-partitioned Parquet storage
- Progress tracking with resume capability
- Cost estimation before download

Usage:
    from ml4t.data.futures.downloader import FuturesDownloader
    from ml4t.data.futures.config import FuturesDownloadConfig

    config = FuturesDownloadConfig(
        products=["ES", "CL", "GC"],
        start="2016-01-01",
        end="2025-12-13",
        storage_path="~/ml4t-data/futures",
    )
    downloader = FuturesDownloader(config)
    downloader.download_all()
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import polars as pl
import structlog
from databento import Historical
from databento.common.error import BentoClientError, BentoServerError

from .config import (
    DEFAULT_PRODUCTS,
    DefinitionsConfig,
    DownloadProgress,
    FuturesCategory,
    FuturesDownloadConfig,
    load_definitions_config,
    load_yaml_config,
)

logger = structlog.get_logger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "FuturesDownloader",
    "FuturesDownloadConfig",
    "FuturesCategory",
    "DownloadProgress",
    "DefinitionsDownloader",
    "DefinitionsConfig",
    "load_yaml_config",
    "load_definitions_config",
    "DEFAULT_PRODUCTS",
]


class FuturesDownloader:
    """Download futures data from Databento for ML4T book.

    This downloader fetches historical futures data using Databento's parent
    symbology ({PRODUCT}.FUT) to get all listed contracts for each product.

    Example:
        >>> config = FuturesDownloadConfig(
        ...     products=["ES", "CL"],
        ...     start="2024-01-01",
        ...     end="2024-12-31",
        ... )
        >>> downloader = FuturesDownloader(config)
        >>> downloader.download_all()
    """

    # Approximate cost per product per year per schema (in USD)
    # Based on ~$52 for 25 products × 3 schemas × 10 years = $0.07/product-year-schema
    COST_PER_PRODUCT_YEAR: ClassVar[float] = 0.07

    def __init__(self, config: FuturesDownloadConfig) -> None:
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
        self.progress = DownloadProgress()
        self._ensure_storage_dirs()

        logger.info(
            "Initialized FuturesDownloader",
            products=len(config.get_product_list()),
            start=config.start,
            end=config.end,
            schemas=config.schemas,
        )

    def _ensure_storage_dirs(self) -> None:
        """Create storage directory structure."""
        base = self.config.storage_path
        base.mkdir(parents=True, exist_ok=True)

        for schema in self.config.schemas:
            schema_dir = base / schema.replace("-", "_")
            schema_dir.mkdir(exist_ok=True)

    def _get_product_dir(self, schema: str, product: str) -> Path:
        """Get storage directory for a product and schema."""
        schema_dir = schema.replace("-", "_")
        return self.config.storage_path / schema_dir / f"product={product}"

    def _product_exists(self, product: str) -> bool:
        """Check if product data already exists for all schemas."""
        for schema in self.config.schemas:
            product_dir = self._get_product_dir(schema, product)
            parquet_file = product_dir / f"{schema.replace('-', '_')}.parquet"
            if not parquet_file.exists():
                return False
        return True

    def estimate_cost(self) -> dict[str, float]:
        """Estimate download cost before fetching.

        Returns:
            Dictionary with cost breakdown and total.
        """
        products = self.config.get_product_list()
        n_products = len(products)
        n_schemas = len(self.config.schemas)

        # Calculate years
        start = datetime.strptime(self.config.start, "%Y-%m-%d")
        end = datetime.strptime(self.config.end, "%Y-%m-%d")
        n_years = (end - start).days / 365.0

        estimated_cost = n_products * n_schemas * n_years * self.COST_PER_PRODUCT_YEAR

        return {
            "products": n_products,
            "schemas": n_schemas,
            "years": round(n_years, 1),
            "cost_per_product_year": self.COST_PER_PRODUCT_YEAR,
            "estimated_total_usd": round(estimated_cost, 2),
        }

    def download_product(self, product: str, skip_existing: bool = True) -> bool:
        """Download all schemas for a single product.

        Args:
            product: Product symbol (e.g., "ES", "CL").
            skip_existing: Skip if data already exists.

        Returns:
            True if successful, False otherwise.
        """
        if skip_existing and self._product_exists(product):
            logger.info("Skipping existing product", product=product)
            self.progress.mark_complete(product)
            return True

        logger.info("Downloading product", product=product)

        try:
            for schema in self.config.schemas:
                self._download_schema(product, schema)

            self.progress.mark_complete(product)
            logger.info("Completed product download", product=product)
            return True

        except Exception as e:
            error_msg = str(e)
            self.progress.mark_failed(product, error_msg)
            logger.error("Failed to download product", product=product, error=error_msg)
            return False

    def _download_schema(self, product: str, schema: str, save: bool = True) -> pl.DataFrame:
        """Download a single schema for a product.

        Args:
            product: Product symbol.
            schema: Schema name (ohlcv-1d, definition, statistics).
            save: Whether to save to file. Set False for update operations
                  that need to merge with existing data first.

        Returns:
            Downloaded data as Polars DataFrame.
        """
        product_dir = self._get_product_dir(schema, product)
        product_dir.mkdir(parents=True, exist_ok=True)

        parent_symbol = f"{product}.FUT"

        logger.debug(
            "Fetching schema",
            product=product,
            schema=schema,
            symbol=parent_symbol,
        )

        try:
            data = self.client.timeseries.get_range(
                dataset=self.config.dataset,
                symbols=parent_symbol,
                stype_in="parent",
                schema=schema,
                start=self.config.start,
                end=self.config.end,
            )

            # Convert to DataFrame
            df_pandas = data.to_df().reset_index()
            df = pl.from_pandas(df_pandas)

            # Fix type inference issue: when DataFrame is empty, symbol becomes Float64
            # Cast symbol to String to ensure consistent schema for merging
            if "symbol" in df.columns and df.schema["symbol"] != pl.String:
                df = df.with_columns(pl.col("symbol").cast(pl.String))

            # Add product column for easier filtering
            df = df.with_columns(pl.lit(product).alias("product"))

            # Filter out spreads (contain '-' in symbol) for OHLCV and definition
            # Spreads have symbols like "ES-NQ" or "CLZ4-CLF5"
            if schema in ("ohlcv-1d", "definition") and "symbol" in df.columns:
                original_rows = df.height
                df = df.filter(~pl.col("symbol").str.contains("-"))
                filtered_rows = original_rows - df.height
                if filtered_rows > 0:
                    logger.debug(
                        "Filtered spread contracts",
                        product=product,
                        schema=schema,
                        spreads_removed=filtered_rows,
                    )

            # Save to Parquet (only if save=True)
            output_file = product_dir / f"{schema.replace('-', '_')}.parquet"

            if save:
                # Don't overwrite existing data with empty DataFrame
                # This prevents data loss when API returns no new data for discontinued products
                if df.height == 0 and output_file.exists():
                    logger.debug(
                        "Skipping save - no new data and file exists",
                        product=product,
                        schema=schema,
                    )
                    return df

                df.write_parquet(output_file)

                logger.debug(
                    "Saved schema data",
                    product=product,
                    schema=schema,
                    rows=df.height,
                    file=str(output_file),
                )
            else:
                logger.debug(
                    "Downloaded schema (not saving - update mode)",
                    product=product,
                    schema=schema,
                    rows=df.height,
                )

            return df

        except BentoClientError as e:
            if "not found" in str(e).lower() or "no data" in str(e).lower():
                logger.warning(
                    "No data available",
                    product=product,
                    schema=schema,
                    error=str(e),
                )
                # Only save error marker if we're in save mode
                if save:
                    df = pl.DataFrame({"product": [product], "error": [str(e)]})
                    output_file = product_dir / f"{schema.replace('-', '_')}.parquet"
                    df.write_parquet(output_file)
                return pl.DataFrame()
            raise

        except BentoServerError as e:
            logger.error(
                "Databento server error",
                product=product,
                schema=schema,
                error=str(e),
            )
            raise

    def download_all(
        self,
        skip_existing: bool = True,
        continue_on_error: bool = True,
    ) -> DownloadProgress:
        """Download all products and schemas.

        Args:
            skip_existing: Skip products that already have data.
            continue_on_error: Continue downloading other products on failure.

        Returns:
            DownloadProgress with completed and failed products.
        """
        products = self.config.get_product_list()
        total = len(products)

        cost_estimate = self.estimate_cost()
        logger.info(
            "Starting bulk download",
            total_products=total,
            estimated_cost_usd=cost_estimate["estimated_total_usd"],
            schemas=self.config.schemas,
        )

        for i, product in enumerate(products, 1):
            logger.info(
                "Processing product",
                product=product,
                progress=f"{i}/{total}",
            )

            try:
                success = self.download_product(product, skip_existing=skip_existing)
                if not success and not continue_on_error:
                    logger.error("Stopping due to error", product=product)
                    break

            except Exception as e:
                self.progress.mark_failed(product, str(e))
                logger.error("Unexpected error", product=product, error=str(e))
                if not continue_on_error:
                    break

        # Summary
        n_completed = len(self.progress.completed_products)
        n_failed = len(self.progress.failed_products)

        logger.info(
            "Download complete",
            completed=n_completed,
            failed=n_failed,
            total=total,
        )

        if self.progress.failed_products:
            logger.warning(
                "Failed products",
                products=list(self.progress.failed_products.keys()),
            )

        return self.progress

    def download_all_parallel(
        self,
        max_workers: int = 4,
        skip_existing: bool = True,
    ) -> DownloadProgress:
        """Download all products in parallel using thread pool.

        Args:
            max_workers: Maximum concurrent downloads (default: 4).
            skip_existing: Skip products that already have data.

        Returns:
            DownloadProgress with completed and failed products.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        products = self.config.get_product_list()
        total = len(products)

        # Filter out existing products upfront
        if skip_existing:
            products_to_download = [p for p in products if not self._product_exists(p)]
            skipped = total - len(products_to_download)
            if skipped > 0:
                logger.info(
                    "Skipping existing products",
                    skipped=skipped,
                    remaining=len(products_to_download),
                )
                for p in products:
                    if self._product_exists(p):
                        self.progress.mark_complete(p)
        else:
            products_to_download = products

        if not products_to_download:
            logger.info("All products already downloaded")
            return self.progress

        cost_estimate = self.estimate_cost()
        logger.info(
            "Starting parallel download",
            total_products=len(products_to_download),
            max_workers=max_workers,
            estimated_cost_usd=cost_estimate["estimated_total_usd"],
            schemas=self.config.schemas,
        )

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_product = {
                executor.submit(self._download_product_safe, product): product
                for product in products_to_download
            }

            # Process completed futures as they finish
            for completed, future in enumerate(as_completed(future_to_product), 1):
                product = future_to_product[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(
                            "Completed product",
                            product=product,
                            progress=f"{completed}/{len(products_to_download)}",
                        )
                except Exception as e:
                    self.progress.mark_failed(product, str(e))
                    logger.error(
                        "Failed product",
                        product=product,
                        error=str(e),
                        progress=f"{completed}/{len(products_to_download)}",
                    )

        # Summary
        n_completed = len(self.progress.completed_products)
        n_failed = len(self.progress.failed_products)

        logger.info(
            "Parallel download complete",
            completed=n_completed,
            failed=n_failed,
            total=total,
        )

        if self.progress.failed_products:
            logger.warning(
                "Failed products",
                products=list(self.progress.failed_products.keys()),
            )

        return self.progress

    def _download_product_safe(self, product: str) -> bool:
        """Thread-safe wrapper for download_product.

        Creates a new Databento client per thread to avoid connection issues.
        """
        try:
            # Create thread-local client
            thread_client = Historical(self.api_key)

            for schema in self.config.schemas:
                self._download_schema_with_client(product, schema, thread_client)

            self.progress.mark_complete(product)
            return True

        except Exception as e:
            self.progress.mark_failed(product, str(e))
            logger.error("Failed to download product", product=product, error=str(e))
            return False

    def _download_schema_with_client(
        self, product: str, schema: str, client: Historical, save: bool = True
    ) -> pl.DataFrame:
        """Download a schema using a specific client instance.

        Args:
            product: Product symbol.
            schema: Schema name.
            client: Databento Historical client instance.
            save: Whether to save to file. Set False for update operations.
        """
        product_dir = self._get_product_dir(schema, product)
        product_dir.mkdir(parents=True, exist_ok=True)

        parent_symbol = f"{product}.FUT"

        logger.debug(
            "Fetching schema",
            product=product,
            schema=schema,
            symbol=parent_symbol,
        )

        try:
            data = client.timeseries.get_range(
                dataset=self.config.dataset,
                symbols=parent_symbol,
                stype_in="parent",
                schema=schema,
                start=self.config.start,
                end=self.config.end,
            )

            # Convert to DataFrame
            df_pandas = data.to_df().reset_index()
            df = pl.from_pandas(df_pandas)

            # Fix type inference issue: when DataFrame is empty, symbol becomes Float64
            # Cast symbol to String to ensure consistent schema for merging
            if "symbol" in df.columns and df.schema["symbol"] != pl.String:
                df = df.with_columns(pl.col("symbol").cast(pl.String))

            # Add product column for easier filtering
            df = df.with_columns(pl.lit(product).alias("product"))

            # Filter out spreads (contain '-' in symbol) for OHLCV and definition
            # Spreads have symbols like "ES-NQ" or "CLZ4-CLF5"
            if schema in ("ohlcv-1d", "definition") and "symbol" in df.columns:
                original_rows = df.height
                df = df.filter(~pl.col("symbol").str.contains("-"))
                filtered_rows = original_rows - df.height
                if filtered_rows > 0:
                    logger.debug(
                        "Filtered spread contracts",
                        product=product,
                        schema=schema,
                        spreads_removed=filtered_rows,
                    )

            # Save to Parquet (only if save=True)
            output_file = product_dir / f"{schema.replace('-', '_')}.parquet"

            if save:
                # Don't overwrite existing data with empty DataFrame
                # This prevents data loss when API returns no new data for discontinued products
                if df.height == 0 and output_file.exists():
                    logger.debug(
                        "Skipping save - no new data and file exists",
                        product=product,
                        schema=schema,
                    )
                    return df

                df.write_parquet(output_file)

                logger.debug(
                    "Saved schema data",
                    product=product,
                    schema=schema,
                    rows=df.height,
                    file=str(output_file),
                )
            else:
                logger.debug(
                    "Downloaded schema (not saving - update mode)",
                    product=product,
                    schema=schema,
                    rows=df.height,
                )

            return df

        except BentoClientError as e:
            if "not found" in str(e).lower() or "no data" in str(e).lower():
                logger.warning(
                    "No data available",
                    product=product,
                    schema=schema,
                    error=str(e),
                )
                # Only save error marker if we're in save mode
                if save:
                    df = pl.DataFrame({"product": [product], "error": [str(e)]})
                    output_file = product_dir / f"{schema.replace('-', '_')}.parquet"
                    df.write_parquet(output_file)
                return pl.DataFrame()
            raise

        except BentoServerError as e:
            logger.error(
                "Databento server error",
                product=product,
                schema=schema,
                error=str(e),
            )
            raise

    def download_test(self, products: list[str] | None = None) -> DownloadProgress:
        """Download a small test set to verify configuration.

        Args:
            products: Products to test (default: ES, CL, GC).

        Returns:
            DownloadProgress for test products.
        """
        test_products = products or ["ES", "CL", "GC"]

        # Create temporary config with shorter date range
        test_config = FuturesDownloadConfig(
            products=test_products,
            start="2024-11-01",
            end="2024-11-30",
            storage_path=self.config.storage_path / "test",
            dataset=self.config.dataset,
            schemas=self.config.schemas,
            api_key=self.api_key,
        )

        test_downloader = FuturesDownloader(test_config)
        return test_downloader.download_all()

    def list_downloaded(self) -> dict[str, list[str]]:
        """List all downloaded products by schema.

        Returns:
            Dictionary mapping schema to list of downloaded products.
        """
        result: dict[str, list[str]] = {}

        for schema in self.config.schemas:
            schema_dir = self.config.storage_path / schema.replace("-", "_")
            if not schema_dir.exists():
                result[schema] = []
                continue

            products = []
            for product_dir in schema_dir.iterdir():
                if product_dir.is_dir() and product_dir.name.startswith("product="):
                    product = product_dir.name.replace("product=", "")
                    parquet_file = product_dir / f"{schema.replace('-', '_')}.parquet"
                    if parquet_file.exists():
                        products.append(product)

            result[schema] = sorted(products)

        return result

    def get_latest_date(self, product: str | None = None) -> datetime | None:
        """Get the latest date in existing OHLCV data.

        Args:
            product: Specific product to check, or None for all products.

        Returns:
            Latest date found, or None if no data exists.
        """
        ohlcv_dir = self.config.storage_path / "ohlcv_1d"
        if not ohlcv_dir.exists():
            return None

        latest_date: datetime | None = None

        products_to_check = [product] if product else self.config.get_product_list()

        for p in products_to_check:
            parquet_file = ohlcv_dir / f"product={p}" / "ohlcv_1d.parquet"
            if parquet_file.exists():
                try:
                    df = pl.read_parquet(parquet_file)
                    if "ts_event" in df.columns and df.height > 0:
                        max_date = df.select(pl.col("ts_event").max()).item()
                        if max_date is not None:
                            if isinstance(max_date, datetime):
                                product_latest = max_date
                            else:
                                product_latest = datetime.fromisoformat(str(max_date))

                            if latest_date is None or product_latest > latest_date:
                                latest_date = product_latest
                except Exception as e:
                    logger.warning("Could not read date from file", product=p, error=str(e))

        return latest_date

    def update(self, end_date: str | None = None) -> DownloadProgress:
        """Update existing data by downloading only new data since last download.

        This method:
        1. Finds the latest date in existing data
        2. Downloads data from that date to end_date (or today)
        3. Merges new data with existing data

        Args:
            end_date: End date for update (default: today). Format: YYYY-MM-DD

        Returns:
            DownloadProgress with results.

        Example:
            >>> downloader.update()  # Update to today
            >>> downloader.update("2025-12-31")  # Update to specific date
        """
        from datetime import timedelta

        # Find latest existing date
        latest_date = self.get_latest_date()

        if latest_date is None:
            logger.info("No existing data found, performing full download")
            return self.download_all()

        # Start from day after latest date
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        # Check if already up to date
        if start_date >= end_date:
            logger.info(
                "Data already up to date",
                latest_date=latest_date.strftime("%Y-%m-%d"),
                end_date=end_date,
            )
            return self.progress

        logger.info(
            "Starting incremental update",
            from_date=start_date,
            to_date=end_date,
            latest_existing=latest_date.strftime("%Y-%m-%d"),
        )

        # Create update config
        original_start = self.config.start
        original_end = self.config.end
        self.config.start = start_date
        self.config.end = end_date

        try:
            # Download new data (will overwrite - we'll merge later)
            products = self.config.get_product_list()

            for product in products:
                self._update_product(product)

            return self.progress

        finally:
            # Restore original config
            self.config.start = original_start
            self.config.end = original_end

    def _update_product(self, product: str) -> bool:
        """Update a single product by downloading new data and merging.

        Args:
            product: Product symbol.

        Returns:
            True if successful.
        """
        try:
            for schema in self.config.schemas:
                existing_file = (
                    self._get_product_dir(schema, product) / f"{schema.replace('-', '_')}.parquet"
                )

                # Download new data WITHOUT saving (save=False prevents overwriting existing data)
                new_data = self._download_schema(product, schema, save=False)

                # Skip if no new data
                if new_data.height == 0:
                    logger.debug(
                        "No new data to merge",
                        product=product,
                        schema=schema,
                    )
                    continue

                # Filter out error rows from new data
                if "error" in new_data.columns:
                    new_data = new_data.filter(pl.col("error").is_null())

                # If existing file exists, merge with it
                if existing_file.exists():
                    existing_data = pl.read_parquet(existing_file)

                    # Filter out error rows from existing data
                    if "error" in existing_data.columns:
                        existing_data = existing_data.filter(pl.col("error").is_null())

                    # Merge and deduplicate
                    if existing_data.height > 0:
                        # Ensure same columns
                        common_cols = set(existing_data.columns) & set(new_data.columns)
                        existing_data = existing_data.select(sorted(common_cols))
                        new_data = new_data.select(sorted(common_cols))

                        merged = pl.concat([existing_data, new_data])

                        # Deduplicate based on key columns
                        if schema == "ohlcv-1d":
                            merged = merged.unique(subset=["ts_event", "symbol"], keep="last")
                        elif schema == "definition":
                            merged = merged.unique(subset=["ts_recv", "raw_symbol"], keep="last")

                        merged = merged.sort(
                            "ts_event" if "ts_event" in merged.columns else "ts_recv"
                        )
                        merged.write_parquet(existing_file)

                        logger.info(
                            "Merged data",
                            product=product,
                            schema=schema,
                            existing_rows=existing_data.height,
                            new_rows=new_data.height,
                            merged_rows=merged.height,
                        )
                    else:
                        # Existing file had only error rows, save new data
                        new_data.write_parquet(existing_file)
                        logger.info(
                            "Replaced error-only file with new data",
                            product=product,
                            schema=schema,
                            new_rows=new_data.height,
                        )
                else:
                    # No existing file, save new data directly
                    new_data.write_parquet(existing_file)
                    logger.info(
                        "Created new file",
                        product=product,
                        schema=schema,
                        rows=new_data.height,
                    )

            self.progress.mark_complete(product)
            return True

        except Exception as e:
            self.progress.mark_failed(product, str(e))
            logger.error("Failed to update product", product=product, error=str(e))
            return False


# Import DefinitionsDownloader for backwards compatibility
from .definitions import DefinitionsDownloader

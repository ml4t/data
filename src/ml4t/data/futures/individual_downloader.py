"""Download individual futures contracts from Databento.

This module downloads specific contract symbols (ESH24, CLF24, etc.) for
demonstrating roll mechanics. Unlike ContinuousDownloader which downloads
pre-rolled contracts, this downloads the actual individual contracts.

Key features:
- Downloads specific contract symbols (not parent symbology)
- Hourly OHLCV data (ohlcv-1h) matching continuous contracts
- Supports quarterly (H, M, U, Z) and monthly products
- Generates contract symbols for specified years
- Single file output per product

Usage:
    from ml4t.data.futures.individual_downloader import (
        IndividualDownloader,
        IndividualDownloadConfig,
    )

    config = IndividualDownloadConfig(
        products={
            "ES": {"months": [3, 6, 9, 12]},   # Quarterly
            "CL": {"months": list(range(1, 13))},  # Monthly
        },
        years=[2024, 2025],
        storage_path="~/ml4t-data/futures/individual",
    )
    downloader = IndividualDownloader(config)
    downloader.download_all()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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

from .databento_parser import MONTH_TO_CODE

logger = structlog.get_logger(__name__)


@dataclass
class IndividualProductConfig:
    """Configuration for a single product's individual contracts."""

    months: list[int]  # Contract months (1-12)
    download_window_months: int = 6  # How many months before expiry to download

    @property
    def is_quarterly(self) -> bool:
        """Check if product has quarterly contracts."""
        return set(self.months) == {3, 6, 9, 12}

    @property
    def is_monthly(self) -> bool:
        """Check if product has monthly contracts."""
        return len(self.months) == 12


@dataclass
class IndividualDownloadConfig:
    """Configuration for individual contract downloads.

    Attributes:
        products: Dict mapping product to its config (months, window).
        years: List of years to download contracts for.
        storage_path: Base path for storing downloaded data.
        dataset: Databento dataset (default: GLBX.MDP3 for CME).
        schema: OHLCV schema (default: ohlcv-1h for hourly).
        api_key: Databento API key (defaults to DATABENTO_API_KEY env var).
    """

    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    years: list[int] = field(default_factory=lambda: [2024, 2025])
    storage_path: str | Path = "~/ml4t-data/futures/individual"
    dataset: str = "GLBX.MDP3"
    schema: str = "ohlcv-1h"
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.storage_path = Path(self.storage_path).expanduser()

    def get_product_config(self, product: str) -> IndividualProductConfig:
        """Get configuration for a specific product."""
        if product not in self.products:
            raise ValueError(f"Product {product} not in configuration")

        product_data = self.products[product]
        return IndividualProductConfig(
            months=product_data.get("months", [3, 6, 9, 12]),
            download_window_months=product_data.get("download_window_months", 6),
        )


def load_individual_config(yaml_path: str | Path) -> IndividualDownloadConfig:
    """Load individual download configuration from YAML file.

    Expected YAML structure (within databento_futures.yaml):
        individual:
          schema: "ohlcv-1h"
          output_dir: "futures/individual"
          products:
            ES:
              months: [3, 6, 9, 12]
              download_window_months: 6
            CL:
              months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
              download_window_months: 2
          years: [2024, 2025]

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        IndividualDownloadConfig instance.
    """
    yaml_path = Path(yaml_path).expanduser()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Get individual section (can be nested under 'individual' key)
    individual_data = data.get("individual", data)

    # Extract storage path
    storage_path = individual_data.get("output_dir", "futures/individual")
    if not storage_path.startswith("/") and not storage_path.startswith("~"):
        # Relative path - prepend default data dir
        base_storage = data.get("storage", {}).get("path", "~/ml4t-data")
        storage_path = f"{base_storage}/{storage_path}"

    return IndividualDownloadConfig(
        products=individual_data.get("products", {}),
        years=individual_data.get("years", [2024, 2025]),
        storage_path=storage_path,
        dataset=data.get("dataset", "GLBX.MDP3"),
        schema=individual_data.get("schema", "ohlcv-1h"),
    )


class IndividualDownloader:
    """Download individual futures contracts from Databento.

    Downloads specific contract symbols (ESH24, CLF24, etc.) for demonstrating
    roll mechanics. Unlike ContinuousDownloader which uses pre-rolled symbology,
    this downloads actual individual contracts using raw_symbol.

    Storage format:
        {storage_path}/{PRODUCT}/data.parquet

    Example:
        >>> config = IndividualDownloadConfig(
        ...     products={
        ...         "ES": {"months": [3, 6, 9, 12]},
        ...         "CL": {"months": list(range(1, 13))},
        ...     },
        ...     years=[2024, 2025],
        ... )
        >>> downloader = IndividualDownloader(config)
        >>> downloader.download_all()
    """

    # Approximate cost per contract (hourly data)
    COST_PER_CONTRACT: ClassVar[float] = 0.10

    def __init__(self, config: IndividualDownloadConfig) -> None:
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

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized IndividualDownloader",
            products=list(config.products.keys()),
            years=config.years,
            schema=config.schema,
        )

    def _generate_contract_symbols(self, product: str) -> list[str]:
        """Generate contract symbols for a product across configured years.

        Args:
            product: Product symbol (e.g., "ES", "CL").

        Returns:
            List of contract symbols (e.g., ["ESH24", "ESM24", "ESU24", "ESZ24"]).
        """
        product_config = self.config.get_product_config(product)
        symbols = []

        for year in self.config.years:
            year_suffix = str(year)[-2:]  # 2024 -> "24"

            for month in product_config.months:
                month_code = MONTH_TO_CODE[month]
                symbol = f"{product}{month_code}{year_suffix}"
                symbols.append(symbol)

        return sorted(symbols)

    def _get_output_path(self, product: str) -> Path:
        """Get output path for a product."""
        return self.config.storage_path / product / "data.parquet"

    def _product_exists(self, product: str) -> bool:
        """Check if product data already exists and has rows."""
        path = self._get_output_path(product)
        if not path.exists():
            return False
        try:
            df = pl.read_parquet(path)
            return len(df) > 0
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((BentoServerError, ConnectionError, OSError)),
    )
    def _download_contracts(self, product: str, symbols: list[str]) -> pl.DataFrame:
        """Download OHLCV data for specific contract symbols.

        Args:
            product: Product symbol.
            symbols: List of contract symbols to download.

        Returns:
            DataFrame with OHLCV data for all contracts.
        """
        # Calculate date range based on contracts
        # Start 6 months before first contract's year, end at last contract's year end
        first_year = min(self.config.years)
        last_year = max(self.config.years)

        start_date = f"{first_year - 1}-07-01"  # Start mid-year before first year
        end_date = f"{last_year}-12-31"

        logger.debug(
            "Downloading contracts",
            product=product,
            symbols=symbols[:5],
            total_symbols=len(symbols),
            start=start_date,
            end=end_date,
        )

        data = self.client.timeseries.get_range(
            dataset=self.config.dataset,
            symbols=symbols,
            schema=self.config.schema,
            stype_in="raw_symbol",
            start=start_date,
            end=end_date,
        )

        df = pl.from_pandas(data.to_df().reset_index())

        # Add product column
        df = df.with_columns(pl.lit(product).alias("product"))

        return df

    def download_product(
        self,
        product: str,
        skip_existing: bool = True,
        force: bool = False,
    ) -> tuple[int, str]:
        """Download all contracts for a product.

        Args:
            product: Product symbol.
            skip_existing: Skip if product data already exists.
            force: Force re-download even if exists.

        Returns:
            Tuple of (rows_downloaded, status_message).
        """
        if skip_existing and not force and self._product_exists(product):
            logger.info("Product already exists", product=product)
            return 0, f"Skipped {product} (already exists)"

        symbols = self._generate_contract_symbols(product)

        if not symbols:
            return 0, f"No symbols generated for {product}"

        logger.info(
            "Downloading individual contracts",
            product=product,
            contracts=len(symbols),
            symbols=symbols[:4],
        )

        try:
            df = self._download_contracts(product, symbols)

            if df.height == 0:
                logger.warning("No data returned", product=product)
                return 0, f"No data for {product}"

            # Save to parquet
            output_path = self._get_output_path(product)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Sort by timestamp and symbol
            df = df.sort(["ts_event", "symbol"])
            df.write_parquet(output_path)

            logger.info(
                "Saved individual contracts",
                product=product,
                rows=df.height,
                contracts=df["symbol"].n_unique(),
                path=str(output_path),
            )

            return df.height, f"Downloaded {product}: {df.height:,} rows, {df['symbol'].n_unique()} contracts"

        except BentoClientError as e:
            error_msg = str(e)
            logger.warning("Client error", product=product, error=error_msg)
            return 0, f"Error for {product}: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            logger.error("Failed to download", product=product, error=error_msg)
            return 0, f"Failed {product}: {error_msg}"

    def download_all(
        self,
        skip_existing: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Download all configured products.

        Args:
            skip_existing: Skip products that already exist.
            force: Force re-download even if exists.

        Returns:
            Summary with rows, products downloaded, errors.
        """
        products = list(self.config.products.keys())
        total = len(products)

        logger.info(
            "Starting individual contracts download",
            products=products,
            years=self.config.years,
        )

        results = {
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_rows": 0,
            "messages": [],
        }

        for i, product in enumerate(products, 1):
            logger.info(f"[{i}/{total}] Processing {product}")

            rows, msg = self.download_product(product, skip_existing=skip_existing, force=force)
            results["messages"].append(msg)

            if "Skipped" in msg:
                results["skipped"] += 1
            elif rows > 0:
                results["downloaded"] += 1
                results["total_rows"] += rows
            else:
                results["failed"] += 1

        logger.info(
            "Download complete",
            downloaded=results["downloaded"],
            skipped=results["skipped"],
            failed=results["failed"],
            total_rows=results["total_rows"],
        )

        return results

    def estimate_cost(self) -> dict[str, float]:
        """Estimate download cost before fetching.

        Returns:
            Dictionary with cost breakdown and total.
        """
        total_contracts = 0

        for product in self.config.products:
            symbols = self._generate_contract_symbols(product)
            total_contracts += len(symbols)

        return {
            "products": len(self.config.products),
            "total_contracts": total_contracts,
            "years": len(self.config.years),
            "cost_per_contract": self.COST_PER_CONTRACT,
            "estimated_total_usd": round(total_contracts * self.COST_PER_CONTRACT, 2),
        }

    def list_symbols(self) -> dict[str, list[str]]:
        """List all contract symbols that would be downloaded.

        Returns:
            Dictionary mapping product to list of symbols.
        """
        result = {}
        for product in self.config.products:
            result[product] = self._generate_contract_symbols(product)
        return result

    def get_status(self) -> dict[str, dict]:
        """Get download status for all products.

        Returns:
            Dictionary mapping product to status info.
        """
        status = {}

        for product in self.config.products:
            path = self._get_output_path(product)

            if not path.exists():
                status[product] = {
                    "exists": False,
                    "rows": 0,
                    "contracts": 0,
                }
            else:
                try:
                    df = pl.read_parquet(path)
                    status[product] = {
                        "exists": True,
                        "rows": df.height,
                        "contracts": df["symbol"].n_unique(),
                        "date_range": (
                            df["ts_event"].min().isoformat(),
                            df["ts_event"].max().isoformat(),
                        ),
                    }
                except Exception as e:
                    status[product] = {
                        "exists": True,
                        "error": str(e),
                    }

        return status

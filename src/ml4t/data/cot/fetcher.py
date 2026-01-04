"""CFTC Commitment of Traders (COT) data fetcher.

Fetches weekly COT reports from the CFTC via the cot_reports library.
Provides mapping from CME/ICE product codes to CFTC market names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

logger = logging.getLogger(__name__)


# Report type definitions
COTReportType = Literal[
    "legacy_fut",
    "legacy_futopt",
    "disaggregated_fut",
    "disaggregated_futopt",
    "traders_in_financial_futures_fut",
    "traders_in_financial_futures_futopt",
    "supplemental_futopt",
]


@dataclass
class ProductMapping:
    """Mapping from exchange product code to CFTC COT market name."""

    code: str  # Exchange product code (e.g., 'ES')
    cot_name: str  # CFTC market name pattern to match
    report_type: COTReportType  # Which COT report contains this product
    description: str = ""  # Human-readable description


# Product mappings for CME/ICE/CBOT products
# Financial futures use 'traders_in_financial_futures_fut' report
# Commodities use 'disaggregated_fut' report
PRODUCT_MAPPINGS: dict[str, ProductMapping] = {
    # Equity Index Futures (Financial)
    "ES": ProductMapping(
        "ES",
        "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "E-mini S&P 500",
    ),
    "NQ": ProductMapping(
        "NQ",
        "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "E-mini NASDAQ-100",
    ),
    "RTY": ProductMapping(
        "RTY",
        "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "E-mini Russell 2000",
    ),
    "YM": ProductMapping(
        "YM",
        "DJIA Consolidated - CHICAGO BOARD OF TRADE",
        "traders_in_financial_futures_fut",
        "E-mini Dow Jones",
    ),
    # Currency Futures (Financial)
    "6E": ProductMapping(
        "6E",
        "EURO FX - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Euro FX",
    ),
    "6J": ProductMapping(
        "6J",
        "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Japanese Yen",
    ),
    "6B": ProductMapping(
        "6B",
        "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "British Pound",
    ),
    "6C": ProductMapping(
        "6C",
        "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Canadian Dollar",
    ),
    "6A": ProductMapping(
        "6A",
        "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Australian Dollar",
    ),
    "6S": ProductMapping(
        "6S",
        "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Swiss Franc",
    ),
    "6M": ProductMapping(
        "6M",
        "MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Mexican Peso",
    ),
    "6N": ProductMapping(
        "6N",
        "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "New Zealand Dollar",
    ),
    # Interest Rate Futures (Financial)
    "ZN": ProductMapping(
        "ZN",
        "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
        "traders_in_financial_futures_fut",
        "10-Year T-Note",
    ),
    "ZB": ProductMapping(
        "ZB",
        "UST BOND - CHICAGO BOARD OF TRADE",
        "traders_in_financial_futures_fut",
        "30-Year T-Bond",
    ),
    "ZF": ProductMapping(
        "ZF",
        "UST 5Y NOTE - CHICAGO BOARD OF TRADE",
        "traders_in_financial_futures_fut",
        "5-Year T-Note",
    ),
    "ZT": ProductMapping(
        "ZT",
        "UST 2Y NOTE - CHICAGO BOARD OF TRADE",
        "traders_in_financial_futures_fut",
        "2-Year T-Note",
    ),
    "SR3": ProductMapping(
        "SR3",
        "SOFR-3M - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "3-Month SOFR",
    ),
    # Crypto (Financial)
    "BTC": ProductMapping(
        "BTC",
        "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Bitcoin",
    ),
    "ETH": ProductMapping(
        "ETH",
        "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE",
        "traders_in_financial_futures_fut",
        "Ether",
    ),
    # VIX
    "VX": ProductMapping(
        "VX",
        "VIX FUTURES - CBOE FUTURES EXCHANGE",
        "traders_in_financial_futures_fut",
        "VIX Futures",
    ),
    # Energy Futures (Commodity/Disaggregated)
    "CL": ProductMapping(
        "CL",
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "WTI Crude Oil",
    ),
    "NG": ProductMapping(
        "NG",
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Natural Gas",
    ),
    "RB": ProductMapping(
        "RB",
        "RBOB GASOLINE - NEW YORK MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "RBOB Gasoline",
    ),
    "HO": ProductMapping(
        "HO",
        "NO. 2 HEATING OIL, NY HARBOR - NEW YORK MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Heating Oil",
    ),
    # Metals (Commodity/Disaggregated)
    "GC": ProductMapping(
        "GC",
        "GOLD - COMMODITY EXCHANGE INC.",
        "disaggregated_fut",
        "Gold",
    ),
    "SI": ProductMapping(
        "SI",
        "SILVER - COMMODITY EXCHANGE INC.",
        "disaggregated_fut",
        "Silver",
    ),
    "HG": ProductMapping(
        "HG",
        "COPPER- #1 - COMMODITY EXCHANGE INC.",
        "disaggregated_fut",
        "Copper",
    ),
    "PL": ProductMapping(
        "PL",
        "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Platinum",
    ),
    # Agricultural (Commodity/Disaggregated)
    "ZC": ProductMapping(
        "ZC",
        "CORN - CHICAGO BOARD OF TRADE",
        "disaggregated_fut",
        "Corn",
    ),
    "ZW": ProductMapping(
        "ZW",
        "WHEAT-SRW - CHICAGO BOARD OF TRADE",
        "disaggregated_fut",
        "Wheat (SRW)",
    ),
    "ZS": ProductMapping(
        "ZS",
        "SOYBEANS - CHICAGO BOARD OF TRADE",
        "disaggregated_fut",
        "Soybeans",
    ),
    "ZM": ProductMapping(
        "ZM",
        "SOYBEAN MEAL - CHICAGO BOARD OF TRADE",
        "disaggregated_fut",
        "Soybean Meal",
    ),
    "ZL": ProductMapping(
        "ZL",
        "SOYBEAN OIL - CHICAGO BOARD OF TRADE",
        "disaggregated_fut",
        "Soybean Oil",
    ),
    # Livestock (Commodity/Disaggregated)
    "LE": ProductMapping(
        "LE",
        "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Live Cattle",
    ),
    "HE": ProductMapping(
        "HE",
        "LEAN HOGS - CHICAGO MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Lean Hogs",
    ),
    "GF": ProductMapping(
        "GF",
        "FEEDER CATTLE - CHICAGO MERCANTILE EXCHANGE",
        "disaggregated_fut",
        "Feeder Cattle",
    ),
}


@dataclass
class COTConfig:
    """Configuration for COT data download."""

    products: list[str] = field(default_factory=list)
    start_year: int = 2020
    end_year: int | None = None  # None = current year
    storage_path: Path = field(default_factory=lambda: Path("~/ml4t-data/cot").expanduser())
    include_options: bool = False  # Use *_futopt reports instead of *_fut

    def __post_init__(self):
        if self.end_year is None:
            self.end_year = datetime.now().year
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path).expanduser()

    def get_years(self) -> list[int]:
        """Get list of years to download."""
        return list(range(self.start_year, self.end_year + 1))


def load_cot_config(config_path: str | Path) -> COTConfig:
    """Load COT configuration from YAML file.

    Example YAML:
        products:
          - ES
          - CL
          - GC
        start_year: 2020
        end_year: 2024
        storage_path: ~/ml4t-data/cot
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return COTConfig(
        products=data.get("products", []),
        start_year=data.get("start_year", 2020),
        end_year=data.get("end_year"),
        storage_path=Path(data.get("storage_path", "~/ml4t-data/cot")).expanduser(),
        include_options=data.get("include_options", False),
    )


class COTFetcher:
    """Fetches CFTC Commitment of Traders data.

    Downloads COT reports and extracts positioning data for specified products.
    Data is stored in Hive-partitioned Parquet format.

    Attributes:
        config: COT download configuration
        _cache: Cache of downloaded COT reports by (report_type, year)
    """

    # Column mappings for standardized output
    # Financial futures (TFF) columns
    TFF_COLUMNS = {
        "Report_Date_as_YYYY-MM-DD": "report_date",
        "Open_Interest_All": "open_interest",
        # Dealer/Intermediary (banks, swap dealers)
        "Dealer_Positions_Long_All": "dealer_long",
        "Dealer_Positions_Short_All": "dealer_short",
        "Dealer_Positions_Spread_All": "dealer_spread",
        # Asset Manager/Institutional
        "Asset_Mgr_Positions_Long_All": "asset_mgr_long",
        "Asset_Mgr_Positions_Short_All": "asset_mgr_short",
        "Asset_Mgr_Positions_Spread_All": "asset_mgr_spread",
        # Leveraged Money (hedge funds)
        "Lev_Money_Positions_Long_All": "lev_money_long",
        "Lev_Money_Positions_Short_All": "lev_money_short",
        "Lev_Money_Positions_Spread_All": "lev_money_spread",
        # Other Reportables
        "Other_Rept_Positions_Long_All": "other_rept_long",
        "Other_Rept_Positions_Short_All": "other_rept_short",
        "Other_Rept_Positions_Spread_All": "other_rept_spread",
        # Non-Reportables (small traders)
        "NonRept_Positions_Long_All": "nonrept_long",
        "NonRept_Positions_Short_All": "nonrept_short",
        # Changes
        "Change_in_Open_Interest_All": "oi_change",
        "Change_in_Dealer_Long_All": "dealer_long_change",
        "Change_in_Dealer_Short_All": "dealer_short_change",
        "Change_in_Asset_Mgr_Long_All": "asset_mgr_long_change",
        "Change_in_Asset_Mgr_Short_All": "asset_mgr_short_change",
        "Change_in_Lev_Money_Long_All": "lev_money_long_change",
        "Change_in_Lev_Money_Short_All": "lev_money_short_change",
    }

    # Disaggregated (commodity) columns
    DISAGG_COLUMNS = {
        "Report_Date_as_YYYY-MM-DD": "report_date",
        "Open_Interest_All": "open_interest",
        # Producer/Merchant/Processor/User (commercials)
        "Prod_Merc_Positions_Long_All": "commercial_long",
        "Prod_Merc_Positions_Short_All": "commercial_short",
        # Swap Dealers
        "Swap_Positions_Long_All": "swap_long",
        "Swap__Positions_Short_All": "swap_short",
        "Swap__Positions_Spread_All": "swap_spread",
        # Managed Money (hedge funds, CTAs)
        "M_Money_Positions_Long_All": "managed_money_long",
        "M_Money_Positions_Short_All": "managed_money_short",
        "M_Money_Positions_Spread_All": "managed_money_spread",
        # Other Reportables
        "Other_Rept_Positions_Long_All": "other_rept_long",
        "Other_Rept_Positions_Short_All": "other_rept_short",
        "Other_Rept_Positions_Spread_All": "other_rept_spread",
        # Non-Reportables (small traders)
        "NonRept_Positions_Long_All": "nonrept_long",
        "NonRept_Positions_Short_All": "nonrept_short",
        # Changes
        "Change_in_Open_Interest_All": "oi_change",
        "Change_in_Prod_Merc_Long_All": "commercial_long_change",
        "Change_in_Prod_Merc_Short_All": "commercial_short_change",
        "Change_in_M_Money_Long_All": "managed_money_long_change",
        "Change_in_M_Money_Short_All": "managed_money_short_change",
    }

    def __init__(self, config: COTConfig | None = None):
        """Initialize COT fetcher.

        Args:
            config: COT download configuration. If None, uses defaults.
        """
        self.config = config or COTConfig()
        self._cache: dict[tuple[str, int], pl.DataFrame] = {}

    def _get_report(self, report_type: COTReportType, year: int) -> pl.DataFrame:
        """Get COT report data for a specific year.

        Uses caching to avoid re-downloading the same data.

        Args:
            report_type: Type of COT report
            year: Year to fetch

        Returns:
            DataFrame with COT data
        """
        cache_key = (report_type, year)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import cot_reports as cot
        except ImportError:
            raise ImportError(
                "cot_reports library not installed. Install with: pip install cot_reports"
            )

        logger.info(f"Downloading COT {report_type} for {year}")
        df_pandas = cot.cot_year(year=year, cot_report_type=report_type)

        # Convert to Polars
        df = pl.from_pandas(df_pandas)
        self._cache[cache_key] = df

        return df

    def _get_report_for_years(self, report_type: COTReportType, years: list[int]) -> pl.DataFrame:
        """Get COT report data for multiple years.

        Args:
            report_type: Type of COT report
            years: List of years to fetch

        Returns:
            Combined DataFrame with COT data
        """
        dfs = []
        for year in years:
            df = self._get_report(report_type, year)
            dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        # Use diagonal_relaxed to handle schema differences between years
        # (some years may have columns with different types or missing columns)
        return pl.concat(dfs, how="diagonal_relaxed")

    def fetch_product(
        self,
        product_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Fetch COT data for a specific product.

        Args:
            product_code: Exchange product code (e.g., 'ES', 'CL')
            start_year: Start year (default: config start_year)
            end_year: End year (default: config end_year)

        Returns:
            DataFrame with COT positioning data

        Raises:
            ValueError: If product code not found in mappings
        """
        if product_code not in PRODUCT_MAPPINGS:
            raise ValueError(
                f"Unknown product code: {product_code}. Available: {list(PRODUCT_MAPPINGS.keys())}"
            )

        mapping = PRODUCT_MAPPINGS[product_code]
        start_year = start_year or self.config.start_year
        end_year = end_year or self.config.end_year
        years = list(range(start_year, end_year + 1))

        # Determine report type (use *_futopt if include_options)
        report_type = mapping.report_type
        if self.config.include_options and report_type.endswith("_fut"):
            report_type = report_type + "opt"

        # Get raw COT data
        df = self._get_report_for_years(report_type, years)

        if df.is_empty():
            logger.warning(f"No COT data found for {product_code}")
            return pl.DataFrame()

        # Filter for this product
        df_filtered = df.filter(
            pl.col("Market_and_Exchange_Names").str.contains(mapping.cot_name.split(" - ")[0])
        )

        if df_filtered.is_empty():
            logger.warning(f"Product {product_code} ({mapping.cot_name}) not found in COT data")
            return pl.DataFrame()

        # Select and rename columns based on report type
        if "traders_in_financial_futures" in report_type:
            column_map = self.TFF_COLUMNS
        else:
            column_map = self.DISAGG_COLUMNS

        # Select available columns
        available_cols = [c for c in column_map if c in df_filtered.columns]
        df_selected = df_filtered.select(available_cols)

        # Rename columns
        rename_map = {old: new for old, new in column_map.items() if old in available_cols}
        df_renamed = df_selected.rename(rename_map)

        # Add product code and metadata
        df_final = df_renamed.with_columns(
            pl.lit(product_code).alias("product"),
            pl.lit(mapping.report_type).alias("report_type"),
        )

        # Parse date and sort
        df_final = df_final.with_columns(
            pl.col("report_date").str.to_date("%Y-%m-%d").alias("report_date")
        ).sort("report_date")

        # Add computed columns (net positions)
        df_final = self._add_computed_columns(df_final, report_type)

        return df_final

    def _add_computed_columns(self, df: pl.DataFrame, report_type: str) -> pl.DataFrame:
        """Add computed positioning columns.

        Args:
            df: DataFrame with COT data
            report_type: Type of COT report

        Returns:
            DataFrame with additional computed columns
        """
        exprs = []

        if "traders_in_financial_futures" in report_type:
            # Net positions for financial futures
            if "dealer_long" in df.columns and "dealer_short" in df.columns:
                exprs.append((pl.col("dealer_long") - pl.col("dealer_short")).alias("dealer_net"))
            if "asset_mgr_long" in df.columns and "asset_mgr_short" in df.columns:
                exprs.append(
                    (pl.col("asset_mgr_long") - pl.col("asset_mgr_short")).alias("asset_mgr_net")
                )
            if "lev_money_long" in df.columns and "lev_money_short" in df.columns:
                exprs.append(
                    (pl.col("lev_money_long") - pl.col("lev_money_short")).alias("lev_money_net")
                )
        else:
            # Net positions for disaggregated (commodity) futures
            if "commercial_long" in df.columns and "commercial_short" in df.columns:
                exprs.append(
                    (pl.col("commercial_long") - pl.col("commercial_short")).alias("commercial_net")
                )
            if "managed_money_long" in df.columns and "managed_money_short" in df.columns:
                exprs.append(
                    (pl.col("managed_money_long") - pl.col("managed_money_short")).alias(
                        "managed_money_net"
                    )
                )

        # Non-reportable net (both report types)
        if "nonrept_long" in df.columns and "nonrept_short" in df.columns:
            exprs.append((pl.col("nonrept_long") - pl.col("nonrept_short")).alias("nonrept_net"))

        if exprs:
            df = df.with_columns(exprs)

        return df

    def fetch_all(self) -> dict[str, pl.DataFrame]:
        """Fetch COT data for all configured products.

        Returns:
            Dictionary mapping product code to DataFrame
        """
        results = {}
        for product in self.config.products:
            try:
                df = self.fetch_product(product)
                if not df.is_empty():
                    results[product] = df
                    logger.info(f"Fetched {len(df)} rows for {product}")
            except Exception as e:
                logger.error(f"Failed to fetch {product}: {e}")

        return results

    def save_to_hive(self, df: pl.DataFrame, product_code: str) -> Path:
        """Save COT data to Hive-partitioned Parquet.

        Args:
            df: DataFrame with COT data
            product_code: Product code for partitioning

        Returns:
            Path to saved data
        """
        output_path = self.config.storage_path / f"product={product_code}"
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / "data.parquet"
        df.write_parquet(file_path)

        logger.info(f"Saved {len(df)} rows to {file_path}")
        return file_path

    def download_all(self, skip_existing: bool = True) -> dict[str, Path]:
        """Download and save COT data for all configured products.

        Args:
            skip_existing: Skip products that already have data

        Returns:
            Dictionary mapping product code to saved file path
        """
        results = {}

        for product in self.config.products:
            output_path = self.config.storage_path / f"product={product}" / "data.parquet"

            if skip_existing and output_path.exists():
                logger.info(f"Skipping {product} (already exists)")
                results[product] = output_path
                continue

            try:
                df = self.fetch_product(product)
                if not df.is_empty():
                    path = self.save_to_hive(df, product)
                    results[product] = path
            except Exception as e:
                logger.error(f"Failed to download {product}: {e}")

        return results

    def list_available_products(self) -> list[str]:
        """List all available product codes."""
        return list(PRODUCT_MAPPINGS.keys())

    def get_product_info(self, product_code: str) -> ProductMapping | None:
        """Get mapping info for a product."""
        return PRODUCT_MAPPINGS.get(product_code)

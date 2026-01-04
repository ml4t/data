"""AQR factor data provider - comprehensive factor data from AQR Research.

This module provides access to 16 AQR datasets covering academically-rigorous
factor research. AQR Capital Management is a pioneer in systematic, factor-based
investing and provides these datasets freely to support academic research.

## Dataset Categories

### Core Equity Factors (QMJ, BAB, HML Devil)
Factor returns for quality, low-volatility, and value anomalies across
US and 23 international equity markets. Both monthly and daily frequencies.

### Equity Portfolios
Sorted portfolios for building factor strategies: 6 portfolios (2×3 size × quality)
and 10 decile portfolios sorted on quality.

### Cross-Asset Factors (VME, TSMOM, Momentum Indices)
Value and momentum across 8 asset classes, trend-following across 58 futures,
and equity momentum indices.

### Long-History Datasets
100+ years of factor data for out-of-sample validation and regime analysis.

## Academic Foundation

Each dataset is based on peer-reviewed academic research:

- **QMJ**: Asness, Frazzini & Pedersen (2014) - "Quality Minus Junk"
- **BAB**: Frazzini & Pedersen (2014) - "Betting Against Beta"
- **HML Devil**: Asness & Frazzini (2013) - "The Devil in HML's Details"
- **VME**: Asness, Moskowitz & Pedersen (2013) - "Value and Momentum Everywhere"
- **TSMOM**: Moskowitz, Ooi & Pedersen (2012) - "Time Series Momentum"

## Data Source

All data from AQR's public research library: https://www.aqr.com/Insights/Datasets
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Literal

import httpx
import pandas as pd
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


# Available AQR datasets - organized by category
AQRDataset = Literal[
    # Core equity factors (daily and monthly)
    "qmj_factors",
    "qmj_factors_daily",
    "bab_factors",
    "bab_factors_daily",
    "hml_devil",
    "hml_devil_daily",
    # Equity portfolios
    "qmj_6_portfolios",
    "qmj_10_portfolios",
    # Cross-asset factors
    "vme_factors",
    "vme_portfolios",
    "tsmom",
    "momentum_indices",
    # Long-history & alternative
    "century_premia",
    "commodities",
    # Optional (static)
    "esg_frontier",
    "credit_premium",
]

# Category groupings for list_datasets(category=...)
AQR_CATEGORIES = {
    "equity_factors": [
        "qmj_factors",
        "qmj_factors_daily",
        "bab_factors",
        "bab_factors_daily",
        "hml_devil",
        "hml_devil_daily",
    ],
    "portfolios": ["qmj_6_portfolios", "qmj_10_portfolios"],
    "cross_asset": ["vme_factors", "vme_portfolios", "tsmom", "momentum_indices"],
    "long_history": ["century_premia", "commodities"],
    "optional": ["esg_frontier", "credit_premium"],
}


class AQRFactorProvider(BaseProvider):
    """
    Provider for AQR Capital Management factor data.

    AQR Capital Management is a global investment management firm built around
    the core principle that a systematic, disciplined approach to managing money
    adds value. Founded in 1998, AQR pioneered many factor investing strategies
    and provides high-quality research data freely to support academic research.

    ## Why AQR Factor Data?

    - **Academic Rigor**: Each dataset is based on peer-reviewed research
    - **Global Coverage**: US + 23 international equity markets
    - **Long History**: Some datasets start from 1877 (commodities) or 1920 (century premia)
    - **Cross-Asset**: Factors across equities, bonds, currencies, commodities
    - **Active Updates**: Monthly updates (except static/historical datasets)
    - **Free Access**: No API key required, just download and use

    ## Dataset Categories

    **Core Equity Factors** (16 datasets, actively updated):

    - **QMJ (Quality Minus Junk)**: Long quality stocks (profitable, growing, safe),
      short junk stocks. Quality earns positive risk-adjusted returns.

    - **BAB (Betting Against Beta)**: Exploits the low-volatility anomaly. Many
      investors can't leverage, so they overweight risky stocks. BAB goes long
      leveraged low-beta and short high-beta stocks.

    - **HML Devil**: Improved value factor using more timely price data. Earns
      305-378 bps/year alpha over standard HML.

    **Equity Portfolios**:
    - QMJ 6 Portfolios: Size × Quality sorted portfolios (2×3 sort)
    - QMJ 10 Portfolios: Quality decile portfolios for monotonicity tests

    **Cross-Asset Factors**:

    - **VME (Value & Momentum Everywhere)**: Factors across 8 asset classes.
      Value and momentum are negatively correlated, providing diversification.

    - **TSMOM (Time Series Momentum)**: Trend-following across 58 futures.
      12-month lookback, 1-month hold. Foundation for CTA strategies.

    - **Momentum Indices**: Index-level momentum (US Large Cap, Small Cap, Intl)

    **Long-History Datasets**:

    - **Century of Factor Premia**: 100+ years of value, momentum, carry, defensive
      factors. Essential for out-of-sample validation.

    - **Commodities for the Long Run**: 140+ years of commodity data.
      Shows carry (term structure) is key driver of returns.

    ## Usage

    ```python
    from ml4t.data.providers import AQRFactorProvider

    # First time: download data from AQR website
    AQRFactorProvider.download()

    # Initialize provider (uses downloaded parquet files)
    provider = AQRFactorProvider()

    # List available datasets
    provider.list_datasets()
    provider.list_datasets(category="equity_factors")
    provider.list_categories()

    # Fetch factor data
    qmj = provider.fetch("qmj_factors", region="USA")
    bab = provider.fetch("bab_factors", region="Global")
    tsmom = provider.fetch("tsmom")
    century = provider.fetch("century_premia")

    # Get educational information about a dataset
    info = provider.get_dataset_info("qmj_factors")
    print(info["description"])  # Rich educational description
    print(info["paper"])        # Academic reference
    print(info["use_cases"])    # Practical applications
    ```

    ## Data Sources

    All data from AQR's public research data library:
    https://www.aqr.com/Insights/Datasets

    ## Academic References

    - QMJ: Asness, Frazzini & Pedersen (2014), "Quality Minus Junk"
    - BAB: Frazzini & Pedersen (2014), "Betting Against Beta"
    - HML Devil: Asness & Frazzini (2013), "The Devil in HML's Details"
    - VME: Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
    - TSMOM: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum"
    - Century: Ilmanen et al. (2021), "A Century of Factor Premia"
    - Commodities: Levine et al. (2018), "Commodities for the Long Run"

    ## Notes

    - Returns are in decimal format (0.01 = 1%)
    - Monthly data is normalized to month-start dates for consistency with Fama-French
    - Daily data available for QMJ, BAB, and HML Devil factors
    """

    # Default data location
    DEFAULT_PATH: ClassVar[Path] = Path("~/ml4t/data/aqr_factors").expanduser()

    # AQR base URL
    BASE_URL: ClassVar[str] = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"

    # AQR download URLs - complete list
    DOWNLOAD_URLS: ClassVar[dict[str, str]] = {
        # Core equity factors
        "qmj_factors": "Quality-Minus-Junk-Factors-Monthly.xlsx",
        "qmj_factors_daily": "Quality-Minus-Junk-Factors-Daily.xlsx",
        "bab_factors": "Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
        "bab_factors_daily": "Betting-Against-Beta-Equity-Factors-Daily.xlsx",
        "hml_devil": "The-Devil-in-HMLs-Details-Factors-Monthly.xlsx",
        "hml_devil_daily": "The-Devil-in-HMLs-Details-Factors-Daily.xlsx",
        # Equity portfolios
        "qmj_6_portfolios": "Quality-Minus-Junk-Six-Portfolios-Formed-on-Size-and-Quality-Monthly.xlsx",
        "qmj_10_portfolios": "Quality-Minus-Junk-10-Quality-Sorted-Portfolios-Monthly.xlsx",
        # Cross-asset
        "vme_factors": "Value-and-Momentum-Everywhere-Factors-Monthly.xlsx",
        "vme_portfolios": "Value-and-Momentum-Everywhere-Portfolios-Monthly.xlsx",
        "tsmom": "Time-Series-Momentum-Factors-Monthly.xlsx",
        "momentum_indices": "Momentum-Indices-Monthly.xlsx",
        # Long-history
        "century_premia": "Century-of-Factor-Premia-Monthly.xlsx",
        "commodities": "Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx",
        # Optional (static)
        "esg_frontier": "Responsible-Investing-ESG-Efficient-Frontier.xlsx",
        "credit_premium": "Credit-Risk-Premium-Preliminary-Paper-Data.xlsx",
    }

    # Dataset metadata - comprehensive details for each dataset
    DATASETS: ClassVar[dict[str, dict]] = {
        # ===== CORE EQUITY FACTORS =====
        "qmj_factors": {
            "name": "Quality Minus Junk: Factors, Monthly",
            "category": "equity_factors",
            "frequency": "monthly",
            "start_date": "1957-07",
            "description": (
                "Quality stocks — those of companies that are profitable, growing, and well "
                "managed — command higher prices on average than those of unprofitable, stagnant, "
                "or poorly managed companies, which we refer to as 'junk.' QMJ goes long quality "
                "stocks and shorts junk stocks, measuring quality based on profitability, growth, "
                "safety, and payout characteristics."
            ),
            "paper": "Asness, Frazzini & Pedersen (2014), 'Quality Minus Junk'",
            "regions": [
                "USA",
                "Global",
                "Global Ex USA",
                "Europe",
                "North America",
                "Pacific",
                "AUS",
                "AUT",
                "BEL",
                "CAN",
                "CHE",
                "DEU",
                "DNK",
                "ESP",
                "FIN",
                "FRA",
                "GBR",
                "GRC",
                "HKG",
                "IRL",
                "ISR",
                "ITA",
                "JPN",
                "NLD",
                "NOR",
                "NZL",
                "PRT",
                "SGP",
                "SWE",
            ],
            "use_cases": [
                "Quality premium analysis",
                "6-factor model construction",
                "Defensive equity strategies",
                "Factor timing research",
            ],
            "skiprows": 18,
        },
        "qmj_factors_daily": {
            "name": "Quality Minus Junk: Factors, Daily",
            "category": "equity_factors",
            "frequency": "daily",
            "start_date": "1957-07",
            "description": "Daily quality factor returns",
            "paper": "Asness, Frazzini & Pedersen (2014), 'Quality Minus Junk'",
            "regions": [
                "USA",
                "Global",
                "Global Ex USA",
                "Europe",
                "North America",
                "Pacific",
            ],
            "use_cases": [
                "High-frequency quality analysis",
                "Event studies",
                "Daily risk management",
            ],
            "skiprows": 18,
        },
        "bab_factors": {
            "name": "Betting Against Beta: Equity Factors, Monthly",
            "category": "equity_factors",
            "frequency": "monthly",
            "start_date": "1931-01",
            "description": (
                "A basic premise of the CAPM is that all agents invest in the highest Sharpe ratio "
                "portfolio and leverage to suit risk preferences. However, many investors (individuals, "
                "pension funds, mutual funds) are constrained in leverage and therefore overweight "
                "riskier securities. This behavior suggests high-beta assets require lower risk-adjusted "
                "returns than low-beta assets. BAB constructs market-neutral factors that are long "
                "leveraged low-beta assets and short high-beta assets, exploiting the low-volatility anomaly."
            ),
            "paper": "Frazzini & Pedersen (2014), 'Betting Against Beta'",
            "regions": [
                "USA",
                "Global",
                "Global Ex USA",
                "Europe",
                "North America",
                "Pacific",
                "AUS",
                "AUT",
                "BEL",
                "CAN",
                "CHE",
                "DEU",
                "DNK",
                "ESP",
                "FIN",
                "FRA",
                "GBR",
                "GRC",
                "HKG",
                "IRL",
                "ISR",
                "ITA",
                "JPN",
                "NLD",
                "NOR",
                "NZL",
                "PRT",
                "SGP",
                "SWE",
            ],
            "use_cases": [
                "Low-volatility anomaly analysis",
                "Risk parity strategies",
                "Leverage constraint research",
                "Beta-neutral portfolios",
            ],
            "skiprows": 18,
        },
        "bab_factors_daily": {
            "name": "Betting Against Beta: Equity Factors, Daily",
            "category": "equity_factors",
            "frequency": "daily",
            "start_date": "1931-01",
            "description": "Daily BAB factor returns",
            "paper": "Frazzini & Pedersen (2014), 'Betting Against Beta'",
            "regions": ["USA", "Global", "Global Ex USA", "Europe", "North America", "Pacific"],
            "use_cases": ["Daily low-vol analysis", "Intramonth BAB timing"],
            "skiprows": 18,
        },
        "hml_devil": {
            "name": "The Devil in HML's Details: Factors, Monthly",
            "category": "equity_factors",
            "frequency": "monthly",
            "start_date": "1926-07",
            "description": (
                "Challenges the standard academic method for measuring 'value' (book-to-price). "
                "The standard HML uses lagged book data with lagged price data, ignoring recent price "
                "movements. HML Devil uses more timely price data while retaining the necessary lag for "
                "book. This improved measure earns statistically significant alphas of 305-378 basis "
                "points per year against 5-factor models containing the standard HML."
            ),
            "paper": "Asness & Frazzini (2013), 'The Devil in HML's Details'",
            "regions": [
                "USA",
                "Global",
                "Global Ex USA",
                "Europe",
                "North America",
                "Pacific",
            ],
            "use_cases": [
                "Improved value factor",
                "Replacing standard HML",
                "Value timing strategies",
            ],
            "skiprows": 18,
        },
        "hml_devil_daily": {
            "name": "The Devil in HML's Details: Factors, Daily",
            "category": "equity_factors",
            "frequency": "daily",
            "start_date": "1926-07",
            "description": "Daily improved HML factor returns",
            "paper": "Asness & Frazzini (2013), 'The Devil in HML's Details'",
            "regions": ["USA", "Global", "Global Ex USA", "Europe", "North America", "Pacific"],
            "use_cases": ["Daily value analysis", "High-frequency value timing"],
            "skiprows": 18,
        },
        # ===== EQUITY PORTFOLIOS =====
        "qmj_6_portfolios": {
            "name": "Quality Minus Junk: Six Portfolios (Size × Quality)",
            "category": "portfolios",
            "frequency": "monthly",
            "start_date": "1957-07",
            "description": "Six portfolios formed on size and quality (2×3 sort)",
            "paper": "Asness, Frazzini & Pedersen (2014), 'Quality Minus Junk'",
            "portfolios": [
                "SMALL LO QUAL",
                "SMALL MED QUAL",
                "SMALL HI QUAL",
                "BIG LO QUAL",
                "BIG MED QUAL",
                "BIG HI QUAL",
            ],
            "use_cases": [
                "Factor construction",
                "Long-only quality strategies",
                "Size-quality interaction",
            ],
            "skiprows": 18,
        },
        "qmj_10_portfolios": {
            "name": "Quality Minus Junk: 10 Quality-Sorted Portfolios",
            "category": "portfolios",
            "frequency": "monthly",
            "start_date": "1957-07",
            "description": "Ten portfolios sorted on quality (deciles)",
            "paper": "Asness, Frazzini & Pedersen (2014), 'Quality Minus Junk'",
            "portfolios": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
            "use_cases": ["Monotonicity tests", "Spread analysis", "Quality premium by decile"],
            "skiprows": 18,
        },
        # ===== CROSS-ASSET FACTORS =====
        "vme_factors": {
            "name": "Value and Momentum Everywhere: Factors",
            "category": "cross_asset",
            "frequency": "monthly",
            "start_date": "1972-01",
            "description": (
                "Consistent value and momentum return premia across eight diverse markets and asset "
                "classes: individual stocks in the US, UK, Continental Europe, and Japan; equity index "
                "futures; government bonds; currencies; and commodity futures. Value and momentum are "
                "negatively correlated with each other (both within and across asset classes), providing "
                "natural diversification. A common factor structure exists among their returns."
            ),
            "paper": "Asness, Moskowitz & Pedersen (2013), 'Value and Momentum Everywhere'",
            "factors": [
                "VAL US",
                "MOM US",
                "VAL UK",
                "MOM UK",
                "VAL EU",
                "MOM EU",
                "VAL JP",
                "MOM JP",
                "VAL EQ",
                "MOM EQ",
                "VAL FX",
                "MOM FX",
                "VAL FI",
                "MOM FI",
                "VAL CM",
                "MOM CM",
                "VAL EVERYWHERE",
                "MOM EVERYWHERE",
            ],
            "use_cases": [
                "Cross-asset strategies",
                "Diversification analysis",
                "Common factor structure",
                "Multi-asset momentum/value",
            ],
            "skiprows": 17,
        },
        "vme_portfolios": {
            "name": "Value and Momentum Everywhere: Portfolios",
            "category": "cross_asset",
            "frequency": "monthly",
            "start_date": "1972-01",
            "description": "48 long-only portfolios across 8 asset classes",
            "paper": "Asness, Moskowitz & Pedersen (2013), 'Value and Momentum Everywhere'",
            "use_cases": [
                "Long-only cross-asset strategies",
                "Return attribution",
                "Diversified alternatives",
            ],
            "skiprows": 17,
        },
        "tsmom": {
            "name": "Time Series Momentum: Factors",
            "category": "cross_asset",
            "frequency": "monthly",
            "start_date": "1985-01",
            "description": (
                "An asset-pricing anomaly termed 'time series momentum' that is consistent across "
                "different asset classes and markets. The past 12-month excess return of each instrument "
                "is a positive predictor of its future return. This 'trend' effect persists for about "
                "a year and then partially reverses over longer horizons. Covers 58 diverse futures and "
                "forward contracts including country equity indices, currencies, commodities, and "
                "sovereign bonds. Forms the academic foundation for trend-following/CTA strategies."
            ),
            "paper": "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum'",
            "asset_classes": ["Equities", "Bonds", "Currencies", "Commodities"],
            "use_cases": [
                "Trend-following strategies",
                "Managed futures replication",
                "CTA strategies",
                "Crisis alpha analysis",
            ],
            "skiprows": 17,
        },
        "momentum_indices": {
            "name": "AQR Momentum Indices",
            "category": "cross_asset",
            "frequency": "monthly",
            "start_date": "1927-01",  # Varies by index
            "description": "Index-level momentum for US, Small Cap, and International equities",
            "paper": "AQR Momentum Indices Methodology",
            "indices": [
                "AQR US Large Cap Momentum",
                "AQR US Small Cap Momentum",
                "AQR International Momentum",
            ],
            "use_cases": [
                "Index momentum strategies",
                "Momentum ETF benchmarking",
                "Factor timing",
            ],
            "skiprows": 17,
        },
        # ===== LONG-HISTORY DATASETS =====
        "century_premia": {
            "name": "Century of Factor Premia",
            "category": "long_history",
            "frequency": "monthly",
            "start_date": "1920-01",
            "description": (
                "Over 100 years of factor data across six asset classes. Provides out-of-sample "
                "validation for factor premia that were largely discovered using post-1963 data. "
                "Factors include value, momentum, carry, and defensive across equities, bonds, "
                "currencies, commodities, credit, and multi-asset composites. Essential for "
                "understanding factor robustness across different economic regimes."
            ),
            "paper": "Ilmanen et al. (2021), 'A Century of Factor Premia'",
            "factors": ["Value", "Momentum", "Carry", "Defensive"],
            "asset_classes": [
                "Equities",
                "Bonds",
                "Currencies",
                "Commodities",
                "Credit",
                "Multi-Asset",
            ],
            "use_cases": [
                "Out-of-sample validation",
                "Regime analysis",
                "Factor timing research",
                "Long-term strategy evaluation",
            ],
            "skiprows": 17,
        },
        "commodities": {
            "name": "Commodities for the Long Run",
            "category": "long_history",
            "frequency": "monthly",
            "start_date": "1877-01",
            "description": (
                "Over 140 years of commodity data starting from 1877. Includes equal-weighted portfolios, "
                "long-short strategies exploiting backwardation/contango (carry), and spot vs. futures "
                "decomposition. Essential for understanding commodities as an asset class over multiple "
                "economic regimes including wars, depressions, and inflationary periods. Shows the term "
                "structure (carry) is a key driver of commodity returns."
            ),
            "paper": "Levine, Ooi, Richardson & Sasseville (2018), 'Commodities for the Long Run'",
            "portfolios": ["Equal Weight", "Long-Short (Backwardation)", "Spot Return", "Carry"],
            "use_cases": [
                "Commodity allocation",
                "Inflation hedging",
                "Alternative risk premia",
                "Term structure strategies",
            ],
            "skiprows": 17,
        },
        # ===== OPTIONAL/STATIC DATASETS =====
        "esg_frontier": {
            "name": "ESG-Efficient Frontier",
            "category": "optional",
            "frequency": "monthly",
            "start_date": "2010-01",
            "description": "ESG-efficient portfolios (static 2020 data)",
            "paper": "Pedersen, Fitzgibbons & Pomorski (2020), 'Responsible Investing'",
            "note": "Static dataset, not actively updated",
            "use_cases": ["ESG integration research", "Sustainable investing"],
            "skiprows": 17,
        },
        "credit_premium": {
            "name": "Credit Risk Premium",
            "category": "optional",
            "frequency": "monthly",
            "start_date": "1927-01",
            "end_date": "2014-12",
            "description": "Credit risk premium data (static, 1927-2014)",
            "paper": "AQR Credit Risk Premium Working Paper",
            "note": "Static dataset, not actively updated",
            "use_cases": ["Credit factor research", "Bond factor models"],
            "skiprows": 17,
        },
    }

    def __init__(self, data_path: str | Path | None = None) -> None:
        """
        Initialize AQR factor provider.

        Args:
            data_path: Path to AQR data directory (default: ~/ml4t/data/aqr_factors/)
                      Should contain parquet files from download().

        Raises:
            FileNotFoundError: If data_path doesn't exist
        """
        super().__init__(rate_limit=None)

        self.data_path = Path(data_path or self.DEFAULT_PATH).expanduser()

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"AQR data directory not found at {self.data_path}. "
                f"Run AQRFactorProvider.download() to fetch data from AQR."
            )

        self.logger.info("Initialized AQR factor provider", data_path=str(self.data_path))

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "aqr"

    def list_datasets(self, category: str | None = None) -> list[str]:
        """
        List available AQR datasets.

        Args:
            category: Filter by category (equity_factors, portfolios, cross_asset,
                     long_history, optional). If None, returns all datasets.

        Returns:
            List of dataset identifiers

        Example:
            >>> provider = AQRFactorProvider()
            >>> provider.list_datasets()
            ['qmj_factors', 'qmj_factors_daily', 'bab_factors', ...]
            >>> provider.list_datasets(category="equity_factors")
            ['qmj_factors', 'qmj_factors_daily', 'bab_factors', ...]
        """
        if category:
            if category not in AQR_CATEGORIES:
                raise ValueError(
                    f"Unknown category '{category}'. Available: {list(AQR_CATEGORIES.keys())}"
                )
            return AQR_CATEGORIES[category]
        return list(self.DATASETS.keys())

    def list_categories(self) -> list[str]:
        """List dataset categories."""
        return list(AQR_CATEGORIES.keys())

    def get_dataset_info(self, dataset: AQRDataset) -> dict:
        """
        Get metadata for a specific dataset.

        Args:
            dataset: Dataset identifier

        Returns:
            Dictionary with name, description, paper, start_date, etc.
        """
        self._validate_dataset(dataset)
        return self.DATASETS[dataset].copy()

    def fetch(
        self,
        dataset: AQRDataset,
        region: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch AQR factor data.

        Args:
            dataset: Dataset identifier (e.g., 'qmj_factors', 'bab_factors')
            region: Region/country filter (e.g., 'USA', 'Global'). Only for equity factors.
                   For VME, use the factor name (e.g., 'VAL US', 'MOM EVERYWHERE')
            start: Start date (YYYY-MM-DD or YYYY-MM), optional
            end: End date (YYYY-MM-DD or YYYY-MM), optional

        Returns:
            Polars DataFrame with 'timestamp' column and factor return columns.
            Returns are in decimal format (0.01 = 1%).

        Example:
            >>> provider = AQRFactorProvider()
            >>> qmj = provider.fetch("qmj_factors", region="USA")
            >>> bab = provider.fetch("bab_factors", region="Global")
            >>> tsmom = provider.fetch("tsmom")
        """
        self._validate_dataset(dataset)

        # Find parquet file
        parquet_path = self.data_path / f"{dataset}.parquet"

        if not parquet_path.exists():
            raise DataNotAvailableError(
                provider="aqr",
                symbol=dataset,
                details={
                    "reason": f"Parquet file not found: {parquet_path}",
                    "suggestion": "Run AQRFactorProvider.download() to fetch data",
                },
            )

        # Load data
        df = pl.read_parquet(parquet_path)

        # Filter by region if specified (for equity factors)
        if region and region in df.columns:
            df = df.select(["timestamp", region])

        # Filter by date range
        if start:
            df = df.filter(pl.col("timestamp") >= start)
        if end:
            df = df.filter(pl.col("timestamp") <= end)

        self.logger.info(
            "Fetched AQR data",
            dataset=dataset,
            region=region,
            rows=len(df),
            columns=len(df.columns) - 1,
        )

        return df.sort("timestamp")

    # Alias methods for backward compatibility
    def fetch_factor(
        self,
        dataset: AQRDataset,
        region: str = "USA",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Alias for fetch() - backward compatibility."""
        return self.fetch(dataset, region=region, start=start, end=end)

    def fetch_factors(
        self,
        dataset: AQRDataset,
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Fetch all columns for a dataset (no region filter)."""
        return self.fetch(dataset, region=None, start=start, end=end)

    def _validate_dataset(self, dataset: str) -> None:
        """Validate dataset name."""
        if dataset not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {list(self.DATASETS.keys())}"
            )

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Base class method - not used for factor data."""
        raise NotImplementedError("AQR provides factor data, not OHLCV. Use fetch() instead.")

    @classmethod
    def download(
        cls,
        output_path: str | Path | None = None,
        datasets: list[AQRDataset] | None = None,
        include_optional: bool = False,
    ) -> Path:
        """
        Download AQR factor data from the AQR website.

        Downloads Excel files from AQR's research data library, processes them,
        and saves as Parquet files for efficient querying.

        Args:
            output_path: Directory to save data (default: ~/ml4t/data/aqr_factors/)
            datasets: List of datasets to download (default: all except optional)
            include_optional: Include optional/static datasets (default: False)

        Returns:
            Path to the data directory

        Example:
            >>> # Download all active datasets
            >>> path = AQRFactorProvider.download()

            >>> # Download specific datasets
            >>> path = AQRFactorProvider.download(datasets=["qmj_factors", "bab_factors"])

            >>> # Include optional datasets
            >>> path = AQRFactorProvider.download(include_optional=True)
        """
        log = structlog.get_logger()

        output_dir = Path(output_path or cls.DEFAULT_PATH).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which datasets to download
        if datasets:
            datasets_to_download = datasets
        else:
            datasets_to_download = list(cls.DOWNLOAD_URLS)
            if not include_optional:
                datasets_to_download = [
                    d for d in datasets_to_download if cls.DATASETS[d]["category"] != "optional"
                ]

        log.info(
            "Starting AQR data download",
            output_path=str(output_dir),
            datasets=datasets_to_download,
            count=len(datasets_to_download),
        )

        # Create source directory for original Excel files
        source_dir = output_dir / "source"
        source_dir.mkdir(exist_ok=True)

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            for dataset in datasets_to_download:
                if dataset not in cls.DOWNLOAD_URLS:
                    log.warning(f"Unknown dataset: {dataset}")
                    continue

                filename = cls.DOWNLOAD_URLS[dataset]
                url = cls.BASE_URL + filename
                log.info(f"Downloading {dataset}...", url=url)

                try:
                    response = client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as e:
                    log.error(f"Failed to download {dataset}: {e}")
                    continue

                # Save source Excel
                excel_path = source_dir / filename
                with open(excel_path, "wb") as f:
                    f.write(response.content)
                log.debug(f"Saved source: {excel_path}")

                # Parse Excel file
                try:
                    df = cls._parse_aqr_excel(dataset, BytesIO(response.content))
                except Exception as e:
                    log.error(f"Failed to parse {dataset} Excel: {e}")
                    continue

                # Save parquet
                parquet_path = output_dir / f"{dataset}.parquet"
                df.write_parquet(parquet_path)
                log.info(f"Saved: {parquet_path} ({len(df)} rows, {len(df.columns) - 1} columns)")

        # Save metadata
        metadata = {
            "source": "AQR Capital Management",
            "website": "https://www.aqr.com/Insights/Datasets",
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "datasets": {
                k: {"name": v["name"], "category": v["category"]}
                for k, v in cls.DATASETS.items()
                if k in datasets_to_download
            },
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(
            "Download complete",
            output_path=str(output_dir),
            datasets_downloaded=len(datasets_to_download),
        )
        return output_dir

    @classmethod
    def _parse_aqr_excel(cls, dataset: str, file: BytesIO) -> pl.DataFrame:
        """
        Parse AQR Excel file into Polars DataFrame.

        AQR Excel files have complex headers that need special handling.
        Returns are already in decimal format (0.01 = 1%).
        """
        info = cls.DATASETS[dataset]
        skiprows = info.get("skiprows", 18)

        # Read with pandas (better Excel support), then convert to Polars
        df_pd = pd.read_excel(file, sheet_name=0, skiprows=skiprows, engine="openpyxl")

        # Some files (VME, Century) have column names in the data rows
        # Check if first column name looks like a placeholder
        first_col = str(df_pd.columns[0]).lower()
        if "unnamed" in first_col or "please" in first_col or "disclosures" in first_col:
            # Column names are in a data row - find them
            for idx, row in df_pd.iterrows():
                first_val = str(row.iloc[0]).upper() if pd.notna(row.iloc[0]) else ""
                # Check if this row has "DATE" or looks like a date
                if first_val == "DATE":
                    # This row has the column names
                    df_pd.columns = [
                        str(v) if pd.notna(v) else f"col_{i}" for i, v in enumerate(row)
                    ]
                    df_pd = df_pd.iloc[idx + 1 :].reset_index(drop=True)
                    break
                # Check if row 0 contains column names (not dates)
                elif idx == 0 and pd.isna(row.iloc[0]) and not pd.isna(row.iloc[1]):
                    # Row 0 has column names starting from column 1
                    new_cols = ["date"] + [
                        str(v) if pd.notna(v) else f"col_{i}" for i, v in enumerate(row.iloc[1:], 1)
                    ]
                    df_pd.columns = new_cols
                    df_pd = df_pd.iloc[1:].reset_index(drop=True)
                    break

        # First column is date
        df_pd.columns = ["date"] + list(df_pd.columns[1:])

        # Parse dates (various formats)
        df_pd["date"] = pd.to_datetime(df_pd["date"], errors="coerce")

        # Drop rows with invalid dates
        df_pd = df_pd.dropna(subset=["date"])

        # Convert to Polars and rename
        df = pl.from_pandas(df_pd)
        df = df.rename({"date": "timestamp"})

        # Cast numeric columns to float (data is already in decimal format)
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in numeric_cols])

        # Normalize month-end dates to month-start for consistency
        # (AQR uses month-end, French uses month-start)
        if info.get("frequency") == "monthly":
            df = df.with_columns(pl.col("timestamp").dt.month_start().alias("timestamp"))

        return df.drop_nulls(subset=["timestamp"]).sort("timestamp")

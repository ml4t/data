"""Fama-French factor data provider - comprehensive factor data from Ken French's Data Library.

This module provides access to 70+ datasets from Kenneth R. French's Data Library,
the canonical source for academic factor research data.

## Dataset Categories

### Core Factors
The Fama-French factor models that revolutionized asset pricing:
- **FF3** (1993): Market, Size (SMB), Value (HML)
- **FF5** (2015): Adds Profitability (RMW) and Investment (CMA)
- **Momentum** (Carhart, 1997): Winners minus losers (UMD)
- **Reversal**: Short-term (1-month) and long-term (3-5 year)

### Sorted Portfolios
Portfolios sorted on characteristics - essential for factor construction:
- **Univariate**: Size, Book-to-Market, Profitability, Investment, Momentum, Beta
- **Bivariate**: Size×Value (6, 25, 100 portfolios), Size×Momentum, etc.
- **Trivariate**: Size×Value×Profitability (32 portfolios)

### Industry Portfolios
Fama-French industry classifications (5, 10, 12, 17, 30, 38, 48, 49 industries).
The FF48 is most commonly used in academic research.

### International Factors
Global factor data for developed and emerging markets.

## Academic Foundation

- Fama & French (1993): "Common risk factors in the returns on stocks and bonds"
- Fama & French (2015): "A five-factor asset pricing model"
- Carhart (1997): "On persistence in mutual fund performance"

## Data Source

Kenneth R. French Data Library:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


# Dataset categories
FFCategory = Literal["factors", "portfolios", "industry", "international", "breakpoints"]
FFFrequency = Literal["monthly", "weekly", "daily"]

# Complete list of available datasets
FFDataset = Literal[
    # === CORE FACTORS ===
    "ff3",
    "ff5",
    "mom",
    "st_rev",
    "lt_rev",
    # Combined
    "ff3_mom",
    "ff5_mom",
    # === UNIVARIATE PORTFOLIOS ===
    "port_size",
    "port_bm",
    "port_op",
    "port_inv",
    "port_ep",
    "port_cfp",
    "port_dp",
    "port_mom",
    "port_strev",
    "port_ltrev",
    "port_beta",
    "port_var",
    "port_resvar",
    "port_accruals",
    "port_ni",
    # === BIVARIATE PORTFOLIOS ===
    "port_size_bm_6",
    "port_size_bm_25",
    "port_size_bm_100",
    "port_size_op_6",
    "port_size_op_25",
    "port_size_inv_6",
    "port_size_inv_25",
    "port_size_mom_6",
    "port_size_mom_25",
    "port_bm_op_25",
    "port_bm_inv_25",
    "port_op_inv_25",
    # === TRIVARIATE PORTFOLIOS ===
    "port_size_bm_op_32",
    "port_size_bm_inv_32",
    "port_size_op_inv_32",
    # === INDUSTRY PORTFOLIOS ===
    "ind_5",
    "ind_10",
    "ind_12",
    "ind_17",
    "ind_30",
    "ind_38",
    "ind_48",
    "ind_49",
    # === INTERNATIONAL ===
    "ff3_developed",
    "ff3_developed_ex_us",
    "ff3_europe",
    "ff3_japan",
    "ff3_asia_pacific",
    "ff3_north_america",
    "ff5_developed",
    "ff5_emerging",
    "mom_developed",
    "mom_emerging",
    # === BREAKPOINTS ===
    "bp_me",
    "bp_bm",
    "bp_op",
    "bp_inv",
    "bp_prior",
]

# Category mappings
FF_CATEGORIES = {
    "factors": ["ff3", "ff5", "mom", "st_rev", "lt_rev", "ff3_mom", "ff5_mom"],
    "portfolios": [
        "port_size",
        "port_bm",
        "port_op",
        "port_inv",
        "port_ep",
        "port_cfp",
        "port_dp",
        "port_mom",
        "port_strev",
        "port_ltrev",
        "port_beta",
        "port_var",
        "port_resvar",
        "port_accruals",
        "port_ni",
        "port_size_bm_6",
        "port_size_bm_25",
        "port_size_bm_100",
        "port_size_op_6",
        "port_size_op_25",
        "port_size_inv_6",
        "port_size_inv_25",
        "port_size_mom_6",
        "port_size_mom_25",
        "port_bm_op_25",
        "port_bm_inv_25",
        "port_op_inv_25",
        "port_size_bm_op_32",
        "port_size_bm_inv_32",
        "port_size_op_inv_32",
    ],
    "industry": ["ind_5", "ind_10", "ind_12", "ind_17", "ind_30", "ind_38", "ind_48", "ind_49"],
    "international": [
        "ff3_developed",
        "ff3_developed_ex_us",
        "ff3_europe",
        "ff3_japan",
        "ff3_asia_pacific",
        "ff3_north_america",
        "ff5_developed",
        "ff5_emerging",
        "mom_developed",
        "mom_emerging",
    ],
    "breakpoints": ["bp_me", "bp_bm", "bp_op", "bp_inv", "bp_prior"],
}


class FamaFrenchProvider(BaseProvider):
    """
    Provider for Fama-French factor data from Ken French's Data Library.

    Kenneth French's Data Library is the canonical source for academic factor data,
    freely available with no authentication required. The data has been meticulously
    constructed following the methodologies described in the foundational factor
    investing papers.

    ## Why Fama-French Data?

    - **Academic Standard**: The definitive source used in thousands of academic papers
    - **Long History**: Some datasets start from 1926
    - **Free Access**: No API key required
    - **Multiple Frequencies**: Monthly, weekly, and daily data
    - **Comprehensive Coverage**: Factors, portfolios, industries, international

    ## Core Factor Models

    **FF3 (Fama-French 3-Factor Model, 1993)**:
    - Mkt-RF: Market return minus risk-free rate
    - SMB: Small Minus Big (size premium)
    - HML: High Minus Low (value premium)
    - RF: Risk-free rate (1-month T-bill)

    **FF5 (Fama-French 5-Factor Model, 2015)**:
    - Adds RMW: Robust Minus Weak (profitability premium)
    - Adds CMA: Conservative Minus Aggressive (investment premium)

    **Momentum (Carhart, 1997)**:
    - MOM/UMD: Up Minus Down (momentum premium, 2-12 month returns)

    **Reversal**:
    - ST_Rev: Short-term reversal (1-month)
    - LT_Rev: Long-term reversal (13-60 months)

    ## Portfolio Sorts

    **Univariate Sorts**: Portfolios formed on a single characteristic
    - Size (ME), Book-to-Market, Profitability, Investment
    - Momentum, Beta, Variance, Accruals, Net Issues

    **Bivariate Sorts**: Portfolios formed on two characteristics
    - 6 Portfolios (2×3): Used for factor construction
    - 25 Portfolios (5×5): Standard for research
    - 100 Portfolios (10×10): Fine-grained analysis

    **Industry Portfolios**: FF48 is most commonly used
    - 48 industries based on SIC codes
    - Use for sector exposure analysis, industry momentum

    ## Usage

    ```python
    from ml4t.data.providers import FamaFrenchProvider

    provider = FamaFrenchProvider()

    # Core factors
    ff3 = provider.fetch("ff3")
    ff5 = provider.fetch("ff5")
    mom = provider.fetch("mom")

    # Combined models
    ff4 = provider.fetch_combined(["ff3", "mom"])  # Carhart 4-factor
    ff6 = provider.fetch_combined(["ff5", "mom"])  # 6-factor

    # Portfolios for factor construction
    size_value = provider.fetch("port_size_bm_25")

    # Industries
    ind48 = provider.fetch("ind_48")

    # International
    ff3_europe = provider.fetch("ff3_europe")

    # Daily data
    ff3_daily = provider.fetch("ff3", frequency="daily")

    # List available datasets
    provider.list_datasets()
    provider.list_datasets(category="factors")
    provider.list_categories()
    ```

    ## Data Source

    Kenneth R. French Data Library:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    ## Notes

    - Returns are in decimal format (0.01 = 1%)
    - Dates use month-start format (first trading day)
    - Missing values (shown as -99.99 in source) are converted to null
    """

    # Ken French Data Library base URL
    BASE_URL: ClassVar[str] = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

    # Complete file mapping
    FILES: ClassVar[dict[str, dict]] = {
        # === CORE FACTORS ===
        "ff3": {
            "file": "F-F_Research_Data_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "weekly", "daily"],
        },
        "ff5": {
            "file": "F-F_Research_Data_5_Factors_2x3",
            "columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "mom": {
            "file": "F-F_Momentum_Factor",
            "columns": ["MOM"],
            "frequencies": ["monthly", "daily"],
        },
        "st_rev": {
            "file": "F-F_ST_Reversal_Factor",
            "columns": ["ST_Rev"],
            "frequencies": ["monthly", "daily"],
        },
        "lt_rev": {
            "file": "F-F_LT_Reversal_Factor",
            "columns": ["LT_Rev"],
            "frequencies": ["monthly", "daily"],
        },
        # === UNIVARIATE PORTFOLIOS ===
        "port_size": {"file": "Portfolios_Formed_on_ME", "frequencies": ["monthly", "daily"]},
        "port_bm": {"file": "Portfolios_Formed_on_BE-ME", "frequencies": ["monthly", "daily"]},
        "port_op": {"file": "Portfolios_Formed_on_OP", "frequencies": ["monthly"]},
        "port_inv": {"file": "Portfolios_Formed_on_INV", "frequencies": ["monthly"]},
        "port_ep": {"file": "Portfolios_Formed_on_E-P", "frequencies": ["monthly"]},
        "port_cfp": {"file": "Portfolios_Formed_on_CF-P", "frequencies": ["monthly"]},
        "port_dp": {"file": "Portfolios_Formed_on_D-P", "frequencies": ["monthly"]},
        "port_mom": {"file": "10_Portfolios_Prior_12_2", "frequencies": ["monthly", "daily"]},
        "port_strev": {"file": "10_Portfolios_Prior_1_0", "frequencies": ["monthly", "daily"]},
        "port_ltrev": {"file": "10_Portfolios_Prior_60_13", "frequencies": ["monthly"]},
        "port_beta": {"file": "Portfolios_Formed_on_BETA", "frequencies": ["monthly"]},
        "port_var": {"file": "Portfolios_Formed_on_VAR", "frequencies": ["monthly"]},
        "port_resvar": {"file": "Portfolios_Formed_on_RESVAR", "frequencies": ["monthly"]},
        "port_accruals": {"file": "Portfolios_Formed_on_AC", "frequencies": ["monthly"]},
        "port_ni": {"file": "Portfolios_Formed_on_NI", "frequencies": ["monthly"]},
        # === BIVARIATE PORTFOLIOS ===
        "port_size_bm_6": {"file": "6_Portfolios_2x3", "frequencies": ["monthly", "daily"]},
        "port_size_bm_25": {"file": "25_Portfolios_5x5", "frequencies": ["monthly", "daily"]},
        "port_size_bm_100": {"file": "100_Portfolios_10x10", "frequencies": ["monthly", "daily"]},
        "port_size_op_6": {"file": "6_Portfolios_ME_OP_2x3", "frequencies": ["monthly"]},
        "port_size_op_25": {"file": "25_Portfolios_ME_OP_5x5", "frequencies": ["monthly"]},
        "port_size_inv_6": {"file": "6_Portfolios_ME_INV_2x3", "frequencies": ["monthly"]},
        "port_size_inv_25": {"file": "25_Portfolios_ME_INV_5x5", "frequencies": ["monthly"]},
        "port_size_mom_6": {"file": "6_Portfolios_ME_Prior_12_2", "frequencies": ["monthly"]},
        "port_size_mom_25": {"file": "25_Portfolios_ME_Prior_12_2", "frequencies": ["monthly"]},
        "port_bm_op_25": {"file": "25_Portfolios_BEME_OP_5x5", "frequencies": ["monthly"]},
        "port_bm_inv_25": {"file": "25_Portfolios_BEME_INV_5x5", "frequencies": ["monthly"]},
        "port_op_inv_25": {"file": "25_Portfolios_OP_INV_5x5", "frequencies": ["monthly"]},
        # === TRIVARIATE PORTFOLIOS ===
        "port_size_bm_op_32": {
            "file": "32_Portfolios_ME_BEME_OP_2x4x4",
            "frequencies": ["monthly"],
        },
        "port_size_bm_inv_32": {
            "file": "32_Portfolios_ME_BEME_INV_2x4x4",
            "frequencies": ["monthly"],
        },
        "port_size_op_inv_32": {
            "file": "32_Portfolios_ME_OP_INV_2x4x4",
            "frequencies": ["monthly"],
        },
        # === INDUSTRY PORTFOLIOS ===
        "ind_5": {"file": "5_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_10": {"file": "10_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_12": {"file": "12_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_17": {"file": "17_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_30": {"file": "30_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_38": {"file": "38_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_48": {"file": "48_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        "ind_49": {"file": "49_Industry_Portfolios", "frequencies": ["monthly", "daily"]},
        # === INTERNATIONAL - DEVELOPED ===
        "ff3_developed": {
            "file": "Developed_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff3_developed_ex_us": {
            "file": "Developed_ex_US_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff3_europe": {
            "file": "Europe_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff3_japan": {
            "file": "Japan_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff3_asia_pacific": {
            "file": "Asia_Pacific_ex_Japan_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff3_north_america": {
            "file": "North_America_3_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff5_developed": {
            "file": "Developed_5_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            "frequencies": ["monthly", "daily"],
        },
        "ff5_emerging": {
            "file": "Emerging_5_Factors",
            "columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            "frequencies": ["monthly"],
        },
        "mom_developed": {
            "file": "Developed_Mom_Factor",
            "columns": ["WML"],
            "frequencies": ["monthly", "daily"],
        },
        "mom_emerging": {
            "file": "Emerging_MOM_Factor",
            "columns": ["WML"],
            "frequencies": ["monthly"],
        },
        # === BREAKPOINTS ===
        "bp_me": {"file": "ME_Breakpoints", "frequencies": ["monthly"]},
        "bp_bm": {"file": "BE-ME_Breakpoints", "frequencies": ["monthly"]},
        "bp_op": {"file": "OP_Breakpoints", "frequencies": ["monthly"]},
        "bp_inv": {"file": "INV_Breakpoints", "frequencies": ["monthly"]},
        "bp_prior": {"file": "Prior_2-12_Breakpoints", "frequencies": ["monthly"]},
    }

    # Dataset metadata with educational descriptions
    DATASETS: ClassVar[dict[str, dict]] = {
        # === CORE FACTORS ===
        "ff3": {
            "name": "Fama-French 3 Factors",
            "category": "factors",
            "start_date": "1926-07",
            "description": (
                "The foundational 3-factor model that revolutionized asset pricing. Explains "
                "stock returns using market risk (Mkt-RF), size premium (SMB: small stocks "
                "outperform large), and value premium (HML: high book-to-market outperforms low). "
                "Use for basic risk adjustment and calculating 3-factor alpha."
            ),
            "paper": "Fama & French (1993), 'Common risk factors in the returns on stocks and bonds'",
            "use_cases": [
                "Risk adjustment (calculating alpha)",
                "Size premium analysis",
                "Value premium analysis",
                "Factor exposure measurement",
            ],
        },
        "ff5": {
            "name": "Fama-French 5 Factors",
            "category": "factors",
            "start_date": "1963-07",
            "description": (
                "Extended factor model adding profitability (RMW: stocks of firms with robust "
                "profitability outperform weak) and investment (CMA: conservative firms that "
                "invest less outperform aggressive investors). Together with FF3, this creates "
                "the modern 5-factor model that explains most cross-sectional return variation."
            ),
            "paper": "Fama & French (2015), 'A five-factor asset pricing model'",
            "use_cases": [
                "Comprehensive factor models",
                "Alpha evaluation",
                "Profitability/investment premium analysis",
                "Strategy attribution",
            ],
        },
        "mom": {
            "name": "Momentum Factor",
            "category": "factors",
            "start_date": "1927-01",
            "description": (
                "The momentum anomaly: stocks with high returns over the past 2-12 months "
                "continue to outperform those with low past returns. This is the 'UMD' (Up "
                "Minus Down) or 'WML' (Winners Minus Losers) factor. Combined with FF3, it "
                "creates the Carhart 4-factor model."
            ),
            "paper": "Carhart (1997), 'On persistence in mutual fund performance'",
            "use_cases": [
                "Momentum strategy construction",
                "4-factor model (FF3 + MOM)",
                "Factor timing research",
                "Cross-sectional momentum analysis",
            ],
        },
        "st_rev": {
            "name": "Short-Term Reversal Factor",
            "category": "factors",
            "start_date": "1926-07",
            "description": (
                "Short-term reversal: stocks that performed well in the prior month tend to "
                "reverse and underperform in the current month. This is the opposite of momentum "
                "at very short horizons, driven by liquidity provision and overreaction."
            ),
            "paper": "Jegadeesh (1990), 'Evidence of Predictable Behavior of Security Returns'",
            "use_cases": ["Short-term trading strategies", "Market microstructure research"],
        },
        "lt_rev": {
            "name": "Long-Term Reversal Factor",
            "category": "factors",
            "start_date": "1926-07",
            "description": (
                "Long-term reversal: stocks that performed poorly over 3-5 years tend to "
                "outperform those that performed well. This contrarian effect is related to "
                "mean reversion in valuations and is distinct from both momentum and value."
            ),
            "paper": "DeBondt & Thaler (1985), 'Does the Stock Market Overreact?'",
            "use_cases": ["Contrarian strategies", "Long-horizon mean reversion research"],
        },
        # === KEY PORTFOLIOS ===
        "port_size_bm_25": {
            "name": "25 Portfolios Formed on Size and Book-to-Market (5×5)",
            "category": "portfolios",
            "start_date": "1926-07",
            "description": (
                "The standard 5×5 sort on size (ME) and value (BE/ME). Creates 25 portfolios "
                "from Small-Value to Big-Growth. Essential for understanding the size-value "
                "interaction and for constructing custom factors."
            ),
            "paper": "Fama & French (1993)",
            "use_cases": [
                "Factor construction",
                "Size-value interaction analysis",
                "Custom portfolio research",
            ],
        },
        "port_size_bm_6": {
            "name": "6 Portfolios Formed on Size and Book-to-Market (2×3)",
            "category": "portfolios",
            "start_date": "1926-07",
            "description": (
                "The building blocks for SMB and HML factors. 2×3 sort creates 6 portfolios: "
                "Small/Big × Value/Neutral/Growth. SMB = (SV + SN + SG)/3 - (BV + BN + BG)/3. "
                "HML = (SV + BV)/2 - (SG + BG)/2."
            ),
            "paper": "Fama & French (1993)",
            "use_cases": ["Understanding factor construction", "Replicating SMB and HML"],
        },
        "ind_48": {
            "name": "48 Industry Portfolios",
            "category": "industry",
            "start_date": "1926-07",
            "description": (
                "The most commonly used Fama-French industry classification. 48 industries "
                "based on SIC codes, covering all major sectors. Use for sector exposure "
                "analysis, industry momentum strategies, and controlling for industry effects."
            ),
            "paper": "Fama & French (1997), 'Industry Costs of Equity'",
            "use_cases": [
                "Sector exposure analysis",
                "Industry momentum",
                "Sector rotation strategies",
                "Controlling for industry effects",
            ],
        },
        # Add descriptions for other datasets as needed
    }

    # Default cache location
    DEFAULT_CACHE_PATH: ClassVar[Path] = Path("~/ml4t/data/french_factors").expanduser()

    def __init__(
        self,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize Fama-French provider.

        Args:
            cache_path: Directory to cache downloaded data (default: ~/ml4t/data/french_factors/)
            use_cache: Whether to use cached data if available (default: True)
        """
        super().__init__(rate_limit=None)

        self.cache_path = Path(cache_path or self.DEFAULT_CACHE_PATH).expanduser()
        self.use_cache = use_cache

        if use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Initialized Fama-French provider",
            cache_path=str(self.cache_path) if use_cache else None,
            use_cache=use_cache,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "fama_french"

    def list_categories(self) -> list[str]:
        """List dataset categories."""
        return list(FF_CATEGORIES.keys())

    def list_datasets(self, category: str | None = None) -> list[str]:
        """
        List available datasets.

        Args:
            category: Filter by category (factors, portfolios, industry, international, breakpoints)

        Returns:
            List of dataset identifiers
        """
        if category:
            if category not in FF_CATEGORIES:
                raise ValueError(
                    f"Unknown category '{category}'. Available: {list(FF_CATEGORIES.keys())}"
                )
            return FF_CATEGORIES[category]
        return list(self.FILES.keys())

    def get_dataset_info(self, dataset: str) -> dict:
        """
        Get metadata for a specific dataset.

        Args:
            dataset: Dataset identifier

        Returns:
            Dictionary with name, description, paper, use_cases, etc.
        """
        # Check for combined datasets
        if dataset == "ff3_mom":
            return {
                "name": "Fama-French 3 Factors + Momentum (Carhart 4-Factor)",
                "category": "factors",
                "start_date": "1927-01",
                "columns": ["Mkt-RF", "SMB", "HML", "RF", "MOM"],
                "description": (
                    "The Carhart 4-factor model: FF3 plus momentum. This is the standard "
                    "model for mutual fund performance evaluation and factor analysis."
                ),
                "paper": "Carhart (1997), 'On persistence in mutual fund performance'",
            }
        if dataset == "ff5_mom":
            return {
                "name": "Fama-French 5 Factors + Momentum (6-Factor)",
                "category": "factors",
                "start_date": "1963-07",
                "columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"],
                "description": "Comprehensive 6-factor model for modern alpha evaluation.",
            }

        if dataset not in self.FILES:
            raise ValueError(f"Unknown dataset '{dataset}'. Use list_datasets() to see available.")

        # Get base info from FILES
        file_info = self.FILES[dataset].copy()

        # Merge with DATASETS if we have rich descriptions
        if dataset in self.DATASETS:
            file_info.update(self.DATASETS[dataset])

        return file_info

    def fetch(
        self,
        dataset: str,
        frequency: FFFrequency = "monthly",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch Fama-French data.

        Args:
            dataset: Dataset identifier (e.g., 'ff3', 'ff5', 'port_size_bm_25', 'ind_48')
            frequency: 'monthly', 'weekly', or 'daily' (default: 'monthly')
            start: Start date (YYYY-MM-DD or YYYY-MM), optional
            end: End date (YYYY-MM-DD or YYYY-MM), optional

        Returns:
            Polars DataFrame with 'timestamp' column and factor/portfolio return columns.
            Returns are in decimal format (0.01 = 1%).

        Example:
            >>> provider = FamaFrenchProvider()
            >>> ff3 = provider.fetch("ff3")
            >>> ff5_daily = provider.fetch("ff5", frequency="daily")
            >>> port = provider.fetch("port_size_bm_25")
        """
        # Handle combined datasets
        if dataset == "ff3_mom":
            return self.fetch_combined(["ff3", "mom"], frequency=frequency, start=start, end=end)
        if dataset == "ff5_mom":
            return self.fetch_combined(["ff5", "mom"], frequency=frequency, start=start, end=end)

        if dataset not in self.FILES:
            raise ValueError(f"Unknown dataset '{dataset}'. Use list_datasets() to see available.")

        file_info = self.FILES[dataset]

        # Validate frequency
        if frequency not in file_info.get("frequencies", ["monthly"]):
            available = file_info.get("frequencies", ["monthly"])
            raise ValueError(
                f"Frequency '{frequency}' not available for {dataset}. Available: {available}"
            )

        # Check cache first
        cache_file = self._get_cache_path(dataset, frequency)
        if self.use_cache and cache_file.exists():
            self.logger.debug("Loading from cache", file=str(cache_file))
            df = pl.read_parquet(cache_file)
        else:
            # Download from Ken French website
            df = self._download_dataset(dataset, frequency)

            # Cache if enabled
            if self.use_cache:
                df.write_parquet(cache_file)
                self.logger.info("Cached data", file=str(cache_file))

        # Filter by date range (convert strings to date for comparison)
        if start:
            start_date = (
                datetime.strptime(start[:10], "%Y-%m-%d").date()
                if len(start) >= 10
                else datetime.strptime(start + "-01", "%Y-%m-%d").date()
            )
            df = df.filter(pl.col("timestamp") >= start_date)
        if end:
            end_date = (
                datetime.strptime(end[:10], "%Y-%m-%d").date()
                if len(end) >= 10
                else datetime.strptime(end + "-28", "%Y-%m-%d").date()
            )
            df = df.filter(pl.col("timestamp") <= end_date)

        self.logger.info(
            "Fetched Fama-French data",
            dataset=dataset,
            frequency=frequency,
            rows=len(df),
            columns=len(df.columns),
        )

        return df.sort("timestamp")

    def fetch_combined(
        self,
        datasets: list[str],
        frequency: FFFrequency = "monthly",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch and merge multiple datasets on timestamp.

        Args:
            datasets: List of dataset identifiers to combine
            frequency: 'monthly', 'weekly', or 'daily'
            start: Start date, optional
            end: End date, optional

        Returns:
            Merged DataFrame with all columns from the requested datasets

        Example:
            >>> provider = FamaFrenchProvider()
            >>> # Carhart 4-factor model
            >>> ff4 = provider.fetch_combined(["ff3", "mom"])
            >>> # 6-factor model
            >>> ff6 = provider.fetch_combined(["ff5", "mom"])
        """
        if not datasets:
            raise ValueError("Must provide at least one dataset")

        # Fetch first dataset
        result = self.fetch(datasets[0], frequency=frequency, start=start, end=end)

        # Join additional datasets
        for dataset in datasets[1:]:
            df = self.fetch(dataset, frequency=frequency, start=start, end=end)
            result = result.join(df, on="timestamp", how="full", coalesce=True)

        return result.sort("timestamp")

    # Backward compatibility aliases
    def fetch_factors(
        self,
        dataset: str = "ff3",
        start: str | None = None,
        end: str | None = None,
        frequency: FFFrequency = "monthly",
    ) -> pl.DataFrame:
        """Alias for fetch() - backward compatibility."""
        return self.fetch(dataset, frequency=frequency, start=start, end=end)

    def clear_cache(self) -> None:
        """Clear cached data files."""
        if not self.cache_path.exists():
            return

        for f in self.cache_path.glob("*.parquet"):
            f.unlink()
            self.logger.info("Deleted cache file", file=str(f))

    def _get_cache_path(self, dataset: str, frequency: str) -> Path:
        """Get cache file path for a dataset."""
        return self.cache_path / f"{dataset}_{frequency}.parquet"

    def _download_dataset(self, dataset: str, frequency: str = "monthly") -> pl.DataFrame:
        """Download dataset from Ken French's website."""
        file_info = self.FILES[dataset]
        base_file = file_info["file"]

        # Build filename with frequency suffix
        if frequency == "daily":
            filename = f"{base_file}_daily_CSV.zip"
        elif frequency == "weekly":
            filename = f"{base_file}_weekly_CSV.zip"
        else:
            filename = f"{base_file}_CSV.zip"

        url = f"{self.BASE_URL}/{filename}"
        self.logger.info("Downloading from Ken French", url=url, dataset=dataset)

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise DataNotAvailableError(
                provider="fama_french",
                symbol=dataset,
                details={"reason": str(e), "url": url},
            ) from e

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                raise DataNotAvailableError(
                    provider="fama_french",
                    symbol=dataset,
                    details={"reason": "No CSV file found in ZIP"},
                )

            csv_name = csv_files[0]
            with zf.open(csv_name) as f:
                content = f.read().decode("utf-8")

        # Parse the CSV
        expected_columns = file_info.get("columns")
        df = self._parse_french_csv(content, expected_columns, frequency)

        return df

    def _parse_french_csv(
        self, content: str, expected_columns: list[str] | None, frequency: str
    ) -> pl.DataFrame:
        """
        Parse Ken French CSV format into Polars DataFrame.

        French CSV format:
        - Multiple header lines (variable count)
        - Data rows start with YYYYMM (monthly) or YYYYMMDD (daily)
        - Annual data section follows monthly (stop at "Annual" marker)
        - Values are in PERCENT (divide by 100)
        - Missing values shown as -99.99 or -999
        """
        lines = content.strip().split("\n")

        # Find the header row and data start
        data_start = 0
        header_cols = None

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]

            # Check if this looks like a data row (starts with a number)
            if parts[0].isdigit() and len(parts[0]) >= 6:
                data_start = i
                break

            # Check if this is a header row (has column names)
            if len(parts) > 1 and not parts[0].isdigit():
                # Could be header
                header_cols = parts

        # If we have expected columns, use those; otherwise try to parse from header
        if expected_columns:
            columns = expected_columns
        elif header_cols:
            # Clean header columns
            columns = [c.strip() for c in header_cols[1:] if c.strip()]
        else:
            columns = None

        # Parse data rows
        records = []
        for line in lines[data_start:]:
            if not line.strip():
                continue

            # Stop at annual data section or non-numeric first column
            parts = [p.strip() for p in line.split(",")]
            if not parts[0] or not parts[0].replace(".", "").replace("-", "").isdigit():
                if "Annual" in line or not parts[0]:
                    break
                continue

            if not parts[0].isdigit():
                break

            date_str = parts[0]
            values = parts[1:]

            # Parse date
            try:
                if frequency == "monthly":
                    if len(date_str) == 6:
                        year = int(date_str[:4])
                        month = int(date_str[4:])
                        timestamp = f"{year:04d}-{month:02d}-01"
                    else:
                        continue
                elif frequency == "weekly":
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:])
                        timestamp = f"{year:04d}-{month:02d}-{day:02d}"
                    else:
                        continue
                else:  # daily
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:])
                        timestamp = f"{year:04d}-{month:02d}-{day:02d}"
                    else:
                        continue

                # Parse numeric values
                parsed_values = []
                for v in values:
                    try:
                        val = float(v)
                        # Handle missing value indicators (-99 etc) and convert to decimal
                        parsed_val = None if val < -90 else val / 100
                        parsed_values.append(parsed_val)
                    except ValueError:
                        parsed_values.append(None)

                record = {"timestamp": timestamp}

                # Assign column names
                if columns:
                    for j, col in enumerate(columns):
                        if j < len(parsed_values):
                            record[col] = parsed_values[j]
                else:
                    # Generate column names
                    for j, val in enumerate(parsed_values):
                        record[f"col_{j + 1}"] = val

                if len(record) > 1:  # Has at least timestamp and one value
                    records.append(record)

            except (ValueError, IndexError):
                continue

        if not records:
            raise ValueError("No valid data records parsed")

        # Create DataFrame with full schema inference to handle mixed None/float columns
        df = pl.DataFrame(records, infer_schema_length=None)

        # Convert timestamp to datetime
        df = df.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))

        return df.sort("timestamp")

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """
        Implement base class method (not used for factor data).

        For factor data, use fetch() or fetch_combined() instead.
        """
        raise NotImplementedError(
            "Fama-French provides factor data, not OHLCV. Use fetch() or fetch_combined() instead."
        )

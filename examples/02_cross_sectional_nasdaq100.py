"""
Example 2: Multi-Asset Cross-Sectional Analysis with NASDAQ 100
===============================================================

This example demonstrates advanced cross-sectional factor modeling using
104 NASDAQ stocks with minute-level data. It showcases proper multi-asset
backtesting with sector neutralization and Combinatorial Purged CV.

Key Concepts:
- Cross-sectional feature engineering at scale
- Sector-neutral factor construction
- Proper handling of survivorship bias
- Combinatorial Purged Cross-Validation (CPCV)
- Deflated Sharpe Ratio for multiple testing

Author: QuantLab Team
Date: August 2025
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Import qfeatures modules
import ml4t.engineer as qf
from ml4t.evaluation import Evaluator, Tier
from ml4t.evaluation.evaluation.stats import deflated_sharpe_ratio
from ml4t.evaluation.evaluation.viz import create_factor_tearsheet
from ml4t.evaluation.splitters import CombinatorialPurgedKFold

warnings.filterwarnings("ignore")

# Configure paths
DATA_PATH = Path.home() / "ml3t/data/equities/algoseek"
TICKER_MAP_PATH = DATA_PATH / "eq_sec_master/ticker_to_secid_lookup.csv"
OUTPUT_PATH = Path("./output/cross_sectional_analysis")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class NASDAQ100DataLoader:
    """Load and prepare NASDAQ 100 minute data."""

    def __init__(self, year: int = 2021):
        self.year = year
        self.data = None
        self.ticker_map = None
        self.sector_map = None

    def load_data(self) -> pl.DataFrame:
        """Load minute data for all NASDAQ 100 stocks."""
        print(f"Loading NASDAQ 100 data for {self.year}...")

        # Load the parquet file
        data_file = DATA_PATH / f"{self.year}.parquet"
        self.data = pl.read_parquet(data_file)

        # Load ticker mapping
        self.ticker_map = pl.read_csv(TICKER_MAP_PATH)

        print(f"Loaded {len(self.data):,} minute bars for {self.data['sec_id'].n_unique()} stocks")

        # Add ticker symbols
        self.data = self.data.join(
            self.ticker_map.select(["sec_id", "ticker"]),
            on="sec_id",
            how="left",
        )

        # Define sectors (simplified for example)
        self.sector_map = self._create_sector_mapping()
        self.data = self.data.join(
            self.sector_map,
            on="ticker",
            how="left",
        )

        return self.data

    def _create_sector_mapping(self) -> pl.DataFrame:
        """Create sector mapping for NASDAQ 100 stocks."""
        # Simplified sector classification
        sector_dict = {
            # Technology
            "AAPL": "Technology",
            "MSFT": "Technology",
            "NVDA": "Technology",
            "GOOGL": "Technology",
            "GOOG": "Technology",
            "META": "Technology",
            "ADBE": "Technology",
            "CSCO": "Technology",
            "INTC": "Technology",
            "AVGO": "Technology",
            "TXN": "Technology",
            "QCOM": "Technology",
            "AMD": "Technology",
            "MU": "Technology",
            "AMAT": "Technology",
            "LRCX": "Technology",
            "KLAC": "Technology",
            "SNPS": "Technology",
            "CDNS": "Technology",
            "MRVL": "Technology",
            "ADI": "Technology",
            "NXPI": "Technology",
            "MCHP": "Technology",
            "XLNX": "Technology",
            "SWKS": "Technology",
            "FISV": "Technology",
            "INTU": "Technology",
            "ADSK": "Technology",
            "WDAY": "Technology",
            "SPLK": "Technology",
            "TEAM": "Technology",
            "DOCU": "Technology",
            "OKTA": "Technology",
            "CRWD": "Technology",
            "ZM": "Technology",
            # Healthcare
            "AMGN": "Healthcare",
            "GILD": "Healthcare",
            "VRTX": "Healthcare",
            "REGN": "Healthcare",
            "BIIB": "Healthcare",
            "ILMN": "Healthcare",
            "DXCM": "Healthcare",
            "IDXX": "Healthcare",
            "ALGN": "Healthcare",
            "MRNA": "Healthcare",
            "SGEN": "Healthcare",
            "INCY": "Healthcare",
            # Consumer
            "AMZN": "Consumer",
            "TSLA": "Consumer",
            "NFLX": "Consumer",
            "BKNG": "Consumer",
            "MELI": "Consumer",
            "JD": "Consumer",
            "PDD": "Consumer",
            "SBUX": "Consumer",
            "LULU": "Consumer",
            "ROST": "Consumer",
            "ORLY": "Consumer",
            "MNST": "Consumer",
            "KDP": "Consumer",
            "PEP": "Consumer",
            "COST": "Consumer",
            "DLTR": "Consumer",
            "WBA": "Consumer",
            # Communications
            "CMCSA": "Communications",
            "TMUS": "Communications",
            "CHTR": "Communications",
            "NTES": "Communications",
            "BIDU": "Communications",
            "FOX": "Communications",
            "FOXA": "Communications",
            "MTCH": "Communications",
            "TCOM": "Communications",
            "SIRI": "Communications",
            # Other
            "PYPL": "Financials",
            "MAR": "Consumer",
            "HON": "Industrials",
            "CSX": "Industrials",
            "ADP": "Industrials",
            "PCAR": "Industrials",
            "PAYX": "Industrials",
            "CTAS": "Industrials",
            "VRSK": "Industrials",
            "ANSS": "Industrials",
            "CTSH": "Industrials",
            "CDW": "Industrials",
            "CPRT": "Industrials",
            "FAST": "Industrials",
            "AEP": "Utilities",
            "XEL": "Utilities",
            "EXC": "Utilities",
            "EA": "Communications",
            "ATVI": "Communications",
            "EBAY": "Consumer",
            "KHC": "Consumer",
            "MDLZ": "Consumer",
            "CEG": "Utilities",
            "ASML": "Technology",
            "CHKP": "Technology",
            "CERN": "Healthcare",
            "ALXN": "Healthcare",
            "ISRG": "Healthcare",
            "VRSN": "Technology",
            "PTON": "Consumer",
            "MXIM": "Technology",
        }

        return pl.DataFrame(
            {
                "ticker": list(sector_dict.keys()),
                "sector": list(sector_dict.values()),
            },
        )

    def get_trading_universe(self, date: datetime) -> list[str]:
        """Get valid trading universe for a specific date."""
        # Filter stocks with sufficient data on the given date
        daily_data = self.data.filter(
            pl.col("bar_start").dt.date() == date.date(),
        )

        # Require at least 300 minute bars (half day of trading)
        valid_tickers = (
            daily_data.group_by("ticker")
            .agg(pl.count("close").alias("bar_count"))
            .filter(pl.col("bar_count") >= 300)
            .select("ticker")
            .to_series()
            .to_list()
        )

        return valid_tickers


class CrossSectionalFeatureEngineer:
    """Generate cross-sectional features for multi-asset analysis."""

    def __init__(self, data: pl.DataFrame):
        self.data = data
        self.features = None

    def create_base_features(self) -> pl.DataFrame:
        """Create base features for each asset."""
        print("Creating base features...")

        # Group by ticker for individual asset features
        features_list = []

        for ticker in self.data["ticker"].unique():
            if ticker is None:
                continue

            ticker_data = self.data.filter(pl.col("ticker") == ticker).sort("bar_start")

            # Calculate various features
            ticker_features = ticker_data.with_columns(
                [
                    # Returns at multiple horizons
                    pl.col("close").pct_change(1).alias("ret_1m"),
                    pl.col("close").pct_change(5).alias("ret_5m"),
                    pl.col("close").pct_change(30).alias("ret_30m"),
                    pl.col("close").pct_change(60).alias("ret_60m"),
                    # Volume features
                    pl.col("volume").rolling_mean(window_size=30).alias("volume_ma30"),
                    (pl.col("volume") / pl.col("volume").rolling_mean(window_size=30)).alias(
                        "volume_ratio",
                    ),
                    # Price features
                    (pl.col("close") / pl.col("close").rolling_mean(window_size=60)).alias(
                        "price_ratio_60m",
                    ),
                    (pl.col("high") - pl.col("low")).alias("range"),
                    ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
                    # Microstructure
                    pl.col("bid_ask_spread_weighted")
                    .rolling_mean(window_size=30)
                    .alias("spread_ma30"),
                    pl.col("effective_spread")
                    .rolling_mean(window_size=30)
                    .alias("eff_spread_ma30"),
                    # Technical indicators
                    qf.ta.rsi(pl.col("close"), 14).alias("rsi_14"),
                    qf.ta.adx(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("adx_14"),
                    # Volatility
                    pl.col("ret_1m").rolling_std(window_size=30).alias("vol_30m"),
                    # Order flow
                    (pl.col("volume_at_ask") - pl.col("volume_at_bid")).alias("order_flow"),
                    ((pl.col("volume_at_ask") - pl.col("volume_at_bid")) / pl.col("volume")).alias(
                        "order_flow_ratio",
                    ),
                ],
            )

            features_list.append(ticker_features)

        # Combine all features
        self.features = pl.concat(features_list).sort(["bar_start", "ticker"])

        return self.features

    def add_cross_sectional_features(self) -> pl.DataFrame:
        """Add cross-sectional features."""
        print("Adding cross-sectional features...")

        # Group by timestamp for cross-sectional calculations
        cs_features = []

        for timestamp in self.features["bar_start"].unique():
            # Get cross-section at this timestamp
            cs_data = self.features.filter(pl.col("bar_start") == timestamp)

            if len(cs_data) < 20:  # Need minimum stocks
                continue

            # Calculate cross-sectional statistics
            cs_stats = cs_data.with_columns(
                [
                    # Rank features
                    pl.col("ret_5m").rank(method="average").alias("ret_5m_rank"),
                    pl.col("ret_30m").rank(method="average").alias("ret_30m_rank"),
                    pl.col("volume_ratio").rank(method="average").alias("volume_rank"),
                    pl.col("rsi_14").rank(method="average").alias("rsi_rank"),
                    # Z-scores
                    ((pl.col("ret_5m") - pl.col("ret_5m").mean()) / pl.col("ret_5m").std()).alias(
                        "ret_5m_zscore",
                    ),
                    (
                        (pl.col("volume_ratio") - pl.col("volume_ratio").mean())
                        / pl.col("volume_ratio").std()
                    ).alias("volume_zscore"),
                    # Percentiles
                    pl.col("ret_5m").quantile(0.2).alias("ret_5m_p20"),
                    pl.col("ret_5m").quantile(0.8).alias("ret_5m_p80"),
                ],
            )

            # Add market-wide statistics
            market_stats = {
                "market_ret_5m": cs_data["ret_5m"].mean(),
                "market_vol_30m": cs_data["vol_30m"].mean(),
                "market_spread": cs_data["spread_ma30"].mean(),
                "n_stocks": len(cs_data),
            }

            for key, value in market_stats.items():
                cs_stats = cs_stats.with_columns(pl.lit(value).alias(key))

            cs_features.append(cs_stats)

        # Combine cross-sectional features
        self.features = pl.concat(cs_features).sort(["bar_start", "ticker"])

        # Add relative features
        self.features = self.features.with_columns(
            [
                (pl.col("ret_5m") - pl.col("market_ret_5m")).alias("ret_5m_relative"),
                (pl.col("vol_30m") / pl.col("market_vol_30m")).alias("vol_relative"),
            ],
        )

        return self.features

    def add_sector_neutral_features(self) -> pl.DataFrame:
        """Create sector-neutral versions of key features."""
        print("Creating sector-neutral features...")

        # Group by timestamp and sector
        sector_neutral_features = []

        for timestamp in self.features["bar_start"].unique():
            ts_data = self.features.filter(pl.col("bar_start") == timestamp)

            for sector in ts_data["sector"].unique():
                if sector is None:
                    continue

                sector_data = ts_data.filter(pl.col("sector") == sector)

                if len(sector_data) < 3:  # Need minimum stocks per sector
                    continue

                # Calculate sector-neutral features
                sector_data = sector_data.with_columns(
                    [
                        # Sector-relative returns
                        (pl.col("ret_5m") - pl.col("ret_5m").mean()).alias("ret_5m_sector_neutral"),
                        (pl.col("ret_30m") - pl.col("ret_30m").mean()).alias(
                            "ret_30m_sector_neutral",
                        ),
                        # Sector z-scores
                        (
                            (pl.col("volume_ratio") - pl.col("volume_ratio").mean())
                            / pl.col("volume_ratio").std()
                        ).alias("volume_sector_zscore"),
                    ],
                )

                sector_neutral_features.append(sector_data)

        # Combine sector-neutral features
        self.features = pl.concat(sector_neutral_features).sort(["bar_start", "ticker"])

        return self.features


class FactorPortfolioStrategy:
    """Multi-factor portfolio strategy with proper risk controls."""

    def __init__(self, features: pl.DataFrame):
        self.features = features
        self.factor_weights = {
            "momentum": 0.3,
            "value": 0.2,
            "quality": 0.2,
            "low_vol": 0.15,
            "reversal": 0.15,
        }

    def calculate_factor_scores(self) -> pl.DataFrame:
        """Calculate composite factor scores."""
        print("Calculating factor scores...")

        # Define factor compositions
        factor_scores = self.features.with_columns(
            [
                # Momentum: Recent returns and RSI
                (pl.col("ret_30m_rank") * 0.6 + pl.col("rsi_rank") * 0.4).alias("momentum_score"),
                # Value: Price ratios (inverted for value)
                (1 - pl.col("price_ratio_60m").rank(method="average") / pl.col("n_stocks")).alias(
                    "value_score",
                ),
                # Quality: Low volatility and tight spreads
                (
                    1
                    - pl.col("vol_30m").rank(method="average") / pl.col("n_stocks") * 0.5
                    + 1
                    - pl.col("spread_ma30").rank(method="average") / pl.col("n_stocks") * 0.5
                ).alias("quality_score"),
                # Low volatility
                (1 - pl.col("vol_30m").rank(method="average") / pl.col("n_stocks")).alias(
                    "low_vol_score",
                ),
                # Short-term reversal
                (1 - pl.col("ret_5m_rank") / pl.col("n_stocks")).alias("reversal_score"),
            ],
        )

        # Calculate composite score
        composite_score = pl.lit(0.0)
        for factor, weight in self.factor_weights.items():
            composite_score = composite_score + pl.col(f"{factor}_score") * weight

        factor_scores = factor_scores.with_columns(
            composite_score.alias("composite_score"),
        )

        # Normalize to portfolio weights (long-short)
        factor_scores = factor_scores.with_columns(
            [
                # Demean scores
                (pl.col("composite_score") - pl.col("composite_score").mean()).alias(
                    "score_demeaned",
                ),
            ],
        )

        # Calculate portfolio weights
        factor_scores = factor_scores.with_columns(
            [
                # Scale to sum to zero (market neutral)
                (pl.col("score_demeaned") / pl.col("score_demeaned").abs().sum()).alias(
                    "portfolio_weight",
                ),
            ],
        )

        return factor_scores

    def apply_risk_controls(self, scores: pl.DataFrame) -> pl.DataFrame:
        """Apply position limits and sector constraints."""

        # Position limits (max 5% per stock)
        scores = scores.with_columns(
            [
                pl.when(pl.col("portfolio_weight") > 0.05)
                .then(0.05)
                .when(pl.col("portfolio_weight") < -0.05)
                .then(-0.05)
                .otherwise(pl.col("portfolio_weight"))
                .alias("portfolio_weight_limited"),
            ],
        )

        # Sector neutrality adjustment
        sector_weights = scores.group_by(["bar_start", "sector"]).agg(
            [
                pl.col("portfolio_weight_limited").sum().alias("sector_weight"),
            ],
        )

        scores = scores.join(
            sector_weights,
            on=["bar_start", "sector"],
            how="left",
        )

        # Adjust weights to be sector neutral
        scores = scores.with_columns(
            [
                (
                    pl.col("portfolio_weight_limited")
                    - pl.col("sector_weight") / pl.col("sector").count().over("bar_start")
                ).alias("final_weight"),
            ],
        )

        return scores


def backtest_factor_strategy(features_with_weights: pl.DataFrame):
    """Backtest the multi-factor strategy using ml4t.evaluation."""

    print("\nRunning multi-factor strategy backtest...")

    # Prepare data for backtesting
    # Select key features
    feature_cols = [
        "momentum_score",
        "value_score",
        "quality_score",
        "low_vol_score",
        "reversal_score",
        "composite_score",
        "ret_5m_sector_neutral",
        "ret_30m_sector_neutral",
        "volume_sector_zscore",
        "vol_relative",
    ]

    # Create forward returns for labeling
    features_with_weights = features_with_weights.sort(["ticker", "bar_start"])
    features_with_weights = features_with_weights.with_columns(
        [
            pl.col("close").shift(-30).alias("close_future"),
        ],
    )
    features_with_weights = features_with_weights.with_columns(
        [
            ((pl.col("close_future") - pl.col("close")) / pl.col("close")).alias(
                "forward_return_30m",
            ),
        ],
    )

    # Apply labeling based on forward returns
    features_with_weights = features_with_weights.with_columns(
        [
            pl.when(pl.col("forward_return_30m") > 0.001)
            .then(1)
            .when(pl.col("forward_return_30m") < -0.001)
            .then(-1)
            .otherwise(0)
            .alias("label"),
        ],
    )

    # Filter valid data
    valid_data = features_with_weights.drop_nulls(
        subset=[*feature_cols, "label", "forward_return_30m"],
    )

    # Prepare for qeval
    X = valid_data.select(feature_cols)
    y = valid_data.select("label")
    groups = valid_data.select("ticker")  # For group-aware CV
    timestamps = valid_data.select("bar_start")
    weights = valid_data.select("final_weight")
    returns = valid_data.select("forward_return_30m")

    # Configure Combinatorial Purged CV
    cv = CombinatorialPurgedKFold(
        n_folds=10,
        n_test_folds=2,
        purge_window=timedelta(minutes=30),  # 30-minute purge
        embargo_window=timedelta(minutes=5),
        seed=42,
    )

    # Create evaluator with comprehensive metrics
    evaluator = Evaluator(
        tier=Tier.TIER_1,  # Most rigorous
        splitter=cv,
        metrics=[
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "information_coefficient",
            "max_drawdown",
            "turnover",
            "hit_rate",
        ],
        confidence_level=0.95,
        n_bootstrap=1000,
    )

    # Use portfolio weights directly (no ML model needed)
    class FactorPortfolio:
        def __init__(self, weights):
            self.weights = weights

        def predict(self, X_idx):
            # Return pre-calculated weights
            return self.weights.iloc[X_idx].to_numpy()

    portfolio = FactorPortfolio(weights)

    # Run backtest
    results = evaluator.evaluate(
        X,
        y,
        portfolio,
        groups=groups,
        timestamps=timestamps,
        returns=returns,
    )

    # Calculate Deflated Sharpe Ratio
    n_trials = len(cv.get_n_splits())  # Number of CPCV paths
    dsr = deflated_sharpe_ratio(
        results.metrics["sharpe_ratio"]["mean"],
        results.metrics["sharpe_ratio"]["std"],
        n_trials=n_trials,
    )

    print("\n=== Multi-Factor Strategy Results ===")
    print(results.summary())
    print(f"\nDeflated Sharpe Ratio: {dsr:.3f}")
    print(f"Probability of Skill: {1 - dsr:.2%}")

    # Generate comprehensive tearsheet
    create_factor_tearsheet(results, output_path=OUTPUT_PATH)

    return results, dsr


def analyze_factor_exposures(features_with_scores: pl.DataFrame):
    """Analyze factor exposures and correlations."""

    print("\nAnalyzing factor exposures...")

    # Calculate factor returns
    factor_returns = {}

    for factor in ["momentum", "value", "quality", "low_vol", "reversal"]:
        # Create factor portfolio
        factor_data = features_with_scores.with_columns(
            [
                pl.when(pl.col(f"{factor}_score") > pl.col(f"{factor}_score").quantile(0.8))
                .then(1)
                .when(pl.col(f"{factor}_score") < pl.col(f"{factor}_score").quantile(0.2))
                .then(-1)
                .otherwise(0)
                .alias("factor_position"),
            ],
        )

        # Calculate factor returns
        factor_data = factor_data.group_by("bar_start").agg(
            [
                (pl.col("forward_return_30m") * pl.col("factor_position"))
                .mean()
                .alias("factor_return"),
            ],
        )

        factor_returns[factor] = factor_data

    # Create correlation matrix
    returns_df = pl.DataFrame(
        {
            "timestamp": factor_returns["momentum"]["bar_start"],
            **{f"{factor}_return": df["factor_return"] for factor, df in factor_returns.items()},
        },
    )

    # Plot correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Factor correlation
    corr_matrix = returns_df.select(
        [col for col in returns_df.columns if col.endswith("_return")],
    ).corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax1)
    ax1.set_title("Factor Return Correlations")

    # Cumulative factor returns
    for factor in factor_returns:
        cumulative = (1 + factor_returns[factor]["factor_return"]).cumprod()
        ax2.plot(factor_returns[factor]["bar_start"], cumulative, label=factor)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title("Factor Performance Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "factor_analysis.png", dpi=300)
    plt.close()


def sector_performance_analysis(features_with_scores: pl.DataFrame):
    """Analyze performance by sector."""

    print("\nAnalyzing sector performance...")

    # Calculate sector returns
    sector_returns = features_with_scores.group_by(["bar_start", "sector"]).agg(
        [
            pl.col("forward_return_30m").mean().alias("sector_return"),
            pl.col("final_weight").sum().alias("sector_weight"),
            pl.count("ticker").alias("n_stocks"),
        ],
    )

    # Plot sector exposures over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Sector weights over time
    for sector in sector_returns["sector"].unique():
        if sector is None:
            continue
        sector_data = sector_returns.filter(pl.col("sector") == sector)
        ax1.plot(sector_data["bar_start"], sector_data["sector_weight"], label=sector)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Portfolio Weight")
    ax1.set_title("Sector Exposures Over Time")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Average sector returns
    avg_sector_returns = (
        sector_returns.group_by("sector")
        .agg(
            [
                pl.col("sector_return").mean().alias("avg_return"),
                pl.col("sector_return").std().alias("return_std"),
            ],
        )
        .sort("avg_return", descending=True)
    )

    sectors = avg_sector_returns["sector"]
    returns = avg_sector_returns["avg_return"] * 10000  # Convert to bps
    stds = avg_sector_returns["return_std"] * 10000

    ax2.bar(sectors, returns, yerr=stds, capsize=5)
    ax2.set_xlabel("Sector")
    ax2.set_ylabel("Average Return (bps)")
    ax2.set_title("Average 30-Minute Returns by Sector")
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "sector_analysis.png", dpi=300)
    plt.close()


def create_execution_analysis(features_with_scores: pl.DataFrame):
    """Analyze execution and turnover characteristics."""

    print("\nAnalyzing execution characteristics...")

    # Calculate turnover
    features_sorted = features_with_scores.sort(["ticker", "bar_start"])
    features_sorted = features_sorted.with_columns(
        [
            pl.col("final_weight").shift(1).over("ticker").alias("prev_weight"),
        ],
    )

    turnover = features_sorted.group_by("bar_start").agg(
        [
            (pl.col("final_weight") - pl.col("prev_weight")).abs().sum().alias("turnover"),
        ],
    )

    # Calculate spread costs
    spread_costs = (
        features_sorted.with_columns(
            [
                (pl.col("final_weight") - pl.col("prev_weight")).abs()
                * pl.col("spread_ma30")
                / 10000,  # Convert to decimal
            ],
        )
        .group_by("bar_start")
        .agg(
            [
                pl.col("spread_costs").sum().alias("total_spread_cost"),
            ],
        )
    )

    # Plot execution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Turnover over time
    axes[0, 0].plot(turnover["bar_start"], turnover["turnover"] * 100)
    axes[0, 0].set_ylabel("Turnover (%)")
    axes[0, 0].set_title("Portfolio Turnover Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Spread costs
    axes[0, 1].plot(spread_costs["bar_start"], spread_costs["total_spread_cost"] * 10000)
    axes[0, 1].set_ylabel("Spread Cost (bps)")
    axes[0, 1].set_title("Estimated Trading Costs")
    axes[0, 1].grid(True, alpha=0.3)

    # Position distribution
    weights = features_with_scores["final_weight"]
    axes[1, 0].hist(weights, bins=50, alpha=0.7)
    axes[1, 0].set_xlabel("Portfolio Weight")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Position Size Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Number of positions
    n_positions = features_sorted.group_by("bar_start").agg(
        [
            pl.col("final_weight")
            .filter(pl.col("final_weight").abs() > 0.001)
            .count()
            .alias("n_long"),
            pl.col("final_weight").filter(pl.col("final_weight") < -0.001).count().alias("n_short"),
        ],
    )

    axes[1, 1].plot(n_positions["bar_start"], n_positions["n_long"], label="Long", color="green")
    axes[1, 1].plot(n_positions["bar_start"], n_positions["n_short"], label="Short", color="red")
    axes[1, 1].set_ylabel("Number of Positions")
    axes[1, 1].set_title("Long/Short Position Counts")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "execution_analysis.png", dpi=300)
    plt.close()


def main():
    """Run the complete cross-sectional analysis example."""

    print("=== Multi-Asset Cross-Sectional Analysis ===\n")

    # 1. Load NASDAQ 100 data
    print("Step 1: Loading NASDAQ 100 minute data...")
    loader = NASDAQ100DataLoader(year=2021)
    data = loader.load_data()

    # Sample first month for faster execution
    data = data.filter(
        (pl.col("bar_start") >= datetime(2021, 1, 4))
        & (pl.col("bar_start") < datetime(2021, 2, 1)),
    )

    print(f"Using {len(data):,} minute bars for {data['ticker'].n_unique()} stocks")

    # 2. Create features
    print("\nStep 2: Engineering features...")
    engineer = CrossSectionalFeatureEngineer(data)
    features = engineer.create_base_features()
    features = engineer.add_cross_sectional_features()
    features = engineer.add_sector_neutral_features()

    print(f"Created {len(features.columns)} features")

    # 3. Calculate factor scores
    print("\nStep 3: Calculating factor scores...")
    strategy = FactorPortfolioStrategy(features)
    factor_scores = strategy.calculate_factor_scores()
    factor_scores = strategy.apply_risk_controls(factor_scores)

    # 4. Backtest strategy
    print("\nStep 4: Backtesting multi-factor strategy...")
    results, dsr = backtest_factor_strategy(factor_scores)

    # 5. Analyze results
    print("\nStep 5: Analyzing results...")
    analyze_factor_exposures(factor_scores)
    sector_performance_analysis(factor_scores)
    create_execution_analysis(factor_scores)

    # 6. Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Universe: {data['ticker'].n_unique()} stocks")
    print(f"Time Period: {data['bar_start'].min()} to {data['bar_start'].max()}")
    print(f"Total observations: {len(data):,}")
    print(
        f"Average stocks per timestamp: {features.group_by('bar_start').count()['count'].mean():.1f}",
    )

    if results:
        print("\nStrategy Performance:")
        print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']['mean']:.3f}")
        print(f"Deflated Sharpe Ratio: {dsr:.3f}")
        print(f"Information Coefficient: {results.metrics['information_coefficient']['mean']:.4f}")
        print(f"Maximum Drawdown: {results.metrics['max_drawdown']['mean']:.2%}")
        print(f"Average Turnover: {results.metrics['turnover']['mean']:.1%}")
        print(f"Hit Rate: {results.metrics['hit_rate']['mean']:.1%}")

    print("\n✅ Cross-sectional analysis complete!")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

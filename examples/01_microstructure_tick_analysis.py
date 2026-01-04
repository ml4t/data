"""
Example 1: Advanced Microstructure Analysis with Tick Data
==========================================================

This example demonstrates how to extract sophisticated market microstructure
features from tick data using qfeatures and validate trading signals with ml4t.evaluation.

Key Concepts:
- Processing high-frequency tick data efficiently
- Calculating microstructure indicators (VPIN, Kyle's Lambda, Roll Spread)
- Detecting toxic order flow and liquidity events
- Proper backtesting with microsecond-precision data

Author: QuantLab Team
Date: August 2025
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml4t.engineer.bars import DollarBar, VolumeBar
from ml4t.engineer.features import microstructure as ms
from ml4t.engineer.labeling import BarrierConfig, triple_barrier_labels

# Import qeval modules
# Import qfeatures modules
from ml4t.evaluation import Evaluator, Tier
from ml4t.evaluation.splitters import PurgedWalkForwardCV

# Configure paths
DATA_PATH = Path.home() / "ml3t/data/equities/data_bento/parquet/SPY"
OUTPUT_PATH = Path("./output/microstructure_analysis")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class TickDataProcessor:
    """Process raw tick data into analysis-ready format."""

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.tick_data = None
        self.trades = None
        self.quotes = None

    def load_tick_data(self, date: str) -> pl.DataFrame:
        """Load tick data for a specific date."""
        filename = f"xnas-itch-{date}.mbo.dbn.parquet"
        filepath = DATA_PATH / filename

        print(f"Loading tick data from {filepath}")
        self.tick_data = pl.read_parquet(filepath)

        # Separate trades and quotes
        self.trades = self.tick_data.filter(
            pl.col("action").is_in(["T", "F"])  # Trade and Fill
        )

        self.quotes = self.tick_data.filter(
            pl.col("action").is_in(["A", "R", "C"])  # Add, Replace, Cancel
        )

        print(f"Loaded {len(self.tick_data):,} tick records")
        print(f"  - {len(self.trades):,} trades")
        print(f"  - {len(self.quotes):,} quotes")

        return self.tick_data

    def aggregate_to_bars(self, bar_type: str = "time", threshold: int = 60) -> pl.DataFrame:
        """Aggregate tick data into bars."""
        if bar_type == "time":
            # Aggregate to 1-minute bars
            bars = self.trades.group_by_dynamic("ts_event", every=f"{threshold}s").agg(
                [
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("size").sum().alias("volume"),
                    pl.col("price").count().alias("trade_count"),
                ]
            )

        elif bar_type == "volume":
            # Create volume bars
            bar_builder = VolumeBar(threshold=threshold)
            bars = bar_builder.build(self.trades)

        elif bar_type == "dollar":
            # Create dollar bars
            bar_builder = DollarBar(threshold=threshold)
            bars = bar_builder.build(self.trades)

        return bars


class MicrostructureFeatureEngineer:
    """Generate advanced microstructure features."""

    def __init__(self, tick_processor: TickDataProcessor):
        self.processor = tick_processor
        self.features = None

    def calculate_order_flow_features(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Calculate order flow and toxicity metrics."""

        # 1. Trade Classification (Lee-Ready algorithm)
        # Classify trades as buyer or seller initiated
        trades_classified = self.processor.trades.with_columns(
            [
                # Simple classification based on trade location vs quotes
                pl.when(pl.col("side") == "B")
                .then(1)
                .when(pl.col("side") == "A")
                .then(-1)
                .otherwise(0)
                .alias("trade_sign")
            ]
        )

        # 2. Order Flow Imbalance
        ofi = (
            trades_classified.group_by_dynamic("ts_event", every="60s")
            .agg(
                [
                    (pl.col("size") * pl.col("trade_sign")).sum().alias("signed_volume"),
                    pl.col("size").sum().alias("total_volume"),
                ]
            )
            .with_columns(
                (pl.col("signed_volume") / pl.col("total_volume")).alias("order_flow_imbalance")
            )
        )

        # 3. VPIN (Volume-Synchronized Probability of Informed Trading)
        # Using qfeatures implementation
        vpin_bars = bars.with_columns(
            [
                ms.vpin(
                    "volume",
                    pl.lit(1).alias("trade_sign"),  # Simplified for example
                    bucket_size=50,
                ).alias("vpin")
            ]
        )

        # 4. Kyle's Lambda (Price Impact)
        kyle_lambda = bars.with_columns(
            [ms.kyle_lambda("close", "volume", "signed_volume", window=20).alias("kyle_lambda")]
        )

        # 5. Amihud Illiquidity
        amihud = bars.with_columns(
            [ms.amihud_illiquidity("returns", "volume", window=20).alias("amihud")]
        )

        # Combine all features
        features = bars.with_columns(
            [
                pl.col("close").pct_change().alias("returns"),
            ]
        )

        # Add microstructure features
        for df, cols in [
            (ofi, ["order_flow_imbalance"]),
            (vpin_bars, ["vpin"]),
            (kyle_lambda, ["kyle_lambda"]),
            (amihud, ["amihud"]),
        ]:
            for col in cols:
                if col in df.columns:
                    features = features.join(
                        df.select(["ts_event", col]), on="ts_event", how="left"
                    )

        return features

    def calculate_spread_metrics(self) -> pl.DataFrame:
        """Calculate bid-ask spread metrics."""

        # Get best bid and ask at each timestamp
        best_quotes = self.processor.quotes.group_by("ts_event").agg(
            [
                pl.col("price").filter(pl.col("side") == "B").max().alias("best_bid"),
                pl.col("price").filter(pl.col("side") == "A").min().alias("best_ask"),
            ]
        )

        # Calculate spreads
        spreads = best_quotes.with_columns(
            [
                (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
                (
                    (pl.col("best_ask") - pl.col("best_bid"))
                    / ((pl.col("best_ask") + pl.col("best_bid")) / 2)
                ).alias("spread_pct"),
            ]
        )

        # Roll's implicit spread estimator
        trade_prices = self.processor.trades.select(["ts_event", "price"])
        roll_spread = ms.roll_measure(trade_prices["price"], window=100)

        return spreads, roll_spread


class MicrostructureAlphaStrategy:
    """Trading strategy based on microstructure signals."""

    def __init__(self, features: pl.DataFrame):
        self.features = features
        self.signals = None
        self.labels = None

    def generate_signals(self) -> pl.DataFrame:
        """Generate trading signals from microstructure features."""

        # Signal 1: VPIN spike indicates informed trading
        vpin_signal = (
            pl.when(pl.col("vpin") > pl.col("vpin").quantile(0.9))
            .then(-1)  # Adverse selection, avoid trading
            .otherwise(0)
        )

        # Signal 2: Order flow persistence
        ofi_signal = (
            pl.when(pl.col("order_flow_imbalance").rolling_mean(5) > 0.2)
            .then(1)  # Follow the flow
            .when(pl.col("order_flow_imbalance").rolling_mean(5) < -0.2)
            .then(-1)
            .otherwise(0)
        )

        # Signal 3: Liquidity provision opportunity
        spread_signal = (
            pl.when(pl.col("spread_pct") > pl.col("spread_pct").quantile(0.8))
            .then(1)  # Wide spread, provide liquidity
            .otherwise(0)
        )

        # Combine signals
        self.signals = self.features.with_columns(
            [
                vpin_signal.alias("vpin_signal"),
                ofi_signal.alias("ofi_signal"),
                spread_signal.alias("spread_signal"),
                # Composite signal
                (vpin_signal * 0.3 + ofi_signal * 0.5 + spread_signal * 0.2).alias(
                    "composite_signal"
                ),
            ]
        )

        return self.signals

    def apply_labeling(self, horizon_minutes: int = 5) -> pl.DataFrame:
        """Apply triple-barrier labeling for strategy evaluation."""

        # Calculate recent volatility for dynamic barriers
        recent_vol = self.signals["returns"].std() * np.sqrt(252 * 390)  # Annualized

        # Configure barriers
        config = BarrierConfig(
            upper_barrier=2 * recent_vol / np.sqrt(252 * 390 / horizon_minutes),
            lower_barrier=-2 * recent_vol / np.sqrt(252 * 390 / horizon_minutes),
            max_holding_period=horizon_minutes,
        )

        # Apply labeling
        self.labels = triple_barrier_labels(
            self.signals, config, price_col="close", timestamp_col="ts_event"
        )

        return self.labels


def backtest_microstructure_strategy(labeled_data: pl.DataFrame):
    """Backtest the microstructure strategy using ml4t.evaluation."""

    # Select features for the model
    feature_cols = [
        "vpin",
        "order_flow_imbalance",
        "kyle_lambda",
        "amihud",
        "spread_pct",
        "composite_signal",
    ]

    # Prepare data
    valid_data = labeled_data.drop_nulls(subset=[*feature_cols, "label"])

    if len(valid_data) < 1000:
        print("Warning: Limited data for backtesting")
        return None

    x = valid_data.select(feature_cols)
    y = valid_data.select("label")
    timestamps = valid_data.select("ts_event")

    # Configure evaluation with high-frequency considerations
    evaluator = Evaluator(
        tier=Tier.TIER_1,  # Rigorous backtesting
        splitter=PurgedWalkForwardCV(
            n_splits=10,
            test_size=0.1,
            label_horizon=timedelta(minutes=5),
            embargo_size=timedelta(minutes=1),
            expanding=False,  # Rolling window for intraday
        ),
        metrics=["sharpe_ratio", "information_coefficient", "max_drawdown", "turnover", "hit_rate"],
        confidence_level=0.95,
    )

    # Simple signal-based strategy (no ML model needed)
    # In practice, you might use XGBoost or similar
    class SignalStrategy:
        def predict(self, x):
            return np.sign(x["composite_signal"].to_numpy())

    strategy = SignalStrategy()

    # Run backtest
    results = evaluator.evaluate(x, y, strategy, timestamps=timestamps)

    # Generate detailed report
    print("\n=== Microstructure Strategy Backtest Results ===")
    print(results.summary())

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Cumulative returns by signal component
    results.plot_cumulative_returns(ax=axes[0, 0], by_signal=True)
    axes[0, 0].set_title("Cumulative Returns by Signal Component")

    # 2. Rolling Sharpe ratio
    results.plot_rolling_metric("sharpe_ratio", window=100, ax=axes[0, 1])
    axes[0, 1].set_title("Rolling Sharpe Ratio (100-bar window)")

    # 3. Feature importance
    results.plot_feature_importance(ax=axes[1, 0])
    axes[1, 0].set_title("Microstructure Feature Importance")

    # 4. Turnover analysis
    results.plot_turnover_decay(ax=axes[1, 1])
    axes[1, 1].set_title("Signal Turnover and Decay")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "microstructure_backtest_results.png", dpi=300)
    plt.close()

    return results


def analyze_intraday_patterns(features: pl.DataFrame):
    """Analyze intraday patterns in microstructure features."""

    # Add time-of-day features
    features_tod = features.with_columns(
        [
            pl.col("ts_event").dt.hour().alias("hour"),
            pl.col("ts_event").dt.minute().alias("minute"),
            (pl.col("ts_event").dt.hour() * 60 + pl.col("ts_event").dt.minute()).alias(
                "minute_of_day"
            ),
        ]
    )

    # Group by time of day
    tod_patterns = features_tod.group_by("minute_of_day").agg(
        [
            pl.col("vpin").mean().alias("avg_vpin"),
            pl.col("spread_pct").mean().alias("avg_spread"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("order_flow_imbalance").mean().alias("avg_ofi"),
        ]
    )

    # Plot intraday patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Convert minute_of_day to time for x-axis
    tod_patterns = tod_patterns.sort("minute_of_day")
    time_labels = [f"{m // 60:02d}:{m % 60:02d}" for m in tod_patterns["minute_of_day"]]

    # VPIN pattern
    axes[0, 0].plot(tod_patterns["avg_vpin"])
    axes[0, 0].set_title("Intraday VPIN Pattern")
    axes[0, 0].set_xlabel("Time of Day")
    axes[0, 0].set_ylabel("Average VPIN")

    # Spread pattern
    axes[0, 1].plot(tod_patterns["avg_spread"])
    axes[0, 1].set_title("Intraday Spread Pattern")
    axes[0, 1].set_xlabel("Time of Day")
    axes[0, 1].set_ylabel("Average Spread %")

    # Volume pattern
    axes[1, 0].plot(tod_patterns["avg_volume"])
    axes[1, 0].set_title("Intraday Volume Pattern")
    axes[1, 0].set_xlabel("Time of Day")
    axes[1, 0].set_ylabel("Average Volume")

    # Order flow pattern
    axes[1, 1].plot(tod_patterns["avg_ofi"])
    axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Intraday Order Flow Imbalance")
    axes[1, 1].set_xlabel("Time of Day")
    axes[1, 1].set_ylabel("Average OFI")

    # Set x-axis labels (show every 30 minutes)
    for ax in axes.flat:
        ax.set_xticks(range(0, len(time_labels), 30))
        ax.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), 30)], rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "intraday_microstructure_patterns.png", dpi=300)
    plt.close()


def main():
    """Run the complete microstructure analysis example."""

    print("=== Microstructure Analysis with Tick Data ===\n")

    # 1. Load and process tick data
    print("Step 1: Loading tick data...")
    processor = TickDataProcessor("SPY")
    processor.load_tick_data("20240903")

    # 2. Aggregate to different bar types
    print("\nStep 2: Creating bars...")
    time_bars = processor.aggregate_to_bars("time", 60)  # 1-minute bars
    volume_bars = processor.aggregate_to_bars("volume", 50000)  # 50K share bars
    dollar_bars = processor.aggregate_to_bars("dollar", 10_000_000)  # $10M bars

    print(f"Created {len(time_bars)} time bars")
    print(f"Created {len(volume_bars)} volume bars")
    print(f"Created {len(dollar_bars)} dollar bars")

    # 3. Calculate microstructure features
    print("\nStep 3: Engineering microstructure features...")
    engineer = MicrostructureFeatureEngineer(processor)
    features = engineer.calculate_order_flow_features(time_bars)
    spreads, roll_spread = engineer.calculate_spread_metrics()

    # Add spread metrics to features
    features = features.join(spreads.select(["ts_event", "spread_pct"]), on="ts_event", how="left")

    # 4. Generate trading signals
    print("\nStep 4: Generating trading signals...")
    strategy = MicrostructureAlphaStrategy(features)
    strategy.generate_signals()

    # 5. Apply labeling
    print("\nStep 5: Applying triple-barrier labeling...")
    labeled_data = strategy.apply_labeling(horizon_minutes=5)

    # 6. Backtest strategy
    print("\nStep 6: Backtesting microstructure strategy...")
    results = backtest_microstructure_strategy(labeled_data)

    # 7. Analyze patterns
    print("\nStep 7: Analyzing intraday patterns...")
    analyze_intraday_patterns(features)

    # 8. Generate summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total trades analyzed: {len(processor.trades):,}")
    print(f"Average VPIN: {features['vpin'].mean():.4f}")
    print(f"Average spread: {features['spread_pct'].mean() * 10000:.2f} bps")
    print(f"Order flow imbalance std: {features['order_flow_imbalance'].std():.4f}")

    if results:
        print(f"\nStrategy Sharpe Ratio: {results.metrics['sharpe_ratio']['mean']:.3f}")
        print(f"Information Coefficient: {results.metrics['information_coefficient']['mean']:.4f}")
        print(f"Maximum Drawdown: {results.metrics['max_drawdown']['mean']:.2%}")
        print(f"Average Turnover: {results.metrics['turnover']['mean']:.2%}")

    print("\n✅ Microstructure analysis complete!")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

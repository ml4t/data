"""COT + OHLCV Integration Workflow.

Combines daily futures OHLCV data with weekly COT positioning data.

PUBLICATION TIMING (Critical for Point-in-Time Modeling):
    - Positions snapshot: Tuesday close of business
    - CFTC publishes: Friday 3:30 PM ET (3-day lag)
    - First tradeable: Friday 3:30 PM ET or Monday open

    For backtesting, COT data from report_date X should NOT be used until
    at least Friday of that week. Using it earlier introduces look-ahead bias.

    Timeline:
        Mon  Tue  Wed  Thu  Fri  Sat  Sun  Mon
              [positions]      [published]
                               ↑
                         Data available here

Usage:
    from ml4t.data.cot import COTFetcher, combine_cot_ohlcv, create_cot_features

    # Combine daily OHLCV with weekly COT
    combined = combine_cot_ohlcv(ohlcv_df, cot_df)

    # Create ML features from COT data
    features = create_cot_features(combined)

    # For point-in-time correct backtesting, shift COT by publication lag:
    # report_date + 3 days = publication date (Friday)
    # report_date + 6 days = conservative (Monday after publication)
"""

from __future__ import annotations

import polars as pl


def combine_cot_ohlcv(
    ohlcv: pl.DataFrame,
    cot: pl.DataFrame,
    date_col: str = "timestamp",
    cot_date_col: str = "report_date",
) -> pl.DataFrame:
    """Combine daily OHLCV with weekly COT data.

    COT data is released weekly (Tuesday positions, released Friday 3:30 PM ET).
    This function forward-fills COT data to align with daily OHLCV.

    WARNING - Point-in-Time Consideration:
        This function does a simple asof join on report_date, which does NOT
        account for publication lag. For backtesting, you should either:
        1. Shift report_date forward by 3-6 days before joining
        2. Filter combined data to only use COT after publication date

        Example (conservative approach - available Monday after publication):
            cot = cot.with_columns(
                (pl.col("report_date") + pl.duration(days=6)).alias("available_date")
            )
            # Then join on available_date instead of report_date

    Args:
        ohlcv: Daily OHLCV DataFrame with timestamp column
        cot: Weekly COT DataFrame from COTFetcher
        date_col: Date column name in OHLCV data
        cot_date_col: Date column name in COT data

    Returns:
        Combined DataFrame with COT columns forward-filled to daily frequency

    Example:
        >>> ohlcv = storage.load("ES", provider="databento")
        >>> cot = fetcher.fetch_product("ES", start_year=2020)
        >>> combined = combine_cot_ohlcv(ohlcv, cot)
    """
    # Ensure date types are compatible
    ohlcv = ohlcv.with_columns(pl.col(date_col).cast(pl.Date).alias("_date"))
    cot = cot.with_columns(pl.col(cot_date_col).cast(pl.Date).alias("_date"))

    # Drop metadata columns from COT that we don't need for joining
    cot_cols_to_keep = [c for c in cot.columns if c not in ("product", "report_type", cot_date_col)]
    if "_date" not in cot_cols_to_keep:
        cot_cols_to_keep.append("_date")

    cot_subset = cot.select(cot_cols_to_keep)

    # Asof join: for each OHLCV date, get most recent COT data
    combined = ohlcv.join_asof(
        cot_subset,
        on="_date",
        strategy="backward",  # Get most recent COT report <= OHLCV date
    )

    # Clean up temporary column
    combined = combined.drop("_date")

    return combined


def create_cot_features(
    df: pl.DataFrame,
    prefix: str = "cot_",
) -> pl.DataFrame:
    """Create ML features from COT positioning data.

    Creates normalized and derived features suitable for ML models.

    Features created:
    - Net position as % of open interest
    - Z-scores of net positions (52-week lookback)
    - Change in positioning (1-week, 4-week)
    - Positioning extremes (percentile ranks)

    Args:
        df: DataFrame with COT columns (from combine_cot_ohlcv)
        prefix: Prefix for output column names

    Returns:
        DataFrame with additional COT feature columns

    Note:
        This function detects whether the data is financial futures (TFF)
        or commodity futures (disaggregated) based on available columns.
    """
    exprs = []

    # Detect report type based on columns
    is_financial = "dealer_net" in df.columns or "lev_money_net" in df.columns
    is_commodity = "commercial_net" in df.columns or "managed_money_net" in df.columns

    oi_col = "open_interest"

    # === Financial Futures Features (TFF) ===
    if is_financial:
        # Leveraged Money (hedge funds) - most predictive for financials
        if "lev_money_net" in df.columns and oi_col in df.columns:
            # Net as % of OI
            exprs.append(
                (pl.col("lev_money_net") / pl.col(oi_col) * 100).alias(f"{prefix}lev_money_pct_oi")
            )
            # Z-score (52-week)
            exprs.append(
                (
                    (pl.col("lev_money_net") - pl.col("lev_money_net").rolling_mean(52))
                    / pl.col("lev_money_net").rolling_std(52)
                ).alias(f"{prefix}lev_money_zscore_52w")
            )
            # 4-week change
            exprs.append(
                (pl.col("lev_money_net") - pl.col("lev_money_net").shift(4)).alias(
                    f"{prefix}lev_money_chg_4w"
                )
            )

        # Asset Managers (institutional) - contrarian signal
        if "asset_mgr_net" in df.columns and oi_col in df.columns:
            exprs.append(
                (pl.col("asset_mgr_net") / pl.col(oi_col) * 100).alias(f"{prefix}asset_mgr_pct_oi")
            )
            exprs.append(
                (
                    (pl.col("asset_mgr_net") - pl.col("asset_mgr_net").rolling_mean(52))
                    / pl.col("asset_mgr_net").rolling_std(52)
                ).alias(f"{prefix}asset_mgr_zscore_52w")
            )

        # Dealer (banks/swap dealers) - often hedging flow
        if "dealer_net" in df.columns and oi_col in df.columns:
            exprs.append(
                (pl.col("dealer_net") / pl.col(oi_col) * 100).alias(f"{prefix}dealer_pct_oi")
            )

    # === Commodity Futures Features (Disaggregated) ===
    if is_commodity:
        # Managed Money (hedge funds, CTAs) - trend followers
        if "managed_money_net" in df.columns and oi_col in df.columns:
            exprs.append(
                (pl.col("managed_money_net") / pl.col(oi_col) * 100).alias(
                    f"{prefix}managed_money_pct_oi"
                )
            )
            exprs.append(
                (
                    (pl.col("managed_money_net") - pl.col("managed_money_net").rolling_mean(52))
                    / pl.col("managed_money_net").rolling_std(52)
                ).alias(f"{prefix}managed_money_zscore_52w")
            )
            exprs.append(
                (pl.col("managed_money_net") - pl.col("managed_money_net").shift(4)).alias(
                    f"{prefix}managed_money_chg_4w"
                )
            )

        # Commercial (producers/hedgers) - informed flow, contrarian
        if "commercial_net" in df.columns and oi_col in df.columns:
            exprs.append(
                (pl.col("commercial_net") / pl.col(oi_col) * 100).alias(
                    f"{prefix}commercial_pct_oi"
                )
            )
            exprs.append(
                (
                    (pl.col("commercial_net") - pl.col("commercial_net").rolling_mean(52))
                    / pl.col("commercial_net").rolling_std(52)
                ).alias(f"{prefix}commercial_zscore_52w")
            )

    # === Universal Features ===
    # Non-reportables (small traders) - often contrarian indicator
    if "nonrept_net" in df.columns and oi_col in df.columns:
        exprs.append(
            (pl.col("nonrept_net") / pl.col(oi_col) * 100).alias(f"{prefix}nonrept_pct_oi")
        )

    # Open Interest change (market participation)
    if "oi_change" in df.columns and oi_col in df.columns:
        exprs.append((pl.col("oi_change") / pl.col(oi_col) * 100).alias(f"{prefix}oi_change_pct"))

    if not exprs:
        return df

    return df.with_columns(exprs)


def combine_cot_ohlcv_pit(
    ohlcv: pl.DataFrame,
    cot: pl.DataFrame,
    date_col: str = "timestamp",
    cot_date_col: str = "report_date",
    publication_lag_days: int = 6,
) -> pl.DataFrame:
    """Combine OHLCV with COT data, respecting point-in-time constraints.

    This is the RECOMMENDED function for backtesting. It accounts for the
    publication lag between when positions are measured (Tuesday) and when
    data becomes publicly available (Friday 3:30 PM ET).

    Publication Timeline:
        - Tuesday: CFTC measures trader positions (report_date)
        - Friday 3:30 PM ET: Data published (report_date + 3 days)
        - Monday: Conservative first use (report_date + 6 days)

    Args:
        ohlcv: Daily OHLCV DataFrame with timestamp column
        cot: Weekly COT DataFrame from COTFetcher
        date_col: Date column name in OHLCV data
        cot_date_col: Date column name in COT data
        publication_lag_days: Days after report_date when data is available.
            Default 6 (Monday after Friday publication) is conservative.
            Use 3 for Friday publication, 4 for Saturday availability.

    Returns:
        Combined DataFrame where COT features are only available after
        their publication date (no look-ahead bias).

    Example:
        >>> # For backtesting (point-in-time safe)
        >>> combined = combine_cot_ohlcv_pit(ohlcv, cot)
        >>>
        >>> # For analysis (ignore publication lag)
        >>> combined = combine_cot_ohlcv(ohlcv, cot)
    """
    # Shift report_date forward by publication lag
    cot_shifted = cot.with_columns(
        (pl.col(cot_date_col) + pl.duration(days=publication_lag_days)).alias("_available_date")
    )

    # Ensure date types are compatible
    ohlcv = ohlcv.with_columns(pl.col(date_col).cast(pl.Date).alias("_date"))
    cot_shifted = cot_shifted.with_columns(pl.col("_available_date").cast(pl.Date))

    # Drop metadata columns from COT that we don't need for joining
    cot_cols_to_keep = [
        c for c in cot_shifted.columns if c not in ("product", "report_type", cot_date_col)
    ]
    if "_available_date" not in cot_cols_to_keep:
        cot_cols_to_keep.append("_available_date")

    cot_subset = cot_shifted.select(cot_cols_to_keep)

    # Asof join using available_date (when data was actually published)
    combined = ohlcv.join_asof(
        cot_subset,
        left_on="_date",
        right_on="_available_date",
        strategy="backward",
    )

    # Clean up temporary columns
    combined = combined.drop(["_date", "_available_date"])

    return combined


def load_combined_futures_data(
    product: str,
    ohlcv_path: str = "~/ml4t-data/futures/ohlcv-1d",
    cot_path: str = "~/ml4t-data/cot",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load and combine futures OHLCV + COT data for a product.

    Convenience function that loads data from standard paths and combines them.

    Args:
        product: Product code (e.g., 'ES', 'CL')
        ohlcv_path: Path to OHLCV Hive-partitioned storage
        cot_path: Path to COT Hive-partitioned storage
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Combined DataFrame with OHLCV + COT features

    Example:
        >>> df = load_combined_futures_data("ES", start_date="2020-01-01")
        >>> print(df.columns)
        ['timestamp', 'open', 'high', 'low', 'close', 'volume',
         'open_interest', 'lev_money_net', 'cot_lev_money_pct_oi', ...]
    """
    from pathlib import Path

    ohlcv_path = Path(ohlcv_path).expanduser()
    cot_path = Path(cot_path).expanduser()

    # Load OHLCV data
    ohlcv_file = ohlcv_path / f"product={product}" / "data.parquet"
    if not ohlcv_file.exists():
        raise FileNotFoundError(f"OHLCV data not found: {ohlcv_file}")

    ohlcv = pl.read_parquet(ohlcv_file)

    # Load COT data
    cot_file = cot_path / f"product={product}" / "data.parquet"
    if not cot_file.exists():
        raise FileNotFoundError(f"COT data not found: {cot_file}")

    cot = pl.read_parquet(cot_file)

    # Combine
    combined = combine_cot_ohlcv(ohlcv, cot)

    # Add features
    combined = create_cot_features(combined)

    # Filter by date if specified
    if start_date:
        combined = combined.filter(pl.col("timestamp") >= start_date)
    if end_date:
        combined = combined.filter(pl.col("timestamp") <= end_date)

    return combined


if __name__ == "__main__":
    # Example usage
    from ml4t.data.cot import COTConfig, COTFetcher

    # Create fetcher with default config
    config = COTConfig(products=["ES"], start_year=2020)
    fetcher = COTFetcher(config)

    # Fetch COT data
    cot_df = fetcher.fetch_product("ES")
    print(f"COT data shape: {cot_df.shape}")
    print(f"COT columns: {cot_df.columns}")
    print(f"Date range: {cot_df['report_date'].min()} to {cot_df['report_date'].max()}")

    # Show sample data
    print("\nSample COT data:")
    print(cot_df.head(5))

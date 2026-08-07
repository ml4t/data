"""Point-in-time COT and OHLCV integration.

COT report dates describe the position snapshot, not public availability.
Every report must be mapped to its official, timezone-aware release timestamp
before it can be joined to market observations. CFTC holiday and exceptional
release schedules are not fixed calendar-day offsets.

Usage:
    from ml4t.data.cot import attach_cot_release_schedule, combine_cot_ohlcv

    cot = attach_cot_release_schedule(cot, official_schedule)
    combined = combine_cot_ohlcv(ohlcv, cot)
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from ml4t.data.core.config import resolve_storage_path


def _timestamp_expr(
    frame: pl.DataFrame,
    column: str,
    *,
    naive_timezone: str,
    require_timezone: bool,
) -> pl.Expr:
    if column not in frame.columns:
        raise ValueError(f"Required timestamp column '{column}' is missing")

    dtype = frame.schema[column]
    expression = pl.col(column)
    if dtype == pl.Null and frame.is_empty():
        return expression.cast(pl.Datetime("us", "UTC"))
    if dtype == pl.Date:
        return (
            expression.cast(pl.Datetime("us"))
            .dt.replace_time_zone(naive_timezone)
            .dt.convert_time_zone("UTC")
        )
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"'{column}' must contain Date or Datetime values, got {dtype}")
    if dtype.time_zone is None:
        if require_timezone:
            raise ValueError(f"'{column}' must contain timezone-aware timestamps")
        expression = expression.dt.replace_time_zone(naive_timezone)
    return expression.dt.convert_time_zone("UTC")


def attach_cot_release_schedule(
    cot: pl.DataFrame,
    schedule: pl.DataFrame,
    *,
    cot_date_col: str = "report_date",
    schedule_date_col: str = "report_date",
    available_at_col: str = "available_at",
) -> pl.DataFrame:
    """Attach authoritative release timestamps to COT reports.

    ``schedule`` must map every report date in ``cot`` to one timezone-aware
    publication timestamp. Extra schedule rows are allowed.
    """
    if cot_date_col not in cot.columns:
        raise ValueError(f"Required COT report date column '{cot_date_col}' is missing")
    if schedule_date_col not in schedule.columns:
        raise ValueError(f"Required schedule date column '{schedule_date_col}' is missing")
    if available_at_col not in schedule.columns:
        raise ValueError(f"Required schedule timestamp column '{available_at_col}' is missing")
    if available_at_col in cot.columns:
        raise ValueError(f"COT data already contains '{available_at_col}'")
    if schedule.get_column(schedule_date_col).null_count():
        raise ValueError("Release schedule report dates cannot contain null values")
    if schedule.get_column(schedule_date_col).n_unique() != schedule.height:
        raise ValueError("Release schedule must contain exactly one row per report date")

    release_expr = _timestamp_expr(
        schedule,
        available_at_col,
        naive_timezone="UTC",
        require_timezone=True,
    )
    schedule_for_join = schedule.select(
        pl.col(schedule_date_col).cast(pl.Date).alias("_cot_schedule_date"),
        release_expr.alias(available_at_col),
    )
    result = (
        cot.with_row_index("_cot_row_index")
        .with_columns(pl.col(cot_date_col).cast(pl.Date).alias("_cot_schedule_date"))
        .join(schedule_for_join, on="_cot_schedule_date", how="left")
        .sort("_cot_row_index")
        .drop("_cot_row_index", "_cot_schedule_date")
    )
    missing = result.filter(pl.col(available_at_col).is_null()).get_column(cot_date_col).unique()
    if not missing.is_empty():
        dates = ", ".join(str(value) for value in missing.sort().to_list())
        raise ValueError(f"Release schedule is missing COT report dates: {dates}")
    return result


def _prepare_cot_for_join(
    cot: pl.DataFrame,
    ohlcv_columns: set[str],
    *,
    cot_date_col: str,
    available_at_col: str,
) -> pl.DataFrame:
    if cot_date_col not in cot.columns:
        raise ValueError(f"Required COT report date column '{cot_date_col}' is missing")
    if available_at_col not in cot.columns:
        raise ValueError(
            f"COT data must include '{available_at_col}' with the official release timestamp"
        )
    if cot.get_column(cot_date_col).null_count() or cot.get_column(available_at_col).null_count():
        raise ValueError("COT report dates and release timestamps cannot contain null values")
    if cot.get_column(cot_date_col).n_unique() != cot.height:
        raise ValueError("COT data must contain exactly one row per report date")
    if cot.get_column(available_at_col).n_unique() != cot.height:
        raise ValueError("Each COT report must have a distinct release timestamp")

    release_expr = _timestamp_expr(
        cot,
        available_at_col,
        naive_timezone="UTC",
        require_timezone=True,
    )
    rename_map = {
        cot_date_col: "cot_report_date",
        available_at_col: "cot_available_at",
    }
    standard_names = {
        "open_interest": "cot_open_interest",
        "product": "cot_product",
        "report_type": "cot_report_type",
    }
    for source, target in standard_names.items():
        if source in cot.columns:
            rename_map[source] = target
    for column in cot.columns:
        if column not in rename_map and column in ohlcv_columns:
            rename_map[column] = f"cot_{column}"

    return (
        cot.with_columns(release_expr.alias(available_at_col))
        .rename(rename_map)
        .sort("cot_available_at")
    )


def combine_cot_ohlcv(
    ohlcv: pl.DataFrame,
    cot: pl.DataFrame,
    date_col: str = "timestamp",
    cot_date_col: str = "report_date",
    available_at_col: str = "available_at",
    observation_timezone: str = "UTC",
) -> pl.DataFrame:
    """Combine OHLCV with COT reports at their official availability timestamps.

    The COT input must contain one timezone-aware release timestamp for every
    report. This explicit schedule handles holiday delays and prevents a report
    from becoming visible on its position date.

    Args:
        ohlcv: Daily OHLCV DataFrame with timestamp column
        cot: Weekly COT DataFrame from COTFetcher
        date_col: Date column name in OHLCV data
        cot_date_col: Date column name in COT data
        available_at_col: Official release timestamp column in COT data
        observation_timezone: Timezone assigned to naive OHLCV timestamps

    Returns:
        Combined DataFrame with COT columns point-in-time forward-filled

    Example:
        >>> ohlcv = storage.load("ES", provider="databento")
        >>> cot = fetcher.fetch_product("ES", start_year=2020)
        >>> cot = attach_cot_release_schedule(cot, official_schedule)
        >>> combined = combine_cot_ohlcv(ohlcv, cot)
    """
    observation_expr = _timestamp_expr(
        ohlcv,
        date_col,
        naive_timezone=observation_timezone,
        require_timezone=False,
    )
    cot_for_join = _prepare_cot_for_join(
        cot,
        set(ohlcv.columns),
        cot_date_col=cot_date_col,
        available_at_col=available_at_col,
    )
    observations = (
        ohlcv.with_row_index("_cot_row_index")
        .with_columns(observation_expr.alias("_cot_observation_at"))
        .sort("_cot_observation_at")
    )
    return (
        observations.join_asof(
            cot_for_join,
            left_on="_cot_observation_at",
            right_on="cot_available_at",
            strategy="backward",
        )
        .sort("_cot_row_index")
        .drop("_cot_row_index", "_cot_observation_at")
    )


def create_cot_features(
    df: pl.DataFrame,
    prefix: str = "cot_",
    report_date_col: str | None = None,
    open_interest_col: str | None = None,
) -> pl.DataFrame:
    """Create ML features from COT positioning data.

    Creates normalized and derived features suitable for ML models.

    Features created:
    - Net position as % of open interest
    - Z-scores of net positions (52-week lookback)
    - Four-week change in leveraged or managed money positioning

    Args:
        df: DataFrame with COT columns (from combine_cot_ohlcv)
        prefix: Prefix for output column names
        report_date_col: Report date used to establish weekly observations
        open_interest_col: COT open-interest denominator

    Returns:
        DataFrame with additional COT feature columns

    Note:
        This function detects whether the data is financial futures (TFF)
        or commodity futures (disaggregated) based on available columns.
    """
    if report_date_col is None:
        report_date_col = next(
            (column for column in ("cot_report_date", "report_date") if column in df.columns),
            None,
        )
    if open_interest_col is None:
        open_interest_col = next(
            (column for column in ("cot_open_interest", "open_interest") if column in df.columns),
            None,
        )

    # Detect report type based on columns
    is_financial = "dealer_net" in df.columns or "lev_money_net" in df.columns
    is_commodity = "commercial_net" in df.columns or "managed_money_net" in df.columns
    feature_inputs = [
        column
        for column in (
            "lev_money_net",
            "asset_mgr_net",
            "dealer_net",
            "managed_money_net",
            "commercial_net",
            "nonrept_net",
            "oi_change",
            open_interest_col,
        )
        if column is not None and column in df.columns
    ]
    if not feature_inputs:
        return df

    report_frame = df
    if report_date_col is not None:
        report_rows = (
            df.select(report_date_col, *feature_inputs)
            .filter(pl.col(report_date_col).is_not_null())
            .unique()
        )
        if report_rows.get_column(report_date_col).n_unique() != report_rows.height:
            raise ValueError("COT feature inputs conflict within a report date")
        report_frame = report_rows.sort(report_date_col)

    exprs: list[pl.Expr] = []

    def percentage(numerator: str, name: str) -> pl.Expr:
        if open_interest_col is None:
            raise ValueError("COT open interest is required for percentage features")
        return (
            pl.when(pl.col(open_interest_col) > 0)
            .then(pl.col(numerator) / pl.col(open_interest_col) * 100)
            .otherwise(None)
            .alias(f"{prefix}{name}")
        )

    # === Financial Futures Features (TFF) ===
    if is_financial:
        # Leveraged Money (hedge funds) - most predictive for financials
        if "lev_money_net" in df.columns and open_interest_col is not None:
            # Net as % of OI
            exprs.append(percentage("lev_money_net", "lev_money_pct_oi"))
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
        if "asset_mgr_net" in df.columns and open_interest_col is not None:
            exprs.append(percentage("asset_mgr_net", "asset_mgr_pct_oi"))
            exprs.append(
                (
                    (pl.col("asset_mgr_net") - pl.col("asset_mgr_net").rolling_mean(52))
                    / pl.col("asset_mgr_net").rolling_std(52)
                ).alias(f"{prefix}asset_mgr_zscore_52w")
            )

        # Dealer (banks/swap dealers) - often hedging flow
        if "dealer_net" in df.columns and open_interest_col is not None:
            exprs.append(percentage("dealer_net", "dealer_pct_oi"))

    # === Commodity Futures Features (Disaggregated) ===
    if is_commodity:
        # Managed Money (hedge funds, CTAs) - trend followers
        if "managed_money_net" in df.columns and open_interest_col is not None:
            exprs.append(percentage("managed_money_net", "managed_money_pct_oi"))
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
        if "commercial_net" in df.columns and open_interest_col is not None:
            exprs.append(percentage("commercial_net", "commercial_pct_oi"))
            exprs.append(
                (
                    (pl.col("commercial_net") - pl.col("commercial_net").rolling_mean(52))
                    / pl.col("commercial_net").rolling_std(52)
                ).alias(f"{prefix}commercial_zscore_52w")
            )

    # === Universal Features ===
    # Non-reportables (small traders) - often contrarian indicator
    if "nonrept_net" in df.columns and open_interest_col is not None:
        exprs.append(percentage("nonrept_net", "nonrept_pct_oi"))

    # Open Interest change (market participation)
    if "oi_change" in df.columns and open_interest_col is not None:
        exprs.append(percentage("oi_change", "oi_change_pct"))

    if not exprs:
        return df

    reports_with_features = report_frame.with_columns(exprs)
    if report_date_col is None:
        return reports_with_features

    feature_names = [expression.meta.output_name() for expression in exprs]
    base = df.drop(*[name for name in feature_names if name in df.columns])
    return base.join(
        reports_with_features.select(report_date_col, *feature_names),
        on=report_date_col,
        how="left",
        maintain_order="left",
    )


def combine_cot_ohlcv_pit(
    ohlcv: pl.DataFrame,
    cot: pl.DataFrame,
    date_col: str = "timestamp",
    cot_date_col: str = "report_date",
    available_at_col: str = "available_at",
    observation_timezone: str = "UTC",
) -> pl.DataFrame:
    """Compatibility alias for :func:`combine_cot_ohlcv`.

    All COT joins are point-in-time safe. Calendar-day lag arithmetic is not
    supported because it cannot represent exact or holiday-delayed releases.

    Args:
        ohlcv: Daily OHLCV DataFrame with timestamp column
        cot: Weekly COT DataFrame from COTFetcher
        date_col: Date column name in OHLCV data
        cot_date_col: Date column name in COT data
        available_at_col: Official release timestamp column in COT data
        observation_timezone: Timezone assigned to naive OHLCV timestamps

    Returns:
        Point-in-time combined DataFrame
    """
    return combine_cot_ohlcv(
        ohlcv,
        cot,
        date_col=date_col,
        cot_date_col=cot_date_col,
        available_at_col=available_at_col,
        observation_timezone=observation_timezone,
    )


def load_combined_futures_data(
    product: str,
    ohlcv_path: str | None = None,
    cot_path: str | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
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
         'open_interest', 'cot_open_interest', 'lev_money_net', ...]
    """
    ohlcv_path = resolve_storage_path(ohlcv_path, "futures", "ohlcv-1d")
    cot_path = resolve_storage_path(cot_path, "cot")

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

    cot = create_cot_features(cot)
    combined = combine_cot_ohlcv(ohlcv, cot)

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

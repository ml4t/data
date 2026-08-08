"""Corporate-action adjustment functions."""

import polars as pl


def _validate_adjustment_inputs(
    df: pl.DataFrame,
    *,
    split_col: str,
    dividend_col: str | None,
    price_cols: list[str],
    volume_col: str | None,
) -> None:
    required = {"date", split_col, *price_cols}
    if dividend_col is not None:
        required.update(("close", dividend_col))
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required adjustment columns: {', '.join(missing)}")
    if df.get_column("date").null_count():
        raise ValueError("Corporate-action dates cannot contain null values")
    if df.get_column("date").n_unique() != df.height:
        raise ValueError("Corporate-action input must contain exactly one row per date")

    finite_columns = [split_col, *price_cols]
    if dividend_col is not None:
        finite_columns.extend(("close", dividend_col))
    if volume_col is not None and volume_col in df.columns:
        finite_columns.append(volume_col)
    for column in dict.fromkeys(finite_columns):
        invalid = df.filter(pl.col(column).is_null() | ~pl.col(column).is_finite())
        if not invalid.is_empty():
            raise ValueError(f"Adjustment column '{column}' must contain finite values")

    if not df.filter(pl.col(split_col) <= 0).is_empty():
        raise ValueError(f"Split ratios in '{split_col}' must be positive")
    if dividend_col is not None and not df.filter(pl.col("close") <= 0).is_empty():
        raise ValueError("Close prices must be positive")
    if dividend_col is not None and not df.filter(pl.col(dividend_col) < 0).is_empty():
        raise ValueError(f"Dividends in '{dividend_col}' cannot be negative")
    if volume_col is not None and volume_col in df.columns:
        if not df.filter(pl.col(volume_col) < 0).is_empty():
            raise ValueError(f"Volume in '{volume_col}' cannot be negative")


def _future_cumulative_product(column: str) -> pl.Expr:
    return pl.col(column).shift(-1, fill_value=1.0).reverse().cum_prod().reverse()


def _apply_canonical_adjustments(
    prices: pl.DataFrame,
    *,
    split_col: str,
    dividend_col: str | None,
    price_cols: list[str],
    volume_col: str | None,
) -> pl.DataFrame:
    _validate_adjustment_inputs(
        prices,
        split_col=split_col,
        dividend_col=dividend_col,
        price_cols=price_cols,
        volume_col=volume_col,
    )
    df = prices.sort("date").clone()

    price_event_factor = (
        pl.col("close")
        / (pl.col(split_col).cast(pl.Float64) * (pl.col("close") + pl.col(dividend_col)))
        if dividend_col is not None
        else 1.0 / pl.col(split_col).cast(pl.Float64)
    )
    df = df.with_columns(
        price_event_factor.alias("_price_event_factor"),
        pl.col(split_col).cast(pl.Float64).alias("_volume_event_factor"),
    ).with_columns(
        _future_cumulative_product("_price_event_factor").alias("price_adjustment_factor"),
        _future_cumulative_product("_volume_event_factor").alias("volume_adjustment_factor"),
    )

    adjustments = [
        (pl.col(column) * pl.col("price_adjustment_factor")).alias(f"adj_{column}")
        for column in price_cols
    ]
    if volume_col is not None and volume_col in df.columns:
        adjustments.append(
            (pl.col(volume_col) * pl.col("volume_adjustment_factor")).alias(f"adj_{volume_col}")
        )
    return df.with_columns(adjustments).drop("_price_event_factor", "_volume_event_factor")


def apply_corporate_actions(
    prices: pl.DataFrame,
    split_col: str = "split_ratio",
    dividend_col: str = "ex-dividend",
    price_cols: list[str] | None = None,
    volume_col: str | None = "volume",
) -> pl.DataFrame:
    """Adjust prices and volume to the share basis of the latest observation.

    A split ratio on date ``t`` is new shares per old share and applies between
    the preceding observation and ``t``. The row on ``t`` is already on the new
    share basis. Earlier prices are divided by subsequent split ratios and
    earlier volume is multiplied by them. A dividend is cash per post-event
    share on its ex-date. Adjusted close-to-close returns include that cash.
    The dividend factor uses the ex-date close. This differs from data vendors
    that discount dividends using the prior close.

    Args:
        prices: DataFrame with date-sorted prices and corporate action data
        split_col: Column with split ratios (default: 'split_ratio')
        dividend_col: Column with dividend amounts (default: 'ex-dividend')
        price_cols: Price columns to adjust (default: ['open', 'high', 'low', 'close'])
        volume_col: Volume column to adjust (default: 'volume')

    Returns:
        DataFrame retaining raw columns and adding adjusted columns plus explicit
        price and volume adjustment factors
    """
    if price_cols is None:
        price_cols = [
            column for column in ("open", "high", "low", "close") if column in prices.columns
        ]
    return _apply_canonical_adjustments(
        prices,
        split_col=split_col,
        dividend_col=dividend_col,
        price_cols=price_cols,
        volume_col=volume_col,
    )


def apply_splits(
    prices: pl.DataFrame,
    split_col: str = "split_ratio",
    price_cols: list[str] | None = None,
    volume_col: str | None = "volume",
) -> pl.DataFrame:
    """Apply the canonical split-only adjustment convention.

    Args:
        prices: DataFrame with date-sorted prices and split_ratio column
        split_col: Name of column containing split ratios (default: 'split_ratio')
        price_cols: List of price columns to adjust (default: ['open', 'high', 'low', 'close'])
        volume_col: Volume column name to adjust

    Returns:
        DataFrame retaining raw values and adding adjusted values and factors
    """
    if price_cols is None:
        price_cols = [
            column for column in ("open", "high", "low", "close") if column in prices.columns
        ]
    return _apply_canonical_adjustments(
        prices,
        split_col=split_col,
        dividend_col=None,
        price_cols=price_cols,
        volume_col=volume_col,
    )


def apply_dividends(
    prices: pl.DataFrame,
    dividend_col: str = "ex-dividend",
    price_cols: list[str] | None = None,
    close_col: str = "adj_close",
) -> pl.DataFrame:
    """Apply dividend-only adjustments that preserve total close-to-close returns.

    A dividend on date ``t`` is cash per share paid between the preceding
    observation and ``t``. The event row remains unchanged. This adapter is for
    data whose split adjustments have already been handled consistently. When
    split factors are present, dividends are converted to the latest share basis
    and the dividend factor is composed with the existing price factor.

    Args:
        prices: DataFrame with date-sorted prices and ex-dividend column
        dividend_col: Name of column containing dividend amounts
        price_cols: List of price columns to adjust
        close_col: Column to use for dividend factor calculation (default: 'adj_close')

    Returns:
        DataFrame with adjusted price columns and an explicit adjustment factor
    """
    if price_cols is None:
        price_cols = [
            column
            for column in ("adj_open", "adj_high", "adj_low", "adj_close")
            if column in prices.columns
        ]

    required = {"date", dividend_col, close_col, *price_cols}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"Missing required dividend columns: {', '.join(missing)}")
    df = prices.sort("date").clone()
    if df.get_column("date").null_count():
        raise ValueError("Dividend dates cannot contain null values")
    if df.get_column("date").n_unique() != df.height:
        raise ValueError("Dividend input must contain exactly one row per date")
    for column in (dividend_col, close_col, *price_cols):
        if not df.filter(pl.col(column).is_null() | ~pl.col(column).is_finite()).is_empty():
            raise ValueError(f"Dividend adjustment column '{column}' must contain finite values")
    if not df.filter(pl.col(dividend_col) < 0).is_empty():
        raise ValueError(f"Dividends in '{dividend_col}' cannot be negative")
    if not df.filter(pl.col(close_col) <= 0).is_empty():
        raise ValueError(f"Close prices in '{close_col}' must be positive")

    for factor_column in ("price_adjustment_factor", "volume_adjustment_factor"):
        if (
            factor_column in df.columns
            and not df.filter(
                pl.col(factor_column).is_null()
                | ~pl.col(factor_column).is_finite()
                | (pl.col(factor_column) <= 0)
            ).is_empty()
        ):
            raise ValueError(f"Adjustment factor '{factor_column}' must be finite and positive")

    rebased_dividend = pl.col(dividend_col)
    if "volume_adjustment_factor" in df.columns:
        rebased_dividend = rebased_dividend / pl.col("volume_adjustment_factor")
    existing_factor = (
        pl.col("price_adjustment_factor")
        if "price_adjustment_factor" in df.columns
        else pl.lit(1.0)
    )

    return (
        df.with_columns(rebased_dividend.alias("_rebased_dividend"))
        .with_columns(
            (pl.col(close_col) / (pl.col(close_col) + pl.col("_rebased_dividend"))).alias(
                "_dividend_event_factor"
            )
        )
        .with_columns(
            _future_cumulative_product("_dividend_event_factor").alias(
                "_dividend_adjustment_factor"
            )
        )
        .with_columns(
            *[
                (pl.col(column) * pl.col("_dividend_adjustment_factor")).alias(column)
                for column in price_cols
            ],
            (existing_factor * pl.col("_dividend_adjustment_factor")).alias(
                "price_adjustment_factor"
            ),
        )
        .drop("_rebased_dividend", "_dividend_event_factor", "_dividend_adjustment_factor")
    )

"""DataFrame conversions that work without optional Arrow dependencies."""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl


def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas data to Polars without requiring PyArrow."""
    columns: dict[str, list[Any]] = {}

    for name, series in df.items():
        if pd.api.types.is_datetime64_any_dtype(series):
            columns[str(name)] = [
                None
                if pd.isna(value)
                else value.to_pydatetime()
                if isinstance(value, pd.Timestamp)
                else value
                for value in series
            ]
        else:
            columns[str(name)] = series.astype(object).where(series.notna(), None).tolist()

    return pl.DataFrame(columns, strict=False, nan_to_null=True)

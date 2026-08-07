"""DataFrame conversions that work without optional Arrow dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def _datetime_to_polars(name: str, series: pd.Series) -> pl.Series:
    """Preserve pandas' integer timestamp representation without Arrow."""
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        unit = dtype.unit
        time_zone = str(dtype.tz)
    else:
        unit = np.datetime_data(dtype)[0]
        time_zone = None

    values = series.array.asi8
    target_unit = unit if unit in {"ms", "us", "ns"} else "ms"
    multiplier = 1_000 if unit == "s" else 1
    nat = np.iinfo(np.int64).min
    normalized = [None if value == nat else int(value) * multiplier for value in values]
    return pl.Series(
        name,
        normalized,
        dtype=pl.Datetime(target_unit, time_zone),
        strict=False,
    )


def _all_null_dtype(dtype: Any) -> pl.DataType | None:
    """Map typed all-null pandas columns to their Polars scalar dtype."""
    if pd.api.types.is_float_dtype(dtype):
        return pl.Float32 if dtype.itemsize == 4 else pl.Float64
    if pd.api.types.is_integer_dtype(dtype):
        prefix = "UInt" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int"
        return getattr(pl, f"{prefix}{dtype.itemsize * 8}")
    if pd.api.types.is_bool_dtype(dtype):
        return pl.Boolean
    if isinstance(dtype, pd.StringDtype):
        return pl.String
    return None


def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas data to Polars without requiring PyArrow."""
    columns: dict[str, pl.Series | list[Any]] = {}

    for name, series in df.items():
        column_name = str(name)
        if pd.api.types.is_datetime64_any_dtype(series):
            columns[column_name] = _datetime_to_polars(column_name, series)
        else:
            values = series.astype(object).where(series.notna(), None).tolist()
            dtype = _all_null_dtype(series.dtype) if series.isna().all() else None
            columns[column_name] = (
                pl.Series(column_name, values, dtype=dtype, strict=False) if dtype else values
            )

    return pl.DataFrame(columns, strict=False, nan_to_null=True)

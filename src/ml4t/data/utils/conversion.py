"""DataFrame conversions that work without optional Arrow dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def _datetime_to_polars(name: str, series: pd.Series) -> pl.Series:
    """Preserve timestamp precision and normalize timezone-aware data to UTC."""
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        unit = dtype.unit
        series = series.dt.tz_convert("UTC")
        time_zone = "UTC"
    elif isinstance(dtype, np.dtype):
        unit = np.datetime_data(dtype)[0]
        time_zone = None
    else:
        arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
        unit = getattr(arrow_dtype, "unit", None)
        arrow_time_zone = getattr(arrow_dtype, "tz", None)
        if arrow_time_zone is not None:
            series = series.dt.tz_convert("UTC")
        time_zone = "UTC" if arrow_time_zone is not None else None
        target_unit = unit if unit in {"ms", "us", "ns"} else "ms"
        normalized = [
            None
            if pd.isna(value)
            else int(pd.Timestamp(value).as_unit(target_unit).asm8.astype("int64"))
            for value in series
        ]
        return pl.Series(
            name,
            normalized,
            dtype=pl.Datetime(target_unit, time_zone),
            strict=False,
        )

    values = getattr(series.array, "asi8", None)
    if not isinstance(values, np.ndarray):
        raise TypeError(f"Cannot access integer datetime values for column '{name}'")

    target_unit = unit if unit in {"ms", "us", "ns"} else "ms"
    nat = np.iinfo(np.int64).min
    if unit == "s":
        normalized = [None if value == nat else int(value) * 1_000 for value in values]
    elif unit in {"ms", "us", "ns"}:
        normalized = [None if value == nat else int(value) for value in values]
    else:
        normalized = [
            None
            if pd.isna(value)
            else int(pd.Timestamp(value).as_unit(target_unit).asm8.astype("int64"))
            for value in series
        ]
    return pl.Series(
        name,
        normalized,
        dtype=pl.Datetime(target_unit, time_zone),
        strict=False,
    )


def _all_null_dtype(dtype: Any) -> pl.DataType | type[pl.DataType] | None:
    """Map typed all-null pandas columns to their Polars scalar dtype."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    if arrow_dtype is not None and str(arrow_dtype).startswith("duration["):
        unit = getattr(arrow_dtype, "unit", "ms")
        return pl.Duration(unit if unit in {"ms", "us", "ns"} else "ms")
    if pd.api.types.is_float_dtype(dtype):
        return pl.Float32 if dtype.itemsize == 4 else pl.Float64
    if pd.api.types.is_integer_dtype(dtype):
        prefix = "UInt" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int"
        return getattr(pl, f"{prefix}{dtype.itemsize * 8}")
    if pd.api.types.is_bool_dtype(dtype):
        return pl.Boolean
    if pd.api.types.is_timedelta64_dtype(dtype):
        arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
        unit = getattr(arrow_dtype, "unit", None)
        if unit is None:
            unit = np.datetime_data(dtype)[0] if isinstance(dtype, np.dtype) else "ms"
        return pl.Duration(unit if unit in {"ms", "us", "ns"} else "ms")
    if isinstance(dtype, pd.CategoricalDtype):
        if dtype.categories.empty:
            return pl.String
        return _all_null_dtype(dtype.categories.dtype) or pl.String
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

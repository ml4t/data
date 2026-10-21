"""DataFrame conversions that work without optional Arrow dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.data.core.feeds import resolve_feed_spec
from ml4t.specs import FeedSpec


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
    if arrow_dtype is not None and str(arrow_dtype).startswith(("date32[", "date64[")):
        return pl.Date
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
        arrow_dtype = getattr(series.dtype, "pyarrow_dtype", None)
        is_timestamp = (
            str(arrow_dtype).startswith("timestamp[")
            if arrow_dtype is not None
            else pd.api.types.is_datetime64_any_dtype(series.dtype)
        )
        if is_timestamp:
            columns[column_name] = _datetime_to_polars(column_name, series)
        else:
            values = series.astype(object).where(series.notna(), None).tolist()
            dtype = _all_null_dtype(series.dtype) if series.isna().all() else None
            columns[column_name] = (
                pl.Series(column_name, values, dtype=dtype, strict=False) if dtype else values
            )

    return pl.DataFrame(columns, strict=False, nan_to_null=True)


def _polars_series_to_pandas(series: pl.Series) -> pd.Series:
    """Convert one Polars series without Arrow while retaining its logical dtype."""
    dtype = series.dtype
    name = series.name
    values = series.to_numpy()

    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            timestamps = pd.to_datetime(values, utc=True).tz_convert(dtype.time_zone)
            return pd.Series(timestamps, name=name)
        return pd.Series(values, name=name)
    if dtype == pl.Date:
        return pd.Series(values.astype("datetime64[ms]"), name=name)
    if isinstance(dtype, pl.Duration):
        return pd.Series(values, name=name)
    if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
        return pd.Series(pd.Categorical(series.to_list()), name=name)
    if dtype == pl.String:
        return pd.Series(series.to_list(), name=name, dtype="str")
    if dtype.is_integer() and series.null_count():
        return pd.Series(pd.array(series.to_list(), dtype=str(dtype)), name=name)
    return pd.Series(values, name=name)


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars data to pandas without requiring PyArrow."""
    return pd.DataFrame({name: _polars_series_to_pandas(df[name]) for name in df.columns})


def _axis_keys(index: pd.Index) -> list[str]:
    return [f"{type(value).__name__}:{value!r}" for value in index.tolist()]


def to_price_panel(
    data: pl.DataFrame | pd.DataFrame,
    *,
    feed_spec: Any | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Project a price field to a pandas clock-by-asset panel without filling cells.

    Clock and asset order follow first occurrence. Duplicate/null keys and ambiguous mappings
    fail. Other observation fields remain in the caller's original table, not in the price panel.
    ``attrs['ml4t_price_panel']`` retains the feed, source dtype, metadata and an observation mask
    distinguishing an observed null from an absent asset/clock pair. Keep these attrs for a sparse
    inverse. The caller must keep per-observation availability and revision facts separately.
    """
    feed = resolve_feed_spec(feed_spec, timestamp_col, entity_col, price_col)
    columns = list(data.columns)
    if feed.entity_col is None:
        candidates = [column for column in ("asset", "symbol", "ticker") if column in columns]
        if len(candidates) != 1:
            raise ValueError("Declare entity_col when asset identity is absent or ambiguous")
        feed = feed.with_overrides(entity_col=candidates[0])
    feed = feed.resolve(columns, ())
    names = [feed.timestamp_col, str(feed.entity_col), feed.price_col]
    if len(set(names)) != 3 or any(name not in columns for name in names):
        raise ValueError("Panel timestamp, entity and price must map to distinct existing columns")
    frame = (
        polars_to_pandas(data.select(names))
        if isinstance(data, pl.DataFrame)
        else data[names].copy()
    )
    if frame[names[:2]].isna().any().any():
        raise ValueError("Panel observation keys cannot be null")
    if frame.duplicated(names[:2]).any():
        raise ValueError("Duplicate timestamp/entity observation keys")
    price_dtype = frame[names[2]].dtype
    if pd.api.types.is_integer_dtype(price_dtype) and not isinstance(
        price_dtype, pd.api.extensions.ExtensionDtype
    ):
        nullable = str(price_dtype).replace("uint", "UInt").replace("int", "Int")
        frame[names[2]] = frame[names[2]].astype(nullable)
    clock = pd.Index(frame[names[0]].drop_duplicates(), name=names[0])
    assets = pd.Index(frame[names[1]].drop_duplicates(), name=names[1])
    indexed = frame.set_index(names[:2])[names[2]]
    grid = pd.MultiIndex.from_product([clock, assets])
    panel = pd.DataFrame(
        indexed.reindex(grid).to_numpy(dtype=object).reshape(len(clock), len(assets)),
        index=clock,
        columns=assets,
    )
    panel = panel.astype(indexed.dtype)
    present = pd.Series(True, index=indexed.index).reindex(grid, fill_value=False)
    panel.attrs["ml4t_price_panel"] = {
        "feed_spec": feed.to_dict(),
        "metadata": deepcopy(dict(metadata or {})),
        "observed": present.to_numpy().reshape(panel.shape).tolist(),
        "clock": _axis_keys(clock),
        "assets": _axis_keys(assets),
        "schema": (
            data.select(names).head(0).write_ipc(None).getvalue().hex()
            if isinstance(data, pl.DataFrame)
            else None
        ),
    }
    if isinstance(data, pl.DataFrame) and data.schema[feed.price_col].is_float():
        nulls = pd.Series(data[feed.price_col].is_null().to_numpy(), index=indexed.index)
        panel.attrs["ml4t_price_panel"]["null_cells"] = (
            nulls.reindex(grid, fill_value=False).to_numpy().reshape(panel.shape).tolist()
        )
    return panel


def _panel_mask(values: Any, shape: tuple[int, int]) -> np.ndarray:
    if (
        not isinstance(values, list)
        or len(values) != shape[0]
        or any(
            not isinstance(row, list)
            or len(row) != shape[1]
            or any(not isinstance(value, bool) for value in row)
            for row in values
        )
    ):
        raise ValueError("Panel cell mask does not match its axes")
    return np.asarray(values, dtype=bool).reshape(shape)


def from_price_panel(
    panel: pd.DataFrame,
    *,
    feed_spec: Any | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    price_col: str | None = None,
) -> pl.DataFrame:
    """Return clock-major native price observations, retaining nulls and typed asset IDs.

    Panels from ``to_price_panel`` retain sparse presence and source dtypes in attrs. An ordinary
    pandas panel without those attrs treats every cell, including a null, as an observation.
    A stale mask after changing panel axes is rejected, never silently realigned or filled.
    """
    contract = panel.attrs.get("ml4t_price_panel", {})
    declared = contract.get("feed_spec")
    if feed_spec is not None and declared is not None:
        if FeedSpec.from_any(feed_spec).to_dict() != declared:
            raise ValueError("feed_spec conflicts with panel declaration")
    feed = resolve_feed_spec(
        declared if feed_spec is None else feed_spec, timestamp_col, entity_col, price_col
    )
    asset_col = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    names = [feed.timestamp_col, asset_col, feed.price_col]
    if len(set(names)) != 3:
        raise ValueError("Panel timestamp, entity and price mappings must be distinct")
    if not panel.index.is_unique or not panel.columns.is_unique:
        raise ValueError("Duplicate panel clock or asset keys")
    if isinstance(panel.index, pd.MultiIndex) or isinstance(panel.columns, pd.MultiIndex):
        raise ValueError("Panel axes must each represent a single declared key")
    if panel.index.hasnans or panel.columns.hasnans:
        raise ValueError("Panel keys cannot be null")
    observed = contract.get("observed")
    if observed is None:
        mask = np.ones(panel.shape, dtype=bool)
    else:
        if contract.get("clock") != _axis_keys(panel.index) or contract.get("assets") != _axis_keys(
            panel.columns
        ):
            raise ValueError("Panel observation mask does not match its axes")
        mask = _panel_mask(observed, panel.shape)
        if panel.notna().to_numpy()[~mask].any():
            raise ValueError("An absent observation mask cell contains a price")
    rows, cols = np.nonzero(mask)
    frame = pd.DataFrame(
        {
            names[0]: panel.index.take(rows),
            names[1]: panel.columns.take(cols),
            names[2]: panel.to_numpy(dtype=object)[rows, cols],
        }
    )
    result = pandas_to_polars(frame)
    timestamp_dtype = frame[names[0]].dtype
    if isinstance(timestamp_dtype, pd.DatetimeTZDtype):
        result = result.with_columns(pl.col(names[0]).dt.convert_time_zone(str(timestamp_dtype.tz)))
    schema = contract.get("schema")
    if isinstance(schema, str):
        schema = dict(pl.read_ipc(BytesIO(bytes.fromhex(schema))).schema)
    if schema is not None:
        expressions = []
        for name, dtype in schema.items():
            expr = pl.col(name)
            if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
                expr = expr.dt.convert_time_zone(dtype.time_zone)
            expressions.append(expr.cast(dtype, strict=True))
        result = result.select(expressions)
        if "null_cells" in contract:
            nulls = _panel_mask(contract["null_cells"], panel.shape)[rows, cols]
            values = [
                value if not pd.isna(value) else None if was_null else float("nan")
                for value, was_null in zip(frame[names[2]], nulls, strict=True)
            ]
            result = result.with_columns(pl.Series(names[2], values, dtype=schema[names[2]]))
    return result

"""Opt-in acquisition compatibility, not Empyrical's analytical functions."""

from __future__ import annotations

import csv
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.utils.atomic import atomic_write
from ml4t.data.utils.conversion import polars_to_pandas
from ml4t.data.utils.locking import file_lock


def cache_dir(environ: Mapping[str, str] = os.environ) -> str:
    """Use legacy EMPYRICAL_CACHE_DIR, then XDG_CACHE_HOME/empyrical."""
    return environ.get(
        "EMPYRICAL_CACHE_DIR",
        str(Path(environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "empyrical"),
    )


def data_path(name: str) -> str:
    return str(Path(cache_dir()) / name)


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Date bounds must be known")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _1_bday_ago() -> pd.Timestamp:
    return pd.Timestamp.now().normalize() - pd.offsets.BDay()


def _date(value: Any) -> str | None:
    return None if value is None else pd.Timestamp(value).date().isoformat()


def get_fama_french(
    start: Any = "1-1-1970",
    end: Any = None,
    datasets: Sequence[str] = ("F-F_Research_Data_Factors_daily", "F-F_Momentum_Factor_daily"),
) -> pd.DataFrame:
    """Return decimal factors/RF on the legacy complete-case UTC date sample."""
    from ml4t.data.providers.fama_french import FF_CATEGORIES, FamaFrenchProvider

    if isinstance(datasets, str) or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Specify distinct Fama-French dataset names")
    names = {}
    for dataset, information in FamaFrenchProvider.FILES.items():
        for frequency in information.get("frequencies", ["monthly"]):
            suffix = "" if frequency == "monthly" else "_" + frequency
            names[information["file"] + suffix] = (dataset, frequency)
    if unknown := set(datasets).difference(names):
        raise ValueError(f"Unsupported Fama-French datasets: {sorted(unknown)}")
    selected = [names[name] for name in datasets]
    if len({frequency for _, frequency in selected}) != 1:
        raise ValueError("Do not combine different factor frequencies in one table")
    frequency = selected[0][1]
    lower = _date(start)
    if frequency == "monthly" and start is not None:
        lower = pd.Timestamp(start).replace(day=1).date().isoformat()
    upper = _date(pd.Timestamp.now() if end is None else end)
    with FamaFrenchProvider(use_cache=False) as provider:
        parts = [
            polars_to_pandas(
                provider.fetch(name, frequency=frequency, start=lower, end=upper)
            ).set_index("timestamp")
            for name, _ in selected
        ]
    result = pd.concat(parts, axis=1).dropna().rename(columns={"MOM": "Mom"})
    period = "M" if frequency == "monthly" else "D"
    result.index = (
        pd.PeriodIndex(result.index, freq=period, name="Date").to_timestamp().tz_localize("UTC")
    )
    result.attrs.update(
        source="Kenneth R. French Data Library",
        frequency=frequency,
        units=None
        if any(name in FF_CATEGORIES["breakpoints"] for name, _ in selected)
        else "decimal_returns",
        source_value_conversion="divide_by_100",
        risk_free_column="RF" if "RF" in result else None,
        timestamp_semantics="session_label",
        missingness_policy="legacy_complete_case",
        availability="unknown; capture time is not historical publication",
    )
    return result


def _validated_returns(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("A returns cache update callback must return a pandas DataFrame")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise ValueError("A returns cache requires a DatetimeIndex")
    result = value.copy(deep=True)
    result.index = pd.to_datetime(result.index, utc=True)
    if (
        result.index.hasnans
        or not result.index.is_unique
        or not result.index.is_monotonic_increasing
    ):
        raise ValueError("A returns cache requires ordered, unique and known timestamps")
    if not result.columns.is_unique or any(
        not isinstance(name, str) or not name for name in result.columns
    ):
        raise ValueError("A returns cache requires distinct named columns")
    if not len(result.columns):
        raise ValueError("A returns cache requires at least one value column")
    for name in result:
        result[name] = pd.to_numeric(result[name], errors="raise")
        values = result[name].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Known cached returns must be finite")
    return result


def get_returns_cached(
    filepath: str | Path, update_func: Callable[..., pd.DataFrame], latest_dt: Any, **kwargs: Any
) -> pd.DataFrame:
    """Keep legacy file-mtime freshness, with validated UTC CSV and atomic replacement."""
    path = Path(filepath).expanduser()
    latest = _utc(latest_dt)
    try:
        modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    except FileNotFoundError:
        modified = None
    if modified is not None and modified >= latest:
        with path.open(newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) < 2 or len(header[1:]) != len(set(header[1:])):
            raise ValueError("Corrupt returns cache header")
        result = pd.read_csv(path, index_col=0)
        result.index = pd.to_datetime(result.index, utc=True, errors="raise")
        return _validated_returns(result)
    result = _validated_returns(update_func(**kwargs))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            atomic_write(path, lambda temporary: result.to_csv(temporary))
    except OSError as error:
        warnings.warn(f"could not update cache {path}: {error}", UserWarning, stacklevel=2)
    return result


def load_portfolio_risk_factors(
    filepath_prefix: str | Path | None = None, start: Any = None, end: Any = None
) -> pd.DataFrame:
    start = _utc("1970-01-01" if start is None else start)
    end = _utc(_1_bday_ago() if end is None else end)
    filepath = data_path("factors.csv") if filepath_prefix is None else filepath_prefix
    return get_returns_cached(filepath, get_fama_french, end).loc[start:end]


def get_treasury_yield(start: Any = None, end: Any = None, period: str = "3MO") -> pd.DataFrame:
    """Return annual percent yield; legacy filling only affects existing reported rows."""
    from ml4t.data.providers.fred import FREDProvider

    if period not in FREDProvider.TREASURY_SERIES:
        raise ValueError("Supported Treasury maturities:1MO,3MO,6MO,1,5,10")
    name = FREDProvider.TREASURY_SERIES[period]
    with FREDProvider(anonymous=not bool(os.getenv("FRED_API_KEY"))) as provider:
        native = provider.fetch_series(
            name,
            _date("1970-01-01" if start is None else start),
            _date(_1_bday_ago() if end is None else end),
        )
    if native.height and set(native["units"].to_list()) != {"Percent"}:
        raise DataValidationError("fred", f"Expected Percent source units for {name}")
    values = polars_to_pandas(native).set_index("timestamp")["value"]
    missing = tuple(bool(value) for value in values.isna())
    result = values.ffill().to_frame(name)
    labels = [stamp.isoformat() for stamp in native["timestamp"]]
    result.index = pd.DatetimeIndex(pd.to_datetime(labels), name="DATE")
    result.attrs.update(
        units="annual_percent_yield",
        source="FRED",
        series_id=name,
        source_missing_mask=missing,
        missingness_policy="legacy_forward_fill_existing_rows",
        timestamp_semantics="session_label",
        availability="unknown",
    )
    return result


def get_symbol_returns_from_yahoo(symbol: str, start: Any = None, end: Any = None) -> pd.DataFrame:
    """Return adjusted-close simple returns with upstream exclusive-end semantics and no fill."""
    from ml4t.data.providers.yahoo import YahooFinanceProvider

    with YahooFinanceProvider() as provider:
        try:
            native = provider.fetch_adjusted_close(symbol, start=start, end=end)
        except Exception as error:
            warnings.warn(f"Yahoo Finance read failed: {error}", UserWarning, stacklevel=2)
            raise
        index_name = provider.last_fetch_metadata["source_index_name"]
        index_unit = provider.last_fetch_metadata["source_index_unit"]
    prices = polars_to_pandas(native).set_index("timestamp")["close"]
    result = prices.pct_change(fill_method=None).dropna().to_frame(symbol)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise DataValidationError("yahoo", "Adjusted-price percentage returns must be finite")
    result.index = pd.DatetimeIndex(result.index).as_unit(index_unit).rename(index_name)
    result.attrs.update(
        price_basis="yahoo_auto_adjusted",
        missingness_policy="no_fill",
        end_boundary="exclusive",
        source="Yahoo Finance/yfinance",
    )
    return result


def default_returns_func(symbol: str, start: Any = None, end: Any = None) -> pd.Series:
    start = _utc("1970-01-01" if start is None else start)
    end = _utc(_1_bday_ago() if end is None else end)
    if symbol == "SPY":
        result = get_returns_cached(
            data_path("spy.csv"),
            get_symbol_returns_from_yahoo,
            end,
            symbol="SPY",
            start="1/1/1970",
            end=pd.Timestamp.now(),
        ).loc[start:end]
    else:
        result = get_symbol_returns_from_yahoo(symbol, start=start, end=end)
    return result[symbol]

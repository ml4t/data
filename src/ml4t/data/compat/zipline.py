"""Optional exchange through Zipline's public bundle readers and ingestion writers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.data.adjustments import prepare_action_inputs
from ml4t.data.storage import InputSnapshot, SnapshotTable, export_snapshot, read_snapshot
from ml4t.data.utils.conversion import pandas_to_polars, polars_to_pandas
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

_FIELDS = ["open", "high", "low", "close", "volume"]
_ASSET_DATES = (
    "start_date",
    "end_date",
    "first_traded",
    "auto_close_date",
    "notice_date",
    "expiration_date",
)
_FACTOR_TABLES = ("splits", "mergers", "dividends")
_PAYOUT_TABLES = ("dividend_payouts", "stock_dividend_payouts")


def _require_zipline() -> None:
    try:
        import zipline  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Bundle exchange requires Zipline Reloaded in the conversion environment; native snapshots do not"
        ) from error


def _identity(namespace: str, sid: int) -> str:
    return f"{namespace}:{sid}"


def _source_metadata(namespace: str, revision: str, role: str) -> dict[str, Any]:
    return {
        "source_revision": revision,
        "identity_namespace": namespace,
        "units": {"prices": None, "cash": None, "volume": "source_reader_values"},
        "as_of": None,
        "price_basis": "raw" if role.endswith("bars") else None,
        "share_basis": "source",
        "revisions": {"bundle": revision},
        "source_software": {"zipline-reloaded": version("zipline-reloaded")},
    }


def _frame(pandas: pd.DataFrame) -> pl.DataFrame:
    return pandas_to_polars(pandas.reset_index(drop=True))


def _clock(value: Any, *, daily: bool) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Bundle export bounds must be known")
    if daily:
        if stamp != stamp.normalize():
            raise ValueError("Daily bundle bounds must be civil session labels at midnight")
        return stamp.tz_localize(None)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _asset_tables(bundle: Any, sids: list[int], namespace: str) -> dict[str, pl.DataFrame]:
    finder = bundle.asset_finder
    asset_types = finder.lookup_asset_types(sids)
    records = {"equities": [], "futures": []}
    for asset in finder.retrieve_all(sids):
        record = asset.to_dict()
        record.pop("exchange_info", None)
        record["canonical_exchange"] = record["exchange"]
        record["exchange"] = record.pop("exchange_full")
        record["asset"] = _identity(namespace, asset.sid)
        record["source_sid"] = int(asset.sid)
        records["futures" if asset_types[asset.sid] == "future" else "equities"].append(record)
    tables = {}
    for kind, rows in records.items():
        if rows:
            frame = pd.DataFrame(rows)
            for name in _ASSET_DATES:
                if name in frame:
                    frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_localize(None)
            tables["zipline_" + kind] = _frame(frame)
    exchanges = [
        {"exchange": name, "canonical_name": item.canonical_name, "country_code": item.country_code}
        for name, item in finder.exchange_info.items()
    ]
    tables["zipline_exchanges"] = _frame(
        pd.DataFrame(exchanges, columns=["exchange", "canonical_name", "country_code"])
    )
    lookups = []
    for (company, share_class), periods in finder.symbol_ownership_map.items():
        for period in periods:
            if period.sid in sids:
                lookups.append(
                    {
                        "sid": int(period.sid),
                        "company_symbol": company,
                        "share_class_symbol": share_class,
                        "symbol": period.value,
                        "start_date": period.start,
                        "end_date": period.end,
                    }
                )
    if lookups:
        tables["zipline_symbol_lookup"] = _frame(pd.DataFrame(lookups))
    return tables


def _bars_and_factors(
    bundle: Any,
    sids: list[int],
    start: Any,
    end: Any,
    namespace: str,
    revision: str,
    frequency: str,
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    try:
        from zipline.data.history_loader import HistoryCompatibleUSEquityAdjustmentReader
        from zipline.lib.adjusted_array import AdjustedArray
    except ImportError as error:
        raise ImportError("Factor capture needs Zipline's supported history readers") from error

    daily = frequency == "daily"
    reader = bundle.equity_daily_bar_reader if daily else bundle.equity_minute_bar_reader
    lower, upper = _clock(start, daily=daily), _clock(end, daily=daily)
    if lower > upper:
        raise ValueError("Bundle export start must not follow end")
    source_calendar = reader.trading_calendar
    if source_calendar is None:
        raise ValueError("The source bundle must declare its trading calendar")
    dates = (
        reader.sessions[(reader.sessions >= lower) & (reader.sessions <= upper)]
        if daily
        else source_calendar.minutes_in_range(lower, upper)
    )
    if not len(dates) or dates[0] != lower or dates[-1] != upper:
        raise ValueError("Export bounds must identify available source sessions or minutes")
    if not sids:
        clock_type = pl.Date if daily else pl.Datetime("ns", "UTC")
        keys = {
            "timestamp": clock_type,
            "asset": pl.String,
            "source_sid": pl.Int64,
            "source": pl.String,
            "revision_id": pl.String,
            "available_at": pl.Datetime("ns", "UTC"),
        }
        empty_bars = pl.DataFrame(schema={**keys, **dict.fromkeys(_FIELDS, pl.Float64)})
        empty_factors = pl.DataFrame(
            schema={
                **keys,
                "price_adjustment_factor": pl.Float64,
                "volume_adjustment_factor": pl.Float64,
                "basis_at": clock_type,
            }
        )
        return empty_bars, empty_factors, source_calendar.name
    arrays = reader.load_raw_arrays(_FIELDS, lower, upper, sids)
    expected = (len(dates), len(sids))
    if any(array.shape != expected for array in arrays):
        raise ValueError("Source reader arrays do not match the declared calendar and sid sample")
    adjustment_reader = HistoryCompatibleUSEquityAdjustmentReader(bundle.adjustment_reader)
    bars, factors = [], []
    for column, sid in enumerate(sids):
        adjustments = adjustment_reader.load_pricing_adjustments(
            ["close", "volume"], dates, [bundle.asset_finder.retrieve_asset(sid)]
        )
        curves = [
            next(
                AdjustedArray(np.ones((len(dates), 1)), mapping or {}, np.nan).traverse(len(dates))
            )
            for mapping in adjustments
        ]
        clock = pd.Series(dates.date) if daily else pd.Series(dates)
        frame = pd.DataFrame(
            {"timestamp": clock, "asset": _identity(namespace, sid), "source_sid": sid}
        )
        for name, array in zip(_FIELDS, arrays, strict=True):
            frame[name] = array[:, column]
        frame["source"] = "zipline_bundle"
        frame["revision_id"] = revision
        frame["available_at"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        bars.append(_frame(frame))
        supplied = frame[
            ["timestamp", "asset", "source_sid", "source", "revision_id", "available_at"]
        ].copy()
        supplied["price_adjustment_factor"] = curves[0][:, 0]
        supplied["volume_adjustment_factor"] = curves[1][:, 0]
        supplied["basis_at"] = upper.date() if daily else upper
        factors.append(_frame(supplied))
    return pl.concat(bars), pl.concat(factors), source_calendar.name


def _action_facts(
    raw: Mapping[str, pl.DataFrame], namespace: str, revision: str, calendar: str | None
) -> pl.DataFrame | None:
    rows = []
    for name in ("splits", *_PAYOUT_TABLES):
        for record in raw[name].iter_rows(named=True):
            payout = name != "splits"
            date_field = "ex_date" if payout else "effective_date"
            stamp = pd.Timestamp(record[date_field], unit="s").date()
            row = {
                "asset": _identity(namespace, record["sid"]),
                "source_sid": record["sid"],
                "source": "zipline_bundle",
                "event_id": f"{name}:{record['source_row']}:{record['sid']}",
                "source_record_id": record["source_row"],
                "revision_id": revision,
                "effective_at": stamp,
                "action_kind": "split"
                if not payout
                else "cash_dividend"
                if name == "dividend_payouts"
                else "stock_dividend",
                "ex_date": stamp,
                "payment_date": None,
                "entitlement_date": None,
                "available_at": None,
                "announced_at": None,
            }
            if name == "splits":
                row["split_ratio"] = record["ratio"]
            else:
                for source, target in (
                    ("pay_date", "payment_date"),
                    ("record_date", "entitlement_date"),
                    ("declared_date", "declared_date"),
                ):
                    epoch = record[source]
                    row[target] = (
                        None
                        if epoch == np.iinfo(np.int64).min
                        else pd.Timestamp(epoch, unit="s").date()
                    )
                if name == "dividend_payouts":
                    row["cash_per_share"] = record["amount"]
                else:
                    row["quantity_ratio"] = record["ratio"]
                    row["destination_asset"] = _identity(namespace, record["payment_sid"])
            rows.append(row)
    if not rows:
        return None
    return prepare_action_inputs(
        pl.from_dicts(rows, infer_schema_length=None),
        split_convention="price_factor",
        feed_spec=FeedSpec(
            timestamp_col="effective_at",
            entity_col="asset",
            time_contract_version=2,
            calendar=calendar,
            timestamp_semantics="session_label",
        ),
    )


def export_bundle(
    bundle: Any,
    path: str | Path,
    *,
    namespace: str,
    source_revision: str,
    sids: Sequence[int],
    start: Any,
    end: Any,
    frequencies: Sequence[str] = ("daily",),
    frequency_bounds: Mapping[str, tuple[Any, Any]] | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    effective_config: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | None] | None = None,
) -> InputSnapshot:
    """Read a bounded loaded BundleData without opening private stores or mutating it.

    The caller owns BundleData.close(). Namespace and source revision must be persistent.
    Unknown currency/publication/merger terms stay unknown. Exported symbol intervals are the
    reader's lookup policy, not inferred listing intervals. Daily and minute bounds are inclusive.
    """
    _require_zipline()
    if (
        not isinstance(namespace, str)
        or not namespace.strip()
        or not isinstance(source_revision, str)
        or not source_revision.strip()
    ):
        raise ValueError("Declare a persistent source namespace and captured source revision")
    if (
        isinstance(sids, str | bytes)
        or any(isinstance(sid, bool) or not isinstance(sid, Integral) or sid < 0 for sid in sids)
        or len(set(sids)) != len(sids)
    ):
        raise ValueError("Select distinct non-negative integer sids")
    sids = [int(sid) for sid in sids]
    if (
        not frequencies
        or len(set(frequencies)) != len(frequencies)
        or set(frequencies) - {"daily", "minute"}
    ):
        raise ValueError("Select daily and/or minute source frequencies")
    bounds = frequency_bounds or {}
    if set(bounds) - set(frequencies) or any(len(pair) != 2 for pair in bounds.values()):
        raise ValueError(
            "Frequency bounds must name selected frequencies and contain start/end pairs"
        )
    components = bundle.adjustment_reader.unpack_db_to_component_dfs(convert_dates=False)
    payouts = components["stock_dividend_payouts"]
    recipients = {int(sid) for sid in payouts.loc[payouts["sid"].isin(sids), "payment_sid"]}
    metadata_sids = [*sids, *sorted(recipients.difference(sids))]
    sources = _asset_tables(bundle, metadata_sids, namespace)
    tables = {
        name: SnapshotTable(frame, name, _source_metadata(namespace, source_revision, name))
        for name, frame in sources.items()
    }
    calendars = set()
    for frequency in frequencies:
        lower, upper = bounds.get(frequency, (start, end))
        bars, factors, source_calendar = _bars_and_factors(
            bundle, sids, lower, upper, namespace, source_revision, frequency
        )
        calendars.add(source_calendar)
        feed = FeedSpec(
            time_contract_version=2,
            calendar=source_calendar,
            entity_col="asset",
            data_frequency=frequency,
            timestamp_semantics="session_label" if frequency == "daily" else "bar_close",
            timezone="UTC",
            bar_type="time",
            bar_grid_anchor=None if frequency == "daily" else "session_open",
            bar_label=None if frequency == "daily" else "right",
        )
        for suffix, frame in (("bars", bars), ("factors", factors)):
            role = frequency + "_" + suffix
            metadata = _source_metadata(namespace, source_revision, role)
            metadata["source_reader_calendar"] = source_calendar
            metadata["source_schedule_provider"] = "exchange-calendars"
            metadata["source_schedule_provider_version"] = version("exchange-calendars")
            metadata["range_start"] = str(_clock(lower, daily=frequency == "daily"))
            metadata["range_end"] = str(_clock(upper, daily=frequency == "daily"))
            metadata["reader_missingness"] = (
                "missing raw prices retained as null; reader volume zeros retained without inferring no-trade"
            )
            if suffix == "factors":
                metadata["price_basis"] = "zipline_history_supplied_pricing_adjustments"
                metadata["factor_perspective"] = str(_clock(upper, daily=frequency == "daily"))
            tables[role] = SnapshotTable(frame, role, metadata, feed_spec=feed)
    raw = {}
    for name in (*_FACTOR_TABLES, *_PAYOUT_TABLES):
        pandas = components[name].copy()
        pandas.insert(0, "source_index", pandas.index.to_numpy())
        pandas.insert(0, "source_row", np.arange(len(pandas), dtype=np.int64))
        pandas = pandas[pandas["sid"].isin(sids)]
        raw[name] = _frame(pandas)
        tables["zipline_" + name] = SnapshotTable(
            raw[name], name, _source_metadata(namespace, source_revision, name)
        )
    if len(calendars) != 1:
        raise ValueError("The requested source readers use conflicting calendars")
    source_calendar = next(iter(calendars))
    selected_calendar = FeedSpec(
        time_contract_version=2, calendar=source_calendar
    ).resolve_calendar(calendar=calendar)
    actions = _action_facts(raw, namespace, source_revision, selected_calendar)
    if actions is not None:
        tables["actions"] = SnapshotTable(
            actions,
            "actions",
            _source_metadata(namespace, source_revision, "actions"),
            feed_spec=FeedSpec(
                time_contract_version=2,
                calendar=source_calendar,
                timestamp_col="effective_at",
                entity_col="asset",
                timestamp_semantics="session_label",
            ),
        )
    return export_snapshot(
        path,
        tables,
        effective_config={} if effective_config is None else effective_config,
        artifacts={} if artifacts is None else artifacts,
        calendar=calendar,
        unavailable={
            "historical_publication_instants": "Bundle capture and civil declared dates do not establish historical availability",
            "merger_consideration_terms": "Merger adjustment factors do not identify destination assets or cash/stock terms",
            "original_symbol_mapping_endpoints": "Public symbol ownership maps extend lookup intervals; they are not raw listing intervals",
            "quote_currency": "The source reader does not establish quote currency or price denomination",
            "root_sector_and_description": "Not exposed by the supported contract readers",
        },
    )


def _epoch_asset_frame(frame: pl.DataFrame) -> pd.DataFrame:
    result = polars_to_pandas(frame).drop(
        columns=["asset", "source_sid", "canonical_exchange"], errors="ignore"
    )
    for name in _ASSET_DATES:
        if name in result:
            result[name] = pd.to_datetime(result[name]).astype("datetime64[ns]").astype("int64")
    return result.set_index("sid")


def make_bundle_ingest(snapshot: str | Path) -> Callable[..., None]:
    """Return a standard Zipline ingest callable; validate before invoking any writer."""
    _require_zipline()
    path = Path(snapshot)
    expected_id = read_snapshot(path).snapshot_id

    def ingest(
        environ: Any,
        asset_db_writer: Any,
        minute_bar_writer: Any,
        daily_bar_writer: Any,
        adjustment_db_writer: Any,
        calendar: Any,
        start_session: Any,
        end_session: Any,
        cache: Any,
        show_progress: bool,
        output_dir: Any,
    ) -> None:
        del environ, cache, output_dir
        captured = read_snapshot(path)
        if captured.snapshot_id != expected_id:
            raise ValueError("The registered snapshot identity changed before ingestion")
        tables = captured.tables
        entries = captured.manifest["content"]["tables"]
        for name in ("daily_bars", "minute_bars"):
            if name in tables:
                if calendar.name != entries[name]["metadata"]["source_reader_calendar"]:
                    raise ValueError("Ingest calendar differs from the captured source schedule")
                low = (
                    pd.Timestamp(entries[name]["metadata"]["range_start"])
                    .tz_localize(None)
                    .normalize()
                )
                high = (
                    pd.Timestamp(entries[name]["metadata"]["range_end"])
                    .tz_localize(None)
                    .normalize()
                )
                if low < pd.Timestamp(start_session).tz_localize(None) or high > pd.Timestamp(
                    end_session
                ).tz_localize(None):
                    raise ValueError("Ingest range does not cover the captured bars")
        kwargs = {
            "exchanges": polars_to_pandas(tables["zipline_exchanges"]).set_index(
                "exchange", drop=False
            )
        }
        if "zipline_equities" in tables:
            kwargs["equities"] = _epoch_asset_frame(tables["zipline_equities"]).drop(
                columns=["symbol", "tick_size", "multiplier"], errors="ignore"
            )
            mappings = polars_to_pandas(tables["zipline_symbol_lookup"])
            for name in ("start_date", "end_date"):
                mappings[name] = (
                    pd.to_datetime(mappings[name]).astype("datetime64[ns]").astype("int64")
                )
            kwargs["equity_symbol_mappings"] = mappings
        if "zipline_futures" in tables:
            future = _epoch_asset_frame(tables["zipline_futures"])
            kwargs["futures"] = future
            kwargs["root_symbols"] = (
                future[["root_symbol", "exchange"]].drop_duplicates().set_index("root_symbol")
            )
        prepared_bars = []
        for name, writer in (("daily_bars", daily_bar_writer), ("minute_bars", minute_bar_writer)):
            if name not in tables:
                continue
            pandas = polars_to_pandas(tables[name])
            chunks = []
            for sid, group in pandas.groupby("source_sid", sort=False):
                frame = group.set_index(pd.DatetimeIndex(group["timestamp"]))[_FIELDS].copy()
                if name == "daily_bars":
                    frame.index = frame.index.tz_localize(None)
                scale = 1000 if name == "daily_bars" else writer.ohlc_ratio_for_sid(int(sid))
                prices = frame[_FIELDS[:4]].to_numpy()
                present = ~np.isnan(prices)
                encoded = np.rint(prices[present] * scale)
                if (
                    not np.isfinite(encoded).all()
                    or (encoded <= 0).any()
                    or (encoded >= np.iinfo(np.uint32).max).any()
                    or not np.array_equal(encoded * (1.0 / scale), prices[present])
                ):
                    raise ValueError(
                        f"The {name} writer cannot preserve price precision for sid {sid}; "
                        "configure a compatible minute price scale or retain the native snapshot"
                    )
                chunks.append((int(sid), frame))
            prepared_bars.append((writer, chunks))
        prepared_adjustments = {}
        for name in _FACTOR_TABLES:
            frame = (
                polars_to_pandas(tables["zipline_" + name])
                .set_index("source_index")
                .rename_axis(None)
                .drop(columns="source_row")
            )
            frame["effective_date"] = pd.to_datetime(frame["effective_date"], unit="s")
            prepared_adjustments[name] = frame
        cash = (
            polars_to_pandas(tables["zipline_dividend_payouts"])
            .set_index("source_index")
            .rename_axis(None)
            .drop(columns="source_row")
        )
        stock = (
            polars_to_pandas(tables["zipline_stock_dividend_payouts"])
            .set_index("source_index")
            .rename_axis(None)
            .drop(columns="source_row")
        )
        asset_db_writer.write_direct(**kwargs)
        for writer, chunks in prepared_bars:
            writer.write(chunks, show_progress=show_progress, invalid_data_behavior="raise")
        for name, frame in prepared_adjustments.items():
            adjustment_db_writer.write_frame(name, frame)
        adjustment_db_writer.write_dividend_payouts(cash)
        adjustment_db_writer.write_stock_dividend_payouts(stock)
        adjustment_db_writer.write()

    return ingest

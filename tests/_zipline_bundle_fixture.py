"""A real offline bundle producer using Zipline's normal ingestion mechanism."""

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def produce_bundle(root: Path, name: str, *, minute_scale: int | None = None):
    from zipline.data import bundles

    start, end = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    source = {}

    def ingest(
        environ,
        asset_db_writer,
        minute_bar_writer,
        daily_bar_writer,
        adjustment_db_writer,
        calendar,
        start_session,
        end_session,
        cache,
        show_progress,
        output_dir,
    ):
        del environ, cache, show_progress, output_dir
        sessions = calendar.sessions_in_range(start_session, end_session)
        equities = pd.DataFrame(
            {
                "sid": [1, 1, 2],
                "symbol": ["OLD", "NEW", "OTHER"],
                "asset_name": ["Renamed Company", "Renamed Company", "Other Venue"],
                "start_date": [start, sessions[1], start],
                "end_date": [start, end, end],
                "exchange": ["NYSE", "NYSE", "NASDAQ"],
            }
        ).set_index("sid")
        futures = pd.DataFrame(
            {
                "sid": [101, 102],
                "symbol": ["ESH24", "ESM24"],
                "root_symbol": ["ES", "ES"],
                "asset_name": ["March index future", "June index future"],
                "start_date": [start, start],
                "end_date": [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-06-21")],
                "exchange": ["CME", "CME"],
                "notice_date": [pd.Timestamp("2024-03-14"), pd.Timestamp("2024-06-20")],
                "expiration_date": [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-06-21")],
                "auto_close_date": [pd.Timestamp("2024-03-13"), pd.Timestamp("2024-06-19")],
                "tick_size": [0.25, 0.25],
                "multiplier": [50.0, 50.0],
            }
        ).set_index("sid")
        exchanges = pd.DataFrame(
            {
                "exchange": ["NYSE", "NASDAQ", "CME"],
                "canonical_name": ["XNYS", "XNYS", "XNYS"],
                "country_code": ["US", "US", "US"],
            }
        )
        roots = pd.DataFrame({"root_symbol": ["ES"], "exchange": ["CME"]}).set_index("root_symbol")
        asset_db_writer.write(
            equities=equities, futures=futures, exchanges=exchanges, root_symbols=roots
        )
        prices = {
            1: [100.0, 51.0, 52.0],
            2: [50.0, 52.0, 53.0],
            101: [4000.0, 4001.0, 4010.0],
            102: [4010.0, 4012.0, 4020.0],
        }
        daily = []
        for sid, close in prices.items():
            values = np.asarray(close)
            frame = pd.DataFrame(
                {
                    "open": values,
                    "high": values + 1,
                    "low": values - 1,
                    "close": values,
                    "volume": np.array([100, 200, 300], dtype=np.uint32),
                },
                index=sessions,
            )
            daily.append((sid, frame))
        daily_bar_writer.write(daily, invalid_data_behavior="raise")
        minute_dates = calendar.session_minutes(start)[:3]
        minutes = []
        for sid, close in prices.items():
            values = np.array([close[0], close[0] + 0.25, close[0] + 0.5])
            if minute_scale is not None and sid == 1:
                values = np.array([1.0001, 1.0002, 1.0003])
            frame = pd.DataFrame(
                {
                    "open": values,
                    "high": values + 0.25,
                    "low": values - 0.25,
                    "close": values,
                    "volume": np.array([10, 0, 30], dtype=np.uint32),
                },
                index=minute_dates,
            )
            if sid == 2:
                frame.loc[minute_dates[1], ["open", "high", "low", "close"]] = np.nan
            minutes.append((sid, frame))
        minute_bar_writer.write(minutes, invalid_data_behavior="raise")
        splits = pd.DataFrame(
            {"sid": np.array([1], dtype=np.int64), "effective_date": [sessions[1]], "ratio": [0.5]}
        )
        dividends = pd.DataFrame(
            {
                "sid": np.array([1], dtype=np.int64),
                "ex_date": [sessions[2]],
                "declared_date": [sessions[0]],
                "record_date": [sessions[2]],
                "pay_date": [pd.Timestamp("2024-01-10")],
                "amount": [1.0],
            }
        )
        stocks = pd.DataFrame(
            {
                "sid": np.array([1], dtype=np.int64),
                "payment_sid": np.array([2], dtype=np.int64),
                "ex_date": [sessions[2]],
                "declared_date": [sessions[0]],
                "record_date": [sessions[2]],
                "pay_date": [pd.Timestamp("2024-01-11")],
                "ratio": [0.25],
            }
        )
        adjustment_db_writer.write(splits=splits, dividends=dividends, stock_dividends=stocks)
        source.update(sessions=sessions, minute_dates=minute_dates)

    bundles.register(name, ingest, calendar_name="XNYS", start_session=start, end_session=end)
    environ = {"ZIPLINE_ROOT": str(root)}
    from zipline.data.bcolz_minute_bars import BcolzMinuteBarWriter

    configured = (
        nullcontext()
        if minute_scale is None
        else patch(
            "zipline.data.bundles.core.BcolzMinuteBarWriter",
            partial(BcolzMinuteBarWriter, default_ohlc_ratio=minute_scale),
        )
    )
    with configured:
        bundles.ingest(name, environ=environ, timestamp=pd.Timestamp("2024-02-01", tz="UTC"))
    return (
        bundles.load(name, environ=environ, timestamp=pd.Timestamp("2024-02-02", tz="UTC")),
        source,
        environ,
    )

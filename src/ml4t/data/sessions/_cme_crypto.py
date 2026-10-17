"""CME crypto's May 2026 transition, without changing the upstream registry."""

from __future__ import annotations

from typing import Any

import pandas as pd


class CMEGlobexCryptoSchedule:
    def __init__(self, legacy: Any, trade_dates: Any) -> None:
        self._legacy = legacy
        self._trade_dates = trade_dates
        self._business_day = trade_dates.holidays()
        self.tz = legacy.tz
        self.name = legacy.name

    def schedule(
        self, start_date: Any, end_date: Any, tz: Any = "UTC", *, interruptions: bool = False
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).tz_localize(None).normalize()
        end = pd.Timestamp(end_date).tz_localize(None).normalize()
        if start > end:
            raise ValueError("start_date must not be later than end_date")
        cutover = pd.Timestamp("2026-05-29")
        frames = []
        if start <= cutover:
            frames.append(
                self._legacy.schedule(start, min(end, cutover), tz=tz, interruptions=interruptions)
            )
        if end > cutover:
            labels = self._trade_dates.valid_days(
                max(start, cutover + pd.Timedelta(days=1)), end, tz=None
            )
            rows = []
            for label in labels:
                previous = label - self._business_day
                opening = previous + pd.Timedelta(hours=16, minutes=2)
                if previous == cutover:
                    opening = previous + pd.Timedelta(hours=16, minutes=30)
                closing = label + pd.Timedelta(hours=16)
                row = {
                    "market_open": opening.tz_localize(self.tz).tz_convert(tz),
                    "market_close": closing.tz_localize(self.tz).tz_convert(tz),
                }
                if interruptions:
                    closures = []
                    for day in pd.date_range(previous, label):
                        if day.dayofweek < 5:
                            halt = day + pd.Timedelta(hours=16)
                            reopen = halt + pd.Timedelta(minutes=2)
                        elif day.dayofweek == 5:
                            halt = day + pd.Timedelta(hours=2)
                            reopen = halt + pd.Timedelta(hours=2)
                        else:
                            continue
                        if opening < reopen and halt < closing:
                            closures.append((halt, reopen))
                    for index, (halt, reopen) in enumerate(closures, start=1):
                        row[f"interruption_start_{index}"] = halt.tz_localize(self.tz).tz_convert(
                            tz
                        )
                        row[f"interruption_end_{index}"] = reopen.tz_localize(self.tz).tz_convert(
                            tz
                        )
                rows.append(row)
            frames.append(
                pd.DataFrame(
                    rows, index=labels, columns=None if rows else ["market_open", "market_close"]
                )
            )
        return pd.concat(frames).sort_index()

    def schedule_for_instants(self, start: Any, end: Any) -> pd.DataFrame:
        start_timestamp, end_timestamp = pd.Timestamp(start), pd.Timestamp(end)
        start_local = (
            start_timestamp.tz_convert(self.tz) if start_timestamp.tzinfo else start_timestamp
        )
        end_local = end_timestamp.tz_convert(self.tz) if end_timestamp.tzinfo else end_timestamp
        next_trade_date = pd.Timestamp(end_local.date()) + self._business_day
        return self.schedule(
            pd.Timestamp(start_local.date()) - pd.Timedelta(days=1),
            next_trade_date,
            interruptions=True,
        )

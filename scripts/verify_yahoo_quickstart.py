"""Exercise single and batch conversion using only the installed Yahoo extra."""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import polars as pl

from ml4t.data.providers import YahooFinanceProvider


def main() -> None:
    with YahooFinanceProvider() as provider:
        for timezone, utc_hour in [(None, 0), ("America/New_York", 5)]:
            index = pd.DatetimeIndex(["2025-11-03"], tz=timezone, name="Date")
            frame = pd.DataFrame(
                {
                    field: pd.array([value], dtype="Int64")
                    for field, value in [
                        ("Open", 100),
                        ("High", 101),
                        ("Low", 99),
                        ("Close", 100),
                        ("Volume", 150),
                    ]
                },
                index=index,
            )
            multi = pd.concat({"SPY": frame, "QQQ": frame}, axis=1).swaplevel(axis=1)
            with patch("yfinance.download", return_value=multi):
                single = provider.fetch_ohlcv("SPY", "2025-11-03", "2025-11-03", "daily")
                batch = provider.fetch_batch_ohlcv(["SPY", "QQQ"], "2025-11-03", "2025-11-03")
            stamp = datetime(2025, 11, 3, utc_hour, tzinfo=UTC)
            for data, symbols in [(single, ["SPY"]), (batch, ["QQQ", "SPY"])]:
                assert data.schema["timestamp"] == pl.Datetime("us", "UTC"), data.schema
                assert data["timestamp"].to_list() == [stamp] * len(symbols)
                assert data["symbol"].to_list() == symbols
                assert data["volume"].to_list() == [150.0] * len(symbols)


if __name__ == "__main__":
    main()

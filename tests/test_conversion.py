"""Tests for DataFrame conversion utilities."""

from datetime import UTC, datetime

import pandas as pd
import polars as pl

from ml4t.data.utils.conversion import pandas_to_polars


def test_pandas_to_polars_preserves_mixed_types_and_nulls() -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", None], utc=True),
            "count": pd.Series([1, None], dtype="Int64"),
            "state": pd.Series(["open", None], dtype="string"),
        }
    )

    result = pandas_to_polars(source)

    assert result.schema == {
        "timestamp": pl.Datetime("us", "UTC"),
        "count": pl.Int64,
        "state": pl.String,
    }
    assert result.to_dicts() == [
        {"timestamp": datetime(2024, 1, 1, tzinfo=UTC), "count": 1, "state": "open"},
        {"timestamp": None, "count": None, "state": None},
    ]


def test_pandas_to_polars_preserves_nanosecond_timestamps() -> None:
    source = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01T00:00:00.123456789Z"], utc=True)}
    )

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("ns", "UTC")}
    assert result["timestamp"].cast(pl.Int64).item() == 1_704_067_200_123_456_789


def test_pandas_to_polars_preserves_all_null_float_dtype() -> None:
    source = pd.DataFrame({"value": pd.Series([None, None], dtype="float64")})

    result = pandas_to_polars(source)

    assert result.schema == {"value": pl.Float64}
    assert result["value"].null_count() == 2

"""Tests for DataFrame conversion utilities."""

from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest

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

    assert dict(result.schema) == {
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


def test_pandas_to_polars_normalizes_fixed_offset_timestamps() -> None:
    source = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T00:00:00.123456-05:00"])})

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("us", "UTC")}
    assert result["timestamp"].dt.hour().item() == 5


def test_pandas_to_polars_preserves_arrow_timestamp_dtype() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    dtype = pd.ArrowDtype(pyarrow.timestamp("ns", tz="UTC"))
    source = pd.DataFrame({"timestamp": pd.Series(["2024-01-01T00:00:00.123456789Z"], dtype=dtype)})

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("ns", "UTC")}
    assert result["timestamp"].cast(pl.Int64).item() == 1_704_067_200_123_456_789


def test_pandas_to_polars_preserves_all_null_duration_and_category_dtypes() -> None:
    source = pd.DataFrame(
        {
            "duration": pd.Series([None], dtype="timedelta64[ns]"),
            "category": pd.Series([None], dtype="category"),
        }
    )

    result = pandas_to_polars(source)

    assert dict(result.schema) == {
        "duration": pl.Duration("ns"),
        "category": pl.Categorical,
    }

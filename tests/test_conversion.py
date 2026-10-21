"""Tests for DataFrame conversion utilities."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.data.utils.conversion import pandas_to_polars


@pytest.mark.parametrize(
    "unit,tick", [("us", 1_704_067_200_000_000), ("ns", 1_704_067_200_000_000_000)]
)
def test_pandas_to_polars_preserves_mixed_types_and_nulls(unit: str, tick: int) -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", None], utc=True).as_unit(unit),
            "count": pd.Series([1, None], dtype="Int64"),
            "state": pd.Series(["open", None], dtype="string"),
        }
    )

    result = pandas_to_polars(source)

    assert dict(result.schema) == {
        "timestamp": pl.Datetime(unit, "UTC"),
        "count": pl.Int64,
        "state": pl.String,
    }
    assert result.to_dicts() == [
        {"timestamp": datetime(2024, 1, 1, tzinfo=UTC), "count": 1, "state": "open"},
        {"timestamp": None, "count": None, "state": None},
    ]
    assert result["timestamp"].cast(pl.Int64).to_list() == [tick, None]


def test_pandas_to_polars_preserves_nanosecond_timestamps() -> None:
    source = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01T00:00:00.123456789Z"], utc=True)}
    )

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("ns", "UTC")}
    assert result["timestamp"].cast(pl.Int64).item() == 1_704_067_200_123_456_789


def test_pandas_parquet_arrow_preserves_adjacent_nanosecond_observations(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    source_file = tmp_path / "observations.parquet"
    pl.DataFrame(
        {
            "timestamp": pl.Series(
                [1_704_067_200_123_456_789, 1_704_067_200_123_456_790],
                dtype=pl.Datetime("ns", "UTC"),
            ),
            "asset": [17, 17],
            "close": [100, 101],
        }
    ).write_parquet(source_file)
    captured = pd.read_parquet(source_file, dtype_backend="pyarrow")

    result = pandas_to_polars(captured)

    assert result.schema["timestamp"] == pl.Datetime("ns", "UTC")
    assert result["timestamp"].cast(pl.Int64).to_list() == [
        1_704_067_200_123_456_789,
        1_704_067_200_123_456_790,
    ]
    assert result["asset"].to_list() == [17, 17]
    assert result["close"].to_list() == [100, 101]


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize(
    "values", [[date(2025, 7, 3), None], [None, None], []], ids=["partial", "unknown", "empty"]
)
def test_pandas_feather_arrow_dates_remain_session_labels_not_instants(
    tmp_path, bits, values
) -> None:
    arrow = pytest.importorskip("pyarrow")
    dtype = arrow.date32() if bits == 32 else arrow.date64()
    source_file = tmp_path / "daily_labels.feather"
    pd.DataFrame({"timestamp": pd.Series(values, dtype=pd.ArrowDtype(dtype))}).to_feather(
        source_file
    )
    captured = pd.read_feather(source_file, dtype_backend="pyarrow")
    assert captured["timestamp"].dtype == pd.ArrowDtype(dtype)

    result = pandas_to_polars(captured)

    assert result.schema["timestamp"] == pl.Date
    assert result["timestamp"].to_list() == values


def test_pandas_to_polars_preserves_all_null_float_dtype() -> None:
    source = pd.DataFrame({"value": pd.Series([None, None], dtype="float64")})

    result = pandas_to_polars(source)

    assert result.schema == {"value": pl.Float64}
    assert result["value"].null_count() == 2


@pytest.mark.parametrize(
    "unit,timestamp,tick",
    [
        ("us", "2024-01-01T00:00:00.123456-05:00", 1_704_085_200_123_456),
        ("ns", "2024-01-01T00:00:00.123456789-05:00", 1_704_085_200_123_456_789),
    ],
)
def test_pandas_to_polars_normalizes_fixed_offset_timestamps(
    unit: str, timestamp: str, tick: int
) -> None:
    source = pd.DataFrame({"timestamp": pd.to_datetime([timestamp]).as_unit(unit)})

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime(unit, "UTC")}
    assert result["timestamp"].dt.hour().item() == 5
    assert result["timestamp"].cast(pl.Int64).item() == tick


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
        "category": pl.String,
    }


def test_pandas_to_polars_uses_one_dtype_for_partial_and_all_null_categories() -> None:
    partial = pandas_to_polars(
        pd.DataFrame({"category": pd.Series(["open", None], dtype="category")})
    )
    all_null = pandas_to_polars(
        pd.DataFrame({"category": pd.Series(pd.Categorical([None], categories=["open", "closed"]))})
    )

    assert partial.schema["category"] == pl.String
    assert all_null.schema["category"] == pl.String


@pytest.mark.parametrize(
    "unit,tick", [("us", 1_704_085_200_000_000), ("ns", 1_704_085_200_000_000_000)]
)
def test_pandas_to_polars_normalizes_named_timezones_to_utc(unit: str, tick: int) -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.Series(
                pd.date_range("2024-01-01", periods=1, tz="America/New_York").as_unit(unit)
            )
        }
    )

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime(unit, "UTC")}
    assert result["timestamp"].dt.hour().item() == 5
    assert result["timestamp"].cast(pl.Int64).item() == tick


def test_pandas_to_polars_rescales_second_timestamps() -> None:
    source = pd.DataFrame(
        {"timestamp": pd.Series(np.array(["2024-01-01T00:00:01"], dtype="datetime64[s]"))}
    )

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("ms")}
    assert result["timestamp"].cast(pl.Int64).item() == 1_704_067_201_000


def test_pandas_to_polars_preserves_tz_naive_arrow_timestamp() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    dtype = pd.ArrowDtype(pyarrow.timestamp("us"))
    source = pd.DataFrame({"timestamp": pd.Series(["2024-01-01T00:00:00.123456"], dtype=dtype)})

    result = pandas_to_polars(source)

    assert result.schema == {"timestamp": pl.Datetime("us")}
    assert result["timestamp"].cast(pl.Int64).item() == 1_704_067_200_123_456


def test_pandas_to_polars_preserves_all_null_arrow_duration_unit() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    dtype = pd.ArrowDtype(pyarrow.duration("ns"))
    source = pd.DataFrame({"duration": pd.Series([None], dtype=dtype)})

    result = pandas_to_polars(source)

    assert result.schema == {"duration": pl.Duration("ns")}

"""Exercise the public NASDAQ session recipe without private files or a working-directory dependency."""

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

EXAMPLE = Path(__file__).parents[1] / "examples/nasdaq_bars_sessions.py"
SAMPLE = EXAMPLE.parent / "data/nasdaq_bars_sample.csv"
EXPECTED_SESSIONS = [date(2025, 3, 7)] * 4 + [date(2025, 3, 10)] * 5


def run_example(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *map(str, args)],
        cwd=tmp_path,
        env={**os.environ, "HOME": str(tmp_path), "PYTHONWARNINGS": "error::ResourceWarning"},
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_default_example_runs_without_private_data_and_preserves_close_labeled_bars(tmp_path):
    result = run_example(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "illustrative" in result.stdout
    assigned = pl.read_parquet(tmp_path / "nasdaq_bars_with_sessions.parquet")
    assert assigned["session_date"].to_list() == EXPECTED_SESSIONS
    assert_frame_equal(assigned.drop("session_date"), pl.read_csv(SAMPLE, try_parse_dates=True))


def test_external_naive_chicago_bars_cross_dst_without_changing_values(tmp_path):
    bars = pl.read_csv(SAMPLE, try_parse_dates=True).with_columns(
        pl.col("datetime").dt.convert_time_zone("America/Chicago").dt.replace_time_zone(None)
    )
    source, output = tmp_path / "bars.parquet", tmp_path / "assigned.parquet"
    bars.write_parquet(source)
    result = run_example(
        tmp_path, "--input", source, "--timezone", "America/Chicago", "--output", output
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assigned = pl.read_parquet(output)
    assert assigned["session_date"].to_list() == EXPECTED_SESSIONS
    assert_frame_equal(assigned.drop("session_date"), bars)
    assert_frame_equal(pl.read_parquet(source), bars)


def test_external_naive_bars_require_timezone_instead_of_silently_assuming_utc(tmp_path):
    bars = pl.read_csv(SAMPLE, try_parse_dates=True).with_columns(
        pl.col("datetime").dt.replace_time_zone(None)
    )
    source = tmp_path / "bars.parquet"
    bars.write_parquet(source)
    result = run_example(tmp_path, "--input", source)
    assert result.returncode == 2
    assert "Naive timestamps require --timezone" in result.stderr
    assert not (tmp_path / "nasdaq_bars_with_sessions.parquet").exists()


@pytest.mark.parametrize("instant", ["2025-03-08T12:00:00Z", "2025-03-10T20:20:00Z"])
def test_example_rejects_weekend_and_break_observations_without_silent_filtering(tmp_path, instant):
    bars = (
        pl.read_csv(SAMPLE, try_parse_dates=True)
        .head(1)
        .with_columns(pl.lit(datetime.fromisoformat(instant)).alias("datetime"))
    )
    source = tmp_path / "bars.parquet"
    bars.write_parquet(source)
    result = run_example(tmp_path, "--input", source)
    assert result.returncode != 0
    assert "Observations outside a trading session" in result.stderr
    assert not (tmp_path / "nasdaq_bars_with_sessions.parquet").exists()


def test_open_labels_include_sunday_open_and_reject_monday_close(tmp_path):
    bars = (
        pl.read_csv(SAMPLE, try_parse_dates=True)
        .head(1)
        .with_columns(pl.lit(datetime.fromisoformat("2025-03-09T22:00:00Z")).alias("datetime"))
    )
    source = tmp_path / "bars.parquet"
    bars.write_parquet(source)
    result = run_example(tmp_path, "--input", source, "--bar-label", "open")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "nasdaq_bars_with_sessions.parquet"
    assert pl.read_parquet(output)["session_date"].to_list() == [date(2025, 3, 10)]
    output.unlink()
    bars.with_columns(
        pl.lit(datetime.fromisoformat("2025-03-10T21:00:00Z")).alias("datetime")
    ).write_parquet(source)
    result = run_example(tmp_path, "--input", source, "--bar-label", "open")
    assert result.returncode != 0
    assert "Observations outside a trading session" in result.stderr
    assert not output.exists()


def test_example_never_overwrites_its_input(tmp_path):
    source = tmp_path / "bars.parquet"
    bars = pl.read_csv(SAMPLE, try_parse_dates=True)
    bars.write_parquet(source)
    result = run_example(tmp_path, "--input", source, "--output", source)
    assert result.returncode == 2
    assert "--output must differ from --input" in result.stderr
    assert_frame_equal(pl.read_parquet(source), bars)


def test_example_prints_session_totals_on_windows_cp1252_console(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    result = run_example(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "session_date,n_bars,total_volume,total_trades\n" in result.stdout
    assert "2025-03-07,4,900,70\n" in result.stdout
    assert "2025-03-10,5,990,73\n" in result.stdout

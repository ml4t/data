"""
Windows Compatibility Test Suite for ml4t-data.
Validates that both data and metadata atomic writes work seamlessly 
without raising WinError 32 on Windows environments.
"""

import shutil
import polars as pl
from pathlib import Path
from datetime import datetime
import pytest

from ml4t.data.storage import HiveStorage, StorageConfig


@pytest.fixture
def test_storage_root():
    """Fixture to manage a clean temporary test storage directory."""
    path = Path.cwd() / "test_ml4t_storage_output"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    
    yield path
    
    # Cleanup after test completes
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def mock_ohlcv_df():
    """Fixture to generate standard Polars DataFrame matching ml4t-data schema."""
    dates = pl.date_range(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 5),
        interval="1d",
        eager=True
    )
    return pl.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [102.0, 103.0, 103.5, 102.5, 104.0],
        "low": [99.0, 100.5, 101.0, 100.0, 102.0],
        "close": [101.5, 102.5, 102.2, 101.8, 103.5],
        "volume": [1000, 1200, 1100, 1300, 1400]
    })


def test_hive_storage_windows_atomic_write(test_storage_root, mock_ohlcv_df):
    """
    Test that HiveStorage can execute sequential duplicate writes 
    onto both Parquet data and JSON metadata without triggering 
    NTFS file-locking conflicts (WinError 32).
    """
    # Initialize Storage Config matching ml4t specs
    config = StorageConfig(
        base_path=str(test_storage_root),
        partition_granularity="month"
    )
    storage = HiveStorage(config)
    
    logic_path = "mock_provider/daily/TEST_SYMBOL"

    # ---- PHASE 1: Initial Write (Creates Data and Metadata) ----
    try:
        storage.write(mock_ohlcv_df, logic_path)
    except PermissionError as e:
        pytest.fail(f"Initial write failed due to premature Windows file-locking: {e}")

    # ---- PHASE 2: Duplicate Overwrite (The core regression test) ----
    # On unpatched Windows, this step instantly crashes with WinError 32
    try:
        storage.write(mock_ohlcv_df, logic_path)
    except PermissionError as e:
        pytest.fail(f"Duplicate overwrite failed! Windows WinError 32 deadlock detected: {e}")

    # ---- PHASE 3: Physical Structure Assertions ----
    metadata_dir = test_storage_root / ".metadata"
    expected_json = metadata_dir / "mock_provider_daily_TEST_SYMBOL.json"
    assert expected_json.exists(), "Metadata JSON file was not generated correctly!"
    
    parquet_files = list(test_storage_root.rglob("*.parquet"))
    assert len(parquet_files) > 0, "No partitioned Parquet files were found!"

    # ---- PHASE 4: Data Load Verification (Aligned with native .read() API) ----
    # ml4t-data returns a Polars LazyFrame, we need to collect() it to get a DataFrame
    lazy_df = storage.read(logic_path)
    assert lazy_df is not None, "Loaded LazyFrame is None!"
    
    # 核心修复：通过 collect() 转换为 DataFrame 后再读取长度
    loaded_df = lazy_df.collect()
    assert len(loaded_df) == 5, f"Loaded DataFrame row count mismatch: expected 5, got {len(loaded_df)}"


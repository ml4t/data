"""Simple validation tests."""

from datetime import datetime

import polars as pl


def test_ohlcv_validator():
    """Test OHLCV validator basic functionality."""
    from ml4t.data.validation.ohlcv import OHLCVValidator

    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "volume": [1000000, 1100000],
        }
    )

    validator = OHLCVValidator()
    result = validator.validate(df)
    assert result.passed is True

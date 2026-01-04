"""ETF data management for ML4T book.

This module provides a unified interface for downloading and managing ETF data
from Yahoo Finance. It is designed to be simple for book readers to use while
providing robust data management features.

Example:
    >>> from ml4t.data.etfs import ETFDataManager
    >>> manager = ETFDataManager.from_config("configs/ml4t_etfs.yaml")
    >>> manager.download_all()
    >>> spy_data = manager.load_ohlcv("SPY")
"""

from ml4t.data.etfs.downloader import ETFDataManager

__all__ = ["ETFDataManager"]

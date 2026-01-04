"""Macro/economic data management for ML4T book.

This module provides a unified interface for downloading and managing
macroeconomic data from FRED (Federal Reserve Economic Data).

Example:
    >>> from ml4t.data.macro import MacroDataManager
    >>> manager = MacroDataManager.from_config("configs/ml4t_etfs.yaml")
    >>> manager.download_treasury_yields()
    >>> yields = manager.load_treasury_yields()
"""

from ml4t.data.macro.downloader import MacroDataManager

__all__ = ["MacroDataManager"]

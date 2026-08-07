"""
Futures data handling for ML4T.

This module provides tools for working with futures contracts:
- Contract specifications and metadata
- Continuous contract construction
- Roll logic and adjustment methods
- Databento data downloading and parsing
"""

from ml4t.data.futures.adjustment import (
    AdjustmentMethod,
    BackAdjustment,
    NoAdjustment,
    RatioAdjustment,
)
from ml4t.data.futures.book_downloader import (
    FuturesConfig,
    FuturesDataManager,
    download_futures_data,
    update_futures_data,
)
from ml4t.data.futures.config import (
    DEFAULT_PRODUCTS,
    DefinitionsConfig,
    DownloadProgress,
    FuturesCategory,
    FuturesDownloadConfig,
    load_definitions_config,
    load_yaml_config,
)
from ml4t.data.futures.continuous import (
    ContinuousContractBuilder,
    build_continuous_contract,
)
from ml4t.data.futures.databento_parser import (
    STAT_TYPE_CLEARED_VOLUME,
    # Stat type constants
    STAT_TYPE_OPEN_INTEREST,
    STAT_TYPE_SETTLEMENT_PRICE,
    ContractInfo,
    get_contract_chain,
    get_expiration_dates,
    get_front_back_contracts,
    load_databento_definitions,
    load_databento_ohlcv,
    load_databento_open_interest,
    load_databento_statistics,
    parse_contract_symbol,
    parse_databento,
    parse_databento_raw,
)
from ml4t.data.futures.parser import parse_quandl_chris, parse_quandl_chris_raw
from ml4t.data.futures.roll import (
    # Databento-compatible selection-based roll strategies
    CalendarRoll,
    # Original crossover-based roll strategies
    FirstNoticeDateRoll,
    HighestOpenInterestRoll,
    HighestVolumeRoll,
    OpenInterestBasedRoll,
    RollEvent,
    RollStrategy,
    TimeBasedRoll,
    VolumeBasedRoll,
)
from ml4t.data.futures.schema import (
    MAJOR_CONTRACTS,
    AssetClass,
    ContractSpec,
    ExchangeInfo,
    FuturesAssetClass,
    SettlementType,
)

_DATABENTO_EXPORTS = [
    "FuturesDownloader",
    "ContinuousDownloader",
    "ContinuousDownloadConfig",
    "ContinuousDownloadProgress",
    "load_continuous_config",
    "IndividualDownloader",
    "IndividualDownloadConfig",
    "IndividualProductConfig",
    "load_individual_config",
    "DefinitionsDownloader",
]
_DATABENTO_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from ml4t.data.futures import continuous_downloader as _continuous_downloader
    from ml4t.data.futures import downloader as _downloader
    from ml4t.data.futures import individual_downloader as _individual_downloader
except ModuleNotFoundError as error:
    missing_module = error.name or ""
    if missing_module != "databento" and not missing_module.startswith("databento."):
        raise
    _DATABENTO_IMPORT_ERROR = error
else:
    ContinuousDownloadConfig = _continuous_downloader.ContinuousDownloadConfig
    ContinuousDownloader = _continuous_downloader.ContinuousDownloader
    ContinuousDownloadProgress = _continuous_downloader.ContinuousDownloadProgress
    load_continuous_config = _continuous_downloader.load_continuous_config
    DefinitionsDownloader = _downloader.DefinitionsDownloader
    FuturesDownloader = _downloader.FuturesDownloader
    IndividualDownloadConfig = _individual_downloader.IndividualDownloadConfig
    IndividualDownloader = _individual_downloader.IndividualDownloader
    IndividualProductConfig = _individual_downloader.IndividualProductConfig
    load_individual_config = _individual_downloader.load_individual_config


def __getattr__(name: str):
    """Explain how to install public Databento-backed downloader symbols."""
    if name in _DATABENTO_EXPORTS and _DATABENTO_IMPORT_ERROR is not None:
        raise AttributeError(
            f"{name} requires the Databento extra: uv add 'ml4t-data[databento]'"
        ) from _DATABENTO_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Schema
    "AssetClass",  # Backward compat alias for FuturesAssetClass
    "FuturesAssetClass",
    "ContractSpec",
    "ExchangeInfo",
    "MAJOR_CONTRACTS",
    "SettlementType",
    "FuturesDownloadConfig",
    "FuturesCategory",
    "DownloadProgress",
    "DefinitionsConfig",
    "DEFAULT_PRODUCTS",
    "load_yaml_config",
    "load_definitions_config",
    # Parser (Quandl)
    "parse_quandl_chris",
    "parse_quandl_chris_raw",
    # Parser (Databento)
    "parse_databento",
    "parse_databento_raw",
    "load_databento_ohlcv",
    "load_databento_definitions",
    "load_databento_open_interest",
    "load_databento_statistics",
    "get_expiration_dates",
    "get_contract_chain",
    "get_front_back_contracts",
    "parse_contract_symbol",
    "ContractInfo",
    # Databento stat type constants
    "STAT_TYPE_OPEN_INTEREST",
    "STAT_TYPE_SETTLEMENT_PRICE",
    "STAT_TYPE_CLEARED_VOLUME",
    # Roll strategies (original crossover-based)
    "RollStrategy",
    "RollEvent",
    "VolumeBasedRoll",
    "OpenInterestBasedRoll",
    "TimeBasedRoll",
    "FirstNoticeDateRoll",
    # Roll strategies (Databento-compatible selection-based)
    "CalendarRoll",
    "HighestVolumeRoll",
    "HighestOpenInterestRoll",
    # Adjustment methods
    "AdjustmentMethod",
    "BackAdjustment",
    "RatioAdjustment",
    "NoAdjustment",
    # Continuous contract builder
    "ContinuousContractBuilder",
    "build_continuous_contract",
    # Book downloader (simplified interface for ML4T readers)
    "FuturesDataManager",
    "FuturesConfig",
    "download_futures_data",
    "update_futures_data",
] + ([] if _DATABENTO_IMPORT_ERROR is not None else _DATABENTO_EXPORTS)

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
from ml4t.data.futures.downloader import (
    DEFAULT_PRODUCTS,
    DefinitionsConfig,
    DefinitionsDownloader,
    DownloadProgress,
    FuturesCategory,
    FuturesDownloadConfig,
    FuturesDownloader,
    load_definitions_config,
    load_yaml_config,
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
    RollStrategy,
    TimeBasedRoll,
    VolumeBasedRoll,
)
from ml4t.data.futures.schema import (
    MAJOR_CONTRACTS,
    AssetClass,
    ContractSpec,
    ExchangeInfo,
    SettlementType,
)

__all__ = [
    # Schema
    "AssetClass",
    "ContractSpec",
    "ExchangeInfo",
    "MAJOR_CONTRACTS",
    "SettlementType",
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
    # Downloader
    "FuturesDownloader",
    "FuturesDownloadConfig",
    "FuturesCategory",
    "DownloadProgress",
    "DEFAULT_PRODUCTS",
    "load_yaml_config",
    # Definitions downloader
    "DefinitionsDownloader",
    "DefinitionsConfig",
    "load_definitions_config",
    # Book downloader (simplified interface for ML4T readers)
    "FuturesDataManager",
    "FuturesConfig",
    "download_futures_data",
    "update_futures_data",
]

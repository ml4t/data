"""
Continuous futures contract builder.

Combines parser, roll strategy, and adjustment method to build
continuous futures time series from individual contract months.

Supports multiple data sources:
- quandl_chris: Quandl CHRIS futures data (legacy)
- databento: Databento futures data (recommended)
"""

from pathlib import Path

import polars as pl

from ml4t.data.futures.adjustment import AdjustmentMethod, BackAdjustment
from ml4t.data.futures.parser import parse_quandl_chris, parse_quandl_chris_raw
from ml4t.data.futures.roll import RollStrategy, VolumeBasedRoll
from ml4t.data.futures.schema import ContractSpec


class ContinuousContractBuilder:
    """
    Build continuous futures contracts from individual contract months.

    Orchestrates the process of:
    1. Parsing raw futures data
    2. Identifying roll dates (when to switch contracts)
    3. Adjusting historical prices to create continuous series
    """

    def __init__(
        self,
        contract_spec: ContractSpec | None = None,
        roll_strategy: RollStrategy | None = None,
        adjustment_method: AdjustmentMethod | None = None,
    ):
        """
        Initialize continuous contract builder.

        Args:
            contract_spec: Optional contract specifications (from schema.py)
            roll_strategy: Strategy for identifying roll dates (default: VolumeBasedRoll)
            adjustment_method: Method for adjusting prices (default: BackAdjustment)
        """
        self.contract_spec = contract_spec
        self.roll_strategy = roll_strategy or VolumeBasedRoll()
        self.adjustment_method = adjustment_method or BackAdjustment()

    def build(
        self,
        ticker: str,
        data_source: str = "quandl_chris",
        storage_path: str | Path | None = None,
    ) -> pl.DataFrame:
        """
        Build continuous contract for a ticker.

        Args:
            ticker: Contract ticker (e.g., "CL" for crude oil, "ES" for E-mini S&P 500)
            data_source: Data source identifier:
                - "quandl_chris": Quandl CHRIS data (legacy)
                - "databento": Databento downloaded data
            storage_path: For databento, path where data was downloaded.
                         Defaults to ~/ml4t-data/futures

        Returns:
            DataFrame with columns:
            - date: pl.Date
            - open, high, low, close: float (unadjusted prices)
            - adjusted_open, adjusted_high, adjusted_low, adjusted_close: float
            - volume: float
            - open_interest: float (nullable)
            - is_roll_date: bool (True on roll dates)

        Examples:
            >>> # Build ES continuous with default settings (Quandl)
            >>> builder = ContinuousContractBuilder()
            >>> es_continuous = builder.build("ES")

            >>> # Build ES continuous from Databento data
            >>> builder = ContinuousContractBuilder(roll_strategy=TimeBasedRoll())
            >>> es_continuous = builder.build("ES", data_source="databento")

            >>> # Build CL continuous with ratio adjustment
            >>> from ml4t.data.futures.adjustment import RatioAdjustment
            >>> builder = ContinuousContractBuilder(adjustment_method=RatioAdjustment())
            >>> cl_continuous = builder.build("CL")
        """
        # 1. Parse data based on source
        if data_source == "quandl_chris":
            # Get raw multi-contract data for roll detection
            raw_data = parse_quandl_chris_raw(ticker)

            # Get clean continuous data (front month only)
            continuous_data = parse_quandl_chris(ticker)

        elif data_source == "databento":
            from ml4t.data.futures.databento_parser import (
                parse_databento,
                parse_databento_raw,
            )

            # Default storage path
            if storage_path is None:
                storage_path = Path("~/ml4t-data/futures").expanduser()
            else:
                storage_path = Path(storage_path).expanduser()

            # Get raw multi-contract data (includes expiration dates)
            raw_data = parse_databento_raw(ticker, storage_path)

            # Get clean continuous data (front month only)
            continuous_data = parse_databento(ticker, storage_path)

        else:
            raise ValueError(
                f"Unknown data source: {data_source}. Supported: 'quandl_chris', 'databento'"
            )

        # 2. Identify rolls using raw multi-contract data
        roll_dates = self.roll_strategy.identify_rolls(raw_data, self.contract_spec)

        # 3. Apply adjustment to continuous data
        adjusted_data = self.adjustment_method.adjust(continuous_data, roll_dates)

        # 4. Add is_roll_date column
        result = adjusted_data.with_columns(pl.col("date").is_in(roll_dates).alias("is_roll_date"))

        return result

    def build_multiple(
        self,
        tickers: list[str],
        data_source: str = "quandl_chris",
        storage_path: str | Path | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Build continuous contracts for multiple tickers.

        Args:
            tickers: List of contract tickers
            data_source: Data source identifier
            storage_path: For databento, path where data was downloaded

        Returns:
            Dictionary mapping ticker -> continuous DataFrame

        Examples:
            >>> builder = ContinuousContractBuilder()
            >>> data = builder.build_multiple(["ES", "CL", "GC"])
            >>> es_data = data["ES"]
            >>> cl_data = data["CL"]

            >>> # With Databento data
            >>> builder = ContinuousContractBuilder(roll_strategy=TimeBasedRoll())
            >>> data = builder.build_multiple(["ES", "CL"], data_source="databento")
        """
        return {ticker: self.build(ticker, data_source, storage_path) for ticker in tickers}


def build_continuous_contract(
    ticker: str,
    roll_strategy: RollStrategy | None = None,
    adjustment_method: AdjustmentMethod | None = None,
    contract_spec: ContractSpec | None = None,
    data_source: str = "quandl_chris",
    storage_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    Convenience function to build a continuous contract.

    Equivalent to:
        builder = ContinuousContractBuilder(contract_spec, roll_strategy, adjustment_method)
        return builder.build(ticker, data_source, storage_path)

    Args:
        ticker: Contract ticker
        roll_strategy: Roll strategy (default: VolumeBasedRoll)
        adjustment_method: Adjustment method (default: BackAdjustment)
        contract_spec: Contract specifications
        data_source: Data source ("quandl_chris" or "databento")
        storage_path: For databento, path where data was downloaded

    Returns:
        Continuous contract DataFrame

    Examples:
        >>> # Simple usage with defaults (Quandl)
        >>> es_data = build_continuous_contract("ES")

        >>> # With Databento data and time-based rolling
        >>> es_data = build_continuous_contract(
        ...     "ES",
        ...     roll_strategy=TimeBasedRoll(days_before_expiration=5),
        ...     data_source="databento",
        ... )

        >>> # With specific strategies
        >>> from ml4t.data.futures.roll import OpenInterestBasedRoll
        >>> from ml4t.data.futures.adjustment import RatioAdjustment
        >>> cl_data = build_continuous_contract(
        ...     "CL",
        ...     roll_strategy=OpenInterestBasedRoll(),
        ...     adjustment_method=RatioAdjustment()
        ... )
    """
    builder = ContinuousContractBuilder(contract_spec, roll_strategy, adjustment_method)
    return builder.build(ticker, data_source, storage_path)

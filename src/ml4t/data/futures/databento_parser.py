"""
Databento futures data parser.

Parse downloaded Databento futures data (OHLCV + definitions) for building
continuous contracts. Works with data downloaded via FuturesDownloader.

Key features:
- Load individual contract OHLCV data
- Parse expiration dates from definition schema
- Contract symbol parsing (ESH25 -> ES + March 2025)
- Compatible with ContinuousContractBuilder architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from ml4t.data.futures.schema import ContractSpec

# Month code to month number mapping (CME standard)
MONTH_CODES = {
    "F": 1,  # January
    "G": 2,  # February
    "H": 3,  # March
    "J": 4,  # April
    "K": 5,  # May
    "M": 6,  # June
    "N": 7,  # July
    "Q": 8,  # August
    "U": 9,  # September
    "V": 10,  # October
    "X": 11,  # November
    "Z": 12,  # December
}

# Reverse mapping
MONTH_TO_CODE = {v: k for k, v in MONTH_CODES.items()}


@dataclass
class ContractInfo:
    """Parsed contract information from symbol."""

    raw_symbol: str  # Full symbol (ESH25)
    product: str  # Root symbol (ES)
    month_code: str  # Month code (H)
    month: int  # Month number (3)
    year: int  # Full year (2025)
    expiration: date | None = None  # From definition schema

    @property
    def contract_month(self) -> str:
        """Contract month in YYYY-MM format."""
        return f"{self.year}-{self.month:02d}"

    def __hash__(self) -> int:
        return hash(self.raw_symbol)


def parse_contract_symbol(symbol: str) -> ContractInfo:
    """
    Parse Databento contract symbol into components.

    Supports two formats:
    - 2-digit year: {PRODUCT}{MONTH}{YY} (e.g., ESH25 -> ES + H + 2025)
    - 1-digit year: {PRODUCT}{MONTH}{Y} (e.g., ZMK9 -> ZM + K + 2019)

    Examples:
        - ESH25 -> ES (E-mini S&P), H (March), 25 (2025)
        - CLZ24 -> CL (Crude), Z (December), 24 (2024)
        - ZMK9 -> ZM (Soybean Meal), K (May), 9 (2019)
        - 6EH25 -> 6E (Euro FX), H (March), 25 (2025)

    Args:
        symbol: Raw contract symbol (e.g., "ESH25" or "ZMK9")

    Returns:
        ContractInfo with parsed components

    Raises:
        ValueError: If symbol format is invalid
    """
    if len(symbol) < 4:
        raise ValueError(f"Invalid symbol format: {symbol}")

    # Try 2-digit year format first (more common in newer data)
    # Check if last char is a month code - if so, it's 1-digit year format
    if symbol[-2] in MONTH_CODES:
        # 1-digit year format: {PRODUCT}{MONTH}{Y}
        year_str = symbol[-1]
        try:
            year_short = int(year_str)
        except ValueError:
            raise ValueError(f"Invalid year in symbol: {symbol}")

        # Single digit year - assume 2010s or 2020s based on context
        # 0-5 = 2020-2025, 6-9 = 2016-2019 (reasonable for current trading)
        year = 2020 + year_short if year_short <= 5 else 2010 + year_short

        month_code = symbol[-2]
        product = symbol[:-2]
    else:
        # 2-digit year format: {PRODUCT}{MONTH}{YY}
        year_str = symbol[-2:]
        try:
            year_short = int(year_str)
        except ValueError:
            raise ValueError(f"Invalid year in symbol: {symbol}")

        # Convert 2-digit year to 4-digit
        # Assume 00-29 = 2000-2029, 30-99 = 1930-1999
        year = 2000 + year_short if year_short < 30 else 1900 + year_short

        month_code = symbol[-3]
        product = symbol[:-3]

    if month_code not in MONTH_CODES:
        raise ValueError(f"Invalid month code '{month_code}' in symbol: {symbol}")
    month = MONTH_CODES[month_code]

    if not product:
        raise ValueError(f"No product symbol in: {symbol}")

    return ContractInfo(
        raw_symbol=symbol,
        product=product,
        month_code=month_code,
        month=month,
        year=year,
    )


def load_databento_definitions(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> pl.DataFrame:
    """
    Load contract definitions for a product from downloaded Databento data.

    Returns DataFrame with expiration dates and other contract metadata.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded

    Returns:
        DataFrame with columns:
        - raw_symbol: str (contract symbol)
        - product: str (root symbol)
        - expiration: datetime (contract expiration)
        - activation: datetime (when contract became active)
        - instrument_class: str
        Plus original Databento columns

    Raises:
        FileNotFoundError: If definition data not found
    """
    storage_path = Path(storage_path).expanduser()
    definition_file = storage_path / "definition" / f"product={product}" / "definition.parquet"

    if not definition_file.exists():
        raise FileNotFoundError(
            f"Definition data not found for {product}. Expected at: {definition_file}"
        )

    df = pl.read_parquet(definition_file)

    # Filter out errors (from failed downloads)
    if "error" in df.columns:
        df = df.filter(pl.col("error").is_null())

    if df.height == 0:
        raise ValueError(f"No definition data available for {product}")

    return df


def load_databento_ohlcv(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> pl.DataFrame:
    """
    Load OHLCV data for all contracts of a product.

    Returns multi-contract data (one row per contract per date).

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded

    Returns:
        DataFrame with columns:
        - date: pl.Date
        - symbol: str (specific contract, e.g., "ESH25")
        - open, high, low, close: float
        - volume: int

    Raises:
        FileNotFoundError: If OHLCV data not found
    """
    storage_path = Path(storage_path).expanduser()
    ohlcv_file = storage_path / "ohlcv_1d" / f"product={product}" / "ohlcv_1d.parquet"

    if not ohlcv_file.exists():
        raise FileNotFoundError(f"OHLCV data not found for {product}. Expected at: {ohlcv_file}")

    df = pl.read_parquet(ohlcv_file)

    # Filter out errors
    if "error" in df.columns:
        df = df.filter(pl.col("error").is_null())

    if df.height == 0:
        raise ValueError(f"No OHLCV data available for {product}")

    return df


# Databento stat_type values (from databento_dbn.StatType enum)
STAT_TYPE_OPENING_PRICE = 1
STAT_TYPE_INDICATIVE_OPENING_PRICE = 2
STAT_TYPE_SETTLEMENT_PRICE = 3
STAT_TYPE_TRADING_SESSION_LOW_PRICE = 4
STAT_TYPE_TRADING_SESSION_HIGH_PRICE = 5
STAT_TYPE_CLEARED_VOLUME = 6
STAT_TYPE_LOWEST_OFFER = 7
STAT_TYPE_HIGHEST_BID = 8
STAT_TYPE_OPEN_INTEREST = 9
STAT_TYPE_FIXING_PRICE = 10


def load_databento_statistics(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
    stat_types: list[int] | None = None,
) -> pl.DataFrame:
    """
    Load statistics data (OI, settlement, etc.) from Databento data.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded
        stat_types: Optional filter for specific stat types.
                   Default: [STAT_TYPE_OPEN_INTEREST] (OI only)

    Returns:
        DataFrame with columns:
        - date: pl.Date (from ts_event)
        - symbol: str (contract symbol)
        - stat_type: int
        - value: float (price for prices, quantity for OI/volume)

    Raises:
        FileNotFoundError: If statistics data not found
    """
    storage_path = Path(storage_path).expanduser()
    stats_file = storage_path / "statistics" / f"product={product}" / "statistics.parquet"

    if not stats_file.exists():
        raise FileNotFoundError(
            f"Statistics data not found for {product}. Expected at: {stats_file}"
        )

    df = pl.read_parquet(stats_file)

    # Filter out errors
    if "error" in df.columns:
        df = df.filter(pl.col("error").is_null())

    if df.height == 0:
        raise ValueError(f"No statistics data available for {product}")

    # Filter by stat_type if specified
    if stat_types is None:
        stat_types = [STAT_TYPE_OPEN_INTEREST]

    df = df.filter(pl.col("stat_type").is_in(stat_types))

    return df


def load_databento_open_interest(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> pl.DataFrame:
    """
    Load daily open interest data for all contracts of a product.

    OI is typically published once per day at end of session.
    Returns the latest OI reading for each contract per day.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded

    Returns:
        DataFrame with columns:
        - date: pl.Date
        - symbol: str (contract symbol)
        - open_interest: int

    Raises:
        FileNotFoundError: If statistics data not found
    """
    stats = load_databento_statistics(product, storage_path, stat_types=[STAT_TYPE_OPEN_INTEREST])

    # Extract date and latest OI per contract per day
    oi_data = (
        stats.with_columns(pl.col("ts_event").cast(pl.Date).alias("date"))
        .sort(["symbol", "date", "ts_event"])
        # Take latest reading per day per contract
        .group_by(["symbol", "date"])
        .agg(pl.col("quantity").last().alias("open_interest"))
        .sort(["date", "symbol"])
    )

    return oi_data


def get_expiration_dates(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> dict[str, date]:
    """
    Get expiration dates for all contracts of a product.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded

    Returns:
        Dictionary mapping contract symbol to expiration date
        {
            "ESH24": date(2024, 3, 15),
            "ESM24": date(2024, 6, 21),
            ...
        }
    """
    definitions = load_databento_definitions(product, storage_path)

    # Extract expiration dates
    # Databento stores expiration as Unix nanoseconds
    expirations = {}

    # Get unique contracts with their expiration
    # Handle both 'raw_symbol' (Databento native) and 'symbol' (our downloaded data) columns
    symbol_col = "raw_symbol" if "raw_symbol" in definitions.columns else "symbol"

    if symbol_col in definitions.columns and "expiration" in definitions.columns:
        # Group by symbol and get the expiration
        contract_expirations = (
            definitions.select([symbol_col, "expiration"])
            .unique(subset=[symbol_col])
            .filter(pl.col("expiration").is_not_null())
        )

        for row in contract_expirations.iter_rows(named=True):
            symbol = row[symbol_col]
            exp = row["expiration"]

            # Convert to date if needed
            if isinstance(exp, datetime):
                expirations[symbol] = exp.date()
            elif isinstance(exp, date):
                expirations[symbol] = exp
            elif isinstance(exp, int | float):
                # Unix nanoseconds
                expirations[symbol] = datetime.fromtimestamp(exp / 1e9).date()

    return expirations


def parse_databento_raw(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
    _contract_spec: ContractSpec | None = None,
    include_open_interest: bool = True,
) -> pl.DataFrame:
    """
    Parse Databento futures data without deduplication (keeps all contracts).

    Returns multi-contract data with duplicate dates, useful for roll analysis.
    Compatible with existing ContinuousContractBuilder interface.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded
        _contract_spec: Optional contract specifications (unused, for API compat)
        include_open_interest: Whether to load and merge OI from statistics (default True)

    Returns:
        DataFrame with potentially multiple rows per date (one per contract):
        - date: pl.Date
        - symbol: str (contract symbol)
        - open, high, low, close: float
        - volume: float
        - open_interest: float (from statistics if available, else null)
        - expiration: date (from definition schema)
    """
    storage_path = Path(storage_path).expanduser()
    ohlcv = load_databento_ohlcv(product, storage_path)

    # Get expiration dates
    try:
        expirations = get_expiration_dates(product, storage_path)
    except FileNotFoundError:
        expirations = {}

    # Standardize column names
    # Databento uses: ts_event (timestamp), open, high, low, close, volume
    rename_map = {}

    # Handle timestamp column
    if "ts_event" in ohlcv.columns:
        rename_map["ts_event"] = "timestamp"

    if rename_map:
        ohlcv = ohlcv.rename(rename_map)

    # Extract date from timestamp
    if "timestamp" in ohlcv.columns:
        ohlcv = ohlcv.with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))

    # Get symbol column (Databento uses 'symbol' or 'raw_symbol')
    symbol_col = "symbol" if "symbol" in ohlcv.columns else "raw_symbol"

    # Add expiration column
    if expirations:
        ohlcv = ohlcv.with_columns(
            pl.col(symbol_col)
            .map_elements(
                lambda s: expirations.get(s),
                return_dtype=pl.Date,
            )
            .alias("expiration")
        )
    else:
        ohlcv = ohlcv.with_columns(pl.lit(None).cast(pl.Date).alias("expiration"))

    # Load and merge open interest from statistics
    oi_merged = False
    if include_open_interest:
        try:
            oi_data = load_databento_open_interest(product, storage_path)
            if oi_data.height > 0:
                # Join OI data with OHLCV
                ohlcv = ohlcv.join(
                    oi_data,
                    left_on=["date", symbol_col],
                    right_on=["date", "symbol"],
                    how="left",
                )
                oi_merged = True
        except FileNotFoundError:
            pass  # No statistics data available

    # Ensure open_interest column exists
    if not oi_merged and "open_interest" not in ohlcv.columns:
        ohlcv = ohlcv.with_columns(pl.lit(None).cast(pl.Float64).alias("open_interest"))

    # Select and standardize output columns
    result = ohlcv.select(
        [
            pl.col("date"),
            pl.col(symbol_col).alias("symbol"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("open_interest").cast(pl.Float64),
            pl.col("expiration"),
        ]
    ).sort(["date", "symbol"])

    return result


def parse_databento(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
    _contract_spec: ContractSpec | None = None,
) -> pl.DataFrame:
    """
    Parse Databento futures data for a specific product.

    Returns front-month only data (single row per date).
    Compatible with existing ContinuousContractBuilder interface.

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded
        _contract_spec: Optional contract specifications (unused, for API compat)

    Returns:
        Clean DataFrame with single row per date, columns:
        - date: pl.Date
        - open, high, low, close: float
        - volume: float
        - open_interest: float (nullable)
    """
    # Get raw multi-contract data
    raw_data = parse_databento_raw(product, storage_path, _contract_spec)

    # Select front month by highest volume
    result = (
        raw_data.sort(["date", "volume"], descending=[False, True])
        .group_by("date")
        .agg(pl.all().first())
        .sort("date")
    )

    # Select only standard OHLCV columns
    return result.select(
        [
            pl.col("date"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            pl.col("open_interest"),
        ]
    )


def get_contract_chain(
    product: str,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> list[ContractInfo]:
    """
    Get the full contract chain with expiration dates.

    Returns contracts sorted by expiration date (front to back).

    Args:
        product: Product symbol (e.g., "ES", "CL")
        storage_path: Base path where Databento data was downloaded

    Returns:
        List of ContractInfo objects sorted by expiration
    """
    expirations = get_expiration_dates(product, storage_path)

    contracts = []
    for symbol, exp_date in expirations.items():
        try:
            info = parse_contract_symbol(symbol)
            info.expiration = exp_date
            contracts.append(info)
        except ValueError:
            # Skip invalid symbols (e.g., spreads)
            continue

    # Sort by expiration
    contracts.sort(key=lambda c: c.expiration or date.max)

    return contracts


def get_front_back_contracts(
    product: str,
    as_of_date: date,
    storage_path: str | Path = "~/ml4t-data/futures",
) -> tuple[ContractInfo | None, ContractInfo | None]:
    """
    Get front and back month contracts as of a specific date.

    Front month = nearest unexpired contract
    Back month = second nearest unexpired contract

    Args:
        product: Product symbol
        as_of_date: Date to evaluate
        storage_path: Base path where Databento data was downloaded

    Returns:
        Tuple of (front_contract, back_contract)
        Either can be None if not available
    """
    chain = get_contract_chain(product, storage_path)

    # Filter to unexpired contracts
    unexpired = [c for c in chain if c.expiration and c.expiration > as_of_date]

    front = unexpired[0] if len(unexpired) > 0 else None
    back = unexpired[1] if len(unexpired) > 1 else None

    return front, back

"""
Futures contract specifications and metadata schema.

This module defines the data structures for representing futures contract
specifications, including multipliers, tick sizes, expiration rules, and
other contract-specific metadata.

Educational Note:
    Understanding contract specifications is critical for:
    - Correct price conversions (cents vs dollars)
    - Accurate P&L calculations (contract multiplier)
    - Proper roll timing (expiration dates)
    - Risk management (tick value, position sizing)
"""

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """Asset class classification for futures contracts."""

    EQUITY_INDEX = "equity_index"
    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    CURRENCY = "currency"
    FIXED_INCOME = "fixed_income"
    VOLATILITY = "volatility"


class SettlementType(str, Enum):
    """Settlement method for futures contracts."""

    CASH = "cash"  # Cash-settled (e.g., equity indices)
    PHYSICAL = "physical"  # Physical delivery (e.g., commodities)


class ExchangeInfo:
    """Exchange-specific information."""

    CME = "CME"  # Chicago Mercantile Exchange
    CBOT = "CBOT"  # Chicago Board of Trade
    NYMEX = "NYMEX"  # New York Mercantile Exchange
    COMEX = "COMEX"  # Commodity Exchange
    ICE = "ICE"  # Intercontinental Exchange
    EUREX = "EUREX"  # European Exchange


@dataclass
class ContractSpec:
    """
    Futures contract specification.

    This dataclass captures all the essential metadata needed to work with
    a futures contract correctly.

    Attributes:
        ticker: Contract root symbol (e.g., "ES", "CL", "GC")
        name: Full contract name (e.g., "E-mini S&P 500")
        exchange: Exchange where contract trades
        asset_class: Asset class classification
        multiplier: Contract multiplier (points to dollars)
        tick_size: Minimum price movement
        tick_value: Dollar value of one tick
        price_quote_unit: Unit for price quotes ("dollars", "cents", "index_points")
        settlement_type: Cash or physical settlement
        contract_months: Trading months (e.g., "HMUZ" for Mar/Jun/Sep/Dec)
        first_notice_days: Days before expiration for first notice
        last_trading_offset: Offset from month-end for last trading day
        currency: Currency denomination
        trading_hours: Trading hours description
        point_value: Dollar value per point (alternative to multiplier)

    Example:
        >>> es_spec = ContractSpec(
        ...     ticker="ES",
        ...     name="E-mini S&P 500",
        ...     exchange=ExchangeInfo.CME,
        ...     asset_class=AssetClass.EQUITY_INDEX,
        ...     multiplier=50.0,
        ...     tick_size=0.25,
        ...     tick_value=12.50,
        ...     price_quote_unit="index_points",
        ...     settlement_type=SettlementType.CASH,
        ...     contract_months="HMUZ",
        ... )
        >>> es_spec.point_value
        50.0
        >>> es_spec.tick_value
        12.5
    """

    ticker: str
    name: str
    exchange: str
    asset_class: AssetClass
    multiplier: float
    tick_size: float
    tick_value: float
    price_quote_unit: str  # "dollars", "cents", "index_points", etc.
    settlement_type: SettlementType
    contract_months: str  # e.g., "HMUZ" for Mar/Jun/Sep/Dec

    # Optional fields
    first_notice_days: int | None = None
    last_trading_offset: int | None = None
    currency: str = "USD"
    trading_hours: str | None = None
    point_value: float | None = None  # Alternative to multiplier

    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.point_value is None:
            self.point_value = self.multiplier

        # Validate tick_value calculation
        expected_tick_value = self.tick_size * self.multiplier
        if abs(self.tick_value - expected_tick_value) > 0.01:
            raise ValueError(
                f"Tick value {self.tick_value} doesn't match "
                f"tick_size ({self.tick_size}) * multiplier ({self.multiplier}) = {expected_tick_value}"
            )

    @property
    def is_cash_settled(self) -> bool:
        """Check if contract is cash-settled."""
        return self.settlement_type == SettlementType.CASH

    @property
    def is_physical_settled(self) -> bool:
        """Check if contract is physically settled."""
        return self.settlement_type == SettlementType.PHYSICAL

    def convert_price(self, price: float, from_unit: str, to_unit: str = "dollars") -> float:
        """
        Convert price between different units.

        Args:
            price: Price value to convert
            from_unit: Source unit ("cents", "dollars", "index_points")
            to_unit: Target unit (default: "dollars")

        Returns:
            Converted price

        Example:
            >>> cl_spec = ContractSpec(..., price_quote_unit="cents", multiplier=1000)
            >>> cl_spec.convert_price(6522, from_unit="cents", to_unit="dollars")
            65.22
        """
        # Normalize to dollars first
        if from_unit == "cents":
            price_dollars = price / 100.0
        elif from_unit == "index_points":
            price_dollars = price  # Index points are the base unit
        elif from_unit == "dollars":
            price_dollars = price
        else:
            raise ValueError(f"Unknown unit: {from_unit}")

        # Convert to target unit
        if to_unit == "cents":
            return price_dollars * 100.0
        elif to_unit == "dollars" or to_unit == "index_points":
            return price_dollars
        else:
            raise ValueError(f"Unknown unit: {to_unit}")

    def calculate_contract_value(self, price: float) -> float:
        """
        Calculate total contract value (notional).

        Args:
            price: Contract price

        Returns:
            Total contract value in dollars

        Example:
            >>> es_spec = ContractSpec(..., multiplier=50.0)
            >>> es_spec.calculate_contract_value(4200.0)  # ES at 4200
            210000.0  # $210,000 notional
        """
        # Ensure price is in the right unit
        if self.price_quote_unit == "cents":
            price = self.convert_price(price, from_unit="cents", to_unit="dollars")

        return price * self.multiplier

    def calculate_tick_pnl(self, ticks: float) -> float:
        """
        Calculate P&L for a given number of ticks.

        Args:
            ticks: Number of ticks of movement

        Returns:
            P&L in dollars

        Example:
            >>> es_spec = ContractSpec(..., tick_value=12.50)
            >>> es_spec.calculate_tick_pnl(4)  # 4 ticks = 1 point
            50.0  # $50 per point (4 * $12.50)
        """
        return ticks * self.tick_value


# Predefined contract specifications for major futures
MAJOR_CONTRACTS = {
    "ES": ContractSpec(
        ticker="ES",
        name="E-mini S&P 500",
        exchange=ExchangeInfo.CME,
        asset_class=AssetClass.EQUITY_INDEX,
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.50,
        price_quote_unit="index_points",
        settlement_type=SettlementType.CASH,
        contract_months="HMUZ",  # Mar, Jun, Sep, Dec
        last_trading_offset=3,  # 3rd Friday
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "CL": ContractSpec(
        ticker="CL",
        name="WTI Crude Oil",
        exchange=ExchangeInfo.NYMEX,
        asset_class=AssetClass.ENERGY,
        multiplier=1000.0,  # 1000 barrels
        tick_size=0.01,
        tick_value=10.0,
        price_quote_unit="dollars",  # Note: Quandl might have it in cents!
        settlement_type=SettlementType.PHYSICAL,
        contract_months="FGHJKMNQUVXZ",  # All 12 months
        first_notice_days=25,  # Approximate
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "GC": ContractSpec(
        ticker="GC",
        name="Gold",
        exchange=ExchangeInfo.COMEX,
        asset_class=AssetClass.METALS,
        multiplier=100.0,  # 100 troy ounces
        tick_size=0.10,
        tick_value=10.0,
        price_quote_unit="dollars",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="GJMQVZ",  # Feb, Apr, Jun, Aug, Oct, Dec
        first_notice_days=28,
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "SI": ContractSpec(
        ticker="SI",
        name="Silver",
        exchange=ExchangeInfo.COMEX,
        asset_class=AssetClass.METALS,
        multiplier=5000.0,  # 5000 troy ounces
        tick_size=0.005,
        tick_value=25.0,
        price_quote_unit="dollars",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="HKNUZ",  # Mar, May, Jul, Sep, Dec
        first_notice_days=28,
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "NG": ContractSpec(
        ticker="NG",
        name="Natural Gas",
        exchange=ExchangeInfo.NYMEX,
        asset_class=AssetClass.ENERGY,
        multiplier=10000.0,  # 10,000 MMBtu
        tick_size=0.001,
        tick_value=10.0,
        price_quote_unit="dollars",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="FGHJKMNQUVXZ",  # All 12 months
        first_notice_days=25,
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "HG": ContractSpec(
        ticker="HG",
        name="Copper",
        exchange=ExchangeInfo.COMEX,
        asset_class=AssetClass.METALS,
        multiplier=25000.0,  # 25,000 pounds
        tick_size=0.0005,
        tick_value=12.50,
        price_quote_unit="dollars",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="HKNUZ",  # Mar, May, Jul, Sep, Dec
        first_notice_days=28,
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "C": ContractSpec(
        ticker="C",
        name="Corn",
        exchange=ExchangeInfo.CBOT,
        asset_class=AssetClass.AGRICULTURE,
        multiplier=50.0,  # 5000 bushels * $0.01/bushel
        tick_size=0.25,  # 1/4 cent
        tick_value=12.50,
        price_quote_unit="cents",  # Cents per bushel
        settlement_type=SettlementType.PHYSICAL,
        contract_months="HKNUZ",  # Mar, May, Jul, Sep, Dec
        first_notice_days=28,
        trading_hours="Day session",
    ),
    "W": ContractSpec(
        ticker="W",
        name="Wheat",
        exchange=ExchangeInfo.CBOT,
        asset_class=AssetClass.AGRICULTURE,
        multiplier=50.0,  # 5000 bushels
        tick_size=0.25,  # 1/4 cent
        tick_value=12.50,
        price_quote_unit="cents",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="HKNUZ",  # Mar, May, Jul, Sep, Dec
        first_notice_days=28,
        trading_hours="Day session",
    ),
    "S": ContractSpec(
        ticker="S",
        name="Soybeans",
        exchange=ExchangeInfo.CBOT,
        asset_class=AssetClass.AGRICULTURE,
        multiplier=50.0,  # 5000 bushels
        tick_size=0.25,  # 1/4 cent
        tick_value=12.50,
        price_quote_unit="cents",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="FHKNQUX",  # Jan, Mar, May, Jul, Aug, Sep, Nov
        first_notice_days=28,
        trading_hours="Day session",
    ),
    "EC": ContractSpec(
        ticker="EC",
        name="Euro FX",
        exchange=ExchangeInfo.CME,
        asset_class=AssetClass.CURRENCY,
        multiplier=125000.0,  # 125,000 euros
        tick_size=0.00005,  # Half pip
        tick_value=6.25,
        price_quote_unit="dollars",  # USD per EUR
        settlement_type=SettlementType.CASH,
        contract_months="HMUZ",  # Mar, Jun, Sep, Dec
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "JY": ContractSpec(
        ticker="JY",
        name="Japanese Yen",
        exchange=ExchangeInfo.CME,
        asset_class=AssetClass.CURRENCY,
        multiplier=12500000.0,  # 12,500,000 yen
        tick_size=0.0000005,
        tick_value=6.25,
        price_quote_unit="dollars",  # USD per JPY
        settlement_type=SettlementType.CASH,
        contract_months="HMUZ",  # Mar, Jun, Sep, Dec
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
    "ZN": ContractSpec(
        ticker="ZN",
        name="10-Year T-Note",
        exchange=ExchangeInfo.CBOT,
        asset_class=AssetClass.FIXED_INCOME,
        multiplier=1000.0,  # $100,000 face value, quoted in 32nds
        tick_size=0.015625,  # 1/64 of a point (half of 1/32)
        tick_value=15.625,
        price_quote_unit="index_points",  # Points and 32nds
        settlement_type=SettlementType.PHYSICAL,
        contract_months="HMUZ",  # Mar, Jun, Sep, Dec
        trading_hours="Nearly 24 hours (Sunday-Friday)",
    ),
}


def get_contract_spec(ticker: str) -> ContractSpec | None:
    """
    Get contract specification for a ticker.

    Args:
        ticker: Contract ticker symbol

    Returns:
        ContractSpec if known, None otherwise

    Example:
        >>> spec = get_contract_spec("ES")
        >>> spec.multiplier
        50.0
    """
    return MAJOR_CONTRACTS.get(ticker)

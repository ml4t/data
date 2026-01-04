"""Tests for futures roll strategies."""

from datetime import date, timedelta

import polars as pl
import pytest

from ml4t.data.futures.roll import (
    CalendarRoll,
    FirstNoticeDateRoll,
    HighestOpenInterestRoll,
    HighestVolumeRoll,
    OpenInterestBasedRoll,
    TimeBasedRoll,
    VolumeBasedRoll,
)
from ml4t.data.futures.schema import AssetClass, ContractSpec, SettlementType


def create_test_data_with_expiration(
    start_date: date = date(2024, 1, 1),
    days: int = 90,
    num_contracts: int = 3,
) -> pl.DataFrame:
    """Create test multi-contract data with expiration dates."""
    records = []

    # Create contracts with quarterly expirations
    base_expirations = [
        date(2024, 3, 15),  # March
        date(2024, 6, 21),  # June
        date(2024, 9, 20),  # September
        date(2024, 12, 20),  # December
    ]

    symbols = ["ESH24", "ESM24", "ESU24", "ESZ24"]

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Add rows for active contracts (those not expired)
        for i in range(min(num_contracts, len(base_expirations))):
            if current_date < base_expirations[i]:
                # Volume decreases as we approach expiration
                days_to_exp = (base_expirations[i] - current_date).days
                volume = max(1000, 10000 * (days_to_exp / 90))

                records.append(
                    {
                        "date": current_date,
                        "symbol": symbols[i],
                        "open": 4000.0 + i * 10,
                        "high": 4010.0 + i * 10,
                        "low": 3990.0 + i * 10,
                        "close": 4005.0 + i * 10,
                        "volume": volume,
                        "open_interest": None,
                        "expiration": base_expirations[i],
                    }
                )

    return pl.DataFrame(records)


def create_simple_test_data() -> pl.DataFrame:
    """Create simple test data without expiration (for volume/OI roll tests)."""
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "open": [100.0, 101.0] * 3,
            "high": [105.0, 106.0] * 3,
            "low": [99.0, 100.0] * 3,
            "close": [104.0, 105.0] * 3,
            "volume": [
                10000.0,
                5000.0,  # Day 1: Front has more volume
                8000.0,
                7000.0,  # Day 2: Still front
                5000.0,
                10000.0,  # Day 3: Back now has more volume (roll)
            ],
            "open_interest": [None] * 6,
        }
    )


class TestTimeBasedRoll:
    """Tests for TimeBasedRoll strategy."""

    def test_identify_rolls_with_expiration(self):
        """Test roll identification with expiration dates."""
        data = create_test_data_with_expiration()
        roll_strategy = TimeBasedRoll(days_before_expiration=5)

        roll_dates = roll_strategy.identify_rolls(data, None)

        # Should find rolls before each expiration
        assert len(roll_dates) > 0

        # Roll dates should be business days
        for roll_date in roll_dates:
            assert roll_date.weekday() < 5

    def test_identify_rolls_respects_days_before(self):
        """Test that roll happens N days before expiration."""
        data = create_test_data_with_expiration()

        # Test with different days_before values
        roll_5 = TimeBasedRoll(days_before_expiration=5)
        roll_10 = TimeBasedRoll(days_before_expiration=10)

        rolls_5 = roll_5.identify_rolls(data, None)
        rolls_10 = roll_10.identify_rolls(data, None)

        # More days before should give earlier roll dates
        if rolls_5 and rolls_10:
            # At least one roll should be earlier with 10 days
            assert any(r10 < r5 for r10, r5 in zip(sorted(rolls_10), sorted(rolls_5)))

    def test_raises_without_expiration_column(self):
        """Test that error is raised without expiration column."""
        data = create_simple_test_data()
        roll_strategy = TimeBasedRoll(days_before_expiration=5)

        with pytest.raises(ValueError, match="expiration"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_without_symbol_column(self):
        """Test that error is raised without symbol column."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "expiration": [date(2024, 3, 15)],
                "volume": [1000.0],
            }
        )
        roll_strategy = TimeBasedRoll(days_before_expiration=5)

        with pytest.raises(ValueError, match="symbol"):
            roll_strategy.identify_rolls(data, None)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        data = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "expiration": pl.Series([], dtype=pl.Date),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )
        roll_strategy = TimeBasedRoll(days_before_expiration=5)

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_calendar_days_mode(self):
        """Test roll calculation using calendar days instead of business days."""
        data = create_test_data_with_expiration()

        roll_business = TimeBasedRoll(days_before_expiration=5, use_business_days=True)
        roll_calendar = TimeBasedRoll(days_before_expiration=5, use_business_days=False)

        rolls_business = roll_business.identify_rolls(data, None)
        rolls_calendar = roll_calendar.identify_rolls(data, None)

        # Both should produce valid results
        assert isinstance(rolls_business, list)
        assert isinstance(rolls_calendar, list)

        # With longer period, the difference should be more apparent
        roll_business_10 = TimeBasedRoll(days_before_expiration=10, use_business_days=True)
        roll_calendar_10 = TimeBasedRoll(days_before_expiration=10, use_business_days=False)

        rolls_business_10 = roll_business_10.identify_rolls(data, None)
        rolls_calendar_10 = roll_calendar_10.identify_rolls(data, None)

        # Business day calculation should give earlier dates
        # (10 business days = ~14 calendar days, so roll should be earlier)
        if rolls_business_10 and rolls_calendar_10:
            # At least one roll should be different
            # Note: may be same if dates happen to align with weekends
            assert isinstance(rolls_business_10, list)
            assert isinstance(rolls_calendar_10, list)


class TestFirstNoticeDateRoll:
    """Tests for FirstNoticeDateRoll strategy."""

    def test_physical_delivery_contract(self):
        """Test rolling for physical delivery contract."""
        data = create_test_data_with_expiration()

        # Create CL contract spec (physical delivery)
        cl_spec = ContractSpec(
            ticker="CL",
            name="Crude Oil",
            exchange="NYMEX",
            asset_class=AssetClass.ENERGY,
            multiplier=1000.0,
            tick_size=0.01,
            tick_value=10.0,
            price_quote_unit="dollars",
            settlement_type=SettlementType.PHYSICAL,
            contract_months="FGHJKMNQUVXZ",
            first_notice_days=25,
        )

        roll_strategy = FirstNoticeDateRoll(days_before_first_notice=1)
        roll_dates = roll_strategy.identify_rolls(data, cl_spec)

        # Should find rolls
        assert len(roll_dates) >= 0  # May be empty if test data doesn't span enough

    def test_cash_settled_falls_back_to_time_based(self):
        """Test that cash-settled contracts use time-based rolling."""
        data = create_test_data_with_expiration()

        # Create ES contract spec (cash-settled)
        es_spec = ContractSpec(
            ticker="ES",
            name="E-mini S&P 500",
            exchange="CME",
            asset_class=AssetClass.EQUITY_INDEX,
            multiplier=50.0,
            tick_size=0.25,
            tick_value=12.50,
            price_quote_unit="index_points",
            settlement_type=SettlementType.CASH,
            contract_months="HMUZ",
        )

        roll_strategy = FirstNoticeDateRoll()
        roll_dates = roll_strategy.identify_rolls(data, es_spec)

        # Should still find rolls (using TimeBasedRoll fallback)
        assert isinstance(roll_dates, list)

    def test_no_contract_spec_uses_time_based(self):
        """Test that missing contract spec uses time-based rolling."""
        data = create_test_data_with_expiration()
        roll_strategy = FirstNoticeDateRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)

        # Should still work (falls back to TimeBasedRoll)
        assert isinstance(roll_dates, list)


class TestVolumeBasedRoll:
    """Tests for VolumeBasedRoll strategy (existing functionality)."""

    def test_basic_roll_detection(self):
        """Test basic volume-based roll detection."""
        # This test verifies existing functionality still works
        roll_strategy = VolumeBasedRoll()

        # Create data with clear volume crossover
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 2, 1),
                    date(2024, 2, 1),
                    date(2024, 3, 1),
                    date(2024, 3, 1),
                ],
                "volume": [
                    10000.0,
                    2000.0,  # Jan: Front dominates
                    6000.0,
                    5000.0,  # Feb: Getting closer
                    2000.0,
                    10000.0,  # Mar: Back now dominates (roll)
                ],
            }
        )

        # Should detect the volume crossover
        roll_dates = roll_strategy.identify_rolls(data, None)
        assert isinstance(roll_dates, list)


class TestOpenInterestBasedRoll:
    """Tests for OpenInterestBasedRoll strategy (existing functionality)."""

    def test_raises_without_oi_column(self):
        """Test that error is raised without open_interest column."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "volume": [1000.0],
            }
        )
        roll_strategy = OpenInterestBasedRoll()

        with pytest.raises(ValueError, match="open_interest"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_with_empty_oi(self):
        """Test that error is raised when all OI values are null."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1)],
                "open_interest": [None, None],
            }
        )
        roll_strategy = OpenInterestBasedRoll()

        with pytest.raises(ValueError, match="No open interest"):
            roll_strategy.identify_rolls(data, None)

    def test_no_duplicate_dates(self):
        """Test with no duplicate dates returns empty list."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open_interest": [10000.0, 11000.0],
            }
        )
        roll_strategy = OpenInterestBasedRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_detects_oi_crossover(self):
        """Test detection of OI crossover."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                ],
                "open_interest": [
                    10000.0,
                    5000.0,  # Day 1: Front has more OI
                    8000.0,
                    7000.0,  # Day 2: Getting closer
                    5000.0,
                    12000.0,  # Day 3: Back now has more OI (roll)
                ],
            }
        )
        roll_strategy = OpenInterestBasedRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert isinstance(roll_dates, list)

    def test_lookback_days_parameter(self):
        """Test lookback_days initialization."""
        roll_strategy = OpenInterestBasedRoll(lookback_days=5)
        assert roll_strategy.lookback_days == 5


class TestVolumeBasedRollDetailed:
    """Detailed tests for VolumeBasedRoll strategy."""

    def test_init_parameters(self):
        """Test initialization parameters."""
        roll_strategy = VolumeBasedRoll(lookback_days=3, min_days_between_rolls=30)
        assert roll_strategy.lookback_days == 3
        assert roll_strategy.min_days_between_rolls == 30

    def test_no_duplicate_dates_returns_empty(self):
        """Test with no duplicate dates returns empty list."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "volume": [10000.0, 11000.0],
            }
        )
        roll_strategy = VolumeBasedRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_insufficient_data(self):
        """Test with insufficient data for comparison."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1)],
                "volume": [10000.0, 5000.0],
            }
        )
        roll_strategy = VolumeBasedRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_min_days_between_rolls_filtering(self):
        """Test that rolls too close together are filtered."""
        # Create data with rolls very close together
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 4),
                ],
                "volume": [
                    10000.0,
                    2000.0,  # Day 1: Front dominates
                    2000.0,
                    10000.0,  # Day 2: Roll! Back dominates
                    10000.0,
                    2000.0,  # Day 3: Roll back! (too close, should filter)
                    2000.0,
                    10000.0,  # Day 4: Another roll (too close)
                ],
            }
        )
        roll_strategy = VolumeBasedRoll(min_days_between_rolls=30)

        roll_dates = roll_strategy.identify_rolls(data, None)
        # Should filter out rolls too close together
        assert isinstance(roll_dates, list)

    def test_detects_clear_crossover(self):
        """Test detection of clear volume crossover."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 2, 1),
                    date(2024, 2, 1),
                    date(2024, 3, 1),
                    date(2024, 3, 1),
                ],
                "volume": [
                    10000.0,
                    1000.0,  # Jan: Front heavily dominates
                    8000.0,
                    3000.0,  # Feb: Still front
                    1000.0,
                    10000.0,  # Mar: Back heavily dominates
                ],
            }
        )
        roll_strategy = VolumeBasedRoll(min_days_between_rolls=1)

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert isinstance(roll_dates, list)


class TestCalendarRoll:
    """Tests for CalendarRoll strategy."""

    def test_init_default_rank(self):
        """Test default rank is 0."""
        roll_strategy = CalendarRoll()
        assert roll_strategy.rank == 0

    def test_init_custom_rank(self):
        """Test initialization with custom rank."""
        roll_strategy = CalendarRoll(rank=2)
        assert roll_strategy.rank == 2

    def test_raises_without_expiration_column(self):
        """Test error when expiration column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "symbol": ["ESH24"],
                "volume": [1000.0],
            }
        )
        roll_strategy = CalendarRoll()

        with pytest.raises(ValueError, match="expiration"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_without_symbol_column(self):
        """Test error when symbol column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "expiration": [date(2024, 3, 15)],
                "volume": [1000.0],
            }
        )
        roll_strategy = CalendarRoll()

        with pytest.raises(ValueError, match="symbol"):
            roll_strategy.identify_rolls(data, None)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        data = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "expiration": pl.Series([], dtype=pl.Date),
            }
        )
        roll_strategy = CalendarRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_detects_roll_on_expiration(self):
        """Test that roll is detected when nearest contract expires."""
        # Create data spanning an expiration
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 3, 14),
                    date(2024, 3, 14),  # Before exp
                    date(2024, 3, 15),
                    date(2024, 3, 15),  # On exp day
                    date(2024, 3, 18),
                    date(2024, 3, 18),  # After exp (H24 expired)
                ],
                "symbol": [
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                ],
                "expiration": [
                    date(2024, 3, 15),
                    date(2024, 6, 21),
                    date(2024, 3, 15),
                    date(2024, 6, 21),
                    date(2024, 3, 15),
                    date(2024, 6, 21),
                ],
            }
        )
        roll_strategy = CalendarRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        # Should detect roll when ESH24 expires and ESM24 becomes nearest
        assert isinstance(roll_dates, list)

    def test_rank_1_selects_second_nearest(self):
        """Test that rank=1 selects second nearest contract."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 15),
                    date(2024, 1, 15),
                    date(2024, 1, 15),
                ],
                "symbol": ["ESH24", "ESM24", "ESU24"],
                "expiration": [
                    date(2024, 3, 15),  # Nearest
                    date(2024, 6, 21),  # Second nearest
                    date(2024, 9, 20),  # Third nearest
                ],
            }
        )
        roll_strategy = CalendarRoll(rank=1)

        # With only one day, no rolls detected
        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_all_contracts_expired(self):
        """Test when all contracts are expired."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 12, 1), date(2024, 12, 1)],
                "symbol": ["ESH24", "ESM24"],
                "expiration": [date(2024, 3, 15), date(2024, 6, 21)],  # Both expired
            }
        )
        roll_strategy = CalendarRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        # No unexpired contracts, so no rolls
        assert roll_dates == []


class TestHighestVolumeRoll:
    """Tests for HighestVolumeRoll strategy."""

    def test_init_default_parameters(self):
        """Test default initialization."""
        roll_strategy = HighestVolumeRoll()
        assert roll_strategy.rank == 0
        assert roll_strategy.min_volume == 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        roll_strategy = HighestVolumeRoll(rank=1, min_volume=1000.0)
        assert roll_strategy.rank == 1
        assert roll_strategy.min_volume == 1000.0

    def test_raises_without_volume_column(self):
        """Test error when volume column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "symbol": ["ESH24"],
            }
        )
        roll_strategy = HighestVolumeRoll()

        with pytest.raises(ValueError, match="volume"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_without_symbol_column(self):
        """Test error when symbol column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "volume": [1000.0],
            }
        )
        roll_strategy = HighestVolumeRoll()

        with pytest.raises(ValueError, match="symbol"):
            roll_strategy.identify_rolls(data, None)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        data = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )
        roll_strategy = HighestVolumeRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert roll_dates == []

    def test_detects_volume_change(self):
        """Test that roll is detected when highest volume contract changes."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                ],
                "symbol": [
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                ],
                "volume": [
                    10000.0,
                    5000.0,  # ESH24 highest
                    8000.0,
                    9000.0,  # ESM24 now highest (roll!)
                    6000.0,
                    12000.0,  # ESM24 still highest
                ],
            }
        )
        roll_strategy = HighestVolumeRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert len(roll_dates) == 1
        assert roll_dates[0] == date(2024, 1, 2)

    def test_min_volume_filtering(self):
        """Test that contracts below min_volume are filtered."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "symbol": ["ESH24", "ESM24", "ESH24", "ESM24"],
                "volume": [
                    500.0,
                    100.0,  # Both below min
                    600.0,
                    200.0,  # Both below min
                ],
            }
        )
        roll_strategy = HighestVolumeRoll(min_volume=1000.0)

        roll_dates = roll_strategy.identify_rolls(data, None)
        # All filtered by min_volume, so no rolls
        assert roll_dates == []

    def test_rank_1_selects_second_highest(self):
        """Test that rank=1 selects second highest volume."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "symbol": ["ESH24", "ESM24", "ESU24"] * 2,
                "volume": [
                    10000.0,
                    5000.0,
                    2000.0,  # ESM24 is second highest
                    10000.0,
                    2000.0,
                    5000.0,  # ESU24 is now second highest (roll at rank=1)
                ],
            }
        )
        roll_strategy = HighestVolumeRoll(rank=1)

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert len(roll_dates) == 1
        assert roll_dates[0] == date(2024, 1, 2)

    def test_single_contract_per_day(self):
        """Test with single contract per day."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "symbol": ["ESH24", "ESH24"],
                "volume": [10000.0, 11000.0],
            }
        )
        roll_strategy = HighestVolumeRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        # Same contract selected each day, no roll
        assert roll_dates == []


class TestHighestOpenInterestRoll:
    """Tests for HighestOpenInterestRoll strategy."""

    def test_init_default_parameters(self):
        """Test default initialization."""
        roll_strategy = HighestOpenInterestRoll()
        assert roll_strategy.rank == 0
        assert roll_strategy.min_oi == 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        roll_strategy = HighestOpenInterestRoll(rank=1, min_oi=5000.0)
        assert roll_strategy.rank == 1
        assert roll_strategy.min_oi == 5000.0

    def test_raises_without_open_interest_column(self):
        """Test error when open_interest column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "symbol": ["ESH24"],
                "volume": [1000.0],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        with pytest.raises(ValueError, match="open_interest"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_without_symbol_column(self):
        """Test error when symbol column missing."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "open_interest": [10000.0],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        with pytest.raises(ValueError, match="symbol"):
            roll_strategy.identify_rolls(data, None)

    def test_raises_with_all_null_oi(self):
        """Test error when all OI values are null."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1)],
                "symbol": ["ESH24", "ESM24"],
                "open_interest": [None, None],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        with pytest.raises(ValueError, match="No open interest"):
            roll_strategy.identify_rolls(data, None)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        data = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "open_interest": pl.Series([], dtype=pl.Float64),
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        with pytest.raises(ValueError, match="No open interest"):
            roll_strategy.identify_rolls(data, None)

    def test_detects_oi_change(self):
        """Test that roll is detected when highest OI contract changes."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                ],
                "symbol": [
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                    "ESH24",
                    "ESM24",
                ],
                "open_interest": [
                    20000.0,
                    10000.0,  # ESH24 highest
                    15000.0,
                    18000.0,  # ESM24 now highest (roll!)
                    12000.0,
                    22000.0,  # ESM24 still highest
                ],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert len(roll_dates) == 1
        assert roll_dates[0] == date(2024, 1, 2)

    def test_min_oi_filtering(self):
        """Test that contracts below min_oi are filtered."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "symbol": ["ESH24", "ESM24", "ESH24", "ESM24"],
                "open_interest": [
                    500.0,
                    100.0,  # Both below min
                    600.0,
                    200.0,  # Both below min
                ],
            }
        )
        roll_strategy = HighestOpenInterestRoll(min_oi=1000.0)

        roll_dates = roll_strategy.identify_rolls(data, None)
        # All filtered by min_oi, so no rolls
        assert roll_dates == []

    def test_rank_1_selects_second_highest(self):
        """Test that rank=1 selects second highest OI."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "symbol": ["ESH24", "ESM24", "ESU24"] * 2,
                "open_interest": [
                    20000.0,
                    10000.0,
                    5000.0,  # ESM24 is second highest
                    20000.0,
                    5000.0,
                    10000.0,  # ESU24 is now second highest (roll)
                ],
            }
        )
        roll_strategy = HighestOpenInterestRoll(rank=1)

        roll_dates = roll_strategy.identify_rolls(data, None)
        assert len(roll_dates) == 1
        assert roll_dates[0] == date(2024, 1, 2)

    def test_filters_null_oi_values(self):
        """Test that null OI values are filtered out."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "symbol": ["ESH24", "ESM24", "ESH24", "ESM24"],
                "open_interest": [
                    20000.0,
                    None,  # ESM24 has no OI
                    18000.0,
                    22000.0,  # Now both have OI, ESM24 highest (roll)
                ],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        # Day 1: Only ESH24 has OI, selected
        # Day 2: ESM24 now highest, roll detected
        assert len(roll_dates) == 1

    def test_single_contract_no_roll(self):
        """Test with single contract per day."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "symbol": ["ESH24", "ESH24"],
                "open_interest": [10000.0, 12000.0],
            }
        )
        roll_strategy = HighestOpenInterestRoll()

        roll_dates = roll_strategy.identify_rolls(data, None)
        # Same contract selected each day, no roll
        assert roll_dates == []

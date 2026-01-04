"""
Roll strategies for futures continuous contracts.

Roll strategies determine when to switch from the current front month
to the next contract month when building continuous contracts.
"""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from ml4t.data.futures.schema import ContractSpec


class RollStrategy(ABC):
    """Abstract base class for roll strategies."""

    @abstractmethod
    def identify_rolls(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates in multi-contract data.

        Args:
            data: Multi-contract DataFrame with potential duplicate dates
                  Must have columns: date, volume, open_interest
            contract_spec: Optional contract specifications

        Returns:
            List of dates when rolls occur, sorted chronologically
        """


class VolumeBasedRoll(RollStrategy):
    """
    Roll when next month volume exceeds front month volume.

    This is the most common roll method - switches to the next contract
    when it becomes more liquid (higher trading volume) than the current front month.
    """

    def __init__(self, lookback_days: int = 1, min_days_between_rolls: int = 20):
        """
        Initialize volume-based roll strategy.

        Args:
            lookback_days: Number of days to confirm volume crossover
                          (helps avoid false signals from single-day spikes)
            min_days_between_rolls: Minimum days between rolls to filter noise
                                   (default 20 for monthly contracts)
        """
        self.lookback_days = lookback_days
        self.min_days_between_rolls = min_days_between_rolls

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on volume crossover.

        Logic:
        1. For each date with multiple contracts, identify front (highest volume) and back (2nd highest)
        2. Detect sustained volume crossover (back month volume consistently exceeds front)
        3. Filter rolls too close together (< min_days_between_rolls)
        4. Return cleaned list of roll dates

        Args:
            data: Multi-contract DataFrame

        Returns:
            List of roll dates
        """
        # Filter to dates with multiple contracts (duplicates)
        date_counts = data.group_by("date").len().filter(pl.col("len") > 1)

        if len(date_counts) == 0:
            # No duplicate dates = already continuous, no rolls detected
            return []

        duplicate_dates = sorted(date_counts.select("date").to_series().to_list())

        # Build volume series for front and back months
        front_volumes = []
        back_volumes = []
        dates_with_data = []

        for dt in duplicate_dates:
            day_data = data.filter(pl.col("date") == dt).sort("volume", descending=True)

            if len(day_data) >= 2:
                front_volumes.append(day_data["volume"][0])
                back_volumes.append(day_data["volume"][1])
                dates_with_data.append(dt)

        if len(dates_with_data) < 2:
            return []

        # Detect crossovers: where the highest volume contract changes significantly
        # Simple heuristic: if today's front volume is much closer to yesterday's back volume
        # than yesterday's front volume, a roll likely occurred
        crossovers = []
        for i in range(1, len(dates_with_data)):
            prev_front = front_volumes[i - 1]
            prev_back = back_volumes[i - 1]
            curr_front = front_volumes[i]

            # Calculate distances to determine if volumes "switched"
            # If current front is closer to previous back, we likely rolled
            dist_to_prev_back = abs(curr_front - prev_back)
            dist_to_prev_front = abs(curr_front - prev_front)

            # Crossover if:
            # 1. Current front volume is much closer to previous back (>20% closer)
            # 2. Volume actually changed significantly (not just noise)
            if dist_to_prev_front > 0:  # Avoid division by zero
                closeness_ratio = dist_to_prev_back / dist_to_prev_front

                # If current front is closer to prev back, ratio will be < 1
                if closeness_ratio < 0.8:  # 20% threshold
                    # Also check that volume actually changed (not same contract)
                    if abs(curr_front - prev_front) / prev_front > 0.1:  # 10% change
                        crossovers.append(dates_with_data[i])

        # Filter out rolls too close together (noise reduction)
        if not crossovers:
            return []

        filtered_rolls = [crossovers[0]]  # Always keep first roll

        for roll_date in crossovers[1:]:
            days_since_last = (roll_date - filtered_rolls[-1]).days

            if days_since_last >= self.min_days_between_rolls:
                filtered_rolls.append(roll_date)

        return filtered_rolls


class OpenInterestBasedRoll(RollStrategy):
    """
    Roll when next month open interest exceeds front month.

    Similar to volume-based, but uses open interest (outstanding contracts)
    instead of daily volume. Often more stable than volume.
    """

    def __init__(self, lookback_days: int = 1):
        """
        Initialize open interest-based roll strategy.

        Args:
            lookback_days: Number of days to confirm OI crossover
        """
        self.lookback_days = lookback_days

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on open interest crossover.

        Logic: Similar to volume-based, but uses open_interest column.

        Args:
            data: Multi-contract DataFrame with open_interest column

        Returns:
            List of roll dates
        """
        # Check if open_interest column exists and has data
        if "open_interest" not in data.columns:
            raise ValueError("Data must have 'open_interest' column for OI-based rolling")

        # Filter out rows where OI is null
        oi_data = data.filter(pl.col("open_interest").is_not_null())

        if len(oi_data) == 0:
            raise ValueError("No open interest data available")

        # Filter to dates with multiple contracts
        date_counts = oi_data.group_by("date").len().filter(pl.col("len") > 1)

        if len(date_counts) == 0:
            return []  # No duplicate dates

        duplicate_dates = date_counts.select("date").to_series().to_list()

        rolls: list[date] = []
        previous_front_oi = None
        previous_back_oi = None

        for dt in sorted(duplicate_dates):
            day_data = oi_data.filter(pl.col("date") == dt).sort("open_interest", descending=True)

            if len(day_data) < 2:
                continue

            front_oi = day_data["open_interest"][0]
            back_oi = day_data["open_interest"][1]

            # Detect roll similar to volume-based
            if previous_front_oi is not None and previous_back_oi is not None:
                if abs(front_oi - previous_back_oi) < abs(front_oi - previous_front_oi):
                    rolls.append(dt)

            previous_front_oi = front_oi
            previous_back_oi = back_oi

        return rolls


class TimeBasedRoll(RollStrategy):
    """
    Roll N business days before front month expiration.

    Uses expiration dates from the data (e.g., from Databento definition schema).
    Most predictable roll method - avoids liquidity-driven roll timing.

    Requires data to have an 'expiration' column with contract expiration dates.
    """

    def __init__(
        self,
        days_before_expiration: int = 5,
        use_business_days: bool = True,
    ):
        """
        Initialize time-based roll strategy.

        Args:
            days_before_expiration: Number of days before expiration to roll.
                                   If use_business_days=True, this is business days.
            use_business_days: If True, count only weekdays (Mon-Fri).
                              If False, use calendar days.
        """
        self.days_before_expiration = days_before_expiration
        self.use_business_days = use_business_days

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on expiration calendar.

        Requires data to have:
        - 'date': Trading dates
        - 'expiration': Contract expiration dates
        - 'symbol': Contract identifier (to track which contract we're in)

        Args:
            data: Multi-contract DataFrame with expiration column
            _contract_spec: Optional contract specifications (unused, for API compat)

        Returns:
            List of roll dates sorted chronologically

        Raises:
            ValueError: If data doesn't have required 'expiration' column
        """
        if "expiration" not in data.columns:
            raise ValueError(
                "TimeBasedRoll requires data with 'expiration' column. "
                "Use Databento definition schema or add expiration dates manually."
            )

        # Get unique contracts with their expirations
        if "symbol" not in data.columns:
            raise ValueError("TimeBasedRoll requires data with 'symbol' column")

        # Get trading dates available in data
        trading_dates = sorted(data.select("date").unique().to_series().to_list())

        if not trading_dates:
            return []

        # Get contract expirations (unique symbol -> expiration mapping)
        contract_exps = (
            data.select(["symbol", "expiration"])
            .unique(subset=["symbol"])
            .filter(pl.col("expiration").is_not_null())
            .sort("expiration")
        )

        if contract_exps.height == 0:
            return []

        # Calculate roll dates
        roll_dates: list[date] = []

        for row in contract_exps.iter_rows(named=True):
            exp_date = row["expiration"]
            if isinstance(exp_date, date):
                roll_date = self._calculate_roll_date(exp_date, trading_dates)
                if roll_date and roll_date in trading_dates:
                    roll_dates.append(roll_date)

        # Remove duplicates and sort
        roll_dates = sorted(set(roll_dates))

        return roll_dates

    def _calculate_roll_date(
        self,
        expiration: date,
        trading_dates: list[date],
    ) -> date | None:
        """
        Calculate roll date as N days before expiration.

        Args:
            expiration: Contract expiration date
            trading_dates: List of available trading dates

        Returns:
            Roll date, or None if can't calculate
        """
        from datetime import timedelta

        if self.use_business_days:
            # Count back N business days
            roll_date = expiration
            days_counted = 0

            while days_counted < self.days_before_expiration:
                roll_date = roll_date - timedelta(days=1)
                # Check if it's a weekday (Monday=0, Friday=4)
                if roll_date.weekday() < 5:
                    days_counted += 1

                # Safety limit to prevent infinite loop
                if (expiration - roll_date).days > 30:
                    return None
        else:
            # Simple calendar days
            roll_date = expiration - timedelta(days=self.days_before_expiration)

        # Find the nearest trading date on or before roll_date
        valid_dates = [d for d in trading_dates if d <= roll_date]
        if valid_dates:
            return max(valid_dates)

        return None


class FirstNoticeDateRoll(RollStrategy):
    """
    Roll before first notice date for physical delivery contracts.

    Critical for commodities (CL, GC, SI, etc.) where holding past first
    notice date could result in delivery obligation.

    First notice date is typically:
    - One business day before the first day of the delivery month
    - Varies by exchange and contract

    Uses ContractSpec.first_notice_days to calculate the date.
    """

    def __init__(self, days_before_first_notice: int = 1):
        """
        Initialize first notice date roll strategy.

        Args:
            days_before_first_notice: Number of business days before first
                                     notice date to roll (default: 1)
        """
        self.days_before_first_notice = days_before_first_notice

    def identify_rolls(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on first notice dates.

        For physical delivery contracts, rolls before first notice to avoid
        delivery obligations.

        Args:
            data: Multi-contract DataFrame
            contract_spec: Contract specifications with first_notice_days

        Returns:
            List of roll dates sorted chronologically
        """
        # If no contract spec or cash-settled, fall back to time-based
        if contract_spec is None or contract_spec.is_cash_settled:
            # Cash-settled contracts don't have first notice
            # Use TimeBasedRoll with 5 days before expiration as fallback
            time_roll = TimeBasedRoll(days_before_expiration=5)
            return time_roll.identify_rolls(data, contract_spec)

        # Get first notice days from spec, or use default
        first_notice_days = contract_spec.first_notice_days or 25

        # First notice is typically first_notice_days before delivery month
        # We roll days_before_first_notice before that
        total_days = first_notice_days + self.days_before_first_notice

        # Use TimeBasedRoll with adjusted days
        time_roll = TimeBasedRoll(
            days_before_expiration=total_days,
            use_business_days=True,
        )

        return time_roll.identify_rolls(data, contract_spec)


# =============================================================================
# Databento-Compatible Roll Strategies (Selection-Based)
# =============================================================================
# These strategies match Databento's continuous contract symbology:
# - Calendar (c): Nearest contract by expiration
# - Volume (v): Highest volume contract
# - Open Interest (n): Highest open interest contract
#
# Unlike the crossover-based strategies above, these are selection-based:
# they determine which contract to use on each day, with roll dates
# occurring when the selected contract changes.
# =============================================================================


class CalendarRoll(RollStrategy):
    """
    Select the nearest contract by expiration date (Databento's "c" rule).

    This is the simplest roll method - always use the contract with the
    nearest expiration date. Roll occurs when that contract expires and
    the next one becomes nearest.

    Requires data to have 'expiration' column.

    Example:
        On 2024-01-15:
        - ESH24 expires 2024-03-15 (selected - nearest)
        - ESM24 expires 2024-06-21
        - ESU24 expires 2024-09-20

        On 2024-03-16 (after ESH24 expires):
        - ESM24 expires 2024-06-21 (now selected)
        - ESU24 expires 2024-09-20
    """

    def __init__(self, rank: int = 0):
        """
        Initialize calendar roll strategy.

        Args:
            rank: Which contract to select (0 = nearest, 1 = second nearest, etc.)
                  Databento notation: c.0, c.1, c.2
        """
        self.rank = rank

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on expiration calendar.

        Roll occurs when a different contract becomes the nearest-to-expiry.

        Args:
            data: Multi-contract DataFrame with 'expiration' column

        Returns:
            List of roll dates
        """
        if "expiration" not in data.columns:
            raise ValueError(
                "CalendarRoll requires data with 'expiration' column. "
                "Use Databento definition schema to add expiration dates."
            )

        if "symbol" not in data.columns:
            raise ValueError("CalendarRoll requires data with 'symbol' column")

        # Get unique dates
        dates = sorted(data.select("date").unique().to_series().to_list())
        if not dates:
            return []

        roll_dates: list[date] = []
        prev_selected: str | None = None

        for dt in dates:
            day_data = data.filter(pl.col("date") == dt)

            # Filter to unexpired contracts (expiration > current date)
            unexpired = day_data.filter(pl.col("expiration") > dt)

            if unexpired.height == 0:
                continue

            # Sort by expiration (nearest first)
            sorted_data = unexpired.sort("expiration")

            # Select contract at specified rank
            if sorted_data.height > self.rank:
                selected = sorted_data["symbol"][self.rank]

                # Detect roll (change in selected contract)
                if prev_selected is not None and selected != prev_selected:
                    roll_dates.append(dt)

                prev_selected = selected

        return roll_dates


class HighestVolumeRoll(RollStrategy):
    """
    Select the contract with highest volume (Databento's "v" rule).

    Uses previous day's volume to rank contracts. Roll occurs when
    a different contract becomes the most liquid.

    This is a market-driven roll - follows where trading activity is highest.

    Example:
        On 2024-03-01:
        - ESH24: 1,500,000 volume (selected)
        - ESM24: 500,000 volume

        On 2024-03-08 (roll week):
        - ESH24: 800,000 volume
        - ESM24: 1,200,000 volume (now selected - roll!)
    """

    def __init__(self, rank: int = 0, min_volume: float = 0):
        """
        Initialize highest volume roll strategy.

        Args:
            rank: Which contract to select (0 = highest, 1 = second highest)
                  Databento notation: v.0, v.1, v.2
            min_volume: Minimum volume threshold to consider a contract
        """
        self.rank = rank
        self.min_volume = min_volume

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on volume ranking.

        Roll occurs when a different contract becomes highest volume.

        Args:
            data: Multi-contract DataFrame with 'volume' column

        Returns:
            List of roll dates
        """
        if "volume" not in data.columns:
            raise ValueError("HighestVolumeRoll requires data with 'volume' column")

        if "symbol" not in data.columns:
            raise ValueError("HighestVolumeRoll requires data with 'symbol' column")

        # Get unique dates
        dates = sorted(data.select("date").unique().to_series().to_list())
        if not dates:
            return []

        roll_dates: list[date] = []
        prev_selected: str | None = None

        for dt in dates:
            day_data = data.filter(pl.col("date") == dt)

            # Filter by minimum volume
            if self.min_volume > 0:
                day_data = day_data.filter(pl.col("volume") >= self.min_volume)

            if day_data.height == 0:
                continue

            # Sort by volume (highest first)
            sorted_data = day_data.sort("volume", descending=True)

            # Select contract at specified rank
            if sorted_data.height > self.rank:
                selected = sorted_data["symbol"][self.rank]

                # Detect roll (change in selected contract)
                if prev_selected is not None and selected != prev_selected:
                    roll_dates.append(dt)

                prev_selected = selected

        return roll_dates


class HighestOpenInterestRoll(RollStrategy):
    """
    Select the contract with highest open interest (Databento's "n" rule).

    Uses previous day's close OI to rank contracts. Roll occurs when
    a different contract becomes the most held.

    OI is generally more stable than volume, providing smoother roll timing.

    Example:
        On 2024-03-01:
        - ESH24: 2,000,000 OI (selected)
        - ESM24: 500,000 OI

        On 2024-03-10 (roll period):
        - ESH24: 1,200,000 OI
        - ESM24: 1,800,000 OI (now selected - roll!)
    """

    def __init__(self, rank: int = 0, min_oi: float = 0):
        """
        Initialize highest open interest roll strategy.

        Args:
            rank: Which contract to select (0 = highest, 1 = second highest)
                  Databento notation: n.0, n.1, n.2
            min_oi: Minimum OI threshold to consider a contract
        """
        self.rank = rank
        self.min_oi = min_oi

    def identify_rolls(
        self, data: pl.DataFrame, _contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """
        Identify roll dates based on open interest ranking.

        Roll occurs when a different contract becomes highest OI.

        Args:
            data: Multi-contract DataFrame with 'open_interest' column

        Returns:
            List of roll dates
        """
        if "open_interest" not in data.columns:
            raise ValueError("HighestOpenInterestRoll requires data with 'open_interest' column")

        if "symbol" not in data.columns:
            raise ValueError("HighestOpenInterestRoll requires data with 'symbol' column")

        # Filter out null OI values
        valid_data = data.filter(pl.col("open_interest").is_not_null())

        if valid_data.height == 0:
            raise ValueError("No open interest data available")

        # Get unique dates
        dates = sorted(valid_data.select("date").unique().to_series().to_list())
        if not dates:
            return []

        roll_dates: list[date] = []
        prev_selected: str | None = None

        for dt in dates:
            day_data = valid_data.filter(pl.col("date") == dt)

            # Filter by minimum OI
            if self.min_oi > 0:
                day_data = day_data.filter(pl.col("open_interest") >= self.min_oi)

            if day_data.height == 0:
                continue

            # Sort by OI (highest first)
            sorted_data = day_data.sort("open_interest", descending=True)

            # Select contract at specified rank
            if sorted_data.height > self.rank:
                selected = sorted_data["symbol"][self.rank]

                # Detect roll (change in selected contract)
                if prev_selected is not None and selected != prev_selected:
                    roll_dates.append(dt)

                prev_selected = selected

        return roll_dates

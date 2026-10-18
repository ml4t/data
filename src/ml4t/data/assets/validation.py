"""Asset-specific validation rules."""

from __future__ import annotations

from typing import Any

import polars as pl
import structlog

from ml4t.data.assets.asset_class import AssetClass, AssetInfo
from ml4t.data.assets.schemas import AssetSchema
from ml4t.data.calendar.crypto import CryptoSessionValidator
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

logger = structlog.get_logger()


class AssetValidator:
    """Validator for asset-specific data quality rules."""

    def __init__(
        self,
        asset_info: AssetInfo,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> None:
        """
        Initialize validator for specific asset.

        Args:
            asset_info: Information about the asset
        """
        self.asset_info = asset_info
        self.asset_class = asset_info.asset_class
        self._calendar = calendar
        self.feed_spec = FeedSpec.from_any(feed_spec)
        self.calendar = self.feed_spec.resolve_calendar(calendar=calendar)

        # Initialize specialized validators
        if self.asset_class == AssetClass.CRYPTO:
            self.session_validator = CryptoSessionValidator(self.calendar, feed_spec=self.feed_spec)
        else:
            self.session_validator = None

    def validate(
        self,
        df: pl.DataFrame,
        frequency: str | None = None,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> dict[str, list[str]]:
        """
        Perform comprehensive validation.

        Args:
            df: DataFrame to validate
            frequency: Data frequency

        Returns:
            Dictionary of validation category to issues
        """
        feed = self.feed_spec if feed_spec is None else FeedSpec.from_any(feed_spec)
        if calendar is UNSET_CALENDAR:
            calendar = self._calendar
        selected = feed.resolve_calendar(calendar=calendar)
        frequency = frequency or feed.data_frequency or "daily"
        results = {}

        # Schema validation
        is_valid, schema_issues = AssetSchema.validate_dataframe(df, self.asset_class)
        if schema_issues:
            results["schema"] = schema_issues

        # Asset-specific validation
        if self.asset_class == AssetClass.CRYPTO:
            results.update(self._validate_crypto(df, frequency, selected, feed))
        elif self.asset_class == AssetClass.EQUITY:
            results.update(self._validate_equity(df))
        elif self.asset_class == AssetClass.FOREX:
            results.update(self._validate_forex(df))

        # Common validations
        results.update(self._validate_common(df))
        if not df.is_empty() and "timestamp" in df.columns:
            from ml4t.data.validation.cross_validation import CrossValidator

            temporal = CrossValidator(
                check_price_continuity=False,
                check_volume_spikes=False,
                calendar=selected,
                feed_spec=feed,
            ).validate(df, frequency=frequency)
            category = {
                AssetClass.EQUITY: "equity_specific",
                AssetClass.FOREX: "forex_specific",
            }.get(self.asset_class, "sessions")
            for issue in temporal.issues:
                if issue.check == "weekend_trading":
                    message = f"Found {issue.row_count} weekend data points"
                else:
                    count = issue.row_count - issue.details["weekend_count"]
                    if count == 0 or (
                        self.asset_class == AssetClass.EQUITY and count <= len(df) * 0.1
                    ):
                        continue
                    message = f"Found {count} data points outside market hours"
                results.setdefault(category, []).append(message)

        # Remove empty categories
        return {k: v for k, v in results.items() if v}

    def _validate_crypto(
        self, df: pl.DataFrame, frequency: str, calendar: str | None, feed: FeedSpec
    ) -> dict[str, list[str]]:
        """Validate crypto-specific rules."""
        results = {}

        # 24/7 continuity check
        if self.session_validator:
            continuity_issues = self.session_validator.validate_continuity(
                df, frequency, calendar=calendar, feed_spec=feed
            )
            if continuity_issues:
                results["continuity"] = continuity_issues

            # Volume profile check
            volume_issues = self.session_validator.validate_volume_profile(df)
            if volume_issues:
                results["volume"] = volume_issues

        # Stablecoin-specific checks
        if self.asset_info.is_stablecoin:
            stablecoin_issues = self._validate_stablecoin(df)
            if stablecoin_issues:
                results["stablecoin"] = stablecoin_issues

        # Check for extreme price movements (>50% in one period)
        if "close" in df.columns and len(df) > 1:
            price_changes = df.with_columns(
                (pl.col("close").pct_change().abs() * 100).alias("pct_change")
            )
            extreme_moves = price_changes.filter(pl.col("pct_change") > 50)

            if len(extreme_moves) > 0:
                results["price_movements"] = [
                    f"Found {len(extreme_moves)} extreme price movements (>50%)"
                ]

        return results

    def _validate_equity(self, df: pl.DataFrame) -> dict[str, list[str]]:
        """Validate equity-specific rules."""
        results = {}
        issues = []

        # Check for reasonable price ranges
        if "close" in df.columns:
            prices = df["close"].to_numpy()
            if len(prices) > 0:
                if prices.min() <= 0:
                    issues.append("Found non-positive prices")
                if prices.max() > 1000000:  # $1M per share is extremely rare
                    issues.append("Found unusually high prices (>$1M)")

        if issues:
            results["equity_specific"] = issues

        return results

    def _validate_forex(self, df: pl.DataFrame) -> dict[str, list[str]]:
        """Validate forex-specific rules."""
        results = {}
        issues = []

        # Check for reasonable exchange rates
        if "close" in df.columns:
            rates = df["close"].to_numpy()
            if len(rates) > 0:
                # Most forex pairs should be between 0.0001 and 10000
                if rates.min() < 0.0001:
                    issues.append("Found unusually low exchange rates (<0.0001)")
                if rates.max() > 10000:
                    issues.append("Found unusually high exchange rates (>10000)")

        # Check spread if available
        if "bid" in df.columns and "ask" in df.columns:
            spreads = (df["ask"] - df["bid"]).to_numpy()
            if len(spreads) > 0:
                negative_spreads = spreads[spreads < 0]
                if len(negative_spreads) > 0:
                    issues.append(f"Found {len(negative_spreads)} negative spreads")

        if issues:
            results["forex_specific"] = issues

        return results

    def _validate_stablecoin(self, df: pl.DataFrame) -> list[str]:
        """Validate stablecoin-specific rules."""
        issues = []

        if "close" not in df.columns:
            return issues

        prices = df["close"].to_numpy()
        if len(prices) == 0:
            return issues

        # Stablecoins should stay close to $1 (or their peg)
        peg_value = 1.0  # USD peg
        tolerance = 0.1  # 10% tolerance

        # Check for de-pegging events
        depegged = df.filter(
            (pl.col("close") < peg_value - tolerance) | (pl.col("close") > peg_value + tolerance)
        )

        if len(depegged) > 0:
            min_price = prices.min()
            max_price = prices.max()
            issues.append(
                f"Stablecoin depegged {len(depegged)} times "
                f"(range: ${min_price:.4f} - ${max_price:.4f})"
            )

        # Check volatility (should be low for stablecoins)
        if len(prices) > 10:
            returns = df.select(pl.col("close").pct_change().abs()).to_numpy().flatten()
            high_volatility = returns[returns > 0.05]  # >5% moves

            if len(high_volatility) > len(returns) * 0.1:
                issues.append(
                    f"High volatility for stablecoin: {len(high_volatility)} periods with >5% moves"
                )

        return issues

    def _validate_common(self, df: pl.DataFrame) -> dict[str, list[str]]:
        """Validate common rules for all assets."""
        results = {}
        issues = []

        # Check for data completeness
        if df.is_empty():
            issues.append("DataFrame is empty")
            return {"data_quality": issues}

        # Check for null values in required columns
        required = AssetSchema.get_required_columns(self.asset_class)
        for col in required:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    issues.append(f"Column {col} has {null_count} null values")

        # Check OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            ohlc_issues = self._validate_ohlc_relationships(df)
            if ohlc_issues:
                issues.extend(ohlc_issues)

        # Check for duplicate timestamps
        if "timestamp" in df.columns:
            duplicates = df.group_by("timestamp").len()
            duplicates = duplicates.filter(pl.col("len") > 1)
            if len(duplicates) > 0:
                issues.append(f"Found {len(duplicates)} duplicate timestamps")

        # Check chronological order
        if "timestamp" in df.columns and len(df) > 1:
            timestamps = df["timestamp"].to_list()
            if timestamps != sorted(timestamps):
                issues.append("Timestamps are not in chronological order")

        if issues:
            results["data_quality"] = issues

        return results

    def _validate_ohlc_relationships(self, df: pl.DataFrame) -> list[str]:
        """Validate OHLC price relationships."""
        issues = []

        # High >= Low
        invalid_hl = df.filter(pl.col("high") < pl.col("low"))
        if len(invalid_hl) > 0:
            issues.append(f"Found {len(invalid_hl)} rows where high < low")

        # High >= Open, Close
        invalid_high = df.filter(
            (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        )
        if len(invalid_high) > 0:
            issues.append(f"Found {len(invalid_high)} rows where high < open or close")

        # Low <= Open, Close
        invalid_low = df.filter(
            (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
        )
        if len(invalid_low) > 0:
            issues.append(f"Found {len(invalid_low)} rows where low > open or close")

        return issues

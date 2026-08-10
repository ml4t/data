"""Anomaly detector implementations."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal

import polars as pl
import structlog

from ml4t.data.anomaly.base import Anomaly, AnomalyDetector, AnomalySeverity, AnomalyType
from ml4t.data.anomaly.config import (
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)

logger = structlog.get_logger()


def _optional_float(value: object) -> float | None:
    """Convert a Polars numeric scalar while preserving null statistics."""
    if value is None:
        return None
    if isinstance(value, int | float | Decimal):
        return float(value)
    raise TypeError(f"Expected a numeric scalar, received {type(value).__name__}")


def _float(value: object) -> float:
    result = _optional_float(value)
    if result is None:
        raise TypeError("Expected a numeric scalar, received null")
    return result


def _datetime(value: object) -> datetime:
    """Convert a Polars temporal scalar, accepting a Date column as midnight.

    A daily bar's timestamp is a ``pl.Date``, so requiring ``pl.Datetime`` here rejected
    every daily series a detector was given. ``datetime`` is a subclass of ``date``, so
    it has to be tested first.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"Expected a date or datetime scalar, received {type(value).__name__}")


class ReturnOutlierDetector(AnomalyDetector):
    """Detect return outliers using MAD or z-score methods."""

    def __init__(self, config: ReturnOutlierConfig | None = None):
        """
        Initialize return outlier detector.

        Args:
            config: Detector configuration
        """
        self.config = config or ReturnOutlierConfig()
        super().__init__(enabled=self.config.enabled)

    @property
    def name(self) -> str:
        """Return detector name."""
        return "return_outliers"

    def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """
        Detect return outliers in the data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being analyzed

        Returns:
            List of detected anomalies
        """
        if not self.enabled or len(df) < self.config.min_samples:
            return []

        anomalies = []

        # Calculate returns
        df_with_returns = df.with_columns(pl.col("close").pct_change().alias("returns"))

        # Remove first row (no return) and filter out nulls
        df_with_returns = df_with_returns.filter(pl.col("returns").is_not_null())

        if self.config.method == "mad":
            anomalies.extend(self._detect_mad(df_with_returns, symbol))
        elif self.config.method == "zscore":
            anomalies.extend(self._detect_zscore(df_with_returns, symbol))
        elif self.config.method == "iqr":
            anomalies.extend(self._detect_iqr(df_with_returns, symbol))

        return anomalies

    def _detect_mad(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """Detect outliers using Median Absolute Deviation."""
        returns = df["returns"]

        # Calculate MAD
        median = _optional_float(returns.median())
        if median is None:
            return []
        mad = _optional_float((returns - median).abs().median())

        # Handle case where MAD is 0
        if mad == 0 or mad is None:
            absolute_median = _optional_float(returns.abs().median())
            if absolute_median is None:
                return []
            mad = 1.4826 * absolute_median

        if mad == 0 or mad is None:  # Still 0, no variation
            return []

        # Modified z-score using MAD
        # 0.6745 is the scaling factor for normal distribution
        modified_z = 0.6745 * (returns - median) / mad

        # Find outliers
        outliers = modified_z.abs() > self.config.threshold

        anomalies = []
        for idx in range(len(df)):
            if outliers[idx]:
                row = df[idx]
                z_score = _float(modified_z[idx])
                return_value = _float(row["returns"][0])
                close_price = _float(row["close"][0])

                # Determine severity based on z-score
                if abs(z_score) > 5:
                    severity = AnomalySeverity.CRITICAL
                elif abs(z_score) > 4:
                    severity = AnomalySeverity.ERROR
                elif abs(z_score) > 3:
                    severity = AnomalySeverity.WARNING
                else:
                    severity = AnomalySeverity.INFO

                anomalies.append(
                    Anomaly(
                        timestamp=_datetime(row["timestamp"][0]),
                        symbol=symbol,
                        type=AnomalyType.RETURN_OUTLIER,
                        severity=severity,
                        value=return_value * 100,  # Convert to percentage
                        expected_range=(
                            float(median - self.config.threshold * mad),
                            float(median + self.config.threshold * mad),
                        ),
                        threshold=self.config.threshold,
                        message=(
                            f"Unusual return of {return_value * 100:.2f}% "
                            f"(MAD z-score: {z_score:.2f})"
                        ),
                        metadata={
                            "method": "mad",
                            "z_score": float(z_score),
                            "close_price": close_price,
                        },
                    )
                )

        return anomalies

    def _detect_zscore(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """Detect outliers using standard z-score."""
        returns = df["returns"]

        # Calculate z-scores
        mean = _optional_float(returns.mean())
        std = _optional_float(returns.std())

        if mean is None or std is None or std == 0:  # No variation
            return []

        z_scores = (returns - mean) / std

        # Find outliers
        outliers = z_scores.abs() > self.config.threshold

        anomalies = []
        for idx in range(len(df)):
            if outliers[idx]:
                row = df[idx]
                z_score = _float(z_scores[idx])
                return_value = _float(row["returns"][0])

                # Determine severity
                if abs(z_score) > 4:
                    severity = AnomalySeverity.CRITICAL
                elif abs(z_score) > 3:
                    severity = AnomalySeverity.ERROR
                else:
                    severity = AnomalySeverity.WARNING

                anomalies.append(
                    Anomaly(
                        timestamp=_datetime(row["timestamp"][0]),
                        symbol=symbol,
                        type=AnomalyType.RETURN_OUTLIER,
                        severity=severity,
                        value=return_value * 100,
                        expected_range=(
                            float(mean - self.config.threshold * std),
                            float(mean + self.config.threshold * std),
                        ),
                        threshold=self.config.threshold,
                        message=(
                            f"Unusual return of {return_value * 100:.2f}% (z-score: {z_score:.2f})"
                        ),
                        metadata={
                            "method": "zscore",
                            "z_score": float(z_score),
                            "close_price": _float(row["close"][0]),
                        },
                    )
                )

        return anomalies

    def _detect_iqr(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """Detect outliers using Interquartile Range."""
        returns = df["returns"]

        # Calculate IQR
        q1 = _optional_float(returns.quantile(0.25))
        q3 = _optional_float(returns.quantile(0.75))
        if q1 is None or q3 is None:
            return []
        iqr = q3 - q1

        if iqr == 0:  # No variation
            return []

        # Define outlier bounds
        lower_bound = q1 - self.config.threshold * iqr
        upper_bound = q3 + self.config.threshold * iqr

        # Find outliers
        outliers = (returns < lower_bound) | (returns > upper_bound)

        anomalies = []
        for idx in range(len(df)):
            if outliers[idx]:
                row = df[idx]
                return_val = _float(row["returns"][0])

                # Determine severity
                extreme_lower = q1 - 3 * iqr
                extreme_upper = q3 + 3 * iqr

                if return_val < extreme_lower or return_val > extreme_upper:
                    severity = AnomalySeverity.CRITICAL
                else:
                    severity = AnomalySeverity.WARNING

                anomalies.append(
                    Anomaly(
                        timestamp=_datetime(row["timestamp"][0]),
                        symbol=symbol,
                        type=AnomalyType.RETURN_OUTLIER,
                        severity=severity,
                        value=return_val * 100,
                        expected_range=(float(lower_bound), float(upper_bound)),
                        threshold=self.config.threshold,
                        message=f"Unusual return of {return_val * 100:.2f}% (outside IQR bounds)",
                        metadata={
                            "method": "iqr",
                            "iqr": float(iqr),
                            "close_price": _float(row["close"][0]),
                        },
                    )
                )

        return anomalies


class VolumeSpikeDetector(AnomalyDetector):
    """Detect unusual volume spikes."""

    def __init__(self, config: VolumeSpikeConfig | None = None):
        """
        Initialize volume spike detector.

        Args:
            config: Detector configuration
        """
        self.config = config or VolumeSpikeConfig()
        super().__init__(enabled=self.config.enabled)

    @property
    def name(self) -> str:
        """Return detector name."""
        return "volume_spikes"

    def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """
        Detect volume spikes in the data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being analyzed

        Returns:
            List of detected anomalies
        """
        if not self.enabled or len(df) < self.config.window + 1:
            return []

        anomalies = []

        # Calculate rolling statistics
        df_with_stats = df.with_columns(
            [
                pl.col("volume").rolling_mean(self.config.window).alias("volume_mean"),
                pl.col("volume").rolling_std(self.config.window).alias("volume_std"),
            ]
        )

        # Calculate z-scores
        df_with_stats = df_with_stats.with_columns(
            ((pl.col("volume") - pl.col("volume_mean")) / pl.col("volume_std")).alias(
                "volume_zscore"
            )
        )

        # Filter for minimum volume
        df_with_stats = df_with_stats.filter(pl.col("volume") > self.config.min_volume)

        # Find spikes
        spikes = df_with_stats.filter(
            (pl.col("volume_zscore").abs() > self.config.threshold)
            & pl.col("volume_zscore").is_not_null()
            & pl.col("volume_zscore").is_not_nan()
        )

        for row in spikes.iter_rows(named=True):
            z_score = _float(row["volume_zscore"])
            volume = _float(row["volume"])
            volume_mean = _float(row["volume_mean"])
            volume_std = _float(row["volume_std"])

            # Determine severity
            if abs(z_score) > 5:
                severity = AnomalySeverity.ERROR
            elif abs(z_score) > 4:
                severity = AnomalySeverity.WARNING
            else:
                severity = AnomalySeverity.INFO

            # Calculate percentage change
            pct_change = ((volume - volume_mean) / volume_mean) * 100

            anomalies.append(
                Anomaly(
                    timestamp=_datetime(row["timestamp"]),
                    symbol=symbol,
                    type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    value=volume,
                    expected_range=(
                        volume_mean - self.config.threshold * volume_std,
                        volume_mean + self.config.threshold * volume_std,
                    ),
                    threshold=self.config.threshold,
                    message=(
                        f"Volume spike: {volume:,.0f} "
                        f"({pct_change:+.1f}% vs {self.config.window}-day avg)"
                    ),
                    metadata={
                        "z_score": float(z_score),
                        "average_volume": volume_mean,
                        "window": self.config.window,
                    },
                )
            )

        return anomalies


class PriceStalenessDetector(AnomalyDetector):
    """Detect stale/unchanged prices."""

    def __init__(self, config: PriceStalenessConfig | None = None):
        """
        Initialize price staleness detector.

        Args:
            config: Detector configuration
        """
        self.config = config or PriceStalenessConfig()
        super().__init__(enabled=self.config.enabled)

    @property
    def name(self) -> str:
        """Return detector name."""
        return "price_staleness"

    def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """
        Detect stale prices in the data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being analyzed

        Returns:
            List of detected anomalies
        """
        if not self.enabled or len(df) < 2:
            return []

        anomalies = []

        if self.config.check_close_only:
            # Check only close prices
            df_with_changes = df.with_columns(pl.col("close").diff().alias("close_change"))

            # Find consecutive unchanged prices
            df_with_groups = df_with_changes.with_columns(
                (pl.col("close_change") != 0).cum_sum().alias("change_group")
            )
        else:
            # Check if all OHLC prices are unchanged
            df_with_changes = df.with_columns(
                [
                    pl.col("open").diff().alias("open_change"),
                    pl.col("high").diff().alias("high_change"),
                    pl.col("low").diff().alias("low_change"),
                    pl.col("close").diff().alias("close_change"),
                ]
            )

            df_with_changes = df_with_changes.with_columns(
                (
                    (pl.col("open_change") == 0)
                    & (pl.col("high_change") == 0)
                    & (pl.col("low_change") == 0)
                    & (pl.col("close_change") == 0)
                ).alias("all_unchanged")
            )

            # Group consecutive unchanged periods
            df_with_groups = df_with_changes.with_columns(
                (~pl.col("all_unchanged")).cum_sum().alias("change_group")
            )

        # Find stale periods
        group_sizes = df_with_groups.group_by("change_group").agg(
            [
                pl.col("timestamp").min().alias("start_date"),
                pl.col("timestamp").max().alias("end_date"),
                pl.col("close").first().alias("stale_price"),
                pl.len().alias("days_unchanged"),
            ]
        )

        # Filter for periods exceeding threshold
        # (subtract 1 since the first occurrence doesn't count)
        stale_periods = group_sizes.filter(
            (pl.col("days_unchanged") - 1) > self.config.max_unchanged_days
        )

        for row in stale_periods.iter_rows(named=True):
            days = int(_float(row["days_unchanged"])) - 1
            stale_price = _float(row["stale_price"])
            start_date = _datetime(row["start_date"])
            end_date = _datetime(row["end_date"])

            # Determine severity based on staleness duration
            if days > 20:
                severity = AnomalySeverity.CRITICAL
            elif days > 10:
                severity = AnomalySeverity.ERROR
            elif days > 5:
                severity = AnomalySeverity.WARNING
            else:
                severity = AnomalySeverity.INFO

            anomalies.append(
                Anomaly(
                    timestamp=end_date,
                    symbol=symbol,
                    type=AnomalyType.PRICE_STALE,
                    severity=severity,
                    value=stale_price,
                    threshold=float(self.config.max_unchanged_days),
                    message=f"Price unchanged for {days} days at {stale_price:.2f}",
                    metadata={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days_unchanged": days,
                        "stale_price": stale_price,
                    },
                )
            )

        return anomalies

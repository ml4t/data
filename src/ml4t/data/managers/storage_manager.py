"""Storage operations for DataManager.

This module handles data storage operations including:
- Initial data loading
- Data import from external sources
- Incremental updates with gap detection
- Data validation and merging
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl
import structlog
from pydantic import ValidationError
from tenacity import RetryError

from ml4t.data.core.schemas import align_frames_for_concat, timestamp_bounds
from ml4t.data.sessions.assigner import SessionAssigner
from ml4t.data.storage.backend import normalize_storage_metadata
from ml4t.data.utils.conversion import pandas_to_polars
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

if TYPE_CHECKING:
    from ml4t.data.managers.fetch_manager import FetchManager

logger = structlog.get_logger()

_TEMPORAL_FIELDS = (
    "calendar",
    "time_contract_version",
    "timezone",
    "timestamp_semantics",
    "session_start_time",
    "bar_grid_anchor",
    "bar_label",
    "partial_bar_policy",
)


class StorageManager:
    """Manages data storage operations.

    This class handles all storage-related operations including loading,
    importing, and updating data with validation and gap detection.

    Attributes:
        storage: Storage backend instance
        fetch_manager: FetchManager for data fetching
        enable_validation: Whether to validate data
        progress_callback: Optional progress reporting callback

    Example:
        >>> storage_mgr = StorageManager(storage, fetch_manager)
        >>> key = storage_mgr.load("AAPL", "2024-01-01", "2024-12-31")
        >>> key = storage_mgr.update("AAPL")
    """

    def __init__(
        self,
        storage: Any,
        fetch_manager: FetchManager,
        enable_validation: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """Initialize StorageManager.

        Args:
            storage: Storage backend instance
            fetch_manager: FetchManager instance
            enable_validation: Enable data validation
            progress_callback: Optional callback for progress updates
        """
        self.storage = storage
        self.fetch_manager = fetch_manager
        self.enable_validation = enable_validation
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is configured.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(message, progress=f"{progress:.0%}")

    def _validate_data(
        self,
        df: pl.DataFrame,
        symbol: str,
        asset_class: str = "equity",
        *,
        frequency: str | None = None,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> Any:
        """Validate data using OHLCV and cross-validation.

        Args:
            df: DataFrame to validate
            symbol: Symbol being validated
            asset_class: Asset class for validation rules

        Returns:
            ValidationReport with results
        """
        from ml4t.data.validation import OHLCVValidator, ValidationReport
        from ml4t.data.validation.cross_validation import CrossValidator
        from ml4t.data.validation.rules import ValidationRulePresets

        report = ValidationReport(symbol=symbol, provider="unknown")

        rules = ValidationRulePresets.for_asset_class(asset_class)
        if calendar is UNSET_CALENDAR:
            calendar = rules.calendar
        if feed_spec is None:
            feed_spec = rules

        # Run OHLCV validation
        ohlcv_validator = OHLCVValidator(
            check_nulls=rules.check_nulls,
            check_price_consistency=rules.check_price_consistency,
            negative_price_policy=rules.negative_price_policy,
            check_negative_volume=rules.check_negative_volume,
            check_duplicate_timestamps=rules.check_duplicate_timestamps,
            check_chronological_order=rules.check_chronological_order,
            check_price_staleness=rules.check_price_staleness,
            check_extreme_returns=rules.check_extreme_returns,
            max_return_threshold=rules.max_return_threshold,
            staleness_threshold=rules.staleness_threshold,
        )

        result = ohlcv_validator.validate(df)
        report.add_result(result, "OHLCVValidator")

        # Run cross-validation
        cross_validator = CrossValidator(
            check_price_continuity=rules.check_price_continuity,
            check_volume_spikes=rules.check_volume_spikes,
            check_weekend_trading=rules.check_weekend_trading,
            check_market_hours=rules.check_market_hours,
            volume_spike_threshold=rules.volume_spike_threshold,
            price_gap_threshold=rules.price_gap_threshold,
            calendar=calendar,
            feed_spec=feed_spec,
        )

        result = cross_validator.validate(df, frequency=frequency)
        report.add_result(result, "CrossValidator")

        return report

    def _merge_data(self, existing: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
        """Merge existing and new data, handling duplicates.

        Args:
            existing: Existing DataFrame
            new: New DataFrame to merge

        Returns:
            Merged DataFrame with duplicates removed
        """
        existing, new = align_frames_for_concat(
            existing,
            new,
            left_fill_values={"dividends": 0.0, "splits": 1.0},
            right_fill_values={"dividends": 0.0, "splits": 1.0},
        )

        # Merge data: concatenate and remove duplicates
        merged_df = pl.concat([existing, new])

        # Remove duplicates, keeping the last (most recent) occurrence
        merged_df = merged_df.unique(
            subset=["timestamp"],
            keep="last",
        ).sort("timestamp")

        return merged_df

    def _ensure_polars_df(self, df: Any) -> pl.DataFrame:
        """Convert any DataFrame type to Polars DataFrame.

        Args:
            df: DataFrame (Polars, LazyFrame, or pandas)

        Returns:
            Polars DataFrame
        """
        if isinstance(df, pl.DataFrame):
            return df
        if hasattr(df, "collect"):  # LazyFrame
            return df.collect()
        if hasattr(df, "columns"):
            return pandas_to_polars(df)
        return df

    @staticmethod
    def _ingestion_feed(
        calendar: str | None | CalendarUnset,
        feed_spec: Any | None,
        frequency: str,
        bar_type: str,
    ) -> tuple[FeedSpec, dict[str, Any]]:
        source = normalize_storage_metadata(feed_spec) if isinstance(feed_spec, dict) else feed_spec
        feed = (
            FeedSpec.from_any(source) if source is not None else FeedSpec(time_contract_version=2)
        )
        aliases = {
            "hour": "hourly",
            "1h": "hourly",
            "1hour": "hourly",
            "1d": "daily",
            "1day": "daily",
            "1m": "minute",
            "1min": "minute",
            "1minute": "minute",
        }
        requested = aliases.get(frequency.lower(), frequency.lower())
        declared = str(feed.data_frequency).lower()
        if feed.data_frequency is not None and aliases.get(declared, declared) != requested:
            raise ValueError("feed data_frequency must match the requested frequency")
        if feed.bar_type not in (None, bar_type):
            raise ValueError("feed bar_type must match the requested bar_type")
        if calendar is not UNSET_CALENDAR:
            feed = feed.with_overrides(calendar=calendar, time_contract_version=2)
        SessionAssigner(feed_spec=feed)
        wire = feed.to_dict()
        return feed, {name: wire[name] for name in _TEMPORAL_FIELDS if name in wire}

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        provider: str | None = None,
        bar_type: str = "time",
        bar_threshold: int | None = None,
        exchange: str = "UNKNOWN",
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        *,
        feed_spec: Any | None = None,
    ) -> str:
        """Load data from provider and store it.

        Args:
            symbol: Symbol to load
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (for time bars)
            asset_class: Asset class
            provider: Optional provider override
            bar_type: Type of bars ("time", "volume", "trade", "dollar", "tick")
            bar_threshold: Threshold for non-time bars
            exchange: Exchange code
            calendar: Optional calendar override

        Returns:
            Storage key for the loaded data

        Raises:
            ValueError: If storage not configured or no data returned
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        feed, temporal_values = self._ingestion_feed(calendar, feed_spec, frequency, bar_type)

        logger.info(
            "Starting data load",
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            asset_class=asset_class,
        )

        self._report_progress(f"Loading {symbol}", 0.0)

        try:
            # Fetch data from provider
            self._report_progress(f"Fetching data for {symbol}", 0.2)
            df = self.fetch_manager.fetch_raw(symbol, start, end, frequency, provider)

            if df.is_empty():
                logger.warning("Provider returned empty data", symbol=symbol)
                raise ValueError(f"No data returned for {symbol}")

            logger.info(f"Fetched {len(df)} rows of data")
            self._report_progress(f"Fetched {len(df)} rows", 0.5)

            min_ts, max_ts = timestamp_bounds(df)

            # Validate data if enabled
            if self.enable_validation:
                validation_report = self._validate_data(
                    df, symbol, asset_class, frequency=frequency, feed_spec=feed
                )
                if not validation_report.passed:
                    logger.warning(
                        "Data validation issues found",
                        symbol=symbol,
                        critical_count=validation_report.critical_count,
                        error_count=validation_report.error_count,
                        warning_count=validation_report.warning_count,
                    )

            # Create metadata
            from ml4t.data.core.models import DataObject, Metadata

            bar_params = {}
            if bar_type == "time":
                bar_params["frequency"] = frequency
            elif bar_threshold is not None:
                bar_params["threshold"] = bar_threshold

            metadata = Metadata(
                provider=provider or "auto",
                symbol=symbol,
                asset_class=asset_class,
                bar_type=bar_type,
                bar_params=bar_params,
                exchange=exchange,
                **temporal_values,
                start_date=min_ts,
                end_date=max_ts,
                last_updated=datetime.now(UTC),
                data_range={
                    "start": str(min_ts),
                    "end": str(max_ts),
                },
            )

            # Create data object
            data_obj = DataObject(data=df, metadata=metadata)

            # Check if data already exists
            key = f"{asset_class}/{frequency}/{symbol}"
            if self.storage.exists(key):
                logger.warning(f"Data already exists for {key}, overwriting")

            # Store data
            self._report_progress(f"Storing data for {symbol}", 0.8)

            metadata_dict = data_obj.metadata.model_dump() if data_obj.metadata else None
            self.storage.write(data_obj.data, key, metadata_dict)

            logger.info("Data load completed", key=key, rows=len(df))
            self._report_progress(f"Completed loading {symbol}", 1.0)

            return key

        except RetryError as e:
            logger.error("Data load failed after retries", symbol=symbol, error=str(e))
            raise ValueError(f"Failed to load {symbol} after retries") from e

        except Exception as e:
            logger.error("Data load failed", symbol=symbol, error=str(e), exc_info=True)
            raise

    def import_data(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        symbol: str,
        provider: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        bar_type: str = "time",
        bar_threshold: int | None = None,
        exchange: str = "UNKNOWN",
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        *,
        feed_spec: Any | None = None,
    ) -> str:
        """Import external data into storage with metadata.

        Args:
            data: DataFrame or LazyFrame with OHLCV data
            symbol: Symbol identifier
            provider: Provider name (for future updates)
            frequency: Data frequency
            asset_class: Asset class
            bar_type: Type of bars
            bar_threshold: Threshold for non-time bars
            exchange: Exchange code
            calendar: Optional calendar override

        Returns:
            Storage key for the imported data

        Raises:
            ValueError: If storage not configured or data is empty
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        feed, temporal_values = self._ingestion_feed(calendar, feed_spec, frequency, bar_type)

        logger.info(
            "Importing external data",
            symbol=symbol,
            provider=provider,
            rows=len(data) if isinstance(data, pl.DataFrame) else "unknown",
        )

        try:
            df = self._ensure_polars_df(data)

            if df.is_empty():
                raise ValueError(f"Cannot import empty data for {symbol}")

            # Validate schema
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            min_ts, max_ts = timestamp_bounds(df)

            # Validate data if enabled
            if self.enable_validation:
                validation_report = self._validate_data(
                    df, symbol, asset_class, frequency=frequency, feed_spec=feed
                )
                if not validation_report.passed:
                    logger.warning(
                        "Imported data has validation issues",
                        symbol=symbol,
                        critical_count=validation_report.critical_count,
                    )

            # Create metadata
            from ml4t.data.core.models import DataObject, Metadata

            bar_params = {}
            if bar_type == "time":
                bar_params["frequency"] = frequency
            elif bar_threshold is not None:
                bar_params["threshold"] = bar_threshold

            metadata = Metadata(
                provider=provider,
                symbol=symbol,
                asset_class=asset_class,
                bar_type=bar_type,
                bar_params=bar_params,
                exchange=exchange,
                **temporal_values,
                start_date=min_ts,
                end_date=max_ts,
                last_updated=datetime.now(UTC),
                data_range={
                    "start": str(min_ts),
                    "end": str(max_ts),
                },
            )

            # Create data object and store
            data_obj = DataObject(data=df, metadata=metadata)
            key = f"{asset_class}/{frequency}/{symbol}"

            metadata_dict = data_obj.metadata.model_dump()
            self.storage.write(data_obj.data, key, metadata_dict)

            logger.info("Data import completed", key=key, rows=len(df))
            return key

        except Exception as e:
            logger.error("Data import failed", symbol=symbol, error=str(e), exc_info=True)
            raise

    def update(
        self,
        symbol: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        lookback_days: int = 7,
        fill_gaps: bool | None = None,
        provider: str | None = None,
        initial_start: str | None = None,
        initial_end: str | None = None,
        initial_load_days: int = 365,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
        check_gaps: bool = True,
    ) -> str:
        """Update existing data with incremental fetch.

        Args:
            symbol: Symbol to update
            frequency: Data frequency
            asset_class: Asset class
            lookback_days: Days to look back for updates
            fill_gaps: Legacy False disables checks; True is rejected, never imputed.
            check_gaps: Report calendar gaps without changing observations.
            provider: Optional provider override
            initial_start: Optional start date for first load when no data exists
            initial_end: Optional end date for first load when no data exists
            initial_load_days: Days to fetch on first load when initial_start is not set

        Returns:
            Storage key for the updated data

        Raises:
            ValueError: If storage not configured
        """
        if fill_gaps is True:
            raise ValueError(
                "Synthetic gap imputation is unsupported; inspect gaps and download source observations"
            )
        if fill_gaps is False:
            check_gaps = False
        if not self.storage:
            raise ValueError("Storage not configured")

        logger.info(
            "Starting incremental update",
            symbol=symbol,
            frequency=frequency,
            lookback_days=lookback_days,
        )

        self._report_progress(f"Updating {symbol}", 0.0)

        try:
            key = f"{asset_class}/{frequency}/{symbol}"

            if not self.storage.exists(key):
                logger.warning("No existing data found, performing initial load", symbol=symbol)
                today = datetime.now(UTC)
                end_date = initial_end or today.strftime("%Y-%m-%d")
                effective_load_days = initial_load_days
                if initial_start is None:
                    max_history_days = self.fetch_manager.get_max_history_days(symbol, provider)
                    if max_history_days is not None and effective_load_days > max_history_days:
                        logger.warning(
                            "Initial load limited by provider history",
                            provider=provider,
                            requested_days=effective_load_days,
                            max_history_days=max_history_days,
                        )
                        effective_load_days = max_history_days
                start_date = initial_start or (
                    today - timedelta(days=effective_load_days)
                ).strftime("%Y-%m-%d")
                return self.load(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    frequency=frequency,
                    asset_class=asset_class,
                    provider=provider,
                    calendar=calendar,
                    feed_spec=feed_spec,
                )

            # Read existing data
            self._report_progress(f"Reading existing data for {symbol}", 0.1)
            existing_lazy = self.storage.read(key)
            existing_df = existing_lazy.collect()

            if existing_df.is_empty():
                raise ValueError(f"Existing data for {symbol} is empty")

            logger.info("Found existing data", rows=len(existing_df))

            existing_metadata = (
                normalize_storage_metadata(self.storage.get_metadata(key), key) or {}
            )
            stored_bar_type = existing_metadata.get("bar_type")
            effective_bar_type = stored_bar_type if isinstance(stored_bar_type, str) else "time"
            effective_feed, temporal_values = self._ingestion_feed(
                calendar,
                existing_metadata if feed_spec is None else feed_spec,
                frequency,
                effective_bar_type,
            )
            existing_provider = existing_metadata.get("provider")
            if (
                not provider
                and existing_provider is not None
                and not isinstance(existing_provider, str)
            ):
                raise ValueError(
                    f"Stored metadata provider for '{key}' must be a string, "
                    f"got {type(existing_provider).__name__}"
                )
            if not provider and existing_provider not in (None, "", "auto"):
                provider = existing_provider

            # Get the date range from existing data
            _, last_timestamp = timestamp_bounds(existing_df)
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.replace(tzinfo=UTC)
            else:
                last_timestamp = last_timestamp.astimezone(UTC)

            # Calculate fetch range
            fetch_start = last_timestamp - timedelta(days=lookback_days)
            fetch_end = datetime.now(UTC)
            provider_history_limited = False
            max_history_days = self.fetch_manager.get_max_history_days(symbol, provider)
            if max_history_days is not None:
                earliest_fetch_start = fetch_end - timedelta(days=max_history_days)
                if fetch_start < earliest_fetch_start:
                    logger.warning(
                        "Incremental update limited by provider history",
                        provider=provider,
                        requested_start=fetch_start,
                        effective_start=earliest_fetch_start,
                        max_history_days=max_history_days,
                    )
                    fetch_start = earliest_fetch_start
                    provider_history_limited = fetch_start.date() > last_timestamp.date()

            # Skip update if data is already current
            if frequency == "daily" and (fetch_end - last_timestamp).days < 1:
                logger.info("Data is already up to date", last_timestamp=last_timestamp)
                self._report_progress(f"Data for {symbol} is already current", 1.0)
                return key

            self._report_progress(f"Fetching new data for {symbol}", 0.3)

            # Fetch new data
            new_df = self.fetch_manager.fetch_raw(
                symbol=symbol,
                start=fetch_start.strftime("%Y-%m-%d"),
                end=fetch_end.strftime("%Y-%m-%d"),
                frequency=frequency,
                provider=provider,
            )

            if new_df.is_empty():
                logger.warning("No new data available", symbol=symbol)
                return key

            logger.info(f"Fetched {len(new_df)} rows of new data")
            self._report_progress(f"Merging data for {symbol}", 0.5)

            # Merge data
            merged_df = self._merge_data(existing_df, new_df)

            logger.info(
                "Merged data",
                original_rows=len(existing_df),
                new_rows=len(new_df),
                merged_rows=len(merged_df),
                added_rows=len(merged_df) - len(existing_df),
            )

            # Gap checks never change observations, even outside provider history limits.
            gaps = []
            gap_check_performed = check_gaps and effective_feed.resolve_calendar() is not None
            if gap_check_performed:
                self._report_progress(f"Checking for gaps in {symbol}", 0.6)

                from ml4t.data.utils.gaps import GapDetector

                selected_calendar = effective_feed.resolve_calendar()
                if selected_calendar is not None and effective_bar_type != "time":
                    raise NotImplementedError("calendar gap checks require fixed time bars")
                gap_detector = GapDetector()
                gaps = gap_detector.detect_gaps(
                    merged_df,
                    frequency=frequency,
                    calendar=selected_calendar,
                    feed_spec=effective_feed,
                )

                if gaps:
                    gap_summary = gap_detector.summarize_gaps(gaps)
                    logger.warning(
                        "Gaps detected in data",
                        gap_count=gap_summary["count"],
                        total_missing=gap_summary["total_missing_periods"],
                    )

            self._report_progress(f"Storing updated data for {symbol}", 0.8)

            # Update metadata with new range
            from ml4t.data.core.models import DataObject, Metadata

            min_ts, max_ts = timestamp_bounds(merged_df)
            existing_attributes = existing_metadata.get("attributes")
            updated_attributes = (
                existing_attributes.copy() if isinstance(existing_attributes, dict) else {}
            )
            updated_at = datetime.now(UTC)
            updated_attributes.update(
                {
                    # BulkManager compares this legacy field with naive local cutoffs.
                    "last_update": updated_at.astimezone().replace(tzinfo=None).isoformat(),
                    "update_type": "incremental",
                    "gaps_filled": False,
                    "gap_check_performed": gap_check_performed,
                    "gap_count": len(gaps) if gap_check_performed else None,
                    "missing_observations": (
                        sum(gap.missing_periods for gap in gaps) if gap_check_performed else None
                    ),
                    "provider_history_limited": provider_history_limited,
                }
            )

            existing_model_values = {
                name: value
                for name in Metadata.model_fields
                if name in existing_metadata
                and (
                    (value := existing_metadata[name]) is not None
                    or (name == "calendar" and existing_metadata.get("time_contract_version") == 2)
                )
            }
            resolved_provider = (
                provider
                or (existing_provider if isinstance(existing_provider, str) else None)
                or "auto"
            )
            existing_bar_type = existing_metadata.get("bar_type")
            existing_bar_params = existing_metadata.get("bar_params")
            metadata_values = {
                **existing_model_values,
                "provider": resolved_provider,
                "symbol": symbol,
                "asset_class": asset_class,
                "bar_type": existing_bar_type if isinstance(existing_bar_type, str) else "time",
                "bar_params": (
                    existing_bar_params
                    if isinstance(existing_bar_params, dict)
                    else {"frequency": frequency}
                ),
                "start_date": min_ts,
                "end_date": max_ts,
                "last_updated": updated_at,
                "data_range": {
                    "start": str(min_ts),
                    "end": str(max_ts),
                },
                "attributes": updated_attributes,
            }

            for name in _TEMPORAL_FIELDS:
                metadata_values.pop(name, None)
            metadata_values.update(temporal_values)

            try:
                updated_metadata = Metadata.model_validate(metadata_values)
            except ValidationError as error:
                invalid_fields = {
                    location[0]
                    for detail in error.errors()
                    if (location := detail.get("loc")) and isinstance(location[0], str)
                }
                removable_fields = invalid_fields - {
                    "provider",
                    "symbol",
                    "asset_class",
                    *_TEMPORAL_FIELDS,
                }
                if not removable_fields:
                    raise
                logger.warning(
                    "Ignoring invalid optional fields in stored metadata",
                    key=key,
                    fields=sorted(removable_fields),
                )
                for field in removable_fields:
                    metadata_values.pop(field, None)
                updated_metadata = Metadata.model_validate(metadata_values)

            updated_obj = DataObject(data=merged_df, metadata=updated_metadata)

            metadata_dict = updated_obj.metadata.model_dump() if updated_obj.metadata else None
            self.storage.write(updated_obj.data, key, metadata_dict, preserve_metadata=True)

            logger.info(
                "Incremental update completed",
                key=key,
                total_rows=len(merged_df),
                added_rows=len(merged_df) - len(existing_df),
            )
            self._report_progress(f"Completed updating {symbol}", 1.0)

            return key

        except RetryError as e:
            logger.error("Update failed after retries", symbol=symbol, error=str(e))
            raise ValueError(f"Failed to update {symbol} after retries") from e

        except Exception as e:
            logger.error("Update failed", symbol=symbol, error=str(e), exc_info=True)
            raise

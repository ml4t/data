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
from tenacity import RetryError

from ml4t.data.core.schemas import align_frames_for_concat

if TYPE_CHECKING:
    from ml4t.data.managers.fetch_manager import FetchManager

logger = structlog.get_logger()


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

        # Select validation rules based on asset class
        if asset_class.lower() == "crypto":
            rules = ValidationRulePresets.crypto_rules()
            is_crypto = True
        elif asset_class.lower() == "forex":
            rules = ValidationRulePresets.forex_rules()
            is_crypto = False
        elif asset_class.lower() == "commodity":
            rules = ValidationRulePresets.commodity_rules()
            is_crypto = False
        else:
            rules = ValidationRulePresets.equity_rules()
            is_crypto = False

        # Run OHLCV validation
        ohlcv_validator = OHLCVValidator(
            check_nulls=rules.check_nulls,
            check_price_consistency=rules.check_price_consistency,
            check_negative_prices=rules.check_negative_prices,
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
            is_crypto=is_crypto,
        )

        result = cross_validator.validate(df)
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
        if hasattr(df, "to_polars"):  # pandas
            return pl.from_pandas(df)
        return df

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
        calendar: str | None = None,
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

            # Validate data if enabled
            if self.enable_validation:
                validation_report = self._validate_data(df, symbol, asset_class)
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

            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

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
                calendar=calendar,
                start_date=min_ts,
                end_date=max_ts,
                last_updated=datetime.now(UTC),
                frequency=frequency,
                data_range={
                    "start": str(min_ts) if min_ts is not None else "",
                    "end": str(max_ts) if max_ts is not None else "",
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

            if hasattr(self.storage, "transaction"):
                with self.storage.transaction() as txn:
                    txn.write(data_obj)
            else:
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
        calendar: str | None = None,
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

            # Validate data if enabled
            if self.enable_validation:
                validation_report = self._validate_data(df, symbol, asset_class)
                if not validation_report.passed:
                    logger.warning(
                        "Imported data has validation issues",
                        symbol=symbol,
                        critical_count=validation_report.critical_count,
                    )

            # Create metadata
            from ml4t.data.core.models import DataObject, Metadata

            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

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
                calendar=calendar,
                start_date=min_ts,
                end_date=max_ts,
                last_updated=datetime.now(UTC),
                frequency=frequency,
                data_range={
                    "start": str(min_ts) if min_ts is not None else "",
                    "end": str(max_ts) if max_ts is not None else "",
                },
            )

            # Create data object and store
            data_obj = DataObject(data=df, metadata=metadata)
            key = f"{asset_class}/{frequency}/{symbol}"

            if hasattr(self.storage, "transaction"):
                with self.storage.transaction() as txn:
                    txn.write(data_obj)
            else:
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
        fill_gaps: bool = True,
        provider: str | None = None,
    ) -> str:
        """Update existing data with incremental fetch.

        Args:
            symbol: Symbol to update
            frequency: Data frequency
            asset_class: Asset class
            lookback_days: Days to look back for updates
            fill_gaps: If True, detect and fill gaps in data
            provider: Optional provider override

        Returns:
            Storage key for the updated data

        Raises:
            ValueError: If storage not configured
        """
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
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                return self.load(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    frequency=frequency,
                    asset_class=asset_class,
                    provider=provider,
                )

            # Read existing data
            self._report_progress(f"Reading existing data for {symbol}", 0.1)
            existing_lazy = self.storage.read(key)
            existing_df = existing_lazy.collect()

            if existing_df.is_empty():
                raise ValueError(f"Existing data for {symbol} is empty")

            logger.info("Found existing data", rows=len(existing_df))

            # Get the date range from existing data
            last_timestamp = existing_df["timestamp"].max()

            # Calculate fetch range
            fetch_start = last_timestamp - timedelta(days=lookback_days)
            fetch_end = datetime.now(UTC)

            # Normalize timezone: stored timestamps may be naive while now(UTC) is aware
            if last_timestamp.tzinfo is None and fetch_end.tzinfo is not None:
                fetch_end = fetch_end.replace(tzinfo=None)

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

            # Detect and optionally fill gaps
            gaps = []
            if fill_gaps:
                self._report_progress(f"Checking for gaps in {symbol}", 0.6)

                from ml4t.data.utils.gaps import GapDetector

                is_crypto = asset_class.lower() == "crypto"
                gap_detector = GapDetector()
                gaps = gap_detector.detect_gaps(merged_df, frequency=frequency, is_crypto=is_crypto)

                if gaps:
                    gap_summary = gap_detector.summarize_gaps(gaps)
                    logger.warning(
                        "Gaps detected in data",
                        gap_count=gap_summary["count"],
                        total_missing=gap_summary["total_missing_periods"],
                    )

                    merged_df = gap_detector.fill_gaps(merged_df, gaps, method="forward")
                    logger.info("Filled gaps using forward fill", final_rows=len(merged_df))

            self._report_progress(f"Storing updated data for {symbol}", 0.8)

            # Update metadata with new range
            from ml4t.data.core.models import DataObject, Metadata

            min_ts = merged_df["timestamp"].min()
            max_ts = merged_df["timestamp"].max()

            updated_metadata = Metadata(
                provider=provider or "auto",
                symbol=symbol,
                asset_class=asset_class,
                bar_type="time",
                bar_params={"frequency": frequency},
                data_range={
                    "start": str(min_ts) if min_ts is not None else "",
                    "end": str(max_ts) if max_ts is not None else "",
                },
                attributes={
                    "last_update": datetime.now().isoformat(),
                    "update_type": "incremental",
                    "gaps_filled": fill_gaps and len(gaps) > 0,
                },
            )

            updated_obj = DataObject(data=merged_df, metadata=updated_metadata)

            if hasattr(self.storage, "transaction"):
                with self.storage.transaction() as txn:
                    if hasattr(txn, "update"):
                        txn.update(key, updated_obj)
                    else:
                        txn.write(updated_obj)
            else:
                metadata_dict = updated_obj.metadata.model_dump() if updated_obj.metadata else None
                self.storage.write(updated_obj.data, key, metadata_dict)

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

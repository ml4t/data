"""Provider-based incremental updater using Template Method pattern.

Ported from crypto-data-pipeline's proven BaseUpdater pattern.
This integrates provider fetching, transformation, and storage in one workflow.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import structlog

from ml4t.data.storage.protocols import IncrementalStorageBackend

logger = structlog.get_logger(__name__)


class ProviderUpdater(ABC):
    """Abstract base class for provider-based incremental updates using Template Method pattern.

    This class implements the common incremental update algorithm while delegating
    provider-specific implementation details to subclasses.

    The update workflow:
    1. Determine time range (use latest timestamp if incremental)
    2. Fetch data from provider
    3. Transform data to standard format
    4. Save chunk and update combined file
    5. Update metadata

    Example:
        class YahooUpdater(ProviderUpdater):
            def __init__(self, provider: YahooFinanceProvider, storage: HiveStorage):
                super().__init__("yahoo", storage)
                self.provider = provider

            def _fetch_data(self, symbol, start_time, end_time, **kwargs):
                return self.provider.fetch_ohlcv(symbol, start_time, end_time)

            def _transform_data(self, data, symbol, **kwargs):
                return data  # Already in correct format

            def _get_default_start_time(self, symbol):
                return datetime.now() - timedelta(days=365)
    """

    def __init__(
        self,
        provider_name: str,
        storage: IncrementalStorageBackend,
        safety_margin_minutes: int = 5,
    ):
        """Initialize provider updater.

        Args:
            provider_name: Name of data provider (e.g., "yahoo", "polygon")
            storage: Storage backend supporting incremental updates
            safety_margin_minutes: Minutes to subtract from latest timestamp
                                   to avoid missing data due to delayed updates
        """
        self.provider_name = provider_name
        self.storage = storage
        self.safety_margin = timedelta(minutes=safety_margin_minutes)
        self.logger = logger.bind(provider=provider_name)

    def update_symbol(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        incremental: bool = True,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update data for a single symbol using Template Method pattern.

        This method defines the overall algorithm while delegating specific
        steps to abstract methods implemented by subclasses.

        Args:
            symbol: Symbol to update (e.g., "AAPL", "BTC-USD")
            start_time: Start time (uses latest timestamp if None and incremental)
            end_time: End time (uses current time if None)
            incremental: If True, fetch only data after latest timestamp
            dry_run: If True, don't write files
            **kwargs: Additional arguments passed to subclass methods

        Returns:
            Update statistics dictionary with keys:
                - symbol: Symbol updated
                - start_time: Start of update period
                - end_time: End of update period
                - records_fetched: Records received from provider
                - records_added: New records added to storage
                - success: Whether update succeeded
                - errors: List of error messages
        """
        symbol = symbol.upper()
        stats = self._initialize_stats(symbol, **kwargs)

        try:
            # Step 1: Determine time range
            start_time, end_time = self._get_time_range(symbol, start_time, end_time, incremental)

            # Check if no update needed
            if start_time is None and end_time is None:
                self.logger.info("No update needed - already up to date", symbol=symbol)
                stats["start_time"] = None
                stats["end_time"] = None
                stats["records_fetched"] = 0
                stats["records_added"] = 0
                stats["success"] = True
                stats["skip_reason"] = "already_up_to_date"
                return stats

            stats["start_time"] = start_time
            stats["end_time"] = end_time

            # Step 2: Fetch data from provider
            self.logger.info(
                "Fetching data",
                symbol=symbol,
                start=start_time,
                end=end_time,
                incremental=incremental,
            )

            data = self._fetch_data(symbol, start_time, end_time, **kwargs)
            stats["records_fetched"] = len(data) if data is not None else 0

            if data is None or len(data) == 0:
                self.logger.info("No data fetched", symbol=symbol)
                stats["success"] = True
                return stats

            # Step 3: Transform data to standard format
            transformed = self._transform_data(data, symbol, **kwargs)

            if len(transformed) == 0:
                self.logger.info("No data after transformation", symbol=symbol)
                stats["success"] = True
                return stats

            # Step 4: Save data
            if not dry_run:
                records_added = self._save_data(transformed, symbol, start_time, end_time)
                stats["records_added"] = records_added

                self.logger.info(
                    "Update completed successfully",
                    symbol=symbol,
                    records_fetched=stats["records_fetched"],
                    records_added=records_added,
                )
            else:
                stats["records_added"] = len(transformed)
                self.logger.info(
                    "Dry run - no data saved",
                    symbol=symbol,
                    records=len(transformed),
                )

            stats["success"] = True

        except Exception as e:
            error_msg = str(e)
            stats["errors"].append(error_msg)
            stats["success"] = False

            self.logger.error(
                "Update failed",
                symbol=symbol,
                error=error_msg,
                exc_info=True,
            )

        return stats

    def update_symbols(
        self,
        symbols: list[str],
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Update multiple symbols with concurrent processing.

        Args:
            symbols: List of symbols to update
            max_workers: Maximum parallel workers (default: min(4, len(symbols)))
            **kwargs: Arguments passed to update_symbol

        Returns:
            Dictionary mapping symbols to their update results
        """
        if not symbols:
            return {}

        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(4, len(symbols))
        max_workers = max(1, min(max_workers, len(symbols)))

        self.logger.info(
            "Starting concurrent symbol updates",
            symbols=symbols,
            max_workers=max_workers,
        )

        results = {}

        if max_workers == 1 or len(symbols) == 1:
            # Sequential processing
            for symbol in symbols:
                results[symbol] = self.update_symbol(symbol, **kwargs)
        else:
            # Concurrent processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self.update_symbol, symbol, **kwargs): symbol
                    for symbol in symbols
                }

                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result

                        if result.get("success", False):
                            self.logger.info(
                                "Symbol update completed",
                                symbol=symbol,
                                records_added=result.get("records_added", 0),
                            )
                        else:
                            self.logger.error(
                                "Symbol update failed",
                                symbol=symbol,
                                errors=result.get("errors", []),
                            )
                    except Exception as e:
                        results[symbol] = {
                            "symbol": symbol,
                            "success": False,
                            "errors": [str(e)],
                            "records_fetched": 0,
                            "records_added": 0,
                        }
                        self.logger.error(
                            "Symbol update failed with exception",
                            symbol=symbol,
                            error=str(e),
                        )

        # Log summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        total_added = sum(r.get("records_added", 0) for r in results.values())

        self.logger.info(
            "Concurrent updates completed",
            symbols_processed=len(symbols),
            successful=successful,
            failed=len(symbols) - successful,
            total_records_added=total_added,
        )

        return results

    def _get_time_range(
        self,
        symbol: str,
        start_time: datetime | None,
        end_time: datetime | None,
        incremental: bool,
    ) -> tuple[datetime | None, datetime | None]:
        """Determine the time range for data fetching.

        Args:
            symbol: Symbol being updated
            start_time: Requested start time
            end_time: Requested end time
            incremental: Whether to do incremental update

        Returns:
            Tuple of (start_time, end_time), or (None, None) if no update needed
        """
        # Determine start time
        if incremental and start_time is None:
            # Get latest existing timestamp
            latest_ts = self.storage.get_latest_timestamp(symbol, self.provider_name)

            if latest_ts:
                # Start from after latest, with safety margin
                start_time = latest_ts - self.safety_margin + timedelta(minutes=1)

                self.logger.info(
                    "Incremental update from latest",
                    symbol=symbol,
                    latest=latest_ts,
                    start=start_time,
                )
            else:
                # No existing data - use subclass default
                start_time = self._get_default_start_time(symbol)
                self.logger.info(
                    "No existing data, starting from default",
                    symbol=symbol,
                    start=start_time,
                )

        # Determine end time
        if end_time is None:
            end_time = datetime.now().replace(microsecond=0)

        # Validate range
        if start_time >= end_time:
            # Already up to date
            return None, None

        self.logger.info(
            "Time range determined",
            symbol=symbol,
            start=start_time,
            end=end_time,
            duration_hours=((end_time - start_time).total_seconds() / 3600),
        )

        return start_time, end_time

    def _save_data(
        self,
        data: pl.DataFrame,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Save data using storage backend.

        Args:
            data: Transformed data to save
            symbol: Symbol being updated
            start_time: Start time for chunk file naming
            end_time: End time for chunk file naming

        Returns:
            Number of records added to combined file
        """
        if len(data) == 0:
            return 0

        # Save as chunk file for audit trail
        chunk_file = self.storage.save_chunk(data, symbol, self.provider_name, start_time, end_time)

        # Update combined file
        records_added = self.storage.update_combined_file(data, symbol, self.provider_name)

        # Update metadata
        self.storage.update_metadata(
            symbol,
            self.provider_name,
            end_time,
            records_added,
            chunk_file.name,
        )

        return int(records_added)

    def _initialize_stats(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Initialize statistics dictionary.

        Args:
            symbol: Symbol being updated
            **kwargs: Additional arguments

        Returns:
            Initial statistics dictionary
        """
        return {
            "symbol": symbol,
            "provider": self.provider_name,
            "records_fetched": 0,
            "records_added": 0,
            "start_time": None,
            "end_time": None,
            "chunk_file": None,
            "errors": [],
            "success": False,
        }

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def _fetch_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch raw data from the data source.

        Args:
            symbol: Symbol to fetch
            start_time: Start time
            end_time: End time
            **kwargs: Additional arguments

        Returns:
            Raw data DataFrame
        """

    @abstractmethod
    def _transform_data(
        self,
        data: pl.DataFrame,
        symbol: str,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Transform raw data to standardized format.

        Args:
            data: Raw data DataFrame
            symbol: Symbol being processed
            **kwargs: Additional arguments

        Returns:
            Transformed DataFrame with standard schema:
                - timestamp: datetime
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: float
        """

    @abstractmethod
    def _get_default_start_time(self, symbol: str) -> datetime:
        """Get default start time for symbol when no data exists.

        Args:
            symbol: Symbol to get default start time for

        Returns:
            Default start datetime
        """

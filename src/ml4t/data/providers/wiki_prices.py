"""Wiki Prices data provider - local Parquet archive (1962-2018)."""

from __future__ import annotations

import os
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError, DataValidationError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class WikiPricesProvider(BaseProvider):
    """
    Provider for Quandl WIKI Prices dataset (local Parquet archive).

    Dataset: Community-curated US equities end-of-day prices
    Date Range: 1962-01-02 to 2018-03-27 (DEPRECATED - no updates)
    Coverage: 3,199 US companies
    Size: 632MB Parquet, 15.4M rows

    ## Why Use Wiki Prices?

    **Best For**:
    - Historical backtesting (pre-2018)
    - Long-term market studies (56 years)
    - Academic research requiring survivorship-bias-free data
    - Teaching quantitative finance (free, offline, comprehensive)

    **Not Suitable For**:
    - Live trading or recent analysis (7-year data gap)
    - Post-2018 strategies validation
    - Current market conditions study

    ## Dataset Background

    Wiki Prices was deprecated by Quandl (now NASDAQ Data Link) in April 2018
    when primary data sources became unavailable. While no longer updated, it
    remains valuable for historical research due to:

    - 56 years of high-quality data (1962-2018)
    - Survivorship-bias free (includes delisted companies)
    - Adjusted prices (splits/dividends handled)
    - Institutional-grade curation
    - Public domain (no usage restrictions)

    See: docs/wiki_prices_status.md for comprehensive analysis

    ## Usage

    ```python
    from ml4t.data.providers.wiki_prices import WikiPricesProvider

    # Default location (auto-detect)
    provider = WikiPricesProvider()

    # Custom Parquet location
    provider = WikiPricesProvider(parquet_path="/path/to/wiki_prices.parquet")

    # Fetch historical data (pre-2018 only)
    aapl = provider.fetch_ohlcv("AAPL", "2010-01-01", "2018-03-27")

    # Use in multi-provider fallback strategy
    def fetch_with_fallback(symbol, start, end):
        wiki_cutoff = "2018-03-27"

        if end <= wiki_cutoff:
            # Pure historical - use Wiki Prices
            return wiki_provider.fetch_ohlcv(symbol, start, end)
        elif start > wiki_cutoff:
            # Pure recent - use Yahoo/EODHD
            return yahoo_provider.fetch_ohlcv(symbol, start, end)
        else:
            # Spans cutoff - combine sources
            wiki_data = wiki_provider.fetch_ohlcv(symbol, start, wiki_cutoff)
            yahoo_data = yahoo_provider.fetch_ohlcv(symbol, "2018-03-28", end)
            return pl.concat([wiki_data, yahoo_data]).sort("timestamp")
    ```

    ## Performance Characteristics

    **Initialization**:
    - Lazy loading: Parquet mapped to memory (no upfront load)
    - First query: ~100-200ms (scan Parquet schema)
    - Subsequent queries: ~5-20ms (memory-mapped access)

    **Memory Usage**:
    - Lazy mode (default): Minimal (<100MB)
    - Eager mode (cache_in_memory=True): ~1.2GB (full dataset in RAM)

    **Query Performance**:
    - Single symbol: 5-20ms (lazy scan)
    - Batch symbols: 100-500ms for 10 stocks (parallel scans)
    - Full dataset scan: ~2-3 seconds (15.4M rows)

    ## Comparison to Alternatives

    | Provider       | Date Range    | Cost  | Speed       | Quality      |
    |----------------|---------------|-------|-------------|--------------|
    | Wiki Prices    | 1962-2018     | Free  | Very Fast   | Excellent    |
    | Yahoo Finance  | 2000-present  | Free  | Fast        | Good         |
    | EODHD          | 1980-present  | €20/mo| Fast        | Excellent    |
    | DataBento      | 2000-present  | $9+/mo| Very Fast   | Professional |

    ## Schema

    Wiki Prices Parquet schema (14 columns):
    ```
    ticker        (String)     - Stock symbol
    date          (Datetime)   - Trading date
    open          (Float64)    - Opening price (unadjusted)
    high          (Float64)    - High price (unadjusted)
    low           (Float64)    - Low price (unadjusted)
    close         (Float64)    - Closing price (unadjusted)
    volume        (Float64)    - Trading volume (unadjusted)
    ex-dividend   (Float64)    - Dividend amount
    split_ratio   (Float64)    - Stock split ratio
    adj_open      (Float64)    - Adjusted opening price
    adj_high      (Float64)    - Adjusted high price
    adj_low       (Float64)    - Adjusted low price
    adj_close     (Float64)    - Adjusted closing price
    adj_volume    (Float64)    - Adjusted volume
    ```

    Transformed to ml4t-data standard schema:
    ```
    timestamp     (Datetime)   - Trading date
    open          (Float64)    - Adjusted opening price
    high          (Float64)    - Adjusted high price
    low           (Float64)    - Adjusted low price
    close         (Float64)    - Adjusted closing price
    volume        (Float64)    - Adjusted volume
    ```

    ## Known Limitations

    - **Frozen Dataset**: No data after March 27, 2018
    - **Daily Only**: No intraday/minute data available
    - **US Equities Only**: No international stocks
    - **End-of-Life Quality**: Some gaps/errors in final months (2018)
    - **No Real-Time**: Static archive, not live API
    """

    # Default locations to search for Wiki Prices Parquet
    DEFAULT_PATHS: ClassVar[list[Path]] = [
        Path("~/ml4t/data/wiki/wiki_prices.parquet").expanduser(),
        Path("~/ml3t/data/equities/nasdaq/wiki_prices.parquet").expanduser(),
        Path("./wiki_prices.parquet"),
    ]

    # Default download location
    DEFAULT_DOWNLOAD_PATH: ClassVar[Path] = Path("~/ml4t/data/wiki").expanduser()

    # NASDAQ Data Link API endpoint for Wiki Prices export
    NASDAQ_EXPORT_URL: ClassVar[str] = "https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES.csv"

    # Dataset metadata
    DATASET_START: ClassVar[str] = "1962-01-02"
    DATASET_END: ClassVar[str] = "2018-03-27"
    DATASET_SYMBOLS: ClassVar[int] = 3199
    DATASET_ROWS: ClassVar[int] = 15_389_314

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        """
        Initialize Wiki Prices provider.

        Args:
            parquet_path: Path to wiki_prices.parquet file (auto-detects if None)
            cache_in_memory: Load full dataset into memory on init (default: False)
                           - False: Lazy loading, minimal memory (~100MB)
                           - True: Eager loading, fast queries (~1.2GB RAM)

        Raises:
            FileNotFoundError: If parquet_path not found and auto-detection fails
        """
        # No rate limiting needed - local file access
        super().__init__(rate_limit=None)

        # Resolve Parquet path
        self.parquet_path = self._resolve_parquet_path(parquet_path)
        self.logger.info(
            "Initialized Wiki Prices provider",
            parquet_path=str(self.parquet_path),
            cache_in_memory=cache_in_memory,
        )

        # Load dataset (lazy or eager)
        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.logger.info("Loading full dataset into memory (~1.2GB)...")
            self._data = pl.read_parquet(self.parquet_path)
            self.logger.info(
                "Dataset loaded",
                rows=len(self._data),
                memory_mb=self._data.estimated_size("mb"),
            )
        else:
            # Lazy loading - just store path
            self._data = None
            self.logger.debug("Using lazy loading (minimal memory footprint)")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "wiki_prices"

    def _resolve_parquet_path(self, provided_path: str | Path | None) -> Path:
        """
        Resolve Wiki Prices Parquet file location.

        Args:
            provided_path: User-provided path (takes precedence)

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If file not found in provided or default locations
        """
        if provided_path:
            path = Path(provided_path).expanduser().resolve()
            if path.exists():
                return path
            raise FileNotFoundError(
                f"Wiki Prices Parquet not found at provided path: {provided_path}"
            )

        # Auto-detect: try default locations
        for path in self.DEFAULT_PATHS:
            resolved = path.expanduser().resolve()
            if resolved.exists():
                self.logger.debug("Auto-detected Wiki Prices Parquet", path=str(resolved))
                return resolved

        # Not found anywhere
        raise FileNotFoundError(
            "Wiki Prices Parquet not found. Tried locations:\n"
            + "\n".join(f"  - {p}" for p in self.DEFAULT_PATHS)
            + "\n\nProvide explicit path: WikiPricesProvider(parquet_path='/path/to/file.parquet')"
        )

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """
        Fetch and transform OHLCV data from Wiki Prices Parquet.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (only "daily" supported)

        Returns:
            Polars DataFrame with standardized schema:
            - timestamp (Datetime): Trading date
            - open (Float64): Adjusted opening price
            - high (Float64): Adjusted high price
            - low (Float64): Adjusted low price
            - close (Float64): Adjusted closing price
            - volume (Float64): Adjusted volume

        Raises:
            ValueError: If frequency is not "daily" or date range invalid
            DataNotAvailableError: If symbol not found or no data in range
            DataValidationError: If Parquet file corrupted or schema mismatch
        """
        # Validate frequency
        if frequency != "daily":
            raise ValueError(
                f"Wiki Prices only supports daily frequency, got '{frequency}'. "
                f"Dataset is end-of-day only (no intraday data)."
            )

        # Warn if date range is outside dataset coverage
        if end < self.DATASET_START or start > self.DATASET_END:
            raise DataNotAvailableError(
                provider="wiki_prices",
                symbol=symbol,
                start=start,
                end=end,
                frequency=frequency,
                details={
                    "reason": f"Date range [{start}, {end}] is completely outside Wiki Prices coverage "
                    f"[{self.DATASET_START}, {self.DATASET_END}]. "
                    f"Dataset was deprecated in 2018 - use Yahoo Finance or EODHD for recent data."
                },
            )

        if start < self.DATASET_START:
            self.logger.warning(
                "Start date before dataset coverage, adjusting",
                requested=start,
                adjusted=self.DATASET_START,
            )
            start = self.DATASET_START

        if end > self.DATASET_END:
            self.logger.warning(
                "End date after dataset coverage, adjusting",
                requested=end,
                adjusted=self.DATASET_END,
            )
            end = self.DATASET_END

        # Load and filter data
        try:
            # Convert string dates to datetime for comparison
            start_dt = pl.lit(start).str.to_datetime()
            end_dt = pl.lit(end).str.to_datetime()

            if self.cache_in_memory and self._data is not None:
                # Query cached in-memory data
                filtered = self._data.filter(
                    (pl.col("ticker") == symbol)
                    & (pl.col("date") >= start_dt)
                    & (pl.col("date") <= end_dt)
                ).sort("date")
            else:
                # Lazy scan Parquet with pushdown filters
                filtered = (
                    pl.scan_parquet(self.parquet_path)
                    .filter(
                        (pl.col("ticker") == symbol)
                        & (pl.col("date") >= start_dt)
                        & (pl.col("date") <= end_dt)
                    )
                    .sort("date")
                    .collect()
                )

        except Exception as e:
            raise DataValidationError(
                provider="wiki_prices",
                message=f"Failed to read Wiki Prices Parquet at {self.parquet_path}: {e}",
            ) from e

        # Check if data found
        if filtered.is_empty():
            raise DataNotAvailableError(
                provider="wiki_prices",
                symbol=symbol,
                start=start,
                end=end,
                frequency=frequency,
                details={
                    "reason": f"No data found for symbol '{symbol}' in date range [{start}, {end}]. "
                    f"Symbol may not exist in Wiki Prices dataset (3,199 US stocks, 1962-2018). "
                    f"Check symbol spelling or try Yahoo Finance for recent/international stocks."
                },
            )

        # Transform to standard schema (use adjusted prices)
        try:
            standardized = filtered.select(
                [
                    pl.col("date").alias("timestamp"),
                    pl.col("adj_open").alias("open"),
                    pl.col("adj_high").alias("high"),
                    pl.col("adj_low").alias("low"),
                    pl.col("adj_close").alias("close"),
                    pl.col("adj_volume").alias("volume"),
                ]
            )
        except Exception as e:
            raise DataValidationError(
                provider="wiki_prices",
                message=f"Schema mismatch in Wiki Prices Parquet. Expected columns: "
                f"[ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume]. "
                f"Error: {e}",
            ) from e

        self.logger.info(
            "Fetched Wiki Prices data",
            symbol=symbol,
            rows=len(standardized),
            date_range=f"{standardized['timestamp'].min()} to {standardized['timestamp'].max()}",
        )

        return standardized

    def list_available_symbols(self) -> list[str]:
        """
        Get list of all symbols in Wiki Prices dataset.

        Returns:
            Sorted list of 3,199 stock symbols

        Example:
            >>> provider = WikiPricesProvider()
            >>> symbols = provider.list_available_symbols()
            >>> len(symbols)
            3199
            >>> "AAPL" in symbols
            True
        """
        if self.cache_in_memory and self._data is not None:
            symbols = self._data["ticker"].unique().sort().to_list()
        else:
            symbols = (
                pl.scan_parquet(self.parquet_path)
                .select("ticker")
                .unique()
                .sort("ticker")
                .collect()["ticker"]
                .to_list()
            )

        self.logger.debug("Listed available symbols", count=len(symbols))
        return symbols

    def get_date_range(self, symbol: str) -> tuple[str, str]:
        """
        Get date range for specific symbol in dataset.

        Args:
            symbol: Stock symbol to query

        Returns:
            Tuple of (start_date, end_date) as YYYY-MM-DD strings

        Raises:
            DataNotAvailableError: If symbol not found

        Example:
            >>> provider = WikiPricesProvider()
            >>> start, end = provider.get_date_range("AAPL")
            >>> print(f"AAPL: {start} to {end}")
            AAPL: 1980-12-12 to 2018-03-27
        """
        if self.cache_in_memory and self._data is not None:
            symbol_data = self._data.filter(pl.col("ticker") == symbol)
        else:
            symbol_data = (
                pl.scan_parquet(self.parquet_path)
                .filter(pl.col("ticker") == symbol)
                .select("date")
                .collect()
            )

        if symbol_data.is_empty():
            raise DataNotAvailableError(
                provider="wiki_prices",
                symbol=symbol,
                details={
                    "reason": f"Symbol '{symbol}' not found in Wiki Prices dataset. "
                    f"Use list_available_symbols() to see available tickers."
                },
            )

        start = symbol_data["date"].min().strftime("%Y-%m-%d")
        end = symbol_data["date"].max().strftime("%Y-%m-%d")

        return start, end

    def get_dataset_stats(self) -> dict[str, any]:
        """
        Get statistics about Wiki Prices dataset.

        Returns:
            Dictionary with dataset metadata:
            - total_rows: Total number of rows
            - total_symbols: Number of unique symbols
            - date_range: Overall date range (start, end)
            - file_size_mb: Parquet file size
            - memory_size_mb: In-memory size (if cached)

        Example:
            >>> provider = WikiPricesProvider()
            >>> stats = provider.get_dataset_stats()
            >>> print(f"Dataset: {stats['total_rows']:,} rows, {stats['total_symbols']} symbols")
            Dataset: 15,389,314 rows, 3,199 symbols
        """
        file_size_mb = self.parquet_path.stat().st_size / (1024 * 1024)

        if self.cache_in_memory and self._data is not None:
            total_rows = len(self._data)
            total_symbols = self._data["ticker"].n_unique()
            start = self._data["date"].min().strftime("%Y-%m-%d")
            end = self._data["date"].max().strftime("%Y-%m-%d")
            memory_size_mb = self._data.estimated_size("mb")
        else:
            # Lazy scan for stats
            df = pl.scan_parquet(self.parquet_path)
            total_rows = df.select(pl.len()).collect().item()
            total_symbols = df.select(pl.col("ticker").n_unique()).collect().item()
            date_stats = df.select(
                [
                    pl.col("date").min().alias("min_date"),
                    pl.col("date").max().alias("max_date"),
                ]
            ).collect()
            start = date_stats["min_date"][0].strftime("%Y-%m-%d")
            end = date_stats["max_date"][0].strftime("%Y-%m-%d")
            memory_size_mb = None

        return {
            "total_rows": total_rows,
            "total_symbols": total_symbols,
            "date_range": (start, end),
            "file_size_mb": round(file_size_mb, 1),
            "memory_size_mb": round(memory_size_mb, 1) if memory_size_mb else None,
            "cached_in_memory": self.cache_in_memory,
        }

    @classmethod
    def download(
        cls,
        output_path: str | Path | None = None,
        api_key: str | None = None,
        env_file: str | Path | None = None,
    ) -> Path:
        """
        Download Wiki Prices dataset from NASDAQ Data Link.

        Downloads the full WIKI/PRICES dataset (~650MB CSV, ~620MB Parquet) and
        saves it as a Parquet file for efficient querying.

        Args:
            output_path: Directory or file path for the downloaded data.
                        If directory, saves as 'wiki_prices.parquet' in that directory.
                        Defaults to ~/ml4t/data/wiki/wiki_prices.parquet
            api_key: NASDAQ Data Link API key. If not provided, looks for:
                    1. QUANDL_API_KEY environment variable
                    2. NASDAQ_DATA_LINK_API_KEY environment variable
                    3. API key in env_file
            env_file: Path to .env file containing API key (default: ~/.env or
                     ~/ml4t/software/data/.env)

        Returns:
            Path to the downloaded Parquet file

        Raises:
            ValueError: If no API key found
            RuntimeError: If download or conversion fails

        Example:
            >>> # Using environment variable
            >>> path = WikiPricesProvider.download()
            >>> print(f"Downloaded to: {path}")

            >>> # Using explicit API key
            >>> path = WikiPricesProvider.download(api_key="your-api-key")

            >>> # Custom output location
            >>> path = WikiPricesProvider.download(output_path="./data/wiki_prices.parquet")
        """
        log = structlog.get_logger()

        # Resolve API key
        resolved_key = cls._resolve_api_key(api_key, env_file)
        if not resolved_key:
            raise ValueError(
                "No NASDAQ Data Link API key found. Provide via:\n"
                "  1. api_key parameter\n"
                "  2. QUANDL_API_KEY environment variable\n"
                "  3. NASDAQ_DATA_LINK_API_KEY environment variable\n"
                "  4. .env file (QUANDL_API_KEY=...)\n\n"
                "Get a free API key at: https://data.nasdaq.com/sign-up"
            )

        # Resolve output path
        if output_path is None:
            output_dir = cls.DEFAULT_DOWNLOAD_PATH
            output_file = output_dir / "wiki_prices.parquet"
        else:
            output_path = Path(output_path).expanduser()
            if output_path.suffix == ".parquet":
                output_file = output_path
                output_dir = output_path.parent
            else:
                output_dir = output_path
                output_file = output_dir / "wiki_prices.parquet"

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Starting Wiki Prices download",
            output_path=str(output_file),
            api_key_prefix=resolved_key[:8] + "...",
        )

        # Request export from NASDAQ Data Link API
        log.info("Requesting export from NASDAQ Data Link API...")

        with httpx.Client(timeout=300.0) as client:
            # Step 1: Request export (returns CSV with S3 download link)
            params = {
                "api_key": resolved_key,
                "qopts.export": "true",
            }

            try:
                response = client.get(cls.NASDAQ_EXPORT_URL, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise ValueError("Invalid API key. Check your NASDAQ Data Link API key.") from e
                elif e.response.status_code == 429:
                    raise RuntimeError("Rate limited by NASDAQ Data Link. Try again later.") from e
                else:
                    raise RuntimeError(
                        f"Export request failed: {e.response.status_code} - {e.response.text}"
                    ) from e

            # Step 2: Parse export metadata CSV to get download link
            meta_df = pl.read_csv(BytesIO(response.content))
            if meta_df.is_empty() or "file.link" not in meta_df.columns:
                raise RuntimeError(f"Unexpected export response format. Columns: {meta_df.columns}")

            download_url = meta_df["file.link"][0]
            file_status = (
                meta_df["file.status"][0] if "file.status" in meta_df.columns else "unknown"
            )
            log.info(f"Export status: {file_status}")

            # Step 3: Poll until export is ready (if status is 'generating')
            max_wait = 600  # 10 minutes max
            wait_interval = 10  # seconds
            total_wait = 0

            while file_status == "generating" and total_wait < max_wait:
                log.info(f"Export generating, waiting {wait_interval}s... ({total_wait}s elapsed)")
                time.sleep(wait_interval)
                total_wait += wait_interval

                response = client.get(cls.NASDAQ_EXPORT_URL, params=params)
                response.raise_for_status()
                meta_df = pl.read_csv(BytesIO(response.content))
                download_url = meta_df["file.link"][0]
                file_status = (
                    meta_df["file.status"][0] if "file.status" in meta_df.columns else "unknown"
                )

            if file_status == "generating":
                raise RuntimeError(f"Export timed out after {max_wait}s. Try again later.")

            # Step 4: Download the ZIP file from S3
            log.info("Downloading ZIP file from S3 (this may take a few minutes)...")
            zip_response = client.get(download_url, timeout=600.0)
            zip_response.raise_for_status()

            log.info(f"Downloaded {len(zip_response.content) / 1024 / 1024:.1f} MB")

            # Step 5: Extract CSV from ZIP
            log.info("Extracting CSV from ZIP...")
            with zipfile.ZipFile(BytesIO(zip_response.content)) as zf:
                csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_files:
                    raise RuntimeError("No CSV file found in ZIP export")
                csv_name = csv_files[0]
                log.info(f"Extracting {csv_name}...")
                with zf.open(csv_name) as f:
                    df = pl.read_csv(f.read())

        if df.is_empty():
            raise RuntimeError("No data received from NASDAQ Data Link API")

        log.info(f"Total rows downloaded: {len(df):,}")

        # Standardize column names (lowercase, no spaces)
        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        # Ensure date column is datetime
        if "date" in df.columns and df["date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.to_datetime())

        # Sort by ticker and date
        df = df.sort(["ticker", "date"])

        # Save as Parquet
        log.info(f"Saving to {output_file}...")
        df.write_parquet(output_file)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        log.info(
            "Download complete",
            output_path=str(output_file),
            rows=len(df),
            tickers=df["ticker"].n_unique(),
            file_size_mb=round(file_size_mb, 1),
        )

        return output_file

    @classmethod
    def _resolve_api_key(cls, api_key: str | None, env_file: str | Path | None) -> str | None:
        """Resolve API key from various sources."""
        # 1. Explicit parameter
        if api_key:
            return api_key

        # 2. Environment variables
        for env_var in ["QUANDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY"]:
            key = os.environ.get(env_var)
            if key:
                return key

        # 3. Load from .env file
        env_paths = []
        if env_file:
            env_paths.append(Path(env_file).expanduser())
        env_paths.extend(
            [
                Path("~/ml4t/software/data/.env").expanduser(),
                Path("~/.env").expanduser(),
                Path(".env"),
            ]
        )

        for env_path in env_paths:
            if env_path.exists():
                try:
                    from dotenv import dotenv_values

                    values = dotenv_values(env_path)
                    for env_var in ["QUANDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY"]:
                        if env_var in values and values[env_var]:
                            return values[env_var]
                except ImportError:
                    # python-dotenv not installed, try manual parsing
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#") or "=" not in line:
                                continue
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key in ["QUANDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY"]:
                                return value

        return None

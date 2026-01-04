"""ML4T Macro Data Downloader for Book Readers.

This module provides a unified interface for downloading and managing
macroeconomic data from FRED for regime filtering and analysis.

Features:
- Download Treasury yield data for regime filtering
- Compute derived series (yield curve slope)
- Config-driven approach for reproducibility

Usage:
    from ml4t.data.macro.downloader import MacroDataManager

    # Initialize with config
    manager = MacroDataManager.from_config("configs/ml4t_etfs.yaml")

    # Download Treasury yields
    manager.download_treasury_yields()

    # Load data
    yields = manager.load_treasury_yields()
    slope = yields["YIELD_CURVE_SLOPE"]

References:
    - FRED: https://fred.stlouisfed.org
    - FRED API: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class MacroConfig:
    """Configuration for macro data download."""

    provider: str = "fred"
    start: str = "2000-01-01"
    end: str = "2025-12-31"
    storage_path: Path = field(default_factory=lambda: Path.home() / "ml4t-data" / "macro")
    series: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.storage_path = Path(self.storage_path).expanduser()

    def get_treasury_symbols(self) -> list[str]:
        """Get list of Treasury yield series IDs."""
        treasury = self.series.get("treasury_yields", {})
        return treasury.get("symbols", ["DGS2", "DGS5", "DGS10", "DGS30"])

    def get_derived_series(self) -> list[dict[str, str]]:
        """Get list of derived series definitions."""
        return self.series.get("derived", [])


class MacroDataManager:
    """Manages macro/economic data download and storage for ML4T book.

    This class provides a simple interface for book readers to:
    1. Download Treasury yield data from FRED
    2. Compute derived series (yield curve slope)
    3. Load data for analysis

    Data is stored as:
        {storage_path}/treasury_yields.parquet
    """

    def __init__(self, config: MacroConfig):
        """Initialize the macro data manager.

        Args:
            config: Configuration object with series and storage path
        """
        self.config = config
        self._provider = None

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> MacroDataManager:
        """Create manager from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized MacroDataManager
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        macro_config = raw_config.get("macro", {})

        config = MacroConfig(
            provider=macro_config.get("provider", "fred"),
            start=macro_config.get("start", "2000-01-01"),
            end=macro_config.get("end", "2025-12-31"),
            storage_path=Path(macro_config.get("storage_path", "~/ml4t-data/macro")).expanduser(),
            series=macro_config.get("series", {}),
        )

        return cls(config)

    def _get_provider(self):
        """Lazily initialize FRED provider."""
        if self._provider is None:
            from ml4t.data.providers.fred import FREDProvider

            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                logger.warning(
                    "FRED_API_KEY not set. Using yfinance fallback for Treasury yields. "
                    "For full FRED access, get free API key at: "
                    "https://fred.stlouisfed.org/docs/api/api_key.html"
                )
                return None

            self._provider = FREDProvider(api_key=api_key)

        return self._provider

    def download_treasury_yields(self) -> pl.DataFrame:
        """Download Treasury yield data.

        Uses FRED API if FRED_API_KEY is set, otherwise falls back to
        yfinance Treasury yield proxies.

        Returns:
            DataFrame with Treasury yield data
        """
        symbols = self.config.get_treasury_symbols()

        logger.info(
            "Downloading Treasury yields",
            symbols=symbols,
            start=self.config.start,
            end=self.config.end,
        )

        provider = self._get_provider()

        if provider is not None:
            # Use FRED provider
            df = self._download_from_fred(provider, symbols)
        else:
            # Fall back to yfinance
            df = self._download_from_yfinance(symbols)

        if df.is_empty():
            logger.warning("No Treasury yield data downloaded")
            return df

        # Compute derived series
        df = self._compute_derived_series(df)

        # Forward fill missing values (yields have gaps on holidays)
        df = df.fill_null(strategy="forward")

        # Save
        self._save_treasury_yields(df)

        logger.info(
            "Download complete",
            rows=len(df),
            columns=df.columns,
        )

        return df

    def _download_from_fred(self, provider, symbols: list[str]) -> pl.DataFrame:
        """Download Treasury yields from FRED API.

        Args:
            provider: FREDProvider instance
            symbols: List of FRED series IDs

        Returns:
            DataFrame with yields
        """
        try:
            df = provider.fetch_multiple(
                series_ids=symbols,
                start=self.config.start,
                end=self.config.end,
                frequency="daily",
                forward_fill=True,
            )

            # Rename columns from {series}_close to just {series}
            rename_map = {f"{s.upper()}_close": s.upper() for s in symbols}
            for old_name, new_name in rename_map.items():
                if old_name in df.columns:
                    df = df.rename({old_name: new_name})

            return df

        except Exception as e:
            logger.error(f"FRED download failed: {e}")
            return pl.DataFrame()

    def _download_from_yfinance(self, symbols: list[str]) -> pl.DataFrame:
        """Download Treasury yields using yfinance as fallback.

        Uses Yahoo Finance Treasury yield indices as proxies:
        - ^IRX: 13-week T-bill (proxy for DGS2)
        - ^FVX: 5-year Treasury
        - ^TNX: 10-year Treasury
        - ^TYX: 30-year Treasury

        Args:
            symbols: List of FRED-style series IDs

        Returns:
            DataFrame with yields
        """
        import yfinance as yf

        # Map FRED series to Yahoo Finance tickers
        yf_mapping = {
            "DGS2": "^IRX",  # 13-week T-bill (closest proxy)
            "DGS5": "^FVX",  # 5-year Treasury
            "DGS10": "^TNX",  # 10-year Treasury
            "DGS30": "^TYX",  # 30-year Treasury
        }

        logger.info("Using yfinance fallback for Treasury yields")

        dfs = []
        for series_id in symbols:
            if series_id not in yf_mapping:
                logger.warning(f"No yfinance equivalent for {series_id}")
                continue

            ticker = yf_mapping[series_id]
            try:
                df = yf.download(
                    ticker,
                    start=self.config.start,
                    end=self.config.end,
                    progress=False,
                    auto_adjust=True,
                )

                if df.empty:
                    logger.warning(f"No data for {series_id} ({ticker})")
                    continue

                # Handle MultiIndex columns from yfinance
                if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)

                # Extract Close as the yield value
                df = df.reset_index()
                series = pl.DataFrame(
                    {
                        "timestamp": df["Date"].values,
                        series_id: df["Close"].values,
                    }
                )
                dfs.append(series)
                logger.debug(f"Downloaded {series_id} ({ticker}): {len(series)} rows")

            except Exception as e:
                logger.warning(f"Failed to download {series_id}: {e}")

        if not dfs:
            return pl.DataFrame()

        # Join all series on timestamp
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on="timestamp", how="outer_coalesce")

        return result.sort("timestamp")

    def _compute_derived_series(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute derived series like yield curve slope.

        Args:
            df: DataFrame with Treasury yields

        Returns:
            DataFrame with derived columns added
        """
        derived = self.config.get_derived_series()

        for series_def in derived:
            name = series_def.get("name")
            formula = series_def.get("formula")

            if not name or not formula:
                continue

            try:
                # Simple formula parsing for "A - B" patterns
                if "-" in formula:
                    parts = formula.split("-")
                    col1 = parts[0].strip()
                    col2 = parts[1].strip()

                    if col1 in df.columns and col2 in df.columns:
                        df = df.with_columns((pl.col(col1) - pl.col(col2)).alias(name))
                        logger.debug(f"Computed {name} = {formula}")
                    else:
                        logger.warning(f"Missing columns for {name}: need {col1} and {col2}")

            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")

        # Also compute standard derived series if columns exist
        if "DGS10" in df.columns and "DGS2" in df.columns:
            if "YIELD_CURVE_SLOPE" not in df.columns:
                df = df.with_columns((pl.col("DGS10") - pl.col("DGS2")).alias("YIELD_CURVE_SLOPE"))
                logger.debug("Computed YIELD_CURVE_SLOPE (DGS10 - DGS2)")

        if "DGS10" in df.columns and "DGS5" in df.columns:
            if "YIELD_CURVE_5_10" not in df.columns:
                df = df.with_columns((pl.col("DGS10") - pl.col("DGS5")).alias("YIELD_CURVE_5_10"))
                logger.debug("Computed YIELD_CURVE_5_10 (DGS10 - DGS5)")

        return df

    def _save_treasury_yields(self, df: pl.DataFrame) -> None:
        """Save Treasury yields to parquet file."""
        output_file = self.config.storage_path / "treasury_yields.parquet"
        df.write_parquet(output_file)
        logger.info(f"Saved: {output_file}")

    def load_treasury_yields(self) -> pl.DataFrame:
        """Load Treasury yield data.

        Returns:
            DataFrame with Treasury yields and derived series
        """
        data_file = self.config.storage_path / "treasury_yields.parquet"

        if not data_file.exists():
            logger.warning("No Treasury yield data found. Run download_treasury_yields() first.")
            return pl.DataFrame()

        df = pl.read_parquet(data_file)

        # Handle column naming (date vs timestamp)
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename({"date": "timestamp"})

        return df.sort("timestamp")

    def get_yield_curve_slope(self) -> pl.DataFrame:
        """Get yield curve slope time series.

        The yield curve slope (10Y - 2Y) is a key regime indicator:
        - Slope > 0.5%: Risk-on environment
        - Slope < 0.5%: Risk-off environment

        Returns:
            DataFrame with timestamp and YIELD_CURVE_SLOPE columns
        """
        df = self.load_treasury_yields()

        if df.is_empty() or "YIELD_CURVE_SLOPE" not in df.columns:
            logger.warning("No yield curve slope data available")
            return pl.DataFrame()

        return df.select(["timestamp", "YIELD_CURVE_SLOPE"])

    def get_regime(self, threshold: float = 0.5) -> pl.DataFrame:
        """Get regime classification based on yield curve slope.

        Args:
            threshold: Slope threshold in percentage points (default 0.5%)

        Returns:
            DataFrame with timestamp, slope, and regime columns
        """
        slope_df = self.get_yield_curve_slope()

        if slope_df.is_empty():
            return pl.DataFrame()

        return slope_df.with_columns(
            pl.when(pl.col("YIELD_CURVE_SLOPE") > threshold)
            .then(pl.lit("risk_on"))
            .otherwise(pl.lit("risk_off"))
            .alias("regime")
        )


def main():
    """CLI entry point for macro data download."""
    import argparse

    parser = argparse.ArgumentParser(description="ML4T Macro Data Downloader")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ml4t_etfs.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = package_dir / args.config

    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        return 1

    # Initialize manager
    manager = MacroDataManager.from_config(config_path)

    # Download Treasury yields
    df = manager.download_treasury_yields()
    print(f"Downloaded Treasury yields: {len(df)} rows")

    # Show recent data
    print("\nRecent values:")
    print(df.tail(5))

    return 0


if __name__ == "__main__":
    exit(main())

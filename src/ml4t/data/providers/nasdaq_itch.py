"""NASDAQ ITCH sample data provider - tick-level market data.

This provider downloads and manages NASDAQ TotalView-ITCH 5.0 sample data files,
enabling tick-level market microstructure analysis.

ITCH 5.0 Protocol: Binary format for NASDAQ order and trade messages
Data Source: https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar
from urllib.parse import urljoin

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError

logger = structlog.get_logger()


class ITCHSampleProvider:
    """
    Provider for NASDAQ TotalView-ITCH 5.0 sample data.

    Downloads ITCH binary files from NASDAQ's public FTP server and provides
    access to parsed message data for market microstructure analysis.

    ## Why Use ITCH Sample Data?

    **Best For**:
    - Learning market microstructure concepts
    - Testing order book reconstruction algorithms
    - Demonstrating tick-level data analysis
    - Educational purposes (free, official NASDAQ data)

    **Limitations**:
    - Single trading day per file (~5-6GB compressed)
    - Requires parsing (binary format)
    - No guaranteed long-term availability

    ## Available Files (as of 2024)

    | File | Date | Compressed Size |
    |------|------|-----------------|
    | 01302019.NASDAQ_ITCH50.gz | Jan 30, 2019 | 4.76 GB |
    | 01302020.NASDAQ_ITCH50.gz | Jan 30, 2020 | 5.60 GB |
    | 03272019.NASDAQ_ITCH50.gz | Mar 27, 2019 | 5.51 GB |
    | 07302019.NASDAQ_ITCH50.gz | Jul 30, 2019 | 3.66 GB |
    | 08302019.NASDAQ_ITCH50.gz | Aug 30, 2019 | 4.08 GB |
    | 10302019.NASDAQ_ITCH50.gz | Oct 30, 2019 | 3.87 GB |
    | 12302019.NASDAQ_ITCH50.gz | Dec 30, 2019 | 3.52 GB |

    ## Usage

    ```python
    from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider

    # Download a sample file
    provider = ITCHSampleProvider()
    path = provider.download("01302019")  # Downloads to ~/ml4t/data/itch/

    # List available files on NASDAQ server
    files = provider.list_available_files()

    # Load pre-parsed messages (if using Rust parser)
    trades = provider.load_parsed_messages("P", parsed_dir="/path/to/messages")
    ```

    ## ITCH Message Types

    The ITCH 5.0 protocol defines 20+ message types:

    | Code | Message Type | Description |
    |------|-------------|-------------|
    | S | System Event | Market open/close signals |
    | R | Stock Directory | Stock metadata |
    | H | Stock Trading Action | Halts, pauses |
    | Y | Reg SHO Restriction | Short sale restrictions |
    | A | Add Order (No MPID) | New limit order |
    | F | Add Order (MPID) | New order with market participant |
    | E | Order Executed | Partial/full execution |
    | C | Order Executed with Price | Execution at different price |
    | X | Order Cancel | Order cancellation |
    | D | Order Delete | Order removal |
    | U | Order Replace | Order modification |
    | P | Trade (Non-Cross) | Regular trade |
    | Q | Cross Trade | Opening/closing cross |
    | B | Broken Trade | Trade cancellation |
    | I | NOII | Net Order Imbalance |

    ## Parsing ITCH Data

    The raw binary data requires parsing. Options:

    1. **Rust Parser (Recommended)**: ~5 minutes for full day
       Location: See Chapter 4 notebooks for parsing example

    2. **Python Parser**: ~1 hour for full day
       Use libraries like `itch-parser` or implement custom

    This provider focuses on downloading and loading data;
    parsing is handled separately.
    """

    # NASDAQ ITCH FTP base URL
    BASE_URL: ClassVar[str] = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"

    # Known sample files (filename: expected_size_bytes)
    KNOWN_FILES: ClassVar[dict[str, int]] = {
        "01302019.NASDAQ_ITCH50.gz": 5_112_000_000,  # ~4.76 GB
        "01302020.NASDAQ_ITCH50.gz": 6_012_000_000,  # ~5.60 GB
        "03272019.NASDAQ_ITCH50.gz": 5_916_000_000,  # ~5.51 GB
        "07302019.NASDAQ_ITCH50.gz": 3_929_000_000,  # ~3.66 GB
        "08302019.NASDAQ_ITCH50.gz": 4_380_000_000,  # ~4.08 GB
        "10302019.NASDAQ_ITCH50.gz": 4_155_000_000,  # ~3.87 GB
        "12302019.NASDAQ_ITCH50.gz": 3_780_000_000,  # ~3.52 GB
    }

    # Default locations (configurable via ML4T_DATA_DIR env var or ~/.ml4t/data/)
    DEFAULT_DOWNLOAD_PATH: ClassVar[Path] = (
        Path(os.environ.get("ML4T_DATA_DIR", "~/.ml4t/data")).expanduser() / "equities/nasdaq_itch"
    )
    DEFAULT_PARSED_PATH: ClassVar[Path] = (
        Path(os.environ.get("ML4T_DATA_DIR", "~/.ml4t/data")).expanduser()
        / "equities/nasdaq_itch/messages"
    )

    # Message type descriptions
    MESSAGE_TYPES: ClassVar[dict[str, str]] = {
        "S": "System Event",
        "R": "Stock Directory",
        "H": "Stock Trading Action",
        "Y": "Reg SHO Restriction",
        "L": "Market Participant Position",
        "V": "MWCB Decline Level",
        "W": "MWCB Status",
        "K": "IPO Quoting Period",
        "J": "LULD Auction Collar",
        "h": "Operational Halt",
        "A": "Add Order (No MPID)",
        "F": "Add Order (MPID)",
        "E": "Order Executed",
        "C": "Order Executed with Price",
        "X": "Order Cancel",
        "D": "Order Delete",
        "U": "Order Replace",
        "P": "Trade (Non-Cross)",
        "Q": "Cross Trade",
        "B": "Broken Trade",
        "I": "NOII",
    }

    def __init__(
        self,
        download_path: str | Path | None = None,
        parsed_path: str | Path | None = None,
    ) -> None:
        """
        Initialize ITCH sample provider.

        Args:
            download_path: Directory to store downloaded ITCH files
                          (default: ~/ml4t/data/itch)
            parsed_path: Directory containing pre-parsed message parquets
                        (default: $ML4T_DATA_DIR/equities/nasdaq_itch/messages or ~/.ml4t/data/...)
        """
        self.logger = structlog.get_logger(name=self.__class__.__name__)

        self.download_path = (
            Path(download_path).expanduser() if download_path else self.DEFAULT_DOWNLOAD_PATH
        )
        self.parsed_path = (
            Path(parsed_path).expanduser() if parsed_path else self.DEFAULT_PARSED_PATH
        )

        self.logger.info(
            "Initialized ITCH sample provider",
            download_path=str(self.download_path),
            parsed_path=str(self.parsed_path),
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "nasdaq_itch"

    def list_available_files(self) -> list[dict[str, str]]:
        """
        List known ITCH sample files available for download.

        Returns:
            List of dicts with file info (name, date, size_gb)

        Example:
            >>> provider = ITCHSampleProvider()
            >>> files = provider.list_available_files()
            >>> for f in files:
            ...     print(f"{f['name']}: {f['date']} ({f['size_gb']} GB)")
        """
        files = []
        for filename, size_bytes in self.KNOWN_FILES.items():
            # Parse date from filename (MMDDYYYY format)
            date_str = filename.split(".")[0]
            month = date_str[:2]
            day = date_str[2:4]
            year = date_str[4:8]

            files.append(
                {
                    "name": filename,
                    "date": f"{year}-{month}-{day}",
                    "size_gb": round(size_bytes / 1e9, 2),
                    "url": urljoin(self.BASE_URL, filename),
                }
            )

        return sorted(files, key=lambda x: x["date"])

    def download(
        self,
        date_or_filename: str,
        output_path: str | Path | None = None,
        verify_size: bool = True,
        progress_callback: callable | None = None,
    ) -> Path:
        """
        Download an ITCH sample file from NASDAQ.

        Args:
            date_or_filename: Either a date string (MMDDYYYY) or full filename
                             Examples: "01302019", "01302019.NASDAQ_ITCH50.gz"
            output_path: Custom output path (default: download_path/filename)
            verify_size: Check downloaded file size against expected
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Path to the downloaded file

        Raises:
            RuntimeError: If download fails or file size mismatch

        Example:
            >>> provider = ITCHSampleProvider()
            >>> path = provider.download("01302019")
            >>> print(f"Downloaded to: {path}")
        """
        # Resolve filename
        if date_or_filename.endswith(".gz"):
            filename = date_or_filename
        else:
            filename = f"{date_or_filename}.NASDAQ_ITCH50.gz"

        if filename not in self.KNOWN_FILES:
            available = ", ".join(self.KNOWN_FILES.keys())
            raise ValueError(f"Unknown ITCH file: {filename}\nAvailable files: {available}")

        # Resolve output path
        if output_path:
            out_path = Path(output_path).expanduser()
        else:
            self.download_path.mkdir(parents=True, exist_ok=True)
            out_path = self.download_path / filename

        # Check if already downloaded
        if out_path.exists():
            existing_size = out_path.stat().st_size
            expected_size = self.KNOWN_FILES[filename]
            if abs(existing_size - expected_size) < 1e8:  # Within 100MB
                self.logger.info(
                    "File already exists, skipping download",
                    path=str(out_path),
                    size_gb=round(existing_size / 1e9, 2),
                )
                return out_path
            else:
                self.logger.warning(
                    "Existing file size mismatch, re-downloading",
                    existing=existing_size,
                    expected=expected_size,
                )

        # Download
        url = urljoin(self.BASE_URL, filename)
        self.logger.info(
            "Starting ITCH download",
            url=url,
            expected_size_gb=round(self.KNOWN_FILES[filename] / 1e9, 2),
        )

        try:
            with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(out_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192 * 1024):  # 8MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)

                        # Log progress every ~500MB
                        if downloaded % (500 * 1024 * 1024) < 8192 * 1024:
                            self.logger.info(
                                "Download progress",
                                downloaded_gb=round(downloaded / 1e9, 2),
                                total_gb=round(total_size / 1e9, 2) if total_size else "unknown",
                            )

        except httpx.HTTPError as e:
            # Clean up partial download
            if out_path.exists():
                out_path.unlink()
            raise RuntimeError(f"Download failed: {e}") from e

        # Verify size
        if verify_size:
            actual_size = out_path.stat().st_size
            expected_size = self.KNOWN_FILES[filename]
            if abs(actual_size - expected_size) > 1e8:  # More than 100MB difference
                self.logger.warning(
                    "Downloaded file size differs from expected",
                    actual_gb=round(actual_size / 1e9, 2),
                    expected_gb=round(expected_size / 1e9, 2),
                )

        self.logger.info(
            "Download complete",
            path=str(out_path),
            size_gb=round(out_path.stat().st_size / 1e9, 2),
        )

        return out_path

    def get_local_file(self, date_or_filename: str | None = None) -> Path | None:
        """
        Get path to a local ITCH file if it exists.

        Args:
            date_or_filename: Date string or filename (checks default if None)

        Returns:
            Path to local file, or None if not found

        Example:
            >>> provider = ITCHSampleProvider()
            >>> path = provider.get_local_file("01302019")
            >>> if path:
            ...     print(f"Found: {path}")
        """
        # Check for any file in download directory
        if date_or_filename is None:
            if self.download_path.exists():
                for filename in self.KNOWN_FILES:
                    path = self.download_path / filename
                    if path.exists():
                        return path

            # Check for itch.bin.gz (ml4t-data download format)
            itch_bin = self.download_path / "itch.bin.gz"
            if itch_bin.exists():
                return itch_bin

            return None

        # Check specific file
        if date_or_filename.endswith(".gz"):
            filename = date_or_filename
        else:
            filename = f"{date_or_filename}.NASDAQ_ITCH50.gz"

        path = self.download_path / filename
        if path.exists():
            return path

        return None

    def load_parsed_messages(
        self,
        message_type: str,
        parsed_dir: str | Path | None = None,
        symbol: str | None = None,
    ) -> pl.DataFrame:
        """
        Load pre-parsed ITCH messages from Parquet files.

        Requires the Rust ITCH parser to have run on the raw data first.

        Args:
            message_type: ITCH message type code (e.g., "P" for trades, "A" for adds)
            parsed_dir: Directory containing parsed message parquets
                       (default: parsed_path)
            symbol: Filter to specific stock symbol (optional)

        Returns:
            Polars DataFrame with parsed messages

        Raises:
            DataNotAvailableError: If parsed data not found

        Example:
            >>> provider = ITCHSampleProvider()
            >>> trades = provider.load_parsed_messages("P")  # Non-cross trades
            >>> print(f"Loaded {len(trades)} trades")

            >>> # Filter to specific symbol
            >>> aapl_trades = provider.load_parsed_messages("P", symbol="AAPL")
        """
        msg_dir = Path(parsed_dir or self.parsed_path).expanduser() / message_type

        if not msg_dir.exists():
            raise DataNotAvailableError(
                provider="nasdaq_itch",
                symbol=message_type,
                details={
                    "reason": f"Parsed message directory not found: {msg_dir}",
                    "message_type": message_type,
                    "description": self.MESSAGE_TYPES.get(message_type, "Unknown"),
                    "hint": "Run the Rust ITCH parser first to generate parquet files",
                },
            )

        # Load all parquet parts
        parquet_files = list(msg_dir.glob("*.parquet"))
        if not parquet_files:
            raise DataNotAvailableError(
                provider="nasdaq_itch",
                symbol=message_type,
                details={
                    "reason": f"No parquet files found in {msg_dir}",
                    "hint": "Run the Rust ITCH parser first",
                },
            )

        self.logger.info(
            "Loading parsed ITCH messages",
            message_type=message_type,
            description=self.MESSAGE_TYPES.get(message_type, "Unknown"),
            num_files=len(parquet_files),
        )

        # Use lazy loading and filter
        df = pl.scan_parquet(msg_dir / "*.parquet")

        if symbol:
            # Try common column names for stock symbol
            for col in ["stock", "symbol", "ticker"]:
                if col in df.collect_schema().names():
                    df = df.filter(pl.col(col).str.strip_chars() == symbol)
                    break

        result = df.collect()

        self.logger.info(
            "Loaded ITCH messages",
            message_type=message_type,
            rows=len(result),
            columns=result.columns,
        )

        return result

    def list_parsed_message_types(
        self, parsed_dir: str | Path | None = None
    ) -> list[dict[str, str]]:
        """
        List available parsed message types in the messages directory.

        Args:
            parsed_dir: Directory containing parsed messages

        Returns:
            List of dicts with message type info

        Example:
            >>> provider = ITCHSampleProvider()
            >>> types = provider.list_parsed_message_types()
            >>> for t in types:
            ...     print(f"{t['code']}: {t['description']} ({t['files']} files)")
        """
        msg_root = Path(parsed_dir or self.parsed_path).expanduser()

        if not msg_root.exists():
            return []

        result = []
        for subdir in sorted(msg_root.iterdir()):
            if subdir.is_dir() and len(subdir.name) == 1:
                code = subdir.name
                parquet_files = list(subdir.glob("*.parquet"))
                if parquet_files:
                    result.append(
                        {
                            "code": code,
                            "description": self.MESSAGE_TYPES.get(code, "Unknown"),
                            "files": len(parquet_files),
                            "path": str(subdir),
                        }
                    )

        return result

    def get_dataset_info(self) -> dict[str, any]:
        """
        Get information about local ITCH data.

        Returns:
            Dict with local file and parsed data info
        """
        info = {
            "download_path": str(self.download_path),
            "parsed_path": str(self.parsed_path),
            "local_files": [],
            "parsed_message_types": [],
        }

        # Check downloaded files
        if self.download_path.exists():
            for filename in self.KNOWN_FILES:
                path = self.download_path / filename
                if path.exists():
                    info["local_files"].append(
                        {
                            "name": filename,
                            "size_gb": round(path.stat().st_size / 1e9, 2),
                            "path": str(path),
                        }
                    )

        # Check for itch.bin.gz (ml4t-data download format)
        itch_bin = self.download_path / "itch.bin.gz"
        if itch_bin.exists() and str(itch_bin) not in [f["path"] for f in info["local_files"]]:
            info["local_files"].append(
                {
                    "name": "itch.bin.gz",
                    "size_gb": round(itch_bin.stat().st_size / 1e9, 2),
                    "path": str(itch_bin),
                }
            )

        # Check parsed messages
        info["parsed_message_types"] = self.list_parsed_message_types()

        return info

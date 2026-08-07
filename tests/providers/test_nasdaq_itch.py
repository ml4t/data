"""Tests for NASDAQ ITCH sample data provider."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider


class TestITCHSampleProviderInit:
    """Tests for provider initialization."""

    def test_default_paths(self):
        """Test default download and parsed paths."""
        provider = ITCHSampleProvider()

        assert provider.download_path == ITCHSampleProvider.default_download_path()
        assert provider.parsed_path == ITCHSampleProvider.default_parsed_path()

    def test_custom_download_path(self, tmp_path):
        """Test custom download path."""
        provider = ITCHSampleProvider(download_path=tmp_path)
        assert provider.download_path == tmp_path

    def test_custom_parsed_path(self, tmp_path):
        """Test custom parsed path."""
        provider = ITCHSampleProvider(parsed_path=tmp_path)
        assert provider.parsed_path == tmp_path

    def test_name_property(self):
        """Test provider name."""
        provider = ITCHSampleProvider()
        assert provider.name == "nasdaq_itch"


class TestListAvailableFiles:
    """Tests for list_available_files method."""

    def test_returns_list(self):
        """Should return a list of file info dicts."""
        provider = ITCHSampleProvider()
        files = provider.list_available_files()

        assert isinstance(files, list)
        assert len(files) > 0

    def test_file_info_structure(self):
        """Each file info should have required fields."""
        provider = ITCHSampleProvider()
        files = provider.list_available_files()

        for file_info in files:
            assert "name" in file_info
            assert "date" in file_info
            assert "size_gb" in file_info
            assert "url" in file_info

    def test_files_sorted_by_date(self):
        """Files should be sorted by date."""
        provider = ITCHSampleProvider()
        files = provider.list_available_files()

        dates = [f["date"] for f in files]
        assert dates == sorted(dates)

    def test_known_files_present(self):
        """Known sample files should be present."""
        provider = ITCHSampleProvider()
        files = provider.list_available_files()
        filenames = [f["name"] for f in files]

        assert "01302019.NASDAQ_ITCH50.gz" in filenames
        assert "01302020.NASDAQ_ITCH50.gz" in filenames

    def test_url_format(self):
        """URLs should point to NASDAQ ITCH server."""
        provider = ITCHSampleProvider()
        files = provider.list_available_files()

        for file_info in files:
            assert file_info["url"].startswith("https://emi.nasdaq.com/ITCH/")


class TestGetLocalFile:
    """Tests for get_local_file method."""

    def test_no_local_file_specific(self, tmp_path):
        """Return None when specific file doesn't exist."""
        provider = ITCHSampleProvider(download_path=tmp_path)
        # Ask for a specific file that doesn't exist in tmp_path
        result = provider.get_local_file("07302019")
        assert result is None

    def test_finds_local_file(self, tmp_path):
        """Find existing local file."""
        # Create a known file
        itch_file = tmp_path / "01302019.NASDAQ_ITCH50.gz"
        itch_file.touch()

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.get_local_file()

        assert result is not None
        assert result == itch_file

    def test_finds_specific_file(self, tmp_path):
        """Find specific file by date."""
        itch_file = tmp_path / "01302020.NASDAQ_ITCH50.gz"
        itch_file.touch()

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.get_local_file("01302020")

        assert result == itch_file

    def test_finds_specific_file_full_name(self, tmp_path):
        """Find specific file by full filename."""
        itch_file = tmp_path / "01302020.NASDAQ_ITCH50.gz"
        itch_file.touch()

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.get_local_file("01302020.NASDAQ_ITCH50.gz")

        assert result == itch_file

    def test_returns_none_for_missing_file(self, tmp_path):
        """Return None when specific file doesn't exist."""
        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.get_local_file("01302019")
        assert result is None


class TestDownload:
    """Tests for download method."""

    def test_unknown_file_raises_error(self, tmp_path):
        """Raise error for unknown file."""
        provider = ITCHSampleProvider(download_path=tmp_path)

        with pytest.raises(ValueError, match="Unknown ITCH file"):
            provider.download("99999999")

    def test_skip_existing_file(self, monkeypatch, tmp_path):
        """Skip download if file already exists with correct size."""
        itch_file = tmp_path / "01302019.NASDAQ_ITCH50.gz"
        monkeypatch.setitem(ITCHSampleProvider.KNOWN_FILES, itch_file.name, 4)
        itch_file.write_bytes(b"data")

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.download("01302019")

        assert result == itch_file
        assert itch_file.read_bytes() == b"data"

    def test_custom_output_path(self, monkeypatch, tmp_path):
        """Allow custom output path."""
        custom_path = tmp_path / "custom" / "output.gz"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(ITCHSampleProvider.KNOWN_FILES, "01302019.NASDAQ_ITCH50.gz", 4)
        custom_path.write_bytes(b"data")

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.download("01302019", output_path=custom_path)

        assert result == custom_path

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_existing_file_requires_exact_size(self, mock_stream, monkeypatch, tmp_path):
        """Replace a truncated destination even when its size is close to the expected size."""
        filename = "01302019.NASDAQ_ITCH50.gz"
        monkeypatch.setitem(ITCHSampleProvider.KNOWN_FILES, filename, 4)
        destination = tmp_path / filename
        destination.write_bytes(b"old")

        response = MagicMock(status_code=200)
        response.headers = {"content-length": "4"}
        response.iter_bytes.return_value = [b"data"]
        mock_stream.return_value.__enter__.return_value = response

        provider = ITCHSampleProvider(download_path=tmp_path)
        result = provider.download("01302019")

        assert result == destination
        assert destination.read_bytes() == b"data"

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_download_creates_directory(self, mock_stream, tmp_path):
        """Download should create directory if needed."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_bytes.return_value = [b"x" * 1000]
        mock_stream.return_value.__enter__.return_value = mock_response

        download_path = tmp_path / "new_dir"
        provider = ITCHSampleProvider(download_path=download_path)

        # This would fail without directory creation
        try:
            provider.download("01302019", verify_size=False)
        except Exception:
            pass  # Expected to fail due to mock limitations

        assert download_path.exists()

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_download_cleans_up_on_error(self, mock_stream, tmp_path):
        """Download should clean up partial file on HTTP error."""
        import httpx

        # Setup mock to raise error
        mock_stream.return_value.__enter__.side_effect = httpx.HTTPError("Test error")

        provider = ITCHSampleProvider(download_path=tmp_path)
        expected_path = tmp_path / "01302019.NASDAQ_ITCH50.gz"

        with pytest.raises(RuntimeError, match="Download failed"):
            provider.download("01302019")

        # Partial file should be cleaned up
        assert not expected_path.exists()

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_download_uses_bounded_network_timeouts(self, mock_stream, tmp_path):
        """Every network phase has a finite timeout."""
        mock_response = MagicMock(status_code=200)
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_bytes.return_value = [b"data"]
        mock_stream.return_value.__enter__.return_value = mock_response

        provider = ITCHSampleProvider(download_path=tmp_path)
        provider.download("01302019", verify_size=False)

        timeout = mock_stream.call_args.kwargs["timeout"]
        assert timeout is not None
        assert timeout.connect == provider.CONNECT_TIMEOUT
        assert timeout.read == provider.READ_TIMEOUT
        assert timeout.write == provider.WRITE_TIMEOUT
        assert timeout.pool == provider.POOL_TIMEOUT

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_download_enforces_overall_elapsed_time(self, mock_stream, tmp_path):
        """A slow but active stream cannot exceed the total operation bound."""
        mock_response = MagicMock(status_code=200)
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_bytes.return_value = [b"data"]
        mock_stream.return_value.__enter__.return_value = mock_response

        provider = ITCHSampleProvider(download_path=tmp_path)
        with (
            patch(
                "ml4t.data.providers.nasdaq_itch.monotonic",
                side_effect=[0.0, provider.MAX_DOWNLOAD_SECONDS + 1],
            ),
            pytest.raises(RuntimeError, match="download exceeded"),
        ):
            provider.download("01302019", verify_size=False)

    @patch("ml4t.data.providers.nasdaq_itch.httpx.stream")
    def test_failed_download_preserves_destination_and_resumes_partial(self, mock_stream, tmp_path):
        """A failed stream preserves existing data and its safe resume checkpoint."""
        import httpx

        destination = tmp_path / "01302019.NASDAQ_ITCH50.gz"
        destination.write_bytes(b"original")
        partial = destination.with_name(f"{destination.name}.part")

        def interrupted_chunks():
            yield b"new-"
            raise httpx.ReadTimeout("stalled")

        failed_response = MagicMock(status_code=200)
        failed_response.headers = {"content-length": "8"}
        failed_response.iter_bytes.return_value = interrupted_chunks()

        resumed_response = MagicMock(status_code=206)
        resumed_response.headers = {
            "content-length": "4",
            "content-range": "bytes 4-7/8",
        }
        resumed_response.iter_bytes.return_value = [b"data"]
        mock_stream.return_value.__enter__.return_value = failed_response

        provider = ITCHSampleProvider(download_path=tmp_path)
        with pytest.raises(RuntimeError, match="Download failed"):
            provider.download("01302019", verify_size=False)

        assert destination.read_bytes() == b"original"
        assert partial.read_bytes() == b"new-"

        mock_stream.return_value.__enter__.return_value = resumed_response
        result = provider.download("01302019", verify_size=False)

        assert result == destination
        assert destination.read_bytes() == b"new-data"
        assert not partial.exists()
        assert mock_stream.call_args.kwargs["headers"] == {"Range": "bytes=4-"}


class TestLoadParsedMessages:
    """Tests for load_parsed_messages method."""

    def test_missing_directory_raises_error(self, tmp_path):
        """Raise error when parsed directory doesn't exist."""
        provider = ITCHSampleProvider(parsed_path=tmp_path)

        with pytest.raises(DataNotAvailableError) as exc_info:
            provider.load_parsed_messages("P")

        assert exc_info.value.provider == "nasdaq_itch"

    def test_empty_directory_raises_error(self, tmp_path):
        """Raise error when no parquet files in directory."""
        # Create empty message type directory
        msg_dir = tmp_path / "P"
        msg_dir.mkdir()

        provider = ITCHSampleProvider(parsed_path=tmp_path)

        with pytest.raises(DataNotAvailableError) as exc_info:
            provider.load_parsed_messages("P")

        assert "No parquet files" in str(exc_info.value.details)

    def test_load_trade_messages(self, tmp_path):
        """Load trade messages from parquet."""
        # Create mock trade data
        msg_dir = tmp_path / "P"
        msg_dir.mkdir()

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 30, 10, 0, 0)],
                "stock": ["AAPL"],
                "price": [300.50],
                "shares": [100],
            }
        )
        df.write_parquet(msg_dir / "part-0.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.load_parsed_messages("P")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "stock" in result.columns

    def test_filter_by_symbol(self, tmp_path):
        """Filter messages by symbol."""
        msg_dir = tmp_path / "P"
        msg_dir.mkdir()

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 30, 10, 0, 0),
                    datetime(2020, 1, 30, 10, 0, 1),
                ],
                "stock": ["AAPL", "MSFT"],
                "price": [300.50, 200.25],
                "shares": [100, 50],
            }
        )
        df.write_parquet(msg_dir / "part-0.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.load_parsed_messages("P", symbol="AAPL")

        assert result.height == 1
        assert result.row(0, named=True)["stock"] == "AAPL"

    def test_load_multiple_parquet_files(self, tmp_path):
        """Load from multiple parquet files."""
        msg_dir = tmp_path / "A"  # Add Order
        msg_dir.mkdir()

        df1 = pl.DataFrame({"order_ref": [1, 2], "stock": ["AAPL", "AAPL"]})
        df2 = pl.DataFrame({"order_ref": [3, 4], "stock": ["AAPL", "AAPL"]})

        df1.write_parquet(msg_dir / "part-0.parquet")
        df2.write_parquet(msg_dir / "part-1.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.load_parsed_messages("A")

        assert result.height == 4


class TestListParsedMessageTypes:
    """Tests for list_parsed_message_types method."""

    def test_empty_when_no_messages(self, tmp_path):
        """Return empty list when no parsed messages."""
        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.list_parsed_message_types()

        assert result == []

    def test_lists_message_types(self, tmp_path):
        """List available message types."""
        # Create directories for message types
        for msg_type in ["A", "P", "D"]:
            msg_dir = tmp_path / msg_type
            msg_dir.mkdir()
            # Add a parquet file
            pl.DataFrame({"dummy": [1]}).write_parquet(msg_dir / "data.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.list_parsed_message_types()

        assert len(result) == 3
        codes = [r["code"] for r in result]
        assert "A" in codes
        assert "P" in codes
        assert "D" in codes

    def test_includes_description(self, tmp_path):
        """Message type info should include description."""
        msg_dir = tmp_path / "P"
        msg_dir.mkdir()
        pl.DataFrame({"dummy": [1]}).write_parquet(msg_dir / "data.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.list_parsed_message_types()

        p_info = next(r for r in result if r["code"] == "P")
        assert p_info["description"] == "Trade (Non-Cross)"

    def test_skips_non_message_directories(self, tmp_path):
        """Skip directories with non-single-character names."""
        # Create a non-message directory
        (tmp_path / "other").mkdir()
        pl.DataFrame({"dummy": [1]}).write_parquet(tmp_path / "other" / "data.parquet")

        # Create valid message directory
        (tmp_path / "A").mkdir()
        pl.DataFrame({"dummy": [1]}).write_parquet(tmp_path / "A" / "data.parquet")

        provider = ITCHSampleProvider(parsed_path=tmp_path)
        result = provider.list_parsed_message_types()

        assert len(result) == 1
        assert result[0]["code"] == "A"


class TestGetDatasetInfo:
    """Tests for get_dataset_info method."""

    def test_basic_structure(self, tmp_path):
        """Return info with required fields."""
        provider = ITCHSampleProvider(download_path=tmp_path, parsed_path=tmp_path)
        info = provider.get_dataset_info()

        assert "download_path" in info
        assert "parsed_path" in info
        assert "local_files" in info
        assert "parsed_message_types" in info
        # May or may not have files depending on system state (legacy files may exist)
        assert isinstance(info["local_files"], list)
        assert isinstance(info["parsed_message_types"], list)

    def test_includes_local_files(self, tmp_path):
        """Include info about local ITCH files in download path."""
        itch_file = tmp_path / "01302019.NASDAQ_ITCH50.gz"
        itch_file.write_bytes(b"x" * 1000)

        provider = ITCHSampleProvider(download_path=tmp_path, parsed_path=tmp_path)
        info = provider.get_dataset_info()

        # Find our file in the list (may also include legacy files)
        our_file = next(
            (f for f in info["local_files"] if f["name"] == "01302019.NASDAQ_ITCH50.gz"),
            None,
        )
        assert our_file is not None
        assert str(tmp_path) in our_file["path"]

    def test_includes_parsed_types(self, tmp_path):
        """Include info about parsed message types."""
        msg_dir = tmp_path / "P"
        msg_dir.mkdir()
        pl.DataFrame({"dummy": [1]}).write_parquet(msg_dir / "data.parquet")

        provider = ITCHSampleProvider(download_path=tmp_path, parsed_path=tmp_path)
        info = provider.get_dataset_info()

        assert len(info["parsed_message_types"]) == 1


class TestKnownConstants:
    """Tests for provider constants."""

    def test_base_url(self):
        """Base URL should point to NASDAQ ITCH server."""
        assert ITCHSampleProvider.BASE_URL == "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"

    def test_known_files_not_empty(self):
        """Should have known sample files defined."""
        assert len(ITCHSampleProvider.KNOWN_FILES) > 0

    def test_message_types_complete(self):
        """Should have common message types defined."""
        expected_types = ["A", "F", "E", "C", "X", "D", "U", "P", "Q", "S", "R"]
        for msg_type in expected_types:
            assert msg_type in ITCHSampleProvider.MESSAGE_TYPES

    def test_message_type_descriptions(self):
        """Message types should have human-readable descriptions."""
        assert ITCHSampleProvider.MESSAGE_TYPES["P"] == "Trade (Non-Cross)"
        assert ITCHSampleProvider.MESSAGE_TYPES["A"] == "Add Order (No MPID)"
        assert ITCHSampleProvider.MESSAGE_TYPES["D"] == "Order Delete"

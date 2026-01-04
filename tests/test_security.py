"""Tests for security features including path traversal protection."""

import tempfile
from pathlib import Path

import pytest

from ml4t.data.security import PathTraversalError, PathValidator
from ml4t.data.storage.filesystem import FileSystemBackend


class TestPathValidator:
    """Test path validation and sanitization."""

    def test_valid_storage_keys(self):
        """Test that valid storage keys are accepted."""
        valid_keys = [
            "equities/daily/AAPL",
            "crypto/minute/BTC-USD",
            "forex/hourly/EUR_USD",
            "commodities/weekly/GOLD",
        ]

        for key in valid_keys:
            asset_class, frequency, symbol = PathValidator.validate_storage_key(key)
            assert asset_class
            assert frequency
            assert symbol

    def test_path_traversal_patterns(self):
        """Test that path traversal patterns are rejected."""
        dangerous_keys = [
            "../../../etc/passwd",
            "equities/../../../etc/passwd",
            "equities/daily/../../../../../../etc/passwd",
            "equities/./daily/../../../AAPL",
            "equities/daily/AAPL/..",
            "equities/daily/AAPL/../../../",
            ".../equities/daily/AAPL",
            "equities/.../daily/AAPL",
            "equities/daily/.../AAPL",
            "~/equities/daily/AAPL",
            "equities/~/daily/AAPL",
            "equities/daily/~/AAPL",
            "$HOME/equities/daily/AAPL",
            "equities/$HOME/daily/AAPL",
            "equities/daily/$HOME",
        ]

        for key in dangerous_keys:
            with pytest.raises(PathTraversalError):
                PathValidator.validate_storage_key(key)

    def test_url_encoded_attacks(self):
        """Test that URL-encoded path traversal is detected."""
        encoded_attacks = [
            "%2e%2e/equities/daily/AAPL",  # ../
            "equities/%2e%2e/daily/AAPL",  # ../
            "equities/daily/%2e%2e",  # ../
            "equities/daily/AAPL%2f%2e%2e",  # /..
            "%2e%2e%2f%2e%2e%2f%2e%2e",  # ../../../
            "equities/daily/AAPL%00.txt",  # Null byte
            "equities/daily/AAPL%252e%252e",  # Double encoded
        ]

        for key in encoded_attacks:
            with pytest.raises((PathTraversalError, ValueError)):
                PathValidator.validate_storage_key(key)

    def test_invalid_key_formats(self):
        """Test that invalid key formats are rejected."""
        invalid_keys = [
            "",  # Empty
            "equities",  # Missing components
            "equities/daily",  # Missing symbol
            "equities/daily/AAPL/extra",  # Too many components
            "/equities/daily/AAPL",  # Leading slash
            "equities/daily/AAPL/",  # Trailing slash
            "equities//daily/AAPL",  # Double slash
            "equities/daily//AAPL",  # Double slash
            "equities daily AAPL",  # Spaces instead of slashes
            "equities|daily|AAPL",  # Pipe separator
            "equities;daily;AAPL",  # Semicolon separator
            "equities&daily&AAPL",  # Ampersand separator
        ]

        for key in invalid_keys:
            with pytest.raises((ValueError, PathTraversalError)):
                PathValidator.validate_storage_key(key)

    def test_special_characters_rejected(self):
        """Test that special characters are rejected."""
        special_char_keys = [
            "equities/daily/AAPL; rm -rf /",  # Command injection
            "equities/daily/AAPL && echo hacked",  # Command chaining
            "equities/daily/AAPL | cat /etc/passwd",  # Pipe
            "equities/daily/AAPL > /tmp/output",  # Redirect
            "equities/daily/AAPL < /etc/passwd",  # Redirect
            "equities/daily/AAPL`whoami`",  # Command substitution
            "equities/daily/AAPL$(whoami)",  # Command substitution
            "equities/daily/AAPL*",  # Wildcard
            "equities/daily/AAPL?",  # Wildcard
            "equities/daily/[AAPL]",  # Character class
            "equities/daily/{AAPL,GOOGL}",  # Brace expansion
        ]

        for key in special_char_keys:
            with pytest.raises((PathTraversalError, ValueError)):
                PathValidator.validate_storage_key(key)

    def test_control_characters_rejected(self):
        """Test that control characters are rejected."""
        # Test null bytes and other control characters
        control_char_keys = [
            "equities/daily/AAPL\x00",  # Null byte
            "equities/daily/AAPL\n",  # Newline
            "equities/daily/AAPL\r",  # Carriage return
            "equities/daily/AAPL\t",  # Tab
            "equities/daily/AAPL\x1b",  # Escape
        ]

        for key in control_char_keys:
            with pytest.raises((PathTraversalError, ValueError)):
                PathValidator.validate_storage_key(key)

    def test_component_validation(self):
        """Test individual component validation."""
        # Test empty components
        with pytest.raises(ValueError):
            PathValidator._validate_component("", "test")

        # Test components that are too long
        with pytest.raises(ValueError):
            PathValidator._validate_component("a" * 256, "test")

        # Test special directory names
        for special in [".", "..", "~"]:
            with pytest.raises(PathTraversalError):
                PathValidator._validate_component(special, "test")

        # Test path separators in components
        for sep in ["/", "\\"]:
            with pytest.raises(PathTraversalError):
                PathValidator._validate_component(f"test{sep}path", "test")

    def test_sanitize_path(self):
        """Test path sanitization."""
        base = Path("/tmp/test_data")

        # Valid paths should work
        safe_path = PathValidator.sanitize_path(base, "equities/daily")
        # Resolve both paths to handle symlinks (e.g., /tmp -> /private/tmp on macOS)
        assert safe_path.resolve() == (base / "equities" / "daily").resolve()

        # Paths trying to escape should fail
        with pytest.raises(PathTraversalError):
            PathValidator.sanitize_path(base, "../etc/passwd")

        with pytest.raises(PathTraversalError):
            PathValidator.sanitize_path(base, "data/../../etc/passwd")

        with pytest.raises(PathTraversalError):
            PathValidator.sanitize_path(base, "/etc/passwd")

    def test_safe_filename_check(self):
        """Test safe filename validation."""
        # Safe filenames
        assert PathValidator.is_safe_filename("data.parquet")
        assert PathValidator.is_safe_filename("AAPL_2024.json")
        assert PathValidator.is_safe_filename("test-file.txt")

        # Unsafe filenames
        assert not PathValidator.is_safe_filename("data\x00.txt")  # Null byte
        assert not PathValidator.is_safe_filename("../data.txt")  # Path separator
        assert not PathValidator.is_safe_filename("data/file.txt")  # Path separator
        assert not PathValidator.is_safe_filename(".")  # Special name
        assert not PathValidator.is_safe_filename("..")  # Special name
        assert not PathValidator.is_safe_filename("CON")  # Windows reserved
        assert not PathValidator.is_safe_filename("PRN")  # Windows reserved


class TestFileSystemBackendSecurity:
    """Test filesystem backend with security features."""

    def test_storage_rejects_traversal_keys(self):
        """Test that storage operations reject path traversal attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemBackend(data_root=Path(tmpdir))

            # Test exists with dangerous keys
            dangerous_keys = [
                "../../../etc/passwd",
                "equities/../../../etc/passwd",
                "equities/daily/../../etc/passwd",
            ]

            for key in dangerous_keys:
                # exists should return False for invalid keys
                assert not storage.exists(key)

                # read should raise error
                with pytest.raises((PathTraversalError, ValueError)):
                    storage.read(key)

                # delete should raise error
                with pytest.raises((PathTraversalError, ValueError)):
                    storage.delete(key)

    def test_metadata_validation_on_write(self):
        """Test that write operations validate metadata for path traversal."""
        from datetime import datetime

        import polars as pl

        from ml4t.data.core.models import DataObject, Metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemBackend(data_root=Path(tmpdir))

            # Create test data
            df = pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                }
            )

            # Try to write with dangerous metadata
            # Note: frequency is now a property derived from bar_params, not a field
            dangerous_metadata = [
                ("../../../etc", "passwd"),  # Dangerous asset_class
                ("equities", "../../../etc/passwd"),  # Dangerous symbol
                ("equities", "AAPL/../../../etc/passwd"),  # Dangerous symbol with prefix
            ]

            for asset_class, symbol in dangerous_metadata:
                metadata = Metadata(
                    provider="test",
                    symbol=symbol,
                    asset_class=asset_class,
                    bar_type="time",
                    bar_params={"frequency": "daily"},
                    schema_version="1.0",
                )

                data = DataObject(data=df, metadata=metadata)

                with pytest.raises(PathTraversalError):
                    storage.write(data)

    def test_no_escape_from_data_root(self):
        """Test that all operations stay within data_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            data_root.mkdir()

            # Create a file outside data_root
            secret_file = Path(tmpdir) / "secret.txt"
            secret_file.write_text("secret content")

            storage = FileSystemBackend(data_root=data_root)

            # Try various attempts to access the secret file
            attempts = [
                "../secret.txt",
                "equities/daily/../../secret.txt",
                "equities/../secret.txt",
            ]

            for attempt in attempts:
                # Should not be able to check existence
                assert not storage.exists(attempt)

                # Should not be able to read
                with pytest.raises((PathTraversalError, ValueError)):
                    storage.read(attempt)

                # Should not be able to delete
                with pytest.raises((PathTraversalError, ValueError)):
                    storage.delete(attempt)

            # Verify secret file is still there
            assert secret_file.exists()
            assert secret_file.read_text() == "secret content"

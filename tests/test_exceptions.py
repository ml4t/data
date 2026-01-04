"""Test core exception classes."""


class TestCoreExceptions:
    """Test core exception functionality."""

    def test_qldm_error_basic(self):
        """Test basic QldmError functionality."""
        from ml4t.data.core.exceptions import QldmError

        # Test basic exception
        exc = QldmError("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"

        # Test with details
        exc_with_details = QldmError("Test error", details={"key": "value"})
        assert exc_with_details.details == {"key": "value"}

    def test_data_validation_error(self):
        """Test DataValidationError functionality."""
        from ml4t.data.core.exceptions import DataValidationError

        # Test basic validation error
        exc = DataValidationError("symbol", "Validation failed")
        assert "symbol" in str(exc)
        assert "Validation failed" in str(exc)

    def test_provider_error(self):
        """Test ProviderError functionality."""
        from ml4t.data.core.exceptions import ProviderError

        # Test basic provider error
        exc = ProviderError("yahoo", "Connection failed")
        assert exc.provider == "yahoo"
        assert "yahoo" in str(exc)
        assert "Connection failed" in str(exc)

    def test_storage_error(self):
        """Test StorageError functionality."""
        from ml4t.data.core.exceptions import StorageError

        # Test basic storage error
        exc = StorageError("File not found")
        assert "File not found" in str(exc)

    def test_configuration_error(self):
        """Test ConfigurationError functionality."""
        from ml4t.data.core.exceptions import ConfigurationError

        # Test basic config error
        exc = ConfigurationError("Invalid configuration")
        assert "Invalid configuration" in str(exc)

    def test_network_error(self):
        """Test NetworkError functionality."""
        from ml4t.data.core.exceptions import NetworkError

        # Test basic network error
        exc = NetworkError("yahoo", "Connection failed")
        assert exc.provider == "yahoo"
        assert "yahoo" in str(exc)
        assert "Connection failed" in str(exc)

    def test_rate_limit_error(self):
        """Test RateLimitError functionality."""
        from ml4t.data.core.exceptions import RateLimitError

        # Test basic rate limit error
        exc = RateLimitError("yahoo", "Rate limit exceeded")
        assert exc.provider == "yahoo"
        assert "Rate limit exceeded" in str(exc)

    def test_authentication_error(self):
        """Test AuthenticationError functionality."""
        from ml4t.data.core.exceptions import AuthenticationError

        # Test basic auth error
        exc = AuthenticationError("yahoo", "Invalid credentials")
        assert exc.provider == "yahoo"
        assert "Invalid credentials" in str(exc)

    def test_symbol_not_found_error(self):
        """Test SymbolNotFoundError functionality."""
        from ml4t.data.core.exceptions import SymbolNotFoundError

        # Test symbol not found error
        exc = SymbolNotFoundError("yahoo", "INVALID")
        assert exc.provider == "yahoo"
        assert exc.symbol == "INVALID"
        assert "INVALID" in str(exc)

    def test_data_not_available_error(self):
        """Test DataNotAvailableError functionality."""
        from ml4t.data.core.exceptions import DataNotAvailableError

        # Test data not available error
        exc = DataNotAvailableError("yahoo", "No data available")
        assert exc.provider == "yahoo"
        assert "No data available" in str(exc)

    def test_lock_error(self):
        """Test LockError functionality."""
        from ml4t.data.core.exceptions import LockError

        # Test lock error
        exc = LockError("test_key", 5.0)
        assert "Could not acquire lock for key test_key within 5.0 seconds" in str(exc)

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        from ml4t.data.core.exceptions import (
            AuthenticationError,
            ConfigurationError,
            DataValidationError,
            NetworkError,
            ProviderError,
            QldmError,
            RateLimitError,
            StorageError,
        )

        # All should inherit from QldmError
        assert issubclass(DataValidationError, QldmError)
        assert issubclass(ProviderError, QldmError)
        assert issubclass(StorageError, QldmError)
        assert issubclass(ConfigurationError, QldmError)
        assert issubclass(NetworkError, ProviderError)  # NetworkError inherits from ProviderError
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(AuthenticationError, ProviderError)

        # All should inherit from Exception
        assert issubclass(QldmError, Exception)

"""Test retry utility functionality."""

import pytest


class TestRetryUtility:
    """Test retry utility functions."""

    def test_retry_import(self):
        """Test retry utility can be imported."""
        from ml4t.data.utils.retry import with_retry

        assert with_retry is not None

    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        from ml4t.data.utils.retry import with_retry

        @with_retry(max_attempts=3, min_wait=0.1)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_retry_decorator_with_failure(self):
        """Test retry decorator with retryable error."""
        from ml4t.data.utils.retry import with_retry

        call_count = 0

        @with_retry(max_attempts=2, min_wait=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")  # This is retryable
            return "success after retry"

        result = failing_function()
        assert result == "success after retry"
        assert call_count == 2

    def test_retry_decorator_max_attempts(self):
        """Test retry decorator respects max attempts."""
        from tenacity import RetryError

        from ml4t.data.utils.retry import with_retry

        call_count = 0

        @with_retry(max_attempts=2, min_wait=0.01)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")  # This is retryable

        with pytest.raises(RetryError):
            always_failing_function()

        assert call_count == 2

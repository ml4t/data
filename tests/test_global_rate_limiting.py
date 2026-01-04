"""Tests for global rate limiting functionality."""

import threading
import time
from unittest.mock import patch

import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.utils.global_rate_limit import GlobalRateLimitManager, global_rate_limit_manager

# Mark all tests in this module as slow (they use actual sleep/waits)
pytestmark = pytest.mark.slow


class TestGlobalRateLimitManager:
    """Test the global rate limiting manager."""

    def test_singleton_behavior(self):
        """Test that GlobalRateLimitManager is a singleton."""
        manager1 = GlobalRateLimitManager()
        manager2 = GlobalRateLimitManager()
        assert manager1 is manager2
        assert manager1 is global_rate_limit_manager

    def test_rate_limiter_sharing(self):
        """Test that same provider configs share rate limiters."""
        manager = GlobalRateLimitManager()

        # Same config should return same limiter
        limiter1 = manager.get_rate_limiter("test", 10, 60.0)
        limiter2 = manager.get_rate_limiter("test", 10, 60.0)
        assert limiter1 is limiter2

        # Different config should return different limiter
        limiter3 = manager.get_rate_limiter("test", 20, 60.0)
        assert limiter1 is not limiter3

    def test_provider_isolation(self):
        """Test that different providers have separate rate limiters."""
        manager = GlobalRateLimitManager()

        yahoo_limiter = manager.get_rate_limiter("yahoo", 10, 60.0)
        binance_limiter = manager.get_rate_limiter("binance", 10, 60.0)

        assert yahoo_limiter is not binance_limiter

    def test_reset_provider_limits(self):
        """Test resetting rate limits for specific provider."""
        manager = GlobalRateLimitManager()

        # Create limiters and use them
        yahoo_limiter = manager.get_rate_limiter("yahoo", 2, 10.0)
        binance_limiter = manager.get_rate_limiter("binance", 2, 10.0)

        # Use up the limits
        yahoo_limiter.acquire()
        yahoo_limiter.acquire()
        binance_limiter.acquire()
        binance_limiter.acquire()

        # Both should be at limit
        assert not yahoo_limiter.acquire(blocking=False)
        assert not binance_limiter.acquire(blocking=False)

        # Reset only yahoo
        manager.reset_provider_limits("yahoo")

        # Yahoo should be reset, binance should still be limited
        assert yahoo_limiter.acquire(blocking=False)
        assert not binance_limiter.acquire(blocking=False)

    def test_reset_all_limits(self):
        """Test resetting all rate limits."""
        manager = GlobalRateLimitManager()

        # Create limiters and use them
        yahoo_limiter = manager.get_rate_limiter("yahoo", 1, 10.0)
        binance_limiter = manager.get_rate_limiter("binance", 1, 10.0)

        # Use up the limits
        yahoo_limiter.acquire()
        binance_limiter.acquire()

        # Both should be at limit
        assert not yahoo_limiter.acquire(blocking=False)
        assert not binance_limiter.acquire(blocking=False)

        # Reset all
        manager.reset_all_limits()

        # Both should be reset
        assert yahoo_limiter.acquire(blocking=False)
        assert binance_limiter.acquire(blocking=False)

    def test_limiter_status(self):
        """Test getting rate limiter status."""
        manager = GlobalRateLimitManager()

        # Clear any existing state and create fresh manager
        manager.reset_all_limits()
        manager._rate_limiters.clear()

        # Create some limiters with explicit configs
        yahoo_limiter = manager.get_rate_limiter("test_yahoo", 5, 60.0)
        binance_limiter = manager.get_rate_limiter("test_binance", 10, 30.0)

        # Use some calls
        yahoo_limiter.acquire()
        yahoo_limiter.acquire()
        binance_limiter.acquire()

        status = manager.get_limiter_status()

        # Check structure
        assert "test_yahoo" in status
        assert "test_binance" in status

        # Check yahoo status
        yahoo_status = status["test_yahoo"][0]
        assert yahoo_status["max_calls"] == 5
        assert yahoo_status["period"] == 60.0
        assert yahoo_status["current_calls"] == 2

        # Check binance status
        binance_status = status["test_binance"][0]
        assert binance_status["max_calls"] == 10
        assert binance_status["period"] == 30.0
        assert binance_status["current_calls"] == 1


class TestProviderIntegration:
    """Test integration with actual providers."""

    def test_provider_shares_global_limits(self):
        """Test that multiple provider instances share global rate limits."""
        # Create DataManager instances with explicitly configured same rate limits
        config = {"providers": {"yahoo": {"rate_limit": (60, 60.0)}}}

        dm1 = DataManager(providers=config["providers"])
        dm2 = DataManager(providers=config["providers"])

        # Get yahoo providers from both
        provider1 = dm1._get_provider("yahoo")
        provider2 = dm2._get_provider("yahoo")

        # They should share the same rate limiter since they have the same config
        assert provider1.rate_limiter is provider2.rate_limiter

    def test_different_providers_separate_limits(self):
        """Test that different provider types have separate limits."""
        dm = DataManager()

        yahoo_provider = dm._get_provider("yahoo")
        mock_provider = dm._get_provider("mock")

        # They should have different rate limiters
        assert yahoo_provider.rate_limiter is not mock_provider.rate_limiter

    def test_rate_limiting_enforcement(self):
        """Test that rate limiting is actually enforced."""
        # Create a provider with very restrictive rate limit
        with patch.object(DataManager, "_get_provider"):
            dm = DataManager()

            # Mock a provider with 1 call per 5 seconds
            mock_provider = dm._get_provider("mock")
            mock_provider.rate_limiter = global_rate_limit_manager.get_rate_limiter(
                "test_provider", 1, 5.0
            )

            # First call should succeed immediately
            start_time = time.time()
            mock_provider.rate_limiter.acquire()
            first_call_time = time.time() - start_time
            assert first_call_time < 0.1  # Should be immediate

            # Second call should be blocked/delayed
            start_time = time.time()
            result = mock_provider.rate_limiter.acquire(blocking=False)
            assert not result  # Should be blocked

    def test_concurrent_access_thread_safety(self):
        """Test thread safety of global rate limiting."""
        manager = GlobalRateLimitManager()
        limiter = manager.get_rate_limiter("test_concurrent", 5, 1.0)

        results = []

        def worker():
            """Worker function to test concurrent access."""
            result = limiter.acquire(blocking=False)
            results.append(result)

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)

        # Start all threads at roughly the same time
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Should have exactly 5 successful acquisitions
        successful = sum(results)
        assert successful == 5
        assert len(results) == 10


def test_module_import():
    """Test that the module imports correctly."""
    from ml4t.data.utils.global_rate_limit import GlobalRateLimitManager, global_rate_limit_manager

    assert GlobalRateLimitManager is not None
    assert global_rate_limit_manager is not None
    assert isinstance(global_rate_limit_manager, GlobalRateLimitManager)


if __name__ == "__main__":
    # Run basic functionality test
    def test_basic_functionality():
        """Quick test of basic functionality."""
        print("Testing global rate limiting...")

        manager = GlobalRateLimitManager()
        limiter = manager.get_rate_limiter("test", 2, 5.0)

        # Should allow 2 calls
        assert limiter.acquire(blocking=False)
        assert limiter.acquire(blocking=False)

        # Third call should be blocked
        assert not limiter.acquire(blocking=False)

        print("✅ Basic global rate limiting test passed!")

    test_basic_functionality()

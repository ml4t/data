"""Tests for global rate limit manager."""

import threading

from ml4t.data.utils.global_rate_limit import (
    GlobalRateLimitManager,
    global_rate_limit_manager,
)


class TestGlobalRateLimitManagerSingleton:
    """Tests for singleton behavior."""

    def test_singleton_same_instance(self):
        """Test that multiple instantiations return same instance."""
        manager1 = GlobalRateLimitManager()
        manager2 = GlobalRateLimitManager()

        assert manager1 is manager2

    def test_singleton_survives_multiple_calls(self):
        """Test that singleton pattern works across many calls."""
        instances = [GlobalRateLimitManager() for _ in range(10)]

        # All should be the same instance
        for instance in instances:
            assert instance is instances[0]

    def test_global_instance_is_singleton(self):
        """Test that global instance matches singleton."""
        manager = GlobalRateLimitManager()
        assert global_rate_limit_manager is manager


class TestGlobalRateLimitManagerGetRateLimiter:
    """Tests for get_rate_limiter method."""

    def test_creates_new_limiter(self):
        """Test that new limiters are created."""
        manager = GlobalRateLimitManager()

        limiter = manager.get_rate_limiter(
            provider_name="test_provider",
            max_calls=10,
            period=60.0,
        )

        assert limiter is not None
        assert limiter.max_calls == 10
        assert limiter.period == 60.0

    def test_returns_same_limiter_for_same_config(self):
        """Test that same config returns same limiter."""
        manager = GlobalRateLimitManager()

        limiter1 = manager.get_rate_limiter(
            provider_name="test_provider2",
            max_calls=10,
            period=60.0,
        )
        limiter2 = manager.get_rate_limiter(
            provider_name="test_provider2",
            max_calls=10,
            period=60.0,
        )

        assert limiter1 is limiter2

    def test_different_config_returns_different_limiter(self):
        """Test that different config returns different limiter."""
        manager = GlobalRateLimitManager()

        limiter1 = manager.get_rate_limiter(
            provider_name="test_provider3",
            max_calls=10,
            period=60.0,
        )
        limiter2 = manager.get_rate_limiter(
            provider_name="test_provider3",
            max_calls=20,  # Different max_calls
            period=60.0,
        )

        assert limiter1 is not limiter2

    def test_burst_parameter(self):
        """Test limiter creation with burst parameter."""
        manager = GlobalRateLimitManager()

        limiter = manager.get_rate_limiter(
            provider_name="test_provider4",
            max_calls=10,
            period=60.0,
            burst=5,
        )

        assert limiter is not None
        assert limiter.burst == 5

    def test_different_providers_get_different_limiters(self):
        """Test that different providers get different limiters."""
        manager = GlobalRateLimitManager()

        limiter1 = manager.get_rate_limiter(
            provider_name="provider_a",
            max_calls=10,
            period=60.0,
        )
        limiter2 = manager.get_rate_limiter(
            provider_name="provider_b",
            max_calls=10,
            period=60.0,
        )

        assert limiter1 is not limiter2


class TestGlobalRateLimitManagerResetProviderLimits:
    """Tests for reset_provider_limits method."""

    def test_resets_provider_limits(self):
        """Test that provider limits are reset."""
        manager = GlobalRateLimitManager()

        limiter = manager.get_rate_limiter(
            provider_name="reset_test",
            max_calls=10,
            period=60.0,
        )

        # Simulate some usage
        for _ in range(5):
            limiter.acquire()

        # Verify calls were made
        assert len(limiter.calls) > 0

        # Reset and verify
        manager.reset_provider_limits("reset_test")

        # Calls should be cleared
        assert len(limiter.calls) == 0

    def test_only_resets_matching_provider(self):
        """Test that only matching provider is reset."""
        manager = GlobalRateLimitManager()

        limiter1 = manager.get_rate_limiter(
            provider_name="reset_target",
            max_calls=10,
            period=60.0,
        )
        limiter2 = manager.get_rate_limiter(
            provider_name="other_provider",
            max_calls=10,
            period=60.0,
        )

        # Use both limiters
        limiter1.acquire()
        limiter2.acquire()

        # Reset only one
        manager.reset_provider_limits("reset_target")

        # Only target should be reset
        assert len(limiter1.calls) == 0
        assert len(limiter2.calls) > 0


class TestGlobalRateLimitManagerResetAllLimits:
    """Tests for reset_all_limits method."""

    def test_resets_all_limits(self):
        """Test that all limits are reset."""
        manager = GlobalRateLimitManager()

        limiter1 = manager.get_rate_limiter(
            provider_name="all_reset_1",
            max_calls=10,
            period=60.0,
        )
        limiter2 = manager.get_rate_limiter(
            provider_name="all_reset_2",
            max_calls=10,
            period=60.0,
        )

        # Use both
        limiter1.acquire()
        limiter2.acquire()

        # Reset all
        manager.reset_all_limits()

        # Both should be reset
        assert len(limiter1.calls) == 0
        assert len(limiter2.calls) == 0


class TestGlobalRateLimitManagerGetLimiterStatus:
    """Tests for get_limiter_status method."""

    def test_returns_empty_for_no_limiters(self):
        """Test status with no active limiters."""
        # Create fresh manager for this test
        manager = GlobalRateLimitManager()

        # Clear any existing limiters for clean test
        manager._rate_limiters.clear()

        status = manager.get_limiter_status()

        assert status == {}

    def test_returns_status_for_single_limiter(self):
        """Test status with single limiter."""
        manager = GlobalRateLimitManager()

        # Clear and create fresh limiter
        manager._rate_limiters.clear()

        _limiter = manager.get_rate_limiter(  # noqa: F841
            provider_name="status_test",
            max_calls=10,
            period=60.0,
            burst=5,
        )

        status = manager.get_limiter_status()

        assert "status_test" in status
        assert len(status["status_test"]) == 1
        assert status["status_test"][0]["max_calls"] == 10
        assert status["status_test"][0]["period"] == 60.0
        assert status["status_test"][0]["burst"] == 5

    def test_returns_status_for_multiple_limiters(self):
        """Test status with multiple limiters."""
        manager = GlobalRateLimitManager()

        # Clear and create fresh limiters
        manager._rate_limiters.clear()

        manager.get_rate_limiter(
            provider_name="status_multi_1",
            max_calls=10,
            period=60.0,
        )
        manager.get_rate_limiter(
            provider_name="status_multi_2",
            max_calls=20,
            period=120.0,
        )

        status = manager.get_limiter_status()

        assert "status_multi_1" in status
        assert "status_multi_2" in status

    def test_groups_by_provider_name(self):
        """Test that status groups limiters by provider name."""
        manager = GlobalRateLimitManager()

        # Clear and create fresh limiters
        manager._rate_limiters.clear()

        # Create two limiters for same provider with different configs
        manager.get_rate_limiter(
            provider_name="grouped_provider",
            max_calls=10,
            period=60.0,
        )
        manager.get_rate_limiter(
            provider_name="grouped_provider",
            max_calls=20,
            period=120.0,
        )

        status = manager.get_limiter_status()

        assert "grouped_provider" in status
        assert len(status["grouped_provider"]) == 2


class TestGlobalRateLimitManagerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_get_limiter(self):
        """Test concurrent access to get_rate_limiter."""
        manager = GlobalRateLimitManager()
        results = []
        errors = []

        def get_limiter():
            try:
                limiter = manager.get_rate_limiter(
                    provider_name="concurrent_test",
                    max_calls=10,
                    period=60.0,
                )
                results.append(limiter)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_limiter) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All results should be the same limiter instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)


class TestGlobalModuleInstance:
    """Tests for the global module-level instance."""

    def test_global_instance_exists(self):
        """Test that global instance exists."""
        assert global_rate_limit_manager is not None

    def test_global_instance_is_manager(self):
        """Test that global instance is GlobalRateLimitManager."""
        assert isinstance(global_rate_limit_manager, GlobalRateLimitManager)

    def test_global_instance_usable(self):
        """Test that global instance is usable."""
        limiter = global_rate_limit_manager.get_rate_limiter(
            provider_name="global_test",
            max_calls=5,
            period=30.0,
        )

        assert limiter is not None

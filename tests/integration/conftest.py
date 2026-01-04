"""Configuration for integration tests."""

import os
import warnings

import pytest
from structlog import get_logger

logger = get_logger(__name__)


def pytest_configure(config):
    """Configure pytest for integration testing."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may use real APIs)"
    )
    config.addinivalue_line("markers", "expensive: mark test as expensive (high API costs)")
    config.addinivalue_line("markers", "requires_api_key: mark test as requiring specific API key")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    # Check if we're in CI environment
    is_ci = os.getenv("CI", "false").lower() == "true"

    # Check available API keys
    {
        "cryptocompare": bool(os.getenv("CRYPTOCOMPARE_API_KEY")),
        "databento": bool(os.getenv("DATABENTO_API_KEY")),
        "oanda": bool(os.getenv("OANDA_API_KEY")),
    }

    skip_expensive = pytest.mark.skip(reason="Skipping expensive tests in CI")
    pytest.mark.skip(reason="Required API key not available")

    for item in items:
        # Skip expensive tests in CI
        if is_ci and "expensive" in item.keywords:
            item.add_marker(skip_expensive)

        # Log which tests are being run
        if "integration" in item.keywords:
            logger.debug(f"Integration test collected: {item.name}")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for integration tests."""
    # Suppress warnings from API libraries
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*BentoWarning.*")

    # Log test session start
    logger.info("Starting integration test session")

    # Report available API keys (without exposing values)
    available_providers = []
    if os.getenv("CRYPTOCOMPARE_API_KEY"):
        available_providers.append("CryptoCompare")
    if os.getenv("DATABENTO_API_KEY"):
        available_providers.append("Databento")
    if os.getenv("OANDA_API_KEY"):
        available_providers.append("OANDA")

    if available_providers:
        logger.info(f"Available providers for testing: {', '.join(available_providers)}")
    else:
        logger.warning("No API keys found - integration tests will be skipped")

    yield

    # Log test session end
    logger.info("Integration test session completed")


@pytest.fixture(scope="session")
def api_key_manager() -> dict[str, str | None]:
    """Manage API keys for integration tests."""
    keys = {
        "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
        "databento": os.getenv("DATABENTO_API_KEY"),
        "oanda": os.getenv("OANDA_API_KEY"),
    }

    # Validate keys format (basic validation without exposing values)
    for provider, key in keys.items():
        if key and provider == "databento" and not key.startswith("db-"):
            logger.warning(f"Invalid {provider} API key format")
            keys[provider] = None

    return keys


@pytest.fixture
def cost_tracker():
    """Track estimated API costs for tests."""

    class CostTracker:
        def __init__(self):
            self.costs = {
                "cryptocompare": 0.0,  # Free tier
                "databento": 0.0,  # Pay per request
                "oanda": 0.0,  # Free practice account
            }
            self.requests = {
                "cryptocompare": 0,
                "databento": 0,
                "oanda": 0,
            }

        def record_request(self, provider: str, estimated_cost: float = 0.0):
            """Record an API request and its estimated cost."""
            if provider in self.costs:
                self.costs[provider] += estimated_cost
                self.requests[provider] += 1

        def report(self):
            """Generate cost report."""
            total_cost = sum(self.costs.values())
            total_requests = sum(self.requests.values())

            report = "API Usage Report:\n"
            report += f"Total Requests: {total_requests}\n"
            report += f"Estimated Total Cost: ${total_cost:.4f}\n"

            for provider in self.costs:
                if self.requests[provider] > 0:
                    report += f"  {provider}: {self.requests[provider]} requests, ${self.costs[provider]:.4f}\n"

            return report

    tracker = CostTracker()
    yield tracker

    # Report costs at end of test
    if any(tracker.requests.values()):
        logger.info(tracker.report())

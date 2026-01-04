"""ML4T Data - Modern financial data management library with unified provider interface."""

try:
    from ml4t.data._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
__author__ = "ML4T Team"
__email__ = "info@ml4trading.io"

# Contract specifications (always available - no external dependencies)
# Asset classes
from ml4t.data.assets.asset_class import AssetClass
from ml4t.data.assets.contracts import (
    FUTURES_REGISTRY,
    ContractSpec,
    get_contract_spec,
    load_contract_specs,
    register_contract_spec,
)

__all__ = [
    # Contract specifications
    "ContractSpec",
    "FUTURES_REGISTRY",
    "get_contract_spec",
    "load_contract_specs",
    "register_contract_spec",
    "AssetClass",
]

# Core imports (may have additional dependencies)
try:
    from ml4t.data.core.config import Config  # noqa: F401
    from ml4t.data.data_manager import DataManager  # noqa: F401
    from ml4t.data.providers.base import BaseProvider  # noqa: F401

    __all__.extend(["BaseProvider", "Config", "DataManager"])
except ImportError:
    # Try to import what's available
    try:
        from ml4t.data.data_manager import DataManager  # noqa: F401

        __all__.append("DataManager")
    except ImportError:
        pass

    try:
        from ml4t.data.providers.base import BaseProvider  # noqa: F401

        __all__.append("BaseProvider")
    except ImportError:
        pass

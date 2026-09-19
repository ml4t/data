"""Asset class abstractions for ml4t-data."""

from ml4t.data.assets.asset_class import AssetClass, AssetInfo
from ml4t.data.assets.contracts import (
    FUTURES_REGISTRY,
    ContractSpec,
    get_contract_spec,
    load_contract_specs,
    register_contract_spec,
)
from ml4t.data.assets.schemas import AssetSchema, get_asset_schema
from ml4t.data.assets.validation import AssetValidator

__all__ = [
    "AssetClass",
    "AssetInfo",
    "AssetSchema",
    "AssetValidator",
    "get_asset_schema",
    # Contract specifications
    "ContractSpec",
    "FUTURES_REGISTRY",
    "get_contract_spec",
    "load_contract_specs",
    "register_contract_spec",
]

"""Tests for contracts module."""

from __future__ import annotations

from ml4t.data.assets.asset_class import AssetClass
from ml4t.data.assets.contracts import (
    FUTURES_REGISTRY,
    ContractSpec,
    get_contract_spec,
    load_contract_specs,
    register_contract_spec,
)


class TestContractSpec:
    """Tests for ContractSpec dataclass."""

    def test_create_contract_spec(self):
        """Test creating a ContractSpec."""
        spec = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=50.0,
            tick_size=0.25,
            margin=15000.0,
            exchange="CME",
            name="E-mini S&P 500",
        )
        assert spec.symbol == "ES"
        assert spec.asset_class == AssetClass.FUTURE
        assert spec.multiplier == 50.0
        assert spec.tick_size == 0.25
        assert spec.margin == 15000.0
        assert spec.exchange == "CME"

    def test_default_values(self):
        """Test ContractSpec default values."""
        spec = ContractSpec(symbol="AAPL")
        assert spec.asset_class == AssetClass.EQUITY
        assert spec.multiplier == 1.0
        assert spec.tick_size == 0.01
        assert spec.margin is None
        assert spec.currency == "USD"

    def test_tick_value_property(self):
        """Test tick_value calculation."""
        # ES: $50 * 0.25 = $12.50 per tick
        spec = ContractSpec(symbol="ES", multiplier=50.0, tick_size=0.25)
        assert spec.tick_value == 12.50

        # CL: $1000 * 0.01 = $10 per tick
        spec = ContractSpec(symbol="CL", multiplier=1000.0, tick_size=0.01)
        assert spec.tick_value == 10.0

        # Equity: $1 * 0.01 = $0.01 per tick
        spec = ContractSpec(symbol="AAPL")
        assert spec.tick_value == 0.01


class TestFuturesRegistry:
    """Tests for FUTURES_REGISTRY."""

    def test_registry_contains_major_contracts(self):
        """Test registry contains major futures contracts."""
        major_contracts = ["ES", "NQ", "CL", "GC", "ZN", "6E", "BTC"]
        for symbol in major_contracts:
            assert symbol in FUTURES_REGISTRY

    def test_es_contract_spec(self):
        """Test E-mini S&P 500 contract spec."""
        spec = FUTURES_REGISTRY["ES"]
        assert spec.multiplier == 50.0
        assert spec.tick_size == 0.25
        assert spec.exchange == "CME"
        assert spec.tick_value == 12.50

    def test_cl_contract_spec(self):
        """Test Crude Oil contract spec."""
        spec = FUTURES_REGISTRY["CL"]
        assert spec.multiplier == 1000.0
        assert spec.tick_size == 0.01
        assert spec.exchange == "NYMEX"
        assert spec.tick_value == 10.0

    def test_gc_contract_spec(self):
        """Test Gold contract spec."""
        spec = FUTURES_REGISTRY["GC"]
        assert spec.multiplier == 100.0
        assert spec.tick_size == 0.10
        assert spec.exchange == "COMEX"
        assert spec.tick_value == 10.0

    def test_all_contracts_have_required_fields(self):
        """Test all contracts have required fields."""
        for symbol, spec in FUTURES_REGISTRY.items():
            assert spec.symbol == symbol
            assert spec.asset_class == AssetClass.FUTURE
            assert spec.multiplier > 0
            assert spec.tick_size > 0
            assert spec.exchange is not None

    def test_micro_contracts_smaller_multiplier(self):
        """Test micro contracts have smaller multipliers."""
        assert FUTURES_REGISTRY["MES"].multiplier < FUTURES_REGISTRY["ES"].multiplier
        assert FUTURES_REGISTRY["MNQ"].multiplier < FUTURES_REGISTRY["NQ"].multiplier
        assert FUTURES_REGISTRY["MGC"].multiplier < FUTURES_REGISTRY["GC"].multiplier


class TestGetContractSpec:
    """Tests for get_contract_spec function."""

    def test_get_existing_contract(self):
        """Test getting an existing contract."""
        spec = get_contract_spec("ES")
        assert spec is not None
        assert spec.symbol == "ES"

    def test_get_nonexistent_contract(self):
        """Test getting a non-existent contract returns None."""
        spec = get_contract_spec("NONEXISTENT")
        assert spec is None

    def test_get_contract_case_sensitive(self):
        """Test symbol lookup is case sensitive."""
        assert get_contract_spec("ES") is not None
        assert get_contract_spec("es") is None


class TestLoadContractSpecs:
    """Tests for load_contract_specs function."""

    def test_load_specific_symbols(self):
        """Test loading specific symbols."""
        specs = load_contract_specs(symbols=["ES", "CL", "GC"])
        assert len(specs) == 3
        assert "ES" in specs
        assert "CL" in specs
        assert "GC" in specs

    def test_load_include_all(self):
        """Test loading all contracts."""
        specs = load_contract_specs(include_all=True)
        assert len(specs) == len(FUTURES_REGISTRY)

    def test_load_none_symbols_returns_empty(self):
        """Test loading with None symbols returns empty dict."""
        specs = load_contract_specs(symbols=None)
        assert specs == {}

    def test_load_empty_list_returns_empty(self):
        """Test loading with empty list returns empty dict."""
        specs = load_contract_specs(symbols=[])
        assert specs == {}

    def test_load_skips_unknown_symbols(self):
        """Test loading skips unknown symbols."""
        specs = load_contract_specs(symbols=["ES", "UNKNOWN", "CL"])
        assert len(specs) == 2
        assert "UNKNOWN" not in specs


class TestRegisterContractSpec:
    """Tests for register_contract_spec function."""

    def test_register_new_contract(self):
        """Test registering a new contract."""
        spec = ContractSpec(
            symbol="TEST",
            asset_class=AssetClass.FUTURE,
            multiplier=100.0,
            tick_size=0.01,
        )
        register_contract_spec(spec)

        assert "TEST" in FUTURES_REGISTRY
        assert FUTURES_REGISTRY["TEST"].multiplier == 100.0

        # Cleanup
        del FUTURES_REGISTRY["TEST"]

    def test_register_overwrites_existing(self):
        """Test registering overwrites existing contract."""
        original = FUTURES_REGISTRY["ES"].multiplier

        # Register with different multiplier
        spec = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=999.0,
            tick_size=0.25,
        )
        register_contract_spec(spec)

        assert FUTURES_REGISTRY["ES"].multiplier == 999.0

        # Restore original
        FUTURES_REGISTRY["ES"] = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=original,
            tick_size=0.25,
            margin=15000.0,
            exchange="CME",
            name="E-mini S&P 500",
        )

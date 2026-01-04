"""Tests for validation rules module."""

import tempfile
from pathlib import Path

import yaml

from ml4t.data.validation.rules import (
    AssetClass,
    ValidationRuleConfig,
    ValidationRulePresets,
    ValidationRuleSet,
    create_rule_set_from_config,
)


class TestAssetClass:
    """Test AssetClass enumeration."""

    def test_asset_class_values(self):
        """Test that asset class values are correct."""
        assert AssetClass.EQUITY == "equity"
        assert AssetClass.CRYPTO == "crypto"
        assert AssetClass.FOREX == "forex"
        assert AssetClass.COMMODITY == "commodity"
        assert AssetClass.INDEX == "index"
        assert AssetClass.FUTURE == "future"
        assert AssetClass.OPTION == "option"


class TestValidationRuleConfig:
    """Test ValidationRuleConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationRuleConfig()

        # Check default boolean settings
        assert config.check_nulls is True
        assert config.check_price_consistency is True
        assert config.check_negative_prices is True
        assert config.check_negative_volume is True
        assert config.check_duplicate_timestamps is True
        assert config.check_chronological_order is True
        assert config.check_price_staleness is True
        assert config.check_extreme_returns is True

        # Check default thresholds
        assert config.max_return_threshold == 0.5
        assert config.staleness_threshold == 5
        assert config.volume_spike_threshold == 10.0
        assert config.price_gap_threshold == 0.1

        # Check cross-validation settings
        assert config.check_price_continuity is True
        assert config.check_volume_spikes is True
        assert config.check_weekend_trading is True
        assert config.check_market_hours is True

        # Check default asset class
        assert config.asset_class == AssetClass.EQUITY

    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidationRuleConfig(
            check_nulls=False,
            max_return_threshold=0.3,
            asset_class=AssetClass.CRYPTO,
        )

        assert config.check_nulls is False
        assert config.max_return_threshold == 0.3
        assert config.asset_class == AssetClass.CRYPTO
        # Other values should still be defaults
        assert config.check_price_consistency is True
        assert config.staleness_threshold == 5


class TestValidationRuleSet:
    """Test ValidationRuleSet class."""

    def test_rule_set_creation(self):
        """Test creating a rule set."""
        rule_set = ValidationRuleSet(name="test_rules", description="Test rule set")

        assert rule_set.name == "test_rules"
        assert rule_set.description == "Test rule set"
        assert len(rule_set.rules) == 0
        assert rule_set.default_rule is None

    def test_get_rules_default(self):
        """Test getting rules when no specific rule exists."""
        rule_set = ValidationRuleSet(name="test_rules")

        # Should return default ValidationRuleConfig
        rules = rule_set.get_rules("AAPL")
        assert isinstance(rules, ValidationRuleConfig)
        assert rules.asset_class == AssetClass.EQUITY  # Default

    def test_get_rules_with_default_rule(self):
        """Test getting rules with a default rule set."""
        default_rule = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)
        rule_set = ValidationRuleSet(name="test_rules", default_rule=default_rule)

        rules = rule_set.get_rules("UNKNOWN")
        assert rules.asset_class == AssetClass.CRYPTO

    def test_get_rules_specific_symbol(self):
        """Test getting rules for a specific symbol."""
        rule_set = ValidationRuleSet(name="test_rules")
        crypto_rule = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)
        rule_set.add_rule("BTCUSD", crypto_rule)

        rules = rule_set.get_rules("BTCUSD")
        assert rules.asset_class == AssetClass.CRYPTO

    def test_get_rules_pattern_matching(self):
        """Test pattern matching for symbols."""
        rule_set = ValidationRuleSet(name="test_rules")
        crypto_rule = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)
        rule_set.add_rule("BTC*", crypto_rule)

        # Should match BTCUSD via pattern
        rules = rule_set.get_rules("BTCUSD")
        assert rules.asset_class == AssetClass.CRYPTO

        # Should not match ETHUSD
        rules = rule_set.get_rules("ETHUSD")
        assert rules.asset_class == AssetClass.EQUITY  # Default

    def test_add_rule(self):
        """Test adding rules to the set."""
        rule_set = ValidationRuleSet(name="test_rules")
        crypto_rule = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)

        rule_set.add_rule("CRYPTO_*", crypto_rule)

        assert "CRYPTO_*" in rule_set.rules
        assert rule_set.rules["CRYPTO_*"] == crypto_rule

    def test_save_and_load_rule_set(self):
        """Test saving and loading rule sets."""
        # Create a rule set with various rules
        rule_set = ValidationRuleSet(name="test_rules", description="Test description")

        default_rule = ValidationRuleConfig(asset_class=AssetClass.EQUITY)
        rule_set.default_rule = default_rule

        crypto_rule = ValidationRuleConfig(
            asset_class=AssetClass.CRYPTO,
            max_return_threshold=0.8,
        )
        rule_set.add_rule("BTC*", crypto_rule)

        forex_rule = ValidationRuleConfig(asset_class=AssetClass.FOREX)
        rule_set.add_rule("EUR*", forex_rule)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_rules.yaml"
            rule_set.save(file_path)

            # Verify file was created
            assert file_path.exists()

            # Load and verify
            loaded_rule_set = ValidationRuleSet.load(file_path)

            assert loaded_rule_set.name == "test_rules"
            assert loaded_rule_set.description == "Test description"

            # Check default rule
            assert loaded_rule_set.default_rule is not None
            assert loaded_rule_set.default_rule.asset_class == AssetClass.EQUITY

            # Check specific rules
            assert "BTC*" in loaded_rule_set.rules
            assert loaded_rule_set.rules["BTC*"].asset_class == AssetClass.CRYPTO
            assert loaded_rule_set.rules["BTC*"].max_return_threshold == 0.8

            assert "EUR*" in loaded_rule_set.rules
            assert loaded_rule_set.rules["EUR*"].asset_class == AssetClass.FOREX

    def test_save_without_default_rule(self):
        """Test saving rule set without default rule."""
        rule_set = ValidationRuleSet(name="minimal_rules")
        crypto_rule = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)
        rule_set.add_rule("CRYPTO", crypto_rule)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "minimal_rules.yaml"
            rule_set.save(file_path)

            # Load and verify
            loaded_rule_set = ValidationRuleSet.load(file_path)
            assert loaded_rule_set.default_rule is None
            assert "CRYPTO" in loaded_rule_set.rules


class TestValidationRulePresets:
    """Test ValidationRulePresets class."""

    def test_equity_rules(self):
        """Test equity rule preset."""
        rules = ValidationRulePresets.equity_rules()

        assert rules.asset_class == AssetClass.EQUITY
        assert rules.check_weekend_trading is True
        assert rules.check_market_hours is True
        assert rules.max_return_threshold == 0.2
        assert rules.staleness_threshold == 5
        assert rules.volume_spike_threshold == 10.0
        assert rules.price_gap_threshold == 0.05

    def test_crypto_rules(self):
        """Test crypto rule preset."""
        rules = ValidationRulePresets.crypto_rules()

        assert rules.asset_class == AssetClass.CRYPTO
        assert rules.check_weekend_trading is False  # Crypto trades 24/7
        assert rules.check_market_hours is False
        assert rules.max_return_threshold == 0.5  # More volatile
        assert rules.staleness_threshold == 3
        assert rules.volume_spike_threshold == 20.0
        assert rules.price_gap_threshold == 0.15

    def test_forex_rules(self):
        """Test forex rule preset."""
        rules = ValidationRulePresets.forex_rules()

        assert rules.asset_class == AssetClass.FOREX
        assert rules.check_weekend_trading is True
        assert rules.check_market_hours is False  # 24/5 market
        assert rules.max_return_threshold == 0.1
        assert rules.staleness_threshold == 10
        assert rules.volume_spike_threshold == 5.0
        assert rules.price_gap_threshold == 0.02

    def test_commodity_rules(self):
        """Test commodity rule preset."""
        rules = ValidationRulePresets.commodity_rules()

        assert rules.asset_class == AssetClass.COMMODITY
        assert rules.check_weekend_trading is True
        assert rules.check_market_hours is True
        assert rules.max_return_threshold == 0.15
        assert rules.staleness_threshold == 5
        assert rules.volume_spike_threshold == 8.0
        assert rules.price_gap_threshold == 0.05

    def test_strict_rules(self):
        """Test strict rule preset."""
        rules = ValidationRulePresets.strict_rules()

        # All checks should be enabled
        assert rules.check_nulls is True
        assert rules.check_price_consistency is True
        assert rules.check_negative_prices is True
        assert rules.check_negative_volume is True
        assert rules.check_duplicate_timestamps is True
        assert rules.check_chronological_order is True
        assert rules.check_price_staleness is True
        assert rules.check_extreme_returns is True
        assert rules.check_price_continuity is True
        assert rules.check_volume_spikes is True

        # Strict thresholds
        assert rules.max_return_threshold == 0.1
        assert rules.staleness_threshold == 3
        assert rules.volume_spike_threshold == 5.0
        assert rules.price_gap_threshold == 0.02

    def test_relaxed_rules(self):
        """Test relaxed rule preset."""
        rules = ValidationRulePresets.relaxed_rules()

        # Basic checks enabled
        assert rules.check_nulls is True
        assert rules.check_price_consistency is True
        assert rules.check_negative_prices is True
        assert rules.check_duplicate_timestamps is True
        assert rules.check_chronological_order is True

        # Relaxed checks disabled
        assert rules.check_negative_volume is False
        assert rules.check_price_staleness is False
        assert rules.check_extreme_returns is False
        assert rules.check_price_continuity is False
        assert rules.check_volume_spikes is False

        # Relaxed thresholds
        assert rules.max_return_threshold == 1.0
        assert rules.staleness_threshold == 10
        assert rules.volume_spike_threshold == 50.0
        assert rules.price_gap_threshold == 0.25


class TestCreateRuleSetFromConfig:
    """Test create_rule_set_from_config function."""

    def test_create_from_config_file(self):
        """Test creating rule set from configuration file."""
        # Create a config file
        config_data = {
            "name": "test_config_rules",
            "description": "Test configuration",
            "default": {
                "asset_class": "equity",
                "max_return_threshold": 0.25,
            },
            "rules": {
                "CRYPTO_*": {
                    "asset_class": "crypto",
                    "check_weekend_trading": False,
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Create rule set from config
            rule_set = create_rule_set_from_config(config_path)

            assert rule_set.name == "test_config_rules"
            assert rule_set.description == "Test configuration"
            assert rule_set.default_rule.asset_class == AssetClass.EQUITY
            assert rule_set.default_rule.max_return_threshold == 0.25

            # Check specific rule
            crypto_rules = rule_set.get_rules("CRYPTO_BTCUSD")
            assert crypto_rules.asset_class == AssetClass.CRYPTO
            assert crypto_rules.check_weekend_trading is False


class TestIntegration:
    """Integration tests for validation rules."""

    def test_complete_workflow(self):
        """Test complete workflow with rule sets."""
        # Create rule set with presets
        rule_set = ValidationRuleSet(name="trading_rules", description="Trading validation rules")

        # Set default to equity rules
        rule_set.default_rule = ValidationRulePresets.equity_rules()

        # Add specific rules for different asset classes
        rule_set.add_rule("BTC*", ValidationRulePresets.crypto_rules())
        rule_set.add_rule("EUR*", ValidationRulePresets.forex_rules())
        rule_set.add_rule("GOLD", ValidationRulePresets.commodity_rules())

        # Test getting rules for different symbols
        btc_rules = rule_set.get_rules("BTCUSD")
        assert btc_rules.asset_class == AssetClass.CRYPTO
        assert btc_rules.check_weekend_trading is False

        eur_rules = rule_set.get_rules("EURUSD")
        assert eur_rules.asset_class == AssetClass.FOREX
        assert eur_rules.max_return_threshold == 0.1

        aapl_rules = rule_set.get_rules("AAPL")  # Should use default
        assert aapl_rules.asset_class == AssetClass.EQUITY
        assert aapl_rules.max_return_threshold == 0.2

        gold_rules = rule_set.get_rules("GOLD")
        assert gold_rules.asset_class == AssetClass.COMMODITY
        assert gold_rules.max_return_threshold == 0.15

        # Test persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "trading_rules.yaml"
            rule_set.save(config_path)

            # Load and verify functionality
            loaded_rule_set = create_rule_set_from_config(config_path)

            # Test that loaded rule set works the same
            loaded_btc_rules = loaded_rule_set.get_rules("BTCUSD")
            assert loaded_btc_rules.asset_class == AssetClass.CRYPTO
            assert loaded_btc_rules.check_weekend_trading is False

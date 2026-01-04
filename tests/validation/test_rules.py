"""Comprehensive tests for validation rules.

Tests cover:
- ValidationRuleConfig Pydantic model
- ValidationRuleSet management and pattern matching
- ValidationRulePresets for different asset classes
- YAML save/load functionality
"""

from pathlib import Path

from ml4t.data.core.models import AssetClass
from ml4t.data.validation.rules import (
    ValidationRuleConfig,
    ValidationRulePresets,
    ValidationRuleSet,
    create_rule_set_from_config,
)


class TestValidationRuleConfig:
    """Test ValidationRuleConfig Pydantic model."""

    def test_default_initialization(self):
        """All checks enabled by default."""
        config = ValidationRuleConfig()
        assert config.check_nulls is True
        assert config.check_price_consistency is True
        assert config.check_negative_prices is True
        assert config.check_negative_volume is True
        assert config.check_duplicate_timestamps is True
        assert config.check_chronological_order is True
        assert config.check_price_staleness is True
        assert config.check_extreme_returns is True

    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        config = ValidationRuleConfig()
        assert config.max_return_threshold == 0.5
        assert config.staleness_threshold == 5
        assert config.volume_spike_threshold == 10.0
        assert config.price_gap_threshold == 0.1

    def test_default_cross_validation_settings(self):
        """Default cross-validation settings are enabled."""
        config = ValidationRuleConfig()
        assert config.check_price_continuity is True
        assert config.check_volume_spikes is True
        assert config.check_weekend_trading is True
        assert config.check_market_hours is True

    def test_default_asset_class(self):
        """Default asset class is EQUITY."""
        config = ValidationRuleConfig()
        assert config.asset_class == AssetClass.EQUITY

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = ValidationRuleConfig(
            check_nulls=False,
            max_return_threshold=0.75,
            staleness_threshold=10,
            asset_class=AssetClass.CRYPTO,
        )
        assert config.check_nulls is False
        assert config.max_return_threshold == 0.75
        assert config.staleness_threshold == 10
        assert config.asset_class == AssetClass.CRYPTO

    def test_model_dump(self):
        """Config can be serialized to dict."""
        config = ValidationRuleConfig(
            check_nulls=False,
            max_return_threshold=0.25,
        )
        data = config.model_dump()

        assert data["check_nulls"] is False
        assert data["max_return_threshold"] == 0.25
        assert "asset_class" in data

    def test_model_from_dict(self):
        """Config can be created from dict."""
        data = {
            "check_nulls": False,
            "max_return_threshold": 0.3,
            "asset_class": "crypto",
        }
        config = ValidationRuleConfig(**data)

        assert config.check_nulls is False
        assert config.max_return_threshold == 0.3


class TestValidationRuleSet:
    """Test ValidationRuleSet management."""

    def test_basic_initialization(self):
        """Basic rule set initialization."""
        rule_set = ValidationRuleSet(
            name="test_rules",
            description="Test rule set",
        )
        assert rule_set.name == "test_rules"
        assert rule_set.description == "Test rule set"
        assert len(rule_set.rules) == 0
        assert rule_set.default_rule is None

    def test_add_rule(self):
        """Add a rule to the set."""
        rule_set = ValidationRuleSet(name="test")
        config = ValidationRuleConfig(max_return_threshold=0.2)

        rule_set.add_rule("AAPL", config)

        assert "AAPL" in rule_set.rules
        assert rule_set.rules["AAPL"].max_return_threshold == 0.2

    def test_get_rules_exact_match(self):
        """Get rules by exact symbol match."""
        rule_set = ValidationRuleSet(name="test")
        config = ValidationRuleConfig(max_return_threshold=0.2)
        rule_set.add_rule("AAPL", config)

        result = rule_set.get_rules("AAPL")

        assert result.max_return_threshold == 0.2

    def test_get_rules_pattern_match(self):
        """Get rules by pattern matching (wildcard)."""
        rule_set = ValidationRuleSet(name="test")
        crypto_config = ValidationRuleConfig(
            asset_class=AssetClass.CRYPTO,
            max_return_threshold=0.5,
        )
        rule_set.add_rule("BTC*", crypto_config)

        # Should match BTCUSD
        result = rule_set.get_rules("BTCUSD")
        assert result.asset_class == AssetClass.CRYPTO
        assert result.max_return_threshold == 0.5

        # Should match BTCEUR
        result = rule_set.get_rules("BTCEUR")
        assert result.asset_class == AssetClass.CRYPTO

    def test_get_rules_default_fallback(self):
        """Fall back to default rule when no match."""
        default_config = ValidationRuleConfig(max_return_threshold=0.3)
        rule_set = ValidationRuleSet(
            name="test",
            default_rule=default_config,
        )
        rule_set.add_rule("AAPL", ValidationRuleConfig(max_return_threshold=0.2))

        # Should return default for unknown symbol
        result = rule_set.get_rules("GOOGL")

        assert result.max_return_threshold == 0.3

    def test_get_rules_no_match_creates_default(self):
        """Create default config when no match and no default rule."""
        rule_set = ValidationRuleSet(name="test")
        rule_set.add_rule("AAPL", ValidationRuleConfig(max_return_threshold=0.2))

        # Should return new default config
        result = rule_set.get_rules("UNKNOWN")

        assert result.max_return_threshold == 0.5  # Default value

    def test_exact_match_priority_over_pattern(self):
        """Exact match takes priority over pattern."""
        rule_set = ValidationRuleSet(name="test")
        rule_set.add_rule("BTC*", ValidationRuleConfig(max_return_threshold=0.5))
        rule_set.add_rule("BTCUSD", ValidationRuleConfig(max_return_threshold=0.3))

        result = rule_set.get_rules("BTCUSD")

        assert result.max_return_threshold == 0.3  # Exact match

    def test_save_and_load(self, tmp_path: Path):
        """Save and load rule set from YAML."""
        # Create rule set
        rule_set = ValidationRuleSet(
            name="test_ruleset",
            description="Test description",
        )
        rule_set.default_rule = ValidationRuleConfig(max_return_threshold=0.4)
        rule_set.add_rule("AAPL", ValidationRuleConfig(max_return_threshold=0.2))
        rule_set.add_rule(
            "BTC*",
            ValidationRuleConfig(
                asset_class=AssetClass.CRYPTO,
                max_return_threshold=0.6,
            ),
        )

        # Save
        path = tmp_path / "rules.yaml"
        rule_set.save(path)

        assert path.exists()

        # Load
        loaded = ValidationRuleSet.load(path)

        assert loaded.name == "test_ruleset"
        assert loaded.description == "Test description"
        assert loaded.default_rule is not None
        assert loaded.default_rule.max_return_threshold == 0.4
        assert "AAPL" in loaded.rules
        assert loaded.rules["AAPL"].max_return_threshold == 0.2

    def test_save_creates_directory(self, tmp_path: Path):
        """Save creates parent directory if needed."""
        rule_set = ValidationRuleSet(name="test")

        path = tmp_path / "nested" / "dir" / "rules.yaml"
        rule_set.save(path)

        assert path.exists()

    def test_load_with_no_default(self, tmp_path: Path):
        """Load rule set without default rule."""
        rule_set = ValidationRuleSet(name="test")
        rule_set.add_rule("AAPL", ValidationRuleConfig())

        path = tmp_path / "rules.yaml"
        rule_set.save(path)

        loaded = ValidationRuleSet.load(path)

        assert loaded.default_rule is None


class TestValidationRulePresets:
    """Test validation rule presets for asset classes."""

    def test_equity_rules(self):
        """Equity preset has correct settings."""
        config = ValidationRulePresets.equity_rules()

        assert config.asset_class == AssetClass.EQUITY
        assert config.check_weekend_trading is True
        assert config.check_market_hours is True
        assert config.max_return_threshold == 0.2
        assert config.staleness_threshold == 5
        assert config.volume_spike_threshold == 10.0
        assert config.price_gap_threshold == 0.05

    def test_crypto_rules(self):
        """Crypto preset has correct settings."""
        config = ValidationRulePresets.crypto_rules()

        assert config.asset_class == AssetClass.CRYPTO
        assert config.check_weekend_trading is False  # 24/7 trading
        assert config.check_market_hours is False  # No market hours
        assert config.max_return_threshold == 0.5  # Higher volatility
        assert config.staleness_threshold == 3
        assert config.volume_spike_threshold == 20.0

    def test_forex_rules(self):
        """Forex preset has correct settings."""
        config = ValidationRulePresets.forex_rules()

        assert config.asset_class == AssetClass.FOREX
        assert config.check_weekend_trading is True
        assert config.check_market_hours is False  # 24/5 market
        assert config.max_return_threshold == 0.1  # Less volatile
        assert config.price_gap_threshold == 0.02

    def test_commodity_rules(self):
        """Commodity preset has correct settings."""
        config = ValidationRulePresets.commodity_rules()

        assert config.asset_class == AssetClass.COMMODITY
        assert config.check_weekend_trading is True
        assert config.check_market_hours is True
        assert config.max_return_threshold == 0.15
        assert config.volume_spike_threshold == 8.0

    def test_strict_rules(self):
        """Strict preset enables all checks with low thresholds."""
        config = ValidationRulePresets.strict_rules()

        # All checks enabled
        assert config.check_nulls is True
        assert config.check_price_consistency is True
        assert config.check_negative_prices is True
        assert config.check_negative_volume is True
        assert config.check_duplicate_timestamps is True
        assert config.check_chronological_order is True
        assert config.check_price_staleness is True
        assert config.check_extreme_returns is True
        assert config.check_price_continuity is True
        assert config.check_volume_spikes is True

        # Low thresholds
        assert config.max_return_threshold == 0.1
        assert config.staleness_threshold == 3
        assert config.volume_spike_threshold == 5.0
        assert config.price_gap_threshold == 0.02

    def test_relaxed_rules(self):
        """Relaxed preset has higher thresholds and fewer checks."""
        config = ValidationRulePresets.relaxed_rules()

        # Core checks enabled
        assert config.check_nulls is True
        assert config.check_price_consistency is True
        assert config.check_negative_prices is True
        assert config.check_duplicate_timestamps is True
        assert config.check_chronological_order is True

        # Some checks disabled
        assert config.check_negative_volume is False
        assert config.check_price_staleness is False
        assert config.check_extreme_returns is False
        assert config.check_price_continuity is False
        assert config.check_volume_spikes is False

        # High thresholds
        assert config.max_return_threshold == 1.0
        assert config.staleness_threshold == 10
        assert config.volume_spike_threshold == 50.0
        assert config.price_gap_threshold == 0.25


class TestCreateRuleSetFromConfig:
    """Test loading rule sets from config files."""

    def test_load_from_yaml(self, tmp_path: Path):
        """Load rule set from YAML config."""
        # Create a YAML config file
        import yaml

        config_data = {
            "name": "production_rules",
            "description": "Production validation rules",
            "default": {
                "check_nulls": True,
                "max_return_threshold": 0.3,
            },
            "rules": {
                "AAPL": {"max_return_threshold": 0.15},
                "BTC*": {
                    "asset_class": "crypto",
                    "check_weekend_trading": False,
                },
            },
        }

        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        # Load using helper function
        rule_set = create_rule_set_from_config(path)

        assert rule_set.name == "production_rules"
        assert rule_set.description == "Production validation rules"
        assert rule_set.default_rule.max_return_threshold == 0.3
        assert rule_set.rules["AAPL"].max_return_threshold == 0.15


class TestAssetClassIntegration:
    """Test asset class integration with rules."""

    def test_all_asset_classes_supported(self):
        """All asset classes can be used in config."""
        for asset_class in AssetClass:
            config = ValidationRuleConfig(asset_class=asset_class)
            assert config.asset_class == asset_class

    def test_asset_class_serialization(self):
        """Asset class serializes correctly."""
        config = ValidationRuleConfig(asset_class=AssetClass.CRYPTO)
        data = config.model_dump()

        # use_enum_values should serialize to string
        assert data["asset_class"] == "crypto"

    def test_asset_class_from_string(self):
        """Asset class can be created from string."""
        config = ValidationRuleConfig(asset_class="forex")
        assert config.asset_class == "forex" or config.asset_class == AssetClass.FOREX

"""Configurable validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, Field

from ml4t.data.assets.asset_class import AssetClass

logger = structlog.get_logger()


class ValidationRuleConfig(BaseModel):
    """Configuration for validation rules."""

    # OHLCV validation settings
    check_nulls: bool = Field(default=True, description="Check for null values")
    check_price_consistency: bool = Field(default=True, description="Check OHLC relationships")
    check_negative_prices: bool = Field(default=True, description="Check for negative prices")
    check_negative_volume: bool = Field(default=True, description="Check for negative volume")
    check_duplicate_timestamps: bool = Field(
        default=True, description="Check for duplicate timestamps"
    )
    check_chronological_order: bool = Field(default=True, description="Check timestamp ordering")
    check_price_staleness: bool = Field(default=True, description="Check for stale prices")
    check_extreme_returns: bool = Field(default=True, description="Check for extreme returns")

    # Thresholds
    max_return_threshold: float = Field(
        default=0.5, description="Maximum return threshold (as fraction)"
    )
    staleness_threshold: int = Field(
        default=5, description="Days of identical prices to flag as stale"
    )
    volume_spike_threshold: float = Field(default=10.0, description="Volume spike multiplier")
    price_gap_threshold: float = Field(default=0.1, description="Price gap threshold (as fraction)")

    # Cross-validation settings
    check_price_continuity: bool = Field(default=True, description="Check price continuity")
    check_volume_spikes: bool = Field(default=True, description="Check for volume spikes")
    check_weekend_trading: bool = Field(default=True, description="Check for weekend trading")
    check_market_hours: bool = Field(default=True, description="Check market hours")

    # Asset class specific
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Asset class")

    model_config = {"use_enum_values": True}


@dataclass
class ValidationRuleSet:
    """Set of validation rules for different scenarios."""

    name: str
    description: str = ""
    rules: dict[str, ValidationRuleConfig] = field(default_factory=dict)
    default_rule: ValidationRuleConfig | None = None

    def get_rules(self, symbol: str) -> ValidationRuleConfig:
        """
        Get validation rules for a symbol.

        Args:
            symbol: Symbol to get rules for

        Returns:
            ValidationRuleConfig for the symbol
        """
        # Check if specific rules exist for this symbol
        if symbol in self.rules:
            return self.rules[symbol]

        # Check for pattern matching (e.g., "BTC*" matches "BTCUSD")
        for pattern, rule in self.rules.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if symbol.startswith(prefix):
                    return rule

        # Return default rule or create standard one
        return self.default_rule or ValidationRuleConfig()

    def add_rule(self, symbol_pattern: str, rule: ValidationRuleConfig) -> None:
        """Add a rule for a symbol pattern."""
        self.rules[symbol_pattern] = rule

    def save(self, path: Path) -> None:
        """Save rule set to YAML file."""
        # Use mode="json" to ensure enums are serialized as strings
        data = {
            "name": self.name,
            "description": self.description,
            "default": self.default_rule.model_dump(mode="json") if self.default_rule else None,
            "rules": {
                pattern: rule.model_dump(mode="json") for pattern, rule in self.rules.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("Saved validation rule set", path=str(path))

    @classmethod
    def load(cls, path: Path) -> ValidationRuleSet:
        """Load rule set from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        rule_set = cls(
            name=data["name"],
            description=data.get("description", ""),
        )

        # Load default rule
        if data.get("default"):
            rule_set.default_rule = ValidationRuleConfig(**data["default"])

        # Load specific rules
        for pattern, rule_data in data.get("rules", {}).items():
            rule_set.rules[pattern] = ValidationRuleConfig(**rule_data)

        return rule_set


class ValidationRulePresets:
    """Preset validation rules for common scenarios."""

    @staticmethod
    def equity_rules() -> ValidationRuleConfig:
        """Standard equity validation rules."""
        return ValidationRuleConfig(
            asset_class=AssetClass.EQUITY,
            check_weekend_trading=True,
            check_market_hours=True,
            max_return_threshold=0.2,  # 20% daily return threshold
            staleness_threshold=5,
            volume_spike_threshold=10.0,
            price_gap_threshold=0.05,  # 5% gap threshold
        )

    @staticmethod
    def crypto_rules() -> ValidationRuleConfig:
        """Cryptocurrency validation rules."""
        return ValidationRuleConfig(
            asset_class=AssetClass.CRYPTO,
            check_weekend_trading=False,  # Crypto trades 24/7
            check_market_hours=False,  # No market hours for crypto
            max_return_threshold=0.5,  # 50% - crypto is volatile
            staleness_threshold=3,  # Less staleness tolerance
            volume_spike_threshold=20.0,  # Higher spike threshold
            price_gap_threshold=0.15,  # 15% gap threshold
        )

    @staticmethod
    def forex_rules() -> ValidationRuleConfig:
        """Forex validation rules."""
        return ValidationRuleConfig(
            asset_class=AssetClass.FOREX,
            check_weekend_trading=True,  # Limited weekend trading
            check_market_hours=False,  # 24/5 market
            max_return_threshold=0.1,  # 10% - forex is less volatile
            staleness_threshold=10,
            volume_spike_threshold=5.0,
            price_gap_threshold=0.02,  # 2% gap threshold
        )

    @staticmethod
    def commodity_rules() -> ValidationRuleConfig:
        """Commodity validation rules."""
        return ValidationRuleConfig(
            asset_class=AssetClass.COMMODITY,
            check_weekend_trading=True,
            check_market_hours=True,
            max_return_threshold=0.15,  # 15% daily return
            staleness_threshold=5,
            volume_spike_threshold=8.0,
            price_gap_threshold=0.05,
        )

    @staticmethod
    def strict_rules() -> ValidationRuleConfig:
        """Strict validation rules for high-quality data."""
        return ValidationRuleConfig(
            check_nulls=True,
            check_price_consistency=True,
            check_negative_prices=True,
            check_negative_volume=True,
            check_duplicate_timestamps=True,
            check_chronological_order=True,
            check_price_staleness=True,
            check_extreme_returns=True,
            check_price_continuity=True,
            check_volume_spikes=True,
            max_return_threshold=0.1,  # Very strict
            staleness_threshold=3,
            volume_spike_threshold=5.0,
            price_gap_threshold=0.02,
        )

    @staticmethod
    def relaxed_rules() -> ValidationRuleConfig:
        """Relaxed validation rules for noisy data."""
        return ValidationRuleConfig(
            check_nulls=True,
            check_price_consistency=True,
            check_negative_prices=True,
            check_negative_volume=False,  # Allow missing volume
            check_duplicate_timestamps=True,
            check_chronological_order=True,
            check_price_staleness=False,  # Don't check staleness
            check_extreme_returns=False,  # Don't check extreme returns
            check_price_continuity=False,
            check_volume_spikes=False,
            max_return_threshold=1.0,  # Very relaxed
            staleness_threshold=10,
            volume_spike_threshold=50.0,
            price_gap_threshold=0.25,
        )


def create_rule_set_from_config(config_path: Path) -> ValidationRuleSet:
    """
    Create a validation rule set from a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ValidationRuleSet loaded from configuration
    """
    return ValidationRuleSet.load(config_path)

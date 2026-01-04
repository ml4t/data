# Multi-Provider Strategies

**Target Audience**: Advanced users building resilient data pipelines
**Time to Read**: 20 minutes
**Prerequisites**: All previous tutorials

## Why Use Multiple Providers?

Relying on a single data provider creates risks:

1. **Provider Outages** - APIs go down
2. **Rate Limit Exhaustion** - You run out of quota
3. **Data Quality Issues** - Provider has bad data
4. **Coverage Gaps** - Provider doesn't have all assets you need
5. **Cost Optimization** - Use cheapest provider for each asset class

**Solution**: Use multiple providers with intelligent fallback and selection logic.

## Strategy 1: Primary + Fallback

Use a primary provider, fall back to secondary if primary fails.

```python
from ml4t.data.providers import TiingoProvider, IEXCloudProvider
from ml4t.data.core.exceptions import NetworkError, RateLimitError

class MultiProviderFetcher:
    def __init__(self):
        # Primary: Tiingo (good quality, 1000/day)
        self.primary = TiingoProvider(api_key="tiingo_key")

        # Fallback: IEX Cloud (50K msg/month)
        self.fallback = IEXCloudProvider(api_key="iex_key")

    def fetch_ohlcv(self, symbol, start, end):
        """Try primary, fall back to secondary."""
        try:
            # Try primary first
            data = self.primary.fetch_ohlcv(symbol, start, end)
            logger.info(f"{symbol}: Fetched from Tiingo (primary)")
            return data

        except (NetworkError, RateLimitError) as e:
            # Primary failed, try fallback
            logger.warning(f"{symbol}: Tiingo failed ({e}), trying IEX Cloud")

            try:
                data = self.fallback.fetch_ohlcv(symbol, start, end)
                logger.info(f"{symbol}: Fetched from IEX Cloud (fallback)")
                return data

            except Exception as e2:
                # Both failed
                logger.error(f"{symbol}: Both providers failed")
                raise Exception(f"All providers failed: {e}, {e2}")

# Usage
fetcher = MultiProviderFetcher()
data = fetcher.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
```

## Strategy 2: Provider Selection by Asset Class

Use the best provider for each asset class.

```python
class AssetClassRouter:
    """Route symbols to optimal provider based on asset class."""

    def __init__(self):
        # Crypto: CoinGecko (free, unlimited)
        self.crypto_provider = CoinGeckoProvider()

        # US Stocks: Tiingo (high quality, 1000/day)
        self.us_stocks_provider = TiingoProvider(api_key="tiingo_key")

        # Global Stocks: EODHD (60+ exchanges, €19.99/mo)
        self.global_stocks_provider = EODHDProvider(api_key="eodhd_key")

        # Forex: OANDA (professional forex)
        self.forex_provider = OANDAProvider(api_key="oanda_key")

    def get_provider(self, symbol):
        """Select provider based on symbol type."""
        if symbol.lower() in ["bitcoin", "ethereum", "cardano", "ripple"]:
            return self.crypto_provider
        elif symbol.count(".") == 1:  # AAPL.US, VOD.LSE format
            return self.global_stocks_provider
        elif "_" in symbol:  # EUR_USD format
            return self.forex_provider
        else:  # AAPL, MSFT format (US stocks)
            return self.us_stocks_provider

    def fetch_ohlcv(self, symbol, start, end):
        """Fetch using optimal provider."""
        provider = self.get_provider(symbol)
        logger.info(f"{symbol}: Using {provider.name()}")
        return provider.fetch_ohlcv(symbol, start, end)

# Usage
router = AssetClassRouter()

btc_data = router.fetch_ohlcv("bitcoin", "2024-01-01", "2024-01-31")
# → Uses CoinGecko

aapl_data = router.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
# → Uses Tiingo

vod_data = router.fetch_ohlcv("VOD.LSE", "2024-01-01", "2024-01-31")
# → Uses EODHD

eur_usd_data = router.fetch_ohlcv("EUR_USD", "2024-01-01", "2024-01-31")
# → Uses OANDA
```

## Strategy 3: Rate Limit Aware Distribution

Distribute symbols across providers to maximize throughput.

```python
class RateLimitAwareDistributor:
    """Distribute symbols across providers based on rate limits."""

    def __init__(self):
        # Provider 1: Tiingo (1000/day)
        self.provider1 = TiingoProvider(api_key="key1")
        self.provider1_limit = 1000
        self.provider1_used = 0

        # Provider 2: IEX Cloud (50K msg/month ≈ 1,666/day)
        self.provider2 = IEXCloudProvider(api_key="key2")
        self.provider2_limit = 1666
        self.provider2_used = 0

        # Provider 3: Alpha Vantage (25/day)
        self.provider3 = AlphaVantageProvider(api_key="key3")
        self.provider3_limit = 25
        self.provider3_used = 0

    def get_available_provider(self):
        """Get provider with remaining quota."""
        if self.provider1_used < self.provider1_limit:
            return self.provider1, "Tiingo"
        elif self.provider2_used < self.provider2_limit:
            return self.provider2, "IEX Cloud"
        elif self.provider3_used < self.provider3_limit:
            return self.provider3, "Alpha Vantage"
        else:
            raise RateLimitError("All providers exhausted")

    def fetch_ohlcv(self, symbol, start, end):
        """Fetch using provider with available quota."""
        provider, provider_name = self.get_available_provider()

        data = provider.fetch_ohlcv(symbol, start, end)

        # Update usage
        if provider_name == "Tiingo":
            self.provider1_used += 1
        elif provider_name == "IEX Cloud":
            self.provider2_used += 1
        elif provider_name == "Alpha Vantage":
            self.provider3_used += 1

        logger.info(f"{symbol}: Used {provider_name} ({self.provider1_used + self.provider2_used + self.provider3_used} total)")
        return data

# Usage
distributor = RateLimitAwareDistributor()

# Can fetch 1000 + 1666 + 25 = 2,691 symbols per day!
for symbol in symbols[:2691]:
    data = distributor.fetch_ohlcv(symbol, start, end)
```

## Strategy 4: Data Quality Consensus

Compare multiple providers and use consensus.

```python
class ConsensusValidator:
    """Validate data against multiple providers."""

    def __init__(self):
        self.providers = [
            ("Tiingo", TiingoProvider(api_key="key1")),
            ("IEX", IEXCloudProvider(api_key="key2")),
            ("Alpha Vantage", AlphaVantageProvider(api_key="key3")),
        ]

    def fetch_with_consensus(self, symbol, start, end, min_agreement=2):
        """Fetch from multiple providers and verify consensus."""
        results = {}
        errors = {}

        # Fetch from all providers
        for name, provider in self.providers:
            try:
                data = provider.fetch_ohlcv(symbol, start, end)
                results[name] = data
            except Exception as e:
                errors[name] = str(e)
                logger.warning(f"{name} failed for {symbol}: {e}")

        if len(results) < min_agreement:
            raise DataValidationError(
                f"Only {len(results)} providers succeeded, need {min_agreement}"
            )

        # Compare close prices
        providers_list = list(results.keys())
        base_provider = providers_list[0]
        base_data = results[base_provider]

        for other_provider in providers_list[1:]:
            other_data = results[other_provider]

            # Merge and compare
            merged = base_data.join(other_data, on="timestamp", suffix=f"_{other_provider}")

            # Calculate price difference
            merged = merged.with_columns(
                ((merged["close"] - merged[f"close_{other_provider}"]).abs() / merged["close"] * 100)
                .alias("price_diff_pct")
            )

            # Check agreement (prices within 0.5%)
            disagreements = merged.filter(merged["price_diff_pct"] > 0.5)

            if len(disagreements) > 0:
                logger.warning(
                    f"{symbol}: {base_provider} vs {other_provider} disagree on "
                    f"{len(disagreements)} dates ({len(disagreements)/len(merged)*100:.1f}%)"
                )
            else:
                logger.info(f"{symbol}: {base_provider} and {other_provider} agree ✅")

        # Return data from base provider (all providers agree)
        return base_data

# Usage
validator = ConsensusValidator()
data = validator.fetch_with_consensus("AAPL", "2024-01-01", "2024-01-31")
# → Validates data across 3 providers
```

## Strategy 5: Cost Optimization

Use cheapest provider that meets requirements.

```python
class CostOptimizer:
    """Select provider based on cost per call."""

    def __init__(self):
        # Define provider costs and limits
        self.providers = [
            {
                "name": "CoinGecko",
                "provider": CoinGeckoProvider(),
                "cost_per_call": 0.00,  # Free
                "daily_limit": 10000,
                "asset_classes": ["crypto"],
            },
            {
                "name": "Tiingo",
                "provider": TiingoProvider(api_key="key"),
                "cost_per_call": 0.00,  # Free tier
                "daily_limit": 1000,
                "asset_classes": ["stocks", "crypto"],
            },
            {
                "name": "EODHD",
                "provider": EODHDProvider(api_key="key"),
                "cost_per_call": 0.00,  # Paid tier (unlimited)
                "monthly_cost": 19.99,  # €19.99/month
                "asset_classes": ["stocks_global"],
            },
            {
                "name": "Finnhub",
                "provider": FinnhubProvider(api_key="key"),
                "cost_per_call": 0.00,  # Paid tier
                "monthly_cost": 59.99,  # $59.99/month
                "asset_classes": ["stocks", "crypto", "forex"],
            },
        ]

    def get_cheapest_provider(self, asset_class, calls_per_day):
        """Get cheapest provider for asset class and usage."""
        eligible = [
            p for p in self.providers
            if asset_class in p["asset_classes"]
        ]

        # Calculate effective cost
        for provider in eligible:
            if "monthly_cost" in provider:
                # Monthly subscription / ~30 days / calls
                provider["effective_cost"] = provider["monthly_cost"] / 30 / calls_per_day
            elif "daily_limit" in provider:
                if calls_per_day <= provider["daily_limit"]:
                    provider["effective_cost"] = 0.00  # Free tier sufficient
                else:
                    provider["effective_cost"] = float('inf')  # Can't handle volume

        # Sort by cost
        eligible.sort(key=lambda p: p["effective_cost"])

        if not eligible:
            raise ValueError(f"No provider for {asset_class}")

        cheapest = eligible[0]
        logger.info(
            f"Cheapest for {asset_class} ({calls_per_day} calls/day): "
            f"{cheapest['name']} (${cheapest['effective_cost']:.4f}/call)"
        )
        return cheapest["provider"]

# Usage
optimizer = CostOptimizer()

# Low volume (100 calls/day) → Use free tier
crypto_provider = optimizer.get_cheapest_provider("crypto", calls_per_day=100)
# → CoinGecko (free)

# High volume (5000 calls/day) → Use paid tier
stocks_provider = optimizer.get_cheapest_provider("stocks", calls_per_day=5000)
# → EODHD (€19.99/month, unlimited)
```

## Strategy 6: Geographic Routing

Route to provider with best coverage for geographic region.

```python
class GeographicRouter:
    """Route based on symbol's exchange/region."""

    def __init__(self):
        self.providers = {
            "US": TiingoProvider(api_key="tiingo_key"),
            "EU": EODHDProvider(api_key="eodhd_key", exchange="LSE"),
            "ASIA": EODHDProvider(api_key="eodhd_key", exchange="TSE"),
            "GLOBAL": EODHDProvider(api_key="eodhd_key"),
        }

    def get_region(self, symbol):
        """Determine region from symbol."""
        if ".US" in symbol or symbol in US_SYMBOLS:
            return "US"
        elif ".LSE" in symbol or ".FRA" in symbol:
            return "EU"
        elif ".TSE" in symbol or ".HKG" in symbol:
            return "ASIA"
        else:
            return "GLOBAL"

    def fetch_ohlcv(self, symbol, start, end):
        """Fetch using region-specific provider."""
        region = self.get_region(symbol)
        provider = self.providers[region]

        logger.info(f"{symbol}: Region={region}, Provider={provider.name()}")
        return provider.fetch_ohlcv(symbol, start, end)
```

## Combining Strategies

Real-world systems often combine multiple strategies:

```python
class ProductionDataFetcher:
    """Production-grade multi-provider fetcher."""

    def __init__(self):
        # Primary providers by asset class
        self.crypto = CoinGeckoProvider()
        self.us_stocks = TiingoProvider(api_key="tiingo_key")
        self.global_stocks = EODHDProvider(api_key="eodhd_key")

        # Fallback provider
        self.fallback = IEXCloudProvider(api_key="iex_key")

        # Rate limit tracking
        self.us_stocks_calls = 0
        self.us_stocks_limit = 1000

    def fetch_ohlcv(self, symbol, start, end, validate=True):
        """Fetch with asset routing, fallback, and validation."""

        # 1. Select primary provider (Strategy 2: Asset Class Routing)
        if self._is_crypto(symbol):
            primary = self.crypto
        elif self._is_us_stock(symbol):
            # Check rate limit (Strategy 3: Rate Limit Aware)
            if self.us_stocks_calls >= self.us_stocks_limit:
                logger.warning("Tiingo limit reached, using fallback")
                primary = self.fallback
            else:
                primary = self.us_stocks
                self.us_stocks_calls += 1
        else:
            primary = self.global_stocks

        # 2. Try primary with fallback (Strategy 1: Primary + Fallback)
        try:
            data = primary.fetch_ohlcv(symbol, start, end)
            provider_used = primary.name()

        except Exception as e:
            logger.warning(f"Primary failed ({e}), trying fallback")
            try:
                data = self.fallback.fetch_ohlcv(symbol, start, end)
                provider_used = self.fallback.name()
            except Exception as e2:
                raise Exception(f"All providers failed: {e}, {e2}")

        # 3. Validate data quality (Strategy 4: Data Quality)
        if validate:
            try:
                self._validate_data(data, symbol)
            except DataValidationError as e:
                logger.error(f"Data validation failed: {e}")
                # Try fallback if primary data was bad
                if provider_used == primary.name():
                    logger.info("Trying fallback due to bad data")
                    data = self.fallback.fetch_ohlcv(symbol, start, end)
                    self._validate_data(data, symbol)  # Validate fallback too

        logger.info(f"{symbol}: Fetched from {provider_used}")
        return data

    def _validate_data(self, df, symbol):
        """Validate OHLCV data."""
        # Check invariants
        if not (df["high"] >= df["low"]).all():
            raise DataValidationError(f"{symbol}: High < Low")
        # ... more validation ...

# Usage
fetcher = ProductionDataFetcher()
data = fetcher.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
```

## Monitoring Multi-Provider Systems

Track provider performance:

```python
class ProviderMonitor:
    """Monitor provider health and performance."""

    def __init__(self):
        self.stats = {}

    def record_fetch(self, provider_name, symbol, success, latency_ms, error=None):
        """Record fetch attempt."""
        if provider_name not in self.stats:
            self.stats[provider_name] = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "latencies": [],
                "errors": [],
            }

        stats = self.stats[provider_name]
        stats["total"] += 1

        if success:
            stats["success"] += 1
            stats["latencies"].append(latency_ms)
        else:
            stats["failed"] += 1
            stats["errors"].append({"symbol": symbol, "error": str(error)})

    def get_provider_stats(self, provider_name):
        """Get statistics for provider."""
        if provider_name not in self.stats:
            return None

        stats = self.stats[provider_name]
        success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0

        return {
            "provider": provider_name,
            "total_calls": stats["total"],
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "recent_errors": stats["errors"][-5:],  # Last 5 errors
        }

    def print_summary(self):
        """Print provider comparison."""
        print("\n" + "=" * 80)
        print("  Provider Performance Summary")
        print("=" * 80)

        for provider_name in self.stats:
            stats = self.get_provider_stats(provider_name)
            print(f"\n{provider_name}:")
            print(f"  Total Calls:   {stats['total_calls']}")
            print(f"  Success Rate:  {stats['success_rate']:.1f}%")
            print(f"  Avg Latency:   {stats['avg_latency_ms']:.0f}ms")

            if stats['recent_errors']:
                print(f"  Recent Errors: {len(stats['recent_errors'])}")
```

## Best Practices

### 1. Plan for Failure

```python
# Always have fallback options
# Never assume a provider will be available
```

### 2. Log Provider Usage

```python
# Track which provider was used for each symbol
# Helps with debugging and cost allocation
logger.info(f"{symbol}: Fetched from {provider_name}")
```

### 3. Validate Cross-Provider

```python
# Periodically validate data from multiple providers
# Catches provider-specific data quality issues
```

### 4. Monitor Costs

```python
# Track API usage and projected costs
# Switch to cheaper providers when hitting limits
```

### 5. Test Failover

```python
# Regularly test fallback paths
# Don't wait for production outage to discover issues
```

## Summary

**Key Strategies**:
1. ✅ **Primary + Fallback** - Resilience against outages
2. ✅ **Asset Class Routing** - Use best provider for each asset type
3. ✅ **Rate Limit Distribution** - Maximize throughput across providers
4. ✅ **Consensus Validation** - Verify data quality across providers
5. ✅ **Cost Optimization** - Use cheapest provider for requirements
6. ✅ **Geographic Routing** - Use providers with best regional coverage

**When to Use Multi-Provider**:
- Production systems requiring high availability
- High-volume data pipelines (>1000 symbols)
- Multi-asset portfolios (stocks + crypto + forex)
- Cost-sensitive operations
- Data quality critical applications

**Trade-offs**:
- **Complexity**: More code, more configuration
- **Cost**: Multiple API subscriptions
- **Latency**: Fallback adds time
- **Consistency**: Different providers may have slightly different data

**Next Steps**:
- Start with Strategy 1 (Primary + Fallback) for simplicity
- Add asset class routing when covering multiple asset types
- Implement consensus validation for critical systems
- Monitor and optimize based on usage patterns

---

**Previous Tutorial**: [04: Data Quality Validation](04_data_quality.md)
**Tutorial Series Complete!** 🎉

**Continue Learning**:
- [Provider Selection Guide](../provider-selection-guide.md)
- [Creating a Provider](../creating_a_provider.md)
- [Contributing to ML4T Data](../../CONTRIBUTING.md)

# ML4T Data Tutorials

Educational tutorials for learning market data concepts and ML4T Data best practices.

## Tutorial Series

### [01: Understanding OHLCV Data](01_understanding_ohlcv.md)
**Target**: Beginners | **Time**: 10 minutes

Learn the fundamentals of OHLCV (Open, High, Low, Close, Volume) data:
- What OHLCV data is and why it matters
- How to interpret candlesticks and price bars
- OHLCV invariants and data quality checks
- Adjusted vs. unadjusted prices
- Working with different frequencies (daily, weekly, monthly)
- Calculating common metrics (returns, volatility, volume analysis)

**Start here if you're new to quantitative finance!**

### [02: Rate Limiting Best Practices](02_rate_limiting.md)
**Target**: Developers | **Time**: 15 minutes

Master API rate limiting to avoid bans and maximize efficiency:
- Understanding rate limit types (RPM, RPD, message credits)
- How ML4T Data's automatic rate limiting works
- Global rate limiters and why they matter
- Best practices for respecting provider limits
- Strategies for high-volume data pipelines
- Provider-specific tips (Alpha Vantage, Tiingo, IEX Cloud, EODHD)

**Essential reading before building production pipelines!**

### [03: Incremental Updates](03_incremental_updates.md)
**Target**: Developers | **Time**: 20 minutes

Reduce API calls by 100-1000x with smart incremental updates:
- The problem with naive full refreshes
- How incremental updates work (check-gap-fetch-merge)
- Using the ProviderUpdater pattern
- Handling gaps and missed updates
- Production patterns (scheduled updates, catch-up, prioritization)
- Batch updates and parallel processing

**Critical for efficient data pipelines!**

### [04: Data Quality Validation](04_data_quality.md)
**Target**: Quant researchers | **Time**: 15 minutes

Ensure data integrity before using it in production:
- Why data quality matters (bad data = lost money)
- OHLCV invariants that must always be true
- Schema validation (columns, types, nulls)
- Anomaly detection (price spikes, zero volume, gaps)
- Provider-specific validation strategies
- Cross-provider data quality checks
- Production validation checklist

**Validate everything, fail loudly, never silently corrupt data!**

### [05: Multi-Provider Strategies](05_multi_provider.md)
**Target**: Advanced users | **Time**: 20 minutes

Build resilient pipelines with multiple data providers:
- Strategy 1: Primary + Fallback (resilience)
- Strategy 2: Asset Class Routing (optimization)
- Strategy 3: Rate Limit Distribution (throughput)
- Strategy 4: Consensus Validation (data quality)
- Strategy 5: Cost Optimization (efficiency)
- Strategy 6: Geographic Routing (coverage)
- Combining strategies for production systems

**For production systems requiring high availability!**

## Learning Path

### For Beginners
1. Start with [01: Understanding OHLCV](01_understanding_ohlcv.md)
2. Read [02: Rate Limiting](02_rate_limiting.md) before heavy usage
3. Implement [03: Incremental Updates](03_incremental_updates.md) to save API calls

### For Researchers
1. Review [01: Understanding OHLCV](01_understanding_ohlcv.md) for fundamentals
2. Master [04: Data Quality](04_data_quality.md) to ensure valid backtests
3. Learn [05: Multi-Provider](05_multi_provider.md) for cross-validation

### For Production Engineers
1. All tutorials are essential reading
2. Focus on [02: Rate Limiting](02_rate_limiting.md) and [03: Incremental Updates](03_incremental_updates.md)
3. Implement patterns from [05: Multi-Provider](05_multi_provider.md) for reliability

## Quick Reference

| Topic | Tutorial | Key Takeaway |
|-------|----------|--------------|
| OHLCV Basics | 01 | High ≥ Low, Open, Close; Volume ≥ 0 |
| Rate Limits | 02 | Use incremental updates, ML4T Data handles limiting automatically |
| Updates | 03 | Incremental mode = 100-1000x fewer API calls |
| Data Quality | 04 | Validate everything, fail loudly, log all issues |
| Multi-Provider | 05 | Primary + Fallback = resilience; Asset routing = optimization |

## Related Documentation

- **Getting Started**: [5-Minute Quickstart](../../README.md#5-minute-quickstart)
- **Provider Selection**: [Provider Selection Guide](../provider-selection-guide.md)
- **Adding Providers**: [Creating a Provider](../creating_a_provider.md)
- **Contributing**: [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Architecture**: [Extending ML4T Data](../extending_ml4t-data.md)

## Examples

See [examples/](../../examples/) directory for runnable code:
- `quickstart.py` - 5-minute Bitcoin data fetch
- `eodhd_example.py` - Global stocks from EODHD
- `finnhub_example.py` - Professional provider usage

## Getting Help

- **Issues**: Report problems at [GitHub Issues](https://github.com/ml4t/data/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/ml4t/data/discussions)
- **Documentation**: Full docs at [User Guide](../user-guide/)

---

**Ready to get started?** Begin with [Tutorial 01: Understanding OHLCV Data](01_understanding_ohlcv.md)!

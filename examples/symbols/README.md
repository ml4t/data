# Symbol Lists for ML4T Book Examples

This directory contains curated symbol lists for use throughout the Machine Learning for Trading (Third Edition) book examples.

## Available Symbol Lists

| File | Symbols | Purpose | Book Tier |
|------|---------|---------|-----------|
| `tech_portfolio_YYYY-MM-DD.txt` | 5 | Quick examples, learning basics | Tier 1 (Free) |
| `sector_etfs_YYYY-MM-DD.txt` | 11 | Sector analysis, diversification | Tier 1 (Free) |
| `nasdaq100_YYYY-MM-DD.txt` | ~100 | Tech-heavy portfolio, sector studies | Tier 2 (Educational) |
| `sp500_full_YYYY-MM-DD.txt` | ~500 | Large-scale backtests, portfolio construction | Tier 2 (Educational) |
| `global_equities_YYYY-MM-DD.txt` | 10 | Multi-market examples (requires EODHD) | Tier 2 (Educational) |

## Understanding Survivorship Bias ⚠️

### What is Survivorship Bias?

**All symbol lists in this directory reflect CURRENT constituents as of the generation date.**

This creates **survivorship bias** in historical analysis:
- If you backtest a 2015-2020 strategy using the 2025 S&P 500 list, you're only testing on companies that survived to 2025
- Delisted companies, bankruptcies, and index removals are excluded
- This artificially inflates backtest performance (you're trading with hindsight)

### Example

Consider a 2015 backtest:
- ✅ Includes: Companies still in S&P 500 in 2025 (survivors)
- ❌ Excludes: Companies that were in S&P 500 in 2015 but were later:
  - Removed from index
  - Acquired/merged
  - Bankrupt/delisted
  - Moved to Russell 2000

**Result**: Your backtest avoids the losers that a real 2015 trader would have encountered.

### When Does This Matter?

**Low Impact** (acceptable for learning):
- **Recent data** (1-3 years) - Minimal constituent changes
- **Concept learning** - Understanding techniques, not production trading
- **Relative comparisons** - Comparing strategies on the same biased dataset
- **Forward testing** - Trading live with current constituents

**High Impact** (use professional data):
- **Long-term backtests** (5+ years) - Significant constituent turnover
- **Production strategies** - Real money depends on accurate simulation
- **Academic research** - Requires rigorous methodology
- **Risk modeling** - Need true historical distributions

### How to Get As-Of Historical Data

For production-grade historical constituent data, use:

1. **DataBento** ($9+/month)
   - Historical S&P 500 membership with as-of dates
   - Full corporate actions history
   - Institutional-grade data quality

2. **Bloomberg Terminal** ($2,000+/month)
   - `MEMB` function for index membership history
   - Complete constituent changes with dates

3. **S&P Dow Jones Indices** (subscription)
   - Official index provider
   - Historical constituent files with effective dates

4. **Compustat/CRSP** (academic/professional)
   - Research-grade databases
   - Survivor-bias-free datasets

## Refreshing Symbol Lists

### Automatic Refresh

Symbol lists are generated with dated filenames (`YYYY-MM-DD`) to track when they were created.

To generate fresh lists:

```bash
# From ml4t-data project root
python scripts/create_symbol_lists.py
```

This will:
- Fetch current S&P 500 constituents from Wikipedia
- Fetch current NASDAQ 100 constituents from Wikipedia
- Generate all 5 symbol list files with today's date
- Output to `examples/symbols/` directory

### Manual Refresh

If Wikipedia scraping fails (403 errors, page structure changes), you can:

1. **Download from reliable sources**:
   - [SPDR S&P 500 ETF (SPY) holdings](https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy)
   - [Invesco QQQ (NASDAQ 100) holdings](https://www.invesco.com/us/financial-products/etfs/holdings?audienceType=Investor&ticker=QQQ)

2. **Extract ticker symbols** (usually column A or B in CSV)

3. **Format as text file**:
   ```
   # S&P 500 Constituents
   # Source: SPDR SPY Holdings
   # Generated: 2025-11-24

   AAPL
   MSFT
   ...
   ```

### Recommended Refresh Frequency

| List | Suggested Refresh | Reasoning |
|------|-------------------|-----------|
| Tech Portfolio | As needed | Manually curated, stable |
| Sector ETFs | Rarely | ETF tickers very stable |
| NASDAQ 100 | Quarterly | ~2-5 changes per quarter |
| S&P 500 | Quarterly | ~5-10 changes per quarter |
| Global Equities | As needed | Manually curated examples |

## Using Symbol Lists in Notebooks

### Loading a Symbol File

```python
from pathlib import Path

def load_symbols(filename: str) -> list[str]:
    """Load symbols from file, ignoring comments and blank lines."""
    symbol_file = Path(__file__).parent / "symbols" / filename
    symbols = []

    with open(symbol_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if line and not line.startswith("#"):
                symbols.append(line)

    return symbols

# Use the most recent dated file
symbols = load_symbols("sp500_full_2025-11-24.txt")
print(f"Loaded {len(symbols)} symbols")
```

### Using with ml4t-data CLI

```yaml
# ml4t-data.yaml
datasets:
  sp500_daily:
    provider: yahoo
    symbols_file: examples/symbols/sp500_full_2025-11-24.txt
    frequency: daily
```

```bash
# Fetch data for all symbols in file
ml4t-data update-all -c ml4t-data.yaml
```

## Symbol File Format

All symbol files follow a consistent format:

```
# Title
# Source: Where symbols came from
# Generated: YYYY-MM-DD
#
# SURVIVORSHIP BIAS WARNING:
# [Standard warning about current constituents]
#
# NOTES:
# - Additional context
# - Use case recommendations

SYMBOL1
SYMBOL2
...
```

**Rules**:
- Lines starting with `#` are comments (ignored)
- Blank lines are ignored
- One symbol per line
- Symbols are uppercase
- No spaces or punctuation (except `.` in SYMBOL.EXCHANGE format)

## Special Symbol Formats

### Yahoo Finance (Default)

Most symbols use standard tickers:
```
AAPL
MSFT
BRK-B    # Note: Periods replaced with hyphens (BRK.B -> BRK-B)
```

### Global Equities (SYMBOL.EXCHANGE)

For multi-market data (requires EODHD or similar):
```
AAPL.US      # US stocks
VOD.LSE      # London Stock Exchange
BMW.DE       # Frankfurt (Deutsche Börse)
7203.T       # Tokyo Stock Exchange
0700.HK      # Hong Kong Stock Exchange
```

## Book Chapter Progression

These symbol lists support a progressive learning path:

| Chapters | List | Data Provider | Reason |
|----------|------|---------------|---------|
| 1-5 | Tech Portfolio (5) | Yahoo Finance | Learn basics, no API limits |
| 3-4 | Sector ETFs (11) | Yahoo Finance | Diversification concepts |
| 6-8 | NASDAQ 100 (100) | Yahoo Finance | Scale up, sector analysis |
| 9-12 | S&P 500 (500) | Yahoo/EODHD | Production-scale portfolios |
| 3, 13+ | Global Equities | EODHD | Multi-market strategies |

## Troubleshooting

### "Symbol not found" errors

**Symptom**: Provider returns 404 or "symbol not found"

**Causes**:
1. Symbol was delisted since list generation
2. Symbol format wrong for provider (e.g., BRK.B vs BRK-B)
3. Symbol recently changed ticker

**Solutions**:
- Remove delisted symbol from your local copy
- Check provider documentation for symbol format
- Refresh symbol list (regenerate with current constituents)

### Stale data warnings

**Symptom**: Using symbol list from >6 months ago

**Impact**: 5-15 constituents may have changed (depending on index)

**Solution**: Regenerate symbol lists quarterly or before major projects

### Wikipedia scraping fails

**Symptom**: Script returns 403 Forbidden or parsing errors

**Solutions**:
1. Check internet connection
2. Try again later (Wikipedia rate limiting)
3. Use manual refresh process (see above)
4. File an issue if persistent

## Contributing

Found issues with symbol lists? Please:

1. Check if symbols are stale (>6 months old)
2. Regenerate lists with current script
3. If issues persist, file a GitHub issue with:
   - Date of symbol list
   - Provider used
   - Error message
   - Symbol(s) affected

## License

Symbol lists are derived from publicly available index constituent data and are provided for educational purposes in accordance with the ML4T book license.

Wikipedia content is used under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

---

**Remember**: For learning and concept exploration, these lists are perfect. For production trading strategies, invest in proper as-of historical constituent data from professional providers.

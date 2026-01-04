# Stress Test: ML4T Data vs Direct yfinance

## Purpose

Test with **200 real tickers** from the Quandl dataset to determine if ML4T Data's YahooFinanceProvider provides actual reliability benefits or is just slower with no gain.

## The Question

> "The 17x slowdown will matter more when you scale. We'll only know with hundreds of tickers if the lower speed actually makes this more reliable or is JUST poor performance."

## Test Design

### Sample
- **200 tickers** from Quandl WIKI dataset (first 200 alphabetically)
- Includes active, delisted, and renamed tickers
- Real-world mix of success/failure scenarios

### Period
- **2024-01-01 to 2024-12-31** (1 year of data)
- ~250 trading days per ticker

### Approaches

**1. Direct yfinance (ml3t style)**
```python
yf_ticker = yf.Ticker(ticker)
hist = yf_ticker.history(start=start, end=end, auto_adjust=False)
time.sleep(0.1)  # 100ms manual delay
```

**2. ML4T Data YahooFinanceProvider**
```python
provider = YahooFinanceProvider(max_requests_per_second=0.5)
df = provider.fetch_ohlcv(ticker, start, end, "daily")
# Automatic rate limiting ~2s per ticker
```

## Metrics Being Measured

### Performance
- Total time
- Average time per ticker
- Throughput (tickers/hour)
- Performance ratio (how much slower)

### Reliability
- Success count
- Failure count
- Success rate
- Failure breakdown (empty, rate limit, network, errors)

### The Verdict

Calculate whether reliability improvement justifies performance cost:

```
Reliability improvement = (ML4T Data_success - yfinance_success) / total * 100
Performance cost = (ML4T Data_time - yfinance_time) / yfinance_time * 100

if reliability_improvement > 2%:
    "Worth it"
elif reliability_improvement > 0%:
    "Marginal benefit"
else:
    "Just slower, no gain"
```

## Expected Results

### Hypothesis 1: ML4T Data is just slower (no benefit)
- Both approaches: similar success rates
- ML4T Data: 10-20x slower
- **Conclusion**: Poor performance, no reliability gain
- **Action**: Increase default rate limit or recommend direct yfinance

### Hypothesis 2: ML4T Data provides reliability
- ML4T Data: 5-10% higher success rate
- ML4T Data: 10-20x slower
- **Conclusion**: Performance cost justified by reliability
- **Action**: Keep defaults, document trade-off

### Hypothesis 3: Direct yfinance gets throttled
- yfinance: many rate limit errors after N tickers
- ML4T Data: smooth throughout
- **Conclusion**: Rate limiting critical at scale
- **Action**: Keep conservative defaults

## Running the Test

```bash
cd /home/stefan/ml4t/software/data
.venv/bin/python examples/stress_test_comparison.py
```

**Expected duration**:
- Direct yfinance: ~20-30 seconds (100ms * 200)
- ML4T Data: ~350-400 seconds (1.75s * 200)
- **Total: ~7-8 minutes**

## Results Location

- **Console output**: Real-time progress and final comparison
- **Log file**: `examples/stress_test_output.log`
- **Detailed results**: `examples/stress_test_results.txt`

## What We'll Learn

1. **Does automatic rate limiting prevent throttling at scale?**
   - If yfinance gets 429 errors after N tickers → Yes
   - If both succeed equally → No benefit

2. **Is the 17x slowdown justified?**
   - If ML4T Data succeeds 10%+ more → Justified
   - If ML4T Data succeeds 1-5% more → Marginal
   - If ML4T Data succeeds equally → Not justified

3. **What's the right default rate limit?**
   - If 0.5 req/sec succeeds 100% → Keep it
   - If 0.5 req/sec still has errors → Too slow, increase it
   - If direct yfinance succeeds 100% → Our limit too conservative

4. **Real-world failure modes**
   - Delisted tickers (both should handle equally)
   - Network errors (ML4T Data should retry better)
   - Rate limiting (ML4T Data should prevent)
   - Invalid symbols (both should handle equally)

## Post-Test Analysis

After test completes, analyze:

### If ML4T Data wins (higher success rate):
- Document reliability benefit percentage
- Calculate cost per additional success (extra seconds per % improvement)
- Update README with "Worth it for production" recommendation

### If both equal:
- ML4T Data is just slower with no benefit
- Increase default rate limit to 1.0 or 2.0 req/sec
- Update README: "Use direct yfinance for small batches, ML4T Data for error handling/logging"

### If yfinance wins (higher success rate):
- Something wrong with ML4T Data implementation
- Debug and fix
- Re-test

## Critical Success Factors

### For ML4T Data to be "worth it":
1. **≥5% higher success rate** OR
2. **Zero rate limit errors while yfinance gets throttled** OR
3. **Better error messages that help debugging**

### If none of above:
- Admit it's just slower
- Recommend direct yfinance for most use cases
- Position ML4T Data for "enterprise features" (logging, monitoring, etc.)

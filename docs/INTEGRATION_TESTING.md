# ML4T Data Integration Testing Guide

## Overview

The ML4T Data integration testing suite provides comprehensive validation of all data providers against real APIs while optimizing for minimal costs and maximum coverage.

## Test Architecture

### Test Categories

1. **Unit Tests** - Mock-based tests for individual components
2. **Integration Tests** - Real API validation with minimal data
3. **Performance Tests** - Baseline measurements and regression detection
4. **Expensive Tests** - Full dataset validation (nightly/manual only)

### Provider Coverage

| Provider | API Type | Cost Model | Test Strategy |
|----------|----------|------------|---------------|
| CryptoCompare | REST | Free tier (10 req/sec) | Single day, 2 symbols |
| Databento | REST | Pay per request | Daily bars, specific contracts |
| OANDA | REST | Free practice account | Hourly bars, major pairs |

## Setup

### Local Development

1. **Install dependencies**:
```bash
uv pip install -e .
uv pip install pytest pytest-cov pytest-asyncio
```

2. **Configure API keys**:
```bash
cp .env.example .env
# Edit .env and add your API keys
source .env
```

3. **Run setup script**:
```bash
./scripts/setup_test_env.sh
```

### CI/CD Configuration

GitHub Actions workflow automatically:
- Runs minimal tests on PRs
- Runs full suite on main branch
- Runs expensive tests nightly
- Tracks performance baselines

## Running Tests

### Quick Test Commands

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run minimal tests (CI mode)
CI=true pytest tests/integration/ -v -m "integration and not expensive"

# Run specific provider
pytest tests/integration/ -v -k cryptocompare

# Run with coverage
pytest tests/integration/ -v --cov=ml4t-data.providers --cov-report=html

# Run performance benchmarks
pytest tests/integration/test_real_api_integration.py::TestRealAPIIntegration::test_performance_baselines -v
```

### Test Markers

- `@pytest.mark.integration` - Requires real API access
- `@pytest.mark.expensive` - High API costs (skip in CI)
- `@pytest.mark.skipif` - Conditional execution based on API keys

## Cost Optimization

### Strategies

1. **Data Minimization**
   - Single day requests (1 bar) for basic tests
   - Maximum 3-day ranges for multi-day tests
   - Limited symbol sets (2-3 per provider)

2. **Smart Test Selection**
   - Skip expensive tests in CI
   - Run full suite only on main branch
   - Nightly runs for comprehensive validation

3. **Caching**
   - Response caching for repeated requests
   - Fixture reuse across test sessions
   - Performance baseline storage

### Cost Estimates

| Provider | Test Type | Requests | Estimated Cost |
|----------|-----------|----------|----------------|
| CryptoCompare | Minimal | 5 | $0 (free tier) |
| CryptoCompare | Full | 20 | $0 (free tier) |
| Databento | Minimal | 3 | ~$0.01 |
| Databento | Full | 10 | ~$0.05 |
| OANDA | All | Unlimited | $0 (practice) |

**Monthly CI Estimate**: < $5 with nightly runs

## Performance Baselines

### Target Metrics

| Provider | Avg Response | Max Response | Throughput |
|----------|--------------|--------------|------------|
| CryptoCompare | < 2s | < 10s | 10 req/s |
| Databento | < 3s | < 15s | 5 req/s |
| OANDA | < 1s | < 5s | 120 req/s |

### Monitoring

Performance is tracked via:
- Test execution times
- API response latencies
- Memory usage patterns
- Error rates

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Check `.env` file exists
   - Verify key format (Databento starts with "db-")
   - Source environment: `source .env`

2. **Rate Limiting**
   - Built-in rate limiters should prevent this
   - If occurs, check provider limits
   - Reduce parallel test execution

3. **Empty Responses**
   - Weekend/holiday dates return empty
   - Invalid symbols return empty
   - Check provider documentation

4. **Test Failures**
   - Verify API keys are valid
   - Check network connectivity
   - Review provider status pages

### Debug Mode

```bash
# Verbose output with logging
pytest tests/integration/ -vv -s --log-cli-level=DEBUG

# Run single test
pytest tests/integration/test_real_api_integration.py::TestRealAPIIntegration::test_cryptocompare_integration -v

# Capture warnings
pytest tests/integration/ -v -W default
```

## CI/CD Integration

### GitHub Actions Secrets

Required secrets:
```
CRYPTOCOMPARE_API_KEY
DATABENTO_API_KEY
OANDA_API_KEY
```

### Workflow Triggers

- **Push to main**: Full integration tests
- **Pull Request**: Minimal tests only
- **Schedule**: Nightly expensive tests
- **Manual**: On-demand full suite

### Test Reports

- Coverage reports uploaded as artifacts
- Performance benchmarks stored
- API usage tracked in logs

## Best Practices

1. **Always use minimal data** for basic validation
2. **Mark expensive tests** with appropriate decorator
3. **Handle empty responses** gracefully
4. **Log API usage** for cost tracking
5. **Cache responses** where appropriate
6. **Document known limitations**
7. **Update baselines** regularly

## Contributing

When adding new providers:

1. Implement provider class inheriting from `BaseProvider`
2. Add integration tests following existing patterns
3. Document API costs and limits
4. Update CI configuration if needed
5. Add to cost optimization strategy

## Monitoring & Alerts

### Metrics to Track

- Test execution time trends
- API error rates
- Coverage percentages
- Cost per test run

### Alert Thresholds

- Coverage drops below 60%
- Performance degrades >50%
- API errors exceed 5%
- Monthly costs exceed $10

## Future Enhancements

- [ ] Response caching layer
- [ ] Parallel test execution optimization
- [ ] Historical performance tracking
- [ ] Automated cost reporting
- [ ] Provider health monitoring
- [ ] Test data generation tools

---

*Last updated: 2025-08-28*
*Maintained by: ML4T Data Development Team*

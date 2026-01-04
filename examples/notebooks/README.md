# ML4T Data Interactive Notebooks

Jupyter notebooks demonstrating ML4T Data usage with real examples and visualizations.

## Available Notebooks

### 1. [Quickstart Notebook](01_quickstart.ipynb) ✅
**Get started in 5 minutes!**

Learn:
- Fetch crypto data (no API key needed!)
- Fetch stock data with free Tiingo API
- Visualize prices and volume
- Calculate returns and volatility
- Compare multiple assets

**Perfect for beginners!**

### 2. Provider Comparison (Coming Soon)
**Compare data quality across providers**

Topics:
- Fetch same symbol from multiple providers
- Compare close prices and volume
- Identify discrepancies
- Measure latency and reliability
- Choose the best provider for your needs

### 3. Incremental Updates (Coming Soon)
**Build efficient data pipelines**

Topics:
- The naive approach vs. incremental updates
- ProviderUpdater pattern demonstration
- Gap detection and backfilling
- Production update scheduling
- Monitoring and alerting

### 4. Multi-Asset Portfolio (Coming Soon)
**Manage diversified portfolios**

Topics:
- Fetch stocks, crypto, and forex
- Asset class routing strategies
- Portfolio analytics and rebalancing
- Risk metrics (volatility, correlation)
- Performance attribution

### 5. Data Quality Validation (Coming Soon)
**Ensure data integrity**

Topics:
- OHLCV invariant validation
- Detecting anomalies (spikes, gaps)
- Cross-provider validation
- Cleaning and filling missing data
- Production data quality monitoring

## Running the Notebooks

### Installation

```bash
# Install ML4T Data with visualization dependencies
pip install ml4t-data matplotlib seaborn jupyter

# Or with all optional dependencies
pip install ml4t-data[viz]
```

### Launch Jupyter

```bash
cd examples/notebooks
jupyter notebook
```

Then open any `.ipynb` file in your browser.

### API Keys

Some notebooks require free API keys:

- **Tiingo** (stocks): Get free key at https://api.tiingo.com/account/api/token
  - 1000 calls/day free tier
  - Set: `export TIINGO_API_KEY="your_key"`

- **EODHD** (global stocks): Get free key at https://eodhd.com/register
  - 500 calls/day free tier
  - Set: `export EODHD_API_KEY="your_key"`

- **CoinGecko** (crypto): No API key needed! ✅

### Tips for Notebooks

1. **Run cells in order** - Each cell builds on previous ones
2. **Restart kernel** if you get errors - `Kernel → Restart & Clear Output`
3. **Save your work** - Changes aren't saved automatically
4. **Experiment!** - Modify the code to explore different symbols/date ranges

## Notebook Features

### Data Fetching
```python
from ml4t-data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()
data = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")
```

### Visualization
```python
import matplotlib.pyplot as plt

df = data.to_pandas()
plt.plot(df['timestamp'], df['close'])
plt.show()
```

### Analysis
```python
# Calculate returns
data = data.with_columns(
    ((data["close"] - data["close"].shift(1)) / data["close"].shift(1) * 100)
    .alias("daily_return")
)

print(f"Avg daily return: {data['daily_return'].mean():.2f}%")
```

## Learning Path

1. **Start with Quickstart** (01_quickstart.ipynb)
   - Learn basic data fetching
   - Understand OHLCV format
   - Create simple visualizations

2. **Read the Tutorials** ([docs/tutorials/](../../docs/tutorials/))
   - Deeper explanations of concepts
   - Best practices and patterns
   - Production-ready code

3. **Explore Advanced Notebooks** (02-05)
   - Provider comparison strategies
   - Production data pipelines
   - Portfolio management
   - Data quality validation

4. **Build Your Own Analysis**
   - Use notebooks as templates
   - Customize for your strategies
   - Share with the community!

## Troubleshooting

### "ModuleNotFoundError: No module named 'ml4t-data'"

```bash
pip install ml4t-data
```

### "No API key found"

Set environment variables:
```bash
export TIINGO_API_KEY="your_key_here"
export EODHD_API_KEY="your_key_here"
```

Or set in notebook:
```python
import os
os.environ["TIINGO_API_KEY"] = "your_key_here"
```

### "Rate limit exceeded"

- You've hit provider's daily limit
- Wait until next day (limits reset at midnight UTC)
- Or upgrade to paid tier
- See [Tutorial 02: Rate Limiting](../../docs/tutorials/02_rate_limiting.md)

### Notebook won't start

```bash
# Reinstall jupyter
pip install --upgrade jupyter notebook

# Or try JupyterLab
pip install jupyterlab
jupyter lab
```

## Contributing

Have a great notebook example? We'd love to see it!

1. Create your notebook in this directory
2. Follow the naming convention: `NN_descriptive_name.ipynb`
3. Add markdown cells explaining what you're doing
4. Test it works from a fresh kernel
5. Submit a PR!

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

## Related Resources

- **[Tutorials](../../docs/tutorials/)** - Step-by-step written guides
- **[Examples](../)** - Python scripts
- **[Provider Selection Guide](../../docs/provider-selection-guide.md)** - Choose providers
- **[API Documentation](../../docs/api/)** - Complete API reference

---

**Happy analyzing!** 📊

# AQR Factor Provider

**Provider**: `AQRFactorProvider`
**Website**: [aqr.com/Insights/Datasets](https://www.aqr.com/Insights/Datasets)
**API Key**: Not required
**Free Tier**: Free (academic use)

---

## Overview

AQR Capital Management provides 16 academic factor datasets for quantitative research, including Quality Minus Junk (QMJ), Betting Against Beta (BAB), and Time Series Momentum (TSMOM).

**Best For**: Factor research, alternative factors, cross-asset strategies

---

## Quick Start

```python
from ml4t.data.providers import AQRFactorProvider

provider = AQRFactorProvider()

# Quality Minus Junk
qmj = provider.fetch("qmj_factors", region="USA")

# Betting Against Beta
bab = provider.fetch("bab_factors")

# Time Series Momentum
tsmom = provider.fetch("tsmom")

provider.close()
```

---

## Available Datasets

### Equity Factors

| Dataset | Description |
|---------|-------------|
| `qmj_factors` | Quality Minus Junk (profitability, growth, safety) |
| `bab_factors` | Betting Against Beta (low-beta outperformance) |
| `hml_devil` | HML Devil (industry-adjusted value) |
| `vme_factors` | Value and Momentum Everywhere |

### Cross-Asset

| Dataset | Description |
|---------|-------------|
| `tsmom` | Time Series Momentum (48 futures, 67 equity indices) |
| `century` | Century of Factor Premia (1920s+) |
| `commodities` | Commodity momentum and carry |

---

## Data Format

- Returns in decimal format (0.01 = 1%)
- Monthly frequency
- Multiple regions: USA, Global, Developed, Emerging

---

## First-Time Setup

AQR data requires initial download (Excel files from AQR website):

```python
# One-time download
AQRFactorProvider.download()

# Then use normally
provider = AQRFactorProvider()
```

---

## Academic Citations

When using AQR data, cite the relevant papers:

- **QMJ**: Asness, Frazzini, and Pedersen (2019)
- **BAB**: Frazzini and Pedersen (2014)
- **TSMOM**: Moskowitz, Ooi, and Pedersen (2012)

---

## See Also

- [AQR Datasets](https://www.aqr.com/Insights/Datasets)
- [Fama-French Provider](fama_french.md)
- [Example Config](../../configs/examples/academic_factors.yaml)
- [Provider README](README.md)

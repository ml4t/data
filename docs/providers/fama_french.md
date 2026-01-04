# Fama-French Provider

**Provider**: `FamaFrenchProvider`
**Website**: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
**API Key**: Not required
**Free Tier**: Free (academic use)

---

## Overview

The Ken French Data Library provides 50+ factor datasets including the famous Fama-French 3-Factor and 5-Factor models.

**Best For**: Academic factors, portfolio research, factor exposure analysis

---

## Quick Start

```python
from ml4t.data.providers import FamaFrenchProvider

provider = FamaFrenchProvider()

# Core factors
ff3 = provider.fetch("ff3", frequency="monthly")   # Mkt-RF, SMB, HML
ff5 = provider.fetch("ff5", frequency="daily")     # +RMW, CMA
mom = provider.fetch("mom", frequency="monthly")   # Momentum

# Combined (Carhart 4-Factor)
carhart = provider.fetch_combined(["ff3", "mom"])

provider.close()
```

---

## Available Datasets

### Core Factors

| Dataset | Factors | Frequency |
|---------|---------|-----------|
| `ff3` | Mkt-RF, SMB, HML | Daily, Monthly |
| `ff5` | Mkt-RF, SMB, HML, RMW, CMA | Daily, Monthly |
| `mom` | MOM (Momentum) | Daily, Monthly |
| `st_rev` | Short-Term Reversal | Monthly |
| `lt_rev` | Long-Term Reversal | Monthly |

### Industry Portfolios

| Dataset | Description |
|---------|-------------|
| `ind_5` | 5 Industries |
| `ind_10` | 10 Industries |
| `ind_12` | 12 Industries |
| `ind_17` | 17 Industries |
| `ind_30` | 30 Industries |
| `ind_48` | 48 Industries (most common) |
| `ind_49` | 49 Industries |

### Size/Value Portfolios

| Dataset | Description |
|---------|-------------|
| `port_size_bm_6` | 6 Size/BM (2x3) |
| `port_size_bm_25` | 25 Size/BM (5x5) |
| `port_size_bm_100` | 100 Size/BM (10x10) |

### International

| Dataset | Region |
|---------|--------|
| `ff3_developed` | Developed Markets |
| `ff3_europe` | Europe |
| `ff3_japan` | Japan |
| `ff3_asia_ex_japan` | Asia Pacific ex Japan |

---

## Data Format

- Returns in decimal format (0.01 = 1%)
- Daily and monthly frequencies
- History from 1926 (US) or 1990 (international)

---

## Combined Factors

Create multi-factor models:

```python
# Carhart 4-Factor (FF3 + Momentum)
ff3_mom = provider.fetch_combined(["ff3", "mom"])

# 6-Factor (FF5 + Momentum)
ff5_mom = provider.fetch_combined(["ff5", "mom"])
```

---

## Academic Citations

- **FF3**: Fama and French (1993)
- **FF5**: Fama and French (2015)
- **Momentum**: Jegadeesh and Titman (1993), Carhart (1997)

---

## See Also

- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [AQR Provider](aqr.md)
- [Example Config](../../configs/examples/academic_factors.yaml)
- [Provider README](README.md)

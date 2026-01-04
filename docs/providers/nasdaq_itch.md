# NASDAQ ITCH Provider

**Provider**: `ITCHSampleProvider`
**Data Source**: NASDAQ TotalView-ITCH sample files
**API Key**: Not required
**Free Tier**: Free (sample data)

---

## Overview

Parses NASDAQ TotalView-ITCH format for tick-level order book data. Uses sample files from NASDAQ.

**Best For**: Order book analysis, market microstructure research

---

## Quick Start

```python
from ml4t.data.providers import ITCHSampleProvider

# Point to ITCH sample file
provider = ITCHSampleProvider(itch_file="~/data/01302019.NASDAQ_ITCH50")

# Parse order book data
df = provider.fetch_ohlcv("AAPL", "2019-01-30", "2019-01-30")

provider.close()
```

---

## Data Format

ITCH files contain tick-level messages:
- Add Order
- Order Executed
- Order Cancelled
- Trade (non-cross)
- System Event

---

## Sample Data

Free sample files available from NASDAQ:
[NASDAQ Historical Data](https://www.nasdaqtrader.com/trader.aspx?id=HistoricalData)

---

## Use Cases

- Order book reconstruction
- Market microstructure analysis
- Execution quality research
- Tick data processing examples

---

## Limitations

- Sample data only (not full historical archive)
- Requires significant processing for large files
- Single-day files

---

## See Also

- [NASDAQ ITCH Specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf)
- [Provider README](README.md)

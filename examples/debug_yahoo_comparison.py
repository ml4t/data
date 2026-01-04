"""Debug the Yahoo comparison to understand the dimension mismatch."""

import sys
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Fetch AAPL data
symbol = "AAPL"
start = "2019-01-01"
end = "2021-12-31"

# Get unadjusted
df_unadj = yf.download(
    symbol, start=start, end=end, auto_adjust=False, actions=False, progress=False
)

# Get adjusted
df_adj = yf.download(symbol, start=start, end=end, auto_adjust=True, actions=False, progress=False)

print("Unadjusted dataframe:")
print(f"  Shape: {df_unadj.shape}")
print(f"  Columns: {df_unadj.columns.tolist()}")
print(f"  Index type: {type(df_unadj.index)}")
print(f"  First row:\n{df_unadj.head(1)}")

print("\nAdjusted dataframe:")
print(f"  Shape: {df_adj.shape}")
print(f"  Columns: {df_adj.columns.tolist()}")
print(f"  Index type: {type(df_adj.index)}")
print(f"  First row:\n{df_adj.head(1)}")

# Extract Close columns
yahoo_adj = df_adj["Close"].values
print("\nYahoo adj array:")
print(f"  Shape: {yahoo_adj.shape}")
print(f"  Type: {type(yahoo_adj)}")
print(f"  Dtype: {yahoo_adj.dtype}")
print(f"  First 3: {yahoo_adj[:3]}")

# Try the comparison
test_arr = np.array([35.0, 36.0, 37.0])  # Simulated "our" values
if len(test_arr) == 3:
    errors = np.abs((test_arr - yahoo_adj[:3]) / yahoo_adj[:3]) * 100
    print("\nTest comparison (first 3):")
    print(f"  Test arr: {test_arr}")
    print(f"  Yahoo: {yahoo_adj[:3]}")
    print(f"  Errors: {errors}")
    print(f"  Num < 0.5%: {(errors < 0.5).sum()}")

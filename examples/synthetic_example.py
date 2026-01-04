"""Synthetic Data Generation with Stochastic Models

This example demonstrates how to generate realistic OHLCV data using
five common stochastic processes from quantitative finance:

1. Geometric Brownian Motion (GBM)
   - The classic model behind Black-Scholes
   - Assumes log-normal returns (returns are normally distributed)
   - Good baseline for equity-like price dynamics

2. Jump-Diffusion (Merton Model)
   - GBM plus occasional large moves (jumps)
   - Creates fat-tailed return distributions
   - Better matches real equity/crypto behavior (crashes, rallies)

3. Ornstein-Uhlenbeck (Mean-Reversion)
   - Prices pulled back toward a long-term mean
   - Used for commodities, interest rates, spread trading
   - Stationary process (bounded variance)

4. Heston (Stochastic Volatility)
   - Two-factor model: price + variance dynamics
   - Captures volatility clustering and leverage effect
   - Industry standard for options pricing

5. GARCH(1,1) (Volatility Clustering)
   - Discrete-time conditional variance model
   - Large moves followed by large moves
   - Foundation of econometric volatility forecasting

Why use synthetic data?
- Testing without network access
- Reproducible experiments (seed parameter)
- Stress testing with different volatility regimes
- Understanding model behavior before using real data
- Educational purposes and book examples

Requirements:
    - ml4t-data installed: pip install ml4t-data
    - No network access needed!

Usage:
    python examples/synthetic_example.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from ml4t.data.providers import SyntheticProvider


def main():
    """Demonstrate synthetic data generation with stochastic models."""
    print("=" * 80)
    print("  Synthetic Data Generation")
    print("  Using Stochastic Processes from Quantitative Finance")
    print("=" * 80)
    print()

    # =========================================================================
    # MODEL 1: GEOMETRIC BROWNIAN MOTION (GBM)
    # =========================================================================
    print("=" * 80)
    print("  Model 1: Geometric Brownian Motion (GBM)")
    print("=" * 80)
    print()
    print("  The foundation of modern option pricing (Black-Scholes).")
    print()
    print("  Mathematical form:")
    print("    dS = μS dt + σS dW")
    print()
    print("  Where:")
    print("    S = stock price")
    print("    μ = drift (expected return)")
    print("    σ = volatility")
    print("    dW = Wiener process increment (random walk)")
    print()
    print("  Key property: Log returns are normally distributed")
    print("    log(S_t / S_0) ~ N((μ - σ²/2)t, σ²t)")
    print()
    print("  Use for: General equity simulation, baseline model")
    print()

    # Generate GBM data
    gbm_provider = SyntheticProvider(
        model="gbm",
        annual_return=0.08,  # 8% expected annual return
        annual_volatility=0.20,  # 20% annual volatility (typical for S&P 500)
        base_price=100.0,
        seed=42,
    )

    gbm_data = gbm_provider.fetch_ohlcv("EQUITY", "2024-01-01", "2024-12-31")
    gbm_returns = gbm_data["close"].pct_change().drop_nulls()

    print(f"  Generated {len(gbm_data)} daily bars")
    print()
    print("  Return Statistics:")
    print(f"    Mean daily return:    {gbm_returns.mean() * 100:+.4f}%")
    print(f"    Daily volatility:     {gbm_returns.std() * 100:.4f}%")
    print(f"    Annualized vol:       {gbm_returns.std() * np.sqrt(252) * 100:.1f}%")
    print(f"    Skewness:             {gbm_returns.skew():.3f}")
    print(f"    Kurtosis:             {gbm_returns.kurtosis():.3f} (Normal = 0)")
    print()
    print("  Price path:")
    print(f"    Start: ${gbm_data['open'][0]:.2f}")
    print(f"    End:   ${gbm_data['close'][-1]:.2f}")
    print(f"    High:  ${gbm_data['high'].max():.2f}")
    print(f"    Low:   ${gbm_data['low'].min():.2f}")
    print()

    # =========================================================================
    # MODEL 2: JUMP-DIFFUSION (MERTON MODEL)
    # =========================================================================
    print("=" * 80)
    print("  Model 2: Jump-Diffusion (Merton Model)")
    print("=" * 80)
    print()
    print("  GBM plus occasional large moves - captures market crashes and rallies.")
    print()
    print("  Mathematical form:")
    print("    dS = μS dt + σS dW + S dJ")
    print()
    print("  Where:")
    print("    dJ = compound Poisson process (random jumps)")
    print("    λ = jump intensity (expected jumps per year)")
    print("    Jump sizes ~ N(0, σ_jump)")
    print()
    print("  Key property: Fat-tailed returns (excess kurtosis)")
    print("    Real markets have more extreme events than GBM predicts")
    print()
    print("  Use for: Equity/crypto with tail risk, stress testing")
    print()

    # Generate jump-diffusion data
    jump_provider = SyntheticProvider(
        model="gbm_jump",
        annual_return=0.08,
        annual_volatility=0.20,
        base_price=100.0,
        seed=42,  # Same seed for comparison
    )

    jump_data = jump_provider.fetch_ohlcv("EQUITY_JUMPY", "2024-01-01", "2024-12-31")
    jump_returns = jump_data["close"].pct_change().drop_nulls()

    print(f"  Generated {len(jump_data)} daily bars")
    print()
    print("  Return Statistics (compare to GBM above):")
    print(f"    Mean daily return:    {jump_returns.mean() * 100:+.4f}%")
    print(f"    Daily volatility:     {jump_returns.std() * 100:.4f}%")
    print(f"    Annualized vol:       {jump_returns.std() * np.sqrt(252) * 100:.1f}%")
    print(f"    Skewness:             {jump_returns.skew():.3f}")
    print(
        f"    Kurtosis:             {jump_returns.kurtosis():.3f} (GBM was {gbm_returns.kurtosis():.3f})"
    )
    print()
    print("  Extreme moves:")
    print(f"    Max daily gain:  {jump_returns.max() * 100:+.2f}%")
    print(f"    Max daily loss:  {jump_returns.min() * 100:+.2f}%")
    print(f"    Days > 3σ:       {(jump_returns.abs() > 3 * jump_returns.std()).sum()}")
    print()

    # =========================================================================
    # MODEL 3: ORNSTEIN-UHLENBECK (MEAN-REVERSION)
    # =========================================================================
    print("=" * 80)
    print("  Model 3: Ornstein-Uhlenbeck (Mean-Reversion)")
    print("=" * 80)
    print()
    print("  Price is pulled back toward a long-term equilibrium level.")
    print()
    print("  Mathematical form:")
    print("    dX = κ(θ - X) dt + σ dW")
    print()
    print("  Where:")
    print("    X = log-price or spread")
    print("    κ = speed of mean reversion (higher = faster reversion)")
    print("    θ = long-term mean level")
    print("    σ = volatility")
    print()
    print("  Key property: Stationary process")
    print("    Variance is bounded: Var(X) = σ² / (2κ)")
    print("    Half-life of shocks: t_1/2 = ln(2) / κ")
    print()
    print("  Use for: Commodities, interest rates, pairs trading, spreads")
    print()

    # Generate mean-reverting data
    mr_provider = SyntheticProvider(
        model="mean_revert",
        annual_volatility=0.15,  # Lower vol typical for commodities
        base_price=100.0,  # Equilibrium price
        seed=42,
    )

    mr_data = mr_provider.fetch_ohlcv("COMMODITY", "2024-01-01", "2024-12-31")
    mr_returns = mr_data["close"].pct_change().drop_nulls()

    print(f"  Generated {len(mr_data)} daily bars")
    print()
    print("  Return Statistics:")
    print(f"    Mean daily return:    {mr_returns.mean() * 100:+.4f}%")
    print(f"    Daily volatility:     {mr_returns.std() * 100:.4f}%")
    print(
        f"    Return autocorr(1):   {mr_returns.to_numpy()[1:].dot(mr_returns.to_numpy()[:-1]) / (len(mr_returns) - 1) / mr_returns.var():.3f}"
    )
    print()
    print("  Price bounded around equilibrium:")
    print(f"    Start: ${mr_data['open'][0]:.2f}")
    print(f"    End:   ${mr_data['close'][-1]:.2f}")
    print(f"    Mean:  ${mr_data['close'].mean():.2f}")
    print(f"    Std:   ${mr_data['close'].std():.2f}")
    print()

    # =========================================================================
    # MODEL 4: HESTON (STOCHASTIC VOLATILITY)
    # =========================================================================
    print("=" * 80)
    print("  Model 4: Heston (Stochastic Volatility)")
    print("=" * 80)
    print()
    print("  Two-factor model with time-varying volatility and leverage effect.")
    print()
    print("  Mathematical form (coupled SDEs):")
    print("    dS = μS dt + √v S dW_S         (price dynamics)")
    print("    dv = κ(θ - v) dt + ξ√v dW_v    (variance dynamics)")
    print()
    print("  Where:")
    print("    S = stock price")
    print("    v = instantaneous variance (stochastic!)")
    print("    κ = mean reversion speed of variance")
    print("    θ = long-term variance (0.04 = 20% annual vol)")
    print("    ξ = volatility of variance (vol-of-vol)")
    print("    ρ = correlation(dW_S, dW_v)")
    print()
    print("  The LEVERAGE EFFECT:")
    print("    When ρ < 0 (typically -0.5 to -0.8):")
    print("    - Price drops → variance increases")
    print("    - Price rises → variance decreases")
    print("    This matches real equity behavior (fear > greed)")
    print()
    print("  Use for: Options pricing, volatility surface modeling, VIX dynamics")
    print()

    # Generate Heston data
    heston_provider = SyntheticProvider(
        model="heston",
        annual_return=0.08,
        annual_volatility=0.20,  # Used for OHLC generation
        base_price=100.0,
        seed=42,
        # Heston-specific parameters
        heston_kappa=2.0,  # Variance mean-reverts with half-life ~4 months
        heston_theta=0.04,  # Long-term variance (20% vol)^2
        heston_xi=0.3,  # Vol-of-vol
        heston_rho=-0.7,  # Strong leverage effect
    )

    heston_data = heston_provider.fetch_ohlcv("HESTON", "2024-01-01", "2024-12-31")
    heston_returns = heston_data["close"].pct_change().drop_nulls()

    print(f"  Generated {len(heston_data)} daily bars")
    print()
    print("  Return Statistics:")
    print(f"    Mean daily return:    {heston_returns.mean() * 100:+.4f}%")
    print(f"    Daily volatility:     {heston_returns.std() * 100:.4f}%")
    print(f"    Annualized vol:       {heston_returns.std() * np.sqrt(252) * 100:.1f}%")
    print(f"    Skewness:             {heston_returns.skew():.3f}")
    print(f"    Kurtosis:             {heston_returns.kurtosis():.3f}")
    print()
    print("  Leverage effect check (does vol increase after drops?):")
    # Calculate rolling volatility and check correlation with prior returns
    returns_arr = heston_returns.to_numpy()
    # Simple check: compare vol in up vs down periods
    down_days = returns_arr[:-1] < 0
    vol_after_down = np.std(returns_arr[1:][down_days])
    vol_after_up = np.std(returns_arr[1:][~down_days])
    print(f"    Vol after down days:  {vol_after_down * 100:.3f}%")
    print(f"    Vol after up days:    {vol_after_up * 100:.3f}%")
    print(f"    Ratio (leverage):     {vol_after_down / vol_after_up:.2f}x")
    print()

    # =========================================================================
    # MODEL 5: GARCH(1,1) (VOLATILITY CLUSTERING)
    # =========================================================================
    print("=" * 80)
    print("  Model 5: GARCH(1,1) (Volatility Clustering)")
    print("=" * 80)
    print()
    print("  Discrete-time conditional variance model - the econometric workhorse.")
    print()
    print("  Mathematical form:")
    print("    r_t = μ + σ_t ε_t,  ε_t ~ N(0,1)")
    print("    σ²_t = ω + α r²_{t-1} + β σ²_{t-1}")
    print()
    print("  Where:")
    print("    r_t = return at time t")
    print("    σ²_t = conditional variance")
    print("    ω = constant term (baseline variance)")
    print("    α = ARCH term (reaction to recent shocks)")
    print("    β = GARCH term (persistence of variance)")
    print()
    print("  Key properties:")
    print("    - Stationarity requires: α + β < 1")
    print("    - Long-run variance: σ²_∞ = ω / (1 - α - β)")
    print("    - Persistence: α + β (typically 0.9-0.99 for daily data)")
    print("    - Higher α = faster reaction to news")
    print("    - Higher β = longer memory")
    print()
    print("  Use for: Volatility forecasting, risk management, VaR")
    print()

    # Generate GARCH data
    garch_provider = SyntheticProvider(
        model="garch",
        annual_return=0.08,
        annual_volatility=0.20,
        base_price=100.0,
        seed=42,
        # GARCH-specific parameters
        garch_alpha=0.10,  # 10% weight on recent squared return
        garch_beta=0.85,  # 85% weight on previous variance
        # Persistence = 0.95 (typical for daily equity returns)
    )

    garch_data = garch_provider.fetch_ohlcv("GARCH", "2024-01-01", "2024-12-31")
    garch_returns = garch_data["close"].pct_change().drop_nulls()

    print(f"  Generated {len(garch_data)} daily bars")
    print()
    print("  Return Statistics:")
    print(f"    Mean daily return:    {garch_returns.mean() * 100:+.4f}%")
    print(f"    Daily volatility:     {garch_returns.std() * 100:.4f}%")
    print(f"    Annualized vol:       {garch_returns.std() * np.sqrt(252) * 100:.1f}%")
    print(f"    Skewness:             {garch_returns.skew():.3f}")
    print(f"    Kurtosis:             {garch_returns.kurtosis():.3f}")
    print()
    print("  Volatility clustering check:")
    garch_arr = garch_returns.to_numpy()
    abs_returns = np.abs(garch_arr)
    # Autocorrelation of absolute returns (measures clustering)
    autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
    print(f"    Autocorr of |returns|: {autocorr:.3f}")
    print("    (Positive = volatility clustering, GBM ≈ 0)")
    print()

    # =========================================================================
    # COMPARING THE MODELS
    # =========================================================================
    print("=" * 80)
    print("  Model Comparison")
    print("=" * 80)
    print()
    print("  Which model to use?")
    print()
    print("  ┌─────────────────┬────────────────────────────────────────────┐")
    print("  │ Model           │ Best For                                   │")
    print("  ├─────────────────┼────────────────────────────────────────────┤")
    print("  │ GBM             │ Baseline equity, index funds, general use  │")
    print("  │ Jump-Diffusion  │ Individual stocks, crypto, stress testing  │")
    print("  │ Mean-Reversion  │ Commodities, rates, spreads, pairs trading │")
    print("  │ Heston          │ Options, volatility products, VIX dynamics │")
    print("  │ GARCH           │ Risk management, VaR, volatility forecast  │")
    print("  └─────────────────┴────────────────────────────────────────────┘")
    print()
    print("  Return distribution comparison:")
    print()
    print(f"  {'Model':<18} {'Vol (ann)':<12} {'Skew':<10} {'Kurtosis':<10}")
    print(f"  {'-' * 18} {'-' * 12} {'-' * 10} {'-' * 10}")
    print(
        f"  {'GBM':<18} {gbm_returns.std() * np.sqrt(252) * 100:>8.1f}%   {gbm_returns.skew():>8.3f}  {gbm_returns.kurtosis():>8.3f}"
    )
    print(
        f"  {'Jump-Diffusion':<18} {jump_returns.std() * np.sqrt(252) * 100:>8.1f}%   {jump_returns.skew():>8.3f}  {jump_returns.kurtosis():>8.3f}"
    )
    print(
        f"  {'Mean-Reversion':<18} {mr_returns.std() * np.sqrt(252) * 100:>8.1f}%   {mr_returns.skew():>8.3f}  {mr_returns.kurtosis():>8.3f}"
    )
    print(
        f"  {'Heston':<18} {heston_returns.std() * np.sqrt(252) * 100:>8.1f}%   {heston_returns.skew():>8.3f}  {heston_returns.kurtosis():>8.3f}"
    )
    print(
        f"  {'GARCH':<18} {garch_returns.std() * np.sqrt(252) * 100:>8.1f}%   {garch_returns.skew():>8.3f}  {garch_returns.kurtosis():>8.3f}"
    )
    print()

    # =========================================================================
    # PRACTICAL EXAMPLES
    # =========================================================================
    print("=" * 80)
    print("  Practical Examples")
    print("=" * 80)
    print()

    # High-volatility crypto
    print("  1. Crypto-like asset (high vol + jumps):")
    crypto = SyntheticProvider(
        model="gbm_jump",
        annual_return=0.50,  # 50% drift (bull market)
        annual_volatility=0.80,  # 80% vol (typical crypto)
        base_price=50000.0,  # BTC-like
        seed=123,
    )
    crypto_data = crypto.fetch_ohlcv("CRYPTO", "2024-01-01", "2024-12-31")
    crypto_returns = crypto_data["close"].pct_change().drop_nulls()
    print(f"     Start: ${crypto_data['open'][0]:,.0f} -> End: ${crypto_data['close'][-1]:,.0f}")
    print(f"     Annual vol: {crypto_returns.std() * np.sqrt(252) * 100:.0f}%")
    print()

    # Low-volatility bond-like
    print("  2. Bond-like asset (low vol, mean-reverting):")
    bond = SyntheticProvider(
        model="mean_revert",
        annual_return=0.03,  # 3% yield
        annual_volatility=0.05,  # 5% vol
        base_price=100.0,  # Par value
        seed=456,
    )
    bond_data = bond.fetch_ohlcv("BOND", "2024-01-01", "2024-12-31")
    print(f"     Range: ${bond_data['low'].min():.2f} - ${bond_data['high'].max():.2f}")
    print(f"     Very stable around ${bond_data['close'].mean():.2f}")
    print()

    # Minute data
    print("  3. High-frequency minute bars:")
    hf = SyntheticProvider(model="gbm", annual_volatility=0.25, seed=789)
    hf_data = hf.fetch_ohlcv("HF_TEST", "2024-01-15", "2024-01-16", frequency="minute")
    print(f"     Generated {len(hf_data)} minute bars for one day")
    print(f"     Sample: {hf_data.head(5)}")
    print()

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================
    print("=" * 80)
    print("  Reproducibility")
    print("=" * 80)
    print()
    print("  Same seed = same data (critical for experiments):")
    print()

    run1 = SyntheticProvider(seed=42).fetch_ohlcv("TEST", "2024-01-01", "2024-03-31")
    run2 = SyntheticProvider(seed=42).fetch_ohlcv("TEST", "2024-01-01", "2024-03-31")

    print(f"    Run 1 final close: ${run1['close'][-1]:.6f}")
    print(f"    Run 2 final close: ${run2['close'][-1]:.6f}")
    print(f"    Match: {run1['close'].equals(run2['close'])}")
    print()
    print("  Different seed = different path:")
    run3 = SyntheticProvider(seed=999).fetch_ohlcv("TEST", "2024-01-01", "2024-03-31")
    print(f"    Seed 42:  ${run1['close'][-1]:.2f}")
    print(f"    Seed 999: ${run3['close'][-1]:.2f}")
    print()

    # =========================================================================
    # IMPLEMENTATION NOTES
    # =========================================================================
    print("=" * 80)
    print("  Implementation Notes")
    print("=" * 80)
    print()
    print("  The SyntheticProvider generates:")
    print()
    print("  1. Close prices from the chosen stochastic model")
    print("  2. Open prices with small gaps from previous close")
    print("  3. High/Low that bracket Open and Close realistically")
    print("  4. Volume correlated with absolute returns (vol-volume relation)")
    print("  5. Volume autocorrelation (volume clusters)")
    print()
    print("  Parameters are scaled from annual to per-bar:")
    print("    σ_bar = σ_annual / sqrt(252 * bars_per_day)")
    print("    μ_bar = μ_annual / (252 * bars_per_day)")
    print()
    print("  Model-specific defaults:")
    print()
    print("  Jump-Diffusion:")
    print("    λ = 5 jumps per year (expected)")
    print("    Jump size ~ N(0, 3%)")
    print()
    print("  Mean-Reversion (OU):")
    print("    κ = 2.0 (half-life ~ 4 months)")
    print()
    print("  Heston (stochastic volatility):")
    print("    κ = 2.0 (variance mean-reversion speed)")
    print("    θ = 0.04 (long-run variance = 20% vol)")
    print("    ξ = 0.3 (vol-of-vol)")
    print("    ρ = -0.7 (leverage effect)")
    print()
    print("  GARCH(1,1):")
    print("    α = 0.10 (ARCH term - reaction to shocks)")
    print("    β = 0.85 (GARCH term - persistence)")
    print("    α + β = 0.95 (typical for daily equity data)")
    print()

    # Clean up
    gbm_provider.close()
    jump_provider.close()
    mr_provider.close()
    heston_provider.close()
    garch_provider.close()
    crypto.close()
    bond.close()
    hf.close()

    print("=" * 80)
    print("  Done! Try experimenting with different parameters.")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as err:
        print(f"\n\nError: {err}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

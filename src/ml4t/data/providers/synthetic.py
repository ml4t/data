"""Synthetic data provider for generating realistic OHLCV data.

This provider generates synthetic market data using financial models,
useful for:
- Testing without network access
- Book examples and tutorials
- Backtesting framework development
- Educational purposes

Models:
- GBM (Geometric Brownian Motion): Classic price dynamics
- GBM with jumps (Merton): Adds occasional large moves (fat tails)
- Mean-reverting (OU): For commodities, pairs trading
- Heston: Stochastic volatility with leverage effect
- GARCH(1,1): Discrete-time volatility clustering

The generated data includes realistic:
- OHLC relationships (High >= Open, Close, Low)
- Volume patterns (higher on volatile days)
- Intraday patterns (for minute data)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

import numpy as np
import polars as pl
import structlog

from ml4t.data.providers.base import BaseProvider
from ml4t.data.synthetic import (
    generate_ohlc_from_close,
    generate_timestamps,
    generate_volume,
    get_bars_per_day,
)

logger = structlog.get_logger()


class SyntheticProvider(BaseProvider):
    """Provider for generating synthetic OHLCV data.

    Generates realistic market data using configurable financial models.
    No network access required - useful for testing and examples.

    Parameters
    ----------
    model : str, default="gbm"
        Price model:
        - "gbm": Geometric Brownian Motion (classic, constant volatility)
        - "gbm_jump": GBM with Poisson jumps (fat tails)
        - "mean_revert": Ornstein-Uhlenbeck (commodities, pairs trading)
        - "heston": Stochastic volatility with leverage effect
        - "garch": GARCH(1,1) volatility clustering
    annual_return : float, default=0.08
        Expected annual return (drift)
    annual_volatility : float, default=0.20
        Annual volatility (standard deviation of returns)
    seed : int, optional
        Random seed for reproducibility
    base_price : float, default=100.0
        Starting price level
    base_volume : float, default=1_000_000
        Base daily volume
    heston_kappa : float, default=2.0
        Heston: Mean reversion speed of variance
    heston_theta : float, default=0.04
        Heston: Long-term variance (0.04 = 20% annual vol)
    heston_xi : float, default=0.3
        Heston: Volatility of variance (vol-of-vol)
    heston_rho : float, default=-0.7
        Heston: Correlation between price and variance shocks
        (negative = leverage effect: vol rises when price falls)
    garch_omega : float, default=0.000002
        GARCH: Constant term (sets long-run variance floor)
    garch_alpha : float, default=0.1
        GARCH: ARCH term weight (reaction to recent shocks)
    garch_beta : float, default=0.85
        GARCH: GARCH term weight (persistence of variance)

    Example
    -------
    >>> provider = SyntheticProvider(seed=42)
    >>> df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")
    >>> df.head()

    >>> # Heston model with strong leverage effect
    >>> heston = SyntheticProvider(model="heston", heston_rho=-0.8, seed=42)
    >>> df = heston.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

    >>> # GARCH model with high persistence
    >>> garch = SyntheticProvider(model="garch", garch_alpha=0.1, garch_beta=0.88, seed=42)
    >>> df = garch.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")
    """

    # No rate limiting needed for synthetic data
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1000, 1.0)

    # Trading days per year
    TRADING_DAYS = 252

    def __init__(
        self,
        model: Literal["gbm", "gbm_jump", "mean_revert", "heston", "garch"] = "gbm",
        annual_return: float = 0.08,
        annual_volatility: float = 0.20,
        seed: int | None = None,
        base_price: float = 100.0,
        base_volume: float = 1_000_000,
        rate_limit: tuple[int, float] | None = None,
        # Heston parameters
        heston_kappa: float = 2.0,
        heston_theta: float = 0.04,
        heston_xi: float = 0.3,
        heston_rho: float = -0.7,
        # GARCH parameters
        garch_omega: float = 0.000002,
        garch_alpha: float = 0.1,
        garch_beta: float = 0.85,
    ) -> None:
        """Initialize synthetic provider."""
        self.model = model
        self.annual_return = annual_return
        self.annual_volatility = annual_volatility
        self.seed = seed
        self.base_price = base_price
        self.base_volume = base_volume

        # Heston model parameters
        self.heston_kappa = heston_kappa  # Mean reversion speed
        self.heston_theta = heston_theta  # Long-term variance
        self.heston_xi = heston_xi  # Vol of vol
        self.heston_rho = heston_rho  # Price-variance correlation

        # GARCH model parameters
        self.garch_omega = garch_omega  # Constant term
        self.garch_alpha = garch_alpha  # ARCH term
        self.garch_beta = garch_beta  # GARCH term

        # Validate GARCH stationarity
        if model == "garch" and (garch_alpha + garch_beta) >= 1.0:
            logger.warning(
                "GARCH parameters may be non-stationary",
                alpha_plus_beta=garch_alpha + garch_beta,
            )

        # Random state for reproducibility
        self._rng = np.random.default_rng(seed)

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "synthetic"

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    def _generate_gbm_returns(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate returns using Geometric Brownian Motion.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step in years (1/252 for daily)

        Returns
        -------
        np.ndarray
            Log returns
        """
        # Daily parameters
        mu = self.annual_return * dt
        sigma = self.annual_volatility * np.sqrt(dt)

        # Generate standard normal shocks
        shocks = self._rng.standard_normal(n_steps)

        # GBM log returns
        returns = (mu - 0.5 * sigma**2) + sigma * shocks

        return returns

    def _generate_gbm_jump_returns(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate returns using GBM with occasional jumps.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step in years

        Returns
        -------
        np.ndarray
            Log returns with jumps
        """
        # Base GBM returns
        returns = self._generate_gbm_returns(n_steps, dt)

        # Add jumps (Poisson process)
        jump_intensity = 5.0  # Expected 5 jumps per year
        jump_prob = jump_intensity * dt
        jump_mean = 0.0  # Average jump is zero (can be up or down)
        jump_std = 0.03  # 3% jump size std

        # Generate jumps
        jump_mask = self._rng.random(n_steps) < jump_prob
        jump_sizes = self._rng.normal(jump_mean, jump_std, n_steps)

        returns[jump_mask] += jump_sizes[jump_mask]

        return returns

    def _generate_mean_revert_returns(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate returns from mean-reverting (Ornstein-Uhlenbeck) log-prices.

        The OU process is applied to log-prices (not returns), ensuring
        prices actually revert to a mean level:

            d(log S) = κ(θ - log S) dt + σ dW

        Where:
            log S = log-price
            θ = long-term mean log-price (log of base_price)
            κ = speed of mean reversion
            σ = volatility

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step in years

        Returns
        -------
        np.ndarray
            Log-returns derived from mean-reverting log-prices
        """
        # Mean reversion parameters
        kappa = 2.0  # Speed of mean reversion (half-life = ln(2)/κ ≈ 0.35 years)
        theta = np.log(self.base_price)  # Long-term mean log-price
        sigma = self.annual_volatility * np.sqrt(dt)

        # Generate log-price path using OU process
        log_prices = np.zeros(n_steps + 1)
        log_prices[0] = theta  # Start at equilibrium

        for t in range(n_steps):
            shock = self._rng.standard_normal()
            # OU: d(logS) = κ(θ - logS)dt + σdW
            log_prices[t + 1] = log_prices[t] + kappa * (theta - log_prices[t]) * dt + sigma * shock

        # Return log-returns (differences in log-prices)
        # These get cumulated in _fetch_and_transform_data to reconstruct prices
        returns = np.diff(log_prices)
        return returns

    def _generate_heston_returns(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate returns using Heston stochastic volatility model.

        The Heston model has two coupled SDEs:
            dS = μS dt + √v S dW_S        (price dynamics)
            dv = κ(θ - v) dt + ξ√v dW_v   (variance dynamics)

        Where:
            S = price
            v = instantaneous variance
            μ = drift (annual_return)
            κ = mean reversion speed of variance (heston_kappa)
            θ = long-term variance (heston_theta)
            ξ = volatility of variance (heston_xi)
            ρ = correlation between dW_S and dW_v (heston_rho)

        The correlation ρ < 0 creates the "leverage effect":
        when prices drop, volatility tends to increase.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step in years

        Returns
        -------
        np.ndarray
            Log returns from Heston model
        """
        kappa = self.heston_kappa
        theta = self.heston_theta
        xi = self.heston_xi
        rho = self.heston_rho
        mu = self.annual_return

        # Initialize variance at long-term mean
        v = np.zeros(n_steps + 1)
        v[0] = theta

        # Generate correlated Brownian motions
        # dW_S and dW_v with correlation rho
        z1 = self._rng.standard_normal(n_steps)
        z2 = self._rng.standard_normal(n_steps)

        # Correlated shocks
        dW_S = z1 * np.sqrt(dt)
        dW_v = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        # Simulate variance process with reflection to keep v >= 0
        # (full truncation scheme for numerical stability)
        for t in range(n_steps):
            v_positive = max(v[t], 0)  # Ensure non-negative
            v[t + 1] = v[t] + kappa * (theta - v_positive) * dt + xi * np.sqrt(v_positive) * dW_v[t]

        # Generate log-returns using simulated variance
        # d(log S) = (μ - v/2) dt + √v dW_S
        sqrt_v = np.sqrt(np.maximum(v[:-1], 0))
        returns = (mu - v[:-1] / 2) * dt + sqrt_v * dW_S

        return returns

    def _generate_garch_returns(self, n_steps: int, dt: float) -> np.ndarray:
        """Generate returns using GARCH(1,1) model.

        GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
        models volatility clustering - large moves tend to follow large moves:

            r_t = μ dt + σ_t ε_t
            σ²_t = ω + α r²_{t-1} + β σ²_{t-1}

        Where:
            r_t = return at time t
            σ²_t = conditional variance at time t
            ω = constant term (garch_omega)
            α = ARCH term (garch_alpha) - reaction to recent shocks
            β = GARCH term (garch_beta) - persistence of variance

        Stationarity requires α + β < 1.
        Long-run variance = ω / (1 - α - β)

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step in years

        Returns
        -------
        np.ndarray
            Log returns from GARCH(1,1) model
        """
        alpha = self.garch_alpha
        beta = self.garch_beta
        mu = self.annual_return * dt

        # Scale omega for the time step
        # Long-run daily variance = annual_vol² / 252
        daily_var = (self.annual_volatility**2) / self.TRADING_DAYS
        # Set omega to achieve this long-run variance
        omega_scaled = daily_var * (1 - alpha - beta)

        # Initialize at unconditional variance
        sigma2 = np.zeros(n_steps + 1)
        sigma2[0] = daily_var

        returns = np.zeros(n_steps)

        # Generate innovations
        epsilon = self._rng.standard_normal(n_steps)

        for t in range(n_steps):
            # Generate return
            returns[t] = mu + np.sqrt(sigma2[t]) * epsilon[t]

            # Update variance for next period
            sigma2[t + 1] = omega_scaled + alpha * returns[t] ** 2 + beta * sigma2[t]

            # Floor variance to prevent numerical issues
            sigma2[t + 1] = max(sigma2[t + 1], 1e-10)

        return returns

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Generate synthetic OHLCV data.

        Parameters
        ----------
        symbol : str
            Symbol name (used as seed modifier for reproducibility)
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        frequency : str
            Data frequency

        Returns
        -------
        pl.DataFrame
            Synthetic OHLCV data
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Generate timestamps using shared utility
        timestamps = generate_timestamps(start_dt, end_dt, frequency)
        n_steps = len(timestamps)

        if n_steps == 0:
            return self._create_empty_dataframe()

        logger.info(
            f"Generating {n_steps} synthetic bars",
            symbol=symbol,
            model=self.model,
            frequency=frequency,
        )

        # Modify RNG state based on symbol for reproducibility
        if self.seed is not None:
            symbol_hash = hash(symbol) % (2**31)
            self._rng = np.random.default_rng(self.seed + symbol_hash)

        # Calculate time step in years
        bars_per_day = get_bars_per_day(frequency)
        dt = 1.0 / (self.TRADING_DAYS * bars_per_day)

        # Generate returns based on model
        if self.model == "gbm":
            returns = self._generate_gbm_returns(n_steps, dt)
        elif self.model == "gbm_jump":
            returns = self._generate_gbm_jump_returns(n_steps, dt)
        elif self.model == "mean_revert":
            returns = self._generate_mean_revert_returns(n_steps, dt)
        elif self.model == "heston":
            returns = self._generate_heston_returns(n_steps, dt)
        elif self.model == "garch":
            returns = self._generate_garch_returns(n_steps, dt)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Convert log returns to prices
        log_prices = np.cumsum(np.concatenate([[np.log(self.base_price)], returns]))
        closes = np.exp(log_prices[1:])  # Skip initial price

        # Generate OHLC using shared utility
        daily_vol = self.annual_volatility / np.sqrt(self.TRADING_DAYS * bars_per_day)
        opens, highs, lows = generate_ohlc_from_close(closes, daily_vol, rng=self._rng)

        # Generate volume using shared utility
        volume = generate_volume(returns, base_volume=self.base_volume, rng=self._rng)

        # Create DataFrame
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            }
        )

        # Ensure correct types
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

        return df

    def get_available_symbols(self) -> list[str]:
        """Return example synthetic symbols.

        Returns
        -------
        list[str]
            List of suggested synthetic symbol names
        """
        return [
            "SYNTH",
            "SYNTH_TECH",
            "SYNTH_BOND",
            "SYNTH_COMMODITY",
            "SYNTH_CRYPTO",
            "SYNTH_FOREX",
        ]

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the random number generator.

        Parameters
        ----------
        seed : int, optional
            New seed value. If None, uses original seed.
        """
        self.seed = seed if seed is not None else self.seed
        self._rng = np.random.default_rng(self.seed)

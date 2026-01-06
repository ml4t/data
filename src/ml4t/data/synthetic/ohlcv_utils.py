"""Shared utilities for generating realistic OHLCV data.

These functions are used by both:
- SyntheticProvider (classical stochastic models)
- LearnedSyntheticProvider (trained generative models)

The goal is to convert any return series into realistic OHLCV data
with proper high/low relationships, volume patterns, and timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def returns_to_prices(
    returns: np.ndarray,
    base_price: float = 100.0,
    log_returns: bool = True,
) -> np.ndarray:
    """Convert return series to price series.

    Parameters
    ----------
    returns : np.ndarray
        Return series of shape (n_steps,) or (n_steps, n_assets)
    base_price : float, default=100.0
        Starting price level
    log_returns : bool, default=True
        If True, returns are log returns. If False, simple returns.

    Returns
    -------
    np.ndarray
        Price series starting from base_price

    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.015])
    >>> prices = returns_to_prices(returns, base_price=100.0)
    >>> print(prices)  # [101.00, 99.00, 100.50] approximately
    """
    if log_returns:
        # Log returns: price = base * exp(cumsum(log_returns))
        if returns.ndim == 1:
            log_prices = np.concatenate([[np.log(base_price)], returns])
            prices = np.exp(np.cumsum(log_prices))[1:]
        else:
            # Multi-asset case
            log_prices = np.zeros((returns.shape[0] + 1, returns.shape[1]))
            log_prices[0] = np.log(base_price)
            log_prices[1:] = returns
            prices = np.exp(np.cumsum(log_prices, axis=0))[1:]
    else:
        # Simple returns: price = base * cumprod(1 + returns)
        if returns.ndim == 1:
            prices = base_price * np.cumprod(1 + returns)
        else:
            prices = base_price * np.cumprod(1 + returns, axis=0)

    return prices


def generate_ohlc_from_close(
    closes: np.ndarray,
    volatility: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic Open, High, Low from Close prices.

    Uses the fact that intrabar price movement follows patterns
    where typically Open != Close and High/Low bracket both.

    Parameters
    ----------
    closes : np.ndarray
        Close prices of shape (n_steps,)
    volatility : float
        Daily volatility for scaling (e.g., 0.02 for 2%)
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Open, High, Low arrays

    Notes
    -----
    The algorithm:
    1. Opens have small gaps from previous close (~10% of daily vol)
    2. High/Low extend beyond max/min of Open/Close
    3. Up days have more extension above, down days more below
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(closes)

    # Generate opening gaps (small random gap from previous close)
    gap_std = volatility * 0.1  # Gap is ~10% of daily vol
    gaps = rng.normal(0, gap_std, n)

    opens = np.zeros(n)
    opens[0] = closes[0] * (1 + gaps[0])
    opens[1:] = closes[:-1] * (1 + gaps[1:])

    # Generate High/Low using intraday range
    # Range is typically 1-2x daily volatility
    range_factor = rng.uniform(0.5, 1.5, n) * volatility

    # Determine if up or down day
    is_up_day = closes > opens

    highs = np.zeros(n)
    lows = np.zeros(n)

    for i in range(n):
        bar_max = max(opens[i], closes[i])
        bar_min = min(opens[i], closes[i])

        # Add extension above max and below min
        extension = closes[i] * range_factor[i]

        # On up days, more extension above; on down days, more below
        if is_up_day[i]:
            highs[i] = bar_max + extension * rng.uniform(0.3, 0.7)
            lows[i] = bar_min - extension * rng.uniform(0.1, 0.4)
        else:
            highs[i] = bar_max + extension * rng.uniform(0.1, 0.4)
            lows[i] = bar_min - extension * rng.uniform(0.3, 0.7)

        # Ensure high >= max(open, close) and low <= min(open, close)
        highs[i] = max(highs[i], bar_max)
        lows[i] = min(lows[i], bar_min)

    return opens, highs, lows


def generate_volume(
    returns: np.ndarray,
    base_volume: float = 1_000_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate realistic volume correlated with absolute returns.

    Higher volatility typically means higher volume (volatility-volume relationship).

    Parameters
    ----------
    returns : np.ndarray
        Return series of shape (n_steps,)
    base_volume : float, default=1_000_000
        Base daily volume
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    np.ndarray
        Volume series

    Notes
    -----
    Volume has:
    - Lognormal base noise
    - Correlation with absolute returns (big moves = high volume)
    - Autocorrelation (volume clusters)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(returns)

    # Base volume with some noise
    base_noise = rng.lognormal(0, 0.3, n)

    # Volume increases with absolute returns (volatility-volume relationship)
    abs_returns = np.abs(returns)
    return_std = np.std(returns)
    # Scale by realized vol or use ones if no variation
    return_factor = 1 + 5 * (abs_returns / return_std) if return_std > 0 else np.ones(n)

    # Add some autocorrelation in volume (volume clusters)
    volume_ar = np.zeros(n)
    volume_ar[0] = 1.0
    for t in range(1, n):
        volume_ar[t] = 0.7 * volume_ar[t - 1] + 0.3 * rng.uniform(0.5, 1.5)

    volume = base_volume * base_noise * return_factor * volume_ar

    # Ensure positive integers
    return np.maximum(volume, 1000).astype(np.float64)


def generate_timestamps(
    start_dt: datetime,
    end_dt: datetime,
    frequency: str,
) -> list[datetime]:
    """Generate trading timestamps for a date range.

    Parameters
    ----------
    start_dt : datetime
        Start datetime (should be timezone-aware)
    end_dt : datetime
        End datetime (should be timezone-aware)
    frequency : str
        Data frequency: "daily", "hourly", "minute", etc.

    Returns
    -------
    list[datetime]
        List of timestamps (weekends excluded for equity-style data)

    Notes
    -----
    - Daily: One timestamp per trading day at 16:00 (market close)
    - Intraday: Bars during 9:30-16:00 trading hours
    - Weekends are excluded
    """
    timestamps = []
    minutes_per_bar = _get_minutes_per_bar(frequency)

    if frequency.lower() in ["daily", "1day"]:
        # Daily: one timestamp per day at market close (16:00)
        current = start_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        while current <= end_dt:
            # Skip weekends
            if current.weekday() < 5:
                timestamps.append(current)
            current += timedelta(days=1)
    else:
        # Intraday: generate bars during trading hours
        current_day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_day <= end_dt:
            if current_day.weekday() < 5:  # Skip weekends
                # Trading hours: 9:30 - 16:00 (simplified)
                bar_time = current_day.replace(hour=9, minute=30)
                end_time = current_day.replace(hour=16, minute=0)

                while bar_time <= end_time:
                    if start_dt <= bar_time <= end_dt:
                        timestamps.append(bar_time)
                    bar_time += timedelta(minutes=minutes_per_bar)

            current_day += timedelta(days=1)

    return timestamps


def _get_minutes_per_bar(frequency: str) -> int:
    """Get minutes per bar for given frequency."""
    freq_lower = frequency.lower()
    mapping = {
        "minute": 1,
        "1minute": 1,
        "5minute": 5,
        "15minute": 15,
        "30minute": 30,
        "hourly": 60,
        "1hour": 60,
        "4hour": 240,
        "daily": 1440,
        "1day": 1440,
    }
    return mapping.get(freq_lower, 1440)


def get_bars_per_day(frequency: str) -> int:
    """Get number of bars per trading day for given frequency.

    Parameters
    ----------
    frequency : str
        Data frequency

    Returns
    -------
    int
        Number of bars per day
    """
    freq_lower = frequency.lower()
    if freq_lower in ["daily", "1day"]:
        return 1
    elif freq_lower in ["hourly", "1hour"]:
        return 24  # Full day for crypto-like
    elif freq_lower in ["4hour"]:
        return 6
    elif freq_lower in ["minute", "1minute"]:
        return 390  # Standard equity trading hours
    elif freq_lower in ["5minute"]:
        return 78
    elif freq_lower in ["15minute"]:
        return 26
    elif freq_lower in ["30minute"]:
        return 13
    else:
        return 1

"""
Monte Carlo price-path simulation — Geometric Brownian Motion.

Used to build the "Forecast Fan" 3D surface (day x path x price) and a 2D
percentile fan chart in the AI Forecast dashboard page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gbm_price_paths(
    spot: float,
    mu: float,
    sigma: float,
    horizon_days: int = 60,
    n_paths: int = 60,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate GBM price paths.

    Args:
        spot:         Current price (path starting value).
        mu:           Annualized drift (e.g. historical mean return * 252).
        sigma:        Annualized volatility (e.g. historical return std * sqrt(252)).
        horizon_days: Number of trading days to simulate forward.
        n_paths:      Number of simulated paths.
        seed:         RNG seed for reproducibility.

    Returns:
        Array of shape (n_paths, horizon_days + 1); column 0 == spot.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = spot
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    for t in range(1, horizon_days + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * z)

    return paths


def forecast_fan_percentiles(
    paths: np.ndarray,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """
    Percentile bands across simulated paths, one row per day-offset.

    Returns a DataFrame indexed 0..horizon_days with one column per percentile,
    e.g. "p5", "p25", "p50", "p75", "p95".
    """
    data = {f"p{p}": np.percentile(paths, p, axis=0) for p in percentiles}
    return pd.DataFrame(data, index=np.arange(paths.shape[1]))

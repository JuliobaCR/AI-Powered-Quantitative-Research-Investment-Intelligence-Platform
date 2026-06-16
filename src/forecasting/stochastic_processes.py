"""
Advanced Stochastic Process Simulations for Monte Carlo Pricing.

Implements three industry-standard models beyond plain GBM:
  - Heston (1993): stochastic volatility with mean-reverting variance
  - Merton (1976): jump-diffusion with compound Poisson jumps
  - CEV: Constant Elasticity of Variance (local-vol model)

All functions return price path arrays of shape (n_paths, horizon_days+1),
matching the interface of gbm_price_paths() in monte_carlo.py so the
dashboard can treat all processes uniformly.

References:
  - Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options."
  - Merton, R.C. (1976). "Option Pricing when Underlying Stock Returns are
    Discontinuous." Journal of Financial Economics 3(1–2), 125–144.
  - MIT Quant Finance course notes (stochastic calculus section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def heston_price_paths(
    spot: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon_days: int = 60,
    n_paths: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Heston stochastic volatility model (Euler-Maruyama discretization).

    dS = mu * S * dt + sqrt(V) * S * dW_S
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    corr(dW_S, dW_V) = rho (leverage effect: typically rho < 0 for equities)

    Full reflection scheme applied to V to keep it non-negative.

    Args:
        spot:         Current spot price.
        mu:           Annual drift (risk-neutral: use r; real-world: use hist. mean).
        v0:           Initial instantaneous variance (sigma_0^2).
        kappa:        Mean-reversion speed of variance.
        theta:        Long-run variance (long-run vol = sqrt(theta)).
        xi:           Vol-of-vol (volatility of the variance process).
        rho:          Correlation between spot and vol Brownian motions.
        horizon_days: Forecast horizon in trading days.
        n_paths:      Number of simulation paths.
        seed:         RNG seed for reproducibility.

    Returns:
        ndarray of shape (n_paths, horizon_days+1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    S = np.empty((n_paths, horizon_days + 1))
    V = np.empty((n_paths, horizon_days + 1))
    S[:, 0] = spot
    V[:, 0] = max(v0, 1e-8)

    sqrt_corr = np.sqrt(max(1.0 - rho**2, 0.0))

    for t in range(1, horizon_days + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_s = z1
        z_v = rho * z1 + sqrt_corr * z2

        v_curr = np.maximum(V[:, t - 1], 1e-8)
        sqrt_v = np.sqrt(v_curr)

        # Full-reflection discretization (prevents negative variance)
        V[:, t] = np.abs(
            v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * np.sqrt(dt) * z_v
        )
        S[:, t] = S[:, t - 1] * np.exp(
            (mu - 0.5 * v_curr) * dt + sqrt_v * np.sqrt(dt) * z_s
        )

    return S


def jump_diffusion_price_paths(
    spot: float,
    mu: float,
    sigma: float,
    lam: float = 0.10,
    mu_j: float = -0.05,
    sigma_j: float = 0.08,
    horizon_days: int = 60,
    n_paths: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Merton (1976) Jump-Diffusion model.

    dS/S = (mu - lambda*k) * dt + sigma * dW + J * dN
    where:
        N ~ Poisson(lambda)               (jump arrival process)
        ln(1+J) ~ Normal(mu_j, sigma_j^2) (log-normal jump size)
        k = E[J] = exp(mu_j + 0.5*sigma_j^2) - 1  (compensator)

    Args:
        spot:         Current spot price.
        mu:           Annual drift (risk-neutral or real-world).
        sigma:        Annual diffusion volatility (excluding jumps).
        lam:          Annual Poisson jump intensity (expected jumps per year).
        mu_j:         Mean of log-jump size (log-space).
        sigma_j:      Std dev of log-jump size.
        horizon_days: Forecast horizon in trading days.
        n_paths:      Number of simulation paths.
        seed:         RNG seed.

    Returns:
        ndarray of shape (n_paths, horizon_days+1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    # Jump compensator: E[J] = exp(mu_j + 0.5*sigma_j^2) - 1
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift = (mu - lam * k - 0.5 * sigma**2) * dt
    diffusion_std = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = spot

    for t in range(1, horizon_days + 1):
        z = rng.standard_normal(n_paths)
        # Number of jumps in this timestep per path
        n_jumps = rng.poisson(lam * dt, n_paths)
        # Aggregate log-jump size for paths with jumps
        jump_log_returns = np.zeros(n_paths)
        jump_mask = n_jumps > 0
        if jump_mask.any():
            for path_i in np.where(jump_mask)[0]:
                nj = int(n_jumps[path_i])
                jump_log_returns[path_i] = float(rng.normal(mu_j, sigma_j, nj).sum())

        log_ret = drift + diffusion_std * z + jump_log_returns
        paths[:, t] = paths[:, t - 1] * np.exp(log_ret)

    return paths


def cev_price_paths(
    spot: float,
    mu: float,
    sigma: float,
    beta: float = 0.5,
    horizon_days: int = 60,
    n_paths: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Constant Elasticity of Variance (CEV) model.

    dS = mu * S * dt + sigma * S^beta * dW

    Special cases:
        beta = 1  → GBM (lognormal)
        beta = 0  → Absolute diffusion (normal)
        beta < 1  → Implied vol skew (higher vol for lower prices — equity leverage effect)

    Args:
        spot:         Current spot price.
        mu:           Annual drift.
        sigma:        CEV scale parameter (units depend on beta).
        beta:         Elasticity parameter (0 ≤ beta ≤ 1).
        horizon_days: Forecast horizon in trading days.
        n_paths:      Number of simulation paths.
        seed:         RNG seed.

    Returns:
        ndarray of shape (n_paths, horizon_days+1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = spot

    for t in range(1, horizon_days + 1):
        z = rng.standard_normal(n_paths)
        S_prev = np.maximum(paths[:, t - 1], 1e-8)
        dS = mu * S_prev * dt + sigma * (S_prev ** beta) * np.sqrt(dt) * z
        paths[:, t] = np.maximum(S_prev + dS, 1e-6)

    return paths


def fit_heston_params(returns: pd.Series) -> dict:
    """
    Quick calibration of Heston parameters from historical return series
    via method-of-moments:
        v0, theta  ← from annualized realized variance
        xi         ← from volatility-of-volatility (rolling std of rolling vol)
        kappa      ← fixed at 2.0 (moderate mean reversion)
        rho        ← from correlation of returns and rolling vol (leverage effect)

    Returns dict ready for heston_price_paths() keyword expansion.
    Provides sensible defaults if data is insufficient (<30 observations).
    """
    r = returns.dropna()
    default_sigma = 0.20

    if len(r) < 30:
        v = default_sigma**2
        return {"v0": v, "kappa": 2.0, "theta": v, "xi": 0.30, "rho": -0.70}

    sigma_ann = float(r.std() * np.sqrt(252))
    v0 = sigma_ann**2

    # Rolling 21-day realized volatility
    rolling_vol = r.rolling(21).std().dropna() * np.sqrt(252)
    if len(rolling_vol) >= 10:
        theta = float(rolling_vol.mean()**2)
        xi = float(rolling_vol.std())
        # Leverage effect: correlation between returns and changes in vol
        vol_changes = rolling_vol.diff().dropna()
        ret_aligned = r.reindex(vol_changes.index)
        if len(vol_changes) > 5:
            corr = float(ret_aligned.corr(vol_changes))
            rho = float(np.clip(corr if np.isfinite(corr) else -0.7, -0.95, -0.05))
        else:
            rho = -0.70
    else:
        theta = v0
        xi = 0.30
        rho = -0.70

    return {
        "v0": round(max(v0, 1e-6), 6),
        "kappa": 2.0,
        "theta": round(max(theta, 1e-6), 6),
        "xi": round(max(xi, 0.05), 3),
        "rho": round(rho, 3),
    }


def fit_jump_params(returns: pd.Series) -> dict:
    """
    Estimate Merton Jump-Diffusion parameters from return series using
    a simple outlier-separation approach:
        - Classify returns > 3σ as "jumps"
        - Diffusion sigma from non-jump returns
        - lambda, mu_j, sigma_j from jump returns

    Returns dict ready for jump_diffusion_price_paths() keyword expansion.
    """
    r = returns.dropna()
    if len(r) < 60:
        return {"sigma": 0.20, "lam": 0.10, "mu_j": -0.05, "sigma_j": 0.08}

    threshold = 3.0 * float(r.std())
    jump_mask = r.abs() > threshold

    if jump_mask.sum() < 3:
        return {
            "sigma": round(float(r.std() * np.sqrt(252)), 4),
            "lam": 0.10,
            "mu_j": -0.05,
            "sigma_j": 0.08,
        }

    jump_returns = r[jump_mask]
    diffusion_returns = r[~jump_mask]

    sigma = round(float(diffusion_returns.std() * np.sqrt(252)), 4) if len(diffusion_returns) > 5 else 0.18
    lam = round(float(jump_mask.mean() * 252), 3)
    mu_j = round(float(np.log1p(jump_returns).mean()), 4)
    sigma_j = round(float(np.log1p(jump_returns).std()), 4) if len(jump_returns) > 2 else 0.08

    return {
        "sigma": max(sigma, 0.05),
        "lam": max(lam, 0.01),
        "mu_j": mu_j,
        "sigma_j": max(sigma_j, 0.02),
    }

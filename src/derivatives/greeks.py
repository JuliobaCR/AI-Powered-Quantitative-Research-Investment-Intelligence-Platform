"""
Options Greeks Calculator — Black-Scholes.

Computes: Delta, Gamma, Vega, Theta, Rho for calls and puts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_greeks(
    S: float,      # Spot price
    K: float,      # Strike
    T: float,      # Time to expiry in years
    r: float,      # Risk-free rate
    sigma: float,  # Implied volatility
    option_type: str = "call",
) -> dict[str, float]:
    """Return all BS Greeks for a European option."""
    if T <= 0 or sigma <= 0:
        return {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)
    Phi_nd1 = norm.cdf(-d1)
    Phi_nd2 = norm.cdf(-d2)

    gamma = phi_d1 / (S * sigma * np.sqrt(T))
    vega = S * phi_d1 * np.sqrt(T) / 100  # per 1% move in vol

    if option_type == "call":
        delta = Phi_d1
        theta = (
            -S * phi_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * Phi_d2
        ) / 365
        rho = K * T * np.exp(-r * T) * Phi_d2 / 100
        price = S * Phi_d1 - K * np.exp(-r * T) * Phi_d2
    else:
        delta = Phi_d1 - 1
        theta = (
            -S * phi_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * Phi_nd2
        ) / 365
        rho = -K * T * np.exp(-r * T) * Phi_nd2 / 100
        price = K * np.exp(-r * T) * Phi_nd2 - S * Phi_nd1

    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "vega": round(vega, 4),
        "theta": round(theta, 4),
        "rho": round(rho, 4),
    }


def greeks_surface(
    spot: float,
    sigma: float,
    r: float = 0.0525,
    option_type: str = "call",
    strike_pct_range: tuple[float, float] = (0.70, 1.30),
    n_strikes: int = 25,
    expirations_days: tuple[int, ...] = (7, 14, 30, 60, 90, 120, 180, 270, 360),
) -> dict[str, np.ndarray]:
    """
    Build Strike x Expiry grids of Black-Scholes Greeks for 3D visualization.

    Returns a dict with 1D arrays "strikes" and "expirations" (days), plus
    2D arrays (shape = [len(expirations), len(strikes)]) for "price",
    "delta", "gamma", "vega", "theta", and "rho".
    """
    strikes = np.linspace(spot * strike_pct_range[0], spot * strike_pct_range[1], n_strikes)
    expirations = np.array(expirations_days, dtype=float)

    grids = {g: np.zeros((len(expirations), len(strikes))) for g in
             ["price", "delta", "gamma", "vega", "theta", "rho"]}

    for i, days in enumerate(expirations):
        T = days / 365.0
        for j, K in enumerate(strikes):
            result = black_scholes_greeks(spot, K, T, r, sigma, option_type)
            for g in grids:
                grids[g][i, j] = result[g]

    return {
        "strikes": strikes,
        "expirations": expirations,
        **grids,
    }

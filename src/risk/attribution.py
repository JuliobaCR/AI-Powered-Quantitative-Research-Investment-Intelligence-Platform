"""
Portfolio Risk Attribution.

Decomposes portfolio risk into per-asset and per-factor contributions
using industry-standard approaches:

  - Component VaR: each asset's contribution to total portfolio VaR
    (sums exactly to portfolio VaR under parametric assumptions)
  - Marginal VaR: sensitivity of portfolio VaR to small position increases
  - Volatility Attribution: each asset's % contribution to portfolio sigma
  - Factor Risk Decomposition: how much Fama-French factors explain
    portfolio-level systematic variance

All functions accept weight dicts and return dicts / DataFrames that
match the platform's defensive pattern (empty containers on failure).

References:
  - Jorion, P. "Value at Risk" (3rd ed.) — component VaR chapter.
  - Roncalli, T. "Risk-Based Portfolios" — risk contribution framework.
  - Goldman Sachs / gs-quant risk decomposition methodology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


# ── Component VaR ─────────────────────────────────────────────────────────────

def component_var(
    weights: dict[str, float],
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Parametric Component VaR for each asset.

    Component VaR_i = w_i * beta_i * VaR_portfolio
    where beta_i = Cov(r_i, r_p) / Var(r_p)

    Property: sum of Component VaR == portfolio VaR (exact decomposition).

    Args:
        weights:    ticker → portfolio weight (should sum to 1).
        returns:    DataFrame of daily returns, columns = tickers.
        confidence: VaR confidence level (e.g. 0.95).

    Returns:
        dict: ticker → Component VaR (daily %, positive = contribution to loss).
    """
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return {}

    w = np.array([weights[t] for t in tickers])
    ret_df = returns[tickers].dropna()
    if len(ret_df) < 20:
        return {t: 0.0 for t in tickers}

    z = abs(_norm.ppf(1 - confidence))
    cov = ret_df.cov().values  # daily covariance

    port_var_daily = float(w @ cov @ w)
    port_vol_daily = float(np.sqrt(max(port_var_daily, 0)))
    port_var_pct = z * port_vol_daily * 100  # daily VaR %

    if port_vol_daily < 1e-10:
        return {t: 0.0 for t in tickers}

    # Beta of each asset to the portfolio
    Sigma_w = cov @ w  # Cov(r_i, r_portfolio) for each i
    betas = Sigma_w / port_var_daily

    comp_var = w * betas * port_var_pct  # w_i * beta_i * VaR_p
    return {t: round(float(cv), 4) for t, cv in zip(tickers, comp_var)}


# ── Marginal VaR ──────────────────────────────────────────────────────────────

def marginal_var(
    weights: dict[str, float],
    returns: pd.DataFrame,
    confidence: float = 0.95,
    delta_w: float = 0.01,
) -> dict[str, float]:
    """
    Numerical marginal VaR: change in portfolio VaR per 1% increase in position.

    Marginal VaR_i = (VaR(w + delta_e_i) − VaR(w)) / delta_w

    Useful for identifying which position adds the most risk per unit of weight.

    Args:
        weights:    ticker → portfolio weight.
        returns:    DataFrame of daily returns, columns = tickers.
        confidence: VaR confidence level.
        delta_w:    Weight bump for numerical differentiation.

    Returns:
        dict: ticker → Marginal VaR (daily %, change per 1% weight increase).
    """
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return {}

    w_base = np.array([weights[t] for t in tickers])
    ret_df = returns[tickers].dropna()
    if len(ret_df) < 20:
        return {t: 0.0 for t in tickers}

    z = abs(_norm.ppf(1 - confidence))
    cov = ret_df.cov().values

    def _port_var(w_vec: np.ndarray) -> float:
        port_var = float(w_vec @ cov @ w_vec)
        return z * np.sqrt(max(port_var, 0)) * 100  # daily VaR %

    base_var = _port_var(w_base)
    result = {}
    for i, ticker in enumerate(tickers):
        w_bump = w_base.copy()
        w_bump[i] += delta_w
        w_bump = w_bump / w_bump.sum()  # renormalize to sum=1
        bumped_var = _port_var(w_bump)
        result[ticker] = round(float((bumped_var - base_var) / delta_w), 4)

    return result


# ── Volatility Attribution ────────────────────────────────────────────────────

def volatility_contribution(
    weights: dict[str, float],
    returns: pd.DataFrame,
) -> dict[str, float]:
    """
    Percentage of portfolio variance attributable to each asset.

    Contribution_i = w_i * (Σw)_i / (w'Σw)

    Returns:
        dict: ticker → % share of total portfolio variance (sums to ~100%).
    """
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return {}

    w = np.array([weights[t] for t in tickers])
    ret_df = returns[tickers].dropna()
    if len(ret_df) < 20:
        return {t: 0.0 for t in tickers}

    cov = ret_df.cov().values
    port_var = float(w @ cov @ w)
    if port_var < 1e-12:
        return {t: 0.0 for t in tickers}

    contrib = w * (cov @ w) / port_var  # contribution to variance
    return {t: round(float(c) * 100, 2) for t, c in zip(tickers, contrib)}


# ── Factor Risk Decomposition ─────────────────────────────────────────────────

def factor_risk_decomposition(
    weights: dict[str, float],
    factor_betas: dict[str, dict[str, float]],
    factor_returns: pd.DataFrame,
    factor_columns: list[str],
) -> dict[str, float]:
    """
    Decompose portfolio variance into Fama-French factor contributions.

    Portfolio-level factor exposure: B = Σ w_i * β_i (vector)
    Systematic variance: V_sys = B' * F * B  where F = factor cov matrix
    Factor contribution_k = B_k^2 * F_{kk} / V_sys  (diagonal approximation)

    Args:
        weights:         ticker → portfolio weight.
        factor_betas:    ticker → {factor → beta} (from run_factor_regression).
        factor_returns:  DataFrame of factor daily returns (from fetch_ff5_factors).
        factor_columns:  List of factor names in order.

    Returns:
        dict: factor → % contribution to systematic portfolio variance.
        Empty dict if inputs are insufficient.
    """
    tickers = [t for t in weights if t in factor_betas]
    if not tickers or factor_returns.empty:
        return {}

    # Portfolio-level beta vector: B = Σ w_i * β_i
    B = np.zeros(len(factor_columns))
    for ticker in tickers:
        w = weights.get(ticker, 0.0)
        betas_i = factor_betas[ticker]
        for j, f in enumerate(factor_columns):
            B[j] += w * float(betas_i.get(f, 0.0))

    # Factor covariance matrix (annualized)
    ff = factor_returns[factor_columns].dropna()
    if len(ff) < 20:
        return {}
    F = ff.cov().values * 252  # annualized

    total_sys_var = float(B @ F @ B)
    if total_sys_var < 1e-12:
        return {f: 0.0 for f in factor_columns}

    # Diagonal approximation: contribution from factor k = B_k^2 * F_{kk}
    contrib = {}
    for j, f in enumerate(factor_columns):
        c = float(B[j]**2 * F[j, j]) / total_sys_var * 100
        contrib[f] = round(c, 2)

    return contrib


# ── Full attribution report ───────────────────────────────────────────────────

def full_attribution_report(
    weights: dict[str, float],
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> dict:
    """
    Convenience wrapper: returns Component VaR, Marginal VaR, and Vol Attribution
    in a single dict for the dashboard.
    """
    return {
        "component_var": component_var(weights, returns, confidence),
        "marginal_var": marginal_var(weights, returns, confidence),
        "vol_contribution": volatility_contribution(weights, returns),
    }

"""
Portfolio Optimization Engine.

Strategies:
  - Equal Weight
  - Mean-Variance (Markowitz)
  - Risk Parity
  - Minimum Variance
  - Black-Litterman (simplified)

Uses cvxpy for convex optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
RISK_FREE = 0.0525
TRADING_DAYS = 252


@dataclass
class PortfolioResult:
    tickers: list[str]
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    strategy: str

    def to_dict(self) -> dict:
        return {
            "tickers": self.tickers,
            "weights": {t: round(float(w), 4) for t, w in zip(self.tickers, self.weights)},
            "expected_return_pct": round(self.expected_return * 100, 2),
            "expected_volatility_pct": round(self.expected_volatility * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "strategy": self.strategy,
        }


def optimize_portfolio(
    returns: pd.DataFrame,
    strategy: str = "mean_variance",
) -> PortfolioResult:
    """
    Optimize portfolio weights.

    Args:
        returns:  DataFrame of daily returns, columns = tickers.
        strategy: "equal_weight" | "mean_variance" | "risk_parity" | "min_variance"
    """
    tickers = list(returns.columns)
    n = len(tickers)
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS

    if strategy == "equal_weight":
        w = np.ones(n) / n
    elif strategy == "min_variance":
        w = _min_variance(cov.values)
    elif strategy == "risk_parity":
        w = _risk_parity(cov.values)
    elif strategy == "mean_variance":
        w = _mean_variance(mu.values, cov.values)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    port_ret = float(w @ mu.values)
    port_vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (port_ret - RISK_FREE) / port_vol if port_vol > 0 else 0.0

    return PortfolioResult(
        tickers=tickers,
        weights=w,
        expected_return=port_ret,
        expected_volatility=port_vol,
        sharpe_ratio=sharpe,
        strategy=strategy,
    )


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 60,
) -> pd.DataFrame:
    """
    Compute the efficient frontier.
    Returns DataFrame with columns: Return, Volatility, Sharpe.
    """
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n = len(mu)

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets, sharpes = [], [], []

    for target in target_returns:
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, cov.values))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mu.values @ w == target,
        ]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status == "optimal" and w.value is not None:
            wv = np.array(w.value)
            vol = float(np.sqrt(wv @ cov.values @ wv))
            ret = float(mu.values @ wv)
            vols.append(vol * 100)
            rets.append(ret * 100)
            sharpes.append((ret - RISK_FREE) / vol if vol > 0 else 0)

    return pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})


def _mean_variance(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    obj = cp.Maximize(ret - 0.5 * risk)
    constraints = [cp.sum(w) == 1, w >= 0.02, w <= 0.50]
    cp.Problem(obj, constraints).solve(solver=cp.ECOS, verbose=False)
    return np.array(w.value) if w.value is not None else np.ones(n) / n


def _min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    cp.Problem(obj, constraints).solve(solver=cp.ECOS, verbose=False)
    return np.array(w.value) if w.value is not None else np.ones(n) / n


def _risk_parity(cov: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Equal risk contribution via sequential quadratic programming."""
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def risk_contrib_diff(w):
        port_var = w @ cov @ w
        mrc = cov @ w  # marginal risk contribution
        rc = w * mrc
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    def grad(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        d_rc_dw = np.diag(mrc) + np.diag(w) @ cov
        return 2 * d_rc_dw.T @ (rc - target)

    result = minimize(
        risk_contrib_diff,
        w0,
        jac=grad,
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        bounds=[(0.01, 0.60)] * n,
        tol=tol,
    )
    w = result.x
    return w / w.sum()

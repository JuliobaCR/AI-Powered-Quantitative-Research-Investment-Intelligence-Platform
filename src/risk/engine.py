"""
Risk Management Engine.

Computes:
  - VaR & CVaR (historical + parametric)
  - Sharpe, Sortino, Calmar ratios
  - Maximum Drawdown + recovery
  - Beta vs benchmark
  - Tail Risk metrics
  - Portfolio-level risk decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Value at Risk (positive number = potential loss).

    Args:
        returns:    Daily return series.
        confidence: e.g. 0.95 for 95% VaR.
        method:     "historical" | "parametric".
    """
    r = returns.dropna()
    if method == "historical":
        return float(-np.percentile(r, (1 - confidence) * 100))
    else:  # parametric
        mu, sigma = r.mean(), r.std()
        z = stats.norm.ppf(1 - confidence)
        return float(-(mu + z * sigma))


def compute_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall) — mean of worst (1-conf)% days."""
    r = returns.dropna()
    var = compute_var(r, confidence, method="historical")
    tail = r[r <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0525) -> float:
    r = returns.dropna()
    excess = r.mean() * 252 - risk_free_rate
    vol = r.std() * np.sqrt(252)
    return float(excess / vol) if vol > 0 else 0.0


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0525) -> float:
    r = returns.dropna()
    excess = r.mean() * 252 - risk_free_rate
    downside = r[r < 0].std() * np.sqrt(252)
    return float(excess / downside) if downside > 0 else 0.0


def compute_max_drawdown(prices: pd.Series) -> tuple[float, int]:
    """
    Returns (max_drawdown_pct, recovery_days).
    max_drawdown_pct is negative.
    """
    cum = (1 + prices.pct_change().fillna(0)).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    trough_idx = drawdown.idxmin()
    recovery_series = cum.loc[trough_idx:]
    recovered = recovery_series[recovery_series >= rolling_max.loc[trough_idx]]
    recovery_days = len(recovered) if len(recovered) > 0 else -1
    return round(max_dd * 100, 2), recovery_days


def compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.DataFrame({"asset": returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 20:
        return 1.0
    cov = np.cov(aligned["asset"], aligned["bench"])
    return float(cov[0, 1] / cov[1, 1])


def compute_calmar(prices: pd.Series) -> float:
    returns = prices.pct_change().dropna()
    annual_return = (1 + returns.mean()) ** 252 - 1
    max_dd, _ = compute_max_drawdown(prices)
    return float(annual_return / abs(max_dd / 100)) if max_dd != 0 else 0.0


def full_risk_report(prices: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """
    Aggregated risk metrics for a single asset.
    """
    returns = prices.pct_change().dropna()
    max_dd, recovery = compute_max_drawdown(prices)
    annual_return = round(((1 + returns.mean()) ** 252 - 1) * 100, 2)
    annual_vol = round(returns.std() * np.sqrt(252) * 100, 2)

    report = {
        "annual_return_pct": annual_return,
        "annual_volatility_pct": annual_vol,
        "sharpe_ratio": round(compute_sharpe(returns), 3),
        "sortino_ratio": round(compute_sortino(returns), 3),
        "calmar_ratio": round(compute_calmar(prices), 3),
        "max_drawdown_pct": max_dd,
        "drawdown_recovery_days": recovery,
        "var_95_daily_pct": round(compute_var(returns, 0.95) * 100, 3),
        "cvar_95_daily_pct": round(compute_cvar(returns, 0.95) * 100, 3),
        "var_99_daily_pct": round(compute_var(returns, 0.99) * 100, 3),
        "skewness": round(float(stats.skew(returns.dropna())), 3),
        "kurtosis": round(float(stats.kurtosis(returns.dropna())), 3),
        "positive_days_pct": round((returns > 0).mean() * 100, 2),
        "risk_score": _risk_score(annual_vol, max_dd, compute_sharpe(returns)),
    }

    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        report["beta"] = round(compute_beta(returns, bench_ret), 3)

    return report


def _risk_score(vol: float, max_dd: float, sharpe: float) -> float:
    """Risk score 0–100: higher = lower risk / better risk-adjusted profile."""
    vol_score = max(0, 40 - vol)
    dd_score = max(0, 30 + max_dd)
    sharpe_score = min(sharpe * 10, 30)
    return round(min(max(vol_score + dd_score + sharpe_score, 0), 100), 2)

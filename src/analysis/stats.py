"""
Quantitative Statistical Analysis Suite.

Implements the core diagnostics used by quantitative analysts and portfolio
managers to characterize return time-series before deploying capital:

  - Hurst Exponent (R/S analysis)        — trending / random / mean-reverting
  - GARCH(1,1) Conditional Volatility    — time-varying risk estimate
  - Simplified ADF Stationarity Test     — unit-root / mean-reverting test
  - Ljung-Box Autocorrelation Test       — serial dependence in returns
  - Jarque-Bera Normality Test           — fat tails / skewness
  - Rolling Autocorrelation              — regime changes in predictability
  - Full Return Distribution Summary     — moments, percentiles

All functions degrade gracefully on short series (< minimum observations)
by returning empty containers or sentinel values — matching the platform's
"never crash the dashboard" contract.

References:
  - Hurst, H.E. (1951). "Long Term Storage Capacity of Reservoirs."
  - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
  - MIT Quant Finance Bible (statistical tests chapter).
  - DeepThinks Finance Quant Guide (signal diagnostics section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Hurst Exponent ────────────────────────────────────────────────────────────

def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via Rescaled Range (R/S) analysis.

    Interpretation:
        H < 0.45  — strongly mean-reverting (anti-persistent)
        0.45–0.55 — random walk (no exploitable memory)
        H > 0.55  — trending / momentum (persistent)

    Returns a float in [0, 1], or 0.5 on failure (assumes RW).
    """
    ts = prices.dropna().values.astype(float)
    if len(ts) < max_lag + 10:
        return 0.5

    rs_pairs: list[tuple[int, float]] = []
    for lag in range(2, min(max_lag, len(ts) // 2)):
        chunks = [ts[i:i + lag] for i in range(0, len(ts) - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            r = dev.max() - dev.min()
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_pairs.append((lag, float(np.mean(rs_chunk))))

    if len(rs_pairs) < 5:
        return 0.5

    log_lags = np.log([p[0] for p in rs_pairs])
    log_rs = np.log([p[1] for p in rs_pairs])
    h, *_ = np.polyfit(log_lags, log_rs, 1)
    return round(float(np.clip(h, 0.0, 1.0)), 4)


def hurst_interpretation(h: float) -> str:
    """Human-readable Hurst interpretation."""
    if h < 0.45:
        return f"H={h:.3f} — Mean-Reverting (anti-persistent; potential for stat-arb strategies)"
    if h > 0.55:
        return f"H={h:.3f} — Trending / Persistent (momentum strategies may apply)"
    return f"H={h:.3f} — Random Walk (no significant serial memory detected)"


# ── GARCH(1,1) Conditional Volatility ────────────────────────────────────────

def garch_vol_estimate(
    returns: pd.Series,
    omega: float = 1e-6,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> pd.Series:
    """
    GARCH(1,1) conditional volatility (annualized %).

    Variance equation: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Uses pre-set parameters (alpha + beta < 1 ensures stationarity).
    Fast enough for real-time dashboard use without MLE fitting.

    Returns:
        pd.Series of annualized conditional volatility (%), aligned to return index.
        Empty Series if returns are too short.
    """
    r = returns.dropna().values.astype(float)
    n = len(r)
    if n < 10:
        return pd.Series(dtype=float, name="garch_vol_pct")

    sigma2 = np.empty(n)
    sigma2[0] = float(np.var(r))

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    vol_ann = np.sqrt(np.maximum(sigma2, 0)) * np.sqrt(252) * 100
    return pd.Series(vol_ann, index=returns.dropna().index, name="garch_vol_pct")


# ── ADF Stationarity Test ─────────────────────────────────────────────────────

def adf_test(prices: pd.Series) -> dict:
    """
    Simplified Augmented Dickey-Fuller test for a unit root.

    Regression: Δy_t = α + β * y_{t-1} + ε
    H0: β = 0 (unit root / non-stationary / random walk)
    H1: β < 0 (stationary / mean-reverting)

    Note: This is a simplified single-lag ADF. A negative t-stat below
    ~−2.86 (5% critical value) suggests stationarity.

    Returns dict with: test_stat, p_value (approximate), is_stationary, interpretation.
    """
    ts = prices.dropna().values.astype(float)
    if len(ts) < 20:
        return {
            "test_stat": 0.0, "p_value": 1.0,
            "is_stationary": False,
            "interpretation": "Insufficient data (need 20+ observations)",
        }

    y = ts
    dy = np.diff(y)
    y_lag = y[:-1]

    X = np.column_stack([np.ones(len(dy)), y_lag])
    coeffs, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ coeffs
    n = len(dy)
    s2 = float(np.sum(resid**2) / max(n - 2, 1))

    XtX_inv = np.linalg.pinv(X.T @ X)
    var_b = s2 * XtX_inv[1, 1]
    t_stat = float(coeffs[1] / np.sqrt(var_b)) if var_b > 0 else 0.0

    # Conservative p-value approximation via standard normal CDF (left tail)
    p_value = float(np.clip(stats.norm.cdf(t_stat), 1e-6, 1.0))
    is_stationary = t_stat < -2.86  # use critical value directly

    if is_stationary:
        interp = "Stationary ✓ — mean-reverting; consider stat-arb / mean-reversion strategies"
    else:
        interp = "Non-stationary — unit root detected; price follows a random walk"

    return {
        "test_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "is_stationary": is_stationary,
        "interpretation": interp,
    }


# ── Ljung-Box Autocorrelation Test ────────────────────────────────────────────

def ljung_box_test(returns: pd.Series, lags: int = 10) -> dict:
    """
    Ljung-Box portmanteau test for autocorrelation in returns.

    H0: No autocorrelation up to `lags` lags.
    H1: At least one lag has significant autocorrelation.

    Q statistic is chi-squared distributed with `lags` degrees of freedom.
    Significant Q (p < 0.05) suggests predictability / strategy opportunity.

    Returns dict with: Q_stat, p_value, has_autocorrelation, lags, acf_values.
    """
    r = returns.dropna().values.astype(float)
    n = len(r)
    if n < lags + 5:
        return {"Q_stat": 0.0, "p_value": 1.0, "has_autocorrelation": False,
                "lags": lags, "acf_values": []}

    acf = np.array([
        float(np.corrcoef(r[:-k], r[k:])[0, 1]) if k > 0 else 1.0
        for k in range(1, lags + 1)
    ])
    q_stat = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1))))
    p_value = float(1.0 - stats.chi2.cdf(q_stat, df=lags))

    return {
        "Q_stat": round(q_stat, 4),
        "p_value": round(p_value, 4),
        "has_autocorrelation": p_value < 0.05,
        "lags": lags,
        "acf_values": [round(float(a), 4) for a in acf],
    }


# ── Jarque-Bera Normality Test ────────────────────────────────────────────────

def jarque_bera_test(returns: pd.Series) -> dict:
    """
    Jarque-Bera test for normality of returns.

    H0: Returns are normally distributed (skewness=0, excess kurtosis=0).
    H1: Returns are non-normal (fat tails, skewness — as observed empirically).

    Rejection (p < 0.05) is nearly universal for daily equity returns.
    The kurtosis value quantifies how much fatter the tails are than Gaussian.

    Returns dict: JB_stat, p_value, is_normal, skewness, excess_kurtosis.
    """
    r = returns.dropna()
    if len(r) < 10:
        return {"JB_stat": 0.0, "p_value": 1.0, "is_normal": True,
                "skewness": 0.0, "excess_kurtosis": 0.0}

    jb_stat, p_value = stats.jarque_bera(r.values)
    skew = float(stats.skew(r.values))
    kurt = float(stats.kurtosis(r.values))  # excess kurtosis (normal = 0)

    return {
        "JB_stat": round(float(jb_stat), 4),
        "p_value": round(float(p_value), 4),
        "is_normal": float(p_value) > 0.05,
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
    }


# ── Rolling Autocorrelation ───────────────────────────────────────────────────

def rolling_autocorrelation(
    returns: pd.Series,
    lag: int = 1,
    window: int = 63,
) -> pd.Series:
    """
    Rolling lag-k autocorrelation over a moving window.

    Useful for detecting regime changes in return predictability —
    a shift from near-zero to significant autocorrelation signals a
    structural break that strategies can exploit.

    Returns:
        pd.Series aligned to returns.index (NaN for warm-up period).
        Empty Series if series is too short.
    """
    r = returns.dropna()
    if len(r) < window + lag:
        return pd.Series(dtype=float, name=f"autocorr_lag{lag}")

    result = pd.Series(np.nan, index=r.index, name=f"autocorr_lag{lag}")

    for i in range(window + lag - 1, len(r)):
        past = r.iloc[i - window - lag + 1: i - lag + 1].values
        curr = r.iloc[i - window + 1: i + 1].values
        if len(past) == len(curr) and np.std(past) > 0 and np.std(curr) > 0:
            result.iloc[i] = float(np.corrcoef(past, curr)[0, 1])

    return result


# ── Return Distribution Summary ───────────────────────────────────────────────

def return_distribution_summary(returns: pd.Series) -> dict:
    """
    Comprehensive distributional characterization of a return series.

    Returns a dict with moments, percentiles, annualized metrics, and
    tail-risk estimates — used as the data backbone for the Quant Lab page.
    """
    r = returns.dropna()
    if len(r) < 5:
        return {}

    pcts = {f"p{p}": round(float(np.percentile(r, p)) * 100, 3)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}

    return {
        "n_obs": len(r),
        "mean_daily_pct": round(float(r.mean()) * 100, 4),
        "std_daily_pct": round(float(r.std()) * 100, 4),
        "annual_return_pct": round(((1 + float(r.mean())) ** 252 - 1) * 100, 2),
        "annual_vol_pct": round(float(r.std() * np.sqrt(252)) * 100, 2),
        "skewness": round(float(stats.skew(r.values)), 4),
        "excess_kurtosis": round(float(stats.kurtosis(r.values)), 4),
        "min_pct": round(float(r.min()) * 100, 3),
        "max_pct": round(float(r.max()) * 100, 3),
        "positive_days_pct": round(float((r > 0).mean()) * 100, 2),
        **pcts,
    }

"""
Statistical Arbitrage — Pairs Trading.

Implements the classic Engle-Granger two-step cointegration framework:
  1. Estimate hedge ratio via OLS (price_A = alpha + beta * price_B)
  2. Test the spread residual for stationarity (ADF test)
  3. Fit Ornstein-Uhlenbeck model to the spread to estimate half-life
  4. Generate entry/exit signals from z-scored spread

Pairs ranking: sorts all watchlist pairs by cointegration strength
(lowest ADF p-value) so the analyst can quickly identify the best
candidates.

All outputs degrade gracefully on insufficient data — same contract as
the rest of the platform (None / empty DataFrame on failure).

References:
  - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and
    Error Correction: Representation, Estimation, and Testing."
  - Gatev, E., Goetzmann, W.N., and Rouwenhorst, K.G. (2006).
    "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
  - Avellaneda, M. and Lee, J.H. (2010). "Statistical Arbitrage in the
    US Equities Market."
  - MIT Quant Bible (statistical arbitrage chapter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class PairAnalysis:
    asset_a: str
    asset_b: str
    hedge_ratio: float          # OLS beta: spread = A - hedge_ratio * B
    ou_half_life_days: float    # mean-reversion half-life in trading days
    adf_stat: float             # ADF t-statistic on the spread (lower = more stationary)
    p_value: float              # approximate p-value (lower = more cointegrated)
    is_cointegrated: bool       # ADF t-stat < −2.86 (5% critical value)
    spread_mean: float          # historical mean of the spread
    spread_std: float           # historical std of the spread (63-day window)
    current_zscore: float       # latest z-score of the spread
    signal: str                 # "BUY_SPREAD" | "SELL_SPREAD" | "HOLD"
    signal_description: str     # human-readable trade description


# ── Internal OLS / ADF helpers ────────────────────────────────────────────────

def _ols_hedge_ratio(price_a: np.ndarray, price_b: np.ndarray) -> float:
    """OLS estimate of beta in price_A = alpha + beta * price_B."""
    X = np.column_stack([np.ones(len(price_b)), price_b])
    coeffs, *_ = np.linalg.lstsq(X, price_a, rcond=None)
    return float(coeffs[1])


def _adf_statistic(series: np.ndarray) -> tuple[float, float]:
    """
    Simplified single-lag ADF test: Δy_t = α + β * y_{t-1} + ε.

    Returns (t_stat, approx_p_value).
    Critical values (MacKinnon 1994): −2.86 at 5%, −3.43 at 1%.
    """
    y = series.astype(float)
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

    # Conservative p-value: standard normal left tail (understates significance —
    # ADF true critical values are more negative than standard normal)
    p_value = float(np.clip(stats.norm.cdf(t_stat), 1e-6, 1.0))
    return t_stat, p_value


# ── Ornstein-Uhlenbeck half-life ─────────────────────────────────────────────

def ou_half_life(spread: np.ndarray) -> float:
    """
    Estimate the Ornstein-Uhlenbeck mean-reversion half-life from the spread.

    Fits: Δspread_t = -kappa * spread_{t-1} * dt + σ * ε
    Half-life τ = log(2) / kappa  (in the same units as the time step, i.e. trading days)

    Returns inf if mean-reversion speed kappa ≤ 0 (non-mean-reverting).
    """
    y = spread.astype(float)
    dy = np.diff(y)
    y_lag = y[:-1]

    if len(dy) < 5 or float(np.std(y_lag)) < 1e-10:
        return float("inf")

    X = np.column_stack([np.ones(len(dy)), y_lag])
    coeffs, *_ = np.linalg.lstsq(X, dy, rcond=None)
    kappa = -float(coeffs[1])  # mean-reversion speed (positive = mean-reverting)

    if kappa <= 0:
        return float("inf")
    return float(np.log(2.0) / kappa)


# ── Core pair analysis ────────────────────────────────────────────────────────

def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Compute the cointegration spread: A - hedge_ratio * B.

    Returns (spread_series, hedge_ratio). If hedge_ratio is None,
    estimates it via OLS over the full aligned sample.
    """
    aligned = pd.DataFrame({"A": price_a, "B": price_b}).dropna()
    if len(aligned) < 20:
        return pd.Series(dtype=float), 0.0

    if hedge_ratio is None:
        hedge_ratio = _ols_hedge_ratio(aligned["A"].values, aligned["B"].values)

    spread = aligned["A"] - hedge_ratio * aligned["B"]
    return spread, hedge_ratio


def spread_zscore(spread: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling z-score of the spread: (spread - rolling_mean) / rolling_std.
    Values outside ±2 are potential entry signals.
    """
    mean = spread.rolling(window, min_periods=window // 2).mean()
    std = spread.rolling(window, min_periods=window // 2).std()
    z = (spread - mean) / std
    return z.rename("z_score")


def analyze_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
    min_history: int = 60,
) -> PairAnalysis | None:
    """
    Full pair analysis: hedge ratio, cointegration test, half-life, signal.

    Returns None if insufficient aligned data (< min_history observations).

    Signal logic (from Gatev et al.):
        z_score < −zscore_entry → BUY spread (A cheap vs B)
        z_score > +zscore_entry → SELL spread (A expensive vs B)
        |z_score| < zscore_exit → HOLD / no position
    """
    aligned = pd.DataFrame({"A": price_a, "B": price_b}).dropna()
    if len(aligned) < min_history:
        return None

    a = aligned["A"].values
    b = aligned["B"].values

    hedge_ratio = _ols_hedge_ratio(a, b)
    spread = a - hedge_ratio * b

    adf_stat, p_value = _adf_statistic(spread)
    hl = ou_half_life(spread)
    is_cointegrated = adf_stat < -2.86  # 5% critical value (MacKinnon)

    # Current z-score: rolling 63-day window
    window = min(63, len(spread) // 2)
    if len(spread) >= window:
        spread_mean = float(np.mean(spread[-window:]))
        spread_std = float(np.std(spread[-window:]))
        current_z = float((spread[-1] - spread_mean) / spread_std) if spread_std > 0 else 0.0
    else:
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))
        current_z = 0.0

    if current_z < -zscore_entry:
        signal = "BUY_SPREAD"
        signal_desc = f"BUY {name_a}, SELL {name_b} (spread too low, z={current_z:.2f})"
    elif current_z > zscore_entry:
        signal = "SELL_SPREAD"
        signal_desc = f"SELL {name_a}, BUY {name_b} (spread too high, z={current_z:.2f})"
    else:
        signal = "HOLD"
        signal_desc = f"No signal — spread within normal range (z={current_z:.2f})"

    return PairAnalysis(
        asset_a=name_a,
        asset_b=name_b,
        hedge_ratio=round(hedge_ratio, 4),
        ou_half_life_days=round(hl, 1) if np.isfinite(hl) else 9999.0,
        adf_stat=round(adf_stat, 4),
        p_value=round(p_value, 4),
        is_cointegrated=is_cointegrated,
        spread_mean=round(spread_mean, 4),
        spread_std=round(spread_std, 4),
        current_zscore=round(current_z, 4),
        signal=signal,
        signal_description=signal_desc,
    )


def rank_pairs(
    prices: pd.DataFrame,
    min_history: int = 252,
    zscore_entry: float = 2.0,
) -> pd.DataFrame:
    """
    Rank all pairs in a price DataFrame by cointegration strength.

    Args:
        prices:       DataFrame with columns = tickers, index = dates.
        min_history:  Minimum observations required per pair.
        zscore_entry: Z-score threshold for signal generation.

    Returns:
        DataFrame sorted by ADF t-statistic (most stationary pairs first),
        with columns: asset_a, asset_b, hedge_ratio, ou_half_life_days,
        adf_stat, p_value, is_cointegrated, current_zscore, signal.
        Empty DataFrame if no pairs qualify.
    """
    tickers = [c for c in prices.columns if prices[c].dropna().__len__() >= min_history]
    rows = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a_name, b_name = tickers[i], tickers[j]
            result = analyze_pair(
                prices[a_name].dropna(),
                prices[b_name].dropna(),
                a_name, b_name,
                zscore_entry=zscore_entry,
                min_history=min_history,
            )
            if result is not None:
                rows.append({
                    "asset_a": result.asset_a,
                    "asset_b": result.asset_b,
                    "hedge_ratio": result.hedge_ratio,
                    "ou_half_life_days": result.ou_half_life_days,
                    "adf_stat": result.adf_stat,
                    "p_value": result.p_value,
                    "is_cointegrated": result.is_cointegrated,
                    "current_zscore": result.current_zscore,
                    "signal": result.signal,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("adf_stat").reset_index(drop=True)
    return df

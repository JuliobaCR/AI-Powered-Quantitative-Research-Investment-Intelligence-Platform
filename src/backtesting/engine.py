"""
Backtesting Framework — vectorized strategy backtests.

Strategies:
  - SMA Crossover (trend-following)
  - RSI Mean Reversion

Performance reporting reuses src.risk.engine (Sharpe, drawdown, VaR, etc.)
so backtest results are directly comparable with the rest of the platform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

from src.risk.engine import compute_sharpe, full_risk_report


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series      # strategy cumulative growth, base = 1.0
    benchmark_curve: pd.Series   # buy & hold cumulative growth, base = 1.0
    positions: pd.Series         # 1.0 = long, 0.0 = flat
    trades: int
    metrics: dict                # full_risk_report on the strategy's equity curve


def _build_result(df: pd.DataFrame, positions: pd.Series, strategy_name: str) -> BacktestResult:
    """Turn a position series into an equity curve + risk report."""
    returns = df["Returns"]
    strat_returns = (positions.shift(1).fillna(0.0) * returns).fillna(0.0)

    equity_curve = (1 + strat_returns).cumprod()
    benchmark_curve = (1 + returns).cumprod()
    trades = int((positions.diff().fillna(0.0) != 0).sum())
    metrics = full_risk_report(equity_curve * 100)

    return BacktestResult(
        strategy_name=strategy_name,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )


def backtest_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> BacktestResult:
    """
    Long when the fast SMA is above the slow SMA, flat otherwise.

    Args:
        df:   OHLCV DataFrame with "Close" and "Returns" columns.
        fast: Fast SMA window (days).
        slow: Slow SMA window (days).
    """
    close = df["Close"]
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()

    positions = pd.Series(np.where(sma_fast > sma_slow, 1.0, 0.0), index=df.index)
    positions.iloc[: max(fast, slow)] = 0.0  # avoid trading on the warm-up window

    return _build_result(df, positions, f"SMA Crossover ({fast}/{slow})")


def backtest_rsi_meanreversion(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lower: int = 30,
    upper: int = 70,
) -> BacktestResult:
    """
    Enter long when RSI drops below `lower` (oversold), exit when RSI rises
    above `upper` (overbought). Flat otherwise.
    """
    rsi = ta.momentum.rsi(df["Close"], window=rsi_period)

    raw_signal = pd.Series(np.nan, index=df.index)
    raw_signal[rsi < lower] = 1.0
    raw_signal[rsi > upper] = 0.0
    positions = raw_signal.ffill().fillna(0.0)

    return _build_result(df, positions, f"RSI Mean Reversion ({rsi_period}, {lower}/{upper})")


def sma_param_sweep(df: pd.DataFrame, fast_range: range, slow_range: range) -> pd.DataFrame:
    """
    Run SMA crossover backtests across a grid of (fast, slow) windows.

    Returns a long-form DataFrame with columns [fast, slow, sharpe], suitable
    for pivoting into a 3D surface.
    """
    rows = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            result = backtest_sma_crossover(df, fast, slow)
            strat_returns = result.equity_curve.pct_change().dropna()
            sharpe = compute_sharpe(strat_returns)
            rows.append({"fast": fast, "slow": slow, "sharpe": round(sharpe, 3)})

    return pd.DataFrame(rows)

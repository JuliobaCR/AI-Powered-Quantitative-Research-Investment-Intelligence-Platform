"""Unit tests for the Backtesting Framework."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (
    BacktestResult,
    backtest_rsi_meanreversion,
    backtest_sma_crossover,
    sma_param_sweep,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 300
    returns = pd.Series(np.random.normal(0.0006, 0.015, n))
    close = (1 + returns).cumprod() * 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close.values, "Returns": returns.values}, index=idx)


def test_sma_crossover_returns_result(sample_df):
    result = backtest_sma_crossover(sample_df, fast=10, slow=30)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(sample_df)
    assert len(result.benchmark_curve) == len(sample_df)
    assert "sharpe_ratio" in result.metrics


def test_sma_crossover_positions_are_binary(sample_df):
    result = backtest_sma_crossover(sample_df, fast=10, slow=30)
    assert set(result.positions.unique()).issubset({0.0, 1.0})


def test_rsi_meanreversion_returns_result(sample_df):
    result = backtest_rsi_meanreversion(sample_df, rsi_period=14, lower=30, upper=70)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(sample_df)
    assert "sharpe_ratio" in result.metrics


def test_sma_param_sweep(sample_df):
    sweep = sma_param_sweep(sample_df, fast_range=range(5, 11, 5), slow_range=range(10, 21, 5))
    assert not sweep.empty
    assert set(sweep.columns) == {"fast", "slow", "sharpe"}
    assert (sweep["fast"] < sweep["slow"]).all()

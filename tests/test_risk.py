"""Unit tests for the Risk Engine."""

import numpy as np
import pandas as pd
import pytest

from src.risk.engine import compute_var, compute_cvar, compute_sharpe, compute_max_drawdown


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0008, 0.015, 252))


@pytest.fixture
def sample_prices(sample_returns):
    return (1 + sample_returns).cumprod() * 100


def test_var_positive(sample_returns):
    var = compute_var(sample_returns, 0.95)
    assert var > 0, "VaR should be a positive loss figure"


def test_cvar_geq_var(sample_returns):
    var = compute_var(sample_returns, 0.95)
    cvar = compute_cvar(sample_returns, 0.95)
    assert cvar >= var, "CVaR should always be >= VaR"


def test_sharpe_type(sample_returns):
    sharpe = compute_sharpe(sample_returns)
    assert isinstance(sharpe, float)


def test_max_drawdown_negative(sample_prices):
    mdd, _ = compute_max_drawdown(sample_prices)
    assert mdd <= 0, "Max drawdown should be non-positive"

"""Unit tests for the Fama-French 5-Factor model (Phase 9)."""

import numpy as np
import pandas as pd
import pytest

from src.factors.fama_french import (
    FACTOR_COLUMNS,
    fetch_ff5_factors,
    rolling_factor_exposures,
    run_factor_regression,
)


@pytest.fixture
def synthetic_factors():
    np.random.seed(42)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    data = {col: np.random.normal(0.0003, 0.01, n) for col in FACTOR_COLUMNS}
    data["RF"] = np.full(n, 0.0001)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def synthetic_returns(synthetic_factors):
    true_alpha_daily = 0.0002
    true_betas = {"Mkt-RF": 1.1, "SMB": 0.3, "HML": -0.2, "RMW": 0.1, "CMA": 0.4}
    noise = np.random.normal(0, 0.0005, len(synthetic_factors))

    excess = true_alpha_daily + sum(
        true_betas[f] * synthetic_factors[f].values for f in FACTOR_COLUMNS
    ) + noise
    returns = excess + synthetic_factors["RF"].values

    return pd.Series(returns, index=synthetic_factors.index), true_alpha_daily, true_betas


def test_run_factor_regression_recovers_betas(synthetic_factors, synthetic_returns):
    returns, true_alpha_daily, true_betas = synthetic_returns
    exposure = run_factor_regression(returns, ticker="TEST", factors=synthetic_factors)

    assert exposure is not None
    assert exposure.ticker == "TEST"
    assert exposure.n_obs == len(synthetic_factors)
    assert exposure.r_squared > 0.9

    for factor, true_beta in true_betas.items():
        assert exposure.betas[factor] == pytest.approx(true_beta, abs=0.1)

    assert exposure.alpha_annual_pct == pytest.approx(true_alpha_daily * 252 * 100, abs=2.0)


def test_run_factor_regression_empty_factors_returns_none():
    returns = pd.Series(
        np.random.normal(0, 0.01, 100),
        index=pd.date_range("2023-01-01", periods=100, freq="B"),
    )
    assert run_factor_regression(returns, factors=pd.DataFrame()) is None


def test_run_factor_regression_insufficient_overlap_returns_none(synthetic_factors):
    short_returns = pd.Series(
        np.random.normal(0, 0.01, 30),
        index=synthetic_factors.index[:30],
    )
    assert run_factor_regression(short_returns, factors=synthetic_factors) is None


def test_rolling_factor_exposures_shape(synthetic_factors, synthetic_returns):
    returns, _, _ = synthetic_returns
    rolling = rolling_factor_exposures(returns, factors=synthetic_factors, window=126, step=5)

    assert not rolling.empty
    assert set(FACTOR_COLUMNS).issubset(rolling.columns)
    assert isinstance(rolling.index, pd.DatetimeIndex)


def test_rolling_factor_exposures_insufficient_history_returns_empty(synthetic_factors, synthetic_returns):
    returns, _, _ = synthetic_returns
    short_returns = returns.iloc[:50]
    short_factors = synthetic_factors.iloc[:50]

    rolling = rolling_factor_exposures(short_returns, factors=short_factors, window=126, step=5)
    assert rolling.empty


def test_fetch_ff5_factors_smoke():
    factors = fetch_ff5_factors()
    if factors.empty:
        pytest.skip("Fama-French data unavailable (no network)")

    assert set(FACTOR_COLUMNS + ["RF"]).issubset(factors.columns)
    assert isinstance(factors.index, pd.DatetimeIndex)

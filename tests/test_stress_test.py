"""Tests for the Stress Testing module."""

import numpy as np
import pandas as pd
import pytest

from src.risk.stress_test import (
    HISTORICAL_SCENARIOS,
    StressResult,
    run_custom_scenario,
    run_stress_scenario,
    stress_scenario_matrix,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    n = 252
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return {
        "AAPL": pd.Series(np.random.normal(0.0005, 0.015, n), index=idx),
        "MSFT": pd.Series(np.random.normal(0.0006, 0.014, n), index=idx),
    }


@pytest.fixture
def equal_weights():
    return {"AAPL": 0.5, "MSFT": 0.5}


def test_historical_scenarios_library_has_entries():
    assert len(HISTORICAL_SCENARIOS) >= 8
    for name, scen in HISTORICAL_SCENARIOS.items():
        assert "equity_shock" in scen
        assert "vol_multiplier" in scen
        assert scen["equity_shock"] < 0 or scen["equity_shock"] == 0


def test_run_stress_scenario_returns_stress_result(equal_weights, sample_returns):
    result = run_stress_scenario(equal_weights, sample_returns, "2008 GFC (Sep–Nov)")
    assert isinstance(result, StressResult)
    assert result.scenario_name == "2008 GFC (Sep–Nov)"
    assert result.equity_shock_pct < 0  # GFC was negative
    assert result.portfolio_pnl_pct < 0
    assert result.var_95_stressed_pct >= 0
    assert set(result.per_asset_pnl.keys()) == {"AAPL", "MSFT"}


def test_all_scenarios_run_without_error(equal_weights, sample_returns):
    for scenario_name in HISTORICAL_SCENARIOS:
        result = run_stress_scenario(equal_weights, sample_returns, scenario_name)
        assert isinstance(result, StressResult)
        assert np.isfinite(result.portfolio_pnl_pct)


def test_run_custom_scenario(equal_weights, sample_returns):
    result = run_custom_scenario(
        equal_weights, sample_returns,
        equity_shock_pct=-15.0,
        vol_multiplier=2.0,
    )
    assert isinstance(result, StressResult)
    assert result.equity_shock_pct == pytest.approx(-15.0, abs=0.01)
    assert result.portfolio_pnl_pct < 0


def test_stress_matrix_shape(equal_weights, sample_returns):
    matrix = stress_scenario_matrix(equal_weights, sample_returns)
    assert not matrix.empty
    assert len(matrix) == len(HISTORICAL_SCENARIOS)
    assert "Portfolio P&L %" in matrix.columns
    assert "AAPL" in matrix.columns


def test_stress_empty_returns():
    result = run_stress_scenario({"X": 1.0}, {}, "COVID-19 Crash (Feb–Mar 2020)")
    assert isinstance(result, StressResult)
    assert result.portfolio_pnl_pct < 0  # shock still applied via equity_shock

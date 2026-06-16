"""Tests for portfolio risk attribution module."""

import numpy as np
import pandas as pd
import pytest

from src.risk.attribution import (
    component_var,
    factor_risk_decomposition,
    full_attribution_report,
    marginal_var,
    volatility_contribution,
)


@pytest.fixture
def sample_portfolio():
    np.random.seed(42)
    n = 252
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "AAPL": np.random.normal(0.0006, 0.016, n),
        "MSFT": np.random.normal(0.0005, 0.014, n),
        "GOOG": np.random.normal(0.0004, 0.015, n),
        "AMZN": np.random.normal(0.0003, 0.018, n),
    }, index=idx)
    weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25}
    return weights, returns


# ── Component VaR ─────────────────────────────────────────────────────────────

class TestComponentVar:
    def test_sums_to_portfolio_var(self, sample_portfolio):
        weights, returns = sample_portfolio
        comp = component_var(weights, returns, confidence=0.95)
        assert comp  # non-empty

        # Sum of component VaRs == portfolio VaR (parametric, exact decomposition)
        from scipy.stats import norm
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        cov = returns[tickers].dropna().cov().values
        z = abs(norm.ppf(0.05))
        port_vol = float(np.sqrt(w @ cov @ w))
        port_var_pct = z * port_vol * 100

        comp_sum = sum(comp.values())
        assert comp_sum == pytest.approx(port_var_pct, rel=0.05)

    def test_keys_match_tickers(self, sample_portfolio):
        weights, returns = sample_portfolio
        comp = component_var(weights, returns)
        assert set(comp.keys()) == set(weights.keys())

    def test_empty_returns(self):
        comp = component_var({"X": 1.0}, pd.DataFrame())
        assert comp == {}

    def test_insufficient_data(self):
        tiny = pd.DataFrame({"X": np.random.normal(0, 0.01, 5)})
        comp = component_var({"X": 1.0}, tiny)
        assert comp == {"X": 0.0}


# ── Marginal VaR ──────────────────────────────────────────────────────────────

class TestMarginalVar:
    def test_keys_match(self, sample_portfolio):
        weights, returns = sample_portfolio
        mvar = marginal_var(weights, returns)
        assert set(mvar.keys()) == set(weights.keys())

    def test_values_finite(self, sample_portfolio):
        weights, returns = sample_portfolio
        mvar = marginal_var(weights, returns)
        for v in mvar.values():
            assert np.isfinite(v)

    def test_highest_vol_asset_highest_marginal_var(self, sample_portfolio):
        weights, returns = sample_portfolio
        mvar = marginal_var(weights, returns)
        # AMZN has highest sigma → should have highest (or near-highest) marginal VaR
        assert mvar["AMZN"] >= mvar["MSFT"]  # AMZN vol > MSFT vol


# ── Volatility Contribution ───────────────────────────────────────────────────

class TestVolatilityContribution:
    def test_sums_to_100(self, sample_portfolio):
        weights, returns = sample_portfolio
        contrib = volatility_contribution(weights, returns)
        total = sum(contrib.values())
        assert total == pytest.approx(100.0, abs=1.0)

    def test_all_positive_for_long_only(self, sample_portfolio):
        weights, returns = sample_portfolio
        contrib = volatility_contribution(weights, returns)
        for v in contrib.values():
            assert v >= 0

    def test_keys_match(self, sample_portfolio):
        weights, returns = sample_portfolio
        contrib = volatility_contribution(weights, returns)
        assert set(contrib.keys()) == set(weights.keys())


# ── Factor Risk Decomposition ─────────────────────────────────────────────────

class TestFactorRiskDecomposition:
    def test_basic_decomposition(self):
        from src.factors.fama_french import FACTOR_COLUMNS
        np.random.seed(42)
        n = 252
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        factor_returns = pd.DataFrame(
            {f: np.random.normal(0.0003, 0.01, n) for f in FACTOR_COLUMNS}, index=idx
        )
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        betas = {
            "AAPL": {"Mkt-RF": 1.2, "SMB": 0.3, "HML": -0.1, "RMW": 0.2, "CMA": 0.1},
            "MSFT": {"Mkt-RF": 1.0, "SMB": -0.1, "HML": 0.0, "RMW": 0.3, "CMA": 0.0},
        }
        result = factor_risk_decomposition(weights, betas, factor_returns, FACTOR_COLUMNS)
        assert set(result.keys()) == set(FACTOR_COLUMNS)
        total = sum(result.values())
        assert total == pytest.approx(100.0, abs=5.0)  # contributions sum to ~100%

    def test_empty_factors_returns_empty(self):
        from src.factors.fama_french import FACTOR_COLUMNS
        result = factor_risk_decomposition({}, {}, pd.DataFrame(), FACTOR_COLUMNS)
        assert result == {}


# ── Full Attribution Report ───────────────────────────────────────────────────

def test_full_attribution_report_keys(sample_portfolio):
    weights, returns = sample_portfolio
    report = full_attribution_report(weights, returns)
    assert "component_var" in report
    assert "marginal_var" in report
    assert "vol_contribution" in report

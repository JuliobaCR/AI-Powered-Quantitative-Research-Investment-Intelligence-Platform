"""Tests for statistical arbitrage pairs trading module."""

import numpy as np
import pandas as pd
import pytest

from src.arbitrage.pairs import (
    PairAnalysis,
    analyze_pair,
    compute_spread,
    ou_half_life,
    rank_pairs,
    spread_zscore,
)


@pytest.fixture
def cointegrated_pair():
    """
    Cointegrated pair: price_B drives price_A with hedge_ratio=2.0 plus a
    stationary noise series — guaranteed cointegration for testing.
    """
    np.random.seed(42)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B")

    # Random walk (B)
    b = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    # A = 2 * B + stationary noise
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.6 * noise[t - 1] + np.random.normal(0, 0.5)
    a = 2.0 * b + noise

    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


@pytest.fixture
def non_cointegrated_pair():
    """Two independent random walks — no cointegration."""
    np.random.seed(7)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    a = 100 + np.cumsum(np.random.normal(0, 1, n))
    b = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.Series(a, index=idx, name="X"), pd.Series(b, index=idx, name="Y")


# ── compute_spread ────────────────────────────────────────────────────────────

def test_compute_spread_returns_series(cointegrated_pair):
    a, b = cointegrated_pair
    spread, hr = compute_spread(a, b)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(a)
    assert hr > 0


def test_compute_spread_hedge_ratio_fixed(cointegrated_pair):
    a, b = cointegrated_pair
    spread, hr = compute_spread(a, b, hedge_ratio=2.0)
    assert hr == pytest.approx(2.0, abs=1e-6)
    # Spread should be stationary (small std relative to price)
    assert float(spread.std()) < 10.0


# ── OU Half-Life ─────────────────────────────────────────────────────────────

def test_ou_half_life_finite_for_stationary():
    np.random.seed(0)
    # Stationary AR(1): x_t = 0.8*x_{t-1} + eps  → half-life = log(2)/log(1/0.8) ≈ 3.1 steps
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.8 * x[t - 1] + np.random.normal(0, 0.5)
    hl = ou_half_life(x)
    assert np.isfinite(hl)
    assert hl > 0


def test_ou_half_life_random_walk_large():
    np.random.seed(1)
    rw = np.cumsum(np.random.normal(0, 1, 200))
    hl = ou_half_life(rw)
    # Random walk has no mean reversion; half-life should be very large or inf
    assert hl > 100 or not np.isfinite(hl)


# ── analyze_pair ──────────────────────────────────────────────────────────────

def test_analyze_pair_cointegrated(cointegrated_pair):
    a, b = cointegrated_pair
    result = analyze_pair(a, b, "A", "B")
    assert isinstance(result, PairAnalysis)
    assert result.asset_a == "A"
    assert result.asset_b == "B"
    # Hedge ratio should be close to 2.0 (true value)
    assert result.hedge_ratio == pytest.approx(2.0, abs=0.3)
    # ADF t-stat should be negative (stationary spread)
    assert result.adf_stat < 0
    # Half-life should be finite and reasonable
    assert np.isfinite(result.ou_half_life_days)
    assert result.ou_half_life_days < 500


def test_analyze_pair_insufficient_data():
    short = pd.Series(np.random.normal(0, 1, 30))
    result = analyze_pair(short, short, min_history=60)
    assert result is None


def test_analyze_pair_signal_in_valid_set(cointegrated_pair):
    a, b = cointegrated_pair
    result = analyze_pair(a, b, "A", "B")
    assert result.signal in {"BUY_SPREAD", "SELL_SPREAD", "HOLD"}


# ── spread_zscore ─────────────────────────────────────────────────────────────

def test_zscore_length(cointegrated_pair):
    a, b = cointegrated_pair
    spread, _ = compute_spread(a, b)
    z = spread_zscore(spread, window=63)
    assert len(z) == len(spread)


def test_zscore_mean_near_zero(cointegrated_pair):
    a, b = cointegrated_pair
    spread, _ = compute_spread(a, b)
    z = spread_zscore(spread, window=63).dropna()
    assert abs(float(z.mean())) < 0.5  # rolling z-score should be near zero


# ── rank_pairs ────────────────────────────────────────────────────────────────

def test_rank_pairs_returns_dataframe(cointegrated_pair):
    a, b = cointegrated_pair
    prices = pd.DataFrame({"A": a, "B": b})
    result = rank_pairs(prices, min_history=100)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "adf_stat" in result.columns
        assert "p_value" in result.columns
        assert "hedge_ratio" in result.columns


def test_rank_pairs_insufficient_history():
    prices = pd.DataFrame({
        "X": np.random.normal(100, 5, 50),
        "Y": np.random.normal(100, 5, 50),
    })
    result = rank_pairs(prices, min_history=252)
    assert result.empty


def test_rank_pairs_sorted_by_adf(cointegrated_pair):
    a, b = cointegrated_pair
    np.random.seed(10)
    c = pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(a))),
                  index=a.index, name="C")
    prices = pd.DataFrame({"A": a, "B": b, "C": c})
    result = rank_pairs(prices, min_history=100)
    if len(result) > 1:
        # Sorted by ADF t-stat (most negative = most cointegrated first)
        assert result["adf_stat"].iloc[0] <= result["adf_stat"].iloc[1]

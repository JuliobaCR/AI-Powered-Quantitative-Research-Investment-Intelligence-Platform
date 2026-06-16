"""Tests for options strategy payoff profiles."""

import numpy as np
import pytest

from src.derivatives.strategies import (
    StrategyProfile,
    build_all_strategies,
    build_butterfly,
    build_bull_call_spread,
    build_covered_call,
    build_iron_condor,
    build_protective_put,
    build_straddle,
    build_strangle,
    estimate_atm_premiums,
    strategy_pnl_curve,
)


SPOT = 100.0
SIGMA = 0.20
DAYS = 30


@pytest.fixture
def atm_premiums():
    return estimate_atm_premiums(SPOT, SIGMA, DAYS)


# ── Premium estimator ─────────────────────────────────────────────────────────

def test_atm_premiums_positive(atm_premiums):
    call_p, put_p = atm_premiums
    assert call_p > 0
    assert put_p > 0


def test_atm_premiums_put_call_parity(atm_premiums):
    """ATM call ≈ ATM put (for low r, short T): rough check."""
    call_p, put_p = atm_premiums
    # ATM call and put should be within 20% of each other
    assert abs(call_p - put_p) / max(call_p, put_p) < 0.20


def test_zero_sigma_returns_zero():
    c, p = estimate_atm_premiums(100.0, 0.0, 30)
    assert c == 0.0 and p == 0.0


# ── Covered Call ──────────────────────────────────────────────────────────────

def test_covered_call_max_profit():
    c_strike, c_prem = 105.0, 2.0
    profile = build_covered_call(SPOT, c_strike, c_prem)
    assert isinstance(profile, StrategyProfile)
    assert profile.max_profit == pytest.approx((c_strike - SPOT) + c_prem, abs=0.01)
    assert profile.name == "Covered Call"


def test_covered_call_pnl_at_expiry():
    profile = build_covered_call(SPOT, 105.0, 2.0)
    spots, pnl = strategy_pnl_curve(profile, (80.0, 120.0))
    # At expiry spot = 110 (above strike): P&L = max_profit
    idx_110 = np.argmin(np.abs(spots - 110.0))
    assert pnl[idx_110] == pytest.approx(profile.max_profit, abs=0.5)


# ── Straddle ─────────────────────────────────────────────────────────────────

def test_long_straddle_max_loss():
    c_p, p_p = 3.0, 3.0
    profile = build_straddle(SPOT, c_p, p_p, long=True)
    assert profile.max_loss == pytest.approx(c_p + p_p, abs=0.01)
    assert profile.max_profit is None  # unlimited


def test_short_straddle_max_profit():
    c_p, p_p = 3.0, 3.0
    profile = build_straddle(SPOT, c_p, p_p, long=False)
    assert profile.max_profit == pytest.approx(c_p + p_p, abs=0.01)
    assert profile.max_loss is None  # unlimited risk


def test_straddle_pnl_at_strike():
    c_p, p_p = 3.0, 3.0
    profile = build_straddle(SPOT, c_p, p_p, long=True)
    spots, pnl = strategy_pnl_curve(profile, (85.0, 115.0))
    # At exactly the strike (100), both legs expire worthless → P&L = -total premium
    idx_100 = np.argmin(np.abs(spots - SPOT))
    assert pnl[idx_100] == pytest.approx(-(c_p + p_p), abs=0.1)


# ── Bull Call Spread ──────────────────────────────────────────────────────────

def test_bull_call_spread_bounded():
    profile = build_bull_call_spread(100.0, 110.0, 5.0, 2.0)
    assert profile.max_profit is not None
    assert profile.max_loss is not None
    assert profile.max_profit > 0
    assert profile.max_loss > 0


# ── Butterfly ─────────────────────────────────────────────────────────────────

def test_butterfly_max_profit_positive():
    profile = build_butterfly(90.0, 100.0, 110.0, 1.0, 4.0, 1.0)
    assert profile.max_profit is not None and profile.max_profit > 0


# ── Iron Condor ───────────────────────────────────────────────────────────────

def test_iron_condor_credit():
    # Iron Condor should be a net credit strategy
    profile = build_iron_condor(85.0, 95.0, 105.0, 115.0, 0.5, 2.0, 2.0, 0.5)
    assert profile.max_profit > 0  # premium collected


# ── build_all_strategies ──────────────────────────────────────────────────────

def test_build_all_strategies_keys():
    strategies = build_all_strategies(SPOT, SIGMA, DAYS)
    expected = {
        "Covered Call", "Protective Put", "Long Straddle", "Short Straddle",
        "Long Strangle", "Short Strangle", "Bull Call Spread", "Bear Put Spread",
        "Long Butterfly", "Iron Condor",
    }
    assert expected == set(strategies.keys())


def test_build_all_strategies_are_profiles():
    strategies = build_all_strategies(SPOT, SIGMA, DAYS)
    for name, profile in strategies.items():
        assert isinstance(profile, StrategyProfile), f"{name} is not a StrategyProfile"


def test_pnl_curves_have_correct_length():
    strategies = build_all_strategies(SPOT, SIGMA, DAYS)
    for name, profile in strategies.items():
        spots, pnl = strategy_pnl_curve(profile, (80.0, 120.0), n_points=100)
        assert len(spots) == 100, f"{name}: spot array wrong length"
        assert len(pnl) == 100, f"{name}: pnl array wrong length"

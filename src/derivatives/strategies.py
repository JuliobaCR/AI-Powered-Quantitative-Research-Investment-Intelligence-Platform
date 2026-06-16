"""
Options Strategy Payoff Profiles.

Implements standard multi-leg options strategies used by institutional
options desks: P&L at expiry, key risk metrics (max profit/loss,
breakeven points), and a premium estimator for rapid scenario building.

All strategies return a StrategyProfile dataclass that the dashboard
uses for payoff diagrams. The payoff_at_expiry / strategy_pnl_curve
functions are pure numpy — no Black-Scholes calls at runtime, so they
are fast enough for interactive sliders.

Strategies implemented:
  - Covered Call       (income / upside cap)
  - Protective Put     (portfolio insurance)
  - Long / Short Straddle (volatility bets)
  - Long / Short Strangle (cheaper vol bets)
  - Bull Call Spread   (bullish limited risk)
  - Bear Put Spread    (bearish limited risk)
  - Long Butterfly     (range-bound bet)
  - Iron Condor        (premium collection in range)

References:
  - Hull, J.C. "Options, Futures, and Other Derivatives."
  - MIT Quant Bible (options strategies chapter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm as _norm


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class StrategyLeg:
    option_type: str   # "call" | "put" | "stock"
    strike: float      # K (ignored for stock legs)
    premium: float     # per-share option premium (always positive)
    quantity: int      # +1 = long, -1 = short
    entry_price: float = 0.0  # for stock legs: entry price paid


@dataclass
class StrategyProfile:
    name: str
    legs: list[StrategyLeg]
    net_premium: float          # total debit (positive) or credit (negative) to enter
    max_profit: float | None    # None = unlimited
    max_loss: float | None      # None = unlimited
    breakeven_points: list[float] = field(default_factory=list)
    description: str = ""


# ── Core payoff engine ───────────────────────────────────────────────────────

def _leg_payoff(leg: StrategyLeg, spots: np.ndarray) -> np.ndarray:
    """Terminal payoff for a single leg at expiry (excluding premium)."""
    if leg.option_type == "stock":
        return leg.quantity * (spots - leg.entry_price)
    if leg.option_type == "call":
        intrinsic = np.maximum(spots - leg.strike, 0.0)
    else:  # put
        intrinsic = np.maximum(leg.strike - spots, 0.0)
    # Long premium is a cost; short premium is income
    return leg.quantity * (intrinsic - leg.premium)


def strategy_pnl_curve(
    profile: StrategyProfile,
    spot_range: tuple[float, float],
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the P&L curve at expiry across a spot price range.

    Returns:
        (spots, pnl) arrays — spots in $/share, pnl in $/share.
    """
    spots = np.linspace(spot_range[0], spot_range[1], n_points)
    pnl = sum(_leg_payoff(leg, spots) for leg in profile.legs)
    return spots, np.array(pnl)


# ── Premium estimator (ATM Black-Scholes) ────────────────────────────────────

def estimate_atm_premiums(
    spot: float,
    sigma: float,
    days: int = 30,
    r: float = 0.0525,
) -> tuple[float, float]:
    """
    Quick ATM call and put premium estimates via Black-Scholes.
    Used to pre-fill strategy sliders with realistic values.

    Returns (call_premium, put_premium).
    """
    T = days / 365.0
    if T <= 0 or sigma <= 0 or spot <= 0:
        return 0.0, 0.0
    d1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = spot * _norm.cdf(d1) - spot * np.exp(-r * T) * _norm.cdf(d2)
    put = spot * np.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
    return round(float(call), 4), round(float(put), 4)


def estimate_otm_premium(
    spot: float,
    strike: float,
    sigma: float,
    option_type: str = "call",
    days: int = 30,
    r: float = 0.0525,
) -> float:
    """Black-Scholes price for an OTM option (any strike)."""
    T = days / 365.0
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = spot * _norm.cdf(d1) - strike * np.exp(-r * T) * _norm.cdf(d2)
    else:
        price = strike * np.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
    return round(float(max(price, 0.0)), 4)


# ── Strategy builders ────────────────────────────────────────────────────────

def build_covered_call(spot: float, call_strike: float, call_premium: float) -> StrategyProfile:
    """Long stock + short OTM call. Income strategy; caps upside at call_strike."""
    legs = [
        StrategyLeg("stock", 0.0, 0.0, +1, entry_price=spot),
        StrategyLeg("call", call_strike, call_premium, -1),
    ]
    net = -call_premium  # credit received
    max_profit = (call_strike - spot) + call_premium
    max_loss = spot - call_premium  # stock falls to zero
    be = [spot - call_premium]
    return StrategyProfile(
        "Covered Call", legs, net, max_profit, max_loss, be,
        "Sell upside for income; loss protection limited to premium received."
    )


def build_protective_put(spot: float, put_strike: float, put_premium: float) -> StrategyProfile:
    """Long stock + long put. Portfolio insurance; unlimited upside."""
    legs = [
        StrategyLeg("stock", 0.0, 0.0, +1, entry_price=spot),
        StrategyLeg("put", put_strike, put_premium, +1),
    ]
    net = put_premium
    max_profit = None  # unlimited
    max_loss = (spot - put_strike) + put_premium
    be = [spot + put_premium]
    return StrategyProfile(
        "Protective Put", legs, net, max_profit, max_loss, be,
        "Downside insurance; pay premium for guaranteed floor at put_strike."
    )


def build_straddle(
    strike: float, call_premium: float, put_premium: float, long: bool = True
) -> StrategyProfile:
    """
    Long straddle: bet on large move in either direction.
    Short straddle: sell volatility (profit if price stays near strike).
    """
    sign = +1 if long else -1
    legs = [
        StrategyLeg("call", strike, call_premium, sign),
        StrategyLeg("put", strike, put_premium, sign),
    ]
    total = call_premium + put_premium
    if long:
        name, desc = "Long Straddle", "Profit from large move; max loss = total premium at strike."
        net, max_p, max_l = total, None, total
        be = [strike - total, strike + total]
    else:
        name, desc = "Short Straddle", "Collect premium if price stays near strike; unlimited risk."
        net, max_p, max_l = -total, total, None
        be = [strike - total, strike + total]
    return StrategyProfile(name, legs, net, max_p, max_l, be, desc)


def build_strangle(
    call_strike: float, put_strike: float,
    call_premium: float, put_premium: float,
    long: bool = True,
) -> StrategyProfile:
    """
    Long strangle: cheaper vol bet — requires larger move than straddle.
    Short strangle: wider profit range than short straddle.
    """
    sign = +1 if long else -1
    legs = [
        StrategyLeg("call", call_strike, call_premium, sign),
        StrategyLeg("put", put_strike, put_premium, sign),
    ]
    total = call_premium + put_premium
    if long:
        name, desc = "Long Strangle", "OTM straddle variant — lower cost, requires bigger move."
        net, max_p, max_l = total, None, total
        be = [put_strike - total, call_strike + total]
    else:
        name, desc = "Short Strangle", "Wider range than short straddle; collect premium if stays between strikes."
        net, max_p, max_l = -total, total, None
        be = [put_strike - total, call_strike + total]
    return StrategyProfile(name, legs, net, max_p, max_l, be, desc)


def build_bull_call_spread(
    low_strike: float, high_strike: float,
    low_premium: float, high_premium: float,
) -> StrategyProfile:
    """
    Long lower-strike call + short higher-strike call.
    Bullish; limits both risk and reward.
    """
    legs = [
        StrategyLeg("call", low_strike, low_premium, +1),
        StrategyLeg("call", high_strike, high_premium, -1),
    ]
    net = low_premium - high_premium  # net debit
    max_p = (high_strike - low_strike) - net
    max_l = net
    be = [low_strike + net]
    return StrategyProfile(
        "Bull Call Spread", legs, net, max_p, max_l, be,
        "Defined risk/reward bullish spread; profit if price rises above low_strike."
    )


def build_bear_put_spread(
    low_strike: float, high_strike: float,
    low_premium: float, high_premium: float,
) -> StrategyProfile:
    """
    Long higher-strike put + short lower-strike put.
    Bearish; limited risk and reward.
    """
    legs = [
        StrategyLeg("put", high_strike, high_premium, +1),
        StrategyLeg("put", low_strike, low_premium, -1),
    ]
    net = high_premium - low_premium  # net debit
    max_p = (high_strike - low_strike) - net
    max_l = net
    be = [high_strike - net]
    return StrategyProfile(
        "Bear Put Spread", legs, net, max_p, max_l, be,
        "Defined risk/reward bearish spread; profit if price falls below high_strike."
    )


def build_butterfly(
    low_strike: float, mid_strike: float, high_strike: float,
    low_prem: float, mid_prem: float, high_prem: float,
) -> StrategyProfile:
    """
    Long butterfly (calls): long low + short 2× mid + long high.
    Profits when underlying price stays close to mid_strike at expiry.
    Symmetric strikes: high - mid = mid - low.
    """
    legs = [
        StrategyLeg("call", low_strike, low_prem, +1),
        StrategyLeg("call", mid_strike, mid_prem, -2),
        StrategyLeg("call", high_strike, high_prem, +1),
    ]
    net = low_prem - 2 * mid_prem + high_prem  # net debit
    max_p = (mid_strike - low_strike) - net
    max_l = net
    be = [low_strike + net, high_strike - net]
    return StrategyProfile(
        "Long Butterfly", legs, net, max_p, max_l, be,
        "Profit if price stays near mid_strike; max loss = net debit at wings."
    )


def build_iron_condor(
    put_low: float, put_high: float, call_low: float, call_high: float,
    put_low_prem: float, put_high_prem: float,
    call_low_prem: float, call_high_prem: float,
) -> StrategyProfile:
    """
    Iron Condor: short put spread + short call spread.
    Net credit strategy; profits when price stays inside [put_high, call_low].

    Structure:
        Long  put  @ put_low   (wing protection)
        Short put  @ put_high  (income)
        Short call @ call_low  (income)
        Long  call @ call_high (wing protection)
    """
    legs = [
        StrategyLeg("put", put_low, put_low_prem, +1),
        StrategyLeg("put", put_high, put_high_prem, -1),
        StrategyLeg("call", call_low, call_low_prem, -1),
        StrategyLeg("call", call_high, call_high_prem, +1),
    ]
    net_credit = (put_high_prem - put_low_prem) + (call_low_prem - call_high_prem)
    max_p = net_credit
    max_l = max(put_high - put_low, call_high - call_low) - net_credit
    be = [put_high - net_credit, call_low + net_credit]
    return StrategyProfile(
        "Iron Condor", legs, -net_credit, max_p, max_l, be,
        "Sell premium in both directions; profit if price stays between short strikes."
    )


# ── Catalogue builder (convenience) ─────────────────────────────────────────

def build_all_strategies(
    spot: float,
    sigma: float,
    days: int = 30,
    r: float = 0.0525,
) -> dict[str, StrategyProfile]:
    """
    Build all standard strategies for a given spot, vol, expiry, and rate.
    OTM strikes set at ±5% and ±10% of spot for strangle / condor wings.
    Used to populate the Quant Lab options strategy tab.
    """
    c_atm, p_atm = estimate_atm_premiums(spot, sigma, days, r)
    c_5up = estimate_otm_premium(spot, spot * 1.05, sigma, "call", days, r)
    c_10up = estimate_otm_premium(spot, spot * 1.10, sigma, "call", days, r)
    p_5dn = estimate_otm_premium(spot, spot * 0.95, sigma, "put", days, r)
    p_10dn = estimate_otm_premium(spot, spot * 0.90, sigma, "put", days, r)

    return {
        "Covered Call": build_covered_call(spot, spot * 1.05, c_5up),
        "Protective Put": build_protective_put(spot, spot * 0.95, p_5dn),
        "Long Straddle": build_straddle(spot, c_atm, p_atm, long=True),
        "Short Straddle": build_straddle(spot, c_atm, p_atm, long=False),
        "Long Strangle": build_strangle(spot * 1.05, spot * 0.95, c_5up, p_5dn, long=True),
        "Short Strangle": build_strangle(spot * 1.05, spot * 0.95, c_5up, p_5dn, long=False),
        "Bull Call Spread": build_bull_call_spread(spot, spot * 1.05, c_atm, c_5up),
        "Bear Put Spread": build_bear_put_spread(spot * 0.95, spot, p_5dn, p_atm),
        "Long Butterfly": build_butterfly(spot * 0.95, spot, spot * 1.05, p_5dn, c_atm, c_5up),
        "Iron Condor": build_iron_condor(
            spot * 0.90, spot * 0.95, spot * 1.05, spot * 1.10,
            p_10dn, p_5dn, c_5up, c_10up,
        ),
    }

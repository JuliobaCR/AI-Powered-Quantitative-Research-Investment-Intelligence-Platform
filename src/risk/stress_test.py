"""
Stress Testing & Scenario Analysis.

Historical scenario library covering major market dislocations, plus a
custom-shock engine for user-defined shocks. All results degrade gracefully
to zero-dicts / empty DataFrames on missing data — same contract as the
rest of the platform.

Inspired by gs-quant's scenario framework and standard industry practice
for market risk (FRTB, Basel III stressed-VaR).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Historical scenario library ───────────────────────────────────────────────
# equity_shock: peak-to-trough aggregate price impact (negative = loss)
# vol_multiplier: VIX / realized vol multiplier applied to sigma
# rate_shock: parallel shift to risk-free rate (as decimal, e.g. -0.015 = -150 bps)
# credit_spread_shock: OAS widening (as decimal)

HISTORICAL_SCENARIOS: dict[str, dict] = {
    "2008 GFC (Sep–Nov)": {
        "description": "Lehman Brothers collapse — global credit freeze",
        "equity_shock": -0.42,
        "vol_multiplier": 2.50,
        "rate_shock": -0.015,
        "credit_spread_shock": +0.040,
        "horizon_days": 63,
    },
    "COVID-19 Crash (Feb–Mar 2020)": {
        "description": "Pandemic-driven market dislocation — fastest -34% S&P 500 in history",
        "equity_shock": -0.34,
        "vol_multiplier": 3.00,
        "rate_shock": -0.015,
        "credit_spread_shock": +0.020,
        "horizon_days": 33,
    },
    "Dot-Com Bust (2000–2002)": {
        "description": "NASDAQ -78% — technology overvaluation unwind",
        "equity_shock": -0.49,
        "vol_multiplier": 1.80,
        "rate_shock": -0.055,
        "credit_spread_shock": +0.030,
        "horizon_days": 504,
    },
    "2022 Rate Shock": {
        "description": "Fed rapid tightening cycle — multi-decade high inflation",
        "equity_shock": -0.19,
        "vol_multiplier": 1.40,
        "rate_shock": +0.0425,
        "credit_spread_shock": +0.015,
        "horizon_days": 252,
    },
    "Black Monday (Oct 1987)": {
        "description": "Single-day 22.6% S&P 500 crash — program trading cascade",
        "equity_shock": -0.226,
        "vol_multiplier": 4.00,
        "rate_shock": +0.000,
        "credit_spread_shock": +0.010,
        "horizon_days": 1,
    },
    "Russian Default / LTCM (1998)": {
        "description": "Emerging market contagion + LTCM deleveraging",
        "equity_shock": -0.20,
        "vol_multiplier": 2.00,
        "rate_shock": -0.0075,
        "credit_spread_shock": +0.035,
        "horizon_days": 45,
    },
    "European Debt Crisis (2011)": {
        "description": "Eurozone sovereign stress — Greece/Italy/Spain spread widening",
        "equity_shock": -0.19,
        "vol_multiplier": 1.60,
        "rate_shock": -0.003,
        "credit_spread_shock": +0.025,
        "horizon_days": 126,
    },
    "Flash Crash (May 2010)": {
        "description": "Algorithmic cascade — intraday 9% drop, recovered same day",
        "equity_shock": -0.09,
        "vol_multiplier": 1.70,
        "rate_shock": +0.000,
        "credit_spread_shock": +0.005,
        "horizon_days": 1,
    },
    "Taper Tantrum (2013)": {
        "description": "Fed tapering announcement — sharp rates-driven selloff",
        "equity_shock": -0.06,
        "vol_multiplier": 1.30,
        "rate_shock": +0.014,
        "credit_spread_shock": +0.008,
        "horizon_days": 30,
    },
    "China Devaluation (Aug 2015)": {
        "description": "CNY devaluation surprise — global EM contagion",
        "equity_shock": -0.12,
        "vol_multiplier": 1.80,
        "rate_shock": -0.002,
        "credit_spread_shock": +0.012,
        "horizon_days": 10,
    },
}


@dataclass
class StressResult:
    scenario_name: str
    description: str
    equity_shock_pct: float
    horizon_days: int
    portfolio_pnl_pct: float
    var_95_unstressed_pct: float
    var_95_stressed_pct: float
    per_asset_pnl: dict[str, float] = field(default_factory=dict)
    vol_multiplier: float = 1.0
    rate_shock_bps: float = 0.0


def _parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Daily VaR (%) using historical method."""
    r = returns.dropna()
    if len(r) < 10:
        return 0.0
    return float(-np.percentile(r, (1 - confidence) * 100) * 100)


def run_stress_scenario(
    portfolio_weights: dict[str, float],
    asset_returns: dict[str, pd.Series],
    scenario_name: str = "2008 GFC (Sep–Nov)",
) -> StressResult:
    """
    Apply a named historical stress scenario to a portfolio.

    Portfolio P&L is approximated as the weighted sum of per-asset P&Ls,
    where each asset's P&L reflects the equity shock applied to its
    average daily return over the scenario horizon.
    """
    scen = HISTORICAL_SCENARIOS.get(scenario_name, list(HISTORICAL_SCENARIOS.values())[0])
    equity_shock = scen["equity_shock"]
    vol_mult = scen["vol_multiplier"]
    horizon = scen["horizon_days"]
    rate_shock = scen["rate_shock"]

    per_asset_pnl: dict[str, float] = {}
    portfolio_pnl = 0.0

    # Baseline combined returns (for portfolio VaR)
    valid_tickers = [t for t in portfolio_weights if t in asset_returns and not asset_returns[t].empty]
    if valid_tickers:
        ret_df = pd.concat(
            {t: asset_returns[t] * portfolio_weights[t] for t in valid_tickers}, axis=1
        ).dropna()
        port_ret = ret_df.sum(axis=1)
        var_base = _parametric_var(port_ret)

        # Stressed VaR: shift returns by equity_shock / horizon (daily shock)
        daily_shock = equity_shock / max(horizon, 1)
        stressed_port_ret = port_ret + daily_shock
        var_stressed = _parametric_var(stressed_port_ret * vol_mult)
    else:
        var_base = 0.0
        var_stressed = abs(equity_shock) * 100 * 0.05

    # Per-asset P&L
    for ticker, weight in portfolio_weights.items():
        ret = asset_returns.get(ticker, pd.Series(dtype=float))
        if ret.empty:
            pnl = equity_shock * 100 * weight
        else:
            # Expected P&L = equity_shock per unit × weight × 100%
            pnl = equity_shock * 100 * weight
        per_asset_pnl[ticker] = round(pnl, 2)
        portfolio_pnl += pnl

    return StressResult(
        scenario_name=scenario_name,
        description=scen["description"],
        equity_shock_pct=round(equity_shock * 100, 1),
        horizon_days=horizon,
        portfolio_pnl_pct=round(portfolio_pnl, 2),
        var_95_unstressed_pct=round(var_base, 3),
        var_95_stressed_pct=round(var_stressed, 3),
        per_asset_pnl=per_asset_pnl,
        vol_multiplier=vol_mult,
        rate_shock_bps=round(rate_shock * 10000, 0),
    )


def run_custom_scenario(
    portfolio_weights: dict[str, float],
    asset_returns: dict[str, pd.Series],
    equity_shock_pct: float = -10.0,
    vol_multiplier: float = 1.5,
    rate_shock_bps: float = 100.0,
    scenario_name: str = "Custom Scenario",
) -> StressResult:
    """
    User-defined scenario with arbitrary equity shock, vol multiplier, and rate move.
    """
    custom_scen = {
        "equity_shock": equity_shock_pct / 100.0,
        "vol_multiplier": vol_multiplier,
        "rate_shock": rate_shock_bps / 10000.0,
        "credit_spread_shock": 0.0,
        "horizon_days": 21,
        "description": f"Custom: {equity_shock_pct:+.1f}% equity, ×{vol_multiplier:.1f} vol, {rate_shock_bps:+.0f}bps rates",
    }
    # Temporarily inject into scenarios dict for reuse
    HISTORICAL_SCENARIOS[scenario_name] = custom_scen
    result = run_stress_scenario(portfolio_weights, asset_returns, scenario_name)
    del HISTORICAL_SCENARIOS[scenario_name]
    return result


def stress_scenario_matrix(
    portfolio_weights: dict[str, float],
    asset_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Run all historical scenarios and return a matrix:
    rows = scenario names, columns = tickers + "Portfolio P&L %".
    Values = estimated P&L % under each scenario.
    """
    rows = []
    for scenario_name in HISTORICAL_SCENARIOS:
        result = run_stress_scenario(portfolio_weights, asset_returns, scenario_name)
        row = {"Scenario": scenario_name, **result.per_asset_pnl,
               "Portfolio P&L %": result.portfolio_pnl_pct}
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Scenario")

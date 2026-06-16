"""
Mathematical models for sports betting value analysis.

This module contains the same class of quantitative tools used by
professional sports trading desks. The goal is to identify mispriced
odds — situations where a bookmaker's implied probability is lower than
our estimated true probability.

Models implemented:
  ─ Implied probability & no-vig (Pinnacle-normalisation) fair odds
  ─ Expected Value (EV%) — the fundamental definition of a value bet
  ─ Kelly Criterion (full and fractional) bet sizing
  ─ Two/three-way arbitrage detection with optimal stake distribution
  ─ Independent Poisson match-outcome model (Dixon-Coles baseline)
  ─ Per-bookmaker, per-outcome analysis with OutcomeAnalysis dataclass
  ─ Bankroll recommendation engine with fractional Kelly
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Core mathematics ──────────────────────────────────────────────────────────

def implied_prob(decimal_odds: float) -> float:
    """
    Convert decimal odds to bookmaker's stated (raw) probability.
    Includes the bookmaker's margin — always over-estimates true probability.
    """
    return 1.0 / decimal_odds if decimal_odds > 1.0 else 0.0


def overround(odds_list: list[float]) -> float:
    """
    Bookmaker overround (vig / juice / margin).
    = Σ(1/odds_i) − 1

    Interpretation:
      5%  → for every $100 wagered, $95 expected return long-run.
      0%  → perfectly fair book (never happens in practice).
      Pinnacle: ~2–4%   Bet365: ~5–8%   Retail shops: 10–15%
    """
    return sum(implied_prob(o) for o in odds_list) - 1.0


def no_vig_probs(odds_list: list[float]) -> list[float]:
    """
    Remove bookmaker margin to recover fair (true) probabilities.

    Method: Pinnacle normalisation — divide each raw implied prob by their
    sum so that they sum to 1. This is a proportional vig removal, which
    is the correct approach when the vig is assumed to be distributed
    equally across all outcomes.
    """
    raw = [implied_prob(o) for o in odds_list]
    total = sum(raw)
    return [p / total for p in raw] if total > 0.0 else raw


def fair_decimal_odds(prob: float) -> float:
    """True probability → fair (no-vig) decimal odds."""
    return round(1.0 / prob, 3) if prob > 0.0 else 0.0


def expected_value(our_prob: float, decimal_odds: float) -> float:
    """
    Expected Value % for a single bet.

    EV% = (p × (odds − 1) − (1 − p)) × 100

    This is the long-run profit per $100 staked at the given odds,
    assuming our probability estimate is accurate.

    EV > 0 → value bet (we have an edge over the bookmaker)
    EV < 0 → bookmaker has the edge (expected loss)
    EV = 0 → fair price (break-even proposition)
    """
    return (our_prob * (decimal_odds - 1.0) - (1.0 - our_prob)) * 100.0


def kelly_fraction(our_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly Criterion fraction of bankroll to wager.

    f* = (p·b − q) / b
    where b = decimal_odds − 1  (net profit per unit staked)
          p = our estimated win probability
          q = 1 − p

    Returns 0 when EV ≤ 0. In practice use 25–50% Kelly (fractional)
    to reduce variance: stake = f* × kelly_mult × bankroll.

    The Kelly Criterion maximises the expected logarithm of wealth
    (geometric growth rate) over time. It is not a fixed-profit formula —
    it is a volatility-adjusted growth maximiser.
    """
    b = decimal_odds - 1.0
    if b <= 0.0 or our_prob <= 0.0:
        return 0.0
    f = (our_prob * b - (1.0 - our_prob)) / b
    return max(0.0, f)


# ── Arbitrage detection ───────────────────────────────────────────────────────

def arbitrage_analysis(outcomes: dict[str, float]) -> dict:
    """
    Detect two- or three-way arbitrage across bookmakers.

    Parameters
    ----------
    outcomes : {label: best_decimal_odds}
        One entry per outcome. Use the BEST available odds for each outcome
        (from any bookmaker) to maximise arb opportunity detection.

    Returns
    -------
    dict:
      is_arbitrage    — True if a risk-free profit exists
      profit_pct      — Guaranteed profit % of total stake (0 if no arb)
      margin_total_pct— Σ(1/best_odds) × 100; <100 means arb exists
      stakes_pct      — % of total bankroll to stake per outcome for equal return
    """
    valid = {k: v for k, v in outcomes.items() if v > 1.0}
    if not valid:
        return {"is_arbitrage": False, "profit_pct": 0.0,
                "margin_total_pct": 100.0, "stakes_pct": {}}
    total_inv = sum(1.0 / v for v in valid.values())
    is_arb = total_inv < 1.0
    profit_pct = (1.0 / total_inv - 1.0) * 100.0 if is_arb else 0.0
    stakes_pct = {k: round((1.0 / v) / total_inv * 100.0, 1) for k, v in valid.items()}
    return {
        "is_arbitrage":     is_arb,
        "profit_pct":       round(profit_pct, 3),
        "margin_total_pct": round(total_inv * 100.0, 2),
        "stakes_pct":       stakes_pct,
    }


# ── Poisson match-outcome model ───────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for Poisson(λ). Direct computation — faster than scipy for small k."""
    if lam <= 0.0 or k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_score_matrix(lam_h: float, lam_a: float, max_goals: int = 7) -> np.ndarray:
    """
    Joint score probability matrix.

    mat[h, a] = P(home scores h goals) × P(away scores a goals)

    Assumes independence of home and away goal counts — the standard
    Dixon-Coles (1997) baseline. The low-score correction (0-0, 1-0, 0-1,
    1-1) is omitted here for transparency; it would require fitting the
    ρ parameter from historical data.
    """
    n = max_goals + 1
    mat = np.zeros((n, n), dtype=float)
    for h in range(n):
        ph = _poisson_pmf(h, lam_h)
        for a in range(n):
            mat[h, a] = ph * _poisson_pmf(a, lam_a)
    return mat


def poisson_match_probs(lam_h: float, lam_a: float, max_goals: int = 7) -> dict:
    """
    Full set of match-outcome probabilities from expected goals.

    Parameters
    ----------
    lam_h, lam_a : expected goals for home and away team respectively.
                   Typical range: 0.5–2.5 per team per match.

    Returns
    -------
    dict with:
      home, draw, away         — 1X2 probabilities (sum ≈ 1)
      over_0_5 … over_4_5     — probability of >N total goals
      btts                     — both teams to score probability
      score_matrix             — raw P(h, a) numpy array
      lam_home, lam_away       — echoed input lambdas
    """
    mat = poisson_score_matrix(lam_h, lam_a, max_goals)
    n = max_goals + 1

    home_win = float(np.sum(np.tril(mat, k=-1)))  # h > a
    draw_val  = float(np.trace(mat))               # h == a
    away_win  = float(np.sum(np.triu(mat, k=1)))  # a > h

    # Total goals distribution P(total = t)
    goals_dist = np.zeros(max_goals * 2 + 2, dtype=float)
    for h in range(n):
        for a in range(n):
            goals_dist[h + a] += mat[h, a]

    over_lines = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        key = f"over_{str(line).replace('.', '_')}"
        over_lines[key] = float(goals_dist[int(math.ceil(line)):].sum())

    # BTTS: P(home ≥ 1 AND away ≥ 1)
    btts = float(1.0 - mat[:, 0].sum() - mat[0, :].sum() + mat[0, 0])

    return {
        "home":        round(home_win, 4),
        "draw":        round(draw_val, 4),
        "away":        round(away_win, 4),
        **{k: round(v, 4) for k, v in over_lines.items()},
        "btts":        round(btts, 4),
        "score_matrix": mat,
        "lam_home":    lam_h,
        "lam_away":    lam_a,
    }


# ── Per-match analysis ────────────────────────────────────────────────────────

@dataclass
class OutcomeAnalysis:
    """Full mathematical analysis for one outcome at one bookmaker."""
    outcome:          str    # 'home' | 'draw' | 'away'
    bookmaker:        str
    decimal_odds:     float
    implied_prob_pct: float  # raw implied probability %
    fair_prob_pct:    float  # no-vig normalised probability %
    model_prob_pct:   float  # Poisson model % (= fair_prob if no model)
    ev_pct:           float  # expected value % (positive = value bet)
    kelly_frac:       float  # full Kelly fraction
    overround_pct:    float  # bookmaker margin %
    is_value_bet:     bool   # ev_pct > 0


def analyze_event(
    event: dict,
    model_probs: dict | None = None,
) -> list[OutcomeAnalysis]:
    """
    Full per-bookmaker, per-outcome analysis for one event from The Odds API.

    Parameters
    ----------
    event       : raw event dict from fetch_odds()
    model_probs : optional {home: float, draw: float, away: float} from
                  the Poisson model. Falls back to no-vig probabilities.
    """
    results: list[OutcomeAnalysis] = []
    home_team = event.get("home_team", "Home")
    away_team = event.get("away_team", "Away")

    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if not outcomes:
                continue

            all_odds = [float(o["price"]) for o in outcomes]
            OR = overround(all_odds)
            fair = no_vig_probs(all_odds)

            for i, o in enumerate(outcomes):
                name     = o["name"]
                dec_odds = float(o["price"])
                imp_p    = implied_prob(dec_odds)
                fair_p   = fair[i]

                if name == home_team:
                    key = "home"
                elif name == away_team:
                    key = "away"
                else:
                    key = "draw"

                our_p = (model_probs or {}).get(key, fair_p)
                ev    = expected_value(our_p, dec_odds)
                kf    = kelly_fraction(our_p, dec_odds)

                results.append(OutcomeAnalysis(
                    outcome=key,
                    bookmaker=bk["title"],
                    decimal_odds=round(dec_odds, 3),
                    implied_prob_pct=round(imp_p * 100.0, 2),
                    fair_prob_pct=round(fair_p * 100.0, 2),
                    model_prob_pct=round(our_p * 100.0, 2),
                    ev_pct=round(ev, 2),
                    kelly_frac=round(kf, 5),
                    overround_pct=round(OR * 100.0, 2),
                    is_value_bet=(ev > 0.0),
                ))
    return results


def bankroll_plan(
    analyses: list[OutcomeAnalysis],
    bankroll: float,
    kelly_mult: float = 0.25,
) -> pd.DataFrame:
    """
    Prioritised betting plan for a bankroll.

    Only includes positive-EV bets. Sorted by EV% descending.
    Stake = f* × kelly_mult × bankroll  (fractional Kelly).

    Default kelly_mult = 0.25 (quarter-Kelly) is the standard conservative
    recommendation to reduce variance while preserving most of the growth.
    """
    rows = []
    for a in analyses:
        if not a.is_value_bet or a.kelly_frac <= 0.0:
            continue
        stake  = round(bankroll * a.kelly_frac * kelly_mult, 2)
        profit = round(stake * (a.decimal_odds - 1.0), 2)
        rows.append({
            "Outcome":       a.outcome.title(),
            "Bookmaker":     a.bookmaker,
            "Odds":          a.decimal_odds,
            "Our Prob %":    a.model_prob_pct,
            "Book Prob %":   a.implied_prob_pct,
            "EV %":          a.ev_pct,
            "Full Kelly %":  round(a.kelly_frac * 100.0, 2),
            "Stake ($)":     stake,
            "Profit if Win": profit,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("EV %", ascending=False).reset_index(drop=True)
    return df


def best_odds_per_outcome(event: dict) -> dict[str, tuple[float, str]]:
    """
    Best available odds for each outcome across all bookmakers in an event.

    Returns {outcome_name: (best_decimal_odds, bookmaker_title)}.
    Used as input to arbitrage_analysis().
    """
    best: dict[str, tuple[float, str]] = {}
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for o in market.get("outcomes", []):
                name  = o["name"]
                price = float(o["price"])
                if name not in best or price > best[name][0]:
                    best[name] = (price, bk["title"])
    return best

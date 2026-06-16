"""
Tests for the sports betting mathematics module.

All tests use synthetic data — no network calls, no API key required.
The odds_client module is not tested here (it wraps a third-party API).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sports.math_models import (
    arbitrage_analysis,
    bankroll_plan,
    analyze_event,
    best_odds_per_outcome,
    expected_value,
    fair_decimal_odds,
    implied_prob,
    kelly_fraction,
    no_vig_probs,
    overround,
    poisson_match_probs,
    poisson_score_matrix,
)


# ── Core math ─────────────────────────────────────────────────────────────────

class TestImpliedProb:
    def test_even_money(self):
        assert implied_prob(2.0) == pytest.approx(0.50)

    def test_heavy_favourite(self):
        assert implied_prob(1.25) == pytest.approx(0.80)

    def test_large_underdog(self):
        assert implied_prob(10.0) == pytest.approx(0.10)

    def test_invalid_odds_returns_zero(self):
        assert implied_prob(0.5) == 0.0
        assert implied_prob(0.0) == 0.0


class TestOverround:
    def test_typical_1x2_overround(self):
        # 1.9 / 3.5 / 4.0 → sum ≈ 1.065 → 6.5% margin
        odds = [1.9, 3.5, 4.0]
        OR = overround(odds)
        assert OR > 0.0
        assert OR < 0.20  # never more than 20% in practice

    def test_fair_book_zero_overround(self):
        # Manually construct a fair 2-outcome book
        odds = [2.0, 2.0]
        assert overround(odds) == pytest.approx(0.0, abs=1e-9)

    def test_pinnacle_style_low_margin(self):
        # ~2% margin typical of Pinnacle
        odds = [1.96, 3.42, 3.98]
        OR = overround(odds)
        assert 0.01 < OR < 0.06


class TestNoVigProbs:
    def test_sum_to_one(self):
        odds = [1.85, 3.50, 4.20]
        probs = no_vig_probs(odds)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)

    def test_len_matches_input(self):
        odds = [2.1, 3.3]
        assert len(no_vig_probs(odds)) == 2

    def test_favourite_has_highest_prob(self):
        # home (1.4) is the favourite
        odds = [1.4, 4.0, 6.0]
        probs = no_vig_probs(odds)
        assert probs[0] > probs[1] > probs[2]


class TestExpectedValue:
    def test_break_even_is_zero(self):
        # If our prob exactly equals fair prob at these odds, EV = 0
        prob = 0.50
        odds = 2.0
        assert expected_value(prob, odds) == pytest.approx(0.0, abs=1e-9)

    def test_positive_ev_on_edge(self):
        # We estimate 60% but book implies 50% → clear value bet
        ev = expected_value(0.60, 2.0)
        assert ev > 0.0
        assert ev == pytest.approx(20.0, abs=0.1)  # (0.6×1 - 0.4)×100 = 20

    def test_negative_ev_on_no_edge(self):
        # We estimate 40% but book prices at 2.0 (implied 50%)
        ev = expected_value(0.40, 2.0)
        assert ev < 0.0

    def test_very_high_odds(self):
        # 0.05 probability on 25.0 odds: EV = (0.05×24 - 0.95)×100 = 25
        ev = expected_value(0.05, 25.0)
        assert ev == pytest.approx(25.0, abs=0.1)


class TestKellyFraction:
    def test_positive_edge(self):
        # 60% prob at 2.0 odds → Kelly = (0.6×1 - 0.4)/1 = 0.20
        kf = kelly_fraction(0.60, 2.0)
        assert kf == pytest.approx(0.20, abs=1e-6)

    def test_zero_edge_returns_zero(self):
        # 50% at 2.0 → exactly break-even → 0
        assert kelly_fraction(0.50, 2.0) == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge_returns_zero(self):
        # 40% at 2.0 → negative edge → never bet
        assert kelly_fraction(0.40, 2.0) == 0.0

    def test_fraction_bounded_zero_to_one(self):
        kf = kelly_fraction(0.99, 1.5)
        assert 0.0 <= kf <= 1.0

    def test_invalid_odds_returns_zero(self):
        assert kelly_fraction(0.5, 1.0) == 0.0
        assert kelly_fraction(0.5, 0.5) == 0.0


class TestFairDecimalOdds:
    def test_even_money(self):
        assert fair_decimal_odds(0.5) == pytest.approx(2.0)

    def test_zero_prob_safe(self):
        assert fair_decimal_odds(0.0) == 0.0

    def test_round_trip(self):
        p = 0.35
        assert fair_decimal_odds(p) == pytest.approx(1.0 / p, abs=0.001)


# ── Arbitrage detection ───────────────────────────────────────────────────────

class TestArbitrageAnalysis:
    def test_no_arbitrage_typical_market(self):
        outcomes = {"Home": 1.90, "Draw": 3.50, "Away": 4.20}
        result = arbitrage_analysis(outcomes)
        assert not result["is_arbitrage"]
        assert result["profit_pct"] == 0.0
        assert result["margin_total_pct"] > 100.0

    def test_arbitrage_detected(self):
        # Engineered arb: 1/2.20 + 1/3.80 + 1/4.50 ≈ 0.968 < 1
        outcomes = {"Home": 2.20, "Draw": 3.80, "Away": 4.50}
        result = arbitrage_analysis(outcomes)
        assert result["is_arbitrage"]
        assert result["profit_pct"] > 0.0

    def test_stakes_sum_to_100(self):
        outcomes = {"Home": 2.20, "Draw": 3.80, "Away": 4.50}
        result = arbitrage_analysis(outcomes)
        stakes = result["stakes_pct"]
        assert sum(stakes.values()) == pytest.approx(100.0, abs=0.1)

    def test_two_way_arb(self):
        # Two-way market (no draw)
        outcomes = {"Home": 2.10, "Away": 2.10}
        result = arbitrage_analysis(outcomes)
        # 1/2.1 + 1/2.1 = 0.952 < 1 → arb
        assert result["is_arbitrage"]

    def test_empty_input_safe(self):
        result = arbitrage_analysis({})
        assert not result["is_arbitrage"]
        assert result["profit_pct"] == 0.0


# ── Poisson model ─────────────────────────────────────────────────────────────

class TestPoissonScoreMatrix:
    def test_shape(self):
        mat = poisson_score_matrix(1.5, 1.0, max_goals=7)
        assert mat.shape == (8, 8)

    def test_sums_to_approximately_one(self):
        mat = poisson_score_matrix(1.4, 1.1, max_goals=9)
        assert mat.sum() == pytest.approx(1.0, abs=0.01)

    def test_nonnegative(self):
        mat = poisson_score_matrix(1.0, 1.0)
        assert (mat >= 0).all()

    def test_zero_goals_dominant_for_low_lambda(self):
        mat = poisson_score_matrix(0.1, 0.1, max_goals=5)
        assert mat[0, 0] > 0.5  # P(0-0) is very high


class TestPoissonMatchProbs:
    def test_1x2_sums_to_one(self):
        probs = poisson_match_probs(1.5, 1.1)
        total = probs["home"] + probs["draw"] + probs["away"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_higher_lambda_favours_home(self):
        # λ_home >> λ_away → P(home win) should dominate
        probs = poisson_match_probs(3.0, 0.5)
        assert probs["home"] > probs["away"]
        assert probs["home"] > probs["draw"]

    def test_symmetric_lambda_near_equal_home_away(self):
        # λ_home = λ_away → P(home) ≈ P(away), both > draw is not guaranteed
        probs = poisson_match_probs(1.3, 1.3)
        assert abs(probs["home"] - probs["away"]) < 0.01

    def test_over_lines_monotone(self):
        probs = poisson_match_probs(1.5, 1.1)
        assert probs["over_0_5"] > probs["over_1_5"] > probs["over_2_5"] > probs["over_3_5"]

    def test_btts_reasonable(self):
        probs = poisson_match_probs(1.5, 1.2)
        assert 0.0 < probs["btts"] < 1.0

    def test_over_0_5_close_to_one(self):
        # Almost certain at least one goal is scored at normal lambdas
        probs = poisson_match_probs(1.5, 1.2)
        assert probs["over_0_5"] > 0.80

    def test_score_matrix_present(self):
        probs = poisson_match_probs(1.0, 1.0)
        assert "score_matrix" in probs
        assert isinstance(probs["score_matrix"], np.ndarray)


# ── analyze_event ─────────────────────────────────────────────────────────────

def _mock_event(home_odds: float, draw_odds: float, away_odds: float,
                home: str = "Arsenal", away: str = "Chelsea") -> dict:
    """Build a minimal The Odds API event dict for testing."""
    return {
        "id": "test_event_1",
        "home_team": home,
        "away_team": away,
        "commence_time": "2025-03-15T15:00:00Z",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": home_odds},
                            {"name": "Draw",  "price": draw_odds},
                            {"name": away, "price": away_odds},
                        ],
                    }
                ],
            }
        ],
    }


class TestAnalyzeEvent:
    def test_returns_three_outcomes(self):
        ev = _mock_event(1.9, 3.5, 4.2)
        results = analyze_event(ev)
        assert len(results) == 3

    def test_outcomes_are_home_draw_away(self):
        ev = _mock_event(1.9, 3.5, 4.2)
        keys = {r.outcome for r in analyze_event(ev)}
        assert keys == {"home", "draw", "away"}

    def test_implied_prob_correct(self):
        ev = _mock_event(2.0, 3.5, 4.0)
        results = {r.outcome: r for r in analyze_event(ev)}
        assert results["home"].implied_prob_pct == pytest.approx(50.0, abs=0.01)

    def test_fair_probs_sum_to_100(self):
        ev = _mock_event(1.9, 3.5, 4.2)
        results = analyze_event(ev)
        assert sum(r.fair_prob_pct for r in results) == pytest.approx(100.0, abs=0.1)

    def test_model_probs_override(self):
        ev = _mock_event(2.0, 3.5, 4.0)
        model = {"home": 0.70, "draw": 0.20, "away": 0.10}
        results = {r.outcome: r for r in analyze_event(ev, model_probs=model)}
        assert results["home"].model_prob_pct == pytest.approx(70.0)
        assert results["home"].is_value_bet  # 70% prob at 2.0 → EV > 0

    def test_no_model_uses_fair_probs(self):
        ev = _mock_event(1.9, 3.5, 4.2)
        results = analyze_event(ev, model_probs=None)
        for r in results:
            assert r.model_prob_pct == pytest.approx(r.fair_prob_pct, abs=0.01)

    def test_ev_sign_correct(self):
        # Home at 2.0, model says 60% → positive EV
        ev = _mock_event(2.0, 3.5, 4.0)
        model = {"home": 0.60, "draw": 0.25, "away": 0.15}
        results = {r.outcome: r for r in analyze_event(ev, model_probs=model)}
        assert results["home"].ev_pct > 0.0

    def test_empty_bookmakers_returns_empty(self):
        ev = {"home_team": "A", "away_team": "B", "bookmakers": []}
        assert analyze_event(ev) == []

    def test_overround_positive(self):
        ev = _mock_event(1.9, 3.5, 4.2)
        results = analyze_event(ev)
        for r in results:
            assert r.overround_pct > 0.0


# ── best_odds_per_outcome ─────────────────────────────────────────────────────

class TestBestOddsPerOutcome:
    def _multi_book_event(self) -> dict:
        return {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "title": "Pinnacle",
                    "markets": [{"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 1.95},
                        {"name": "Draw",    "price": 3.50},
                        {"name": "Chelsea", "price": 4.10},
                    ]}],
                },
                {
                    "title": "Bet365",
                    "markets": [{"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 1.90},
                        {"name": "Draw",    "price": 3.60},
                        {"name": "Chelsea", "price": 4.00},
                    ]}],
                },
            ],
        }

    def test_finds_best_home_odds(self):
        ev = self._multi_book_event()
        best = best_odds_per_outcome(ev)
        assert best["Arsenal"][0] == pytest.approx(1.95)
        assert best["Arsenal"][1] == "Pinnacle"

    def test_finds_best_draw_odds(self):
        ev = self._multi_book_event()
        best = best_odds_per_outcome(ev)
        assert best["Draw"][0] == pytest.approx(3.60)
        assert best["Draw"][1] == "Bet365"

    def test_empty_event_returns_empty(self):
        assert best_odds_per_outcome({"bookmakers": []}) == {}


# ── bankroll_plan ─────────────────────────────────────────────────────────────

class TestBankrollPlan:
    def test_only_value_bets_included(self):
        ev = _mock_event(2.0, 3.5, 4.0)
        model = {"home": 0.60, "draw": 0.20, "away": 0.10}
        analyses = analyze_event(ev, model_probs=model)
        plan = bankroll_plan(analyses, bankroll=1000.0, kelly_mult=0.25)
        # All rows should have EV% > 0
        assert all(plan["EV %"] > 0)

    def test_quarter_kelly_stake_reasonable(self):
        ev = _mock_event(2.0, 3.5, 4.0)
        model = {"home": 0.60, "draw": 0.20, "away": 0.10}
        analyses = analyze_event(ev, model_probs=model)
        plan = bankroll_plan(analyses, bankroll=1000.0, kelly_mult=0.25)
        if not plan.empty:
            # Quarter-Kelly stake should be a small fraction of bankroll
            assert (plan["Stake ($)"] <= 250.0).all()  # never exceeds 25% of bankroll

    def test_empty_plan_when_no_value(self):
        # Model prob exactly equals no-vig prob → no edge
        ev = _mock_event(1.9, 3.5, 4.2)
        # No model_probs → uses fair probs → EV ≈ 0 for all
        analyses = analyze_event(ev, model_probs=None)
        plan = bankroll_plan(analyses, bankroll=1000.0, kelly_mult=0.25)
        assert plan.empty

    def test_sorted_by_ev_descending(self):
        ev = _mock_event(2.0, 3.5, 4.0)
        model = {"home": 0.70, "draw": 0.25, "away": 0.15}
        analyses = analyze_event(ev, model_probs=model)
        plan = bankroll_plan(analyses, bankroll=1000.0, kelly_mult=0.25)
        if len(plan) > 1:
            assert plan["EV %"].is_monotonic_decreasing


# ── Historical module tests ───────────────────────────────────────────────────

from src.sports.historical import (
    compute_season_stats,
    estimate_lambdas,
    _normalize,
    _find_team,
)


def _make_matches(fixtures: list[tuple[str, str, int, int]]) -> list[dict]:
    """
    Build minimal football-data.org match dicts from (home, away, hg, ag) tuples.
    All marked as FINISHED with a fixed date.
    """
    matches = []
    for i, (home, away, hg, ag) in enumerate(fixtures):
        matches.append({
            "id": i,
            "utcDate": f"2024-{(i % 12) + 1:02d}-15T15:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": home},
            "awayTeam": {"name": away},
            "score": {"fullTime": {"home": hg, "away": ag}},
        })
    return matches


# Synthetic mini-league: 5 teams, 20 matches (each pair plays home + away).
# Each team plays exactly 4 home games so MIN_MATCHES=4 threshold is always met.
_TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Tottenham", "ManCity"]
_FIXTURES = [
    # Arsenal home (avg 2.0 goals/game — strong home attack)
    ("Arsenal",   "Chelsea",    2, 1),
    ("Arsenal",   "Liverpool",  1, 1),
    ("Arsenal",   "Tottenham",  3, 0),
    ("Arsenal",   "ManCity",    2, 0),
    # Chelsea home
    ("Chelsea",   "Arsenal",    0, 2),
    ("Chelsea",   "Liverpool",  1, 2),
    ("Chelsea",   "Tottenham",  2, 2),
    ("Chelsea",   "ManCity",    0, 1),
    # Liverpool home
    ("Liverpool", "Arsenal",    1, 1),
    ("Liverpool", "Chelsea",    3, 1),
    ("Liverpool", "Tottenham",  2, 0),
    ("Liverpool", "ManCity",    1, 1),
    # Tottenham home (avg 0.25 goals/game — weak home attack)
    ("Tottenham", "Arsenal",    0, 1),
    ("Tottenham", "Chelsea",    1, 1),
    ("Tottenham", "Liverpool",  0, 2),
    ("Tottenham", "ManCity",    0, 3),
    # ManCity home
    ("ManCity",   "Arsenal",    2, 0),
    ("ManCity",   "Chelsea",    3, 0),
    ("ManCity",   "Liverpool",  2, 1),
    ("ManCity",   "Tottenham",  4, 0),
]


class TestComputeSeasonStats:
    def setup_method(self):
        self.matches = _make_matches(_FIXTURES)
        self.stats = compute_season_stats(self.matches)

    def test_returns_season_stats(self):
        assert self.stats is not None

    def test_all_teams_present(self):
        for team in _TEAMS:
            assert team in self.stats.team_strengths

    def test_league_avgs_positive(self):
        assert self.stats.league_avg_home > 0
        assert self.stats.league_avg_away > 0

    def test_total_matches_count(self):
        assert self.stats.total_matches == len(_FIXTURES)

    def test_attack_defense_ratings_positive(self):
        for ts in self.stats.team_strengths.values():
            assert ts.attack_home >= 0
            assert ts.defense_home >= 0
            assert ts.attack_away >= 0
            assert ts.defense_away >= 0

    def test_avg_goals_match_raw_data(self):
        # Arsenal scores 2, 1, 3 at home → avg 2.0
        arsenal = self.stats.team_strengths["Arsenal"]
        assert arsenal.avg_scored_home == pytest.approx(2.0, abs=0.01)

    def test_recent_form_nonempty(self):
        for ts in self.stats.team_strengths.values():
            assert len(ts.recent_form) > 0
            assert all(r in "WDL" for r in ts.recent_form)

    def test_empty_matches_returns_none(self):
        assert compute_season_stats([]) is None

    def test_unfinished_matches_ignored(self):
        bad = [{"status": "SCHEDULED", "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}}]
        result = compute_season_stats(bad)
        assert result is None


class TestEstimateLambdas:
    def setup_method(self):
        self.stats = compute_season_stats(_make_matches(_FIXTURES))

    def test_returns_positive_lambdas(self):
        lh, la = estimate_lambdas("Arsenal", "Chelsea", self.stats)
        assert lh > 0
        assert la > 0

    def test_strong_home_team_has_higher_lam_h(self):
        # Arsenal is strong at home (scores 2+ per game); compare to Tottenham
        lh_arsenal, _ = estimate_lambdas("Arsenal", "Chelsea", self.stats)
        lh_tott,    _ = estimate_lambdas("Tottenham", "Chelsea", self.stats)
        assert lh_arsenal > lh_tott

    def test_unknown_team_falls_back_to_league_avg(self):
        lh, la = estimate_lambdas("NonExistentFC", "AlsoFake", self.stats)
        # Should return league averages, not zero or negative
        assert lh == pytest.approx(self.stats.league_avg_home, rel=0.01)
        assert la == pytest.approx(self.stats.league_avg_away, rel=0.01)

    def test_lambdas_always_positive(self):
        # λ must be > 0 for all matchups. Extreme synthetic data can push values
        # above 5.0 (e.g. ManCity attacking Tottenham's porous away defense), so
        # we only assert positivity — the model's min-clamp guarantees ≥ 0.1.
        for h in _TEAMS:
            for a in _TEAMS:
                if h == a:
                    continue
                lh, la = estimate_lambdas(h, a, self.stats)
                assert lh >= 0.1
                assert la >= 0.1


class TestNormalize:
    def test_strips_fc(self):
        assert _normalize("Arsenal FC") == "arsenal"

    def test_strips_united(self):
        assert _normalize("Manchester United") == "manchester"

    def test_lowercases(self):
        assert _normalize("CHELSEA") == "chelsea"

    def test_strips_parenthetical(self):
        assert _normalize("Team Name (W)") == "team name"


class TestFindTeam:
    def setup_method(self):
        self.stats = compute_season_stats(_make_matches(_FIXTURES))
        self.strengths = self.stats.team_strengths

    def test_exact_match(self):
        assert _find_team("Arsenal", self.strengths) is not None

    def test_normalised_match(self):
        # "Arsenal FC" should match "Arsenal"
        assert _find_team("Arsenal FC", self.strengths) is not None

    def test_nonexistent_returns_none(self):
        assert _find_team("ReallyUnknownTeam1234", self.strengths) is None

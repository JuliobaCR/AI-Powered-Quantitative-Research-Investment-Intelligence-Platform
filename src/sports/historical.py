"""
Historical match data — football-data.org (free API, no cost).

Provides team attack/defense strength estimation for the Poisson model.
Without historical data, λ values would be guesses. With this module,
λ is derived from each team's actual goal-scoring and defensive record.

Model: simplified Dixon-Coles attack/defense parameterisation.
  λ_home = league_avg_home × attack_home × defense_away_opponent
  λ_away = league_avg_away × attack_away × defense_home_opponent

Free tier: 10 requests/minute, no monthly cap.
Sign up:   https://www.football-data.org/client/register
Add key:   FOOTBALL_DATA_KEY=... in your .env file.

Supported competitions (free tier):
  PL  — Premier League (England)
  PD  — Primera División (La Liga)
  BL1 — Bundesliga
  SA  — Serie A
  FL1 — Ligue 1
  CL  — UEFA Champions League
  DED — Eredivisie
  PPL — Primeira Liga
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

FOOTBALL_DATA_KEY: str = os.getenv("FOOTBALL_DATA_KEY", "")
_BASE_URL = "https://api.football-data.org/v4"

# The Odds API sport key → football-data.org competition ID
COMPETITION_MAP: dict[str, str] = {
    "soccer_epl":                    "PL",
    "soccer_spain_la_liga":          "PD",
    "soccer_germany_bundesliga":     "BL1",
    "soccer_italy_serie_a":          "SA",
    "soccer_france_ligue_one":       "FL1",
    "soccer_uefa_champs_league":     "CL",
    "soccer_uefa_europa_league":     "EL",
}

# Minimum past matches needed to compute a reliable strength estimate
MIN_MATCHES = 4


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TeamStrength:
    """Attack/defense ratings derived from a full season of matches."""
    team:              str
    matches_home:      int
    matches_away:      int
    attack_home:       float   # relative to league avg home attack (>1 = strong finisher)
    defense_home:      float   # relative to league avg away attack (<1 = solid at home)
    attack_away:       float
    defense_away:      float
    avg_scored_home:   float
    avg_conceded_home: float
    avg_scored_away:   float
    avg_conceded_away: float
    recent_form:       str     # e.g. "WWDLW" (most recent first)
    goals_for_total:   int
    goals_against_total: int


@dataclass
class SeasonStats:
    """Aggregated league-level and per-team data for one season."""
    competition_id:    str
    season:            int
    total_matches:     int
    league_avg_home:   float   # avg goals per match by home team
    league_avg_away:   float   # avg goals per match by away team
    team_strengths:    dict[str, TeamStrength] = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _current_season() -> int:
    """Return the start year of the current football season (e.g. 2024 for 2024/25)."""
    now = datetime.date.today()
    return now.year - 1 if now.month < 8 else now.year


def _normalize(name: str) -> str:
    """
    Normalise a team name for fuzzy cross-API matching.
    Strips common suffixes (FC, CF, SC, AC, United, City…) and lowercases.
    """
    s = name.strip()
    for token in (" FC", " CF", " SC", " AC", " AS", " CD", " FK", " SK",
                  " Athletic Club", " Atletico", " United", " City", " Town"):
        s = s.replace(token, "")
    # Remove parenthetical suffixes like "(N)" or "(W)"
    s = re.sub(r"\s*\(.*?\)", "", s)
    return s.strip().lower()


def _find_team(name: str, strengths: dict[str, TeamStrength]) -> TeamStrength | None:
    """Look up a team with exact → normalised → prefix fuzzy matching."""
    # 1. exact
    if name in strengths:
        return strengths[name]
    # 2. normalised exact
    norm = _normalize(name)
    for k, v in strengths.items():
        if _normalize(k) == norm:
            return v
    # 3. 5-char prefix
    for k, v in strengths.items():
        kn = _normalize(k)
        if kn.startswith(norm[:5]) or norm.startswith(kn[:5]):
            return v
    return None


# ── API client ────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> dict | list:
    """Authenticated GET against football-data.org."""
    if not FOOTBALL_DATA_KEY:
        return {}
    try:
        resp = requests.get(
            f"{_BASE_URL}{endpoint}",
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
            params=params or {},
            timeout=15,
        )
        if resp.ok:
            return resp.json()
        return {}
    except Exception:
        return {}


def fetch_season_matches(
    competition_id: str,
    season: int | None = None,
) -> list[dict]:
    """
    Fetch all FINISHED matches for a competition/season.

    Returns a list of raw match dicts from football-data.org.
    Empty list on error or missing API key.

    Cost: 1 API request.
    """
    if not FOOTBALL_DATA_KEY:
        return []
    s = season or _current_season()
    data = _get(f"/competitions/{competition_id}/matches",
                {"season": s, "status": "FINISHED"})
    return data.get("matches", []) if isinstance(data, dict) else []


# ── Strength computation ──────────────────────────────────────────────────────

def compute_season_stats(matches: list[dict]) -> SeasonStats | None:
    """
    Compute per-team attack/defense ratings and league averages.

    Returns None if there are no valid matches.

    Algorithm (Dixon-Coles simplification):
      For each team T playing at home:
        attack_home(T)  = avg_goals_scored_home(T)  / league_avg_home
        defense_home(T) = avg_goals_conceded_home(T) / league_avg_away

      Prediction for match Home(H) vs Away(A):
        λ_home = league_avg_home × attack_home(H) × defense_away(A)
        λ_away = league_avg_away × attack_away(A) × defense_home(H)

      A value < 1 on defense means better-than-average (concedes fewer goals).
      A value > 1 on attack means stronger-than-average (scores more goals).
    """
    raw: dict[str, dict] = {}

    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        score = (m.get("score") or {}).get("fullTime") or {}
        h_goals = score.get("home")
        a_goals = score.get("away")
        if h_goals is None or a_goals is None:
            continue

        home = (m.get("homeTeam") or {}).get("name", "")
        away = (m.get("awayTeam") or {}).get("name", "")
        date = m.get("utcDate", "")
        if not home or not away:
            continue

        for team in (home, away):
            if team not in raw:
                raw[team] = {
                    "hs": [], "hc": [], "as_": [], "ac": [],
                    "results": [],   # (date, "W"|"D"|"L")
                }

        raw[home]["hs"].append(int(h_goals))
        raw[home]["hc"].append(int(a_goals))
        raw[away]["as_"].append(int(a_goals))
        raw[away]["ac"].append(int(h_goals))

        # Result from each team's perspective
        if h_goals > a_goals:
            raw[home]["results"].append((date, "W"))
            raw[away]["results"].append((date, "L"))
        elif h_goals < a_goals:
            raw[home]["results"].append((date, "L"))
            raw[away]["results"].append((date, "W"))
        else:
            raw[home]["results"].append((date, "D"))
            raw[away]["results"].append((date, "D"))

    if not raw:
        return None

    # League averages
    all_hs = [g for s in raw.values() for g in s["hs"]]
    all_as = [g for s in raw.values() for g in s["as_"]]
    if not all_hs:
        return None
    lg_h = sum(all_hs) / len(all_hs)
    lg_a = sum(all_as) / len(all_as) if all_as else lg_h * 0.8

    strengths: dict[str, TeamStrength] = {}
    for team, s in raw.items():
        hs, hc = s["hs"], s["hc"]
        as_, ac = s["as_"], s["ac"]

        # Fall back to league avg if too few matches
        avg_hs  = sum(hs)  / len(hs)  if len(hs)  >= MIN_MATCHES else lg_h
        avg_hc  = sum(hc)  / len(hc)  if len(hc)  >= MIN_MATCHES else lg_a
        avg_as  = sum(as_) / len(as_) if len(as_) >= MIN_MATCHES else lg_a
        avg_ac  = sum(ac)  / len(ac)  if len(ac)  >= MIN_MATCHES else lg_h

        atk_h = avg_hs / lg_h  if lg_h > 0 else 1.0
        def_h = avg_hc / lg_a  if lg_a > 0 else 1.0
        atk_a = avg_as / lg_a  if lg_a > 0 else 1.0
        def_a = avg_ac / lg_h  if lg_h > 0 else 1.0

        # Recent form: sort by date descending, take last 10
        sorted_res = sorted(s["results"], key=lambda x: x[0], reverse=True)
        form_str = "".join(r[1] for r in sorted_res[:8])

        strengths[team] = TeamStrength(
            team=team,
            matches_home=len(hs),
            matches_away=len(as_),
            attack_home=round(atk_h, 3),
            defense_home=round(def_h, 3),
            attack_away=round(atk_a, 3),
            defense_away=round(def_a, 3),
            avg_scored_home=round(avg_hs, 2),
            avg_conceded_home=round(avg_hc, 2),
            avg_scored_away=round(avg_as, 2),
            avg_conceded_away=round(avg_ac, 2),
            recent_form=form_str,
            goals_for_total=sum(hs) + sum(as_),
            goals_against_total=sum(hc) + sum(ac),
        )

    return SeasonStats(
        competition_id="",
        season=_current_season(),
        total_matches=len(matches),
        league_avg_home=round(lg_h, 3),
        league_avg_away=round(lg_a, 3),
        team_strengths=strengths,
    )


# ── Lambda estimation ─────────────────────────────────────────────────────────

def estimate_lambdas(
    home_team: str,
    away_team: str,
    stats: SeasonStats,
) -> tuple[float, float]:
    """
    Estimate expected goals (λ_home, λ_away) from historical strength data.

    Falls back gracefully:
      Both teams found → full Dixon-Coles formula
      One team found   → that team's rating × league average for the other
      Neither found    → league averages (same as the old manual defaults)
    """
    lg_h = stats.league_avg_home
    lg_a = stats.league_avg_away
    strengths = stats.team_strengths

    h = _find_team(home_team, strengths)
    a = _find_team(away_team, strengths)

    if h and a:
        lam_h = lg_h * h.attack_home * a.defense_away
        lam_a = lg_a * a.attack_away * h.defense_home
    elif h:
        lam_h = lg_h * h.attack_home
        lam_a = lg_a * h.defense_home
    elif a:
        lam_h = lg_h * a.defense_away
        lam_a = lg_a * a.attack_away
    else:
        lam_h = lg_h
        lam_a = lg_a

    return round(max(0.10, lam_h), 3), round(max(0.10, lam_a), 3)


def form_color(result: str) -> str:
    """CSS color for a result character (W/D/L)."""
    return {"W": "#1D9E75", "D": "#BA7517", "L": "#D85A30"}.get(result, "#7D8590")

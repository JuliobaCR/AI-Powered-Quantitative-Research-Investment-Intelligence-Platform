"""
The Odds API client — real-time odds from 80+ bookmakers worldwide.

1win does not provide a public API. The Odds API is the industry standard
for odds aggregation and covers Pinnacle (sharpest odds in the market),
Bet365, William Hill, Unibet, and 70+ other bookmakers.

Free tier: 500 requests/month.
Each fetch_odds() call costs 1 request and returns ALL upcoming matches.
The dashboard caches responses for 30 minutes to preserve quota.

Sign up: https://the-odds-api.com
Set ODDS_API_KEY in your .env file to activate this module.
"""

from __future__ import annotations

import os

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
_BASE_URL = "https://api.the-odds-api.com/v4"

# Display name → The Odds API sport key
SPORTS: dict[str, str] = {
    "⚽ Premier League":        "soccer_epl",
    "⚽ La Liga":               "soccer_spain_la_liga",
    "⚽ Champions League":      "soccer_uefa_champs_league",
    "⚽ Europa League":         "soccer_uefa_europa_league",
    "⚽ Bundesliga":            "soccer_germany_bundesliga",
    "⚽ Serie A":               "soccer_italy_serie_a",
    "⚽ Ligue 1":               "soccer_france_ligue_one",
    "⚽ Liga MX":               "soccer_mexico_ligamx",
    "⚽ MLS":                   "soccer_usa_mls",
    "⚽ Copa Libertadores":     "soccer_conmebol_copa_libertadores",
    "🏀 NBA":                   "basketball_nba",
    "🏀 EuroLeague":            "basketball_euroleague",
    "🎾 ATP":                   "tennis_atp",
    "🏈 NFL":                   "americanfootball_nfl",
}

# Historical league-average expected goals per team per match (home, away).
# Used as Poisson model defaults when no user input is provided.
LEAGUE_AVG_XG: dict[str, tuple[float, float]] = {
    "soccer_epl":                        (1.44, 1.18),
    "soccer_spain_la_liga":              (1.36, 1.10),
    "soccer_uefa_champs_league":         (1.50, 1.20),
    "soccer_uefa_europa_league":         (1.40, 1.15),
    "soccer_germany_bundesliga":         (1.58, 1.26),
    "soccer_italy_serie_a":              (1.35, 1.10),
    "soccer_france_ligue_one":           (1.38, 1.08),
    "soccer_mexico_ligamx":              (1.30, 1.05),
    "soccer_usa_mls":                    (1.40, 1.15),
    "soccer_conmebol_copa_libertadores": (1.28, 1.08),
}

# ── Internal quota tracking ────────────────────────────────────────────────────

_last_headers: dict[str, str] = {}


def _get(endpoint: str, params: dict) -> list | dict:
    """Internal GET with quota-header capture."""
    global _last_headers
    full_params = {**params, "apiKey": ODDS_API_KEY}
    try:
        resp = requests.get(f"{_BASE_URL}{endpoint}", params=full_params, timeout=12)
        _last_headers = dict(resp.headers)
        if resp.ok:
            return resp.json()
        return []
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_odds(
    sport_key: str,
    regions: str = "eu,uk,us",
    markets: str = "h2h",
) -> list[dict]:
    """
    Fetch current 1X2 (h2h) odds for all upcoming events in a sport.

    Cost: 1 API request per call.
    Returns a list of event dicts; empty list on error or missing key.

    Event structure:
      {id, sport_key, commence_time, home_team, away_team,
       bookmakers: [{title, markets: [{key:'h2h', outcomes:[{name, price}]}]}]}
    """
    if not ODDS_API_KEY:
        return []
    data = _get(f"/sports/{sport_key}/odds", {
        "regions":    regions,
        "markets":    markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    })
    return data if isinstance(data, list) else []


def fetch_scores(sport_key: str, days_from: int = 1) -> list[dict]:
    """Fetch recent/live scores (events from the past N days)."""
    if not ODDS_API_KEY:
        return []
    data = _get(f"/sports/{sport_key}/scores", {"daysFrom": days_from})
    return data if isinstance(data, list) else []


def requests_remaining() -> int | None:
    """API quota remaining for this month, from the last response headers."""
    for key in ("x-requests-remaining", "X-Requests-Remaining"):
        val = _last_headers.get(key)
        if val is not None:
            return int(val)
    return None


def requests_used() -> int | None:
    """API requests consumed this month."""
    for key in ("x-requests-used", "X-Requests-Used"):
        val = _last_headers.get(key)
        if val is not None:
            return int(val)
    return None

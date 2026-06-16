"""
AlphaForge — Streamlit Dashboard.

Entry point: streamlit run src/dashboard/app.py
"""

import math
import sys
import datetime
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from config.settings import PORTFOLIO_CFG
from src.analysis.stats import (
    adf_test,
    garch_vol_estimate,
    hurst_exponent,
    hurst_interpretation,
    jarque_bera_test,
    ljung_box_test,
    return_distribution_summary,
    rolling_autocorrelation,
)
from src.arbitrage.pairs import (
    analyze_pair,
    compute_spread,
    rank_pairs,
    spread_zscore,
)
from src.backtesting.engine import (
    backtest_rsi_meanreversion,
    backtest_sma_crossover,
    sma_param_sweep,
)
from src.derivatives.greeks import greeks_surface
from src.derivatives.strategies import (
    build_all_strategies,
    estimate_atm_premiums,
    strategy_pnl_curve,
)
from src.explainability.alpha_score import compute_alpha_score
from src.factors.fama_french import (
    FACTOR_COLUMNS,
    fetch_ff5_factors,
    rolling_factor_exposures,
    run_factor_regression,
)
from src.forecasting.deep_models import deep_return_forecast
from src.forecasting.models import simple_return_forecast
from src.forecasting.monte_carlo import forecast_fan_percentiles, gbm_price_paths
from src.forecasting.stochastic_processes import (
    fit_heston_params,
    fit_jump_params,
    heston_price_paths,
    jump_diffusion_price_paths,
)
from src.fundamentals.analyzer import analyze_fundamentals
from src.market_data.fetcher import (
    fetch_company_info,
    fetch_multi_ohlcv,
    fetch_ohlcv,
    fetch_quote,
)
from src.market_data.indicators import add_all_indicators
from src.portfolio.optimizer import efficient_frontier, optimize_portfolio
from src.regime_detection.detector import detect_regimes
from src.risk.attribution import full_attribution_report, factor_risk_decomposition
from src.risk.engine import full_risk_report
from src.risk.stress_test import (
    HISTORICAL_SCENARIOS,
    run_custom_scenario,
    run_stress_scenario,
    stress_scenario_matrix,
)
from src.portfolio.demo import INITIAL_CASH as DEMO_INITIAL_CASH, DemoPortfolio
from src.sports.odds_client import (
    LEAGUE_AVG_XG,
    SPORTS,
    fetch_odds,
    requests_remaining,
    requests_used,
)
from src.sports.historical import (
    COMPETITION_MAP,
    SeasonStats,
    TeamStrength,
    compute_season_stats,
    estimate_lambdas,
    fetch_season_matches,
    form_color,
)
from src.sports.math_models import (
    OutcomeAnalysis,
    analyze_event,
    arbitrage_analysis,
    bankroll_plan,
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
from src.valuation.engine import dcf_sensitivity_table, run_valuation

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────
# GitHub Dark-inspired theme: comfortable for long sessions, low visual fatigue.
# Key decisions: #0D1117 bg (no pure black), #C9D1D9 text (no pure white),
# #30363D borders (subtle separation), #58A6FF accents (cool, not harsh).

st.markdown(
    """
    <style>
    /* ── Global ──────────────────────────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html, body, [data-testid="stApp"] {
        background-color: #0D1117;
        color: #C9D1D9;
        font-size: 15px;
        line-height: 1.6;
    }

    /* ── Main content area ───────────────────────────────────────────────── */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1440px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #C9D1D9;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #7D8590;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding-top: 0.6rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #30363D;
        margin: 0.5rem 0;
    }

    /* ── Headings ────────────────────────────────────────────────────────── */
    h1 { font-size: 1.55rem; color: #E6EDF3; letter-spacing: -0.02em; font-weight: 700; }
    h2 { font-size: 1.2rem;  color: #C9D1D9; letter-spacing: -0.01em; font-weight: 600; }
    h3 { font-size: 1.0rem;  color: #C9D1D9; letter-spacing: -0.01em; font-weight: 600; }

    /* ── Metrics ─────────────────────────────────────────────────────────── */
    .stMetric label,
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        color: #7D8590;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E6EDF3;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.78rem;
    }

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] > div > div > div > div button {
        color: #7D8590;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }
    [data-testid="stTabs"] > div > div > div > div button[aria-selected="true"] {
        color: #58A6FF;
        border-bottom-color: #58A6FF;
        background: transparent;
    }
    [data-testid="stTabs"] > div > div > div > div button:hover {
        color: #C9D1D9;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {
        background: #21262D;
        border: 1px solid #30363D;
        color: #C9D1D9;
        border-radius: 6px;
        font-size: 0.85rem;
        padding: 0.4rem 1rem;
        transition: background 0.12s, border-color 0.12s;
    }
    .stButton > button:hover {
        background: #30363D;
        border-color: #58A6FF;
        color: #E6EDF3;
    }
    .stButton > button[kind="primary"] {
        background: #1D9E75;
        border-color: #1D9E75;
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background: #18855E;
        border-color: #18855E;
    }

    /* ── Form inputs ─────────────────────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #C9D1D9;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #58A6FF;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.15);
    }

    /* ── Selectbox ───────────────────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #C9D1D9;
        border-radius: 6px;
    }

    /* ── Sliders ─────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] p {
        color: #7D8590;
        font-size: 0.8rem;
    }

    /* ── Dividers ────────────────────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid #30363D;
        margin: 0.75rem 0;
    }

    /* ── Alert / banner boxes ────────────────────────────────────────────── */
    [data-testid="stSuccess"]  { background: rgba(29,158,117,0.10); border-left: 3px solid #1D9E75; }
    [data-testid="stWarning"]  { background: rgba(186,117,23,0.10); border-left: 3px solid #BA7517; }
    [data-testid="stInfo"]     { background: rgba(83,74,183,0.10);  border-left: 3px solid #534AB7; }
    [data-testid="stError"]    { background: rgba(216,90,48,0.10);  border-left: 3px solid #D85A30; }

    /* ── Dataframe table ─────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #21262D;
        border-radius: 6px;
        overflow: hidden;
    }

    /* ── Caption / small text ────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #7D8590;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    /* ── Scrollbars ──────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #0D1117; }
    ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484F58; }

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] { color: #58A6FF; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Market helpers ─────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    """Return True if NYSE/NASDAQ are currently open (approx, ignores holidays)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # EDT = UTC-4 (Mar–Nov), EST = UTC-5 (Nov–Mar). Use -4 as conservative EST check.
    now_et = now_utc.astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
    if now_et.weekday() >= 5:
        return False
    open_t  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now_et < close_t


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown("## ⚡ AlphaForge")
    st.caption("AI-Powered Quant Research Platform")

    # Market open / closed badge
    if _is_market_open():
        st.markdown(
            '<span style="color:#1D9E75; font-size:0.72rem; font-weight:600;">'
            '● MARKET OPEN</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#7D8590; font-size:0.72rem;">'
            '○ MARKET CLOSED</span>',
            unsafe_allow_html=True,
        )

    # Last updated timestamp
    _now_str = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f'<span style="color:#484F58; font-size:0.68rem;">Updated {_now_str}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.caption("MARKET")
    page = st.radio(
        "Navigation",
        [
            "📊 Overview", "🔬 Research", "📈 Charts 3D",
            "── MODELS ──",
            "🔮 AI Forecast", "🧪 Backtesting", "📐 Factor Lab",
            "── RISK & QUANT ──",
            "🔥 Stress Test", "📊 Quant Lab", "🔄 Pairs Lab",
            "── PORTFOLIO ──",
            "💼 Portfolio", "📈 Demo Portfolio", "⚡ Alpha Score",
            "── SPORTS ──",
            "⚽ Sports Betting",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Watchlist & period ────────────────────────────────────────────────────
    default_tickers = PORTFOLIO_CFG.default_tickers
    tickers_input = st.text_input(
        "Watchlist (comma-separated)",
        value=", ".join(default_tickers),
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    selected = st.selectbox("Primary ticker", tickers)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

    st.divider()

    # ── Live mode auto-refresh ────────────────────────────────────────────────
    live_mode = st.checkbox("🔴 Live mode (auto-refresh)", value=False)
    if live_mode:
        refresh_secs = st.select_slider(
            "Refresh every", options=[30, 60, 120, 300], value=60,
            format_func=lambda s: f"{s}s",
        )
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.document.querySelector('[data-testid="stApp"]').dispatchEvent(
                    new Event('rerun', {{bubbles: true}})
                );
                window.parent.location.reload();
            }}, {refresh_secs * 1000});
            </script>
            """,
            height=0,
        )
        st.caption(f"Page reloads every {refresh_secs}s")

    st.divider()
    st.caption("v0.1.0 · Julio Ricardo Barrios Amador")


# ── Cached data loaders ──────────────────────────────────────────────────────

# TTL policy:
#   30s   — live quotes (refresh often)
#   600s  — price/indicator data (10 min; markets move slowly for historical)
#   1800s — derived analytics that change with new bars (risk, GARCH, regimes)
#   3600s — expensive computations on slow-changing data (frontier, sweep, stress)
#   7200s — very expensive ML training (deep models)
#   86400s— external reference data (Fama-French factors, downloaded once/day)

@st.cache_data(ttl=30, show_spinner=False)
def load_quote(ticker):
    return fetch_quote(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def load_ohlcv(ticker, period):
    return fetch_ohlcv(ticker, period=period)

@st.cache_data(ttl=600, show_spinner=False)
def load_indicators(ticker, period):
    df = load_ohlcv(ticker, period)
    return add_all_indicators(df) if not df.empty else df

@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker):
    return analyze_fundamentals(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def load_valuation(ticker):
    return run_valuation(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def load_regimes(ticker, period):
    df = load_ohlcv(ticker, period)
    return detect_regimes(df["Close"]) if not df.empty else pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def load_risk(ticker, period):
    df = load_ohlcv(ticker, period)
    spx = load_ohlcv("^GSPC", period)  # shared cache — fetched once for all tickers
    bench = spx["Close"] if not spx.empty else None
    return full_risk_report(df["Close"], bench) if not df.empty else {}

@st.cache_data(ttl=600, show_spinner=False)
def load_multi(tickers_tuple, period):
    return fetch_multi_ohlcv(list(tickers_tuple), period=period)

@st.cache_data(ttl=3600, show_spinner=False)
def load_efficient_frontier(tickers_tuple, period):
    data = load_multi(tickers_tuple, period)
    returns = pd.DataFrame({t: d["Returns"] for t, d in data.items() if not d.empty}).dropna()
    return efficient_frontier(returns) if len(returns.columns) >= 2 else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_greeks_surface(spot, sigma, option_type):
    return greeks_surface(spot, sigma, option_type=option_type)

@st.cache_data(ttl=7200, show_spinner=False)
def load_sma_sweep(ticker, period):
    df = load_indicators(ticker, period)
    if df.empty:
        return pd.DataFrame()
    return sma_param_sweep(df, range(5, 41, 5), range(20, 121, 10))

@st.cache_data(ttl=1800, show_spinner=False)
def load_simple_forecast(ticker, period, horizon_days):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return {}
    return simple_return_forecast(df["Close"], horizon_days=horizon_days)

@st.cache_data(ttl=7200, show_spinner=False)
def load_deep_forecast(ticker, period, model_type, horizon_days, epochs=20):
    """Train deep model. Cached for 2h — expensive CPU training."""
    df = load_ohlcv(ticker, period)
    if df.empty:
        return {}
    return deep_return_forecast(
        df["Close"], model_type=model_type, horizon_days=horizon_days, epochs=epochs
    )

@st.cache_data(ttl=1800, show_spinner=False)
def load_gbm_paths(ticker, period, horizon_days, n_paths):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return np.zeros((0, 0))
    r = df["Returns"]
    spot = float(df["Close"].iloc[-1])
    mu = float(r.mean() * 252)
    sigma = float(r.std() * np.sqrt(252))
    return gbm_price_paths(spot, mu, sigma, horizon_days, n_paths)

@st.cache_data(ttl=86400, show_spinner=False)
def load_ff5_factors():
    return fetch_ff5_factors()

@st.cache_data(ttl=3600, show_spinner=False)
def load_factor_exposure(ticker, period):
    df = load_ohlcv(ticker, period)
    factors = load_ff5_factors()
    if df.empty or factors.empty:
        return None
    return run_factor_regression(df["Returns"], ticker=ticker, factors=factors)

@st.cache_data(ttl=3600, show_spinner=False)
def load_rolling_exposures(ticker, period):
    df = load_ohlcv(ticker, period)
    factors = load_ff5_factors()
    if df.empty or factors.empty:
        return pd.DataFrame()
    return rolling_factor_exposures(df["Returns"], factors)

@st.cache_data(ttl=1800, show_spinner=False)
def load_heston_paths(ticker, period, horizon_days, n_paths):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return np.zeros((0, 0))
    r = df["Returns"]
    spot = float(df["Close"].iloc[-1])
    mu = float(r.mean() * 252)
    params = fit_heston_params(r)
    return heston_price_paths(spot, mu, horizon_days=horizon_days, n_paths=n_paths, **params)

@st.cache_data(ttl=1800, show_spinner=False)
def load_jump_paths(ticker, period, horizon_days, n_paths):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return np.zeros((0, 0))
    r = df["Returns"]
    spot = float(df["Close"].iloc[-1])
    mu = float(r.mean() * 252)
    params = fit_jump_params(r)
    return jump_diffusion_price_paths(spot, mu, horizon_days=horizon_days, n_paths=n_paths, **params)

@st.cache_data(ttl=3600, show_spinner=False)
def load_stress_matrix(tickers_tuple, period):
    data = load_multi(tickers_tuple, period)
    weights = {t: 1.0 / len(tickers_tuple) for t in tickers_tuple}
    asset_returns = {t: d["Returns"] for t, d in data.items() if not d.empty}
    return stress_scenario_matrix(weights, asset_returns)

@st.cache_data(ttl=1800, show_spinner=False)
def load_quant_stats(ticker, period):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return {}
    r = df["Returns"]
    return {
        "hurst": hurst_exponent(df["Close"]),
        "adf": adf_test(df["Close"]),
        "lb": ljung_box_test(r),
        "jb": jarque_bera_test(r),
        "dist": return_distribution_summary(r),
    }

@st.cache_data(ttl=1800, show_spinner=False)
def load_garch_vol(ticker, period):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return pd.Series(dtype=float)
    return garch_vol_estimate(df["Returns"])

@st.cache_data(ttl=1800, show_spinner=False)
def load_rolling_autocorr(ticker, period):
    df = load_ohlcv(ticker, period)
    if df.empty:
        return pd.Series(dtype=float)
    return rolling_autocorrelation(df["Returns"])

@st.cache_data(ttl=1800, show_spinner=False)
def load_pair_analysis(ticker_a, ticker_b, period):
    df_a = load_ohlcv(ticker_a, period)
    df_b = load_ohlcv(ticker_b, period)
    if df_a.empty or df_b.empty:
        return None
    return analyze_pair(df_a["Close"], df_b["Close"], ticker_a, ticker_b)

@st.cache_data(ttl=3600, show_spinner=False)
def load_pairs_ranking(tickers_tuple, period):
    data = load_multi(tickers_tuple, period)
    prices = pd.DataFrame({t: d["Close"] for t, d in data.items() if not d.empty})
    return rank_pairs(prices, min_history=63)

@st.cache_data(ttl=1800, show_spinner=False)
def load_attribution(tickers_tuple, period):
    data = load_multi(tickers_tuple, period)
    returns_df = pd.DataFrame({t: d["Returns"] for t, d in data.items() if not d.empty}).dropna()
    if len(returns_df.columns) < 2:
        return {}
    weights = {t: 1.0 / len(returns_df.columns) for t in returns_df.columns}
    return full_attribution_report(weights, returns_df)

@st.cache_data(ttl=1800, show_spinner=False)
def load_odds(sport_key: str) -> list[dict]:
    """Fetch live odds for a sport. Cached 30 min to preserve free-tier quota."""
    return fetch_odds(sport_key)

@st.cache_data(ttl=86400, show_spinner=False)
def load_season_stats(competition_id: str) -> SeasonStats | None:
    """
    Fetch full season match history and compute team attack/defense ratings.
    Cached 24h — results don't change intra-day.
    Cost: 1 API request to football-data.org.
    """
    matches = fetch_season_matches(competition_id)
    if not matches:
        return None
    return compute_season_stats(matches)

# ── Per-page asset selector ──────────────────────────────────────────────────
# Every analytical page can analyze ANY asset independently of the sidebar
# primary ticker. Common assets are pre-loaded; custom field accepts any symbol.

_QUICK_ASSETS = [
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    # Metals / Commodities
    "GC=F",    "SI=F",    "CL=F",    "NG=F",
    # Broad ETFs
    "QQQ",  "SPY",  "IWM",  "DIA",  "VTI",
    # Sector ETFs
    "XLK",  "XLF",  "XLE",  "XLV",  "XLU",
    # Tech mega-caps
    "NVDA", "TSLA", "AMZN", "META", "GOOGL", "MSFT", "AAPL", "AVGO", "AMD",
    # Finance
    "JPM",  "GS",   "BAC",  "V",    "MA",
    # Other sectors
    "BRK-B", "JNJ", "UNH", "XOM", "CVX",
    # Indices
    "^GSPC", "^IXIC", "^DJI", "^VIX", "^TNX",
    # International / Fixed income
    "EWJ",  "EEM",  "VGK",  "TLT",  "HYG",  "LQD",
]


def _ticker_selector(page_key: str) -> str:
    """
    Inline per-page asset selector rendered as two compact controls.

    Shows the user's watchlist first, then a curated list of common
    assets, plus a free-text field for any ticker yFinance recognises
    (BTC-USD, GC=F, ^VIX, etc.). Returns the effective ticker string.
    """
    options = list(dict.fromkeys(tickers + _QUICK_ASSETS))  # watchlist first, no dupes
    default_idx = options.index(selected) if selected in options else 0

    c1, c2 = st.columns([3, 2])
    with c1:
        choice = st.selectbox(
            "Asset", options, index=default_idx,
            key=f"_tksel_{page_key}",
            help="Pick from watchlist or common assets",
        )
    with c2:
        custom = st.text_input(
            "Custom ticker", placeholder="e.g. GC=F  BTC-USD  ^VIX",
            key=f"_tkcust_{page_key}",
            label_visibility="collapsed",
            help="Enter any yFinance symbol to override the selector",
        ).upper().strip()

    effective = custom if custom else choice
    if custom:
        st.caption(f"Analyzing **{custom}** (custom entry)")
    return effective


# ── Pages ────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("Market Overview")

    # Top metrics — use cached quotes (TTL=30s) to avoid redundant API calls
    quotes = {t: load_quote(t) for t in tickers}
    cols = st.columns(len(tickers))
    for col, ticker in zip(cols, tickers):
        q = quotes[ticker]
        with col:
            st.metric(
                label=ticker,
                value=f"${q['price']:.2f}",
                delta=f"{q['change_pct']:+.2f}%",
            )

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Performance")
        data = load_multi(tuple(tickers), period)
        fig = go.Figure()
        for t, df in data.items():
            if df.empty:
                continue
            normalized = df["Close"] / df["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index, y=normalized, name=t, mode="lines",
                line=dict(width=2),
            ))
        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Indexed (Base = 100)",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=0, r=0, t=10, b=0),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Watchlist")
        for ticker in tickers:
            q = quotes[ticker]
            color = "🟢" if q["change_pct"] >= 0 else "🔴"
            st.markdown(
                f"**{ticker}** {color} `${q['price']:.2f}` "
                f"`{q['change_pct']:+.2f}%`"
            )
        st.divider()
        # Market regime for primary ticker
        st.subheader("Market Regime")
        with st.spinner("Detecting regimes..."):
            regime_df = load_regimes(selected, period)
        if not regime_df.empty:
            current = regime_df.iloc[-1]
            st.markdown(f"**{current['Regime_Label']}**")
            st.progress(float(current["Confidence"]))
            st.caption(f"Confidence: {current['Confidence']:.1%}")


def page_research():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("Research")
    with _tsel:
        page_ticker = _ticker_selector("research")

    info = fetch_company_info(page_ticker)
    st.caption(f"**{info['name']}** · {info['sector']} · {info['industry']}")

    tab1, tab2, tab3 = st.tabs(["📋 Fundamentals", "💲 Valuation", "📰 Technicals"])

    with tab1:
        fund = load_fundamentals(page_ticker)
        if fund:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue Growth", f"{fund.revenue_growth_yoy:.1f}%")
            c2.metric("Gross Margin", f"{fund.gross_margin:.1f}%")
            c3.metric("FCF Margin", f"{fund.fcf_margin:.1f}%")
            c4.metric("ROIC", f"{fund.roic:.1f}%")

            st.subheader("Fundamental Score")
            metrics = {
                "Revenue Growth": fund.revenue_growth_yoy,
                "Gross Margin": fund.gross_margin,
                "Operating Margin": fund.operating_margin,
                "FCF Margin": fund.fcf_margin,
                "ROE": fund.roe,
                "ROIC": fund.roic,
            }
            fig = go.Figure(go.Bar(
                x=list(metrics.values()),
                y=list(metrics.keys()),
                orientation="h",
                marker_color=["#1D9E75" if v >= 20 else "#BA7517" if v >= 0 else "#D85A30"
                              for v in metrics.values()],
            ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Value (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Fundamental Score", f"{fund.fundamental_score:.0f} / 100")
        else:
            st.warning("No fundamental data available. Check ticker or API.")

    with tab2:
        val = load_valuation(page_ticker)
        if val:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P/E", f"{val.pe_ratio:.1f}x")
            c2.metric("PEG", f"{val.peg_ratio:.2f}")
            c3.metric("EV/EBITDA", f"{val.ev_ebitda:.1f}x")
            c4.metric("DCF Fair Value", f"${val.dcf_fair_value:.2f}",
                      delta=f"{val.dcf_margin_of_safety:+.1f}% MoS")

            st.subheader("DCF Sensitivity Analysis")
            sens = dcf_sensitivity_table(page_ticker)
            if sens:
                fig = go.Figure(go.Heatmap(
                    z=sens["values"],
                    x=[f"{g*100:.0f}% growth" for g in sens["growths"]],
                    y=[f"{w*100:.0f}% WACC" for w in sens["waccs"]],
                    colorscale="RdYlGn",
                    text=[[f"${v:.0f}" for v in row] for row in sens["values"]],
                    texttemplate="%{text}",
                    colorbar=dict(title="Fair Value"),
                ))
                fig.update_layout(
                    template="plotly_dark", height=320,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                verdict_color = {"Undervalued": "🟢", "Fair": "🟡", "Overvalued": "🔴"}
                st.markdown(f"**Verdict:** {verdict_color.get(val.verdict, '⚪')} {val.verdict}")

    with tab3:
        df = load_indicators(page_ticker, period)
        if not df.empty:
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.25, 0.20],
                vertical_spacing=0.04,
            )
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="OHLC",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                                     line=dict(color="#534AB7", width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                                     line=dict(color="#BA7517", width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                                     line=dict(color="gray", width=0.8, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                                     line=dict(color="gray", width=0.8, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(128,128,128,0.07)"), row=1, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                                     line=dict(color="#1D9E75")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            # MACD
            colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in df["MACD_Hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
                                 marker_color=colors), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                     line=dict(color="#534AB7")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                     line=dict(color="#BA7517")), row=3, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=620, showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig, use_container_width=True)


def page_charts_3d():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("3D Analytics")
    with _tsel:
        page_ticker = _ticker_selector("charts3d")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Volatility Surface",
        "🔷 Return–Risk–Score",
        "🌀 Regime Timeline 3D",
        "📐 Correlation Cube",
        "🧮 Greeks Lab 3D",
    ])

    with tab1:
        st.subheader("Implied Volatility Surface (Simulated)")
        st.caption("Strikes × Expirations × Implied Vol — based on Black-Scholes term structure")
        df = load_ohlcv(page_ticker, "1y")
        if not df.empty:
            price = df["Close"].iloc[-1]
            hist_vol = df["Returns"].std() * np.sqrt(252)

            strikes = np.linspace(price * 0.70, price * 1.30, 25)
            expirations = np.array([7, 14, 30, 60, 90, 120, 180, 270, 360])
            K, T = np.meshgrid(strikes, expirations)
            moneyness = np.log(K / price)
            smile = hist_vol * (1 + 0.5 * moneyness ** 2 - 0.1 * moneyness)
            term = 1 - 0.15 * np.exp(-T / 90)
            IV = smile * term * np.sqrt(T / 365) * np.sqrt(252)
            IV = np.clip(IV, 0.05, 1.50)

            fig = go.Figure(data=[go.Surface(
                x=strikes, y=expirations, z=IV,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
                },
            )])
            fig.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Expiry",
                    zaxis_title="Implied Vol",
                    camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                ),
                height=580,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Return — Risk — Alpha Score (3D Scatter)")
        data = load_multi(tuple(tickers), period)
        rows = []
        for t, df in data.items():
            if df.empty or len(df) < 30:
                continue
            r = df["Returns"].dropna()
            ann_ret = (1 + r.mean()) ** 252 - 1
            ann_vol = r.std() * np.sqrt(252)
            fund = load_fundamentals(t)   # cached — avoids redundant API calls per ticker
            val = load_valuation(t)
            score = compute_alpha_score(
                ticker=t,
                fundamental_score=fund.fundamental_score if fund else 50,
                market_trend_score=50,
                news_sentiment_score=50,
                valuation_score=val.valuation_score if val else 50,
                risk_score=50,
            ).total_score
            rows.append({
                "Ticker": t,
                "Return (%)": round(ann_ret * 100, 2),
                "Volatility (%)": round(ann_vol * 100, 2),
                "Alpha Score": round(score, 1),
            })

        if rows:
            plot_df = pd.DataFrame(rows)
            fig = go.Figure(data=[go.Scatter3d(
                x=plot_df["Volatility (%)"],
                y=plot_df["Return (%)"],
                z=plot_df["Alpha Score"],
                mode="markers+text",
                text=plot_df["Ticker"],
                textposition="top center",
                marker=dict(
                    size=plot_df["Alpha Score"] / 5 + 5,
                    color=plot_df["Alpha Score"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Alpha Score"),
                    opacity=0.85,
                    line=dict(color="white", width=0.5),
                ),
            )])
            fig.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Volatility (%)",
                    yaxis_title="Return (%)",
                    zaxis_title="Alpha Score",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
                ),
                height=580,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Market Regime Timeline (3D)")
        regime_df = load_regimes(page_ticker, period)
        ohlcv = load_ohlcv(page_ticker, period)
        if not regime_df.empty and not ohlcv.empty:
            merged = pd.concat([
                ohlcv["Close"].rename("Price"),
                ohlcv["Returns"].rename("Returns"),
                regime_df["Regime"],
                regime_df["Confidence"],
            ], axis=1).dropna()
            merged["DateOrd"] = np.arange(len(merged))

            COLOR_MAP = {0: "#1D9E75", 1: "#BA7517", 2: "#D85A30", 3: "#A32D2D"}
            LABEL_MAP = {0: "Bull/LowVol", 1: "Bull/HiVol", 2: "Bear/LowVol", 3: "Bear/HiVol"}

            fig = go.Figure()
            for regime_id in sorted(merged["Regime"].unique()):
                mask = merged["Regime"] == regime_id
                subset = merged[mask]
                fig.add_trace(go.Scatter3d(
                    x=subset["DateOrd"],
                    y=subset["Price"],
                    z=subset["Confidence"],
                    mode="markers",
                    name=LABEL_MAP.get(regime_id, f"State {regime_id}"),
                    marker=dict(
                        size=3,
                        color=COLOR_MAP.get(regime_id, "#888"),
                        opacity=0.75,
                    ),
                ))
            fig.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Trading Days",
                    yaxis_title="Price ($)",
                    zaxis_title="Regime Confidence",
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
                ),
                height=580,
                legend=dict(orientation="h", y=-0.1),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Return Correlation Heatmap")
        data = load_multi(tuple(tickers), period)
        returns_df = pd.DataFrame({
            t: d["Returns"] for t, d in data.items() if not d.empty
        }).dropna()
        if len(returns_df.columns) >= 2:
            corr = returns_df.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                colorbar=dict(title="Pearson r"),
            ))
            fig.update_layout(
                template="plotly_dark",
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 3D version of rolling correlation
            st.subheader("Rolling Correlation Over Time (3D)")
            if len(returns_df.columns) >= 2:
                t1, t2 = returns_df.columns[0], returns_df.columns[1]
                window = 30
                rolling_corr = returns_df[t1].rolling(window).corr(returns_df[t2]).dropna()
                price1 = load_ohlcv(t1, period)["Close"].reindex(rolling_corr.index).ffill()
                price2 = load_ohlcv(t2, period)["Close"].reindex(rolling_corr.index).ffill()

                fig2 = go.Figure(data=[go.Scatter3d(
                    x=np.arange(len(rolling_corr)),
                    y=price1.values,
                    z=rolling_corr.values,
                    mode="lines",
                    line=dict(
                        color=rolling_corr.values,
                        colorscale="RdBu",
                        width=4,
                    ),
                    name=f"{t1}–{t2} rolling corr",
                )])
                fig2.update_layout(
                    template="plotly_dark",
                    scene=dict(
                        xaxis_title="Trading Days",
                        yaxis_title=f"{t1} Price",
                        zaxis_title="30d Correlation",
                    ),
                    height=500,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.subheader(f"Options Greeks Surface (Black-Scholes) — {page_ticker}")
        st.caption(
            "Strikes × Days-to-Expiry → Greek value. Computed from the Black-Scholes "
            "model using historical volatility as a proxy for IV — fully interactive: "
            "drag to rotate, scroll to zoom."
        )
        df = load_ohlcv(page_ticker, "1y")
        if not df.empty:
            spot = float(df["Close"].iloc[-1])
            hist_vol = float(df["Returns"].std() * np.sqrt(252))

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                option_type = st.radio(
                    "Option Type", ["call", "put"], horizontal=True, key="greeks_opt_type"
                )
            with c2:
                greek = st.selectbox(
                    "Greek",
                    ["delta", "gamma", "vega", "theta", "rho"],
                    format_func=lambda g: g.title(),
                    key="greeks_select",
                )
            with c3:
                sigma = st.slider(
                    "Volatility σ (annualized)", 0.05, 1.50,
                    float(round(min(max(hist_vol, 0.05), 1.50), 2)), 0.01,
                    key="greeks_sigma",
                )

            surf = load_greeks_surface(spot, sigma, option_type)

            GREEK_COLORSCALES = {
                "delta": "RdYlGn",
                "gamma": "Viridis",
                "vega": "Viridis",
                "theta": "RdBu",
                "rho": "Cividis",
            }
            GREEK_TITLES = {
                "delta": "Delta (∂V/∂S)",
                "gamma": "Gamma (∂²V/∂S²)",
                "vega": "Vega (∂V/∂σ)",
                "theta": "Theta (∂V/∂t, $/day)",
                "rho": "Rho (∂V/∂r)",
            }

            fig = go.Figure(data=[go.Surface(
                x=surf["strikes"], y=surf["expirations"], z=surf[greek],
                colorscale=GREEK_COLORSCALES.get(greek, "Viridis"),
                colorbar=dict(title=greek.title()),
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
                },
            )])
            fig.update_layout(
                template="plotly_dark",
                title=f"{GREEK_TITLES.get(greek, greek.title())} — {option_type.title()} · Spot ${spot:.2f}",
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Expiry",
                    zaxis_title=GREEK_TITLES.get(greek, greek.title()),
                    camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                ),
                height=580,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


def page_ai_forecast():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("AI Forecast")
    with _tsel:
        page_ticker = _ticker_selector("forecast")
    st.caption(
        "Multi-model return forecasting: Gradient Boosting, LSTM/Transformer (MC-Dropout CI), "
        "GBM, Heston Stochastic Vol, and Merton Jump-Diffusion Monte Carlo. "
        "Statistical outputs only — not investment advice."
    )

    df = load_ohlcv(page_ticker, period)
    if df.empty:
        st.warning("No data available. Check ticker or API.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon_days = st.slider("Forecast horizon (days)", 5, 60, 20)
    with c2:
        model_type = st.radio(
            "Deep model", ["lstm", "transformer"], format_func=lambda m: m.upper(),
            horizontal=True, key="forecast_model_type",
        )
    with c3:
        n_paths = st.slider("MC paths", 10, 100, 30, step=10)
    with c4:
        train_epochs = st.slider("Training epochs", 5, 60, 20, step=5,
                                 help="More epochs = better model, longer wait. Result cached 2h.")

    # ── GBM baseline — always fast ─────────────────────────────────────────
    with st.spinner("Computing GBM forecast..."):
        gbm_fc = load_simple_forecast(page_ticker, period, horizon_days)

    # ── Deep Learning — opt-in to avoid slow training on every page load ───
    train_deep = st.checkbox(
        f"🔮 Train {model_type.upper()} model (first run: ~10–30 s, then cached 2 h)",
        key="forecast_train_deep",
    )
    if train_deep:
        with st.spinner(f"Training {model_type.upper()} ({train_epochs} epochs)…"):
            deep_fc = load_deep_forecast(page_ticker, period, model_type, horizon_days, train_epochs)
    else:
        deep_fc = {}

    st.divider()
    st.subheader("Expected Return — Model Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gradient Boosting (baseline)**")
        st.metric("Expected Return", f"{gbm_fc.get('expected_return_pct', 0):.2f}%")
        st.caption(
            f"95% CI: [{gbm_fc.get('ci_low_pct', 0):.2f}%, {gbm_fc.get('ci_high_pct', 0):.2f}%]"
        )
    with c2:
        st.markdown(f"**{model_type.upper()} (deep learning, MC-Dropout)**")
        if deep_fc:
            st.metric("Expected Return", f"{deep_fc.get('expected_return_pct', 0):.2f}%")
            st.caption(
                f"95% CI: [{deep_fc.get('ci_low_pct', 0):.2f}%, {deep_fc.get('ci_high_pct', 0):.2f}%] · "
                f"val loss: {deep_fc.get('val_loss', 0):.6f}"
            )
        else:
            st.info(f"Enable the checkbox above to train {model_type.upper()}.")
    st.caption(f"Forecast horizon: {horizon_days} trading days")

    st.divider()
    st.subheader("Monte Carlo Price Cone (GBM)")
    paths = load_gbm_paths(page_ticker, period, horizon_days, n_paths)
    if paths.size == 0:
        st.info("Not enough data to run the Monte Carlo simulation.")
        return

    fan = forecast_fan_percentiles(paths)
    hist = df["Close"].tail(90)
    x_fan = pd.bdate_range(start=hist.index[-1], periods=horizon_days + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Historical", line=dict(color="#E5E5E5", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_fan, y=fan["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_fan, y=fan["p5"], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(83,74,183,0.15)", name="5–95% range",
    ))
    fig.add_trace(go.Scatter(
        x=x_fan, y=fan["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_fan, y=fan["p25"], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(83,74,183,0.30)", name="25–75% range",
    ))
    fig.add_trace(go.Scatter(
        x=x_fan, y=fan["p50"], name="Median path", line=dict(color="#534AB7", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Price ($)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Fan (3D) — GBM")
    st.caption(
        f"{n_paths} simulated GBM price paths over {horizon_days} trading days — "
        "drag to rotate, scroll to zoom."
    )
    days = np.arange(paths.shape[1])
    path_idx = np.arange(paths.shape[0])
    fig3d = go.Figure(data=[go.Surface(
        x=days, y=path_idx, z=paths,
        colorscale="Viridis",
        colorbar=dict(title="Price ($)"),
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
        },
    )])
    fig3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Days Ahead",
            yaxis_title="Simulation Path",
            zaxis_title="Price ($)",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
        ),
        height=580,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ── Advanced stochastic process tabs — opt-in ────────────────────────────
    st.divider()
    st.subheader("Advanced Stochastic Process Comparison")
    st.caption(
        "Heston and Merton Jump-Diffusion processes are calibrated automatically "
        "from historical returns. Click to run — results cached 30 min."
    )

    if st.button("🌀 Run Heston + Jump-Diffusion Simulations", key="run_stochastic_btn"):
        st.session_state["_run_stochastic"] = True

    if not st.session_state.get("_run_stochastic", False):
        st.info(
            "Click the button above to run advanced stochastic simulations. "
            "First run may take a few seconds."
        )
        return  # stop rendering the rest of this page until requested

    proc_tab1, proc_tab2 = st.tabs(["🌀 Heston Stochastic Vol", "⚡ Jump-Diffusion (Merton)"])

    with proc_tab1:
        r = df["Returns"]
        spot = float(df["Close"].iloc[-1])
        heston_params = fit_heston_params(r)
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("v₀ (init var)", f"{heston_params['v0']:.4f}", help="Initial instantaneous variance")
        hc2.metric("κ (kappa)", f"{heston_params['kappa']:.2f}", help="Mean-reversion speed")
        hc3.metric("ξ (xi) vol-of-vol", f"{heston_params['xi']:.3f}", help="Volatility of variance")
        hc4.metric("ρ (rho)", f"{heston_params['rho']:.3f}", help="Spot-vol correlation (leverage effect)")

        with st.spinner("Simulating Heston paths..."):
            h_paths = load_heston_paths(page_ticker, period, horizon_days, n_paths)

        if h_paths.size > 0:
            h_fan = forecast_fan_percentiles(h_paths)
            hist_close = df["Close"].tail(90)
            x_fan = pd.bdate_range(start=hist_close.index[-1], periods=horizon_days + 1)

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=hist_close.index, y=hist_close.values,
                                       name="Historical", line=dict(color="#E5E5E5", width=1.5)))
            fig_h.add_trace(go.Scatter(x=x_fan, y=h_fan["p95"], line=dict(width=0),
                                       showlegend=False, hoverinfo="skip"))
            fig_h.add_trace(go.Scatter(x=x_fan, y=h_fan["p5"], line=dict(width=0),
                                       fill="tonexty", fillcolor="rgba(29,158,117,0.15)",
                                       name="5–95% range"))
            fig_h.add_trace(go.Scatter(x=x_fan, y=h_fan["p75"], line=dict(width=0),
                                       showlegend=False, hoverinfo="skip"))
            fig_h.add_trace(go.Scatter(x=x_fan, y=h_fan["p25"], line=dict(width=0),
                                       fill="tonexty", fillcolor="rgba(29,158,117,0.30)",
                                       name="25–75% range"))
            fig_h.add_trace(go.Scatter(x=x_fan, y=h_fan["p50"], name="Median (Heston)",
                                       line=dict(color="#1D9E75", width=2)))
            fig_h.update_layout(template="plotly_dark", yaxis_title="Price ($)",
                                 height=380, legend=dict(orientation="h", y=-0.15),
                                 margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_h, use_container_width=True)

            st.subheader("Heston Fan (3D)")
            st.caption("Stochastic-vol paths — vol itself evolves, producing fat-tailed distributions.")
            h_days = np.arange(h_paths.shape[1])
            h_idx = np.arange(h_paths.shape[0])
            fig_h3d = go.Figure(data=[go.Surface(
                x=h_days, y=h_idx, z=h_paths, colorscale="Teal",
                colorbar=dict(title="Price ($)"),
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )])
            fig_h3d.update_layout(
                template="plotly_dark",
                scene=dict(xaxis_title="Days", yaxis_title="Path",
                           zaxis_title="Price ($)",
                           camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))),
                height=560, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_h3d, use_container_width=True)

    with proc_tab2:
        jump_params = fit_jump_params(df["Returns"])
        jc1, jc2, jc3, jc4 = st.columns(4)
        jc1.metric("σ (diffusion)", f"{jump_params['sigma']:.3f}", help="Diffusion volatility (excl. jumps)")
        jc2.metric("λ (intensity)", f"{jump_params['lam']:.3f}", help="Expected jumps per year")
        jc3.metric("μⱼ (mean jump)", f"{jump_params['mu_j']:.4f}", help="Mean log-jump size")
        jc4.metric("σⱼ (jump vol)", f"{jump_params['sigma_j']:.3f}", help="Jump size std dev")

        with st.spinner("Simulating Jump-Diffusion paths..."):
            jd_paths = load_jump_paths(page_ticker, period, horizon_days, n_paths)

        if jd_paths.size > 0:
            jd_fan = forecast_fan_percentiles(jd_paths)
            x_fan = pd.bdate_range(start=df["Close"].index[-1], periods=horizon_days + 1)
            hist_close = df["Close"].tail(90)

            fig_jd = go.Figure()
            fig_jd.add_trace(go.Scatter(x=hist_close.index, y=hist_close.values,
                                        name="Historical", line=dict(color="#E5E5E5", width=1.5)))
            fig_jd.add_trace(go.Scatter(x=x_fan, y=jd_fan["p95"], line=dict(width=0),
                                        showlegend=False, hoverinfo="skip"))
            fig_jd.add_trace(go.Scatter(x=x_fan, y=jd_fan["p5"], line=dict(width=0),
                                        fill="tonexty", fillcolor="rgba(216,90,48,0.15)",
                                        name="5–95% range"))
            fig_jd.add_trace(go.Scatter(x=x_fan, y=jd_fan["p75"], line=dict(width=0),
                                        showlegend=False, hoverinfo="skip"))
            fig_jd.add_trace(go.Scatter(x=x_fan, y=jd_fan["p25"], line=dict(width=0),
                                        fill="tonexty", fillcolor="rgba(216,90,48,0.30)",
                                        name="25–75% range"))
            fig_jd.add_trace(go.Scatter(x=x_fan, y=jd_fan["p50"], name="Median (Jump-Diffusion)",
                                        line=dict(color="#D85A30", width=2)))
            fig_jd.update_layout(template="plotly_dark", yaxis_title="Price ($)",
                                  height=380, legend=dict(orientation="h", y=-0.15),
                                  margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_jd, use_container_width=True)

            st.subheader("Jump-Diffusion Fan (3D)")
            st.caption("Jump paths show sudden discontinuities — captures crash/rally risk missing from GBM.")
            jd_days = np.arange(jd_paths.shape[1])
            jd_idx = np.arange(jd_paths.shape[0])
            fig_jd3d = go.Figure(data=[go.Surface(
                x=jd_days, y=jd_idx, z=jd_paths, colorscale="Reds",
                colorbar=dict(title="Price ($)"),
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )])
            fig_jd3d.update_layout(
                template="plotly_dark",
                scene=dict(xaxis_title="Days", yaxis_title="Path",
                           zaxis_title="Price ($)",
                           camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))),
                height=560, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_jd3d, use_container_width=True)


def page_portfolio():
    st.title("Portfolio Analytics")

    data = load_multi(tuple(tickers), period)
    returns_df = pd.DataFrame({
        t: d["Returns"] for t, d in data.items() if not d.empty
    }).dropna()

    if len(returns_df.columns) < 2:
        st.warning("Need at least 2 tickers with data to run portfolio optimization.")
        return

    c1, c2 = st.columns([2, 1])

    with c2:
        strategy = st.selectbox(
            "Optimization Strategy",
            ["mean_variance", "risk_parity", "min_variance", "equal_weight"],
            format_func=lambda s: s.replace("_", " ").title(),
        )

    with st.spinner("Optimizing portfolio..."):
        result = optimize_portfolio(returns_df, strategy=strategy)
        ef = load_efficient_frontier(tuple(tickers), period)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return", f"{result.expected_return*100:.1f}%")
    c2.metric("Expected Volatility", f"{result.expected_volatility*100:.1f}%")
    c3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    c4.metric("Strategy", strategy.replace("_", " ").title())

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Weights")
        weights = {t: float(w) for t, w in zip(result.tickers, result.weights)}
        fig = go.Figure(go.Pie(
            labels=list(weights.keys()),
            values=[round(w * 100, 2) for w in weights.values()],
            hole=0.5,
            textinfo="label+percent",
            marker=dict(colors=["#1D9E75", "#534AB7", "#BA7517", "#D85A30"]),
        ))
        fig.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Efficient Frontier")
        if not ef.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ef["Volatility"], y=ef["Return"],
                mode="lines",
                line=dict(color="#534AB7", width=2),
                name="Efficient Frontier",
            ))
            # Mark optimal portfolio
            fig.add_trace(go.Scatter(
                x=[result.expected_volatility * 100],
                y=[result.expected_return * 100],
                mode="markers",
                marker=dict(size=14, color="#1D9E75", symbol="star"),
                name="Optimal Portfolio",
            ))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Individual Risk Reports")
    risk_cols = st.columns(len(tickers))
    for col, ticker in zip(risk_cols, tickers):
        risk = load_risk(ticker, period)
        with col:
            st.markdown(f"**{ticker}**")
            st.metric("Sharpe", f"{risk.get('sharpe_ratio', 0):.2f}")
            st.metric("Max DD", f"{risk.get('max_drawdown_pct', 0):.1f}%")
            st.metric("Ann. Vol", f"{risk.get('annual_volatility_pct', 0):.1f}%")
            st.metric("VaR 95%", f"{risk.get('var_95_daily_pct', 0):.2f}%")

    # ── Risk Attribution ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Portfolio Risk Attribution")
    st.caption("Equal-weight portfolio — component VaR, marginal VaR, and volatility contribution.")

    with st.spinner("Computing risk attribution..."):
        attr = load_attribution(tuple(tickers), period)

    if attr:
        comp_var = attr.get("component_var", {})
        marg_var = attr.get("marginal_var", {})
        vol_contrib = attr.get("vol_contribution", {})

        at1, at2, at3 = st.columns(3)

        with at1:
            st.markdown("**Component VaR (daily %)**")
            if comp_var:
                fig_cv = go.Figure(go.Bar(
                    x=list(comp_var.keys()), y=list(comp_var.values()),
                    marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in comp_var.values()],
                    text=[f"{v:.3f}%" for v in comp_var.values()],
                    textposition="outside",
                ))
                fig_cv.update_layout(template="plotly_dark", height=280,
                                     yaxis_title="VaR %", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_cv, use_container_width=True)

        with at2:
            st.markdown("**Volatility Contribution (%)**")
            if vol_contrib:
                fig_vc = go.Figure(go.Pie(
                    labels=list(vol_contrib.keys()),
                    values=[abs(v) for v in vol_contrib.values()],
                    hole=0.5,
                    textinfo="label+percent",
                    marker=dict(colors=["#1D9E75", "#534AB7", "#BA7517", "#D85A30", "#888"]),
                ))
                fig_vc.update_layout(template="plotly_dark", height=280,
                                     showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vc, use_container_width=True)

        with at3:
            st.markdown("**Marginal VaR per +1% Weight**")
            if marg_var:
                fig_mv = go.Figure(go.Bar(
                    x=list(marg_var.keys()), y=list(marg_var.values()),
                    marker_color=["#D85A30" if v > 0 else "#1D9E75" for v in marg_var.values()],
                    text=[f"{v:+.4f}%" for v in marg_var.values()],
                    textposition="outside",
                ))
                fig_mv.update_layout(template="plotly_dark", height=280,
                                     yaxis_title="ΔVaR / Δw", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_mv, use_container_width=True)


def page_stress_test():
    st.title("Stress Test & Scenario Analysis")
    st.caption(
        "Historical stress scenarios applied to a equal-weight portfolio of your watchlist. "
        "Inspired by Basel III stressed-VaR and industry practice for market risk management."
    )

    data = load_multi(tuple(tickers), period)
    asset_returns = {t: d["Returns"] for t, d in data.items() if not d.empty}
    weights = {t: 1.0 / len(tickers) for t in tickers}

    if not asset_returns:
        st.warning("No price data available.")
        return

    # ── Scenario selector ────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    with col_left:
        scenario_name = st.selectbox("Historical Scenario", list(HISTORICAL_SCENARIOS.keys()))

    with col_right:
        st.markdown("**Custom Scenario**")
        custom_shock = st.slider("Equity shock (%)", -60, +30, -20)
        custom_vol = st.slider("Vol multiplier", 1.0, 5.0, 2.0, step=0.25)
        run_custom = st.button("Run Custom Scenario")

    # ── Individual scenario result ────────────────────────────────────────────
    with st.spinner("Running stress scenario..."):
        if run_custom:
            result = run_custom_scenario(weights, asset_returns,
                                         equity_shock_pct=float(custom_shock),
                                         vol_multiplier=float(custom_vol))
        else:
            result = run_stress_scenario(weights, asset_returns, scenario_name)

    st.divider()
    scen_display = result.scenario_name
    st.markdown(f"### {scen_display}")
    st.caption(result.description)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Equity Shock", f"{result.equity_shock_pct:+.1f}%")
    s2.metric("Portfolio P&L", f"{result.portfolio_pnl_pct:+.2f}%",
              delta=f"{'Loss' if result.portfolio_pnl_pct < 0 else 'Gain'}",
              delta_color="inverse")
    s3.metric("VaR 95% (Normal)", f"{result.var_95_unstressed_pct:.3f}%")
    s4.metric("VaR 95% (Stressed)", f"{result.var_95_stressed_pct:.3f}%",
              delta=f"{result.var_95_stressed_pct - result.var_95_unstressed_pct:+.3f}%",
              delta_color="inverse")

    # Per-asset P&L bar
    st.subheader("Per-Asset P&L Impact")
    pnl = result.per_asset_pnl
    fig_pnl = go.Figure(go.Bar(
        x=list(pnl.keys()), y=list(pnl.values()),
        marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in pnl.values()],
        text=[f"{v:+.2f}%" for v in pnl.values()], textposition="outside",
    ))
    fig_pnl.update_layout(template="plotly_dark", height=320,
                           yaxis_title="P&L (%)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Full scenario matrix heatmap ──────────────────────────────────────────
    st.divider()
    st.subheader("All Scenarios — Portfolio P&L Heatmap")
    st.caption("Equal-weight portfolio loss/gain across all 10 historical scenarios.")

    with st.spinner("Running all scenarios..."):
        matrix = load_stress_matrix(tuple(tickers), period)

    if not matrix.empty:
        # Show heatmap: rows = scenarios, columns = tickers + Portfolio
        display_cols = [c for c in matrix.columns if c != "Portfolio P&L %"] + ["Portfolio P&L %"]
        plot_df = matrix[display_cols]

        fig_heat = go.Figure(go.Heatmap(
            z=plot_df.values,
            x=plot_df.columns.tolist(),
            y=plot_df.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:+.1f}%" for v in row] for row in plot_df.values],
            texttemplate="%{text}",
            colorbar=dict(title="P&L %"),
        ))
        fig_heat.update_layout(
            template="plotly_dark", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Asset / Portfolio",
            yaxis_title="Stress Scenario",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── 3D scenario surface ───────────────────────────────────────────────
        st.subheader("Scenario × Asset P&L Surface (3D)")
        st.caption("Drag to rotate — each ridge shows how one scenario hits all assets simultaneously.")
        asset_cols = [c for c in plot_df.columns if c != "Portfolio P&L %"]
        if asset_cols:
            z_3d = plot_df[asset_cols].values
            x_3d = np.arange(len(asset_cols))
            y_3d = np.arange(len(plot_df))

            fig_3d = go.Figure(data=[go.Surface(
                x=x_3d, y=y_3d, z=z_3d,
                colorscale="RdYlGn",
                cmid=0,
                colorbar=dict(title="P&L %"),
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )])
            tick_y = list(range(len(plot_df)))
            fig_3d.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis=dict(title="Asset", tickvals=list(x_3d), ticktext=asset_cols),
                    yaxis=dict(title="Scenario", tickvals=tick_y,
                               ticktext=[s[:20] for s in plot_df.index.tolist()]),
                    zaxis_title="P&L (%)",
                    camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                ),
                height=600,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_3d, use_container_width=True)


def page_quant_lab():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("Quant Lab")
    with _tsel:
        page_ticker = _ticker_selector("quantlab")
    st.caption(
        "Statistical diagnostics used by quantitative analysts: return distribution analysis, "
        "Hurst exponent, GARCH volatility, hypothesis tests, and options strategy payoffs."
    )

    df = load_ohlcv(page_ticker, period)
    if df.empty:
        st.warning("No data available. Check ticker or API.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Statistical Tests",
        "📈 GARCH & Autocorrelation",
        "📊 Return Distribution",
        "🎯 Options Strategies",
    ])

    with tab1:
        st.subheader("Hurst Exponent (R/S Analysis)")
        with st.spinner("Computing Hurst exponent..."):
            qs = load_quant_stats(page_ticker, period)

        h = qs.get("hurst", 0.5)
        h_color = "#1D9E75" if h > 0.55 else "#D85A30" if h < 0.45 else "#BA7517"
        st.markdown(f"## <span style='color:{h_color}'>{hurst_interpretation(h)}</span>",
                    unsafe_allow_html=True)

        st.divider()
        st.subheader("Statistical Tests Summary")

        adf = qs.get("adf", {})
        lb = qs.get("lb", {})
        jb = qs.get("jb", {})

        tc1, tc2, tc3 = st.columns(3)

        with tc1:
            st.markdown("**ADF Stationarity Test**")
            if adf:
                status = "✅ Stationary" if adf.get("is_stationary") else "❌ Non-Stationary"
                st.metric("ADF t-stat", f"{adf.get('test_stat', 0):.4f}")
                st.metric("p-value", f"{adf.get('p_value', 1):.4f}")
                st.markdown(f"**Result:** {status}")
                st.caption(adf.get("interpretation", ""))

        with tc2:
            st.markdown("**Ljung-Box Autocorrelation**")
            if lb:
                ac_status = "⚠️ Serial Correlation" if lb.get("has_autocorrelation") else "✅ No Autocorrelation"
                st.metric("Q Statistic", f"{lb.get('Q_stat', 0):.4f}")
                st.metric("p-value", f"{lb.get('p_value', 1):.4f}")
                st.markdown(f"**Result:** {ac_status}")
                # ACF bar chart
                acf_vals = lb.get("acf_values", [])
                if acf_vals:
                    fig_acf = go.Figure(go.Bar(
                        x=list(range(1, len(acf_vals) + 1)), y=acf_vals,
                        marker_color=["#D85A30" if abs(v) > 0.1 else "#888" for v in acf_vals],
                    ))
                    fig_acf.add_hline(y=0.1, line_dash="dot", line_color="gray")
                    fig_acf.add_hline(y=-0.1, line_dash="dot", line_color="gray")
                    fig_acf.update_layout(template="plotly_dark", height=200,
                                          xaxis_title="Lag", yaxis_title="ACF",
                                          margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_acf, use_container_width=True)

        with tc3:
            st.markdown("**Jarque-Bera Normality**")
            if jb:
                norm_status = "✅ Normal" if jb.get("is_normal") else "❌ Non-Normal (fat tails)"
                st.metric("JB Statistic", f"{jb.get('JB_stat', 0):.2f}")
                st.metric("Skewness", f"{jb.get('skewness', 0):.4f}")
                st.metric("Excess Kurtosis", f"{jb.get('excess_kurtosis', 0):.4f}")
                st.markdown(f"**Result:** {norm_status}")
                st.caption("Equities almost universally fail normality — fat tails dominate.")

    with tab2:
        st.subheader("GARCH(1,1) Conditional Volatility")
        st.caption("Time-varying risk estimate: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1} (annualized %)")

        with st.spinner("Computing GARCH volatility..."):
            garch = load_garch_vol(page_ticker, period)

        if not garch.empty:
            hist_vol = (df["Returns"].dropna().rolling(21).std() * np.sqrt(252) * 100).dropna()
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=hist_vol.index, y=hist_vol.values, name="21-day Rolling Vol",
                                       line=dict(color="#888", width=1, dash="dot")))
            fig_g.add_trace(go.Scatter(x=garch.index, y=garch.values, name="GARCH(1,1) Vol",
                                       line=dict(color="#1D9E75", width=2)))
            fig_g.update_layout(template="plotly_dark", yaxis_title="Annualized Vol (%)",
                                 height=380, legend=dict(orientation="h", y=-0.15),
                                 margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

            current_garch = float(garch.iloc[-1])
            hist_ann_vol = float(df["Returns"].std() * np.sqrt(252) * 100)
            gc1, gc2 = st.columns(2)
            gc1.metric("Current GARCH Vol (ann.)", f"{current_garch:.2f}%")
            gc2.metric("Historical Vol (ann.)", f"{hist_ann_vol:.2f}%",
                       delta=f"{current_garch - hist_ann_vol:+.2f}%", delta_color="inverse")

        st.divider()
        st.subheader("Rolling Autocorrelation (lag-1)")
        st.caption("Periods where autocorrelation deviates from zero may signal exploitable serial dependence.")

        with st.spinner("Computing rolling autocorrelation..."):
            ac = load_rolling_autocorr(page_ticker, period)

        if not ac.empty:
            fig_ac = go.Figure()
            fig_ac.add_trace(go.Scatter(x=ac.index, y=ac.values, name="Lag-1 Autocorrelation",
                                        line=dict(color="#534AB7", width=1.5)))
            fig_ac.add_hline(y=0, line_color="white", line_width=0.8)
            fig_ac.add_hline(y=0.15, line_dash="dot", line_color="#D85A30", opacity=0.6)
            fig_ac.add_hline(y=-0.15, line_dash="dot", line_color="#D85A30", opacity=0.6)
            fig_ac.update_layout(template="plotly_dark", yaxis_title="Autocorrelation",
                                  height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_ac, use_container_width=True)

    with tab3:
        st.subheader("Return Distribution Analysis")
        qs_data = load_quant_stats(page_ticker, period)
        dist = qs_data.get("dist", {})

        if dist:
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Ann. Return", f"{dist.get('annual_return_pct', 0):.1f}%")
            dc2.metric("Ann. Volatility", f"{dist.get('annual_vol_pct', 0):.1f}%")
            dc3.metric("Skewness", f"{dist.get('skewness', 0):.3f}")
            dc4.metric("Excess Kurtosis", f"{dist.get('excess_kurtosis', 0):.3f}")

            st.divider()
            r = df["Returns"].dropna()

            col_hist, col_qq = st.columns(2)
            with col_hist:
                st.markdown("**Return Histogram vs Normal**")
                r_vals = r.values
                mu_r, sig_r = float(r_vals.mean()), float(r_vals.std())
                x_norm = np.linspace(r_vals.min(), r_vals.max(), 200)
                from scipy.stats import norm as _norm_dist
                y_norm = _norm_dist.pdf(x_norm, mu_r, sig_r)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=r_vals * 100, nbinsx=60, histnorm="probability density",
                    name="Returns", marker_color="#534AB7", opacity=0.7,
                ))
                fig_hist.add_trace(go.Scatter(
                    x=x_norm * 100, y=y_norm / 100, name="Normal Fit",
                    line=dict(color="#D85A30", width=2),
                ))
                fig_hist.update_layout(template="plotly_dark", height=340,
                                       xaxis_title="Daily Return (%)",
                                       yaxis_title="Density",
                                       margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_qq:
                st.markdown("**Q-Q Plot (vs Normal)**")
                from scipy.stats import probplot
                qq_x, qq_y = probplot(r_vals, dist="norm")[0]
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=qq_x, y=qq_y * 100, mode="markers",
                                            marker=dict(size=3, color="#534AB7"), name="Return quantiles"))
                line_x = [float(qq_x.min()), float(qq_x.max())]
                fig_qq.add_trace(go.Scatter(x=line_x, y=[v * 100 for v in line_x],
                                            line=dict(color="#D85A30", dash="dash"), name="Normal line"))
                fig_qq.update_layout(template="plotly_dark", height=340,
                                     xaxis_title="Theoretical Quantiles",
                                     yaxis_title="Sample Quantiles (%)",
                                     margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_qq, use_container_width=True)

            # Percentile table
            pct_labels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
            pct_keys = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
            pct_vals = [dist.get(k, 0) for k in pct_keys]
            fig_pct = go.Figure(go.Bar(
                x=pct_labels, y=pct_vals,
                marker_color=["#D85A30" if v < 0 else "#1D9E75" for v in pct_vals],
                text=[f"{v:.3f}%" for v in pct_vals], textposition="outside",
            ))
            fig_pct.update_layout(template="plotly_dark", height=280,
                                   yaxis_title="Daily Return (%)",
                                   margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pct, use_container_width=True)

    with tab4:
        st.subheader("Options Strategy Payoff Profiles")
        st.caption("P&L at expiry for standard multi-leg options strategies, priced from ATM Black-Scholes.")

        df_ohlcv = load_ohlcv(page_ticker, period)
        if df_ohlcv.empty:
            st.info("No price data.")
        else:
            spot = float(df_ohlcv["Close"].iloc[-1])
            hist_vol = float(df_ohlcv["Returns"].std() * np.sqrt(252))

            sc1, sc2, sc3 = st.columns(3)
            days_exp = sc1.slider("Days to expiry", 7, 180, 30)
            vol_override = sc2.slider("Implied Vol (%)", 10, 100,
                                       int(hist_vol * 100), step=5) / 100.0
            strategy_name = sc3.selectbox("Strategy", [
                "Long Straddle", "Short Straddle", "Long Strangle", "Short Strangle",
                "Bull Call Spread", "Bear Put Spread", "Long Butterfly",
                "Iron Condor", "Covered Call", "Protective Put",
            ])

            all_strategies = build_all_strategies(spot, vol_override, days_exp)
            profile = all_strategies.get(strategy_name)

            if profile:
                # Key metrics
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Net Premium", f"${profile.net_premium:.2f}/share",
                           help="Positive = debit (cost), Negative = credit (income)")
                max_p_str = f"${profile.max_profit:.2f}" if profile.max_profit is not None else "Unlimited"
                max_l_str = f"${profile.max_loss:.2f}" if profile.max_loss is not None else "Unlimited"
                pm2.metric("Max Profit", max_p_str)
                pm3.metric("Max Loss", max_l_str)

                st.caption(profile.description)
                if profile.breakeven_points:
                    be_str = " | ".join([f"${b:.2f}" for b in profile.breakeven_points])
                    st.caption(f"Breakeven: {be_str}")

                # P&L diagram
                spot_range = (spot * 0.70, spot * 1.30)
                spots, pnl = strategy_pnl_curve(profile, spot_range)

                fig_str = go.Figure()
                fig_str.add_trace(go.Scatter(
                    x=spots, y=pnl,
                    mode="lines",
                    line=dict(color="#534AB7", width=2.5),
                    name=strategy_name,
                    fill="tozeroy",
                    fillcolor="rgba(83,74,183,0.12)",
                ))
                fig_str.add_hline(y=0, line_color="white", line_width=0.8)
                fig_str.add_vline(x=spot, line_dash="dot", line_color="#BA7517",
                                  annotation_text=f"Spot ${spot:.0f}",
                                  annotation_position="top right")
                for be in profile.breakeven_points:
                    fig_str.add_vline(x=be, line_dash="dash", line_color="#D85A30", opacity=0.6)

                fig_str.update_layout(
                    template="plotly_dark", height=420,
                    xaxis_title="Spot Price at Expiry ($)",
                    yaxis_title="P&L per Share ($)",
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_str, use_container_width=True)

                # 3D: strategy P&L across spot × volatility surface
                st.subheader("P&L Surface vs Spot × Time (3D)")
                st.caption("How the payoff profile evolves as days to expiry decrease from selected DTE to 0.")
                dte_range = np.array([int(days_exp * f) for f in [1.0, 0.75, 0.5, 0.25, 0.0]])
                dte_range = np.unique(np.maximum(dte_range, 0))
                spot_3d = np.linspace(spot * 0.75, spot * 1.25, 80)
                pnl_3d = np.zeros((len(dte_range), len(spot_3d)))
                for i, dte in enumerate(dte_range):
                    all_s_i = build_all_strategies(spot, vol_override, max(dte, 0))
                    p_i = all_s_i.get(strategy_name)
                    if p_i:
                        _, pl_i = strategy_pnl_curve(p_i, (spot * 0.75, spot * 1.25), n_points=80)
                        pnl_3d[i, :] = pl_i

                fig_str3d = go.Figure(data=[go.Surface(
                    x=spot_3d, y=dte_range, z=pnl_3d,
                    colorscale="RdYlGn",
                    cmid=0,
                    colorbar=dict(title="P&L ($)"),
                    contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
                )])
                fig_str3d.update_layout(
                    template="plotly_dark",
                    scene=dict(
                        xaxis_title="Spot ($)", yaxis_title="Days to Expiry",
                        zaxis_title="P&L ($)",
                        camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                    ),
                    height=560, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_str3d, use_container_width=True)


def page_pairs_lab():
    st.title("Pairs Lab — Statistical Arbitrage")
    st.caption(
        "Engle-Granger cointegration tests, Ornstein-Uhlenbeck half-life estimation, "
        "and spread z-score signals. Classic stat-arb framework from Gatev, Goetzmann & Rouwenhorst (2006)."
    )

    data = load_multi(tuple(tickers), period)
    prices_df = pd.DataFrame({t: d["Close"] for t, d in data.items() if not d.empty})

    if len(prices_df.columns) < 2:
        st.warning("Need at least 2 tickers with data. Add more tickers to the watchlist.")
        return

    # ── Pair selector ────────────────────────────────────────────────────────
    pl1, pl2, pl3 = st.columns(3)
    valid_tickers = list(prices_df.columns)
    ticker_a = pl1.selectbox("Asset A", valid_tickers, index=0)
    ticker_b = pl2.selectbox("Asset B", valid_tickers,
                              index=min(1, len(valid_tickers) - 1))
    zscore_entry = pl3.slider("Z-score entry threshold", 1.0, 3.5, 2.0, step=0.25)

    if ticker_a == ticker_b:
        st.warning("Select two different assets.")
        return

    with st.spinner("Analyzing pair..."):
        pair = load_pair_analysis(ticker_a, ticker_b, period)

    if pair is None:
        st.info("Insufficient overlapping data for pair analysis (need 60+ aligned observations).")
        return

    # ── Pair metrics ─────────────────────────────────────────────────────────
    pa1, pa2, pa3, pa4, pa5 = st.columns(5)
    coint_color = "#1D9E75" if pair.is_cointegrated else "#D85A30"
    pa1.metric("ADF t-stat", f"{pair.adf_stat:.4f}",
               help="More negative = more stationary. 5% critical value ≈ −2.86")
    pa2.metric("Cointegrated?", "✅ Yes" if pair.is_cointegrated else "❌ No")
    pa3.metric("Hedge Ratio β", f"{pair.hedge_ratio:.4f}",
               help=f"Spread = {ticker_a} - {pair.hedge_ratio:.3f} × {ticker_b}")
    pa4.metric("OU Half-Life", f"{pair.ou_half_life_days:.1f} days",
               help="Expected time for spread to revert to half its deviation")
    pa5.metric("Current Z-Score", f"{pair.current_zscore:+.3f}")

    signal_colors = {"BUY_SPREAD": "#1D9E75", "SELL_SPREAD": "#D85A30", "HOLD": "#BA7517"}
    signal_color = signal_colors.get(pair.signal, "#888")
    st.markdown(
        f"<h3 style='color:{signal_color}'>Signal: {pair.signal} — {pair.signal_description}</h3>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Spread & z-score charts ───────────────────────────────────────────────
    spread_series, _ = compute_spread(prices_df[ticker_a], prices_df[ticker_b],
                                       hedge_ratio=pair.hedge_ratio)
    z_series = spread_zscore(spread_series)

    tab_spread, tab_price = st.tabs(["📈 Spread Z-Score", "📊 Price Comparison"])

    with tab_spread:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=z_series.index, y=z_series.values,
                                   name="Z-Score", line=dict(color="#534AB7", width=1.5)))
        fig_z.add_hline(y=zscore_entry, line_dash="dash", line_color="#D85A30",
                        annotation_text=f"+{zscore_entry:.1f} (Sell Spread)")
        fig_z.add_hline(y=-zscore_entry, line_dash="dash", line_color="#1D9E75",
                        annotation_text=f"−{zscore_entry:.1f} (Buy Spread)")
        fig_z.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
        fig_z.add_hline(y=-0.5, line_dash="dot", line_color="gray", opacity=0.5)
        fig_z.add_hline(y=0, line_color="white", line_width=0.6)
        fig_z.update_layout(template="plotly_dark", height=380,
                             yaxis_title="Z-Score", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_z, use_container_width=True)

        st.subheader("Raw Spread")
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=spread_series.index, y=spread_series.values,
                                    name="Spread", line=dict(color="#BA7517", width=1.5)))
        fig_sp.add_hline(y=float(pair.spread_mean), line_dash="dot", line_color="white",
                          annotation_text="Mean")
        fig_sp.update_layout(template="plotly_dark", height=280,
                              yaxis_title=f"Spread ({ticker_a} − β·{ticker_b})",
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sp, use_container_width=True)

    with tab_price:
        fig_px = go.Figure()
        for t in [ticker_a, ticker_b]:
            p = prices_df[t].dropna()
            normalized = p / p.iloc[0] * 100
            fig_px.add_trace(go.Scatter(x=normalized.index, y=normalized.values,
                                        name=t, line=dict(width=2)))
        fig_px.update_layout(template="plotly_dark", height=360,
                              yaxis_title="Indexed (Base = 100)",
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_px, use_container_width=True)

    # ── Pairs ranking table — opt-in (O(n²) cointegration tests) ─────────────
    st.divider()
    st.subheader("All Pairs Ranked by Cointegration Strength")
    if st.button("🔍 Rank All Pairs", key="rank_pairs_btn",
                 help="Runs Engle-Granger on every combination — cached 1 h after first run"):
        st.session_state["_run_pairs_ranking"] = True

    if not st.session_state.get("_run_pairs_ranking", False):
        st.info("Click **Rank All Pairs** to run cointegration tests on every ticker combination.")
    else:
        with st.spinner("Running cointegration tests on all pairs…"):
            ranking = load_pairs_ranking(tuple(valid_tickers), period)
        if not ranking.empty:
            display_rank = ranking.copy()
            display_rank["is_cointegrated"] = display_rank["is_cointegrated"].map(
                {True: "✅ Yes", False: "❌ No"}
            )
            display_rank["signal"] = display_rank["signal"].str.replace("_", " ")
            st.dataframe(display_rank, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to rank pairs — try a longer period or add more tickers.")


def page_backtesting():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("Backtesting")
    with _tsel:
        page_ticker = _ticker_selector("backtest")

    df = load_indicators(page_ticker, period)
    if df.empty:
        st.warning("No data available. Check ticker or API.")
        return

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])

    if strategy == "SMA Crossover":
        s1, s2 = st.columns(2)
        fast = s1.slider("Fast SMA window", 5, 50, 20)
        slow = s2.slider("Slow SMA window", 20, 200, 50)
        if fast >= slow:
            st.warning("Fast window must be smaller than the slow window.")
            return
        result = backtest_sma_crossover(df, fast, slow)
    else:
        s1, s2, s3 = st.columns(3)
        rsi_period = s1.slider("RSI period", 5, 30, 14)
        lower = s2.slider("Oversold threshold (buy <)", 10, 40, 30)
        upper = s3.slider("Overbought threshold (sell >)", 60, 90, 70)
        result = backtest_rsi_meanreversion(df, rsi_period, lower, upper)

    st.divider()

    m = result.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy Return", f"{m['annual_return_pct']:.1f}%")
    c2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    c3.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")
    c4.metric("Positive Days", f"{m['positive_days_pct']:.1f}%")
    c5.metric("Position Changes", result.trades)

    st.divider()
    st.subheader("Equity Curve vs Buy & Hold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve * 100,
        name=result.strategy_name, line=dict(color="#1D9E75", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result.benchmark_curve.index, y=result.benchmark_curve * 100,
        name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Growth of $100",
        height=380,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy Drawdown")
    cum = result.equity_curve
    drawdown = (cum - cum.cummax()) / cum.cummax() * 100
    fig_dd = go.Figure(go.Scatter(
        x=drawdown.index, y=drawdown, fill="tozeroy",
        line=dict(color="#D85A30"), name="Drawdown",
    ))
    fig_dd.update_layout(
        template="plotly_dark",
        yaxis_title="Drawdown (%)",
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    if strategy == "SMA Crossover":
        st.divider()
        st.subheader("Parameter Optimization Surface — Sharpe Ratio (3D)")
        st.caption(
            "Sharpe ratio across the SMA fast/slow parameter grid. "
            "Drag to rotate, scroll to zoom — find the ridge of best combinations."
        )
        with st.spinner("Running parameter sweep..."):
            sweep = load_sma_sweep(page_ticker, period)
        if not sweep.empty:
            pivot = sweep.pivot(index="slow", columns="fast", values="sharpe")
            fig3 = go.Figure(data=[go.Surface(
                x=pivot.columns, y=pivot.index, z=pivot.values,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
                },
            )])
            fig3.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Fast SMA",
                    yaxis_title="Slow SMA",
                    zaxis_title="Sharpe Ratio",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
                ),
                height=550,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)


def page_factor_lab():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("Factor Lab")
    with _tsel:
        page_ticker = _ticker_selector("factorlab")
    st.caption(
        "Fama-French 5-Factor regression (Mkt-RF, SMB, HML, RMW, CMA). Factor data from the "
        "Kenneth French Data Library (Dartmouth Tuck)."
    )

    factors = load_ff5_factors()
    if factors.empty:
        st.warning("Fama-French factor data unavailable — check internet connection.")
        return

    df = load_ohlcv(page_ticker, period)
    if df.empty:
        st.warning("No data available. Check ticker or API.")
        return

    exposure = load_factor_exposure(page_ticker, period)
    if exposure is None:
        st.info("Not enough overlapping history with factor data (need 60+ trading days).")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Alpha", f"{exposure.alpha_annual_pct:.2f}%")
    c2.metric("R²", f"{exposure.r_squared:.3f}")
    c3.metric("Observations", exposure.n_obs)

    st.divider()
    st.subheader("Factor Loadings (Betas)")
    betas = exposure.betas
    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in betas.values()]
    fig = go.Figure(go.Bar(
        x=list(betas.values()), y=list(betas.keys()),
        orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in betas.values()], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        xaxis_title="Beta",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Rolling Factor Exposure (3D)")
    st.caption(
        "126-day rolling betas, stepped every 5 days — drag to rotate, scroll to zoom."
    )
    rolling = load_rolling_exposures(page_ticker, period)
    if rolling.empty:
        st.info("Select a longer period (1y+) for the rolling exposure surface.")
        return

    x = np.arange(len(rolling))
    y = np.arange(len(FACTOR_COLUMNS))
    z = rolling[FACTOR_COLUMNS].to_numpy().T

    tick_idx = np.linspace(0, len(rolling) - 1, min(6, len(rolling))).astype(int)
    fig3d = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale="RdBu",
        cmid=0,
        colorbar=dict(title="Beta"),
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
        },
    )])
    fig3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="Date",
                tickvals=[int(i) for i in tick_idx],
                ticktext=[rolling.index[i].strftime("%Y-%m-%d") for i in tick_idx],
            ),
            yaxis=dict(title="Factor", tickvals=list(y), ticktext=FACTOR_COLUMNS),
            zaxis_title="Beta",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
        ),
        height=580,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)


def page_alpha():
    _tc, _tsel = st.columns([3, 2])
    _tc.title("⚡ Alpha Score")
    with _tsel:
        page_ticker = _ticker_selector("alpha")

    fund = load_fundamentals(page_ticker)
    val = load_valuation(page_ticker)
    risk = load_risk(page_ticker, period)
    df = load_indicators(page_ticker, period)

    momentum_score = float(df["Momentum_Score"].iloc[-1]) if not df.empty else 50.0
    trend_score = min(momentum_score, 100.0)
    news_score = 55.0  # placeholder; real value from news_ai.sentiment

    result = compute_alpha_score(
        ticker=page_ticker,
        fundamental_score=fund.fundamental_score if fund else 50.0,
        market_trend_score=trend_score,
        news_sentiment_score=news_score,
        valuation_score=val.valuation_score if val else 50.0,
        options_score=50.0,
        risk_score=risk.get("risk_score", 50.0),
    )

    # Verdict banner
    VERDICT_COLORS = {"BUY": "🟢", "WAIT": "🟡", "SELL": "🔴"}
    st.markdown(
        f"## {VERDICT_COLORS.get(result.verdict, '⚪')} {result.verdict} — "
        f"{page_ticker} · Score: **{result.total_score:.0f} / 100** · "
        f"Confidence: {result.confidence:.0%}"
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Score Breakdown")
        comp = result.component_scores
        weights_display = {
            "Fundamentals (35%)": comp["Fundamentals"],
            "Market Trend (20%)": comp["Market Trend"],
            "News Sentiment (15%)": comp["News Sentiment"],
            "Valuation (10%)": comp["Valuation"],
            "Options Activity (10%)": comp["Options Activity"],
            "Risk Profile (10%)": comp["Risk Profile"],
        }
        colors = ["#1D9E75" if v >= 65 else "#BA7517" if v >= 45 else "#D85A30"
                  for v in comp.values()]
        fig = go.Figure(go.Bar(
            x=list(weights_display.values()),
            y=list(weights_display.keys()),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in weights_display.values()],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(range=[0, 110]),
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Reasoning")
        for bullet in result.reasoning:
            st.markdown(bullet)

    with col2:
        st.subheader("Weighted Contributions")
        wc = result.weighted_contributions
        fig2 = go.Figure(go.Pie(
            labels=list(wc.keys()),
            values=list(wc.values()),
            hole=0.55,
            textinfo="label+value",
            marker=dict(colors=["#1D9E75", "#534AB7", "#1D9E75", "#BA7517", "#534AB7", "#D85A30"]),
        ))
        fig2.update_layout(
            template="plotly_dark",
            showlegend=False,
            annotations=[dict(text=f"{result.total_score:.0f}", x=0.5, y=0.5,
                              font_size=26, showarrow=False, font_color="white")],
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Invalidation Condition")
        st.info(result.invalidation)


def page_demo_portfolio():
    """Paper-trading portfolio: monitor P&L, execute simulated trades, view history."""
    st.title("Demo Portfolio")
    st.caption(
        "Paper-trading account — $100,000 initial capital, no real money involved. "
        "Prices are fetched live; P&L updates on every page refresh."
    )

    demo = DemoPortfolio()
    snap_prices = {}

    # ── Fetch live prices for held positions ─────────────────────────────────
    pos_tickers = list(demo.positions.keys())
    if pos_tickers:
        for t in pos_tickers:
            try:
                snap_prices[t] = float(fetch_quote(t)["price"])
            except Exception:
                snap_prices[t] = float(demo.positions[t]["avg_cost"])

    snap = demo.get_snapshot(snap_prices)

    # ── Top metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    ret_color = "#1D9E75" if snap["total_return"] >= 0 else "#D85A30"
    m1.metric(
        "Portfolio Value",
        f"${snap['total_value']:,.2f}",
        delta=f"${snap['total_return']:+,.2f}",
        delta_color="normal",
    )
    m2.metric(
        "Total Return",
        f"{snap['total_return_pct']:+.2f}%",
        help=f"vs ${DEMO_INITIAL_CASH:,.0f} initial capital",
    )
    m3.metric("Cash", f"${snap['cash']:,.2f}", help=f"{snap['cash_pct']:.1f}% of portfolio")
    m4.metric("Unrealized P&L", f"${snap['unrealized_pnl']:+,.2f}",
              delta=f"{snap['unrealized_pnl_pct']:+.2f}%")
    m5.metric("Open Positions", snap["n_positions"])

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dp_tab1, dp_tab2, dp_tab3 = st.tabs(["📋 Positions", "📝 Trade", "📜 History"])

    # ── TAB 1: Positions ──────────────────────────────────────────────────────
    with dp_tab1:
        if not snap["positions"]:
            st.info(
                "No open positions. Switch to the **Trade** tab to buy your first asset."
            )
        else:
            # Positions table
            pos_rows = snap["positions"]
            pos_df = pd.DataFrame(pos_rows)

            def _color_pnl(val):
                """Color P&L cells in the dataframe display."""
                try:
                    v = float(val)
                    color = "#1D9E75" if v >= 0 else "#D85A30"
                    return f"color: {color}"
                except (ValueError, TypeError):
                    return ""

            display_df = pos_df[[
                "ticker", "shares", "avg_cost", "current_price",
                "cost_basis", "market_value", "unrealized_pnl",
                "unrealized_pnl_pct", "weight_pct",
            ]].copy()
            display_df.columns = [
                "Ticker", "Shares", "Avg Cost", "Price",
                "Cost Basis", "Market Value", "Unrealized P&L",
                "P&L %", "Weight %",
            ]

            styled = (
                display_df.style
                .format({
                    "Avg Cost": "${:.2f}", "Price": "${:.2f}",
                    "Cost Basis": "${:,.2f}", "Market Value": "${:,.2f}",
                    "Unrealized P&L": "${:+,.2f}", "P&L %": "{:+.2f}%",
                    "Weight %": "{:.1f}%",
                })
                .applymap(_color_pnl, subset=["Unrealized P&L", "P&L %"])
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.divider()

            # Allocation pie + P&L bar side by side
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Allocation by Market Value**")
                fig_alloc = go.Figure(go.Pie(
                    labels=[r["ticker"] for r in pos_rows],
                    values=[r["market_value"] for r in pos_rows],
                    hole=0.5,
                    textinfo="label+percent",
                    marker=dict(colors=["#1D9E75", "#534AB7", "#BA7517", "#D85A30",
                                        "#58A6FF", "#888", "#E6EDF3"]),
                ))
                fig_alloc.update_layout(
                    template="plotly_dark", height=300,
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            with pc2:
                st.markdown("**Unrealized P&L by Position**")
                pnl_vals = [r["unrealized_pnl"] for r in pos_rows]
                fig_pnl = go.Figure(go.Bar(
                    x=[r["ticker"] for r in pos_rows],
                    y=pnl_vals,
                    marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in pnl_vals],
                    text=[f"${v:+,.2f}" for v in pnl_vals],
                    textposition="outside",
                ))
                fig_pnl.update_layout(
                    template="plotly_dark", height=300,
                    yaxis_title="P&L ($)", margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

    # ── TAB 2: Trade ──────────────────────────────────────────────────────────
    with dp_tab2:
        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown("**Buy**")
            with st.form("buy_form", clear_on_submit=True):
                buy_ticker = st.text_input(
                    "Ticker", placeholder="e.g. AAPL"
                ).upper().strip()
                buy_shares = st.number_input("Shares", min_value=0.001, step=1.0, value=1.0)
                buy_price  = st.number_input(
                    "Price per share ($)", min_value=0.01, step=0.01, value=100.0
                )
                buy_cost_preview = buy_shares * buy_price
                st.caption(
                    f"Estimated total: **${buy_cost_preview:,.2f}** · "
                    f"Cash available: ${snap['cash']:,.2f}"
                )
                submitted_buy = st.form_submit_button("Execute Buy", type="primary")

            if submitted_buy:
                if buy_ticker:
                    ok, msg = demo.buy(buy_ticker, buy_shares, buy_price)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Enter a ticker symbol.")

        with tc2:
            st.markdown("**Sell**")
            if not pos_tickers:
                st.info("No positions to sell.")
            else:
                with st.form("sell_form", clear_on_submit=True):
                    sell_ticker = st.selectbox("Ticker", pos_tickers)
                    current_held = float(demo.positions.get(sell_ticker, {}).get("shares", 0))
                    st.caption(f"Currently holding: {current_held:g} shares")
                    sell_shares = st.number_input(
                        "Shares to sell", min_value=0.001, max_value=float(current_held),
                        step=1.0, value=min(1.0, current_held),
                    )
                    sell_price = st.number_input(
                        "Price per share ($)", min_value=0.01, step=0.01,
                        value=float(snap_prices.get(sell_ticker, 100.0)),
                    )
                    proceeds_preview = sell_shares * sell_price
                    st.caption(f"Estimated proceeds: **${proceeds_preview:,.2f}**")
                    submitted_sell = st.form_submit_button("Execute Sell", type="primary")

                if submitted_sell:
                    ok, msg = demo.sell(sell_ticker, sell_shares, sell_price)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.divider()
        st.markdown("**Reset Portfolio**")
        st.caption(
            "Wipes all positions, trades, and history — restores $100,000 cash. "
            "This cannot be undone."
        )
        if st.button("🗑 Reset to $100,000 Cash", key="demo_reset_btn"):
            if "confirm_reset" not in st.session_state:
                st.session_state.confirm_reset = True
                st.warning("Click **Confirm Reset** to proceed, or refresh the page to cancel.")
            if st.button("⚠ Confirm Reset", key="demo_confirm_reset_btn", type="primary"):
                demo.reset()
                st.success("Portfolio reset to $100,000 cash.")
                del st.session_state["confirm_reset"]

    # ── TAB 3: History ────────────────────────────────────────────────────────
    with dp_tab3:
        trades = demo.get_trades()
        if not trades:
            st.info("No trades yet. Use the Trade tab to execute your first order.")
        else:
            trades_df = pd.DataFrame(trades)
            display_trades = trades_df[[
                "timestamp", "action", "ticker", "shares", "price", "total", "cash_after",
            ]].copy()
            display_trades.columns = [
                "Timestamp", "Action", "Ticker", "Shares", "Price ($)", "Total ($)", "Cash After ($)"
            ]

            def _color_action(val):
                return "color: #1D9E75" if val == "BUY" else "color: #D85A30"

            styled_trades = (
                display_trades.style
                .format({
                    "Price ($)": "${:.2f}",
                    "Total ($)": "${:,.2f}",
                    "Cash After ($)": "${:,.2f}",
                })
                .applymap(_color_action, subset=["Action"])
            )
            st.dataframe(styled_trades, use_container_width=True, hide_index=True)

            st.divider()
            csv_data = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Trade History (CSV)",
                data=csv_data,
                file_name="alphaforge_demo_trades.csv",
                mime="text/csv",
            )

            th1, th2 = st.columns(2)
            th1.metric("Total Trades", len(trades))
            buys  = sum(1 for t in trades if t["action"] == "BUY")
            sells = len(trades) - buys
            th2.metric("Buys / Sells", f"{buys} / {sells}")


# ── Sports Betting ───────────────────────────────────────────────────────────

def _fmt_time(iso_str: str) -> str:
    """Format ISO datetime to readable UTC label."""
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %H:%M")
    except Exception:
        return iso_str[:16]


def _events_summary(events: list[dict]) -> pd.DataFrame:
    """Build the overview table: one row per upcoming match with best odds."""
    rows = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        best: dict[str, float] = {}
        n_bk = len(ev.get("bookmakers", []))
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for o in mkt.get("outcomes", []):
                    nm = o["name"]
                    pr = float(o["price"])
                    if nm not in best or pr > best[nm]:
                        best[nm] = pr

        h_odds = best.get(home, 0.0)
        a_odds = best.get(away, 0.0)
        draw_key = next((k for k in best if k not in (home, away)), None)
        d_odds = best.get(draw_key, 0.0) if draw_key else 0.0

        valid_odds = [o for o in (h_odds, d_odds, a_odds) if o > 1.0]
        margin = overround(valid_odds) * 100.0 if valid_odds else float("nan")

        rows.append({
            "Match":      f"{home} vs {away}",
            "Start UTC":  _fmt_time(ev.get("commence_time", "")),
            "Home":       round(h_odds, 2) if h_odds else "-",
            "Draw":       round(d_odds, 2) if d_odds else "-",
            "Away":       round(a_odds, 2) if a_odds else "-",
            "Margin %":   round(margin, 1) if not math.isnan(margin) else "-",
            "Books":      n_bk,
        })
    return pd.DataFrame(rows)


def page_sports_betting() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    h_col, api_col = st.columns([3, 1])
    with h_col:
        st.title("⚽ Sports Betting — Value Analysis")
        st.caption(
            "Identify mispriced odds using Expected Value, Kelly Criterion, and the "
            "independent Poisson model. This is a mathematical tool, not tipster advice."
        )

    # ── API key guard ─────────────────────────────────────────────────────────
    from src.sports import odds_client as _oc
    if not _oc.ODDS_API_KEY:
        with api_col:
            st.metric("API Status", "⚠ No Key")
        st.error("**ODDS_API_KEY not set.** Add your free key to `.env` to activate live odds.")
        st.markdown("""
**Setup (2 minutes):**
1. Sign up at [the-odds-api.com](https://the-odds-api.com) — free tier: 500 requests/month
2. Copy your API key
3. Add to your `.env` file:
```
ODDS_API_KEY=your_key_here
```
4. Restart the dashboard

**Why not 1win?** 1win and most bookmakers do not publish public APIs. The Odds API
aggregates odds from **80+ bookmakers worldwide** including Pinnacle (lowest margin in
the market), Bet365, William Hill, and Unibet — giving us the full market picture needed
for value identification.
        """)
        st.divider()
        st.subheader("📐 Offline Calculator")
        st.caption("Full mathematical toolkit available without an API key.")

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.subheader("EV Calculator")
            ev_prob   = st.number_input("Your probability (%)", 0.1, 99.9, 50.0, 0.1, key="ev_p") / 100
            ev_odds   = st.number_input("Decimal odds",          1.01, 100.0, 2.0, 0.01, key="ev_o")
            ev_result = expected_value(ev_prob, ev_odds)
            color     = "#1D9E75" if ev_result > 0 else "#D85A30"
            st.markdown(
                f'<div style="font-size:2rem;color:{color};font-weight:700;">'
                f'EV = {ev_result:+.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.caption("Positive = edge over bookmaker")

        with oc2:
            st.subheader("Kelly Criterion")
            k_prob  = st.number_input("Your probability (%)", 0.1, 99.9, 55.0, 0.1, key="kelly_p") / 100
            k_odds  = st.number_input("Decimal odds",          1.01, 100.0, 2.1, 0.01, key="kelly_o")
            k_bank  = st.number_input("Bankroll ($)",          1.0, 1e7, 1000.0, 10.0, key="kelly_b")
            k_mult  = st.select_slider("Kelly fraction", [0.10, 0.25, 0.50, 1.0],
                                        value=0.25, format_func=lambda x: f"{int(x*100)}%",
                                        key="kelly_m")
            kf      = kelly_fraction(k_prob, k_odds)
            stake   = k_bank * kf * k_mult
            kcolor  = "#1D9E75" if kf > 0 else "#D85A30"
            st.markdown(
                f'<div style="color:{kcolor};font-size:1.4rem;font-weight:700;">'
                f'Full Kelly: {kf*100:.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.metric("Recommended Stake", f"${stake:,.2f}",
                      delta=f"({k_mult*100:.0f}% × {kf*100:.2f}%)")

        with oc3:
            st.subheader("Arbitrage Check")
            st.caption("Enter best odds available across bookmakers")
            arb_h = st.number_input("Home odds",  1.01, 100.0, 2.10, 0.01, key="arb_h")
            arb_d = st.number_input("Draw odds",  1.01, 100.0, 3.50, 0.01, key="arb_d")
            arb_a = st.number_input("Away odds",  1.01, 100.0, 3.80, 0.01, key="arb_a")
            arb   = arbitrage_analysis({"Home": arb_h, "Draw": arb_d, "Away": arb_a})
            if arb["is_arbitrage"]:
                st.success(f"✅ ARB FOUND — {arb['profit_pct']:.3f}% profit locked")
                for outcome, pct in arb["stakes_pct"].items():
                    st.write(f"{outcome}: {pct:.1f}% of bankroll")
            else:
                deficit = arb["margin_total_pct"] - 100.0
                st.warning(f"No arb — margin total {arb['margin_total_pct']:.2f}% (+{deficit:.2f}% over)")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 1, 1, 1, 1])
    with ctrl1:
        sport_label = st.selectbox("League / Sport", list(SPORTS.keys()),
                                   key="sb_sport")
    sport_key = SPORTS[sport_label]

    with ctrl2:
        bankroll = st.number_input("Bankroll ($)", min_value=10.0, max_value=1e7,
                                   value=1000.0, step=100.0, key="sb_bankroll")
    with ctrl3:
        kelly_mult = st.select_slider(
            "Kelly fraction",
            options=[0.10, 0.25, 0.50, 1.0],
            value=0.25,
            format_func=lambda x: f"{int(x * 100)}%",
            key="sb_kmult",
        )
    with ctrl4:
        regions = st.selectbox("Regions", ["eu,uk,us", "eu,uk", "us", "eu"],
                               key="sb_regions")
    with ctrl5:
        refresh_btn = st.button("🔄 Refresh Odds", key="sb_refresh",
                                help="Consumes 1 API request")

    if refresh_btn:
        load_odds.clear()

    # ── Fetch odds ────────────────────────────────────────────────────────────
    with st.spinner("Fetching live odds…"):
        events = load_odds(sport_key)

    # Show API quota
    rem  = requests_remaining()
    used = requests_used()
    with api_col:
        if rem is not None:
            quota_color = "#1D9E75" if rem > 100 else "#BA7517" if rem > 50 else "#D85A30"
            st.markdown(
                f'<div style="text-align:right">'
                f'<span style="font-size:0.7rem;color:#7D8590;">API quota</span><br>'
                f'<span style="color:{quota_color};font-weight:700;">{rem}</span>'
                f'<span style="color:#7D8590;font-size:0.75rem;"> remaining</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("API Status", "✅ Active")

    if not events:
        st.info("No upcoming events found for this sport right now. Try a different league or check back later.")
        return

    # ── Events overview table ─────────────────────────────────────────────────
    st.subheader(f"Upcoming Matches — {sport_label}")
    overview_df = _events_summary(events)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Match selector ────────────────────────────────────────────────────────
    match_labels = [f"{ev['home_team']} vs {ev['away_team']}" for ev in events]
    sel_match = st.selectbox("Select match to analyse", match_labels, key="sb_match")
    sel_event = events[match_labels.index(sel_match)]

    home_team = sel_event["home_team"]
    away_team = sel_event["away_team"]

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    sb_t1, sb_t2, sb_t3, sb_t4, sb_t5 = st.tabs([
        "📊 Probability Analysis",
        "💡 Value Bets (EV)",
        "💰 Kelly Bankroll Plan",
        "⚡ Arbitrage Scanner",
        "🎯 Poisson Model",
    ])

    # ── TAB 5 first: Poisson model (determines model_probs for other tabs) ───
    # Runs first so model_probs is available for Tabs 1-4.

    with sb_t5:
        st.subheader("🎯 Poisson Match Model — calibrated from historical data")

        # ── Try to load historical season stats ───────────────────────────────
        comp_id = COMPETITION_MAP.get(sport_key)
        season_stats: SeasonStats | None = None
        from src.sports.historical import FOOTBALL_DATA_KEY as _fdk
        if comp_id and _fdk:
            with st.spinner("Loading historical match data…"):
                season_stats = load_season_stats(comp_id)

        # ── Auto-estimate λ from historical data ──────────────────────────────
        avg_h, avg_a = LEAGUE_AVG_XG.get(sport_key, (1.3, 1.1))
        data_source = "league averages (no historical key)"

        if season_stats:
            auto_lam_h, auto_lam_a = estimate_lambdas(home_team, away_team, season_stats)
            avg_h, avg_a = auto_lam_h, auto_lam_a
            data_source = (
                f"Dixon-Coles model · {season_stats.total_matches} matches · "
                f"league avg {season_stats.league_avg_home:.2f}H / {season_stats.league_avg_away:.2f}A"
            )
        elif not comp_id:
            data_source = "league averages (this league not in football-data.org)"
        elif not _fdk:
            data_source = "league averages (add FOOTBALL_DATA_KEY to .env for auto-estimation)"

        st.caption(f"λ source: **{data_source}**")

        # ── Team stats cards (if historical data available) ───────────────────
        if season_stats:
            h_str = season_stats.team_strengths.get(
                next((k for k in season_stats.team_strengths
                      if k == home_team or home_team in k), ""), None
            )
            a_str = season_stats.team_strengths.get(
                next((k for k in season_stats.team_strengths
                      if k == away_team or away_team in k), ""), None
            )
            # Fuzzy fallback
            from src.sports.historical import _find_team, _normalize
            h_str = h_str or _find_team(home_team, season_stats.team_strengths)
            a_str = a_str or _find_team(away_team, season_stats.team_strengths)

            if h_str or a_str:
                hc, ac = st.columns(2)
                for col, ts, label in [(hc, h_str, home_team), (ac, a_str, away_team)]:
                    with col:
                        st.markdown(
                            f'<div style="background:#161B22;border:1px solid #30363D;'
                            f'border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
                            f'<div style="font-weight:700;color:#E6EDF3;font-size:1rem;">{label}</div>',
                            unsafe_allow_html=True,
                        )
                        if ts:
                            # Form badges
                            form_html = " ".join(
                                f'<span style="background:{form_color(r)};color:#fff;'
                                f'border-radius:3px;padding:1px 5px;font-weight:700;">{r}</span>'
                                for r in ts.recent_form
                            )
                            st.markdown(
                                f'<div style="margin:4px 0 8px;">{form_html}</div>'
                                f'<div style="color:#7D8590;font-size:0.82rem;">'
                                f'Home: {ts.avg_scored_home:.2f} scored / {ts.avg_conceded_home:.2f} conceded<br>'
                                f'Away: {ts.avg_scored_away:.2f} scored / {ts.avg_conceded_away:.2f} conceded<br>'
                                f'Attack ×{ts.attack_home:.2f}H ×{ts.attack_away:.2f}A &nbsp;'
                                f'Defense ×{ts.defense_home:.2f}H ×{ts.defense_away:.2f}A'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div style="color:#7D8590;font-size:0.82rem;">'
                                f'No historical data found (name mismatch between APIs)</div></div>',
                                unsafe_allow_html=True,
                            )

        # ── λ sliders (auto-filled from historical, still user-adjustable) ────
        st.caption("Fine-tune λ manually if you have additional information about team news, injuries, etc.")
        pm_c1, pm_c2 = st.columns(2)
        with pm_c1:
            lam_h = st.slider(
                f"λ {home_team} (expected goals)",
                0.1, 4.0, float(round(avg_h, 1)), 0.1,
                key="sb_lam_h",
                help="Expected goals scored by the home team this match",
            )
        with pm_c2:
            lam_a = st.slider(
                f"λ {away_team} (expected goals)",
                0.1, 4.0, float(round(avg_a, 1)), 0.1,
                key="sb_lam_a",
                help="Expected goals scored by the away team this match",
            )

        pm_probs = poisson_match_probs(lam_h, lam_a)
        model_probs = {
            "home": pm_probs["home"],
            "draw": pm_probs["draw"],
            "away": pm_probs["away"],
        }

        # ── 1X2 + market summary ──────────────────────────────────────────────
        pm_m1, pm_m2, pm_m3, pm_m4, pm_m5 = st.columns(5)
        pm_m1.metric(f"Home ({home_team})", f"{pm_probs['home']*100:.1f}%",
                     help="Model probability of home win")
        pm_m2.metric("Draw",               f"{pm_probs['draw']*100:.1f}%")
        pm_m3.metric(f"Away ({away_team})", f"{pm_probs['away']*100:.1f}%")
        pm_m4.metric("BTTS",               f"{pm_probs['btts']*100:.1f}%",
                     help="Both Teams To Score probability")
        pm_m5.metric("O2.5 Goals",         f"{pm_probs['over_2_5']*100:.1f}%")

        # ── Score probability heatmap ─────────────────────────────────────────
        mat = pm_probs["score_matrix"]
        max_disp = 6
        z_data = mat[:max_disp + 1, :max_disp + 1] * 100.0
        labels = [[f"{v:.1f}%" for v in row] for row in z_data]

        fig_heat = go.Figure(go.Heatmap(
            z=z_data,
            x=[str(i) for i in range(max_disp + 1)],
            y=[str(i) for i in range(max_disp + 1)],
            text=labels,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#0D1117"],
                [0.3, "#1a3a5c"],
                [0.6, "#534AB7"],
                [1.0, "#58A6FF"],
            ],
            showscale=True,
            colorbar=dict(title="Prob %", tickfont=dict(color="#C9D1D9")),
        ))
        fig_heat.add_trace(go.Scatter(
            x=[str(i) for i in range(max_disp + 1)],
            y=[str(i) for i in range(max_disp + 1)],
            mode="markers",
            marker=dict(symbol="square", size=38, color="rgba(186,117,23,0.18)",
                        line=dict(color="#BA7517", width=2)),
            showlegend=False,
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            title=dict(
                text=f"{home_team} goals (rows) vs {away_team} goals (cols) — probability %",
                font=dict(color="#C9D1D9", size=13),
            ),
            xaxis=dict(title=f"{away_team} goals", color="#7D8590", tickcolor="#484F58"),
            yaxis=dict(title=f"{home_team} goals", color="#7D8590", tickcolor="#484F58"),
            height=420,
            margin=dict(t=50, b=40, l=50, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Over/Under probability table ──────────────────────────────────────
        ou_data = {
            "Market": ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5", "Over 4.5",
                       "Under 0.5", "Under 1.5", "Under 2.5", "Under 3.5", "Under 4.5"],
            "Probability %": [
                f"{pm_probs['over_0_5']*100:.1f}",
                f"{pm_probs['over_1_5']*100:.1f}",
                f"{pm_probs['over_2_5']*100:.1f}",
                f"{pm_probs['over_3_5']*100:.1f}",
                f"{pm_probs['over_4_5']*100:.1f}",
                f"{(1-pm_probs['over_0_5'])*100:.1f}",
                f"{(1-pm_probs['over_1_5'])*100:.1f}",
                f"{(1-pm_probs['over_2_5'])*100:.1f}",
                f"{(1-pm_probs['over_3_5'])*100:.1f}",
                f"{(1-pm_probs['over_4_5'])*100:.1f}",
            ],
            "Fair Odds": [
                f"{fair_decimal_odds(pm_probs['over_0_5']):.2f}",
                f"{fair_decimal_odds(pm_probs['over_1_5']):.2f}",
                f"{fair_decimal_odds(pm_probs['over_2_5']):.2f}",
                f"{fair_decimal_odds(pm_probs['over_3_5']):.2f}",
                f"{fair_decimal_odds(pm_probs['over_4_5']):.2f}",
                f"{fair_decimal_odds(1-pm_probs['over_0_5']):.2f}",
                f"{fair_decimal_odds(1-pm_probs['over_1_5']):.2f}",
                f"{fair_decimal_odds(1-pm_probs['over_2_5']):.2f}",
                f"{fair_decimal_odds(1-pm_probs['over_3_5']):.2f}",
                f"{fair_decimal_odds(1-pm_probs['over_4_5']):.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(ou_data), use_container_width=True, hide_index=True)

        # ── Full league team stats table ──────────────────────────────────────
        if season_stats and season_stats.team_strengths:
            with st.expander("📊 Full league team stats (all teams this season)"):
                ts_rows = []
                for t, ts in sorted(
                    season_stats.team_strengths.items(),
                    key=lambda x: -(x[1].avg_scored_home + x[1].avg_scored_away),
                ):
                    ts_rows.append({
                        "Team":         t,
                        "Form":         ts.recent_form,
                        "Atk H":        ts.attack_home,
                        "Def H":        ts.defense_home,
                        "Atk A":        ts.attack_away,
                        "Def A":        ts.defense_away,
                        "Avg Sc H":     ts.avg_scored_home,
                        "Avg Cc H":     ts.avg_conceded_home,
                        "Avg Sc A":     ts.avg_scored_away,
                        "Avg Cc A":     ts.avg_conceded_away,
                        "Gms H":        ts.matches_home,
                        "Gms A":        ts.matches_away,
                    })
                st.dataframe(pd.DataFrame(ts_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Atk > 1.0 = scores more than league avg · Def < 1.0 = concedes less than league avg"
                )

    # ── Shared analysis using model_probs from Poisson tab ────────────────────
    analyses = analyze_event(sel_event, model_probs=model_probs)

    # ── TAB 1: Probability Analysis ───────────────────────────────────────────
    with sb_t1:
        st.subheader(f"📊 Probability Analysis — {sel_match}")

        if not analyses:
            st.warning("No bookmaker odds available for this match.")
        else:
            # Aggregate median probabilities per outcome across bookmakers
            outcomes_order = ["home", "draw", "away"]
            outcome_labels = [home_team, "Draw", away_team]

            def _agg_for_outcome(key: str, field: str) -> float:
                vals = [getattr(a, field) for a in analyses if a.outcome == key]
                return float(np.median(vals)) if vals else 0.0

            implied_vals = [_agg_for_outcome(k, "implied_prob_pct") for k in outcomes_order]
            fair_vals    = [_agg_for_outcome(k, "fair_prob_pct")    for k in outcomes_order]
            model_vals   = [pm_probs.get(k, 0.0) * 100.0 for k in outcomes_order]

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                name="Book implied %",
                x=outcome_labels,
                y=implied_vals,
                marker_color="#D85A30",
                text=[f"{v:.1f}%" for v in implied_vals],
                textposition="auto",
            ))
            fig_prob.add_trace(go.Bar(
                name="No-vig fair %",
                x=outcome_labels,
                y=fair_vals,
                marker_color="#58A6FF",
                text=[f"{v:.1f}%" for v in fair_vals],
                textposition="auto",
            ))
            fig_prob.add_trace(go.Bar(
                name="Poisson model %",
                x=outcome_labels,
                y=model_vals,
                marker_color="#1D9E75",
                text=[f"{v:.1f}%" for v in model_vals],
                textposition="auto",
            ))
            fig_prob.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
                legend=dict(font=dict(color="#C9D1D9")),
                xaxis=dict(color="#7D8590"),
                yaxis=dict(title="Probability %", color="#7D8590",
                           gridcolor="#21262D", range=[0, 100]),
                height=360,
                margin=dict(t=20, b=30, l=50, r=20),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # ── Per-bookmaker table ───────────────────────────────────────────
            st.subheader("Odds by Bookmaker")
            # Pivot: rows = bookmakers, cols = outcomes
            pivot_rows: dict[str, dict] = {}
            for a in analyses:
                bk = a.bookmaker
                if bk not in pivot_rows:
                    pivot_rows[bk] = {"Bookmaker": bk}
                label = home_team if a.outcome == "home" else (
                    "Draw" if a.outcome == "draw" else away_team
                )
                pivot_rows[bk][f"{label} odds"]     = a.decimal_odds
                pivot_rows[bk][f"{label} implied %"] = a.implied_prob_pct
                pivot_rows[bk]["Margin %"]           = a.overround_pct

            if pivot_rows:
                pivot_df = pd.DataFrame(pivot_rows.values())
                # Sort by margin ascending (sharpest book first)
                if "Margin %" in pivot_df.columns:
                    pivot_df = pivot_df.sort_values("Margin %").reset_index(drop=True)
                st.dataframe(pivot_df, use_container_width=True, hide_index=True)

            # ── Market summary metrics ────────────────────────────────────────
            all_margins = [a.overround_pct for a in analyses if a.outcome == "home"]
            if all_margins:
                t1m1, t1m2, t1m3 = st.columns(3)
                t1m1.metric("Books covering", len(sel_event.get("bookmakers", [])))
                t1m2.metric("Avg book margin", f"{np.mean(all_margins):.2f}%")
                t1m3.metric("Best (lowest) margin",
                             f"{np.min(all_margins):.2f}%",
                             delta=f"-{np.mean(all_margins) - np.min(all_margins):.2f}% vs avg")

    # ── TAB 2: Value Bets ─────────────────────────────────────────────────────
    with sb_t2:
        st.subheader("💡 Value Bets — Expected Value Analysis")
        st.caption(
            "A value bet occurs when our estimated probability is **higher** than the "
            "bookmaker's implied probability. EV% > 0 = edge. EV% < 0 = bookmaker's "
            "margin eating into fair value."
        )

        if not analyses:
            st.warning("No odds data available.")
        else:
            # Full table: all bookmakers × all outcomes
            full_rows = []
            for a in analyses:
                ev_color = "#1D9E75" if a.is_value_bet else "#D85A30"
                out_label = home_team if a.outcome == "home" else (
                    "Draw" if a.outcome == "draw" else away_team
                )
                full_rows.append({
                    "Outcome":       out_label,
                    "Bookmaker":     a.bookmaker,
                    "Odds":          a.decimal_odds,
                    "Book Prob %":   a.implied_prob_pct,
                    "No-Vig %":      a.fair_prob_pct,
                    "Model Prob %":  a.model_prob_pct,
                    "EV %":          a.ev_pct,
                    "Value":         "✅ YES" if a.is_value_bet else "❌ no",
                })

            full_df = pd.DataFrame(full_rows).sort_values("EV %", ascending=False).reset_index(drop=True)

            def _ev_style(val):
                if isinstance(val, (int, float)):
                    return f"color: {'#1D9E75' if val > 0 else '#D85A30'}; font-weight: 600"
                return ""

            styled = full_df.style.applymap(_ev_style, subset=["EV %"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Value bets bar chart
            value_bets = [r for r in full_rows if "✅" in r["Value"]]
            if value_bets:
                st.success(f"**{len(value_bets)} value bet(s) found** using Poisson model probabilities")
                vb_df = pd.DataFrame(value_bets)
                fig_ev = go.Figure(go.Bar(
                    x=vb_df["EV %"],
                    y=[f"{r['Outcome']} @ {r['Bookmaker']}" for _, r in vb_df.iterrows()],
                    orientation="h",
                    marker_color="#1D9E75",
                    text=[f"+{v:.2f}%" for v in vb_df["EV %"]],
                    textposition="auto",
                ))
                fig_ev.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0D1117",
                    plot_bgcolor="#0D1117",
                    xaxis=dict(title="EV %", color="#7D8590", gridcolor="#21262D"),
                    yaxis=dict(color="#C9D1D9"),
                    height=max(200, len(value_bets) * 45),
                    margin=dict(t=10, b=30, l=180, r=20),
                )
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.info("No value bets found at current odds with this Poisson model. "
                        "Adjust λ values in the Poisson Model tab or verify your probability estimates.")

    # ── TAB 3: Kelly Bankroll Plan ────────────────────────────────────────────
    with sb_t3:
        st.subheader("💰 Kelly Criterion — Bankroll Allocation Plan")
        st.caption(
            f"Bankroll: **${bankroll:,.2f}** · Kelly fraction: **{int(kelly_mult*100)}%** "
            f"({int(kelly_mult*100)}% of full Kelly stake — standard risk management)"
        )

        if not analyses:
            st.warning("No odds data available.")
        else:
            plan_df = bankroll_plan(analyses, bankroll, kelly_mult)
            if plan_df.empty:
                st.info("No positive-EV bets found. Kelly Criterion only recommends bets with mathematical edge.")
                st.markdown("""
**Why no bets?**
- The Poisson model's probabilities don't exceed the bookmaker's no-vig fair probabilities.
- Adjust the λ sliders in the **Poisson Model** tab if you believe the market is mispriging a team.
- Kelly never bets on negative expected value — that is the correct mathematical behavior.
                """)
            else:
                total_stake = plan_df["Stake ($)"].sum()
                total_ev    = (plan_df["EV %"] * plan_df["Stake ($)"]).sum() / total_stake if total_stake else 0
                k_s1, k_s2, k_s3 = st.columns(3)
                k_s1.metric("Recommended bets", len(plan_df))
                k_s2.metric("Total stake",      f"${total_stake:,.2f}",
                             delta=f"{total_stake/bankroll*100:.1f}% of bankroll")
                k_s3.metric("Weighted avg EV",  f"{total_ev:.2f}%")

                def _stake_style(val):
                    if isinstance(val, (int, float)) and val > 0:
                        return "color: #1D9E75; font-weight: 600"
                    return ""

                styled_plan = plan_df.style.applymap(_stake_style, subset=["EV %"])
                st.dataframe(styled_plan, use_container_width=True, hide_index=True)

                # Kelly comparison chart
                st.subheader("Full vs Fractional Kelly Comparison")
                kelly_rows = []
                for _, row in plan_df.iterrows():
                    kf_full = kelly_fraction(row["Our Prob %"] / 100, row["Odds"])
                    label   = f"{row['Outcome']} @ {row['Bookmaker']}"
                    kelly_rows.append({
                        "label":     label,
                        "Full Kelly ($)":    round(bankroll * kf_full, 2),
                        "Half Kelly ($)":    round(bankroll * kf_full * 0.5, 2),
                        "Quarter Kelly ($)": round(bankroll * kf_full * 0.25, 2),
                    })
                if kelly_rows:
                    k_df = pd.DataFrame(kelly_rows)
                    fig_k = go.Figure()
                    for col, color in [
                        ("Full Kelly ($)", "#D85A30"),
                        ("Half Kelly ($)", "#BA7517"),
                        ("Quarter Kelly ($)", "#1D9E75"),
                    ]:
                        fig_k.add_trace(go.Bar(
                            name=col, x=k_df["label"], y=k_df[col],
                            marker_color=color,
                            text=[f"${v:,.0f}" for v in k_df[col]],
                            textposition="auto",
                        ))
                    fig_k.update_layout(
                        barmode="group",
                        template="plotly_dark",
                        paper_bgcolor="#0D1117",
                        plot_bgcolor="#0D1117",
                        legend=dict(font=dict(color="#C9D1D9")),
                        xaxis=dict(color="#7D8590"),
                        yaxis=dict(title="Stake ($)", color="#7D8590", gridcolor="#21262D"),
                        height=320,
                        margin=dict(t=10, b=50, l=60, r=20),
                    )
                    st.plotly_chart(fig_k, use_container_width=True)

                st.caption(
                    "⚠ **Risk note:** Full Kelly maximises long-run growth but produces "
                    "large swings. Quarter Kelly (25%) is the standard professional "
                    "recommendation. Never exceed full Kelly."
                )

    # ── TAB 4: Arbitrage Scanner ──────────────────────────────────────────────
    with sb_t4:
        st.subheader("⚡ Arbitrage Scanner")
        st.caption(
            "Finds the best available odds for each outcome across all bookmakers. "
            "If the implied probabilities sum to less than 100%, a risk-free profit exists."
        )

        best = best_odds_per_outcome(sel_event)
        if not best:
            st.warning("No odds data to scan.")
        else:
            home_key = home_team
            away_key = away_team
            draw_key = next((k for k in best if k not in (home_key, away_key)), None)

            arb_input = {home_key: best[home_key][0]} if home_key in best else {}
            if away_key in best:
                arb_input[away_key] = best[away_key][0]
            if draw_key and draw_key in best:
                arb_input["Draw"] = best[draw_key][0]

            arb_result = arbitrage_analysis(arb_input)

            # Best odds table
            arb_rows = []
            for name, (price, bk) in best.items():
                arb_rows.append({
                    "Outcome":          name,
                    "Best Odds":        price,
                    "Best Bookmaker":   bk,
                    "Implied Prob %":   round(implied_prob(price) * 100, 2),
                    "Fair Prob %":      round(100.0 / price, 2),
                })
            st.dataframe(pd.DataFrame(arb_rows), use_container_width=True, hide_index=True)

            # Margin gauge
            margin = arb_result["margin_total_pct"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=margin,
                delta={"reference": 100.0, "valueformat": ".2f",
                       "decreasing": {"color": "#1D9E75"},
                       "increasing": {"color": "#D85A30"}},
                title={"text": "Total Implied Probability %<br><sub>< 100% = arbitrage</sub>",
                       "font": {"color": "#C9D1D9"}},
                gauge={
                    "axis": {"range": [95, 115], "tickcolor": "#484F58",
                             "tickfont": {"color": "#7D8590"}},
                    "bar": {"color": "#1D9E75" if margin < 100 else "#D85A30"},
                    "bgcolor": "#21262D",
                    "bordercolor": "#30363D",
                    "steps": [
                        {"range": [95, 100], "color": "rgba(29,158,117,0.15)"},
                        {"range": [100, 115], "color": "rgba(216,90,48,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#58A6FF", "width": 3},
                        "thickness": 0.85,
                        "value": 100.0,
                    },
                },
                number={"suffix": "%", "font": {"color": "#E6EDF3", "size": 36}},
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                height=280,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            if arb_result["is_arbitrage"]:
                st.success(
                    f"✅ **ARBITRAGE OPPORTUNITY DETECTED** — "
                    f"Guaranteed profit: **{arb_result['profit_pct']:.3f}%** of total stake"
                )
                a_c1, a_c2 = st.columns(2)
                with a_c1:
                    st.subheader("Optimal stake distribution")
                    for outcome, pct in arb_result["stakes_pct"].items():
                        amt = bankroll * pct / 100.0
                        st.metric(outcome, f"${amt:,.2f}", delta=f"{pct:.1f}% of bankroll")
                with a_c2:
                    st.subheader("Profit calculation")
                    total_return = bankroll / (arb_result["margin_total_pct"] / 100.0)
                    profit_amt   = total_return - bankroll
                    st.metric("Total bankroll staked", f"${bankroll:,.2f}")
                    st.metric("Guaranteed return",     f"${total_return:,.2f}")
                    st.metric("Profit (risk-free)",    f"${profit_amt:,.2f}",
                               delta=f"+{arb_result['profit_pct']:.3f}%")
            else:
                deficit = arb_result["margin_total_pct"] - 100.0
                st.info(
                    f"No arbitrage at current best odds. The market is **{deficit:.2f}%** "
                    f"over fair value. Bookmakers are collectively extracting "
                    f"**{deficit:.2f}%** margin on this match."
                )

                # Show what odds would be needed
                st.subheader("Odds required to create arbitrage")
                if home_key in best and away_key in best:
                    # Keep home and draw odds, solve for away odds that closes arb
                    fixed_inv = sum(
                        1.0 / best[k][0]
                        for k in best
                        if k != away_key and k in best
                    )
                    needed_inv = 1.0 - fixed_inv
                    if needed_inv > 0:
                        needed_away_odds = round(1.0 / needed_inv, 2)
                        current_away = best.get(away_key, (0,))[0]
                        improvement  = needed_away_odds - current_away
                        st.write(
                            f"To achieve arb, **{away_key}** odds would need to be "
                            f"**{needed_away_odds:.2f}** (current best: {current_away:.2f}, "
                            f"gap: +{improvement:.2f})"
                        )


# ── Router ────────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Overview":        page_overview,
    "🔬 Research":        page_research,
    "📈 Charts 3D":       page_charts_3d,
    "🔮 AI Forecast":     page_ai_forecast,
    "🧪 Backtesting":     page_backtesting,
    "📐 Factor Lab":      page_factor_lab,
    "🔥 Stress Test":     page_stress_test,
    "📊 Quant Lab":       page_quant_lab,
    "🔄 Pairs Lab":       page_pairs_lab,
    "💼 Portfolio":       page_portfolio,
    "📈 Demo Portfolio":  page_demo_portfolio,
    "⚡ Alpha Score":     page_alpha,
    "⚽ Sports Betting":  page_sports_betting,
}

# Section-divider pseudo-pages redirect to Overview
if page.startswith("──"):
    page_overview()
else:
    PAGES[page]()

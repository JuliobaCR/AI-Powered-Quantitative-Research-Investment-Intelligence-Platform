"""
AlphaForge — Streamlit Dashboard.

Entry point: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import PORTFOLIO_CFG
from src.explainability.alpha_score import compute_alpha_score
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
from src.risk.engine import full_risk_report
from src.valuation.engine import dcf_sensitivity_table, run_valuation

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stMetric label {font-size: 0.75rem; color: #888;}
    .stMetric [data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 600;}
    div[data-testid="stSidebar"] {background: #0d0d12;}
    .block-container {padding-top: 1.5rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ AlphaForge")
    st.caption("AI-Powered Quant Research Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔬 Research", "📈 Charts 3D", "💼 Portfolio", "⚡ Alpha Score"],
        label_visibility="collapsed",
    )

    st.divider()
    default_tickers = PORTFOLIO_CFG.default_tickers
    tickers_input = st.text_input(
        "Watchlist (comma-separated)",
        value=", ".join(default_tickers),
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    selected = st.selectbox("Primary ticker", tickers)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

    st.divider()
    st.caption(f"v0.1.0 · Julio Ricardo Barrios Amador")


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker, period):
    return fetch_ohlcv(ticker, period=period)

@st.cache_data(ttl=300, show_spinner=False)
def load_indicators(ticker, period):
    df = load_ohlcv(ticker, period)
    return add_all_indicators(df) if not df.empty else df

@st.cache_data(ttl=300, show_spinner=False)
def load_fundamentals(ticker):
    return analyze_fundamentals(ticker)

@st.cache_data(ttl=300, show_spinner=False)
def load_valuation(ticker):
    return run_valuation(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def load_regimes(ticker, period):
    df = load_ohlcv(ticker, period)
    return detect_regimes(df["Close"]) if not df.empty else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_risk(ticker, period):
    df = load_ohlcv(ticker, period)
    spx = fetch_ohlcv("^GSPC", period=period)
    bench = spx["Close"] if not spx.empty else None
    return full_risk_report(df["Close"], bench) if not df.empty else {}

@st.cache_data(ttl=300, show_spinner=False)
def load_multi(tickers_tuple, period):
    return fetch_multi_ohlcv(list(tickers_tuple), period=period)

@st.cache_data(ttl=600, show_spinner=False)
def load_efficient_frontier(tickers_tuple, period):
    data = load_multi(tickers_tuple, period)
    returns = pd.DataFrame({t: d["Returns"] for t, d in data.items() if not d.empty}).dropna()
    return efficient_frontier(returns) if len(returns.columns) >= 2 else pd.DataFrame()

# ── Pages ────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("Market Overview")

    # Top metrics
    quotes = {t: fetch_quote(t) for t in tickers}
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
    st.title(f"Research — {selected}")
    info = fetch_company_info(selected)
    st.caption(f"**{info['name']}** · {info['sector']} · {info['industry']}")

    tab1, tab2, tab3 = st.tabs(["📋 Fundamentals", "💲 Valuation", "📰 Technicals"])

    with tab1:
        fund = load_fundamentals(selected)
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
        val = load_valuation(selected)
        if val:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P/E", f"{val.pe_ratio:.1f}x")
            c2.metric("PEG", f"{val.peg_ratio:.2f}")
            c3.metric("EV/EBITDA", f"{val.ev_ebitda:.1f}x")
            c4.metric("DCF Fair Value", f"${val.dcf_fair_value:.2f}",
                      delta=f"{val.dcf_margin_of_safety:+.1f}% MoS")

            st.subheader("DCF Sensitivity Analysis")
            sens = dcf_sensitivity_table(selected)
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
        df = load_indicators(selected, period)
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
    st.title("3D Analytics")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Volatility Surface",
        "🔷 Return–Risk–Score",
        "🌀 Regime Timeline 3D",
        "📐 Correlation Cube",
    ])

    with tab1:
        st.subheader("Implied Volatility Surface (Simulated)")
        st.caption("Strikes × Expirations × Implied Vol — based on Black-Scholes term structure")
        df = load_ohlcv(selected, "1y")
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
            fund = analyze_fundamentals(t)
            val = run_valuation(t)
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
        regime_df = load_regimes(selected, period)
        ohlcv = load_ohlcv(selected, period)
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


def page_alpha():
    st.title(f"⚡ Alpha Score — {selected}")

    fund = load_fundamentals(selected)
    val = load_valuation(selected)
    risk = load_risk(selected, period)
    df = load_indicators(selected, period)

    momentum_score = float(df["Momentum_Score"].iloc[-1]) if not df.empty else 50.0
    trend_score = min(momentum_score, 100.0)
    news_score = 55.0  # placeholder; real value from news_ai.sentiment

    result = compute_alpha_score(
        ticker=selected,
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
        f"{selected} · Score: **{result.total_score:.0f} / 100** · "
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


# ── Router ────────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Overview": page_overview,
    "🔬 Research": page_research,
    "📈 Charts 3D": page_charts_3d,
    "💼 Portfolio": page_portfolio,
    "⚡ Alpha Score": page_alpha,
}

PAGES[page]()

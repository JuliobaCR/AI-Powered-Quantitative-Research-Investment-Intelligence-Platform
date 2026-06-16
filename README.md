# ⚡ AlphaForge
### AI-Powered Quantitative Research & Investment Intelligence Platform

![Status](https://img.shields.io/badge/status-active%20development-orange)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35-red)
![License](https://img.shields.io/badge/license-MIT-green)

> *"Without data, you're just another person with an opinion."* — W. Edwards Deming

AlphaForge is a quantitative investment research platform combining market intelligence, fundamental analysis, options analytics, machine learning, and explainable AI into a single decision-support system.

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/JuliobaCR/AI-Powered-Quantitative-Research-Investment-Intelligence-Platform.git
cd AI-Powered-Quantitative-Research-Investment-Intelligence-Platform

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your Finnhub and Anthropic keys

# 5. Run the dashboard
streamlit run src/dashboard/app.py
```

Or with Docker:
```bash
docker-compose up --build
# Open http://localhost:8501
```

---

## 🏗️ Architecture

```
Market Data (yFinance + Finnhub)
         │
         ├── Technical Indicators Engine
         ├── News Intelligence (NLP Sentiment)
         ├── Fundamental Analysis Engine
         │         │
         │     Valuation Engine (DCF + Multiples)
         │
         ├── Market Regime Detection (HMM)
         ├── Derivatives Analytics (BS Greeks Surfaces + Options Strategies)
         ├── Forecasting Engine (GBM · Heston · Merton Jump-Diffusion · LSTM · Transformer)
         ├── Statistical Analysis (Hurst · GARCH · ADF · Ljung-Box · Jarque-Bera)
         ├── Risk Engine (VaR · CVaR · Component VaR · Attribution · Stress Testing)
         ├── Backtesting Engine (SMA · RSI · Parameter Sweeps)
         ├── Factor Model (Fama-French 5-Factor + Rolling Exposures)
         ├── Statistical Arbitrage (Cointegration · OU Half-Life · Pairs Signals)
         │
         └── Alpha Scoring System (0–100)
                   │
           Portfolio Optimizer (Mean-Variance · Risk Parity · Black-Litterman)
                   │
         Demo Portfolio Engine (Paper Trading · JSON persistence · Live P&L)
                   │
         Streamlit Dashboard (3D Mouse-Interactive Charts, 12 Pages)
         Auto-Refresh (Live Mode) · GitHub-Dark Comfort Theme
```

---

## 📊 Dashboard Modules (12 Pages)

| Page | Description |
|------|-------------|
| **📊 Overview** | Live quotes, performance chart, watchlist, regime indicator |
| **🔬 Research** | Fundamentals, DCF sensitivity heatmap, candlestick + RSI + MACD |
| **📈 Charts 3D** | Volatility surface, Return-Risk-Score scatter, Regime timeline, Rolling correlation, Greeks Lab (Delta/Gamma/Vega/Theta/Rho surfaces) |
| **🔮 AI Forecast** | GBM / LSTM / Transformer forecasts with MC-Dropout CIs; Heston stochastic-vol and Merton Jump-Diffusion Monte Carlo; 3D rotatable path fans for all 3 processes |
| **🧪 Backtesting** | SMA Crossover & RSI Mean Reversion strategies, equity curve vs buy & hold, drawdown, 3D Sharpe parameter-optimization surface |
| **📐 Factor Lab** | Fama-French 5-Factor OLS regression (alpha, betas, R²), 3D rolling factor-exposure surface |
| **🔥 Stress Test** | 10 historical scenarios (GFC · COVID · Dot-Com · 2022 Rate Shock · Black Monday · LTCM · EU Debt · Flash Crash · Taper Tantrum · China Devaluation), custom shock, heatmap + 3D scenario × asset P&L surface |
| **📊 Quant Lab** | Hurst exponent (R/S), GARCH(1,1) conditional vol, rolling autocorrelation, ADF stationarity test, Ljung-Box, Jarque-Bera, return histogram + Q-Q plot, options strategy payoff diagrams (10 strategies) + 3D P&L × spot × DTE surface |
| **🔄 Pairs Lab** | Engle-Granger cointegration, OU half-life estimation, spread z-score signals, pairs ranking table |
| **💼 Portfolio** | Efficient frontier, optimal weights (4 strategies), individual risk reports, component VaR, marginal VaR, volatility contribution attribution |
| **📈 Demo Portfolio** | Paper-trading account ($100,000 initial capital): live P&L tracking, buy/sell order execution, allocation pie, unrealized gain/loss bars, full trade history with CSV export |
| **⚡ Alpha Score** | Weighted conviction score (0–100), BUY/WAIT/SELL verdict, score breakdown chart |
| **⚽ Sports Betting** | Real-time odds from 80+ bookmakers (The Odds API): implied probability, no-vig fair odds, Expected Value (EV%) identification, Kelly Criterion bankroll calculator, three-way arbitrage scanner, independent Poisson match-outcome model with score-probability heatmap |

---

## ⚡ Alpha Score Framework

The core decision engine aggregates 6 sub-scores into a final conviction score [0–100]:

| Component | Weight | Source |
|-----------|--------|--------|
| Fundamentals | 35% | Revenue growth, FCF margin, ROIC, D/E |
| Market Trend | 20% | RSI, MACD, ADX, SMA positioning |
| News Sentiment | 15% | Finnhub NLP pipeline |
| Valuation | 10% | DCF, PEG, EV/EBITDA |
| Options Activity | 10% | GEX, Put/Call ratio, IV surface |
| Risk Profile | 10% | Sharpe, Max DD, VaR |

**Output:** BUY (≥70) · WAIT (50–70) · SELL (<50) with specific invalidation conditions.

---

## 📁 Repository Structure

```
alphaforge/
├── src/
│   ├── market_data/       # yFinance + Finnhub fetchers, technical indicators
│   ├── news_ai/           # NLP sentiment scoring
│   ├── fundamentals/      # Financial statement analysis
│   ├── valuation/         # DCF + multiples + sensitivity analysis
│   ├── regime_detection/  # Hidden Markov Model market regimes
│   ├── derivatives/       # Black-Scholes Greeks + Options Strategy payoff profiles
│   ├── forecasting/       # GBM · Heston · Jump-Diffusion + LSTM & Transformer
│   ├── analysis/          # Hurst exponent · GARCH(1,1) · ADF · Ljung-Box · JB test
│   ├── arbitrage/         # Pairs trading: cointegration · OU half-life · z-score signals
│   ├── risk/              # VaR · CVaR · Component VaR · Attribution · Stress Testing
│   ├── portfolio/         # Mean-variance · Risk Parity · Min-Var (cvxpy) + Demo paper-trading
│   ├── backtesting/       # Strategy backtests (SMA, RSI) + parameter sweeps
│   ├── factors/           # Fama-French 5-Factor regression + rolling exposures
│   ├── explainability/    # Alpha Score aggregation
│   └── dashboard/         # Streamlit app (12 pages, 3D Plotly charts, Live Mode)
├── config/                # Global settings, API keys management
│   ├── sports/            # Sports betting: The Odds API client, Poisson model, EV/Kelly/arbitrage
├── tests/                 # pytest unit tests (17 test files)
├── data/                  # raw / processed / external data
├── models/                # serialized ML models
├── notebooks/             # Jupyter research notebooks
├── Makefile               # make run / make test / make lint
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🔑 API Keys Required

| Service | Purpose | Free Tier | Link |
|---------|---------|-----------|------|
| **Finnhub** | Real-time quotes, news | 60 calls/min | [finnhub.io](https://finnhub.io) |
| **Anthropic** | AI Research Assistant | Pay-per-use | [console.anthropic.com](https://console.anthropic.com) |
| **The Odds API** | Real-time sports odds (80+ bookmakers) | 500 req/month | [the-odds-api.com](https://the-odds-api.com) |
| yFinance | Historical data, fundamentals | ✅ Free, no key | Built-in |

---

## 🗺️ Roadmap

- [x] Phase 1 — Data Infrastructure (yFinance + Finnhub pipelines)
- [x] Phase 2 — Research Layer (Fundamentals, Valuation, News)
- [x] Phase 3 — Quantitative Layer (HMM Regimes, Risk, Greeks)
- [x] Phase 4 — Alpha Engine (Scoring + Explainability)
- [x] Phase 5 — Portfolio Layer (Optimization, Efficient Frontier)
- [x] Phase 6 — Visualization (3D Dashboard, Streamlit)
- [x] Phase 7 — Deep Learning (LSTM, Transformer forecasting + MC-Dropout CI)
- [x] Phase 8 — Backtesting Framework (SMA Crossover, RSI Mean Reversion, parameter sweep)
- [x] Phase 9 — Factor Modeling (Fama-French 5-factor + rolling exposures)
- [x] Phase 10 — Advanced Stochastic Processes (Heston, Merton Jump-Diffusion, CEV)
- [x] Phase 11 — Stress Testing & Scenario Analysis (10 historical scenarios + custom shocks)
- [x] Phase 12 — Statistical Analysis Suite (Hurst · GARCH · ADF · Ljung-Box · Jarque-Bera)
- [x] Phase 13 — Options Strategies (10 payoff profiles + 3D P&L surface)
- [x] Phase 14 — Statistical Arbitrage / Pairs Lab (Engle-Granger, OU, z-score signals)
- [x] Phase 15 — Risk Attribution (Component VaR, Marginal VaR, Vol Decomposition)
- [x] Phase 16 — UX & Paper Trading (GitHub-Dark comfort theme, Live auto-refresh, Demo Portfolio engine)
- [x] Phase 17 — Sports Betting Analytics (EV identification, Kelly Criterion, Poisson model, arbitrage scanner, The Odds API integration)
- [ ] Phase 18 — REST API (FastAPI + WebSocket streaming)

---

## 👨‍💻 Author

**Julio Ricardo Barrios Amador**  
Computer Engineering Student · Instituto Tecnológico de Costa Rica  
IEEE Costa Rica Section Student Representative  
[LinkedIn](https://linkedin.com/in/julio-ricardo-barrios-amador) · [GitHub](https://github.com/JuliobaCR)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

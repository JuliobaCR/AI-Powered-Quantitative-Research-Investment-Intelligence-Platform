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
         ├── Derivatives Analytics (BS Greeks)
         ├── Forecasting Engine (GBM, LSTM*)
         ├── Risk Engine (VaR, CVaR, Sharpe)
         │
         └── Alpha Scoring System (0–100)
                   │
           Portfolio Optimizer (Mean-Variance, Risk Parity)
                   │
         Streamlit Dashboard (3D Charts, Interactive)
```

---

## 📊 Dashboard Modules

| Page | Description |
|------|-------------|
| **Overview** | Live quotes, performance chart, watchlist, regime indicator |
| **Research** | Fundamentals, DCF sensitivity heatmap, candlestick + RSI + MACD |
| **Charts 3D** | Volatility surface, Return-Risk-Score scatter, Regime timeline, Rolling correlation |
| **Portfolio** | Efficient frontier, optimal weights, risk decomposition per ticker |
| **Alpha Score** | Weighted conviction score, reasoning bullets, invalidation conditions |

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
│   ├── derivatives/       # Black-Scholes Greeks
│   ├── forecasting/       # ML return forecasting (GBM, LSTM*)
│   ├── risk/              # VaR, CVaR, Sharpe, drawdown
│   ├── portfolio/         # Mean-variance, Risk Parity, Min-Var (cvxpy)
│   ├── explainability/    # Alpha Score aggregation
│   └── dashboard/         # Streamlit app with 3D Plotly charts
├── config/                # Global settings, API keys management
├── tests/                 # pytest unit tests
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
| yFinance | Historical data, fundamentals | ✅ Free, no key | Built-in |

---

## 🗺️ Roadmap

- [x] Phase 1 — Data Infrastructure (yFinance + Finnhub pipelines)
- [x] Phase 2 — Research Layer (Fundamentals, Valuation, News)
- [x] Phase 3 — Quantitative Layer (HMM Regimes, Risk, Greeks)
- [x] Phase 4 — Alpha Engine (Scoring + Explainability)
- [x] Phase 5 — Portfolio Layer (Optimization, Efficient Frontier)
- [x] Phase 6 — Visualization (3D Dashboard, Streamlit)
- [ ] Phase 7 — Deep Learning (LSTM, Transformer forecasting)
- [ ] Phase 8 — Backtesting Framework
- [ ] Phase 9 — Factor Modeling (Fama-French 5-factor)
- [ ] Phase 10 — REST API (FastAPI)

---

## 👨‍💻 Author

**Julio Ricardo Barrios Amador**  
Computer Engineering Student · Instituto Tecnológico de Costa Rica  
IEEE Costa Rica Section Student Representative  
[LinkedIn](https://linkedin.com/in/julio-ricardo-barrios-amador) · [GitHub](https://github.com/JuliobaCR)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

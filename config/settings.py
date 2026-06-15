"""
AlphaForge — Global configuration and constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class DataConfig:
    cache_ttl: int = int(os.getenv("CACHE_TTL_SECONDS", 300))
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass(frozen=True)
class PortfolioConfig:
    default_tickers: list[str] = field(
        default_factory=lambda: ["META", "CRDO", "GEV", "IONQ"]
    )
    risk_free_rate: float = 0.0525  # ~US 10Y yield
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class AlphaScoreWeights:
    fundamentals: float = 0.35
    market_trend: float = 0.20
    news_sentiment: float = 0.15
    valuation: float = 0.10
    options_activity: float = 0.10
    risk_profile: float = 0.10


@dataclass(frozen=True)
class RegimeConfig:
    n_states: int = 4  # Bull, Bear, Sideways, HighVol
    lookback_days: int = 252
    min_regime_duration_days: int = 10


DATA_CFG = DataConfig()
PORTFOLIO_CFG = PortfolioConfig()
ALPHA_WEIGHTS = AlphaScoreWeights()
REGIME_CFG = RegimeConfig()

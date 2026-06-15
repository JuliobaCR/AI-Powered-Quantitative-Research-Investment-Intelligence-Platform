"""
Alpha Scoring System.

Central decision engine that aggregates all sub-module scores into
a single conviction score [0–100] with explainable reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import ALPHA_WEIGHTS


@dataclass
class AlphaScoreResult:
    ticker: str
    total_score: float                  # 0–100
    component_scores: dict[str, float]  # raw scores per module
    weighted_contributions: dict[str, float]
    verdict: str                        # BUY | WAIT | SELL
    confidence: float                   # 0–1
    reasoning: list[str]                # human-readable bullets
    invalidation: str                   # specific condition to flip thesis


def compute_alpha_score(
    ticker: str,
    fundamental_score: float = 0.0,
    market_trend_score: float = 0.0,
    news_sentiment_score: float = 0.0,
    valuation_score: float = 0.0,
    options_score: float = 0.0,
    risk_score: float = 0.0,
) -> AlphaScoreResult:
    """
    Combine sub-scores into a weighted Alpha Score.

    All inputs should be in [0, 100].
    """
    raw = {
        "Fundamentals": fundamental_score,
        "Market Trend": market_trend_score,
        "News Sentiment": news_sentiment_score,
        "Valuation": valuation_score,
        "Options Activity": options_score,
        "Risk Profile": risk_score,
    }
    weights = {
        "Fundamentals": ALPHA_WEIGHTS.fundamentals,
        "Market Trend": ALPHA_WEIGHTS.market_trend,
        "News Sentiment": ALPHA_WEIGHTS.news_sentiment,
        "Valuation": ALPHA_WEIGHTS.valuation,
        "Options Activity": ALPHA_WEIGHTS.options_activity,
        "Risk Profile": ALPHA_WEIGHTS.risk_profile,
    }

    weighted = {k: round(raw[k] * weights[k], 2) for k in raw}
    total = round(sum(weighted.values()), 2)

    verdict, confidence = _verdict(total, raw)
    reasoning = _build_reasoning(raw, ticker)
    invalidation = _build_invalidation(ticker, total, raw)

    return AlphaScoreResult(
        ticker=ticker,
        total_score=total,
        component_scores=raw,
        weighted_contributions=weighted,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        invalidation=invalidation,
    )


def _verdict(score: float, raw: dict) -> tuple[str, float]:
    if score >= 70:
        verdict = "BUY"
        confidence = min(0.60 + (score - 70) / 100, 0.95)
    elif score >= 50:
        verdict = "WAIT"
        confidence = 0.50
    else:
        verdict = "SELL"
        confidence = min(0.60 + (50 - score) / 100, 0.95)
    return verdict, round(confidence, 2)


def _build_reasoning(raw: dict, ticker: str) -> list[str]:
    bullets: list[str] = []
    if raw["Fundamentals"] >= 70:
        bullets.append("✅ Fundamentals sólidos: alto FCF margin y eficiencia de capital")
    elif raw["Fundamentals"] < 40:
        bullets.append("⚠️  Debilidad fundamental: márgenes o crecimiento bajo presión")

    if raw["Market Trend"] >= 70:
        bullets.append("✅ Tendencia técnica alcista: momentum y precio sobre medias clave")
    elif raw["Market Trend"] < 40:
        bullets.append("⚠️  Tendencia técnica débil o bearish")

    if raw["News Sentiment"] >= 65:
        bullets.append("✅ Sentiment positivo en noticias recientes")
    elif raw["News Sentiment"] < 35:
        bullets.append("⚠️  Sentiment negativo en noticias — posible catalizador bajista")

    if raw["Valuation"] >= 60:
        bullets.append("✅ Valuación atractiva relativa al valor intrínseco (DCF)")
    elif raw["Valuation"] < 35:
        bullets.append("⚠️  Valuación extendida — múltiplos elevados vs sector")

    if raw["Risk Profile"] >= 65:
        bullets.append("✅ Perfil de riesgo favorable: Sharpe y drawdown controlado")
    elif raw["Risk Profile"] < 35:
        bullets.append("⚠️  Riesgo elevado: alta volatilidad o drawdown significativo")

    if not bullets:
        bullets.append(f"ℹ️  Señales mixtas en {ticker} — monitorear catalizadores")

    return bullets


def _build_invalidation(ticker: str, score: float, raw: dict) -> str:
    if score >= 70:
        return (
            f"Tesis bajista activada si: precio cae >15% desde máximos, "
            f"fundamentales deterioran (FCF negativo por 2 trimestres), "
            f"o sentiment score cae por debajo de 30."
        )
    elif score < 50:
        return (
            f"Tesis alcista activada si: score Alpha supera 70, "
            f"earnings beat + revisión de guidance, "
            f"o precio rompe resistencia clave con volumen."
        )
    return f"Monitorear: cualquier cambio en fundamentals o regime shift altera el score de {ticker}."

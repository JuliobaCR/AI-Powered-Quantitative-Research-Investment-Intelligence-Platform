"""
News Intelligence Engine.

Fetches financial news via Finnhub and scores:
  - Sentiment (positive / neutral / negative)
  - Impact score
  - Confidence score

Falls back to keyword-based scoring when Finnhub unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import finnhub

from config.settings import DATA_CFG

logger = logging.getLogger(__name__)

BULLISH_WORDS = {
    "beat", "beats", "exceeds", "record", "growth", "surge", "rally",
    "upgrade", "outperform", "buy", "strong", "positive", "profit",
    "expansion", "accelerate", "breakthrough", "raised", "bullish",
}
BEARISH_WORDS = {
    "miss", "misses", "decline", "fall", "drop", "downgrade", "sell",
    "loss", "weak", "negative", "layoff", "risk", "concern", "cut",
    "warning", "investigation", "lawsuit", "recall", "bearish",
}


@dataclass
class NewsItem:
    headline: str
    source: str
    datetime: str
    sentiment: float       # -1 to 1
    sentiment_label: str   # "Positivo" | "Neutro" | "Negativo"
    impact: float          # 0–1
    confidence: float      # 0–1
    url: str = ""


def fetch_news(ticker: str, limit: int = 10) -> list[NewsItem]:
    """
    Fetch and score recent news for a ticker.
    Uses Finnhub if key is available, otherwise returns empty list.
    """
    items: list[NewsItem] = []
    fh_client = _get_finnhub()

    if fh_client:
        try:
            raw = fh_client.company_news(ticker, _from="2025-01-01", to="2026-12-31")
            for article in raw[:limit]:
                headline = article.get("headline", "")
                sent, label, conf = _score_sentiment(headline)
                impact = _estimate_impact(headline, article.get("source", ""))
                dt = datetime.fromtimestamp(
                    article.get("datetime", 0), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M")

                items.append(
                    NewsItem(
                        headline=headline,
                        source=article.get("source", "Unknown"),
                        datetime=dt,
                        sentiment=sent,
                        sentiment_label=label,
                        impact=impact,
                        confidence=conf,
                        url=article.get("url", ""),
                    )
                )
        except Exception as exc:
            logger.warning("Finnhub news failed for %s: %s", ticker, exc)

    return items


def aggregate_sentiment(news_items: list[NewsItem]) -> dict:
    """Aggregate news items into a single sentiment signal."""
    if not news_items:
        return {"score": 0.0, "label": "Sin datos", "confidence": 0.0, "count": 0}

    weighted_sent = sum(n.sentiment * n.impact for n in news_items)
    total_weight = sum(n.impact for n in news_items) or 1.0
    avg_sent = weighted_sent / total_weight
    avg_conf = sum(n.confidence for n in news_items) / len(news_items)

    if avg_sent > 0.2:
        label = "Positivo"
    elif avg_sent < -0.2:
        label = "Negativo"
    else:
        label = "Neutro"

    return {
        "score": round(avg_sent, 3),
        "normalized": round((avg_sent + 1) / 2 * 100, 1),  # 0–100
        "label": label,
        "confidence": round(avg_conf, 3),
        "count": len(news_items),
        "positive_pct": round(
            sum(1 for n in news_items if n.sentiment > 0.2) / len(news_items) * 100, 1
        ),
    }


def _score_sentiment(text: str) -> tuple[float, str, float]:
    """Keyword-based sentiment scoring. Returns (score -1..1, label, confidence)."""
    words = set(re.findall(r"\w+", text.lower()))
    pos = len(words & BULLISH_WORDS)
    neg = len(words & BEARISH_WORDS)

    if pos == 0 and neg == 0:
        return 0.0, "Neutro", 0.3

    total = pos + neg
    score = (pos - neg) / total
    confidence = min(0.4 + total * 0.1, 0.9)
    label = "Positivo" if score > 0.2 else "Negativo" if score < -0.2 else "Neutro"
    return round(score, 3), label, round(confidence, 3)


def _estimate_impact(headline: str, source: str) -> float:
    """Estimate news impact [0–1] based on keywords and source credibility."""
    HIGH_IMPACT = {"earnings", "revenue", "guidance", "acquisition", "merger", "fda", "sec"}
    MED_IMPACT = {"analyst", "upgrade", "downgrade", "partnership", "contract"}
    HIGH_CREDIBILITY = {"reuters", "bloomberg", "wsj", "financial times", "cnbc", "ft"}

    words = set(re.findall(r"\w+", headline.lower()))
    if words & HIGH_IMPACT:
        base = 0.75
    elif words & MED_IMPACT:
        base = 0.55
    else:
        base = 0.35

    source_boost = 0.10 if any(h in source.lower() for h in HIGH_CREDIBILITY) else 0.0
    return round(min(base + source_boost, 1.0), 2)


def _get_finnhub() -> finnhub.Client | None:
    if DATA_CFG.finnhub_api_key:
        try:
            return finnhub.Client(api_key=DATA_CFG.finnhub_api_key)
        except Exception:
            pass
    return None

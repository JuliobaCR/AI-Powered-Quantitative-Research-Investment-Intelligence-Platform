"""
Valuation Engine.

Methods:
  - Multiples: P/E, PEG, EV/EBITDA, P/S, P/B
  - DCF: intrinsic value with sensitivity analysis
  - Composite Valuation Score [0–100]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    ticker: str
    price: float
    pe_ratio: float
    peg_ratio: float
    ev_ebitda: float
    ps_ratio: float
    pb_ratio: float
    dcf_fair_value: float
    dcf_margin_of_safety: float   # pct; positive = undervalued
    dcf_confidence: float         # 0–1
    valuation_score: float        # 0–100; higher = cheaper relative to value
    verdict: str                  # "Undervalued" | "Fair" | "Overvalued"


def run_valuation(ticker: str) -> ValuationResult | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

        pe = info.get("trailingPE") or info.get("forwardPE") or 0
        peg = info.get("pegRatio") or 0
        ev_ebitda = info.get("enterpriseToEbitda") or 0
        ps = info.get("priceToSalesTrailing12Months") or 0
        pb = info.get("priceToBook") or 0

        dcf_value, dcf_conf = _dcf_estimate(info)
        mos = ((dcf_value - price) / price * 100) if price else 0.0

        score = _valuation_score(pe, peg, ev_ebitda, ps, pb, mos)
        verdict = "Undervalued" if mos > 15 else "Overvalued" if mos < -15 else "Fair"

        return ValuationResult(
            ticker=ticker,
            price=round(price, 2),
            pe_ratio=round(pe, 2),
            peg_ratio=round(peg, 2),
            ev_ebitda=round(ev_ebitda, 2),
            ps_ratio=round(ps, 2),
            pb_ratio=round(pb, 2),
            dcf_fair_value=round(dcf_value, 2),
            dcf_margin_of_safety=round(mos, 2),
            dcf_confidence=round(dcf_conf, 2),
            valuation_score=round(score, 2),
            verdict=verdict,
        )
    except Exception as exc:
        logger.error("Valuation failed for %s: %s", ticker, exc)
        return None


def _dcf_estimate(info: dict, years: int = 5) -> tuple[float, float]:
    """
    Simplified DCF: grows FCF for `years` at analyst EPS growth rate,
    then applies a terminal multiple. Returns (fair_value, confidence).
    """
    fcf = info.get("freeCashflow", 0) or 0
    shares = info.get("sharesOutstanding", 1) or 1
    wacc = 0.10  # 10% discount rate
    terminal_multiple = 20.0
    growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.10
    growth_rate = min(max(float(growth_rate), -0.30), 0.50)

    if fcf <= 0 or shares <= 0:
        return 0.0, 0.0

    fcf_per_share = fcf / shares
    pv_sum = 0.0
    for yr in range(1, years + 1):
        projected = fcf_per_share * (1 + growth_rate) ** yr
        pv_sum += projected / (1 + wacc) ** yr

    terminal_value = (fcf_per_share * (1 + growth_rate) ** years * terminal_multiple)
    pv_terminal = terminal_value / (1 + wacc) ** years
    fair_value = pv_sum + pv_terminal

    # Confidence: lower if growth rate assumed, FCF negative, or thin data
    confidence = 0.7 if info.get("earningsGrowth") else 0.4
    return fair_value, confidence


def _valuation_score(pe, peg, ev_ebitda, ps, pb, mos) -> float:
    """Higher score = more attractive valuation."""
    score = 50.0  # neutral baseline

    if 0 < pe < 20:
        score += 15
    elif 20 <= pe < 35:
        score += 5
    elif pe > 60:
        score -= 15

    if 0 < peg < 1:
        score += 20
    elif 1 <= peg < 1.5:
        score += 10
    elif peg > 3:
        score -= 10

    score += min(mos / 2, 15)  # up to +15 pts for margin of safety

    return round(min(max(score, 0), 100), 2)


def dcf_sensitivity_table(ticker: str) -> dict:
    """
    Build a 5×5 sensitivity matrix: rows = WACC, cols = growth rate.
    Returns dict with 'waccs', 'growths', 'values'.
    """
    try:
        info = yf.Ticker(ticker).info
        fcf = info.get("freeCashflow", 0) or 0
        shares = info.get("sharesOutstanding", 1) or 1
        if fcf <= 0:
            return {}

        waccs = [0.07, 0.09, 0.10, 0.12, 0.14]
        growths = [0.05, 0.10, 0.15, 0.20, 0.25]
        table = []

        for w in waccs:
            row = []
            for g in growths:
                tmp_info = dict(info)
                tmp_info["earningsGrowth"] = g
                tmp_info["freeCashflow"] = fcf
                tmp_info["sharesOutstanding"] = shares
                val, _ = _dcf_estimate(tmp_info)
                row.append(round(val, 2))
            table.append(row)

        return {"waccs": waccs, "growths": growths, "values": table}
    except Exception as exc:
        logger.error("DCF sensitivity failed for %s: %s", ticker, exc)
        return {}

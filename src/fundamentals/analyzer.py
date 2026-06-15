"""
Fundamental Analysis Engine.

Fetches financial statements via yFinance and computes:
  - Profitability metrics
  - Cash flow metrics
  - Capital efficiency (ROE, ROIC, ROA)
  - Financial stability ratios
  - Composite Fundamental Score [0–100]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    ticker: str
    revenue_growth_yoy: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    ebitda_margin: float
    fcf_margin: float
    fcf_growth_yoy: float
    roe: float
    roa: float
    roic: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    interest_coverage: float
    cash_conversion: float
    fundamental_score: float  # 0–100

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def analyze_fundamentals(ticker: str) -> FundamentalMetrics | None:
    """
    Fetch and compute all fundamental metrics for a given ticker.
    Returns None if data is unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # --- Income Statement ---
        rev = info.get("totalRevenue", 0) or 0
        rev_prev = info.get("revenueGrowth", 0)
        gross_margin = _pct(info.get("grossMargins"))
        op_margin = _pct(info.get("operatingMargins"))
        net_margin = _pct(info.get("profitMargins"))
        ebitda = info.get("ebitda", 0) or 0
        ebitda_margin = (ebitda / rev * 100) if rev else 0.0

        # --- Cash Flow ---
        fcf = info.get("freeCashflow", 0) or 0
        ocf = info.get("operatingCashflow", 0) or 0
        fcf_margin = (fcf / rev * 100) if rev else 0.0
        cash_conversion = (fcf / ocf) if ocf else 0.0
        fcf_growth = _pct(info.get("revenueGrowth"))  # proxy; ideally TTM vs prior

        # --- Capital Efficiency ---
        roe = _pct(info.get("returnOnEquity"))
        roa = _pct(info.get("returnOnAssets"))
        roic = _estimate_roic(info)

        # --- Balance Sheet ---
        dte = info.get("debtToEquity", 0)
        dte = (dte / 100) if dte else 0.0
        current_ratio = info.get("currentRatio", 0) or 0.0
        quick_ratio = info.get("quickRatio", 0) or 0.0
        ebit = info.get("ebit", ebitda * 0.8) or 0
        interest = info.get("interestExpense", 0) or 1
        interest_coverage = abs(ebit / interest) if interest else 10.0

        score = _compute_fundamental_score(
            rev_growth=_pct(rev_prev),
            gross_margin=gross_margin,
            op_margin=op_margin,
            net_margin=net_margin,
            fcf_margin=fcf_margin,
            roe=roe,
            roic=roic,
            dte=dte,
            current_ratio=current_ratio,
        )

        return FundamentalMetrics(
            ticker=ticker,
            revenue_growth_yoy=_pct(rev_prev),
            gross_margin=gross_margin,
            operating_margin=op_margin,
            net_margin=net_margin,
            ebitda_margin=ebitda_margin,
            fcf_margin=fcf_margin,
            fcf_growth_yoy=fcf_growth,
            roe=roe,
            roa=roa,
            roic=roic,
            debt_to_equity=dte,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            interest_coverage=min(interest_coverage, 50.0),
            cash_conversion=cash_conversion,
            fundamental_score=score,
        )

    except Exception as exc:
        logger.error("Fundamental analysis failed for %s: %s", ticker, exc)
        return None


def _pct(val) -> float:
    if val is None:
        return 0.0
    return float(val) * 100 if abs(float(val)) < 5 else float(val)


def _estimate_roic(info: dict) -> float:
    ebit = info.get("ebit", 0) or 0
    tax_rate = info.get("effectiveTaxRate", 0.21) or 0.21
    nopat = ebit * (1 - tax_rate)
    total_assets = info.get("totalAssets", 0) or 0
    total_liabilities = info.get("totalLiab", 0) or 0
    invested_capital = total_assets - total_liabilities
    return (nopat / invested_capital * 100) if invested_capital else 0.0


def _compute_fundamental_score(
    rev_growth: float,
    gross_margin: float,
    op_margin: float,
    net_margin: float,
    fcf_margin: float,
    roe: float,
    roic: float,
    dte: float,
    current_ratio: float,
) -> float:
    """
    Rule-based scoring system. Each dimension contributes to a 0–100 score.
    Weights mirror the Alpha Score framework (fundamentals = 35% of total).
    """
    scores: list[float] = []

    # Revenue growth (0–30 pts)
    scores.append(min(rev_growth / 30 * 30, 30))

    # Profitability (0–25 pts)
    scores.append(min(gross_margin / 80 * 15, 15))
    scores.append(min(max(op_margin, 0) / 30 * 10, 10))

    # Cash generation (0–20 pts)
    scores.append(min(max(fcf_margin, 0) / 25 * 20, 20))

    # Capital efficiency (0–15 pts)
    scores.append(min(max(roe, 0) / 30 * 8, 8))
    scores.append(min(max(roic, 0) / 25 * 7, 7))

    # Balance sheet health (0–10 pts)
    leverage_score = max(0, 10 - dte * 3)
    liquidity_score = min(current_ratio / 2 * 5, 5)
    scores.append(leverage_score * 0.5 + liquidity_score * 0.5)

    return round(min(sum(scores), 100), 2)

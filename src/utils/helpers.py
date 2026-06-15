"""Utility helpers for AlphaForge."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd


def format_large_number(n: float) -> str:
    """Format large numbers: 1_500_000 → '1.5M'."""
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def annualized_return(prices: pd.Series) -> float:
    """CAGR from a price series."""
    r = prices.pct_change().dropna()
    return float((1 + r.mean()) ** 252 - 1)


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 30) -> pd.Series:
    return s1.rolling(window).corr(s2)


def cache_key(*args) -> str:
    """Create a deterministic hash from arguments."""
    payload = "_".join(str(a) for a in args)
    return hashlib.md5(payload.encode()).hexdigest()[:12]

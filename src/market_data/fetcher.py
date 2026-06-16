"""
Market Data Fetcher — yFinance + Finnhub.

Provides a unified interface to fetch:
  - OHLCV price history
  - Real-time quotes
  - Options chains
  - Company info
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

import finnhub
import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import DATA_CFG

logger = logging.getLogger(__name__)

_finnhub_client: Optional[finnhub.Client] = None


def get_finnhub() -> Optional[finnhub.Client]:
    global _finnhub_client
    if _finnhub_client is None and DATA_CFG.finnhub_api_key:
        _finnhub_client = finnhub.Client(api_key=DATA_CFG.finnhub_api_key)
    return _finnhub_client


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data via yFinance.

    Args:
        ticker:   Stock symbol, e.g. "AAPL".
        period:   yFinance period string, e.g. "1y", "6mo", "5y".
        interval: Bar interval, e.g. "1d", "1h", "1wk".

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume, Returns].
    """
    try:
        df: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df.dropna(subset=["Returns"], inplace=True)
        df.attrs["ticker"] = ticker
        return df
    except Exception as exc:
        logger.error("fetch_ohlcv failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_multi_ohlcv(
    tickers: list[str],
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers, returned as a dict keyed by symbol."""
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        result[ticker] = fetch_ohlcv(ticker, period=period)
        time.sleep(0.1)  # polite rate limiting
    return result


# ---------------------------------------------------------------------------
# Real-time quotes
# ---------------------------------------------------------------------------

def fetch_quote(ticker: str) -> dict:
    """
    Fetch real-time quote via Finnhub. Falls back to yFinance fast_info.

    Returns dict with keys: price, change, change_pct, high, low, open, prev_close.
    """
    fh = get_finnhub()
    if fh:
        try:
            q = fh.quote(ticker)
            return {
                "price": q.get("c", 0.0),
                "change": q.get("d", 0.0),
                "change_pct": q.get("dp", 0.0),
                "high": q.get("h", 0.0),
                "low": q.get("l", 0.0),
                "open": q.get("o", 0.0),
                "prev_close": q.get("pc", 0.0),
                "source": "finnhub",
            }
        except Exception as exc:
            logger.warning("Finnhub quote failed for %s: %s. Falling back.", ticker, exc)

    # yFinance fallback
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.last_price or 0.0
        prev = info.previous_close or price
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0.0
        return {
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "high": info.day_high or price,
            "low": info.day_low or price,
            "open": info.open or price,
            "prev_close": prev,
            "source": "yfinance",
        }
    except Exception as exc:
        logger.error("Quote fetch failed for %s: %s", ticker, exc)
        return {"price": 0.0, "change": 0.0, "change_pct": 0.0, "source": "error"}


# ---------------------------------------------------------------------------
# Company info & fundamentals snapshot
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def fetch_company_info(ticker: str) -> dict:
    """Cache company metadata from yFinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "employees": info.get("fullTimeEmployees", 0),
            "description": info.get("longBusinessSummary", ""),
            "website": info.get("website", ""),
            "country": info.get("country", ""),
        }
    except Exception as exc:
        logger.error("fetch_company_info failed for %s: %s", ticker, exc)
        return {"name": ticker, "sector": "N/A", "industry": "N/A"}


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------

def fetch_options_chain(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Fetch options chain (calls + puts) for the nearest expiration.

    Returns dict with 'calls' and 'puts' DataFrames.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
        nearest_exp = expirations[0]
        chain = t.option_chain(nearest_exp)
        return {
            "calls": chain.calls,
            "puts": chain.puts,
            "expiration": nearest_exp,
        }
    except Exception as exc:
        logger.error("fetch_options_chain failed for %s: %s", ticker, exc)
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}

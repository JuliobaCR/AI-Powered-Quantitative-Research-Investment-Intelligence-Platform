"""
Technical Indicators Engine.

Wraps the `ta` library + custom implementations for:
  - Trend: SMA, EMA, MACD, ADX
  - Momentum: RSI, Stochastic, Williams %R, ROC
  - Volatility: Bollinger Bands, ATR, Keltner Channel
  - Volume: OBV, VWAP, CMF
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a comprehensive set of technical indicators to an OHLCV DataFrame."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Trend ---
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    df["ADX"] = ta.trend.adx(high, low, close, window=14)

    # --- Momentum ---
    df["RSI_14"] = ta.momentum.rsi(close, window=14)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["Williams_R"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["ROC_10"] = ta.momentum.roc(close, window=10)

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Pct"] = bb.bollinger_pband()

    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # --- Volume ---
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["CMF_20"] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)

    # --- Custom ---
    df["Price_vs_SMA200"] = (close / df["SMA_200"] - 1) * 100
    df["Volatility_20d"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
    df["Momentum_Score"] = _momentum_score(df)

    return df


def _momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite momentum score [0–100] combining RSI, MACD, and ROC.
    """
    rsi_norm = df["RSI_14"] / 100
    macd_norm = (df["MACD_Hist"] - df["MACD_Hist"].min()) / (
        df["MACD_Hist"].max() - df["MACD_Hist"].min() + 1e-9
    )
    roc_norm = (df["ROC_10"] - df["ROC_10"].min()) / (
        df["ROC_10"].max() - df["ROC_10"].min() + 1e-9
    )
    score = (0.4 * rsi_norm + 0.35 * macd_norm + 0.25 * roc_norm) * 100
    return score.clip(0, 100)

"""
Forecasting Engine — placeholder stubs for ML models.

Full implementations (LSTM, XGBoost, TFT) are in Phase 3 development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def simple_return_forecast(
    prices: pd.Series,
    horizon_days: int = 20,
) -> dict:
    """
    Gradient Boosting forecast using lagged returns as features.
    Returns expected return and a confidence interval.
    """
    r = prices.pct_change().dropna()
    n_lags = 20

    if len(r) < n_lags + horizon_days + 10:
        return {"expected_return": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    X, y = [], []
    for i in range(n_lags, len(r) - horizon_days):
        X.append(r.iloc[i - n_lags:i].values)
        y.append(r.iloc[i:i + horizon_days].sum())

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(X) * 0.8)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_scaled[:split], y[:split])

    X_latest = scaler.transform(r.values[-n_lags:].reshape(1, -1))
    pred = float(model.predict(X_latest)[0])

    std = float(np.std(y[split:] - model.predict(X_scaled[split:])))
    return {
        "expected_return_pct": round(pred * 100, 2),
        "ci_low_pct": round((pred - 1.96 * std) * 100, 2),
        "ci_high_pct": round((pred + 1.96 * std) * 100, 2),
        "horizon_days": horizon_days,
    }

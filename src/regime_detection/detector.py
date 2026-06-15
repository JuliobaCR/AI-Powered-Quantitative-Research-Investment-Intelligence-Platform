"""
Market Regime Detection Engine.

Uses Hidden Markov Models (HMM) to classify market into regimes:
  0 = Bull / Low Volatility
  1 = Bull / High Volatility
  2 = Bear / Low Volatility
  3 = Bear / High Volatility

Features: daily returns + realized volatility (20d rolling std).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "Bull / Low Vol",
    1: "Bull / High Vol",
    2: "Bear / Low Vol",
    3: "Bear / High Vol",
}

REGIME_COLORS = {
    0: "#1D9E75",   # teal — bull calm
    1: "#BA7517",   # amber — bull turbulent
    2: "#D85A30",   # coral — bear calm
    3: "#A32D2D",   # red — bear crisis
}


class RegimeDetector:
    def __init__(self, n_states: int = 4, random_state: int = 42):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, returns: pd.Series) -> "RegimeDetector":
        """Fit HMM on return + vol features."""
        features = self._build_features(returns)
        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        self._fitted = True
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        """Predict regime labels for each observation."""
        if not self._fitted:
            self.fit(returns)
        X = self.scaler.transform(self._build_features(returns))
        return self.model.predict(X)

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """Return posterior state probabilities per observation."""
        if not self._fitted:
            self.fit(returns)
        X = self.scaler.transform(self._build_features(returns))
        return self.model.predict_proba(X)

    def transition_matrix(self) -> np.ndarray:
        """Return the learned transition probability matrix."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.transmat_

    def current_regime(self, returns: pd.Series) -> dict:
        """Return current (latest) regime with label, color, and confidence."""
        labels = self.predict(returns)
        proba = self.predict_proba(returns)
        current = int(labels[-1])
        confidence = float(proba[-1, current])
        return {
            "regime_id": current,
            "label": REGIME_LABELS.get(current, f"State {current}"),
            "color": REGIME_COLORS.get(current, "#888"),
            "confidence": round(confidence, 3),
            "transition_probs": {
                REGIME_LABELS.get(j, f"State {j}"): round(float(self.model.transmat_[current, j]), 3)
                for j in range(self.n_states)
            },
        }

    @staticmethod
    def _build_features(returns: pd.Series) -> np.ndarray:
        r = returns.dropna()
        vol = r.rolling(20, min_periods=5).std().fillna(r.std())
        return np.column_stack([r.values, vol.values])


def detect_regimes(price_series: pd.Series, n_states: int = 4) -> pd.DataFrame:
    """
    Convenience function. Returns a DataFrame with:
      - Date index
      - Returns
      - Regime (int)
      - Regime_Label (str)
      - Regime_Color (str)
      - Confidence (float)
    """
    returns = price_series.pct_change().dropna()
    detector = RegimeDetector(n_states=n_states)
    labels = detector.predict(returns)
    proba = detector.predict_proba(returns)
    confidence = proba[np.arange(len(labels)), labels]

    df = pd.DataFrame(index=returns.index)
    df["Returns"] = returns.values
    df["Regime"] = labels
    df["Regime_Label"] = [REGIME_LABELS.get(r, f"State {r}") for r in labels]
    df["Regime_Color"] = [REGIME_COLORS.get(r, "#888") for r in labels]
    df["Confidence"] = confidence
    return df

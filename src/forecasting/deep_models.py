"""
Deep Learning Forecasting — LSTM and Transformer return forecasters.

Both models predict the cumulative return over `horizon_days` from a window
of `seq_len` lagged daily returns (same windowing idea as
`simple_return_forecast` in models.py). Uncertainty is estimated via
MC-Dropout: dropout stays active at inference time and the forecast is the
distribution over repeated stochastic forward passes.

Output dict shape matches `simple_return_forecast` so the dashboard can
compare GBM / LSTM / Transformer forecasts uniformly:
    {expected_return_pct, ci_low_pct, ci_high_pct, horizon_days, model_type, ...}
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn


def _build_sequences(returns: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of `seq_len` returns -> sum of the next `horizon` returns."""
    X, y = [], []
    for i in range(seq_len, len(returns) - horizon):
        X.append(returns[i - seq_len:i])
        y.append(returns[i:i + horizon].sum())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMForecaster(nn.Module):
    """Two-layer LSTM regressor over a sequence of daily returns."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    """Small Transformer encoder regressor over a sequence of daily returns."""

    def __init__(self, input_size: int = 1, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2, max_len: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(self.dropout(x)).squeeze(-1)


def _train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Train in-place with an 85/15 split. Returns (train_loss, val_loss) at the end."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    split = max(1, int(len(X_t) * 0.85))
    split = min(split, len(X_t) - 1) if len(X_t) > 1 else len(X_t)
    X_train, y_train = X_t[:split], y_t[:split]
    X_val, y_val = X_t[split:], y_t[split:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(X_train), y_train).item())
        val_loss = float(loss_fn(model(X_val), y_val).item()) if len(X_val) > 0 else train_loss

    return train_loss, val_loss


def deep_return_forecast(
    prices: pd.Series,
    model_type: str = "lstm",
    horizon_days: int = 20,
    seq_len: int = 30,
    epochs: int = 60,
    n_mc_samples: int = 40,
) -> dict:
    """
    Forecast cumulative return over `horizon_days` using an LSTM or Transformer,
    with MC-Dropout confidence intervals.

    Args:
        prices:       Close price series.
        model_type:   "lstm" | "transformer".
        horizon_days: Forecast horizon in trading days.
        seq_len:      Number of lagged daily returns fed to the model.
        epochs:       Training epochs (full retrain each call — keep modest for caching).
        n_mc_samples: Number of stochastic forward passes for the confidence interval.
    """
    returns = prices.pct_change().dropna()

    if len(returns) < seq_len + horizon_days + 10:
        return {
            "expected_return_pct": 0.0,
            "ci_low_pct": 0.0,
            "ci_high_pct": 0.0,
            "horizon_days": horizon_days,
            "model_type": model_type,
            "train_loss": 0.0,
            "val_loss": 0.0,
        }

    X, y = _build_sequences(returns.values, seq_len, horizon_days)
    mean, std = float(X.mean()), float(X.std())
    std = std if std > 0 else 1.0
    X_norm = (X - mean) / std

    torch.manual_seed(42)
    if model_type == "transformer":
        model: nn.Module = TransformerForecaster()
    else:
        model = LSTMForecaster()

    train_loss, val_loss = _train_model(model, X_norm[..., None], y, epochs=epochs)

    latest = returns.values[-seq_len:]
    latest_norm = (latest - mean) / std
    x_input = torch.tensor(latest_norm[None, :, None], dtype=torch.float32)

    model.train()  # keep dropout active for MC sampling
    with torch.no_grad():
        samples = np.array([model(x_input).item() for _ in range(n_mc_samples)])

    pred_mean = float(samples.mean())
    ci_low, ci_high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))

    return {
        "expected_return_pct": round(pred_mean * 100, 2),
        "ci_low_pct": round(ci_low * 100, 2),
        "ci_high_pct": round(ci_high * 100, 2),
        "horizon_days": horizon_days,
        "model_type": model_type,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
    }

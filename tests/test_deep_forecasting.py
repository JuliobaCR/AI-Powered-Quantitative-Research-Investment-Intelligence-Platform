"""Unit tests for the Deep Learning Forecasting module (Phase 7)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.forecasting.deep_models import _build_sequences, deep_return_forecast
from src.forecasting.monte_carlo import forecast_fan_percentiles, gbm_price_paths


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    n = 250
    returns = np.random.normal(0.0005, 0.012, n)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_deep_return_forecast_keys(sample_prices, model_type):
    result = deep_return_forecast(
        sample_prices, model_type=model_type, horizon_days=5, seq_len=10,
        epochs=5, n_mc_samples=5,
    )

    expected_keys = {
        "expected_return_pct", "ci_low_pct", "ci_high_pct",
        "horizon_days", "model_type", "train_loss", "val_loss",
    }
    assert expected_keys.issubset(result.keys())
    assert result["model_type"] == model_type
    assert result["horizon_days"] == 5
    assert result["ci_low_pct"] <= result["ci_high_pct"]
    for key in ("expected_return_pct", "ci_low_pct", "ci_high_pct", "train_loss", "val_loss"):
        assert np.isfinite(result[key])


def test_deep_return_forecast_short_series_returns_zero_dict():
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    prices = pd.Series(np.full(20, 100.0), index=idx)

    result = deep_return_forecast(prices, model_type="lstm", horizon_days=5, seq_len=10, epochs=5)

    assert result["expected_return_pct"] == 0.0
    assert result["ci_low_pct"] == 0.0
    assert result["ci_high_pct"] == 0.0
    assert result["model_type"] == "lstm"


def test_build_sequences_shapes():
    returns = np.arange(50, dtype=np.float32)
    X, y = _build_sequences(returns, seq_len=10, horizon=5)

    assert X.shape == (35, 10)
    assert y.shape == (35,)


def test_gbm_price_paths_shape():
    paths = gbm_price_paths(spot=100.0, mu=0.05, sigma=0.2, horizon_days=20, n_paths=15, seed=42)

    assert paths.shape == (15, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_forecast_fan_percentiles_shape():
    paths = gbm_price_paths(spot=100.0, mu=0.05, sigma=0.2, horizon_days=20, n_paths=15, seed=42)
    fan = forecast_fan_percentiles(paths)

    assert fan.shape == (21, 5)
    assert list(fan.columns) == ["p5", "p25", "p50", "p75", "p95"]

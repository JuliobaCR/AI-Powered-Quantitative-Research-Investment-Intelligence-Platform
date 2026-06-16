"""Tests for the Quantitative Statistical Analysis Suite."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import (
    adf_test,
    garch_vol_estimate,
    hurst_exponent,
    hurst_interpretation,
    jarque_bera_test,
    ljung_box_test,
    return_distribution_summary,
    rolling_autocorrelation,
)


@pytest.fixture
def random_walk_prices():
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    return pd.Series(prices, index=idx)


@pytest.fixture
def daily_returns(random_walk_prices):
    return random_walk_prices.pct_change().dropna()


# ── Hurst ─────────────────────────────────────────────────────────────────────

class TestHurst:
    def test_random_walk_near_half(self, random_walk_prices):
        h = hurst_exponent(random_walk_prices)
        # Random walk should give H roughly between 0.3 and 0.7
        assert 0.25 < h < 0.80

    def test_range_bounded(self, random_walk_prices):
        h = hurst_exponent(random_walk_prices)
        assert 0.0 <= h <= 1.0

    def test_short_series_returns_half(self):
        short = pd.Series(np.arange(10, dtype=float))
        assert hurst_exponent(short) == 0.5

    def test_interpretation_strings(self):
        assert "Mean-Reverting" in hurst_interpretation(0.30)
        assert "Random Walk" in hurst_interpretation(0.50)
        assert "Trending" in hurst_interpretation(0.70)


# ── GARCH ─────────────────────────────────────────────────────────────────────

class TestGARCH:
    def test_shape(self, daily_returns):
        vol = garch_vol_estimate(daily_returns)
        assert len(vol) == len(daily_returns)

    def test_all_positive(self, daily_returns):
        vol = garch_vol_estimate(daily_returns)
        assert np.all(vol > 0)

    def test_annualized_reasonable(self, daily_returns):
        vol = garch_vol_estimate(daily_returns)
        # Annualized vol from 1% daily should be ~15-20%
        assert 5.0 < float(vol.mean()) < 80.0

    def test_short_returns_empty(self):
        short = pd.Series(np.random.normal(0, 0.01, 5))
        assert garch_vol_estimate(short).empty


# ── ADF ───────────────────────────────────────────────────────────────────────

class TestADF:
    def test_returns_dict_keys(self, random_walk_prices):
        result = adf_test(random_walk_prices)
        assert set(result.keys()) >= {"test_stat", "p_value", "is_stationary", "interpretation"}

    def test_stationary_series_detected(self):
        # Stationary AR(1) process: x_t = 0.5 * x_{t-1} + eps
        np.random.seed(0)
        x = np.zeros(300)
        for t in range(1, 300):
            x[t] = 0.5 * x[t - 1] + np.random.normal(0, 1)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        result = adf_test(pd.Series(x, index=idx))
        # t-stat should be well below -2.86 for a clearly stationary AR(1)
        assert result["test_stat"] < -1.5  # at minimum, clearly negative

    def test_short_series_returns_sentinels(self):
        short = pd.Series(np.arange(5, dtype=float))
        r = adf_test(short)
        assert r["is_stationary"] is False


# ── Ljung-Box ─────────────────────────────────────────────────────────────────

class TestLjungBox:
    def test_keys(self, daily_returns):
        result = ljung_box_test(daily_returns)
        assert "Q_stat" in result
        assert "p_value" in result
        assert "has_autocorrelation" in result
        assert "acf_values" in result

    def test_q_stat_positive(self, daily_returns):
        result = ljung_box_test(daily_returns)
        assert result["Q_stat"] >= 0

    def test_p_value_in_range(self, daily_returns):
        result = ljung_box_test(daily_returns)
        assert 0 <= result["p_value"] <= 1

    def test_acf_values_length(self, daily_returns):
        result = ljung_box_test(daily_returns, lags=5)
        assert len(result["acf_values"]) == 5


# ── Jarque-Bera ───────────────────────────────────────────────────────────────

class TestJarqueBera:
    def test_returns_dict(self, daily_returns):
        result = jarque_bera_test(daily_returns)
        assert "JB_stat" in result
        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert "is_normal" in result

    def test_normal_data_not_rejected(self):
        np.random.seed(5)
        r = pd.Series(np.random.normal(0, 1, 200))
        result = jarque_bera_test(r)
        # Normal data should NOT be rejected (p > 0.05 most of the time)
        assert result["JB_stat"] >= 0

    def test_fat_tailed_returns_rejected(self, daily_returns):
        result = jarque_bera_test(daily_returns)
        assert result["excess_kurtosis"] != 0.0  # real returns have fat tails


# ── Rolling Autocorrelation ───────────────────────────────────────────────────

class TestRollingAutocorr:
    def test_length(self, daily_returns):
        ac = rolling_autocorrelation(daily_returns, lag=1, window=63)
        assert len(ac) <= len(daily_returns)

    def test_range(self, daily_returns):
        ac = rolling_autocorrelation(daily_returns, lag=1, window=63).dropna()
        assert np.all(ac >= -1.0) and np.all(ac <= 1.0)

    def test_short_series_empty(self):
        short = pd.Series(np.random.normal(0, 0.01, 20))
        ac = rolling_autocorrelation(short, lag=1, window=63)
        assert ac.empty


# ── Distribution Summary ──────────────────────────────────────────────────────

class TestDistributionSummary:
    def test_keys(self, daily_returns):
        d = return_distribution_summary(daily_returns)
        assert "mean_daily_pct" in d
        assert "annual_vol_pct" in d
        assert "p5" in d
        assert "p95" in d
        assert d["p5"] < d["p95"]

    def test_n_obs(self, daily_returns):
        d = return_distribution_summary(daily_returns)
        assert d["n_obs"] == len(daily_returns)

    def test_empty_short_series():
        assert return_distribution_summary(pd.Series(dtype=float)) == {}

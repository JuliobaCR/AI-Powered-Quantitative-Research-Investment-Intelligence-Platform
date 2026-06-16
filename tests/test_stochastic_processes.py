"""Tests for Heston, Merton Jump-Diffusion, and CEV stochastic processes."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.stochastic_processes import (
    cev_price_paths,
    fit_heston_params,
    fit_jump_params,
    heston_price_paths,
    jump_diffusion_price_paths,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    n = 252
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    r = np.random.normal(0.0005, 0.015, n)
    return pd.Series(r, index=idx)


# ── Heston ────────────────────────────────────────────────────────────────────

class TestHeston:
    def test_shape(self):
        paths = heston_price_paths(100.0, 0.05, v0=0.04, kappa=2.0,
                                    theta=0.04, xi=0.3, rho=-0.7,
                                    horizon_days=20, n_paths=15)
        assert paths.shape == (15, 21)

    def test_initial_price(self):
        paths = heston_price_paths(150.0, 0.05, v0=0.04, kappa=2.0,
                                    theta=0.04, xi=0.3, rho=-0.7,
                                    horizon_days=10, n_paths=10)
        assert np.all(paths[:, 0] == 150.0)

    def test_all_positive(self):
        paths = heston_price_paths(100.0, 0.08, v0=0.04, kappa=2.0,
                                    theta=0.04, xi=0.5, rho=-0.5,
                                    horizon_days=30, n_paths=50, seed=0)
        assert np.all(paths > 0)

    def test_reproducible_with_seed(self):
        p1 = heston_price_paths(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7,
                                 horizon_days=10, n_paths=5, seed=99)
        p2 = heston_price_paths(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7,
                                 horizon_days=10, n_paths=5, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_fit_params_keys(self, sample_returns):
        params = fit_heston_params(sample_returns)
        for key in ("v0", "kappa", "theta", "xi", "rho"):
            assert key in params
        assert 0 < params["v0"] < 1
        assert -1 <= params["rho"] <= 0  # leverage effect: rho < 0


# ── Merton Jump-Diffusion ─────────────────────────────────────────────────────

class TestJumpDiffusion:
    def test_shape(self):
        paths = jump_diffusion_price_paths(100.0, 0.07, 0.20,
                                            horizon_days=25, n_paths=12)
        assert paths.shape == (12, 26)

    def test_initial_price(self):
        paths = jump_diffusion_price_paths(200.0, 0.05, 0.18,
                                            horizon_days=10, n_paths=8)
        assert np.all(paths[:, 0] == 200.0)

    def test_all_positive(self):
        paths = jump_diffusion_price_paths(100.0, 0.05, 0.20,
                                            lam=0.5, mu_j=-0.10, sigma_j=0.15,
                                            horizon_days=30, n_paths=100, seed=42)
        assert np.all(paths > 0)

    def test_fit_params_keys(self, sample_returns):
        params = fit_jump_params(sample_returns)
        for key in ("sigma", "lam", "mu_j", "sigma_j"):
            assert key in params
        assert params["sigma"] > 0
        assert params["lam"] > 0
        assert params["sigma_j"] > 0


# ── CEV ───────────────────────────────────────────────────────────────────────

class TestCEV:
    def test_shape(self):
        paths = cev_price_paths(100.0, 0.05, 0.20, beta=0.5,
                                 horizon_days=20, n_paths=10)
        assert paths.shape == (10, 21)

    def test_initial_price(self):
        paths = cev_price_paths(50.0, 0.05, 0.20, beta=0.5,
                                 horizon_days=10, n_paths=5)
        assert np.all(paths[:, 0] == 50.0)

    def test_gbm_special_case_comparable(self):
        # CEV with beta=1 should produce similar distribution to GBM
        from src.forecasting.monte_carlo import gbm_price_paths
        cev = cev_price_paths(100.0, 0.05, 0.20, beta=1.0,
                               horizon_days=252, n_paths=1000, seed=7)
        gbm = gbm_price_paths(100.0, 0.05, 0.20, 252, 1000, seed=7)
        # Medians should be in the same ballpark (within 20%)
        assert abs(np.median(cev[:, -1]) - np.median(gbm[:, -1])) / 100.0 < 0.20

    def test_short_series_fit(self):
        short = pd.Series(np.random.normal(0, 0.01, 10))
        params = fit_heston_params(short)  # should return defaults
        assert params["v0"] > 0

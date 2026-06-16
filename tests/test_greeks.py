"""Unit tests for the Options Greeks engine."""

import numpy as np

from src.derivatives.greeks import black_scholes_greeks, greeks_surface


def test_call_delta_range():
    g = black_scholes_greeks(S=100, K=100, T=0.5, r=0.0525, sigma=0.25, option_type="call")
    assert 0 <= g["delta"] <= 1


def test_put_delta_range():
    g = black_scholes_greeks(S=100, K=100, T=0.5, r=0.0525, sigma=0.25, option_type="put")
    assert -1 <= g["delta"] <= 0


def test_gamma_non_negative():
    g = black_scholes_greeks(S=100, K=100, T=0.5, r=0.0525, sigma=0.25, option_type="call")
    assert g["gamma"] >= 0


def test_greeks_surface_shapes():
    surf = greeks_surface(spot=100, sigma=0.30, option_type="call",
                           n_strikes=10, expirations_days=(7, 30, 90))
    expected_shape = (3, 10)
    for key in ["price", "delta", "gamma", "vega", "theta", "rho"]:
        assert surf[key].shape == expected_shape
    assert len(surf["strikes"]) == 10
    assert len(surf["expirations"]) == 3


def test_greeks_surface_delta_bounds_call():
    surf = greeks_surface(spot=100, sigma=0.30, option_type="call",
                           n_strikes=10, expirations_days=(7, 30, 90))
    assert np.all(surf["delta"] >= 0) and np.all(surf["delta"] <= 1)


def test_greeks_surface_delta_bounds_put():
    surf = greeks_surface(spot=100, sigma=0.30, option_type="put",
                           n_strikes=10, expirations_days=(7, 30, 90))
    assert np.all(surf["delta"] >= -1) and np.all(surf["delta"] <= 0)

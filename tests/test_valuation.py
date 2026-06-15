"""Unit tests for the Valuation Engine."""

import pytest
from src.valuation.engine import _valuation_score, _dcf_estimate


def test_valuation_score_range():
    score = _valuation_score(pe=20, peg=0.9, ev_ebitda=15, ps=5, pb=3, mos=10)
    assert 0 <= score <= 100


def test_dcf_returns_zero_on_no_fcf():
    info = {"freeCashflow": 0, "sharesOutstanding": 1000}
    val, conf = _dcf_estimate(info)
    assert val == 0.0
    assert conf == 0.0

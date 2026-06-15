"""Unit tests for the Alpha Score system."""

from src.explainability.alpha_score import compute_alpha_score


def test_score_range():
    result = compute_alpha_score("TEST", 70, 80, 60, 55, 50, 65)
    assert 0 <= result.total_score <= 100


def test_buy_verdict():
    result = compute_alpha_score("TEST", 90, 85, 80, 75, 70, 80)
    assert result.verdict == "BUY"


def test_sell_verdict():
    result = compute_alpha_score("TEST", 10, 15, 20, 15, 10, 20)
    assert result.verdict == "SELL"


def test_reasoning_not_empty():
    result = compute_alpha_score("TEST", 70, 70, 70, 70, 70, 70)
    assert len(result.reasoning) > 0

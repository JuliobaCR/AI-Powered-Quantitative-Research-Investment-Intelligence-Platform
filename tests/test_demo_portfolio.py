"""Tests for the paper-trading DemoPortfolio engine."""

import json
import pathlib
import tempfile
import pytest

# Monkey-patch the portfolio file path before importing the module
import src.portfolio.demo as _demo_module


@pytest.fixture(autouse=True)
def tmp_portfolio_file(tmp_path, monkeypatch):
    """Redirect PORTFOLIO_FILE to a temp path so tests don't touch data/."""
    monkeypatch.setattr(_demo_module, "PORTFOLIO_FILE", tmp_path / "demo_portfolio.json")
    yield


@pytest.fixture
def fresh():
    """A brand-new DemoPortfolio with $100,000 cash and no positions."""
    from src.portfolio.demo import DemoPortfolio
    return DemoPortfolio()


# ── Initialization ────────────────────────────────────────────────────────────

class TestInitialization:
    def test_initial_cash(self, fresh):
        assert fresh.cash == 100_000.0

    def test_no_positions(self, fresh):
        assert fresh.positions == {}

    def test_persists_to_disk(self, fresh, tmp_path):
        # The file should exist after construction (created on first _load)
        # A buy should write it
        fresh.buy("AAPL", 1, 150.0)
        assert (tmp_path / "demo_portfolio.json").exists()


# ── Buy ───────────────────────────────────────────────────────────────────────

class TestBuy:
    def test_basic_buy_succeeds(self, fresh):
        ok, msg = fresh.buy("AAPL", 10, 150.0)
        assert ok
        assert "AAPL" in msg

    def test_cash_deducted(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        assert fresh.cash == pytest.approx(100_000.0 - 1_500.0)

    def test_position_created(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        assert "AAPL" in fresh.positions
        assert fresh.positions["AAPL"]["shares"] == pytest.approx(10.0)

    def test_avg_cost_correct(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        assert fresh.positions["AAPL"]["avg_cost"] == pytest.approx(150.0)

    def test_second_buy_updates_avg_cost(self, fresh):
        fresh.buy("AAPL", 10, 100.0)  # cost 1000
        fresh.buy("AAPL", 10, 200.0)  # cost 2000 → avg 150
        assert fresh.positions["AAPL"]["avg_cost"] == pytest.approx(150.0)
        assert fresh.positions["AAPL"]["shares"] == pytest.approx(20.0)

    def test_ticker_uppercased(self, fresh):
        fresh.buy("aapl", 1, 100.0)
        assert "AAPL" in fresh.positions

    def test_insufficient_cash_fails(self, fresh):
        ok, msg = fresh.buy("AAPL", 10_000, 200.0)
        assert not ok
        assert "Insufficient" in msg

    def test_zero_shares_fails(self, fresh):
        ok, msg = fresh.buy("AAPL", 0, 100.0)
        assert not ok

    def test_zero_price_fails(self, fresh):
        ok, msg = fresh.buy("AAPL", 1, 0.0)
        assert not ok

    def test_buy_records_trade(self, fresh):
        fresh.buy("AAPL", 5, 120.0)
        trades = fresh.get_trades()
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"
        assert trades[0]["ticker"] == "AAPL"


# ── Sell ──────────────────────────────────────────────────────────────────────

class TestSell:
    def test_basic_sell_succeeds(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        ok, msg = fresh.sell("AAPL", 5, 160.0)
        assert ok
        assert "AAPL" in msg

    def test_cash_restored_after_sell(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        fresh.sell("AAPL", 5, 160.0)
        expected_cash = 100_000.0 - 1_500.0 + 5 * 160.0
        assert fresh.cash == pytest.approx(expected_cash)

    def test_shares_reduced(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        fresh.sell("AAPL", 4, 160.0)
        assert fresh.positions["AAPL"]["shares"] == pytest.approx(6.0)

    def test_full_sell_removes_position(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        fresh.sell("AAPL", 10, 160.0)
        assert "AAPL" not in fresh.positions

    def test_sell_nonexistent_fails(self, fresh):
        ok, msg = fresh.sell("MSFT", 1, 100.0)
        assert not ok
        assert "MSFT" in msg

    def test_oversell_fails(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        ok, msg = fresh.sell("AAPL", 100, 160.0)
        assert not ok

    def test_sell_records_trade(self, fresh):
        fresh.buy("AAPL", 10, 150.0)
        fresh.sell("AAPL", 3, 160.0)
        trades = fresh.get_trades()
        assert trades[0]["action"] == "SELL"  # most recent first


# ── Snapshot ──────────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_no_positions(self, fresh):
        snap = fresh.get_snapshot({})
        assert snap["cash"] == pytest.approx(100_000.0)
        assert snap["n_positions"] == 0
        assert snap["total_value"] == pytest.approx(100_000.0)
        assert snap["total_return"] == pytest.approx(0.0)

    def test_snapshot_with_gain(self, fresh):
        fresh.buy("AAPL", 10, 100.0)  # cost 1000
        snap = fresh.get_snapshot({"AAPL": 120.0})  # now worth 1200
        assert snap["unrealized_pnl"] == pytest.approx(200.0)
        assert snap["total_value"] == pytest.approx(100_000.0 + 200.0)

    def test_snapshot_with_loss(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        snap = fresh.get_snapshot({"AAPL": 80.0})
        assert snap["unrealized_pnl"] == pytest.approx(-200.0)

    def test_snapshot_missing_price_uses_avg_cost(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        snap = fresh.get_snapshot({})  # no price provided
        # Fallback to avg_cost → no P&L
        assert snap["unrealized_pnl"] == pytest.approx(0.0)

    def test_snapshot_weights_sum_to_100(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        fresh.buy("MSFT", 5, 200.0)
        snap = fresh.get_snapshot({"AAPL": 100.0, "MSFT": 200.0})
        weights = [r["weight_pct"] for r in snap["positions"]]
        cash_pct = snap["cash_pct"]
        assert sum(weights) + cash_pct == pytest.approx(100.0, abs=0.5)


# ── Trade history ─────────────────────────────────────────────────────────────

class TestHistory:
    def test_most_recent_first(self, fresh):
        fresh.buy("AAPL", 1, 100.0)
        fresh.buy("MSFT", 1, 200.0)
        trades = fresh.get_trades()
        assert trades[0]["ticker"] == "MSFT"
        assert trades[1]["ticker"] == "AAPL"

    def test_limit_n(self, fresh):
        for _ in range(5):
            fresh.buy("AAPL", 1, 100.0)
        assert len(fresh.get_trades(n=3)) == 3

    def test_no_trades_returns_empty(self, fresh):
        assert fresh.get_trades() == []


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_restores_cash(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        fresh.reset()
        assert fresh.cash == 100_000.0

    def test_reset_clears_positions(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        fresh.reset()
        assert fresh.positions == {}

    def test_reset_clears_trades(self, fresh):
        fresh.buy("AAPL", 10, 100.0)
        fresh.reset()
        assert fresh.get_trades() == []


# ── Persistence (round-trip JSON) ─────────────────────────────────────────────

class TestPersistence:
    def test_state_survives_reload(self, fresh, tmp_path):
        from src.portfolio.demo import DemoPortfolio
        fresh.buy("AAPL", 7, 150.0)
        # Instantiate a second portfolio object — reads same JSON file
        fresh2 = DemoPortfolio()
        assert "AAPL" in fresh2.positions
        assert fresh2.positions["AAPL"]["shares"] == pytest.approx(7.0)
        assert fresh2.cash == pytest.approx(fresh.cash)

    def test_corrupted_json_recovers(self, tmp_path, monkeypatch):
        from src.portfolio.demo import DemoPortfolio, INITIAL_CASH
        pf = tmp_path / "demo_portfolio.json"
        pf.write_text("{ CORRUPTED }}", encoding="utf-8")
        portfolio = DemoPortfolio()
        assert portfolio.cash == INITIAL_CASH

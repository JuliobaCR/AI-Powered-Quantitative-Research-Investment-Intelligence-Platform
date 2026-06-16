"""
Demo Portfolio — Paper Trading Engine.

Simulates a $100,000 paper-trading account with persistent state across
Streamlit sessions. Positions, cash, and trade history are stored in
data/demo_portfolio.json. All operations (buy/sell/snapshot) read and
write atomically to the file so the state survives browser refreshes
and Streamlit reruns.

Design contract:
  - DemoPortfolio.buy()  → (success: bool, message: str)
  - DemoPortfolio.sell() → (success: bool, message: str)
  - DemoPortfolio.get_snapshot(current_prices) → dict with all metrics
  - DemoPortfolio.get_trades() → list of dicts (most recent first)
  - DemoPortfolio.reset() → wipe all positions, restore $100k cash

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

# Resolve relative to project root regardless of cwd
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PORTFOLIO_FILE = _PROJECT_ROOT / "data" / "demo_portfolio.json"

INITIAL_CASH: float = 100_000.0


class DemoPortfolio:
    """
    Paper-trading portfolio with JSON persistence.

    Thread / rerun safety: Streamlit reruns the entire script on every
    interaction, so we always read from disk at __init__ and write on
    every mutation. This makes every operation atomic from the user's
    perspective.
    """

    def __init__(self) -> None:
        self._data = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        if PORTFOLIO_FILE.exists():
            try:
                with open(PORTFOLIO_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted file → reset to defaults
        return {
            "cash": INITIAL_CASH,
            "positions": {},   # ticker → {shares, avg_cost, total_cost}
            "trades": [],      # list of trade dicts
            "created_at": datetime.now().isoformat(),
        }

    def _save(self) -> None:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return float(self._data["cash"])

    @property
    def positions(self) -> dict:
        return self._data["positions"]

    # ── Trading operations ────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: float, price: float) -> tuple[bool, str]:
        """
        Execute a paper buy order.

        Returns (True, confirmation_msg) on success,
                (False, error_msg) if insufficient cash or invalid input.
        """
        ticker = ticker.upper().strip()
        if shares <= 0 or price <= 0:
            return False, "Shares and price must be positive."

        cost = round(shares * price, 2)
        if cost > self.cash:
            return False, (
                f"Insufficient cash. Need ${cost:,.2f} but only ${self.cash:,.2f} available."
            )

        # Update cash
        self._data["cash"] = round(self.cash - cost, 2)

        # Update or create position
        pos = self._data["positions"]
        if ticker in pos:
            old_shares = float(pos[ticker]["shares"])
            old_cost = float(pos[ticker]["total_cost"])
            new_shares = old_shares + shares
            new_cost = old_cost + cost
            pos[ticker] = {
                "shares": round(new_shares, 6),
                "avg_cost": round(new_cost / new_shares, 4),
                "total_cost": round(new_cost, 2),
            }
        else:
            pos[ticker] = {
                "shares": round(shares, 6),
                "avg_cost": round(price, 4),
                "total_cost": round(cost, 2),
            }

        # Record trade
        self._data["trades"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "action": "BUY",
            "shares": round(shares, 4),
            "price": round(price, 2),
            "total": cost,
            "cash_after": self.cash,
        })

        self._save()
        return True, f"✅ Bought {shares:g} × {ticker} @ ${price:.2f} — Total: ${cost:,.2f}"

    def sell(self, ticker: str, shares: float, price: float) -> tuple[bool, str]:
        """
        Execute a paper sell order.

        Returns (True, confirmation_msg) on success,
                (False, error_msg) if position doesn't exist or shares exceed holding.
        """
        ticker = ticker.upper().strip()
        if shares <= 0 or price <= 0:
            return False, "Shares and price must be positive."

        pos = self._data["positions"]
        if ticker not in pos:
            return False, f"{ticker} is not in your portfolio."

        held = float(pos[ticker]["shares"])
        if shares > held + 1e-6:
            return False, (
                f"Cannot sell {shares:g} shares — only {held:g} held."
            )

        proceeds = round(shares * price, 2)
        self._data["cash"] = round(self.cash + proceeds, 2)

        remaining = held - shares
        if remaining < 1e-6:
            del pos[ticker]
        else:
            pos[ticker]["shares"] = round(remaining, 6)
            pos[ticker]["total_cost"] = round(remaining * float(pos[ticker]["avg_cost"]), 2)

        self._data["trades"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "action": "SELL",
            "shares": round(shares, 4),
            "price": round(price, 2),
            "total": proceeds,
            "cash_after": self.cash,
        })

        self._save()
        return True, f"✅ Sold {shares:g} × {ticker} @ ${price:.2f} — Proceeds: ${proceeds:,.2f}"

    # ── Portfolio analytics ───────────────────────────────────────────────────

    def get_snapshot(self, current_prices: dict[str, float]) -> dict:
        """
        Return a complete portfolio snapshot using current market prices.

        current_prices: dict {ticker: price} — any missing ticker falls back
                        to the average cost (shows break-even instead of P&L).
        """
        pos = self._data["positions"]
        rows = []
        total_market_value = 0.0
        total_cost_basis = 0.0

        for ticker, data in pos.items():
            shares = float(data["shares"])
            avg_cost = float(data["avg_cost"])
            cost_basis = float(data["total_cost"])

            price = current_prices.get(ticker, avg_cost)
            market_value = shares * price
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            weight = 0.0  # computed after totals

            rows.append({
                "ticker": ticker,
                "shares": round(shares, 4),
                "avg_cost": round(avg_cost, 2),
                "current_price": round(price, 2),
                "cost_basis": round(cost_basis, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            })
            total_market_value += market_value
            total_cost_basis += cost_basis

        # Compute weights
        total_incl_cash = total_market_value + self.cash
        for row in rows:
            row["weight_pct"] = round(row["market_value"] / total_incl_cash * 100, 1) if total_incl_cash > 0 else 0.0

        total_value = self.cash + total_market_value
        total_return = total_value - INITIAL_CASH
        total_return_pct = (total_return / INITIAL_CASH * 100) if INITIAL_CASH > 0 else 0.0
        unrealized_total = total_market_value - total_cost_basis

        return {
            "cash": round(self.cash, 2),
            "cash_pct": round(self.cash / total_incl_cash * 100, 1) if total_incl_cash > 0 else 100.0,
            "total_market_value": round(total_market_value, 2),
            "total_value": round(total_value, 2),
            "initial_capital": INITIAL_CASH,
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "unrealized_pnl": round(unrealized_total, 2),
            "unrealized_pnl_pct": round(unrealized_total / total_cost_basis * 100, 2) if total_cost_basis > 0 else 0.0,
            "n_positions": len(rows),
            "positions": rows,
        }

    def get_trades(self, n: int | None = None) -> list[dict]:
        """Return trades list, most recent first. n limits the count."""
        trades = list(reversed(self._data["trades"]))
        return trades[:n] if n else trades

    def reset(self) -> None:
        """Wipe all positions and trades, restore initial cash."""
        self._data = {
            "cash": INITIAL_CASH,
            "positions": {},
            "trades": [],
            "created_at": datetime.now().isoformat(),
        }
        self._save()

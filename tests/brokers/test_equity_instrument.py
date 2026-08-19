"""Equity instrument support through the SAME ledger, costs and risk gates.

The seam that makes the platform equities-capable: instrument-typed legs with
multiplier-aware money math. The options path must be bit-identical to before
(multiplier 100 everywhere a literal 100 used to be), and the equity path must
obey the same honesty rules — worse-side fills, never better than mid, $0
equity commissions, shares never expiring.
"""

from datetime import date, datetime

import pytest

from catalyst.core.config import load_config
from catalyst.core.types import (
    AccountState, Direction, EquityKey, ExitRules, Order, OrderIntent, OrderLeg,
    OrderStatus, ProposedTrade, Side,
)
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.costs.model import BetterThanNBBOError, NBBOCostModel, ZeroCostModel
from catalyst.risk.manager import RiskManager

CFG = load_config("backtest")
NOW = datetime(2025, 6, 2, 10, 30)


def _broker(cash=100_000.0):
    b = SimulatedBroker(fill_model=CFG.execution.fill_model,
                        commissions=CFG.execution.commissions, starting_cash=cash)
    b.update_market({}, NOW, equity_quotes={"SPY": (599.98, 600.02)})
    return b


def _order(side, qty, intent=OrderIntent.OPEN, position_id=None):
    return Order(legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=side, qty=qty)],
                 limit_price=600.0, intent=intent,
                 direction=Direction.LONG, exit_rules=ExitRules(),
                 position_id=position_id, tag="eq:test")


class TestEquityLedger:
    def test_buy_shares_costs_price_times_shares_no_commission(self):
        b = _broker()
        r = b.place_order(_order(Side.BUY, 100))
        assert r.status is OrderStatus.FILLED
        assert r.commission == 0.0, "US equities are commission-free"
        spent = 100_000.0 - b.cash
        # never better than mid, never worse than ask + slippage allowance
        assert 100 * 600.00 <= spent <= 100 * 600.30

    def test_round_trip_loses_only_spread_and_slippage(self):
        b = _broker()
        r = b.place_order(_order(Side.BUY, 100))
        pos = b.get_positions()[0]
        b.place_order(_order(Side.SELL, 100, intent=OrderIntent.CLOSE,
                             position_id=pos.position_id))
        loss = 100_000.0 - b.cash
        # 60% crossing of a 4bp-wide spread + 3.5bps/side on ~$60k notional:
        # bounded, small, and strictly positive — a free round trip would mean
        # the cost model leaked.
        assert 0 < loss < 100.0, f"round-trip friction {loss:.2f} out of bounds"

    def test_equity_position_marks_from_quotes_and_never_expires(self):
        b = _broker()
        b.place_order(_order(Side.BUY, 100))
        # move the market up 10 and jump a year ahead: no expiry settlement
        b.update_market({}, datetime(2026, 6, 2, 10, 30),
                        equity_quotes={"SPY": (609.98, 610.02)})
        pos = b.get_positions()[0]
        assert pos.current_value == pytest.approx(610.0, abs=0.05)
        assert pos.multiplier == 1.0
        assert len(b.get_positions()) == 1, "shares must never expire-settle"
        assert pos.unrealized_pnl == pytest.approx((610.0 - pos.entry_price) * 100, rel=0.01)

    def test_equity_account_equity_uses_share_multiplier(self):
        b = _broker()
        b.place_order(_order(Side.BUY, 100))
        acct = b.get_account()
        assert acct.equity == pytest.approx(100_000.0, abs=100.0), \
            "cash + marked shares should ~= starting cash right after the fill"


class TestEquityCostHonesty:
    def test_buy_never_better_than_mid_sell_never_better_than_mid(self):
        m = NBBOCostModel(CFG.execution.fill_model, CFG.execution.commissions)
        buy = m.equity_fill(99.98, 100.02, Side.BUY, 100)
        sell = m.equity_fill(99.98, 100.02, Side.SELL, 100)
        assert buy.price >= 100.0 and sell.price <= 100.0
        assert buy.commission == 0.0 and sell.commission == 0.0

    def test_zero_cost_twin_fills_at_mid(self):
        z = ZeroCostModel()
        assert z.equity_fill(99.98, 100.02, Side.BUY, 100).price == pytest.approx(100.0)

    def test_unusable_quote_raises(self):
        m = NBBOCostModel(CFG.execution.fill_model, CFG.execution.commissions)
        with pytest.raises(ValueError):
            m.equity_fill(0.0, 100.0, Side.BUY, 100)


class TestEquityRiskSizing:
    def test_risk_manager_sizes_shares_at_1x_multiplier(self):
        """A $5 defined-risk stop on shares must size ~100x more units than a
        $5 option premium — the multiplier bug this test pins."""
        rm = RiskManager(CFG.risk.model_copy(update={"cash_floor_fraction": 0.05,
                                                     "max_deployed": 0.95}))
        acct = AccountState(equity=100_000, cash=100_000, buying_power=100_000,
                            timestamp=NOW)
        shares = ProposedTrade(
            engine="t", catalyst_ref="x",
            legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=1)],
            unit_cost=600.0, unit_max_loss=5.0, direction=Direction.LONG,
            exit_rules=ExitRules(), per_trade_risk_fraction=0.01)
        d = rm.size_entry(shares, acct, [])
        # risk budget $1,000 / ($5 x 1.05 buffer x multiplier 1) ~= 190 shares,
        # then trimmed by the cash-cost check (600/share) to ~150.
        assert d.units > 100, f"sized {d.units} — multiplier not applied as 1x"

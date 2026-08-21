"""SimulatedBroker integrity regressions from the v15 audit:
D-006 (per-leg settlement), D-007/D-114 (strict close matching),
D-037/D-039 (NaN marks/equity fills), D-040 (zero-bid credit),
D-041/D-113 (credit collateral), D-043/D-044 (crossed quotes),
D-111/D-038 (zero spot), D-199 (force_close accounting)."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest

from catalyst.brokers.simulated import SimulatedBroker
from catalyst.core.config import CommissionsConfig, FillModelConfig
from catalyst.core.types import (
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    Order,
    OrderIntent,
    OrderLeg,
    OrderStatus,
    Side,
)

NOW = datetime(2024, 6, 3, 15, 45)
FRONT = date(2024, 6, 7)
BACK = date(2024, 7, 19)
KF = OptionKey(underlying="SPY", expiry=FRONT, right=OptionRight.CALL, strike=530.0)
KB = OptionKey(underlying="SPY", expiry=BACK, right=OptionRight.CALL, strike=530.0)
COMM = CommissionsConfig(alpaca_per_contract=0.0,
                         schwab_per_contract_per_leg=0.65,
                         active_profile="alpaca")


def chain(spot=525.0, when=NOW, contracts=None) -> OptionChain:
    return OptionChain(underlying="SPY", underlying_price=spot, timestamp=when,
                       contracts=contracts if contracts is not None else [
                           OptionContract(key=KF, bid=2.00, ask=2.20),
                           OptionContract(key=KB, bid=6.00, ask=6.40)])


def broker(cash=100_000.0) -> SimulatedBroker:
    b = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=0.6,
                                   slippage_pct_of_premium=0.0),
        commissions=COMM, starting_cash=cash)
    b.update_market({"SPY": chain()}, NOW)
    return b


def open_calendar(b) -> str:
    """Short front / long back calendar (a debit)."""
    order = Order(legs=[OrderLeg(key=KF, side=Side.SELL, qty=1),
                        OrderLeg(key=KB, side=Side.BUY, qty=1)],
                  intent=OrderIntent.OPEN, limit_price=4.5, tag="t:cal")
    r = b.place_order(order)
    assert r.status is OrderStatus.FILLED
    return b.get_positions()[0].position_id


class TestPerLegSettlement:
    def test_calendar_front_leg_settles_back_leg_survives(self):
        b = broker()
        pid = open_calendar(b)
        # advance past FRONT expiry only; back month keeps quoting
        later = datetime.combine(FRONT + timedelta(days=3), NOW.time())
        b.update_market({"SPY": chain(when=later, contracts=[
            OptionContract(key=KB, bid=7.00, ask=7.40)])}, later)
        pos = b.get_positions()
        assert len(pos) == 1, "back-month leg must SURVIVE front expiry (D-006)"
        assert [leg.key for leg in pos[0].legs] == [KB]
        assert b.partial_settlements and b.partial_settlements[0][0] == pid
        # short front call, spot 525 < 530 -> expires worthless -> 0 settle cash
        assert b.partial_settlements[0][1] == pytest.approx(0.0)

    def test_zero_spot_never_settles_puts_at_strike(self):
        b = broker()
        order = Order(legs=[OrderLeg(key=OptionKey(underlying="SPY", expiry=FRONT,
                                                   right=OptionRight.PUT, strike=520.0),
                                     side=Side.BUY, qty=1)],
                      intent=OrderIntent.OPEN, limit_price=2.5, tag="t:p")
        put_key = order.legs[0].key
        b.update_market({"SPY": chain(contracts=[
            OptionContract(key=put_key, bid=2.00, ask=2.20)])}, NOW)
        assert b.place_order(order).status is OrderStatus.FILLED
        mark_before = b.get_positions()[0].current_value
        later = datetime.combine(FRONT + timedelta(days=3), NOW.time())
        # chain with spot 0.0 (the D-111 poison): a fully-expired structure
        # falls back to LAST MARK (loud), never to strike-vs-zero intrinsic
        b.update_market({"SPY": chain(spot=0.0, when=later, contracts=[])}, later)
        assert b.settlements, "expired position settles (at last mark)"
        _, value, _ = b.settlements[0]
        assert value == pytest.approx(mark_before)
        assert value < 100, "a 520-strike put must NOT settle at ~520 from spot 0"


class TestStrictCloseMatching:
    def _open_spread(self, b):
        order = Order(legs=[OrderLeg(key=KF, side=Side.BUY, qty=1),
                            OrderLeg(key=KB, side=Side.SELL, qty=1)],
                      intent=OrderIntent.OPEN, limit_price=-3.5, tag="t:s",
                      max_loss=5.0)
        r = b.place_order(order)
        assert r.status is OrderStatus.FILLED
        return b.get_positions()[0]

    def test_correct_mirror_close_fills(self):
        b = broker()
        pos = self._open_spread(b)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KF, side=Side.SELL, qty=1),
                  OrderLeg(key=KB, side=Side.BUY, qty=1)],
            intent=OrderIntent.CLOSE, position_id=pos.position_id,
            limit_price=0.0, tag="t:c"))
        assert r.status is OrderStatus.FILLED
        assert b.get_positions() == []

    def test_same_side_close_rejected(self):
        b = broker()
        pos = self._open_spread(b)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KF, side=Side.BUY, qty=1),      # NOT inverted
                  OrderLeg(key=KB, side=Side.BUY, qty=1)],
            intent=OrderIntent.CLOSE, position_id=pos.position_id,
            limit_price=0.0, tag="t:c"))
        assert r.status is OrderStatus.REJECTED
        assert len(b.get_positions()) == 1                     # untouched

    def test_wrong_strike_close_rejected(self):
        b = broker()
        pos = self._open_spread(b)
        other = OptionKey(underlying="SPY", expiry=FRONT,
                          right=OptionRight.CALL, strike=999.0)
        r = b.place_order(Order(
            legs=[OrderLeg(key=other, side=Side.SELL, qty=1),
                  OrderLeg(key=KB, side=Side.BUY, qty=1)],
            intent=OrderIntent.CLOSE, position_id=pos.position_id,
            limit_price=0.0, tag="t:c"))
        assert r.status is OrderStatus.REJECTED

    def test_single_leg_close_of_spread_rejected(self):
        b = broker()
        pos = self._open_spread(b)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KF, side=Side.SELL, qty=1)],
            intent=OrderIntent.CLOSE, position_id=pos.position_id,
            limit_price=0.0, tag="t:c"))
        assert r.status is OrderStatus.REJECTED


class TestCreditCollateral:
    def test_credit_open_without_max_loss_rejected(self):
        b = broker()
        r = b.place_order(Order(
            legs=[OrderLeg(key=KB, side=Side.SELL, qty=1),
                  OrderLeg(key=KF, side=Side.BUY, qty=1)],
            intent=OrderIntent.OPEN, limit_price=-3.5, tag="t:cr"))
        assert r.status is OrderStatus.REJECTED
        assert "collateralize" in r.message

    def test_credit_open_reserves_collateral(self):
        b = broker()
        r = b.place_order(Order(
            legs=[OrderLeg(key=KB, side=Side.SELL, qty=1),
                  OrderLeg(key=KF, side=Side.BUY, qty=1)],
            intent=OrderIntent.OPEN, limit_price=-3.5, tag="t:cr", max_loss=10.0))
        assert r.status is OrderStatus.FILLED
        assert b.reserved_collateral == pytest.approx(10.0 * 100)
        assert b.get_account().buying_power < b.cash

    def test_credit_selling_is_bounded_by_collateral(self):
        """Before D-041 a credit open ADDED cash, so infinite size passed the
        broker guard. Now collateral binds."""
        b = broker(cash=1_000.0)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KB, side=Side.SELL, qty=1),
                  OrderLeg(key=KF, side=Side.BUY, qty=1)],
            intent=OrderIntent.OPEN, limit_price=-3.5, tag="t:cr",
            max_loss=50.0))   # $5,000 collateral vs $1,000 cash
        assert r.status is OrderStatus.REJECTED

    def test_collateral_released_on_close(self):
        b = broker()
        b.place_order(Order(
            legs=[OrderLeg(key=KB, side=Side.SELL, qty=1),
                  OrderLeg(key=KF, side=Side.BUY, qty=1)],
            intent=OrderIntent.OPEN, limit_price=-3.5, tag="t:cr", max_loss=10.0))
        pid = b.get_positions()[0].position_id
        b.place_order(Order(
            legs=[OrderLeg(key=KB, side=Side.BUY, qty=1),
                  OrderLeg(key=KF, side=Side.SELL, qty=1)],
            intent=OrderIntent.CLOSE, position_id=pid, limit_price=0.0, tag="t:c"))
        assert b.reserved_collateral == pytest.approx(0.0)


class TestQuoteSanitation:
    def test_zero_bid_option_cannot_be_sold(self):
        b = broker()
        dead = OptionContract(key=KF, bid=0.0, ask=0.30)
        b.update_market({"SPY": chain(contracts=[dead,
            OptionContract(key=KB, bid=6.0, ask=6.4)])}, NOW)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KF, side=Side.SELL, qty=1)],
            intent=OrderIntent.OPEN, limit_price=-0.10, tag="t:z", max_loss=1.0))
        assert r.status is OrderStatus.REJECTED   # D-040: no buyer, no credit

    def test_crossed_quote_rejected_not_crashed(self):
        b = broker()
        crossed = OptionContract(key=KF, bid=2.50, ask=2.00)   # bid > ask
        b.update_market({"SPY": chain(contracts=[crossed])}, NOW)
        r = b.place_order(Order(
            legs=[OrderLeg(key=KF, side=Side.BUY, qty=1)],
            intent=OrderIntent.OPEN, limit_price=2.6, tag="t:x"))
        assert r.status is OrderStatus.REJECTED   # D-043/D-044

    def test_nan_equity_quote_cannot_fill(self):
        from catalyst.core.types import EquityKey
        b = broker()
        b.update_market({"SPY": chain()}, NOW,
                        equity_quotes={"SPY": (math.nan, math.nan)})
        r = b.place_order(Order(
            legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=10)],
            intent=OrderIntent.OPEN, limit_price=525.0, tag="t:e"))
        assert r.status is OrderStatus.REJECTED   # D-039


class TestForceClose:
    def test_force_close_charges_commission_and_records(self):
        b = SimulatedBroker(
            fill_model=FillModelConfig(spread_fill_fraction=0.6,
                                       slippage_pct_of_premium=0.0),
            commissions=CommissionsConfig(alpaca_per_contract=0.65,
                                          schwab_per_contract_per_leg=0.65,
                                          active_profile="alpaca"),
            starting_cash=100_000.0)
        b.update_market({"SPY": chain()}, NOW)
        pid = open_calendar(b)
        commissions_before = b.total_commissions
        b.force_close(pid, 4.0)
        assert b.total_commissions > commissions_before        # D-199
        assert b.synthetic_closes and b.synthetic_closes[0][0] == pid

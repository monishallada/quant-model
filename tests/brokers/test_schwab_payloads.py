"""Schwab adapter payload/parsing tests — no network (audit D-005/033/034/035/
036/106/107/108/110). build_payload is public precisely so this file can exist."""

from __future__ import annotations

from datetime import date

import pytest

from catalyst.brokers.schwab import SchwabBroker, SchwabCredentials, SchwabError
from catalyst.core.types import (
    EquityKey,
    OptionKey,
    OptionRight,
    Order,
    OrderIntent,
    OrderLeg,
    OrderStatus,
    Side,
)

K1 = OptionKey(underlying="SPY", expiry=date(2024, 6, 21),
               right=OptionRight.CALL, strike=533.0)
K2 = OptionKey(underlying="SPY", expiry=date(2024, 6, 21),
               right=OptionRight.CALL, strike=540.0)


def _broker() -> SchwabBroker:
    return SchwabBroker(SchwabCredentials(
        client_id="x", client_secret="y", refresh_token="z", account_hash="HASH1234"))


def _order(intent=OrderIntent.OPEN, limit=-1.50, legs=None, position_id=None):
    return Order(
        legs=legs or [OrderLeg(key=K1, side=Side.SELL, qty=1),
                      OrderLeg(key=K2, side=Side.BUY, qty=1)],
        intent=intent, limit_price=limit, position_id=position_id, tag="t:r")


class TestBuildPayload:
    def test_close_intent_instructions(self):
        p = _broker().build_payload(_order(intent=OrderIntent.CLOSE, position_id="x"))
        instr = {leg["instruction"] for leg in p["orderLegCollection"]}
        assert instr == {"SELL_TO_CLOSE", "BUY_TO_CLOSE"}

    def test_credit_order_type_from_sign(self):
        p = _broker().build_payload(_order(limit=-1.50))
        assert p["orderType"] == "NET_CREDIT"
        assert p["price"] == "1.50"

    def test_debit_order_type_from_sign(self):
        p = _broker().build_payload(_order(limit=2.00))
        assert p["orderType"] == "NET_DEBIT"

    def test_close_of_long_structure_is_net_credit(self):
        """Closing a long spread marked at +1.20: the close order's limit is
        -1.20 under the unified convention (D-106), so Schwab must see
        NET_CREDIT — receiving money, not paying it."""
        p = _broker().build_payload(_order(intent=OrderIntent.CLOSE,
                                           position_id="x", limit=-1.20))
        assert p["orderType"] == "NET_CREDIT"

    def test_equity_leg_supported(self):
        p = _broker().build_payload(_order(
            legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=100)],
            limit=525.0))
        inst = p["orderLegCollection"][0]["instrument"]
        assert inst == {"symbol": "SPY", "assetType": "EQUITY"}

    def test_multileg_fills_together(self):
        p = _broker().build_payload(_order())
        assert p["complexOrderStrategyType"] == "CUSTOM"


class TestModifyOrder:
    def test_partial_delta_refused(self):
        b = _broker()
        result = b.modify_order("42", {"price": "1.25"})
        assert result.status is OrderStatus.REJECTED
        assert "complete order payload" in result.message


class TestReconciliation:
    def _positions_response(self, rows):
        return {"securitiesAccount": {"positions": rows}}

    def test_option_per_unit_value(self, monkeypatch):
        b = _broker()
        rows = self._positions_response([{
            "instrument": {"assetType": "OPTION", "symbol": "SPY   240621C00533000"},
            "longQuantity": 0, "shortQuantity": 3,
            "averagePrice": 1.50, "marketValue": -450.0}])
        monkeypatch.setattr(b, "_req", lambda *a, **k: rows)
        pos = b.get_positions()[0]
        assert pos.qty == 3
        assert pos.current_value == pytest.approx(-1.50)

    def test_equity_row_included(self, monkeypatch):
        b = _broker()
        rows = self._positions_response([{
            "instrument": {"assetType": "EQUITY", "symbol": "SPY"},
            "longQuantity": 100, "shortQuantity": 0,
            "averagePrice": 500.0, "marketValue": 52500.0}])
        monkeypatch.setattr(b, "_req", lambda *a, **k: rows)
        pos = b.get_positions()[0]
        assert isinstance(pos.legs[0].key, EquityKey)
        assert pos.current_value == pytest.approx(525.0)

    def test_unparseable_option_fails_loudly(self, monkeypatch):
        b = _broker()
        rows = self._positions_response([{
            "instrument": {"assetType": "OPTION", "symbol": "GARBAGE"},
            "longQuantity": 1, "shortQuantity": 0,
            "averagePrice": 1.0, "marketValue": 100.0}])
        monkeypatch.setattr(b, "_req", lambda *a, **k: rows)
        with pytest.raises(SchwabError):
            b.get_positions()


class TestAccountBalances:
    def test_cash_account_field_names(self, monkeypatch):
        b = _broker()
        data = {"securitiesAccount": {"currentBalances": {
            "liquidationValue": 100000.0,
            "cashAvailableForTrading": 60000.0}}}
        monkeypatch.setattr(b, "_req", lambda *a, **k: data)
        acct = b.get_account()
        assert acct.cash == 60000.0
        assert acct.buying_power == 60000.0

    def test_missing_balances_fail_loudly_not_zero(self, monkeypatch):
        b = _broker()
        data = {"securitiesAccount": {"currentBalances": {"liquidationValue": 1.0}}}
        monkeypatch.setattr(b, "_req", lambda *a, **k: data)
        with pytest.raises(SchwabError):
            b.get_account()

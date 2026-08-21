"""Alpaca adapter payload/parsing tests — no network, stubbed transport.

Regression coverage for the census's worst execution defects:
D-001 (closes sent as opens), D-003 (min//-truncation unit math),
D-004 (abs() destroying the credit/debit sign), D-029 (equity legs),
D-030 (phantom $0 fills), D-031 (silently dropped positions),
D-032 (total-dollar vs per-unit marks), D-104 (async cancel),
D-105 (unknown status defaulting).
"""

from __future__ import annotations

from datetime import date

import pytest

from catalyst.brokers.alpaca import AlpacaBroker, AlpacaCredentials, AlpacaError, _map_status
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


def _broker(monkeypatch, response=None):
    b = AlpacaBroker(AlpacaCredentials(
        key_id="test", secret_key="test",
        endpoint="https://paper-api.alpaca.markets"))
    b.captured = []
    def fake_req(method, path, **kw):
        b.captured.append((method, path, kw))
        return response if response is not None else {
            "id": "oid-1", "status": "new", "filled_qty": "0",
            "filled_avg_price": None}
    monkeypatch.setattr(b, "_req", fake_req)
    return b


def _order(intent=OrderIntent.OPEN, limit=-1.50, legs=None, position_id=None):
    return Order(
        legs=legs or [OrderLeg(key=K1, side=Side.SELL, qty=2),
                      OrderLeg(key=K2, side=Side.BUY, qty=2)],
        intent=intent, limit_price=limit, position_id=position_id,
        tag="test:ref")


class TestPayloadConstruction:
    def test_close_intent_reaches_every_leg(self, monkeypatch):
        b = _broker(monkeypatch)
        b.place_order(_order(intent=OrderIntent.CLOSE, position_id="pos-1"))
        payload = b.captured[0][2]["json"]
        intents = {leg["position_intent"] for leg in payload["legs"]}
        assert intents == {"sell_to_close", "buy_to_close"}

    def test_open_intent_reaches_every_leg(self, monkeypatch):
        b = _broker(monkeypatch)
        b.place_order(_order(intent=OrderIntent.OPEN))
        payload = b.captured[0][2]["json"]
        intents = {leg["position_intent"] for leg in payload["legs"]}
        assert intents == {"sell_to_open", "buy_to_open"}

    def test_credit_limit_price_keeps_sign(self, monkeypatch):
        b = _broker(monkeypatch)
        b.place_order(_order(limit=-1.50))
        assert b.captured[0][2]["json"]["limit_price"] == "-1.50"

    def test_debit_limit_price_positive(self, monkeypatch):
        b = _broker(monkeypatch)
        b.place_order(_order(limit=2.25))
        assert b.captured[0][2]["json"]["limit_price"] == "2.25"

    def test_gcd_unit_decomposition(self, monkeypatch):
        b = _broker(monkeypatch)
        legs = [OrderLeg(key=K1, side=Side.BUY, qty=4),
                OrderLeg(key=K2, side=Side.SELL, qty=6)]
        b.place_order(_order(legs=legs))
        payload = b.captured[0][2]["json"]
        assert payload["qty"] == "2"
        assert [leg["ratio_qty"] for leg in payload["legs"]] == ["2", "3"]

    def test_coprime_ratios_preserved(self, monkeypatch):
        b = _broker(monkeypatch)
        legs = [OrderLeg(key=K1, side=Side.BUY, qty=2),
                OrderLeg(key=K2, side=Side.SELL, qty=3)]
        b.place_order(_order(legs=legs))
        payload = b.captured[0][2]["json"]
        assert payload["qty"] == "1"
        assert [leg["ratio_qty"] for leg in payload["legs"]] == ["2", "3"]

    def test_equity_leg_uses_plain_symbol(self, monkeypatch):
        b = _broker(monkeypatch)
        b.place_order(_order(legs=[OrderLeg(key=EquityKey(underlying="SPY"),
                                            side=Side.BUY, qty=100)], limit=525.0))
        payload = b.captured[0][2]["json"]
        assert payload["symbol"] == "SPY"
        assert "position_intent" not in payload

    def test_unfilled_order_reports_none_fill_price(self, monkeypatch):
        b = _broker(monkeypatch)
        result = b.place_order(_order())
        assert result.status is OrderStatus.ACCEPTED
        assert result.avg_fill_price is None      # NOT a phantom $0.00


class TestStatusAndCancel:
    def test_every_documented_status_mapped(self):
        assert _map_status("held") is OrderStatus.ACCEPTED
        assert _map_status("done_for_day") is OrderStatus.CANCELED
        assert _map_status("replaced") is OrderStatus.CANCELED
        assert _map_status("partially_filled") is OrderStatus.PARTIALLY_FILLED

    def test_unknown_status_logged_not_silent(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            s = _map_status("some_future_status")
        assert s is OrderStatus.ACCEPTED
        assert "UNKNOWN" in caplog.text

    def test_cancel_is_reported_async(self, monkeypatch):
        b = _broker(monkeypatch, response={})
        result = b.cancel_order("oid-9")
        assert result.status is OrderStatus.ACCEPTED   # NOT CANCELED yet
        assert "async" in result.message


class TestReconciliation:
    def test_option_position_per_unit_value(self, monkeypatch):
        rows = [{"symbol": "SPY240621C00533000", "asset_class": "us_option",
                 "qty": "-3", "avg_entry_price": "1.50",
                 "market_value": "-450.0"}]
        b = _broker(monkeypatch, response=rows)
        pos = b.get_positions()[0]
        assert pos.qty == 3
        assert pos.current_value == pytest.approx(-450.0 / (3 * 100))  # per unit

    def test_equity_position_included(self, monkeypatch):
        rows = [{"symbol": "SPY", "asset_class": "us_equity", "qty": "100",
                 "avg_entry_price": "500.0", "market_value": "52500.0"}]
        b = _broker(monkeypatch, response=rows)
        pos = b.get_positions()[0]
        assert isinstance(pos.legs[0].key, EquityKey)
        assert pos.current_value == pytest.approx(525.0)

    def test_adjusted_root_parses_after_symbology_fix(self, monkeypatch):
        rows = [{"symbol": "AAPL1240621C00150000", "asset_class": "us_option",
                 "qty": "1", "avg_entry_price": "1.0", "market_value": "100.0"}]
        b = _broker(monkeypatch, response=rows)
        assert b.get_positions()[0].legs[0].key.underlying == "AAPL1"

    def test_garbage_option_symbol_fails_loudly(self, monkeypatch):
        rows = [{"symbol": "???240621C00150000", "asset_class": "us_option",
                 "qty": "1", "avg_entry_price": "1.0", "market_value": "100.0"}]
        b = _broker(monkeypatch, response=rows)
        with pytest.raises(AlpacaError):
            b.get_positions()

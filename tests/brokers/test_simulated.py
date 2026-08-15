"""SimulatedBroker fill/cost modeling on hand-computed cases."""

from datetime import date, datetime

import pytest

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
from catalyst.brokers.simulated import SimulatedBroker

NOW = datetime(2024, 6, 3, 15, 45)
KEY = OptionKey(underlying="SPY", expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=533.0)
KEY2 = OptionKey(underlying="SPY", expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=540.0)


def make_chain(bid: float = 2.80, ask: float = 3.00, bid2: float = 1.00, ask2: float = 1.10) -> OptionChain:
    return OptionChain(
        underlying="SPY",
        underlying_price=525.94,
        timestamp=NOW,
        contracts=[
            OptionContract(key=KEY, bid=bid, ask=ask),
            OptionContract(key=KEY2, bid=bid2, ask=ask2),
        ],
    )


def make_broker(
    cash: float = 100_000.0,
    frac: float = 0.60,
    slip: float = 0.02,
    commission: float = 0.0,
) -> SimulatedBroker:
    broker = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=frac, slippage_pct_of_premium=slip),
        commissions=CommissionsConfig(
            alpaca_per_contract=commission,
            schwab_per_contract_per_leg=0.65,
            active_profile="alpaca",
        ),
        starting_cash=cash,
    )
    broker.update_market({"SPY": make_chain()}, NOW)
    return broker


def test_buy_fill_price_hand_computed() -> None:
    # mid=2.90, 60% toward ask: 2.90 + 0.6*(3.00-2.90) = 2.96; +2% slip = 3.0192
    broker = make_broker()
    result = broker.place_order(
        Order(legs=[OrderLeg(key=KEY, side=Side.BUY, qty=1)], limit_price=2.90, tag="t:x")
    )
    assert result.status is OrderStatus.FILLED
    assert result.avg_fill_price == pytest.approx(3.0192, abs=1e-6)
    assert broker.cash == pytest.approx(100_000.0 - 301.92, abs=1e-6)


def test_sell_to_close_hand_computed() -> None:
    broker = make_broker()
    opened = broker.place_order(
        Order(legs=[OrderLeg(key=KEY, side=Side.BUY, qty=2)], limit_price=2.90, tag="t:x")
    )
    pos_id = f"pos-{opened.order_id}"
    # Sell fill: mid - 0.6*(mid-bid) = 2.90 - 0.06 = 2.84; -2% slip = 2.7832
    closed = broker.place_order(
        Order(
            legs=[OrderLeg(key=KEY, side=Side.SELL, qty=2)],
            limit_price=2.84,
            intent=OrderIntent.CLOSE,
            position_id=pos_id,
        )
    )
    assert closed.status is OrderStatus.FILLED
    assert closed.avg_fill_price == pytest.approx(2.7832, abs=1e-6)
    assert broker.get_positions() == []
    # Round trip cost: (3.0192 - 2.7832) * 2 * 100
    assert broker.cash == pytest.approx(100_000.0 - 47.20, abs=1e-2)


def test_commission_per_contract_per_leg() -> None:
    broker = make_broker(commission=0.65)
    result = broker.place_order(
        Order(
            legs=[
                OrderLeg(key=KEY, side=Side.BUY, qty=3),
                OrderLeg(key=KEY2, side=Side.SELL, qty=3),
            ],
            limit_price=1.90,
            tag="t:x",
        )
    )
    assert result.commission == pytest.approx(0.65 * 6)


def test_vertical_spread_nets_legs() -> None:
    broker = make_broker(slip=0.0, frac=0.0)  # fills at mid for arithmetic clarity
    result = broker.place_order(
        Order(
            legs=[
                OrderLeg(key=KEY, side=Side.BUY, qty=1),
                OrderLeg(key=KEY2, side=Side.SELL, qty=1),
            ],
            limit_price=1.85,
            tag="t:x",
        )
    )
    # mid(533C)=2.90 paid, mid(540C)=1.05 received -> net debit 1.85
    assert result.avg_fill_price == pytest.approx(1.85, abs=1e-9)
    pos = broker.get_positions()[0]
    assert pos.qty == 1
    assert len(pos.legs) == 2


def test_partial_close_reduces_qty() -> None:
    broker = make_broker()
    opened = broker.place_order(
        Order(legs=[OrderLeg(key=KEY, side=Side.BUY, qty=4)], limit_price=2.90, tag="t:x")
    )
    pos_id = f"pos-{opened.order_id}"
    broker.place_order(
        Order(
            legs=[OrderLeg(key=KEY, side=Side.SELL, qty=2)],
            limit_price=2.84,
            intent=OrderIntent.CLOSE,
            position_id=pos_id,
        )
    )
    assert broker.get_positions()[0].qty == 2


def test_insufficient_cash_rejected() -> None:
    broker = make_broker(cash=100.0)
    result = broker.place_order(
        Order(legs=[OrderLeg(key=KEY, side=Side.BUY, qty=1)], limit_price=2.90, tag="t:x")
    )
    assert result.status is OrderStatus.REJECTED


def test_expiry_settles_at_intrinsic() -> None:
    broker = make_broker(slip=0.0, frac=0.0)
    broker.place_order(
        Order(legs=[OrderLeg(key=KEY, side=Side.BUY, qty=1)], limit_price=2.90, tag="t:x")
    )
    cash_before = broker.cash
    # Advance past expiry with spot above strike: intrinsic = 542 - 533 = 9.
    later = datetime(2024, 6, 24, 15, 45)
    chain = OptionChain(
        underlying="SPY", underlying_price=542.0, timestamp=later,
        contracts=[OptionContract(key=KEY, bid=0.0, ask=0.0)],
    )
    broker.update_market({"SPY": chain}, later)
    assert broker.get_positions() == []
    assert broker.cash == pytest.approx(cash_before + 900.0)

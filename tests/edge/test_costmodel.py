"""Cost model: hand-computed marketable fills, mid as a hard floor, latency
degrades fills, limit orders fill only on a crossing path, and every fill's
cost attribution decomposes exactly. All quotes are synthetic and every
timestamp predates the 2026-02-22 lockbox window."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from edge.core.config import ExecutionConfig
from edge.core.events import QuoteEvent, Side
from edge.execution.costmodel import (
    BetterThanNBBOError,
    CostModel,
    FillStatus,
    assert_not_better_than_mid,
)

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
T0 = datetime(2026, 1, 6, 10, 0, tzinfo=ET)


@pytest.fixture
def model() -> CostModel:
    """Model with hand-computable frictions: 60% crossing, 2% slip, $0.65/ct."""
    cfg = ExecutionConfig(
        spread_fill_fraction=0.6,
        slippage_pct=0.02,
        commission_per_contract=0.65,
        latency_ms=[0, 500],
    )
    return CostModel(cfg)


def quote(bid: float, ask: float, *, ts: datetime = T0,
          bid_size: int = 10, ask_size: int = 10) -> QuoteEvent:
    return QuoteEvent(ts=ts, symbol="XYZ", bid=bid, ask=ask,
                      bid_size=bid_size, ask_size=ask_size)


# ---------------------------------------------------------------------------
# Marketable fills, hand-computed to the cent
# ---------------------------------------------------------------------------


def test_marketable_buy_fill_to_the_cent(model: CostModel) -> None:
    # mid 1.05; crossed = 1.05 + 0.6*(1.10-1.05) = 1.08; * 1.02 = 1.1016
    fill = model.marketable_fill(quote(1.00, 1.10), Side.BUY, 2)
    assert fill.status is FillStatus.FILLED
    assert fill.qty == 2
    assert fill.price == pytest.approx(1.1016, abs=1e-9)
    assert fill.costs.spread_cost == pytest.approx(6.00, abs=1e-6)     # 0.03*2*100
    assert fill.costs.slippage_cost == pytest.approx(4.32, abs=1e-6)   # 0.0216*2*100
    assert fill.costs.commission == pytest.approx(1.30, abs=1e-9)      # 0.65*2
    assert fill.costs.total == pytest.approx(11.62, abs=1e-6)


def test_marketable_sell_fill_to_the_cent(model: CostModel) -> None:
    # mid 0.95; crossed = 0.95 - 0.6*(0.95-0.90) = 0.92; * 0.98 = 0.9016
    fill = model.marketable_fill(quote(0.90, 1.00), Side.SELL, 2)
    assert fill.price == pytest.approx(0.9016, abs=1e-9)
    assert fill.costs.spread_cost == pytest.approx(6.00, abs=1e-6)     # 0.03*2*100
    assert fill.costs.slippage_cost == pytest.approx(3.68, abs=1e-6)   # 0.0184*2*100
    assert fill.costs.commission == pytest.approx(1.30, abs=1e-9)
    assert fill.costs.total == pytest.approx(10.98, abs=1e-6)


def test_vertical_both_legs(model: CostModel) -> None:
    """Buy the 1.00/1.10 leg, sell the 0.90/1.00 leg: a 2-lot debit vertical."""
    long_leg = model.marketable_fill(quote(1.00, 1.10), Side.BUY, 2)
    short_leg = model.marketable_fill(quote(0.90, 1.00), Side.SELL, 2)
    assert long_leg.price is not None and short_leg.price is not None
    # Net debit per share: 1.1016 - 0.9016 = 0.20 (mid-to-mid it was 0.10).
    assert long_leg.price - short_leg.price == pytest.approx(0.20, abs=1e-9)
    # Commission is per contract per leg per side: 2 legs * 2 contracts * 0.65.
    assert long_leg.costs.commission + short_leg.costs.commission == pytest.approx(2.60)
    # Total friction on the structure, to the cent.
    assert long_leg.costs.total + short_leg.costs.total == pytest.approx(22.60, abs=1e-6)


def test_marketable_rejects_crossed_book_and_bad_qty(model: CostModel) -> None:
    with pytest.raises(ValueError, match="crossed"):
        model.marketable_fill(quote(1.10, 1.00), Side.BUY, 1)
    with pytest.raises(ValueError, match="qty"):
        model.marketable_fill(quote(1.00, 1.10), Side.BUY, 0)


# ---------------------------------------------------------------------------
# Mid is a hard floor
# ---------------------------------------------------------------------------


def test_mid_is_hard_floor() -> None:
    q = quote(1.00, 1.10)  # mid 1.05
    with pytest.raises(BetterThanNBBOError):
        assert_not_better_than_mid(1.04, q, Side.BUY)    # buy below mid
    with pytest.raises(BetterThanNBBOError):
        assert_not_better_than_mid(1.06, q, Side.SELL)   # sell above mid
    # At mid is the floor, not a violation (crossing fraction 0 fills there).
    assert_not_better_than_mid(q.mid, q, Side.BUY)
    assert_not_better_than_mid(q.mid, q, Side.SELL)


def test_nan_price_never_reaches_a_fill() -> None:
    with pytest.raises(BetterThanNBBOError, match="NaN"):
        assert_not_better_than_mid(float("nan"), quote(1.00, 1.10), Side.BUY)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_zero_vs_500ms_latency_on_a_moving_quote(model: CostModel) -> None:
    q0 = quote(1.00, 1.10)  # decision quote: zero-latency buy fills 1.1016
    q_later = quote(1.10, 1.20, ts=T0 + timedelta(milliseconds=500))
    # mid 1.15; crossed = 1.15 + 0.6*0.05 = 1.18; * 1.02 = 1.2036

    instant = model.fill_with_latency(q0, q0, 0, Side.BUY, 1)
    delayed = model.fill_with_latency(q0, q_later, 500, Side.BUY, 1)

    assert instant.price == pytest.approx(1.1016, abs=1e-9)
    assert delayed.price == pytest.approx(1.2036, abs=1e-9)
    assert instant.price is not None and delayed.price is not None
    assert delayed.price > instant.price  # latency made the buy strictly worse


def test_latency_missing_or_stale_quote_is_no_fill(model: CostModel) -> None:
    q0 = quote(1.00, 1.10)
    missing = model.fill_with_latency(q0, None, 500, Side.BUY, 1)
    assert missing.status is FillStatus.UNFILLED
    assert missing.qty == 0 and missing.price is None
    assert missing.costs.total == 0.0

    stale = quote(1.00, 1.10, ts=T0 - timedelta(seconds=1))  # predates decision
    result = model.fill_with_latency(q0, stale, 500, Side.BUY, 1)
    assert result.status is FillStatus.UNFILLED


def test_latency_future_quote_is_lookahead_bug(model: CostModel) -> None:
    q0 = quote(1.00, 1.10)
    future = quote(1.00, 1.10, ts=T0 + timedelta(milliseconds=600))
    with pytest.raises(ValueError, match="lookahead"):
        model.fill_with_latency(q0, future, 500, Side.BUY, 1)


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


def test_limit_never_crossed_is_unfilled(model: CostModel) -> None:
    path = [
        quote(1.00, 1.05, ts=T0),
        quote(0.99, 1.02, ts=T0 + timedelta(seconds=1)),
        quote(0.98, 1.01, ts=T0 + timedelta(seconds=2)),
    ]
    result = model.limit_fill(Side.BUY, 1.00, 5, path)  # ask never <= 1.00
    assert result.status is FillStatus.UNFILLED
    assert result.qty == 0
    assert result.price is None
    assert result.costs.total == 0.0


def test_limit_buy_fills_across_crossing_quotes(model: CostModel) -> None:
    path = [
        quote(1.00, 1.10, ts=T0),                                        # no cross
        quote(1.00, 1.04, ts=T0 + timedelta(seconds=1), ask_size=3),     # mid 1.02
        quote(0.98, 1.02, ts=T0 + timedelta(seconds=2), ask_size=10),    # mid 1.00
    ]
    result = model.limit_fill(Side.BUY, 1.05, 5, path)
    assert result.status is FillStatus.FILLED
    assert result.qty == 5
    assert result.price == 1.05  # always AT the limit, never the better touch
    # Tranche 1: 3 @ (1.05-1.02) -> $9.00; tranche 2: 2 @ (1.05-1.00) -> $10.00
    assert result.costs.spread_cost == pytest.approx(19.00, abs=1e-6)
    assert result.costs.slippage_cost == 0.0  # a resting limit pays no slippage
    assert result.costs.commission == pytest.approx(3.25, abs=1e-9)  # 0.65*5


def test_limit_partial_fill_pro_rata_to_displayed_size(model: CostModel) -> None:
    path = [quote(1.10, 1.14, ts=T0, bid_size=4)]  # mid 1.12, only 4 displayed
    result = model.limit_fill(Side.SELL, 1.10, 10, path)
    assert result.status is FillStatus.PARTIAL
    assert result.qty == 4
    assert result.price == 1.10
    assert result.costs.spread_cost == pytest.approx(8.00, abs=1e-6)  # (1.12-1.10)*4*100
    assert result.costs.commission == pytest.approx(2.60, abs=1e-9)   # 0.65*4


def test_limit_min_fill_is_one_contract(model: CostModel) -> None:
    path = [quote(1.00, 1.04, ts=T0, ask_size=0)]  # crossing but zero displayed
    result = model.limit_fill(Side.BUY, 1.05, 5, path)
    assert result.status is FillStatus.PARTIAL
    assert result.qty == 1


def test_limit_path_must_be_time_ordered(model: CostModel) -> None:
    path = [
        quote(1.00, 1.10, ts=T0 + timedelta(seconds=1)),
        quote(1.00, 1.10, ts=T0),  # goes backwards
    ]
    with pytest.raises(ValueError, match="out of order"):
        model.limit_fill(Side.BUY, 1.05, 1, path)


# ---------------------------------------------------------------------------
# Cost attribution: exact decomposition
# ---------------------------------------------------------------------------


def test_attribution_sums_exactly_buy(model: CostModel) -> None:
    q = quote(1.00, 1.10)
    fill = model.marketable_fill(q, Side.BUY, 3)
    assert fill.price is not None
    expected = (fill.price - q.mid) * 3 * 100 + fill.costs.commission
    assert fill.costs.total == expected  # exact, not approx


def test_attribution_sums_exactly_sell(model: CostModel) -> None:
    q = quote(0.90, 1.00)
    fill = model.marketable_fill(q, Side.SELL, 3)
    assert fill.price is not None
    expected = (q.mid - fill.price) * 3 * 100 + fill.costs.commission
    assert fill.costs.total == expected  # exact, not approx

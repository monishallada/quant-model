"""Engine unit tests: fixed chain + signal ⇒ asserted structure."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from catalyst.core.config import load_config
from catalyst.core.models import (
    Catalyst,
    CatalystType,
    Direction,
    OptionRight,
    Side,
    SignalResult,
)
from catalyst.archive.engines.engine_a_convexity import EngineAConvexity
from catalyst.archive.engines.engine_b_crush_spread import EngineBCrushSpread
from catalyst.archive.engines.engine_c_pead import EngineCPead
from catalyst.archive.engines.engine_d_calendar import EngineDCalendar
from tests.conftest import FakeIVRank, build_chain

CFG = load_config("backtest")
AS_OF = datetime(2024, 6, 3, 15, 45)  # Monday
LONG = SignalResult(direction=Direction.LONG, confidence=0.8)
SHORT = SignalResult(direction=Direction.SHORT, confidence=0.8)
NEUTRAL = SignalResult(direction=Direction.NEUTRAL, confidence=0.0)


def catalyst_on(d: date, hour: int = 16, ctype: CatalystType = CatalystType.OTHER, **kw) -> Catalyst:
    return Catalyst(
        symbol="SPY", type=ctype, when=datetime(d.year, d.month, d.day, hour), source="test", **kw
    )


# ---------------------------------------------------------------------------
# Engine A
# ---------------------------------------------------------------------------


def make_engine_a(rank: float | None = 20.0) -> EngineAConvexity:
    cfg = CFG.engines.engine_a.model_copy(update={"enabled": True})  # archived: off in config
    return EngineAConvexity(cfg, CFG.risk, FakeIVRank(rank))


def test_engine_a_proposes_otm_call_on_long() -> None:
    # Catalyst Wed 2024-06-12: 7 trading days from Mon 06-03 -> inside [5, 10].
    cat = catalyst_on(date(2024, 6, 12))
    chain = build_chain(expiry_ivs={date(2024, 6, 21): 0.25, date(2024, 7, 19): 0.24})
    trade = make_engine_a().evaluate(cat, chain, LONG, AS_OF)
    assert trade is not None
    assert len(trade.legs) == 1 and trade.legs[0].side is Side.BUY
    leg_key = trade.legs[0].key
    assert leg_key.right is OptionRight.CALL
    assert leg_key.strike > 100.0  # slightly OTM
    picked = chain.find(leg_key)
    assert 0.25 <= abs(picked.greeks.delta) <= 0.35
    # Max-loss basis is worst-case NBBO (ask), at or above the mid estimate.
    assert trade.unit_max_loss >= trade.unit_cost > 0
    # Default: out the session before the event (2024-06-12 event, AMC ->
    # reaction 06-13, so hard exit 06-12... reaction-1 trading day).
    assert trade.exit_rules.hard_exit_date is not None
    assert trade.exit_rules.hard_exit_date < date(2024, 6, 13)
    assert trade.per_trade_risk_fraction == CFG.engines.engine_a.per_trade_risk


def test_engine_a_skips_when_iv_rank_high() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    assert make_engine_a(rank=55.0).evaluate(cat, build_chain(), LONG, AS_OF) is None


def test_engine_a_skips_on_neutral_signal() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    assert make_engine_a().evaluate(cat, build_chain(), NEUTRAL, AS_OF) is None


def test_engine_a_skips_outside_trigger_window() -> None:
    too_close = catalyst_on(date(2024, 6, 5))  # 2 trading days out
    too_far = catalyst_on(date(2024, 7, 3))  # >10 trading days out
    engine = make_engine_a()
    assert engine.evaluate(too_close, build_chain(), LONG, AS_OF) is None
    assert engine.evaluate(too_far, build_chain(), LONG, AS_OF) is None


def test_engine_a_puts_on_short_signal() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    trade = make_engine_a().evaluate(cat, build_chain(), SHORT, AS_OF)
    assert trade is not None
    assert trade.legs[0].key.right is OptionRight.PUT
    assert trade.legs[0].key.strike < 100.0


# ---------------------------------------------------------------------------
# Engine B
# ---------------------------------------------------------------------------


def make_engine_b() -> EngineBCrushSpread:
    cfg = CFG.engines.engine_b.model_copy(update={"enabled": True})  # archived: off in config
    return EngineBCrushSpread(cfg, CFG.risk)


def test_engine_b_debit_spread_structure() -> None:
    # Catalyst Wed 06-05 AMC: 2 trading days out from Mon 06-03.
    cat = catalyst_on(date(2024, 6, 5))
    chain = build_chain()
    trade = make_engine_b().evaluate(cat, chain, LONG, AS_OF)
    assert trade is not None
    assert len(trade.legs) == 2
    buy = next(l for l in trade.legs if l.side is Side.BUY)
    sell = next(l for l in trade.legs if l.side is Side.SELL)
    assert buy.key.right is OptionRight.CALL and sell.key.right is OptionRight.CALL
    assert buy.key.expiry == sell.key.expiry == date(2024, 6, 21)  # first expiry after event
    assert buy.key.strike < sell.key.strike  # long 0.40d below short 0.20d
    width = sell.key.strike - buy.key.strike
    assert 0 < trade.unit_cost < width
    assert trade.exit_rules.tp_fraction_of_max == CFG.engines.engine_b.tp_fraction_of_max
    # Held through the event; time-stopped after it resolves.
    assert trade.exit_rules.hard_exit_date > cat.when.date()


def test_engine_b_short_uses_puts() -> None:
    cat = catalyst_on(date(2024, 6, 5))
    trade = make_engine_b().evaluate(cat, build_chain(), SHORT, AS_OF)
    assert trade is not None
    assert all(l.key.right is OptionRight.PUT for l in trade.legs)
    buy = next(l for l in trade.legs if l.side is Side.BUY)
    sell = next(l for l in trade.legs if l.side is Side.SELL)
    assert buy.key.strike > sell.key.strike  # put spread mirrors


def test_engine_b_skips_neutral_and_window() -> None:
    engine = make_engine_b()
    assert engine.evaluate(catalyst_on(date(2024, 6, 5)), build_chain(), NEUTRAL, AS_OF) is None
    far = catalyst_on(date(2024, 6, 12))  # 7 trading days out, window is 1-3
    assert engine.evaluate(far, build_chain(), LONG, AS_OF) is None


# ---------------------------------------------------------------------------
# Engine C
# ---------------------------------------------------------------------------


def make_engine_c(rank: float = 30.0) -> EngineCPead:
    cfg = CFG.engines.engine_c.model_copy(update={"enabled": True})  # archived: off in config
    return EngineCPead(cfg, CFG.risk, FakeIVRank(rank))


def earnings(d: date, actual: float, estimate: float, hour: int = 16) -> Catalyst:
    return catalyst_on(d, hour=hour, ctype=CatalystType.EARNINGS,
                       eps_actual=actual, eps_estimate=estimate)


def test_engine_c_long_on_positive_surprise_ignores_signal() -> None:
    # AMC report Fri 05-31 -> reaction Mon 06-03 -> today is day 0 of window.
    cat = earnings(date(2024, 5, 31), actual=1.20, estimate=1.00)
    chain = build_chain()  # 06-21 expiry = 18 days out (inside 2-4 weeks)
    trade = make_engine_c().evaluate(cat, chain, SHORT, AS_OF)  # signal says SHORT
    assert trade is not None
    assert trade.direction is Direction.LONG  # measured surprise wins
    buy = next(l for l in trade.legs if l.side is Side.BUY)
    assert buy.key.right is OptionRight.CALL
    assert trade.exit_rules.max_hold_trading_days == CFG.engines.engine_c.hold_days


def test_engine_c_short_on_negative_surprise() -> None:
    cat = earnings(date(2024, 5, 31), actual=0.80, estimate=1.00)
    trade = make_engine_c().evaluate(cat, build_chain(), NEUTRAL, AS_OF)
    assert trade is not None and trade.direction is Direction.SHORT


def test_engine_c_gates() -> None:
    small = earnings(date(2024, 5, 31), actual=1.02, estimate=1.00)  # 2% < 5%
    assert make_engine_c().evaluate(small, build_chain(), NEUTRAL, AS_OF) is None
    big = earnings(date(2024, 5, 31), actual=1.20, estimate=1.00)
    assert make_engine_c(rank=80.0).evaluate(big, build_chain(), NEUTRAL, AS_OF) is None  # IV not normalized
    not_earnings = catalyst_on(date(2024, 5, 31), ctype=CatalystType.CPI)
    assert make_engine_c().evaluate(not_earnings, build_chain(), NEUTRAL, AS_OF) is None
    future = earnings(date(2024, 6, 10), actual=1.2, estimate=1.0)  # hasn't happened
    assert make_engine_c().evaluate(future, build_chain(), NEUTRAL, AS_OF) is None


# ---------------------------------------------------------------------------
# Engine D
# ---------------------------------------------------------------------------


def make_engine_d(enabled: bool = True) -> EngineDCalendar:
    cfg = CFG.engines.engine_d.model_copy(update={"enabled": enabled})
    return EngineDCalendar(cfg, CFG.risk)


def test_engine_d_calendar_on_elevated_front_iv() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    # Front IV 35% vs back 24% -> ratio 1.46 >= 1.15 threshold.
    chain = build_chain(expiry_ivs={date(2024, 6, 21): 0.35, date(2024, 7, 19): 0.24})
    trade = make_engine_d().evaluate(cat, chain, NEUTRAL, AS_OF)
    assert trade is not None
    sell = next(l for l in trade.legs if l.side is Side.SELL)
    buy = next(l for l in trade.legs if l.side is Side.BUY)
    assert sell.key.expiry == date(2024, 6, 21) and buy.key.expiry == date(2024, 7, 19)
    assert sell.key.strike == buy.key.strike  # true calendar
    assert trade.direction is Direction.NEUTRAL
    assert trade.unit_cost > 0
    assert trade.exit_rules.use_stops  # max-loss guard always armed


def test_engine_d_skips_flat_term_structure() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    chain = build_chain(expiry_ivs={date(2024, 6, 21): 0.25, date(2024, 7, 19): 0.24})
    assert make_engine_d().evaluate(cat, chain, NEUTRAL, AS_OF) is None


def test_engine_d_disabled_by_default_config() -> None:
    cat = catalyst_on(date(2024, 6, 12))
    chain = build_chain(expiry_ivs={date(2024, 6, 21): 0.35, date(2024, 7, 19): 0.24})
    assert make_engine_d(enabled=False).evaluate(cat, chain, NEUTRAL, AS_OF) is None

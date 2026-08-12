"""RiskManager: the authoritative layer. These tests are the spec's teeth —
caps reject over-limit orders, the cash floor is never breached (property
test over randomized order streams), breakers halt entries."""

from __future__ import annotations

import random
from datetime import date, datetime

import pandas as pd

from catalyst.core.config import load_config
from catalyst.core.models import (
    AccountState,
    Direction,
    ExitRules,
    OptionKey,
    OptionRight,
    OrderLeg,
    Position,
    PositionLeg,
    ProposedTrade,
    Side,
)
from catalyst.risk.manager import RiskManager

CFG = load_config("backtest")
TODAY = datetime(2024, 6, 3, 15, 45)
KEY = OptionKey(underlying="SPY", expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=533.0)


def account(
    equity: float = 100_000.0, cash: float | None = None, when: datetime = TODAY
) -> AccountState:
    return AccountState(
        equity=equity, cash=cash if cash is not None else equity, buying_power=0.0,
        timestamp=when,
    )


def proposal(
    unit_cost: float = 3.0,
    risk_fraction: float = 0.03,
    engine: str = "engine_a",
    symbol: str = "SPY",
) -> ProposedTrade:
    key = OptionKey(underlying=symbol, expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=533.0)
    return ProposedTrade(
        engine=engine, catalyst_ref="t:x", legs=[OrderLeg(key=key, side=Side.BUY, qty=1)],
        unit_cost=unit_cost, unit_max_loss=unit_cost, direction=Direction.LONG,
        exit_rules=ExitRules(), per_trade_risk_fraction=risk_fraction,
    )


def open_position(premium: float, engine: str = "engine_a", symbol: str = "SPY") -> Position:
    key = OptionKey(underlying=symbol, expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=500.0)
    return Position(
        position_id=f"p-{random.randint(0, 10**9)}", legs=[PositionLeg(key=key, side=Side.BUY, qty=1)],
        qty=1, entry_price=premium / 100.0, entry_time=TODAY, engine_tag=engine,
        direction=Direction.LONG, exit_rules=ExitRules(), current_value=premium / 100.0,
    )


def test_basic_sizing_matches_fixed_fractional() -> None:
    rm = RiskManager(CFG.risk)
    decision = rm.size_entry(proposal(unit_cost=3.0, risk_fraction=0.03), account(), [])
    # 3000 budget / (300 x 1.05 sizing_cost_buffer) = 9.52 -> 9 units.
    # The buffer keeps the risk budget a hard bound under worse-than-mid fills.
    assert decision.units == 9


def test_cash_floor_rejects_and_trims() -> None:
    rm = RiskManager(CFG.risk)
    # equity 100k, floor 40k. cash 41k -> headroom 1k -> only 3 units of a 3.00 option.
    d = rm.size_entry(proposal(unit_cost=3.0), account(equity=100_000, cash=41_000), [])
    assert d.units == 3
    # cash exactly at floor -> reject.
    d2 = rm.size_entry(proposal(unit_cost=3.0), account(equity=100_000, cash=40_000), [])
    assert d2.units == 0 and "cash floor" in d2.reason


def test_heat_cap_rejects_when_fully_deployed() -> None:
    rm = RiskManager(CFG.risk)
    positions = [open_position(12_500.0), open_position(12_400.0)]  # 24.9k of 25k cap
    d = rm.size_entry(proposal(unit_cost=3.0), account(), positions)
    assert d.units == 0 and "heat" in d.reason


def test_heat_cap_trims_size() -> None:
    rm = RiskManager(CFG.risk)
    positions = [open_position(23_000.0)]  # 2k headroom under the 25k cap
    d = rm.size_entry(proposal(unit_cost=3.0), account(), positions)
    assert d.units == 6  # limited by heat (2000/300), not budget (10)


def test_correlation_cap_blocks_fourth_same_underlying() -> None:
    rm = RiskManager(CFG.risk)
    positions = [open_position(1_000.0, symbol="SPY") for _ in range(3)]
    d = rm.size_entry(proposal(symbol="SPY"), account(), positions)
    assert d.units == 0 and "correlation" in d.reason


def test_uncorrelated_underlyings_not_blocked() -> None:
    rm = RiskManager(CFG.risk)
    # Synthetic histories: SPY/QQQ share identical returns (corr 1), XOM's
    # returns are an independent random walk (corr ~0).
    import numpy as np

    idx = pd.date_range("2023-06-01", periods=200, freq="B")
    rng = np.random.default_rng(9)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx)
    xom = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx)
    rm.set_history({
        "SPY": pd.DataFrame({"close": spy}),
        "QQQ": pd.DataFrame({"close": spy * 2}),
        "XOM": pd.DataFrame({"close": xom}),
    })
    positions = [open_position(1_000.0, symbol="QQQ") for _ in range(3)]
    blocked = rm.size_entry(proposal(symbol="SPY"), account(), positions)
    assert blocked.units == 0  # SPY vs QQQ corr = 1.0
    allowed = rm.size_entry(proposal(symbol="XOM"), account(), positions)
    assert allowed.units > 0  # anti-correlated name is not crowded out


def test_daily_breaker_halts_entries() -> None:
    rm = RiskManager(CFG.risk)
    rm.record_mark(date(2024, 5, 31), 100_000.0)
    d = rm.size_entry(proposal(), account(equity=91_000), [])  # -9% on the day
    assert d.units == 0 and "daily breaker" in d.reason
    ok = rm.size_entry(proposal(), account(equity=95_000), [])  # -5% is fine
    assert ok.units > 0


def test_weekly_breaker_halts_entries() -> None:
    # Grind-down week: no single day trips the -8% daily halt, but the week's
    # cumulative loss crosses -15% -> weekly breaker fires.
    rm = RiskManager(CFG.risk)
    rm.record_mark(date(2024, 5, 31), 90_000.0)  # Friday before this week
    rm.record_mark(date(2024, 6, 3), 86_000.0)   # Mon -4.4%
    rm.record_mark(date(2024, 6, 4), 82_000.0)   # Tue -4.7%
    wednesday = datetime(2024, 6, 5, 15, 45)
    # Wed intraday 76k: daily 76/82 = -7.3% (below halt), weekly 76/90 = -15.6%.
    d = rm.size_entry(proposal(), account(equity=76_000, when=wednesday), [])
    assert d.units == 0 and "weekly breaker" in d.reason
    ok = rm.size_entry(proposal(), account(equity=80_000, when=wednesday), [])
    assert ok.units > 0  # -11% on the week doesn't halt


def test_hedge_budget_capped() -> None:
    rm = RiskManager(CFG.risk)
    hedge_positions = [open_position(9_900.0, engine="hedge")]  # 9.9k of 10k budget
    d = rm.size_entry(
        proposal(unit_cost=3.0, engine="hedge", risk_fraction=0.05), account(), hedge_positions
    )
    assert d.units == 0 and "hedge budget" in d.reason


def test_no_return_target_anywhere_in_api() -> None:
    """Architectural invariant: nothing in the risk API accepts a target."""
    import inspect

    sig = inspect.signature(RiskManager.size_entry)
    banned = {"target", "goal", "benchmark", "desired_return", "performance"}
    for name in sig.parameters:
        assert not any(b in name.lower() for b in banned)
    from catalyst.core.config import RiskConfig

    for field in RiskConfig.model_fields:
        assert not any(b in field.lower() for b in banned)


def test_property_cash_floor_never_breached_under_random_streams() -> None:
    """Property test: across randomized entry streams, approving every gate
    decision never lets cash dip below the floor."""
    rng = random.Random(42)
    for trial in range(30):
        rm = RiskManager(CFG.risk)
        equity = 100_000.0
        cash = equity
        positions: list[Position] = []
        for step in range(60):
            unit_cost = rng.uniform(0.3, 45.0)
            frac = rng.choice([0.02, 0.03, 0.05, 0.10])
            engine = rng.choice(["engine_a", "engine_b", "engine_c", "hedge"])
            p = proposal(unit_cost=unit_cost, risk_fraction=frac, engine=engine)
            d = rm.size_entry(p, account(equity=equity, cash=cash), positions)
            if d.units > 0:
                cost = unit_cost * 100.0 * d.units
                cash -= cost
                positions.append(open_position(cost, engine=engine))
                # Invariant: the floor survives every approved entry.
                assert cash >= CFG.risk.cash_floor_fraction * equity - 1e-6, (
                    f"floor breached on trial {trial} step {step}"
                )
            # Random partial closes free cash back up.
            if positions and rng.random() < 0.3:
                closed = positions.pop(rng.randrange(len(positions)))
                cash += closed.premium_at_risk * rng.uniform(0.2, 1.8)

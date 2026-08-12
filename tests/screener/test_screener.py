"""Screener: implied move math, liquidity gates, ranking order."""

from __future__ import annotations

from datetime import date, datetime

from catalyst.core.config import load_config
from catalyst.core.models import Catalyst, CatalystType
from catalyst.screener.catalyst_screener import CatalystScreener
from tests.conftest import build_chain

CFG = load_config("backtest")
AS_OF = datetime(2024, 6, 3, 15, 45)


def catalyst_on(d: date) -> Catalyst:
    return Catalyst(
        symbol="SPY", type=CatalystType.OTHER, when=datetime(d.year, d.month, d.day, 16),
        source="test",
    )


def test_implied_move_matches_straddle_over_spot() -> None:
    chain = build_chain()
    screener = CatalystScreener(CFG.screener)
    row = screener.screen(catalyst_on(date(2024, 6, 12)), chain, AS_OF)
    assert row is not None
    call = next(c for c in chain.contracts
                if c.key.strike == 100.0 and c.key.expiry == date(2024, 6, 21) and c.key.right.value == "C")
    put = next(c for c in chain.contracts
               if c.key.strike == 100.0 and c.key.expiry == date(2024, 6, 21) and c.key.right.value == "P")
    assert abs(row.implied_move - (call.mid + put.mid) / 100.0) < 1e-9
    assert row.atm_strike == 100.0
    assert row.event_expiry == date(2024, 6, 21)
    assert row.rejected is None  # ~25 IV straddle over 18d ≈ 4.6% >= 3% gate


def test_liquidity_gates_reject() -> None:
    screener = CatalystScreener(CFG.screener)
    wide = build_chain(spread_frac=0.20)  # 20% spreads >> 8% cap
    row = screener.screen(catalyst_on(date(2024, 6, 12)), wide, AS_OF)
    assert row is not None and row.rejected is not None and "spread" in row.rejected

    thin = build_chain(volume=10)
    row2 = screener.screen(catalyst_on(date(2024, 6, 12)), thin, AS_OF)
    assert row2 is not None and "volume" in (row2.rejected or "")

    no_oi = build_chain(open_interest=5)
    row3 = screener.screen(catalyst_on(date(2024, 6, 12)), no_oi, AS_OF)
    assert row3 is not None and "OI" in (row3.rejected or "")


def test_low_implied_move_rejected() -> None:
    quiet = build_chain(expiry_ivs={date(2024, 6, 21): 0.05, date(2024, 7, 19): 0.05})
    screener = CatalystScreener(CFG.screener)
    row = screener.screen(catalyst_on(date(2024, 6, 12)), quiet, AS_OF)
    assert row is not None and "implied move" in (row.rejected or "")


def test_rank_orders_by_score_and_drops_rejected() -> None:
    screener = CatalystScreener(CFG.screener)
    hot = build_chain(expiry_ivs={date(2024, 6, 21): 0.40, date(2024, 7, 19): 0.35})
    mild = build_chain(expiry_ivs={date(2024, 6, 21): 0.22, date(2024, 7, 19): 0.20})
    dead = build_chain(expiry_ivs={date(2024, 6, 21): 0.04, date(2024, 7, 19): 0.04})
    cat = catalyst_on(date(2024, 6, 12))
    ranked = screener.rank([(cat, mild), (cat, hot), (cat, dead)], AS_OF)
    assert len(ranked) == 2  # dead one rejected on implied move
    assert ranked[0].implied_move > ranked[1].implied_move
    assert ranked[0].score >= ranked[1].score

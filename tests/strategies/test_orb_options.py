"""ORB-options: gates refuse for the stated reason, on a planted market."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from catalyst.core.interfaces.intraday import IntradayContext
from catalyst.core.types import Direction, OptionRight
from catalyst.strategies.active.orb_options import OrbOptionsStrategy, OrbParams

SYMBOL = "QQQ"


def _session_bars(day: date, *, closes: list[float], volume: float = 1_000_000.0,
                  start: time = time(9, 30)) -> pd.DataFrame:
    idx = pd.date_range(datetime.combine(day, start), periods=len(closes), freq="1min")
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 0.02, "low": c - 0.02, "close": c,
         "volume": np.full(len(c), volume)},
        index=idx,
    )


def _warm(strategy: OrbOptionsStrategy, days: int, base: float = 400.0,
          follow_through_pct: float = 1.5, pattern: str = "trend") -> date:
    """Feed complete prior sessions so ATR, volume baseline, warmup AND the
    conditional post-breakout excursion sample all form.

    ``pattern="trend"``   breakouts run (the excursion the gate needs)
    ``pattern="fakeout"`` breakouts REVERSE — same volatility, so the range
                          and ATR gates still pass, but the conditional
                          excursion is ~zero. This is the market where a
                          long option cannot pay for itself.
    """
    day = date(2024, 1, 2)
    run = base * follow_through_pct / 100.0
    for i in range(days):
        d = day + timedelta(days=i)
        closes = [base + (1.5 if j % 2 else 0.0) for j in range(15)]
        if pattern == "trend":
            closes += [base + run * min((j + 1) / 100.0, 1.0) for j in range(375)]
        else:  # fakeout: poke above the range, then sell off all session
            closes += [base + 2.5]
            closes += [base + 2.5 - 7.5 * min((j + 1) / 120.0, 1.0) for j in range(374)]
        bars = _session_bars(d, closes=closes)
        # The engine calls on_minute EVERY minute; the strategy accumulates the
        # session as it goes and consumes it at the next roll. Mirror that with
        # an in-window call and a late one, or the accumulated frame would stop
        # at 09:45 and the excursion window would be truncated.
        for hh, mm in ((9, 45), (15, 59)):
            ctx = IntradayContext(
                session=d, now=datetime.combine(d, time(hh, mm)),
                bars={SYMBOL: bars[bars.index <= datetime.combine(d, time(hh, mm))]},
                option_quote=lambda k: (1.0, 1.02))
            strategy.on_minute(ctx)
    return day + timedelta(days=days)


def _quote_fn(mid: float = 1.00, rel: float = 0.02):
    half = mid * rel / 2.0
    return lambda key: (mid - half, mid + half)


def _breakout_ctx(strategy, day, *, up: bool = True, volume: float = 3_000_000.0,
                  range_width: float = 1.5, run: float = 6.0, quote=None):
    """Tight opening range then a decisive break on heavy volume."""
    base = 400.0
    closes = [base + (range_width if j % 2 else 0.0) for j in range(15)]
    closes += [base + (run if up else -run)] * 16
    bars = _session_bars(day, closes=closes, volume=volume)
    return IntradayContext(session=day, now=datetime.combine(day, time(9, 46)),
                           bars={SYMBOL: bars},
                           option_quote=quote or _quote_fn())


class TestBreakoutDetection:
    def test_upside_break_buys_calls(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, up=True))
        assert len(trades) == 1
        assert trades[0].direction is Direction.LONG
        assert trades[0].legs[0].key.right is OptionRight.CALL

    def test_downside_break_buys_puts(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, up=False))
        assert len(trades) == 1
        assert trades[0].direction is Direction.SHORT
        assert trades[0].legs[0].key.right is OptionRight.PUT

    def test_no_break_is_silent(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, run=0.10))
        assert trades == []
        assert s._gates.get("no_breakout", 0) >= 1

    def test_one_entry_per_session(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        assert len(s.on_minute(_breakout_ctx(s, day))) == 1
        assert s.on_minute(_breakout_ctx(s, day)) == []
        assert s._gates.get("already_fired", 0) >= 1


class TestGatesRefuseForTheStatedReason:
    def test_thin_volume_refuses(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, volume=100.0))
        assert trades == []
        assert s._gates.get("volume_unconfirmed", 0) >= 1

    def test_wide_opening_range_refuses(self) -> None:
        """A range already as wide as a day's move has spent the move."""
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, range_width=50.0))
        assert trades == []
        assert s._gates.get("range_too_wide", 0) >= 1

    def test_wide_option_spread_refuses(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, quote=_quote_fn(rel=0.40)))
        assert trades == []
        assert s._gates.get("no_contract", 0) >= 1

    def test_penny_premium_refuses(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        trades = s.on_minute(_breakout_ctx(s, day, quote=_quote_fn(mid=0.02)))
        assert trades == []
        assert s._gates.get("no_contract", 0) >= 1

    def test_breakeven_gate_refuses_expensive_premium(self) -> None:
        """The signature gate: an option that cannot pay for itself."""
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        # a very expensive option needs a huge move to break even
        trades = s.on_minute(_breakout_ctx(s, day, quote=_quote_fn(mid=40.0)))
        assert trades == []
        assert s._gates.get("breakeven_unmet", 0) >= 1

    def test_warmup_blocks_until_history_exists(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 3)
        assert s.on_minute(_breakout_ctx(s, day)) == []
        assert s._gates.get("warmup", 0) >= 1

    def test_outside_window_is_silent(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        ctx = _breakout_ctx(s, day)
        late = IntradayContext(session=ctx.session,
                               now=datetime.combine(day, time(14, 30)),
                               bars=ctx.bars, option_quote=ctx.option_quote)
        assert s.on_minute(late) == []
        assert s._gates.get("outside_window", 0) >= 1


class TestEconomics:
    def test_entry_cost_exceeds_mid(self) -> None:
        """Never a mid fill: crossing + slippage + commission, always."""
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        t = s.on_minute(_breakout_ctx(s, day, quote=_quote_fn(mid=1.00, rel=0.02)))[0]
        assert t.unit_cost > 1.00
        assert t.rationale["mid"] == pytest.approx(1.00)

    def test_long_option_max_loss_is_the_premium(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        t = s.on_minute(_breakout_ctx(s, day))[0]
        assert t.unit_max_loss == pytest.approx(t.unit_cost)

    def test_exits_are_declared_not_discretionary(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        t = s.on_minute(_breakout_ctx(s, day))[0]
        assert t.exit_rules.stop_loss_pct < 0
        assert t.exit_rules.max_hold_minutes == 120
        assert t.exit_rules.close_by_time == time(15, 45)

    def test_rationale_records_the_breakeven_arithmetic(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        t = s.on_minute(_breakout_ctx(s, day))[0]
        assert t.rationale["expected_move"] > t.rationale["breakeven_move"]


class TestPointInTime:
    def test_premarket_bars_never_enter_the_range(self) -> None:
        """The v16 defect that corrupted a whole pilot: 04:00 bars in RTH stats."""
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        ctx = _breakout_ctx(s, day)
        bars = ctx.bars[SYMBOL]
        pre = _session_bars(day, closes=[999.0] * 60, start=time(4, 0))
        merged = pd.concat([pre, bars]).sort_index()
        polluted = IntradayContext(session=day, now=ctx.now,
                                   bars={SYMBOL: merged},
                                   option_quote=ctx.option_quote)
        s2 = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        _warm(s2, 32)
        trades = s2.on_minute(polluted)
        # the 999 premarket prints must not define the opening range
        assert s2._range_hi is None or s2._range_hi < 500.0


class TestConditionalExcursionGate:
    """The gate's whole point: fair-priced options need a CONDITIONAL edge."""

    def test_trades_when_breakouts_historically_follow_through(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32, follow_through_pct=1.5)
        trades = s.on_minute(_breakout_ctx(s, day))
        assert len(trades) == 1
        assert trades[0].rationale["expected_move"] > trades[0].rationale["breakeven_move"]
        assert trades[0].rationale["excursion_obs"] >= 10

    def test_refuses_when_breakouts_historically_fizzle(self) -> None:
        """Same signal, same option price — but no conditional edge to pay for it.

        The gate is an EXPECTATION over the excursion distribution, so a
        fizzle market fails it by having no right tail at all, not merely by
        having a low median.
        """
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 45, pattern="fakeout")
        assert s.on_minute(_breakout_ctx(s, day)) == []
        assert s._gates.get("breakeven_unmet", 0) >= 1

    def test_unformed_excursion_sample_never_trades(self) -> None:
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15, min_excursion_obs=500))
        day = _warm(s, 32)
        assert s.on_minute(_breakout_ctx(s, day)) == []
        assert s._gates.get("excursion_unformed", 0) >= 1


class TestGreeksAccuracy:
    """The breakeven divides by delta, so delta must be real, not a proxy."""

    def test_delta_is_signed_and_put_deltas_are_negative(self) -> None:
        from catalyst.core.types import OptionRight
        s = OrbOptionsStrategy()
        t = 0.0038                      # ~1 session of trading time
        call = s._delta_of(1.00, 400.0, 400.0, OptionRight.CALL, t)
        put = s._delta_of(1.00, 400.0, 400.0, OptionRight.PUT, t)
        assert call is not None and put is not None
        assert call[0] > 0 and put[0] < 0
        assert abs(call[0]) == pytest.approx(abs(put[0]), abs=0.05)  # ATM symmetry

    def test_atm_delta_is_near_half(self) -> None:
        from catalyst.core.types import OptionRight
        s = OrbOptionsStrategy()
        d = s._delta_of(1.00, 400.0, 400.0, OptionRight.CALL, 0.0038)
        assert d is not None and 0.40 < d[0] < 0.60

    def test_deep_otm_delta_is_small(self) -> None:
        from catalyst.core.types import OptionRight
        s = OrbOptionsStrategy()
        near = s._delta_of(1.00, 400.0, 400.0, OptionRight.CALL, 0.0038)
        far = s._delta_of(0.05, 400.0, 412.0, OptionRight.CALL, 0.0038)
        assert near is not None
        if far is not None:
            assert abs(far[0]) < abs(near[0])

    def test_unpriceable_contract_is_refused_not_guessed(self) -> None:
        from catalyst.core.types import OptionRight
        s = OrbOptionsStrategy()
        assert s._delta_of(0.0, 400.0, 400.0, OptionRight.CALL, 0.0038) is None
        assert s._delta_of(1.00, 400.0, 400.0, OptionRight.CALL, 0.0) is None

    def test_time_to_expiry_counts_trading_minutes_to_the_close(self) -> None:
        """A 0DTE option at 10:00 has ~6 hours of life — not 0 days, not 1."""
        s = OrbOptionsStrategy()
        day = date(2024, 3, 15)
        t_morning = s._t_years(day, datetime.combine(day, time(10, 0)), day)
        t_afternoon = s._t_years(day, datetime.combine(day, time(15, 0)), day)
        assert t_morning > t_afternoon > 0
        assert t_morning == pytest.approx(360 / (390 * 252), rel=1e-6)

    def test_theta_raises_the_required_move(self) -> None:
        """Theta-aware breakeven must exceed the naive premium/delta move."""
        s = OrbOptionsStrategy(OrbParams(warmup_sessions=15))
        day = _warm(s, 32)
        t = s.on_minute(_breakout_ctx(s, day))[0]
        naive = (t.unit_cost / abs(t.rationale["delta"])) / t.rationale["spot"]
        assert t.rationale["breakeven_move"] > naive
        assert t.rationale["theta_decay"] > 0

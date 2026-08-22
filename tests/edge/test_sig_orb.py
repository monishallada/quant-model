"""OpeningRangeBreakout: planted breakout + volume surge fires; the same
breakout on thin volume does not; the opening range and the same-time
volume baseline are computed only from bars visible at the decision instant
(prior sessions only for the 21-session median). One engine-level test runs
the signal through the sanctioned run_signal path (trial line BEFORE the
backtest). All data is synthetic, no network, every timestamp is strictly
before the 2026-02-22 lockbox start, and every write lands under tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

from edge.core.config import (
    DataConfig,
    EdgeConfig,
    ExecutionConfig,
    RiskConfig,
    ValidationConfig,
)
from edge.core.events import BarEvent, Side
from edge.data.loader import EdgeDataLoader
from edge.regime.classifier import BUCKETS
from edge.research.registry import TrialRegistry
from edge.runners.engine import BacktestEngine, EngineParams
from edge.signals import registry as sigreg
from edge.signals.base import MIN_HYPOTHESIS_CHARS, SignalContext
from edge.signals.orb import (
    ORB_HYPOTHESIS,
    OpeningRangeBreakout,
    OpeningRangeBreakoutConfig,
)
from edge.signals.registry import run_signal

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

SYM = "ORB"
#: The signal session (a Monday), strictly before the 2026-02-22 lockbox.
SIGNAL_DAY = date(2025, 6, 30)
#: End of the 30-minute opening range on the default config.
OR_END_TOD = time(10, 0)
#: Default breakout instant: 65 minutes after the open, inside the 30-90 window.
BREAKOUT_TOD = time(10, 35)


# ---------------------------------------------------------------------------
# Synthetic tape: 5-minute bars, flat priors, one planted breakout session
# ---------------------------------------------------------------------------


def _grid(last_tod: time) -> list[time]:
    """Bar close times 09:35, 09:40, ... up to and including ``last_tod``."""
    out: list[time] = []
    minute = 9 * 60 + 35
    last = last_tod.hour * 60 + last_tod.minute
    while minute <= last:
        out.append(time(minute // 60, minute % 60))
        minute += 5
    return out


def _bar(
    day: date, tod: time, o: float, h: float, lo: float, c: float, v: int
) -> BarEvent:
    return BarEvent(
        ts=datetime.combine(day, tod, tzinfo=ET),
        symbol=SYM,
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=v,
    )


def build_tape(
    *,
    n_prior: int = 21,
    prior_vol: int = 1000,
    prior_vols: list[int] | None = None,
    signal_vol: int = 3000,
    breakout_close: float | None = 101.0,
    breakout_tod: time = BREAKOUT_TOD,
    post_hold_bars: int = 0,
) -> tuple[list[BarEvent], BarEvent]:
    """(all bars, the bar closing at ``breakout_tod`` on the signal session).

    Prior sessions are flat 100.0 bars (high 100.5 / low 99.5) with
    ``prior_vols`` (oldest->newest) or a constant ``prior_vol`` per bar.
    The signal session repeats the same opening range, drifts inside it at
    100.2, then closes at ``breakout_close`` on the ``breakout_tod`` bar
    (``None`` plants NO breakout: the bar stays inside the range).
    ``post_hold_bars`` appends bars holding beyond the range (no re-cross).
    """
    days = [d.date() for d in pd.bdate_range(end=SIGNAL_DAY, periods=n_prior + 1)]
    prior_days, sig_day = days[:-1], days[-1]
    vols = prior_vols if prior_vols is not None else [prior_vol] * n_prior
    assert len(vols) == n_prior
    final_minute = breakout_tod.hour * 60 + breakout_tod.minute + 5 * post_hold_bars
    final_tod = time(final_minute // 60, final_minute % 60)

    bars: list[BarEvent] = []
    for day, vol in zip(prior_days, vols):
        for tod in _grid(final_tod):
            bars.append(_bar(day, tod, 100.0, 100.5, 99.5, 100.0, vol))

    anchor: BarEvent | None = None
    for tod in _grid(final_tod):
        if tod <= OR_END_TOD:
            b = _bar(sig_day, tod, 100.0, 100.5, 99.5, 100.0, signal_vol)
        elif tod == breakout_tod:
            if breakout_close is None:
                b = _bar(sig_day, tod, 100.2, 100.4, 100.0, 100.2, signal_vol)
            else:
                hi = max(100.2, breakout_close) + 0.2
                lo = min(100.2, breakout_close) - 0.2
                b = _bar(sig_day, tod, 100.2, hi, lo, breakout_close, signal_vol)
            anchor = b
        elif tod < breakout_tod:
            b = _bar(sig_day, tod, 100.2, 100.4, 100.0, 100.2, signal_vol)
        else:  # hold beyond the range without re-crossing it
            assert breakout_close is not None
            hold = breakout_close + 0.1
            b = _bar(
                sig_day, tod, hold, hold + 0.1, breakout_close - 0.1, hold, signal_vol
            )
        bars.append(b)
    assert anchor is not None
    return bars, anchor


class EngineCtx:
    """Engine-shaped ctx: ``.now``, ``.bar``, ``history() -> tuple[BarEvent]``.

    Deliberately serves WHATEVER bars it was given — including future ones —
    so tests can prove the signal itself enforces ``ts <= now``.
    """

    def __init__(self, bars: list[BarEvent], bar: BarEvent) -> None:
        self._bars = list(bars)
        self.bar = bar
        self.now = bar.ts

    def history(self, symbol: str | None = None, n: int | None = None):
        rows = [b for b in self._bars if b.symbol == (symbol or self.bar.symbol)]
        return tuple(rows if n is None else rows[-n:])


def fire(bars: list[BarEvent], at: BarEvent):
    return OpeningRangeBreakout().on_bar(EngineCtx(bars, at))


def bar_at(bars: list[BarEvent], tod: time) -> BarEvent:
    ts = datetime.combine(SIGNAL_DAY, tod, tzinfo=ET)
    return next(b for b in bars if b.ts == ts)


# ---------------------------------------------------------------------------
# Declarations: registration, hypothesis, regimes, config
# ---------------------------------------------------------------------------


def test_registered_under_its_name() -> None:
    assert sigreg.get("opening_range_breakout") is OpeningRangeBreakout
    assert OpeningRangeBreakout.name == "opening_range_breakout"


def test_hypothesis_is_admissible_and_names_the_mechanism() -> None:
    assert OpeningRangeBreakout.hypothesis == ORB_HYPOTHESIS
    assert len(ORB_HYPOTHESIS.strip()) >= MIN_HYPOTHESIS_CHARS
    # Mechanism, counterparty, regime restriction, fixed thresholds — all stated.
    for phrase in ("first 30 minutes", "1.5x", "prior 21 sessions", "passive"):
        assert phrase in ORB_HYPOTHESIS
    assert "chop" in ORB_HYPOTHESIS  # says where the edge dies


def test_trades_trending_buckets_only() -> None:
    allowed = OpeningRangeBreakout.allowed_regimes
    assert allowed == {"high_vol_trending", "low_vol_trending"}
    assert set(allowed) <= BUCKETS  # real classifier vocabulary, no typos
    assert all(bucket.endswith("_trending") for bucket in allowed)


def test_config_forbids_extras() -> None:
    with pytest.raises(ValidationError):
        OpeningRangeBreakoutConfig(bogus_knob=1)  # type: ignore[call-arg]


def test_config_rejects_incoherent_windows() -> None:
    with pytest.raises(ValidationError):
        # Entering before the range is complete is meaningless.
        OpeningRangeBreakoutConfig(entry_start_minutes=10)
    with pytest.raises(ValidationError):
        OpeningRangeBreakoutConfig(entry_end_minutes=30)  # end <= start
    cfg = OpeningRangeBreakoutConfig()  # the hypothesis's fixed choices
    assert cfg.opening_range_minutes == 30
    assert (cfg.entry_start_minutes, cfg.entry_end_minutes) == (30, 90)
    assert cfg.rel_vol_threshold == 1.5
    assert cfg.volume_lookback_sessions == 21
    assert cfg.horizon_minutes == 120


# ---------------------------------------------------------------------------
# The planted breakout: volume surge fires, thin volume does not
# ---------------------------------------------------------------------------


def test_planted_breakout_with_volume_surge_fires_long() -> None:
    bars, anchor = build_tape()  # rel_vol = 3000/1000 = 3.0
    events = fire(bars, anchor)
    assert len(events) == 1
    evt = events[0]
    assert evt.side is Side.BUY
    assert evt.symbol == SYM
    assert evt.signal_name == "opening_range_breakout"
    assert evt.horizon_minutes == 120
    assert evt.ts == anchor.ts  # stamped at the decision instant
    # conviction = min(1, rel_vol / 3.0) = 1.0 at rel_vol 3.0
    assert evt.conviction == pytest.approx(1.0)


def test_same_breakout_on_thin_volume_is_silent() -> None:
    bars, anchor = build_tape(signal_vol=1000)  # rel_vol = 1.0
    assert fire(bars, anchor) == []


def test_rel_vol_exactly_at_threshold_is_silent() -> None:
    # The trigger is STRICT: rel_vol must exceed 1.5, not merely reach it.
    bars, anchor = build_tape(signal_vol=1500)  # rel_vol = 1.5 exactly
    assert fire(bars, anchor) == []


def test_conviction_scales_with_rel_vol() -> None:
    bars, anchor = build_tape(signal_vol=2000)  # rel_vol = 2.0
    (evt,) = fire(bars, anchor)
    assert evt.conviction == pytest.approx(2.0 / 3.0)  # rel_vol / (2 * 1.5)
    bars_hot, anchor_hot = build_tape(signal_vol=3000)  # rel_vol = 3.0 -> cap
    (evt_hot,) = fire(bars_hot, anchor_hot)
    assert evt_hot.conviction == pytest.approx(1.0)
    assert evt_hot.conviction > evt.conviction


def test_breakdown_below_range_low_fires_short() -> None:
    bars, anchor = build_tape(breakout_close=99.0)  # ORL = 99.5
    (evt,) = fire(bars, anchor)
    assert evt.side is Side.SELL
    assert evt.conviction == pytest.approx(1.0)


def test_volume_surge_without_breakout_is_silent() -> None:
    # Abnormal volume alone is not the signal: no range break, no opinion.
    bars, anchor = build_tape(breakout_close=None)
    assert fire(bars, anchor) == []


# ---------------------------------------------------------------------------
# The entry window: only 30-90 minutes after the open
# ---------------------------------------------------------------------------


def test_no_entry_before_window_opens() -> None:
    bars, _ = build_tape()
    early = bar_at(bars, time(9, 55))  # 25 minutes after the open
    assert fire(bars, early) == []


def test_opening_range_bar_cannot_break_its_own_range() -> None:
    # The 10:00 bar is minute 30 (window edge) but is itself part of the
    # range: its close can never exceed the range extremes it helped set.
    bars, _ = build_tape()
    assert fire(bars, bar_at(bars, OR_END_TOD)) == []


def test_no_entry_after_window_closes() -> None:
    bars, anchor = build_tape(breakout_tod=time(11, 5))  # 95 min > 90
    assert fire(bars, anchor) == []


def test_entry_at_window_edge_still_fires() -> None:
    bars, anchor = build_tape(breakout_tod=time(11, 0))  # exactly 90 min
    (evt,) = fire(bars, anchor)
    assert evt.side is Side.BUY


# ---------------------------------------------------------------------------
# Point-in-time: visible bars only, prior sessions only
# ---------------------------------------------------------------------------


def test_range_and_volume_use_only_visible_bars() -> None:
    """A ctx leaking future bars must change NOTHING about the decision."""
    bars, anchor = build_tape(signal_vol=2000)  # rel_vol 2.0 -> conviction 2/3
    next_day = date(2025, 7, 1)
    leaked = bars + [
        # Later the same session: huge volume and a wild range. If either
        # leaked into today's cumulative volume or the range, conviction
        # (or the fire itself) would change.
        _bar(SIGNAL_DAY, time(10, 40), 100.0, 200.0, 50.0, 150.0, 1_000_000),
        # The NEXT session's opening-range window: same time-of-day slot as
        # the range — a naive time-of-day pool would corrupt ORH/ORL.
        _bar(next_day, time(9, 40), 100.0, 300.0, 10.0, 200.0, 1_000_000),
        _bar(next_day, time(9, 45), 100.0, 300.0, 10.0, 200.0, 1_000_000),
    ]
    clean = fire(bars, anchor)
    with_future = fire(leaked, anchor)
    assert len(clean) == len(with_future) == 1
    assert with_future[0].side is Side.BUY
    assert with_future[0].conviction == pytest.approx(clean[0].conviction)
    assert with_future[0].conviction == pytest.approx(2.0 / 3.0)


def test_future_volume_cannot_rescue_a_thin_tape() -> None:
    bars, anchor = build_tape(signal_vol=1000)  # thin: rel_vol = 1.0
    leaked = bars + [
        _bar(SIGNAL_DAY, time(10, 40), 100.0, 100.5, 99.5, 100.2, 1_000_000)
    ]
    assert fire(leaked, anchor) == []


def test_median_uses_prior_sessions_only() -> None:
    """Today's own tape must never enter its own volume baseline.

    Priors (oldest->newest): 11 sessions at 1000/bar, 10 at 4000/bar; today
    3000/bar. Correct (prior-only) median = 13 bars x 1000 = 13000 ->
    rel_vol = 39000/13000 = 3.0 -> fires at full conviction. Had today's
    39000 entered the pool and evicted the oldest sample, the median would
    be 39000 -> rel_vol 1.0 -> silence. Firing at conviction 1.0 proves the
    baseline is prior sessions only.
    """
    bars, anchor = build_tape(prior_vols=[1000] * 11 + [4000] * 10, signal_vol=3000)
    (evt,) = fire(bars, anchor)
    assert evt.side is Side.BUY
    assert evt.conviction == pytest.approx(1.0)


def test_insufficient_history_is_silent() -> None:
    # 5 prior sessions < the 21 the median is defined over: no shortcut.
    bars, anchor = build_tape(n_prior=5)
    assert fire(bars, anchor) == []


def test_no_refire_while_price_holds_beyond_range() -> None:
    # The bar AFTER the breakout closes beyond the range but does not cross
    # it (previous close already outside): no fresh initiative, no signal.
    bars, _ = build_tape(post_hold_bars=1)
    hold = bar_at(bars, time(10, 40))
    assert hold.close > 100.5  # still beyond ORH
    assert fire(bars, hold) == []


# ---------------------------------------------------------------------------
# SignalContext protocol flavor (DataFrame history + decision_ts)
# ---------------------------------------------------------------------------


class FrameCtx:
    """Protocol-shaped ctx: ``decision_ts`` and a DataFrame ``history``."""

    def __init__(self, bars: list[BarEvent], bar: BarEvent) -> None:
        self._bars = bars
        self._bar = bar

    @property
    def decision_ts(self) -> datetime:
        return self._bar.ts

    @property
    def bar(self) -> BarEvent:
        return self._bar

    def history(self, symbol: str, n_bars: int) -> pd.DataFrame:
        rows = [b for b in self._bars if b.symbol == symbol][-n_bars:]
        return pd.DataFrame(
            {
                "ts": [b.ts for b in rows],
                "open": [b.open for b in rows],
                "high": [b.high for b in rows],
                "low": [b.low for b in rows],
                "close": [b.close for b in rows],
                "volume": [b.volume for b in rows],
            }
        )

    def frame(self, kind: str, symbol: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


def test_protocol_dataframe_ctx_is_supported() -> None:
    bars, anchor = build_tape()
    ctx = FrameCtx(bars, anchor)
    assert isinstance(ctx, SignalContext)
    (evt,) = OpeningRangeBreakout().on_bar(ctx)
    assert evt.side is Side.BUY
    assert evt.ts == anchor.ts
    assert evt.conviction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Engine-level: the sanctioned run_signal path on the same planted tape
# ---------------------------------------------------------------------------


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def _frames_from(bars: list[BarEvent]) -> dict[tuple[str, str], pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "ts": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    quotes = pd.DataFrame(
        {
            "ts": frame["ts"],
            "bid": frame["close"] - 0.05,
            "ask": frame["close"] + 0.05,
            "bid_size": 500,
            "ask_size": 500,
        }
    )
    return {(SYM, "bars"): frame, (SYM, "quotes"): quotes}


CONFIG = EdgeConfig(
    data=DataConfig(lockbox_start=date(2026, 2, 22)),
    validation=ValidationConfig(min_oos_trades=300, pbo_max=0.5, bootstrap_resamples=200),
    execution=ExecutionConfig(
        spread_fill_fraction=1.0,
        slippage_pct=0.0,
        commission_per_contract=0.01,
        latency_ms=[100],
    ),
    risk=RiskConfig(
        kelly_fraction=0.25,
        per_trade_cap_pct=2.0,
        daily_loss_halt_pct=3.0,
        target_daily_vol_pct=None,
    ),
)


def test_engine_run_through_run_signal_records_trial_first(tmp_path: Path) -> None:
    bars, _ = build_tape(post_hold_bars=5)  # bars after the break so entry fills
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "edge.yaml").write_text(
        (REPO_ROOT / "config" / "edge.yaml").read_text()
    )
    loader = EdgeDataLoader(TapeBackend(_frames_from(bars)), repo_root=root)
    trials_root = tmp_path / "trials"
    trials_root.mkdir()
    trials = TrialRegistry(root=trials_root)
    params = EngineParams(
        symbols=(SYM,),
        start=bars[0].ts.date(),
        end=SIGNAL_DAY,
        latency_ms=100,
        initial_equity=100_000.0,
        seed=0,
        risk_r_pct=1.0,
        max_hold_minutes=120,
    )
    signal = OpeningRangeBreakout()
    seen_during_backtest: list[int] = []

    def backtest(sig: OpeningRangeBreakout):
        # The trial line must already exist while the engine is running.
        seen_during_backtest.append(trials.cumulative_trial_count())
        return BacktestEngine(loader, CONFIG, params, [sig]).run()

    record, result = run_signal(
        signal,
        backtest,
        trials,
        run_config={
            "symbols": [SYM],
            "start": str(bars[0].ts.date()),
            "end": str(SIGNAL_DAY),
            "latency_ms": [100],
        },
        ts=datetime(2026, 1, 15, 10, 0, tzinfo=ET),
    )
    assert seen_during_backtest == [1]
    assert record.hypothesis == ORB_HYPOTHESIS
    assert record.result == {"status": "pending"}
    # Exactly the one planted breakout fired and executed; the identity holds.
    assert result.emitted == 1
    assert result.executed == 1
    assert result.emitted == result.executed + sum(result.drop_counters.values())
    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    assert row["signal_name"] == OpeningRangeBreakout.name
    assert row["side"] == "buy"
    assert row["conviction"] == pytest.approx(1.0)

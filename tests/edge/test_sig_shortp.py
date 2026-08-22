"""ShortRatioDeviation signal tests: canned short_ratio frames with planted
z patterns. Proves: a 3-day sustained z>2 rise fires exactly once at the
next open after the pattern (the session where the last ratio first becomes
visible at pre-open); same-day ratios are NEVER visible (next-preopen
``available_at`` stamps keep them out — shown both through the real engine
context and by an explicit anti-conservative-stamp counterfactual); a
single-day spike never fires; dedup is one emission per newly visible
pattern day; the multi-day hold runs through the real BacktestEngine via
``EngineParams.max_hold_minutes``; and the trial is recorded through
``run_signal`` BEFORE the engine runs.

ALL data is synthetic, no network, every timestamp strictly predates the
2026-02-22 lockbox wall, and every write lands under pytest's tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

import edge.signals.registry as sigreg
from edge.core.config import (
    DataConfig,
    EdgeConfig,
    ExecutionConfig,
    RiskConfig,
    ValidationConfig,
)
from edge.core.events import BarEvent, Side, SignalEvent
from edge.data.loader import ALL_SYMBOLS, EdgeDataLoader
from edge.regime.classifier import BUCKETS, gate
from edge.research.registry import TrialRegistry
from edge.runners.engine import BacktestEngine, EngineParams, SizeDecision
from edge.signals.base import require_hypothesis
from edge.signals.registry import run_signal
from edge.signals.short_pressure import (
    MINUTES_PER_DAY,
    SHORT_RATIO_KIND,
    ShortRatioDeviation,
    ShortRatioDeviationConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
LATENCY_MS = 100
LATENCY = timedelta(milliseconds=LATENCY_MS)

# Synthetic January 2026 sessions, all weekdays, all WELL before the
# 2026-02-22 lockbox wall. Jan 2 = Fri, 5-9 = Mon-Fri, 12-13 = Mon-Tue.
JAN2, JAN5, JAN6, JAN7 = (date(2026, 1, d) for d in (2, 5, 6, 7))
JAN8, JAN9, JAN12, JAN13 = (date(2026, 1, d) for d in (8, 9, 12, 13))
SESSIONS = [JAN5, JAN6, JAN7, JAN8, JAN9, JAN12, JAN13]


def T(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)


def preopen(day: date) -> pd.Timestamp:
    """The FINRA feed's real stamp: next-session pre-open, 09:00 ET."""
    return pd.Timestamp(T(day, 9, 0))


# ---------------------------------------------------------------------------
# Canned frames
# ---------------------------------------------------------------------------


def ratio_frame(rows: list[tuple[str, date, float, pd.Timestamp]]) -> pd.DataFrame:
    """(symbol, asof_date, short_ratio_z, available_at) -> feed-shaped frame.

    Carries the real feed's point-in-time columns; the raw ``short_ratio``
    level is a constant on purpose — levels are meaningless, deviations are
    the signal.
    """
    return pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(r[1]) for r in rows],
            "symbol": [r[0] for r in rows],
            "short_ratio": 0.55,
            "short_ratio_z": [r[2] for r in rows],
            "available_at": [r[3] for r in rows],
        }
    )


def pattern_frame() -> pd.DataFrame:
    """The planted 3-day z-rise: AAA z>2 on Jan 5/6/7, published next preopen.

    The pattern completes with the Jan 7 row, first visible Jan 8 09:00 ET
    — so the RIGHT firing session is Jan 8. Later rows break the pattern
    (no refire). BBB rows are a DECOY with huge sustained z: a signal that
    failed to filter by symbol would pollute AAA's window with them.
    """
    return ratio_frame(
        [
            ("AAA", JAN2, 0.5, preopen(JAN5)),
            ("AAA", JAN5, 2.5, preopen(JAN6)),
            ("AAA", JAN6, 2.8, preopen(JAN7)),
            ("AAA", JAN7, 3.1, preopen(JAN8)),  # pattern day: visible Jan 8
            ("AAA", JAN8, 1.0, preopen(JAN9)),  # broken: no refire
            ("AAA", JAN9, 0.8, preopen(JAN12)),
            ("AAA", JAN12, 0.9, preopen(JAN13)),
            ("BBB", JAN2, 9.0, preopen(JAN5)),  # decoys (symbol isolation)
            ("BBB", JAN5, 9.1, preopen(JAN6)),
            ("BBB", JAN6, 9.2, preopen(JAN7)),
        ]
    )


def session_bars(days: list[date], price: float = 100.0) -> pd.DataFrame:
    stamps = [T(day, 9, m) for day in days for m in (31, 32, 33, 34, 35)]
    return pd.DataFrame(
        {
            "ts": stamps,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100,
        }
    )


def quotes_from_bars(bars: pd.DataFrame, half_spread: float = 0.10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": bars["ts"],
            "bid": bars["close"] - half_spread,
            "ask": bars["close"] + half_spread,
            "bid_size": 50,
            "ask_size": 50,
        }
    )


def bar(day: date, hh: int, mm: int, symbol: str = "AAA") -> BarEvent:
    return BarEvent(
        ts=T(day, hh, mm), symbol=symbol,
        open=100.0, high=100.2, low=99.8, close=100.0, volume=100,
    )


# ---------------------------------------------------------------------------
# Contexts: engine-shaped fake (pit + now) and base-protocol fake
# (frame + decision_ts). The fakes enforce the SAME visibility join the
# engine's MarketContext does: available_at <= decision instant.
# ---------------------------------------------------------------------------


class EngineShapedCtx:
    """Mirrors MarketContext: ``now``, ``bar``, ``pit`` with the PIT join."""

    def __init__(self, b: BarEvent, frame: pd.DataFrame) -> None:
        self.bar = b
        self.now = b.ts
        self._frame = frame

    def pit(self, kind: str, symbol: str | None = None) -> pd.DataFrame:
        assert kind == SHORT_RATIO_KIND
        visible = self._frame[self._frame["available_at"] <= pd.Timestamp(self.now)]
        if symbol is not None and "symbol" in visible.columns:
            visible = visible[visible["symbol"] == symbol]
        return visible.reset_index(drop=True).copy()


class BaseProtocolCtx:
    """The base SignalContext shape: ``decision_ts``, ``bar``, ``frame``."""

    def __init__(self, b: BarEvent, frame: pd.DataFrame) -> None:
        self._bar = b
        self._frame = frame

    @property
    def decision_ts(self) -> datetime:
        return self._bar.ts

    @property
    def bar(self) -> BarEvent:
        return self._bar

    def history(self, symbol: str, n_bars: int):  # pragma: no cover - unused
        raise NotImplementedError

    def frame(self, kind: str, symbol: str) -> pd.DataFrame:
        assert kind == SHORT_RATIO_KIND
        visible = self._frame[
            self._frame["available_at"] <= pd.Timestamp(self._bar.ts)
        ]
        return visible[visible["symbol"] == symbol].reset_index(drop=True).copy()


def sweep_sessions(
    sig: ShortRatioDeviation, frame: pd.DataFrame, days: list[date]
) -> list[tuple[datetime, SignalEvent]]:
    """Run on_bar over every planted bar of every session; collect emissions."""
    out: list[tuple[datetime, SignalEvent]] = []
    for day in days:
        for mm in (31, 32, 33, 34, 35):
            b = bar(day, 9, mm)
            for event in sig.on_bar(EngineShapedCtx(b, frame)):
                out.append((b.ts, event))
    return out


# ---------------------------------------------------------------------------
# Engine plumbing (mirrors tests/edge/test_engine.py)
# ---------------------------------------------------------------------------


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


class FixedSizer:
    """Deterministic test sizer: fixed qty and risk unit."""

    def __init__(self, qty: int, risk_per_share: float) -> None:
        self.qty = qty
        self.risk_per_share = risk_per_share

    def size(self, signal: SignalEvent, ref_price: float, equity: float) -> SizeDecision:
        return SizeDecision(qty=self.qty, risk_per_share=self.risk_per_share)


def make_config() -> EdgeConfig:
    return EdgeConfig(
        data=DataConfig(lockbox_start=date(2026, 2, 22)),
        validation=ValidationConfig(min_oos_trades=1, pbo_max=0.5, bootstrap_resamples=10),
        execution=ExecutionConfig(
            spread_fill_fraction=1.0,
            slippage_pct=0.0,
            commission_per_contract=0.01,
            latency_ms=[LATENCY_MS],
        ),
        risk=RiskConfig(
            kelly_fraction=0.25,
            per_trade_cap_pct=2.0,
            daily_loss_halt_pct=50.0,
            target_daily_vol_pct=None,
        ),
    )


def make_engine(
    tmp_path: Path,
    sig: ShortRatioDeviation,
    *,
    end: date,
    ratio: pd.DataFrame | None = None,
) -> BacktestEngine:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "edge.yaml").write_text((REPO_ROOT / "config" / "edge.yaml").read_text())
    days = [d for d in SESSIONS if d <= end]
    bars = session_bars(days)
    frames = {
        ("AAA", "bars"): bars,
        ("AAA", "quotes"): quotes_from_bars(bars),
        (ALL_SYMBOLS, SHORT_RATIO_KIND): pattern_frame() if ratio is None else ratio,
    }
    params = EngineParams(
        symbols=("AAA",),
        start=SESSIONS[0],
        end=end,
        latency_ms=LATENCY_MS,
        initial_equity=100_000.0,
        seed=0,
        risk_r_pct=1.0,
        # The platform's first multi-day hold: the engine's MaxHoldExit
        # measures wall-clock minutes, wired from the signal's own horizon.
        max_hold_minutes=sig.max_hold_minutes(),
        pit_kinds=(SHORT_RATIO_KIND,),
    )
    loader = EdgeDataLoader(TapeBackend(frames), repo_root=tmp_path)
    return BacktestEngine(
        loader, make_config(), params, [sig], sizer=FixedSizer(10, 1.0)
    )


# ---------------------------------------------------------------------------
# Contract: registration, hypothesis, regimes, config
# ---------------------------------------------------------------------------


def test_registered_under_its_name_with_admissible_hypothesis() -> None:
    assert sigreg.get("short_ratio_deviation") is ShortRatioDeviation
    assert require_hypothesis(ShortRatioDeviation.hypothesis) == ShortRatioDeviation.hypothesis
    # The fixed economic choices are stated in the hypothesis itself.
    for stated in ("above 2", "3+ consecutive", "Long-only", "1-3 days", "high-vol"):
        assert stated in ShortRatioDeviation.hypothesis


def test_allowed_regimes_is_all_but_high_vol_and_stays_in_sync() -> None:
    expected = {b for b in BUCKETS if not b.startswith("high_vol")}
    assert set(ShortRatioDeviation.allowed_regimes) == expected
    # The classifier's engine hook honors the declaration.
    assert gate(ShortRatioDeviation.allowed_regimes, "low_vol_trending") is True
    assert gate(ShortRatioDeviation.allowed_regimes, "low_vol_chopping") is True
    assert gate(ShortRatioDeviation.allowed_regimes, "high_vol_trending") is False
    assert gate(ShortRatioDeviation.allowed_regimes, "high_vol_chopping") is False


def test_config_defaults_are_the_stated_choices_and_forbid_drift() -> None:
    cfg = ShortRatioDeviationConfig()
    assert cfg.z_threshold == 2.0
    assert cfg.min_sustained_days == 3
    assert cfg.horizon_days == 3
    with pytest.raises(ValidationError):
        ShortRatioDeviationConfig(z_thresh=2.5)  # typo'd knob: error, not silence
    with pytest.raises(ValidationError):
        ShortRatioDeviationConfig().__setattr__("z_threshold", 3.0)  # frozen
    with pytest.raises(ValidationError):
        ShortRatioDeviationConfig(horizon_days=4)  # outside the stated 1-3 band
    with pytest.raises(ValidationError):
        ShortRatioDeviationConfig(min_sustained_days=1)  # multi-day by definition


def test_max_hold_minutes_is_the_wall_clock_horizon() -> None:
    assert ShortRatioDeviation().max_hold_minutes() == 3 * MINUTES_PER_DAY  # 4320
    one_day = ShortRatioDeviation(ShortRatioDeviationConfig(horizon_days=1))
    assert one_day.max_hold_minutes() == MINUTES_PER_DAY


# ---------------------------------------------------------------------------
# The planted 3-day z-rise fires at the RIGHT session (next open after the
# pattern), once, long-only
# ---------------------------------------------------------------------------


def test_three_day_rise_fires_at_next_open_after_pattern() -> None:
    sig = ShortRatioDeviation()
    emissions = sweep_sessions(sig, pattern_frame(), SESSIONS)
    assert len(emissions) == 1
    fired_at, event = emissions[0]
    # The Jan 7 row (pattern completer) is stamped available Jan 8 09:00 —
    # the FIRST bar of Jan 8 is the next open after the pattern.
    assert fired_at == T(JAN8, 9, 31)
    assert event.ts == T(JAN8, 9, 31)
    assert event.symbol == "AAA"
    assert event.side is Side.BUY  # long-only by construction
    assert event.conviction == 1.0
    assert event.horizon_minutes == 3 * MINUTES_PER_DAY
    assert event.signal_name == "short_ratio_deviation"


def test_dedup_is_one_emission_per_newly_visible_pattern_day() -> None:
    # Pattern persists an extra day: Jan 8's z is ALSO > 2, so a NEW pattern
    # day becomes visible Jan 9 — one emission per pattern day, per session.
    frame = ratio_frame(
        [
            ("AAA", JAN5, 2.5, preopen(JAN6)),
            ("AAA", JAN6, 2.8, preopen(JAN7)),
            ("AAA", JAN7, 3.1, preopen(JAN8)),
            ("AAA", JAN8, 2.6, preopen(JAN9)),
        ]
    )
    sig = ShortRatioDeviation()
    emissions = sweep_sessions(sig, frame, [JAN7, JAN8, JAN9])
    assert [ts for ts, _ in emissions] == [T(JAN8, 9, 31), T(JAN9, 9, 31)]
    # Same ctx replayed again: the pattern day was acted on — no re-emission.
    assert sig.on_bar(EngineShapedCtx(bar(JAN9, 9, 31), frame)) == []


def test_base_protocol_ctx_path_fires_identically() -> None:
    sig = ShortRatioDeviation()
    assert sig.on_bar(BaseProtocolCtx(bar(JAN7, 9, 31), pattern_frame())) == []
    events = sig.on_bar(BaseProtocolCtx(bar(JAN8, 9, 31), pattern_frame()))
    assert len(events) == 1 and events[0].side is Side.BUY


# ---------------------------------------------------------------------------
# Same-day ratios are INVISIBLE (available_at = next preopen governs)
# ---------------------------------------------------------------------------


def test_same_day_ratio_is_invisible_during_its_own_session() -> None:
    sig = ShortRatioDeviation()
    # During EVERY bar of Jan 7 the asof-Jan-7 row (which would complete the
    # pattern) exists in the frame but is stamped available Jan 8 09:00 —
    # the signal must not fire anywhere in the Jan 7 session.
    assert sweep_sessions(sig, pattern_frame(), [JAN5, JAN6, JAN7]) == []


def test_only_the_next_preopen_stamp_keeps_the_same_day_row_out() -> None:
    # Counterfactual with a deliberately ANTI-CONSERVATIVE stamp (never the
    # real feed's behavior): were the Jan 7 row visible same-day, the
    # pattern WOULD fire on Jan 7 — proving the pattern was complete and
    # that visibility, not the rule, is what defers the trade to Jan 8.
    frame = ratio_frame(
        [
            ("AAA", JAN5, 2.5, preopen(JAN6)),
            ("AAA", JAN6, 2.8, preopen(JAN7)),
            ("AAA", JAN7, 3.1, pd.Timestamp(T(JAN7, 8, 0))),  # same-day (wrong)
        ]
    )
    events = ShortRatioDeviation().on_bar(EngineShapedCtx(bar(JAN7, 9, 31), frame))
    assert len(events) == 1  # fires a session early ONLY under the bad stamp


# ---------------------------------------------------------------------------
# Non-patterns never fire
# ---------------------------------------------------------------------------


def test_single_day_spike_does_not_fire() -> None:
    frame = ratio_frame(
        [
            ("AAA", JAN2, 0.3, preopen(JAN5)),
            ("AAA", JAN5, 0.4, preopen(JAN6)),
            ("AAA", JAN6, 3.5, preopen(JAN7)),  # one-day spike
            ("AAA", JAN7, 0.4, preopen(JAN8)),
            ("AAA", JAN8, 0.2, preopen(JAN9)),
        ]
    )
    assert sweep_sessions(ShortRatioDeviation(), frame, SESSIONS) == []


def test_two_day_elevation_does_not_fire() -> None:
    frame = ratio_frame(
        [
            ("AAA", JAN2, 0.5, preopen(JAN5)),
            ("AAA", JAN5, 2.5, preopen(JAN6)),
            ("AAA", JAN6, 2.8, preopen(JAN7)),  # only 2 sustained days
            ("AAA", JAN7, 0.4, preopen(JAN8)),
        ]
    )
    assert sweep_sessions(ShortRatioDeviation(), frame, SESSIONS) == []


def test_z_exactly_at_threshold_does_not_fire() -> None:
    frame = ratio_frame(
        [
            ("AAA", JAN5, 2.0, preopen(JAN6)),  # strict >: 2.0 is not > 2.0
            ("AAA", JAN6, 2.0, preopen(JAN7)),
            ("AAA", JAN7, 2.0, preopen(JAN8)),
        ]
    )
    assert sweep_sessions(ShortRatioDeviation(), frame, SESSIONS) == []


def test_nan_z_in_window_does_not_fire() -> None:
    frame = ratio_frame(
        [
            ("AAA", JAN5, 2.5, preopen(JAN6)),
            ("AAA", JAN6, float("nan"), preopen(JAN7)),  # warmup gap
            ("AAA", JAN7, 3.1, preopen(JAN8)),
        ]
    )
    assert sweep_sessions(ShortRatioDeviation(), frame, SESSIONS) == []


def test_fewer_rows_than_sustained_days_does_not_fire() -> None:
    frame = ratio_frame([("AAA", JAN7, 9.9, preopen(JAN8))])
    assert sweep_sessions(ShortRatioDeviation(), frame, SESSIONS) == []


# ---------------------------------------------------------------------------
# Through the real engine: right session, multi-day hold via max_hold,
# same-day rows invisible under the engine's own PIT join
# ---------------------------------------------------------------------------


def test_engine_end_to_end_fires_jan8_and_holds_multi_day(tmp_path: Path) -> None:
    sig = ShortRatioDeviation()
    result = make_engine(tmp_path, sig, end=JAN13).run()
    assert result.emitted == 1 and result.executed == 1
    assert sum(result.drop_counters.values()) == 0
    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    assert row.symbol == "AAA" and row.side == "buy"
    # Entry decision at Jan 8's FIRST bar close (09:31); fill lands at the
    # arrival instant, priced off the NEXT bar's ask.
    assert row.entry_ts == T(JAN8, 9, 31) + LATENCY
    assert row.entry_px == pytest.approx(100.10)
    # Multi-day hold: max_hold (4320 wall-clock minutes) first reachable at
    # a bar on Jan 12 (the weekend defers it) -> exit decision Jan 12 09:31,
    # filled next bar. Held across FOUR sessions.
    assert row.exit_reason == "max_hold"
    assert row.exit_ts == T(JAN12, 9, 31) + LATENCY
    assert row.holding_minutes == pytest.approx(4 * MINUTES_PER_DAY)
    assert row.holding_minutes > MINUTES_PER_DAY  # genuinely multi-day


def test_engine_never_sees_same_day_rows(tmp_path: Path) -> None:
    # Run only through Jan 7: the pattern-completing asof-Jan-7 row is IN
    # the preloaded frame but stamped available Jan 8 09:00 — under the
    # engine's own available_at <= now join nothing ever fires.
    sig = ShortRatioDeviation()
    result = make_engine(tmp_path, sig, end=JAN7).run()
    assert result.emitted == 0 and result.executed == 0
    assert len(result.ledger) == 0


def test_run_signal_records_the_trial_before_the_engine_runs(tmp_path: Path) -> None:
    sig = ShortRatioDeviation()
    engine = make_engine(tmp_path, sig, end=JAN13)
    trials = TrialRegistry(root=tmp_path)
    seen_during_backtest: list[int] = []

    def backtest(s: ShortRatioDeviation):
        assert s is sig
        seen_during_backtest.append(trials.cumulative_trial_count())
        return engine.run()

    record, result = run_signal(
        sig,
        backtest,
        trials,
        run_config={"start": str(SESSIONS[0]), "end": str(JAN13), "universe": ["AAA"]},
        ts=T(JAN5, 9, 0),
    )
    assert seen_during_backtest == [1]  # trial line existed BEFORE the run
    assert record.hypothesis == ShortRatioDeviation.hypothesis
    assert record.config["signal"] == "short_ratio_deviation"
    assert record.config["signal_config"] == {
        "z_threshold": 2.0,
        "min_sustained_days": 3,
        "horizon_days": 3,
    }
    assert sorted(record.config["allowed_regimes"]) == [
        "low_vol_chopping",
        "low_vol_trending",
    ]
    assert result.executed == 1

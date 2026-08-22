"""IndexLeadLag: planted-tape proofs of the lead-lag rule.

Covered here:

* a planted index jump with a FLAT high-beta name fires, same direction,
  ~30-minute horizon (both jump directions);
* a name that has ALREADY repriced past half its beta-implied move does not
  fire — the lag is observed, not assumed;
* beta is measured from PRIOR sessions only (today's bars, however wild,
  cannot change it), needs a minimum history, and gates out low-beta names;
* abstentions: no fresh index bar, not enough ATR history, calm index,
  never on the index symbol itself;
* registration contract: admissible hypothesis, stressed_backwardation
  excluded from allowed_regimes, config frozen with extras forbidden;
* one engine run through the sanctioned run_signal path: the trial line
  exists BEFORE the backtest, the fill is next-bar, and the drop identity
  holds.

ALL data is synthetic, no network, every timestamp is strictly before the
2026-02-22 lockbox wall, and every write lands under pytest tmp dirs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
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
from edge.regime.classifier import VOL_HIGH, VOL_LOW, VOL_MID, VOL_STRESSED
from edge.research.registry import TrialRegistry
from edge.runners.engine import BacktestEngine, EngineParams
from edge.signals import registry as signal_registry
from edge.signals.base import MIN_HYPOTHESIS_CHARS, regime_allowed
from edge.signals.leadlag import (
    MIN_ATR_OBSERVATIONS,
    MIN_BETA_DAYS,
    RET_WINDOW_BARS,
    IndexLeadLag,
    IndexLeadLagConfig,
    measure_beta,
)
from edge.signals.registry import run_signal

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

INDEX = "SPY"
NAME = "HIB"  # the high-beta single name

#: 70 synthetic prior sessions ending Friday 2025-06-13 — all pre-lockbox.
PRIOR_SESSIONS: list[date] = [d.date() for d in pd.bdate_range(end="2025-06-13", periods=70)]
TODAY = date(2025, 6, 16)  # the Monday after

#: Prior-session daily index returns: alternating +/-1% (nonzero variance).
INDEX_PRIOR_RETS: list[float] = [0.01 if i % 2 == 0 else -0.01 for i in range(70)]

#: True beta planted into the name's prior sessions.
PLANTED_BETA = 1.5

INDEX_BASE = 500.0
NAME_BASE = 100.0
INDEX_WIGGLE = 0.01  # ~2e-5 relative minute noise
NAME_WIGGLE = 0.002
INDEX_JUMP = 3.0  # ~ +0.6%: far above 1x the tiny expanding ATR

#: Bar index of the planted jump: leaves exactly MIN_ATR_OBSERVATIONS prior
#: 5-minute observations before the move that ends there.
J = MIN_ATR_OBSERVATIONS + RET_WINDOW_BARS  # = 35 -> 10:06 ET
JUMP_TS = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 31, tzinfo=ET) + timedelta(minutes=J)


# ---------------------------------------------------------------------------
# Tape builders
# ---------------------------------------------------------------------------


def _bar(symbol: str, ts: datetime, close: float, prev_close: float) -> BarEvent:
    return BarEvent(
        ts=ts,
        symbol=symbol,
        open=prev_close,
        high=max(prev_close, close) + 0.01,
        low=min(prev_close, close) - 0.01,
        close=close,
        volume=1_000,
    )


def prior_bars(symbol: str, base: float, rets: list[float], sessions: list[date]) -> list[BarEvent]:
    """One 15:59 bar per prior session; closes follow the given daily returns."""
    bars: list[BarEvent] = []
    prev = base
    close = base
    for day, ret in zip(sessions, rets):
        close = close * (1.0 + ret)
        ts = datetime(day.year, day.month, day.day, 15, 59, tzinfo=ET)
        bars.append(_bar(symbol, ts, close, prev))
        prev = close
    return bars


def today_bars(
    symbol: str,
    base: float,
    n: int,
    *,
    wiggle: float,
    jump_at: int | None = None,
    jump: float = 0.0,
) -> list[BarEvent]:
    """Minute bars from 9:31 ET on TODAY: alternating +/-wiggle around base,
    plus a persistent level shift of ``jump`` from bar ``jump_at`` on."""
    bars: list[BarEvent] = []
    prev = base
    for k in range(n):
        close = base + wiggle * (1 if k % 2 == 0 else -1)
        if jump_at is not None and k >= jump_at:
            close += jump
        ts = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 31, tzinfo=ET) + timedelta(
            minutes=k
        )
        bars.append(_bar(symbol, ts, close, prev))
        prev = close
    return bars


def build_tape(
    *,
    index_jump: float = INDEX_JUMP,
    name_jump: float = 0.0,
    planted_beta: float = PLANTED_BETA,
    sessions: list[date] = PRIOR_SESSIONS,
    n_today: int = J + 1,
) -> dict[str, list[BarEvent]]:
    """Prior sessions with an exact planted daily beta + today's minute tape."""
    rets = INDEX_PRIOR_RETS[: len(sessions)]
    name_rets = [planted_beta * r for r in rets]
    return {
        INDEX: prior_bars(INDEX, INDEX_BASE, rets, sessions)
        + today_bars(INDEX, INDEX_BASE, n_today, wiggle=INDEX_WIGGLE, jump_at=J, jump=index_jump),
        NAME: prior_bars(NAME, NAME_BASE, name_rets, sessions)
        + today_bars(
            NAME,
            NAME_BASE,
            n_today,
            wiggle=NAME_WIGGLE,
            jump_at=J if name_jump else None,
            jump=name_jump,
        ),
    }


class FakeCtx:
    """Engine-shaped ctx over a dict tape; visibility (ts <= now) enforced
    here too, so a peeking signal would fail these tests as well."""

    def __init__(self, tape: dict[str, list[BarEvent]], symbol: str, now: datetime) -> None:
        self._tape = tape
        self.now = now
        visible = [b for b in tape[symbol] if b.ts <= now]
        assert visible and visible[-1].ts == now, "test tape must have a bar at now"
        self.bar = visible[-1]

    def history(self, symbol: str | None = None, n: int | None = None) -> tuple[BarEvent, ...]:
        bars = [b for b in self._tape.get(symbol or self.bar.symbol, []) if b.ts <= self.now]
        return tuple(bars[-n:]) if n is not None else tuple(bars)


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_registered_with_admissible_hypothesis() -> None:
    assert signal_registry.get("index_leadlag") is IndexLeadLag
    assert len(IndexLeadLag.hypothesis.strip()) >= MIN_HYPOTHESIS_CHARS
    # The claim names its fixed choices — they are hypothesis, not tuning.
    for stated in ("5-minute", "63-session", "1.0", "HALF", "30 minutes"):
        assert stated in IndexLeadLag.hypothesis


def test_stressed_backwardation_is_excluded_and_stated() -> None:
    assert IndexLeadLag.allowed_regimes == frozenset({VOL_LOW, VOL_MID, VOL_HIGH})
    assert VOL_STRESSED not in IndexLeadLag.allowed_regimes
    assert regime_allowed(IndexLeadLag, VOL_STRESSED) is False
    for state in (VOL_LOW, VOL_MID, VOL_HIGH):
        assert regime_allowed(IndexLeadLag, state) is True
    # The exclusion is stated in the hypothesis, not silently configured.
    assert VOL_STRESSED in IndexLeadLag.hypothesis


def test_config_is_frozen_and_forbids_extras() -> None:
    with pytest.raises(ValidationError):
        IndexLeadLagConfig(atr_multiple=2.0)  # type: ignore[call-arg]
    cfg = IndexLeadLagConfig()
    assert cfg.index_symbol == INDEX
    assert cfg.atr_k == 1.0
    assert cfg.lag_fraction == 0.5
    assert cfg.min_beta == 1.0
    assert cfg.horizon_minutes == 30
    with pytest.raises(ValidationError):
        cfg.atr_k = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The rule: planted jump + flat name fires; already-moved name does not
# ---------------------------------------------------------------------------


def test_planted_index_jump_fires_flat_high_beta_name() -> None:
    tape = build_tape()  # index jumps ~+0.6% at 10:06; name is flat
    events = IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS))
    assert len(events) == 1
    evt = events[0]
    assert evt.symbol == NAME
    assert evt.side is Side.BUY  # SAME direction as the index move
    assert evt.ts == JUMP_TS
    assert evt.conviction == 1.0
    assert evt.horizon_minutes == 30
    assert evt.signal_name == "index_leadlag"


def test_name_that_already_moved_does_not_fire() -> None:
    # Name jumps +1.0% alongside the index: past half its beta-implied move
    # (0.5 * 1.5 * 0.6% = 0.45%), so there is no stale quote left to trade.
    tape = build_tape(name_jump=1.0)
    assert IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS)) == []


def test_down_jump_fires_sell_and_fallen_name_does_not() -> None:
    tape = build_tape(index_jump=-INDEX_JUMP)
    events = IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS))
    assert len(events) == 1
    assert events[0].side is Side.SELL
    # A name that already fell with the index does not fire.
    fallen = build_tape(index_jump=-INDEX_JUMP, name_jump=-1.0)
    assert IndexLeadLag().on_bar(FakeCtx(fallen, NAME, JUMP_TS)) == []


def test_calm_index_never_fires() -> None:
    # Same tape, no jump: minute noise never exceeds its own expanding ATR.
    tape = build_tape(index_jump=0.0)
    sig = IndexLeadLag()
    start = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 31, tzinfo=ET)
    for k in range(RET_WINDOW_BARS, J + 1):
        assert sig.on_bar(FakeCtx(tape, NAME, start + timedelta(minutes=k))) == []


def test_never_fires_on_the_index_symbol_itself() -> None:
    tape = build_tape()
    assert IndexLeadLag().on_bar(FakeCtx(tape, INDEX, JUMP_TS)) == []


def test_no_fire_without_fresh_index_bar() -> None:
    # The index tape is stale (no bar at now): the trigger is unverifiable.
    tape = build_tape()
    tape[INDEX] = [b for b in tape[INDEX] if b.ts < JUMP_TS]
    assert IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS)) == []


def test_no_fire_before_atr_distribution_exists() -> None:
    # Jump at bar 10: only 5 prior 5-minute observations, far under the
    # MIN_ATR_OBSERVATIONS floor — a jump vs no distribution is no signal.
    early_jump = 10
    assert early_jump - RET_WINDOW_BARS < MIN_ATR_OBSERVATIONS
    rets = INDEX_PRIOR_RETS
    tape = {
        INDEX: prior_bars(INDEX, INDEX_BASE, rets, PRIOR_SESSIONS)
        + today_bars(
            INDEX, INDEX_BASE, early_jump + 1, wiggle=INDEX_WIGGLE, jump_at=early_jump,
            jump=INDEX_JUMP,
        ),
        NAME: prior_bars(NAME, NAME_BASE, [PLANTED_BETA * r for r in rets], PRIOR_SESSIONS)
        + today_bars(NAME, NAME_BASE, early_jump + 1, wiggle=NAME_WIGGLE),
    }
    now = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 31, tzinfo=ET) + timedelta(
        minutes=early_jump
    )
    assert IndexLeadLag().on_bar(FakeCtx(tape, NAME, now)) == []


# ---------------------------------------------------------------------------
# Beta: measured point-in-time from PRIOR days only
# ---------------------------------------------------------------------------


def test_beta_uses_prior_days_only() -> None:
    tape = build_tape(name_jump=5.0)  # today the name goes wild: +5%
    prior_only_index = [b for b in tape[INDEX] if b.ts.date() < TODAY]
    prior_only_name = [b for b in tape[NAME] if b.ts.date() < TODAY]

    beta_prior = measure_beta(prior_only_name, prior_only_index, TODAY)
    assert beta_prior == pytest.approx(PLANTED_BETA, rel=1e-9)

    # Appending today's (wild) bars changes NOTHING: prior days only.
    beta_full = measure_beta(tape[NAME], tape[INDEX], TODAY)
    assert beta_full == beta_prior


def test_low_prior_beta_blocks_the_fire() -> None:
    # Identical firing tape today, but the PRIOR sessions say beta = 0.2:
    # a low-beta name is not hedged against the index, so no mechanism.
    tape = build_tape(planted_beta=0.2)
    assert measure_beta(tape[NAME], tape[INDEX], TODAY) == pytest.approx(0.2, rel=1e-9)
    assert IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS)) == []


def test_unmeasurable_beta_means_no_trade() -> None:
    # Too few prior sessions for a trustworthy beta: abstain, never default.
    short_sessions = PRIOR_SESSIONS[-(MIN_BETA_DAYS // 2) :]
    tape = build_tape(sessions=short_sessions)
    assert measure_beta(tape[NAME], tape[INDEX], TODAY) is None
    assert IndexLeadLag().on_bar(FakeCtx(tape, NAME, JUMP_TS)) == []


# ---------------------------------------------------------------------------
# Engine integration through the sanctioned run_signal path
# ---------------------------------------------------------------------------

ENGINE_CONFIG = EdgeConfig(
    data=DataConfig(lockbox_start=date(2026, 2, 22)),
    validation=ValidationConfig(min_oos_trades=300, pbo_max=0.5, bootstrap_resamples=100),
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

HALF_SPREAD = 0.05


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def _frames_from(bars: dict[str, list[BarEvent]]) -> dict[tuple[str, str], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, events in bars.items():
        frame = pd.DataFrame(
            {
                "ts": [b.ts for b in events],
                "open": [b.open for b in events],
                "high": [b.high for b in events],
                "low": [b.low for b in events],
                "close": [b.close for b in events],
                "volume": [b.volume for b in events],
            }
        )
        frames[(symbol, "bars")] = frame
        frames[(symbol, "quotes")] = pd.DataFrame(
            {
                "ts": frame["ts"],
                "bid": frame["close"] - HALF_SPREAD,
                "ask": frame["close"] + HALF_SPREAD,
                "bid_size": 500,
                "ask_size": 500,
            }
        )
    return frames


def test_engine_run_records_trial_first_and_fills_next_bar(tmp_path: Path) -> None:
    # 75 minute bars today: the 10:06 jump, then room for the 30-minute
    # max-hold exit to fill (decision 10:37, next-bar fill 10:38).
    tape = build_tape(n_today=75)
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "edge.yaml").write_text(
        (REPO_ROOT / "config" / "edge.yaml").read_text()
    )
    loader = EdgeDataLoader(TapeBackend(_frames_from(tape)), repo_root=root)
    trials_root = tmp_path / "trials"
    trials_root.mkdir()
    trials = TrialRegistry(root=trials_root)
    params = EngineParams(
        symbols=(INDEX, NAME),
        start=PRIOR_SESSIONS[0],
        end=TODAY,
        latency_ms=100,
        initial_equity=100_000.0,
        seed=0,
        risk_r_pct=1.0,
        max_hold_minutes=30,
    )
    signal = IndexLeadLag()
    observed: dict[str, int] = {}

    def backtest(sig: IndexLeadLag):
        # The trial line must already be on disk while the backtest runs.
        observed["trials_before_run"] = trials.cumulative_trial_count()
        return BacktestEngine(loader, ENGINE_CONFIG, params, [sig]).run()

    record, result = run_signal(
        signal,
        backtest,
        trials,
        run_config={"symbols": [INDEX, NAME], "start": str(PRIOR_SESSIONS[0]), "end": str(TODAY)},
        ts=datetime(2025, 6, 16, 16, 0, tzinfo=ET),
    )

    assert observed["trials_before_run"] == 1
    assert record.hypothesis == IndexLeadLag.hypothesis
    assert record.result == {"status": "pending"}
    assert record.config["allowed_regimes"] == sorted(IndexLeadLag.allowed_regimes)

    # The jump window straddles bars 10:06-10:10: 5 emissions, 1 executed,
    # the rest dropped as position-already-open. The identity must close.
    assert result.executed == 1
    assert result.emitted == 5
    assert result.drop_counters["position-already-open"] == 4
    assert result.emitted == result.executed + sum(result.drop_counters.values())

    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    assert row["symbol"] == NAME  # never the index itself
    assert row["side"] == "buy"
    assert row["signal_name"] == "index_leadlag"
    assert row["exit_reason"] == "max_hold"
    # Next-bar fill: the decision at 10:06 is priced off the 10:07 book.
    assert pd.Timestamp(row["entry_ts"]) > pd.Timestamp(JUMP_TS)
    next_bar_close = NAME_BASE + NAME_WIGGLE  # k=36 is an even (+wiggle) bar
    assert row["entry_px"] == pytest.approx(next_bar_close + HALF_SPREAD, abs=1e-9)

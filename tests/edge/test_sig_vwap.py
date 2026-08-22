"""VwapReversion: planted mean-reverting stretch fires SHORT at the high,
trending tapes are suppressed via allowed_regimes (the load-bearing regime
clause, exercised through the real engine's regime gate), and the z-score
uses only session-so-far data (prefix invariance: a ctx that over-serves
future bars or prior-session bars changes nothing).

All tapes are synthetic, all writes land in tmp_path dirs, no network, and
every timestamp is strictly before the 2026-02-22 lockbox start."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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
from edge.core.events import BarEvent, Side, SignalEvent
from edge.data.loader import EdgeDataLoader
from edge.regime.classifier import BUCKETS, gate
from edge.runners.engine import DROP_REGIME, BacktestEngine, EngineParams
from edge.signals import registry as signal_registry
from edge.signals.base import MIN_HYPOTHESIS_CHARS, regime_allowed
from edge.signals.vwap_rev import (
    CHOP_BUCKETS,
    VwapReversion,
    VwapReversionConfig,
    current_z,
    spread_series,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

#: Synthetic session strictly before the 2026-02-22 lockbox start.
SESSION = date(2025, 3, 4)
PRIOR_SESSION = date(2025, 3, 3)
SYMBOL = "TTT"


def _ts(minute_index: int, day: date = SESSION) -> datetime:
    """Close time of session minute bar ``minute_index`` (0 -> 09:31 ET)."""
    return datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET) + timedelta(
        minutes=minute_index + 1
    )


def _bars(closes: np.ndarray, day: date = SESSION, volume: int = 1000) -> list[BarEvent]:
    bars = []
    prev = float(closes[0])
    for i, close in enumerate(closes):
        close = float(close)
        bars.append(
            BarEvent(
                ts=_ts(i, day),
                symbol=SYMBOL,
                open=prev,
                high=max(prev, close) + 0.02,
                low=min(prev, close) - 0.02,
                close=close,
                volume=volume,
            )
        )
        prev = close
    return bars


def chop_closes(n: int = 70) -> np.ndarray:
    """Deterministic non-trending chop around 100 (std > 0, |z| < 2 always)."""
    i = np.arange(n)
    return 100.0 + 0.05 * np.sin(1.0 + i * 0.9)


def chop_spike_closes(spike_to: float = 101.5) -> np.ndarray:
    """70 chop bars, then one planted liquidity-demand spike (bar 71)."""
    return np.append(chop_closes(70), spike_to)


def trend_closes() -> np.ndarray:
    """64 linear up-drift bars (+0.05) then 6 accelerating bars (+0.35).

    The pure linear leg sits at z ~ 1.7 (< 2: a steady trend alone never
    trips the threshold); the terminal acceleration crosses z > 2, so the
    raw stretch pattern DOES fire on this tape — suppression must come from
    allowed_regimes, not from the z-score.
    """
    lin = 100.0 + 0.05 * np.arange(64)
    return np.concatenate([lin, lin[-1] + 0.35 * np.arange(1, 7)])


class FakeCtx:
    """Minimal SignalContext: serves exactly the bars it is given.

    Deliberately sloppy on request: ``history`` returns EVERY bar handed to
    it (which may include future bars and prior-session bars) — the signal,
    not the ctx, must enforce the session-so-far window.
    """

    def __init__(self, bars: list[BarEvent], decision_index: int) -> None:
        self._bars = bars
        self._bar = next(b for b in bars if b.ts == _ts(decision_index))

    @property
    def decision_ts(self) -> datetime:
        return self._bar.ts

    @property
    def bar(self) -> BarEvent:
        return self._bar

    def history(self, symbol: str, n_bars: int) -> tuple[BarEvent, ...]:
        return tuple(b for b in self._bars if b.symbol == symbol)[-n_bars:]

    def frame(self, kind: str, symbol: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


def events_at(bars: list[BarEvent], decision_index: int) -> list[SignalEvent]:
    return VwapReversion().on_bar(FakeCtx(bars, decision_index))


# ---------------------------------------------------------------------------
# Declarations: registration, hypothesis, the regime clause, config hygiene
# ---------------------------------------------------------------------------


def test_registered_with_admissible_hypothesis() -> None:
    assert signal_registry.get("vwap_reversion") is VwapReversion
    assert len(VwapReversion.hypothesis.strip()) >= MIN_HYPOTHESIS_CHARS
    # The fixed economic choices are written into the hypothesis text.
    assert "z > 2" in VwapReversion.hypothesis
    assert "60" in VwapReversion.hypothesis
    assert VwapReversion.warmup_bars == 60


def test_allowed_regimes_are_exactly_the_chop_buckets() -> None:
    assert VwapReversion.allowed_regimes == CHOP_BUCKETS
    assert CHOP_BUCKETS == {"low_vol_chopping", "high_vol_chopping"}
    # Real classifier vocabulary — a typo here would gate the signal forever.
    assert CHOP_BUCKETS <= BUCKETS
    for bucket in ("low_vol_trending", "high_vol_trending"):
        assert regime_allowed(VwapReversion, bucket) is False
        assert gate(VwapReversion.allowed_regimes, bucket) is False
    for bucket in CHOP_BUCKETS:
        assert regime_allowed(VwapReversion, bucket) is True
        assert gate(VwapReversion.allowed_regimes, bucket) is True


def test_config_defaults_are_the_declared_fixed_choices() -> None:
    cfg = VwapReversionConfig()
    assert cfg.z_entry == 2.0
    assert cfg.min_session_bars == 60
    assert cfg.horizon_minutes == 60
    assert cfg.conviction_full_z == 4.0
    with pytest.raises(ValidationError):
        VwapReversionConfig(z_threshold=3.0)  # typo'd knob: error, not silence
    with pytest.raises(ValidationError):
        cfg.z_entry = 3.0  # frozen


# ---------------------------------------------------------------------------
# The planted mean-reverting stretch: short at the high, long at the low
# ---------------------------------------------------------------------------


def test_planted_stretch_fires_short_at_the_high() -> None:
    bars = _bars(chop_spike_closes())
    spike_index = 70

    # No opinion anywhere in the chop, even after the 60-bar minimum.
    for k in range(60, spike_index):
        assert events_at(bars, k) == []

    events = events_at(bars, spike_index)
    assert len(events) == 1
    evt = events[0]
    assert evt.side is Side.SELL  # fade the upside stretch: short at the high
    assert evt.symbol == SYMBOL
    assert evt.signal_name == "vwap_reversion"
    assert evt.ts == _ts(spike_index)
    assert evt.horizon_minutes == 60
    # z ~ 8 at the planted spike: conviction saturates at 1.0.
    assert evt.conviction == 1.0


def test_downside_stretch_fires_long_symmetrically() -> None:
    bars = _bars(np.append(chop_closes(70), 98.5))
    events = events_at(bars, 70)
    assert len(events) == 1
    assert events[0].side is Side.BUY


def test_conviction_is_scaled_by_z_between_threshold_and_saturation() -> None:
    # The trending tape's first acceleration bar sits at 2 < z < 4: the
    # emitted conviction must equal |z| / 4 exactly (and saturate at 1).
    closes = trend_closes()
    bars = _bars(closes)
    k = 64  # first acceleration bar
    events = events_at(bars, k)
    assert len(events) == 1
    highs = np.array([b.high for b in bars[: k + 1]])
    lows = np.array([b.low for b in bars[: k + 1]])
    vols = np.array([float(b.volume) for b in bars[: k + 1]])
    z = current_z(spread_series(closes[: k + 1], highs, lows, vols))
    assert z is not None and 2.0 < z < 4.0
    assert events[0].conviction == pytest.approx(z / 4.0)
    assert 0.0 < events[0].conviction < 1.0


def test_min_session_bars_gate_blocks_an_early_spike() -> None:
    # The same planted spike, but only 30 session bars behind it: no z yet.
    bars = _bars(np.append(chop_closes(30), 101.5))
    assert events_at(bars, 30) == []


def test_degenerate_flat_session_emits_nothing() -> None:
    bars = _bars(np.full(65, 100.0))  # spread std is 0: no z exists
    assert events_at(bars, 64) == []


# ---------------------------------------------------------------------------
# Prefix invariance: z uses ONLY session-so-far data
# ---------------------------------------------------------------------------


def test_future_bars_in_history_change_nothing() -> None:
    """FakeCtx serves the ENTIRE tape (future included); the signal must
    reproduce the truncated-tape decision bar-for-bar."""
    full = _bars(chop_spike_closes())
    for k in (60, 65, 70):
        truncated = full[: k + 1]
        e_full = events_at(full, k)
        e_trunc = events_at(truncated, k)
        assert [e.model_dump() for e in e_full] == [e.model_dump() for e in e_trunc]
    # The spike decision specifically: identical opinion despite the ctx
    # over-serving bars that close after the decision instant.
    assert events_at(full, 70)[0].side is Side.SELL


def test_prior_session_bars_are_excluded_from_the_distribution() -> None:
    """A wild prior session in history must not move today's z at all."""
    today = _bars(chop_spike_closes())
    yesterday = _bars(100.0 + 5.0 * np.sin(np.arange(80)), day=PRIOR_SESSION)
    with_prior = yesterday + today
    for k in (60, 70):
        a = [e.model_dump() for e in events_at(today, k)]
        b = [e.model_dump() for e in events_at(with_prior, k)]
        assert a == b


def test_spread_series_is_prefix_invariant_and_volume_weighted() -> None:
    closes = chop_spike_closes()
    highs = closes + 0.02
    lows = closes - 0.02
    vols = np.linspace(500.0, 2000.0, closes.size)  # unequal volumes
    full = spread_series(closes, highs, lows, vols)
    for k in (1, 10, 60, 70):
        prefix = spread_series(closes[: k + 1], highs[: k + 1], lows[: k + 1], vols[: k + 1])
        np.testing.assert_allclose(full[: k + 1], prefix)
    # Hand-computed volume weighting on the first two bars.
    tp = (highs + lows + closes) / 3.0
    expected_1 = closes[1] - (tp[0] * vols[0] + tp[1] * vols[1]) / (vols[0] + vols[1])
    assert full[1] == pytest.approx(expected_1)


def test_current_z_matches_the_definition_and_guards_degeneracy() -> None:
    spreads = np.array([0.1, -0.2, 0.0, 0.3, 1.5])
    z = current_z(spreads)
    assert z == pytest.approx((spreads[-1] - spreads.mean()) / spreads.std())
    assert current_z(np.array([1.0])) is None  # one point: no distribution
    assert current_z(np.zeros(80)) is None  # zero std: no z


# ---------------------------------------------------------------------------
# Trending tape suppressed via allowed_regimes — through the real engine
# ---------------------------------------------------------------------------

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


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def trending_tape_loader(tmp_path: Path) -> EdgeDataLoader:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "edge.yaml").write_text(
        (REPO_ROOT / "config" / "edge.yaml").read_text()
    )
    bars = _bars(trend_closes())
    bars_frame = pd.DataFrame(
        {
            "ts": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    quotes_frame = pd.DataFrame(
        {
            "ts": bars_frame["ts"],
            "bid": bars_frame["close"] - 0.05,
            "ask": bars_frame["close"] + 0.05,
            "bid_size": 500,
            "ask_size": 500,
        }
    )
    frames = {(SYMBOL, "bars"): bars_frame, (SYMBOL, "quotes"): quotes_frame}
    return EdgeDataLoader(TapeBackend(frames), repo_root=root)


def run_engine(tmp_path: Path, session_bucket: str):
    """One engine run over the trending tape; the regime gate consults the
    signal's OWN allowed_regimes declaration against ``session_bucket``."""

    def regime_gate(event: SignalEvent, ctx: object) -> bool:
        return regime_allowed(VwapReversion, session_bucket)

    params = EngineParams(
        symbols=(SYMBOL,),
        start=SESSION,
        end=SESSION,
        latency_ms=100,
        initial_equity=100_000.0,
        seed=0,
        risk_r_pct=1.0,
    )
    engine = BacktestEngine(
        trending_tape_loader(tmp_path),
        CONFIG,
        params,
        [VwapReversion()],
        regime_gate=regime_gate,
    )
    return engine.run()


def test_trending_tape_is_suppressed_via_allowed_regimes(tmp_path: Path) -> None:
    result = run_engine(tmp_path, "low_vol_trending")
    # The raw stretch pattern fires on the acceleration bars...
    assert result.emitted > 0
    # ...and EVERY emission dies at the regime gate: no trade ever happens.
    assert result.drop_counters[DROP_REGIME] == result.emitted
    assert result.executed == 0
    assert result.ledger.empty


def test_same_tape_trades_when_the_session_is_chopping(tmp_path: Path) -> None:
    """Control: identical tape and engine, chopping bucket — the gate opens,
    so the suppression above is attributable to allowed_regimes alone."""
    result = run_engine(tmp_path, "low_vol_chopping")
    assert result.emitted > 0
    assert result.drop_counters[DROP_REGIME] == 0
    assert result.executed >= 1
    assert not result.ledger.empty
    assert (result.ledger["side"] == "sell").all()  # fading the upside stretch
    assert (result.ledger["signal_name"] == "vwap_reversion").all()

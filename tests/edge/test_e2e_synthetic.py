"""END-TO-END synthetic proof of the whole research loop on one crafted tape.

A toy always-long signal (with a real, admissible hypothesis) is REGISTERED
in the signal catalog, instantiated through ``registry.create``, and run
through the sanctioned path:

    run_signal -> TrialRegistry line (BEFORE the backtest)
               -> BacktestEngine over a 60-session, 2-symbol tape with a
                  PLANTED upward drift (every 7th session drifts down)
               -> ledger / equity -> render_tearsheet -> evaluate_promotion
               -> REJECTED.md (and the EXPECTANCY.md leaderboard)

The tape is designed so the drift signal passes EVERY promotion gate except
the minimum out-of-sample trade count (120 trades < the production 300):

* expectancy CI strictly above zero  (planted drift, cheap frictions)
* PBO below the cap                  (upstream CSCV hook value)
* latency survival                   (positive expectancy under EVERY
                                      configured latency scenario)
* >= 3 positive regime buckets       (all four 15-session blocks win)
* trade count                        [FAIL] -- the one designed failure

so the REJECT verdict proves the gates actually gate: relaxing ONLY the
trade-count threshold flips the same numbers to PROMOTE. The rejection is
recorded in a tmp REJECTED.md with ``n_oos`` named among the numbers.

ALL data is synthetic, no network, every timestamp is strictly before
2026-02-22, and every write lands under pytest tmp dirs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.core.config import (
    DataConfig,
    EdgeConfig,
    ExecutionConfig,
    RiskConfig,
    ValidationConfig,
)
from edge.core.events import Side, SignalEvent
from edge.data.loader import EdgeDataLoader
from edge.research.registry import TrialRecord, TrialRegistry
from edge.research.tearsheet import (
    FAIL_TRADES,
    MONTHS_TO_DISTINGUISH_SHARPE_ONE,
    REGIME_BUCKETS,
    VERDICT_PROMOTE,
    VERDICT_REJECT,
    ExpectancyRow,
    append_expectancy_md,
    append_rejected_md,
    compute_monthly_headline,
    compute_regime_breakdown,
    compute_trade_stats,
    evaluate_promotion,
    render_tearsheet,
)
from edge.runners.engine import BacktestEngine, EngineParams, RunResult
from edge.signals import registry as signal_registry
from edge.signals.base import ALL_REGIMES, MIN_HYPOTHESIS_CHARS, Signal, SignalConfig
from edge.signals.registry import run_signal
from edge.validation.gateway import scan_research_side

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

# --------------------------------------------------------------------------
# The crafted tape: 60 business sessions, 2 symbols, planted drift
# --------------------------------------------------------------------------

#: 60 synthetic sessions, all strictly before the 2026-02-22 lockbox wall.
SESSIONS: list[date] = [d.date() for d in pd.bdate_range("2025-01-02", periods=60)]

#: Every 7th session the drift is DOWN — losers for bootstrap variance.
LOSER_EVERY = 7

#: Per-symbol (base price, up-drift, down-drift) of the planted tape.
TAPE_SPEC: dict[str, tuple[float, float, float]] = {
    "AAA": (100.0, +0.50, -0.20),
    "BBB": (60.0, +0.60, -0.25),
}

#: Five minute bars per session; the signal fires on the first two closes.
BAR_MINUTES = (31, 32, 33, 34, 35)
HALF_SPREAD = 0.05
LATENCY_SCENARIOS = (100, 500, 2000)

CONFIG = EdgeConfig(
    data=DataConfig(lockbox_start=date(2026, 2, 22)),
    # Production-shaped promotion gates (min_oos_trades=300 as in edge.yaml),
    # cheap bootstrap for test speed.
    validation=ValidationConfig(min_oos_trades=300, pbo_max=0.5, bootstrap_resamples=200),
    execution=ExecutionConfig(
        spread_fill_fraction=1.0,
        slippage_pct=0.0,
        commission_per_contract=0.01,
        latency_ms=list(LATENCY_SCENARIOS),
    ),
    risk=RiskConfig(
        kelly_fraction=0.25,
        per_trade_cap_pct=2.0,
        daily_loss_halt_pct=3.0,
        target_daily_vol_pct=None,
    ),
)

#: The upstream CSCV hook value (edge.validation.stats.pbo_cscv is exercised
#: in test_stats; the tearsheet takes the computed number as an input).
UPSTREAM_PBO = 0.10


def _session_drift(symbol: str, session_index: int) -> float:
    _, up, down = TAPE_SPEC[symbol]
    return down if session_index % LOSER_EVERY == 0 else up


def build_frames() -> dict[tuple[str, str], pd.DataFrame]:
    """Bars + quotes for both symbols: a linear intra-session drift ramp."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, (base, _up, _down) in TAPE_SPEC.items():
        rows: list[dict] = []
        price = base
        for i, session in enumerate(SESSIONS):
            drift = _session_drift(symbol, i)
            prev_close = price
            for k, minute in enumerate(BAR_MINUTES, start=1):
                close = price + drift * k / len(BAR_MINUTES)
                rows.append(
                    {
                        "ts": datetime(
                            session.year, session.month, session.day, 9, minute, tzinfo=ET
                        ),
                        "open": prev_close,
                        "high": max(prev_close, close) + 0.02,
                        "low": min(prev_close, close) - 0.02,
                        "close": close,
                        "volume": 1_000,
                    }
                )
                prev_close = close
            price += drift
        bars = pd.DataFrame(rows)
        quotes = pd.DataFrame(
            {
                "ts": bars["ts"],
                "bid": bars["close"] - HALF_SPREAD,
                "ask": bars["close"] + HALF_SPREAD,
                "bid_size": 500,
                "ask_size": 500,
            }
        )
        frames[(symbol, "bars")] = bars
        frames[(symbol, "quotes")] = quotes
    return frames


class TapeBackend:
    """In-memory backend serving the pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def make_repo_root(tmp_path: Path) -> Path:
    """Tmp repo root with the real edge.yaml copied in (wall 2026-02-22)."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "edge.yaml").write_text((REPO_ROOT / "config" / "edge.yaml").read_text())
    return tmp_path


# --------------------------------------------------------------------------
# The toy always-long signal (registered) and its engine adapter
# --------------------------------------------------------------------------

E2E_HYPOTHESIS = (
    "Passive rebalancing flow must buy the first prints of every session "
    "regardless of price; the counterparty is the index-tracking desk paying "
    "the spread because tracking error is costlier to it than execution, so "
    "a patient long entered at the open harvests that urgency premium."
)

#: Bar closes at which the always-long toy emits (the 9:32 emission is always
#: dropped as position-already-open, exercising the drop pipeline).
EMIT_TIMES = frozenset({time(9, 31), time(9, 32)})


class DriftRider(Signal):
    """Toy always-long signal: emits a long opinion on the first two closes."""

    name = "drift_rider"
    hypothesis = E2E_HYPOTHESIS
    allowed_regimes = ALL_REGIMES
    warmup_bars = 0
    config_type = SignalConfig

    def on_bar(self, ctx) -> list[SignalEvent]:
        if ctx.now.astimezone(ET).time() not in EMIT_TIMES:
            return []
        return [
            SignalEvent(
                ts=ctx.now,
                symbol=ctx.bar.symbol,
                side=Side.BUY,
                conviction=1.0,
                horizon_minutes=4,
                signal_name=self.name,
            )
        ]


class EngineSignalAdapter:
    """Bridges the base Signal contract (list) to the engine's (event | None)."""

    def __init__(self, signal: Signal) -> None:
        self._signal = signal
        self.name = signal.name

    def on_bar(self, ctx) -> SignalEvent | None:
        events = self._signal.on_bar(ctx)
        return events[0] if events else None


def engine_params(latency_ms: int) -> EngineParams:
    return EngineParams(
        symbols=tuple(TAPE_SPEC),
        start=SESSIONS[0],
        end=SESSIONS[-1],
        latency_ms=latency_ms,
        initial_equity=100_000.0,
        seed=0,
        risk_r_pct=1.0,
        max_hold_minutes=2,
    )


def regime_series() -> pd.Series:
    """Four 15-session blocks, one canonical bucket each, stamped at block start."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(SESSIONS[i], tz=ET) for i in (0, 15, 30, 45)]
    )
    return pd.Series(list(REGIME_BUCKETS), index=idx)


# --------------------------------------------------------------------------
# The one end-to-end run (module-scoped; every test reads from it)
# --------------------------------------------------------------------------


@dataclass
class E2ERun:
    record: TrialRecord
    result: RunResult
    trials: TrialRegistry
    trials_seen_inside_backtest: int
    latency_results: dict[int, RunResult]
    latency_survived: bool
    tearsheet: str
    promotion_strict: object
    promotion_relaxed: object
    positive_buckets: int
    trade_stats: object
    headline: object
    expectancy_md: Path
    rejected_md: Path


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> E2ERun:
    tmp = tmp_path_factory.mktemp("e2e")
    root = make_repo_root(tmp / "repo")
    frames = build_frames()
    loader = EdgeDataLoader(TapeBackend(frames), repo_root=root)

    trials_root = tmp / "trials"
    trials_root.mkdir()
    trials = TrialRegistry(root=trials_root)

    snapshot = signal_registry._snapshot()
    try:
        signal_registry.register(DriftRider)
        signal = signal_registry.create(DriftRider.name)  # catalog round-trip

        observed: dict[str, int] = {}

        def backtest(sig: Signal) -> RunResult:
            # Provable ordering: the trial line must already be on disk HERE.
            observed["trials_before_run"] = trials.cumulative_trial_count()
            engine = BacktestEngine(
                loader, CONFIG, engine_params(LATENCY_SCENARIOS[0]), [EngineSignalAdapter(sig)]
            )
            return engine.run()

        record, result = run_signal(
            signal,
            backtest,
            trials,
            run_config={
                "symbols": sorted(TAPE_SPEC),
                "start": str(SESSIONS[0]),
                "end": str(SESSIONS[-1]),
                "latency_ms": list(LATENCY_SCENARIOS),
            },
            ts=datetime(2026, 1, 15, 10, 0, tzinfo=ET),
        )

        # Latency survival: re-run the SAME experiment under every configured
        # latency scenario; survival = positive expectancy in each.
        latency_results: dict[int, RunResult] = {LATENCY_SCENARIOS[0]: result}
        for latency in LATENCY_SCENARIOS[1:]:
            engine = BacktestEngine(
                loader, CONFIG, engine_params(latency), [EngineSignalAdapter(signal)]
            )
            latency_results[latency] = engine.run()
        latency_survived = all(
            len(run.ledger) > 0 and float(run.ledger["r_multiple"].mean()) > 0.0
            for run in latency_results.values()
        )
    finally:
        signal_registry._restore(snapshot)

    regime = regime_series()
    tearsheet = render_tearsheet(
        DriftRider.name,
        result.ledger,
        result.equity,
        trials,
        regime,
        CONFIG.validation,
        pbo=UPSTREAM_PBO,
        latency_survived=latency_survived,
        seed=0,
    )

    trade_stats = compute_trade_stats(result.ledger, CONFIG.validation, seed=0)
    headline = compute_monthly_headline(result.equity, CONFIG.validation, seed=0)
    _, positive_buckets = compute_regime_breakdown(result.ledger, regime)
    promotion_strict = evaluate_promotion(
        n_trades=trade_stats.n,
        expectancy_ci=(trade_stats.ci_lo, trade_stats.ci_hi),
        pbo=UPSTREAM_PBO,
        latency_survived=latency_survived,
        positive_buckets=positive_buckets,
        config=CONFIG.validation,
    )
    # Identical numbers, trade-count gate relaxed to below what the tape
    # produced: the ONLY failing gate flips, so the verdict must too.
    relaxed = ValidationConfig(
        min_oos_trades=trade_stats.n, pbo_max=0.5, bootstrap_resamples=200
    )
    promotion_relaxed = evaluate_promotion(
        n_trades=trade_stats.n,
        expectancy_ci=(trade_stats.ci_lo, trade_stats.ci_hi),
        pbo=UPSTREAM_PBO,
        latency_survived=latency_survived,
        positive_buckets=positive_buckets,
        config=relaxed,
    )

    expectancy_md = tmp / "EXPECTANCY.md"
    rejected_md = tmp / "REJECTED.md"
    append_expectancy_md(
        expectancy_md,
        [
            ExpectancyRow(
                signal=DriftRider.name,
                geometric_monthly=headline.geometric,
                months=headline.months,
                n_trades=trade_stats.n,
                expectancy_r=trade_stats.expectancy_r,
                verdict=promotion_strict.verdict,
            )
        ],
    )
    append_rejected_md(
        rejected_md,
        DriftRider.name,
        promotion_strict.failures,
        numbers={
            "n_oos": trade_stats.n,
            "min_oos_trades": CONFIG.validation.min_oos_trades,
        },
    )

    return E2ERun(
        record=record,
        result=result,
        trials=trials,
        trials_seen_inside_backtest=observed["trials_before_run"],
        latency_results=latency_results,
        latency_survived=latency_survived,
        tearsheet=tearsheet,
        promotion_strict=promotion_strict,
        promotion_relaxed=promotion_relaxed,
        positive_buckets=positive_buckets,
        trade_stats=trade_stats,
        headline=headline,
        expectancy_md=expectancy_md,
        rejected_md=rejected_md,
    )


# --------------------------------------------------------------------------
# Hypothesis contract + registry ordering
# --------------------------------------------------------------------------


def test_hypothesis_is_admissible_and_recorded(e2e: E2ERun) -> None:
    assert len(E2E_HYPOTHESIS.strip()) >= MIN_HYPOTHESIS_CHARS
    assert e2e.record.hypothesis == E2E_HYPOTHESIS
    assert e2e.record.config["signal"] == DriftRider.name
    assert e2e.record.config["signal_class"].endswith("DriftRider")
    assert e2e.record.duplicate_of is None


def test_registry_line_precedes_results(e2e: E2ERun) -> None:
    """The trial line existed on disk BEFORE the backtest produced anything."""
    assert e2e.trials_seen_inside_backtest == 1
    assert e2e.record.trial_id == 1
    # The pre-run line is pending forever — the registry is append-only.
    assert e2e.record.result == {"status": "pending"}
    assert e2e.trials.cumulative_trial_count() == 1


# --------------------------------------------------------------------------
# Engine accounting on the crafted tape
# --------------------------------------------------------------------------


def test_ledger_drop_identity_holds(e2e: E2ERun) -> None:
    result = e2e.result
    # 60 sessions x 2 symbols x 2 emissions (9:31 executes, 9:32 drops as
    # position-already-open): identity emitted == executed + sum(drops).
    assert result.emitted == 240
    assert result.executed == 120
    assert sum(result.drop_counters.values()) == 120
    assert result.emitted == result.executed + sum(result.drop_counters.values())
    assert result.drop_counters["position-already-open"] == 120
    assert all(
        count == 0
        for stage, count in result.drop_counters.items()
        if stage != "position-already-open"
    )
    assert len(result.ledger) == 120


def test_ledger_economics_close_the_books(e2e: E2ERun) -> None:
    ledger = e2e.result.ledger
    assert set(ledger["symbol"]) == set(TAPE_SPEC)
    assert (ledger["signal_name"] == DriftRider.name).all()
    assert (ledger["exit_reason"] == "max_hold").all()  # nothing stranded
    assert float(e2e.result.equity.iloc[-1]) == pytest.approx(
        100_000.0 + float(ledger["pnl"].sum())
    )
    # The planted drift is real: expectancy is positive and its CI clears zero.
    assert e2e.trade_stats.expectancy_r > 0.0
    assert e2e.trade_stats.ci_lo > 0.0


def test_latency_survival_under_every_scenario(e2e: E2ERun) -> None:
    assert set(e2e.latency_results) == set(LATENCY_SCENARIOS)
    for run in e2e.latency_results.values():
        assert len(run.ledger) == 120
        assert float(run.ledger["r_multiple"].mean()) > 0.0
    assert e2e.latency_survived is True


# --------------------------------------------------------------------------
# Tearsheet: the headline block parses and matches the equity curve
# --------------------------------------------------------------------------


def test_tearsheet_headline_parses(e2e: E2ERun) -> None:
    text = e2e.tearsheet
    assert f"TEARSHEET: {DriftRider.name}" in text

    geo = re.search(r"Geometric \(compounded\):\s+(-?\d+(?:\.\d+)?)%", text)
    assert geo is not None, "headline geometric line missing"
    assert float(geo.group(1)) / 100.0 == pytest.approx(
        e2e.headline.geometric, abs=6e-4  # one-decimal-percent rounding
    )

    months = re.search(r"Months observed:\s+(\d+)", text)
    assert months is not None and int(months.group(1)) == e2e.headline.months
    assert e2e.headline.months == 3  # Jan/Feb/Mar 2025 of the 60-session tape

    # 3 months < 46: the statistical-power warning must be appended verbatim.
    assert (
        f"NOTE: 3 months < {MONTHS_TO_DISTINGUISH_SHARPE_ONE} needed to "
        "distinguish a Sharpe-1.0 strategy from zero." in text
    )
    ci = re.search(r"95% bootstrap CI:\s+\[(-?\d+(?:\.\d+)?)%, (-?\d+(?:\.\d+)?)%\]", text)
    assert ci is not None, "headline CI line missing"


# --------------------------------------------------------------------------
# Promotion: the gates actually gate
# --------------------------------------------------------------------------


def test_promotion_fails_only_the_trade_count_gate(e2e: E2ERun) -> None:
    assert e2e.positive_buckets == 4  # regime gate passes on all four buckets
    assert e2e.trade_stats.n == 120
    assert e2e.trade_stats.n < CONFIG.validation.min_oos_trades
    assert e2e.promotion_strict.promoted is False
    assert e2e.promotion_strict.failures == (FAIL_TRADES,)
    assert e2e.promotion_strict.verdict == VERDICT_REJECT


def test_relaxing_only_that_gate_promotes(e2e: E2ERun) -> None:
    """Same numbers, min-trade gate satisfied -> PROMOTE: every OTHER gate
    was already passing, so the trade-count gate alone caused the reject."""
    assert e2e.promotion_relaxed.promoted is True
    assert e2e.promotion_relaxed.failures == ()
    assert e2e.promotion_relaxed.verdict == VERDICT_PROMOTE


def test_tearsheet_promotion_block_names_the_failure(e2e: E2ERun) -> None:
    text = e2e.tearsheet
    assert f"[FAIL] oos trades 120 >= {CONFIG.validation.min_oos_trades}" in text
    assert f"VERDICT: {VERDICT_REJECT} (failed: {FAIL_TRADES})" in text
    # Every other gate line reads PASS.
    assert "[PASS] expectancy CI excludes zero" in text
    assert f"[PASS] pbo {UPSTREAM_PBO:.3f} < 0.500" in text
    assert "[PASS] latency survival" in text
    assert "[PASS] positive regime buckets 4 >= 3" in text


# --------------------------------------------------------------------------
# Markdown artifacts (tmp files, never the repo's)
# --------------------------------------------------------------------------


def test_rejected_md_names_n_oos(e2e: E2ERun) -> None:
    text = e2e.rejected_md.read_text()
    assert f"## REJECTED: {DriftRider.name}" in text
    assert FAIL_TRADES in text
    assert "n_oos=120" in text
    assert f"min_oos_trades={CONFIG.validation.min_oos_trades}" in text


def test_expectancy_md_ranks_the_rejected_signal(e2e: E2ERun) -> None:
    text = e2e.expectancy_md.read_text()
    assert f"| 1 | {DriftRider.name} |" in text
    assert f"| {VERDICT_REJECT} |" in text  # failures are ranked, not hidden


# --------------------------------------------------------------------------
# The data-gateway wall stays intact after integration
# --------------------------------------------------------------------------


def test_gateway_scan_stays_clean() -> None:
    assert scan_research_side() == []

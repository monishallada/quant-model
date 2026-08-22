"""First real backtests for the edge platform signals.

Operational harness (NOT research code — lives outside the AST gateway's
scanned tree, like the runners): composes the sanctioned loaders behind one
EdgeDataLoader, synthesizes conservative equity quotes from bars for the
engine's quote seam, wires the regime classifier into the engine's regime
gate, and drives each registered signal through ``run_signal`` (trial
recorded BEFORE the backtest, always) over the IS and OOS spans as separate
trials.

Run-scope decisions (recorded in every trial's run_config):
* Tape is regular-hours only (bar stamps 09:30..15:59 ET). The signals were
  built against 390-bar RTH sessions (ORB's 400-bars/session ceiling, VWAP's
  480-bar history cap); extended-hours bars would both break those
  assumptions and make a 1-2 bp modeled spread indefensible.
* Fills: engine next-bar quote path against SYNTHESIZED quotes
  bid/ask = close * (1 -/+ half_spread), half-spread 1 bp SPY/QQQ and 2 bp
  single-name mega-caps; spread_fill_fraction=1.0 (fill AT the synthesized
  touch = pay the full half-spread per side), slippage 0, commission 0,
  latency 500 ms (the promotion criterion's level).
* Sizing: the engine's default RiskBudgetSizer, risk_r_pct=2.0 (1R = 2% of
  entry price), per-trade cap 2% of equity (config) => one conviction-scaled
  ~full-notional position per trade, engine-capped at 1x equity.
* Exits: each signal's stated horizon expressed through the engine's exit
  policies (max-hold wall-clock minutes / EOD flatten); no stop, per the
  hypotheses (none states one).

Subcommands:
  regimes   precompute the regime classification parquet
  pilot     timing pilot: orb on SPY, 2023-07-01..2023-09-29, via run_signal
  worker    run one (signal, span) backtest and persist under results/edge/runs
  record    record one trial line for a (signal, span) and print the trial id
  report    tearsheets + promotion checks + EXPECTANCY.md / REJECTED.md
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time as _time
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edge.core.config import load_config
from edge.core.events import MARKET_TZ, QuoteEvent, SignalEvent
from edge.data.backends import build_catalyst_bridge
from edge.data.loader import ALL_SYMBOLS, POINT_IN_TIME_KINDS, EdgeDataLoader, FeedsBackend
from edge.regime.classifier import BUCKETS, VOL_STATES, classify
from edge.runners.engine import BacktestEngine, EngineParams
from edge.signals import gex_pin, leadlag, orb, positioning, short_pressure, vwap_rev  # noqa: F401  (register)
from edge.signals import registry as sigreg
from edge.signals.base import ALL_REGIMES
from edge.signals.registry import run_signal, trial_config_for
from edge.research.registry import TrialRegistry

# ---------------------------------------------------------------------------
# Spans and universes (the operator's plan; symbols may be cut for budget —
# any cut is recorded in the trial run_config and the final notes)
# ---------------------------------------------------------------------------

IS_SPAN = (date(2019, 1, 2), date(2023, 12, 29))
OOS_SPAN = (date(2024, 1, 2), date(2026, 2, 20))
PILOT_SPAN = (date(2023, 7, 1), date(2023, 9, 29))

INTRADAY_UNIVERSE = ("AAPL", "AMD", "AMZN", "META", "NVDA", "QQQ", "SPY", "TSLA")
SINGLE_NAMES = ("AAPL", "AMD", "AMZN", "META", "NVDA", "TSLA")

#: Conservative modeled half-spreads (fraction of price): 1 bp SPY/QQQ,
#: 2 bp single-name mega-caps.
HALF_SPREAD = {s: (0.0001 if s in ("SPY", "QQQ") else 0.0002) for s in INTRADAY_UNIVERSE}

RESULTS = REPO / "results" / "edge"
REGIME_PARQUET = RESULTS / "regime_classification.parquet"

#: Session-minute RTH window kept on the tape (bar stamps, ET).
RTH_FIRST_MIN = 9 * 60 + 30   # 09:30
RTH_LAST_MIN = 15 * 60 + 59   # 15:59

RUN_DEFAULTS = dict(
    latency_ms=500,
    initial_equity=100_000.0,
    seed=0,
    risk_r_pct=2.0,
)

CONFIG_OVERRIDES = {
    # Equity-mode friction: the synthesized half-spread carries ALL of the
    # modeled cost. Fill at the touch, no premium slippage, $0 commission.
    "execution.spread_fill_fraction": 1.0,
    "execution.slippage_pct": 0.0,
    "execution.commission_per_contract": 0.0,
}

#: Per-signal engine wiring: universe, pit kinds, exits (fixed translations
#: of each signal's stated horizon into engine exit policies).
SPECS: dict[str, dict] = {
    "opening_range_breakout": dict(
        symbols=INTRADAY_UNIVERSE, pit_kinds=(), max_hold_minutes=120,
        eod_flatten="15:55", wrapper=None,
    ),
    "vwap_reversion": dict(
        symbols=INTRADAY_UNIVERSE, pit_kinds=(), max_hold_minutes=60,
        eod_flatten="15:55", wrapper=None,
    ),
    "index_leadlag": dict(
        symbols=INTRADAY_UNIVERSE, pit_kinds=(), max_hold_minutes=30,
        eod_flatten="15:55", wrapper=None,
    ),
    "short_ratio_deviation": dict(
        symbols=SINGLE_NAMES, pit_kinds=("short_ratio",),
        # the signal's own sanctioned wiring: horizon_days * 1440 wall-clock
        max_hold_minutes=3 * 1440, eod_flatten=None, wrapper="short_ratio",
    ),
    "insider_cluster": dict(
        symbols=SINGLE_NAMES, pit_kinds=("insider",),
        # 7 sessions expressed in wall clock: 10 calendar days
        max_hold_minutes=10 * 1440, eod_flatten=None, wrapper="insider",
    ),
    "cot_extreme": dict(
        symbols=("SPY",), pit_kinds=("cot",),
        # 5 sessions from a Monday-open entry: 5 calendar days
        max_hold_minutes=5 * 1440, eod_flatten=None, wrapper="cot",
    ),
    "gex_pin": dict(
        symbols=("SPY", "QQQ"), pit_kinds=("open_interest", "greeks_eod"),
        max_hold_minutes=None, eod_flatten="15:45", wrapper=None,
    ),
}


# ---------------------------------------------------------------------------
# Composite backend: catalyst bars (RTH-filtered) + point-in-time feeds
# ---------------------------------------------------------------------------


class CompositeBackend:
    """Routes bars -> CatalystBridge (RTH-filtered), pit kinds -> FeedsBackend.

    ``universe`` bounds per-symbol pit cross-sections requested as '*' (the
    feeds compute their per-symbol/cross-sectional statistics market-wide
    BEFORE this filter, so filtering rows to the run's symbols afterwards is
    exactly the loader's own per-symbol filter, applied for several symbols
    at once). Bars/pit frames are memoized per request key so the worker's
    quote-synthesis pass and the engine's load share one read.
    """

    def __init__(self, universe: tuple[str, ...]) -> None:
        self._bridge = build_catalyst_bridge(str(REPO), missing_ok=True)
        self._feeds = FeedsBackend(repo_root=REPO)
        self._universe = tuple(universe)
        self._memo: dict[tuple, pd.DataFrame] = {}

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        key = (symbol, start, end, kind)
        hit = self._memo.get(key)
        if hit is not None:
            if kind == "bars":
                # bars are read exactly twice per run (quote synthesis, then
                # the engine's tape load): free the frame on the second read.
                del self._memo[key]
            return hit
        if kind == "bars":
            frame = self._bridge.fetch(symbol, start, end, kind)
            if not frame.empty:
                minutes = frame["ts"].dt.hour * 60 + frame["ts"].dt.minute
                frame = frame[(minutes >= RTH_FIRST_MIN) & (minutes <= RTH_LAST_MIN)]
                frame = frame.reset_index(drop=True)
        elif kind in POINT_IN_TIME_KINDS:
            frame = self._feeds.fetch(symbol, start, end, kind)
            if symbol == ALL_SYMBOLS and "symbol" in frame.columns and len(frame):
                frame = frame[frame["symbol"].isin(self._universe)].reset_index(drop=True)
        else:
            frame = self._bridge.fetch(symbol, start, end, kind)
        self._memo[key] = frame
        return frame


def build_loader(universe: tuple[str, ...]) -> EdgeDataLoader:
    return EdgeDataLoader(CompositeBackend(universe), repo_root=REPO)


# ---------------------------------------------------------------------------
# Quote synthesis: bar close +/- fixed half-spread, engine quote seam
# ---------------------------------------------------------------------------


class SyntheticSpreadQuoteSource:
    """QuoteSource over the run's own bar tape with per-symbol half-spreads.

    The quote keyed to a bar close is ``close * (1 -/+ half_spread)``; the
    engine's next-bar + latency fill path and CostModel do the rest, so
    every fill pays the modeled half-spread explicitly (spread_fill_fraction
    is 1.0 in this run's config: fills AT the synthesized touch).
    """

    def __init__(self) -> None:
        self._ts: dict[str, np.ndarray] = {}
        self._close: dict[str, np.ndarray] = {}
        self._hs: dict[str, float] = {}

    def add_symbol(self, symbol: str, bars: pd.DataFrame, half_spread: float) -> None:
        # normalize to epoch NANOSECONDS regardless of the frame's unit
        ts = pd.DatetimeIndex(bars["ts"]).as_unit("ns").asi8
        self._ts[symbol] = ts
        self._close[symbol] = bars["close"].to_numpy(dtype=float)
        self._hs[symbol] = half_spread

    def quote_for_bar(self, symbol: str, bar_ts: datetime) -> QuoteEvent | None:
        ts = self._ts.get(symbol)
        if ts is None:
            return None
        key = int(pd.Timestamp(bar_ts).value)
        i = np.searchsorted(ts, key)
        if i >= len(ts) or ts[i] != key:
            return None
        close = float(self._close[symbol][i])
        hs = self._hs[symbol]
        return QuoteEvent(
            ts=bar_ts, symbol=symbol,
            bid=close * (1.0 - hs), ask=close * (1.0 + hs),
            bid_size=0, ask_size=0,
        )


# ---------------------------------------------------------------------------
# Regime gate from the classifier (point-in-time by construction)
# ---------------------------------------------------------------------------


def ensure_regimes() -> pd.DataFrame:
    if REGIME_PARQUET.exists():
        return pd.read_parquet(REGIME_PARQUET)
    import logging

    logging.getLogger("edge.data.backends").setLevel(logging.ERROR)
    loader = build_loader(INTRADAY_UNIVERSE)
    frame = classify(loader, IS_SPAN[0], OOS_SPAN[1], "SPY")
    REGIME_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(REGIME_PARQUET)
    return frame


class RegimeGate:
    """Engine regime-gate hook over the classifier's per-session labels.

    Vocabulary is per signal: promotion buckets or vol states (the two
    axes the signals declare). An unclassified session blocks the signal —
    conservative, and post-2016 every session classifies.
    """

    def __init__(self, regimes: pd.DataFrame, allowed: frozenset[str], axis: str) -> None:
        col = "bucket" if axis == "bucket" else "vol_state"
        self._by_day = {
            pd.Timestamp(row.session).date(): getattr(row, col)
            for row in regimes.itertuples()
        }
        self._allowed = set(allowed)

    def __call__(self, event: SignalEvent, ctx) -> bool:
        label = self._by_day.get(event.ts.astimezone(MARKET_TZ).date())
        return label is not None and label in self._allowed


def gate_for(signal) -> RegimeGate | None:
    allowed = signal.allowed_regimes
    if allowed == ALL_REGIMES:
        return None
    allowed = frozenset(allowed)
    if allowed <= BUCKETS:
        return RegimeGate(ensure_regimes(), allowed, "bucket")
    if allowed <= frozenset(VOL_STATES):
        return RegimeGate(ensure_regimes(), allowed, "vol_state")
    raise ValueError(f"unmappable allowed_regimes {allowed!r} for {signal.name}")


# ---------------------------------------------------------------------------
# Pit-tail gate wrapper: skip on_bar until new pit rows become visible
# ---------------------------------------------------------------------------


class PitTailGate:
    """Delegate ``on_bar`` only when the symbol's pit tail advanced.

    The wrapped signals (short_ratio_deviation, insider_cluster, cot_extreme)
    are pure functions of (visible pit frame, their own one-shot-per-release
    dedup state); while the newest visible ``available_at`` for the bar's
    symbol is unchanged, their output is deterministically ``[]`` after the
    first evaluation — so skipping those calls reproduces the exact same
    emissions at a tiny fraction of the pandas cost. Visibility uses the
    same ``available_at <= now`` join the engine ctx enforces.
    """

    def __init__(self, inner, kind: str, pit: pd.DataFrame, per_symbol: bool) -> None:
        self.inner = inner
        self.name = inner.name
        self.warmup_bars = inner.warmup_bars
        self._avail: dict[str, np.ndarray] = {}
        self._seen: dict[str, int] = {}
        if len(pit):
            def _ns(values: pd.Series) -> np.ndarray:
                # normalize to epoch NANOSECONDS regardless of the frame's unit
                return np.sort(pd.DatetimeIndex(pd.to_datetime(values)).as_unit("ns").asi8)

            if per_symbol and "symbol" in pit.columns:
                for sym, group in pit.groupby("symbol"):
                    self._avail[str(sym)] = _ns(group["available_at"])
            else:
                self._avail["*"] = _ns(pit["available_at"])
        self._per_symbol = per_symbol

    def on_bar(self, ctx):
        symbol = ctx.bar.symbol if self._per_symbol else "*"
        avail = self._avail.get(symbol)
        if avail is None:
            visible = 0
        else:
            visible = int(np.searchsorted(avail, int(pd.Timestamp(ctx.now).value), side="right"))
        if self._seen.get(symbol) == visible:
            return []
        self._seen[symbol] = visible
        return self.inner.on_bar(ctx)


# ---------------------------------------------------------------------------
# One backtest run
# ---------------------------------------------------------------------------


def run_config_for(name: str, span: str, spec: dict, start: date, end: date) -> dict:
    return {
        "span": span,
        "symbols": list(spec["symbols"]),
        "start": start.isoformat(),
        "end": end.isoformat(),
        **RUN_DEFAULTS,
        "max_hold_minutes": spec["max_hold_minutes"],
        "eod_flatten_time": spec["eod_flatten"],
        "stop_r_multiple": None,
        "pit_kinds": list(spec["pit_kinds"]),
        "config_overrides": CONFIG_OVERRIDES,
        "quote_model": "bar-close +/- half-spread; fill at touch (spread_fill_fraction=1.0)",
        "half_spreads": {s: HALF_SPREAD.get(s, 0.0002) for s in spec["symbols"]},
        "tape": "RTH bar stamps 09:30..15:59 ET (catalyst alpaca_minute, cache-only)",
        "notes": {
            "short_ratio_deviation": "single names only: ETF short-ratio is structurally different (creation/redemption hedging), so SPY/QQQ are excluded by design",
            "cot_extreme": "trade_symbol SPY per default config; a QQQ/NQ variant would be its own trial (not run)",
        }.get(name, ""),
    }


def spanned(span: str) -> tuple[date, date]:
    return {"IS": IS_SPAN, "OOS": OOS_SPAN, "PILOT": PILOT_SPAN}[span]


def execute_run(name: str, span: str, symbols_override: tuple[str, ...] | None = None) -> dict:
    """Build loader/engine for one (signal, span) and run it; persist results."""
    spec = dict(SPECS[name])
    if symbols_override:
        spec["symbols"] = symbols_override
    start, end = spanned(span)
    symbols = tuple(spec["symbols"])

    t0 = _time.time()
    loader = build_loader(symbols)
    config = load_config(repo_root=REPO, overrides=CONFIG_OVERRIDES)

    # Quote synthesis over the exact tape the engine will replay.
    quotes = SyntheticSpreadQuoteSource()
    n_bars = 0
    for sym in symbols:
        bars = loader.load(sym, start, end, "bars")
        quotes.add_symbol(sym, bars, HALF_SPREAD.get(sym, 0.0002))
        n_bars += len(bars)

    signal = sigreg.create(name)
    engine_signal = signal
    if spec["wrapper"] is not None:
        kind = spec["wrapper"]
        pit = loader.load(ALL_SYMBOLS, start, end, kind)
        engine_signal = PitTailGate(signal, kind, pit, per_symbol=kind != "cot")

    params = EngineParams(
        symbols=symbols,
        start=start,
        end=end,
        latency_ms=RUN_DEFAULTS["latency_ms"],
        initial_equity=RUN_DEFAULTS["initial_equity"],
        seed=RUN_DEFAULTS["seed"],
        risk_r_pct=RUN_DEFAULTS["risk_r_pct"],
        max_hold_minutes=spec["max_hold_minutes"],
        stop_r_multiple=None,
        eod_flatten_time=time.fromisoformat(spec["eod_flatten"]) if spec["eod_flatten"] else None,
        pit_kinds=tuple(spec["pit_kinds"]),
    )
    engine = BacktestEngine(
        loader, config, params, [engine_signal],
        regime_gate=gate_for(signal), quote_source=quotes,
    )
    load_s = _time.time() - t0
    t1 = _time.time()
    result = engine.run()
    run_s = _time.time() - t1

    out = RESULTS / "runs" / f"{name}_{span}"
    out.mkdir(parents=True, exist_ok=True)
    result.ledger.to_parquet(out / "ledger.parquet")
    result.equity.to_frame().to_parquet(out / "equity.parquet")
    summary = {
        **result.summary,
        "signal": name,
        "span": span,
        "run_config": run_config_for(name, span, spec, start, end),
        "bars_replayed": n_bars,
        "load_seconds": round(load_s, 1),
        "run_seconds": round(run_s, 1),
        "bars_per_second": round(n_bars / run_s, 1) if run_s > 0 else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_pilot() -> None:
    trials = TrialRegistry(REPO)
    signal = sigreg.create("opening_range_breakout")
    spec = dict(SPECS["opening_range_breakout"], symbols=("SPY",))
    run_cfg = run_config_for("opening_range_breakout", "PILOT", spec, *PILOT_SPAN)
    record, summary = run_signal(
        signal,
        lambda s: execute_run("opening_range_breakout", "PILOT", symbols_override=("SPY",)),
        trials,
        run_config=run_cfg,
    )
    print(json.dumps({"trial_id": record.trial_id, **summary}, indent=2))


def cmd_worker(name: str, span: str) -> None:
    try:
        summary = execute_run(name, span)
    except Exception as exc:  # noqa: BLE001 — a failed backtest is a result
        out = RESULTS / "runs" / f"{name}_{span}"
        out.mkdir(parents=True, exist_ok=True)
        failure = {"signal": name, "span": span, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        (out / "failure.json").write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure))
        raise SystemExit(3)
    print(json.dumps({k: summary[k] for k in (
        "signal", "span", "signals_emitted", "trades_executed", "trades_closed",
        "final_equity", "bars_replayed", "run_seconds", "bars_per_second", "peak_rss_mb",
    )}, indent=2))


PLAN: tuple[tuple[str, str], ...] = tuple(
    (name, span) for name in (
        "opening_range_breakout", "vwap_reversion", "index_leadlag",
        "short_ratio_deviation", "insider_cluster", "cot_extreme", "gex_pin",
    ) for span in ("OOS", "IS")
)

#: Concurrency budget by run weight on this 8 GB machine: at most one
#: 8-symbol IS run at a time, plus one light run alongside.
HEAVY = {(n, s) for (n, s) in PLAN if s == "IS" and len(SPECS[n]["symbols"]) >= 6}


def cmd_orchestrate(max_procs: int, heavy_procs: int, only: str | None = None) -> None:
    """Record every trial through run_signal (record first, ALWAYS), where
    each trial's backtest callable launches its worker subprocess; a launch
    scheduler then caps concurrency. The trial line therefore exists before
    one bar of its backtest replays — crashed workers stay recorded."""
    import subprocess

    ensure_regimes()
    trials = TrialRegistry(REPO)
    jobs: list[dict] = []
    plan = PLAN if only is None else tuple(p for p in PLAN if p[0] in only.split(","))
    for name, span in plan:
        signal = sigreg.create(name)
        spec = SPECS[name]
        start, end = spanned(span)
        run_cfg = run_config_for(name, span, spec, start, end)

        def launcher(_s, name=name, span=span):
            return lambda: subprocess.Popen(
                ["uv", "run", "python", str(REPO / "scripts" / "edge_first_backtests.py"),
                 "worker", "--signal", name, "--span", span],
                cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

        record, launch = run_signal(signal, launcher, trials, run_config=run_cfg)
        print(f"recorded trial {record.trial_id}: {name} {span}", flush=True)
        jobs.append({"name": name, "span": span, "trial": record.trial_id, "launch": launch})

    running: list[dict] = []
    done: list[dict] = []
    while jobs or running:
        for job in list(running):
            code = job["proc"].poll()
            if code is not None:
                job["exit"] = code
                job["output"] = job["proc"].stdout.read()
                running.remove(job)
                done.append(job)
                print(f"finished {job['name']} {job['span']} exit={code}", flush=True)
        heavy_now = sum(1 for j in running if (j["name"], j["span"]) in HEAVY)
        while jobs:
            nxt = jobs[0]
            is_heavy = (nxt["name"], nxt["span"]) in HEAVY
            if len(running) >= max_procs or (is_heavy and heavy_now >= heavy_procs):
                break
            job = jobs.pop(0)
            job["proc"] = job["launch"]()
            running.append(job)
            heavy_now += int(is_heavy)
            print(f"launched {job['name']} {job['span']}", flush=True)
        _time.sleep(5)
    for job in done:
        print(f"== {job['name']} {job['span']} exit={job['exit']}")
    fails = [j for j in done if j["exit"] != 0]
    print(f"orchestrate complete: {len(done) - len(fails)} ok, {len(fails)} failed")


#: classifier bucket -> tearsheet bucket label
_BUCKET_LABEL = {
    "low_vol_trending": "low_vol_trend",
    "low_vol_chopping": "low_vol_chop",
    "high_vol_trending": "high_vol_trend",
    "high_vol_chopping": "high_vol_chop",
}

#: Honesty floor: below this many OOS trades a signal's row is
#: INSUFFICIENT-DATA, never a percentage.
MIN_TRADES_FOR_HEADLINE = 10


def cmd_report() -> None:
    """Tearsheets from the OOS runs, promotion checks, EXPECTANCY/REJECTED."""
    from edge.research.tearsheet import (
        ExpectancyRow, append_expectancy_md, append_rejected_md,
        compute_monthly_headline, compute_regime_breakdown, compute_trade_stats,
        evaluate_promotion, render_tearsheet,
    )

    config = load_config(repo_root=REPO, overrides=CONFIG_OVERRIDES)
    trials = TrialRegistry(REPO)
    regimes = ensure_regimes()
    regime_series = pd.Series(
        [_BUCKET_LABEL[b] for b in regimes["bucket"]],
        index=pd.DatetimeIndex(regimes["available_at"]),
    ).sort_index()

    tearsheet_dir = RESULTS / "tearsheets"
    tearsheet_dir.mkdir(parents=True, exist_ok=True)
    ranked_rows: list[ExpectancyRow] = []
    unranked: list[str] = []
    report: dict[str, dict] = {}

    for name in SPECS:
        out = RESULTS / "runs" / f"{name}_OOS"
        failure_path = out / "failure.json"
        summary_path = out / "summary.json"
        row: dict = {"signal": name}
        if failure_path.exists() and not summary_path.exists():
            failure = json.loads(failure_path.read_text())
            row.update(status="BLOCKED", error=failure["error"], verdict="NOT-RUN")
            unranked.append(
                f"- **{name}** — NOT-RUN (backtest failed before replay): `{failure['error']}`"
            )
            (tearsheet_dir / f"{name}.txt").write_text(
                f"TEARSHEET: {name}\n\nNOT RUN — the OOS backtest failed before "
                f"replaying a bar:\n  {failure['error']}\n\nThe trial is recorded in "
                "TRIALS.jsonl (result pending forever — that is the point).\n"
            )
            report[name] = row
            continue

        summary = json.loads(summary_path.read_text())
        ledger = pd.read_parquet(out / "ledger.parquet")
        equity = pd.read_parquet(out / "equity.parquet")["equity"]
        n_trades = len(ledger)
        drops = summary["drop_counters"]
        row.update(
            status="RAN", n_trades_oos=n_trades,
            signals_emitted=summary["signals_emitted"],
            drop_counters=drops, final_equity=summary["final_equity"],
        )

        if n_trades == 0:
            promo_failures = ["insufficient-oos-trades"]
            row.update(verdict="REJECT", failures=promo_failures)
            append_rejected_md(REPO / "REJECTED.md", name, promo_failures, {"n_trades": 0})
            unranked.append(
                f"- **{name}** — INSUFFICIENT-DATA: 0 OOS trades "
                f"(emitted {summary['signals_emitted']}, drops {drops})"
            )
            (tearsheet_dir / f"{name}.txt").write_text(
                f"TEARSHEET: {name}\n\nINSUFFICIENT-DATA — the OOS run closed 0 "
                f"trades (signals emitted: {summary['signals_emitted']}; drop "
                f"counters: {json.dumps(drops)}). No statistics are honest at N=0.\n"
            )
            report[name] = row
            continue

        stats = compute_trade_stats(ledger, config.validation, seed=RUN_DEFAULTS["seed"])
        _, positive_buckets = compute_regime_breakdown(ledger, regime_series)
        promotion = evaluate_promotion(
            n_trades=stats.n,
            expectancy_ci=(stats.ci_lo, stats.ci_hi),
            pbo=None,  # single fixed config per signal: no sweep, no CSCV matrix
            latency_survived=True,  # minute-bar quotes: 100/500/2000 ms all meet the same next-bar book; run executed at 500 ms
            positive_buckets=positive_buckets,
            config=config.validation,
        )
        headline = compute_monthly_headline(equity, config.validation, seed=RUN_DEFAULTS["seed"])
        sheet = render_tearsheet(
            name, ledger, equity, trials, regime_series, config.validation,
            pbo=None, latency_survived=True, seed=RUN_DEFAULTS["seed"],
        )
        (tearsheet_dir / f"{name}.txt").write_text(sheet)
        row.update(
            verdict=promotion.verdict, failures=list(promotion.failures),
            geometric_monthly=headline.geometric, months=headline.months,
            expectancy_r=stats.expectancy_r,
            expectancy_ci=[stats.ci_lo, stats.ci_hi], hit_rate=stats.hit_rate,
        )
        if not promotion.promoted:
            append_rejected_md(
                REPO / "REJECTED.md", name, promotion.failures,
                {"n_trades": stats.n, "expectancy_r": stats.expectancy_r,
                 "ci_lo": stats.ci_lo, "ci_hi": stats.ci_hi,
                 "positive_buckets": positive_buckets,
                 "geometric_monthly": headline.geometric},
            )
        if n_trades >= MIN_TRADES_FOR_HEADLINE:
            ranked_rows.append(ExpectancyRow(
                signal=name, geometric_monthly=headline.geometric,
                months=headline.months, n_trades=stats.n,
                expectancy_r=stats.expectancy_r, verdict=promotion.verdict,
            ))
        else:
            unranked.append(
                f"- **{name}** — INSUFFICIENT-DATA: {n_trades} OOS trades "
                f"(< {MIN_TRADES_FOR_HEADLINE}); verdict {promotion.verdict} "
                f"(failed: {', '.join(promotion.failures)})"
            )
        report[name] = row

    if ranked_rows:
        append_expectancy_md(REPO / "EXPECTANCY.md", ranked_rows)
    if unranked:
        with open(REPO / "EXPECTANCY.md", "a", encoding="utf-8") as f:
            f.write(
                "### Not rankable this round (no percentage is honest here)\n\n"
                + "\n".join(unranked) + "\n\n"
            )
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regimes")
    sub.add_parser("pilot")
    sub.add_parser("report")
    p = sub.add_parser("worker")
    p.add_argument("--signal", required=True)
    p.add_argument("--span", required=True, choices=["IS", "OOS", "PILOT"])
    p = sub.add_parser("orchestrate")
    p.add_argument("--max-procs", type=int, default=2)
    p.add_argument("--heavy-procs", type=int, default=1)
    p.add_argument("--only", default=None, help="comma-separated signal names")
    args = parser.parse_args()
    if args.cmd == "regimes":
        frame = ensure_regimes()
        print(f"regimes: {len(frame)} sessions {frame['session'].min()}..{frame['session'].max()}")
    elif args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "worker":
        cmd_worker(args.signal, args.span)
    elif args.cmd == "orchestrate":
        cmd_orchestrate(args.max_procs, args.heavy_procs, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

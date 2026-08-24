"""Research campaign runner: trial -> backtest -> ledger -> tearsheet.

One reusable driver for every signal run, so a campaign is reproducible from
the repository rather than from a scratch script. It owns exactly the wiring
the referee does not:

* :class:`SyntheticQuoteSource` — the sanctioned ``QuoteSource`` seam for
  equity runs off minute bars. There is no NBBO tape for the equity
  universe, so a bar's book is modelled as ``close * (1 -/+ half_spread)``
  and fills cross to the touch (``execution.spread_fill_fraction = 1.0``,
  a full half-spread paid per side). The half-spreads are an INPUT, stated
  per run and recorded in the trial config — never a fitted quantity.
* trial-first ordering — every run goes through
  :func:`edge.signals.registry.run_signal`, so the hypothesis and config
  land in the append-only registry BEFORE the backtest, whether or not the
  result is ever looked at.
* persistence — ledger/equity/summary per run, tearsheet from the OOS run
  only, and the EXPECTANCY.md / REJECTED.md ledgers.

Nothing here may weaken the referee: cost model, validation, promotion
criteria, and the lockbox live elsewhere and are read-only from this module.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from edge.core.config import EdgeConfig
from edge.core.events import MARKET_TZ, QuoteEvent
from edge.data.loader import EdgeDataLoader
from edge.regime.classifier import classify
from edge.research.registry import TrialRegistry
from edge.research.tearsheet import (
    ExpectancyRow,
    append_expectancy_md,
    append_rejected_md,
    render_tearsheet,
)
from edge.runners.engine import BacktestEngine, EngineParams, RunResult
from edge.signals.base import Signal
from edge.signals.registry import run_signal

logger = logging.getLogger(__name__)

#: Nominal displayed size for synthesized quotes. Large enough that the
#: partial-fill path never binds on equity runs; the spread, not depth, is
#: what these runs are measuring.
NOMINAL_QUOTE_SIZE: int = 100_000


class SyntheticQuoteSource:
    """Bar-derived top of book: ``close * (1 -/+ half_spread)``.

    Honest about what it is: a MODEL of the book, not a tape. Half-spreads
    are supplied per symbol (with a default for anything unlisted) and are
    recorded into the run config so a reader can see the assumption that
    produced the fills.
    """

    def __init__(
        self,
        closes: Mapping[tuple[str, datetime], float],
        half_spreads: Mapping[str, float],
        default_half_spread: float,
    ) -> None:
        self._closes = dict(closes)
        self._half_spreads = dict(half_spreads)
        self._default = float(default_half_spread)

    def quote_for_bar(self, symbol: str, bar_ts: datetime) -> QuoteEvent | None:
        close = self._closes.get((symbol, bar_ts))
        if close is None or close <= 0.0:
            return None
        half = self._half_spreads.get(symbol, self._default)
        return QuoteEvent(
            ts=bar_ts,
            symbol=symbol,
            bid=close * (1.0 - half),
            ask=close * (1.0 + half),
            bid_size=NOMINAL_QUOTE_SIZE,
            ask_size=NOMINAL_QUOTE_SIZE,
        )


def build_quote_source(
    loader: EdgeDataLoader,
    symbols: Sequence[str],
    start: date,
    end: date,
    half_spreads: Mapping[str, float],
    default_half_spread: float,
) -> SyntheticQuoteSource:
    """Load bars through the gateway and model each bar's book from its close."""
    closes: dict[tuple[str, datetime], float] = {}
    for symbol in symbols:
        frame = loader.load(symbol, start, end, "bars")
        if frame is None or frame.empty:
            continue
        stamps = pd.to_datetime(frame["ts"] if "ts" in frame.columns else frame.index)
        for stamp, close in zip(stamps, frame["close"]):
            closes[(symbol, stamp.to_pydatetime())] = float(close)
    return SyntheticQuoteSource(closes, half_spreads, default_half_spread)


@dataclass(frozen=True)
class RunSpec:
    """One (signal, span) backtest: everything that shapes it, stated."""

    signal_name: str
    build_signal: Callable[[], Signal]
    span: str                      # 'IS' | 'OOS' — only OOS is ever headlined
    symbols: tuple[str, ...]
    start: date
    end: date
    half_spreads: Mapping[str, float] = field(default_factory=dict)
    default_half_spread: float = 0.0002
    latency_ms: int = 500
    initial_equity: float = 100_000.0
    risk_r_pct: float = 2.0
    max_hold_minutes: int | None = None
    stop_r_multiple: float | None = None
    eod_flatten_time: time | None = None
    #: Execution parameters this run APPLIES (and therefore records). Equity
    #: runs model their whole cost in the synthesized spread, so the
    #: option-oriented percentage slippage and per-contract commission are
    #: switched off and the fill crosses the full half-spread.
    execution_overrides: Mapping[str, float] = field(
        default_factory=lambda: {
            "spread_fill_fraction": 1.0,
            "slippage_pct": 0.0,
            "commission_per_contract": 0.0,
        }
    )
    pit_kinds: tuple[str, ...] = ()
    options_pit: tuple[Mapping[str, Any], ...] = ()
    seed: int = 0

    @property
    def run_id(self) -> str:
        return f"{self.signal_name}_{self.span}"


def _run_config(spec: RunSpec) -> dict[str, Any]:
    """The experiment description recorded in the trial line."""
    return {
        "span": spec.span,
        "symbols": list(spec.symbols),
        "start": spec.start.isoformat(),
        "end": spec.end.isoformat(),
        "latency_ms": spec.latency_ms,
        "initial_equity": spec.initial_equity,
        "seed": spec.seed,
        "risk_r_pct": spec.risk_r_pct,
        "max_hold_minutes": spec.max_hold_minutes,
        "stop_r_multiple": spec.stop_r_multiple,
        "eod_flatten_time": spec.eod_flatten_time.isoformat() if spec.eod_flatten_time else None,
        "pit_kinds": list(spec.pit_kinds),
        "options_pit": [dict(o) for o in spec.options_pit],
        "quote_model": "bar-close +/- half-spread; fill at touch (spread_fill_fraction=1.0)",
        "half_spreads": dict(spec.half_spreads),
        "default_half_spread": spec.default_half_spread,
        "config_overrides": {
            f"execution.{k}": v for k, v in sorted(spec.execution_overrides.items())
        },
    }


def execute(
    spec: RunSpec,
    loader: EdgeDataLoader,
    config: EdgeConfig,
    trials: TrialRegistry,
    out_root: Path,
) -> RunResult | None:
    """Record the trial, run the backtest, persist artifacts.

    Returns the :class:`RunResult`, or ``None`` when the backtest raised —
    in which case ``failure.json`` is written and the trial line still
    stands. A failed experiment is a recorded experiment.
    """
    out_dir = out_root / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    params = EngineParams(
        symbols=spec.symbols,
        start=spec.start,
        end=spec.end,
        latency_ms=spec.latency_ms,
        initial_equity=spec.initial_equity,
        seed=spec.seed,
        risk_r_pct=spec.risk_r_pct,
        max_hold_minutes=spec.max_hold_minutes,
        stop_r_multiple=spec.stop_r_multiple,
        eod_flatten_time=spec.eod_flatten_time,
        pit_kinds=spec.pit_kinds,
        options_pit=spec.options_pit,
    )
    quotes = build_quote_source(
        loader, spec.symbols, spec.start, spec.end,
        spec.half_spreads, spec.default_half_spread,
    )

    # Apply what the run records: the engine sees exactly the execution
    # parameters _run_config() reports. Recording an override without
    # applying it is the one dishonesty this runner must never commit.
    run_config_obj = config.model_copy(
        update={"execution": config.execution.model_copy(
            update=dict(spec.execution_overrides))}
    )

    def backtest(signal: Signal) -> RunResult:
        engine = BacktestEngine(
            loader, run_config_obj, params, [signal], quote_source=quotes
        )
        return engine.run()

    try:
        _record, result = run_signal(
            spec.build_signal(), backtest, trials, run_config=_run_config(spec)
        )
    except Exception as exc:  # noqa: BLE001 — a failed run is data, not a crash
        logger.warning("%s failed: %s: %s", spec.run_id, type(exc).__name__, exc)
        (out_dir / "failure.json").write_text(json.dumps({
            "signal": spec.signal_name,
            "span": spec.span,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2))
        return None

    result.ledger.to_parquet(out_dir / "ledger.parquet")
    result.equity.rename("equity").to_frame().to_parquet(out_dir / "equity.parquet")
    summary = dict(result.summary)
    summary.update({"signal": spec.signal_name, "span": spec.span,
                    "run_config": _run_config(spec)})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return result


def tearsheet_for(
    signal_name: str,
    result: RunResult,
    loader: EdgeDataLoader,
    config: EdgeConfig,
    trials: TrialRegistry,
    spec: RunSpec,
    out_root: Path,
    *,
    index_symbol: str = "SPY",
) -> str:
    """Render and persist the OOS tearsheet for one signal."""
    regime_frame = classify(loader, spec.start, spec.end, symbol_index=index_symbol)
    # The tearsheet ffills each trade's entry_ts onto this index, so it must be
    # tz-aware ET like the ledger stamps: index each session at its own ET
    # midnight, and a trade at any hour of that session picks up its bucket.
    sessions = pd.to_datetime(regime_frame["session"])
    if sessions.dt.tz is None:
        sessions = sessions.dt.tz_localize(MARKET_TZ)
    else:
        sessions = sessions.dt.tz_convert(MARKET_TZ)
    regime = pd.Series(
        regime_frame["bucket"].to_numpy(), index=pd.DatetimeIndex(sessions)
    ).sort_index()
    text = render_tearsheet(
        signal_name,
        result.ledger,
        result.equity,
        trials,
        regime,
        config.validation,
        pbo=None,
        # Minute-bar quote granularity: every configured latency scenario
        # meets the same next-bar book, so the run IS the fill at each.
        latency_survived=True,
        seed=spec.seed,
    )
    out_dir = out_root / "tearsheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{signal_name}.txt").write_text(text)
    return text


__all__ = [
    "NOMINAL_QUOTE_SIZE",
    "ExpectancyRow",
    "RunSpec",
    "SyntheticQuoteSource",
    "append_expectancy_md",
    "append_rejected_md",
    "build_quote_source",
    "execute",
    "tearsheet_for",
]

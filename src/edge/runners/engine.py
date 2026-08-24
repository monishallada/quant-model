"""Event-driven backtest engine: replay, visibility, fills, ledger, drops.

The engine replays minute :class:`~edge.core.events.BarEvent` streams for
every configured symbol, merged into one time-ordered tape and driven
through a :class:`~edge.core.events.BacktestClock`. At each bar close ``t``
it calls every enabled signal's ``on_bar(ctx)`` where the
:class:`MarketContext` exposes ONLY data available at ``t``:

* bars — every bar the context can return closed at or before ``t`` (the
  context reads from the engine's already-replayed history, so a future bar
  is unreachable by construction);
* slow point-in-time feeds — rows join on ``available_at <= t`` (the
  loader's publication-time column), filtered by the ENGINE, never by the
  signal;
* per-expiry options point-in-time frames — the CatalystBridge EOD kinds
  (:data:`OPTIONS_PIT_KINDS`: ``greeks_eod`` / ``open_interest``, symbol
  grammar ``"UNDERLYING:YYYY-MM-DD"``) declared via
  ``EngineParams.options_pit`` and served through
  :meth:`MarketContext.options_pit`. Because the expiry is part of the
  symbol, these cannot be preloaded once via ``ALL_SYMBOLS`` like the feed
  kinds; the engine instead loads each (symbol, ET session) lazily through
  the loader on first request and caches the frame for the rest of that
  session. Visibility is enforced ENGINE-side exactly like the feeds:
  only rows with ``available_at <= t`` are returned (so day-D open
  interest — stamped next-business-day 09:00 ET by the bridge — is never
  visible on day D). The loads are reads through the gateway loader; with
  the default cache-only bridge they can never trigger a terminal fetch.

Signal pipeline (order is fixed and each drop stage has a named counter):

    SignalEvent -> regime gate -> daily halt -> position-already-open
        -> sizing -> risk cap -> execution at t+latency

Execution prices off the NEXT bar's quote via
:meth:`~edge.execution.costmodel.CostModel.fill_with_latency`: the decision
quote is the tape's quote for the decision bar, and the arrival quote is
the next bar's quote re-stamped at the true arrival instant
``t + latency_ms`` (the order lands inside the next bar's interval; that
bar's quote is the book it meets). No next bar means no fill — never a
same-bar fill. Equity mode is first-class (``contract_multiplier=1``,
quantities are shares); an option leg slots in by supplying a different
``quote_source`` and multiplier — the engine loop does not change.

Accounting identity, asserted on every run:

    signals emitted == trades executed + sum(all drop counters)

:meth:`BacktestEngine.run` returns a :class:`RunResult` and NEVER writes to
``results/`` — persistence is the caller's decision.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from itertools import groupby, pairwise
from typing import Any, Final, Literal, NamedTuple, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.config import EdgeConfig
from edge.core.events import MARKET_TZ, BacktestClock, BarEvent, QuoteEvent, Side, SignalEvent
from edge.data.loader import ALL_SYMBOLS, POINT_IN_TIME_KINDS, EdgeDataLoader
from edge.execution.costmodel import CostModel, FillCosts, FillResult, FillStatus
from edge.research.registry import canonical_config_hash

# ---------------------------------------------------------------------------
# Named drop stages (the v16 instrumentation contract)
# ---------------------------------------------------------------------------

DROP_REGIME: str = "regime-gated"
DROP_SIZING: str = "sizing-zero"
DROP_NO_QUOTE: str = "no-quote"
DROP_LATENCY: str = "latency-no-fill"
DROP_POSITION_OPEN: str = "position-already-open"
DROP_RISK_CAP: str = "risk-cap"
DROP_DAILY_HALT: str = "daily-halt"

#: Every stage that can drop a signal. The run summary always carries all of
#: them (zeros included) so a missing counter is a bug, not an absence.
DROP_STAGES: tuple[str, ...] = (
    DROP_REGIME,
    DROP_SIZING,
    DROP_NO_QUOTE,
    DROP_LATENCY,
    DROP_POSITION_OPEN,
    DROP_RISK_CAP,
    DROP_DAILY_HALT,
)

#: Exit reasons emitted by the built-in policies plus the terminal flatten.
EXIT_END_OF_DATA: str = "end_of_data"

#: The ledger's non-negotiable column set, in fixed order.
LEDGER_COLUMNS: tuple[str, ...] = (
    "symbol",
    "side",
    "qty",
    "signal_name",
    "conviction",
    "entry_ts",
    "entry_px",
    "exit_ts",
    "exit_px",
    "exit_reason",
    "pnl",
    "r_multiple",
    "spread_cost",
    "slippage_cost",
    "commission",
    "mfe",
    "mae",
    "holding_minutes",
)


#: Per-expiry options point-in-time kinds the engine can serve lazily via
#: :meth:`MarketContext.options_pit` — the CatalystBridge EOD datasets whose
#: symbol grammar is ``"UNDERLYING:YYYY-MM-DD"`` (the expiration). Disjoint
#: from :data:`~edge.data.loader.POINT_IN_TIME_KINDS` by construction.
OPTIONS_PIT_KINDS: Final[tuple[str, ...]] = ("greeks_eod", "open_interest")


class EngineInvariantError(AssertionError):
    """An engine bookkeeping invariant failed — always a bug, never data."""


# ---------------------------------------------------------------------------
# Run parameters (config-sweepable; no numbers hardcoded in the engine)
# ---------------------------------------------------------------------------


class OptionsPitSpec(BaseModel):
    """One per-expiry options dataset a run declares it will read.

    ``kind`` is a CatalystBridge EOD kind (:data:`OPTIONS_PIT_KINDS`);
    ``symbols`` are the plain UNDERLYINGS (``"SPY"``) requests may name.
    The per-expiry part belongs to the ``ctx.options_pit`` request symbol
    (``"SPY:2026-01-16"``, the bridge grammar), never to the declaration —
    expiries roll with the calendar, so they cannot be enumerated up front;
    that is exactly why these kinds are loaded lazily per (symbol, session)
    instead of once via ``ALL_SYMBOLS`` like ``EngineParams.pit_kinds``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["open_interest", "greeks_eod"]
    symbols: tuple[str, ...] = Field(min_length=1)


class EngineParams(BaseModel):
    """One backtest run's shape. Frozen; sweeps build new instances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: tuple[str, ...] = Field(min_length=1)
    start: date
    end: date
    #: One latency scenario per run (sweep the config list across runs).
    latency_ms: int = Field(ge=0)
    initial_equity: float = Field(gt=0.0)
    #: Recorded into the config hash; hooks may draw from the engine's rng.
    seed: int = 0
    #: Per-share risk unit for the default sizer, as % of the reference price.
    risk_r_pct: float = Field(gt=0.0)
    #: Exit policies — each None disables that policy, so an exit-policy
    #: comparison is a config sweep, never a code change.
    max_hold_minutes: int | None = Field(default=None, gt=0)
    stop_r_multiple: float | None = Field(default=None, gt=0.0)
    eod_flatten_time: time | None = None
    #: Slow point-in-time kinds to preload and expose through the context.
    pit_kinds: tuple[str, ...] = ()
    #: Per-expiry options datasets (kind + underlyings) served lazily
    #: through ``ctx.options_pit``; empty tuple = none served.
    options_pit: tuple[OptionsPitSpec, ...] = ()
    #: 1 = equity shares. The options wrapper passes the OCC multiplier.
    contract_multiplier: int = Field(default=1, gt=0)


# ---------------------------------------------------------------------------
# Hook protocols
# ---------------------------------------------------------------------------


class SizeDecision(NamedTuple):
    """A sizing hook's answer: quantity and the per-share risk unit."""

    qty: int
    risk_per_share: float


class Sizer(Protocol):
    """Sizing hook: signal + reference price + current equity -> size."""

    def size(self, signal: SignalEvent, ref_price: float, equity: float) -> SizeDecision:
        """Return the quantity to trade and the per-share risk at entry."""
        ...


@dataclass(frozen=True)
class RiskBudgetSizer:
    """Default sizer: conviction-scaled risk budget over a per-share risk unit.

    ``risk_per_share = ref_price * risk_r_pct / 100``;
    ``qty = floor(equity * per_trade_cap_pct/100 * conviction / (risk_per_share * mult))``.
    """

    per_trade_cap_pct: float
    risk_r_pct: float
    multiplier: int = 1

    def size(self, signal: SignalEvent, ref_price: float, equity: float) -> SizeDecision:
        risk_per_share = ref_price * self.risk_r_pct / 100.0
        if risk_per_share <= 0.0:
            return SizeDecision(qty=0, risk_per_share=risk_per_share)
        budget = equity * self.per_trade_cap_pct / 100.0 * signal.conviction
        qty = int(budget // (risk_per_share * self.multiplier))
        return SizeDecision(qty=qty, risk_per_share=risk_per_share)


#: Regime gate hook: True lets the signal through, False drops it
#: (counted under ``regime-gated``).
RegimeGate = Callable[[SignalEvent, "MarketContext"], bool]


class Signal(Protocol):
    """A strategy: reads the context at each bar close, may emit a signal."""

    name: str

    def on_bar(self, ctx: MarketContext) -> "SignalEvent | list[SignalEvent] | None":
        """Return SignalEvent(s) stamped at ``ctx.now``, or None.

        A single event, a list (edge.signals.base.Signal returns lists), or
        None are all accepted; the engine unwraps uniformly.
        """
        ...


# ---------------------------------------------------------------------------
# Positions and exit policies
# ---------------------------------------------------------------------------


@dataclass
class OpenPosition:
    """One live position plus the path statistics accumulated while held."""

    symbol: str
    side: Side
    qty: int
    signal_name: str
    conviction: float
    risk_per_share: float
    entry_decision_ts: datetime
    entry_ts: datetime  # fill (arrival) instant: decision + latency
    entry_px: float
    entry_costs: FillCosts
    high_water: float = float("-inf")
    low_water: float = float("inf")

    def observe(self, bar: BarEvent) -> None:
        """Fold one held bar into the MFE/MAE path extremes."""
        self.high_water = max(self.high_water, bar.high)
        self.low_water = min(self.low_water, bar.low)

    @property
    def direction(self) -> int:
        return 1 if self.side is Side.BUY else -1


class ExitPolicy(Protocol):
    """One named exit rule; the engine records ``name`` as ``exit_reason``."""

    name: str

    def should_exit(self, position: OpenPosition, bar: BarEvent) -> bool:
        """True if this bar should trigger an exit for ``position``."""
        ...


@dataclass(frozen=True)
class StopLossExit:
    """Stop at ``k*R`` on the underlying move, checked on the BAR PATH.

    Long: triggers when ``bar.low <= entry_px - k * risk_per_share``;
    short mirrors on ``bar.high``. The exit still executes through the
    next-bar fill path — no intrabar fill at the stop level is assumed.
    """

    r_multiple: float
    name: str = "stop"

    def should_exit(self, position: OpenPosition, bar: BarEvent) -> bool:
        distance = self.r_multiple * position.risk_per_share
        if position.side is Side.BUY:
            return bar.low <= position.entry_px - distance
        return bar.high >= position.entry_px + distance


@dataclass(frozen=True)
class MaxHoldExit:
    """Exit once the position has been held ``minutes`` or longer."""

    minutes: int
    name: str = "max_hold"

    def should_exit(self, position: OpenPosition, bar: BarEvent) -> bool:
        return bar.ts - position.entry_ts >= timedelta(minutes=self.minutes)


@dataclass(frozen=True)
class EodFlattenExit:
    """Exit on the first bar whose ET time-of-day reaches ``flatten_time``."""

    flatten_time: time
    name: str = "eod_flatten"

    def should_exit(self, position: OpenPosition, bar: BarEvent) -> bool:
        return bar.ts.astimezone(MARKET_TZ).time() >= self.flatten_time


@dataclass(frozen=True)
class SignalInvalidationExit:
    """Exit when the strategy's own invalidation callback says the thesis died."""

    invalidated: Callable[[OpenPosition, BarEvent], bool]
    name: str = "signal_invalidation"

    def should_exit(self, position: OpenPosition, bar: BarEvent) -> bool:
        return bool(self.invalidated(position, bar))


def build_exit_policies(
    params: EngineParams,
    invalidation: Callable[[OpenPosition, BarEvent], bool] | None = None,
) -> tuple[ExitPolicy, ...]:
    """Assemble the exit-policy stack from config — first hit wins, in order:
    stop, max-hold, EOD flatten, signal invalidation. Each None in ``params``
    disables its policy, so exit-policy comparison is a config sweep."""
    policies: list[ExitPolicy] = []
    if params.stop_r_multiple is not None:
        policies.append(StopLossExit(params.stop_r_multiple))
    if params.max_hold_minutes is not None:
        policies.append(MaxHoldExit(params.max_hold_minutes))
    if params.eod_flatten_time is not None:
        policies.append(EodFlattenExit(params.eod_flatten_time))
    if invalidation is not None:
        policies.append(SignalInvalidationExit(invalidation))
    return tuple(policies)


# ---------------------------------------------------------------------------
# Quote source seam (equity now; an option leg supplies its own later)
# ---------------------------------------------------------------------------


class QuoteSource(Protocol):
    """Maps (symbol, bar close ts) to that bar's prevailing top-of-book."""

    def quote_for_bar(self, symbol: str, bar_ts: datetime) -> QuoteEvent | None:
        """Quote keyed to the bar closing at ``bar_ts``; None if unquoted."""
        ...


class TapeQuoteSource:
    """Quote tape loaded via the data gateway: one quote row per bar close."""

    def __init__(self, rows: Mapping[tuple[str, datetime], tuple[float, float, int, int]]) -> None:
        self._rows = dict(rows)

    def quote_for_bar(self, symbol: str, bar_ts: datetime) -> QuoteEvent | None:
        row = self._rows.get((symbol, bar_ts))
        if row is None:
            return None
        bid, ask, bid_size, ask_size = row
        return QuoteEvent(
            ts=bar_ts, symbol=symbol, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size
        )


# ---------------------------------------------------------------------------
# Signal-facing context: the ONLY window a signal gets on the world
# ---------------------------------------------------------------------------


class MarketContext:
    """What a signal may see at one bar close. Visibility is enforced HERE.

    Bars come from the engine's already-replayed history (nothing closing
    after ``now`` exists in it); point-in-time frames are filtered to rows
    with ``available_at <= now`` by the engine. Signals never touch the
    loader, a backend, or the future.
    """

    __slots__ = ("_engine", "_state", "bar", "now")

    def __init__(
        self, engine: BacktestEngine, state: _RunState, now: datetime, bar: BarEvent
    ) -> None:
        self._engine = engine
        self._state = state
        self.now = now
        self.bar = bar

    def history(self, symbol: str | None = None, n: int | None = None) -> tuple[BarEvent, ...]:
        """Bars for ``symbol`` (default: the current bar's) closed at/before now."""
        bars = self._state.seen.get(symbol or self.bar.symbol, [])
        return tuple(bars[-n:]) if n is not None else tuple(bars)

    def pit(self, kind: str, symbol: str | None = None) -> pd.DataFrame:
        """Point-in-time rows of ``kind`` with ``available_at <= now``."""
        frame = self._state.pit.get(kind)
        if frame is None:
            raise KeyError(
                f"pit kind {kind!r} was not preloaded; add it to EngineParams.pit_kinds "
                f"(loaded: {sorted(self._state.pit)})"
            )
        visible = frame[frame["available_at"] <= pd.Timestamp(self.now)]
        if symbol is not None and "symbol" in visible.columns:
            visible = visible[visible["symbol"] == symbol]
        return visible.reset_index(drop=True).copy()

    def pit_latest(self, kind: str, symbol: str | None = None) -> pd.Series | None:
        """The NEWEST row of ``kind`` visible at ``now``, or None if none is.

        Same visibility rule as :meth:`pit` (``available_at <= now``), but
        indexed: the frame is split per symbol once and bisected per call,
        so minute-resolution kinds cost O(log n) rather than a full mask and
        copy. Use this when a signal needs the current state of a feed
        rather than its whole history.
        """
        frame = self._state.pit.get(kind)
        if frame is None:
            raise KeyError(
                f"pit kind {kind!r} was not preloaded; add it to EngineParams.pit_kinds "
                f"(loaded: {sorted(self._state.pit)})"
            )
        cache_key = (kind, symbol)
        entry = self._state.pit_index.get(cache_key)
        if entry is None:
            sub = frame
            if symbol is not None and "symbol" in frame.columns:
                sub = frame[frame["symbol"] == symbol]
            sub = sub.sort_values("available_at", kind="stable").reset_index(drop=True)
            values = pd.to_datetime(sub["available_at"]).to_numpy(dtype="datetime64[ns]")
            entry = (values, sub)
            self._state.pit_index[cache_key] = entry
        values, sub = entry
        if len(sub) == 0:
            return None
        cut = int(values.searchsorted(pd.Timestamp(self.now).to_datetime64(), side="right"))
        if cut == 0:
            return None
        return sub.iloc[cut - 1]

    def options_pit(self, kind: str, symbol: str) -> pd.DataFrame:
        """Per-expiry options rows of ``kind`` with ``available_at <= now``.

        ``kind`` must be declared in ``EngineParams.options_pit`` and
        ``symbol`` follows the bridge grammar ``"UNDERLYING:YYYY-MM-DD"``
        with a declared underlying (``KeyError`` otherwise). The engine
        loads each (symbol, ET session) lazily through the gateway loader
        on the first request and serves the cached frame for the rest of
        that session — a read, never a terminal fetch under the default
        cache-only bridge. Visibility is enforced HERE, same as the feed
        kinds: only rows already published at ``now`` are returned, so
        day-D open interest (stamped next-business-day 09:00 ET by the
        bridge) is invisible for all of day D.
        """
        frame = self._engine._options_pit_frame(self._state, kind, symbol, self.now)
        visible = frame[frame["available_at"] <= pd.Timestamp(self.now)]
        return visible.reset_index(drop=True).copy()

    def frame(self, kind: str, symbol: str | None = None) -> pd.DataFrame:
        """:class:`~edge.signals.base.SignalContext`-protocol spelling.

        Routes the per-expiry options kinds (:data:`OPTIONS_PIT_KINDS`) to
        :meth:`options_pit` — they require the per-expiry symbol — and every
        other kind to :meth:`pit`. Signals written against the base protocol
        (``ctx.frame(kind, symbol)``) work against the engine unchanged.
        """
        if kind in OPTIONS_PIT_KINDS:
            if symbol is None:
                raise ValueError(
                    f"options pit kind {kind!r} requires the per-expiry symbol "
                    "('UNDERLYING:YYYY-MM-DD'); none was given"
                )
            return self.options_pit(kind, symbol)
        return self.pit(kind, symbol)

    def has_position(self, symbol: str) -> bool:
        """True if the engine currently holds a position in ``symbol``."""
        return symbol in self._state.positions

    @property
    def equity(self) -> float:
        """Mark-to-market equity at ``now`` (cash + open positions)."""
        return self._engine._equity(self._state)


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced. The engine never persists any of it."""

    ledger: pd.DataFrame
    equity: pd.Series
    drop_counters: dict[str, int]
    config_hash: str
    emitted: int
    executed: int

    @property
    def summary(self) -> dict[str, Any]:
        """The run summary; drop counters are always present, zeros included."""
        return {
            "config_hash": self.config_hash,
            "signals_emitted": self.emitted,
            "trades_executed": self.executed,
            "trades_closed": len(self.ledger),
            "drop_counters": dict(self.drop_counters),
            "final_equity": float(self.equity.iloc[-1]) if len(self.equity) else None,
        }


# ---------------------------------------------------------------------------
# Internal per-run state (rebuilt on every run(): reruns are independent)
# ---------------------------------------------------------------------------


class _OptionsPitEntry(NamedTuple):
    """One lazily loaded per-expiry options frame plus the session it serves."""

    session: date
    frame: pd.DataFrame


@dataclass
class _RunState:
    bars: dict[str, list[BarEvent]] = field(default_factory=dict)
    bar_ts: dict[str, list[datetime]] = field(default_factory=dict)
    quotes: QuoteSource | None = None
    pit: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: (kind, symbol|None) -> (available_at as int64 ns, per-symbol frame),
    #: built lazily so ``pit_latest`` is O(log n) instead of masking the
    #: whole frame on every bar. Minute-resolution kinds make the difference
    #: between a run that finishes and one that does not.
    pit_index: dict[tuple[str, str | None], tuple[Any, pd.DataFrame]] = field(
        default_factory=dict
    )
    #: (kind, per-expiry symbol) -> the frame loaded for the current session.
    options_pit: dict[tuple[str, str], _OptionsPitEntry] = field(default_factory=dict)
    seen: dict[str, list[BarEvent]] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    cash: float = 0.0
    counters: Counter[str] = field(default_factory=Counter)
    emitted: int = 0
    executed: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    equity_ts: list[datetime] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    current_day: date | None = None
    day_start_equity: float = 0.0


class _FillFailure(NamedTuple):
    """Why an attempted fill produced nothing (maps to a drop stage on entry)."""

    reason: str  # "no-next-bar" | "no-quote" | "unfilled"


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Replays the tape, runs signals, executes fills, keeps the books."""

    def __init__(
        self,
        loader: EdgeDataLoader,
        config: EdgeConfig,
        params: EngineParams,
        signals: Sequence[Signal],
        *,
        regime_gate: RegimeGate | None = None,
        sizer: Sizer | None = None,
        exit_policies: Sequence[ExitPolicy] | None = None,
        invalidation: Callable[[OpenPosition, BarEvent], bool] | None = None,
        quote_source: QuoteSource | None = None,
    ) -> None:
        unknown = [k for k in params.pit_kinds if k not in POINT_IN_TIME_KINDS]
        if unknown:
            raise ValueError(
                f"unknown pit kinds {unknown}; valid: {sorted(POINT_IN_TIME_KINDS)}"
            )
        options_pit: dict[str, frozenset[str]] = {}
        for spec in params.options_pit:
            bad = [s for s in spec.symbols if not s or ":" in s]
            if bad:
                raise ValueError(
                    f"options_pit kind {spec.kind!r} declares plain underlyings, "
                    f"got {bad}; the per-expiry part belongs to the "
                    "ctx.options_pit request symbol ('UNDERLYING:YYYY-MM-DD'), "
                    "never to the declaration"
                )
            options_pit[spec.kind] = options_pit.get(spec.kind, frozenset()) | frozenset(
                spec.symbols
            )
        self._options_pit = options_pit
        self._loader = loader
        self._config = config
        self._params = params
        self._signals = list(signals)
        self._regime_gate = regime_gate
        self._sizer = sizer or RiskBudgetSizer(
            per_trade_cap_pct=config.risk.per_trade_cap_pct,
            risk_r_pct=params.risk_r_pct,
            multiplier=params.contract_multiplier,
        )
        self._policies: tuple[ExitPolicy, ...] = (
            tuple(exit_policies)
            if exit_policies is not None
            else build_exit_policies(params, invalidation)
        )
        self._quote_source_override = quote_source
        self._model = CostModel(
            config.execution, contract_multiplier=params.contract_multiplier
        )

    # -- public API ---------------------------------------------------------

    def run(self) -> RunResult:
        """Replay the configured window and return the run's full accounting.

        Never writes to ``results/`` (or anywhere else) — the caller owns
        persistence. Rebuilding all state per call makes reruns independent
        and byte-identical for the same inputs and seed.
        """
        state = _RunState(cash=self._params.initial_equity)
        self._load_tape(state)
        merged = sorted(
            (bar for bars in state.bars.values() for bar in bars),
            key=lambda b: (b.ts, b.symbol),
        )
        if merged:
            clock = BacktestClock(start=merged[0].ts)
            for ts, group_iter in groupby(merged, key=lambda b: b.ts):
                clock.advance(ts)
                group = list(group_iter)
                self._apply_marks(state, group)
                self._roll_day(state, ts)
                for bar in group:
                    self._check_exits(state, bar)
                for bar in group:
                    ctx = MarketContext(self, state, clock.now(), bar)
                    for signal in self._signals:
                        emitted = signal.on_bar(ctx)
                        if emitted is None:
                            continue
                        events = emitted if isinstance(emitted, list) else [emitted]
                        for event in events:
                            self._process_signal(state, event, ctx)
                state.equity_ts.append(ts)
                state.equity_values.append(self._equity(state))
        self._flatten_end_of_data(state)
        total_drops = sum(state.counters.values())
        if state.emitted != state.executed + total_drops:
            raise EngineInvariantError(
                f"drop identity violated: emitted {state.emitted} != "
                f"executed {state.executed} + drops {total_drops} "
                f"({dict(state.counters)})"
            )
        drop_counters = {stage: int(state.counters.get(stage, 0)) for stage in DROP_STAGES}
        ledger = pd.DataFrame(state.rows, columns=list(LEDGER_COLUMNS))
        equity = pd.Series(
            state.equity_values,
            index=pd.DatetimeIndex(state.equity_ts),
            name="equity",
            dtype=float,
        )
        return RunResult(
            ledger=ledger,
            equity=equity,
            drop_counters=drop_counters,
            config_hash=self._config_hash(),
            emitted=state.emitted,
            executed=state.executed,
        )

    # -- loading ------------------------------------------------------------

    def _load_tape(self, state: _RunState) -> None:
        """Pull bars, quotes, and pit frames through the gateway loader."""
        p = self._params
        quote_rows: dict[tuple[str, datetime], tuple[float, float, int, int]] = {}
        for symbol in p.symbols:
            bars = _frame_to_bars(symbol, self._loader.load(symbol, p.start, p.end, "bars"))
            state.bars[symbol] = bars
            state.bar_ts[symbol] = [b.ts for b in bars]
            state.seen[symbol] = []
            if self._quote_source_override is None:
                quotes = self._loader.load(symbol, p.start, p.end, "quotes")
                quote_rows.update(_frame_to_quote_rows(symbol, quotes))
        state.quotes = self._quote_source_override or TapeQuoteSource(quote_rows)
        for kind in p.pit_kinds:
            frame = self._loader.load(ALL_SYMBOLS, p.start, p.end, kind)  # type: ignore[arg-type]
            if not frame.empty:
                available = pd.to_datetime(frame["available_at"])
                if available.dt.tz is None:
                    raise ValueError(
                        f"pit kind {kind!r} has a naive available_at column; "
                        "visibility joins need tz-aware timestamps"
                    )
                frame = frame.sort_values("available_at", kind="stable").reset_index(drop=True)
            state.pit[kind] = frame

    def _options_pit_frame(
        self, state: _RunState, kind: str, symbol: str, now: datetime
    ) -> pd.DataFrame:
        """The declared per-expiry options frame for ``symbol``, load-once
        per (symbol, ET session).

        The first request in a session loads ``[params.start, session]``
        through the gateway loader (never past the current session, so the
        engine never even asks a backend for a future day); later requests
        in the same session are served from the run-state cache, and a new
        session reloads with the extended range. Rows are validated
        (tz-aware ``available_at``) and sorted; the CALLER (the context)
        applies the ``available_at <= now`` visibility join.
        """
        allowed = self._options_pit.get(kind)
        if allowed is None:
            raise KeyError(
                f"options pit kind {kind!r} was not declared; add it to "
                f"EngineParams.options_pit (declared: {sorted(self._options_pit)})"
            )
        underlying = symbol.split(":", 1)[0]
        if underlying not in allowed:
            raise KeyError(
                f"underlying {underlying!r} (from {symbol!r}) is not declared for "
                f"options pit kind {kind!r} (declared: {sorted(allowed)})"
            )
        session = now.astimezone(MARKET_TZ).date()
        cached = state.options_pit.get((kind, symbol))
        if cached is not None and cached.session == session:
            return cached.frame
        end = min(self._params.end, session)
        frame = self._loader.load(symbol, self._params.start, end, kind)  # type: ignore[arg-type]
        if frame.empty:
            if "available_at" not in frame.columns:
                frame = pd.DataFrame(columns=[*frame.columns, "available_at"])
        else:
            available = pd.to_datetime(frame["available_at"])
            if available.dt.tz is None:
                raise ValueError(
                    f"options pit kind {kind!r} ({symbol}) has a naive "
                    "available_at column; visibility joins need tz-aware "
                    "timestamps"
                )
            frame = (
                frame.assign(available_at=available)
                .sort_values("available_at", kind="stable")
                .reset_index(drop=True)
            )
        state.options_pit[(kind, symbol)] = _OptionsPitEntry(session=session, frame=frame)
        return frame

    # -- replay phases ------------------------------------------------------

    def _apply_marks(self, state: _RunState, group: list[BarEvent]) -> None:
        """Phase 1 at ts: record bars as seen, update marks and MFE/MAE paths."""
        for bar in group:
            state.seen[bar.symbol].append(bar)
            state.last_close[bar.symbol] = bar.close
            position = state.positions.get(bar.symbol)
            if position is not None:
                position.observe(bar)

    def _roll_day(self, state: _RunState, ts: datetime) -> None:
        """Reset the daily-halt baseline on the first tape event of an ET day."""
        et_day = ts.astimezone(MARKET_TZ).date()
        if et_day != state.current_day:
            state.current_day = et_day
            state.day_start_equity = self._equity(state)

    def _check_exits(self, state: _RunState, bar: BarEvent) -> None:
        """Phase 2 at ts: first exit policy to fire wins; fill at next bar."""
        position = state.positions.get(bar.symbol)
        if position is None:
            return
        for policy in self._policies:
            if policy.should_exit(position, bar):
                self._execute_exit(state, position, bar.ts, policy.name)
                break

    # -- the signal pipeline ------------------------------------------------

    def _process_signal(self, state: _RunState, event: SignalEvent, ctx: MarketContext) -> None:
        """Run one emitted signal through the gated pipeline to a fill or a drop."""
        now = ctx.now
        if event.ts != now:
            raise EngineInvariantError(
                f"signal {event.signal_name!r} stamped {event.ts}, but the clock "
                f"reads {now}: signals must stamp ctx.now"
            )
        state.emitted += 1
        if self._regime_gate is not None and not self._regime_gate(event, ctx):
            state.counters[DROP_REGIME] += 1
            return
        equity_now = self._equity(state)
        halt_pct = self._config.risk.daily_loss_halt_pct
        if state.day_start_equity - equity_now >= state.day_start_equity * halt_pct / 100.0:
            state.counters[DROP_DAILY_HALT] += 1
            return
        if event.symbol in state.positions:
            state.counters[DROP_POSITION_OPEN] += 1
            return
        ref_price = state.last_close.get(event.symbol)
        if ref_price is None:
            state.counters[DROP_NO_QUOTE] += 1  # never seen a bar: unpriceable
            return
        decision = self._sizer.size(event, ref_price, equity_now)
        if decision.qty <= 0:
            state.counters[DROP_SIZING] += 1
            return
        mult = self._params.contract_multiplier
        cap_qty = int(equity_now // (ref_price * mult)) if ref_price > 0 else 0
        qty = min(decision.qty, cap_qty)
        if qty <= 0:
            state.counters[DROP_RISK_CAP] += 1
            return
        attempt = self._attempt_fill(state, event.symbol, event.side, qty, now)
        if isinstance(attempt, _FillFailure):
            if attempt.reason == "no-quote":
                state.counters[DROP_NO_QUOTE] += 1
            else:  # "no-next-bar" (no next bar, no fill) or "unfilled"
                state.counters[DROP_LATENCY] += 1
            return
        fill, arrival_ts = attempt
        assert fill.price is not None  # FILLED contract of the cost model
        signed = 1 if event.side is Side.BUY else -1
        state.cash -= signed * fill.price * fill.qty * mult
        state.cash -= fill.costs.commission
        state.positions[event.symbol] = OpenPosition(
            symbol=event.symbol,
            side=event.side,
            qty=fill.qty,
            signal_name=event.signal_name,
            conviction=event.conviction,
            risk_per_share=decision.risk_per_share,
            entry_decision_ts=now,
            entry_ts=arrival_ts,
            entry_px=fill.price,
            entry_costs=fill.costs,
        )
        state.executed += 1

    # -- execution ----------------------------------------------------------

    def _attempt_fill(
        self,
        state: _RunState,
        symbol: str,
        side: Side,
        qty: int,
        decision_ts: datetime,
    ) -> tuple[FillResult, datetime] | _FillFailure:
        """Price a marketable order arriving ``latency_ms`` after ``decision_ts``.

        The arrival quote is the NEXT bar's quote re-stamped at the true
        arrival instant — the order lands inside the next bar's interval, so
        that bar's book is what it meets. This is the single fill seam: an
        option leg plugs in via a different quote source, nothing else moves.
        """
        assert state.quotes is not None
        next_bar = self._next_bar(state, symbol, decision_ts)
        if next_bar is None:
            return _FillFailure("no-next-bar")
        decision_quote = state.quotes.quote_for_bar(symbol, decision_ts)
        next_quote = state.quotes.quote_for_bar(symbol, next_bar.ts)
        if decision_quote is None or next_quote is None:
            return _FillFailure("no-quote")
        arrival_ts = decision_ts + timedelta(milliseconds=self._params.latency_ms)
        arrival_quote = QuoteEvent(
            ts=arrival_ts,
            symbol=next_quote.symbol,
            bid=next_quote.bid,
            ask=next_quote.ask,
            bid_size=next_quote.bid_size,
            ask_size=next_quote.ask_size,
        )
        fill = self._model.fill_with_latency(
            decision_quote, arrival_quote, self._params.latency_ms, side, qty
        )
        if fill.status is not FillStatus.FILLED or fill.price is None:
            return _FillFailure("unfilled")
        return fill, arrival_ts

    def _execute_exit(
        self, state: _RunState, position: OpenPosition, decision_ts: datetime, reason: str
    ) -> None:
        """Close ``position`` through the same next-bar fill path as entries.

        A failed exit fill (no next bar / no quote / unfilled) leaves the
        position open: the policy fires again on the next bar, and the
        terminal flatten catches whatever the tape's end strands.
        """
        exit_side = Side.SELL if position.side is Side.BUY else Side.BUY
        attempt = self._attempt_fill(
            state, position.symbol, exit_side, position.qty, decision_ts
        )
        if isinstance(attempt, _FillFailure):
            return
        fill, arrival_ts = attempt
        assert fill.price is not None
        mult = self._params.contract_multiplier
        signed = 1 if exit_side is Side.BUY else -1
        state.cash -= signed * fill.price * fill.qty * mult
        state.cash -= fill.costs.commission
        del state.positions[position.symbol]
        self._book_trade(state, position, arrival_ts, fill.price, reason, fill.costs)

    def _flatten_end_of_data(self, state: _RunState) -> None:
        """Terminal flatten: mark out stranded positions at their last close.

        Not a modeled fill (no quote survives the tape's end): price is the
        symbol's final close, costs are zero, reason is ``end_of_data``.
        """
        for symbol in list(state.positions):
            position = state.positions.pop(symbol)
            last_ts = state.bar_ts[symbol][-1]
            exit_px = state.last_close[symbol]
            signed = -position.direction
            state.cash -= signed * exit_px * position.qty * self._params.contract_multiplier
            self._book_trade(
                state,
                position,
                last_ts,
                exit_px,
                EXIT_END_OF_DATA,
                FillCosts(spread_cost=0.0, slippage_cost=0.0, commission=0.0),
            )

    # -- bookkeeping --------------------------------------------------------

    def _book_trade(
        self,
        state: _RunState,
        position: OpenPosition,
        exit_ts: datetime,
        exit_px: float,
        exit_reason: str,
        exit_costs: FillCosts,
    ) -> None:
        """Write one closed trade into the ledger with full attribution."""
        mult = self._params.contract_multiplier
        scale = position.qty * mult
        commission = position.entry_costs.commission + exit_costs.commission
        pnl = position.direction * (exit_px - position.entry_px) * scale - commission
        risk_dollars = position.risk_per_share * scale
        r_multiple = pnl / risk_dollars if risk_dollars > 0 else float("nan")
        if position.side is Side.BUY:
            mfe = position.high_water - position.entry_px
            mae = position.entry_px - position.low_water
        else:
            mfe = position.entry_px - position.low_water
            mae = position.high_water - position.entry_px
        state.rows.append(
            {
                "symbol": position.symbol,
                "side": position.side.value,
                "qty": position.qty,
                "signal_name": position.signal_name,
                "conviction": position.conviction,
                "entry_ts": position.entry_ts,
                "entry_px": position.entry_px,
                "exit_ts": exit_ts,
                "exit_px": exit_px,
                "exit_reason": exit_reason,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "spread_cost": position.entry_costs.spread_cost + exit_costs.spread_cost,
                "slippage_cost": position.entry_costs.slippage_cost + exit_costs.slippage_cost,
                "commission": commission,
                "mfe": max(0.0, mfe),
                "mae": max(0.0, mae),
                "holding_minutes": (exit_ts - position.entry_ts).total_seconds() / 60.0,
            }
        )

    def _equity(self, state: _RunState) -> float:
        """Mark-to-market equity: cash plus signed open-position value."""
        mult = self._params.contract_multiplier
        value = sum(
            pos.direction * pos.qty * state.last_close[symbol] * mult
            for symbol, pos in state.positions.items()
        )
        return state.cash + value

    def _next_bar(self, state: _RunState, symbol: str, ts: datetime) -> BarEvent | None:
        """The symbol's first bar closing strictly after ``ts``, if any."""
        ts_list = state.bar_ts.get(symbol, [])
        i = bisect_right(ts_list, ts)
        return state.bars[symbol][i] if i < len(ts_list) else None

    def _config_hash(self) -> str:
        """sha256 over everything that shapes this run's behavior."""
        return canonical_config_hash(
            {
                "engine": self._params.model_dump(mode="json"),
                "execution": self._config.execution.model_dump(mode="json"),
                "risk": self._config.risk.model_dump(mode="json"),
                "exit_policies": [policy.name for policy in self._policies],
            }
        )


# ---------------------------------------------------------------------------
# Frame -> event conversion (tape hygiene lives here)
# ---------------------------------------------------------------------------


def _frame_ts(frame: pd.DataFrame) -> pd.Series:
    """The frame's timestamp column (``ts`` column or DatetimeIndex), tz-checked."""
    if "ts" in frame.columns:
        ts = pd.to_datetime(frame["ts"])
    else:
        ts = pd.to_datetime(frame.index.to_series().reset_index(drop=True))
    if ts.dt.tz is None:
        raise ValueError("tape frames need tz-aware timestamps (America/New_York)")
    return ts


def _frame_to_bars(symbol: str, frame: pd.DataFrame) -> list[BarEvent]:
    """Validate and convert one symbol's bars frame into BarEvents, ts-sorted."""
    if frame is None or frame.empty:
        return []
    ts = _frame_ts(frame)
    bars = [
        BarEvent(
            ts=stamp.to_pydatetime(),
            symbol=symbol,
            open=float(o),
            high=float(h),
            low=float(lo),
            close=float(c),
            volume=int(v),
        )
        for stamp, o, h, lo, c, v in zip(
            ts, frame["open"], frame["high"], frame["low"], frame["close"], frame["volume"]
        )
    ]
    bars.sort(key=lambda b: b.ts)
    for prev, cur in pairwise(bars):
        if prev.ts == cur.ts:
            raise ValueError(f"duplicate bar for {symbol} at {cur.ts}: tape is corrupt")
    return bars


def _frame_to_quote_rows(
    symbol: str, frame: pd.DataFrame
) -> dict[tuple[str, datetime], tuple[float, float, int, int]]:
    """One (symbol, bar ts) -> (bid, ask, sizes) row per quote tape entry."""
    if frame is None or frame.empty:
        return {}
    ts = _frame_ts(frame)
    bid_size = frame["bid_size"] if "bid_size" in frame.columns else [0] * len(frame)
    ask_size = frame["ask_size"] if "ask_size" in frame.columns else [0] * len(frame)
    return {
        (symbol, stamp.to_pydatetime()): (float(b), float(a), int(bs), int(asz))
        for stamp, b, a, bs, asz in zip(ts, frame["bid"], frame["ask"], bid_size, ask_size)
    }


def run_backtest(
    loader: EdgeDataLoader,
    config: EdgeConfig,
    params: EngineParams,
    signals: Sequence[Signal],
    **hooks: Any,
) -> RunResult:
    """Convenience one-shot: construct a :class:`BacktestEngine` and run it."""
    return BacktestEngine(loader, config, params, signals, **hooks).run()

"""Opening-range breakout on abnormal volume (ORB).

MECHANISM (the falsifiable claim, in full in :data:`ORB_HYPOTHESIS`):
overnight information gets repriced in the first 30 minutes of the session;
the opening range brackets that auction. A close beyond the range on
abnormal participation — cumulative session volume above 1.5x the same-time
median of the prior 21 sessions — marks initiative flow from participants
forced to reprice (index-arb desks hedging basis, gap-chasers covering),
and their continued execution pushes price further intraday. The other
side is passive resting liquidity anchored to yesterday's value. The edge
lives in trending regimes only; in chop the break is noise and ORB dies.

POINT-IN-TIME DISCIPLINE
------------------------
Everything is computed from bars the ctx serves, re-filtered locally to
``ts <= now`` (defense-in-depth: the engine's ctx cannot serve the future,
but the invariant is enforced HERE too so it is locally testable):

* the opening range uses only the current session's bars closing in
  ``(09:30, 09:30 + opening_range_minutes]`` that are visible at decision
  time — never a later bar;
* the 21-session same-time volume median uses PRIOR sessions only: for each
  of the most recent 21 prior sessions, the cumulative volume of bars whose
  ET time-of-day is at or before the decision bar's time-of-day. Today's
  tape never enters its own baseline.

Every threshold below is a FIXED economic choice written into the
hypothesis (never swept): 30-minute range, entries only 30-90 minutes after
the open, rel_vol > 1.5, 21-session median, ~120-minute horizon, conviction
= min(1, rel_vol / (2 x threshold)) — full conviction once participation is
double the trigger level. A variant is a new trial, not a tune.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime, time, timedelta
from statistics import median
from typing import Final, NamedTuple

import pandas as pd
from pydantic import Field, model_validator

from edge.core.events import MARKET_TZ, BarEvent, Side, SignalEvent
from edge.signals.base import Signal, SignalConfig, SignalContext
from edge.signals.registry import register

__all__ = ["ORB_HYPOTHESIS", "OpeningRangeBreakout", "OpeningRangeBreakoutConfig"]

#: Regular-session open (ET): the anchor for the opening range and for the
#: same-time cumulative-volume comparison.
SESSION_OPEN: Final[time] = time(9, 30)

#: Upper bound on bars per session used to size the history request:
#: 390 regular-session minute bars plus slack for odd tapes.
BARS_PER_SESSION_CEILING: Final[int] = 400

#: The regime buckets this signal may trade: trending only (both vol
#: halves). Chop is where ORB dies — the hypothesis says so explicitly.
TRENDING_BUCKETS: Final[frozenset[str]] = frozenset(
    {"high_vol_trending", "low_vol_trending"}
)

ORB_HYPOTHESIS: Final[str] = (
    "Overnight information gets repriced in the first 30 minutes of the "
    "session; the opening range brackets that auction. A close beyond the "
    "range on abnormal participation — cumulative session volume above 1.5x "
    "the same-time median of the prior 21 sessions — marks initiative flow "
    "from participants FORCED to reprice: index-arbitrage desks hedging "
    "basis and gap-chasers covering, whose continued execution pushes price "
    "further in the break direction over roughly the next 120 minutes. The "
    "other side is passive resting liquidity anchored to yesterday's value; "
    "those orders keep paying because their reference price is stale, not "
    "because they hold new information. The edge exists only while the tape "
    "trends — in chop the break is noise, initiative flow exhausts at the "
    "range edge, and ORB dies — so the signal trades the trending regime "
    "buckets only. Entries are taken 30-90 minutes after the open (earlier "
    "the range is unformed, later the repricing is done), long above the "
    "range high / short below the range low, with conviction scaled by "
    "rel_vol (min(1, rel_vol / 3.0): full size at double the 1.5x trigger). "
    "Every threshold here is a fixed economic choice, not a fitted "
    "parameter."
)


class OpeningRangeBreakoutConfig(SignalConfig):
    """ORB parameters. Defaults ARE the hypothesis — variants are new trials."""

    #: Minutes after the open that define the opening range.
    opening_range_minutes: int = Field(default=30, gt=0)
    #: Entry window, minutes after the open (inclusive bounds).
    entry_start_minutes: int = Field(default=30, gt=0)
    entry_end_minutes: int = Field(default=90, gt=0)
    #: rel_vol must strictly exceed this to count as abnormal participation.
    rel_vol_threshold: float = Field(default=1.5, gt=0.0)
    #: Prior sessions in the same-time cumulative-volume median.
    volume_lookback_sessions: int = Field(default=21, gt=0)
    #: Expected life of the repricing flow, minutes.
    horizon_minutes: int = Field(default=120, gt=0)

    @model_validator(mode="after")
    def _windows_cohere(self) -> "OpeningRangeBreakoutConfig":
        if self.entry_start_minutes < self.opening_range_minutes:
            raise ValueError(
                "entry_start_minutes must be >= opening_range_minutes: the "
                "range must be complete before a break of it can mean anything"
            )
        if self.entry_end_minutes <= self.entry_start_minutes:
            raise ValueError("entry_end_minutes must be > entry_start_minutes")
        return self


class _Bar(NamedTuple):
    """Normalized visible bar: ET close time + the fields ORB reads."""

    ts: datetime
    high: float
    low: float
    close: float
    volume: float


def _history_bars(raw: object) -> list[_Bar]:
    """Normalize a ctx history payload into ts-sorted :class:`_Bar` rows.

    Accepts both context flavors: the engine's ``tuple[BarEvent, ...]`` and
    the :class:`~edge.signals.base.SignalContext` protocol's ``DataFrame``
    (columns ``high``/``low``/``close``/``volume`` plus a ``ts`` column or a
    ``DatetimeIndex``; naive timestamps are a construction error).
    """
    if raw is None:
        return []
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return []
        if "ts" in raw.columns:
            ts = pd.to_datetime(raw["ts"]).reset_index(drop=True)
        else:
            ts = pd.to_datetime(raw.index.to_series().reset_index(drop=True))
        if ts.dt.tz is None:
            raise ValueError(
                "history frames must carry tz-aware timestamps (America/New_York)"
            )
        ts = ts.dt.tz_convert(MARKET_TZ)
        bars = [
            _Bar(
                ts=stamp.to_pydatetime(),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=float(v),
            )
            for stamp, h, lo, c, v in zip(
                ts, raw["high"], raw["low"], raw["close"], raw["volume"]
            )
        ]
    else:
        bars = [
            _Bar(
                ts=b.ts.astimezone(MARKET_TZ),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in raw  # engine ctx: an iterable of BarEvent
        ]
    bars.sort(key=lambda b: b.ts)
    return bars


def _same_time_volume(day_bars: list[_Bar], cutoff_tod: time) -> tuple[float, int]:
    """(cumulative regular-session volume through ``cutoff_tod``, bar count).

    Counts bars whose ET close time-of-day is in ``(09:30, cutoff_tod]`` —
    the same-time slice compared across sessions.
    """
    rows = [b for b in day_bars if SESSION_OPEN < b.ts.time() <= cutoff_tod]
    return sum(b.volume for b in rows), len(rows)


class _DayState:
    """One session's incremental aggregates for one symbol.

    A pure performance memo mirroring what the full recomputation would see
    for this session: same-time rows (ET time-of-day strictly after the
    09:30 open) with their chronological cumulative-volume prefix and global
    ingestion indices (for the history-window truncation arithmetic), the
    opening-range extremes, and the last two closes (for the crossing test).
    """

    __slots__ = (
        "count", "gidx", "last_close", "or_high", "or_low",
        "prefix", "prev_close", "start_idx", "tods", "vols",
    )

    def __init__(self, start_idx: int) -> None:
        self.start_idx = start_idx     # global index of the day's first bar
        self.count = 0                 # ALL bars ingested this day
        self.tods: list[time] = []     # tods of bars with tod > 09:30, in order
        self.gidx: list[int] = []      # global index per same-time row
        self.vols: list[float] = []    # volume per same-time row
        self.prefix: list[float] = []  # chronological cumulative volumes
        self.or_high: float | None = None
        self.or_low: float | None = None
        self.last_close: float | None = None
        self.prev_close: float | None = None


class _SymbolState:
    """Per-symbol incremental tape state (see :class:`_DayState`)."""

    __slots__ = ("day_states", "days", "last_ts", "total")

    def __init__(self) -> None:
        self.total = 0                 # global ingested-bar counter
        self.last_ts: datetime | None = None
        self.days: list[date] = []     # ascending ingestion order
        self.day_states: dict[date, _DayState] = {}


@register
class OpeningRangeBreakout(Signal):
    """Opening-range breakout on abnormal same-time relative volume."""

    name = "opening_range_breakout"
    hypothesis = ORB_HYPOTHESIS
    allowed_regimes = TRENDING_BUCKETS
    #: Bar-count warmup is tape-granularity dependent, so it is 0 here; the
    #: REAL readiness gate is internal and economic: no signal until the
    #: full ``volume_lookback_sessions`` prior sessions exist in history.
    warmup_bars = 0
    config_type = OpeningRangeBreakoutConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: Per-symbol incremental tape memo — a pure performance cache.
        #: Any bar the memo cannot PROVE it has already mirrored (cold
        #: start, replayed/backward timestamp) triggers a full rebuild from
        #: the same capped history request the original per-bar
        #: recomputation used, so outputs are identical by construction.
        self._tape: dict[str, _SymbolState] = {}

    # -- incremental ingestion (pure bookkeeping; no signal logic) ----------

    def _window_start(self, state: _SymbolState) -> int:
        """Global index of the first bar inside the capped history window.

        The original recomputation reads ``ctx.history(symbol, n_request)``
        — the last ``(lookback + 1) * 400`` bars — so any bar older than
        that is invisible to it; the incremental evaluation reproduces the
        same horizon through this bound.
        """
        cfg: OpeningRangeBreakoutConfig = self.config  # type: ignore[assignment]
        n_request = (cfg.volume_lookback_sessions + 1) * BARS_PER_SESSION_CEILING
        return state.total - n_request

    def _fold(self, state: _SymbolState, b: _Bar) -> None:
        """Fold one visible bar (ts-ordered) into the symbol's day memos."""
        cfg: OpeningRangeBreakoutConfig = self.config  # type: ignore[assignment]
        day = b.ts.date()
        ds = state.day_states.get(day)
        if ds is None:
            ds = _DayState(start_idx=state.total)
            state.day_states[day] = ds
            state.days.append(day)
            # Prune days that have fallen entirely below the history window
            # (they can never become visible again: the window only moves
            # forward, and a rebuild resets everything).
            wstart = self._window_start(state)
            while len(state.days) > 1:
                oldest = state.day_states[state.days[0]]
                if oldest.start_idx + oldest.count <= wstart:
                    del state.day_states[state.days.pop(0)]
                else:
                    break
        gi = state.total
        state.total += 1
        state.last_ts = b.ts
        ds.count += 1
        ds.prev_close, ds.last_close = ds.last_close, b.close
        tod = b.ts.time()
        if SESSION_OPEN < tod:
            session_open = b.ts.replace(
                hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute,
                second=0, microsecond=0,
            )
            if b.ts <= session_open + timedelta(minutes=cfg.opening_range_minutes):
                ds.or_high = b.high if ds.or_high is None else max(ds.or_high, b.high)
                ds.or_low = b.low if ds.or_low is None else min(ds.or_low, b.low)
            ds.tods.append(tod)
            ds.gidx.append(gi)
            ds.vols.append(b.volume)
            ds.prefix.append((ds.prefix[-1] if ds.prefix else 0.0) + b.volume)

    def _rebuild(self, ctx: SignalContext, symbol: str, now_et: datetime) -> _SymbolState:
        """Cold rebuild from the SAME capped history request the original
        per-bar recomputation used (visibility re-filter included)."""
        cfg: OpeningRangeBreakoutConfig = self.config  # type: ignore[assignment]
        n_request = (cfg.volume_lookback_sessions + 1) * BARS_PER_SESSION_CEILING
        state = _SymbolState()
        self._tape[symbol] = state
        for b in _history_bars(ctx.history(symbol, n_request)):
            if b.ts <= now_et:
                self._fold(state, b)
        return state

    def _day_same_time(
        self, ds: _DayState, cutoff_tod: time, wstart: int
    ) -> tuple[float, int]:
        """(cumulative volume, row count) of a day's same-time slice within
        the history window — :func:`_same_time_volume` over exactly the rows
        the original's capped request would have served, with identical
        chronological float accumulation."""
        k = bisect_right(ds.tods, cutoff_tod)
        if ds.start_idx >= wstart:
            if k == 0:
                return 0.0, 0
            return ds.prefix[k - 1], k
        lo = bisect_left(ds.gidx, wstart)
        if k <= lo:
            return 0.0, 0
        total = 0.0
        for v in ds.vols[lo:k]:
            total += v
        return total, k - lo

    def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
        cfg: OpeningRangeBreakoutConfig = self.config  # type: ignore[assignment]

        # The decision instant: engine ctx exposes .now, the protocol
        # exposes .decision_ts — both are the just-closed bar's close.
        now: datetime | None = getattr(ctx, "now", None)
        if now is None:
            now = ctx.decision_ts
        now_et = now.astimezone(MARKET_TZ)
        bar: BarEvent = ctx.bar
        symbol = bar.symbol

        # 0. Keep the incremental memo mirrored. The engine feeds every
        #    completed bar of a symbol through on_bar exactly once, in
        #    order, so folding ctx.bar keeps the memo equal to what the
        #    original capped-history recomputation would see; anything
        #    unprovable (cold start, replay) rebuilds from that request.
        state = self._tape.get(symbol)
        bar_et = bar.ts.astimezone(MARKET_TZ)
        if (
            state is None
            or state.last_ts is None
            or bar_et <= state.last_ts
            or bar.ts != now
        ):
            state = self._rebuild(ctx, symbol, now_et)
        else:
            self._fold(
                state,
                _Bar(
                    ts=bar_et,
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                ),
            )

        # 1. Entry window: only 30-90 minutes after the open. Earlier the
        #    range is unformed; later the overnight repricing is done.
        session_open = now_et.replace(
            hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0, microsecond=0
        )
        minutes_since_open = (now_et - session_open).total_seconds() / 60.0
        if not (cfg.entry_start_minutes <= minutes_since_open <= cfg.entry_end_minutes):
            return []

        # 2. Visible bars only (the memo mirrors the capped request).
        if state.total == 0:
            return []

        # 3. Opening range from THIS session's visible bars closing within
        #    (open, open + opening_range_minutes].
        session_day = now_et.date()
        today = state.day_states.get(session_day)
        if today is None or today.or_high is None or today.or_low is None:
            return []
        or_high = today.or_high
        or_low = today.or_low

        # 4. Abnormal participation: today's cumulative regular-session
        #    volume vs the same-time median of the last 21 PRIOR sessions.
        #    Today's tape never enters its own baseline; fewer prior
        #    sessions than the lookback means no signal, never a shortcut.
        cutoff_tod = now_et.time()
        wstart = self._window_start(state)
        baseline: list[float] = []
        for day in reversed(state.days):
            if day >= session_day:
                continue
            ds = state.day_states[day]
            if ds.start_idx + ds.count <= wstart:
                break  # entirely below the window; older days are too
            cum, n_rows = self._day_same_time(ds, cutoff_tod, wstart)
            if n_rows == 0:
                continue  # session carries no same-time information
            baseline.append(cum)
            if len(baseline) == cfg.volume_lookback_sessions:
                break
        if len(baseline) < cfg.volume_lookback_sessions:
            return []
        baseline_median = median(baseline)
        if baseline_median <= 0.0:
            return []
        today_volume, _ = self._day_same_time(today, cutoff_tod, wstart)
        rel_vol = today_volume / baseline_median
        if rel_vol <= cfg.rel_vol_threshold:
            return []

        # 5. The just-closed bar must CROSS the range edge (previous close
        #    inside, this close beyond) — initiative flow at the moment of
        #    the break, not every bar that happens to sit outside the range.
        #    The previous close is the last TODAY bar strictly before this
        #    one: when the memo's newest bar IS this bar, that is the
        #    second-to-last ingested close; when a (protocol) ctx served a
        #    history without the current bar, it is the newest ingested one.
        prev_close = (
            today.prev_close if state.last_ts == bar_et else today.last_close
        )
        if prev_close is None:
            return []
        close = float(bar.close)
        if close > or_high and prev_close <= or_high:
            side = Side.BUY
        elif close < or_low and prev_close >= or_low:
            side = Side.SELL
        else:
            return []

        # 6. Conviction scaled by rel_vol: 0.5 at the 1.5x trigger, full
        #    size once participation doubles it. Fixed mapping, never fitted.
        conviction = min(1.0, rel_vol / (2.0 * cfg.rel_vol_threshold))
        return [
            SignalEvent(
                ts=now,
                symbol=symbol,
                side=side,
                conviction=conviction,
                horizon_minutes=cfg.horizon_minutes,
                signal_name=self.name,
            )
        ]

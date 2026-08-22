"""VwapReversion: fade a z > 2 stretch of price vs session VWAP in chop.

MECHANISM (the falsifiable claim, in full in :attr:`VwapReversion.hypothesis`)
-----------------------------------------------------------------------------
Institutional execution engines are benchmarked to session VWAP and slice
their parents to track it. When price stretches far from the session VWAP
*without new information*, the stretch is a temporary liquidity-demand
imbalance — an impatient taker paid up (or down) through the book — and the
VWAP-pegged flow on the other side absorbs it back toward the benchmark.
This signal FADES that stretch: SELL a stretch above VWAP, BUY a stretch
below, over a ~60-minute horizon, with conviction scaled by |z|.

The LOAD-BEARING clause is the regime gate: in a trending session the same
stretch is information (repricing), not imbalance, so the signal declares
``allowed_regimes`` = the two chopping promotion buckets
(:data:`CHOP_BUCKETS`) and lets the engine's regime gate suppress it
everywhere else. Removing the gate is not a variant of this signal — it is
a different (and rejected) hypothesis.

MEASUREMENT (fixed economic choices — never swept; a variant is a new trial)
----------------------------------------------------------------------------
* Session VWAP is computed strictly from the session's visible bars:
  cumulative volume-weighted typical price ``(high + low + close) / 3``
  over bars of the current ET session with ``ts <= decision_ts``.
* The stretch statistic is the z-score of the CURRENT bar's price-VWAP
  spread (``close - session_vwap``) within the session's OWN spread
  distribution so far (every session bar up to and including the current
  one; mean/std with ddof=0). Prefix-invariant by construction: truncating
  the future never changes a past bar's z, and no bar of a prior session
  ever enters the distribution.
* Entry at ``|z| > 2`` (``z_entry``), only once the session has produced at
  least 60 spread observations (``min_session_bars``) so the distribution
  is the session's own, not three bars of noise.
* Conviction ``= min(1, |z| / 4)`` (``conviction_full_z``): a just-over-
  threshold stretch trades at half conviction, a 4-sigma one at full.
* Horizon 60 minutes (``horizon_minutes``): the absorption timescale, not
  a tuned exit.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd
from pydantic import Field

from edge.core.events import MARKET_TZ, BarEvent, Side, SignalEvent
from edge.signals.base import Signal, SignalConfig, SignalContext
from edge.signals.registry import register

__all__ = [
    "CHOP_BUCKETS",
    "VwapReversion",
    "VwapReversionConfig",
    "current_z",
    "spread_series",
]

#: The two non-trending promotion buckets (must stay a subset of
#: ``edge.regime.classifier.BUCKETS``). Trending buckets are deliberately
#: absent: in a trend the stretch is information, not imbalance.
CHOP_BUCKETS: Final[frozenset[str]] = frozenset(
    {"low_vol_chopping", "high_vol_chopping"}
)

#: Guard against a degenerate (all-equal) spread distribution: no z exists.
_MIN_STD: Final[float] = 1e-12

#: Minute-bar cadence: the incremental session memo continues only across an
#: exactly-one-minute step; any other step rebuilds from history (exactness
#: over speed on gapped or unfamiliar tapes).
_ONE_MINUTE: Final[timedelta] = timedelta(minutes=1)


class VwapReversionConfig(SignalConfig):
    """Fixed economic choices of the VWAP-reversion hypothesis.

    These are declarations, not tunables: each default is written into the
    hypothesis text. Running a different value is a DIFFERENT experiment —
    record it as its own trial; never sweep to find these.
    """

    #: Entry threshold on |z| of the current price-VWAP spread.
    z_entry: float = Field(default=2.0, gt=0.0)
    #: Minimum session spread observations before any z is trusted.
    min_session_bars: int = Field(default=60, ge=2)
    #: The absorption horizon handed to risk/exits.
    horizon_minutes: int = Field(default=60, gt=0)
    #: |z| at which conviction saturates at 1.0.
    conviction_full_z: float = Field(default=4.0, gt=0.0)
    #: Bars requested from the ctx; covers a full 390-minute RTH session.
    max_history_bars: int = Field(default=480, ge=1)


def spread_series(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Per-bar ``close - session_vwap`` for ONE session's bars, in order.

    ``session_vwap`` at bar ``i`` is the cumulative volume-weighted typical
    price ``(high + low + close) / 3`` over bars ``0..i`` — strictly the
    prefix, so element ``i`` of the result depends only on bars ``0..i``
    (prefix-invariant by construction). Bars before any volume has printed
    (cumulative volume 0) have no VWAP and are omitted from the series.
    """
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if closes.size == 0:
        return np.empty(0, dtype=float)
    typical = (highs + lows + closes) / 3.0
    cum_pv = np.cumsum(typical * volumes)
    cum_v = np.cumsum(volumes)
    valid = cum_v > 0.0
    vwap = np.divide(cum_pv, cum_v, out=np.full_like(cum_pv, np.nan), where=valid)
    return (closes - vwap)[valid]


def current_z(spreads: np.ndarray) -> float | None:
    """z of the LAST spread within the whole series (its session so far).

    The distribution is the session's own spreads up to and including the
    current one (mean/std, ddof=0). ``None`` when no z exists: fewer than
    two observations, or a degenerate (near-zero / non-finite) std.
    """
    spreads = np.asarray(spreads, dtype=float)
    if spreads.size < 2 or not np.all(np.isfinite(spreads)):
        return None
    std = float(spreads.std())
    if not np.isfinite(std) or std <= _MIN_STD:
        return None
    return float((spreads[-1] - float(spreads.mean())) / std)


def _decision_ts(ctx: SignalContext) -> datetime:
    """The decision instant: ``ctx.decision_ts`` (base protocol) or engine ``ctx.now``."""
    ts = getattr(ctx, "decision_ts", None)
    if isinstance(ts, datetime):
        return ts
    now = getattr(ctx, "now", None)
    if isinstance(now, datetime):
        return now
    return ctx.bar.ts


def _history_rows(
    raw: object,
) -> list[tuple[datetime, float, float, float, float]]:
    """Normalize a ctx.history() result to ``(ts, close, high, low, volume)`` rows.

    Accepts the base-protocol DataFrame (``ts`` column or DatetimeIndex;
    ``high``/``low``/``volume`` optional) and the engine's tuple of
    :class:`BarEvent`. Naive timestamps are a construction error, never a
    guessed timezone.
    """
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return []
        frame = raw.reset_index() if isinstance(raw.index, pd.DatetimeIndex) else raw
        frame = frame.reset_index(drop=True)
        ts_col = next((c for c in ("ts", "index") if c in frame.columns), None)
        if ts_col is None:
            raise ValueError(
                "bars history frame needs a 'ts' column or a DatetimeIndex"
            )
        stamps = pd.to_datetime(frame[ts_col])
        if stamps.dt.tz is None:
            raise ValueError(
                "bars history needs tz-aware timestamps (America/New_York); "
                "a naive stamp cannot be placed on an ET session"
            )
        stamps = stamps.dt.tz_convert(MARKET_TZ)
        close = frame["close"].astype(float)
        high = frame["high"].astype(float) if "high" in frame.columns else close
        low = frame["low"].astype(float) if "low" in frame.columns else close
        volume = (
            frame["volume"].astype(float)
            if "volume" in frame.columns
            else pd.Series(1.0, index=frame.index)
        )
        return [
            (t.to_pydatetime(), float(c), float(h), float(lo), float(v))
            for t, c, h, lo, v in zip(stamps, close, high, low, volume)
        ]
    rows: list[tuple[datetime, float, float, float, float]] = []
    for bar in raw:  # type: ignore[union-attr]
        rows.append(
            (bar.ts, float(bar.close), float(bar.high), float(bar.low), float(bar.volume))
        )
    return rows


class _SessionState:
    """Per-symbol incremental state for ONE session's spread series.

    A pure performance memo: it holds exactly the values the full
    recomputation over the session's rows would produce (``rows`` mirrors the
    row list, ``cum_pv``/``cum_v`` the cumulative typical-price*volume and
    volume sums, ``spreads`` the output of :func:`spread_series` over those
    rows). Any call that cannot PROVE the memo mirrors the full recomputation
    (new session, replayed/backward timestamp, a gapped tape where history
    may hold bars this state never saw, or a session at the
    ``max_history_bars`` truncation cap) falls back to the full original
    computation and rebuilds the memo from it — so outputs are identical to
    recomputing from scratch on every bar, by construction.
    """

    __slots__ = ("cum_pv", "cum_v", "day", "last_ts", "n_rows", "spreads")

    def __init__(self, day: object) -> None:
        self.day = day
        self.last_ts: datetime | None = None
        self.n_rows = 0
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.spreads: list[float] = []


@register
class VwapReversion(Signal):
    """Fade a z > 2 price-vs-session-VWAP stretch — chop regimes only."""

    name = "vwap_reversion"
    hypothesis = (
        "Institutional execution engines (and the brokers graded against them) "
        "benchmark to session VWAP and slice parent orders to track it; a stretch "
        "of price beyond z > 2 of the session's own price-VWAP spread distribution "
        "so far (minimum 60 session bars, VWAP built strictly from the session's "
        "visible bars) that arrives WITHOUT new information is a temporary "
        "liquidity-demand imbalance, and the VWAP-pegged flow absorbs it back "
        "toward the benchmark over roughly the next hour. We FADE the stretch "
        "(sell above VWAP, buy below) with a 60-minute horizon and conviction "
        "scaled by |z| (saturating at 4). The other side is the impatient "
        "liquidity taker who crossed the book to demand immediacy; they keep "
        "paying because urgency — hedging deadlines, margin, mandate-driven "
        "rebalancing — recurs every session and is price-inelastic while it "
        "lasts. LOAD-BEARING regime clause: chopping (non-trending, "
        "non-stressed) buckets ONLY — in a trending session the same stretch is "
        "information (repricing), not imbalance, so allowed_regimes is the two "
        "chopping buckets and the regime gate, not the z-score, carries that "
        "half of the claim. All thresholds (z > 2, 60-bar minimum, 60-minute "
        "horizon, conviction saturation at |z| = 4) are fixed economic choices "
        "written here, never swept."
    )
    allowed_regimes = CHOP_BUCKETS
    warmup_bars = 60
    config_type = VwapReversionConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: Per-symbol session memo — a pure performance cache (see
        #: :class:`_SessionState`); one instance per run, like the engine.
        self._sessions: dict[str, _SessionState] = {}

    def _rebuild_session(
        self, ctx: SignalContext, bar: BarEvent, now: datetime, session_day: object
    ) -> np.ndarray:
        """The ORIGINAL full computation, verbatim; rebuilds the memo from it.

        Every semantic choice below (visibility re-filter, session filter,
        sort, current-bar append, truncation to ``max_history_bars``) is the
        reference implementation; the incremental path in :meth:`on_bar` is
        valid only when it provably mirrors this.
        """
        cfg: VwapReversionConfig = self.config  # type: ignore[assignment]
        rows = _history_rows(ctx.history(bar.symbol, cfg.max_history_bars))
        # Session-so-far only: current ET session, nothing after the decision
        # instant — even a ctx that over-serves cannot leak the future or a
        # prior session into the spread distribution.
        rows = [
            r
            for r in rows
            if r[0] <= now and r[0].astimezone(MARKET_TZ).date() == session_day
        ]
        rows.sort(key=lambda r: r[0])
        if not rows or rows[-1][0] != bar.ts:
            rows.append(
                (bar.ts, float(bar.close), float(bar.high), float(bar.low), float(bar.volume))
            )
        rows = rows[-cfg.max_history_bars:]

        closes = np.array([r[1] for r in rows], dtype=float)
        highs = np.array([r[2] for r in rows], dtype=float)
        lows = np.array([r[3] for r in rows], dtype=float)
        volumes = np.array([r[4] for r in rows], dtype=float)

        spreads = spread_series(closes, highs, lows, volumes)

        state = _SessionState(session_day)
        state.last_ts = rows[-1][0]
        state.n_rows = len(rows)
        # np.cumsum accumulates sequentially, so continuing these sums one
        # bar at a time reproduces the full recomputation bit for bit.
        typical = (highs + lows + closes) / 3.0
        state.cum_pv = float(np.cumsum(typical * volumes)[-1]) if len(rows) else 0.0
        state.cum_v = float(np.cumsum(volumes)[-1]) if len(rows) else 0.0
        state.spreads = [float(s) for s in spreads]
        self._sessions[bar.symbol] = state
        return spreads

    def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
        cfg: VwapReversionConfig = self.config  # type: ignore[assignment]
        bar: BarEvent = ctx.bar
        now = _decision_ts(ctx)
        session_day = now.astimezone(MARKET_TZ).date()

        state = self._sessions.get(bar.symbol)
        contiguous = (
            state is not None
            and state.day == session_day
            and state.last_ts is not None
            and bar.ts == now
            and bar.ts - state.last_ts == _ONE_MINUTE
            # At the truncation cap the original DROPS the oldest rows; the
            # incremental sums cannot mirror that, so fall back per bar.
            and state.n_rows + 1 < cfg.max_history_bars
        )
        if contiguous:
            # The engine feeds every completed bar of a symbol through
            # on_bar exactly once, in order, so the memo plus the current
            # bar IS the session row list the full recomputation would
            # build. Identical arithmetic, one bar at a time.
            close = float(bar.close)
            typical = (float(bar.high) + float(bar.low) + close) / 3.0
            volume = float(bar.volume)
            state.cum_pv += typical * volume
            state.cum_v += volume
            if state.cum_v > 0.0:
                state.spreads.append(close - state.cum_pv / state.cum_v)
            state.last_ts = bar.ts
            state.n_rows += 1
            spreads = np.asarray(state.spreads, dtype=float)
        else:
            spreads = self._rebuild_session(ctx, bar, now, session_day)

        if spreads.size < cfg.min_session_bars:
            return []
        z = current_z(spreads)
        if z is None or abs(z) <= cfg.z_entry:  # the claim is z > 2, strictly
            return []
        return [
            SignalEvent(
                ts=now,
                symbol=bar.symbol,
                side=Side.SELL if z > 0 else Side.BUY,
                conviction=min(1.0, abs(z) / cfg.conviction_full_z),
                horizon_minutes=cfg.horizon_minutes,
                signal_name=self.name,
            )
        ]

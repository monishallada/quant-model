"""IndexLeadLag: the index absorbs macro flow first; lagging high-beta
single names reprice with a measurable delay.

MECHANISM
---------
Macro order flow (rate prints, index rebalance, futures-led sweeps) hits the
index products (SPY, QQQ) first because that is where the liquidity is.
Single-name liquidity providers in high-beta names hedge their inventory
AGAINST the index: when the index jumps, their first action is to adjust the
index hedge, and only then do they widen and re-center the single-name
quotes. For a window of minutes the high-beta name's book still shows
resting orders priced off the pre-jump index level. The other side of this
trade is that stale resting order — it keeps paying because re-quoting
hundreds of names lags the one-ticket index hedge, structurally and
repeatedly.

The relationship is MEASURED, never assumed: the beta linking a name to the
index is a rolling 63-session daily beta computed point-in-time from prior
sessions' closes only (:func:`measure_beta`), and the "lag" is observed
directly as the shortfall of the name's own trailing 5-minute return versus
its beta-implied move — if the name has already repriced, there is no trade.

FIXED ECONOMIC CHOICES (written into the hypothesis; never swept — a
variant is its own trial):

* trigger: |index trailing 5-minute return| > 1.0x the index's own
  expanding 5-minute ATR (mean absolute 5-minute move over the FULL history
  to date — an expanding window, not a fitted lookback);
* lag: the name's signed 5-minute return in the index's direction is under
  HALF the beta-implied move;
* high-beta: measured prior-days beta >= 1.0;
* horizon: ~30 minutes — the re-quoting window, after which the edge decays.

REGIME EXCLUSION (stated, not tuned): all regimes EXCEPT
``stressed_backwardation``. When the vol curve inverts (VIX9D > VIX) the
lead-lag inverts to contagion — index weakness begets forced single-name
liquidation rather than a catch-up drift, and the stale quote is no longer
the loser. The signal therefore declares every vol state but the stressed
one.

Data access: signals see the world only through the engine-provided ctx
(bars already replayed, point-in-time by construction). This module never
touches a loader or a feed. Instances keep small derived-value memos keyed
by (symbol, timestamp/session); use a fresh instance per run.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta
from typing import Final, Sequence

from pydantic import Field

from edge.core.events import MARKET_TZ, BarEvent, Side, SignalEvent
from edge.regime.classifier import VOL_STATES, VOL_STRESSED
from edge.signals.base import Signal, SignalConfig
from edge.signals.registry import register

__all__ = [
    "BETA_WINDOW_DAYS",
    "IndexLeadLag",
    "IndexLeadLagConfig",
    "MIN_ATR_OBSERVATIONS",
    "MIN_BETA_DAYS",
    "RET_WINDOW_BARS",
    "measure_beta",
]

# ---------------------------------------------------------------------------
# Fixed definitions (constants, not config: nothing here may ever be swept)
# ---------------------------------------------------------------------------

#: The trailing-return window, in minute bars: "the 5-minute move".
RET_WINDOW_BARS: Final[int] = 5

#: Minimum prior 5-minute observations before the expanding ATR is usable —
#: half an hour of tape; before that the distribution is not a distribution.
MIN_ATR_OBSERVATIONS: Final[int] = 30

#: The rolling daily-beta window: 63 sessions, one calendar quarter.
BETA_WINDOW_DAYS: Final[int] = 63

#: Minimum paired daily returns before a measured beta is trusted at all.
MIN_BETA_DAYS: Final[int] = 20

#: Bounded index-history window per incremental sync. Any minute the signal
#: is called, at most a handful of index bars are new since the last sync;
#: a window that cannot PROVE continuity back to the last ingested bar
#: triggers a full rebuild instead of guessing.
_INDEX_SYNC_WINDOW: Final[int] = 64


class IndexLeadLagConfig(SignalConfig):
    """IndexLeadLag parameters. Defaults ARE the hypothesis's fixed economic
    choices; a different value is a different experiment (its own trial),
    never an in-place tune."""

    #: The leading index product the universe is hedged against.
    index_symbol: str = "SPY"
    #: Trigger multiple on the index's expanding 5-minute ATR (k = 1.0).
    atr_k: float = Field(default=1.0, gt=0.0)
    #: "Lagging" = the name's own move is under this fraction of the
    #: beta-implied move (one half).
    lag_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    #: High-beta filter: measured prior-days beta must reach this (1.0).
    min_beta: float = Field(default=1.0, gt=0.0)
    #: Catch-up horizon in minutes (~30: the re-quoting window).
    horizon_minutes: int = Field(default=30, gt=0)


# ---------------------------------------------------------------------------
# Pure measurement helpers (bars in, numbers out; no data access)
# ---------------------------------------------------------------------------


def _session_of(ts: datetime) -> date:
    """The ET session date a bar belongs to."""
    return ts.astimezone(MARKET_TZ).date()


def _trailing_return(bars: Sequence[BarEvent], now: datetime) -> float | None:
    """The strict trailing :data:`RET_WINDOW_BARS`-minute return ending at ``now``.

    Requires the latest bar to close exactly at ``now`` and the base bar to
    close exactly ``RET_WINDOW_BARS`` minutes earlier WITHIN the same ET
    session — a gapped tape or an overnight span is not a 5-minute move, so
    the measurement abstains (``None``) rather than guessing.
    """
    if len(bars) < RET_WINDOW_BARS + 1:
        return None
    last = bars[-1]
    base = bars[-(RET_WINDOW_BARS + 1)]
    if last.ts != now:
        return None
    if last.ts - base.ts != timedelta(minutes=RET_WINDOW_BARS):
        return None
    if _session_of(last.ts) != _session_of(base.ts):
        return None
    if base.close <= 0.0:
        return None
    return last.close / base.close - 1.0


def _expanding_abs_move_mean(bars: Sequence[BarEvent], now: datetime) -> float | None:
    """Expanding mean |5-minute move| of ``bars`` STRICTLY BEFORE ``now``.

    The index's own 5-minute ATR distribution: every intra-session
    RET_WINDOW-minute move in the full history to date (an expanding window,
    never a fitted lookback), excluding the move that ends at ``now`` itself
    — the trigger is compared against its past, not against itself. Needs at
    least :data:`MIN_ATR_OBSERVATIONS` observations; otherwise ``None``.
    """
    values: list[float] = []
    for i in range(RET_WINDOW_BARS, len(bars)):
        cur = bars[i]
        if cur.ts >= now:  # bars are oldest-first: nothing further qualifies
            break
        base = bars[i - RET_WINDOW_BARS]
        if cur.ts - base.ts != timedelta(minutes=RET_WINDOW_BARS):
            continue
        if _session_of(cur.ts) != _session_of(base.ts):
            continue
        if base.close <= 0.0:
            continue
        values.append(abs(cur.close / base.close - 1.0))
    if len(values) < MIN_ATR_OBSERVATIONS:
        return None
    return sum(values) / len(values)


def _daily_closes_before(
    bars: Sequence[BarEvent], session: date
) -> list[tuple[date, float]]:
    """Per-session last closes for every ET session STRICTLY BEFORE ``session``.

    Bars are oldest-first (the engine's replay order); the current session
    and anything after it is excluded — this is what makes every downstream
    beta a prior-days measurement.
    """
    out: list[tuple[date, float]] = []
    for bar in bars:
        day = _session_of(bar.ts)
        if day >= session:
            break
        if out and out[-1][0] == day:
            out[-1] = (day, bar.close)
        else:
            out.append((day, bar.close))
    return out


def _beta_core(
    name_daily_pairs: Sequence[tuple[date, float]],
    index_daily_pairs: Sequence[tuple[date, float]],
) -> float | None:
    """The beta arithmetic over already-extracted per-session close pairs.

    Shared by :func:`measure_beta` (bars in) and the signal's incremental
    daily-close state, so the two paths cannot drift: pair the sessions where
    both symbols closed, keep the most recent :data:`BETA_WINDOW_DAYS`
    returns, ``beta = cov/var``; fewer than :data:`MIN_BETA_DAYS` paired
    returns or a flat index yields ``None``.
    """
    name_daily = dict(name_daily_pairs)
    paired = [(d, name_daily[d], c) for d, c in index_daily_pairs if d in name_daily]
    if len(paired) < 2:
        return None
    name_rets: list[float] = []
    index_rets: list[float] = []
    for (_, n0, i0), (_, n1, i1) in zip(paired, paired[1:]):
        if n0 <= 0.0 or i0 <= 0.0:
            continue
        name_rets.append(n1 / n0 - 1.0)
        index_rets.append(i1 / i0 - 1.0)
    name_rets = name_rets[-BETA_WINDOW_DAYS:]
    index_rets = index_rets[-BETA_WINDOW_DAYS:]
    n = len(index_rets)
    if n < MIN_BETA_DAYS:
        return None
    mean_i = sum(index_rets) / n
    mean_n = sum(name_rets) / n
    var_i = sum((r - mean_i) ** 2 for r in index_rets) / n
    if var_i <= 0.0:
        return None
    cov = sum((a - mean_n) * (b - mean_i) for a, b in zip(name_rets, index_rets)) / n
    return cov / var_i


def measure_beta(
    name_bars: Sequence[BarEvent],
    index_bars: Sequence[BarEvent],
    session: date,
) -> float | None:
    """Rolling 63-session daily beta of the name on the index, point-in-time.

    Uses ONLY sessions strictly before ``session`` (prior days: today's
    bars, however many have printed, never contaminate the measurement —
    truncating or extending today's tape cannot change the answer). Daily
    returns are computed over the sessions where BOTH symbols closed, the
    most recent :data:`BETA_WINDOW_DAYS` returns are kept, and
    ``beta = cov(name, index) / var(index)``. Fewer than
    :data:`MIN_BETA_DAYS` paired returns, or a flat index, yields ``None``
    — an unmeasurable beta is never defaulted.
    """
    return _beta_core(
        _daily_closes_before(name_bars, session),
        _daily_closes_before(index_bars, session),
    )


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------


@register
class IndexLeadLag(Signal):
    """Same-direction catch-up on high-beta names lagging an index jump."""

    name = "index_leadlag"
    hypothesis = (
        "Index products (SPY/QQQ) absorb macro flow first; single-name "
        "liquidity providers in high-beta names hedge their inventory "
        "against the index, so after an index jump they adjust the one-"
        "ticket index hedge before re-centering hundreds of single-name "
        "quotes — for minutes the stale resting order on the name's book is "
        "still priced off the pre-jump index level. The other side is that "
        "stale resting order, and it keeps paying because re-quoting "
        "structurally lags the index hedge. Fixed choices: trigger when the "
        "index's trailing 5-minute return exceeds 1.0x its own expanding "
        "5-minute ATR (full history to date, no fitted lookback); trade the "
        "SAME direction on names whose own 5-minute return is under HALF "
        "the beta-implied move, where beta is a rolling 63-session daily "
        "beta measured point-in-time from prior sessions only — the lag is "
        "re-measured every session, never assumed; require measured beta "
        ">= 1.0; horizon ~30 minutes (the re-quoting window). Excluded in "
        "stressed_backwardation: with the vol curve inverted the lead-lag "
        "inverts to contagion — index weakness triggers forced single-name "
        "liquidation instead of a catch-up drift — so the stressed regime "
        "is declared off-limits rather than traded and hedged."
    )
    #: Every vol regime EXCEPT stressed backwardation (see module docstring).
    allowed_regimes = frozenset(VOL_STATES) - {VOL_STRESSED}
    #: Enough tape for the expanding ATR floor plus the trailing window.
    warmup_bars = MIN_ATR_OBSERVATIONS + RET_WINDOW_BARS + 1
    config_type = IndexLeadLagConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        # Derived-value memos and incremental measurement state (pure
        # functions of replayed history, ingested bar-by-bar so the engine's
        # replay costs O(1) per bar instead of O(history)). One instance per
        # run: a reused instance fed a DIFFERENT tape would serve stale
        # values; a backward timestamp resets everything defensively.
        self._beta_memo: dict[tuple[str, date], float | None] = {}
        self._reset_measurement_state()

    def _reset_measurement_state(self) -> None:
        #: newest index bar folded into the ATR/daily state (None = cold).
        self._idx_last: datetime | None = None
        #: the last RET_WINDOW_BARS+1 ingested index bars: (ts, close, session).
        self._idx_tail: deque[tuple[datetime, float, date]] = deque(
            maxlen=RET_WINDOW_BARS + 1
        )
        #: qualifying |5-minute move| values not yet folded (ts, value).
        self._atr_pending: deque[tuple[datetime, float]] = deque()
        #: chronological running sum/count of folded values — identical
        #: arithmetic to summing the full list left to right.
        self._atr_sum: float = 0.0
        self._atr_count: int = 0
        #: per-symbol per-session last closes, oldest first (index + names).
        self._daily: dict[str, list[tuple[date, float]]] = {}
        #: newest bar folded into a symbol's daily-close list.
        self._daily_last: dict[str, datetime] = {}
        #: monotonicity guard: a backward clock means a different replay.
        self._last_now: datetime | None = None

    # -- incremental ingestion (pure bookkeeping; no signal logic) ----------

    def _fold_daily(self, symbol: str, ts: datetime, close: float) -> None:
        """Fold one bar into ``symbol``'s per-session last-close list —
        the same last-close-wins construction as :func:`_daily_closes_before`."""
        day = _session_of(ts)
        rows = self._daily.setdefault(symbol, [])
        if rows and rows[-1][0] == day:
            rows[-1] = (day, close)
        else:
            rows.append((day, close))

    def _ingest_index_bar(self, bar: BarEvent) -> None:
        """Fold one NEW index bar (strictly after ``_idx_last``) into state.

        The tail deque holds the last RET_WINDOW_BARS+1 ingested bars, so
        ``tail[0]`` is exactly ``bars[i - RET_WINDOW_BARS]`` of the full
        sequence — the same base bar :func:`_expanding_abs_move_mean` uses.
        """
        session = _session_of(bar.ts)
        self._idx_tail.append((bar.ts, bar.close, session))
        self._idx_last = bar.ts
        tail = self._idx_tail
        if len(tail) == RET_WINDOW_BARS + 1:
            base_ts, base_close, base_session = tail[0]
            if (
                bar.ts - base_ts == timedelta(minutes=RET_WINDOW_BARS)
                and session == base_session
                and base_close > 0.0
            ):
                self._atr_pending.append((bar.ts, abs(bar.close / base_close - 1.0)))

    def _sync_index(self, ctx, index_symbol: str, now: datetime) -> None:
        """Bring index state up to ``now`` from bounded history reads.

        Cold start (or a bounded window that provably cannot reach back to
        the last ingested bar) rebuilds from the full history — exactness
        over speed on unfamiliar tapes; the engine's minute cadence stays on
        the bounded fast path.
        """
        if self._idx_last is not None:
            recent = ctx.history(index_symbol, _INDEX_SYNC_WINDOW)
            if not recent:
                return
            if len(recent) == _INDEX_SYNC_WINDOW and recent[0].ts > self._idx_last:
                self._idx_last = None  # window cannot prove continuity: rebuild
            else:
                for b in recent:
                    if b.ts <= self._idx_last:
                        continue
                    if b.ts > now:
                        break
                    self._ingest_index_bar(b)
                    self._fold_daily(index_symbol, b.ts, b.close)
                return
        # cold rebuild over the full visible history
        self._idx_tail.clear()
        self._atr_pending.clear()
        self._atr_sum = 0.0
        self._atr_count = 0
        self._daily[index_symbol] = []
        for b in ctx.history(index_symbol):
            if b.ts > now:
                break
            self._ingest_index_bar(b)
            self._fold_daily(index_symbol, b.ts, b.close)

    def _sync_name_daily(self, ctx, bar: BarEvent, now: datetime) -> None:
        """Fold the name's bar into its daily-close list (cold start: rebuild)."""
        symbol = bar.symbol
        last = self._daily_last.get(symbol)
        if last is None or bar.ts <= last:
            self._daily[symbol] = []
            newest: datetime | None = None
            for b in ctx.history(symbol):
                if b.ts > now:
                    break
                self._fold_daily(symbol, b.ts, b.close)
                newest = b.ts
            if newest is not None:
                self._daily_last[symbol] = newest
            return
        self._fold_daily(symbol, bar.ts, bar.close)
        self._daily_last[symbol] = bar.ts

    # -- memoized measurements ---------------------------------------------

    def _atr_at(self, now: datetime) -> float | None:
        """The expanding mean |5-minute move| strictly before ``now``.

        Folds pending qualifying moves with ``ts < now`` in chronological
        order — the identical addition sequence to summing
        :func:`_expanding_abs_move_mean`'s value list left to right.
        """
        pending = self._atr_pending
        while pending and pending[0][0] < now:
            _, value = pending.popleft()
            self._atr_sum += value
            self._atr_count += 1
        if self._atr_count < MIN_ATR_OBSERVATIONS:
            return None
        return self._atr_sum / self._atr_count

    def _beta(self, symbol: str, index_symbol: str, session: date) -> float | None:
        key = (symbol, session)
        if key not in self._beta_memo:
            name_rows = [p for p in self._daily.get(symbol, []) if p[0] < session]
            index_rows = [p for p in self._daily.get(index_symbol, []) if p[0] < session]
            self._beta_memo[key] = _beta_core(name_rows, index_rows)
        return self._beta_memo[key]

    # -- the rule -----------------------------------------------------------

    def on_bar(self, ctx) -> list[SignalEvent]:
        cfg: IndexLeadLagConfig = self.config  # type: ignore[assignment]
        bar: BarEvent = ctx.bar
        now = bar.ts
        if bar.symbol == cfg.index_symbol:
            return []  # the index leads; it is never the laggard we trade

        if self._last_now is not None and now < self._last_now:
            # A backward clock is a different replay: stale incremental
            # state (and betas measured off it) must not survive.
            self._beta_memo.clear()
            self._reset_measurement_state()
        self._last_now = now

        self._sync_name_daily(ctx, bar, now)

        # 1) The index must have printed THIS minute and jumped over 1x its
        #    own expanding 5-minute ATR. Stale index tape -> abstain.
        #    (_trailing_return reads only the last RET_WINDOW_BARS+1 bars,
        #    so the bounded read is the full-history computation.)
        index_recent = ctx.history(cfg.index_symbol, RET_WINDOW_BARS + 1)
        if not index_recent or index_recent[-1].ts != now:
            return []
        self._sync_index(ctx, cfg.index_symbol, now)
        r_index = _trailing_return(index_recent, now)
        if r_index is None or r_index == 0.0:
            return []
        atr = self._atr_at(now)
        if atr is None or atr <= 0.0 or abs(r_index) <= cfg.atr_k * atr:
            return []

        # 2) The name's own trailing 5-minute move must be measurable.
        name_recent = ctx.history(bar.symbol, RET_WINDOW_BARS + 1)
        r_name = _trailing_return(name_recent, now)
        if r_name is None:
            return []

        # 3) Beta: measured point-in-time from PRIOR sessions only; the name
        #    must be high-beta (hedged against the index) for the mechanism
        #    to apply at all.
        beta = self._beta(bar.symbol, cfg.index_symbol, _session_of(now))
        if beta is None or beta < cfg.min_beta:
            return []

        # 4) The lag itself, observed: the name's signed move in the index's
        #    direction is under half the beta-implied move. A name that
        #    already repriced has no stale quote left to trade against.
        direction = 1 if r_index > 0.0 else -1
        if direction * r_name >= cfg.lag_fraction * beta * abs(r_index):
            return []

        return [
            SignalEvent(
                ts=now,
                symbol=bar.symbol,
                side=Side.BUY if direction > 0 else Side.SELL,
                conviction=1.0,
                horizon_minutes=cfg.horizon_minutes,
                signal_name=self.name,
            )
        ]

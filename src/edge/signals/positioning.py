"""Positioning signals: who is crowded (COT) and who is informed (Form 4).

Two registered signals over the loader's point-in-time feed kinds — data
reaches them ONLY through the engine-provided ctx (which is built on
:class:`edge.data.loader.EdgeDataLoader`); this module never touches a feed,
a backend, or the loader itself.

:class:`CotExtreme`
    Fades 2-year z-extremes (|z| > 2 over 104 weekly TFF reports) in
    leveraged-fund ES net positioning, trading the index proxy with a
    multi-day horizon. Entries happen only at the release that created the
    extreme: the CFTC report is as of Tuesday but ``available_at`` is Friday
    16:00 ET, and the signal re-filters on ``available_at`` itself, so a
    Tuesday report can never trigger before its Friday release.

:class:`InsiderCluster`
    Goes long a symbol after a CLUSTER of open-market officer buys — at
    least 2 distinct officers inside 21 calendar days with positive net
    open-market dollars (buys minus sales) — read from the ``insider`` feed
    kind, whose ``available_at`` is each filing's EDGAR acceptanceDateTime
    + 5 minutes. A late-filed Form 4 therefore cannot backdate the entry:
    the signal acts on filing availability, never on transaction dates.

DISCIPLINE — every threshold below (z > 2, 104 weeks, >= 2 officers, 21
days, horizons) is a FIXED economic choice written into the hypothesis; a
different setting is a different trial, never a sweep.

Both signals defensively re-filter every frame to
``available_at <= decision_ts`` even though the engine's ctx already
enforces it — availability discipline is part of the signal's own contract
and is unit-testable without an engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import pandas as pd
from pydantic import Field

from edge.core.events import Side, SignalEvent
from edge.regime.classifier import VOL_STATES, VOL_STRESSED
from edge.signals.base import ALL_REGIMES, Signal, SignalConfig
from edge.signals.registry import register

__all__ = [
    "MINUTES_PER_SESSION",
    "CotExtreme",
    "CotExtremeConfig",
    "InsiderCluster",
    "InsiderClusterConfig",
]

#: Regular NYSE session length in minutes; multi-day horizons are declared in
#: sessions and converted to the engine's ``horizon_minutes`` with this.
MINUTES_PER_SESSION = 390


# ---------------------------------------------------------------------------
# ctx helpers: tolerate both context spellings, enforce availability locally
# ---------------------------------------------------------------------------


def _decision_ts(ctx) -> datetime:
    """The decision instant: the engine ctx calls it ``now``, the base
    :class:`~edge.signals.base.SignalContext` protocol ``decision_ts``."""
    now = getattr(ctx, "now", None)
    return now if now is not None else ctx.decision_ts


def _visible_rows(ctx, kind: str, symbol: str | None) -> pd.DataFrame:
    """Point-in-time rows of ``kind`` the signal may see at the decision ts.

    Reads through ``ctx.pit`` (engine context) or ``ctx.frame`` (base
    protocol), then RE-FILTERS to ``available_at <= decision_ts`` and, when
    the frame carries a ``symbol`` column, to ``symbol``. The engine already
    serves point-in-time frames, so the re-filter is normally a no-op — but
    the availability discipline belongs to the signal too, and doing it here
    makes it directly testable against an unfiltered fake ctx.
    """
    getter = getattr(ctx, "pit", None)
    frame = getter(kind, symbol) if getter is not None else ctx.frame(kind, symbol)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["asof_date", "available_at"])
    frame = frame.reset_index(drop=True)
    available = pd.to_datetime(frame["available_at"])
    keep = available <= pd.Timestamp(_decision_ts(ctx))
    if symbol is not None and "symbol" in frame.columns:
        keep &= frame["symbol"] == symbol
    return frame[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CotExtreme: fade crowded fast money in ES at the Friday release
# ---------------------------------------------------------------------------


class CotExtremeConfig(SignalConfig):
    """Fixed economic choices of the COT-extreme fade (never swept — a
    different setting is its own trial)."""

    #: TFF market whose leveraged-fund positioning is measured.
    market: str = "ES"
    #: The index proxy actually traded (the equity engine trades shares).
    trade_symbol: str = "SPY"
    #: The extreme: |z| strictly greater than 2 — the hypothesis's number.
    z_threshold: float = Field(default=2.0, gt=0.0)
    #: 2 years of weekly reports; the z at the latest report uses exactly
    #: this many trailing reports (full window required, like the feed).
    z_window_weeks: int = Field(default=104, gt=1)
    #: Multi-day fade horizon, in sessions.
    horizon_sessions: int = Field(default=5, gt=0)
    #: Entries happen only at the release open following the stamp: a report
    #: first seen more than this many calendar days after its release is
    #: history, not news, and is never traded. 4 days spans Friday 16:00 ET
    #: to the open after a long weekend.
    max_entry_age_days: int = Field(default=4, gt=0)


@register
class CotExtreme(Signal):
    """Fade 2y leveraged-fund positioning extremes in ES on the index proxy.

    ``allowed_regimes`` is ALL, deliberately and stated: crowded-positioning
    unwinds are triggered BY regime shocks, so gating the fade out of any
    regime would remove exactly the sessions the mechanism pays in — the
    positioning extreme itself is the filter.

    One emission per weekly report, at the first bar of ``trade_symbol``
    whose close is at/after the report's ``available_at`` (Friday 16:00 ET;
    the engine's next-bar fill then puts the entry at the following open).
    Instance state tracks the last release considered, so a report is acted
    on exactly once per run.
    """

    name = "cot_extreme"
    hypothesis = (
        "Leveraged funds in ES at a 2-year z-extreme (|z| > 2 over 104 weekly "
        "TFF reports of lev_money net position / open interest) are maximally "
        "crowded fast money whose stop-driven, margin-forced unwinds overshoot "
        "fair value on shocks; the other side is asset managers — slow, "
        "mandate-constrained, calendar-rebalancing capital that cannot absorb "
        "the unwind at fair prices and keeps paying because its flows are "
        "benchmark-driven, not price-driven. Fade the extreme on the index "
        "proxy (SPY) for 5 sessions, entering only at the open following the "
        "Friday 16:00 ET release that revealed it."
    )
    allowed_regimes: ClassVar[str] = ALL_REGIMES
    warmup_bars = 0
    config_type = CotExtremeConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: available_at of the newest release already considered this run.
        self._last_seen_release: pd.Timestamp | None = None

    def on_bar(self, ctx) -> list[SignalEvent]:
        cfg: CotExtremeConfig = self.config  # type: ignore[assignment]
        if ctx.bar.symbol != cfg.trade_symbol:
            return []
        now = _decision_ts(ctx)
        rows = _visible_rows(ctx, "cot", None)
        if rows.empty or "market" not in rows.columns:
            return []
        rows = rows[rows["market"] == cfg.market]
        if rows.empty:
            return []
        rows = rows.sort_values(["available_at", "asof_date"], kind="stable")
        latest_release = pd.Timestamp(rows["available_at"].iloc[-1])
        if self._last_seen_release is not None and latest_release <= self._last_seen_release:
            return []  # this release was already considered (acted or skipped)
        self._last_seen_release = latest_release
        # Entries only at the release open following the stamp: a report that
        # is already old when first seen (e.g. a run starting mid-history) is
        # history, not news.
        if pd.Timestamp(now) - latest_release > pd.Timedelta(days=cfg.max_entry_age_days):
            return []
        weekly = (
            rows.drop_duplicates("asof_date", keep="last")
            .sort_values("asof_date", kind="stable")["lev_money_net_oi"]
            .astype(float)
        )
        window = weekly.tail(cfg.z_window_weeks)
        if len(window) < cfg.z_window_weeks:
            return []  # a z on fewer weeks is a different statistic
        std = float(window.std(ddof=1))
        if not (std > 0.0):  # zero variance or NaN: no defensible z
            return []
        z = (float(window.iloc[-1]) - float(window.mean())) / std
        if abs(z) <= cfg.z_threshold:
            return []
        return [
            SignalEvent(
                ts=now,
                symbol=cfg.trade_symbol,
                # FADE: crowded longs (z > +2) get sold, crowded shorts bought.
                side=Side.SELL if z > 0 else Side.BUY,
                conviction=1.0,
                horizon_minutes=cfg.horizon_sessions * MINUTES_PER_SESSION,
                signal_name=self.name,
            )
        ]


# ---------------------------------------------------------------------------
# InsiderCluster: follow clustered officer buying from filing availability
# ---------------------------------------------------------------------------


class InsiderClusterConfig(SignalConfig):
    """Fixed economic choices of the insider-cluster long (never swept)."""

    #: Distinct officers with open-market buys required inside the window.
    min_officers: int = Field(default=2, gt=1)
    #: Trailing calendar-day window of the feed's served columns; must match
    #: the ``insider`` kind's feature suffix (``*_{window_days}d``).
    window_days: int = Field(default=21, gt=0)
    #: Long-bias horizon, in sessions, inside the hypothesis's 5-10 day band.
    horizon_sessions: int = Field(default=7, gt=0)
    #: A cluster row first seen more than this many calendar days after its
    #: availability is stale news and is never traded (4 spans a filing
    #: accepted Friday evening to the open after a long weekend).
    max_entry_age_days: int = Field(default=4, gt=0)


@register
class InsiderCluster(Signal):
    """Long bias after >= 2 distinct officers net-buy within 21 days.

    Reads the loader's ``insider`` kind (per-symbol rows with
    ``net_insider_dollars_21d`` and ``officer_buyers_21d``; ``available_at``
    is each filing's acceptanceDateTime + 5min, maxed over the window). The
    cluster test runs on the newest AVAILABLE row for the bar's symbol, so a
    late-filed Form 4 moves the entry LATER, never earlier.

    ``allowed_regimes``: every vol regime EXCEPT ``stressed_backwardation``
    (the classifier's vol-curve-inverted state). In acute stress, forced
    liquidation flow is indiscriminate — the diversified seller is dumping
    everything for liquidity reasons, so officer buys stop identifying the
    informed side of the trade and single-name longs just absorb the panic.
    These are :data:`edge.regime.classifier.VOL_STATES` names (checked with
    :func:`edge.signals.base.regime_allowed` against the session's
    ``vol_state``), not promotion buckets.

    One emission per (symbol, newest available row), tracked in per-symbol
    instance state, so a cluster is acted on exactly once per run.
    """

    name = "insider_cluster"
    hypothesis = (
        "When at least 2 distinct officers make open-market Form 4 buys in "
        "the same stock within 21 calendar days and the window's net "
        "open-market dollars (buys minus sales) are positive, the cluster "
        "carries private information about improving fundamentals — officers "
        "buy with their own cash only when they expect to be paid for it; the "
        "other side is the diversified seller pricing only public "
        "information, who keeps paying because index rebalancing, liquidity, "
        "and tax sales are made regardless of firm-specific prospects. Long "
        "bias for 7 sessions (inside the 5-10 day drift band) from the "
        "filings' availability, i.e. EDGAR acceptanceDateTime + 5 minutes — "
        "never from the transaction date, so late filings cannot backdate "
        "the entry."
    )
    allowed_regimes: ClassVar[frozenset[str]] = frozenset(VOL_STATES) - {VOL_STRESSED}
    warmup_bars = 0
    config_type = InsiderClusterConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: symbol -> available_at of the newest row already considered.
        self._last_seen: dict[str, pd.Timestamp] = {}

    def on_bar(self, ctx) -> list[SignalEvent]:
        cfg: InsiderClusterConfig = self.config  # type: ignore[assignment]
        symbol = ctx.bar.symbol
        now = _decision_ts(ctx)
        rows = _visible_rows(ctx, "insider", symbol)
        if rows.empty:
            return []
        net_col = f"net_insider_dollars_{cfg.window_days}d"
        buyers_col = f"officer_buyers_{cfg.window_days}d"
        missing = [c for c in (net_col, buyers_col) if c not in rows.columns]
        if missing:
            raise ValueError(
                f"{self.name}: insider frame is missing {missing}; the feed "
                f"serves *_{cfg.window_days}d columns only when window_days "
                "matches its build window"
            )
        rows = rows.sort_values(["available_at", "asof_date"], kind="stable")
        latest = rows.iloc[-1]
        latest_avail = pd.Timestamp(latest["available_at"])
        last = self._last_seen.get(symbol)
        if last is not None and latest_avail <= last:
            return []  # newest row already considered for this symbol
        self._last_seen[symbol] = latest_avail
        if pd.Timestamp(now) - latest_avail > pd.Timedelta(days=cfg.max_entry_age_days):
            return []  # stale news: the drift window starts at availability
        buyers = int(latest[buyers_col])
        net_dollars = float(latest[net_col])
        # Long-only by construction: clustered selling has too many innocent
        # explanations (diversification, tax, options hygiene) to short.
        if buyers < cfg.min_officers or not (net_dollars > 0.0):
            return []
        return [
            SignalEvent(
                ts=now,
                symbol=symbol,
                side=Side.BUY,
                conviction=1.0,
                horizon_minutes=cfg.horizon_sessions * MINUTES_PER_SESSION,
                signal_name=self.name,
            )
        ]

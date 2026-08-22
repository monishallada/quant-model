"""ShortRatioDeviation: sustained FINRA short-ratio elevation as squeeze fuel.

MECHANISM (see also the interpretation caveat in the FINRA feed's module
docstring, served through the loader as kind ``short_ratio``): the daily
FINRA short-volume ratio is dominated by market-maker shorting — a market
maker filling a customer BUY prints a short sale — so the ratio proxies how
much of the tape was ABSORBED by liquidity providers rather than met by
natural sellers. Levels are meaningless; deviations are the signal. When a
symbol's own-history z-score of the ratio (the feed's ``short_ratio_z``,
computed against the symbol's trailing 63 published sessions) holds above
+2 for three or more consecutive published sessions, aggressive buying has
been persistently absorbed into market-maker inventory. The other side of
this trade is the MM inventory manager, who must eventually re-hedge by
buying back — squeeze susceptibility resolving over the next 1-3 sessions.

FIXED ECONOMIC CHOICES (written into the hypothesis, NOT for sweeping — a
variant is its own trial): z > 2 against the symbol's own 63-day feed
baseline, sustained for 3+ consecutive published sessions, long-only,
1-3 day horizon (default 3). "Rise" here is the fixed elevation test —
each of the last 3 published sessions above the +2 sigma band — never a
fitted slope threshold, which would be a knob begging to be swept.

TIMING / POINT-IN-TIME: the feed stamps every ratio row ``available_at`` =
the NEXT NYSE session's pre-open (09:00 ET), so the signal can NEVER see a
same-day ratio — the pattern completed by session D first becomes visible
at D+1's pre-open, and the signal fires on the first bar of session D+1
(the next open after the pattern). Visibility is enforced by the engine's
context (``available_at <= decision_ts``); this module re-filters
defensively but the engine remains the enforcer.

MULTI-DAY HOLD (the platform's first): the engine supports multi-day
positions via ``EngineParams.max_hold_minutes`` — its :class:`MaxHoldExit`
measures WALL-CLOCK ``bar.ts - entry_ts``, so pass
:meth:`ShortRatioDeviation.max_hold_minutes` (``horizon_days * 1440``
wall-clock minutes; 4320 for the default 3-day horizon). Weekend/holiday
gaps simply defer the exit to the first bar at/past the threshold.
``SignalEvent.horizon_minutes`` is stamped with the same wall-clock figure.

REGIMES: allowed everywhere EXCEPT the high-vol buckets. Stated reason: in
stress regimes an elevated short ratio marks REAL directional selling
pressure (natural sellers hitting bids through intermediaries), not
absorbed buying — the mechanism inverts, so the signal must stand down.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

import pandas as pd
from pydantic import Field

from edge.core.events import Side, SignalEvent
from edge.signals.base import Signal, SignalConfig, SignalContext
from edge.signals.registry import register

__all__ = ["ShortRatioDeviation", "ShortRatioDeviationConfig"]

#: Loader kind this signal consumes (preload via ``EngineParams.pit_kinds``).
SHORT_RATIO_KIND: Final[str] = "short_ratio"

#: The feed's own-history rolling z column (63 published sessions baseline).
Z_COLUMN: Final[str] = "short_ratio_z"

#: Wall-clock minutes per day — the unit MaxHoldExit measures holds in.
MINUTES_PER_DAY: Final[int] = 1440


class ShortRatioDeviationConfig(SignalConfig):
    """Fixed economic choices — recorded in the trial line, never swept.

    Each default IS the hypothesis's stated number. Running a different
    value is a different experiment: record it as its own trial.
    """

    #: Own-history z the ratio must exceed each day (strict >): the +2 sigma
    #: band against the feed's 63-published-session baseline.
    z_threshold: float = Field(default=2.0, gt=0.0)
    #: Consecutive published sessions the elevation must be sustained for.
    #: >= 2 structurally: a single-day spike is noise, not absorption.
    min_sustained_days: int = Field(default=3, ge=2)
    #: Holding horizon in wall-clock days; the stated economic band is 1-3.
    horizon_days: int = Field(default=3, ge=1, le=3)


@register
class ShortRatioDeviation(Signal):
    """Long-only squeeze-susceptibility bias from sustained short-ratio z-rise.

    Fires one BUY per symbol on the first bar at which the pattern — the
    symbol's ``short_ratio_z`` strictly above ``z_threshold`` on each of its
    ``min_sustained_days`` most recent visible published sessions — is
    newly visible (i.e. the next open after the pattern day, since ratios
    publish at the next session's pre-open). Re-fires only when a NEW
    pattern day becomes visible; sizing, regime gating, and exits belong to
    the engine.
    """

    name = "short_ratio_deviation"
    hypothesis = (
        "FINRA daily short-volume is dominated by market-maker shorting (an MM "
        "filling a customer buy prints a short sale), so a symbol's short-volume "
        "ratio proxies how much of the tape liquidity providers absorbed rather "
        "than natural sellers met; when the ratio's z-score against the symbol's "
        "own 63-day published history holds strictly above 2 for 3+ consecutive "
        "published sessions, aggressive buying has been persistently absorbed "
        "into MM inventory. The other side is the MM inventory manager, who must "
        "re-hedge the accumulated short inventory by buying back and keeps "
        "paying because inventory risk limits force covering regardless of price "
        "— squeeze susceptibility. Long-only at the next open after the pattern "
        "becomes visible (ratios publish next-session pre-open, so same-day "
        "ratios are never seen), horizon 1-3 days (default 3). Excluded in "
        "high-vol regimes: under stress an elevated short ratio marks real "
        "directional selling routed through intermediaries, not absorbed buying."
    )
    #: All buckets EXCEPT high-vol (stated in the hypothesis: the mechanism
    #: inverts under stress). Kept in sync with the regime classifier's
    #: BUCKETS by test, not by import.
    allowed_regimes = frozenset({"low_vol_trending", "low_vol_chopping"})
    warmup_bars = 0
    config_type = ShortRatioDeviationConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: Per-symbol latest pattern day already acted on (emission dedup):
        #: one emission per newly visible pattern day, not per bar.
        self._last_trigger: dict[str, date] = {}

    # ------------------------------------------------------------------
    # context seams: works against both the engine's MarketContext
    # (``now`` + ``pit``) and the base SignalContext protocol
    # (``decision_ts`` + ``frame``). Both serve point-in-time frames.
    # ------------------------------------------------------------------

    @staticmethod
    def _decision_instant(ctx: SignalContext) -> datetime:
        now = getattr(ctx, "now", None)
        return now if now is not None else ctx.decision_ts

    @staticmethod
    def _ratio_frame(ctx: SignalContext, symbol: str) -> pd.DataFrame:
        pit = getattr(ctx, "pit", None)
        if pit is not None:
            return pit(SHORT_RATIO_KIND, symbol)
        return ctx.frame(SHORT_RATIO_KIND, symbol)

    # ------------------------------------------------------------------
    # the rule
    # ------------------------------------------------------------------

    def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
        """Emit one BUY when the sustained-elevation pattern is newly visible."""
        symbol = ctx.bar.symbol
        ts = self._decision_instant(ctx)
        frame = self._ratio_frame(ctx, symbol)
        if frame is None or frame.empty:
            return []
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] == symbol]
        if "available_at" in frame.columns:
            # Defense-in-depth only: the ctx already served a point-in-time
            # frame — this can only see LESS, never more.
            visible = pd.to_datetime(frame["available_at"]) <= pd.Timestamp(ts)
            frame = frame[visible]
        if frame.empty:
            return []
        if Z_COLUMN not in frame.columns:
            raise ValueError(
                f"signal {self.name!r} needs the feed column {Z_COLUMN!r} in the "
                f"{SHORT_RATIO_KIND!r} frame; got columns {list(frame.columns)}"
            )
        cfg: ShortRatioDeviationConfig = self.config  # type: ignore[assignment]
        rows = (
            frame.sort_values("asof_date", kind="stable")
            .drop_duplicates("asof_date", keep="last")
            .reset_index(drop=True)
        )
        if len(rows) < cfg.min_sustained_days:
            return []
        window = rows.tail(cfg.min_sustained_days)
        z = pd.to_numeric(window[Z_COLUMN], errors="coerce")
        # NaN (warmup / zero-dispersion baseline) fails the comparison:
        # an unknowable elevation never counts as a sustained one.
        if not bool((z > cfg.z_threshold).all()):
            return []
        trigger = pd.Timestamp(rows["asof_date"].iloc[-1]).date()
        last = self._last_trigger.get(symbol)
        if last is not None and trigger <= last:
            return []  # this pattern day was already acted on
        self._last_trigger[symbol] = trigger
        return [
            SignalEvent(
                ts=ts,
                symbol=symbol,
                side=Side.BUY,  # long-only by construction: the mechanism
                # (forced MM re-hedge buying) has no short mirror here.
                conviction=1.0,
                horizon_minutes=cfg.horizon_days * MINUTES_PER_DAY,
                signal_name=self.name,
            )
        ]

    # ------------------------------------------------------------------
    # engine wiring for the multi-day horizon
    # ------------------------------------------------------------------

    def max_hold_minutes(self) -> int:
        """Wall-clock minutes for ``EngineParams.max_hold_minutes``.

        The engine's :class:`~edge.runners.engine.MaxHoldExit` measures
        wall-clock ``bar.ts - entry_ts``, so the multi-day horizon is
        ``horizon_days * 1440``; weekend gaps defer the exit to the first
        bar at/past the threshold. This is the sanctioned way to run this
        signal's 1-3 day hold through the engine.
        """
        cfg: ShortRatioDeviationConfig = self.config  # type: ignore[assignment]
        return cfg.horizon_days * MINUTES_PER_DAY

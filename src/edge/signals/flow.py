"""Order-flow signals: informed continuation vs uninformed reversion.

Wave-1's signals all read PRICE. These read the TAPE — signed volume and
flow toxicity — and they are deliberately a MATCHED PAIR making OPPOSITE
predictions from the same observation, discriminated by one variable.

The theory, stated so it can be wrong:

* A large one-sided imbalance created by an INFORMED trader is a metaorder
  being worked in child slices. The remaining slices keep pushing price, so
  the imbalance CONTINUES (Kyle 1985; the empirical square-root impact law).
  The other side is the liquidity provider being adversely selected, who
  widens and re-prices — and who accepts that loss because uninformed flow
  pays the spread that funds it.
* A large one-sided imbalance created by an UNINFORMED liquidity demander
  (index rebalance, margin call, ETF create/redeem) is a payment for
  immediacy. Inventory-holding market makers demand a price concession to
  absorb it, and once inventory is redistributed the concession REVERTS
  (Grossman-Miller 1988). The other side is the impatient trader, who keeps
  paying because immediacy is worth more to them than the concession.

Both effects are real; they differ in WHO is transacting. The discriminator
is flow toxicity: VPIN (Easley/Lopez de Prado/O'Hara) estimates the share of
volume that is informed. So the pair is:

    high imbalance + HIGH toxicity  -> FOLLOW  (metaorder continuation)
    high imbalance + LOW  toxicity  -> FADE    (immediacy concession)

If neither fires, the tape carries no exploitable information at this
horizon on this universe, and that is a publishable answer. Thresholds below
are FIXED economic choices written into the hypotheses — a different value
is a different trial, never a sweep.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

import pandas as pd
from pydantic import Field

from edge.core.events import Side, SignalEvent
from edge.signals.base import Signal, SignalConfig
from edge.signals.registry import register

#: Session window: skip the opening auction's price discovery and stop
#: early enough that the horizon fits inside the session.
_ENTRY_START = time(9, 45)
_ENTRY_END = time(15, 0)


class FlowConfig(SignalConfig):
    """Fixed economic choices shared by the pair (never swept)."""

    #: |imbalance| above this is "one-sided" — 35% of net signed volume over
    #: the rolling window is a large deviation from a balanced tape.
    imbalance_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    #: VPIN above/below this splits toxic from benign flow. 0.5 is the
    #: midpoint of VPIN's [0,1] range, not a fitted quantile.
    vpin_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: Trade intensity must be elevated: something is actually happening.
    min_intensity_z: float = Field(default=1.0)
    #: Minutes to hold — long enough for a metaorder to finish working or a
    #: concession to be replenished, short enough to stay intraday.
    horizon_minutes: int = Field(default=30, gt=0)


def _micro_now(ctx: Any, symbol: str) -> pd.Series | None:
    """The newest microstructure row visible at the decision instant."""
    getter = getattr(ctx, "pit_latest", None)
    if getter is not None:
        return getter("micro", symbol)
    frame = ctx.frame("micro", symbol)          # base-protocol spelling
    return None if frame.empty else frame.iloc[-1]


def _decision_ts(ctx: Any) -> datetime:
    ts = getattr(ctx, "now", None)
    return ts if ts is not None else ctx.decision_ts


def _tradeable(ctx: Any, cfg: FlowConfig) -> pd.Series | None:
    """Shared gate: in-window, elevated intensity, formed VPIN, one-sided."""
    now = _decision_ts(ctx)
    if not (_ENTRY_START <= now.timetz().replace(tzinfo=None) <= _ENTRY_END):
        return None
    row = _micro_now(ctx, ctx.bar.symbol)
    if row is None:
        return None
    vpin, imbalance, intensity = row.get("vpin"), row.get("imbalance"), row.get("intensity_z")
    if pd.isna(vpin) or pd.isna(imbalance) or pd.isna(intensity):
        return None                              # unformed statistics never trade
    if intensity < cfg.min_intensity_z:
        return None
    if abs(float(imbalance)) < cfg.imbalance_threshold:
        return None
    # the row must belong to THIS minute: a stale row means the tape went
    # quiet, and a quiet tape carries no flow signal
    row_ts = pd.Timestamp(row["ts"])
    if (pd.Timestamp(now) - row_ts).total_seconds() > 90:
        return None
    return row


CONTINUATION_HYPOTHESIS = (
    "A large one-sided trade imbalance accompanied by HIGH flow toxicity "
    "(VPIN above 0.5) is an informed metaorder being worked in child slices: "
    "the remaining slices keep pushing price in the same direction over the "
    "next half hour. The other side is the market maker being adversely "
    "selected, who accepts the loss because uninformed flow pays the spread "
    "that funds it, and who must re-hedge inventory in the same direction. "
    "Follows the imbalance; excluded in stressed backwardation where the "
    "toxicity measure reflects panic liquidation rather than information."
)

REVERSION_HYPOTHESIS = (
    "A large one-sided trade imbalance accompanied by LOW flow toxicity "
    "(VPIN below 0.5) is uninformed demand for immediacy — an index "
    "rebalance, a margin call, an ETF create/redeem — and the price "
    "concession the inventory-holding market maker demanded to absorb it "
    "reverts once that inventory is redistributed over the next half hour. "
    "The other side is the impatient trader, who keeps paying because "
    "immediacy is worth more to them than the concession. Fades the "
    "imbalance; excluded in stressed backwardation where a concession can "
    "keep widening instead of reverting."
)


@register
class FlowContinuation(Signal):
    """Follow toxic one-sided flow (informed metaorder continuation)."""

    name = "flow_continuation"
    hypothesis = CONTINUATION_HYPOTHESIS
    allowed_regimes = frozenset({"low_vol_trending", "low_vol_chopping", "high_vol_trending"})
    warmup_bars = 30

    def __init__(self, config: FlowConfig | None = None) -> None:
        self.config = config or FlowConfig()

    def on_bar(self, ctx: Any) -> list[SignalEvent]:
        cfg = self.config
        row = _tradeable(ctx, cfg)
        if row is None or float(row["vpin"]) < cfg.vpin_threshold:
            return []
        imbalance = float(row["imbalance"])
        return [SignalEvent(
            ts=_decision_ts(ctx),
            symbol=ctx.bar.symbol,
            side=Side.BUY if imbalance > 0 else Side.SELL,
            conviction=min(abs(imbalance), 1.0),
            horizon_minutes=cfg.horizon_minutes,
            signal_name=self.name,
        )]


@register
class FlowReversion(Signal):
    """Fade benign one-sided flow (uninformed immediacy concession)."""

    name = "flow_reversion"
    hypothesis = REVERSION_HYPOTHESIS
    allowed_regimes = frozenset({"low_vol_trending", "low_vol_chopping", "high_vol_trending"})
    warmup_bars = 30

    def __init__(self, config: FlowConfig | None = None) -> None:
        self.config = config or FlowConfig()

    def on_bar(self, ctx: Any) -> list[SignalEvent]:
        cfg = self.config
        row = _tradeable(ctx, cfg)
        if row is None or float(row["vpin"]) >= cfg.vpin_threshold:
            return []
        imbalance = float(row["imbalance"])
        return [SignalEvent(
            ts=_decision_ts(ctx),
            symbol=ctx.bar.symbol,
            side=Side.SELL if imbalance > 0 else Side.BUY,   # FADE
            conviction=min(abs(imbalance), 1.0),
            horizon_minutes=cfg.horizon_minutes,
            signal_name=self.name,
        )]

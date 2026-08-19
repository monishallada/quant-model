"""Catalyst variance play — a NEGATIVE-EV, right-tail-seeking gamble.

READ THIS BEFORE READING ANY NUMBER THIS STRATEGY PRODUCES.

This is not an edge strategy and is not intended to become one. It buys long
strangles across many catalysts, sized tiny, and wins only when the realized
post-event move is large enough to beat BOTH the premium paid AND the post-event
IV collapse. Most positions lose. A minority win big. That asymmetry, spread
across many small independent shots, is the entire mechanism.

**The safeguards bound how fast it loses. They do not make it profitable.**
A 50% cash floor and 1.5% position sizing guarantee survival, not growth. If a
segment comes back positive, the correct reading is right-tail variance, not
edge — which is exactly what the mandatory concentration check is here to
characterise.

Why long strangles are structurally against us at a catalyst:

- We are LONG volatility into an event whose implied vol is at its annual peak.
  The post-print IV crush works against both legs simultaneously.
- The options market prices the expected move. Selecting names whose realized
  move has historically exceeded their implied move is a weak magnitude lean,
  NOT a predictor — this project has measured repeatedly that its directional
  and magnitude signals do not survive costs.
- Every prior options campaign here lost money. There is no reason to expect
  this one to be the exception, and it is not being built on that hope.

What it IS good for: a paper competition scored on the ceiling. A bounded-loss
lottery ticket with an uncapped right tail can place; a compounding strategy
with +1%/month cannot. That is the only claim being made.

Flagged in metadata as paper-competition / variance / negative-EV and
explicitly BARRED from the live deployment path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from catalyst.core.chains import quotable
from catalyst.core.interfaces import (
    Cadence,
    CatalystStrategy,
    Opportunity,
    StrategyContext,
)
from catalyst.core.types import (
    Catalyst,
    Direction,
    ExitRules,
    OptionChain,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)
from catalyst.strategies.registry import StrategyMeta, register

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VarianceParams:
    """Every number lives here; nothing hardcoded in the logic."""

    otm_delta: float = 0.20          # target |delta| per leg
    delta_tolerance: float = 0.12    # accept 0.08-0.32 before giving up
    dte_min_days: int = 3            # first expiry >= catalyst + 3d (never 0DTE)
    dte_max_days: int = 21
    entry_lead_days: int = 1         # enter 1 session before the print
    entry_lead_max: int = 2
    per_trade_risk_fraction: float = 0.015   # 1.5% of equity per shot
    min_ask: float = 0.05
    max_spread_pct_of_mid: float = 0.10      # liquidity gate
    use_debit_spread: bool = False           # alternative vehicle, default OFF
    # Exits: scale out at +100%, let the remainder RUN. Capping the runner
    # destroys the only outcome that justifies the structure — v6 measured a
    # 3x cap collapsing P(10x) from 26% to 5.2%.
    tp1_gain: float = 1.00
    tp1_fraction: float = 0.50
    use_stops: bool = False
    close_after_event_days: int = 2


class CatalystVarianceStrategy(CatalystStrategy):
    """Long strangle into a catalyst. Magnitude bet, direction-agnostic."""

    name = "catalyst_variance"
    cadence = Cadence.CATALYST

    def __init__(self, params: VarianceParams | None = None) -> None:
        self._p = params or VarianceParams()
        self.skipped_no_expiry = 0
        self.skipped_no_legs = 0
        self.skipped_liquidity = 0
        self.skipped_window = 0
        self.built = 0

    # -- which expiries are worth fetching --------------------------------
    def catalyst_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        event = catalyst.when.date()
        if not self._in_entry_window(as_of, event):
            return []
        lo = event + timedelta(days=self._p.dte_min_days)
        hi = event + timedelta(days=self._p.dte_max_days)
        elig = [e for e in sorted(available) if lo <= e <= hi]
        return elig[:1]      # shortest expiry that contains the event + runway

    def _in_entry_window(self, as_of: date, event: date) -> bool:
        """1-2 sessions before the print, when OTM premium is cheaper than at
        peak pre-event IV."""
        lead = (event - as_of).days
        return self._p.entry_lead_days <= lead <= self._p.entry_lead_max

    # -- the bet -----------------------------------------------------------
    def evaluate(
        self, catalyst: Catalyst, chain: OptionChain, signal: SignalResult,
        as_of: datetime,
    ) -> ProposedTrade | None:
        p = self._p
        event = catalyst.when.date()
        if not self._in_entry_window(as_of.date(), event):
            self.skipped_window += 1
            return None

        expiries = self.catalyst_expiries(catalyst, chain.expirations(), as_of.date())
        if not expiries:
            self.skipped_no_expiry += 1
            return None
        expiry = expiries[0]

        call = self._leg(chain, expiry, OptionRight.CALL)
        put = self._leg(chain, expiry, OptionRight.PUT)
        if call is None or put is None:
            self.skipped_no_legs += 1
            return None

        # Liquidity gate: a cheap OTM ticket nobody will trade is not a
        # position, it is a modelling artifact.
        for leg in (call, put):
            if leg.mid <= 0 or leg.spread_pct_of_mid > p.max_spread_pct_of_mid:
                self.skipped_liquidity += 1
                return None

        # Debit paid = both legs at the ASK. This IS the max loss: a long
        # strangle cannot lose more than its premium, which is what makes the
        # downside bounded and the sizing meaningful.
        unit_cost = call.ask + put.ask
        if unit_cost <= 0:
            self.skipped_no_legs += 1
            return None

        self.built += 1
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[OrderLeg(key=call.key, side=Side.BUY, qty=1),
                  OrderLeg(key=put.key, side=Side.BUY, qty=1)],
            unit_cost=unit_cost,
            unit_max_loss=unit_cost,
            # NEUTRAL is the honest label: this is a magnitude bet with no
            # directional opinion whatsoever.
            direction=Direction.NEUTRAL,
            exit_rules=ExitRules(
                tp1_gain=p.tp1_gain,
                tp1_fraction=p.tp1_fraction,
                use_stops=p.use_stops,
                # Close once the event has resolved — holding a decaying
                # post-event ticket is paying theta for a catalyst that has
                # already happened.
                hard_exit_date=event + timedelta(days=p.close_after_event_days),
                close_before_expiry_days=1,
            ),
            per_trade_risk_fraction=p.per_trade_risk_fraction,
            rationale={"structure": "long_strangle", "expiry": str(expiry),
                       "call_strike": call.key.strike, "put_strike": put.key.strike,
                       "debit": round(unit_cost, 4),
                       "event": str(event), "lead_days": (event - as_of.date()).days},
        )

    def _leg(self, chain: OptionChain, expiry: date, right: OptionRight):
        """Nearest contract to the target |delta| within tolerance."""
        p = self._p
        candidates = [c for c in quotable(chain.slice(expiry=expiry, right=right))
                      if c.ask > p.min_ask and c.greeks is not None]
        if not candidates:
            return None
        best = min(candidates, key=lambda c: abs(abs(c.greeks.delta) - p.otm_delta))
        if abs(abs(best.greeks.delta) - p.otm_delta) > p.delta_tolerance:
            return None
        return best


def build(cfg) -> CatalystVarianceStrategy:
    section = getattr(cfg, "catalyst_variance", None)
    if section is None:
        return CatalystVarianceStrategy()
    return CatalystVarianceStrategy(VarianceParams(
        otm_delta=getattr(section, "otm_delta", VarianceParams.otm_delta),
        dte_min_days=getattr(section, "dte_min_days", VarianceParams.dte_min_days),
        entry_lead_days=getattr(section, "entry_lead_days", VarianceParams.entry_lead_days),
        per_trade_risk_fraction=getattr(section, "per_trade_risk_fraction",
                                        VarianceParams.per_trade_risk_fraction),
        use_debit_spread=getattr(section, "use_debit_spread",
                                 VarianceParams.use_debit_spread),
    ))


register(
    StrategyMeta(
        name="catalyst_variance",
        module="catalyst.strategies.active.catalyst_variance",
        status="active",
        notes=("PAPER-COMPETITION / VARIANCE / NEGATIVE-EV. Long strangles into "
               "catalysts. No edge by design — bounded-loss lottery ticket built "
               "for right-tail frequency, not growth. BARRED FROM LIVE: the "
               "framework's no-live-without-validated-edge rule applies and this "
               "strategy has no edge by construction."),
    ),
    build,
)

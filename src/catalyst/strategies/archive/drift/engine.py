"""Drift Harvest — the three-arm engine.

One class, three structures, identical gates. Running the arms on the same
event set with the same filters is what makes the comparison an experiment
rather than three unrelated backtests:

- ``credit``  Arm A, the strategy: an OTM credit spread on the FAR side of the
  drift (gap up → sell a put spread below; gap down → sell a call spread
  above). Wins if the stock drifts with the gap, goes nowhere, or drifts
  mildly against — and collects theta instead of paying it.
- ``condor``  Arm B, direction-neutral control: both spreads. Isolates the
  variance risk premium alone, with the drift tilt removed.
- ``debit``   Arm C, drift-direction control: a debit spread WITH the drift
  (Engine C's structure). Isolates the drift alone, still paying premium.

If A beats both controls, VRP and drift are real and additive. If A only
matches B, the drift tilt is worthless. If A loses to C, the credit structure
is the problem. Every outcome is informative.

Risk reporting: credit structures report ``unit_max_loss = width - credit``
(a positive number) so the RiskManager's caps, cash floor, and sizing bind
exactly as they do for debit premium. The engine never sizes itself.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

from catalyst.core.config import DriftConfig, RiskConfig
from catalyst.core.interfaces import CatalystStrategy
from catalyst.core.types import (
    Catalyst,
    CatalystType,
    Direction,
    ExitRules,
    OptionChain,
    OptionContract,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)
from catalyst.core.tradingcal import add_trading_days
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.strategies.archive.drift.screener import DriftScreener
from catalyst.core.chains import nearest_delta, quotable

logger = logging.getLogger(__name__)

Arm = Literal["credit", "condor", "debit"]


def _wing(chain: OptionChain, expiry: date, right: OptionRight, short_strike: float,
          spot: float, wing_pct: float) -> OptionContract | None:
    """Long protective wing, ``wing_pct`` of spot beyond the short strike."""
    offset = spot * wing_pct
    target = short_strike - offset if right is OptionRight.PUT else short_strike + offset
    candidates = [c for c in quotable(chain.slice(expiry=expiry, right=right))
                  if (c.key.strike < short_strike if right is OptionRight.PUT
                      else c.key.strike > short_strike)]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.key.strike - target))


class DriftHarvestEngine(CatalystStrategy):
    def __init__(self, cfg: DriftConfig, risk: RiskConfig, screener: DriftScreener,
                 arm: Arm = "credit") -> None:
        self._cfg = cfg
        self._risk = risk
        self._screener = screener
        self._arm = arm
        self.name = f"drift_{arm}"
        self.skipped_no_expiry = 0
        self.skipped_no_structure = 0
        self.skipped_iv_not_rich = 0
        self.skipped_credit_bounds = 0
        self.gate_passed = 0

    # ------------------------------------------------------------------

    def _select_expiry(self, available: list[date], session: date) -> date | None:
        lo, hi = self._cfg.expiry_days
        eligible = [e for e in sorted(available) if lo <= (e - session).days <= hi]
        if not eligible:
            return None
        target = (lo + hi) / 2.0
        return min(eligible, key=lambda e: abs((e - session).days - target))

    def catalyst_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        if catalyst.type is not CatalystType.EARNINGS:
            return []
        reaction = resolve_reaction_session(catalyst)
        if not self._screener.entry_window_open(reaction, as_of):
            return []
        setup = self._screener.setup_for(catalyst)
        if setup.direction is None:
            return []
        expiry = self._select_expiry(available, as_of)
        return [expiry] if expiry else []

    # ------------------------------------------------------------------

    def _credit_spread(self, chain: OptionChain, expiry: date, right: OptionRight,
                       spot: float) -> tuple[OptionContract, OptionContract] | None:
        """(short leg at target delta, long protective wing)."""
        cfg = self._cfg
        delta = cfg.credit.short_delta if self._arm == "credit" else cfg.condor.short_delta
        wing_pct = cfg.credit.wing_width_pct if self._arm == "credit" else cfg.condor.wing_width_pct
        short = nearest_delta(chain, expiry, right, delta)
        if short is None:
            return None
        long_wing = _wing(chain, expiry, right, short.key.strike, spot, wing_pct)
        if long_wing is None:
            return None
        return short, long_wing

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,  # unused: direction is the MEASURED gap, never a forecast
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        if not cfg.enabled or catalyst.type is not CatalystType.EARNINGS:
            return None
        session = as_of.date()
        reaction = resolve_reaction_session(catalyst)
        if not self._screener.entry_window_open(reaction, session):
            return None

        setup = self._screener.setup_for(catalyst)
        if setup.direction is None:
            return None

        expiry = self._select_expiry(chain.expirations(), session)
        if expiry is None:
            self.skipped_no_expiry += 1
            return None

        # Gate: only sell premium that is rich relative to realized movement.
        richness = self._screener.iv_richness(chain, expiry, setup.realized_vol)
        if richness is None or richness < cfg.min_iv_richness:
            self.skipped_iv_not_rich += 1
            return None
        self.gate_passed += 1

        spot = chain.underlying_price
        drift_up = setup.direction == "up"
        legs: list[OrderLeg] = []
        rationale: dict[str, object] = {
            "gap": setup.gap, "iv_richness": richness,
            "realized_vol": setup.realized_vol, "arm": self._arm,
            "dte": (expiry - session).days,
        }

        if self._arm == "debit":
            # Control C: pay premium in the drift direction (Engine C's shape).
            right = OptionRight.CALL if drift_up else OptionRight.PUT
            long_leg = nearest_delta(chain, expiry, right, cfg.debit.long_delta)
            short_leg = nearest_delta(chain, expiry, right, cfg.debit.short_delta)
            if long_leg is None or short_leg is None or long_leg.key.strike == short_leg.key.strike:
                self.skipped_no_structure += 1
                return None
            net_debit = long_leg.mid - short_leg.mid
            worst_debit = long_leg.ask - short_leg.bid
            width = abs(long_leg.key.strike - short_leg.key.strike)
            if net_debit <= 0 or worst_debit >= width:
                self.skipped_no_structure += 1
                return None
            legs = [OrderLeg(key=long_leg.key, side=Side.BUY, qty=1),
                    OrderLeg(key=short_leg.key, side=Side.SELL, qty=1)]
            exit_rules = ExitRules(
                tp_fraction_of_max=cfg.exits.tp_debit_fraction_of_max,
                stop_loss_pct=self._risk.stop_loss_pct, use_stops=True,
                max_hold_trading_days=cfg.exits.hold_trading_days,
                close_before_expiry_days=cfg.exits.close_before_expiry_days,
            )
            return ProposedTrade(
                engine=self.name, catalyst_ref=catalyst.ref, legs=legs,
                unit_cost=net_debit, unit_max_loss=worst_debit,
                direction=Direction.LONG if drift_up else Direction.SHORT,
                exit_rules=exit_rules, per_trade_risk_fraction=cfg.per_trade_risk,
                rationale={**rationale, "width": width, "net_debit": net_debit},
            )

        # --- Credit arms: sell the far side (A) or both sides (B) ---
        sides: list[OptionRight]
        if self._arm == "credit":
            # Gap up ⇒ drift up ⇒ sell PUTS below. Gap down ⇒ sell CALLS above.
            sides = [OptionRight.PUT if drift_up else OptionRight.CALL]
        else:
            sides = [OptionRight.PUT, OptionRight.CALL]

        credit = 0.0
        worst_credit = 0.0  # credit received under worst-case NBBO fills
        width = 0.0
        for right in sides:
            pair = self._credit_spread(chain, expiry, right, spot)
            if pair is None:
                self.skipped_no_structure += 1
                return None
            short_leg, long_leg = pair
            legs.append(OrderLeg(key=short_leg.key, side=Side.SELL, qty=1))
            legs.append(OrderLeg(key=long_leg.key, side=Side.BUY, qty=1))
            credit += short_leg.mid - long_leg.mid
            worst_credit += short_leg.bid - long_leg.ask
            # An iron condor can only lose on ONE side, so the risk basis is the
            # widest single spread, not the sum.
            width = max(width, abs(short_leg.key.strike - long_leg.key.strike))
            rationale[f"{right.value}_short_strike"] = short_leg.key.strike
            rationale[f"{right.value}_long_strike"] = long_leg.key.strike

        if width <= 0 or credit <= 0 or worst_credit <= 0:
            self.skipped_credit_bounds += 1
            return None
        credit_fraction = credit / width
        if self._arm == "credit" and not (
            cfg.credit.min_credit_fraction <= credit_fraction < cfg.credit.max_credit_fraction
        ):
            self.skipped_credit_bounds += 1
            return None

        # Max loss uses the WORST-CASE credit: if fills come in worse than mid,
        # less premium is collected and the loss is correspondingly larger.
        max_loss = width - worst_credit
        if max_loss <= 0:
            self.skipped_credit_bounds += 1
            return None

        return ProposedTrade(
            engine=self.name, catalyst_ref=catalyst.ref, legs=legs,
            unit_cost=-credit,  # negative: premium received
            unit_max_loss=max_loss,
            direction=(Direction.NEUTRAL if self._arm == "condor"
                       else (Direction.LONG if drift_up else Direction.SHORT)),
            exit_rules=ExitRules(
                tp_credit_fraction=cfg.exits.tp_credit_fraction,
                stop_credit_multiple=cfg.exits.stop_credit_multiple,
                use_stops=True,
                max_hold_trading_days=cfg.exits.hold_trading_days,
                close_before_expiry_days=cfg.exits.close_before_expiry_days,
            ),
            per_trade_risk_fraction=cfg.per_trade_risk,
            rationale={**rationale, "width": width, "credit": credit,
                       "credit_fraction": credit_fraction, "max_loss": max_loss},
        )

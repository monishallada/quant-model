"""State 3 — the Kinetic Ignition strategy engine.

Implements the shared ``Strategy`` interface so it flows through the same
screener → strategy → RiskManager → fill engine → exit manager chain as the
archived engines. Given a fired signal and the 09:35 option chain:

- expiration: nearest with ``0 <= DTE <= max_dte`` (calendar days). Days with
  no qualifying expiry are SKIPPED and counted, never silently relaxed —
  which structurally excludes most Monday gap days (Mon→Fri = 4 DTE).
- strike: closest to ``P_0935`` (ATM), call for long, put for short.
- size: requests ``nav_fraction`` of NAV. The RiskManager is authoritative and
  may trim or reject; this engine has no way to override it.

Sizing note: ``unit_max_loss`` is the ASK (worst-case NBBO entry), since a
long option's max loss is the premium actually paid, and the entry fills at
the ask by spec.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from catalyst.core.config import KineticConfig, RiskConfig
from catalyst.core.interfaces import Strategy
from catalyst.core.models import (
    Catalyst,
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
from catalyst.kinetic.screener import KineticScreener

logger = logging.getLogger(__name__)


class KineticIgnitionEngine(Strategy):
    name = "kinetic"

    def __init__(self, cfg: KineticConfig, risk: RiskConfig, screener: KineticScreener) -> None:
        self._cfg = cfg
        self._risk = risk
        self._screener = screener
        self.skipped_no_expiry = 0
        self.skipped_no_contract = 0
        self.skipped_thin_premium = 0

    # ------------------------------------------------------------------

    def select_expiry(self, available: list[date], session: date) -> date | None:
        """Nearest expiration with 0 <= DTE <= max_dte, else None."""
        eligible = [e for e in sorted(available)
                    if 0 <= (e - session).days <= self._cfg.execution.max_dte]
        return eligible[0] if eligible else None

    def select_contract(
        self, chain: OptionChain, expiry: date, right: OptionRight, spot: float
    ) -> OptionContract | None:
        """ATM contract: strike closest to ``spot`` with a usable two-sided quote."""
        candidates = [c for c in chain.slice(expiry=expiry, right=right)
                      if c.ask > 0 and c.bid >= 0]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c.key.strike - spot))

    def required_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        expiry = self.select_expiry(available, as_of)
        return [expiry] if expiry else []

    # ------------------------------------------------------------------

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,  # unused: direction comes from the measured gap+momentum
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        if not cfg.enabled:
            return None
        session = as_of.date()
        kinetic_signal = self._screener.evaluate(catalyst.symbol, session)
        if not kinetic_signal.fires or kinetic_signal.price_0935 is None:
            return None

        expiry = self.select_expiry(chain.expirations(), session)
        if expiry is None:
            self.skipped_no_expiry += 1
            return None

        right = (OptionRight.CALL if kinetic_signal.direction == "long"
                 else OptionRight.PUT)
        contract = self.select_contract(chain, expiry, right, kinetic_signal.price_0935)
        if contract is None:
            self.skipped_no_contract += 1
            return None
        if contract.ask < cfg.execution.min_premium:
            self.skipped_thin_premium += 1
            return None

        direction = (Direction.LONG if kinetic_signal.direction == "long"
                     else Direction.SHORT)
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=f"kinetic:{catalyst.symbol}:{session.isoformat()}",
            legs=[OrderLeg(key=contract.key, side=Side.BUY, qty=1)],
            unit_cost=contract.ask,
            unit_max_loss=contract.ask,  # long premium paid at the ask
            direction=direction,
            exit_rules=ExitRules(use_stops=True, close_before_expiry_days=0),
            per_trade_risk_fraction=cfg.execution.nav_fraction,
            rationale={
                "gap": kinetic_signal.gap,
                "premarket_volume": kinetic_signal.premarket_volume,
                "momentum": kinetic_signal.momentum,
                "price_0935": kinetic_signal.price_0935,
                "strike": contract.key.strike,
                "dte": (expiry - session).days,
                "entry_ask": contract.ask,
                "entry_bid": contract.bid,
            },
        )

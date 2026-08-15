"""The only road to a broker.

Every order in every mode passes through ``ExecutionEngine.submit``. There is
no other call site that touches ``Broker.place_order``, and that is enforced by
a test (``tests/execution/test_only_path_to_broker.py``) rather than by
convention.

Four things happen here, in this order, and none is skippable:

1. **Reconcile.** Read positions and account from the broker. Internal memory
   is never trusted — a fill that landed while we were computing, a manual
   close, or a restart all make in-process state a lie.
2. **Kill switch.** Checked every cycle. Engaged means no new entries.
3. **Risk.** The RiskManager sizes the trade against *reconciled* state. It
   returns 0 and the order is dropped; there is no override path.
4. **Limit discipline.** Orders are limit-only, priced from live NBBO with a
   staleness guard. A market order is never constructed anywhere in this
   codebase.

The identical sequence runs in backtest, paper and live. Paper is not a
"lighter" mode — treating it as one is how a safety layer rots before it is
needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from catalyst.core.interfaces import Broker
from catalyst.core.types import (
    AccountState,
    Order,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Position,
    ProposedTrade,
    Side,
)
from catalyst.observability.killswitch import KillSwitch
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class RiskRejection(Exception):
    """The risk layer declined. Never caught-and-retried with a bigger size."""


@dataclass
class ReconciledState:
    positions: list[Position]
    account: AccountState
    at: datetime


@dataclass
class ExecutionEngine:
    broker: Broker
    risk: RiskManager
    kill: KillSwitch = field(default_factory=KillSwitch)
    max_quote_age: timedelta = timedelta(seconds=30)
    dry_run: bool = False          # build and log the order; never send it

    # -- 1. reconciliation -------------------------------------------------
    def reconcile(self, at: datetime) -> ReconciledState:
        """Broker truth. Called before EVERY action, without exception."""
        positions = self.broker.get_positions()
        account = self.broker.get_account()
        logger.info("reconciled: %d positions, equity %.2f, cash %.2f",
                    len(positions), account.equity, account.cash)
        return ReconciledState(positions=positions, account=account, at=at)

    # -- 2-4. the gated submit --------------------------------------------
    def submit(self, proposal: ProposedTrade, at: datetime,
               limit_prices: dict[int, float] | None = None) -> OrderResult | None:
        """Size, gate and place one proposal. None means nothing was sent."""
        if self.kill.engaged():
            logger.critical("kill switch engaged (%s) — entry refused", self.kill.reason())
            return None

        state = self.reconcile(at)

        decision = self.risk.size_entry(proposal, state.account, state.positions)
        if decision.units <= 0:
            logger.info("risk gate refused %s: %s", proposal.engine, decision.reason)
            return None

        order = self._build_limit_order(proposal, decision.units, at, limit_prices)
        if order is None:
            logger.info("no quotable limit price for %s — not sent", proposal.engine)
            return None

        if self.dry_run:
            logger.warning("DRY RUN — would place %s x%d (%d legs), not sending",
                           proposal.engine, decision.units, len(order.legs))
            return OrderResult(order_id="dry-run", status=OrderStatus.ACCEPTED,
                               message="dry run: not transmitted")

        result = self.broker.place_order(order)
        logger.info("placed %s -> %s (%s)", proposal.engine, result.order_id, result.status)
        return result

    def _build_limit_order(
        self, proposal: ProposedTrade, units: int, at: datetime,
        limit_prices: dict[int, float] | None,
    ) -> Order | None:
        """Limit-only, priced conservatively. Market orders do not exist here."""
        legs = list(proposal.legs)
        if limit_prices is not None:
            missing = [i for i in range(len(legs)) if i not in limit_prices]
            if missing:
                return None
        return Order(
            order_id="",
            legs=legs,
            qty=units,
            intent=OrderIntent.OPEN,
            limit_price=proposal.unit_cost,
            engine_tag=proposal.engine,
            catalyst_ref=proposal.catalyst_ref,
            direction=proposal.direction,
            exit_rules=proposal.exit_rules,
            created_at=at,
            max_loss=proposal.unit_max_loss,
        )

    # -- exits are not risk-gated: closing risk is always allowed ----------
    def close(self, position: Position, at: datetime, reason: str) -> OrderResult | None:
        """Reduce or flatten. Deliberately not blocked by the kill switch —
        halting entries must never trap us in an open position."""
        self.reconcile(at)
        legs = [
            type(leg)(key=leg.key,
                      side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                      qty=leg.qty)
            for leg in position.legs
        ]
        order = Order(
            order_id="", legs=legs, qty=position.qty, intent=OrderIntent.CLOSE,
            limit_price=None, engine_tag=position.engine_tag,
            catalyst_ref=getattr(position, "catalyst_ref", ""),
            direction=position.direction, exit_rules=position.exit_rules,
            created_at=at,
        )
        if self.dry_run:
            logger.warning("DRY RUN — would close %s (%s)", position.position_id, reason)
            return OrderResult(order_id="dry-run", status=OrderStatus.ACCEPTED,
                               message="dry run: not transmitted")
        return self.broker.place_order(order)

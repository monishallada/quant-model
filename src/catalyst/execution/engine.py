"""The only road to a broker in paper and live mode.

Every paper/live order passes through ``ExecutionEngine``. The backtest
engines drive ``SimulatedBroker`` directly by design (they own the event loop
and construct orders from broker-truth positions); that split — and the claim
that nothing else touches a broker — is enforced by
``tests/execution/test_only_path_to_broker.py``, which fails if any new module
grows a ``place_order`` call site.

Four things happen on every entry, in this order, and none is skippable:

1. **Reconcile.** Read positions and account from the broker. Internal memory
   is never trusted — a fill that landed while we were computing, a manual
   close, or a restart all make in-process state a lie.
2. **Kill switch.** Checked on every entry. Engaged means no new entries.
3. **Risk.** The RiskManager sizes the trade against *reconciled* state. Zero
   units means the order is dropped; there is no override path.
4. **Limit discipline.** Orders are limit-only at the proposal's worst-case
   net price, refused when the quotes backing that price are stale
   (``max_quote_age``). A market order is never constructed here.

Exits are deliberately NOT risk-gated or kill-gated: halting entries must
never trap an open position (audit v15: this contract is now tested, not
asserted).
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
    OrderLeg,
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


class StaleQuoteError(Exception):
    """The quotes backing a limit price are older than max_quote_age."""


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
    #: refuse to price an order from quotes older than this (audit D-058: this
    #: guard was documented but unwired; it is now enforced in submit()).
    max_quote_age: timedelta = timedelta(seconds=30)
    dry_run: bool = False          # build and log the order; never send it
    #: orders placed but not FILLED at last sight, keyed by (engine, ref).
    #: Live fills are asynchronous: without this, two submits in one cycle
    #: could both pass the cash check against the same unspent cash (D-142).
    _pending: dict[tuple[str, str], str] = field(default_factory=dict)

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
               quote_time: datetime | None = None) -> OrderResult | None:
        """Size, gate and place one proposal. None means nothing was sent.

        ``quote_time`` is when the quotes behind ``proposal.unit_cost`` were
        read; older than ``max_quote_age`` refuses the order rather than
        pricing it from a stale book.
        """
        if self.kill.engaged():
            logger.critical("kill switch engaged (%s) — entry refused", self.kill.reason())
            return None

        if quote_time is not None and at - quote_time > self.max_quote_age:
            logger.warning("stale quotes (%.0fs old) — entry refused",
                           (at - quote_time).total_seconds())
            return None

        key = (proposal.engine, proposal.catalyst_ref)
        if key in self._pending:
            logger.warning("entry for %s already in flight (%s) — refused",
                           key, self._pending[key])
            return None

        state = self.reconcile(at)

        decision = self.risk.size_entry(proposal, state.account, state.positions)
        if decision.units <= 0:
            logger.info("risk gate refused %s: %s", proposal.engine, decision.reason)
            return None

        order = self._build_limit_order(proposal, decision.units)
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
        if result.status is OrderStatus.ACCEPTED:
            self._pending[key] = result.order_id
        elif result.status is OrderStatus.FILLED:
            self._pending.pop(key, None)
        return result

    def _build_limit_order(self, proposal: ProposedTrade, units: int) -> Order | None:
        """Limit-only, at the proposal's worst-case net price.

        Legs carry TOTAL contracts (per-unit ratio x sized units) — the Order
        model has no separate qty field (audit D-012/D-059: the old builder
        passed five fields that do not exist on the model and dropped the
        sizing decision entirely; construction is now covered by tests).
        """
        if units <= 0 or not proposal.legs:
            return None
        legs = [OrderLeg(key=leg.key, side=leg.side, qty=leg.qty * units)
                for leg in proposal.legs]
        return Order(
            legs=legs,
            intent=OrderIntent.OPEN,
            limit_price=proposal.unit_cost,
            tag=f"{proposal.engine}:{proposal.catalyst_ref}",
            direction=proposal.direction,
            exit_rules=proposal.exit_rules,
            max_loss=proposal.unit_max_loss,
        )

    # -- exits are not risk-gated: closing risk is always allowed ----------
    def close(self, position: Position, at: datetime, reason: str,
              close_units: int | None = None) -> OrderResult | None:
        """Reduce or flatten. Deliberately not blocked by the kill switch —
        halting entries must never trap us in an open position.

        The close is priced at the position's current mark as a limit and
        refused if the reconciled broker no longer knows the position
        (audit D-013: the old builder crashed on construction; it also never
        checked the position still existed).
        """
        state = self.reconcile(at)
        live = {p.position_id for p in state.positions}
        if position.position_id not in live:
            logger.warning("close(%s) refused: position not at broker (%s)",
                           position.position_id, reason)
            return None

        units = position.qty if close_units is None else min(close_units, position.qty)
        if units <= 0:
            return None
        legs = [
            OrderLeg(key=leg.key,
                     side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                     qty=leg.qty * units)
            for leg in position.legs
        ]
        order = Order(
            legs=legs,
            intent=OrderIntent.CLOSE,
            position_id=position.position_id,
            # Order-level sign convention (debit > 0, credit < 0): closing
            # a long structure is a SALE (credit), closing a short is a
            # buy-back (debit) — both are -current_value (audit D-106).
            limit_price=-position.current_value,
            tag=f"{position.engine_tag}:{position.catalyst_ref}",
            direction=position.direction,
            exit_rules=position.exit_rules,
        )
        if self.dry_run:
            logger.warning("DRY RUN — would close %s x%d (%s)",
                           position.position_id, units, reason)
            return OrderResult(order_id="dry-run", status=OrderStatus.ACCEPTED,
                               message="dry run: not transmitted")
        result = self.broker.place_order(order)
        logger.info("close %s x%d (%s) -> %s", position.position_id, units,
                    reason, result.status)
        return result

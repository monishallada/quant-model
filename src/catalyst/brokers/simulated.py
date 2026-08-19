"""SimulatedBroker: the backtest fill engine + cash/position ledger.

Fill model (config-driven, conservative by construction):
- price: mid moved ``spread_fill_fraction`` toward the worse side of the
  NBBO per leg (default 60% toward the ask on buys / toward the bid on sells)
- slippage: ``slippage_pct_of_premium`` applied per leg against the trader
- commission: ``per_contract`` for the active profile, per contract per leg

Orders fill immediately at the modeled price: the limit price is what the
execution layer *asked* for; the model already assumes a worse-than-mid fill,
which is the backtest's stand-in for limit-order reality. Positions are
broker-truth here exactly as they are live: strategy/risk/exit layers read
``get_positions()`` and never trust their own memory.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from functools import reduce
from typing import Any

from catalyst import costs
from catalyst.core.config import CommissionsConfig, FillModelConfig
from catalyst.core.interfaces import Broker
from catalyst.core.types import (
    EquityKey,
    AccountState,
    ExitRules,
    OptionChain,
    OptionRight,
    Order,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Position,
    PositionLeg,
    Side,
)

logger = logging.getLogger(__name__)


class SimulatedBroker(Broker):
    def __init__(
        self,
        fill_model: FillModelConfig,
        commissions: CommissionsConfig,
        starting_cash: float,
        cost_model: costs.CostModel | None = None,
    ) -> None:
        self._fill = fill_model
        self._commissions = commissions
        self._cost_model = cost_model or costs.build(fill_model, commissions)
        self.cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._chains: dict[str, OptionChain] = {}
        self._equity_quotes: dict[str, tuple[float, float]] = {}
        self._now: datetime | None = None
        self._order_seq = 0
        self.total_commissions = 0.0

    # ------------------------------------------------------------------
    # Backtester market-state hooks
    # ------------------------------------------------------------------

    def update_market(self, chains: dict[str, OptionChain], now: datetime,
                      equity_quotes: dict[str, tuple[float, float]] | None = None) -> None:
        """Advance simulated time: refresh marks and settle anything expired.

        ``equity_quotes`` maps symbol -> (bid, ask) for share instruments; the
        intraday loop supplies these per minute. Options-only callers omit it
        and behave exactly as before.
        """
        self._chains = chains
        if equity_quotes is not None:
            self._equity_quotes = equity_quotes
        self._now = now
        for pos in list(self._positions.values()):
            self._mark_position(pos)
        self._settle_expired(now)

    def _leg_mid(self, pos_leg: PositionLeg) -> float | None:
        if isinstance(pos_leg.key, EquityKey):
            q = self._equity_quotes.get(pos_leg.key.underlying)
            if q is None or q[0] <= 0 or q[1] <= 0:
                return None
            return (q[0] + q[1]) / 2.0
        chain = self._chains.get(pos_leg.key.underlying)
        if chain is None:
            return None
        contract = chain.find(pos_leg.key)
        if contract is None or contract.ask <= 0:
            return None
        return contract.mid

    def _mark_position(self, pos: Position) -> None:
        value = 0.0
        for leg in pos.legs:
            mid = self._leg_mid(leg)
            if mid is None:
                return  # keep last mark when a leg has no quote today
            value += mid * leg.qty * (1 if leg.side is Side.BUY else -1)
        pos.current_value = value
        pos.high_water_value = max(pos.high_water_value, value)

    def _settle_expired(self, now: datetime) -> None:
        """Cash-settle at intrinsic any structure held through expiry.

        The exit layer closes positions ``close_before_expiry_days`` ahead, so
        this is a safety net, not a normal path.
        """
        for pid, pos in list(self._positions.items()):
            if any(isinstance(leg.key, EquityKey) for leg in pos.legs):
                continue  # shares do not expire
            expiry = min(leg.key.expiry for leg in pos.legs)
            if expiry >= now.date():
                continue
            chain = self._chains.get(pos.underlying)
            spot = chain.underlying_price if chain else None
            value = 0.0
            if spot is not None:
                for leg in pos.legs:
                    if leg.key.right is OptionRight.CALL:
                        intrinsic = max(spot - leg.key.strike, 0.0)
                    else:
                        intrinsic = max(leg.key.strike - spot, 0.0)
                    value += intrinsic * leg.qty * (1 if leg.side is Side.BUY else -1)
            else:
                logger.warning("Settling %s at last mark; no spot for %s", pid, pos.underlying)
                value = pos.current_value
            self.cash += value * pos.qty * pos.multiplier
            pos.realized_pnl += (value - pos.entry_price) * pos.qty * pos.multiplier
            logger.warning("Position %s expired in simulation; settled at intrinsic %.2f", pid, value)
            del self._positions[pid]

    def force_close(self, position_id: str, value_per_unit: float) -> None:
        """Settle a position at an externally determined per-unit value.

        Exists ONLY for the intraday engine's flagged synthetic-fill path
        (mandatory EOD flatten with no live quote). Callers must count every
        use — an unflagged force_close is a fabricated fill.
        """
        pos = self._positions.pop(position_id, None)
        if pos is None:
            return
        self.cash += value_per_unit * pos.qty * pos.multiplier
        pos.realized_pnl += (value_per_unit - pos.entry_price) * pos.qty * pos.multiplier

    # ------------------------------------------------------------------
    # Fill modeling
    # ------------------------------------------------------------------

    def _leg_fill_price(self, order: Order) -> dict[int, float] | None:
        """Per-leg modeled fill prices (per share), or None if unquotable.

        Delegates to ``costs.CostModel`` — the broker owns the ledger, not the
        cost math. That separation is what lets the zero-cost diagnostic run
        the identical path with frictions switched off.
        """
        prices: dict[int, float] = {}
        for i, leg in enumerate(order.legs):
            if isinstance(leg.key, EquityKey):
                q = self._equity_quotes.get(leg.key.underlying)
                if q is None or q[0] <= 0 or q[1] <= 0:
                    return None
                prices[i] = self._cost_model.equity_fill(q[0], q[1], leg.side, leg.qty).price
                continue
            chain = self._chains.get(leg.key.underlying)
            contract = chain.find(leg.key) if chain else None
            if contract is None or contract.ask <= 0 or contract.bid < 0:
                return None
            prices[i] = self._cost_model.leg_fill(contract, leg.side, leg.qty).price
        return prices

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        if self._now is None:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message="No market data loaded")
        self._order_seq += 1
        order_id = f"sim-{self._order_seq}"

        leg_prices = self._leg_fill_price(order)
        if leg_prices is None:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message="Leg unquotable in current snapshot")

        # Per-contract commission applies to OPTION contracts only; US equity
        # routing is commission-free at both configured brokers. Without this
        # branch a 500-share order would be charged 500 x $0.65.
        total_contracts = sum(leg.qty for leg in order.legs
                              if not isinstance(leg.key, EquityKey))
        commission = self._commissions.per_contract * total_contracts
        # Signed cash flow: buys pay, sells receive.
        cash_flow = 0.0
        for i, leg in enumerate(order.legs):
            sign = -1.0 if leg.side is Side.BUY else 1.0
            cash_flow += sign * leg_prices[i] * leg.qty * leg.key.multiplier
        cash_flow -= commission

        if order.intent is OrderIntent.OPEN:
            if self.cash + cash_flow < 0:
                return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                                   message="Insufficient cash (broker-level guard)")
            units = reduce(math.gcd, (leg.qty for leg in order.legs))
            unit_net = -cash_flow_ex_commission(cash_flow, commission) / (units * order.multiplier)
            pos = Position(
                position_id=f"pos-{order_id}",
                legs=[
                    PositionLeg(key=leg.key, side=leg.side, qty=leg.qty // units)
                    for leg in order.legs
                ],
                qty=units,
                entry_price=unit_net,
                entry_time=self._now,
                engine_tag=order.tag.split(":", 1)[0] if order.tag else "",
                direction=order.direction,
                catalyst_ref=order.tag.split(":", 1)[1] if ":" in order.tag else "",
                exit_rules=order.exit_rules or ExitRules(),
                current_value=unit_net,
                high_water_value=unit_net,
                max_loss=order.max_loss,
            )
            self._positions[pos.position_id] = pos
            self.cash += cash_flow
            self.total_commissions += commission
            self._mark_position(pos)
            return OrderResult(
                order_id=order_id, status=OrderStatus.FILLED, filled_qty=units,
                avg_fill_price=unit_net, commission=commission,
            )

        # CLOSE
        pos = self._positions.get(order.position_id or "")
        if pos is None:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=f"Unknown position {order.position_id}")
        units = reduce(math.gcd, (leg.qty for leg in order.legs))
        if units > pos.qty:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=f"Close qty {units} exceeds open {pos.qty}")
        credit_per_unit = cash_flow_ex_commission(cash_flow, commission) / (units * order.multiplier)
        pos.realized_pnl += (credit_per_unit - pos.entry_price) * units * pos.multiplier
        pos.qty -= units
        self.cash += cash_flow
        self.total_commissions += commission
        if pos.qty == 0:
            del self._positions[pos.position_id]
        return OrderResult(
            order_id=order_id, status=OrderStatus.FILLED, filled_qty=units,
            avg_fill_price=credit_per_unit, commission=commission,
        )

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult:
        # Fills are immediate in simulation; nothing is ever resting.
        return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                           message="Simulated orders fill immediately")

    def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                           message="Simulated orders fill immediately")

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> AccountState:
        pos_value = sum(p.current_value * p.qty * p.multiplier for p in self._positions.values())
        equity = self.cash + pos_value
        return AccountState(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,
            timestamp=self._now or datetime.min,
        )


def cash_flow_ex_commission(cash_flow: float, commission: float) -> float:
    """The pure premium cash flow (commission removed) for per-unit pricing."""
    return cash_flow + commission

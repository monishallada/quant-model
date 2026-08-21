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
    OptionKey,
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
        self._equity_fill_quotes: dict[str, tuple[float, float]] | None = None
        self._equity_quotes: dict[str, tuple[float, float]] = {}
        # underlying -> {date: spot}, last few sessions. A single last-spot
        # tuple was overwritten by the NEXT session's chain in update_market
        # BEFORE _settle_expired ran, so the expiry-day guard could never
        # match precisely when a chain was fetched (verifier catch).
        self._spot_by_date: dict[str, dict] = {}
        #: settlement telemetry: (position_id, per-unit value, realized pnl $)
        #: — the trade-record layer reconciles against THIS, not against a
        #: guess (audit MEDIUM: deep-ITM settles were recorded as -$1.30).
        self.settlements: list[tuple[str, float, float]] = []
        #: partial-expiry telemetry: (position_id, per-unit cash settled) for
        #: structures where only SOME legs expired (audit D-006). The record
        #: layer folds these into exit cash; the position lives on.
        self.partial_settlements: list[tuple[str, float]] = []
        #: synthetic force_close telemetry (intraday flatten path, D-199)
        self.synthetic_closes: list[tuple[str, float, float]] = []
        #: collateral reserved against credit structures, per position (D-041)
        self._reserved: dict[str, float] = {}
        #: marks kept stale because a leg had no quote (D-198)
        self.stale_marks = 0
        self._now: datetime | None = None
        self._order_seq = 0
        self.total_commissions = 0.0

    # ------------------------------------------------------------------
    # Backtester market-state hooks
    # ------------------------------------------------------------------

    def update_market(self, chains: dict[str, OptionChain], now: datetime,
                      equity_quotes: dict[str, tuple[float, float]] | None = None,
                      equity_fill_quotes: dict[str, tuple[float, float]] | None = None) -> None:
        """Advance simulated time: refresh marks and settle anything expired.

        ``equity_quotes`` maps symbol -> (bid, ask) for share instruments; the
        intraday loop supplies these per minute. Options-only callers omit it
        and behave exactly as before.
        """
        self._chains = chains
        if equity_quotes is not None:
            self._equity_quotes = equity_quotes
        # Fills may use ONLY the current-bar book (audit D-023): when provided,
        # a symbol missing from it is UNQUOTABLE this minute even if an older
        # mark exists. Omitting the param PRESERVES the current book (mid-tick
        # refreshes must not silently revert fills to mark quotes).
        if equity_fill_quotes is not None:
            self._equity_fill_quotes = equity_fill_quotes
        for u, ch in chains.items():
            if ch.underlying_price and ch.underlying_price > 0:
                hist = self._spot_by_date.setdefault(u, {})
                hist[now.date()] = ch.underlying_price
                for d in sorted(hist)[:-5]:      # keep a short window
                    del hist[d]
        self._now = now
        for pos in list(self._positions.values()):
            self._mark_position(pos)
        self._settle_expired(now)

    def _leg_mid(self, pos_leg: PositionLeg) -> float | None:
        # NaN guards are explicit on BOTH branches: NaN <= 0 is False, so NaN
        # quotes previously marked positions and filled equity legs (D-037/39).
        if isinstance(pos_leg.key, EquityKey):
            q = self._equity_quotes.get(pos_leg.key.underlying)
            if (q is None or q[0] != q[0] or q[1] != q[1]
                    or q[0] <= 0 or q[1] <= 0):
                return None
            return (q[0] + q[1]) / 2.0
        chain = self._chains.get(pos_leg.key.underlying)
        if chain is None:
            return None
        contract = chain.find(pos_leg.key)
        if (contract is None or contract.bid != contract.bid
                or contract.ask != contract.ask or contract.ask <= 0):
            return None
        return contract.mid

    def _mark_position(self, pos: Position) -> None:
        value = 0.0
        for leg in pos.legs:
            mid = self._leg_mid(leg)
            if mid is None:
                # keep last mark, but COUNT it — invisible staleness let stale
                # marks pass as fresh in the daily engine (audit D-198)
                self.stale_marks += 1
                logger.debug("stale mark kept for %s (no quote on %s)",
                             pos.position_id, leg.key)
                return
            value += mid * leg.qty * (1 if leg.side is Side.BUY else -1)
        pos.current_value = value
        pos.high_water_value = max(pos.high_water_value, value)

    def _settle_expired(self, now: datetime) -> None:
        """Cash-settle at intrinsic any LEG held through its expiry.

        Only expired legs settle (audit D-006: min-expiry settlement liquidated
        live back-month legs of calendars at intrinsic, deleting their time
        value). Survivors stay open as a reduced structure; their proceeds are
        reported via partial_settlements so trade records stay exact.
        """
        for pid, pos in list(self._positions.items()):
            option_legs = [leg for leg in pos.legs
                           if isinstance(leg.key, OptionKey)]
            expired = [leg for leg in option_legs if leg.key.expiry < now.date()]
            if not expired:
                continue

            def _spot_for(expiry, underlying):
                chain = self._chains.get(underlying)
                spot = chain.underlying_price if chain else None
                on_expiry = self._spot_by_date.get(underlying, {}).get(expiry)
                if on_expiry and on_expiry > 0:
                    spot = on_expiry
                # 0.0 and NaN are NOT spots (audit D-038/D-111): a zero spot
                # settled every put at full strike.
                if spot is None or spot != spot or spot <= 0:
                    return None
                return spot

            def _leg_intrinsic(leg):
                if isinstance(leg.key, EquityKey):
                    return None                      # shares never expire
                spot = _spot_for(leg.key.expiry, leg.key.underlying)
                if spot is None:
                    return None
                if leg.key.right is OptionRight.CALL:
                    intrinsic = max(spot - leg.key.strike, 0.0)
                else:
                    intrinsic = max(leg.key.strike - spot, 0.0)
                return intrinsic * leg.qty * (1 if leg.side is Side.BUY else -1)

            if len(expired) == len(pos.legs):
                # whole structure expired: settle and delete (original path)
                values = [_leg_intrinsic(leg) for leg in expired]
                if any(v is None for v in values):
                    logger.warning("Settling %s at last mark; no valid spot for %s",
                                   pid, pos.underlying)
                    value = pos.current_value
                else:
                    value = sum(values)
                self.cash += value * pos.qty * pos.multiplier
                settle_pnl = (value - pos.entry_price) * pos.qty * pos.multiplier
                pos.realized_pnl += settle_pnl
                self.settlements.append((pid, value, settle_pnl))
                self._release_reserve(pid, pos.qty, pos.qty)
                logger.warning("Position %s expired in simulation; settled at "
                               "intrinsic %.2f", pid, value)
                del self._positions[pid]
                continue

            # PARTIAL: settle only the expired legs; the reduced structure
            # lives on with the original entry basis (total P&L exact: the
            # record layer folds this cash into exit proceeds).
            values = [_leg_intrinsic(leg) for leg in expired]
            if any(v is None for v in values):
                logger.error("Cannot settle expired legs of %s (no valid spot); "
                             "retrying next session", pid)
                continue
            per_unit = sum(values)
            self.cash += per_unit * pos.qty * pos.multiplier
            pos.realized_pnl += per_unit * pos.qty * pos.multiplier
            self.partial_settlements.append((pid, per_unit))
            pos.legs = [leg for leg in pos.legs if leg not in expired]
            self._mark_position(pos)
            logger.warning("Front leg(s) of %s expired; settled %.2f/unit at "
                           "intrinsic, %d leg(s) remain live", pid, per_unit,
                           len(pos.legs))

    def _release_reserve(self, pid: str, units: int, total_units: int) -> None:
        """Free collateral proportionally as a credit structure closes."""
        held = self._reserved.get(pid)
        if held is None:
            return
        if units >= total_units:
            self._reserved.pop(pid, None)
        else:
            self._reserved[pid] = held * (1 - units / total_units)

    @property
    def reserved_collateral(self) -> float:
        return sum(self._reserved.values())

    def force_close(self, position_id: str, value_per_unit: float) -> None:
        """Settle a position at an externally determined per-unit value.

        Exists ONLY for the intraday engine's flagged synthetic-fill path
        (mandatory EOD flatten with no live quote). Callers must count every
        use — an unflagged force_close is a fabricated fill.
        """
        pos = self._positions.pop(position_id, None)
        if pos is None:
            return
        # Synthetic flattens pay commissions like every other close and are
        # counted in their own telemetry (audit D-199: they were free and
        # invisible).
        contracts = sum(leg.qty for leg in pos.legs
                        if not isinstance(leg.key, EquityKey)) * pos.qty
        commission = self._commissions.per_contract * contracts
        self.cash += value_per_unit * pos.qty * pos.multiplier - commission
        self.total_commissions += commission
        pnl = (value_per_unit - pos.entry_price) * pos.qty * pos.multiplier - commission
        pos.realized_pnl += pnl
        self.synthetic_closes.append((position_id, value_per_unit, pnl))
        self._release_reserve(position_id, pos.qty, pos.qty)

    # ------------------------------------------------------------------
    # Fill modeling
    # ------------------------------------------------------------------

    def _leg_fills(self, order: Order) -> list["costs.Fill"] | None:
        """Per-leg modeled fills (price + commission), or None if unquotable.

        Delegates to ``costs.CostModel`` — the broker owns the ledger, not the
        cost math (audit D-112: the broker used to RECOMPUTE commissions,
        a second copy that could drift from the cost truth). A crossed book or
        an NBBO-assertion failure inside the model makes the leg unquotable
        and REJECTS the order rather than crashing the session (D-043/D-044).
        """
        fills: list[costs.Fill] = []
        fill_book = (self._equity_fill_quotes
                     if self._equity_fill_quotes is not None
                     else self._equity_quotes)
        for leg in order.legs:
            if isinstance(leg.key, EquityKey):
                q = fill_book.get(leg.key.underlying)
                if (q is None or q[0] != q[0] or q[1] != q[1]
                        or q[0] <= 0 or q[1] <= 0):
                    return None
                try:
                    fills.append(self._cost_model.equity_fill(q[0], q[1],
                                                              leg.side, leg.qty))
                except (ValueError, costs.BetterThanNBBOError):
                    return None
                continue
            chain = self._chains.get(leg.key.underlying)
            contract = chain.find(leg.key) if chain else None
            # NaN guard is explicit: NaN <= 0 is False, so NaN quotes sailed
            # through and "filled" (audit MEDIUM).
            if (contract is None or contract.ask != contract.ask
                    or contract.bid != contract.bid
                    or contract.ask <= 0 or contract.bid < 0):
                return None
            # A zero-bid option has NO buyer: selling it for model-priced
            # credit fabricates money (audit D-040).
            if leg.side is Side.SELL and contract.bid <= 0:
                return None
            try:
                fills.append(self._cost_model.leg_fill(contract, leg.side, leg.qty))
            except (ValueError, costs.BetterThanNBBOError):
                return None
        return fills

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        if self._now is None:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message="No market data loaded")
        self._order_seq += 1
        order_id = f"sim-{self._order_seq}"

        fills = self._leg_fills(order)
        if fills is None:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message="Leg unquotable in current snapshot")

        # ONE commission source: the cost model's per-leg Fill.commission
        # (equity legs are $0 there; audit D-112 removed the broker's own
        # recomputation, which was a second copy that could drift).
        commission = sum(f.commission for f in fills)
        # Signed cash flow: buys pay, sells receive.
        cash_flow = 0.0
        for leg, fill in zip(order.legs, fills):
            sign = -1.0 if leg.side is Side.BUY else 1.0
            cash_flow += sign * fill.price * leg.qty * leg.key.multiplier
        cash_flow -= commission

        if order.intent is OrderIntent.OPEN:
            units_pre = reduce(math.gcd, (leg.qty for leg in order.legs))
            # Credit structures must post collateral: a net-credit open RAISES
            # cash, so without a reserve the cash guard mathematically cannot
            # reject and credit selling is unbounded (audit D-041/D-113).
            new_reserve = 0.0
            if cash_flow > 0:
                if order.max_loss is None or order.max_loss != order.max_loss:
                    return OrderResult(
                        order_id=order_id, status=OrderStatus.REJECTED,
                        message="credit open refused: no max_loss to collateralize")
                new_reserve = order.max_loss * units_pre * order.multiplier
            if self.cash + cash_flow - self.reserved_collateral - new_reserve < 0:
                return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                                   message="Insufficient cash/collateral "
                                           "(broker-level guard)")
            # Unit semantics: units = gcd of leg quantities; legs are stored as
            # per-unit ratios. An unreduced proposal (2x/4x sized to 3) is thus
            # recorded as 6 units of a (1x/2x) structure at half the per-unit
            # price — internally consistent (cash, marks, ratio exits all use
            # the same basis); TradeRecord qty/prices are in THESE units.
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
            if new_reserve > 0:
                self._reserved[pos.position_id] = new_reserve
            self.cash += cash_flow
            self.total_commissions += commission
            self._mark_position(pos)
            # AUDIT LOW: high-water was seeded with the FILL (mid + crossing +
            # slippage) while marks are mid-based, so the first mark sat ~cost
            # drag below high water and tight trails fired instantly in a flat
            # market. Anchor the trail at the first mark instead.
            if pos.current_value == pos.current_value:  # not NaN
                pos.high_water_value = pos.current_value
            return OrderResult(
                order_id=order_id, status=OrderStatus.FILLED, filled_qty=units,
                avg_fill_price=unit_net, commission=commission,
                position_id=pos.position_id,
            )

        # CLOSE
        maybe_pos = self._positions.get(order.position_id or "")
        if maybe_pos is None:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=f"Unknown position {order.position_id}")
        pos = maybe_pos
        units = reduce(math.gcd, (leg.qty for leg in order.legs))
        if units > pos.qty:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=f"Close qty {units} exceeds open {pos.qty}")
        # STRICT close matching (audit D-007/D-114): every closing leg must
        # mirror a position leg — same key, INVERTED side, exact ratio. A
        # wrong-strike, same-side, or partial-leg "close" corrupted cash and
        # realized P&L silently while reporting FILLED.
        want = {(leg.key, Side.SELL if leg.side is Side.BUY else Side.BUY): leg.qty
                for leg in pos.legs}
        got = {(leg.key, leg.side): leg.qty for leg in order.legs}
        if set(got) != set(want) or any(got[k] != want[k] * units for k in want):
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                message="close legs do not mirror the open position "
                        f"{pos.position_id} (keys/sides/ratios must match)")
        credit_per_unit = cash_flow_ex_commission(cash_flow, commission) / (units * order.multiplier)
        total_before = pos.qty
        pos.realized_pnl += (credit_per_unit - pos.entry_price) * units * pos.multiplier
        pos.qty -= units
        self.cash += cash_flow
        self.total_commissions += commission
        self._release_reserve(pos.position_id, units, total_before)
        if pos.qty == 0:
            del self._positions[pos.position_id]
        return OrderResult(
            order_id=order_id, status=OrderStatus.FILLED, filled_qty=units,
            avg_fill_price=credit_per_unit, commission=commission,
            position_id=order.position_id,
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
            # collateral held against credit structures is not spendable (D-041)
            buying_power=self.cash - self.reserved_collateral,
            timestamp=self._now or datetime.min,
        )


def cash_flow_ex_commission(cash_flow: float, commission: float) -> float:
    """The pure premium cash flow (commission removed) for per-unit pricing."""
    return cash_flow + commission

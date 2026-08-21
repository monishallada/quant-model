"""A live/paper trading session — the loop that did not exist.

Audit D-066/D-067: paper mode connected, granted ``paper_tested``, printed
"Session live" and exited the process; the strategy was loaded and never
invoked. This module is the actual loop, built from the SAME components the
backtester validates: proposals from the strategy, sizing by the RiskManager,
orders through ExecutionEngine, exits by the mechanical ``evaluate_exits``
interpreter. There is no live-only trading logic anywhere in it.

The session tracks its own positions (broker rows are leg-level reconciliation
truth and carry no ExitRules — audit D-109), reconciles them against the
broker every cycle, and counts entry/exit round trips. ``paper_tested`` is
granted from THESE counts, never from connecting (audit D-016).
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Callable
from datetime import UTC, datetime

from catalyst.core.interfaces import DataSource, DirectionalSignal, Strategy
from catalyst.core.types import Catalyst, OrderStatus, Position, PositionLeg
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.execution.engine import ExecutionEngine
from catalyst.exits.manager import evaluate_exits

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    cycles: int = 0
    proposals: int = 0
    entries_filled: int = 0
    exits_filled: int = 0
    orders_rejected: int = 0
    round_trips: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class TradingSession:
    """One bounded trading session (paper or live — the wiring decides)."""

    strategy: Strategy
    signal: DirectionalSignal
    execution: ExecutionEngine
    data: DataSource
    catalysts: list[Catalyst]
    #: seconds between cycles; tests pass 0
    interval_seconds: float = 60.0
    #: hard bound on cycles so a session always terminates
    max_cycles: int = 390
    #: injectable clock for tests
    now_fn: Callable[[], datetime] | None = None

    def _now(self) -> datetime:
        return self.now_fn() if self.now_fn else datetime.now(UTC)

    # ------------------------------------------------------------------
    def run(self) -> SessionStats:
        stats = SessionStats()
        open_positions: dict[str, Position] = {}

        for _cycle in range(self.max_cycles):
            stats.cycles += 1
            at = self._now()
            try:
                self._one_cycle(at, open_positions, stats)
            except Exception as e:                      # noqa: BLE001
                # A cycle error must not kill the session with positions open,
                # but it must never be silent either.
                logger.exception("session cycle error")
                stats.errors.append(str(e)[:200])
            if self.execution.kill.engaged() and not open_positions:
                logger.warning("kill switch engaged and book flat — session ends")
                break
            if self.interval_seconds:
                _time.sleep(self.interval_seconds)
        return stats

    # ------------------------------------------------------------------
    def _one_cycle(self, at: datetime, open_positions: dict[str, Position],
                   stats: SessionStats) -> None:
        state = self.execution.reconcile(at)
        live_ids = {p.position_id for p in state.positions}
        # equity marks feed the breakers (audit D-065: the live RiskManager
        # never received a single mark, so breakers could not fire)
        if hasattr(self.execution.risk, "record_mark"):
            self.execution.risk.record_mark(at.date(), state.account.equity)

        # --- exits first, always ---------------------------------------
        broker_marks = {p.position_id: p for p in state.positions}
        for pid, pos in list(open_positions.items()):
            if pid not in live_ids:
                # closed/expired at the broker outside our loop
                open_positions.pop(pid)
                continue
            # refresh mark from broker truth before evaluating rules
            pos.current_value = self._structure_mark(pos, broker_marks)
            for action in evaluate_exits(pos, at.date()):
                result = self.execution.close(pos, at, action.reason)
                if result is not None and result.status is OrderStatus.FILLED:
                    stats.exits_filled += 1
                    stats.round_trips += 1
                    open_positions.pop(pid, None)
                elif result is not None:
                    stats.orders_rejected += 1
                break

        # --- entries ----------------------------------------------------
        for catalyst in self._due_catalysts(at):
            proposal = self._evaluate(catalyst, at)
            if proposal is None:
                continue
            stats.proposals += 1
            result = self.execution.submit(proposal, at, quote_time=at)
            if result is None:
                continue
            if result.status is OrderStatus.FILLED:
                stats.entries_filled += 1
                pos = self._position_from(proposal, result, at)
                open_positions[pos.position_id] = pos
            elif result.status is OrderStatus.REJECTED:
                stats.orders_rejected += 1

    # ------------------------------------------------------------------
    def _due_catalysts(self, at: datetime) -> list[Catalyst]:
        today = at.date()
        out = []
        for c in self.catalysts:
            try:
                reaction = resolve_reaction_session(c)
            except Exception:                           # noqa: BLE001
                continue
            if abs((reaction - today).days) <= 5:
                out.append(c)
        return out

    def _evaluate(self, catalyst: Catalyst, at: datetime):
        """Chain snapshot -> strategy.evaluate, mirroring the backtester."""
        expiries = None
        if hasattr(self.strategy, "catalyst_expiries"):
            try:
                available = self.data.list_expirations(catalyst.symbol)
                expiries = self.strategy.catalyst_expiries(
                    catalyst, available, at.date())
            except Exception as e:                      # noqa: BLE001
                logger.warning("expiry selection failed for %s: %s",
                               catalyst.ref, e)
                return None
        if not expiries:
            return None
        try:
            chain = self.data.get_chain(catalyst.symbol, at, expiries=expiries)
        except Exception as e:                          # noqa: BLE001
            logger.warning("chain fetch failed for %s: %s", catalyst.ref, e)
            return None
        try:
            history = self.data.get_history(catalyst.symbol,
                                            at.date().replace(year=at.year - 1),
                                            at.date())
        except Exception:                               # noqa: BLE001
            import pandas as pd
            history = pd.DataFrame()
        sig = self.signal.evaluate(catalyst.symbol, history, chain)
        evaluate = getattr(self.strategy, "evaluate", None)
        if evaluate is None:
            logger.warning("strategy %s has no evaluate(); catalyst skipped",
                           getattr(self.strategy, "name", "?"))
            return None
        return evaluate(catalyst, chain, sig, at)

    @staticmethod
    def _position_from(proposal, result, at: datetime) -> Position:
        return Position(
            position_id=f"pos-{result.order_id}",
            legs=[PositionLeg(key=leg.key, side=leg.side, qty=leg.qty)
                  for leg in proposal.legs],
            qty=result.filled_qty,
            entry_price=(result.avg_fill_price
                         if result.avg_fill_price is not None
                         else proposal.unit_cost),
            entry_time=at,
            engine_tag=proposal.engine,
            catalyst_ref=proposal.catalyst_ref,
            direction=proposal.direction,
            exit_rules=proposal.exit_rules,
            current_value=(result.avg_fill_price
                           if result.avg_fill_price is not None
                           else proposal.unit_cost),
        )

    @staticmethod
    def _structure_mark(pos: Position, broker_marks: dict) -> float:
        """Net per-unit mark of our structure from broker leg rows.

        Broker positions are leg-level (one row per contract); our structure's
        mark is the signed sum of matching leg marks. A missing leg keeps the
        last mark (counted upstream by the simulated broker; live brokers
        always quote open positions).
        """
        row = broker_marks.get(pos.position_id)
        if row is not None:          # SimulatedBroker: structure-level row
            return row.current_value
        total, found = 0.0, 0
        for leg in pos.legs:
            for pid, bp in broker_marks.items():
                if bp.legs and bp.legs[0].key == leg.key:
                    sign = 1 if leg.side.value == "buy" else -1
                    total += sign * bp.current_value * leg.qty
                    found += 1
                    break
        return total if found == len(pos.legs) else pos.current_value

"""Event-driven daily backtester.

One pass per trading session at the configured snapshot time:

1. pull chain slices (open-position expiries + engine-required expiries only)
2. ``broker.update_market`` — refresh marks, settle anything expired
3. exits first (mechanical ExitRules interpreter)
4. entries: catalyst × engine → proposal → entry gate (sizing/risk) → order
5. record equity

The entry gate is a protocol: M1 uses plain fixed-fractional sizing, M3 swaps
in the authoritative RiskManager without touching this loop.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Sequence

import pandas as pd

from catalyst.core.config import Config
from catalyst.core.interfaces import (Cadence, DataSource, DirectionalSignal,
                                     Opportunity, Strategy, StrategyContext)
from catalyst.core.types import (
    BacktestResult,
    Catalyst,
    Direction,
    OptionChain,
    Order,
    OrderIntent,
    OrderLeg,
    OrderStatus,
    Position,
    ProposedTrade,
    Side,
    TradeRecord,
)
from catalyst.core.tradingcal import sessions_in_range
from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.data.thetadata_historical import DataUnavailableError
from catalyst.exits.manager import evaluate_exits
from catalyst.risk.gate import EntryGate

logger = logging.getLogger(__name__)

# How many days around "today" a catalyst is considered in play for engines.
# Wide on purpose: each engine applies its own precise trigger window.
_CATALYST_LOOKAHEAD_DAYS = 21
_CATALYST_LOOKBACK_DAYS = 5


@dataclass
class _OpenTradeState:
    engine: str
    catalyst_ref: str
    underlying: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    max_qty: int
    commissions: float = 0.0
    exit_cash_weighted: float = 0.0  # Σ credit_per_unit × qty_closed
    multiplier: float = 100.0        # instrument dollars-per-point (audit LOW)
    qty_closed: int = 0
    exit_reasons: list[str] = field(default_factory=list)
    last_exit_time: datetime | None = None


class Backtester:
    def __init__(
        self,
        cfg: Config,
        data: DataSource,
        strategies: Sequence[Strategy],
        signal: DirectionalSignal,
        catalysts: Sequence[Catalyst],
        gate: EntryGate,
        hedger: object | None = None,  # HedgeManager (M3); None = no hedge sleeve
        screener: object | None = None,  # CatalystScreener; gates PRE-event entries
        catalyst_lookback_days: int = _CATALYST_LOOKBACK_DAYS,
        label: str = "backtest",
    ) -> None:
        self._cfg = cfg
        self._data = data
        self._strategies = list(strategies)
        self._signal = signal
        self._catalysts = list(catalysts)
        self._gate = gate
        self._hedger = hedger
        self._screener = screener
        self._lookback_days = catalyst_lookback_days
        self._label = label
        hh, mm = cfg.data.snapshot_time.split(":")
        self._snap = time(int(hh), int(mm))

    # ------------------------------------------------------------------

    def run(self, start: date, end: date) -> BacktestResult:
        cfg = self._cfg
        broker = SimulatedBroker(
            fill_model=cfg.execution.fill_model,
            commissions=cfg.execution.commissions,
            starting_cash=cfg.account.starting_capital,
        )
        sessions = sessions_in_range(start, end)
        if not sessions:
            raise ValueError(f"No trading sessions in {start}..{end}")

        # Underlying history once per symbol (signals slice it lookahead-free).
        symbols = sorted({c.symbol for c in self._catalysts})
        history: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                history[sym] = self._data.get_history(
                    sym, self._lookback_start(start), end
                )
            except DataUnavailableError:
                logger.warning("No history for %s; its signals will be neutral", sym)

        catalysts_by_symbol: dict[str, list[Catalyst]] = defaultdict(list)
        for c in self._catalysts:
            catalysts_by_symbol[c.symbol].append(c)

        if hasattr(self._gate, "set_history"):
            self._gate.set_history(history)

        open_state: dict[str, _OpenTradeState] = {}
        trades: list[TradeRecord] = []
        equity_curve: dict[date, float] = {}

        for i, session in enumerate(sessions):
            if i % 100 == 0:
                logger.info(
                    "progress %s: session %d/%d (%s)", self._label, i + 1, len(sessions), session
                )
            at = datetime.combine(session, self._snap)
            chains = self._pull_chains(session, at, broker, catalysts_by_symbol)
            broker.update_market(chains, at)
            self._sweep_settled(broker, open_state, trades, at)

            self._process_exits(broker, chains, open_state, trades, at)
            self._process_hedge(broker, chains, open_state, at)
            self._process_entries(
                broker, chains, history, catalysts_by_symbol, open_state, trades, at
            )
            self._process_scheduled_entries(
                broker, chains, history, open_state, trades, at
            )

            equity = broker.get_account().equity
            equity_curve[session] = equity
            if hasattr(self._gate, "record_mark"):
                self._gate.record_mark(session, equity)

        # Liquidate remaining marks into trade records for reporting.
        self._finalize_open(broker, open_state, trades, datetime.combine(sessions[-1], self._snap))

        return self._build_result(broker, trades, equity_curve, start, end)

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------

    def _lookback_start(self, start: date) -> date:
        # Enough history for slow signal lookbacks + IV-rank style windows.
        return date(start.year - 2, start.month, 1)

    def _catalysts_in_play(self, symbol_catalysts: list[Catalyst], session: date) -> list[Catalyst]:
        out = []
        for c in symbol_catalysts:
            delta_days = (c.when.date() - session).days
            if -self._lookback_days <= delta_days <= _CATALYST_LOOKAHEAD_DAYS:
                out.append(c)
        return out

    @staticmethod
    def _pit_history(history: dict, session) -> dict:
        """Point-in-time view: bars strictly BEFORE the session.

        AUDIT HIGH: the full backtest-window frames were handed to every
        StrategyContext — ctx.history[sym].iloc[-1] was the END of the
        backtest, a straight lookahead for any strategy reading history
        (the signal path was sliced; the context path was not).
        """
        import pandas as _pd
        cutoff = _pd.Timestamp(session)
        return {k: v[v.index < cutoff] for k, v in history.items()}

    def _pull_chains(
        self,
        session: date,
        at: datetime,
        broker: SimulatedBroker,
        catalysts_by_symbol: dict[str, list[Catalyst]],
    ) -> dict[str, OptionChain]:
        """Fetch only the chain slices needed today: expiries of open positions
        plus expiries any engine declares for an in-play catalyst."""
        needed: dict[str, set[date]] = defaultdict(set)

        for pos in broker.get_positions():
            for leg in pos.legs:
                if leg.key.expiry >= session:
                    needed[pos.underlying].add(leg.key.expiry)

        # Non-catalyst strategies (SCHEDULED / DAILY) enumerate their own work,
        # so the pipeline asks them what to fetch. opportunities() is deliberately
        # chain-free: it answers "what would you look at today", which must not
        # require paying for a chain to find out.
        for strat in self._strategies:
            if strat.cadence is Cadence.CATALYST:
                continue
            probe = StrategyContext(as_of=at, data=self._data)
            for opp in strat.opportunities(session, probe):
                for sym in opp.symbols:
                    available = [e for e in self._data.list_expirations(sym) if e > session]
                    req = strat.required_expiries(opp, available, session)
                    if req:
                        needed[sym].update(req)
                    elif available:
                        needed[sym].add(available[0])

        for sym, cat_list in catalysts_by_symbol.items():
            in_play = self._catalysts_in_play(cat_list, session)
            if not in_play:
                continue
            available = [e for e in self._data.list_expirations(sym) if e > session]
            for catalyst in in_play:
                # The screener measures implied move on the first expiry that
                # captures the event — make sure it's in the snapshot.
                if self._screener is not None and catalyst.when.date() > session:
                    event_expiries = [
                        e for e in available if e >= resolve_reaction_session(catalyst)
                    ]
                    if event_expiries and any(
                        strat.required_expiries(
                            Opportunity(session=session, symbols=(sym,), catalyst=catalyst),
                            available, session)
                        for strat in self._strategies
                    ):
                        needed[sym].add(event_expiries[0])
                for strat in self._strategies:
                    req = strat.required_expiries(
                            Opportunity(session=session, symbols=(sym,), catalyst=catalyst),
                            available, session)
                    if req is None:
                        horizon_needed = [
                            e for e in available
                            if (e - session).days <= self._cfg.data.chain_max_dte
                        ]
                        needed[sym].update(horizon_needed)
                    else:
                        needed[sym].update(req)

        if self._hedger is not None and self._hedger.wants_check(  # type: ignore[attr-defined]
            session, broker.get_positions()
        ):
            hedge_sym = self._hedger.symbol  # type: ignore[attr-defined]
            available = [e for e in self._data.list_expirations(hedge_sym) if e > session]
            needed[hedge_sym].update(
                self._hedger.required_expiries(available, session)  # type: ignore[attr-defined]
            )

        chains: dict[str, OptionChain] = {}
        for sym, expiries in needed.items():
            if not expiries:
                continue
            try:
                chains[sym] = self._data.get_chain(sym, at, expiries=sorted(expiries))
            except DataUnavailableError as exc:
                logger.debug("No chain for %s on %s: %s", sym, session, exc)
        return chains

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------

    def _process_exits(
        self,
        broker: SimulatedBroker,
        chains: dict[str, OptionChain],
        open_state: dict[str, _OpenTradeState],
        trades: list[TradeRecord],
        at: datetime,
    ) -> None:
        for pos in list(broker.get_positions()):
            for action in evaluate_exits(pos, at.date()):
                qty = max(1, round(pos.qty * action.close_fraction))
                qty = min(qty, pos.qty)
                order = Order(
                    legs=[
                        OrderLeg(
                            key=leg.key,
                            side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                            qty=leg.qty * qty,
                        )
                        for leg in pos.legs
                    ],
                    limit_price=pos.current_value,
                    tag=f"{pos.engine_tag}:{pos.catalyst_ref}",
                    intent=OrderIntent.CLOSE,
                    position_id=pos.position_id,
                )
                result = broker.place_order(order)
                if result.status is not OrderStatus.FILLED:
                    logger.debug("Exit rejected for %s: %s", pos.position_id, result.message)
                    continue
                state = open_state.get(pos.position_id)
                if state is not None:
                    state.commissions += result.commission
                    state.exit_cash_weighted += (result.avg_fill_price or 0.0) * result.filled_qty
                    state.qty_closed += result.filled_qty
                    state.exit_reasons.append(action.reason)
                    state.last_exit_time = at
                    if state.qty_closed >= state.max_qty:
                        trades.append(self._to_record(pos.position_id, state))
                        del open_state[pos.position_id]
                if action.is_tp1 and pos.qty > 0:
                    pos.tp1_done = True

    def _sweep_settled(
        self,
        broker: SimulatedBroker,
        open_state: dict[str, _OpenTradeState],
        trades: list[TradeRecord],
        at: datetime,
    ) -> None:
        """Fold positions the broker settled at expiry into trade records."""
        live_ids = {p.position_id for p in broker.get_positions()}
        # AUDIT MEDIUM: a deep-ITM settle that paid $6,000 into cash was
        # recorded as pnl=-$1.30 (exit approximated as entry). The broker now
        # keeps settlement telemetry; the record reconciles against it.
        settled = {pid: (value, pnl) for pid, value, pnl in
                   getattr(broker, "settlements", [])}
        for pid in list(open_state):
            if pid in live_ids:
                continue
            state = open_state.pop(pid)
            state.exit_reasons.append("expired_settled")
            state.last_exit_time = at
            if pid in settled:
                value, _pnl = settled[pid]
                remaining = state.max_qty - state.qty_closed
                state.exit_cash_weighted += value * remaining
                state.qty_closed += remaining
                trades.append(self._to_record(pid, state))
            else:
                trades.append(self._to_record(pid, state, settled_remainder=True))

    def _finalize_open(
        self,
        broker: SimulatedBroker,
        open_state: dict[str, _OpenTradeState],
        trades: list[TradeRecord],
        at: datetime,
    ) -> None:
        for pos in list(broker.get_positions()):
            state = open_state.pop(pos.position_id, None)
            if state is None:
                continue
            remaining = pos.qty
            state.exit_cash_weighted += pos.current_value * remaining
            state.qty_closed += remaining
            state.exit_reasons.append("end_of_backtest")
            state.last_exit_time = at
            trades.append(self._to_record(pos.position_id, state))

    def _to_record(
        self, position_id: str, state: _OpenTradeState, settled_remainder: bool = False
    ) -> TradeRecord:
        qty = state.qty_closed if state.qty_closed else state.max_qty
        if state.qty_closed:
            exit_price = state.exit_cash_weighted / state.qty_closed
        else:
            exit_price = state.entry_price  # settled remainder approximation
        closed_qty = state.qty_closed if state.qty_closed else state.max_qty
        mult = getattr(state, "multiplier", 100.0)
        pnl = (
            state.exit_cash_weighted * mult
            - state.entry_price * closed_qty * mult
            - state.commissions
        ) if state.qty_closed else -state.commissions
        return TradeRecord(
            position_id=position_id,
            engine=state.engine,
            catalyst_ref=state.catalyst_ref,
            underlying=state.underlying,
            direction=state.direction,
            entry_time=state.entry_time,
            exit_time=state.last_exit_time or state.entry_time,
            entry_price=state.entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl=pnl,
            exit_reason="+".join(state.exit_reasons) if state.exit_reasons else "unknown",
            max_qty=state.max_qty,
        )

    # ------------------------------------------------------------------
    # Hedge sleeve (M3): evaluated after exits, before new driver entries,
    # through the same entry gate as everything else.
    # ------------------------------------------------------------------

    def _process_hedge(
        self,
        broker: SimulatedBroker,
        chains: dict[str, OptionChain],
        open_state: dict[str, _OpenTradeState],
        at: datetime,
    ) -> None:
        if self._hedger is None:
            return
        positions = broker.get_positions()
        if not self._hedger.wants_check(at.date(), positions):  # type: ignore[attr-defined]
            return
        chain = chains.get(self._hedger.symbol)  # type: ignore[attr-defined]
        if chain is None:
            return
        account = broker.get_account()
        proposal = self._hedger.evaluate(chain, account.equity, positions, at)  # type: ignore[attr-defined]
        if proposal is not None:
            self._enter(broker, proposal, open_state, at)

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def _process_entries(
        self,
        broker: SimulatedBroker,
        chains: dict[str, OptionChain],
        history: dict[str, pd.DataFrame],
        catalysts_by_symbol: dict[str, list[Catalyst]],
        open_state: dict[str, _OpenTradeState],
        trades: list[TradeRecord],
        at: datetime,
    ) -> None:
        session = at.date()
        # One entry per (engine, catalyst): don't re-enter the same bet daily,
        # and never re-enter a bet that already closed.
        already = {(s.engine, s.catalyst_ref) for s in open_state.values()}
        already |= {(t.engine, t.catalyst_ref) for t in trades}

        for sym, cat_list in catalysts_by_symbol.items():
            chain = chains.get(sym)
            if chain is None:
                continue
            in_play = self._catalysts_in_play(cat_list, session)
            if not in_play:
                continue
            signal = self._signal_for(sym, history, session, chain)
            for catalyst in in_play:
                if not self._passes_screener(catalyst, chain, at):
                    continue
                ctx = StrategyContext(as_of=at, data=self._data, signal=signal,
                                      history=self._pit_history(history, session),
                                      chains={sym: chain})
                opp = Opportunity(session=session, symbols=(sym,), catalyst=catalyst)
                for strat in self._strategies:
                    if (strat.name, catalyst.ref) in already:
                        continue
                    # plan(), never evaluate() directly: the pipeline calls the
                    # strategy, so there is no path for a strategy to reach the
                    # broker, the risk layer or the cost model.
                    proposal = strat.plan(opp, ctx)
                    if proposal is None:
                        continue
                    self._enter(broker, proposal, open_state, at)
                    already.add((strat.name, catalyst.ref))

    def _process_scheduled_entries(
        self,
        broker: SimulatedBroker,
        chains: dict[str, OptionChain],
        history: dict[str, pd.DataFrame],
        open_state: dict[str, _OpenTradeState],
        trades: list[TradeRecord],
        at: datetime,
    ) -> None:
        """SCHEDULED / DAILY strategies. Identical downstream path to catalyst
        entries: plan() -> risk gate -> cost model -> exits. Only the way the
        opportunity was discovered differs."""
        session = at.date()
        open_refs = {(st.engine, st.catalyst_ref) for st in open_state.values()}

        for strat in self._strategies:
            if strat.cadence is Cadence.CATALYST:
                continue
            probe = StrategyContext(as_of=at, data=self._data,
                                    history=self._pit_history(history, session),
                                    chains=chains)
            for opp in strat.opportunities(session, probe):
                sym = opp.symbol
                if sym not in chains:
                    continue
                signal = self._signal_for(sym, history, session, chains[sym])
                ctx = StrategyContext(as_of=at, data=self._data, signal=signal,
                                      history=self._pit_history(history, session),
                                      chains=chains)
                proposal = strat.plan(opp, ctx)
                if proposal is None:
                    continue
                if (proposal.engine, proposal.catalyst_ref) in open_refs:
                    continue
                self._enter(broker, proposal, open_state, at)
                open_refs.add((proposal.engine, proposal.catalyst_ref))

    def _passes_screener(self, catalyst: Catalyst, chain: OptionChain, at: datetime) -> bool:
        """Screener gates PRE-event entries (implied move + liquidity). Post-
        event evaluation (Engine C's PEAD window) bypasses it: post-crush
        straddles can't measure the event's implied move, and C carries its
        own surprise/IV gates."""
        if self._screener is None or catalyst.when <= at:
            return True
        row = self._screener.screen(catalyst, chain, at)  # type: ignore[attr-defined]
        if row is None:
            return False
        if row.rejected:
            logger.debug("Screener rejected %s: %s", catalyst.ref, row.rejected)
            return False
        return True

    def _signal_for(
        self,
        symbol: str,
        history: dict[str, pd.DataFrame],
        session: date,
        chain: OptionChain,
    ):
        hist = history.get(symbol)
        if hist is None:
            hist = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        else:
            # Strictly before today's session: the 15:45 snapshot must not see
            # information from bars that close after it.
            hist = hist.loc[: pd.Timestamp(session) - pd.Timedelta(days=1)]
        return self._signal.evaluate(symbol, hist, chain)

    def _enter(
        self,
        broker: SimulatedBroker,
        proposal: ProposedTrade,
        open_state: dict[str, _OpenTradeState],
        at: datetime,
    ) -> None:
        decision = self._gate.size_entry(
            proposal, broker.get_account(), broker.get_positions()
        )
        if decision.units <= 0:
            logger.debug("Entry skipped (%s): %s", proposal.engine, decision.reason)
            return
        order = Order(
            legs=[
                OrderLeg(key=leg.key, side=leg.side, qty=leg.qty * decision.units)
                for leg in proposal.legs
            ],
            limit_price=proposal.unit_cost,
            tag=f"{proposal.engine}:{proposal.catalyst_ref}",
            intent=OrderIntent.OPEN,
            exit_rules=proposal.exit_rules,
            direction=proposal.direction,
            max_loss=proposal.unit_max_loss,
        )
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED:
            logger.debug("Entry rejected (%s): %s", proposal.engine, result.message)
            return
        position_id = f"pos-{result.order_id}"
        open_state[position_id] = _OpenTradeState(
            engine=proposal.engine,
            catalyst_ref=proposal.catalyst_ref,
            underlying=proposal.legs[0].key.underlying,
            direction=proposal.direction,
            multiplier=proposal.multiplier,
            entry_time=at,
            entry_price=result.avg_fill_price or proposal.unit_cost,
            max_qty=result.filled_qty,
            commissions=result.commission,
        )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(
        self,
        broker: SimulatedBroker,
        trades: list[TradeRecord],
        equity_curve: dict[date, float],
        start: date,
        end: date,
    ) -> BacktestResult:
        cfg = self._cfg
        equity = pd.Series(
            list(equity_curve.values()),
            index=pd.to_datetime(list(equity_curve.keys())),
            dtype=float,
        )
        daily_returns = equity.pct_change().dropna()
        tstats = m.trade_stats(trades)
        ruin = probability_of_ruin(daily_returns, cfg.backtest.monte_carlo)

        per_engine_pnl: dict[str, float] = defaultdict(float)
        per_engine_trades: dict[str, int] = defaultdict(int)
        for t in trades:
            per_engine_pnl[t.engine] += t.pnl
            per_engine_trades[t.engine] += 1

        starting = cfg.account.starting_capital
        ending = float(equity.iloc[-1]) if len(equity) else starting
        return BacktestResult(
            label=self._label,
            start=start,
            end=end,
            starting_capital=starting,
            ending_equity=ending,
            total_return=ending / starting - 1.0,
            n_trades=len(trades),
            win_rate=tstats["win_rate"],
            avg_win=tstats["avg_win"],
            avg_loss=tstats["avg_loss"],
            win_loss_ratio=tstats["win_loss_ratio"],
            max_drawdown=m.max_drawdown(equity),
            sharpe=m.sharpe(daily_returns),
            sortino=m.sortino(daily_returns),
            calmar=m.calmar(equity),
            probability_of_ruin=ruin,
            mc_survival_rate=1.0 - ruin,
            monthly=m.monthly_stats(equity),
            per_engine_pnl=dict(per_engine_pnl),
            per_engine_trades=dict(per_engine_trades),
            trades=trades,
            equity_curve={d.isoformat(): float(v) for d, v in equity_curve.items()},
        )

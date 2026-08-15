"""Minute-level driver for the Kinetic Ignition Engine.

The shared daily ``Backtester`` samples one 15:45 snapshot per session and so
cannot express an intraday strategy; this driver runs the minute loop while
reusing the *validated* shared infrastructure unchanged:

- ``SimulatedBroker`` — fill engine, cash/position ledger, commissions
- ``RiskManager``     — authoritative sizing, caps, circuit breakers
- ``metrics`` / ``montecarlo`` — the same metric suite as every prior gate
- ``ThetaDataHistorical`` / ``AlpacaMinuteBars`` — the same data layer

Per-session flow (no overnight holds, so each day starts flat):

    09:35  screen watchlist → fire signals → chain snapshot → RiskManager
           sizes → SimulatedBroker fills entry at the ASK (+flat slippage)
    09:36..time_stop  each minute: mark to NBBO, feed completed 5-min candles
           to the EMA tracker, evaluate exits, fill exits at the BID
    close  record equity; feed the RiskManager its daily mark for breakers

Cost model: entries at ask / exits at bid come from real minute NBBO. Where a
contract-minute has no NBBO, the configured synthetic haircut is applied to
the last known price and the trade is FLAGGED (``synthetic_fill=True``) so
synthetic fills can never masquerade as real ones in the report.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from catalyst.core.config import Config, FillModelConfig
from catalyst.core.types import (
    AccountState,
    BacktestResult,
    Catalyst,
    CatalystType,
    Direction,
    OptionChain,
    OptionContract,
    OptionKey,
    Order,
    OrderIntent,
    OrderLeg,
    OrderStatus,
    Side,
    TradeRecord,
)
from catalyst.core.tradingcal import sessions_in_range
from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.data.intraday import AlpacaMinuteBars, ThetaMinuteQuotes, five_minute_candles
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical
from catalyst.strategies.archive.kinetic.engine import KineticIgnitionEngine
from catalyst.strategies.archive.kinetic.exits import EmaTracker, KineticExitState
from catalyst.strategies.archive.kinetic.screener import KineticScreener, _parse_time
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class _OpenTrade:
    symbol: str
    position_id: str
    direction: str
    key: OptionKey
    entry_price: float
    entry_time: datetime
    contracts: int
    dte: int
    quotes: pd.DataFrame
    exit_state: KineticExitState
    synthetic_fill: bool
    gap: float
    requested_contracts: int  # what nav_fraction alone would have bought
    last_good_mid: float = 0.0  # last valid NBBO mid, for synthetic fallback


class KineticBacktester:
    def __init__(
        self,
        cfg: Config,
        data: ThetaDataHistorical,
        bars: AlpacaMinuteBars,
        quotes: ThetaMinuteQuotes,
        catalysts: list[Catalyst],
        zero_cost: bool = False,
        uncapped_sizing: bool = False,
        daily_closes: dict[str, pd.Series] | None = None,
        label: str = "kinetic",
    ) -> None:
        self._cfg = cfg
        self._data = data
        self._bars = bars
        self._quotes = quotes
        self._zero_cost = zero_cost
        self._uncapped = uncapped_sizing
        self._label = label
        self.screener = KineticScreener(cfg.kinetic, bars, daily_closes)
        self.engine = KineticIgnitionEngine(cfg.kinetic, cfg.risk, self.screener)

        self._signal_time = _parse_time(cfg.kinetic.signal.signal_time)
        self._session_open = _parse_time(cfg.kinetic.signal.session_open)
        self._time_stop = _parse_time(cfg.kinetic.exits.time_stop)

        # Earnings reaction days per symbol.
        self._reactions: dict[date, list[str]] = defaultdict(list)
        for c in catalysts:
            if c.type is CatalystType.EARNINGS:
                self._reactions[_reaction_session(c)].append(c.symbol)

        # Diagnostics surfaced in the report.
        self.watchlisted = 0
        self.fired = 0
        self.entered = 0
        self.risk_rejected = 0
        self.risk_trimmed = 0
        self.synthetic_fills = 0
        self.no_expiry_days = 0

    # ------------------------------------------------------------------

    def _fill_model(self) -> FillModelConfig:
        if self._zero_cost:
            # Mid fills, no slippage: isolates signal quality from friction.
            return FillModelConfig(spread_fill_fraction=0.0, slippage_pct_of_premium=0.0,
                                   slippage_per_contract=0.0)
        # Spec: entries at the ask, exits at the bid, plus flat per-contract slippage.
        return FillModelConfig(
            spread_fill_fraction=1.0,
            slippage_pct_of_premium=0.0,
            slippage_per_contract=self._cfg.kinetic.execution.slippage_per_contract,
        )

    def _chain_at(self, symbol: str, when: datetime, expiries: list[date] | None = None
                  ) -> OptionChain | None:
        try:
            return self._data.get_chain(symbol, when, expiries=expiries,
                                        max_dte=self._cfg.kinetic.execution.max_dte,
                                        include_liquidity=False)
        except DataUnavailableError:
            return None

    def _mark_chain(self, trade: _OpenTrade, now: datetime) -> tuple[OptionChain | None, bool]:
        """Single-contract chain at ``now`` from minute NBBO. Second element
        flags that a synthetic (haircut) quote had to be used."""
        ts = pd.Timestamp(now)
        row = trade.quotes[trade.quotes.index <= ts]
        if row.empty:
            return None, False
        last = row.iloc[-1]
        bid, ask = float(last["bid"]), float(last["ask"])
        # NaN-safe: a NaN quote must never reach a position mark, or it
        # silently poisons equity (and every metric downstream).
        valid = (bid == bid and ask == ask and ask > 0 and bid >= 0 and ask >= bid)
        synthetic = False
        if valid:
            trade.last_good_mid = (bid + ask) / 2.0
        else:
            haircut = self._cfg.kinetic.execution.synthetic_haircut
            reference = trade.last_good_mid or trade.entry_price
            bid, ask = reference * (1 - haircut), reference * (1 + haircut)
            synthetic = True
        contract = OptionContract(key=trade.key, bid=bid, ask=ask)
        return OptionChain(underlying=trade.symbol, underlying_price=0.0,
                           timestamp=now, contracts=[contract]), synthetic

    # ------------------------------------------------------------------

    def run(self, start: date, end: date) -> tuple[BacktestResult, dict[str, object]]:
        cfg = self._cfg
        rm = RiskManager(cfg.risk)
        broker = SimulatedBroker(fill_model=self._fill_model(),
                                 commissions=cfg.execution.commissions,
                                 starting_cash=cfg.account.starting_capital)
        trades: list[TradeRecord] = []
        equity_curve: dict[date, float] = {}
        sessions = [s for s in sessions_in_range(start, end) if s in self._reactions]

        for session in sessions:
            symbols = sorted(set(self._reactions[session]))
            open_trades = self._open_session(broker, rm, symbols, session)
            if open_trades:
                self._run_minute_loop(broker, open_trades, session, trades)
            equity = broker.get_account().equity
            equity_curve[session] = equity
            rm.record_mark(session, equity)

        extras = {
            "watchlisted": self.watchlisted, "fired": self.fired, "entered": self.entered,
            "risk_rejected": self.risk_rejected, "risk_trimmed": self.risk_trimmed,
            "synthetic_fills": self.synthetic_fills,
            "skipped_no_expiry": self.engine.skipped_no_expiry,
            "skipped_no_contract": self.engine.skipped_no_contract,
            "skipped_thin_premium": self.engine.skipped_thin_premium,
            "commissions": broker.total_commissions,
        }
        return self._build_result(trades, equity_curve, start, end), extras

    # ------------------------------------------------------------------

    def _open_session(self, broker: SimulatedBroker, rm: RiskManager,
                      symbols: list[str], session: date) -> list[_OpenTrade]:
        cfg = self._cfg
        entry_dt = datetime.combine(session, self._signal_time)
        opened: list[_OpenTrade] = []

        for symbol in symbols:
            sig = self.screener.evaluate(symbol, session)
            if sig.watchlisted and sig.reject_reason != "no_premarket":
                self.watchlisted += 1
            if not sig.fires:
                continue
            self.fired += 1

            chain = self._chain_at(symbol, entry_dt)
            if chain is None:
                self.engine.skipped_no_contract += 1
                continue
            catalyst = Catalyst(symbol=symbol, type=CatalystType.EARNINGS,
                                when=datetime.combine(session, self._session_open),
                                source="kinetic")
            proposal = self.engine.evaluate(catalyst, chain, _NO_SIGNAL, entry_dt)
            if proposal is None:
                continue

            account = broker.get_account()
            equity = account.equity
            if not (equity == equity) or equity <= 0:  # NaN or wiped out
                logger.warning("Non-finite equity at %s %s; skipping entry", symbol, session)
                continue
            account = AccountState(equity=equity, cash=account.cash,
                                   buying_power=account.buying_power, timestamp=entry_dt)
            unit_cost = proposal.unit_cost
            if not (unit_cost == unit_cost) or unit_cost <= 0:
                continue
            requested = int((equity * cfg.kinetic.execution.nav_fraction) // (unit_cost * 100.0))
            if self._uncapped:
                units = max(requested, 0)
                reason = "uncapped_diagnostic"
            else:
                decision = rm.size_entry(proposal, account, broker.get_positions())
                units, reason = decision.units, decision.reason
            if units <= 0:
                self.risk_rejected += 1
                logger.debug("Kinetic entry blocked %s %s: %s", symbol, session, reason)
                continue
            if units < requested:
                self.risk_trimmed += 1

            broker.update_market({symbol: chain}, entry_dt)
            order = Order(legs=[OrderLeg(key=proposal.legs[0].key, side=Side.BUY, qty=units)],
                          limit_price=proposal.unit_cost,
                          tag=f"kinetic:{symbol}:{session.isoformat()}",
                          intent=OrderIntent.OPEN, exit_rules=proposal.exit_rules,
                          direction=proposal.direction)
            result = broker.place_order(order)
            if result.status is not OrderStatus.FILLED:
                self.risk_rejected += 1
                continue
            self.entered += 1

            key = proposal.legs[0].key
            quotes = self._quotes.contract_day(key, session, self._signal_time,
                                               _end_of_day(self._time_stop))
            entry_price = result.avg_fill_price or proposal.unit_cost
            opened.append(_OpenTrade(
                symbol=symbol, position_id=f"pos-{result.order_id}",
                direction=sig.direction or "long", key=key,
                entry_price=entry_price, entry_time=entry_dt, contracts=units,
                dte=(key.expiry - session).days, quotes=quotes,
                exit_state=KineticExitState(
                    direction=sig.direction or "long", entry_price=entry_price,
                    hard_stop_loss=cfg.kinetic.exits.hard_stop_loss,
                    time_stop=self._time_stop,
                    ema=EmaTracker(period=cfg.kinetic.exits.ema_period),
                ),
                synthetic_fill=False, gap=sig.gap or 0.0, requested_contracts=requested,
            ))
        return opened

    def _run_minute_loop(self, broker: SimulatedBroker, open_trades: list[_OpenTrade],
                         session: date, trades: list[TradeRecord]) -> None:
        cfg = self._cfg
        candle_minutes = cfg.kinetic.exits.candle_minutes
        live = {t.position_id: t for t in open_trades}

        # Candle completion schedule per symbol: (completion_time, close).
        # Precomputed once so the minute loop is a pointer advance rather than
        # a full frame scan per minute per position.
        schedule: dict[str, list[tuple[datetime, float]]] = {}
        for t in open_trades:
            bars = self._bars.get_day(t.symbol, session)
            candles = five_minute_candles(bars, self._session_open, candle_minutes)
            schedule[t.symbol] = [
                ((ts + pd.Timedelta(minutes=candle_minutes)).to_pydatetime(), float(close))
                for ts, close in zip(candles.index, candles["close"])
            ]

        minute = datetime.combine(session, self._signal_time) + timedelta(minutes=1)
        stop_dt = datetime.combine(session, self._time_stop)
        cursor: dict[str, int] = defaultdict(int)

        while minute <= stop_dt and live:
            for pid, trade in list(live.items()):
                # Feed every candle that COMPLETED at or before this minute.
                sym_schedule = schedule.get(trade.symbol, [])
                while (cursor[pid] < len(sym_schedule)
                       and sym_schedule[cursor[pid]][0] <= minute):
                    trade.exit_state.on_candle_close(sym_schedule[cursor[pid]][1])
                    cursor[pid] += 1

                chain, synthetic = self._mark_chain(trade, minute)
                if chain is None:
                    continue
                if synthetic and not trade.synthetic_fill:
                    trade.synthetic_fill = True
                    self.synthetic_fills += 1
                broker.update_market({trade.symbol: chain}, minute)
                mid = chain.contracts[0].mid
                decision = trade.exit_state.evaluate_minute(minute.time(), mid)
                if not decision.should_exit:
                    continue
                self._close(broker, trade, minute, decision.reason, trades)
                del live[pid]
            minute += timedelta(minutes=1)

        # Safety net: anything still open at the stop is liquidated (no overnight holds).
        for pid, trade in list(live.items()):
            chain, _ = self._mark_chain(trade, stop_dt)
            if chain is not None:
                broker.update_market({trade.symbol: chain}, stop_dt)
            self._close(broker, trade, stop_dt, "time_stop", trades)

    def _close(self, broker: SimulatedBroker, trade: _OpenTrade, when: datetime,
               reason: str, trades: list[TradeRecord]) -> None:
        position = next((p for p in broker.get_positions()
                         if p.position_id == trade.position_id), None)
        if position is None:
            return
        order = Order(legs=[OrderLeg(key=trade.key, side=Side.SELL, qty=position.qty)],
                      limit_price=position.current_value,
                      tag=f"kinetic:{trade.symbol}", intent=OrderIntent.CLOSE,
                      position_id=trade.position_id)
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED:
            logger.warning("Kinetic exit rejected %s: %s", trade.symbol, result.message)
            return
        exit_price = result.avg_fill_price or 0.0
        pnl = (exit_price - trade.entry_price) * result.filled_qty * 100.0 - result.commission
        trades.append(TradeRecord(
            position_id=trade.position_id, engine="kinetic",
            catalyst_ref=f"{trade.symbol}:{when.date().isoformat()}:dte{trade.dte}",
            underlying=trade.symbol,
            direction=Direction.LONG if trade.direction == "long" else Direction.SHORT,
            entry_time=trade.entry_time, exit_time=when,
            entry_price=trade.entry_price, exit_price=exit_price,
            qty=result.filled_qty, pnl=pnl, exit_reason=reason,
            max_qty=trade.contracts,
        ))

    # ------------------------------------------------------------------

    def _build_result(self, trades: list[TradeRecord], equity_curve: dict[date, float],
                      start: date, end: date) -> BacktestResult:
        cfg = self._cfg
        equity = pd.Series(list(equity_curve.values()),
                           index=pd.to_datetime(list(equity_curve.keys())), dtype=float)
        daily = equity.pct_change().dropna()
        tstats = m.trade_stats(trades)
        ruin = probability_of_ruin(daily, cfg.backtest.monte_carlo) if len(daily) > 2 else 0.0
        per_bucket: dict[str, float] = defaultdict(float)
        per_bucket_n: dict[str, int] = defaultdict(int)
        for t in trades:
            bucket = "0DTE" if t.catalyst_ref.endswith("dte0") else "1-3DTE"
            per_bucket[bucket] += t.pnl
            per_bucket_n[bucket] += 1
        starting = cfg.account.starting_capital
        ending = float(equity.iloc[-1]) if len(equity) else starting
        return BacktestResult(
            label=self._label, start=start, end=end,
            starting_capital=starting, ending_equity=ending,
            total_return=ending / starting - 1.0,
            n_trades=len(trades), win_rate=tstats["win_rate"],
            avg_win=tstats["avg_win"], avg_loss=tstats["avg_loss"],
            win_loss_ratio=tstats["win_loss_ratio"],
            max_drawdown=m.max_drawdown(equity), sharpe=m.sharpe(daily),
            sortino=m.sortino(daily), calmar=m.calmar(equity),
            probability_of_ruin=ruin, mc_survival_rate=1.0 - ruin,
            monthly=m.monthly_stats(equity),
            per_engine_pnl=dict(per_bucket), per_engine_trades=dict(per_bucket_n),
            trades=trades,
            equity_curve={d.isoformat(): float(v) for d, v in equity_curve.items()},
        )


def _reaction_session(catalyst: Catalyst) -> date:
    """First session that can react: AMC reports move the NEXT day."""
    from catalyst.data.catalysts import resolve_reaction_session

    return resolve_reaction_session(catalyst)


def _end_of_day(stop: time) -> time:
    """Quote window end — a few minutes past the time stop for safety."""
    return time(min(stop.hour + 0, 23), min(stop.minute + 5, 59))


class _NoSignal:
    direction = Direction.NEUTRAL
    confidence = 0.0


_NO_SIGNAL = _NoSignal()

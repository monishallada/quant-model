"""Version 2 — the same cointegration signal expressed through 1-2 DTE ATM
options, run entirely through the validated SimulatedBroker.

Structure per signal:
- Z > +entry (A rich):  buy ATM put on A  + buy ATM call on B
- Z < -entry (A cheap): buy ATM call on A + buy ATM put on B
- Equal dollar premium per leg (small-integer contract ratio within
  tolerance); total premium sized by the authoritative RiskManager.

Costs are modeled punitively per the test's central question: fills use the
configured fill model with ``spread_fill_fraction=1.0`` overrides — entries
pay the FULL ASK (+slippage), exits hit the FULL BID (−slippage), per leg,
each way. Theta decay is inherent in marking against real quoted chains.

Lifecycle:
- signals evaluated on closes through t−1; orders execute at t's 15:45 ET
  chain snapshot (lookahead-free)
- Z-reversion exit: |Z| < z_exit  → sell both legs
- expiry-day stop: any position whose leg expires today is liquidated at the
  14:00 ET snapshot, NEVER held to the close — and if the signal is still
  open and the pair still cointegrated, the position is ROLLED (re-entered
  next session with fresh 1-2 DTE options, paying the full entry costs
  again). Rolls are counted and reported: the reversion outliving the
  option's life is a core cost of this structure.
- pair deactivation (ADF p >= alpha): flatten immediately, no roll.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from collections import defaultdict

import pandas as pd

from catalyst.core.config import Config
from catalyst.core.models import (
    AccountState,
    BacktestResult,
    Direction,
    ExitRules,
    OptionChain,
    OptionRight,
    Order,
    OrderIntent,
    OrderLeg,
    OrderStatus,
    ProposedTrade,
    Side,
    TradeRecord,
)
from catalyst.core.tradingcal import sessions_in_range, trading_days_between
from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical
from catalyst.engines.util import atm_contract
from catalyst.pairs.cointegration import CointegrationEngine, PairState
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)

_ENTRY_SNAP = time(15, 45)


@dataclass
class _OpenOptionPair:
    pair: tuple[str, str]
    position_id: str
    direction: str
    entry_session: date
    entry_z: float
    rolls: int  # how many times this signal has been re-entered after expiry


class PairsOptionsBacktester:
    def __init__(
        self,
        cfg: Config,
        data: ThetaDataHistorical,
        closes: dict[str, pd.Series],
        zero_cost: bool = False,
        label: str = "pairs-options",
    ) -> None:
        self._cfg = cfg
        self._data = data
        self._closes = closes
        self._zero_cost = zero_cost
        self._label = label
        self.engine = CointegrationEngine(cfg.pairs, closes)
        hh, mm = cfg.pairs.options.expiry_liquidation_time.split(":")
        self._liq_snap = time(int(hh), int(mm))
        self.roll_count = 0
        self.skipped_no_expiry = 0
        self.skipped_thin_premium = 0
        self.trades_on_decoupled = 0

    # ------------------------------------------------------------------

    def _fill_overrides(self):
        from catalyst.core.config import FillModelConfig

        if self._zero_cost:
            return FillModelConfig(spread_fill_fraction=0.0, slippage_pct_of_premium=0.0)
        # Punitive per spec: full worse-side NBBO + configured slippage.
        return FillModelConfig(
            spread_fill_fraction=1.0,
            slippage_pct_of_premium=self._cfg.execution.fill_model.slippage_pct_of_premium,
        )

    def _valid_expiry(self, symbol: str, session: date) -> date | None:
        lo, hi = self._cfg.pairs.options.dte_window_trading_days
        for expiry in self._data.list_expirations(symbol):
            if expiry <= session:
                continue
            dte = trading_days_between(session, expiry)
            if lo <= dte <= hi:
                return expiry
            if dte > hi:
                return None
        return None

    def _chain(self, symbol: str, session: date, snap: time, expiry: date) -> OptionChain | None:
        try:
            return self._data.get_chain(
                symbol, datetime.combine(session, snap), expiries=[expiry],
                include_liquidity=False,
            )
        except DataUnavailableError:
            return None

    def _marks_for(self, positions, session: date, snap: time) -> dict[str, OptionChain]:
        """Chains covering every open leg's expiry at the given snapshot."""
        chains: dict[str, OptionChain] = {}
        needed: dict[str, set[date]] = defaultdict(set)
        for pos in positions:
            for leg in pos.legs:
                if leg.key.expiry >= session:
                    needed[leg.key.underlying].add(leg.key.expiry)
        for sym, exps in needed.items():
            try:
                chains[sym] = self._data.get_chain(
                    sym, datetime.combine(session, snap), expiries=sorted(exps),
                    include_liquidity=False,
                )
            except DataUnavailableError:
                pass
        return chains

    # ------------------------------------------------------------------

    def run(self, start: date, end: date) -> tuple[BacktestResult, dict[str, object]]:
        cfg = self._cfg
        pcfg = cfg.pairs
        rm = RiskManager(cfg.risk)
        rm.set_history({s: pd.DataFrame({"close": c}) for s, c in self._closes.items()})
        broker = SimulatedBroker(
            fill_model=self._fill_overrides(),
            commissions=cfg.execution.commissions,
            starting_cash=cfg.account.starting_capital,
        )

        sessions = sessions_in_range(start, end)
        states: dict[tuple[str, str], dict[date, PairState]] = {}
        for pair in [tuple(p) for p in pcfg.universe]:
            states[pair] = {st.session: st for st in self.engine.compute_states(pair, sessions)}

        open_pairs: dict[tuple[str, str], _OpenOptionPair] = {}
        pending_entry: dict[tuple[str, str], tuple[str, float, int]] = {}  # dir, z, rolls
        open_meta: dict[str, _OpenOptionPair] = {}
        trades: list[TradeRecord] = []
        trade_entries: dict[str, tuple[float, float, datetime, str, int]] = {}
        equity_curve: dict[date, float] = {}

        def close_position(op: _OpenOptionPair, session: date, snap: time, reason: str) -> None:
            pos = next((p for p in broker.get_positions() if p.position_id == op.position_id), None)
            if pos is None:
                return
            order = Order(
                legs=[
                    OrderLeg(key=leg.key,
                             side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                             qty=leg.qty * pos.qty)
                    for leg in pos.legs
                ],
                limit_price=pos.current_value,
                tag=f"pairs_v2:{op.pair[0]}/{op.pair[1]}",
                intent=OrderIntent.CLOSE,
                position_id=op.position_id,
            )
            result = broker.place_order(order)
            if result.status is not OrderStatus.FILLED:
                logger.warning("V2 close rejected %s (%s): %s", op.pair, reason, result.message)
                return
            if reason == "decoupled":
                self.trades_on_decoupled += 1
            entry_net, unit_entry, entry_time, _, max_qty = trade_entries.pop(op.position_id)
            pnl = (result.avg_fill_price - unit_entry) * result.filled_qty * 100.0 - result.commission
            trades.append(TradeRecord(
                position_id=op.position_id,
                engine="pairs_v2",
                catalyst_ref=f"{op.pair[0]}/{op.pair[1]}",
                underlying=op.pair[0],
                direction=Direction.NEUTRAL,
                entry_time=entry_time,
                exit_time=datetime.combine(session, snap),
                entry_price=unit_entry,
                exit_price=result.avg_fill_price or 0.0,
                qty=result.filled_qty,
                pnl=pnl,
                exit_reason=reason,
                max_qty=max_qty,
            ))
            del open_pairs[op.pair]
            open_meta.pop(op.position_id, None)

        for session in sessions:
            prev_states = {
                pair: next(
                    (by_day[s] for s in sorted(by_day, reverse=True) if s < session and by_day[s] is not None),
                    None,
                )
                for pair, by_day in states.items()
            }

            # ---- 14:00 expiry-day liquidation (before anything else). ----
            expiring = [
                op for op in list(open_pairs.values())
                if any(
                    leg.key.expiry == session
                    for p in broker.get_positions() if p.position_id == op.position_id
                    for leg in p.legs
                )
            ]
            if expiring:
                positions = [p for p in broker.get_positions()
                             if any(p.position_id == op.position_id for op in expiring)]
                broker.update_market(self._marks_for(positions, session, self._liq_snap),
                                     datetime.combine(session, self._liq_snap))
                for op in expiring:
                    st = prev_states.get(op.pair)
                    still_wants = (
                        st is not None and st.active and st.zscore is not None
                        and abs(st.zscore) >= pcfg.z_exit
                        and op.rolls < pcfg.options.max_consecutive_rolls
                    )
                    close_position(op, session, self._liq_snap, "expiry_liquidation")
                    if still_wants:
                        direction = op.direction
                        pending_entry[op.pair] = (direction, st.zscore or 0.0, op.rolls + 1)
                        self.roll_count += 1

            # ---- 15:45: mark everything, exits, then entries. ----
            at = datetime.combine(session, _ENTRY_SNAP)
            broker.update_market(self._marks_for(broker.get_positions(), session, _ENTRY_SNAP), at)

            for pair, op in list(open_pairs.items()):
                st = prev_states.get(pair)
                if st is None:
                    continue
                if not st.active:
                    close_position(op, session, _ENTRY_SNAP, "decoupled")
                elif st.zscore is not None and abs(st.zscore) < pcfg.z_exit:
                    close_position(op, session, _ENTRY_SNAP, "z_reverted")

            # New signals from t-1 closes.
            for pair, st in prev_states.items():
                if st is None or not st.active or st.zscore is None:
                    continue
                if pair in open_pairs or pair in pending_entry:
                    continue
                if st.zscore > pcfg.z_entry:
                    pending_entry[pair] = ("short_A_long_B", st.zscore, 0)
                elif st.zscore < -pcfg.z_entry:
                    pending_entry[pair] = ("long_A_short_B", st.zscore, 0)

            for pair, (direction, z, rolls) in list(pending_entry.items()):
                del pending_entry[pair]
                if pair in open_pairs:
                    continue
                self._try_enter(
                    broker, rm, pair, direction, z, rolls, session,
                    open_pairs, open_meta, trade_entries,
                )

            equity = broker.get_account().equity
            equity_curve[session] = equity
            rm.record_mark(session, equity)

        # Final liquidation for clean records.
        at = datetime.combine(sessions[-1], _ENTRY_SNAP)
        broker.update_market(self._marks_for(broker.get_positions(), sessions[-1], _ENTRY_SNAP), at)
        for op in list(open_pairs.values()):
            close_position(op, sessions[-1], _ENTRY_SNAP, "end_of_backtest")

        extras = {
            "stability": self.engine.ledger,
            "rolls": self.roll_count,
            "skipped_no_expiry": self.skipped_no_expiry,
            "skipped_thin_premium": self.skipped_thin_premium,
            "trades_on_decoupled": self.trades_on_decoupled,
            "total_commissions": broker.total_commissions,
        }
        return self._build_result(trades, equity_curve, start, end), extras

    # ------------------------------------------------------------------

    def _try_enter(self, broker, rm, pair, direction, z, rolls, session,
                   open_pairs, open_meta, trade_entries) -> None:
        pcfg = self._cfg.pairs
        a_sym, b_sym = pair
        exp_a = self._valid_expiry(a_sym, session)
        exp_b = self._valid_expiry(b_sym, session)
        if exp_a is None or exp_b is None:
            self.skipped_no_expiry += 1
            return
        chain_a = self._chain(a_sym, session, _ENTRY_SNAP, exp_a)
        chain_b = self._chain(b_sym, session, _ENTRY_SNAP, exp_b)
        if chain_a is None or chain_b is None:
            self.skipped_no_expiry += 1
            return
        right_a = OptionRight.PUT if direction == "short_A_long_B" else OptionRight.CALL
        right_b = OptionRight.CALL if direction == "short_A_long_B" else OptionRight.PUT
        leg_a = atm_contract(chain_a, exp_a, right_a)
        leg_b = atm_contract(chain_b, exp_b, right_b)
        if leg_a is None or leg_b is None:
            self.skipped_no_expiry += 1
            return
        if min(leg_a.mid, leg_b.mid) < pcfg.options.min_premium:
            self.skipped_thin_premium += 1
            return

        # Equal-dollar premium: small integer ratio within tolerance.
        qty_a, qty_b = 1, 1
        if leg_a.mid > leg_b.mid:
            qty_b = max(1, round(leg_a.mid / leg_b.mid))
        else:
            qty_a = max(1, round(leg_b.mid / leg_a.mid))
        mismatch = abs(qty_a * leg_a.mid - qty_b * leg_b.mid) / max(qty_a * leg_a.mid, qty_b * leg_b.mid)
        if mismatch > pcfg.options.equal_premium_tolerance:
            self.skipped_thin_premium += 1
            return

        unit_cost = qty_a * leg_a.mid + qty_b * leg_b.mid
        unit_worst = qty_a * leg_a.ask + qty_b * leg_b.ask
        proposal = ProposedTrade(
            engine="pairs_v2",
            catalyst_ref=f"{a_sym}/{b_sym}",
            legs=[
                OrderLeg(key=leg_a.key, side=Side.BUY, qty=qty_a),
                OrderLeg(key=leg_b.key, side=Side.BUY, qty=qty_b),
            ],
            unit_cost=unit_cost,
            unit_max_loss=unit_worst,
            direction=Direction.NEUTRAL,
            exit_rules=ExitRules(use_stops=False, close_before_expiry_days=0),
            per_trade_risk_fraction=pcfg.per_trade_risk,
        )
        account = broker.get_account()
        decision = rm.size_entry(proposal, account, broker.get_positions())
        if decision.units <= 0:
            logger.debug("V2 entry blocked (%s): %s", pair, decision.reason)
            return
        chains = {a_sym: chain_a, b_sym: chain_b}
        # Merge with existing marks so open positions stay marked.
        merged = self._marks_for(broker.get_positions(), session, _ENTRY_SNAP)
        merged.update(chains)
        broker.update_market(merged, datetime.combine(session, _ENTRY_SNAP))
        order = Order(
            legs=[
                OrderLeg(key=leg_a.key, side=Side.BUY, qty=qty_a * decision.units),
                OrderLeg(key=leg_b.key, side=Side.BUY, qty=qty_b * decision.units),
            ],
            limit_price=unit_cost,
            tag=f"pairs_v2:{a_sym}/{b_sym}",
            intent=OrderIntent.OPEN,
            exit_rules=proposal.exit_rules,
            direction=Direction.NEUTRAL,
        )
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED:
            logger.debug("V2 entry rejected (%s): %s", pair, result.message)
            return
        position_id = f"pos-{result.order_id}"
        op = _OpenOptionPair(
            pair=pair, position_id=position_id, direction=direction,
            entry_session=session, entry_z=z, rolls=rolls,
        )
        open_pairs[pair] = op
        open_meta[position_id] = op
        trade_entries[position_id] = (
            result.avg_fill_price or unit_cost,
            result.avg_fill_price or unit_cost,
            datetime.combine(session, _ENTRY_SNAP),
            direction,
            result.filled_qty,
        )

    def _build_result(self, trades, equity_curve, start, end) -> BacktestResult:
        cfg = self._cfg
        equity = pd.Series(list(equity_curve.values()),
                           index=pd.to_datetime(list(equity_curve.keys())), dtype=float)
        daily = equity.pct_change().dropna()
        tstats = m.trade_stats(trades)
        ruin = probability_of_ruin(daily, cfg.backtest.monte_carlo)
        per_pair = defaultdict(float)
        for t in trades:
            per_pair[t.catalyst_ref] += t.pnl
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
            per_engine_pnl=dict(per_pair),
            per_engine_trades={k: sum(1 for t in trades if t.catalyst_ref == k) for k in per_pair},
            trades=trades,
            equity_curve={d.isoformat(): float(v) for d, v in equity_curve.items()},
        )

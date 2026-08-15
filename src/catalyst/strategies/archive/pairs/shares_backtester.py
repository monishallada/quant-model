"""Version 1 — pure stat-arb on the underlying shares (the edge test).

Dollar-neutral: at entry, long and short legs carry equal dollar notional.
Fills at the NEXT session's open after a signal (signals use closes through
t-1; no lookahead). Costs: half-spread + slippage per side per leg,
commission per share, and daily borrow accrual on the short leg. Marks and
equity at session closes.

Sizing routes through the authoritative RiskManager: one "unit" of a pair is
GROSS notional whose assumed worst adverse spread move (``max_adverse_move``)
equals the risk basis, so the per-trade cap, heat cap, cash floor, and
breakers all bind exactly as they do for options premium. (An adapter
constructs the ProposedTrade/Position shapes RiskManager understands; the
risk code itself is untouched.)

The SimulatedBroker is an options fill engine and is not used here; V2 (the
options expression) runs through it fully. This module's ledger is the
minimal share-market simulator the comparison requires, unit-tested on
hand-computed cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from collections import defaultdict

import pandas as pd

from catalyst.core.config import Config
from catalyst.core.types import (
    AccountState,
    BacktestResult,
    Direction,
    ExitRules,
    OptionKey,
    OptionRight,
    OrderLeg,
    Position,
    PositionLeg,
    ProposedTrade,
    Side,
    TradeRecord,
)
from catalyst.core.tradingcal import sessions_in_range
from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.strategies.archive.pairs.cointegration import CointegrationEngine, PairState
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)

_SENTINEL_EXPIRY = date(2099, 1, 1)
_MARK_TIME = time(16, 0)


def _sentinel_key(symbol: str) -> OptionKey:
    """Identity-only key so RiskManager's correlation cap can group pair
    positions by underlying without any change to risk code."""
    return OptionKey(underlying=symbol, expiry=_SENTINEL_EXPIRY, right=OptionRight.CALL, strike=1.0)


@dataclass
class _OpenPair:
    pair: tuple[str, str]
    direction: str  # "short_A_long_B" (Z>+entry) or "long_A_short_B" (Z<-entry)
    qty_a: int  # shares of A (long or short per direction)
    qty_b: int
    entry_a: float  # effective fill prices (costs included)
    entry_b: float
    entry_session: date
    risk_dollars: float  # risk basis registered with RiskManager
    borrow_accrued: float = 0.0
    entry_z: float = 0.0


class PairsSharesBacktester:
    def __init__(self, cfg: Config, closes: dict[str, pd.Series], opens: dict[str, pd.Series],
                 zero_cost: bool = False, label: str = "pairs-shares") -> None:
        self._cfg = cfg
        self._closes = closes
        self._opens = opens
        self._zero_cost = zero_cost
        self._label = label
        self.engine = CointegrationEngine(cfg.pairs, closes)
        self.trades_on_decoupled = 0  # entered, then pair broke before Z-exit

    # ------------------------------------------------------------------

    def _px(self, series: dict[str, pd.Series], symbol: str, session: date) -> float | None:
        s = series.get(symbol)
        if s is None:
            return None
        ts = pd.Timestamp(session)
        if ts in s.index:
            v = float(s.loc[ts])
            return v if v == v else None
        return None

    def _fill_price(self, raw: float, side: Side) -> float:
        if self._zero_cost:
            return raw
        bps = (self._cfg.pairs.shares.half_spread_bps + self._cfg.pairs.shares.slippage_bps) / 1e4
        return raw * (1 + bps) if side is Side.BUY else raw * (1 - bps)

    def _commission(self, shares: int) -> float:
        if self._zero_cost:
            return 0.0
        return self._cfg.pairs.shares.commission_per_share * shares

    def _borrow_apr(self, symbol: str) -> float:
        sh = self._cfg.pairs.shares
        return sh.borrow_apr_overrides.get(symbol, sh.borrow_apr_default)

    # ------------------------------------------------------------------

    def run(self, start: date, end: date) -> tuple[BacktestResult, dict[str, object]]:
        cfg = self._cfg
        pcfg = cfg.pairs
        rm = RiskManager(cfg.risk)
        rm.set_history({s: pd.DataFrame({"close": c}) for s, c in self._closes.items()})

        sessions = sessions_in_range(start, end)
        states: dict[tuple[str, str], dict[date, PairState]] = {}
        for pair in [tuple(p) for p in pcfg.universe]:
            states[pair] = {st.session: st for st in self.engine.compute_states(pair, sessions)}

        cash = cfg.account.starting_capital
        open_pairs: dict[tuple[str, str], _OpenPair] = {}
        pending: dict[tuple[str, str], tuple[str, float]] = {}  # fill next open: (direction, entry_z)
        pending_close: dict[tuple[str, str], str] = {}  # close next open: reason
        trades: list[TradeRecord] = []
        equity_curve: dict[date, float] = {}

        def mark_equity(session: date) -> float:
            total = cash
            for op in open_pairs.values():
                pa = self._px(self._closes, op.pair[0], session)
                pb = self._px(self._closes, op.pair[1], session)
                if pa is None or pb is None:
                    continue
                sign_a = -1 if op.direction == "short_A_long_B" else 1
                total += sign_a * op.qty_a * (pa - op.entry_a)
                total += -sign_a * op.qty_b * (pb - op.entry_b)
                total -= op.borrow_accrued
            return total

        def risk_positions() -> list[Position]:
            """Adapter: open pairs as Position shapes for RiskManager caps."""
            out = []
            for op in open_pairs.values():
                out.append(Position(
                    position_id=f"pair-{op.pair[0]}-{op.pair[1]}-{op.entry_session}",
                    legs=[PositionLeg(key=_sentinel_key(op.pair[0]), side=Side.BUY, qty=1)],
                    qty=1,
                    entry_price=op.risk_dollars / 100.0,  # premium_at_risk == risk_dollars
                    entry_time=datetime.combine(op.entry_session, _MARK_TIME),
                    engine_tag="pairs_v1",
                    direction=Direction.NEUTRAL,
                    exit_rules=ExitRules(),
                    current_value=op.risk_dollars / 100.0,
                ))
            return out

        def close_pair(op: _OpenPair, session: date, reason: str) -> None:
            nonlocal cash
            pa_raw = self._px(self._opens, op.pair[0], session) or self._px(self._closes, op.pair[0], session)
            pb_raw = self._px(self._opens, op.pair[1], session) or self._px(self._closes, op.pair[1], session)
            if pa_raw is None or pb_raw is None:
                logger.warning("No exit prices for %s on %s; holding", op.pair, session)
                return
            sign_a = -1 if op.direction == "short_A_long_B" else 1
            # Closing side is the opposite of the holding side per leg.
            exit_a = self._fill_price(pa_raw, Side.BUY if sign_a < 0 else Side.SELL)
            exit_b = self._fill_price(pb_raw, Side.SELL if sign_a < 0 else Side.BUY)
            pnl = (
                sign_a * op.qty_a * (exit_a - op.entry_a)
                + -sign_a * op.qty_b * (exit_b - op.entry_b)
                - op.borrow_accrued
                - self._commission(op.qty_a + op.qty_b)
            )
            cash += pnl
            if reason == "decoupled":
                self.trades_on_decoupled += 1
            gross = op.qty_a * op.entry_a + op.qty_b * op.entry_b
            trades.append(TradeRecord(
                position_id=f"pair-{op.pair[0]}-{op.pair[1]}-{op.entry_session}",
                engine="pairs_v1",
                catalyst_ref=f"{op.pair[0]}/{op.pair[1]}",
                underlying=op.pair[0],
                direction=Direction.NEUTRAL,
                entry_time=datetime.combine(op.entry_session, _MARK_TIME),
                exit_time=datetime.combine(session, _MARK_TIME),
                entry_price=gross / 100.0,
                exit_price=(gross + pnl) / 100.0,
                qty=1,
                pnl=pnl,
                exit_reason=reason,
                max_qty=1,
            ))
            del open_pairs[op.pair]

        for session in sessions:
            # 1. Execute pending closes at today's open.
            for pair, reason in list(pending_close.items()):
                if pair in open_pairs:
                    close_pair(open_pairs[pair], session, reason)
                del pending_close[pair]

            # 2. Execute pending entries at today's open (through RiskManager).
            for pair, (direction, entry_z) in list(pending.items()):
                del pending[pair]
                if pair in open_pairs:
                    continue
                a_sym, b_sym = pair
                short_sym = a_sym if direction == "short_A_long_B" else b_sym
                if short_sym in pcfg.shares.hard_to_borrow_skip:
                    continue
                pa_raw = self._px(self._opens, a_sym, session)
                pb_raw = self._px(self._opens, b_sym, session)
                if pa_raw is None or pb_raw is None or min(pa_raw, pb_raw) < pcfg.min_price:
                    continue
                equity = mark_equity(session)
                # Risk basis: one unit = gross notional whose assumed worst
                # adverse spread move equals the unit risk.
                unit_gross = 2_000.0  # $1k per side per unit (integer-share friendly)
                unit_risk = unit_gross * pcfg.max_adverse_move
                proposal = ProposedTrade(
                    engine="pairs_v1",
                    catalyst_ref=f"{a_sym}/{b_sym}",
                    legs=[OrderLeg(key=_sentinel_key(a_sym), side=Side.BUY, qty=1)],
                    unit_cost=unit_risk / 100.0,
                    unit_max_loss=unit_risk / 100.0,
                    direction=Direction.NEUTRAL,
                    exit_rules=ExitRules(),
                    per_trade_risk_fraction=pcfg.per_trade_risk,
                )
                account = AccountState(
                    equity=equity, cash=cash, buying_power=cash,
                    timestamp=datetime.combine(session, _MARK_TIME),
                )
                decision = rm.size_entry(proposal, account, risk_positions())
                if decision.units <= 0:
                    logger.debug("Pair entry blocked (%s): %s", pair, decision.reason)
                    continue
                per_side = decision.units * unit_gross / 2.0
                sign_a = -1 if direction == "short_A_long_B" else 1
                fill_a = self._fill_price(pa_raw, Side.SELL if sign_a < 0 else Side.BUY)
                fill_b = self._fill_price(pb_raw, Side.BUY if sign_a < 0 else Side.SELL)
                qty_a = int(per_side // fill_a)
                qty_b = int(per_side // fill_b)
                if qty_a < 1 or qty_b < 1:
                    continue
                cash -= self._commission(qty_a + qty_b)
                open_pairs[pair] = _OpenPair(
                    pair=pair, direction=direction, qty_a=qty_a, qty_b=qty_b,
                    entry_a=fill_a, entry_b=fill_b, entry_session=session,
                    risk_dollars=decision.units * unit_risk, entry_z=entry_z,
                )

            # 3. Evaluate signals at today's close -> queue actions for next open.
            for pair, by_day in states.items():
                st = by_day.get(session)
                if st is None:
                    continue
                op = open_pairs.get(pair)
                if op is not None:
                    if not st.active:
                        pending_close[pair] = "decoupled"
                    elif st.zscore is not None and abs(st.zscore) < self._cfg.pairs.z_exit:
                        pending_close[pair] = "z_reverted"
                elif st.active and st.zscore is not None and pair not in pending:
                    if st.zscore > self._cfg.pairs.z_entry:
                        pending[pair] = ("short_A_long_B", st.zscore)
                    elif st.zscore < -self._cfg.pairs.z_entry:
                        pending[pair] = ("long_A_short_B", st.zscore)

            # 4. Accrue borrow on shorts; mark equity; feed breakers.
            if not self._zero_cost:
                for op in open_pairs.values():
                    short_sym = op.pair[0] if op.direction == "short_A_long_B" else op.pair[1]
                    short_qty = op.qty_a if op.direction == "short_A_long_B" else op.qty_b
                    short_px = self._px(self._closes, short_sym, session)
                    if short_px:
                        op.borrow_accrued += short_qty * short_px * self._borrow_apr(short_sym) / 252.0
            equity = mark_equity(session)
            equity_curve[session] = equity
            rm.record_mark(session, equity)

        # Liquidate leftovers at final closes for clean records.
        for pair in list(open_pairs):
            close_pair(open_pairs[pair], sessions[-1], "end_of_backtest")

        return self._build_result(trades, equity_curve, start, end), {
            "stability": self.engine.ledger,
            "trades_on_decoupled": self.trades_on_decoupled,
        }

    def _build_result(self, trades: list[TradeRecord], equity_curve: dict[date, float],
                      start: date, end: date) -> BacktestResult:
        cfg = self._cfg
        equity = pd.Series(list(equity_curve.values()),
                           index=pd.to_datetime(list(equity_curve.keys())), dtype=float)
        daily = equity.pct_change().dropna()
        tstats = m.trade_stats(trades)
        ruin = probability_of_ruin(daily, cfg.backtest.monte_carlo)
        per_pair: dict[str, float] = defaultdict(float)
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
            per_engine_pnl=dict(per_pair), per_engine_trades={
                k: sum(1 for t in trades if t.catalyst_ref == k) for k in per_pair
            },
            trades=trades,
            equity_curve={d.isoformat(): float(v) for d, v in equity_curve.items()},
        )

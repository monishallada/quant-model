"""v8 backtester: systematic index VRP harvest, priced on real chains.

Daily loop. Every ``entry_every_days`` sessions it opens a defined-risk put
credit spread per symbol (subject to concurrency and RiskManager limits), then
marks every open position daily against real NBBO and applies mechanical
exits: take profit at a share of the credit, stop at a multiple of it, and a
mandatory close before expiry.

Sizing runs through the authoritative RiskManager exactly as every prior
campaign did: max loss is (width - credit) at worst-case fills, so the
per-trade cap, portfolio heat cap and cash floor all bind normally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from catalyst.core.config import Config
from catalyst.core.models import (AccountState, BacktestResult, Direction, ExitRules,
                                  OptionKey, OptionRight, OrderLeg, Position, PositionLeg,
                                  ProposedTrade, Side, TradeRecord)
from catalyst.core.tradingcal import add_trading_days, sessions_in_range
from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.index_vrp.engine import IndexVRPEngine, VRPPosition
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)

MARK_TIME = time(15, 45)


@dataclass
class _Open:
    pos: VRPPosition
    entry_liability: float  # what it would have cost to close at entry (= credit received)
    last_liability: float


@dataclass
class VRPResult:
    equity: pd.Series
    trades: list[TradeRecord]
    benchmark: pd.Series
    diagnostics: dict = field(default_factory=dict)

    def report(self, label: str) -> str:
        h = m.headline(self.equity)
        bh = m.headline(self.benchmark)
        wins = [t for t in self.trades if t.pnl > 0]
        wr = len(wins) / len(self.trades) if self.trades else 0.0
        conc = m.concentration(self.trades, 3)
        share = conc["share"]
        lines = [
            f"===== {label} =====",
            f"  AVG MONTHLY RETURN: {h['avg_monthly_return']:+.2%}"
            f"    (SPY benchmark: {bh['avg_monthly_return']:+.2%})",
            f"  median month {h['median_monthly_return']:+.2%} | best {h['best_month']:+.2%} | "
            f"worst {h['worst_month']:+.2%} | months positive {h['pct_months_positive']:.0%}",
            f"  CAGR {h['cagr']:+.2%} | max drawdown {h['max_drawdown']:+.2%} | "
            f"trades {len(self.trades)} | win rate {wr:.1%}",
            f"  profit factor {m.profit_factor(self.trades):.2f} | "
            f"EV/trade ${m.expected_value(self.trades):,.0f}",
        ]
        if share is not None:
            flag = "  *** LOTTERY FLAG ***" if share > 0.5 else ""
            lines.append(f"  top-3 concentration: {share:.0%} of total P&L{flag}")
        lines.append(f"  diagnostics: {self.diagnostics}")
        if len(self.trades) < 100:
            lines.append(f"  *** WARNING: N={len(self.trades)} < 100 ***")
        return "\n".join(lines)


def _risk_position(op: _Open, contracts: int) -> Position:
    key = OptionKey(underlying=op.pos.symbol, expiry=op.pos.expiry,
                    right=OptionRight.PUT, strike=op.pos.short_strike)
    return Position(
        position_id=f"vrp-{op.pos.symbol}-{op.pos.entry_date}-{op.pos.short_strike:g}",
        legs=[PositionLeg(key=key, side=Side.SELL, qty=1)], qty=contracts,
        entry_price=-op.pos.credit,
        entry_time=datetime.combine(op.pos.entry_date, MARK_TIME),
        engine_tag="index_vrp", direction=Direction.NEUTRAL, exit_rules=ExitRules(),
        current_value=-op.last_liability, max_loss=op.pos.max_loss,
    )


def run_index_vrp(cfg: Config, data, benchmark: pd.Series, start: date, end: date,
                  label: str = "v8-index-vrp") -> VRPResult:
    vcfg = cfg.index_vrp
    engine = IndexVRPEngine(vcfg, data)
    rm = RiskManager(cfg.risk)

    expiries = {s: data.list_expirations(s) for s in vcfg.symbols}
    sessions = sessions_in_range(start, end)
    equity = cfg.account.starting_capital
    cash = equity
    open_positions: dict[str, tuple[_Open, int]] = {}
    trades: list[TradeRecord] = []
    curve: dict[date, float] = {}
    blocked = 0

    for i, session in enumerate(sessions):
        # ---- mark and manage open positions ----
        for pid, (op, contracts) in list(open_positions.items()):
            liability = engine.mark(op.pos, session)
            if liability is None:
                continue
            op.last_liability = liability
            captured = (op.pos.credit - liability) / op.pos.credit
            reason = None
            if captured >= vcfg.tp_credit_fraction:
                reason = "take_profit_credit"
            elif liability >= vcfg.stop_credit_multiple * op.pos.credit:
                reason = "stop_loss_credit"
            elif session >= add_trading_days(op.pos.expiry, -vcfg.close_before_expiry_days):
                reason = "expiry_buffer"
            if reason:
                pnl = (op.pos.credit - liability) * contracts * 100.0
                # Closing a credit spread COSTS the liability. The credit was
                # already added to cash at entry, so adding the P&L here would
                # count it twice — which inflates equity even when the trades
                # themselves lose money (profit factor < 1 with a rising curve
                # is the signature of exactly this bug).
                cash -= liability * contracts * 100.0
                trades.append(TradeRecord(
                    position_id=pid, engine="index_vrp",
                    catalyst_ref=f"{op.pos.symbol}:{op.pos.expiry}",
                    underlying=op.pos.symbol, direction=Direction.NEUTRAL,
                    entry_time=datetime.combine(op.pos.entry_date, MARK_TIME),
                    exit_time=datetime.combine(session, MARK_TIME),
                    entry_price=-op.pos.credit, exit_price=-liability,
                    qty=contracts, pnl=pnl, exit_reason=reason, max_qty=contracts))
                del open_positions[pid]

        # ---- open new positions on the entry schedule ----
        if i % vcfg.entry_every_days == 0 and len(open_positions) < vcfg.max_concurrent:
            for symbol in vcfg.symbols:
                if len(open_positions) >= vcfg.max_concurrent:
                    break
                built = engine.propose(symbol, session, expiries.get(symbol, []))
                if built is None:
                    continue
                pos, _chain = built
                key = OptionKey(underlying=symbol, expiry=pos.expiry,
                                right=OptionRight.PUT, strike=pos.short_strike)
                wing = OptionKey(underlying=symbol, expiry=pos.expiry,
                                 right=OptionRight.PUT, strike=pos.long_strike)
                proposal = ProposedTrade(
                    engine="index_vrp", catalyst_ref=f"{symbol}:{pos.expiry}",
                    legs=[OrderLeg(key=key, side=Side.SELL, qty=1),
                          OrderLeg(key=wing, side=Side.BUY, qty=1)],
                    unit_cost=-pos.credit, unit_max_loss=pos.max_loss,
                    direction=Direction.NEUTRAL, exit_rules=ExitRules(),
                    per_trade_risk_fraction=vcfg.per_trade_risk)
                account = AccountState(equity=equity, cash=cash, buying_power=cash,
                                       timestamp=datetime.combine(session, MARK_TIME))
                existing = [_risk_position(o, c) for o, c in open_positions.values()]
                decision = rm.size_entry(proposal, account, existing)
                if decision.units <= 0:
                    blocked += 1
                    continue
                op = _Open(pos=pos, entry_liability=pos.credit, last_liability=pos.credit)
                pid = f"vrp-{symbol}-{session}-{pos.short_strike:g}"
                open_positions[pid] = (op, decision.units)
                cash += pos.credit * decision.units * 100.0  # credit received

        # ---- mark equity: cash plus the liability of open spreads ----
        liabilities = sum(o.last_liability * c * 100.0 for o, c in open_positions.values())
        equity = cash - liabilities
        curve[session] = equity
        rm.record_mark(session, equity)

    eq = pd.Series(curve)
    eq.index = pd.to_datetime(eq.index)
    bench = benchmark.reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * cfg.account.starting_capital
    diagnostics = {"built": engine.built, "no_chain": engine.skipped_no_chain,
                   "no_structure": engine.skipped_no_structure,
                   "thin_credit": engine.skipped_thin_credit, "risk_blocked": blocked}
    return VRPResult(eq, trades, bench, diagnostics)

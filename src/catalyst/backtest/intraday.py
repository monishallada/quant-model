"""IntradayBacktester — the minute-resolution event loop.

Generalized from the archived Kinetic backtester (the only prior minute-level
loop, validated in v3) into the shared engine family: multi-strategy,
multi-position, instrument-aware (shares and options through the same
SimulatedBroker ledger and the same cost truth), with the mandatory honesty
mechanics carried over intact:

- fills through ``costs.CostModel`` only (worse-side + slippage; equities also
  pay their bps; zero-cost twin runs the identical path)
- option marks from the contract's own minute NBBO; a stale quote (> max
  staleness) freezes the mark; a missing exit quote falls back to
  last-good-mid x haircut and is ALWAYS counted as a synthetic fill —
  ``synthetic_fills: N`` in diagnostics is the difference between a result and
  a fabrication (the pandas-alignment bug of v3 was caught by exactly this)
- no overnight exposure: everything force-flattens at ``eod_flatten`` and the
  loop asserts flat before ending the session
- the RiskManager gates every entry against reconciled broker state, and an
  intra-session breaker (mirroring ``risk.daily_loss_halt``) blocks new
  entries after a session drawdown breach

Timestamp discipline (see interfaces/intraday.py): decisions at ts see bars
labeled <= ts-1min and fill at ts — equities at the ts bar's OPEN (the first
tradeable print after the decision), options at the contract NBBO row at ts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import pandas as pd

from catalyst.core.interfaces.intraday import IntradayContext, IntradayStrategy
from catalyst.core.types import (
    EquityKey,
    OrderLeg,
    OptionChain,
    OptionContract,
    OptionKey,
    Order,
    OrderIntent,
    OrderStatus,
    Position,
    Side,
    TradeRecord,
)
from catalyst.core.tradingcal import sessions_in_range
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)

SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


@dataclass(frozen=True)
class IntradayParams:
    """Engine parameters. Config-owned numbers; nothing hidden in the loop."""

    eod_flatten: time = time(15, 58)       # mandatory: no overnight exposure
    equity_half_spread_bps: float = 1.0    # SPY/QQQ-class quote proxy from bars
    option_quote_staleness_min: int = 5    # freeze marks beyond this age
    synthetic_haircut: float = 0.90        # last-good-mid x this when exit quote missing
    decision_every_min: int = 1            # decide every N minutes


@dataclass
class _OpenState:
    strategy: str
    symbol: str
    entry_ts: datetime
    position_id: str
    proposal_rationale: dict = field(default_factory=dict)


@dataclass
class IntradayResult:
    equity: pd.Series                      # one mark per session close
    trades: list[TradeRecord]
    diagnostics: dict


class IntradayBacktester:
    """Drives IntradayStrategy instances through real minute data."""

    def __init__(self, cfg, bars_provider, option_quotes_provider,
                 data=None, params: IntradayParams | None = None) -> None:
        """``bars_provider.get_day(symbol, day) -> DataFrame(open..close)`` ET-indexed;
        ``option_quotes_provider.contract_day(key, day, start, end) -> DataFrame(bid, ask)``.
        Injected so tests run on synthetic frames with zero network."""
        self._cfg = cfg
        self._bars = bars_provider
        self._oq = option_quotes_provider
        self._data = data
        self._p = params or IntradayParams()

    # ------------------------------------------------------------------
    def run(self, strategies: list[IntradayStrategy], start: date, end: date,
            broker_factory) -> IntradayResult:
        rm = RiskManager(self._cfg.risk)
        broker = broker_factory()
        trades: list[TradeRecord] = []
        curve: dict[pd.Timestamp, float] = {}
        diag = {"sessions": 0, "entries": 0, "risk_blocked": 0, "breaker_blocked": 0,
                "unquotable_entries": 0, "synthetic_fills": 0, "stale_mark_minutes": 0,
                "forced_flattens": 0}

        for session in sessions_in_range(start, end):
            self._run_session(session, strategies, broker, rm, trades, diag)
            equity = broker.get_account().equity
            curve[pd.Timestamp(session)] = equity
            rm.record_mark(session, equity)
            diag["sessions"] += 1
            # No-overnight invariant: a position surviving the flatten is a bug,
            # not a policy choice.
            assert not broker.get_positions(), (
                f"positions survived EOD flatten on {session}")

        return IntradayResult(pd.Series(curve).sort_index(), trades, diag)

    # ------------------------------------------------------------------
    def _run_session(self, session, strategies, broker, rm, trades, diag) -> None:
        p = self._p
        universe: set[str] = set()
        for s in strategies:
            universe.update(s.session_universe(session))
        day_bars = {sym: self._bars.get_day(sym, session) for sym in universe}
        day_bars = {k: v for k, v in day_bars.items() if v is not None and not v.empty}
        if not day_bars:
            return

        option_frames: dict[OptionKey, pd.DataFrame] = {}
        open_state: dict[str, _OpenState] = {}      # position_id -> state
        session_open_equity = broker.get_account().equity
        breaker_hit = False

        start_ts = datetime.combine(session, SESSION_OPEN) + timedelta(minutes=1)
        end_ts = datetime.combine(session, SESSION_CLOSE)
        ts = start_ts
        while ts <= end_ts:
            # -- market state at ts ----------------------------------------
            visible = {sym: df[df.index <= ts - timedelta(minutes=1)]
                       for sym, df in day_bars.items()}
            fill_quotes = self._equity_fill_quotes(day_bars, ts)
            mark_quotes = self._equity_mark_quotes(visible)
            chains = self._option_chains_at(option_frames, ts, diag)
            broker.update_market(chains, ts, equity_quotes={**mark_quotes, **fill_quotes})

            # -- intra-session breaker (engine-level mirror of daily halt) --
            equity_now = broker.get_account().equity
            if (not breaker_hit and session_open_equity > 0
                    and equity_now / session_open_equity - 1.0 <= self._cfg.risk.daily_loss_halt):
                breaker_hit = True
                logger.warning("intraday breaker: %.1f%% session drawdown at %s",
                               100 * (equity_now / session_open_equity - 1), ts)

            # -- exits BEFORE entries --------------------------------------
            self._process_exits(broker, open_state, trades, ts, diag)

            # -- entries ---------------------------------------------------
            if ts.time() < p.eod_flatten:
                ctx = IntradayContext(session=session, now=ts, bars=visible,
                                      data=self._data)
                for strat in strategies:
                    for proposal in (strat.on_minute(ctx) or []):
                        sym = proposal.legs[0].key.underlying
                        if any(o.strategy == strat.name and o.symbol == sym
                               for o in open_state.values()):
                            continue        # one position per (strategy, symbol)
                        if breaker_hit:
                            diag["breaker_blocked"] += 1
                            continue
                        self._enter(proposal, strat.name, sym, broker, rm,
                                    option_frames, open_state, ts, diag)

            ts += timedelta(minutes=p.decision_every_min)

        # -- mandatory flatten ---------------------------------------------
        self._flatten_all(broker, open_state, trades,
                          datetime.combine(session, p.eod_flatten), diag,
                          reason="eod_flatten")

    # ------------------------------------------------------------------
    def _equity_fill_quotes(self, day_bars, ts):
        """Executable market at ts: the ts-labeled bar's OPEN ± half spread —
        the first tradeable print after a decision made on bars <= ts-1min."""
        hs = self._p.equity_half_spread_bps / 1e4
        out = {}
        for sym, df in day_bars.items():
            if ts in df.index:
                px = float(df.loc[ts, "open"])
                if px > 0:
                    out[sym] = (px * (1 - hs), px * (1 + hs))
        return out

    def _equity_mark_quotes(self, visible):
        hs = self._p.equity_half_spread_bps / 1e4
        out = {}
        for sym, df in visible.items():
            if not df.empty:
                px = float(df["close"].iloc[-1])
                if px > 0:
                    out[sym] = (px * (1 - hs), px * (1 + hs))
        return out

    def _option_chains_at(self, option_frames, ts, diag):
        """Single-contract chains from minute NBBO (the Kinetic pattern).

        Quote rows at-or-before ts within the staleness cap; a too-stale quote
        freezes the mark (counted) rather than pretending freshness.
        """
        by_underlying: dict[str, list[OptionContract]] = {}
        cap = timedelta(minutes=self._p.option_quote_staleness_min)
        for key, frame in option_frames.items():
            rows = frame[frame.index <= ts]
            if rows.empty:
                continue
            if ts - rows.index[-1] > cap:
                diag["stale_mark_minutes"] += 1
                continue
            bid, ask = float(rows["bid"].iloc[-1]), float(rows["ask"].iloc[-1])
            if ask <= 0:
                continue
            by_underlying.setdefault(key.underlying, []).append(
                OptionContract(key=key, bid=max(bid, 0.0), ask=ask))
        return {u: OptionChain(underlying=u, timestamp=ts, underlying_price=0.0,
                               contracts=cs)
                for u, cs in by_underlying.items()}

    # ------------------------------------------------------------------
    def _enter(self, proposal, strat_name, sym, broker, rm, option_frames,
               open_state, ts, diag) -> None:
        # Pull minute NBBO for any option legs we have not seen yet, so the
        # fill is priced from the contract's OWN market at ts, never a stale
        # EOD snapshot.
        for leg in proposal.legs:
            if isinstance(leg.key, OptionKey) and leg.key not in option_frames:
                frame = self._oq.contract_day(leg.key, ts.date(),
                                              SESSION_OPEN, SESSION_CLOSE)
                option_frames[leg.key] = frame if frame is not None else pd.DataFrame()

        chains = self._option_chains_at(option_frames, ts, diag)
        # refresh broker market so the fill sees the just-fetched contract
        broker.update_market(chains, ts, equity_quotes=None)

        account = broker.get_account()
        decision = rm.size_entry(proposal, account, broker.get_positions())
        if decision.units <= 0:
            diag["risk_blocked"] += 1
            return
        # Order legs carry TOTAL quantities; the broker derives units as the
        # gcd. Proposal legs are per-unit ratios, so scale by the risk gate's
        # sizing decision here.
        sized_legs = [type(leg)(key=leg.key, side=leg.side,
                                qty=leg.qty * decision.units)
                      for leg in proposal.legs]
        order = Order(legs=sized_legs,
                      limit_price=proposal.unit_cost, intent=OrderIntent.OPEN,
                      direction=proposal.direction, exit_rules=proposal.exit_rules,
                      tag=f"{strat_name}:{proposal.catalyst_ref}",
                      max_loss=proposal.unit_max_loss)
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED:
            diag["unquotable_entries"] += 1
            return
        pos = broker.get_positions()[-1]
        open_state[pos.position_id] = _OpenState(
            strategy=strat_name, symbol=sym, entry_ts=ts,
            position_id=pos.position_id,
            proposal_rationale=dict(proposal.rationale))
        diag["entries"] += 1

    # ------------------------------------------------------------------
    def _process_exits(self, broker, open_state, trades, ts, diag) -> None:
        for pos in list(broker.get_positions()):
            st = open_state.get(pos.position_id)
            if st is None:
                continue
            rules = pos.exit_rules
            reason = None
            if rules.close_by_time is not None and ts.time() >= rules.close_by_time:
                reason = "close_by_time"
            elif (rules.max_hold_minutes is not None
                  and (ts - st.entry_ts) >= timedelta(minutes=rules.max_hold_minutes)):
                reason = "max_hold_minutes"
            elif (rules.use_stops and rules.stop_loss_pct is not None
                  and pos.entry_price > 0 and pos.current_value > 0
                  and pos.current_value / pos.entry_price - 1.0 <= rules.stop_loss_pct):
                reason = "stop_loss"
            elif (rules.use_stops and rules.trail_stop_pct is not None
                  and pos.high_water_value > 0 and pos.current_value > 0
                  and pos.current_value / pos.high_water_value - 1.0
                      <= -abs(rules.trail_stop_pct)):
                reason = "trail_stop"
            if reason:
                self._close(broker, pos, st, trades, ts, diag, reason)
                open_state.pop(pos.position_id, None)

    def _flatten_all(self, broker, open_state, trades, ts, diag, reason) -> None:
        for pos in list(broker.get_positions()):
            st = open_state.get(pos.position_id) or _OpenState(
                strategy=pos.engine_tag, symbol=pos.underlying,
                entry_ts=pos.entry_time, position_id=pos.position_id)
            self._close(broker, pos, st, trades, ts, diag, reason, force=True)
            open_state.pop(pos.position_id, None)
            diag["forced_flattens"] += 1

    def _close(self, broker, pos: Position, st: _OpenState, trades, ts, diag,
               reason: str, force: bool = False) -> None:
        # Capture BEFORE the close: the broker decrements pos.qty to zero on a
        # full close, so P&L computed afterwards would silently read as $0 —
        # a fabricated break-even on every single trade.
        qty, entry_price, multiplier = pos.qty, pos.entry_price, pos.multiplier
        legs = [OrderLeg(key=leg.key,
                         side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                         qty=leg.qty * qty) for leg in pos.legs]
        order = Order(legs=legs, intent=OrderIntent.CLOSE,
                      position_id=pos.position_id, limit_price=0.0,
                      direction=pos.direction, tag=f"{st.strategy}:close")
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED and force:
            # Missing exit quote on a mandatory flatten: last-good-mid x
            # haircut, ALWAYS flagged. Silence here is how backtests lie.
            diag["synthetic_fills"] += 1
            value = pos.current_value * self._p.synthetic_haircut
            broker.force_close(pos.position_id, value)
            result_price = value
        elif result.status is not OrderStatus.FILLED:
            return                                  # try again next minute
        else:
            result_price = result.avg_fill_price or pos.current_value
        pnl = (result_price - entry_price) * qty * multiplier
        trades.append(TradeRecord(
            position_id=pos.position_id, engine=st.strategy,
            catalyst_ref=str(st.proposal_rationale.get("ref", st.symbol)),
            underlying=st.symbol, direction=pos.direction,
            entry_time=st.entry_ts, exit_time=ts,
            entry_price=entry_price, exit_price=result_price,
            qty=qty, pnl=pnl, exit_reason=reason, max_qty=qty))

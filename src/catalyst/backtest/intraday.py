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
from catalyst.exits.manager import evaluate_intraday_exits
from catalyst.core.tradingcal import session_close_time, sessions_in_range
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
    catalyst_ref: str = ""
    proposal_rationale: dict = field(default_factory=dict)


@dataclass
class IntradayResult:
    equity: pd.Series                      # one mark per session close
    trades: list[TradeRecord]
    diagnostics: dict


class _GuardedData:
    """The lookahead wall for the raw data handle.

    AUDIT MEDIUM: ctx.data was the raw DataSource — at a 10:00 decision a
    strategy could request the SAME session's 15:45 chain (or next week's).
    Chains are now restricted to strictly-prior sessions; listing expirations
    is timeless metadata and passes through.
    """

    def __init__(self, data, session):
        self._data, self._session = data, session

    def list_expirations(self, symbol):
        return self._data.list_expirations(symbol)

    def get_chain(self, symbol, when, **kw):
        if when.date() >= self._session:
            raise LookaheadError(
                f"strategy requested chain as-of {when} during session "
                f"{self._session} — only strictly-prior sessions are visible")
        return self._data.get_chain(symbol, when, **kw)


class LookaheadError(RuntimeError):
    """A strategy tried to read data from the present or future."""


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
        # AUDIT MEDIUM: set iteration order is per-process random — a strategy
        # scanning ctx.bars and taking the first qualifying symbol produced
        # different trades for identical inputs. Sorted = deterministic.
        day_bars = {sym: self._bars.get_day(sym, session) for sym in sorted(universe)}
        missing = [k for k, v in day_bars.items() if v is None or v.empty]
        if missing:
            # AUDIT HIGH: silence here means an invisibly mutilated dataset.
            diag["missing_bar_days"] = diag.get("missing_bar_days", 0) + len(missing)
        day_bars = {k: v for k, v in day_bars.items() if v is not None and not v.empty}
        if not day_bars:
            diag["skipped_sessions_no_bars"] = diag.get("skipped_sessions_no_bars", 0) + 1
            return

        option_frames: dict[OptionKey, pd.DataFrame] = {}
        open_state: dict[str, _OpenState] = {}      # position_id -> state
        session_open_equity = broker.get_account().equity
        breaker_hit = False

        # AUDIT HIGH: honor early closes (13:00 half days). Everything —
        # the loop end, the flatten, and quote visibility — clamps to the
        # session's REAL close; otherwise the engine trades forward-filled
        # phantom quotes for up to 3 hours on ~4 sessions/year.
        real_close = session_close_time(session)
        end_ts = datetime.combine(session, min(SESSION_CLOSE.replace(), real_close))
        flatten_at = min(
            datetime.combine(session, p.eod_flatten),
            end_ts - timedelta(minutes=2),
        )
        start_ts = datetime.combine(session, SESSION_OPEN) + timedelta(minutes=1)
        ts = start_ts
        while ts <= end_ts:
            # -- market state at ts ----------------------------------------
            visible = {sym: df[df.index <= ts - timedelta(minutes=1)]
                       for sym, df in day_bars.items()}
            fill_quotes = self._equity_fill_quotes(day_bars, ts)
            mark_quotes = self._equity_mark_quotes(visible)
            chains = self._option_chains_at(option_frames, ts, diag)
            # marks from visible bars; FILLS only from the current bar — merging
            # let a missing ts bar fill at a stale mark (audit D-023)
            broker.update_market(chains, ts, equity_quotes=mark_quotes,
                                 equity_fill_quotes=fill_quotes)

            # -- intra-session breaker (engine-level mirror of daily halt) --
            equity_now = broker.get_account().equity
            if (not breaker_hit and session_open_equity > 0
                    and equity_now / session_open_equity - 1.0 <= self._cfg.risk.daily_loss_halt):
                breaker_hit = True
                logger.warning("intraday breaker: %.1f%% session drawdown at %s",
                               100 * (equity_now / session_open_equity - 1), ts)

            # -- mandatory flatten happens IN the loop at its true time ------
            # (AUDIT LOW: it previously executed after the loop against the
            # 16:00 market state while stamped 15:58 — two minutes of
            # post-"flatten" price action leaked into the fill.)
            if ts >= flatten_at:
                self._flatten_all(broker, open_state, trades, ts, diag,
                                  reason="eod_flatten")

            # -- exits BEFORE entries --------------------------------------
            self._process_exits(broker, open_state, trades, ts, diag)

            # -- entries ---------------------------------------------------
            if ts < flatten_at:
                ctx = IntradayContext(session=session, now=ts, bars=visible,
                                      data=_GuardedData(self._data, session) if self._data is not None else None,
                                      option_quote=self._make_quote_fn(
                                          option_frames, session, ts, diag))
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
        # refresh market state AT the flatten timestamp (audit D-184: the
        # flatten previously priced from whatever minute the loop last saw)
        visible = {sym: df[df.index <= flatten_at - timedelta(minutes=1)]
                   for sym, df in day_bars.items()}
        broker.update_market(
            self._option_chains_at(option_frames, flatten_at, diag),
            flatten_at,
            equity_quotes=self._equity_mark_quotes(visible),
            equity_fill_quotes=self._equity_fill_quotes(day_bars, flatten_at))
        self._flatten_all(broker, open_state, trades, flatten_at, diag,
                          reason="eod_flatten")

    # ------------------------------------------------------------------
    def _make_quote_fn(self, option_frames, session, ts, diag):
        """Read-only quote access for strategy SELECTION.

        Same leak discipline as bars: only quotes at or before ts-1min are
        visible to a decision at ts. First touch of a contract fetches its
        whole cached day-frame (0.6s cold, free warm) — counted, because
        request volume is a real budget."""
        from datetime import timedelta as _td

        def quote(key):
            if key not in option_frames:
                frame = self._oq.contract_day(key, session, SESSION_OPEN, SESSION_CLOSE)
                if frame is not None and not frame.empty:
                    close_dt = datetime.combine(session, session_close_time(session))
                    frame = frame[frame.index <= close_dt]
                option_frames[key] = frame if frame is not None else pd.DataFrame()
                diag["quote_fetches"] = diag.get("quote_fetches", 0) + 1
            frame = option_frames[key]
            if frame.empty:
                return None
            rows = frame[frame.index <= ts - _td(minutes=1)]
            if rows.empty:
                return None
            bid, ask = float(rows["bid"].iloc[-1]), float(rows["ask"].iloc[-1])
            return (bid, ask) if ask > 0 else None
        return quote

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

    def _option_chains_at(self, option_frames, ts, diag, count: bool = True):
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
                if count:
                    diag["stale_mark_minutes"] += 1
                continue
            bid, ask = float(rows["bid"].iloc[-1]), float(rows["ask"].iloc[-1])
            # AUDIT MEDIUM: NaN defeats every guard (all comparisons False) —
            # a NaN quote "fills", cash goes NaN, and no counter increments.
            if bid != bid or ask != ask or ask <= 0:
                if count:
                    diag["nan_quotes_dropped"] = diag.get("nan_quotes_dropped", 0) + 1
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
                if frame is not None and not frame.empty:
                    close_dt = datetime.combine(ts.date(), session_close_time(ts.date()))
                    frame = frame[frame.index <= close_dt]
                option_frames[leg.key] = frame if frame is not None else pd.DataFrame()

        # count=False: the loop already counted this tick's staleness; a
        # second pass double-counted every entry minute (audit D-185)
        chains = self._option_chains_at(option_frames, ts, diag, count=False)
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
        pid = result.position_id or ""
        if not pid:
            raise RuntimeError("broker filled an order without naming the "
                               "position (audit D-080)")
        open_state[pid] = _OpenState(
            strategy=strat_name, symbol=sym, entry_ts=ts,
            position_id=pid,
            catalyst_ref=proposal.catalyst_ref,
            proposal_rationale=dict(proposal.rationale))
        diag["entries"] += 1

    # ------------------------------------------------------------------
    def _process_exits(self, broker, open_state, trades, ts, diag) -> None:
        for pos in list(broker.get_positions()):
            st = open_state.get(pos.position_id)
            if st is None:
                continue
            # single exit truth: exits/manager interprets the rules (D-091)
            hit = evaluate_intraday_exits(pos, st.entry_ts, ts)
            if hit is not None:
                # AUDIT HIGH: pop state ONLY on a filled close. Popping on a
                # rejected close created a zombie — the exit never retried,
                # the position rode to the flatten, and the dedup (which reads
                # open_state) let a SECOND concurrent position open: doubled
                # exposure the risk gate never approved.
                if self._close(broker, pos, st, trades, ts, diag, hit.reason):
                    open_state.pop(pos.position_id, None)

    def _entry_commission_for(self, legs, qty: int) -> float:
        """Entry-side commission from CAPTURED qty — never from pos.qty, which
        the broker zeroes on a full close."""
        from catalyst.core.types import EquityKey
        per = self._cfg.execution.commissions.per_contract
        return sum(per * leg.qty * qty for leg in legs
                   if not isinstance(leg.key, EquityKey))

    def _flatten_all(self, broker, open_state, trades, ts, diag, reason) -> None:
        for pos in list(broker.get_positions()):
            st = open_state.get(pos.position_id) or _OpenState(
                strategy=pos.engine_tag, symbol=pos.underlying,
                entry_ts=pos.entry_time, position_id=pos.position_id)
            self._close(broker, pos, st, trades, ts, diag, reason, force=True)
            open_state.pop(pos.position_id, None)   # force path always settles
            diag["forced_flattens"] += 1

    def _close(self, broker, pos: Position, st: _OpenState, trades, ts, diag,
               reason: str, force: bool = False) -> bool:
        """Returns True when the position actually left the book."""
        # Capture BEFORE the close: the broker decrements pos.qty to zero on a
        # full close, so P&L computed afterwards would silently read as $0 —
        # a fabricated break-even on every single trade.
        qty, entry_price, multiplier = pos.qty, pos.entry_price, pos.multiplier
        # VERIFIER CATCH: _entry_commission(pos) after place_order read
        # pos.qty == 0 (the broker zeroes it on a full close) — the entry-side
        # commission was always $0 and records missed it by exactly that much.
        entry_comm = self._entry_commission_for(pos.legs, qty)
        legs = [OrderLeg(key=leg.key,
                         side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                         qty=leg.qty * qty) for leg in pos.legs]
        order = Order(legs=legs, intent=OrderIntent.CLOSE,
                      position_id=pos.position_id,
                      # order-level convention: close cash direction = -value (D-106)
                      limit_price=-pos.current_value,
                      direction=pos.direction, tag=f"{st.strategy}:close")
        result = broker.place_order(order)
        if result.status is not OrderStatus.FILLED and force:
            # Missing exit quote on a mandatory flatten: last-good-mid x
            # haircut, ALWAYS flagged. Silence here is how backtests lie.
            diag["synthetic_fills"] += 1
            # AUDIT LOW->fixed: the haircut must be ADVERSE for BOTH signs. A
            # credit position's mark is negative; multiplying by 0.9 SHRANK the
            # liability — a 10% gift on exactly the fills that are already
            # fabricated. Long: receive less. Short: pay more.
            h = self._p.synthetic_haircut
            value = pos.current_value * (h if pos.current_value >= 0 else (2.0 - h))
            # broker.force_close charges the close commission in CASH (P0 fix
            # for audit D-199); the record must carry the same charge or cash
            # and records desync by exactly that amount (audit D-187)
            from catalyst.core.types import EquityKey as _EK
            contracts = sum(leg.qty for leg in pos.legs
                            if not isinstance(leg.key, _EK)) * pos.qty
            synthetic_comm = self._cfg.execution.commissions.per_contract * contracts
            broker.force_close(pos.position_id, value)
            result_price = value
        elif result.status is not OrderStatus.FILLED:
            diag["exit_retries"] = diag.get("exit_retries", 0) + 1
            return False                            # try again next minute
        else:
            result_price = (result.avg_fill_price
                            if result.avg_fill_price is not None
                            else pos.current_value)
        # AUDIT MEDIUM: the LEDGER charged commissions on both sides but the
        # TradeRecord ignored them — records must reconcile with cash.
        close_comm = getattr(result, "commission", 0.0) or 0.0
        close_comm += locals().get("synthetic_comm", 0.0)
        pnl = (result_price - entry_price) * qty * multiplier - close_comm - entry_comm
        trades.append(TradeRecord(
            position_id=pos.position_id, engine=st.strategy,
            catalyst_ref=st.catalyst_ref or st.symbol,
            underlying=st.symbol,
            direction=pos.direction or Direction.NEUTRAL,
            entry_time=st.entry_ts, exit_time=ts,
            entry_price=entry_price, exit_price=result_price,
            qty=qty, pnl=pnl, exit_reason=reason, max_qty=qty))
        return True

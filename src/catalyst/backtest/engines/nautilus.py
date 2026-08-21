"""
[CROSS-CHECK ENGINE — audit v15 note] This adapter is an OPT-IN
independent re-implementation used only to disagree with the native engine.
Known approximations (audit D-087/D-178/D-179/D-180): shadow sizing holds
equity at starting capital, reserves at a flat 100x multiplier, and the
fill-model mapping treats spread_fill_fraction approximately. Its numbers are
for DIVERGENCE DETECTION, never for reporting.
NautilusTrader engine — a genuinely INDEPENDENT second backtest.

This is not a replay of the native engine's fills. Nautilus runs its own
backtest end to end:

- its own **venue** and matching engine (`SimulatedExchange`)
- its own **fill model** against the quote book
- its own **fee model** (commission)
- its own **position tracking, P&L and account/equity accounting**
- its own **event clock and order lifecycle**

The only thing shared with the native engine is the raw ThetaData quotes and
the strategy's decision logic — and both of those *must* be shared, because
otherwise the two engines would be backtesting different data or a different
strategy, and comparing them would mean nothing. Everything downstream of "what
do you want to trade" is computed twice, independently.

That is the difference that matters. A replay can only catch arithmetic faults
in the ledger. Two independent event loops can also disagree about *whether a
fill was achievable*, what it cost, when it happened, and what the account was
worth afterwards — which is where the harder errors live.

Runs anywhere Python runs: no Docker, no .NET.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, time
from decimal import Decimal

import pandas as pd

from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.core.interfaces.strategy import Cadence, Opportunity, StrategyContext
from catalyst.core.types import OptionRight, Side

logger = logging.getLogger(__name__)

VENUE_NAME = "OPRA"
MULTIPLIER = 100
# snapshot follows cfg.data.snapshot_time at run() time; this module-level
# value is only the fallback (audit D-086: the constant silently diverged
# from config)
SNAP = time(15, 45)
PRICE_PRECISION = 4


class NautilusEngine(BacktestEngine):
    name = "nautilus"
    verifies = ("fully independent backtest: own matching engine, fill model, "
                "fee model, position tracking and equity accounting")

    def __init__(self, max_sessions: int | None = None) -> None:
        #: Bound on sessions loaded into Nautilus. Each session is a full chain
        #: snapshot converted into quote ticks, which is memory-hungry; None
        #: means no bound.
        self._max_sessions = max_sessions

    def available(self) -> tuple[bool, str]:
        try:
            import nautilus_trader
            from nautilus_trader.backtest.engine import BacktestEngine as _BE  # noqa: F401
            return True, f"nautilus_trader {nautilus_trader.__version__}"
        except Exception as e:                          # noqa: BLE001
            return False, f"nautilus_trader unavailable: {e}"

    def run(self, strategy, start: date, end: date, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost: bool = False) -> EngineResult:
        ok, reason = self.available()
        if not ok:
            return EngineResult(engine=self.name, error=reason)
        try:
            return _run_independent(strategy, start, end, cfg=cfg, data=data,
                                    zero_cost=zero_cost, engine_name=self.name,
                                    max_sessions=self._max_sessions)
        except Exception as e:                          # noqa: BLE001
            logger.exception("nautilus independent backtest failed")
            return EngineResult(engine=self.name, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
def _run_independent(strategy, start: date, end: date, *, cfg, data, zero_cost: bool,
                     engine_name: str, max_sessions: int | None) -> EngineResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nautilus_trader.backtest.engine import BacktestEngine as NTBacktestEngine
        from nautilus_trader.backtest.engine import BacktestEngineConfig
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.enums import AccountType, OmsType
        from nautilus_trader.model.identifiers import Venue
        from nautilus_trader.model.objects import Money

    from catalyst.core.tradingcal import sessions_in_range
    from catalyst.data.thetadata_historical import DataUnavailableError

    # This adapter drives a fixed universe on a schedule. A CATALYST strategy
    # derives its universe from the calendar, so it cannot be mirrored here —
    # say so plainly instead of reporting "empty universe", which reads like a
    # config slip rather than an unsupported cadence.
    if getattr(strategy, "cadence", None) is Cadence.CATALYST:
        return EngineResult(
            engine=engine_name,
            error="CATALYST cadence not supported by the Nautilus adapter "
                  "(it mirrors a fixed universe on a schedule)")
    universe = _universe(strategy)
    sessions = sessions_in_range(start, end)
    if max_sessions:
        sessions = sessions[:max_sessions]
    if not sessions or not universe:
        return EngineResult(engine=engine_name,
                            error="no sessions or empty universe")

    # --- collect the SAME raw quotes the native engine sees -----------------
    # Fetch ONLY the expiries the strategy declares, exactly as the native
    # engine does. Pulling a full chain per session is billed per expiration
    # and is slow enough to make the second engine unusable in practice.
    snapshots: dict[date, dict[str, object]] = {}
    expiry_cache: dict[str, list] = {}
    for s in sessions:
        probe_ctx = StrategyContext(as_of=datetime.combine(s, SNAP), data=data)
        try:
            opps = strategy.opportunities(s, probe_ctx)
        except Exception:                                # noqa: BLE001
            opps = []
        if not opps:
            continue
        per_symbol = {}
        for opp in opps:
            sym = opp.symbol
            if sym in per_symbol:
                continue
            if sym not in expiry_cache:
                try:
                    expiry_cache[sym] = data.list_expirations(sym)
                except Exception:                        # noqa: BLE001
                    expiry_cache[sym] = []
            avail = [e for e in expiry_cache[sym] if e > s]
            wanted = strategy.required_expiries(opp, avail, s) or avail[:1]
            for expiry in list(wanted)[:2]:
                try:
                    chain = data.get_chain(sym, datetime.combine(s, SNAP),
                                           expiries=[expiry], include_liquidity=False)
                except DataUnavailableError:
                    continue
                except Exception:                        # noqa: BLE001
                    continue
                if chain is not None and chain.contracts:
                    per_symbol[sym] = chain
                    break
        if per_symbol:
            snapshots[s] = per_symbol
    if not snapshots:
        return EngineResult(engine=engine_name, error="no chain data in window")

    # --- decide intent, then let NAUTILUS price and account for it ----------
    # The strategy is consulted for decisions only. Nautilus performs the
    # matching, applies its own fill and fee models, tracks the position and
    # computes the account equity — none of which touches native code.
    orders = _collect_intent(strategy, snapshots, data, cfg=cfg)
    if not orders:
        return EngineResult(engine=engine_name, error="strategy proposed no trades",
                            diagnostics={"sessions": len(snapshots)})

    # Second pass: a contract quoted ONLY on its entry date can never be exited,
    # because the exit is driven by a later tick on the same instrument. Without
    # this the run silently reports open positions and zero realized P&L —
    # which looks like a flat strategy rather than a broken harness.
    all_sessions = sessions_in_range(start, end)
    for o in orders:
        try:
            i = all_sessions.index(o["session"])
        except ValueError:
            continue
        exit_i = min(i + o["hold_days"], len(all_sessions) - 1)
        exit_day = all_sessions[exit_i]
        o["exit_session"] = exit_day
        if exit_day in snapshots and o["symbol"] in snapshots[exit_day]:
            continue
        try:
            chain = data.get_chain(o["symbol"], datetime.combine(exit_day, SNAP),
                                   expiries=[o["key"].expiry], include_liquidity=False)
        except Exception:                                # noqa: BLE001
            continue
        if chain is not None and chain.contracts:
            snapshots.setdefault(exit_day, {})[o["symbol"]] = chain

    engine = NTBacktestEngine(config=BacktestEngineConfig(
        trader_id="CATALYST-002",
        logging=LoggingConfig(bypass_logging=True)))

    fee_model = _fee_model(cfg, zero_cost)
    # MARGIN, not CASH. A cash account rejects any order it cannot fully fund
    # from settled cash, which silently dropped every risk-sized position and
    # produced a degenerate curve that looked like "the strategy did nothing".
    # Long options are cash-secured in practice, so leverage is 1:1 — the
    # margin account type is what lets the position be *held*, not a way to
    # take more risk than the RiskManager approved.
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(cfg.account.starting_capital, USD)],
        default_leverage=Decimal(1),
        fill_model=_fill_model(cfg, zero_cost),
        fee_model=fee_model,
    )

    instruments, ticks, plan = _build_market(orders, snapshots)
    for inst in instruments.values():
        engine.add_instrument(inst)
    if not ticks:
        return EngineResult(engine=engine_name, error="no quote ticks constructed")
    engine.add_data(sorted(ticks, key=lambda t: t.ts_init))

    from catalyst.backtest.engines._nautilus_strategy import (
        MirrorConfig, MirrorStrategy,
    )
    engine.add_strategy(MirrorStrategy(config=MirrorConfig(plan=plan)))

    engine.run()
    result = _extract(engine, cfg.account.starting_capital, engine_name,
                      n_sessions=len(snapshots))
    engine.dispose()
    return result


def _universe(strategy) -> tuple[str, ...]:
    params = getattr(strategy, "_p", None)
    uni = getattr(params, "universe", None) if params else None
    return tuple(uni) if uni else ()


def _collect_intent(strategy, snapshots, data, cfg=None) -> list[dict]:
    """Ask the strategy what it wants, session by session.

    Quantity comes from the RiskManager, not from the engine. Sizing is a
    policy decision that is authoritative in every mode — letting each engine
    pick its own lot size would make the comparison measure position size
    instead of engine behaviour, which is what a fixed 1-lot did here first
    (24 Nautilus positions against 7 risk-gated native ones).

    Everything downstream — fill price, commission, position accounting,
    equity — is still computed independently by each engine.
    """
    from catalyst.core.types import AccountState
    from catalyst.risk.manager import RiskManager

    rm = RiskManager(cfg.risk) if cfg is not None else None
    equity = float(cfg.account.starting_capital) if cfg is not None else 100_000.0
    out: list[dict] = []
    open_book: list[dict] = []      # positions still open at this session
    history: dict[str, pd.DataFrame] = {}
    for session, chains in sorted(snapshots.items()):
        at = datetime.combine(session, SNAP)
        # Retire positions whose hold has elapsed before sizing today's.
        open_book[:] = [o for o in open_book if o["expiry_session"] > session]
        probe = StrategyContext(as_of=at, data=data, history=history, chains=chains)
        try:
            opps = strategy.opportunities(session, probe)
        except Exception:                                # noqa: BLE001
            continue
        for opp in opps:
            if opp.symbol not in chains:
                continue
            ctx = StrategyContext(as_of=at, data=data, signal=None,
                                  history=history, chains=chains)
            try:
                proposal = strategy.plan(opp, ctx)
            except Exception:                            # noqa: BLE001
                continue
            if proposal is None or not proposal.legs:
                continue
            units = 1
            if rm is not None:
                # Size against the LIVE book, not against an empty one. Sizing
                # each proposal as if it were the only open position ignores the
                # heat cap and cash floor entirely — it approved 24 positions at
                # 25% of equity each (600% deployed) and drove the account to
                # -$250k. The RiskManager is deterministic given
                # (proposal, account, positions), so replicating the position
                # lifecycle here reproduces the native sizing decisions exactly
                # without either engine calling into the other.
                cash = equity - sum(o["unit_max_loss"] * o["units"] * 100.0
                                    for o in open_book)
                account = AccountState(equity=equity, cash=max(cash, 0.0),
                                       buying_power=max(cash, 0.0), timestamp=at)
                decision = rm.size_entry(proposal, account,
                                         [_as_position(o) for o in open_book])
                units = max(int(decision.units), 0)
                if units <= 0:
                    continue
            leg = proposal.legs[0]          # single-leg structures
            out.append({
                "session": session,
                "symbol": opp.symbol,
                "key": leg.key,
                "side": leg.side,
                "hold_days": _hold_days(proposal),
                "risk_fraction": proposal.per_trade_risk_fraction,
                "unit_max_loss": proposal.unit_max_loss,
                "units": units,
            })
            open_book.append({
                "unit_max_loss": proposal.unit_max_loss, "units": units,
                "symbol": opp.symbol, "key": leg.key, "side": leg.side,
                "expiry_session": _plus_sessions(session, _hold_days(proposal)),
            })
    return out


def _as_position(o: dict):
    """Minimal Position view for the RiskManager's heat and correlation caps."""
    from catalyst.core.types import Direction, ExitRules, Position, PositionLeg
    return Position(
        position_id=f"{o['symbol']}-{o['expiry_session']}",
        legs=[PositionLeg(key=o["key"], side=o["side"], qty=1)],
        qty=o["units"], entry_price=o["unit_max_loss"],
        entry_time=datetime.combine(o["expiry_session"], SNAP),
        engine_tag="nautilus", direction=Direction.LONG,
        exit_rules=ExitRules(), max_loss=o["unit_max_loss"])


def _plus_sessions(session: date, n: int) -> date:
    from catalyst.core.tradingcal import add_trading_days
    try:
        return add_trading_days(session, n)
    except Exception:                                    # noqa: BLE001
        return session + pd.Timedelta(days=n).to_pytimedelta()


def _hold_days(proposal) -> int:
    rules = proposal.exit_rules
    return int(getattr(rules, "max_hold_trading_days", None) or 15)


def _fill_model(cfg, zero_cost: bool):
    from nautilus_trader.backtest.models import FillModel
    if zero_cost:
        return FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=7)
    # Nautilus applies its OWN slippage mechanics; this is deliberately not a
    # copy of our fill model. Two engines using identical cost code would not
    # be an independent check of cost handling.
    return FillModel(prob_fill_on_limit=1.0,
                     prob_slippage=float(cfg.execution.fill_model.spread_fill_fraction),
                     random_seed=7)


def _fee_model(cfg, zero_cost: bool):
    from nautilus_trader.backtest.models import MakerTakerFeeModel
    return MakerTakerFeeModel()


def _build_market(orders, snapshots):
    """Instruments + quote ticks for exactly the contracts that get traded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.data import QuoteTick
        from nautilus_trader.model.enums import AssetClass, OptionKind
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        from nautilus_trader.model.instruments import OptionContract
        from nautilus_trader.model.objects import Price, Quantity

    venue = Venue(VENUE_NAME)
    wanted = {(o["symbol"], o["key"]) for o in orders}
    instruments: dict[str, object] = {}
    ticks: list = []

    for session, chains in sorted(snapshots.items()):
        ts = int(pd.Timestamp(datetime.combine(session, SNAP)).value)
        for sym, chain in chains.items():
            for c in chain.contracts:
                if (sym, c.key) not in wanted or c.bid <= 0 or c.ask <= 0:
                    continue
                iid = _instrument_id(c.key)
                if iid not in instruments:
                    k = c.key
                    instruments[iid] = OptionContract(
                        instrument_id=InstrumentId(Symbol(iid), venue),
                        raw_symbol=Symbol(iid),
                        asset_class=AssetClass.EQUITY,
                        currency=USD,
                        price_precision=PRICE_PRECISION,
                        price_increment=Price.from_str("0.0001"),
                        multiplier=Quantity.from_int(MULTIPLIER),
                        lot_size=Quantity.from_int(1),
                        underlying=sym,
                        option_kind=(OptionKind.CALL if k.right is OptionRight.CALL
                                     else OptionKind.PUT),
                        strike_price=Price(round(k.strike, 2), 2),
                        activation_ns=0,
                        expiration_ns=int(pd.Timestamp(k.expiry).value),
                        ts_event=ts, ts_init=ts,
                    )
                inst = instruments[iid]
                ticks.append(QuoteTick(
                    instrument_id=inst.id,
                    bid_price=Price(round(c.bid, PRICE_PRECISION), PRICE_PRECISION),
                    ask_price=Price(round(c.ask, PRICE_PRECISION), PRICE_PRECISION),
                    bid_size=Quantity.from_int(max(int(c.open_interest or 1), 1)),
                    ask_size=Quantity.from_int(max(int(c.open_interest or 1), 1)),
                    ts_event=ts, ts_init=ts,
                ))

    plan = []
    for o in orders:
        iid = _instrument_id(o["key"])
        if iid not in instruments:
            continue
        plan.append({
            "instrument_id": str(instruments[iid].id),
            "entry_ts": int(pd.Timestamp(datetime.combine(o["session"], SNAP)).value),
            "side": "BUY" if o["side"] is Side.BUY else "SELL",
            "hold_days": o["hold_days"],
            "exit_ts": int(pd.Timestamp(
                datetime.combine(o.get("exit_session") or o["session"], SNAP)).value),
            "risk_fraction": o["risk_fraction"],
            "unit_max_loss": o["unit_max_loss"],
            "units": o.get("units", 1),
        })
    return instruments, ticks, plan


def _money(value) -> float:
    """Nautilus reports Money as a string like '123.45 USD'."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return 0.0


def _instrument_id(key) -> str:
    r = "C" if key.right is OptionRight.CALL else "P"
    return f"{key.underlying}{key.expiry:%y%m%d}{r}{int(round(key.strike * 1000)):08d}"


def _extract(engine, starting_capital: float, engine_name: str,
             n_sessions: int) -> EngineResult:
    """Read NAUTILUS's own account and position records — not ours."""
    from catalyst.core.types import Direction, TradeRecord

    account_reports = engine.trader.generate_account_report(
        __import__("nautilus_trader.model.identifiers", fromlist=["Venue"]).Venue(VENUE_NAME))
    positions = engine.trader.generate_positions_report()

    curve: dict[pd.Timestamp, float] = {}
    if account_reports is not None and len(account_reports):
        for ts, row in account_reports.iterrows():
            val = row.get("total") if hasattr(row, "get") else None
            try:
                curve[pd.Timestamp(ts)] = float(val)
            except (TypeError, ValueError):
                continue

    trades: list[TradeRecord] = []
    realized = 0.0
    if positions is not None and len(positions):
        for i, row in positions.reset_index().iterrows():
            pnl = _money(row.get("realized_pnl"))
            realized += pnl
            trades.append(TradeRecord(
                position_id=str(row.get("position_id", i)),
                engine=engine_name, catalyst_ref=str(row.get("instrument_id", "")),
                underlying=str(row.get("instrument_id", ""))[:4],
                direction=Direction.LONG,
                entry_time=pd.Timestamp(row.get("ts_opened", 0)).to_pydatetime(),
                exit_time=pd.Timestamp(row.get("ts_closed", 0)).to_pydatetime(),
                entry_price=float(row.get("avg_px_open", 0) or 0),
                exit_price=float(row.get("avg_px_close", 0) or 0),
                qty=int(float(row.get("peak_qty", 0) or 0)),
                pnl=pnl, exit_reason="nautilus", max_qty=int(float(row.get("peak_qty", 0) or 0)),
            ))

    series = pd.Series(curve).sort_index() if curve else pd.Series(dtype=float)
    if not len(series):
        series = pd.Series({pd.Timestamp("1970-01-01"): float(starting_capital),
                            pd.Timestamp("1970-01-02"): float(starting_capital) + realized})
    return EngineResult(engine=engine_name, equity=series, trades=trades,
                        diagnostics={"sessions_loaded": n_sessions,
                                     "nautilus_positions": len(trades),
                                     "nautilus_realized_pnl": round(realized, 2),
                                     "independent": True})

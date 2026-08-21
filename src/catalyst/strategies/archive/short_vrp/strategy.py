"""v14 — Concentrated short-VRP: defined-risk iron condors on the richest
catalyst premium.

THE BET. Implied volatility is systematically overpriced around known
catalysts; sellers of defined-risk premium collect the spread as IV collapses
post-event. Direction-independent and structural. This is the untested mirror
of the family this repo has already measured from the other side:

- Engines A/B BOUGHT this premium at events and lost (-0.45%/-2.25%/mo).
- v10 bought strangles at every catalyst: PF 0.87 real, 0.96 frictionless —
  the buyer loses even before costs, which is exactly the seller's opening.
- v8 sold index VRP UNCONDITIONALLY at EOD and broke even (+0.07%/mo).
- The one prior validated edge (Engine C, ~+1%/yr) is this premium's cousin.

What is genuinely new here: CONCENTRATION. Trade only events where the premium
is measurably fattest — IV rank >= 80th percentile of the trailing year, AND
IV/RV >= 1.3, AND the option-implied move >= 1.1x what the stock actually
delivered over its last 8 earnings. Every gate is point-in-time by
construction (IV rank from the provider's trailing series; realized vol and
historical moves from closes strictly before entry; prior-events list frozen
before the current event).

Structure is defined-risk ALWAYS: short iron condor, ~0.16-delta shorts,
wings one listed strike further out. Max loss = max(width) - credit, known at
entry, and the RiskManager sizes on that number — mandatory for premium
selling. Four legs pay eight spread crossings round trip; the honest cost
model is where this family of edges usually dies, which is precisely what the
backtest exists to measure.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from catalyst.core.chains import nearest_delta, quotable
from catalyst.core.interfaces import CatalystStrategy
from catalyst.core.tradingcal import previous_trading_day
from catalyst.core.types import (
    Catalyst,
    Direction,
    ExitRules,
    OptionChain,
    OptionContract,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.strategies.registry import StrategyMeta, register

logger = logging.getLogger(__name__)

# one id per process: sidecar rows from different runs are distinguishable
# (audit D-158: six segment-runs appended indistinguishable duplicates)
import os as _os
_RUN_ID = f"{_os.getpid()}"


class ShortVRPStrategy(CatalystStrategy):
    """One defined-risk iron condor per qualifying (richest-premium) catalyst."""

    name = "short_vrp"

    def __init__(self, cfg, iv_rank, closes: dict[str, pd.Series],
                 earnings_dates: dict[str, list[date]]) -> None:
        self._p = cfg.short_vrp
        self._iv_rank = iv_rank                    # IVRankProvider (injected)
        self._closes = closes                      # symbol -> daily close series
        self._earnings = earnings_dates            # symbol -> sorted event dates
        self.gates = {"seen": 0, "not_entry_day": 0, "iv_rank": 0, "iv_rv": 0,
                      "implied_vs_hist": 0, "structure": 0, "liquidity": 0,
                      "passed": 0}

    # ------------------------------------------------------------------
    def catalyst_expiries(self, catalyst: Catalyst, available: list[date],
                          as_of: date) -> list[date] | None:
        """First listed expiry strictly after the event, capped at max_dte."""
        event = resolve_reaction_session(catalyst)
        elig = [e for e in sorted(available)
                if e > event and (e - as_of).days <= self._p.max_dte]
        return elig[:1]

    # ------------------------------------------------------------------
    def evaluate(self, catalyst: Catalyst, chain: OptionChain,
                 signal: SignalResult, as_of: datetime) -> ProposedTrade | None:
        p = self._p
        self.gates["seen"] += 1

        # Entry timing: the session BEFORE the reaction session, at the 15:45
        # snapshot — peak pre-event IV per the spec's entry_lead.
        reaction = resolve_reaction_session(catalyst)
        if as_of.date() != previous_trading_day(reaction):
            self.gates["not_entry_day"] += 1
            return None

        sym = catalyst.symbol
        spot = chain.underlying_price
        if spot <= 0:
            return None

        # All three gate metrics are computed up-front (not lazily inside the
        # gates) so the sidecar records them for REJECTED events too — the
        # sensitivity sweep needs outcomes across the whole threshold grid,
        # not just inside the default gates.
        # Scale contracts (each was a silent no-op gate before v14.1):
        #   - provider iv_rank is 0..100; config thresholds are 0..1
        #   - IV/RV must be tenor-matched: ~30-DTE IV over 30d RV. The
        #     event-expiry ATM IV is mechanically inflated by the event.
        #   - hist moves must be measured on REACTION sessions, not the
        #     (pre-announcement) calendar date of after-close reporters.
        raw_rank = self._iv_rank.iv_rank(sym, as_of.date())
        rank = raw_rank / 100.0 if raw_rank is not None else None
        iv30 = self._iv_rank.atm_iv30(sym, as_of.date())  # tenor-matched (~30 DTE)
        rv30 = self._realized_vol(sym, as_of.date())
        implied_move = _implied_move(chain, spot, after=reaction)
        hist_move = self._hist_avg_move(sym, as_of.date())
        iv_rv = (iv30 / rv30) if (iv30 and rv30 and rv30 > 0) else None
        imp_hist = (implied_move / hist_move) if (implied_move and hist_move
                                                  and hist_move > 0) else None
        self._sidecar_row(catalyst.ref, sym, as_of.date(), spot, rank, iv30,
                          rv30, iv_rv, implied_move, hist_move, imp_hist)

        # ---- VRP gate 1: IV rank vs trailing year ------------------------
        if rank is None or rank < p.iv_rank_min:
            self.gates["iv_rank"] += 1
            return None

        # ---- VRP gate 2: IV / 30d realized -------------------------------
        if iv_rv is None or iv_rv < p.iv_rv_min:
            self.gates["iv_rv"] += 1
            return None

        # ---- VRP gate 3: implied move vs history --------------------------
        if imp_hist is None or imp_hist < p.implied_vs_hist_min:
            self.gates["implied_vs_hist"] += 1
            return None

        # ---- structure: iron condor at ~0.16-delta shorts ----------------
        # The chain can carry expiries beyond the requested one (open
        # positions on the same symbol inject theirs): min() could select an
        # expiry BEFORE the event and sell a structure that never sees the
        # catalyst (audit D-017). Re-derive the event-capture expiry.
        event_expiries = [e for e in chain.expirations()
                          if e > reaction and (e - as_of.date()).days <= p.max_dte]
        if not event_expiries:
            self.gates["structure"] += 1
            return None
        expiry = min(event_expiries)
        legs = _iron_condor(chain, expiry, p.short_delta)
        if legs is None:
            self.gates["structure"] += 1
            return None
        sc, lc, sp_, lp = legs

        # per-leg liquidity gate (spec: <= 8% of mid on TRADED strikes)
        for leg in (sc, lc, sp_, lp):
            if leg.mid <= 0 or leg.spread_pct_of_mid > p.max_leg_spread_pct_of_mid:
                self.gates["liquidity"] += 1
                return None

        # worst-case credit: shorts at BID, longs at ASK
        credit = (sc.bid + sp_.bid) - (lc.ask + lp.ask)
        call_width = lc.key.strike - sc.key.strike
        put_width = sp_.key.strike - lp.key.strike
        width = max(call_width, put_width)
        if credit <= 0.05 or width <= 0 or credit >= width:
            self.gates["structure"] += 1
            return None

        self.gates["passed"] += 1
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[OrderLeg(key=sc.key, side=Side.SELL, qty=1),
                  OrderLeg(key=lc.key, side=Side.BUY, qty=1),
                  OrderLeg(key=sp_.key, side=Side.SELL, qty=1),
                  OrderLeg(key=lp.key, side=Side.BUY, qty=1)],
            unit_cost=-credit,
            unit_max_loss=width - credit,
            direction=Direction.NEUTRAL,
            exit_rules=ExitRules(
                tp_credit_fraction=p.tp_credit_fraction,
                stop_credit_multiple=(p.stop_credit_multiple if p.use_stops else None),
                use_stops=p.use_stops,
                # the crush edge is spent once IV normalizes: out by the end
                # of the day after the event, no matter what
                hard_exit_date=_next_session(reaction),
                close_before_expiry_days=0,
            ),
            per_trade_risk_fraction=p.per_trade_risk_fraction,
            rationale={"iv_rank": round(rank, 3), "iv_rv": round(iv_rv, 2),
                       "implied_move": round(implied_move, 4),
                       "hist_move": round(hist_move, 4),
                       "credit": round(credit, 2), "width": width,
                       "short_call": sc.key.strike, "short_put": sp_.key.strike},
        )

    # ------------------------------------------------------------------
    def _sidecar_row(self, ref, sym, asof, spot, rank, atm_iv, rv30, iv_rv,
                     implied_move, hist_move, imp_hist) -> None:
        """Append this event's gate metrics to a diagnostics CSV.

        Purely observational (never read back during the run): the sensitivity
        sweep joins these rows to the trade ledger by catalyst_ref to evaluate
        the whole threshold grid, and rejected events must appear too.
        Failures are swallowed — diagnostics must never kill a backtest.
        """
        try:
            from pathlib import Path
            path = Path("results/archive/short_vrp/gate_metrics.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            fmt = lambda v: "" if v is None else f"{v:.6g}"  # noqa: E731
            with path.open("a") as f:
                if new:
                    f.write("run_id,catalyst_ref,symbol,as_of,spot,iv_rank,iv30,"
                            "rv30,iv_rv,implied_move,hist_move,implied_vs_hist\n")
                f.write(f"{_RUN_ID},{ref},{sym},{asof},{fmt(spot)},{fmt(rank)},"
                        f"{fmt(atm_iv)},{fmt(rv30)},{fmt(iv_rv)},"
                        f"{fmt(implied_move)},{fmt(hist_move)},{fmt(imp_hist)}\n")
        except Exception:                                   # noqa: BLE001
            logger.debug("sidecar write failed", exc_info=True)

    def _realized_vol(self, sym: str, asof: date) -> float | None:
        px = self._closes.get(sym)
        if px is None:
            return None
        window = px[px.index < pd.Timestamp(asof)].tail(31)
        if len(window) < 21:
            return None
        r = np.log(window).diff().dropna()
        return float(r.std() * np.sqrt(252))

    def _hist_avg_move(self, sym: str, asof: date) -> float | None:
        """Average |close-to-close move| across the PRIOR <=8 REACTION sessions
        — strictly before entry (point-in-time). ``self._earnings`` holds
        reaction dates (resolve_reaction_session applied in build()), so an
        after-close reporter's move is measured on the day the stock actually
        reacted, not the announcement date's (pre-event) close."""
        px = self._closes.get(sym)
        reactions = [r for r in self._earnings.get(sym, []) if r < asof][-8:]
        if px is None or len(reactions) < 4:
            return None
        moves = []
        for r in reactions:
            pre = px[px.index < pd.Timestamp(r)]
            post = px[px.index >= pd.Timestamp(r)]
            if not len(pre) or not len(post):
                continue
            p0, p1 = float(pre.iloc[-1]), float(post.iloc[0])
            if p0 > 0:
                moves.append(abs(p1 / p0 - 1))
        return float(np.mean(moves)) if len(moves) >= 4 else None


def _implied_move(chain: OptionChain, spot: float, after=None) -> float | None:
    """ATM straddle mid / spot, on the FIRST event-capture expiry only —
    mixing strikes across expiries paired calls and puts from different
    tenors (audit D-159)."""
    expiries = [e for e in chain.expirations() if after is None or e > after]
    if not expiries:
        return None
    exp = min(expiries)
    contracts = chain.slice(expiry=exp)
    strikes = sorted({c.key.strike for c in contracts})
    if not strikes:
        return None
    k = min(strikes, key=lambda x: abs(x - spot))
    call = next((c for c in contracts
                 if c.key.strike == k and c.key.right is OptionRight.CALL), None)
    put = next((c for c in contracts
                if c.key.strike == k and c.key.right is OptionRight.PUT), None)
    if call is None or put is None or call.ask <= 0 or put.ask <= 0:
        return None
    return (call.mid + put.mid) / spot


def _iron_condor(chain: OptionChain, expiry: date, short_delta: float):
    """(short_call, long_call, short_put, long_put) or None."""
    sc = nearest_delta(chain, expiry, OptionRight.CALL, short_delta)
    sp_ = nearest_delta(chain, expiry, OptionRight.PUT, short_delta)
    if sc is None or sp_ is None or sp_.key.strike >= sc.key.strike:
        return None
    calls = sorted([c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.CALL))
                    if c.key.strike > sc.key.strike], key=lambda c: c.key.strike)
    puts = sorted([c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.PUT))
                   if c.key.strike < sp_.key.strike], key=lambda c: c.key.strike,
                  reverse=True)
    if not calls or not puts:
        return None
    return sc, calls[0], sp_, puts[0]     # wings one listed increment out


def _next_session(d: date) -> date:
    from catalyst.core.tradingcal import next_trading_day
    try:
        return next_trading_day(d)
    except Exception:                                   # noqa: BLE001
        return d + timedelta(days=1)


def build(cfg) -> ShortVRPStrategy:
    """Registry entry point: injects IV-rank provider, closes, earnings dates.

    All injected data is READ-ONLY history the strategy could have had on file
    — same class of injection as Engine A's iv_rank (established precedent).
    """
    from catalyst.core.tradingcal import sessions_in_range
    from catalyst.data.cache import ParquetCache
    from catalyst.data.intraday import AlpacaMinuteBars
    from catalyst.data.iv_history import IVRankProvider
    from catalyst.data.thetadata_client import ThetaDataClient
    from catalyst.runners.deploy_runner import _load_catalysts

    cache = ParquetCache(cfg.data.cache_dir)
    ivr = IVRankProvider(cfg.data, ThetaDataClient(cfg.data.thetadata), cache,
                         lookback_days=252)
    # prepare() is mandatory before iv_rank() (gate3 precedent); series are
    # already in the iv_eod cache for the full watchlist, so this is fast.
    hist_start = date(2016, 1, 4)
    raw_end = getattr(getattr(cfg, "backtest", None), "end_date", None)
    if isinstance(raw_end, str):
        hist_end = date.fromisoformat(raw_end)
    elif isinstance(raw_end, date):
        hist_end = raw_end
    else:
        hist_end = date.today()
    for sym in cfg.watchlist:
        try:
            ivr.prepare(sym, hist_start, hist_end)
        except Exception as exc:                    # noqa: BLE001
            logger.warning("IV prepare failed for %s: %s (gate will skip it)",
                           sym, exc)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    closes: dict[str, pd.Series] = {}
    for sym in cfg.watchlist:
        df = bars.daily_bars(sym, hist_start, hist_end)
        if not df.empty:
            ser = df["close"]
            ser.index = pd.to_datetime(pd.Index(ser.index))
            closes[sym] = ser
    cats = _load_catalysts(cfg, hist_start, hist_end)
    # ALL catalyst REACTION sessions per symbol: earnings for single names,
    # CPI/FOMC for index products — the "historical move" gate is
    # event-type-agnostic (prior same-symbol events), so macro events aren't
    # silently excluded. Reaction resolution matters: an after-close reporter
    # moves the NEXT session, and using announcement dates here measured
    # pre-event drift (~0.8%) instead of event moves (~4%), silencing gate 3.
    earnings: dict[str, list[date]] = {}
    for c in cats:
        try:
            earnings.setdefault(c.symbol, []).append(resolve_reaction_session(c))
        except Exception:                               # noqa: BLE001
            earnings.setdefault(c.symbol, []).append(c.when.date())
    for sym in earnings:
        earnings[sym] = sorted(set(earnings[sym]))
    return ShortVRPStrategy(cfg, ivr, closes, earnings)


register(
    StrategyMeta(
        name="short_vrp",
        module="catalyst.strategies.archive.short_vrp.strategy",
        status="archived",
        notes=("v14. Concentrated short-VRP: defined-risk iron condors on "
               "catalysts gated to the richest premium (IV rank>=80th pct, "
               "IV/RV>=1.3, implied>=1.1x hist-8 moves). The seller-side mirror "
               "of measured buyer losses; the honest 8-crossing cost model is "
               "the referee."),
    ),
    build,
)

"""v9 — straight long calls and puts. THE ACTIVE STRATEGY.

Selection logic is unchanged from the standalone v9 campaign: momentum picks
the direction, strikes are chosen at a moneyness offset from the chain's own
contemporaneous spot, tenor is 3-5 weeks, and positions are held to a fixed
exit with no profit-taking.

What changed is everything *around* it. The standalone version ran its own
loop, priced fills by reading ``.ask``/``.bid`` directly, and never built an
``Order`` — so it paid no commission, no slippage, and never met the
RiskManager. Here it emits a ``ProposedTrade`` and the shared pipeline does the
rest, which means its numbers are now comparable with every other campaign for
the first time. That comparison is reported explicitly rather than quietly
substituted; see ``results/active/long_options/``.

Design choices, each traceable to something this project measured:

- **Direction from momentum** (IC ~0.035 at the 5-day horizon, validated
  against a random control) — the only directional edge measured anywhere here.
  The v9 sweep then showed it is *harmful* on this structure: calls-only beat
  signal-directed in all 12 paired configs. ``use_signal=False`` reproduces
  that control.
- **3-5 week tenor.** Under ~3 days is fatal — Kinetic's 0DTE lost $1,209 per
  trade frictionless, pure theta.
- **No profit-taking.** Capping the upside removes the outcomes that justify
  the structure; v6 measured a 3x cap collapsing P(10x) from 26% to 5.2%.
- **Strike from ``chain.underlying_price``.** Never from an adjusted price
  series: that mismatch bought deep-ITM contracts on every pre-split date
  (AMZN 20.03x, TSLA 14.93x, NVDA 10.04x) while unsplit MSFT looked correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

from catalyst.core.chains import quotable
from catalyst.core.interfaces import Cadence, Opportunity, Strategy, StrategyContext
from catalyst.core.types import (
    Direction,
    ExitRules,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
)
from catalyst.strategies.registry import StrategyMeta, register

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LongOptionsParams:
    """Every number lives in config; nothing is hardcoded in the logic."""

    universe: tuple[str, ...] = ("TSLA", "AMD", "NVDA", "AMZN", "META", "MSFT")
    hold_days: int = 15
    dte_days: int = 35
    dte_tolerance: int = 12
    moneyness: float = 1.05
    entry_every_days: int = 15
    momentum_lookback: int = 60
    use_signal: bool = True
    allow_puts: bool = True
    min_ask: float = 0.05
    per_trade_risk_fraction: float = 0.25


class LongOptionsStrategy(Strategy):
    """Buy a call when momentum is positive, a put when negative; hold; repeat."""

    name = "long_options"
    cadence = Cadence.SCHEDULED

    def __init__(self, params: LongOptionsParams | None = None) -> None:
        self._p = params or LongOptionsParams()
        self._session_index: dict[date, int] = {}
        self._counter = 0

    # -- enumeration -------------------------------------------------------
    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        """One cycle every ``entry_every_days`` sessions, across the universe.

        Chain-free by construction: answering "would you look today" must not
        cost a chain request.
        """
        idx = self._session_index.get(session)
        if idx is None:
            idx = self._counter
            self._session_index[session] = idx
            self._counter += 1
        if idx % self._p.entry_every_days != 0:
            return []
        return [Opportunity(session=session, symbols=(sym,),
                            meta={"cycle": idx // self._p.entry_every_days})
                for sym in self._p.universe]

    def required_expiries(
        self, opp: Opportunity, available: list[date], as_of: date
    ) -> list[date] | None:
        return _candidate_expiries(available, as_of, self._p.dte_days,
                                   self._p.dte_tolerance)[:3]

    # -- intent ------------------------------------------------------------
    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        p = self._p
        chain = ctx.chain(opp.symbol)
        if chain is None or chain.underlying_price <= 0:
            return None

        right = self._direction(opp.symbol, ctx)
        # Strike target comes from the CHAIN's spot, never from an adjusted
        # price series — see the module docstring.
        raw_spot = chain.underlying_price
        target = (raw_spot * p.moneyness if right is OptionRight.CALL
                  else raw_spot * (2.0 - p.moneyness))

        expiries = _candidate_expiries(chain.expirations(), opp.session,
                                       p.dte_days, p.dte_tolerance)
        for expiry in expiries[:3]:
            candidates = [c for c in quotable(chain.slice(expiry=expiry, right=right))
                          if c.ask > p.min_ask]
            if not candidates:
                continue
            pick = min(candidates, key=lambda c: abs(c.key.strike - target))
            # A long option's max loss IS the premium, so unit_cost and
            # unit_max_loss coincide. The RiskManager sizes from that; the
            # strategy does not choose a quantity.
            return ProposedTrade(
                engine=self.name,
                catalyst_ref=f"{opp.symbol}:{opp.session}",
                legs=[OrderLeg(key=pick.key, side=Side.BUY, qty=1)],
                unit_cost=pick.ask,
                unit_max_loss=pick.ask,
                direction=(Direction.LONG if right is OptionRight.CALL
                           else Direction.SHORT),
                # Held to a fixed exit with NO profit-taking and NO stop: capping
                # the upside removes the outcomes that justify the structure
                # (v6 measured a 3x cap collapsing P(10x) from 26% to 5.2%).
                exit_rules=ExitRules(max_hold_trading_days=p.hold_days,
                                     use_stops=False),
                per_trade_risk_fraction=p.per_trade_risk_fraction,
                rationale={"moneyness": p.moneyness, "right": right.value,
                           "spot": raw_spot, "strike": pick.key.strike,
                           "cycle": opp.meta.get("cycle")},
            )
        return None

    # -- direction ---------------------------------------------------------
    def _direction(self, symbol: str, ctx: StrategyContext) -> OptionRight:
        p = self._p
        if not p.use_signal or not p.allow_puts:
            return OptionRight.CALL
        hist = ctx.history.get(symbol)
        if hist is None or len(hist) <= p.momentum_lookback:
            return OptionRight.CALL
        closes = hist["close"]
        mom = float(closes.iloc[-1] / closes.iloc[-p.momentum_lookback] - 1.0)
        return OptionRight.CALL if mom >= 0 else OptionRight.PUT


def _candidate_expiries(available: list[date], today: date, dte: int, tol: int) -> list[date]:
    """Fridays first, then nearest-to-target DTE."""
    elig = [e for e in sorted(available) if dte - tol <= (e - today).days <= dte + tol]
    fri = [e for e in elig if e.weekday() == 4]
    ordered = fri + [e for e in elig if e not in fri]
    return sorted(ordered, key=lambda e: abs((e - today).days - dte))


def build(cfg) -> LongOptionsStrategy:
    """Registry entry point. Reads config; falls back to measured defaults."""
    section = getattr(cfg, "long_options", None)
    if section is None:
        return LongOptionsStrategy()
    return LongOptionsStrategy(LongOptionsParams(
        universe=tuple(getattr(section, "universe", LongOptionsParams.universe)),
        hold_days=getattr(section, "hold_days", LongOptionsParams.hold_days),
        dte_days=getattr(section, "dte_days", LongOptionsParams.dte_days),
        moneyness=getattr(section, "moneyness", LongOptionsParams.moneyness),
        entry_every_days=getattr(section, "entry_every_days",
                                 LongOptionsParams.entry_every_days),
        use_signal=getattr(section, "use_signal", LongOptionsParams.use_signal),
        allow_puts=getattr(section, "allow_puts", LongOptionsParams.allow_puts),
        per_trade_risk_fraction=getattr(section, "per_trade_risk_fraction",
                                        LongOptionsParams.per_trade_risk_fraction),
    ))


register(
    StrategyMeta(
        name="long_options",
        module="catalyst.strategies.active.long_options",
        status="active",
        date_tested="2026-08-15",
        verdict=("first options structure with a real right tail (TSLA P(10x)=7.3%), "
                 "but no alpha: calls-only beat signal-directed in all 12 configs"),
        avg_monthly_return=0.0098,
        baseline_annual=0.01,
        validated=False,      # set by the pipeline once it passes the standard report
        paper_tested=False,   # set only after an actual --mode paper session
        notes="Standalone v9 measured without commissions/slippage; now routed "
              "through costs/ so numbers are comparable. Both are reported.",
    ),
    build,
)

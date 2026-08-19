"""ASCENT — all-options, target-seeking convexity engine.

THE CLAIM, STATED EXACTLY. This architecture maximizes the probability of a
30-50%/month outcome over an 8-month window given the best payoff distribution
this project has ever measured. That probability is approximately 15% for the
8.2x (30%/mo) target under the DP-optimal sizing policy, with a median outcome
near zero and roughly an 80% chance of losing more than half the portfolio.
It does NOT average 30%/month — no configuration of the measured distribution
does, and the report states the full outcome distribution rather than its tail.

Architecture, with each component's selection evidence:

- **Structure: long calls at moneyness 1.20, ~35 DTE, 15-session hold.**
  Measured head-to-head against 1.05/1.15 calls and 1x2 backspreads on 785
  identical entry events: same mean as the baseline (1.29 vs 1.27) but P(10x)
  3.3% vs 0.3% and max 34x vs 13.5x at a quarter of the premium — 5.8x the
  tail-per-dollar. The backspread's headline number was a denominator artifact
  and was rejected on its true risk basis.
- **Universe: 6 cost-screened names (TSLA AMD NVDA AMZN META MSFT), equal
  split per cycle.** Cycle-level averaging with REAL same-cycle correlation
  raised P(target) from 11.4% to 15.2% versus single-name — with dynamic
  sizing, the policy can always add variance via f, so the better-shaped
  diversified base wins.
- **No direction signal.** Calls-only beat signal-directed in all 12 paired
  configs (v9). Puts destroyed value on every name.
- **No IV conditioning.** Significant on the near-the-money mixed book
  (Spearman -0.148, p~0), it does NOT transfer to deep-OTM calls
  (Spearman +0.025 on m1.20) — killed in joint synthesis.
- **Sizing: DP-optimal dynamic policy (Dubins-Savage bold play, generalized),
  computed by backward induction on the real-cost distribution.** Dynamic
  sizing nearly quadruples P(target) vs the best static fraction (15.2% vs
  ~4%). The policy lives OUTSIDE this class: strategies cannot see account
  equity by construction, so the per-cycle fraction is supplied via config and
  the policy table ships with the report. In backtest the pipeline runs the
  structure at a static fraction; the dynamic overlay is evaluated by exact
  replay and Monte Carlo in the report analytics.

Note on the founding spec: distance-to-target sizing is precisely what the
original governing invariant prohibited ("no module may raise size in response
to underperformance relativeive to any goal"). The user's explicit later
instructions supersede it; the policy is precomputed and reported, not a
runtime override of the risk layer — the RiskManager still gates every order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

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
class AscentParams:
    universe: tuple[str, ...] = ("TSLA", "AMD", "NVDA", "AMZN", "META", "MSFT")
    moneyness: float = 1.20
    dte_days: int = 35
    dte_tolerance: int = 12
    hold_days: int = 15
    entry_every_days: int = 15
    min_ask: float = 0.05
    #: Per-name fraction for the pipeline's static-structure validation run.
    #: The dynamic policy replaces this in the overlay analytics; 0.0833/name
    #: ~= 50% deployed, the near-optimal static point on the measured curve.
    per_trade_risk_fraction: float = 0.0833


class AscentStrategy(Strategy):
    """Deep-OTM calls across six names on a fixed cycle. No signal, no filter:
    every removable component was removed because measurement said so."""

    name = "ascent"
    cadence = Cadence.SCHEDULED

    def __init__(self, params: AscentParams | None = None) -> None:
        self._p = params or AscentParams()
        self._seen: dict[date, int] = {}
        self._n = 0

    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        idx = self._seen.get(session)
        if idx is None:
            idx = self._n
            self._seen[session] = idx
            self._n += 1
        if idx % self._p.entry_every_days != 0:
            return []
        return [Opportunity(session=session, symbols=(s,),
                            meta={"cycle": idx // self._p.entry_every_days})
                for s in self._p.universe]

    def required_expiries(
        self, opp: Opportunity, available: list[date], as_of: date
    ) -> list[date] | None:
        return _expiries(available, as_of, self._p.dte_days, self._p.dte_tolerance)[:3]

    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        p = self._p
        chain = ctx.chain(opp.symbol)
        if chain is None or chain.underlying_price <= 0:
            return None
        # Strike from the CHAIN'S own spot — never an adjusted price series
        # (the split-mismatch rule, learned at full-campaign cost).
        target = chain.underlying_price * p.moneyness
        for expiry in _expiries(chain.expirations(), opp.session,
                                p.dte_days, p.dte_tolerance)[:3]:
            cands = [c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.CALL))
                     if c.ask > p.min_ask]
            if not cands:
                continue
            pick = min(cands, key=lambda c: abs(c.key.strike - target))
            return ProposedTrade(
                engine=self.name,
                catalyst_ref=f"{opp.symbol}:{opp.session}",
                legs=[OrderLeg(key=pick.key, side=Side.BUY, qty=1)],
                unit_cost=pick.ask,
                unit_max_loss=pick.ask,     # long call: premium IS the max loss
                direction=Direction.LONG,
                # Winners run to the cycle exit. No profit-taking, no stop:
                # capping the tail is the one thing this structure cannot afford
                # (v6: a 3x cap collapsed P(10x) from 26% to 5.2%).
                exit_rules=ExitRules(max_hold_trading_days=p.hold_days,
                                     use_stops=False),
                per_trade_risk_fraction=p.per_trade_risk_fraction,
                rationale={"moneyness": p.moneyness, "spot": chain.underlying_price,
                           "strike": pick.key.strike, "cycle": opp.meta.get("cycle")},
            )
        return None


def _expiries(available: list[date], today: date, dte: int, tol: int) -> list[date]:
    elig = [e for e in sorted(available) if dte - tol <= (e - today).days <= dte + tol]
    fri = [e for e in elig if e.weekday() == 4]
    ordered = fri + [e for e in elig if e not in fri]
    return sorted(ordered, key=lambda e: abs((e - today).days - dte))


def build(cfg) -> AscentStrategy:
    section = getattr(cfg, "ascent", None)
    if section is None:
        return AscentStrategy()
    return AscentStrategy(AscentParams(
        universe=tuple(getattr(section, "universe", AscentParams.universe)),
        moneyness=getattr(section, "moneyness", AscentParams.moneyness),
        hold_days=getattr(section, "hold_days", AscentParams.hold_days),
        entry_every_days=getattr(section, "entry_every_days",
                                 AscentParams.entry_every_days),
        per_trade_risk_fraction=getattr(section, "per_trade_risk_fraction",
                                        AscentParams.per_trade_risk_fraction),
    ))


register(
    StrategyMeta(
        name="ascent",
        module="catalyst.strategies.active.ascent",
        status="active",
        notes=("ALL-OPTIONS TARGET-SEEKING ENGINE. Maximizes P(30-50%/mo over 8mo): "
               "~15% under the DP-optimal dynamic sizing policy, median outcome ~0, "
               "~80% chance of losing more than half. NOT an edge strategy — no "
               "configuration of the measured distribution averages 30%/mo. "
               "Live path requires validated positive edge, which this does not have."),
    ),
    build,
)

"""new-strategy — scaffold a strategy that already satisfies the contract.

    uv run python -m catalyst.runners.new_strategy --name mean_revert_vol \
        --cadence scheduled

Writes `strategies/active/<name>.py` with the interface implemented, the
registry call in place, and the parameter dataclass wired to config — so the
only thing left to write is the actual idea.

The scaffold is deliberately opinionated about what it does NOT include: no
broker, no sizing, no fill pricing, no place to put a stop-loss override. A
scaffold that left those doors open would invite exactly the per-campaign
divergence this framework exists to prevent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ACTIVE = Path("src/catalyst/strategies/active")

TEMPLATE = '''"""{title}

<one paragraph: what the idea is, and what it is betting on>

Evidence this relies on (see `design_brief`):
- <rule or campaign this builds on>
- <rule this deliberately contradicts, and why the old evidence no longer applies>

Design choices, each traceable to something measured:
- <choice> -> <the measurement that justifies it>
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
class {cls}Params:
    """Every number lives in config; nothing hardcoded in the logic."""

    universe: tuple[str, ...] = ("SPY",)
    dte_days: int = 35
    dte_tolerance: int = 12
    hold_days: int = 15
    entry_every_days: int = 15
    per_trade_risk_fraction: float = 0.02


class {cls}(Strategy):
    """<one line: what it trades and when>"""

    name = "{name}"
    cadence = Cadence.{cadence}

    def __init__(self, params: {cls}Params | None = None) -> None:
        self._p = params or {cls}Params()
        self._seen: dict[date, int] = {{}}
        self._n = 0

    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        """What would you look at today? No chain access, no trade decision.

        Chain-free on purpose: answering "would you look" must not cost a
        chain request.
        """
        idx = self._seen.get(session)
        if idx is None:
            idx = self._n
            self._seen[session] = idx
            self._n += 1
        if idx % self._p.entry_every_days != 0:
            return []
        return [Opportunity(session=session, symbols=(s,)) for s in self._p.universe]

    def required_expiries(
        self, opp: Opportunity, available: list[date], as_of: date
    ) -> list[date] | None:
        """Narrow the fetch — historical chains are billed per expiration."""
        lo, hi = self._p.dte_days - self._p.dte_tolerance, self._p.dte_days + self._p.dte_tolerance
        elig = [e for e in sorted(available) if lo <= (e - as_of).days <= hi]
        return elig[:3] or None

    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        """Unsized trade intent, or None when the entry gates are not met.

        You do NOT choose quantity — the RiskManager sizes from unit_max_loss.
        You do NOT price the fill — costs/ does that.
        """
        chain = ctx.chain(opp.symbol)
        if chain is None or chain.underlying_price <= 0:
            return None

        # Strike selection MUST use the chain's own spot, never an adjusted
        # price series (that mismatch bought deep-ITM contracts for years).
        spot = chain.underlying_price

        # TODO: implement the idea. Pick contracts, then return intent:
        #
        # candidates = [c for c in quotable(chain.slice(expiry=e, right=OptionRight.CALL))
        #               if c.ask > 0.05]
        # pick = min(candidates, key=lambda c: abs(c.key.strike - target))
        # return ProposedTrade(
        #     engine=self.name,
        #     catalyst_ref=f"{{opp.symbol}}:{{opp.session}}",
        #     legs=[OrderLeg(key=pick.key, side=Side.BUY, qty=1)],
        #     unit_cost=pick.ask,
        #     unit_max_loss=pick.ask,        # worst case; credit structures: width - credit
        #     direction=Direction.LONG,
        #     exit_rules=ExitRules(max_hold_trading_days=self._p.hold_days),
        #     per_trade_risk_fraction=self._p.per_trade_risk_fraction,
        #     rationale={{"spot": spot}},     # logged, never parsed
        # )
        return None


def build(cfg) -> {cls}:
    """Registry entry point. Reads config; falls back to defaults."""
    section = getattr(cfg, "{name}", None)
    if section is None:
        return {cls}()
    return {cls}({cls}Params(
        universe=tuple(getattr(section, "universe", {cls}Params.universe)),
        dte_days=getattr(section, "dte_days", {cls}Params.dte_days),
        hold_days=getattr(section, "hold_days", {cls}Params.hold_days),
        entry_every_days=getattr(section, "entry_every_days",
                                 {cls}Params.entry_every_days),
        per_trade_risk_fraction=getattr(section, "per_trade_risk_fraction",
                                        {cls}Params.per_trade_risk_fraction),
    ))


register(
    StrategyMeta(
        name="{name}",
        module="catalyst.strategies.active.{name}",
        status="active",
        notes="<what this is testing>",
    ),
    build,
)
'''


def scaffold(name: str, cadence: str) -> Path:
    if not name.isidentifier():
        raise SystemExit(f"'{name}' is not a valid module name")
    path = ACTIVE / f"{name}.py"
    if path.exists():
        raise SystemExit(f"{path} already exists — archive the current strategy first:\n"
                         f"  uv run python -m catalyst.runners.archive_strategy --name <current>")
    cls = "".join(part.capitalize() for part in name.split("_")) + "Strategy"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.format(
        title=f"{name} — <one-line summary>.",
        cls=cls, name=name, cadence=cadence.upper()))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="snake_case strategy name")
    ap.add_argument("--cadence", default="scheduled",
                    choices=["catalyst", "scheduled", "daily"])
    args = ap.parse_args(argv)

    path = scaffold(args.name, args.cadence)
    print(f"created {path}")
    print("\nnext:")
    print("  1. read the brief   : uv run python -m catalyst.runners.design_brief")
    print(f"  2. implement plan() in {path}")
    print("  3. add its parameters to config/base.yaml (no numbers in code)")
    print(f"  4. test  : uv run python -m catalyst.runners.deploy_runner "
          f"--mode backtest --strategy {args.name}")
    print("     (registration is automatic — anything in active/ is discovered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

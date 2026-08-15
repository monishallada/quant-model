"""Cross-engine comparison — where the value of a second engine actually is.

The report shows every engine's numbers side by side so decisions are made on
both. But the section that matters most is DIVERGENCE, and it is framed
deliberately:

- **Divergence is a finding.** Two independent implementations computing
  different P&L from the same trades means at least one is wrong. Every wrong
  number in this project's history — the v8 credit double-count, the v9 split
  mismatch, the row-count time base — was exactly this shape, and each was
  caught only by an internal inconsistency that happened to be noticed.
- **Agreement is weak evidence.** It says neither engine has an obvious ledger
  fault. It does not validate the strategy, the data, or the assumption that
  a fill was achievable. Two engines agreeing on a wrong input agree perfectly.
- **An unavailable engine is reported, never omitted.** Silently dropping an
  engine makes "agreed" and "never ran" look identical on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from catalyst.backtest import metrics as m
from catalyst.core.interfaces.engine import EngineResult

#: Relative gap in average monthly return above which engines are called
#: divergent. Tight on purpose: engines running the same fills should agree to
#: rounding, so anything above this is a real difference in arithmetic.
DIVERGENCE_TOLERANCE = 0.01       # 1%/month. The v8 double-count was
                                  # +4.01% vs a true +0.07% — a 3.94%
                                  # gap that must never slip through.

#: WHY TWO INDEPENDENT ENGINES CANNOT MATCH TO THE CENT
#:
#: Position size depends on account equity; equity depends on realized P&L;
#: P&L depends on the engine's own fill prices. So sizing is path-dependent on
#: exactly the thing that must stay independent. Two genuinely separate engines
#: will therefore take slightly different position counts and sizes, and their
#: equity curves will drift apart even when both are correct.
#:
#: Forcing them to agree would mean sharing P&L — which turns the second engine
#: back into a replay of the first. The divergence bands below are set with
#: that in mind: they are tuned to catch a LEDGER FAULT (wrong sign, missing
#: multiplier, double-counted premium), not to demand identical paths.
#:
#: Final equity is the sharpest check: two engines summing the same fills must
#: land on the same dollar. Monthly return and max drawdown legitimately differ
#: by marking FREQUENCY — the native engine marks daily and sees intra-hold
#: swings, while a fill-replay engine steps only at exits — so a small gap
#: there is curve shape, not arithmetic. Equity has no such excuse.
EQUITY_TOLERANCE = 0.35           # 35%: path drift is expected; a
                                  # 2x gap is not, and that is the
                                  # signature of a ledger fault


@dataclass
class EngineView:
    """One engine's headline numbers for a segment."""

    engine: str
    available: bool
    avg_monthly_return: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    final_equity: float | None = None
    n_trades: int = 0
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_result(cls, r: EngineResult) -> EngineView:
        if not r.ok:
            return cls(engine=r.engine, available=False, error=r.error,
                       diagnostics=r.diagnostics)
        h = m.headline(r.equity)
        return cls(engine=r.engine, available=True,
                   avg_monthly_return=h["avg_monthly_return"], cagr=h["cagr"],
                   max_drawdown=h["max_drawdown"], final_equity=r.final_equity,
                   n_trades=len(r.trades), diagnostics=r.diagnostics)


@dataclass
class EngineComparison:
    """Side-by-side engine results plus an explicit divergence verdict."""

    strategy: str
    segment: str
    cost_profile: str
    views: list[EngineView] = field(default_factory=list)

    @property
    def reference(self) -> EngineView | None:
        """The native engine is the reference: it owns risk, costs and exits."""
        for v in self.views:
            if v.engine == "native" and v.available:
                return v
        return next((v for v in self.views if v.available), None)

    @property
    def comparable(self) -> list[EngineView]:
        ref = self.reference
        return [v for v in self.views if v.available and ref and v.engine != ref.engine]

    @property
    def divergences(self) -> list[dict]:
        ref = self.reference
        if ref is None or ref.avg_monthly_return is None:
            return []
        out = []
        for v in self.comparable:
            if v.avg_monthly_return is None:
                continue
            # Primary check: same fills must sum to the same dollar.
            if (v.final_equity is not None and ref.final_equity
                    and abs(v.final_equity - ref.final_equity)
                    > abs(ref.final_equity) * EQUITY_TOLERANCE):
                out.append({"engine": v.engine, "reference": ref.engine,
                            "reference_final_equity": ref.final_equity,
                            "engine_final_equity": v.final_equity,
                            "equity_gap": v.final_equity - ref.final_equity})

            gap = v.avg_monthly_return - ref.avg_monthly_return
            if abs(gap) > DIVERGENCE_TOLERANCE:
                out.append({"engine": v.engine, "reference": ref.engine,
                            "gap_monthly": gap,
                            "reference_monthly": ref.avg_monthly_return,
                            "engine_monthly": v.avg_monthly_return})
            # Per-trade divergence counted inside the engine adapter is a
            # stronger signal than the aggregate: aggregates can cancel.
            per_trade = v.diagnostics.get("per_trade_divergences", 0)
            if per_trade:
                out.append({"engine": v.engine, "reference": ref.engine,
                            "per_trade_divergences": per_trade,
                            "sample": v.diagnostics.get("divergence_sample", [])})
        return out

    @property
    def agree(self) -> bool:
        return not self.divergences and len(self.comparable) > 0

    def to_text(self) -> str:
        L = [f"  ENGINE COMPARISON — {self.segment}/{self.cost_profile}", ""]
        L.append(f"    {'engine':<12}{'monthly':>10}{'CAGR':>10}{'maxDD':>10}"
                 f"{'final eq':>13}{'N':>6}")
        L.append("    " + "-" * 61)
        for v in self.views:
            if not v.available:
                L.append(f"    {v.engine:<12}{'UNAVAILABLE':>10}   {(v.error or '')[:44]}")
                continue
            L.append(f"    {v.engine:<12}{v.avg_monthly_return:>+9.2%}"
                     f"{v.cagr:>+10.1%}{v.max_drawdown:>+10.1%}"
                     f"{v.final_equity:>13,.0f}{v.n_trades:>6}")
        L.append("")

        if not self.comparable:
            L.append("    only one engine ran — no cross-check performed.")
            L.append("    (this is NOT agreement; nothing was compared)")
        elif self.agree:
            ref = self.reference
            L.append("    ENGINES AGREE — same story, no ledger fault detected.")
            L.append("    Independent engines size off their OWN equity, so paths drift")
            L.append("    even when both are right. Agreement here means neither shows a")
            L.append("    sign error, missing multiplier or double-counted premium.")
            if ref and ref.final_equity:
                L.append(f"    final equity matches across engines "
                         f"({ref.final_equity:,.2f}); monthly/maxDD may differ "
                         "slightly by marking frequency, not arithmetic.")
            L.append("    Agreement means neither implementation miscomputed P&L from")
            L.append("    the same fills. It does NOT validate the signal, the data,")
            L.append("    or whether those fills were achievable.")
        else:
            L.append("    *** ENGINES DIVERGE — at least one is wrong. ***")
            for d in self.divergences:
                if "equity_gap" in d:
                    L.append(f"      {d['engine']} vs {d['reference']}: final equity "
                             f"{d['engine_final_equity']:,.2f} vs "
                             f"{d['reference_final_equity']:,.2f} "
                             f"({d['equity_gap']:+,.2f}) — same fills, different sum")
                elif "gap_monthly" in d:
                    L.append(f"      {d['engine']} vs {d['reference']}: "
                             f"{d['engine_monthly']:+.2%} vs {d['reference_monthly']:+.2%} "
                             f"({d['gap_monthly']:+.2%}/mo)")
                else:
                    L.append(f"      {d['engine']}: {d['per_trade_divergences']} "
                             f"per-trade P&L mismatches vs {d['reference']}")
                    for s in d.get("sample", [])[:3]:
                        L.append(f"        {s}")
            L.append("      Do not act on either number until this is explained.")
        return "\n".join(L)

    def to_dict(self) -> dict:
        return {
            "segment": self.segment, "cost_profile": self.cost_profile,
            "engines": [{"engine": v.engine, "available": v.available,
                         "avg_monthly_return": v.avg_monthly_return,
                         "cagr": v.cagr, "max_drawdown": v.max_drawdown,
                         "final_equity": v.final_equity, "n_trades": v.n_trades,
                         "error": v.error, "diagnostics": v.diagnostics}
                        for v in self.views],
            "agree": self.agree, "divergences": self.divergences,
        }

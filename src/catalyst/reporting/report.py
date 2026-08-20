"""The standardized strategy report — every strategy judged identically.

Report generation lives in the shared pipeline, not in strategy code, because
the audit found each campaign re-implementing (or quietly skipping) its own
checks: the N>100 warning existed in four files, the concentration check in
six, the zero-cost diagnostic in eight. A strategy that writes its own report
can decline to run the check that would have embarrassed it.

Here, no strategy can opt out. Every run produces:

- **Average monthly return** first, always — the project's headline convention.
- **Real vs zero-cost** side by side. The gap separates "no signal" from
  "signal eaten by friction", which is the distinction that decided v3 and v4.
- **70/30 chronological out-of-sample split.** In-sample numbers alone have
  been wrong here often enough to be worthless on their own.
- **N>100 flag.** Engine C looked like +$125/trade at N=85 and was
  gross-negative at N=926.
- **Concentration check** (top-3 share of P&L). A "winning" strategy resting
  on three trades is a lottery ticket wearing a strategy's clothes.
- **Verdict vs the Engine C baseline** (~1%/yr), so results are ranked against
  the archived reference rather than against hope.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.backtest import metrics as m
from catalyst.core.types import TradeRecord

#: Engine C at N=85 implied roughly +1%/yr before it was refuted at scale.
#: Everything is measured against it so "better than nothing" is a number.
ENGINE_C_BASELINE_ANNUAL = 0.01
MIN_TRUSTWORTHY_N = 100
CONCENTRATION_FLAG = 0.50


@dataclass
class SegmentReport:
    """One (segment, cost-profile) cell of the standard report."""

    segment: str            # full | train | test
    cost_profile: str       # real | zero
    avg_monthly_return: float
    cagr: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expected_value: float
    concentration_share: float | None
    pct_months_positive: float

    @property
    def sample_too_small(self) -> bool:
        return self.n_trades < MIN_TRUSTWORTHY_N

    @property
    def lottery_flagged(self) -> bool:
        return self.concentration_share is not None and self.concentration_share > CONCENTRATION_FLAG


@dataclass
class StrategyReport:
    strategy: str
    start: date
    end: date
    segments: list[SegmentReport] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    #: One EngineComparison per (segment, cost profile). The verdict is still
    #: computed from the NATIVE engine, because it is the only one that owns
    #: risk sizing, the cost model and exits — but a divergence downgrades the
    #: verdict, since a number two engines disagree about is not a result.
    comparisons: list = field(default_factory=list)

    @property
    def engines_diverge(self) -> bool:
        return any(not c.agree and c.comparable for c in self.comparisons)

    def get(self, segment: str, cost_profile: str = "real") -> SegmentReport | None:
        for s in self.segments:
            if s.segment == segment and s.cost_profile == cost_profile:
                return s
        return None

    # -- verdict ----------------------------------------------------------
    @property
    def verdict(self) -> str:
        """Judged on OUT-OF-SAMPLE real-cost performance, never in-sample.

        Ranking on the training segment is how a curve-fit gets promoted, so
        the verdict deliberately ignores it.
        """
        if self.engines_diverge:
            return ("HOLD — engines disagree on the P&L; resolve before "
                    "reading any number here")
        # AUDIT HIGH: never fall back to the FULL segment — that is 70%
        # training data wearing an out-of-sample label, and it once granted
        # validated=True from an in-sample number. No test segment, no verdict.
        test = self.get("test", "real")
        if test is None:
            return "NO RESULT — test segment missing; nothing here is out-of-sample"
        annual = (1.0 + test.avg_monthly_return) ** 12 - 1.0
        if test.n_trades == 0:
            return "NO TRADES"
        if test.sample_too_small:
            return f"INCONCLUSIVE — N={test.n_trades} < {MIN_TRUSTWORTHY_N}"
        if test.lottery_flagged:
            return (f"REJECT — top-3 trades are {test.concentration_share:.0%} of P&L "
                    "(lottery artifact)")
        if annual <= 0:
            return f"REJECT — negative out-of-sample ({annual:+.1%}/yr)"
        if annual <= ENGINE_C_BASELINE_ANNUAL:
            return (f"REJECT — {annual:+.1%}/yr does not beat the Engine C "
                    f"baseline ({ENGINE_C_BASELINE_ANNUAL:+.1%}/yr)")
        return f"CANDIDATE — {annual:+.1%}/yr out-of-sample, beats baseline"

    @property
    def cost_gap(self) -> float | None:
        """Zero-cost minus real-cost monthly return: what friction takes."""
        real, zero = self.get("full", "real"), self.get("full", "zero")
        if real is None or zero is None:
            return None
        return zero.avg_monthly_return - real.avg_monthly_return

    # -- rendering --------------------------------------------------------
    def to_text(self) -> str:
        L = [f"{'=' * 74}", f"  STRATEGY REPORT — {self.strategy}",
             f"  {self.start} to {self.end}", "=" * 74]
        full = self.get("full", "real")
        if full:
            L += [f"\n  AVG MONTHLY RETURN: {full.avg_monthly_return:+.2%}"
                  f"     (CAGR {full.cagr:+.1%}, max DD {full.max_drawdown:+.1%})"]
        gap = self.cost_gap
        if gap is not None:
            L.append(f"  cost drag: {gap:+.2%}/mo  "
                     f"(zero-cost {self.get('full','zero').avg_monthly_return:+.2%}/mo)")
        L.append("")
        hdr = (f"  {'segment':<8}{'cost':<7}{'monthly':>10}{'CAGR':>9}{'maxDD':>9}"
               f"{'N':>6}{'win':>7}{'PF':>7}{'top3':>7}")
        L += [hdr, "  " + "-" * (len(hdr) - 2)]
        for s in self.segments:
            top3 = f"{s.concentration_share:.0%}" if s.concentration_share is not None else "n/a"
            L.append(f"  {s.segment:<8}{s.cost_profile:<7}{s.avg_monthly_return:>+9.2%}"
                     f"{s.cagr:>+9.1%}{s.max_drawdown:>+9.1%}{s.n_trades:>6}"
                     f"{s.win_rate:>6.0%}{s.profit_factor:>7.2f}{top3:>7}")
        L.append("")
        for s in self.segments:
            if s.cost_profile != "real":
                continue
            if s.sample_too_small and s.n_trades:
                L.append(f"  *** WARNING: {s.segment} N={s.n_trades} < {MIN_TRUSTWORTHY_N} "
                         "— sample too small to trust ***")
            if s.lottery_flagged:
                L.append(f"  *** LOTTERY FLAG: {s.segment} top-3 trades are "
                         f"{s.concentration_share:.0%} of total P&L ***")
        if self.comparisons:
            L.append("")
            for c in self.comparisons:
                L.append(c.to_text())
                L.append("")
        for k, v in self.extras.items():
            L.append(f"  {k}: {v}")
        L += ["", f"  VERDICT: {self.verdict}", "=" * 74]
        return "\n".join(L)

    def to_json(self) -> str:
        return json.dumps(
            {"strategy": self.strategy, "start": str(self.start), "end": str(self.end),
             "verdict": self.verdict, "cost_gap": self.cost_gap,
             "segments": [asdict(s) for s in self.segments], "extras": self.extras,
             "engines_diverge": self.engines_diverge,
             "engine_comparisons": [c.to_dict() for c in self.comparisons]},
            indent=2, default=str)

    def save(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "report.txt").write_text(self.to_text())
        path = root / "report.json"
        path.write_text(self.to_json())
        return path


def build_segment(
    segment: str, cost_profile: str, equity: pd.Series, trades: list[TradeRecord]
) -> SegmentReport:
    """Compute one cell. Every check runs here — none is optional."""
    h = m.headline(equity) if len(equity) > 1 else {
        "avg_monthly_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0,
        "pct_months_positive": 0.0}
    ts = m.trade_stats(trades)
    return SegmentReport(
        segment=segment, cost_profile=cost_profile,
        avg_monthly_return=h["avg_monthly_return"], cagr=h["cagr"],
        max_drawdown=h["max_drawdown"], n_trades=len(trades),
        win_rate=ts["win_rate"], profit_factor=m.profit_factor(trades),
        expected_value=m.expected_value(trades),
        concentration_share=m.concentration(trades, 3)["share"],
        pct_months_positive=h["pct_months_positive"])

"""Accumulated evidence — the most valuable asset in this repository.

Roughly 5,000 trades across ten campaigns, bought with real data and real
compute. The point of keeping it queryable rather than buried in prose is that
designing the *next* strategy should start from what has already been measured,
not from a blank page.

Two kinds of knowledge live here:

- **Rules** — findings expensive enough that re-testing them is waste. Each
  carries the evidence that produced it, so a rule can be challenged on its
  merits rather than treated as folklore.
- **Campaigns** — what every strategy actually returned, its instrument, and
  its verdict, so a new idea can be compared against the real distribution of
  outcomes instead of against hope.

Used by `runners/design_brief.py` to print a standing brief before any new
strategy is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    """A finding that should not be re-litigated without new evidence."""

    rule: str
    evidence: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Campaign:
    name: str
    instrument: str          # OPTIONS | SHARES
    avg_monthly: float
    verdict: str
    tags: tuple[str, ...] = ()


#: Rules earned the hard way. Mirrors archive/LEARNINGS.md, kept as data so it
#: can be filtered by the design brief.
RULES: tuple[Rule, ...] = (
    Rule("Directional forecasting from technical signals does not work.",
         "Trend and mean-reversion were edgeless at Gate 2; gap-direction lost "
         "gross at N=926. Every strategy needing a direction call failed.",
         ("signal", "direction")),
    Rule("Never buy options held under ~3 days.",
         "Kinetic 0DTE lost $1,209/trade FRICTIONLESS — pure theta, not costs.",
         ("options", "tenor", "theta")),
    Rule("Intraday direction (15-120 min) is a coin flip.",
         "8 signals, 525 symbol-days: zero cleared |t|>2.5. At 40 min the best "
         "signal (t=1.70) was indistinguishable from a random control (t=1.69).",
         ("signal", "intraday")),
    Rule("Option spreads are the dominant cost.",
         "$18-28 per crossing on single names; 53% of collected credit in v4. "
         "Single-name round trips cost 4-10% of premium. Index options are "
         "3.1x cheaper (SPY 0.65% of mid vs 4.43% single-name average).",
         ("options", "costs")),
    Rule("Small samples lie.",
         "Engine C looked like +$125/trade at N=85; the same structure at "
         "N=926 was gross-negative. Nothing under N=100 is a finding.",
         ("methodology", "sample")),
    Rule("Overlapping windows fabricate significance.",
         "A 63-day horizon sampled every 5 days produced t=5.98 that was "
         "really t~1.68. Always sample non-overlapping.",
         ("methodology", "statistics")),
    Rule("Always include a random control.",
         "It has validated every harness we trust and would have caught any "
         "lookahead.",
         ("methodology",)),
    Rule("Cross-check metrics against each other.",
         "v8 reported +4.01%/mo alongside profit factor 0.98 — impossible, and "
         "it exposed a credit double-count. Trade stats and the equity curve "
         "must agree or one is lying.",
         ("methodology",)),
    Rule("Never mix split-ADJUSTED prices with RAW option strikes.",
         "v9: Alpaca bars are split-adjusted, ThetaData strikes are raw. AMZN "
         "ratio 20.03x, TSLA 14.93x, NVDA 10.04x, unsplit MSFT 1.00x — half the "
         "universe looked correct, which is why it survived a whole campaign. "
         "Always select strikes from chain.underlying_price.",
         ("data", "options")),
    Rule("Never fabricate an exit price for a missing contract.",
         "v9's intrinsic fallback marked a vanished pre-split strike against a "
         "post-split spot and manufactured a 924x put. Absent at exit means "
         "unpriceable: skip and count it.",
         ("data", "methodology")),
    Rule("Derive elapsed time from DATES, never from row count.",
         "A curve marked only at exits (~16 rows/year) read as 0.8 months and "
         "turned a +10% year into '+12.8% per month'.",
         ("methodology", "metrics")),
    Rule("Sizing decides ruin, not edge.",
         "v9 identical strategy: +2.93%/mo at 10% of equity per cycle, "
         "+3.74% at 25%, -4.68% at 50%, -5.77% at 100%. At 50% all six names "
         "wiped out. A negative per-position edge cannot be sized into profit.",
         ("risk", "sizing")),
    Rule("Capping upside destroys a threshold objective.",
         "v6: a 3x cap collapsed P(10x) from 26% to 5.2%. Covered calls and "
         "credit spreads maximise average return and are wrong for a "
         "competition — they sell the outcomes that place.",
         ("structure", "tournament")),
    Rule("Options amplify friction as well as signal.",
         "Best measured directional edge is IC ~0.035 against 4-10% round-trip "
         "cost. Expressing a small edge through options converts a slightly "
         "positive strategy into a clearly negative one — shown seven ways.",
         ("options", "costs")),
)

#: Every campaign, measured. avg_monthly is real-cost, full period.
CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign("v1 Engine A — pre-catalyst long options", "OPTIONS", -0.0045,
             "No edge", ("catalyst", "long-premium")),
    Campaign("v1 Engine B — debit spreads", "OPTIONS", -0.0225, "No edge",
             ("catalyst", "spreads")),
    Campaign("v1 Engine C — PEAD", "OPTIONS", 0.0008,
             "Refuted at scale (N=85 -> N=926)", ("catalyst", "pead")),
    Campaign("v2 Pairs V2 — 1-2 DTE options", "OPTIONS", -0.0012,
             "Signal fake; 81% force-flattened", ("pairs",)),
    Campaign("v2 Pairs V1 — shares", "SHARES", -0.0007, "No edge", ("pairs",)),
    Campaign("v3 Kinetic — 0-3 DTE options", "OPTIONS", -0.0145,
             "Break-even signal, friction fatal", ("intraday", "theta")),
    Campaign("v4 Drift — credit spreads", "OPTIONS", -0.0106,
             "Real gross edge, 15x too small vs costs", ("credit",)),
    Campaign("v5 Alpha Platform — shares + SPY", "SHARES", 0.0139,
             "FIRST REAL EDGE — beats SPY (+1.00%)", ("cross-sectional", "shares")),
    Campaign("v6 Tournament — 3-5wk OTM calls", "OPTIONS", -0.0389,
             "Model said P(10x)=31%, real chains 0%", ("tournament",)),
    Campaign("v7 Combined Allocator", "OPTIONS", 0.0249,
             "Concentration-flagged", ("allocator",)),
    Campaign("v8 Index VRP — SPY/QQQ put spreads", "OPTIONS", 0.0007,
             "Break-even; cost thesis right but insufficient", ("index", "credit")),
    Campaign("v9 Long calls/puts (calls-only)", "OPTIONS", 0.0098,
             "First real right tail (TSLA P(10x)=7.3%) but no alpha; "
             "measured without commissions", ("tournament", "long-premium")),
    Campaign("Equal-weight buy & hold, six names", "SHARES", 0.0288,
             "Benchmark any single-name options idea must beat", ("benchmark",)),
    Campaign("SPY buy-and-hold", "SHARES", 0.0100, "Baseline", ("benchmark",)),
)


def rules_for(*tags: str) -> list[Rule]:
    """Rules matching any tag; all rules when no tag is given."""
    if not tags:
        return list(RULES)
    want = {t.lower() for t in tags}
    return [r for r in RULES if want & {t.lower() for t in r.tags}]


def campaigns_for(*tags: str) -> list[Campaign]:
    if not tags:
        return list(CAMPAIGNS)
    want = {t.lower() for t in tags}
    return [c for c in CAMPAIGNS if want & {t.lower() for t in c.tags}]


def best_measured(instrument: str | None = None) -> Campaign:
    pool = [c for c in CAMPAIGNS
            if instrument is None or c.instrument == instrument.upper()]
    return max(pool, key=lambda c: c.avg_monthly)


def load_reports(root: Path = Path("results")) -> list[dict]:
    """Every standard report on disk — actual runs, not curated summaries."""
    out = []
    for p in sorted(root.rglob("report.json")):
        try:
            data = json.loads(p.read_text())
            data["_path"] = str(p)
            out.append(data)
        except Exception:                               # noqa: BLE001
            continue
    return out

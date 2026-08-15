"""brief — the standing brief to read BEFORE designing a strategy.

    uv run python -m catalyst.runners.design_brief
    uv run python -m catalyst.runners.design_brief --tags options costs

Prints what has already been measured: the rules that are expensive to
re-learn, every campaign's realized average monthly return, the benchmark any
new idea must beat, and the current promotion status of everything in the repo.

The purpose is narrow and important — a new strategy should start from ten
campaigns of evidence rather than from a blank page, and should be able to say
which prior finding it is relying on or deliberately contradicting.
"""

from __future__ import annotations

import argparse
import sys

from catalyst.knowledge.base import (
    best_measured,
    campaigns_for,
    load_reports,
    rules_for,
)


def render(tags: tuple[str, ...] = ()) -> str:
    L: list[str] = []
    W = 78
    L += ["=" * W, "  DESIGN BRIEF — what this project has already measured", "=" * W]

    rules = rules_for(*tags)
    L += ["", f"  RULES ({len(rules)}) — do not re-test without new evidence", ""]
    for i, r in enumerate(rules, 1):
        L.append(f"  {i:2d}. {r.rule}")
        L.append(f"      evidence: {r.evidence}")
        L.append("")

    camps = sorted(campaigns_for(*tags), key=lambda c: c.avg_monthly, reverse=True)
    L += [f"  CAMPAIGNS ({len(camps)}) — realized avg monthly return, real costs", ""]
    L.append(f"  {'campaign':<46}{'instr':<9}{'monthly':>9}")
    L.append("  " + "-" * (W - 4))
    for c in camps:
        L.append(f"  {c.name[:44]:<46}{c.instrument:<9}{c.avg_monthly:>+8.2%}")
    L.append("")

    opts = best_measured("OPTIONS")
    shares = best_measured("SHARES")
    L += ["  THE BAR", "",
          f"    best options campaign : {opts.name} ({opts.avg_monthly:+.2%}/mo)",
          f"    best shares campaign  : {shares.name} ({shares.avg_monthly:+.2%}/mo)",
          "    a new options idea must beat BOTH the shares book and simply",
          "    owning the underlying, or it is adding risk for nothing.", ""]

    reports = load_reports()
    if reports:
        L += [f"  REPORTS ON DISK ({len(reports)})", ""]
        for r in reports:
            L.append(f"    {r.get('strategy','?'):<24} {r.get('verdict','')[:46]}")
        L.append("")

    try:
        from catalyst.strategies.registry import registry
        L += ["  PROMOTION STATUS", ""]
        for name, m in sorted(registry().items()):
            flag = ("LIVE-ELIGIBLE" if m.validated and m.paper_tested
                    else "validated" if m.validated else "not validated")
            L.append(f"    {name:<26} {m.status:<10} {flag}")
        L.append("")
    except Exception:                                   # noqa: BLE001
        pass

    L += ["  WRITING A NEW STRATEGY", "",
          "    1. Say which rule above it relies on, or which it contradicts",
          "       and why the old evidence no longer applies.",
          "    2. Scaffold it:",
          "       uv run python -m catalyst.runners.new_strategy --name <name>",
          "    3. Test it (real + zero cost, 70/30 OOS, N>100, concentration):",
          "       uv run python -m catalyst.runners.deploy_runner \\",
          "           --mode backtest --strategy <name>",
          "    4. A CANDIDATE verdict grants `validated`. Then paper, then live.",
          "=" * W]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=[],
                    help="filter, e.g. options costs risk signal tournament")
    args = ap.parse_args(argv)
    print(render(tuple(args.tags)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

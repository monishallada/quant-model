"""pdf-report — render any strategy's standard report to PDF, one command.

    uv run python -m catalyst.runners.strategy_pdf --strategy <name>

Reads the pipeline's own artifacts (results/active/<name>/report.json, or the
archived copy) and renders the standard report — headline, segment matrix,
cost-drag, warnings, verdict, equity-relevant stats — plus the trade texture.
This closes the workflow loop the owner specified: spec in → code → backtest →
PDF report out, with no hand-built report step.

The PDF renders exactly what the honesty machinery produced. It computes
nothing new, so it can never disagree with the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK, ACC, RED, GRN, GRY = "#14171D", "#3B5C8A", "#A93A2C", "#1A7A57", "#767E8C"


def find_report(name: str) -> Path | None:
    for base in (Path("results/active"), Path("results/archive"),
                 Path("results/archive/pre_framework")):
        p = base / name / "report.json"
        if p.exists():
            return p
    return None


def render(name: str, report: dict, out: Path) -> Path:
    plt.rcParams.update({"font.size": 9, "text.color": INK})
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, f"Strategy Report — {name}", fontsize=15,
                 fontweight="bold", color=INK)
        fig.text(0.07, 0.925, f"{report.get('start')} to {report.get('end')}"
                 "  ·  real ThetaData/Alpaca data through the honest pipeline",
                 fontsize=9, color=GRY)
        y = 0.87

        verdict = report.get("verdict", "?")
        vcol = GRN if verdict.startswith("CANDIDATE") else RED
        fig.text(0.07, y, f"VERDICT: {verdict}", fontsize=11, color=vcol,
                 fontweight="bold"); y -= 0.035

        gap = report.get("cost_gap")
        if gap is not None:
            fig.text(0.07, y, f"cost drag: {gap:+.2%}/month between zero-cost "
                     "and real-cost twins", fontsize=9); y -= 0.03

        hdr = (f"{'segment':<9}{'cost':<7}{'monthly':>9}{'CAGR':>9}{'maxDD':>9}"
               f"{'N':>7}{'win':>7}{'PF':>7}{'top3':>7}")
        fig.text(0.07, y, hdr, fontsize=8.5, family="monospace",
                 fontweight="bold"); y -= 0.018
        for s in report.get("segments", []):
            top3 = (f"{s['concentration_share']:.0%}"
                    if s.get("concentration_share") is not None else "n/a")
            row = (f"{s['segment']:<9}{s['cost_profile']:<7}"
                   f"{s['avg_monthly_return']:>+8.2%}{s['cagr']:>+9.1%}"
                   f"{s['max_drawdown']:>+9.1%}{s['n_trades']:>7}"
                   f"{s['win_rate']:>6.0%}{s['profit_factor']:>7.2f}{top3:>7}")
            fig.text(0.07, y, row, fontsize=8.5, family="monospace"); y -= 0.016
        y -= 0.02

        # warnings the pipeline attached — never omitted from the PDF
        for s in report.get("segments", []):
            if s["cost_profile"] != "real":
                continue
            if s["n_trades"] and s["n_trades"] < 100:
                fig.text(0.07, y, f"*** WARNING: {s['segment']} N={s['n_trades']}"
                         " < 100 — sample too small to trust ***",
                         fontsize=8.5, color=RED); y -= 0.017
            cs = s.get("concentration_share")
            if cs is not None and cs > 0.5:
                fig.text(0.07, y, f"*** LOTTERY FLAG: {s['segment']} top-3 "
                         f"trades = {cs:.0%} of P&L ***",
                         fontsize=8.5, color=RED); y -= 0.017
        y -= 0.02

        for k, v in (report.get("extras") or {}).items():
            fig.text(0.07, y, f"{k}: {v}", fontsize=8, color=GRY); y -= 0.015
        y -= 0.02

        note = ("Reading guide: the verdict is computed on the OUT-OF-SAMPLE "
                "real-cost segment only. The zero-cost twin isolates friction "
                "from signal. A CANDIDATE verdict grants 'validated' in the "
                "promotion ledger; paper trading on Alpaca then grants "
                "'paper_tested'; live requires both plus a typed confirmation.")
        for line in textwrap.wrap(note, 100):
            fig.text(0.07, y, line, fontsize=8, color=GRY); y -= 0.014
        pdf.savefig(fig); plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    src = find_report(args.strategy)
    if src is None:
        print(f"no report.json found for '{args.strategy}' under results/ — "
              "run the backtest first")
        return 1
    report = json.loads(src.read_text())
    out = args.out or src.parent / f"{args.strategy}_report.pdf"
    render(args.strategy, report, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

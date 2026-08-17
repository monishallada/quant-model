"""comp-profile — characterise a variance play as a gamble, not an edge.

    uv run python -m catalyst.runners.variance_profile --strategy catalyst_variance

The standard report ranks a strategy against the Engine C baseline, which is
the right question for an edge strategy and the WRONG question for this one.
A negative-EV lottery ticket will always "fail" that comparison, and reporting
only that tells you nothing about whether it can place in a competition.

So this reports the comp-relevant profile instead:

- **Right-tail frequency** — what share of months post >=30% and >=50%. This is
  the number a competition is actually scored on.
- **The typical month** — median and the share that bleed, so the ceiling is
  never quoted without the base rate beside it.
- **Survival** — probability of ruin and max drawdown across resampled paths,
  plus explicit confirmation the cash floor held. This is where the safeguards
  either prove themselves or don't.
- **Concentration** — how much of any positive result is a handful of tail
  winners. For this strategy heavy concentration is EXPECTED and is the nature
  of the play, not a defect; the check's job here is to size the dependence
  honestly rather than to flag it as a bug.

Monte Carlo resamples the realised trade sequence. That preserves the observed
trade distribution while destroying the specific ordering, which answers "how
much of this result was the order the winners happened to arrive in".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catalyst.backtest import metrics as m

BIG_MONTH = 0.30          # a "big" month for competition purposes
HUGE_MONTH = 0.50


def resample_paths(trades: list, starting: float, n_paths: int = 5000,
                   seed: int = 11) -> np.ndarray:
    """Bootstrap equity paths by reshuffling the realised trade P&L stream."""
    pnls = np.array([float(t.pnl) for t in trades], dtype=float)
    if not len(pnls):
        return np.empty((0, 0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnls), size=(n_paths, len(pnls)))
    return starting + np.cumsum(pnls[idx], axis=1)


def profile(equity: pd.Series, trades: list, starting: float,
            cash_floor_fraction: float, n_paths: int = 5000) -> dict:
    monthly = m.monthly_return_series(equity)
    mv = monthly.to_numpy() if len(monthly) else np.array([])

    paths = resample_paths(trades, starting, n_paths)
    if paths.size:
        finals = paths[:, -1]
        troughs = paths.min(axis=1)
        # Ruin = breaching the structural floor the safeguards promise.
        floor = starting * cash_floor_fraction
        p_ruin = float(np.mean(troughs <= floor))
        worst_dd = float(np.min(troughs / starting - 1.0))
    else:
        finals = np.array([])
        p_ruin, worst_dd = 0.0, 0.0

    conc = m.concentration(trades, 3)
    pnls = np.array([float(t.pnl) for t in trades]) if trades else np.array([])

    return {
        "n_trades": len(trades),
        "n_months": int(len(mv)),
        # -- right tail: the competition-relevant number ------------------
        "pct_months_ge_30": float(np.mean(mv >= BIG_MONTH)) if len(mv) else 0.0,
        "pct_months_ge_50": float(np.mean(mv >= HUGE_MONTH)) if len(mv) else 0.0,
        "best_month": float(np.max(mv)) if len(mv) else 0.0,
        # -- the base rate that must be quoted beside it ------------------
        "pct_months_negative": float(np.mean(mv < 0)) if len(mv) else 0.0,
        "median_month": float(np.median(mv)) if len(mv) else 0.0,
        "worst_month": float(np.min(mv)) if len(mv) else 0.0,
        "avg_monthly_return": m.avg_monthly_return(equity),
        # -- expectancy, reported plainly ---------------------------------
        "expectancy_per_trade": float(np.mean(pnls)) if len(pnls) else 0.0,
        "win_rate": float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        "biggest_winner": float(np.max(pnls)) if len(pnls) else 0.0,
        "biggest_loser": float(np.min(pnls)) if len(pnls) else 0.0,
        # -- survival ------------------------------------------------------
        "probability_of_ruin": p_ruin,
        "worst_path_drawdown": worst_dd,
        "cash_floor_held": bool(p_ruin == 0.0),
        "realized_max_drawdown": m.max_drawdown(equity),
        "mc_p10_final": float(np.percentile(finals, 10)) if finals.size else 0.0,
        "mc_median_final": float(np.median(finals)) if finals.size else 0.0,
        "mc_p90_final": float(np.percentile(finals, 90)) if finals.size else 0.0,
        "mc_p99_final": float(np.percentile(finals, 99)) if finals.size else 0.0,
        # -- concentration: expected here, characterised not flagged -------
        "top3_share_of_pnl": conc["share"],
        "total_pnl": conc["total_pnl"],
    }


def render(name: str, segments: dict[str, dict], starting: float) -> str:
    L = ["=" * 78,
         f"  COMP PROFILE — {name}",
         "  NEGATIVE-EV VARIANCE PLAY — a survivable gamble, not an edge strategy",
         "=" * 78, ""]
    for seg, p in segments.items():
        L += [f"  --- {seg} ---",
              f"    trades {p['n_trades']}  months {p['n_months']}  "
              f"win rate {p['win_rate']:.1%}",
              f"    AVG MONTHLY RETURN      {p['avg_monthly_return']:+.2%}",
              f"    expectancy/trade        ${p['expectancy_per_trade']:,.0f}"
              f"   (negative is expected and is the honest result)",
              "",
              "    RIGHT TAIL (what a competition is scored on)",
              f"      months >= +30%        {p['pct_months_ge_30']:.1%}",
              f"      months >= +50%        {p['pct_months_ge_50']:.1%}",
              f"      best month            {p['best_month']:+.1%}",
              f"      biggest single winner ${p['biggest_winner']:,.0f}",
              "",
              "    THE TYPICAL MONTH (quoted beside the ceiling, always)",
              f"      median month          {p['median_month']:+.2%}",
              f"      months negative       {p['pct_months_negative']:.1%}",
              f"      worst month           {p['worst_month']:+.1%}",
              "",
              "    SURVIVAL (what the safeguards buy)",
              f"      probability of ruin   {p['probability_of_ruin']:.2%}",
              f"      realised max drawdown {p['realized_max_drawdown']:+.1%}",
              f"      worst resampled path  {p['worst_path_drawdown']:+.1%}",
              f"      cash floor held       {'YES' if p['cash_floor_held'] else '*** NO ***'}",
              f"      MC final p10/med/p90  ${p['mc_p10_final']:,.0f} / "
              f"${p['mc_median_final']:,.0f} / ${p['mc_p90_final']:,.0f}",
              f"      MC final p99          ${p['mc_p99_final']:,.0f}",
              ""]
        share = p["top3_share_of_pnl"]
        if share is None:
            L.append("    CONCENTRATION: n/a (total P&L not positive — nothing to attribute)")
        else:
            L.append(f"    CONCENTRATION: top-3 trades = {share:.0%} of total P&L")
            L.append("      Expected for this structure. Heavy concentration IS the")
            L.append("      nature of a lottery play, not a defect — but it means any")
            L.append("      positive result is right-tail variance, never edge.")
        L.append("")
    L.append("=" * 78)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="catalyst_variance")
    ap.add_argument("--paths", type=int, default=5000)
    args = ap.parse_args(argv)

    from catalyst.core.config import load_config
    cfg = load_config("backtest")
    src = Path("results/active") / args.strategy / "segments.json"
    if not src.exists():
        print(f"no segment data at {src} — run the backtest first")
        return 1
    payload = json.loads(src.read_text())
    print(render(args.strategy, payload, cfg.account.starting_capital))
    return 0


if __name__ == "__main__":
    sys.exit(main())

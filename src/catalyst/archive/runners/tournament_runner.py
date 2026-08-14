"""Tournament simulator: P(hitting target) across rolling competition windows.

Reports the honest distribution — probability of each threshold AND probability
of ruin — for a range of architecture configurations. Research only.
"""

from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.alpha.data import load_price_panel
from catalyst.alpha.signals import build_signal_panels, cross_sectional_zscore
from catalyst.tournament.engine import TournamentConfig, run_tournament_path
from catalyst.tournament.objective import evaluate_paths, rolling_windows

logger = logging.getLogger(__name__)


def composite_signal(prices, market):
    """Equal-weight blend of the momentum signals validated in v5."""
    sig = build_signal_panels(prices, market)
    use = ["mom_12_1", "vol_adj_mom", "residual_mom"]
    return cross_sectional_zscore(sum(sig[k].fillna(0.0) for k in use))


def run_config(label, prices, signal, windows, cfg, months):
    paths = []
    for i, (s, e) in enumerate(windows):
        p = run_tournament_path(prices, signal, s, e, cfg, seed=i)
        if not p.empty:
            paths.append(p)
    return evaluate_paths(label, paths, months)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=None)
    ap.add_argument("--end", type=date.fromisoformat, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/tournament"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    t = cfg.tournament
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)

    prices, market = load_price_panel(cfg, t.universe, start, end, min_coverage=0.25)
    logger.info("panel: %d names x %d dates", prices.shape[1], prices.shape[0])
    signal = composite_signal(prices, market)
    windows = rolling_windows(prices.index, t.window_months)
    logger.info("%d rolling %d-month windows", len(windows), t.window_months)

    base = dict(n_positions=t.n_positions, hold_days=t.hold_days, dte_days=t.dte_days,
                moneyness=t.moneyness, capital_fraction=t.capital_fraction,
                spread_pct=t.spread_pct, cost_per_contract=t.cost_per_contract)

    variants = {
        "A. baseline (3 pos, 10% OTM, 90% deployed)": TournamentConfig(**base),
        "B. max concentration (1 position)": TournamentConfig(**{**base, "n_positions": 1}),
        "C. deep OTM (1.25 strike)": TournamentConfig(**{**base, "moneyness": 1.25}),
        "D. deep OTM + 1 position": TournamentConfig(**{**base, "n_positions": 1, "moneyness": 1.25}),
        "E. full deployment (100%)": TournamentConfig(**{**base, "capital_fraction": 1.0}),
        "F. NO SIGNAL control (random picks)": TournamentConfig(**{**base, "use_signal": False}),
        "G. take profit at 3x (caps tail)": TournamentConfig(**{**base, "take_profit_multiple": 3.0}),
    }

    rows = []
    print()
    for label, vcfg in variants.items():
        res = run_config(label, prices, signal, windows, vcfg, t.window_months)
        print(res.report()); print()
        rows.append({"config": label, **res.summary()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "variants.json").write_text(json.dumps(rows, indent=2))
    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()

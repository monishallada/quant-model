"""Alpha research lab — signal diagnostics before any portfolio is built.

Answers the question that should precede every strategy: does this signal
know anything about future returns? Runs the signal library against forward
returns at several horizons and reports information coefficients, t-stats,
and decile spreads. Backtest/research only; no broker of any kind.

Usage:
    uv run python -m catalyst.runners.alpha_lab_runner --out results/alpha
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.strategies.archive.alpha.data import load_price_panel
from catalyst.strategies.archive.alpha.evaluation import evaluate_library
from catalyst.strategies.archive.alpha.signals import build_signal_panels

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, default=None)
    p.add_argument("--end", type=date.fromisoformat, default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=[5, 21, 63])
    p.add_argument("--split", action="store_true", help="also report train/test separately")
    p.add_argument("--out", type=Path, default=Path("results/alpha"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config("backtest")
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)

    prices, market = load_price_panel(cfg, cfg.drift.universe, start, end)
    logger.info("panel: %d symbols x %d dates (%s -> %s)", prices.shape[1], prices.shape[0],
                prices.index.min().date(), prices.index.max().date())

    signals = build_signal_panels(prices, market)
    args.out.mkdir(parents=True, exist_ok=True)

    segments = [("all", prices)]
    if args.split:
        boundary = prices.index[int(len(prices) * cfg.backtest.train_test_split)]
        segments = [("train", prices.loc[:boundary]), ("test", prices.loc[boundary:]),
                    ("all", prices)]

    for seg_name, seg_prices in segments:
        seg_signals = {k: v.loc[seg_prices.index] for k, v in signals.items()}
        table = evaluate_library(seg_signals, seg_prices, args.horizons)
        table = table.sort_values(["horizon_d", "t_stat"], ascending=[True, False])
        print(f"\n===== SIGNAL DIAGNOSTICS [{seg_name}] "
              f"{seg_prices.index.min().date()} .. {seg_prices.index.max().date()} =====")
        print(table.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        table.to_json(args.out / f"ic-{seg_name}.json", orient="records", indent=2)

    print(f"\nartifacts -> {args.out}/")


if __name__ == "__main__":
    main()

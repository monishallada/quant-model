"""v7 Combined Allocator runner — the single active strategy.

80% stable alpha sleeve + 20% convex sleeve, reported with AVERAGE MONTHLY
RETURN as the headline number. Backtest only; no live broker.
"""

from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.alpha.data import load_price_panel
from catalyst.alpha.portfolio import PortfolioConfig, run_portfolio_backtest
from catalyst.alpha.signals import build_signal_panels
from catalyst.allocator.combined import (AllocatorConfig, combine_sleeves, to_daily_returns)
from catalyst.backtest import metrics as m
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.tournament.engine import TournamentConfig
from catalyst.tournament.real_chain import run_real_chain_path
from catalyst.archive.runners.tournament_runner import composite_signal

logger = logging.getLogger(__name__)


def build_stable_sleeve(cfg, prices, market):
    """v5 portable alpha: market-neutral overlay on SPY."""
    signals = build_signal_panels(prices, market)
    pcfg = PortfolioConfig(rebalance_days=5, n_long=20, n_short=20, gross_exposure=1.0,
                           market_neutral=True, score_weighted=True,
                           signal_smoothing_days=5, beta_neutralize=True)
    out = run_portfolio_backtest(prices, market, signals, pcfg)
    alpha_r = out.equity.pct_change().fillna(0.0)
    spy_r = out.benchmark.pct_change().fillna(0.0)
    return (alpha_r + spy_r), out.benchmark


def build_convex_sleeve(cfg, prices_hv, market_hv, data, start, end):
    """v6 tournament engine on real chains, balanced configuration."""
    t = cfg.tournament
    signal = composite_signal(prices_hv, market_hv)
    expiries = {s: data.list_expirations(s) for s in prices_hv.columns}
    tcfg = TournamentConfig(n_positions=8, moneyness=1.20, capital_fraction=0.50,
                            hold_days=t.hold_days, dte_days=t.dte_days,
                            cost_per_contract=t.cost_per_contract)
    path = run_real_chain_path(prices_hv, signal, data, pd.Timestamp(start),
                               pd.Timestamp(end), tcfg, expiries)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 4))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--convex", type=float, default=0.20)
    ap.add_argument("--no-rebalance", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/allocator"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)

    print("building stable sleeve (v5 portable alpha)...")
    prices, market = load_price_panel(cfg, cfg.drift.universe, args.start, args.end)
    stable_r, bench = build_stable_sleeve(cfg, prices, market)

    print("building convex sleeve (v6 tournament, real chains)...")
    prices_hv, market_hv = load_price_panel(cfg, cfg.tournament.universe, args.start,
                                            args.end, min_coverage=0.5)
    convex_eq = build_convex_sleeve(cfg, prices_hv, market_hv, data, args.start, args.end)

    idx = stable_r.index
    convex_r = to_daily_returns(convex_eq, idx) if not convex_eq.empty else pd.Series(0.0, index=idx)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    variants = {
        f"v7 combined ({int(args.convex*100)}% convex, rebalanced)":
            AllocatorConfig(convex_fraction=args.convex, rebalance_sleeves=True),
        f"v7 combined ({int(args.convex*100)}% convex, let run)":
            AllocatorConfig(convex_fraction=args.convex, rebalance_sleeves=False),
        "stable sleeve only (100% alpha)":
            AllocatorConfig(convex_fraction=0.0, rebalance_sleeves=True),
        "convex sleeve only (100% tournament)":
            AllocatorConfig(convex_fraction=1.0, rebalance_sleeves=False),
    }
    print()
    for label, acfg in variants.items():
        res = combine_sleeves(stable_r, convex_r, bench, acfg)
        print(res.report(label)); print()
        rows.append(res.summary(label))
        if "rebalanced" in label:
            pd.DataFrame({"equity": res.equity, "benchmark": res.benchmark,
                          "stable": res.stable_sleeve, "convex": res.convex_sleeve}
                         ).to_json(args.out / "curve-v7.json", orient="split", indent=2)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()

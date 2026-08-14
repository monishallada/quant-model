"""Market-neutral alpha portfolio backtest with walk-forward signal weighting."""

from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.alpha.data import load_price_panel
from catalyst.alpha.portfolio import PortfolioConfig, run_portfolio_backtest
from catalyst.alpha.signals import build_signal_panels

logger = logging.getLogger(__name__)


def report(name: str, out, seg: str) -> dict:
    s = out.stats()
    print(f"\n===== {name} [{seg}] =====")
    print(f"  CAGR {s['cagr']:+7.2%}   |  SPY {s['benchmark_cagr']:+7.2%}   "
          f"|  EXCESS {s['excess_cagr']:+7.2%}")
    print(f"  vol {s['vol']:6.2%} | sharpe {s['sharpe']:5.2f} | maxDD {s['max_drawdown']:+7.2%}")
    print(f"  beta {s['beta']:+5.2f} | alpha(ann) {s['alpha_ann']:+7.2%} | "
          f"avg turnover {s['avg_turnover']:.2f}")
    return {"name": name, "segment": seg, **s}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, default=None)
    p.add_argument("--end", type=date.fromisoformat, default=None)
    p.add_argument("--out", type=Path, default=Path("results/alpha"))
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--rebalance", type=int, default=5)
    p.add_argument("--n-side", type=int, default=20)
    p.add_argument("--gross", type=float, default=1.0)
    p.add_argument("--label", default="market_neutral")
    p.add_argument("--score-weighted", action="store_true")
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--beta-neutral", action="store_true")
    p.add_argument("--overlay-on-spy", action="store_true",
                   help="portable alpha: hold SPY and run the market-neutral book on top")
    p.add_argument("--drop-top-winners", type=int, default=0,
                   help="survivorship-bias probe: exclude the N best total-return names")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)
    prices, market = load_price_panel(cfg, cfg.drift.universe, start, end)
    if args.drop_top_winners:
        # Our universe was selected in 2026 for being liquid TODAY, which tilts it
        # toward names that already won. If the alpha is a survivorship artifact,
        # removing the biggest winners should destroy it.
        total_ret = (prices.iloc[-1] / prices.iloc[0]).dropna().sort_values()
        drop = list(total_ret.index[-args.drop_top_winners:])
        logger.info("dropping %d best performers: %s", len(drop), ", ".join(drop))
        prices = prices.drop(columns=drop)
    signals = build_signal_panels(prices, market)

    pcfg = PortfolioConfig(rebalance_days=args.rebalance, n_long=args.n_side,
                           n_short=args.n_side, gross_exposure=args.gross,
                           market_neutral=not args.long_only,
                           score_weighted=args.score_weighted,
                           signal_smoothing_days=args.smooth,
                           beta_neutralize=args.beta_neutral)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    boundary = prices.index[int(len(prices) * cfg.backtest.train_test_split)]
    for seg, px in (("all", prices), ("train", prices.loc[:boundary]),
                    ("test", prices.loc[boundary:])):
        sig = {k: v.loc[px.index] for k, v in signals.items()}
        out = run_portfolio_backtest(px, market.loc[px.index], sig, pcfg)
        if args.overlay_on_spy:
            # Portable alpha: the market-neutral book is beta-0 and uncorrelated,
            # so its returns ADD to a passive SPY holding rather than replacing it.
            # This is how a modest standalone alpha becomes a high absolute return
            # without leverage on either component alone.
            alpha_r = out.equity.pct_change().fillna(0.0)
            spy_r = out.benchmark.pct_change().fillna(0.0)
            combined = (1.0 + alpha_r + spy_r).cumprod() * out.equity.iloc[0]
            out.equity = combined
        rows.append(report(args.label, out, seg))
        if seg == "all":
            pd.DataFrame({"equity": out.equity, "benchmark": out.benchmark}).to_json(
                args.out / f"curve-{args.label}.json", orient="split", indent=2)
    (args.out / f"portfolio-{args.label}.json").write_text(json.dumps(rows, indent=2))
    print(f"\nartifacts -> {args.out}/")


if __name__ == "__main__":
    main()

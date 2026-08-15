"""v9 sweep — map the configuration frontier for straight long calls/puts.

Two questions the single baseline cannot answer:

1. **Does the momentum signal do anything?** The ``calls_only`` control buys a
   call every cycle regardless of signal. If signal-directed and calls-only
   score the same, the signal is decoration and the returns are just the fact
   that these six names rose over the sample.

2. **What does the aggressiveness dial actually buy?** Moneyness and capital
   fraction trade average return against tail. For a threshold objective the
   right dial setting is the one that maximizes P(target), which is NOT the one
   that maximizes the mean — and this sweep measures the gap directly.

A portfolio mode is included because it changes the answer: six names running
concurrently at 1/6 weight is a fundamentally different outcome distribution
from one name at full weight, and the portfolio is what would actually trade.

Every variant reuses the chains the baseline already cached, so the sweep costs
no additional data requests.
"""

from __future__ import annotations

import argparse, itertools, json, logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from catalyst.core.config import load_config
from catalyst.backtest import metrics as m
from catalyst.data.cache import ParquetCache
from catalyst.data.intraday import AlpacaMinuteBars
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.strategies.archive.long_options_standalone.engine import LongOptionConfig, run_long_options
from catalyst.strategies.archive.tournament.objective import rolling_windows

logger = logging.getLogger(__name__)


def window_multiples(equity: pd.Series, months: int) -> list[float]:
    out = []
    for s, e in rolling_windows(pd.DatetimeIndex(equity.index), months, step_days=21):
        seg = equity.loc[s:e]
        if len(seg) > 2 and seg.iloc[0] > 0:
            out.append(float(seg.iloc[-1] / seg.iloc[0]))
    return out


def combine_equal_weight(curves: list[pd.Series], starting: float = 100_000.0) -> pd.Series:
    """Equal-weight portfolio: each name gets 1/N of capital, rebalanced never.

    No rebalancing is deliberate — rebalancing into a losing convex sleeve is a
    way to feed a whole account into decay, and letting winners run is the
    entire point of the structure.
    """
    if not curves:
        return pd.Series(dtype=float)
    idx = curves[0].index
    for c in curves[1:]:
        idx = idx.union(c.index)
    w = starting / len(curves)
    total = None
    for c in curves:
        norm = (c / c.iloc[0] * w).reindex(idx).ffill().bfill()
        total = norm if total is None else total + norm
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--symbols", nargs="+",
                    default=["TSLA", "AMD", "NVDA", "AMZN", "META", "MSFT"])
    ap.add_argument("--window-months", type=int, default=7, help="competition length")
    ap.add_argument("--out", type=Path, default=Path("results/long_options/sweep"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")

    prices, expiries = {}, {}
    for s in args.symbols:
        df = bars.daily_bars(s, args.start, args.end)
        if df.empty:
            continue
        prices[s] = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(pd.Index(df.index)))
        expiries[s] = data.list_expirations(s)
    print(f"loaded {len(prices)} symbols\n")

    # Capital fraction spans two orders of behaviour: 0.10 is survivable
    # sizing, 1.00 is one zero away from a wipeout. A threshold objective
    # rewards boldness, but only up to the point where ruin probability
    # swallows the tail — this grid is wide enough to show where that turns.
    grid = list(itertools.product([1.00, 1.05, 1.10], [0.10, 0.25, 0.50, 1.00], [True, False]))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for moneyness, frac, use_signal in grid:
        lcfg = LongOptionConfig(moneyness=moneyness, capital_fraction=frac,
                                use_signal=use_signal, allow_puts=use_signal)
        tag = f"m{moneyness:.2f}_f{frac:.2f}_{'signal' if use_signal else 'callsonly'}"
        curves, per_symbol, ruined = [], {}, []
        for s in prices:
            r = run_long_options(s, prices[s], data, lcfg, expiries[s])
            if not r.equity.empty and r.trades:
                curves.append(r.equity)
                per_symbol[s] = m.avg_monthly_return(r.equity)
                if r.ruined:
                    ruined.append(s)
        if not curves:
            continue
        port = combine_equal_weight(curves)
        h = m.headline(port)
        mults = window_multiples(port, args.window_months)
        p = lambda x: float(np.mean([mm >= x for mm in mults])) if mults else 0.0
        row = {"config": tag, "moneyness": moneyness, "capital_fraction": frac,
               "use_signal": use_signal, **h, "n_windows": len(mults),
               "p_2x": p(2.0), "p_3x": p(3.0), "p_5x": p(5.0), "p_10x": p(10.0),
               "median_window": float(np.median(mults)) if mults else 0.0,
               "best_window": float(max(mults)) if mults else 0.0,
               "per_symbol": per_symbol, "ruined_symbols": ruined,
               "n_ruined": len(ruined)}
        rows.append(row)
        print(f"{tag:32s} {h['avg_monthly_return']:+7.2%}/mo  maxDD {h['max_drawdown']:+6.1%}  "
              f"P(2x) {p(2.0):4.0%}  P(3x) {p(3.0):4.0%}  P(5x) {p(5.0):4.0%}  "
              f"P(10x) {p(10.0):4.0%}  median {row['median_window']:.2f}x"
              + (f"  RUINED {len(ruined)}/{len(prices)}" if ruined else ""))
        (args.out / "sweep.json").write_text(json.dumps(rows, indent=2, default=str))

    if not rows:
        print("no results"); return

    df = pd.DataFrame(rows)
    print("\n\n===== SIGNAL VALUE: signal-directed vs calls-only =====")
    for sig, g in df.groupby("use_signal"):
        lbl = "signal-directed" if sig else "calls-only (control)"
        print(f"  {lbl:24s} mean {g['avg_monthly_return'].mean():+7.2%}/mo   "
              f"P(3x) {g['p_3x'].mean():4.0%}   P(5x) {g['p_5x'].mean():4.0%}")

    print("\n===== THE GAP: best by average return vs best by P(target) =====")
    ba = df.loc[df["avg_monthly_return"].idxmax()]
    bp = df.loc[df["p_5x"].idxmax()]
    print(f"  max avg return : {ba['config']:32s} {ba['avg_monthly_return']:+7.2%}/mo  P(5x) {ba['p_5x']:.0%}")
    print(f"  max P(5x)      : {bp['config']:32s} {bp['avg_monthly_return']:+7.2%}/mo  P(5x) {bp['p_5x']:.0%}")
    print(f"\nartifacts -> {args.out}/")


if __name__ == "__main__":
    main()

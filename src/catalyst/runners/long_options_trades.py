"""Dump the per-position multiple distribution — the number that explains
everything else. Equity paths are sizing-dependent; the multiple is not."""
from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from catalyst.core.config import load_config
from catalyst.data.cache import ParquetCache
from catalyst.data.intraday import AlpacaMinuteBars
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.strategies.archive.long_options_standalone.engine import LongOptionConfig, run_long_options


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+",
                    default=["TSLA", "AMD", "NVDA", "AMZN", "META", "MSFT"])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--moneyness", type=float, default=1.05)
    ap.add_argument("--out", type=Path, default=Path("results/long_options/trades"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config("backtest")
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    # ruin_floor=0 so every scheduled cycle runs: we want the full population of
    # positions, not the truncated path that a sized account would have died on.
    lcfg = LongOptionConfig(moneyness=args.moneyness, capital_fraction=0.01)

    args.out.mkdir(parents=True, exist_ok=True)
    allt = []
    for s in args.symbols:
        df = bars.daily_bars(s, args.start, args.end)
        if df.empty:
            continue
        px = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(pd.Index(df.index)))
        r = run_long_options(s, px, data, lcfg, data.list_expirations(s), ruin_floor=0.0)
        mults = np.array([t.multiple for t in r.trades])
        if not len(mults):
            continue
        calls = [t.multiple for t in r.trades if t.right == "C"]
        puts = [t.multiple for t in r.trades if t.right == "P"]
        print(f"{s:6s} n={len(mults):3d}  mean {mults.mean():.3f}x  median {np.median(mults):.3f}x  "
              f"win {np.mean(mults>1)*100:4.1f}%  zeros {np.mean(mults<0.02)*100:4.1f}%  "
              f"p90 {np.percentile(mults,90):.2f}x  max {mults.max():.1f}x  "
              f"| calls {np.mean(calls) if calls else 0:.2f}x (n={len(calls)})  "
              f"puts {np.mean(puts) if puts else 0:.2f}x (n={len(puts)})")
        allt += [{"symbol": s, "entry": str(t.entry), "right": t.right,
                  "entry_price": t.entry_price, "exit_price": t.exit_price,
                  "multiple": t.multiple} for t in r.trades]
        (args.out / "trades.json").write_text(json.dumps(allt, indent=2))

    if allt:
        m = np.array([t["multiple"] for t in allt])
        print(f"\n===== ALL {len(m)} POSITIONS =====")
        print(f"  mean {m.mean():.3f}x   median {np.median(m):.3f}x   win rate {np.mean(m>1)*100:.1f}%")
        print(f"  expired worthless: {np.mean(m<0.02)*100:.1f}%   >=2x: {np.mean(m>=2)*100:.1f}%   "
              f">=5x: {np.mean(m>=5)*100:.1f}%   max {m.max():.1f}x")
        verb = "gain" if m.mean() >= 1 else "loss"
        print(f"  --> the average long option returns {m.mean():.3f}x, a "
              f"{abs(m.mean()-1)*100:.1f}% {verb} per 3-week position; "
              f"median {np.median(m):.3f}x")


if __name__ == "__main__":
    main()

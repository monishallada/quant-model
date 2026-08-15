"""v9 per-symbol lab runner — avg monthly return, stock by stock. Backtest only."""
from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.backtest import metrics as m
from catalyst.data.cache import ParquetCache
from catalyst.data.intraday import AlpacaMinuteBars
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.persymbol.lab import buy_and_hold, covered_calls, put_credit_spreads

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/persymbol"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    pcfg = cfg.persymbol
    symbols = args.symbols or list(pcfg.universe)
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in symbols:
        df = bars.daily_bars(sym, args.start, args.end)
        if df.empty:
            print(f"{sym}: no price data"); continue
        px = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(pd.Index(df.index)))
        expiries = data.list_expirations(sym)
        print(f"\n########## {sym} ##########")
        for fn, name in ((buy_and_hold, "buy_and_hold"),
                         (covered_calls, "covered_calls"),
                         (put_credit_spreads, "put_credit_spread")):
            res = fn(px, starting=100_000.0) if name == "buy_and_hold" else \
                  fn(sym, px, data, pcfg, expiries)
            res.symbol = sym
            if res.equity.empty: 
                print(f"  {name:20} no result"); continue
            h = res.headline()
            print(f"  {name:20} AVG MONTHLY {h['avg_monthly_return']:+7.2%} | "
                  f"CAGR {h['cagr']:+7.1%} | maxDD {h['max_drawdown']:+7.1%} | "
                  f"trades {h['n_trades']:3} | win {h['win_rate']:5.0%}")
            rows.append({"symbol": sym, "strategy": name, **h})
            (args.out / "results.json").write_text(json.dumps(rows, indent=2, default=str))

    if rows:
        print("\n\n=========== AVG MONTHLY RETURN, STOCK BY STOCK ===========")
        t = pd.DataFrame(rows).pivot(index="symbol", columns="strategy",
                                     values="avg_monthly_return")
        print((t*100).round(2).to_string())
    print(f"\nartifacts -> {args.out}/")


if __name__ == "__main__":
    main()

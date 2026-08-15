"""v8 Index VRP runner — the single active strategy. Backtest only."""
from __future__ import annotations

import argparse, json, logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import load_config
from catalyst.backtest import metrics as m
from catalyst.backtest.walkforward import chronological_split
from catalyst.data.cache import ParquetCache
from catalyst.data.intraday import AlpacaMinuteBars
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.strategies.archive.index_vrp.backtester import run_index_vrp

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--out", type=Path, default=Path("results/index_vrp"))
    ap.add_argument("--symbols", nargs="+", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    if args.symbols:
        cfg = load_config("backtest", overrides={"index_vrp.symbols": args.symbols})
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    spy = bars.daily_bars("SPY", args.start, args.end)
    benchmark = pd.Series(spy["close"].to_numpy(), index=pd.to_datetime(pd.Index(spy.index)))

    boundary, _ = chronological_split(args.start, args.end, cfg.backtest.train_test_split)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg, s, e in (("all", args.start, args.end), ("train", args.start, boundary),
                      ("test", boundary, args.end)):
        res = run_index_vrp(cfg, data, benchmark, s, e, label=f"v8-index-vrp [{seg}]")
        print(res.report(f"v8 Index VRP [{seg}]  {s} .. {e}")); print()
        h = m.headline(res.equity)
        rows.append({"segment": seg, **h, "n_trades": len(res.trades),
                     "profit_factor": m.profit_factor(res.trades),
                     "benchmark_monthly": m.headline(res.benchmark)["avg_monthly_return"],
                     "diagnostics": res.diagnostics})
        if seg == "all":
            pd.DataFrame({"equity": res.equity, "benchmark": res.benchmark}).to_json(
                args.out / "curve.json", orient="split", indent=2)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()

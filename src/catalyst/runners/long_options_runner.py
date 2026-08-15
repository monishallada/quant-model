"""v9 — straight long calls/puts per symbol. Avg monthly return AND threshold
probabilities, because those two metrics disagree for a convex structure and
the disagreement is the decision. Backtest only."""
from __future__ import annotations

import argparse, json, logging
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


def window_multiples(equity: pd.Series, months: int = 7) -> list[float]:
    """Competition-style outcome distribution over rolling windows."""
    out = []
    for s, e in rolling_windows(pd.DatetimeIndex(equity.index), months, step_days=21):
        seg = equity.loc[s:e]
        if len(seg) > 2 and seg.iloc[0] > 0:
            out.append(float(seg.iloc[-1] / seg.iloc[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 1))
    ap.add_argument("--symbols", nargs="+",
                    default=["TSLA", "AMD", "NVDA", "AMZN", "META", "MSFT"])
    ap.add_argument("--moneyness", type=float, default=1.05)
    ap.add_argument("--capital-fraction", type=float, default=0.25)
    ap.add_argument("--calls-only", action="store_true",
                    help="ignore the momentum signal; always buy calls (control)")
    ap.add_argument("--out", type=Path, default=Path("results/long_options"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config("backtest")
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata), cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    lcfg = LongOptionConfig(moneyness=args.moneyness, capital_fraction=args.capital_fraction,
                            use_signal=not args.calls_only,
                            allow_puts=not args.calls_only)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in args.symbols:
        df = bars.daily_bars(sym, args.start, args.end)
        if df.empty:
            print(f"{sym}: no price data"); continue
        px = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(pd.Index(df.index)))
        res = run_long_options(sym, px, data, lcfg, data.list_expirations(sym))
        if res.unpriceable:
            print(f"{sym}: {res.unpriceable} cycles skipped (contract absent at exit)")
        if res.equity.empty or not res.trades:
            print(f"{sym}: no trades"); continue
        h = m.headline(res.equity)
        bh = m.headline(px / px.iloc[0] * 100_000)
        mults = window_multiples(res.equity)
        p = lambda x: float(np.mean([mm >= x for mm in mults])) if mults else 0.0
        print(f"\n########## {sym} ##########")
        print(f"  AVG MONTHLY RETURN: {h['avg_monthly_return']:+7.2%}   "
              f"(buy & hold: {bh['avg_monthly_return']:+7.2%})")
        print(f"  CAGR {h['cagr']:+8.1%} | maxDD {h['max_drawdown']:+7.1%} | "
              f"months positive {h['pct_months_positive']:.0%}")
        print(f"  trades {len(res.trades)} | win rate {res.win_rate:.1%} | "
              f"wipeouts {res.wipeout_rate:.0%} | best single option {res.best_multiple:.1f}x")
        if res.ruined:
            print(f"  *** RUINED {res.ruined_on} — equity fell below 1% of starting; "
                  f"trading stopped. The low trade count is the wipeout, not missing data. ***")
        print(f"  7-month windows (n={len(mults)}): P(2x) {p(2):.0%} | P(5x) {p(5):.0%} | "
              f"P(10x) {p(10):.0%} | median {np.median(mults) if mults else 0:.2f}x")
        rows.append({"symbol": sym, **h, "buyhold_monthly": bh["avg_monthly_return"],
                     "n_trades": len(res.trades), "win_rate": res.win_rate,
                     "ruined": res.ruined, "ruined_on": res.ruined_on,
                     "wipeout_rate": res.wipeout_rate, "best_option": res.best_multiple,
                     "p_2x": p(2), "p_5x": p(5), "p_10x": p(10),
                     "median_window": float(np.median(mults)) if mults else 0.0})
        (args.out / "results.json").write_text(json.dumps(rows, indent=2, default=str))

    if rows:
        print("\n\n=========== AVG MONTHLY RETURN, STOCK BY STOCK ===========")
        t = pd.DataFrame(rows)[["symbol","avg_monthly_return","buyhold_monthly",
                                "win_rate","p_5x","p_10x","max_drawdown"]]
        t.columns = ["symbol","LONG OPTIONS","buy&hold","win rate","P(5x)","P(10x)","maxDD"]
        for c in t.columns[1:]:
            t[c] = (t[c]*100).round(1)
        print(t.to_string(index=False))
    print(f"\nartifacts -> {args.out}/")


if __name__ == "__main__":
    main()

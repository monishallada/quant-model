"""paper-sim — MOSAIC paper-trading SIMULATION. No broker. Ever.

    uv run python -m catalyst.runners.paper_sim --start 2026-05-01 --end 2026-08-01

This is the pre-deployment simulation the v16 directive requires: the machine
behaves as though it is trading the $100,000 account — minute loop, fills,
slippage, rejections, position management, P&L — but every order terminates in
the audited SimulatedBroker. There is NO code path from this runner to
AlpacaBroker or SchwabBroker: it never imports them, never reads credentials,
and runs entirely from historical/cached data through the same intraday engine
the backtests use. Deployment to any real (even paper) account requires the
separate deploy_runner path, which stays gated behind the operator's explicit
"APPROVE PAPER DEPLOYMENT".

The window passed here must be the UNTOUCHED HOLDOUT — sessions never used in
any fitting, sweep, or look at any point of the campaign. The run report says
so on its face.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from catalyst.core.config import load_config
from catalyst.observability.killswitch import configure_json_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/active/mosaic_paper_sim"))
    args = ap.parse_args(argv)

    configure_json_logging(logging.INFO)
    cfg = load_config("backtest")           # backtest env: SimulatedBroker only

    from catalyst.backtest.engines import IntradayNativeEngine
    from catalyst.data.alpaca_history import AlpacaDailyBars
    from catalyst.data.cache import ParquetCache
    from catalyst.data.thetadata_client import ThetaDataClient
    from catalyst.data.thetadata_historical import ThetaDataHistorical
    from catalyst.strategies.registry import load_strategy

    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata),
                               cache=cache,
                               history_provider=AlpacaDailyBars(cfg.data.alpaca, cache))
    strategy = load_strategy("mosaic", cfg)

    print("=" * 70)
    print("  MOSAIC PAPER SIMULATION — RESEARCH MODE")
    print("  broker: SimulatedBroker (no external connectivity of any kind)")
    print(f"  window: {args.start} .. {args.end}  (must be the untouched holdout)")
    print(f"  account: ${cfg.account.starting_capital:,.0f} simulated")
    print("=" * 70)

    engine = IntradayNativeEngine()
    result = engine.run(strategy, args.start, args.end, cfg=cfg, data=data,
                        signal=None, catalysts=[], screener=None,
                        zero_cost=False)
    if result.error:
        print(f"SIMULATION FAILED: {result.error}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    equity = result.equity
    trades = result.trades
    import pandas as pd
    pd.DataFrame([t.model_dump() for t in trades]).to_csv(
        args.out / "paper_sim_trades.csv", index=False)
    equity.rename("equity").to_csv(args.out / "paper_sim_equity.csv",
                                   index_label="date")

    daily = equity.pct_change().dropna()
    summary = {
        "window": f"{args.start}..{args.end}",
        "mode": "PAPER SIMULATION (SimulatedBroker; no broker connectivity)",
        "sessions": int(len(equity)),
        "n_trades": len(trades),
        "final_equity": float(equity.iloc[-1]) if len(equity) else None,
        "total_return": (float(equity.iloc[-1] / equity.iloc[0] - 1)
                         if len(equity) > 1 else None),
        "best_day": float(daily.max()) if len(daily) else None,
        "worst_day": float(daily.min()) if len(daily) else None,
        "win_days": float((daily > 0).mean()) if len(daily) else None,
        "diagnostics": result.diagnostics,
    }
    (args.out / "paper_sim_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("\nSIMULATION COMPLETE. No orders were sent anywhere. Deployment "
          "remains gated on the operator's explicit approval phrase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

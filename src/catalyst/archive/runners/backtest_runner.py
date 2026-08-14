"""Backtest runner CLI.

Gate 1 mode (default): trivial test strategy on synthetic weekly catalysts —
plumbing validation on real ThetaData. Later gates plug real engines/signals
into the same assembly.

Usage:
    uv run python -m catalyst.runners.backtest_runner \
        --start 2024-01-02 --end 2025-06-30 --symbol SPY
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from catalyst.core.config import load_config
from catalyst.core.models import BacktestResult
from catalyst.backtest.backtester import Backtester
from catalyst.backtest.synthetic import weekly_synthetic_catalysts
from catalyst.backtest.walkforward import chronological_split
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.engines.trivial import TrivialTestStrategy
from catalyst.risk.gate import FixedFractionalGate
from catalyst.signals.neutral import NeutralSignal


def format_result(r: BacktestResult) -> str:
    lines = [
        f"===== {r.label} | {r.start} .. {r.end} =====",
        f"equity: {r.starting_capital:,.0f} -> {r.ending_equity:,.0f}  "
        f"(total return {r.total_return:+.2%})",
        f"trades: {r.n_trades} | win rate {r.win_rate:.1%} | "
        f"avg win {r.avg_win:,.0f} | avg loss {r.avg_loss:,.0f} | W/L {r.win_loss_ratio:.2f}",
        f"max drawdown {r.max_drawdown:.2%} | sharpe {r.sharpe:.2f} | "
        f"sortino {r.sortino:.2f} | calmar {r.calmar:.2f}",
        f"monte carlo: P(ruin) {r.probability_of_ruin:.2%} | survival {r.mc_survival_rate:.2%}",
        "monthly returns: "
        f"mean {r.monthly.mean:+.2%} | median {r.monthly.median:+.2%} | "
        f"p05 {r.monthly.p05:+.2%} | p95 {r.monthly.p95:+.2%} | "
        f"min {r.monthly.min:+.2%} | max {r.monthly.max:+.2%}",
        f"per-engine pnl: { {k: round(v) for k, v in r.per_engine_pnl.items()} }",
        f"per-engine trades: {r.per_engine_trades}",
    ]
    return "\n".join(lines)


def build_gate1_backtester(cfg, start: date, end: date, symbol: str) -> Backtester:
    cache = ParquetCache(cfg.data.cache_dir)
    data = ThetaDataHistorical(
        cfg.data,
        cache=cache,
        history_provider=AlpacaDailyBars(cfg.data.alpaca, cache),
    )
    ea = cfg.engines.engine_a
    strategy = TrivialTestStrategy(
        target_delta=ea.target_delta,
        per_trade_risk=ea.per_trade_risk,
        tp1_gain=ea.tp1_gain,
        tp1_fraction=ea.tp1_fraction,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        max_hold_trading_days=10,
        close_before_expiry_days=cfg.risk.close_before_expiry_days,
    )
    catalysts = weekly_synthetic_catalysts(symbol, start, end)
    return Backtester(
        cfg=cfg,
        data=data,
        strategies=[strategy],
        signal=NeutralSignal(),
        catalysts=catalysts,
        gate=FixedFractionalGate(),
        label=f"gate1-trivial-{symbol}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument(
        "--mode",
        choices=["single", "split"],
        default="single",
        help="single = one run; split = chronological train/test per config",
    )
    parser.add_argument("--out", type=Path, default=None, help="write full result JSON here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config("backtest")
    results: list[BacktestResult] = []
    if args.mode == "split":
        boundary, _ = chronological_split(args.start, args.end, cfg.backtest.train_test_split)
        segments = [("train", args.start, boundary), ("test", boundary, args.end)]
    else:
        segments = [("full", args.start, args.end)]

    for seg_label, seg_start, seg_end in segments:
        bt = build_gate1_backtester(cfg, seg_start, seg_end, args.symbol)
        result = bt.run(seg_start, seg_end)
        result = result.model_copy(update={"label": f"{result.label}-{seg_label}"})
        results.append(result)
        print(format_result(result))
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([r.model_dump(mode="json") for r in results], indent=2)
        )
        print(f"full results written to {args.out}")


if __name__ == "__main__":
    main()

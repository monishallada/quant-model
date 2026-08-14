"""Cointegration pairs strategy runner — backtest and report only.

Runs Version 1 (dollar-neutral shares) and/or Version 2 (1-2 DTE ATM
options via SimulatedBroker) over the config universe, each in real-cost and
zero-cost form, with the standard chronological train/test split and
optional walk-forward. Simulated broker + historical data ONLY: this runner
never touches a live broker.

Usage:
    uv run python -m catalyst.runners.pairs_runner --version 1 --out results/pairs
    uv run python -m catalyst.runners.pairs_runner --version 2 --out results/pairs
    uv run python -m catalyst.runners.pairs_runner --version 1 --walkforward
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import Config, load_config
from catalyst.core.models import BacktestResult
from catalyst.backtest.walkforward import chronological_split, walk_forward_windows
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.archive.pairs.options_backtester import PairsOptionsBacktester
from catalyst.archive.pairs.shares_backtester import PairsSharesBacktester
from catalyst.runners.backtest_runner import format_result

logger = logging.getLogger(__name__)


def load_price_frames(cfg: Config, start: date, end: date) -> tuple[dict, dict]:
    """Daily closes and opens for every symbol in the pair universe, with a
    lookback pad for the rolling window warmup."""
    cache = ParquetCache(cfg.data.cache_dir)
    bars = AlpacaDailyBars(cfg.data.alpaca, cache)
    pad_start = date(start.year - 1, 1, 1)
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    symbols = sorted({s for pair in cfg.pairs.universe for s in pair})
    for sym in symbols:
        df = bars.get_history(sym, pad_start, end)
        if df.empty:
            logger.warning("No price history for %s — its pairs will be inactive", sym)
            continue
        closes[sym] = df["close"]
        opens[sym] = df["open"]
        logger.info("%s: %d bars (%s -> %s)", sym, len(df),
                    df.index.min().date(), df.index.max().date())
    return closes, opens


def stability_dict(extras: dict) -> dict:
    ledger = extras.pop("stability")
    return {
        **extras,
        "activations": ledger.activations,
        "deactivations": ledger.deactivations,
        "active_session_fraction": (
            ledger.active_sessions / ledger.total_sessions if ledger.total_sessions else 0.0
        ),
        "deactivation_dates": {
            "/".join(k): [d.isoformat() for d in v] for k, v in ledger.deactivation_dates.items()
        },
    }


def run_segment(cfg: Config, version: int, zero_cost: bool, closes, opens, data,
                start: date, end: date, label: str) -> tuple[BacktestResult, dict]:
    if version == 1:
        bt = PairsSharesBacktester(cfg, closes, opens, zero_cost=zero_cost, label=label)
        return bt.run(start, end)
    bt = PairsOptionsBacktester(cfg, data, closes, zero_cost=zero_cost, label=label)
    return bt.run(start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=[1, 2], required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/pairs"))
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--costs", choices=["real", "zero", "both"], default="both")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config("backtest")
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)
    boundary, _ = chronological_split(start, end, cfg.backtest.train_test_split)

    closes, opens = load_price_frames(cfg, start, end)
    data = None
    if args.version == 2:
        cache = ParquetCache(cfg.data.cache_dir)
        data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata),
                                   cache=cache)

    args.out.mkdir(parents=True, exist_ok=True)
    cost_modes = {"real": [False], "zero": [True], "both": [False, True]}[args.costs]

    for zero_cost in cost_modes:
        cost_tag = "zerocost" if zero_cost else "realcost"
        segments = [("train", start, boundary), ("test", boundary, end), ("all", start, end)]
        for seg, s, e in segments:
            label = f"pairs-v{args.version}-{cost_tag}-{seg}"
            result, extras = run_segment(cfg, args.version, zero_cost, closes, opens, data, s, e, label)
            print(format_result(result))
            print(f"  stability/extras: {stability_dict(dict(extras))}")
            print()
            payload = result.model_dump(mode="json")
            payload["extras"] = stability_dict(dict(extras))
            (args.out / f"{label}.json").write_text(json.dumps(payload, indent=2))

    if args.walkforward:
        rows = []
        for i, w in enumerate(walk_forward_windows(start, end, cfg.backtest.walk_forward)):
            result, _ = run_segment(cfg, args.version, False, closes, opens, data,
                                    w.test_start, w.test_end, f"wf-{i}")
            rows.append({
                "window": i, "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(), "return": result.total_return,
                "sharpe": result.sharpe, "max_dd": result.max_drawdown,
                "trades": result.n_trades,
            })
        (args.out / f"pairs-v{args.version}-walkforward.json").write_text(json.dumps(rows, indent=2))
        print(f"walk-forward rows -> {args.out}/pairs-v{args.version}-walkforward.json")


if __name__ == "__main__":
    main()

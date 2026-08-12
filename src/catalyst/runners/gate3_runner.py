"""Gate 3 runner: FULL integrated backtest — all enabled engines + the
authoritative RiskManager + hedge sleeve + mechanical exits — over the full
history, with the parameter sweep and the aggressiveness dial.

Modes:
    full     one integrated run over the whole range, train/test reported
    sweep    parameter sweep (config backtest.sweep) on the TRAIN segment,
             top configs re-evaluated on the untouched TEST segment
    dial     aggressiveness dial: per-trade size x cash floor grid, reporting
             return distribution vs survival probability shifts
    walkforward  rolling out-of-sample windows with base config

Usage:
    uv run python -m catalyst.runners.gate3_runner --mode full --signal trend
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from catalyst.core.config import Config, load_config
from catalyst.core.models import BacktestResult
from catalyst.backtest.backtester import Backtester
from catalyst.backtest.sweep import iter_combinations
from catalyst.backtest.walkforward import chronological_split, walk_forward_windows
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.iv_history import IVRankProvider
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.risk.hedge import HedgeManager
from catalyst.risk.manager import RiskManager
from catalyst.screener.catalyst_screener import CatalystScreener
from catalyst.runners.backtest_runner import format_result
from catalyst.runners.gate2_runner import build_engine, build_signal, load_catalysts

logger = logging.getLogger(__name__)


class Gate3Assembly:
    """Shared data plumbing across many runs (cache/memo reuse)."""

    def __init__(self, base_cfg: Config) -> None:
        self.cache = ParquetCache(base_cfg.data.cache_dir)
        self.client = ThetaDataClient(base_cfg.data.thetadata)
        self.data = ThetaDataHistorical(
            base_cfg.data, client=self.client, cache=self.cache,
            history_provider=AlpacaDailyBars(base_cfg.data.alpaca, self.cache),
        )
        self.iv_rank = IVRankProvider(
            base_cfg.data, self.client, self.cache,
            lookback_days=base_cfg.engines.engine_a.iv_rank_lookback_days,
        )
        self._iv_prepared: set[str] = set()

    def prepare_iv(self, symbols: list[str], start: date, end: date) -> None:
        for sym in symbols:
            if sym not in self._iv_prepared:
                self.iv_rank.prepare(sym, start, end)
                self._iv_prepared.add(sym)


def run_one(
    assembly: Gate3Assembly,
    cfg: Config,
    signal_name: str,
    start: date,
    end: date,
    label: str,
    overrides: dict[str, Any] | None = None,
) -> BacktestResult:
    catalysts = load_catalysts(cfg, assembly.cache, start, end)
    engines = [
        build_engine(name, cfg, assembly.iv_rank)
        for name, engine_cfg in (
            ("engine_a", cfg.engines.engine_a),
            ("engine_b", cfg.engines.engine_b),
            ("engine_c", cfg.engines.engine_c),
        )
        if engine_cfg.enabled
    ]
    bt = Backtester(
        cfg=cfg,
        data=assembly.data,
        strategies=engines,
        signal=build_signal(signal_name, cfg),
        catalysts=catalysts,
        gate=RiskManager(cfg.risk),
        hedger=HedgeManager(cfg.risk),
        screener=CatalystScreener(cfg.screener),
        label=label,
    )
    result = bt.run(start, end)
    if overrides:
        result = result.model_copy(update={"config_overrides": overrides})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "sweep", "dial", "walkforward"], default="full")
    parser.add_argument("--signal", default="trend")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/gate3"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    base_cfg = load_config("backtest")
    start = args.start or date.fromisoformat(base_cfg.backtest.start_date)
    end = args.end or date.fromisoformat(base_cfg.backtest.end_date)
    boundary, _ = chronological_split(start, end, base_cfg.backtest.train_test_split)
    assembly = Gate3Assembly(base_cfg)

    catalysts = load_catalysts(base_cfg, assembly.cache, start, end)
    assembly.prepare_iv(sorted({c.symbol for c in catalysts}), start, end)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        for seg, s, e in (("train", start, boundary), ("test", boundary, end),
                          ("all", start, end)):
            result = run_one(assembly, base_cfg, args.signal, s, e, f"gate3-{seg}")
            print(format_result(result))
            print()
            (args.out / f"gate3-{seg}.json").write_text(
                json.dumps(result.model_dump(mode="json"), indent=2))

    elif args.mode == "sweep":
        rows = []
        for i, overrides in enumerate(iter_combinations(base_cfg.backtest.sweep)):
            cfg = load_config("backtest", overrides=overrides)
            label = f"sweep-{i}"
            train_res = run_one(assembly, cfg, args.signal, start, boundary,
                                f"{label}-train", overrides)
            test_res = run_one(assembly, cfg, args.signal, boundary, end,
                               f"{label}-test", overrides)
            row = {
                **{f"param:{k}": str(v) for k, v in overrides.items()},
                "train_return": train_res.total_return, "train_sharpe": train_res.sharpe,
                "train_ruin": train_res.probability_of_ruin,
                "test_return": test_res.total_return, "test_sharpe": test_res.sharpe,
                "test_ruin": test_res.probability_of_ruin,
                "test_max_dd": test_res.max_drawdown, "test_trades": test_res.n_trades,
            }
            rows.append(row)
            logger.info("sweep %d done: %s", i, row)
        (args.out / "sweep.json").write_text(json.dumps(rows, indent=2))
        print(f"{len(rows)} sweep rows -> {args.out}/sweep.json")

    elif args.mode == "dial":
        rows = []
        sizes = [0.01, 0.02, 0.03, 0.05]
        floors = [0.30, 0.40, 0.50]
        for size in sizes:
            for floor in floors:
                overrides = {
                    "engines.engine_a.per_trade_risk": size,
                    "engines.engine_b.per_trade_risk": size,
                    "engines.engine_c.per_trade_risk": max(size * 2 / 3, 0.01),
                    "risk.cash_floor_fraction": floor,
                }
                cfg = load_config("backtest", overrides=overrides)
                res = run_one(assembly, cfg, args.signal, start, end,
                              f"dial-{size}-{floor}", overrides)
                rows.append({
                    "per_trade_risk": size, "cash_floor": floor,
                    "total_return": res.total_return, "monthly_mean": res.monthly.mean,
                    "monthly_p05": res.monthly.p05, "max_drawdown": res.max_drawdown,
                    "sharpe": res.sharpe, "prob_ruin": res.probability_of_ruin,
                    "mc_survival": res.mc_survival_rate, "n_trades": res.n_trades,
                })
                logger.info("dial size=%s floor=%s done", size, floor)
        (args.out / "dial.json").write_text(json.dumps(rows, indent=2))
        print(f"{len(rows)} dial rows -> {args.out}/dial.json")

    else:  # walkforward
        windows = walk_forward_windows(start, end, base_cfg.backtest.walk_forward)
        rows = []
        for i, w in enumerate(windows):
            res = run_one(assembly, base_cfg, args.signal, w.test_start, w.test_end,
                          f"wf-{i}")
            rows.append({
                "window": i, "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(),
                "return": res.total_return, "sharpe": res.sharpe,
                "max_dd": res.max_drawdown, "trades": res.n_trades,
            })
        (args.out / "walkforward.json").write_text(json.dumps(rows, indent=2))
        print(f"{len(rows)} walk-forward windows -> {args.out}/walkforward.json")


if __name__ == "__main__":
    main()

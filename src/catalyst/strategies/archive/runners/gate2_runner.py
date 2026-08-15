"""Gate 2 runner: Engines A and B independently, each under every baseline
signal, over the full ThetaData history, reported train/test.

Usage:
    uv run python -m catalyst.runners.gate2_runner \
        [--engines engine_a engine_b] [--signals trend mean_reversion neutral] \
        [--start 2018-01-01] [--end 2026-08-01] [--out results/gate2]

The (engine, signal) grid runs sequentially; the first run warms the shared
parquet cache, the rest are disk-bound.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from catalyst.core.config import Config, load_config
from catalyst.core.interfaces import DirectionalSignal, Strategy
from catalyst.core.models import BacktestResult, Catalyst
from catalyst.backtest.backtester import Backtester
from catalyst.backtest.walkforward import chronological_split
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.catalysts import StaticEconomicCalendar, YFinanceEarnings
from catalyst.data.iv_history import IVRankProvider
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.archive.engines.engine_a_convexity import EngineAConvexity
from catalyst.archive.engines.engine_b_crush_spread import EngineBCrushSpread
from catalyst.archive.engines.engine_c_pead import EngineCPead
from catalyst.risk.gate import FixedFractionalGate
from catalyst.archive.runners.backtest_runner import format_result
from catalyst.signals.mean_reversion import MeanReversionSignal
from catalyst.signals.neutral import NeutralSignal
from catalyst.signals.trend import TrendSignal

logger = logging.getLogger(__name__)


def load_catalysts(cfg: Config, cache: ParquetCache, start: date, end: date) -> list[Catalyst]:
    earnings_symbols = [s for s in cfg.watchlist if s not in cfg.catalysts.economic_symbols]
    providers = [
        StaticEconomicCalendar(cfg.catalysts.calendars_dir, cfg.catalysts.economic_symbols),
        YFinanceEarnings(earnings_symbols, cache),
    ]
    out: list[Catalyst] = []
    for p in providers:
        out.extend(p.get_catalyst_calendar(start, end))
    logger.info("Loaded %d catalysts (%s..%s)", len(out), start, end)
    return sorted(out, key=lambda c: c.when)


def build_signal(name: str, cfg: Config) -> DirectionalSignal:
    if name == "trend":
        return TrendSignal(cfg.signals.trend)
    if name == "mean_reversion":
        return MeanReversionSignal(cfg.signals.mean_reversion)
    if name == "neutral":
        return NeutralSignal()
    raise ValueError(f"Unknown signal {name}")


def build_engine(name: str, cfg: Config, iv_rank: IVRankProvider) -> Strategy:
    if name == "engine_a":
        return EngineAConvexity(cfg.engines.engine_a, cfg.risk, iv_rank)
    if name == "engine_b":
        return EngineBCrushSpread(cfg.engines.engine_b, cfg.risk)
    if name == "engine_c":
        return EngineCPead(cfg.engines.engine_c, cfg.risk, iv_rank)
    raise ValueError(f"Unknown engine {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["engine_a", "engine_b"])
    parser.add_argument("--signals", nargs="+", default=["trend", "mean_reversion", "neutral"])
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/gate2"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config("backtest")
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)
    boundary, _ = chronological_split(start, end, cfg.backtest.train_test_split)

    cache = ParquetCache(cfg.data.cache_dir)
    client = ThetaDataClient(cfg.data.thetadata)
    data = ThetaDataHistorical(
        cfg.data, client=client, cache=cache,
        history_provider=AlpacaDailyBars(cfg.data.alpaca, cache),
    )
    catalysts = load_catalysts(cfg, cache, start, end)

    iv_rank = IVRankProvider(
        cfg.data, client, cache, lookback_days=cfg.engines.engine_a.iv_rank_lookback_days
    )
    needs_iv = {"engine_a", "engine_c"} & set(args.engines)
    if needs_iv:
        symbols = sorted({c.symbol for c in catalysts})
        for i, sym in enumerate(symbols):
            logger.info("Preparing IV history %s (%d/%d)", sym, i + 1, len(symbols))
            iv_rank.prepare(sym, start, end)

    args.out.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for engine_name in args.engines:
        for signal_name in args.signals:
            for seg_label, s, e in (("train", start, boundary), ("test", boundary, end)):
                label = f"{engine_name}+{signal_name}-{seg_label}"
                engine = build_engine(engine_name, cfg, iv_rank)
                bt = Backtester(
                    cfg=cfg,
                    data=data,
                    strategies=[engine],
                    signal=build_signal(signal_name, cfg),
                    catalysts=[c for c in catalysts if s <= c.when.date() <= e],
                    gate=FixedFractionalGate(),
                    label=label,
                )
                result: BacktestResult = bt.run(s, e)
                print(format_result(result))
                print()
                (args.out / f"{label}.json").write_text(
                    json.dumps(result.model_dump(mode="json"), indent=2)
                )
                summary.append({
                    "engine": engine_name, "signal": signal_name, "segment": seg_label,
                    "n_trades": result.n_trades, "total_return": result.total_return,
                    "win_rate": result.win_rate, "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "monthly_mean": result.monthly.mean,
                    "prob_ruin": result.probability_of_ruin,
                })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Gate 2 artifacts in {args.out}/")


if __name__ == "__main__":
    main()

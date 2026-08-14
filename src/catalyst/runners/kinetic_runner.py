"""Kinetic Ignition Engine runner — backtest and report only.

Runs the intraday PEAD momentum strategy over the config universe with the
standard chronological train/test split, walk-forward, and a zero-cost
diagnostic alongside every real-cost run. Simulated broker + historical data
ONLY: this runner never touches a live broker.

Usage:
    uv run python -m catalyst.runners.kinetic_runner --out results/kinetic
    uv run python -m catalyst.runners.kinetic_runner --walkforward
    uv run python -m catalyst.runners.kinetic_runner --uncapped   # spec-literal 10% NAV
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from catalyst.core.config import Config, load_config
from catalyst.core.models import BacktestResult
from catalyst.backtest import metrics as m
from catalyst.backtest.walkforward import chronological_split, walk_forward_windows
from catalyst.data.cache import ParquetCache
from catalyst.data.catalysts import YFinanceEarnings
from catalyst.data.intraday import AlpacaMinuteBars, ThetaMinuteQuotes
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.kinetic.backtester import KineticBacktester

logger = logging.getLogger(__name__)


def build(cfg: Config) -> tuple[ThetaDataHistorical, AlpacaMinuteBars, ThetaMinuteQuotes, ParquetCache]:
    cache = ParquetCache(cfg.data.cache_dir)
    client = ThetaDataClient(cfg.data.thetadata)
    data = ThetaDataHistorical(cfg.data, client=client, cache=cache)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    quotes = ThetaMinuteQuotes(cfg.data.thetadata, cache, client=client)
    return data, bars, quotes, cache


def format_kinetic(result: BacktestResult, extras: dict) -> str:
    trades = result.trades
    pf = m.profit_factor(trades)
    ev = m.expected_value(trades)
    conc = m.concentration(trades, 3)
    lines = [
        f"===== {result.label} | {result.start} .. {result.end} =====",
        f"equity: {result.starting_capital:,.0f} -> {result.ending_equity:,.0f} "
        f"({result.total_return:+.2%})",
        f"trades: {result.n_trades} | win rate {result.win_rate:.1%} | "
        f"EV/trade ${ev:,.0f} | profit factor {pf:.2f}",
        f"avg win ${result.avg_win:,.0f} | avg loss ${result.avg_loss:,.0f} | "
        f"max DD {result.max_drawdown:.2%} | sharpe {result.sharpe:.2f}",
        f"top-3 concentration: ${conc['top_n_pnl']:,.0f} of ${conc['total_pnl']:,.0f}"
        + (f" = {conc['share']:.0%} of total P&L"
           + ("  *** LOTTERY-TICKET FLAG ***" if conc['share'] > 0.5 else "")
           if conc['share'] is not None else "  (share n/a: total P&L not positive)"),
        f"DTE buckets: pnl={ {k: round(v) for k, v in result.per_engine_pnl.items()} } "
        f"n={result.per_engine_trades}",
        f"funnel: {extras}",
    ]
    if result.n_trades < 100:
        lines.append(f"  *** WARNING: N={result.n_trades} < 100 — sample too small to trust ***")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/kinetic"))
    parser.add_argument("--costs", choices=["real", "zero", "both"], default="both")
    parser.add_argument("--uncapped", action="store_true",
                        help="spec-literal 10%% NAV sizing, bypassing RiskManager caps (diagnostic)")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config("backtest")
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)
    boundary, _ = chronological_split(start, end, cfg.backtest.train_test_split)

    data, bars, quotes, cache = build(cfg)
    logger.info("Loading earnings calendar for %d symbols...", len(cfg.kinetic.universe))
    catalysts = YFinanceEarnings(cfg.kinetic.universe, cache).get_catalyst_calendar(start, end)
    logger.info("Loaded %d earnings events", len(catalysts))

    # One cached daily-close series per symbol supplies every prior close.
    symbols = sorted({c.symbol for c in catalysts})
    daily_closes = {}
    for i, sym in enumerate(symbols, 1):
        daily_closes[sym] = bars.daily_closes(sym, start, end)
        if i % 25 == 0:
            logger.info("daily closes %d/%d", i, len(symbols))

    args.out.mkdir(parents=True, exist_ok=True)
    cost_modes = {"real": [False], "zero": [True], "both": [False, True]}[args.costs]
    summary: list[dict] = []

    for zero_cost in cost_modes:
        tag = "zerocost" if zero_cost else "realcost"
        if args.uncapped:
            tag += "-uncapped"
        for seg, s, e in (("train", start, boundary), ("test", boundary, end), ("all", start, end)):
            label = f"kinetic-{tag}-{seg}"
            bt = KineticBacktester(cfg, data, bars, quotes, catalysts,
                                   zero_cost=zero_cost, uncapped_sizing=args.uncapped,
                                   daily_closes=daily_closes, label=label)
            result, extras = bt.run(s, e)
            print(format_kinetic(result, extras))
            print()
            payload = result.model_dump(mode="json")
            payload["extras"] = extras
            payload["profit_factor"] = m.profit_factor(result.trades)
            payload["ev_per_trade"] = m.expected_value(result.trades)
            payload["concentration_top3"] = m.concentration(result.trades, 3)
            (args.out / f"{label}.json").write_text(json.dumps(payload, indent=2))
            summary.append({"label": label, "segment": seg, "cost": tag,
                            "n_trades": result.n_trades, "return": result.total_return,
                            "win_rate": result.win_rate,
                            "profit_factor": m.profit_factor(result.trades),
                            "ev_per_trade": m.expected_value(result.trades),
                            "max_dd": result.max_drawdown,
                            "top3_share": m.concentration(result.trades, 3)["share"]})

    if args.walkforward:
        rows = []
        for i, w in enumerate(walk_forward_windows(start, end, cfg.backtest.walk_forward)):
            bt = KineticBacktester(cfg, data, bars, quotes, catalysts,
                                   zero_cost=False, daily_closes=daily_closes,
                                   label=f"kinetic-wf-{i}")
            result, _ = bt.run(w.test_start, w.test_end)
            rows.append({"window": i, "test_start": w.test_start.isoformat(),
                         "test_end": w.test_end.isoformat(), "return": result.total_return,
                         "trades": result.n_trades, "sharpe": result.sharpe,
                         "max_dd": result.max_drawdown})
            logger.info("walk-forward %d: %+.2f%% (%d trades)", i,
                        result.total_return * 100, result.n_trades)
        (args.out / "kinetic-walkforward.json").write_text(json.dumps(rows, indent=2))

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()

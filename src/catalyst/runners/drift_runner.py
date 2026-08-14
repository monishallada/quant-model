"""Drift Harvest runner — backtest and report only.

Runs the three arms (credit / condor / debit) over an identical event set so
the comparison isolates each effect, with the standard chronological
train/test split and a zero-cost diagnostic alongside every real-cost run.
Simulated broker + historical data ONLY; never touches a live broker.

Usage:
    uv run python -m catalyst.runners.drift_runner --out results/drift
    uv run python -m catalyst.runners.drift_runner --arms credit --costs real
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.core.config import Config, load_config
from catalyst.core.models import BacktestResult
from catalyst.backtest import metrics as m
from catalyst.backtest.backtester import Backtester
from catalyst.backtest.walkforward import chronological_split
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.catalysts import YFinanceEarnings
from catalyst.data.intraday import AlpacaMinuteBars
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical
from catalyst.drift.engine import DriftHarvestEngine
from catalyst.drift.screener import DriftScreener
from catalyst.risk.manager import RiskManager
from catalyst.signals.neutral import NeutralSignal

logger = logging.getLogger(__name__)


class _InMemoryHistory:
    """Serves the daily frames the runner already pulled.

    Without this the backtester would re-request every symbol's history from
    Alpaca once per segment (130 symbols x 9 segments), which trips the rate
    limiter and costs nothing but time — the data is already in memory.
    """

    def __init__(self, daily: dict[str, pd.DataFrame]) -> None:
        self._daily = daily

    def get_history(self, symbol: str, start: date, end: date,
                    interval: str = "1d") -> pd.DataFrame:
        df = self._daily.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        idx = pd.to_datetime(pd.Series(list(df.index)))
        out = pd.DataFrame(
            {"open": df["open"].to_numpy(dtype=float),
             "high": df["close"].to_numpy(dtype=float),
             "low": df["close"].to_numpy(dtype=float),
             "close": df["close"].to_numpy(dtype=float),
             "volume": 0},
            index=idx,
        )
        return out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]


# Entry lands up to ~5 sessions after the reaction day, so the backtester must
# keep an earnings catalyst "in play" well past its report date.
_CATALYST_LOOKBACK_DAYS = 20


def format_drift(result: BacktestResult, engine: DriftHarvestEngine) -> str:
    trades = result.trades
    pf = m.profit_factor(trades)
    ev = m.expected_value(trades)
    conc = m.concentration(trades, 3)
    share = conc["share"]
    lines = [
        f"===== {result.label} | {result.start} .. {result.end} =====",
        f"equity: {result.starting_capital:,.0f} -> {result.ending_equity:,.0f} "
        f"({result.total_return:+.2%})",
        f"trades: {result.n_trades} | win rate {result.win_rate:.1%} | "
        f"EV/trade ${ev:,.0f} | profit factor {pf:.2f}",
        f"avg win ${result.avg_win:,.0f} | avg loss ${result.avg_loss:,.0f} | "
        f"max DD {result.max_drawdown:.2%} | sharpe {result.sharpe:.2f}",
        f"top-3: ${conc['top_n_pnl']:,.0f} of ${conc['total_pnl']:,.0f}"
        + (f" = {share:.0%}" + ("  *** LOTTERY-TICKET FLAG ***" if share > 0.5 else "")
           if share is not None else "  (share n/a: total not positive)"),
        f"gates: passed={engine.gate_passed} no_expiry={engine.skipped_no_expiry} "
        f"iv_not_rich={engine.skipped_iv_not_rich} no_structure={engine.skipped_no_structure} "
        f"credit_bounds={engine.skipped_credit_bounds}",
    ]
    if result.n_trades < 100:
        lines.append(f"  *** WARNING: N={result.n_trades} < 100 ***")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["credit", "condor", "debit"])
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--costs", choices=["real", "zero", "both"], default="both")
    parser.add_argument("--out", type=Path, default=Path("results/drift"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    base_cfg = load_config("backtest")
    start = args.start or date.fromisoformat(base_cfg.backtest.start_date)
    end = args.end or date.fromisoformat(base_cfg.backtest.end_date)
    boundary, _ = chronological_split(start, end, base_cfg.backtest.train_test_split)

    cache = ParquetCache(base_cfg.data.cache_dir)
    client = ThetaDataClient(base_cfg.data.thetadata)
    bars = AlpacaMinuteBars(base_cfg.data.alpaca, cache, feed="sip")

    logger.info("Loading earnings for %d symbols...", len(base_cfg.drift.universe))
    catalysts = YFinanceEarnings(base_cfg.drift.universe, cache).get_catalyst_calendar(start, end)
    symbols = sorted({c.symbol for c in catalysts})
    logger.info("%d earnings events across %d symbols", len(catalysts), len(symbols))

    # One cached daily pull per symbol, padded a year for correlation warmup.
    pad_start = date(start.year - 1, 1, 1)
    daily: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        daily[sym] = bars.daily_bars(sym, pad_start, end)
        if i % 25 == 0:
            logger.info("daily bars %d/%d", i, len(symbols))
    data = ThetaDataHistorical(base_cfg.data, client=client, cache=cache,
                               history_provider=_InMemoryHistory(daily))

    args.out.mkdir(parents=True, exist_ok=True)
    cost_modes = {"real": [False], "zero": [True], "both": [False, True]}[args.costs]
    summary: list[dict] = []

    for arm in args.arms:
        for zero_cost in cost_modes:
            cfg = base_cfg
            if zero_cost:
                cfg = load_config("backtest", overrides={
                    "execution.fill_model.spread_fill_fraction": 0.0,
                    "execution.fill_model.slippage_pct_of_premium": 0.0,
                })
            tag = "zerocost" if zero_cost else "realcost"
            for seg, s, e in (("train", start, boundary), ("test", boundary, end),
                              ("all", start, end)):
                label = f"drift-{arm}-{tag}-{seg}"
                screener = DriftScreener(cfg.drift, daily)
                engine = DriftHarvestEngine(cfg.drift, cfg.risk, screener, arm=arm)
                bt = Backtester(
                    cfg=cfg, data=data, strategies=[engine], signal=NeutralSignal(),
                    catalysts=[c for c in catalysts if s <= c.when.date() <= e],
                    gate=RiskManager(cfg.risk),
                    catalyst_lookback_days=_CATALYST_LOOKBACK_DAYS,
                    label=label,
                )
                result = bt.run(s, e)
                print(format_drift(result, engine))
                print()
                payload = result.model_dump(mode="json")
                payload["profit_factor"] = m.profit_factor(result.trades)
                payload["ev_per_trade"] = m.expected_value(result.trades)
                payload["concentration_top3"] = m.concentration(result.trades, 3)
                payload["gates"] = {
                    "passed": engine.gate_passed, "no_expiry": engine.skipped_no_expiry,
                    "iv_not_rich": engine.skipped_iv_not_rich,
                    "no_structure": engine.skipped_no_structure,
                    "credit_bounds": engine.skipped_credit_bounds,
                }
                (args.out / f"{label}.json").write_text(json.dumps(payload, indent=2))
                summary.append({
                    "arm": arm, "cost": tag, "segment": seg, "n_trades": result.n_trades,
                    "return": result.total_return, "win_rate": result.win_rate,
                    "ev_per_trade": m.expected_value(result.trades),
                    "profit_factor": m.profit_factor(result.trades),
                    "max_dd": result.max_drawdown, "sharpe": result.sharpe,
                    "top3_share": m.concentration(result.trades, 3)["share"],
                })
                (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()

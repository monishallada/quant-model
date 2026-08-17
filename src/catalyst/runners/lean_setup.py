"""lean-setup — prepare the LEAN engine using this repo's ThetaData key.

    uv run python -m catalyst.runners.lean_setup            # scaffold project
    uv run python -m catalyst.runners.lean_setup --convert  # write ThetaData chains

LEAN reads its own on-disk data format, so ThetaData chains are converted into
it here. Critically this uses the SAME ThetaData credentials and the SAME
snapshots the native engine consumes — one subscription, one source of truth
for prices. If the two engines then disagree, the difference is engine
behaviour, not two different price feeds, which is the only way the comparison
means anything.

Docker must be running for LEAN itself (`open -a Docker` on macOS).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time
from pathlib import Path

from catalyst.backtest.engines.lean import DATA_ROOT, LEAN_ROOT, ThetaToLeanWriter
from catalyst.core.config import load_config

SNAP = time(15, 45)

ALGORITHM = '''from AlgorithmImports import *


class CatalystMirror(QCAlgorithm):
    """Runs the active strategy's RULE inside LEAN, independently.

    LEAN selects its own contracts from its own chain, prices its own fills,
    applies its own fee and margin models, and computes its own equity. Nothing
    here reads a native-engine result — only the rule is shared, which is
    unavoidable if the two engines are to be testing the same strategy.

    What LEAN contributes that the native engine cannot: corporate actions
    (splits), early assignment and exercise, dividends and margin.
    """

    def Initialize(self):
        self.SetStartDate({y0}, {m0}, {d0})
        self.SetEndDate({y1}, {m1}, {d1})
        self.SetCash({cash})
        self.moneyness = {moneyness}
        self.hold_days = {hold_days}
        self.entry_every = {entry_every}
        self._session = 0
        self._opened = {{}}
        self.symbols = []
        for ticker in {universe}:
            eq = self.AddEquity(ticker, Resolution.Minute)
            opt = self.AddOption(ticker, Resolution.Minute)
            opt.SetFilter(-15, 15, timedelta({dte_lo}), timedelta({dte_hi}))
            self.symbols.append(opt.Symbol)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(15, 45), self.Rebalance)

    def Rebalance(self):
        self._session += 1
        # Close anything past its hold, exactly as the native rule does.
        for symbol, opened in list(self._opened.items()):
            if (self._session - opened) >= self.hold_days:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
                self._opened.pop(symbol, None)
        if (self._session - 1) % self.entry_every != 0:
            return
        for canonical in self.symbols:
            chain = self.CurrentSlice.OptionChains.get(canonical)
            if chain is None:
                continue
            underlying = chain.Underlying.Price
            if underlying <= 0:
                continue
            target = underlying * self.moneyness
            calls = [c for c in chain if c.Right == OptionRight.Call and c.AskPrice > 0.05]
            if not calls:
                continue
            pick = sorted(calls, key=lambda c: (abs(c.Strike - target), c.Expiry))[0]
            if pick.Symbol in self._opened:
                continue
            # LEAN sizes against its OWN portfolio and margin model.
            qty = self.CalculateOrderQuantity(pick.Symbol, {risk_fraction})
            if qty and qty > 0:
                self.MarketOrder(pick.Symbol, max(int(qty), 1))
                self._opened[pick.Symbol] = self._session
'''


def scaffold(cfg, start: date, end: date, universe: list[str],
             mirrors: str = "long_options") -> Path:
    project = LEAN_ROOT / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "main.py").write_text(ALGORITHM.format(
        y0=start.year, m0=start.month, d0=start.day,
        y1=end.year, m1=end.month, d1=end.day,
        cash=int(cfg.account.starting_capital),
        universe=universe, dte_lo=23, dte_hi=47,
        moneyness=1.05, hold_days=15, entry_every=15, risk_fraction=0.05))
    # Stamp what this algorithm mirrors so the engine can refuse to report a
    # stale run against a different strategy.
    (project / "MIRRORS").write_text(mirrors + "\n")
    (project / "config.json").write_text(
        '{\n  "algorithm-language": "Python",\n'
        '  "parameters": {},\n'
        '  "description": "Cross-check mirror of the active catalyst strategy"\n}\n')
    return project


def convert(cfg, symbols: list[str], start: date, end: date, limit: int) -> int:
    """Write ThetaData chains into LEAN's option layout."""
    from catalyst.core.tradingcal import sessions_in_range
    from catalyst.data.cache import ParquetCache
    from catalyst.data.thetadata_client import ThetaDataClient
    from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical

    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata),
                               cache=ParquetCache(cfg.data.cache_dir))
    writer = ThetaToLeanWriter()
    written = 0
    sessions = sessions_in_range(start, end)[:limit]
    for sym in symbols:
        for s in sessions:
            try:
                chain = data.get_chain(sym, datetime.combine(s, SNAP),
                                       include_liquidity=False)
            except DataUnavailableError:
                continue
            written += writer.write_chain(sym, datetime.combine(s, SNAP), chain)
        print(f"  {sym}: {writer.coverage(sym)} session files")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert", action="store_true", help="write ThetaData into LEAN format")
    ap.add_argument("--symbols", nargs="+", default=["SPY"])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2024, 3, 28))
    ap.add_argument("--limit", type=int, default=20, help="max sessions to convert")
    ap.add_argument("--strategy", default="long_options",
                    help="which strategy this algorithm mirrors")
    args = ap.parse_args(argv)

    cfg = load_config("backtest")
    project = scaffold(cfg, args.start, args.end, args.symbols,
                       mirrors=args.strategy)
    print(f"LEAN project scaffolded at {project}")

    if args.convert:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        n = convert(cfg, args.symbols, args.start, args.end, args.limit)
        print(f"converted {n} contract quotes into {DATA_ROOT}")
    else:
        print("run again with --convert to write ThetaData chains into LEAN format")

    from catalyst.backtest.engines.lean_docker import LeanDockerEngine as LeanEngine
    ok, why = LeanEngine().available()
    print(f"\nLEAN available: {ok} — {why}")
    if not ok:
        print("start Docker with:  open -a Docker")
    return 0


if __name__ == "__main__":
    sys.exit(main())

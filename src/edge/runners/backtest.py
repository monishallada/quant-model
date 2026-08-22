"""Backtest entrypoint: ``python -m edge.runners.backtest``.

A thin argparse front over :mod:`edge.runners.engine` — all machinery lives
there. This module only parses arguments, assembles
:class:`~edge.runners.engine.EngineParams` from the CLI plus
``config/edge.yaml``, and prints the run summary. It never persists results:
:class:`~edge.runners.engine.RunResult` stays in memory for the caller.

Data access goes through :class:`~edge.data.loader.EdgeDataLoader` (the only
sanctioned gateway). The CLI constructs the sanctioned feeds-backed loader,
which today serves only the slow point-in-time kinds — until a bars-capable
backend is wired behind the gateway, a CLI run reports that limitation and
exits nonzero instead of pretending to backtest. Programmatic callers pass
their own bars-capable loader to the engine directly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, time

from edge.core.config import load_config
from edge.data.loader import EdgeDataLoader
from edge.runners.engine import BacktestEngine, EngineParams, Signal

#: Named strategies the CLI can run. Signal modules register here as they
#: land; an unknown name is an error listing what exists.
KNOWN_SIGNALS: dict[str, type[Signal]] = {}


def build_parser() -> argparse.ArgumentParser:
    """CLI surface for the backtest runner."""
    parser = argparse.ArgumentParser(
        prog="edge.runners.backtest",
        description="Run an edge backtest over historical events.",
    )
    parser.add_argument("--config", default=None, help="path to edge.yaml (default: config/edge.yaml)")
    parser.add_argument("--repo-root", default=None, help="repository root (default: this repo)")
    parser.add_argument("--strategy", default=None, help="registered strategy name to run")
    parser.add_argument("--symbols", default=None, help="comma-separated symbols, e.g. AAPL,MSFT")
    parser.add_argument("--start", default=None, help="first session, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="last session, YYYY-MM-DD")
    parser.add_argument("--latency-ms", type=int, default=None,
                        help="latency scenario (default: first entry of execution.latency_ms)")
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="starting equity in dollars")
    parser.add_argument("--risk-r-pct", type=float, default=1.0,
                        help="per-share risk unit as %% of entry reference price")
    parser.add_argument("--max-hold-minutes", type=int, default=None,
                        help="max-hold exit policy (omit to disable)")
    parser.add_argument("--stop-r", type=float, default=None,
                        help="stop exit at k*R on the underlying move (omit to disable)")
    parser.add_argument("--eod-flatten", default=None,
                        help="EOD flatten time HH:MM ET (omit to disable)")
    parser.add_argument("--seed", type=int, default=0, help="run seed (recorded in config hash)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the engine, print the summary. Persists nothing."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv:  # bare invocation: show the surface, an informational exit
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    missing = [name for name in ("symbols", "start", "end") if getattr(args, name) is None]
    if missing:
        print(f"missing required arguments: {', '.join('--' + m for m in missing)}",
              file=sys.stderr)
        return 2
    config = load_config(config_path=args.config, repo_root=args.repo_root)
    signals: list[Signal] = []
    if args.strategy is not None:
        builder = KNOWN_SIGNALS.get(args.strategy)
        if builder is None:
            print(f"unknown strategy {args.strategy!r}; registered: "
                  f"{sorted(KNOWN_SIGNALS) or '(none yet)'}", file=sys.stderr)
            return 2
        signals.append(builder())
    params = EngineParams(
        symbols=tuple(s.strip() for s in args.symbols.split(",") if s.strip()),
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        latency_ms=args.latency_ms if args.latency_ms is not None else config.execution.latency_ms[0],
        initial_equity=args.equity,
        seed=args.seed,
        risk_r_pct=args.risk_r_pct,
        max_hold_minutes=args.max_hold_minutes,
        stop_r_multiple=args.stop_r,
        eod_flatten_time=time.fromisoformat(args.eod_flatten) if args.eod_flatten else None,
    )
    loader = EdgeDataLoader.with_feeds(repo_root=args.repo_root, config_path=args.config)
    engine = BacktestEngine(loader, config, params, signals)
    try:
        result = engine.run()
    except ValueError as exc:
        print(f"backtest unavailable through the CLI loader: {exc}", file=sys.stderr)
        print("wire a bars-capable backend behind EdgeDataLoader, or drive "
              "BacktestEngine programmatically with your own loader.", file=sys.stderr)
        return 2
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

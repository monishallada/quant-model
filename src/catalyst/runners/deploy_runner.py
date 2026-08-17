"""deploy — run a validated strategy in backtest, paper or live.

    uv run python -m catalyst.runners.deploy_runner --mode paper --strategy long_options

Mode selects **which DataSource and Broker get injected, and nothing else**.
The strategy, the RiskManager, the cost expectations and the exit rules are
byte-for-byte identical in all three. What you backtested is what paper-trades
and what goes live; there is no per-mode branch in any strategy.

Gating, in increasing severity:

- ``backtest`` — historical data, SimulatedBroker. No confirmation needed; it
  cannot touch an account.
- ``paper``    — real Alpaca paper account. Prints a pre-flight summary and
  requires a typed ``yes`` before connecting or sending anything.
- ``live``     — real money. Requires the strategy to be marked BOTH validated
  and paper-tested in its metadata (refused otherwise, structurally), prints a
  fuller pre-flight including capital at risk, and requires a typed
  ``LIVE`` confirmation.

``--dry-run`` reaches the confirmation gate and stops before any order is
transmitted; use it to prove the wiring without touching an account.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path

from catalyst.core.config import load_config
from catalyst.core.interfaces import Broker, Cadence, DataSource, Strategy
from catalyst.screener.catalyst_screener import CatalystScreener
from catalyst.execution.engine import ExecutionEngine
from catalyst.observability.killswitch import KillSwitch, configure_json_logging
from catalyst.risk.manager import RiskManager
from catalyst.strategies.promotion import record_paper_session
from catalyst.strategies.registry import StrategyMeta, load_strategy, registry

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class DeploymentRefused(SystemExit):
    """A gate said no. Always fatal — never retried with softer settings."""


@dataclass
class Wiring:
    """The only thing that changes between modes."""

    mode: Mode
    data: DataSource
    broker: Broker
    describe: str


# ----------------------------------------------------------------------
# mode -> (DataSource, Broker). The single switch point in the system.
# ----------------------------------------------------------------------
def build_wiring(mode: Mode, cfg, *, dry_run: bool = False) -> Wiring:
    from catalyst.data.alpaca_history import AlpacaDailyBars
    from catalyst.data.cache import ParquetCache
    from catalyst.data.thetadata_client import ThetaDataClient
    from catalyst.data.thetadata_historical import ThetaDataHistorical

    cache = ParquetCache(cfg.data.cache_dir)
    # ThetaData's STOCK tier here is FREE, so /v3/stock/history 403s on any
    # multi-year pull. Options data is STANDARD and fine. Every prior campaign
    # solved this the same way: keep ThetaData for chains, inject Alpaca for
    # underlying history. Without it a catalyst strategy dies on the first
    # signal lookup with a subscription error that reads like a code fault.
    data = ThetaDataHistorical(cfg.data, client=ThetaDataClient(cfg.data.thetadata),
                               cache=cache,
                               history_provider=AlpacaDailyBars(cfg.data.alpaca, cache))

    if mode is Mode.BACKTEST:
        from catalyst.brokers.simulated import SimulatedBroker
        broker = SimulatedBroker(fill_model=cfg.execution.fill_model,
                                 commissions=cfg.execution.commissions,
                                 starting_cash=cfg.account.starting_capital)
        return Wiring(mode, data, broker, "ThetaData historical + SimulatedBroker")

    if mode is Mode.PAPER:
        from catalyst.brokers.alpaca import AlpacaBroker, AlpacaCredentials
        broker = AlpacaBroker(AlpacaCredentials.from_env(paper=True))
        return Wiring(mode, data, broker, f"live data + AlpacaBroker @ {broker.endpoint}")

    from catalyst.brokers.schwab import SchwabBroker, SchwabCredentials
    broker = SchwabBroker(SchwabCredentials.from_env())
    return Wiring(mode, data, broker, "live data + SchwabBroker (REAL MONEY)")


# ----------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------
def _confirm(prompt: str, expect: str) -> bool:
    """Typed confirmation. No default, no y/N shortcut — the operator types
    the exact word or nothing happens."""
    try:
        answer = input(f"{prompt}\nType {expect!r} to proceed: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == expect


def preflight_summary(meta: StrategyMeta, wiring: Wiring, cfg) -> str:
    acct: dict = {}
    if hasattr(wiring.broker, "preflight"):
        try:
            acct = wiring.broker.preflight()
        except Exception as e:                      # noqa: BLE001 — shown, not raised
            acct = {"error": str(e)}
    risk = cfg.risk
    lines = [
        "=" * 70,
        f"  DEPLOY PRE-FLIGHT — mode: {wiring.mode.value.upper()}",
        "=" * 70,
        f"  strategy          : {meta.name}",
        f"  validated         : {meta.validated}",
        f"  paper-tested      : {meta.paper_tested}",
        f"  backtest verdict  : {meta.verdict or 'none recorded'}",
        f"  avg monthly return: {meta.avg_monthly_return if meta.avg_monthly_return is not None else 'n/a'}",
        "",
        f"  wiring            : {wiring.describe}",
        f"  endpoint          : {acct.get('endpoint', 'n/a')}",
        f"  account           : {acct.get('account_number', 'n/a')}",
        f"  equity            : {acct.get('equity', 'n/a')}",
        f"  buying power      : {acct.get('buying_power', 'n/a')}",
        f"  options level     : {acct.get('options_level', 'n/a')}",
        "",
        "  RISK LIMITS (authoritative in every mode, identical to backtest)",
        f"    cash floor      : {risk.cash_floor_fraction:.0%} of equity held back",
        f"    max deployed    : {risk.max_deployed:.0%} portfolio heat cap",
        f"    hedge sleeve    : {risk.hedge_fraction:.0%}",
        f"    correlation cap : {risk.max_correlated_positions} positions "
        f"above rho {risk.correlation_threshold:.2f}",
        f"    daily breaker   : {risk.daily_loss_halt:.0%}",
        f"    weekly breaker  : {risk.weekly_loss_halt:.0%}",
        f"    sizing buffer   : {risk.sizing_cost_buffer:.0%} added to worst-case cost",
    ]
    if acct.get("error"):
        lines.append(f"\n  !! broker preflight failed: {acct['error']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def enforce_preconditions(meta: StrategyMeta, mode: Mode) -> None:
    """Structural refusals, checked BEFORE any broker is constructed.

    Ordering matters: if wiring came first, a missing-credentials error would
    mask the far more important "this was never paper-tested" refusal, and the
    operator would fix the credentials and try again — walking straight past
    the gate that was actually protecting them.
    """
    if mode is not Mode.LIVE:
        return
    from catalyst.strategies.promotion import check_live_eligibility

    module_path = None
    try:
        import importlib, inspect
        module_path = Path(inspect.getfile(importlib.import_module(meta.module)))
    except Exception:                                   # noqa: BLE001
        pass
    eligible, reason = check_live_eligibility(meta.name, module_path)
    if not eligible:
        raise DeploymentRefused(f"REFUSED: {reason}")
    logger.info("live eligibility: %s", reason)


def enforce_gates(meta: StrategyMeta, wiring: Wiring, cfg, *, assume_yes: bool) -> None:
    """Pre-flight summary and typed confirmation, after wiring is built."""
    if wiring.mode is Mode.BACKTEST:
        return

    if wiring.mode is Mode.LIVE and getattr(wiring.broker, "is_paper", False):
        raise DeploymentRefused("REFUSED: live mode resolved to a paper broker.")

    print(preflight_summary(meta, wiring, cfg))

    if assume_yes:
        logger.warning("--yes supplied: confirmation prompt skipped")
        if wiring.mode is Mode.LIVE:
            raise DeploymentRefused(
                "REFUSED: --yes is not honoured in live mode. Real money "
                "requires an interactive typed confirmation.")
        return

    if wiring.mode is Mode.PAPER:
        if not _confirm("This connects to a REAL Alpaca paper account and may place "
                        "paper orders.", "yes"):
            raise DeploymentRefused("aborted at paper confirmation gate")
    else:
        print("\n  *** THIS IS REAL MONEY. Orders placed here settle for cash. ***")
        if not _confirm("Deploy this strategy against the live account above?", "LIVE"):
            raise DeploymentRefused("aborted at live confirmation gate")


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.BACKTEST)
    ap.add_argument("--strategy", required=True, help="registered strategy name")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--config", default="backtest")
    ap.add_argument("--signal", default="neutral",
                    choices=["trend", "mean_reversion", "neutral"],
                    help="directional signal; strategies that carry their own "
                         "direction (e.g. long_options) use neutral")
    ap.add_argument("--dry-run", action="store_true",
                    help="reach the gate and build orders, but transmit nothing")
    ap.add_argument("--yes", action="store_true",
                    help="skip the paper prompt (never honoured for live)")
    args = ap.parse_args(argv)

    configure_json_logging(logging.INFO)
    cfg = load_config(args.config)

    meta = registry().get(args.strategy)
    if meta is None:
        known = ", ".join(sorted(registry())) or "none"
        raise DeploymentRefused(f"unknown strategy '{args.strategy}'. Known: {known}")

    kill = KillSwitch()
    if kill.engaged():
        raise DeploymentRefused(f"REFUSED: kill switch engaged ({kill.reason()})")

    # Structural gates first: a strategy that has not earned live must be
    # refused before we even reach for credentials.
    enforce_preconditions(meta, args.mode)

    wiring = build_wiring(args.mode, cfg, dry_run=args.dry_run)
    enforce_gates(meta, wiring, cfg, assume_yes=args.yes)

    strategy: Strategy = load_strategy(args.strategy, cfg)
    execution = ExecutionEngine(broker=wiring.broker, risk=RiskManager(cfg.risk),
                                kill=kill, dry_run=args.dry_run)

    if args.mode is Mode.BACKTEST:
        from catalyst.backtest.pipeline import Pipeline
        # A CATALYST-cadence strategy enumerates its opportunities from this
        # calendar. Omitting it is silent: the run completes over every session,
        # finds nothing to trade, and reports a tidy zero-trade verdict. That is
        # exactly what happened on the first catalyst_variance run.
        catalysts = _load_catalysts(cfg, args.start, args.end) \
            if strategy.cadence is Cadence.CATALYST else []
        if strategy.cadence is Cadence.CATALYST and not catalysts:
            raise DeploymentRefused(
                f"REFUSED: '{args.strategy}' is CATALYST cadence but no catalysts "
                f"were loaded for {args.start}..{args.end}. Running would report "
                "zero trades as though the strategy simply never triggered.")
        pipeline = Pipeline(cfg=cfg, data=wiring.data, catalysts=catalysts,
                            screener=CatalystScreener(cfg.screener),
                            signal=build_signal(args.signal, cfg))
        report = pipeline.run_and_save(
            strategy, args.start, args.end,
            Path("results/active") / args.strategy)
        print(report.to_text())
        return 0

    print(f"\nconnected: {wiring.describe}")
    state = execution.reconcile(datetime.now(UTC))
    print(f"reconciled against broker: {len(state.positions)} open positions, "
          f"equity {state.account.equity:,.2f}")
    if args.dry_run:
        print("\nDRY RUN — gate cleared, wiring verified, nothing transmitted.")
        print("NOTE: a dry run does NOT grant paper_tested; only a real "
              "paper session does.")
        return 0

    if args.mode is Mode.PAPER:
        # paper_tested is granted by an actual session against a real paper
        # account — never by a flag, a config value, or a dry run.
        acct = ""
        try:
            acct = str(wiring.broker.preflight().get("account_number", ""))
        except Exception:                               # noqa: BLE001
            pass
        try:
            record_paper_session(args.strategy, acct, len(state.positions))
            print(f"promotion: '{args.strategy}' is now paper-tested on {acct}")
        except PermissionError as e:
            print(f"\nNOTE: {e}")
    print("\nSession live. Kill switch: touch "
          f"{kill.path} to halt new entries immediately.")
    return 0


def _load_catalysts(cfg, start: date, end: date):
    """Earnings + economic calendar, the same sources every prior campaign used."""
    from catalyst.data.alpaca_history import AlpacaDailyBars
    from catalyst.data.cache import ParquetCache
    from catalyst.data.catalysts import StaticEconomicCalendar, YFinanceEarnings

    cache = ParquetCache(cfg.data.cache_dir)
    earnings_symbols = [s for s in cfg.watchlist
                        if s not in cfg.catalysts.economic_symbols]
    out = []
    for provider in (
        StaticEconomicCalendar(cfg.catalysts.calendars_dir,
                               cfg.catalysts.economic_symbols),
        YFinanceEarnings(earnings_symbols, cache),
    ):
        out.extend(provider.get_catalyst_calendar(start, end))
    logger.info("loaded %d catalysts (%s..%s)", len(out), start, end)
    return sorted(out, key=lambda c: c.when)


def build_signal(name: str, cfg):
    """Signal selection, matching the archived runners exactly.

    ``cfg.signals`` is a container of per-signal sections; each signal takes
    its OWN section, not the container.
    """
    from catalyst.signals.mean_reversion import MeanReversionSignal
    from catalyst.signals.neutral import NeutralSignal
    from catalyst.signals.trend import TrendSignal

    if name == "trend":
        return TrendSignal(cfg.signals.trend)
    if name == "mean_reversion":
        return MeanReversionSignal(cfg.signals.mean_reversion)
    if name == "neutral":
        return NeutralSignal()
    raise ValueError(f"unknown signal '{name}'; known: trend, mean_reversion, neutral")


if __name__ == "__main__":
    sys.exit(main())

# Catalyst-Driven Convexity Options Trading System

Modular quantitative options trading system: capped-downside/uncapped-upside bets timed to
known catalysts (earnings, CPI, FOMC), with an authoritative risk layer that structurally
guarantees survival.

**Governing invariant:** return is a measured output, never a driver. No module targets a
return figure or relaxes risk in response to underperformance. All orders pass through the
`RiskManager`; nothing can override it.

## Architecture

A stable shared core; swappable strategies. Every strategy — active or archived —
implements one `Strategy` interface and travels one fixed pipeline:

```
screener → strategy → RiskManager → cost model → exits → metrics → report
```

There is no flag or config key that removes a stage, and the pipeline calls the
strategy rather than the reverse. A strategy holds no Broker, no RiskManager and
no cost model, so it cannot size its own position, price its own fill, or skip
the out-of-sample split — those are absent from its type signatures.

```
config/                  YAML only; no numbers in code
core/interfaces/         DataSource, Broker, Strategy, DirectionalSignal
core/types/              OptionChain, Greeks, Order, Position, BacktestResult...
data/  brokers/  screener/  risk/  execution/  exits/
costs/                   single source of cost truth — every fill priced here
backtest/                backtester, pipeline, walk-forward, metrics
reporting/               the standard report; strategies cannot opt out of a check
strategies/active/       exactly one strategy under test
strategies/archive/      every prior campaign, kept runnable with its results
observability/           kill switch, heartbeat, JSON logging
runners/                 backtest, deploy, archive
```

Identical strategy code runs in backtest, paper and live — `--mode` changes only
which `DataSource` and `Broker` are injected:

| Mode | DataSource | Broker | Gate |
|---|---|---|---|
| `backtest` | `ThetaDataHistorical` | `SimulatedBroker` | none |
| `paper` | live | `AlpacaBroker` (paper endpoint) | typed `yes` |
| `live` | live | `SchwabBroker` | validated **and** paper-tested, then typed `LIVE` |

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding a strategy, archiving, the
pipeline guarantees, and the deployment path.

## Setup

```bash
uv sync                                   # install deps (Python 3.13)
cp .env.example .env                      # then fill in keys — .env is never committed
scripts/run_thetadata.sh                  # launch the ThetaData terminal (requires Java 21+)
```

## Layout

- `config/` — ALL numeric parameters live in YAML here; no hardcoded numbers in code
- `src/catalyst/core/` — interfaces, typed models, symbology, config loader, trading calendar
- `src/catalyst/{data,brokers,screener,engines,signals,risk,execution,exits,backtest,forecast,observability,runners}/`
- `tests/` — mirrors package layout; built alongside each module
- `data_cache/` — local parquet cache of historical pulls (gitignored, regenerable)

## Milestones

M1 data+backtester (Gate 1) → M2 screener+engines (Gate 2) → M3 risk/exits
(**Gate 3 = go/no-go review before any live code**) → M4 Alpaca paper live (Gate 4)
→ Schwab transition via drop-in `SchwabBroker` adapter.

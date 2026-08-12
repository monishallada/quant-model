# Catalyst-Driven Convexity Options Trading System

Modular quantitative options trading system: capped-downside/uncapped-upside bets timed to
known catalysts (earnings, CPI, FOMC), with an authoritative risk layer that structurally
guarantees survival.

**Governing invariant:** return is a measured output, never a driver. No module targets a
return figure or relaxes risk in response to underperformance. All orders pass through the
`RiskManager`; nothing can override it.

## Architecture

Identical strategy code runs in backtest, paper, and live — only the injected `DataSource`
and `Broker` implementations differ:

| Environment | DataSource            | Broker            |
|-------------|-----------------------|-------------------|
| backtest    | `ThetaDataHistorical` | `SimulatedBroker` |
| paper       | `LiveDataSource`      | `AlpacaBroker`    |
| live        | `LiveDataSource`      | `AlpacaBroker` → `SchwabBroker` (post-validation) |

Pipeline: screener → engines (A convexity / B crush-spread / C PEAD / D calendar)
→ directional signal → **RiskManager** (three-tier allocation, caps, breakers — authoritative)
→ execution (limit-only discipline) → broker adapter. Exits are mechanical rules attached
to every position at entry.

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

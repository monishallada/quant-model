# Archived: catalyst-engine strategy set (v1, 2026-08-13)

The original catalyst-driven engine set was set aside after its Gate 2/Gate 3
validation campaign, in favor of testing a cointegration pairs strategy.
**Nothing was deleted** — code, tests, and every result artifact are preserved
here and remain runnable for comparison.

## What lives where

| Piece | Location |
|---|---|
| Engine code (A convexity, B crush-spread, C PEAD, D calendar) | `src/catalyst/archive/engines/` |
| Gate 2 / Gate 3 runners | `src/catalyst/archive/runners/` |
| Engine unit tests (still run in CI) | `tests/archive/test_engines.py` |
| Gate 2 results (12 segments + summary) | `archive/results/gate2/` |
| Gate 2 zero-cost diagnostic | `archive/results/gate2_diagnostic/` |
| Gate 3 full/sweep results | `archive/results/gate3/` |
| Gate 3 aggressiveness dial | `archive/results/gate3_dial/` |
| Cost-decomposition script | `archive/scripts/gate2_cost_decomposition.py` |
| Go/no-go review document | https://claude.ai/code/artifact/2939bba4-333b-4aba-98f0-b00899aae4c2 |

All four engines are disabled in `config/base.yaml` (`engines.*.enabled: false`)
so no new backtest runs them.

## Headline findings (full detail in the review artifact)

- Engine C (post-earnings drift, measured-surprise direction) was the only
  validated edge: positive train AND test, 85 trades, 35% win, 2.61 W/L,
  ≈ +$125/trade ≈ **~1%/yr at safe (2%) sizing** — the baseline any new
  strategy must beat.
- Engines A/B lose gross under baseline signals (B: −51% frictionless);
  every sweep combo with B enabled was a disaster.
- Sizing dial: 2% per-trade compounds, 3.3% destroys the same expectancy.

## To re-run for comparison

```bash
uv run python -m catalyst.archive.runners.gate2_runner --out archive/results/gate2_rerun
uv run python -m catalyst.archive.runners.gate3_runner --mode full \
    --set engines.engine_c.enabled=true --out archive/results/gate3_rerun
```

(The gate runners read the same config system; re-enable engines per-run with
`--set engines.engine_X.enabled=true` rather than editing config.)

Shared infrastructure (data layer, SimulatedBroker, screener, RiskManager,
exits, backtester, metrics, config) was **not** archived — it is the validated
platform every subsequent strategy plugs into.

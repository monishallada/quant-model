# Archived strategy campaigns

## v1 — catalyst-engine set (archived 2026-08-13)

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


---

## v2 — cointegration pairs (archived 2026-08-13)

Tested and set aside the same day. **Nothing deleted.**

| Piece | Location |
|---|---|
| Cointegration engine, V1 shares + V2 options backtesters | `src/catalyst/archive/pairs/` |
| Runner | `src/catalyst/archive/runners/pairs_runner.py` |
| Tests (still in CI) | `tests/archive/pairs/` |
| Results (V1/V2 x real/zero cost x train/test/all + walk-forward) | `archive/results/pairs/` |
| Verdict document | https://claude.ai/code/artifact/4de7f3ec-b526-41dc-928c-dabde7ec904b |

**Findings:** V1 (shares) lost gross — −5.3% zero-cost over 8.6y — so the signal
itself had no edge; 81% of trades were force-flattened when pairs decoupled
(pairs held cointegration only ~14% of sessions). V2 (1–2 DTE ATM options)
−11.9% real-cost, with 90% of positions dying at the expiry stop and just 1 of
107 exiting via Z-reversion; its positive test cell was 3 trades = 706% of
segment P&L (lottery-ticket variance, not edge).

`pairs.enabled: false` in config; the strategy runs only via its archived runner:

```bash
uv run python -m catalyst.archive.runners.pairs_runner --version 1 --out archive/results/pairs_rerun
```

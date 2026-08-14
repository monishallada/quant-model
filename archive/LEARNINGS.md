# Cumulative learnings — read this before designing any new strategy

Every campaign below is archived and **not running**. What they measured is the
most valuable asset in this repository: roughly 5,000 trades of evidence about
what does and does not work, bought with real data and real compute. This file
is the standing brief for every future strategy test.

**Headline metric convention:** every strategy report leads with **average
monthly return %**, computed geometrically from the equity curve
(`catalyst.backtest.metrics.headline`). Benchmark alongside it, always.

---

## ⚠️ The instrument finding — read this first

**Every options campaign lost money. The only positive edge found is in shares.**
Measured avg monthly return, real costs, full period:

| Campaign | Instrument | Avg monthly |
|---|---|---|
| v1 Engine A — pre-catalyst long options | OPTIONS | −0.45% |
| v1 Engine B — debit spreads | OPTIONS | −2.25% |
| v1 Integrated A+B+C+hedge | OPTIONS | −0.88% |
| v2 Pairs V2 — 1–2 DTE options | OPTIONS | −0.12% |
| v3 Kinetic — 0–3 DTE options | OPTIONS | −1.45% |
| v4 Drift — credit spreads | OPTIONS | −1.06% |
| v6 Tournament — 3–5wk OTM calls (standalone) | OPTIONS | −3.89% |
| v2 Pairs V1 — shares | SHARES | −0.07% |
| **v5 Alpha Platform — shares + SPY** | **SHARES** | **+1.39%** |
| *SPY buy-and-hold* | *SHARES* | *+1.00%* |

**Why:** options are an amplifier, not a source of edge. They multiply both the
signal and the friction. Our best measured directional edge is IC ≈ 0.035, while
single-name option round trips cost 4–10% of premium ($18–28 per spread
crossing). The amplification does not overcome the cost, so expressing a small
edge through options converts a slightly positive strategy into a clearly
negative one — demonstrated seven separate ways.

**The honest role for options here** is the convex sleeve: not the return
engine, but a bounded-cost lottery ticket that supplies a right tail a
compounding book cannot. That is exactly how v7 uses them (20% allocation).

**The one untested options path worth trying:** index options (SPY/SPX) are
penny-wide and 5–10× cheaper to trade than single names, which is the variable
that killed every campaign above.

## Rules earned the hard way — do not re-test these

| Rule | Evidence |
|---|---|
| **Directional forecasting from technical signals does not work.** | Trend and mean-reversion were edgeless at Gate 2; gap-direction lost gross at N=926. Every strategy that needed a direction call failed. |
| **Never buy options held under ~3 days.** | Fatal four separate ways. Kinetic 0DTE lost $1,209/trade *frictionless* — pure theta, not costs. |
| **Intraday direction (15–120 min) is a coin flip.** | 8 signals, 525 symbol-days: zero cleared \|t\|>2.5 at any horizon. At 40 min the best signal (t=1.70) was indistinguishable from a random control (t=1.69). |
| **Option spreads are the dominant cost.** | $18–28 per crossing on single names; 53% of collected credit in v4. Any strategy crossing spreads often is dead on arrival. |
| **Small samples lie.** | Engine C looked like +$125/trade at N=85. The same structure at N=926 was gross-negative. |
| **Overlapping windows fabricate significance.** | A 63-day horizon sampled every 5 days produced t=5.98 that was really t≈1.68. Always sample non-overlapping. |
| **Always include a random control.** | It has validated every harness we trust and would have caught any lookahead. |

## What actually works

- **Cross-sectional momentum at multi-day horizons.** IC ≈ 0.035, t ≈ 2.4–2.6 at
  the 5-day horizon over 380+ independent observations. Modest but real, stable
  out-of-sample, and survives dropping the 25 best-performing names.
- **Full-breadth score-weighted construction**, not decile buckets. Same signal:
  Sharpe 0.15 → 0.41, drawdown −32% → −19.6%, out-of-sample alpha +1.0% → +3.9%.
- **Portable alpha.** A beta-0 overlay on a passive index position converts a
  modest uncorrelated alpha into a strong absolute return.
- **Variance risk premium is real** but tiny: +$7/trade gross, ~15× too small to
  survive single-name option costs.

## Open constraints (the real bottlenecks)

1. **Signal diversity.** Our library is one signal in five costumes — momentum
   variants correlate +0.74 (mom_12_1 × residual_mom = **+0.957**). The
   orthogonal signals we have (low-vol, reversal) carry no IC. This caps Sharpe
   and is the single highest-value thing to fix.
2. **No fundamental data.** FMP caps `limit=5` records, so value/quality/revisions
   signals cannot be built without a data upgrade.
3. **No order flow.** Alpaca gives OHLCV bars, ThetaData gives NBBO and OI. Order
   book depth and block prints — which Transformer order-flow models require —
   do not exist on this stack.
4. **Untested and buildable:** options-implied cross-sectional signals (IV skew,
   call−put IV spread, IV−RV) from 8 years of cached chains, and universe
   expansion from 130 to ~500 names (√breadth should lift IR ~2×).

## Campaign index

| Campaign | Result (avg monthly) | Verdict | Code | Report |
|---|---|---|---|---|
| v1 catalyst engines A/B/D | negative | No edge | `archive/engines/` | Gate 3 artifact |
| v1 Engine C (PEAD) | ≈ +0.08%/mo (N=85) | Refuted at scale by v4 | `archive/engines/` | Gate 3 artifact |
| v2 cointegration pairs | negative | Signal fake; 81% force-flattened | `archive/pairs/` | pairs verdict |
| v3 Kinetic Ignition | −7.6%/mo | Break-even signal, friction fatal | `archive/kinetic/` | kinetic verdict |
| v4 Drift Harvest | −9.5%/mo | Real gross edge, 15× too small | `archive/drift/` | drift verdict |
| v5 Alpha Platform | **+1.35%/mo** | **First real, robust edge** | `catalyst/alpha/` | alpha v5 report |
| v6 Tournament Engine | −3.89%/mo standalone | Model said P(10x)=31%, real chains 0% | `catalyst/tournament/` | tournament verdict |
| **v7 Combined Allocator** | **+2.49%/mo (concentration-flagged)** | **ACTIVE** | `catalyst/allocator/` | this campaign |

## Re-running an archived campaign

All archived code stays importable and runnable:

```bash
uv run python -m catalyst.archive.runners.gate3_runner --mode full
uv run python -m catalyst.archive.runners.pairs_runner --version 1
uv run python -m catalyst.archive.runners.kinetic_runner --costs both
uv run python -m catalyst.archive.runners.drift_runner --arms credit condor debit
uv run python -m catalyst.archive.runners.alpha_lab_runner        # IC diagnostics
uv run python -m catalyst.archive.runners.tournament_runner       # P(threshold) grid
```

Shared infrastructure — data layer, SimulatedBroker, RiskManager, exit manager,
backtester, metrics — was never archived. It is the validated platform every
campaign plugs into, and it now carries credit-structure support, flat
per-contract slippage, and the monthly headline metric.

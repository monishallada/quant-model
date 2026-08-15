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
| v8 Index VRP — SPY/QQQ put credit spreads | OPTIONS | +0.07% |
| **v5 Alpha Platform — shares + SPY** | **SHARES** | **+1.39%** |
| v9 Long calls/puts — signal-directed, 6 names, f=0.25 | OPTIONS | −1.85% (avg of 6) |
| **v9 Long CALLS ONLY — 6 names, f=0.25** | **OPTIONS** | **+0.98% (avg of 6); AMD +4.77%, NVDA +4.87%** |
| *SPY buy-and-hold* | *SHARES* | *+1.00%* |
| *Equal-weight buy-and-hold, the same six names* | *SHARES* | *+2.88%* |

**v9 is the first options campaign to produce a real right tail.** Straight long
calls at 25% of equity per 3-week cycle: TSLA P(5x)=15.6% and **P(10x)=7.3%**
over rolling 7-month windows, NVDA P(10x)=5.2%, AMD P(10x)=3.1% — against
P(10x)=0% for every earlier campaign and 0% for buy-and-hold. The portfolio
tail-maximizing config (ATM, 50% sizing, calls only) reaches **P(5x)=12%,
P(10x)=6%** — at +0.07%/month, −98.5% drawdown, and ruin on 4 of 6 names.

**But there is no alpha in it — only amplified beta.** The momentum signal is
*actively harmful*: calls-only beat signal-directed in every one of the 12
paired configurations (mean −0.67%/mo vs −1.96%/mo). "Calls only on six mega-cap
tech names, 2018–2026" is a leveraged long bet on the largest tech bull run in
history, on a universe selected with hindsight. Only AMD (+4.77%) and NVDA
(+4.87%) beat simply owning the same share (+3.73%, +3.66%), and they do it at
−93% and −91% drawdown versus the stock's own much shallower path.

**Per-position economics (785 positions, split-corrected):** mean 1.131x, median
**0.462x**, win rate 34.8%, 6.5% expire worthless, 18.2% ≥2x, 4.1% ≥5x, best
13.5x. The median position loses 54% while the mean gains 13% — textbook
convexity. Puts (0.60–1.04x) systematically underperform calls (1.03–1.54x),
which is precisely why the signal destroys value.

**Sizing is the whole game for a threshold objective.** The identical strategy
returns +2.93%/mo at 10% of equity per cycle, +3.74%/mo at 25%, −4.68%/mo at
50%, and −5.77%/mo at 100%. The first per-symbol run at f=0.50 wiped out all six
names and reported −4.4 to −4.8%/month; that was the sizing, not the strategy.

**Why:** options are an amplifier, not a source of edge. They multiply both the
signal and the friction. Our best measured directional edge is IC ≈ 0.035, while
single-name option round trips cost 4–10% of premium ($18–28 per spread
crossing). The amplification does not overcome the cost, so expressing a small
edge through options converts a slightly positive strategy into a clearly
negative one — demonstrated seven separate ways.

**The honest role for options here** is the convex sleeve: not the return
engine, but a bounded-cost lottery ticket that supplies a right tail a
compounding book cannot. That is exactly how v7 uses them (20% allocation).

**The index-options path was tested (v8) and the cost thesis was CORRECT but
insufficient.** Measured spreads: SPY 0.65% of mid, QQQ 0.95%, vs 4.43% average
across single names — **3.1× cheaper**. Moving the identical credit-spread trade
to that venue improved per-trade economics by ~$110 (v4 single-name: −$96/trade;
v8 index: +$14/trade), which took options from clearly negative to roughly
**break-even (+0.07%/month over 8.6 years)**. 78.8% win rate across 532 trades,
but profit factor 1.04 in-sample and **0.95 out-of-sample**. The venue change
was worth about 1.1%/month — almost exactly the cost saving — and the underlying
premium is still too thin to clear even the reduced friction.

**Conclusion across eight campaigns: no options structure has produced a
positive out-of-sample edge.** The cheapest venue in the options market gets you
to zero, not to profit.

## Framework refactor (2026-08-15)

The repository is now a stable shared core with swappable strategies. Every
strategy travels one fixed pipeline — screener -> strategy -> RiskManager ->
cost model -> exits -> metrics -> report — and no strategy can decline a check.

**Verified behaviour-preserving:** the Drift campaign re-run through the
refactored pipeline reproduces its stored artifact bit-for-bit (690 trades,
ending equity $33,484.328, max DD -66.5187%, win rate 52.029%). Structure
changed; measurement did not.

**Found during the refactor:**

| Finding | Status |
|---|---|
| `chronological_split` returns `(boundary, boundary)`; callers run train `[start, boundary]` and test `[boundary, end]`, so the boundary session sits in BOTH segments — a one-session leak out of ~2160. | Legacy function left untouched so archived results reproduce; the pipeline uses the new `chronological_split_exclusive`. **Open decision: whether to re-run archived campaigns on the corrected split.** |
| 4 of 9 campaigns bypassed the RiskManager; 4 of 9 priced their own fills by reading `.bid`/`.ask`, paying no commission or slippage. | Fixed structurally — strategies cannot import `catalyst.risk`, `catalyst.costs`, `catalyst.brokers` or `catalyst.execution`, enforced by AST test. |
| The N>100 flag was duplicated in 4 files, the concentration check in 6, the zero-cost diagnostic in 8 — each opt-in. | All moved into `reporting/`; every strategy is judged identically with no opt-out. |

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
| **Cross-check metrics against each other.** | v8 first reported +4.01%/mo alongside profit factor 0.98 — impossible, and it exposed a credit double-count in the cash ledger. Trade-level stats and the equity curve must agree or one of them is lying. |
| **Never mix split-ADJUSTED prices with RAW option strikes.** | v9: Alpaca daily bars are split-adjusted, ThetaData strikes are raw as-of-date. Targeting a strike off the adjusted price bought deep-ITM contracts on every pre-split date — AMZN ratio 20.03x, TSLA 14.93x, NVDA 10.04x, while unsplit MSFT was 1.00x. **Half the universe looked correct, which is why it survived a whole campaign.** Always select strikes from `chain.underlying_price`, the chain's own contemporaneous spot. |
| **Never fabricate an exit price for a missing contract.** | v9 fell back to intrinsic when a contract was absent from the exit chain, marking a vanished pre-split strike against a post-split spot and manufacturing a **924x put**. A contract absent at exit is *unpriceable* — skip the position and count it. |
| **Derive elapsed time from DATES, never from row count.** | v9 marked equity only at exits (~16 rows/year); `len(equity)/21` read that as 0.8 months and turned a +10% year into "+12.8% per month". Any curve not sampled daily breaks row-count time math. |
| **Ruin must be a reported outcome, not a silent early exit.** | v9 stopped trading at a 1%-of-capital floor; TSLA logged 13 of 139 scheduled cycles, which reads as missing data unless the wipeout is stated outright. |

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
| v7 Combined Allocator | +2.49%/mo (concentration-flagged) | Archived | `catalyst/allocator/` | v7 campaign |
| v8 Index VRP | +0.07%/mo (test −0.07%) | Archived — break-even | `catalyst/index_vrp/` | v8 campaign |
| **v9 Long calls/puts** | **+0.98%/mo calls-only; −1.85% signal-directed** | **ACTIVE — first real right tail (P(10x)=7.3% TSLA), but no alpha** | `catalyst/persymbol/long_options.py` | this campaign |

## Re-running an archived campaign

All archived code stays importable and runnable:

```bash
uv run python -m catalyst.archive.runners.gate3_runner --mode full
uv run python -m catalyst.archive.runners.pairs_runner --version 1
uv run python -m catalyst.archive.runners.kinetic_runner --costs both
uv run python -m catalyst.archive.runners.drift_runner --arms credit condor debit
uv run python -m catalyst.archive.runners.alpha_lab_runner        # IC diagnostics
uv run python -m catalyst.archive.runners.tournament_runner       # P(threshold) grid
uv run python -m catalyst.runners.long_options_runner --calls-only  # v9 per-symbol table
uv run python -m catalyst.runners.long_options_sweep              # v9 24-config frontier
uv run python -m catalyst.runners.long_options_trades             # v9 per-position distribution
```

Shared infrastructure — data layer, SimulatedBroker, RiskManager, exit manager,
backtester, metrics — was never archived. It is the validated platform every
campaign plugs into, and it now carries credit-structure support, flat
per-contract slippage, and the monthly headline metric.

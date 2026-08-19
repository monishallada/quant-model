# Intraday Platform (v12) — Architecture & Research Plan

*STEP 5 deliverable. Everything below is grounded in the verified capability
matrix (`scratchpad/intraday/capability_matrix.md`), the architecture map, two
cost-ranked literature reviews with citations, and eleven prior campaigns of
measured evidence. Nothing is assumed.*

---

## 1. Data reality (verified by live probes, not assumption)

| Capability | Status | Depth | Verified by |
|---|---|---|---|
| Options minute NBBO (equity options) | ✅ ThetaData `/v3/option/history/quote` | 2016-01 → present | 391 rows/0.63s per contract-day; bulk 183k rows/5.8s |
| **Index options minute NBBO (SPXW, XSP)** | ✅ same endpoint, index symbols | ≥2018 → present | 0DTE 2024 + 2018 probes, 200 OK |
| EOD option chains + greeks (15:45 snap) | ✅ warm cache | 145 symbols, 2018-01 → 2026-07 | 84,749 cached chains |
| Equity minute bars (SIP, incl. pre-market) | ✅ Alpaca | 2016-01 → present | 2016/2024 probes; IEX-no-premarket trap re-confirmed |
| Options trade prints / signed flow | ❌ not on any current subscription | — | GEX/flow strategies closed |
| Equity tick/quote data | ❌ | — | microstructure strategies closed |
| Options data pre-2016 | ❌ (PROFESSIONAL tier) | — | 2008-2015 regimes untestable |
| Historical earnings/CPI/FOMC calendar | ✅ yfinance + static CSVs | 2018 → present | 763 catalysts loaded in v10 |

**Request economics:** full-chain minute pulls for a multi-year campaign ≈ 161k
requests (days of terminal time). Per-contract-day pulls are 0.6s. Therefore:
**strategies select contracts from cached EOD chains; the backtester pulls
minute NBBO only for contracts actually traded.** This is the load-bearing
data-architecture decision.

## 2. Alpha menu (evidence-ranked, cost-aware)

Two venues with a 100–400x cost difference define the design: equities at
~1–2bps round trip (SPY/QQQ) vs options at 0.65–4.4% of premium. Directional
alpha is expressed in the cheap venue; options are used only where a
**structural premium** pays their toll.

### Equities sleeve (cheap venue — direction lives here)
| # | Strategy | Documented net edge | Confidence | Citation anchor |
|---|---|---|---|---|
| E1 | Intraday time-series momentum (first 30min → last hour), **vol-gated** | +2–5bps/trade uncond., +5–15 gated | HIGH (JFE 2018 + 2024 net-of-cost replication) | Gao/Han/Li/Zhou; Baltussen et al. JFE 2021 |
| E2 | Vol-regime gating overlay on E1/E3 | +0.2–0.4 Sharpe as multiplier | HIGH | Moreira-Muir JF 2017 |
| E3 | Event-day continuation on "in-play" names (gap + ≥3–5x rel. volume) | +5–20bps, fat-tailed | MEDIUM | mechanism-solid, regime-concentrated |
| E4 | Post-FOMC/CPI drift on SPY (~20 events/yr) | +10–30bps/event | MEDIUM, satellite only | post-2015 evidence OK |
| E5 | Gap-conditioned open behavior (SPY/QQQ) | +3–8bps | MEDIUM | tug-of-war literature |

**Skipped as arbitraged/data-infeasible:** minute-scale lead-lag, sub-5-min
cross-sectional reversal, classic ORB, VWAP-band folklore, overnight-premium
capture (falsified live — NightShares closure), pre-FOMC drift (dead post-2015),
closing-auction imbalance (needs proprietary feeds).

### Options sleeve (expensive venue — structural premium only)
| # | Strategy | Documented net edge | Confidence | Citation anchor |
|---|---|---|---|---|
| O1 | Conditional 0DTE **defined-risk premium selling** on SPXW/XSP (put-ratio / broken-wing family), 10:00+ ET entry, hold-to-expiry bias | Only peer-quality net-positive candidate; must survive 65–100% spread crossing | MEDIUM-HIGH | Vilkov SSRN 4641356; Beckmeyer et al. 4404704 |
| O2 | FOMC/CPI event-day index vol selling (0/1DTE defined-risk) | Best per-trade edge/toll ratio; pooled with O1, N too small standalone | MEDIUM | Nasdaq FOMC vol premium; event studies |
| O3 | Post-earnings first-hour IV-bleed selling, tightest mega-cap chains only | marginal; kill if toll >3% of premium | LOW | — |

**Documented unviable (will not burn backtest cycles):** long options intraday
on any directional signal (coin-flip × theta × spread = triple-negative — also
measured here: Kinetic −$1,156/trade real-cost), unconditional short strangles,
GEX-anything (no signed flow data; bp-scale post-2020), unusual-options-activity
following, pinning, earnings straddle buying.

### Measured constraints the design engineers around (this repo's own evidence)
- Intraday direction is a coin flip (8 signals × 525 symbol-days, zero |t|>2.5).
  → No standalone directional model. Direction appears only as *conditioned*
  effects (E1's documented conditional structure, E3/E4 event windows).
- 0DTE **long** options lose $1,209/trade frictionless. → O1 is *short*
  premium, defined-risk, hold-to-expiry-biased (one toll, not two).
- Option cycling drag compounds fatally. → Options positions ≤2/day, mostly 1,
  hold-to-expiry bias; all cycling lives in equities at 1–2bps.

## 3. Architecture (reuse-first; the map found ~80% exists)

**Reused unchanged:** costs/ (NBBO cost truth + zero-cost twin), risk/
(RiskManager, shrink-only, breakers), execution/ (sole path to broker,
AST-enforced), reporting/ + promotion ledger, pipeline (mandatory OOS +
standard report), registry/deploy gates, ThetaMinuteQuotes + AlpacaMinuteBars,
Kinetic's minute-loop patterns (NaN-safe marks, flagged synthetic fills),
the 8-signal IC gate harness (non-overlapping, random control, |t|>2.5).

**Built new (v12):**
1. **Equity instrument support** — the one structural gap: instrument-typed
   legs (EquityKey | OptionKey), bps-based equity fill model under the same
   cost truth, SimulatedBroker equity ledger, shares risk-basis (defined stop).
2. `Cadence.INTRADAY` + minute-resolution StrategyContext conventions.
3. **IntradayBacktester** — generalized from Kinetic: multi-strategy,
   multi-position minute loop; equity marks per minute; option marks from
   per-contract-day NBBO frames; synthetic-fill accounting always flagged.
4. **IntradayNativeEngine(BacktestEngine)** — the wrapper Kinetic never had;
   buys mandatory OOS split, zero-cost twin, N>100/concentration checks,
   promotion writes, for free.
5. Feature engine (returns/RV/VWAP-distance/rel-volume/time-of-day,
   strictly point-in-time), regime engine (RV percentile bands + event flags).
6. Alpha engines E1–E5, O1–O2 as separate registered strategies.
7. Options contract selector for O1: chain optimizer on credit/width, EV at
   65–100% spread crossing, strike distance in σ, liquidity gates.
8. Portfolio layer: rank by edge×confidence/risk; Greeks caps + correlated-
   exposure caps in RiskManager config; capital flows to top-ranked, cash is
   the default state.
9. Live loop: market-hours minute scheduler around ExecutionEngine (paper mode
   first; same code path).

**Execution realism rules (non-negotiable, enforced in the engine):**
options fill at 65–100% of quoted spread (configurable, never better than
NBBO — asserted), equities at close±cost with no intra-bar heroics, stale
quotes skipped and counted, missing exit contract = unpriceable skip, all
fills flagged if synthetic. Every strategy also runs its zero-cost twin.

## 4. Validation gauntlet (unchanged standards, now intraday)
Signal IC gate (random control, non-overlapping, |t|>2.5) → pipeline backtest
2018–2026 (real + zero cost, 70/30 chronological OOS) → walk-forward across
regimes (2018 vol, 2020 crash, 2021 melt-up, 2022 bear, 2023-24 chop/rally,
2025-26) → Monte Carlo + cost/slippage stress (spread ×1.5, delayed entries)
→ alpha-decay split-half → concentration check → baselines (SPY B&H, simple
momentum, random-entry twin). Verdicts on OOS only. Nothing skips a stage.

## 5. The target, honestly framed now
40%/month on $100k is not supported by any documented net edge in either
literature review: the best equities candidate compounds single-digit
percent per *year* at ≤1 RT/day on index ETFs; documented 0DTE premium-selling
Sharpe, sized at defined-risk ≤2% of account, is single-digit percent per
*month* in good months with real tail risk. The build proceeds because the
system's value is the measured frontier, not the target: every sleeve gets
built, gated, and stress-tested, and the final report states exactly what the
frontier is and how it was measured. If measurement surprises us upward, the
report will show it — earned, not manufactured.

## 6. Build order (STEPS 6–20 mapped)
1. Equity instrument support + tests (core seam)
2. IntradayBacktester + IntradayNativeEngine + tests
3. Feature + regime engines + point-in-time tests
4. E1 (flagship) → IC gate → pipeline
5. O1 contract selector + engine → pipeline
6. E3/E4/E5, O2 → gates
7. Survivor ensemble + portfolio layer
8. Walk-forward, MC, decay, baselines
9. Paper-trading loop
10. Final report (architecture, data map, alpha library, math spec, results,
    failure analysis, deployment config — conservative defaults first)

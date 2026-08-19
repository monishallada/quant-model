# v12 Intraday Platform — Final Report

*Phase 27 deliverable. Every number below was measured on this repository's
data through its honesty machinery: pre-specified gates with random controls,
real-cost + zero-cost twins, 70/30 chronological out-of-sample splits, N and
concentration checks, and a promotion ledger no strategy can write to by hand.*

---

## 1. The headline conclusions

1. **The platform is built and validated** — a minute-resolution, instrument-
   aware (equities + options) research and execution stack on the existing
   honest pipeline. 519 tests. Every component below is production-grade.
2. **No intraday strategy survived honest costs.** Six literature-ranked
   candidates entered the gauntlet; all six died — five at pre-specified
   gates, one (O1) at real execution costs after its gross edge was confirmed.
3. **The 40%/month target is not achievable on this data**, and per the spec
   this is reported plainly: the best gross edge found intraday is +0.21%/month
   before frictions. No configuration, sizing, or ensemble of measured edges
   reaches within two orders of magnitude of the target. This conclusion is a
   measurement, not an opinion — the numbers are in §5–7.

## 2. Architecture (as built)

```
data/        ThetaData minute NBBO (options, incl. SPXW/XSP index) + EOD chains
             Alpaca SIP minute bars (equities, incl. pre-market) - all parquet-cached
core/        instrument-typed legs (EquityKey | OptionKey), multiplier-aware money math
costs/       single cost truth: worse-side NBBO crossing + slippage + commissions;
             equities bps-based; zero-cost twin runs the identical path
engines/     IntradayBacktester: minute event loop, leak-tested visibility rule
             (bar T visible at T+1min; decide at ts, fill at ts), flagged synthetic
             fills, mandatory EOD flatten (assert-flat), intra-session breaker
risk/        RiskManager gates every entry vs reconciled broker state; risk-basis
             vs cash-basis split (options: coincide; shares: stop distance vs price)
pipeline/    mandatory full/train/test x real/zero-cost, exclusive OOS split,
             standard report, promotion ledger (evidence-written only)
strategies/  IntradayStrategy contract: strategies see visible bars + read-only
             option quotes; return intent only. No broker/risk/cost access,
             AST-enforced.
```

## 3. Data map (verified by live probes)

| Source | Dataset | Depth | Status |
|---|---|---|---|
| ThetaData | option minute NBBO (equity underlyings) | 2016-01 → present | ✅ verified |
| ThetaData | **SPXW/XSP index option minute NBBO** | ≥2018 → present | ✅ verified |
| ThetaData | EOD chains + greeks (15:45) | 2018 → present, 145 syms | ✅ warm cache |
| Alpaca | SIP minute bars incl. pre-market | 2016-01 → present | ✅ verified |
| — | options trade prints / signed flow | — | ❌ closes GEX/flow strategies |
| — | equity tick/quote data | — | ❌ closes microstructure strategies |
| — | anything pre-2016 | — | ❌ tier-gated |

Request economics (measured): per-contract-day minute NBBO 0.6s; SPY full-day
bulk 166s; SPXW full-day bulk times out (>300s); SPXW EOD chain ~5.5min. These
numbers dictated the quote-driven architecture (≈4 targeted requests/session).

## 4. Alpha library — every candidate, every verdict

| # | Candidate (literature-ranked) | Pre-specified gate result | Verdict |
|---|---|---|---|
| E1 | First-30min → last-30min momentum (GHLZ JFE 2018), SPY/QQQ, 5,314 sym-days | Regression real (t=4.13, control clean) but **not monotone** (rank IC −0.02) and not tradeable: honest expanding-window version **−2.3bps/trade net** both symbols | **DEAD** — decayed post-publication |
| E3 | Event-day "in-play" continuation (gap ≥2% + 3× rel-volume), 6 mega-caps | **−48bps/trade** (t=−1.5, n=73); the rel-volume condition *flips the sign*; gap-only variant (+12.6bps, t=1.94) fails the gate and would be post-hoc | **DEAD** |
| E4 | Post-FOMC/CPI drift, SPY, n=183 events | FOMC t=0.41; CPI wrong-signed; pooled t=0.05 | **DEAD** — underpowered as warned |
| E5 | Gap-conditioned open behavior (follow moderate / fade extreme), n=4,251 | t=−0.02 / −0.08 on both pre-specified arms | **DEAD** — flat zero |
| E2 | Vol-regime gating overlay | An overlay; nothing survived to overlay | moot |
| O1 | **0DTE SPXW defined-risk put-spread premium selling** (Vilkov-anchored) | Gross edge CONFIRMED (+0.21%/mo, PF 1.16, 73% win, N=1,992, concentration 4–5%); **real-cost −0.18%/mo, OOS −0.29%/mo, PF 0.88** | **DEAD at retail execution** — the entire edge sits inside the spread |

## 5. O1 deep-dive — the one result with information in it

Full period 2016–2026, one defined-risk SPXW put credit spread per qualifying
session, entry 10:30 ET, quote-driven selection (synthetic parity spot +
straddle implied move), 3× credit stop, 15:55 mandatory buy-back:

| segment | cost | monthly | PF | N | win | maxDD |
|---|---|---|---|---|---|---|
| full | real | −0.18% | 0.90 | 1,807 | 70% | −20.8% |
| full | zero | **+0.21%** | **1.16** | 1,992 | 73% | −4.8% |
| test (OOS) | real | −0.29% | 0.88 | 793 | 70% | −12.1% |
| test (OOS) | zero | +0.15% | 1.07 | 793 | 72% | −4.5% |

**The fill-quality frontier** (the operative robustness test — full-period
re-price at each spread-crossing fraction):

| crossing of quoted spread | slippage | monthly | PF |
|---|---|---|---|
| 0% (mid) | 0 | +0.21% | 1.16 |
| 20% | 0 | +0.16% | 1.11 |
| 40% | 0 | +0.09% | 1.06 |
| **60%** | 2% | **−0.18%** | 0.90 |
| 80% | 2% | −0.21% | 0.86 |
| 100% | 2% | −0.19% | 0.85 |

Break-even ≈ 45% of quoted spread. A retail marketable order on SPX options
cannot be *assumed* to fill better than that band; claiming it would require
measured execution-quality data (Rule-605-style) for the actual broker. Until
such a measurement exists, the honest verdict stands: REJECT.

This reproduces v8's EOD finding at intraday frequency and 3.7× the sample:
**the index variance premium is real, thin, and fully consumed by the spread
at retail access.** Who keeps it: whoever pays no spread — market makers and
wholesalers. That is not a strategy defect; it is the market's structure.

## 6. Trading behavior (O1, real-cost path)

~1 trade/session · 0.6 positions held at any time · avg hold ~5h · capital
utilization ~2.5% of equity at risk per position · win 70% · avg win ≈ +$96 ·
avg loss ≈ −$305 · largest loss −$2,414 (capped by structure) · commissions +
spread = 0.39%/mo drag on a $100k account.

## 7. Why the 40%/month target fails, quantitatively

- Best measured intraday gross edge: **+0.21%/mo** (O1, before frictions).
- Best prior campaign gross edge anywhere in this repo: +2.88%/mo (equal-weight
  mega-cap shares, buy & hold — not intraday, not options).
- 40%/mo = 5,600%/yr compounded. The gap to the measured frontier is ~200×.
- Leverage does not close it: v9/v11 measured the geometric ceiling of the
  best convex distribution at **+0.53%/mo** (growth-optimal f), and the
  bold-play maximum P(8.2× in 8 months) at ~14% with median outcome ≈ 0.
- Every mechanism that promised more (intraday cycling, signal-directed
  options, event drift) was measured and died at its gate in this build.

## 8. Failure analysis — when this system loses money

O1 (if ever run): loses on gap-down/vol-spike days (the 30% of stopped or
expired-ITM trades cluster on drawdown days: maxDD −20.8% real / −4.8% gross);
its worst case is a fast morning selloff through both strikes before the 3×
stop can act — bounded at (width − credit) by construction. The equity book:
flat by design; the platform holds cash when no gate-passing edge exists,
which is the current, correct state.

## 9. Recommended deployment configuration (conservative defaults first)

1. **Deploy nothing to live.** No strategy holds `validated=True`; the deploy
   gate enforces this without exceptions. This is the system working.
2. **Optionally paper-trade O1 for execution-quality measurement only** —
   not for P&L: paper fills vs quoted spread would produce the effective/quoted
   ratio that decides whether the 45% break-even is reachable at a real broker.
   That measurement, not more backtesting, is the only path that could revive O1.
3. **Data purchases that would expand the viable menu** (in value order):
   Cboe Open-Close signed volume (unlocks GEX/flow properly), options trade
   prints (calibrates the cost model with realized effective spreads).
4. The strongest evidence-backed use of this account remains the non-intraday
   result: the shares core (+2.88%/mo measured) with at most a small,
   growth-optimally-sized convex sleeve (+0.5%/mo ceiling, measured).

## 10. What the platform is worth going forward

The deliverable is a research machine that kills bad ideas for ~$0 in hours:
six candidates gated in two days, one full options campaign priced on 8,083
contract-days of real minute NBBO, every kill recorded with its evidence in
LEARNINGS.md. The next hypothesis — yours or the literature's — plugs in as
one file and faces the same gauntlet.

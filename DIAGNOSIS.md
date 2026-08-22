# DIAGNOSIS — why the current system produces few trades and near-zero returns

*2026-08-22. Every number below was computed from this repo's code and ledgers,
then independently re-derived by a second pass before landing here. Sources are
cited as file:line or ledger path. Nothing in this document is an estimate.*

## The five numbers

| # | Question | Answer |
|---|---|---|
| 1 | Signals per symbol per day | Active gen: **67 evaluated → 5.0 emitted → 0.79 executed** (champion). Archived directional gen: **0.0128/symbol/day** — one trade per symbol every 78 trading days. |
| 2 | Friction share of gross P&L | **99.1%** (champion 2024). 62.6% (2023), 283.9% (2025-Q1), 400%+ at high frequency. The backtest does **not** fill at mid. |
| 3 | Lookahead | **None found** — 42 sites audited, 0 leaks in the active path. Fills are zero-latency (optimistic, not lookahead). |
| 4 | Exit ladder capping winners | **No.** The described 10/15/20 trim ladder does not exist in the code. The only trim touched 6/276 trades and cost ~$1.7k against a −$59.8k total. Winners ran to +941%. |
| 5 | Per-trade expectancy (honest OOS) | **+0.0016R, 95% CI [−0.043, +0.046], N=204** — indistinguishable from zero. |

**The one-line diagnosis:** the flat equity curve is not caused by mid-fill
fantasy, lookahead, or exit truncation — the backtester is honest and the exits
are near-irrelevant. It is caused by (a) a gross edge of ~1%/month that is the
same size as the modeled cost of trading it, and (b) in the archived
directional generation, entry signals with no measurable edge at all. The
system trades rarely because its gates correctly refuse almost everything;
when the gates were loosened (817-trade variants), friction reached 4× gross.

---

## 1. How many signals fire per symbol per day?

**Active generation (MOSAIC, the current strategy):** one decision every 5
minutes inside 09:45–15:15 ([mosaic.py:117-119](src/catalyst/strategies/active/mosaic.py#L117-L119),
cadence gate at :303-306) = **67 theoretical decisions/symbol/day**; the
measured counter is 16,800 over 252 sessions = 66.7/day
(`report.json extras.gates_full_real`, funnel closes exactly).

Champion (`qqq_efficient`, QQQ 2024) funnel per day:

```
67 decision minutes
 → 41.9 die at the smile-fit gate        (no_smile 10,567/yr)
 →  8.1 die with no tradeable candidate  (no_candidates 2,053)
 → 10.9 die at the EV gate               (ev_rejected 2,743)
 →  5.04 proposals emitted               (1,270/yr = 7.6% of decisions)
 →  0.79 trades executed                 (200/yr; 84% of emissions dropped
                                          because a position is already open)
```

The loosest variants (SPY, pre-tightening) emitted 21–23/day and executed
~3.2/day. **Instrumentation gap:** the biggest attrition stage — 1,070 of
1,270 champion emissions dropped by the one-position-per-symbol rule — is a
bare `continue` with no counter ([intraday.py:230](src/catalyst/backtest/intraday.py#L230));
the archived generation persists no rejection counts at all (transient
`logger.debug` only, [backtester.py:621,652,668](src/catalyst/backtest/backtester.py#L621)).

**Archived directional generation (what the operator describes):** 276 trades
over 2,156 sessions × 10 symbols (`archive/results/gate3/gate3-all.json`,
2018-01→2026-07) = **0.128 trades/day for the whole book**, 0.0128 per symbol
per day. The rebuild's "many candidate signals per day" goal is a ~400×
frequency increase over this generation at the evaluation stage — which MOSAIC
already achieves (67/day evaluated); the funnel, not the generator, is where
frequency goes to die, and it dies there because of #2.

## 2. What fraction of gross P&L does friction consume?

**The backtest does not fill at mid — mid is a hard floor.**
`NBBOCostModel.leg_fill` ([model.py:94-115](src/catalyst/costs/model.py#L94-L115)):
buy = mid + 0.60·(ask−mid), sell = mid − 0.60·(mid−bid), then 2%-of-premium
slippage against the trader, then $0.65/contract/leg commission (schwab
profile forced by [backtest.yaml:11-12](config/backtest.yaml#L11-L12)). An
assertion raises `BetterThanNBBOError` if any fill lands better than mid
(model.py:62-82). Only the diagnostic zero-cost twin fills at mid.

Friction share of frictionless gross (zero-cost twin vs real, same machine):

| Run | Gross (zero-cost) | Friction | Net | Friction % of gross |
|---|---|---|---|---|
| Champion 2024 (N=200) | +$15,787 | $15,642 | +$145 | **99.1%** |
| 2023 out-of-year (N=128) | +$11,357 | $7,104 | +$4,253 | **62.6%** |
| 2025-Q1 holdout (N=76) | +$1,986 | $5,637 | −$3,651 | **284%** |
| pilot_v2 SPY (N=817) | +$10,003 | $40,081 | −$30,078 | **401%** |
| comm_free SPY (N=801, $0 comm) | +$6,697 | $28,166 | −$21,469 | **421%** |

Per champion trade: **$78.21 total friction** = $16.73 commissions (verified
to 1e-6 against qty × 2 legs × 2 sides × $0.65 on all 200 rows) + $61.48
spread-crossing + slippage (jointly attributed; ledgers carry no per-fill
spread column — an instrumentation gap the rebuild's cost-attribution report
must close). The comm_free row is decisive: **even at $0 commissions,
crossing+slippage alone is 4.2× gross at ~3 trades/day.** This alone explains
the flat curve.

## 3. Is there lookahead?

**No — verdict CLEAN across 42 audited sites** in the active path
(pipeline → IntradayNativeEngine → IntradayBacktester → MosaicStrategy →
SimulatedBroker):

- All 8 `.shift(` hits in `src/catalyst/` (including the negative-shift
  forward-return labels) live under `strategies/archive/` — unreachable from
  the active path. Zero pandas `merge`/`join`/`merge_asof` in `data/` or
  `backtest/`.
- Bar visibility: decisions at ts see bars ≤ ts−1min
  ([intraday.py:190-191](src/catalyst/backtest/intraday.py#L190-L191)); the
  strategy's quote accessor is capped the same way (:273).
- Fills price from the ts snapshot: equity at the ts bar's open (:287),
  options at the ts NBBO boundary snapshot (:311) — confirmed against a
  cached quote frame (391 boundary rows, 09:30 row empty).
- Exits are evaluated before entries against the same ts-minute NBBO mid
  (staleness-capped 5 min), never intrabar extremes — no option OHLC exists
  in the path at all.
- MOSAIC's fitted state is strictly point-in-time: sessions append only after
  turnover, refits use completed sessions only, and backwards session movement
  resets all state so the full→train→test pipeline order cannot leak
  ([mosaic.py:183-191, 231-233, 286-291](src/catalyst/strategies/active/mosaic.py#L183-L191)).

**The honest caveat is latency, not lookahead:** fills are zero-latency at the
decision snapshot. That is optimistic, and it is exactly what the rebuild's
100ms/500ms/2s latency sensitivity must quantify. Four low-severity
non-lookahead issues were logged (as-of-today expiration listing, cross-session
staleness in two caches, a train/predict vol-state mismatch); none plausibly
inflates P&L.

## 4. Is the exit logic capping winners?

**The described ladder ("10% stop, trim 50% at +10%, 30% at +15%, 20% at
+20%") does not exist anywhere in this repo** — grep over `src/` and `config/`
finds no such parameters. What actually exists:

- Archived gen: one scale-out — sell 50% at **+100%** (engine_a only,
  [base.yaml:93-95](config/base.yaml#L93-L95), applied at
  [manager.py:106-113](src/catalyst/exits/manager.py#L106-L113)); stop at
  −50% of premium; 25–30% trailing stops; engine_b full-close TP at 60% of
  max width.
- Active gen (MOSAIC): time exits (345-min max hold, 15:45 flatten) + credit
  stop at 1.8× credit ([mosaic.py:134-136](src/catalyst/strategies/active/mosaic.py#L134-L136)).

**Quantified:** the trim touched **6 of 276** archived trades (2.2%).
No-trim counterfactual on the reconstructable trades: +$9,103 vs +$7,383
actual — the ladder cost **~$1,719** (≤$2,416 under worse fill assumptions)
against a **−$59,753 total backtest P&L**. The right tail was alive and
uncapped: 40 trades exceeded +100%, four exceeded +250%, the best ran to
+941%. The losses came from entries (stop-loss exits: 64 trades, −$71,295,
averaging −69% on a −50% stop — positions gapping far through the stop) — the
exit ladder is second-order by two orders of magnitude. For MOSAIC, "remove
the stop" was measured directly: the stops_off variant returns −2.96%/mo
(PF 0.61, maxDD −30%) vs the champion's +0.01%/mo — expensive, but
load-bearing. One real gap: per-trade MFE is not recorded anywhere
(`TradeRecord` has entry/exit only), so the rebuild must log MFE/MAE per
trade to make exit-policy comparison a measurement instead of a re-run.

## 5. Measured per-trade expectancy in R

R = structural max loss at entry (width − credit for credit verticals,
premium for debits; ×100 per-share convention verified at
[sizing.py:38](src/catalyst/risk/sizing.py#L38); ledger identity
pnl = (exit−entry)×qty×100 − commissions verified to zero residual on all
rows). Bootstrap: 10,000 resamples, percentile CI.

| Window | N | Expectancy (R) | 95% CI | Hit rate | Payoff | $/trade | CI excludes 0? |
|---|---|---|---|---|---|---|---|
| 2023 out-of-year (untouched) | 128 | +0.020 | [−0.035, +0.073] | 53.9% | 0.97 | +$33 | **No** |
| 2025-Q1 holdout (untouched) | 76 | −0.029 | [−0.109, +0.051] | 48.7% | 0.89 | −$48 | **No** |
| **Pooled honest OOS** | **204** | **+0.0016** | **[−0.043, +0.046]** | **52.0%** | **0.94** | **+$2.95** | **No** |
| 2024 test segment (selection-biased) | 36 | +0.061 | [−0.043, +0.164] | 58.3% | 1.12 | +$119 | No |
| Archived Engine C test (directional) | 28 | +0.056 | [−0.192, +0.337] | 35.7% | 2.26 | +$91 | No |

At the observed 13.9 trades/month, pooled honest OOS expectancy implies
**+0.04%/month on $100k**. Every CI includes zero. By the rebuild's own
promotion criteria (95% CI excluding zero after costs, ≥300 OOS trades),
**nothing currently in this repo qualifies for paper trading** — including
the best thing 16 campaigns produced.

---

## Constraints the target architecture must respect (data audit)

Verified against the data layer (8 ThetaData endpoints used, 1 Alpaca, 1 FMP)
and the cache (89k greeks files, 82k OI files, 40k option minute-quote files,
2016/2018→2026):

| Signal family | Status |
|---|---|
| Skew, term structure, charm/vanna windows, IV/RV spread | **Feasible now** (EOD greeks incl. vanna/charm columns + 1-min NBBO + existing smile engine) |
| GEX / dealer gamma | Mechanically feasible now (OI × EOD gamma), but the v16 research verdict stands: OI-only GEX's sign assumption is unverifiable without signed positioning data — usable as a *regime* feature to be tested, not assumed |
| OFI, tick-rule trade signs, VPIN (equities) | **Adapter not built** — Alpaca SIP historical quotes/trades exist on the API the repo already authenticates against; entitlement unprobed. Canonical VPIN needs the trades adapter; a bulk-volume proxy is possible now on 1-min bars (degraded) |
| Options tape: sweeps, blocks, aggressor side | **Not feasible on current subscriptions** — ThetaData STANDARD serves no options trade prints (verified: zero /trade endpoints callable; repo capability matrix concurs). Needs a tier upgrade or Cboe Open-Close |
| ES futures lead-lag | **Not feasible** — no source carries futures (Alpaca equities-only). Nearest proxy: SPY/QQQ SIP minute bars 04:00–20:00 |
| High-ADR universe screener | Feasible with a small adapter (daily bars work for any symbol; needs a broad candidate listing — Alpaca /v2/assets; FMP plan is forward-only, capped) |

## Lockbox declaration (binding as of this document)

**2026-02-22 → present is the lockbox.** No fit, sweep, feature selection,
sanity check, or research-loop read touches it, effective immediately and
enforced in the rebuilt data loader. It is spent once, on the finalist.
Prior campaigns' runs ended 2026-08-01 or earlier and did not fit on this
window, but the archived ledgers overlapping it (gate3 spans to 2026-07-31)
mean the *archived* strategies cannot claim a clean lockbox; the rebuilt
system's candidates can.

## What this diagnosis implies for the rebuild

1. **The referee already exists and is honest.** The cost model
   (worse-than-mid + slippage + commissions, assertion-guarded), the
   zero-cost twin, and the point-in-time discipline survived a hostile
   re-audit. The rebuild should port them as the referee, not rewrite them.
2. **The binding constraint is gross-edge-per-trade vs ~$40–80 friction, not
   signal count.** Candidate generation is easy (67/day already). Promotion
   must price every candidate against its own friction, per contract, at the
   gate — which the EV-gate architecture already does; what's missing is
   breadth of *orthogonal* signal families and per-fill cost attribution.
3. **Three instrumentation gaps to close in the rebuild:** per-fill
   spread/slippage columns in the ledger; MFE/MAE per trade; counters on
   every silent drop stage (the 84% one-position-per-symbol drop first).
4. **Latency sensitivity is mandatory** — current fills are zero-latency
   optimistic; nothing measured here survives contact with the market until
   the 100ms/500ms/2s sweep says so.
5. **Part of the specified signal library needs data that isn't owned.**
   Options-tape and futures signals require new subscriptions; equity
   microstructure needs a new (probably free) adapter. This is an operator
   decision, flagged in the open questions below.

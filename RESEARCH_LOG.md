# RESEARCH LOG

*Written as-it-happens. Newest entries at the bottom. The headline metric for
any strategy result in this log is average monthly return (out-of-sample,
geometric, with CI and month count) — no strategy results exist yet; nothing
is reported until the harness produces them.*

---

## 2026-08-22 — Diagnosis, foundation, hardening, data

**Diagnosis (DIAGNOSIS.md, all five numbers verified by independent
re-derivation).** The old system's flat curve is explained: friction consumes
99.1% of gross in the best configuration ever measured here (and 400%+ at
high frequency); honest pooled OOS expectancy is +0.0016R, CI [−0.043,
+0.046], N=204 — zero. Not explained by mid-fills (backtest provably fills
worse than mid), not lookahead (42 sites audited clean), not exit truncation
(the described trim ladder doesn't exist; the real one cost $1.7k against
−$59.8k of entry-driven losses). Conclusion driving everything below: the
constraint is gross-edge-per-trade vs ~$40–80 friction, and orthogonal signal
families — not more parameter tuning on the one family we had.

**Lockbox declared: 2026-02-22 → present.** Enforced in code from day one.

**Foundation built (src/edge/), referee first.** Deflated Sharpe + PBO
(CSCV) + purged k-fold with embargo + block bootstrap; cost model ported
bitwise-identical to the audited catalyst stack (1000/1000 differential
fills), extended with latency fills, limit/partial-fill modeling, and
per-fill cost attribution; append-only TRIALS.jsonl registry; event-driven
core with one code path for backtest/paper/live. Calibration verified
adversarially: best-of-2000-noise fails the DSR in 20/20 seeds; PBO ≈ 0.5 on
pure noise, 0.0 on planted skill (cross-checked against an independent
brute-force implementation); purged CV reduces a provably-leaking naive-CV
0.64 accuracy to 0.50.

**Hardening.** A dedicated adversarial pass breached the first lockbox/
registry implementation 9 ways (forgeable key, deletable spent-marker, no
timezone anchor on the wall, unkeyed hash chain, grep-evadable import
gateway, ...). All fixed and re-attacked: 12 vectors blocked, 0 breached,
4 residuals documented in module docstrings with mitigation hooks. Design
principle adopted and written down: accidental violations impossible;
deliberate bypass requires a conspicuous act that permanently marks the
lockbox spent and leaves a registry trace (forge = burn). Suite: 844 passed.

**Data.** Probed five free sources live; all five BUILD. Key finds: FINRA
short-sale history sits on a rolling ~8-year CDN window (eroding daily — full
archive job running now, oldest-first), Cboe VIX/SKEW reach 1990, COT's
Tuesday-data/Friday-release lag confirmed empirically, FRED works keyless via
fredgraph export, Alpaca's existing free keys serve historical SIP quotes AND
trades to 2019+ (probed: 200s with data on all four checks). Adapter wave
building now under the point-in-time convention: every row carries asof_date
AND available_at (ET); all joins key on available_at.

**Next:** feeds wave lands → feature layer → first signal family
implementations, each with its own hypothesis entry here and a TRIALS.jsonl
record before any result is looked at.

## 2026-08-22 (later) — Feeds wave: eight adapters, point-in-time audited

Raw archive completed first (FINRA 2,012 daily files 2018-08→2026-08 now
safe locally — the CDN's rolling window can no longer erode our history;
Cboe 1,730 P/C days + 6 index histories to 1990; Treasury 37 years).

Eight modules built and integrated: FINRA short-ratio, Cboe vol-indices +
P/C, Treasury/FRED rates, CFTC COT/TFF, EDGAR Form 4, market-internals
engine, the CatalystBridge backend (options NBBO/greeks/OI through the
lockbox wall, cache-only by default), earnings-proximity. Suite: 1001
passed / 88 skipped; AST gateway scan clean. Every adapter emits asof_date
AND available_at (ET), and EdgeDataLoader clamps row-level on available_at.
The stamps encode real publication physics — e.g. open interest is stamped
NEXT business day 09:00 ET (day-D OI is not knowable on day D), COT stamps
Friday 16:00 ET for Tuesday data, EDGAR uses acceptanceDateTime+5min and a
late-filed Form 4 delays its whole feature window.

Adversarial PIT audit: 5 sources clean, 3 issues found and being fixed —
(1) COT holiday-Friday weeks stamped ~1 trading day early (~3 wks/yr);
(2) earnings features defaulted to a mode where moved dates leak into
historical joins (strict mode becomes the default; opting out now requires
allow_snapshot_lookahead=True and warns); (3) Cboe bad-print cleaning used a
CENTERED median — a lookahead inside the cleaning step itself (now trailing).
Lesson recorded: lookahead hides in janitorial code, not just signals.

## 2026-08-22 (later still) — Wave 1: first real signal results. Audit: CLEAN. Verdict: 0/7 alive on this tape

Seven hypotheses went in; none survived. Every headline below is OOS-only
(2024-01-02..2026-02-20, 26 months), full universes, nothing cut. The
discipline audit (independent, primary-evidence: registry chain walk,
ledger timestamps, own bootstrap) found ZERO violations.

**Ran and rejected (3):**

- **opening_range_breakout** — geo monthly −1.5%, 342 trades, expectancy
  −0.08R, 95% CI [−0.14R, −0.01R] — strictly negative: the break itself
  loses (gross −$24.8k before the −$6.9k modeled spread). 0/4 regime
  buckets positive.
- **vwap_reversion** — geo monthly −2.2%, 1,738 trades, expectancy −0.03R,
  CI [−0.04R, −0.01R]. The chop-only regime gate discarded 84% of 99k
  emissions and the survivors still lose. Well-measured negative.
- **index_leadlag** — geo monthly −20.8% (equity $100k → $232), 13,772
  trades, expectancy −0.02R, CI strictly negative. A small per-trade edge
  deficit compounded through 30-minute round trips at 4–8 bp; stale-quote
  catch-up does not survive even minute-bar latency.

**Insufficient data, no percentage headlined (2):**

- **short_ratio_deviation** — 8 OOS trades (< 10): INSUFFICIENT-DATA. What
  little exists is bad (2/8 winners). Single names only by design (ETF
  short-ratio is creation/redemption noise). First run pair (trials 9/10)
  was suppressed by a harness epoch-unit bug; re-run honestly as trials
  17/18, flagged duplicate_of, same config.
- **insider_cluster** — 0 trades because the loader's insider kind serves
  zero rows (no cached Form 4 parquets). DATA GAP, not signal evidence.
  Needs the EDGAR archive warmed.

**Blocked before a single bar (2):**

- **cot_extreme** — FileNotFoundError: no CFTC TFF raw archives cached.
  Trials 13/14 recorded, failure.json persisted. Needs the archive warmed.
- **gex_pin** — engine-capability gap, not data: the pit mechanism serves
  only the six feed kinds and cannot route per-expiry greeks/OI frames.
  The data exists (32k+ cached SPY OI frames). Running it honestly needs a
  foundation extension (per-expiry pit routing) — stopped rather than
  worked around. Trials 15/16 recorded.

**Audit detail (all six checks passed):** (1) TRIALS.jsonl chain-verifies
end to end — 18 trials (14 planned + 2 pilot + 2 flagged re-runs), every
hypothesis ≥ 50 chars, every config_hash canonical, duplicate_of links
correct, every result artifact written at/after its trial line's ts, every
summary's run_config byte-identical to its trial's recorded run config.
(2) Lockbox intact: detect_tamper() passes, LOCKBOX_SPENT.json absent, max
timestamp across every ledger/equity/regime artifact is 2026-02-20 ET —
nothing at/after the 2026-02-22 wall. (3) EXPECTANCY.md lists all 7
including failures, ranked, no IS numbers. (4) Tearsheets: cost
attribution sums exactly, regime breakdowns present, deflated Sharpe cites
n_trials=18 (= registry count). (5) Independent bootstrap of ORB OOS from
its ledger: −0.0753R, CI [−0.138, −0.012] — matches to sampling noise;
hit rate and geo monthly reproduce too. (6) Drop identity
(emitted = executed + Σdrops) holds in all 11 persisted summaries.
Suite after everything: 1284 passed / 88 skipped. Referee untouched.
Registry anchor (rolling_digest of TRIALS.jsonl at audit close):
`ec0546a623fef88b305126bde8293229acc3af6a1a8675be44905e89ecc42407`.

**Operator's read.** The platform did its job: seven cheap, honest
funerals in one day, every negative result priced and recorded. Intraday
price-only families on liquid mega-caps at retail latency are dead here —
consistent with the diagnosis (friction vs gross-edge-per-trade). The two
positioning hypotheses remain UNTESTED, not rejected: warm the EDGAR and
CFTC archives before wave 2. gex_pin needs an engine extension first;
that is foundation work, to be attacked as its own task, referee-reviewed.
Nothing gets promoted; nothing gets tuned; the thresholds stay as written
in their hypotheses.

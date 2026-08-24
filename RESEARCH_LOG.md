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

## 2026-08-22 (session continuation) — Closing the wave-1 gaps

Three wave-1 hypotheses were never actually tested (two blocked on missing
archives, one on an engine capability gap). Everything below is plumbing to
make them testable — no thresholds touched, no results yet.

**Archives warmed.** CFTC TFF 2010–2026 (17 yearly zips; the 2010–2012 files
use a legacy `Report_Date_as_MM_DD_YYYY` header, now accepted as an alias —
found by real data, not by guessing) and EDGAR Form 4 for the eight
single names (5,952 filings, 34,179 transaction rows, 2019-01→2026-02-20,
zero parse failures). The Form 4 backfill exposed a real parser bug: the
fixed-width index parser derived column bounds from header labels, but the
data columns are padded wider, truncating dates to `2019-01`. Fixed with a
right-anchored row regex plus a drift guard that raises rather than silently
shrinking an index. Caveat recorded: filings carry the issuer symbol AS
FILED, so Meta's pre-rename rows say `FB` and most Alphabet rows say `GOOG`.

**Engine extension.** Per-expiry point-in-time routing (`options_pit`), the
gap that blocked gex_pin: per-(symbol, session) lazy loads through the
gateway, visibility enforced engine-side, and day-D open interest provably
invisible on day D (it is stamped next-business-day 09:00 ET by the bridge).

**Two platform holes found and fixed while re-running.**
1. The wave-1 run harness lived only in a subagent's scratch directory and
   was gone — the campaign was not reproducible from the repo. It is now
   `src/edge/runners/campaign.py`: RunSpec → trial-first `run_signal` →
   engine → persisted ledger/equity/summary → tearsheet, with the synthetic
   quote source (bar close ± half-spread, fill at touch) as a named,
   recorded modelling assumption rather than an invisible one.
2. `EdgeDataLoader` had no constructor serving market data AND feeds
   together — `with_feeds()` refuses `bars`. Added `ResearchBackend` +
   `with_research()`: bars/quotes/options-EOD via the cache-only
   CatalystBridge, feed kinds via FeedsBackend, one lockbox wall over both.
   (First attempt at this patch inserted a class into the middle of another
   class body and broke 105 tests; reverted and redone. Suite green: 609
   edge tests.)

Two failed trials (19, 20) are recorded in TRIALS.jsonl from the harness bug
before the loader fix. They stand — the registry records attempts, and a
higher trial count only raises the deflated-Sharpe bar. Reruns now executing.

## 2026-08-23 — Wave 1b: the three untested hypotheses, honestly tested

All three are now runnable. None is promotable, but only ONE of the three is
a statement about markets — the other two are statements about coverage.

**A bug of mine corrupted the first attempt, and the correction matters.**
`campaign.py` RECORDED execution overrides it never APPLIED, so the reruns
charged option-grade costs on equity trades: 2% of the *share price* in
slippage per side, plus per-share commissions. cot_extreme's in-sample
result read $16,388 (of which $72,772 was phantom slippage); corrected, it
is **$89,316** — a modest ~11% loss over 13.5 years, not a wipeout. gex_pin's
OOS read −39%; corrected, it is −1.2%. Wave-1 numbers were unaffected
(their harness applied the overrides; ledgers show slippage $0). Fixed so it
cannot recur: the recorded config is now DERIVED from the applied config
object, with regression tests pinning record==apply. The lesson is the
project's own thesis turned on itself — a config that claims one thing while
the engine does another is exactly the failure this platform exists to catch,
and it caught it in my own code.

**cot_extreme — INSUFFICIENT-DATA (structurally trade-starved).**
38 signals in 13.5 years IS; **0 in the 26-month OOS window**. A 2-sigma
leveraged-fund extreme over a 104-week window is rare by construction, so
this hypothesis cannot reach the 300-trade promotion bar in any realistic
sample. Not rejected on edge — unfalsifiable at this frequency.

**insider_cluster — UNTESTABLE ON THIS UNIVERSE (a data-shape finding).**
0 emissions across both spans, and the reason is structural, not plumbing:
across 3,455 feature rows and 7 years, `officer_buyers_21d` **never once
reaches 2** (max = 1; only 8 rows carry any officer buy at all). Mega-cap
officers receive awards and sell; they do not buy on the open market. The
mechanism may well exist — it just does not exist HERE. Testing it needs a
small/mid-cap universe and a much broader Form 4 backfill.

**gex_pin — COVERAGE-LIMITED PROBE, not a verdict on GEX.**
OOS: 12 trades, geometric monthly −0.0%, expectancy −0.05R,
95% CI [−0.18R, +0.06R] — includes zero, hit rate 50%, payoff 0.47. So: no
measurable pinning effect either way. But every trade landed on a MONTHLY
expiry (2024-03-15, 2024-07-19, 2024-11-15, 2025-01-17, ...) because the EOD
greeks cache holds only monthly expiries, while SPY/QQQ expire daily. The
hypothesis was therefore probed on roughly 4% of its opportunity set. The
v16 question ("is OI-only dealer gamma alpha?") remains open; answering it
needs a greeks_eod warm for daily expiries.

**Two more platform defects found and fixed.** (1) The regime classifier and
the tearsheet had drifted into two spellings of the same four buckets
(`high_vol_trending` vs `high_vol_trend`); wave-1's harness had been silently
translating, so the drift only surfaced once the harness was rebuilt in-repo.
One vocabulary now, producer's spelling wins. (2) The tearsheet's regime join
needs a tz-aware ET index; the campaign runner now builds it, so a trade at
any hour picks up its own session's bucket.

**Microstructure layer built** (the family the diagnosis actually pointed
at, and the only one wave 1 could not see). 568 sessions of SPY+QQQ trade
prints staged (3.34GB, zero skips, 2025-01-02..2026-02-20); tick-rule
signing, size-weighted trade-sign imbalance, VPIN on an EXACT volume clock
(boundary prints split proportionally, so buckets hold identical volume),
and trade-intensity z. 17 tests. Two of my own tests failed first and both
were the TEST's error, not the code's: minutes are left-open/right-closed
(a print exactly on 09:30:00.000 belongs to the minute ENDING 09:30), and a
truncated tape's last minute is legitimately incomplete. Corrected the
tests, kept the behaviour. Per-minute frames now precomputing into a compact
derived cache.

**Standing count: 7 hypotheses tested, 0 promoted, 0 tuned.**

## 2026-08-23 (later) — Wave 2: the microstructure family. Both sides of the pair fail.

The first hypotheses in this project that read WHO is transacting rather than
what price did — and the first designed as a matched pair, so that the test
could distinguish "wrong direction" from "no information".

**Design.** Same observation (a large one-sided trade imbalance), opposite
predictions, one discriminator. High VPIN => informed metaorder being worked
in slices => FOLLOW (Kyle; square-root impact law). Low VPIN => uninformed
demand for immediacy => the inventory concession REVERTS (Grossman-Miller).
Unit-tested property: across the whole VPIN range exactly one of the pair
fires — never both, never neither — and they take opposite sides on the same
imbalance. If the discriminator carried information, one should have worked
where the other failed.

**Result: neither worked, and the discriminator carried nothing.**

- **flow_continuation** — OOS geometric monthly **-0.3%**, 95% CI
  [-0.5%, -0.1%], 8 months, **N=191**. Expectancy -0.01R, CI [-0.02R, -0.00R]
  — excludes zero on the NEGATIVE side. Hit rate 36.1%, payoff 1.17.
- **flow_reversion** — OOS geometric monthly **-0.3%**, 95% CI [-0.5%, -0.0%],
  8 months, **N=145**. Expectancy -0.02R, CI [-0.04R, +0.01R] — includes zero.
  Hit rate 42.8%, payoff 0.89.

Both REJECT. Neither reaches the 300-trade bar, and 8 months is far short of
the 46 needed to distinguish a Sharpe-1.0 strategy from zero — so the honest
statement is "no measurable edge", not "proven absent".

**The cost attribution is the finding, again, and it is now unambiguous.**
flow_continuation: gross **-$417**, spread **-$1,761**, net -$2,177.
flow_reversion: gross **-$773**, spread **-$1,344**, net -$2,117.
Gross P&L is approximately ZERO in both directions — the tape's information,
if any, is smaller than one half-spread at a 30-minute horizon. This is the
same physics the diagnosis found in the options book (friction 99.1% of
gross) reproduced in a completely different data family, by signals built on
a completely different mechanism. Four independent families now agree.

**What this does and does not settle.** It does NOT refute market
microstructure theory: metaorder continuation and immediacy concession are
well-evidenced effects, and both plausibly live at horizons (seconds) and
in venues (the actual book) this setup cannot reach with minute bars, trade
prints only, and no quote data. What it settles is narrower and more useful:
**at a 30-minute horizon on SPY/QQQ, trade-tape imbalance and VPIN do not
produce a signal large enough to pay a 1bp half-spread.** Testing the faster
horizon would need L1 quote data (OFI, queue imbalance) and a sub-minute
engine — a different platform, not a parameter change.

**Standing count: 9 hypotheses tested, 0 promoted, 0 tuned.**

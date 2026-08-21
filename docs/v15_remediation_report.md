# v15 Remediation — Execution Report

*Full remediation of the 228-defect census, executed 2026-08-21. 53 files
changed, +1,919/−588 lines. Every fix carries either a regression test or a
loud-failure conversion; the full suite grew from 532 to 645 tests, all green.*

## What was fixed, by phase

### P0 — Execution & money path (all 17 CRITICALs + execution HIGHs)
- **ExecutionEngine rebuilt** against the real Order model: total-quantity legs,
  broker-named position ids, real limit prices, a WIRED staleness guard, and
  in-flight-order dedup. The paper/live path constructs, sizes, places, and
  closes orders for the first time — proven end-to-end against the real
  SimulatedBroker + RiskManager (9 tests, no mocks in the money path).
- **A real trading loop exists** (`execution/session.py`): reconcile → breaker
  marks → exits (via the single exit interpreter) → catalyst entries, all
  through ExecutionEngine. `paper_tested` now requires an OBSERVED round trip;
  connecting grants nothing.
- **Alpaca adapter**: closes transmit as closes (`*_to_close`), the credit/debit
  sign survives (no more `abs()`), gcd unit math, `None`-preserving fill
  prices, loud reconciliation (unparseable = refuse, equities included,
  per-unit marks), async-honest cancels, every documented status mapped.
  15 payload tests. Schwab adapter mirrored (12 tests).
- **SimulatedBroker**: strict close-leg matching (mirror legs or REJECT),
  per-leg expiry settlement (calendars keep their back legs), credit
  **collateral encumbrance** (credit selling is now bounded), zero-bid options
  unsellable, crossed quotes refused at the cost model, NaN guards on every
  quote path, single commission source, synthetic closes pay commissions.
- **Promotion ledger**: research (`--set`) runs never write it; atomic saves;
  corrupt files quarantined not overwritten; code-hash resolution failure
  refuses live; repo-anchored paths.
- **Mode/config binding**: `--mode paper` REQUIRES the paper config;
  **risk limits are now byte-identical across backtest/paper/live and enforced
  by a test** — before this, paper/live inherited the research-era limits
  (5% floor, 95% heat): 8× the deployment the backtests validated.

### P1 — Data integrity
- Atomic cache writes + corruption quarantine + collision-proof keys.
- All five residual cache-poisoning paths killed (daily bars, in-progress
  sessions, yfinance empty results, HTTP 472/empty bodies, partial IV series).
- IV series: partial assemblies never cached; far-from-money "ATM" days are
  holes, not samples.
- Greeks: all served fields NaN-validated; the dead py_vollib path formally
  retired (probe-at-import; the in-house solver — which produced every
  measured number — now verified against Hull's textbook values, put-call
  parity, and finite differences: 14 reference tests).
- Catalyst hygiene: phantom 2025-shutdown CPI rows removed, same-day
  CPI+FOMC collapsed, coverage-gap warnings, PIT hazards documented loudly on
  the model, reaction-session boundary semantics under test (10 tests).
- SIP feed unified for daily+minute data (feed-scoped cache keys).

### P2 — Methodology
- Daily engine clamps the snapshot on 18 early-close sessions.
- Announcement-day entries get their event-capture expiry.
- Walk-forward boundaries exclusive (no shared session).
- The six-segment matrix is genuinely mandatory: a failed segment makes the
  verdict say NO RESULT instead of silently omitting the diagnostic.
- Metrics corrected against definitions: standard Sortino denominator,
  bankrupt curves report −100% (not flat 0.0), CAGR/Calmar guards, scratch
  trades excluded from losses, P(ruin) refuses corrupt input loudly,
  Sharpe's rf assumption is now a parameter. 13 hand-computed reference tests.
- Ledger records the TEST-segment number (not 70%-in-sample) next to the
  test-judged verdict.
- Same-symbol/same-session catalyst dedup; fill-reconciled max-loss; broker
  names position ids; exit-fill prices required on FILLED results.
- Intraday: single exit interpreter shared with the daily engine (dead-zone
  stops fixed), fills can't use stale mark quotes, flatten re-marks at the
  flatten timestamp, telemetry counted once, synthetic closes reconcile.

### P3 — Types & tests
- **Money-path bug-class mypy errors: 93 → 0** (arg-type/union-attr/call-arg/
  attr-defined/assignment/operator/index across execution, brokers, risk,
  costs, backtest core, reporting, exits).
- Coverage where verdicts are made: pipeline 0→81%, report 37→92%, execution
  engine 35→87%, Alpaca 0→74%, Schwab 0→61%, kill switch 45→77%, metrics
  72→82%. Suite: 532 → 645 tests.
- The only-path-to-broker invariant is enforced by an AST test that EXISTS.
- Deploy gates under test: mode/config binding, --set refusals, wiring.

### P4 — Structure & truth
- Single sources: kill-switch path (config-wired), commissions (cost model),
  unit decomposition (gcd everywhere), position naming (broker), close-limit
  sign convention (−value), annualization conventions documented.
- Dead code removed (PipelineRun, _equity_series, dead hedge param, dead
  status branches); every docstring invariant now either enforced by a named
  existing test or rewritten to the truth.
- Calendar edges fail loudly (left-edge wrap, holiday close-time).
- Cross-check engines (nautilus/lean) carry explicit divergence-detection-only
  banners; the LEAN zero-cost fee switch keys correctly.
- Archived in-sample "dial"/sweep outputs labeled as such; archiving preserves
  the promotion ledger; the PDF runner picks the newest report and says which.

## Deliberately documented rather than "fixed" (18 items)
Vendor PIT limitations (yfinance current-knowledge calendars, post-hoc EPS),
the IV series' sawtooth tenor, BS q=0/flat-r fallback limits, percentile-vs-
rank naming, per-engine gate-counter attribution, and the nautilus shim's
approximations: these are data/method realities that code cannot un-know.
Each now carries a loud in-code warning at the exact consumption site, so no
future strategy can trip over them silently.

## Acceptance test
The v14 baseline (gated stops-on short-VRP, 2018–2026) re-run on the fixed
engine — results appended below when complete. Expected: small legitimate
drift from the half-day clamp (18 sessions), same-session dedup, and the
correlation cutoff fix; verdict unchanged (INCONCLUSIVE, negative economics).

## Acceptance test — RESULT (v14 baseline, pre-fix vs post-fix engine)

| cell | PRE-FIX | POST-FIX | N |
|---|---|---|---|
| full/real | −0.122%/mo | −0.150%/mo | 34 → 52 |
| full/zero | −0.026%/mo | +0.018%/mo | 34 → 52 |
| test/real | −0.238%/mo | −0.261%/mo | 16 → 26 |
| test/zero | −0.085%/mo | +0.035%/mo | 16 → 26 |
| verdict | INCONCLUSIVE — N=16 | INCONCLUSIVE — N=26 | — |

Every delta traces to a named fix:

1. **N grew 34 → 52** because D-021 was silently SUPPRESSING valid spec
   entries: announcement-day entries (every AMC reporter's designed entry
   session) never got their event-capture expiry injected, so the structure
   gate failed invisibly. The "true" spec strategy always traded ~52 times;
   the pre-fix engine could only see 34. (D-068's macro floor also admits
   CPI/FOMC events to screening — evaluations rose 3,612 → 5,151 — but all
   still fail the vol gates, as v14 concluded.)
2. **Zero-cost flipped mildly positive** (+0.018%/mo full): the recovered
   same-day entries are the crush-capture entries the spec intended, and they
   are gross-positive. **Real-cost stays negative everywhere** (−0.150%/mo
   full): friction (~0.17%/mo drag) still swamps the gross edge — the v14
   REJECT stands, now measured on the strategy the spec actually described.
3. Verdict unchanged: INCONCLUSIVE on sample size, negative economics at
   tradeable prices. No prior conclusion flips; one prior number (N) was
   materially understated by a defect this remediation found and fixed.

Also observed during acceptance: the terminal's session died twice mid-run and
the NEW failure machinery behaved exactly as designed — prefetch aborted
loudly (D-217), the engine refused to degrade segment-by-segment (D-085), no
cache poisoning, and a supervised retry converged on attempt 2 from cache.

## Final state
- 645 tests passing, 92 skipped (all skips scoped and documented).
- Money-path bug-class type errors: 0.
- Promotion ledger: `validated=false` for all strategies; paper_tested
  grantable only by observed round trips.
- System quiesced: no backtests running, terminal idle, kill switch clear,
  `strategies/active/` empty, everything ready to run.

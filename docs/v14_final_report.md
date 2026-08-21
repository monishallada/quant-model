# v14 — Concentrated Short-VRP (Event Iron Condors): Final Report

*Campaign complete. Every number below was produced by the standard pipeline
(real ThetaData NBBO + Alpaca closes, real/zero-cost twins, 70/30 chronological
OOS split, promotion-ledger verdict) and then independently re-derived by a
10-agent adversarial verification pass over the persisted trade ledgers. Four
defects that pass found in my own first draft — including a lookahead leak in
the IV provider — were fixed and every affected run was re-executed.*

---

## 1. Headline: AVERAGE MONTHLY RETURN

Definitive runs, leak-free code, 2018-01-02 → 2026-08-01:

| variant | full REAL | full ZERO | test (OOS) REAL | test ZERO | N full | verdict |
|---|---|---|---|---|---|---|
| **Gated, stops-on** (the spec) | **−0.12%/mo** | −0.03%/mo | −0.24%/mo | −0.09%/mo | 34 | INCONCLUSIVE — N=16 < 100 |
| Gated, stops-off | −0.16%/mo | −0.03%/mo | −0.28%/mo | −0.06%/mo | 34 | INCONCLUSIVE — N=16 < 100 |
| Ungated, stops-on | −0.85%/mo | −0.17%/mo | −1.25%/mo | −0.25%/mo | 171 | INCONCLUSIVE — N=69 < 100 |
| Ungated, stops-off | −0.95%/mo | −0.23%/mo | −1.27%/mo | −0.28%/mo | 169 | INCONCLUSIVE — N=69 < 100 |

Two orderings hold in **both** regimes, which is what makes them believable:
**stops-on beats stops-off** (the spec's open question, answered by the
backtester), and **gating beats not gating** — concentration cuts the loss ~7×
by trading ~5× less. Neither ordering crosses zero.

**Every segment of every variant, at both real and zero cost, is negative.**
The formal pipeline verdict is INCONCLUSIVE on sample size (test N=16 < 100);
the economic verdict is REJECT — the strategy loses ~$2,624 before any
friction and ~$11,801 after it, over 8.6 years on a $100k account.

*Note on convention: `avg_monthly_return` in this repo is geometric,
(1+CAGR)^(1/12)−1. The arithmetic mean of monthly returns agrees to within
1e-5 on every segment (verified).*

Stops-on vs stops-off, as the spec required the backtester to decide:
**stops-on wins** (−0.12% vs −0.16%/mo real), but both lose. The stop is not
the disease.

## 2. What was specified vs what was measured

The v14 bet: implied volatility is systematically overpriced at the *richest*
catalysts, so concentrating defined-risk condor selling on IV rank ≥ 80th
pctile AND IV/RV ≥ 1.3 AND implied ≥ 1.1× the historical event move should
collect a crush that unconditional selling cannot.

Measured answer: **concentration does not rescue the seller.** The premium at
"rich" events is not rich enough to cover the moves those events deliver, and
eight bid/ask crossings on a four-leg structure bury the remainder.

## 3. Campaign integrity — four defects found and fixed

This campaign's most valuable output may be the bugs it exposed:

1. **Three gates were silent no-ops** in the first run (IV-rank compared on a
   0–100 scale against a 0–1 threshold; IV/RV computed from event-expiry IV,
   which is mechanically inflated by the very event being traded; historical
   moves measured on announcement dates instead of reaction sessions — 0.8%
   "history" vs the true ~4%). Caught by the new gate-metrics sidecar.
2. **A lookahead leak in the IV provider** (found by adversarial verification,
   not by me): `iv_rank`/`atm_iv30` selected `series.index <= day`, but the
   series is EOD greeks stamped ~15:59 while every decision is taken at the
   15:45 snapshot — so the entry day's own closing IV, ~15 minutes in the
   future, was setting gate 1 and the numerator of gate 2. Present on 82% of
   evaluated events. Impact was *adverse* (leaked-in trades lost money) and no
   verdict changed, but it is now fixed: the provider defaults to
   strictly-before, with `include_day=True` as an explicit opt-in.

All four fixes are locked in by regression tests (535 passing) covering the
evaluate() passing path and the provider's point-in-time contract — the paths
that had no coverage, which is exactly why the bugs shipped.

## 4. Sensitivity across gate thresholds (the spec's overfit test)

Per-trade outcomes over the full 6×7×6 threshold grid, joined per-event to
corrected gate metrics. One-at-a-time marginals (each gate alone):

| IV-rank ≥ | zero-cost total | | IV/RV ≥ | zero-cost total | | implied/hist ≥ | zero-cost total |
|---|---|---|---|---|---|---|---|
| 0.0 | −$18,025 | | 1.0 | −$16,037 | | 0.9 | −$13,513 |
| 0.8 | −$11,672 | | 1.3 | −$8,628 | | 1.1 | −$11,493 |
| 0.9 | −$6,013 | | 1.5 | −$5,556 | | 1.3 | −$5,446 |

Every marginal is negative at every threshold — tightening only shrinks the
loss by trading less. The full joint grid, 237 cells with n ≥ 20:

- **Zero cost: 14 cells positive** — a contiguous band at implied/hist ≥ 1.3
  with the IV-rank gate *removed*, best +$89/trade (n=24).
- **Real cost: 0 of 237 cells positive.** Best cell −$102/trade. The median
  zero→real gap is $221/trade, and the positive band sits precisely where
  friction is heaviest.
- The spec's own cell (0.80/1.3/1.1): n=34, **−$95/trade zero-cost**,
  −$367/trade real.

Per the spec's rule — "robust across a range = real; one value = overfit" —
the robust finding is that **the edge is negative everywhere at tradeable
prices**. The zero-cost band is a friction-free artifact, not a strategy.

## 5. Why it loses: the anatomy (gated stops-on, real cost, N=34)

| exit reason | n | total P&L | avg | win rate |
|---|---|---|---|---|
| stop at 2× credit | 11 | −$14,568 | −$1,324 | 0% |
| time stop (day after event) | 7 | −$763 | −$109 | 43% |
| expiry buffer | 1 | −$80 | −$80 | 0% |
| take-profit at 50% credit | 15 | +$3,610 | +$241 | 100% |
| **net** | **34** | **−$11,801** | −$347 | 53% |

Three compounding mechanisms, each measured:

1. **The payoff geometry is upside-capped, downside-heavy.** TP at half the
   credit caps every winner near +$241; losers average −$1,324 at the stop.
   A 1:5 payoff needs an ~84% win rate; the structure delivers 53–65%.
2. **The stop fires on the event gap itself** — 11 of 34 trades, all losers,
   the single largest bucket. Turning it off does not help (−0.16%/mo): those
   trades then die at the hard exit instead (expiry_buffer −$1,619 avg,
   time_stop −$712 avg). Lose-lose, because the *entry* is mispriced, not the
   exit.
3. **Friction is ~$270/trade** (verified: mean $270, median $270 on the gated
   book; $256/$233 ungated). Commissions are only 15–19% of it; the bid/ask
   crossings are the tax. The zero-cost book already loses −$2,624; friction
   multiplies the loss ~4.5×.

## 5b. The scale-free explanation: friction as a share of credit

The cleanest statement of why this family fails needs no statistics and no
sample-size caveat. Measured on the ledgers, per share of a four-leg round
trip:

| short Δ | median credit | friction | **friction / credit** | gross edge | win rate |
|---|---|---|---|---|---|
| 0.10 | $0.29 | $0.18 | **61%** | +$66/trade | 78% |
| 0.16 (spec) | $0.52 | $0.29 | **55%** | −$77/trade | 65% |
| 0.25 | $0.80 | $0.53 | **66%** | −$213/trade | 47% |

**You surrender 55–66% of every dollar of credit to the spread**, at every
strike distance tested. The ratio is essentially flat in delta: moving the
shorts further out collects less premium and pays proportionally less
friction, so it does not escape the toll.

For a short-VRP structure to profit, the variance risk premium would have to
overprice options by *more than 60% of premium*. The literature — and this
repository's own prior measurements — put the overpricing at a few percent.
The strategy is not marginally unprofitable; it misses by an order of
magnitude, and no gate, stop, or strike selection changes that arithmetic.

## 5c. Strike-delta dose response

The one structural knob the gate grid cannot reach, swept end to end:

| short Δ | N | gross $/trade | real $/trade | friction | win % | credit/width |
|---|---|---|---|---|---|---|
| 0.05 | **14** | +$51 | −$47 | $98 | 86% | 0.043 |
| 0.07 | 24 | +$40 | −$84 | $124 | 79% | 0.071 |
| **0.10** | 27 | **+$66** ← gross peak | −$80 | $145 | 78% | 0.108 |
| 0.16 (spec) | 34 | −$77 | −$347 | $270 | 65% | 0.206 |
| 0.25 | 36 | −$213 | −$921 | $708 | 47% | 0.349 |

Across a **5× range of strike distance**, real-cost P&L per trade never
crosses zero. Three things move together and explain each other:

1. **Gross edge peaks at 0.10Δ and falls away on both sides.** Nearer the
   money the shorts are breached too often; further out the credit shrinks
   faster than the spread does. A genuine interior optimum — and it is
   +$66/trade, against $145 of friction at the same point.
2. **Win rate rises monotonically** (47% → 86%) as the shorts move out,
   exactly as the geometry requires, and it never rescues the P&L: the payoff
   ratio deteriorates in step with it.
3. **The apparent improvement at 0.05Δ is an artifact of not trading.** N
   collapses to 14 trades in 8.6 years as the minimum-credit and liquidity
   gates reject premium that thin. The curve approaches break-even by
   converging on inactivity — losing less because it risks less, never
   because it earns.

The correlation between delta and gross P&L per trade is −0.96 over five
points. A consistent ordering across the whole sweep is real evidence in a way
no single cell is: every individual cell's t-statistic is insignificant
(§7b), but the *shape* is not an accident. And the shape says the spec's
0.16Δ was not the problem, and that no strike selection is the solution.

## 5d. Fill-quality frontier — what execution would this need?

Run at the sweep's optimum (0.10Δ), re-pricing the same 27 trades at each
spread-crossing fraction. The three crossing-only rows isolate the spread term
(slippage and commissions zeroed); the last row is the production cost model.

| execution assumption | friction/trade | P&L/trade | PF | monthly |
|---|---|---|---|---|
| mid pricing (0% crossing) | $0 | +$66 | 1.47 | +0.02% |
| 20% of quoted spread | $45 | +$21 | 1.14 | +0.005% |
| **40% of quoted spread** | $60 | **+$5** | **1.03** | ≈0.00% |
| 60% crossing + 2% slippage + commissions | $145 | −$80 | 0.54 | −0.02% |

**Break-even sits at roughly 40% of the quoted spread** — and that is the
*generous* reading, since the crossing-only rows carry no slippage and no
commissions. Add those back and the bar tightens further.

The comparison that matters: v12's index 0DTE campaign measured break-even at
~45% of quoted spread. Two different instrument classes, two different
structures, two different campaigns — the same threshold band. That converts a
strategy-specific result into a statement about retail options execution:

> Defined-risk premium selling at retail breaks even somewhere around 40–45%
> effective/quoted spread capture. The production cost model assumes 60%,
> which is why every version of this trade lands under water.

Whether a real broker fills better than 40% on four-leg single-name spreads is
**not answerable by backtesting**. It requires Rule-605-style execution-quality
data, or a measured paper/live sample of actual fills against the NBBO at the
moment of each order. That measurement — not more simulation — is the only
thing that could revive this family.

## 6. Tail-event report (the defined-risk audit)

- **At mid pricing: ZERO breaches of the structural cap** (width − credit) ×
  qty × 100, across every losing trade in every ledger. Defined-risk was
  defined-risk; the engine's cap arithmetic is exact.
- **At real cost, forced exits overshoot the theoretical cap**: 2 breaches
  (gated stops-on, max $1,768 over), 5 (gated stops-off, max $1,497), 33
  (ungated, max $2,300 — 2.4× cap). Every breach is a pre-expiry exit through
  post-event quotes; none is a settlement. **Live implication: "max loss known
  at entry" is an expiry-day number, not an early-exit number.**
- Median realized loss is ~50% of cap (gated) — losses are ordinary, not
  catastrophic; there are simply too many of them.

## 7. Concentration, both directions

Gated stops-on real: top-3 gains = 29% of gross gains; top-3 losses = 46% of
gross losses. Ungated (N=157): 9% and 13%. No lottery artifact — at N=157 the
P&L is diffuse in both directions, confirming the loss is systematic rather
than a few disasters. The high gated shares are a small-sample effect and one
more reason the N>100 bar exists.

## 7b. What the sample can and cannot resolve

The campaign's numbers split into one quantity that is certain and one that is
not, and the distinction decides how much weight the verdict can bear.

**Friction is deterministic.** Bootstrapped 95% CIs on per-trade cost drag are
entirely positive in every variant — $62–229 (0.10Δ), $130–416 (spec 0.16Δ),
$197–306 (ungated, N=171). Friction is not a random variable; it is a toll
paid on every trade, measured with confidence.

**The gross edge is not resolvable at these sample sizes.** Bootstrapped 95%
CIs on zero-cost P&L per trade straddle zero in *both* gated cells: −$109 to
+$210 at 0.10Δ (t = 0.79), −$344 to +$165 at 0.16Δ (t = −0.59). Neither the
sign of the edge nor the difference between deltas is distinguishable from
noise at N = 27–36. The monotone dose response in §5c is the evidence that
survives this — a consistent ordering across five configurations is not
something any single insignificant cell can claim.

**Where the sample IS adequate, the answer is decisive.** At N = 171
(ungated), real-cost P&L per trade is −$342, 95% CI [−$454, −$237],
bootstrapped probability of profitability **0.0%**.

So the verdict does not rest on the noisy quantity. The robust claim is not
"the edge is negative" but:

> Whatever the gross edge is, it is smaller than the measured friction, and
> the sample cannot resolve its sign. A design that trades four times a year
> cannot be validated on 8.6 years of data — which is a finding about the
> design, not a limitation of the test.

## 8. Walk-forward (fixed rules, rolling yearly windows)

Ungated real-cost: **negative every year, 2018–2026** — structural, not
regime-dependent. Zero cost is mixed (2020–21, 2023, 2025 positive; 2018,
2022, 2024, 2026 negative), i.e. even the frictionless edge is unstable in
sign. Gated (N=34): too few trades per window for inference, which is the
point of the N>100 rule.

## 9. Why N>100 is unreachable at the spec gates

Of ~600 in-range events, 188 reach an entry-day evaluation. The three gates
reject 85 / 49 / 15, liquidity 5 → **34 trades in 8.6 years ≈ 4/year**. The
gates are highly correlated (105/99/122 pass individually, 40 jointly).
Macro events are excluded twice over: only 15 of the CPI/FOMC calendar even
reaches evaluation (index condors with one-strike wings fail the credit
floor), and none passes the vol gates — index IV rarely ranks ≥80th pctile on
a scheduled macro day. Reaching N>100 out-of-sample at this rate needs ~70
years or a ~10× wider watchlist, at which point liquidity, not the thesis,
binds.

## 10. Where this sits in the repo's measured history

| campaign | side | result |
|---|---|---|
| Engines A/B | **buy** event premium | −0.45% / −2.25%/mo |
| v10 | **buy** strangles at every catalyst | PF 0.96 *frictionless* |
| v8 | **sell** index VRP unconditionally, EOD | +0.07%/mo (breakeven) |
| O1/v12 | **sell** 0DTE SPXW spreads intraday | +0.21%/mo gross, −0.18% real |
| **v14** | **sell** event premium, concentrated | **−0.03% zero / −0.12% real** |

The family portrait is now complete from both sides. **Single-name event
premium is priced close to fair — buyers lose a little, sellers lose a
little, and the bid/ask spread consumes both.** The only positive gross edge
this repo has ever measured in the VRP family (O1, +0.21%/mo) also died inside
the spread. At retail access the spread is the only durable edge in this game,
and it belongs to the market maker.

## 11. Verdict and disposition

- Pipeline verdict: **INCONCLUSIVE — N=16 < 100**; economics negative at zero
  cost, so more sample would not change the sign.
- vs Engine C baseline (+1%/yr): **fails**.
- Promotion ledger: `validated=false` written by the pipeline. Not eligible
  for paper or live, and should not be.
- **Tuning** (the spec's closing instruction) was done through the
  pre-specified sensitivity grid and a structural delta sweep, not post-hoc
  selection: no threshold combination and no strike-delta is worth promoting.
  The honest tune is "don't trade it."
- **The one path that could revive this**: measure your broker's actual
  effective/quoted fill ratio on four-leg single-name spreads. If it is
  reliably below ~40%, the 0.10Δ configuration becomes worth re-testing — with
  a watchlist wide enough to reach N>100, since at 4 trades/year the current
  universe can never validate it. If it is at or above 40% (the retail norm),
  this family is closed.
- Reusable assets earned this campaign: the corrected IV-provider contract
  (scale, tenor, point-in-time), reaction-session event moves, ledger/equity/
  gate-metric persistence on every pipeline run, the `--set` research override
  (backtest-only), and regression tests over the previously-uncovered paths.

*Generated 2026-08-20. Verification: 10-agent adversarial pass — ledger↔report
reconciliation ×3, cap-integrity, full grid re-derivation, lookahead hunt,
twin symmetry, cost decomposition, completeness critique.*

# v16 — MOSAIC: unified intraday options algorithm (design)

*One master decision engine. Research/simulation mode ONLY: the brokerage
execution layer stays untouched until the explicit phrase "APPROVE PAPER
DEPLOYMENT" is given by the operator. Nothing in this campaign calls
place_order against any non-simulated broker.*

## 0. Honest frame (what 15 campaigns already measured)

This build stands on measured ground, not hope:

- The ONLY confirmed gross intraday options edge on this data: short-dated
  index premium (O1: +0.21%/mo gross, PF 1.16, N=1,992) — killed at retail
  execution (break-even ≈45% of quoted spread).
- Single-name event premium: fair-priced both directions; friction 55–66% of
  collected credit (v14).
- Intraday direction: statistically indistinguishable from coin-flip at every
  horizon tested (v12 E1/E3/E5).
- The research target ($20–30k/day on $100k = 20–30%/day) sits ~4 orders of
  magnitude beyond the best measured gross edge anywhere in this repo
  (+2.88%/mo). It is treated as an optimization DIRECTION; the report will
  state plainly what the data supports.

MOSAIC's thesis is therefore NOT "predict direction." It is: **price the
short-horizon distribution better than the option market prices it, minute by
minute, and only trade when the mispricing exceeds measured friction.** Every
prior campaign priced ONE structure with ONE rule; MOSAIC prices the whole
local surface against a conditional distribution and optimizes the structure.

## 1. Universe & data (designed to the data we actually have)

- Core: SPY + QQQ options, 0–2 DTE, ATM ±6 strikes, 1-minute NBBO
  (ThetaData, cached per contract-day). Both have daily expiries.
- Underlying state: 1-minute SIP bars (Alpaca), full session + premarket.
- Context: EOD chains + greeks (15:45), daily OI, IV-rank provider.
- NOT available (and therefore NOT claimed): options trade prints, signed
  flow, tick data. "Flow" features are volume/OI-based only; dealer-gamma is
  an OI-only proxy carried with explicit uncertainty and admitted only if it
  survives its keep/drop test.

## 2. Architecture — one decision engine, seven measured organs

```
minute bars ─► RV ENGINE          (bipower/RK realized vol, diurnal curve,
                                   rest-of-horizon forecast)
minute bars ─► CONDITIONAL DIST   (empirical conditional quantiles of returns
                                   by time-of-day × vol-state × trend-state;
                                   fit on train window only, frozen for test)
minute NBBO ─► SURFACE ENGINE     (per-minute smile fit on the strike window;
                                   ATM IV, skew, curvature; local dislocation
                                   z-scores per contract)
      both  ─► IV/RV RELATIVE     (conditional P(RV beats IV) by state)
EOD chain   ─► CONTEXT            (term structure, IV rank, OI gamma proxy*)
minute bars ─► REGIME             (vol-level × trend buckets; refit per
                                   walk-forward window)
            ─► PAYOFF TRANSFORM   (underlying quantile grid × IV-dynamics map
                                   → option P&L distribution per candidate)
            ─► STRUCTURE OPTIMIZER (long call/put, debit vertical: maximize
                                   risk-adjusted EV net of modeled friction)
            ─► MASTER EV RANKER   (EV / max-loss, liquidity caps, portfolio
                                   greeks, capital rotation vs open positions)
            ─► EXECUTION DECISION (cross / limit / skip from spread state and
                                   alpha-decay estimate)
```
(*) admitted only on out-of-sample evidence.

Master output per minute: NO TRADE, or {structure, legs, qty, max price,
exit conditions, expected hold}.

## 3. Where the P&L physics must come from

Net edge per trade must exceed ~60% of the quoted spread per leg round-trip
(measured friction). Long options/verticals on SPY 0-1DTE ATM: spreads
typically $0.01–0.05 on ~$1–3 premium → friction floor ~1–4% of premium per
round trip. Therefore the machine hunts effects of ≥ several % of premium per
hour: distribution-vs-smile mispricing at short horizon, conditional IV/RV
gaps, post-shock vol dislocations. The EV gate refuses anything smaller.
This is the narrowest honest funnel that could clear costs — and the report
will show exactly what survives it.

## 4. Anti-overfitting protocol (pre-specified)

- Chronological 70/30 via the standard pipeline (full/train/test × real/zero).
- All fitted components (quantile tables, IV/RV conditionals, diurnal curves,
  regime thresholds) fit on data strictly before each decision (expanding
  window, ≥1-session lag) — enforced in code, not convention.
- Walk-forward: rolling windows via the existing machinery.
- Parameter perturbation: EV threshold, quantile bins, surface window, exit
  params — each swept ±50%; a single-point edge = overfit = reported as such.
- Monte Carlo: trade resequencing + bootstrap on daily P&L (existing engine).
- Controls: (a) shuffled-signal control (same trade times, random contracts);
  (b) the EV gate at threshold 0 (trade everything) as the friction baseline.
- Final untouched holdout: the LAST 3 months are excluded from ALL fitting
  and sweeps; touched exactly once, for the paper-simulation run.

## 5. Phases

A. Research sweep (8 parallel briefs) — running.
B. Engines with unit tests against known values (RV, distribution+calibration,
   surface fit, payoff transform, optimizer).
C. Strategy `mosaic` (IntradayStrategy) through the audited intraday engine.
D. Pilot: SPY 2024 (data machine validation + first honest read).
E. Scale: SPY+QQQ, 2021→holdout-start; full pipeline matrix.
F. Robustness: walk-forward, perturbation, MC, stress (2022 bear, vol spikes,
   2024-08-05, 2025 windows in-range), slippage/fill frontier, liquidity caps.
G. Paper simulation (SimulatedBroker only, holdout window, live-shaped loop).
H. PDF (the 40-section report) → STOP → await "APPROVE PAPER DEPLOYMENT".

## 6. Step 0 record
Engine clear: no processes, active/ empty, kill switch clear, working tree
clean at start. ThetaData terminal restarted (session flakiness persists;
supervised-retry pattern used for all long fetches).


## Frontier grid (final — 2024 pilot + confirmations)

| variant              | full real | full zero | test real | N |
|---|---|---|---|---|
| comm_free            | -1.96% | +0.58% | -2.52% | 801 |
| composite            | +0.32% | +0.37% | +0.23% | 208 |
| efficient            | -0.24% | +0.57% | +0.23% | 170 |
| efficient_noD        | -0.27% | +0.52% | +0.31% | 162 |
| ev_frac_10           | -2.98% | +0.15% | -3.65% | 660 |
| ev_frac_20           | -0.65% | +0.41% | -0.27% | 218 |
| ev_frac_30           | -0.18% | -0.00% | -0.13% | 27 |
| ev_frac_40           | -0.04% | -0.00% | -0.04% | 5 |
| fill_00              | -0.28% | +0.79% | -1.12% | 798 |
| fill_20              | -0.61% | +0.79% | -1.33% | 800 |
| hold_close           | -0.96% | +1.06% | +0.33% | 387 |
| holdout_2025q1       | -1.25% | +0.66% | +3.04% | 76 |
| pilot_v1             | -3.57% | +1.36% | -3.18% | 909 |
| pilot_v2_aligned     | -2.94% | +0.79% | -3.62% | 817 |
| qqq_2023_confirm     | +0.35% | +0.91% | +0.78% | 128 |
| qqq_composite        | -0.40% | -0.37% | -0.25% | 83 |
| qqq_efficient        | +0.01% | +1.23% | +1.02% | 200 |
| qqq_ev20             | -0.81% | -0.37% | -0.47% | 83 |
| stops_off            | -2.96% | +0.55% | -3.88% | 728 |
| width_4              | -1.34% | +0.12% | -1.79% | 605 |

Champion (qqq_efficient) across the three windows, FULL Schwab pricing:
- 2024 (development year): +0.01%/mo net, +1.23%/mo gross
- 2023 (untouched, out-of-year): +0.35%/mo net, +0.91%/mo gross
- 2025-Q1 (untouched holdout, single touch): -1.25%/mo net, +0.66%/mo gross

VERDICT: the gross edge is real and reproducible (~0.7-1.2%/mo in every
window); retail friction (~1%/mo at Schwab option pricing) consumes it.
Net at full retail execution is statistically zero and can be negative
for a quarter. Double-digit monthly returns are NOT supported.

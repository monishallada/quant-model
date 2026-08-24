# REJECTED SIGNALS

Signals that failed promotion, with the named gates they failed and the numbers that failed them.

## REJECTED: opening_range_breakout

- failures: expectancy-ci-includes-zero, pbo-missing, regime-instability
- numbers: n_trades=342, expectancy_r=-0.0753097, ci_lo=-0.138897, ci_hi=-0.0112441, positive_buckets=0, geometric_monthly=-0.0145716

## REJECTED: vwap_reversion

- failures: expectancy-ci-includes-zero, pbo-missing, regime-instability
- numbers: n_trades=1738, expectancy_r=-0.0252689, ci_lo=-0.0405653, ci_hi=-0.0105373, positive_buckets=0, geometric_monthly=-0.0220076

## REJECTED: index_leadlag

- failures: expectancy-ci-includes-zero, pbo-missing, regime-instability
- numbers: n_trades=13772, expectancy_r=-0.0227394, ci_lo=-0.0277583, ci_hi=-0.0178154, positive_buckets=0, geometric_monthly=-0.208071

## REJECTED: short_ratio_deviation

- failures: expectancy-ci-includes-zero, insufficient-oos-trades, pbo-missing, regime-instability
- numbers: n_trades=8, expectancy_r=-0.983394, ci_lo=-2.21646, ci_hi=0.116684, positive_buckets=0, geometric_monthly=-0.00627617

## REJECTED: insider_cluster

- failures: insufficient-oos-trades
- numbers: n_trades=0

## REJECTED: gex_pin

- failures: insufficient-oos-trades, pbo-missing, regime-instability, coverage-limited-to-monthly-expiries
- numbers: n_trades=12, expectancy_r=-0.05, ci_lo=-0.18, ci_hi=0.06, expiries_tested=26, expiries_available_daily=500

## REJECTED: cot_extreme

- failures: insufficient-oos-trades, structurally-trade-starved
- numbers: oos_trades=0, is_trades_13_5_years=38, is_final_equity=89316

## REJECTED: insider_cluster

- failures: setup-never-occurs-on-this-universe
- numbers: oos_trades=0, max_officer_buyers_21d_observed=1, required_officers=2, feature_rows_scanned=3455

## REJECTED: flow_continuation

- failures: expectancy-ci-excludes-zero-negative, insufficient-oos-trades, pbo-missing, regime-instability
- numbers: n_trades=191, expectancy_r=-0.01, ci_lo=-0.02, ci_hi=0, gross_pnl=-416.64, spread_cost=-1760.71, hit_rate=0.361

## REJECTED: flow_reversion

- failures: expectancy-ci-includes-zero, insufficient-oos-trades, pbo-missing, regime-instability
- numbers: n_trades=145, expectancy_r=-0.02, ci_lo=-0.04, ci_hi=0.01, gross_pnl=-773.29, spread_cost=-1343.8, hit_rate=0.428


# EXPECTANCY

Every signal ever tested, ranked by out-of-sample geometric monthly return. Failures included — hiding them is survivorship bias.

## EXPECTANCY RANKING — 3 signals

| Rank | Signal | Geo monthly | Months | Trades | Expectancy (R) | Verdict |
|-----:|:-------|------------:|-------:|-------:|---------------:|:--------|
| 1 | opening_range_breakout | -1.5% | 26 | 342 | -0.08R | REJECT |
| 2 | vwap_reversion | -2.2% | 26 | 1738 | -0.03R | REJECT |
| 3 | index_leadlag | -20.8% | 26 | 13772 | -0.02R | REJECT |

### Not rankable this round (no percentage is honest here)

- **short_ratio_deviation** — INSUFFICIENT-DATA: 8 OOS trades (< 10); verdict REJECT (failed: expectancy-ci-includes-zero, insufficient-oos-trades, pbo-missing, regime-instability)
- **insider_cluster** — INSUFFICIENT-DATA: 0 OOS trades (emitted 0, drops {'regime-gated': 0, 'sizing-zero': 0, 'no-quote': 0, 'latency-no-fill': 0, 'position-already-open': 0, 'risk-cap': 0, 'daily-halt': 0})
- **cot_extreme** — NOT-RUN (backtest failed before replay): `FileNotFoundError: no TFF archives (fut_fin_txt_*.zip / FinFut*.txt) under /Users/monishallada/quant-model/data_cache/edge/raw/cot`
- **gex_pin** — NOT-RUN (backtest failed before replay): `ValueError: unknown pit kinds ['open_interest', 'greeks_eod']; valid: ['cot', 'earnings', 'insider', 'rates', 'short_ratio', 'vol_indices']`

## EXPECTANCY RANKING — 4 signals

| Rank | Signal | Geo monthly | Months | Trades | Expectancy (R) | Verdict |
|-----:|:-------|------------:|-------:|-------:|---------------:|:--------|
| 1 | gex_pin | -0.0% | 26 | 12 | -0.05R | REJECT |
| 2 | opening_range_breakout | -1.5% | 26 | 342 | -0.08R | REJECT |
| 3 | vwap_reversion | -2.2% | 26 | 1738 | -0.03R | REJECT |
| 4 | index_leadlag | -20.8% | 26 | 13772 | -0.02R | REJECT |

## EXPECTANCY RANKING — 6 signals

| Rank | Signal | Geo monthly | Months | Trades | Expectancy (R) | Verdict |
|-----:|:-------|------------:|-------:|-------:|---------------:|:--------|
| 1 | gex_pin | -0.0% | 26 | 12 | -0.05R | REJECT |
| 2 | flow_continuation | -0.3% | 8 | 191 | -0.01R | REJECT |
| 3 | flow_reversion | -0.3% | 8 | 145 | -0.02R | REJECT |
| 4 | opening_range_breakout | -1.5% | 26 | 342 | -0.08R | REJECT |
| 5 | vwap_reversion | -2.2% | 26 | 1738 | -0.03R | REJECT |
| 6 | index_leadlag | -20.8% | 26 | 13772 | -0.02R | REJECT |


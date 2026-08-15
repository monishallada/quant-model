"""Regression: elapsed time must come from dates, not row count.

A curve marked only at trade exits has ~16 rows per year. Deriving months from
len() read that as 0.8 months and turned a +10% YEAR into "+12.8% per MONTH".
Same class of bug as the v8 credit double-count: the number looked plausible,
and only an internal cross-check exposed it.
"""

import pandas as pd

from catalyst.backtest.metrics import avg_monthly_return


def test_sparse_curve_uses_calendar_span_not_row_count():
    """16 marks spanning a full year is a year, not sixteen days."""
    idx = pd.to_datetime([f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in (5, 20)])
    equity = pd.Series([100_000 * (1.10 ** (i / len(idx))) for i in range(len(idx))], index=idx)
    monthly = avg_monthly_return(equity)
    # +10% over ~11.5 months ~= +0.83%/mo. The bug reported ~ +12%/mo.
    assert 0.005 < monthly < 0.012, f"got {monthly:.2%}/mo"


def test_dense_and_sparse_curves_agree():
    """Same economics, different sampling, same headline."""
    dense_idx = pd.bdate_range("2024-01-01", "2024-12-31")
    dense = pd.Series([100_000 * (1.5 ** (i / (len(dense_idx) - 1))) for i in range(len(dense_idx))],
                      index=dense_idx)
    sparse = dense.iloc[::15]
    assert abs(avg_monthly_return(dense) - avg_monthly_return(sparse)) < 0.002

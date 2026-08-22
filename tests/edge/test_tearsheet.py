"""Tearsheet: format-exact headline golden, the <46-months power note,
promotion boundary cases (299 vs 300 trades; CI touching zero), cost
attribution summing to net, regime breakdown, and the EXPECTANCY/REJECTED
markdown writers. All data is synthetic; every timestamp predates the
2026-02-22 lockbox start."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edge.core.config import ValidationConfig
from edge.research.registry import TrialRegistry
from edge.research.tearsheet import (
    FAIL_CI,
    FAIL_LATENCY,
    FAIL_PBO_HIGH,
    FAIL_PBO_MISSING,
    FAIL_REGIME,
    FAIL_TRADES,
    MONTHS_TO_DISTINGUISH_SHARPE_ONE,
    REGIME_BUCKETS,
    ExpectancyRow,
    MonthlyHeadline,
    append_expectancy_md,
    append_rejected_md,
    compute_cost_attribution,
    compute_monthly_headline,
    compute_regime_breakdown,
    compute_trade_stats,
    evaluate_promotion,
    monthly_returns,
    render_headline_block,
    render_tearsheet,
)
from edge.validation.stats import bootstrap_ci

ET = ZoneInfo("America/New_York")
# Synthetic study year, well BEFORE the 2026-02-22 lockbox start.
T0 = pd.Timestamp("2025-01-02 10:00", tz=ET)

# Test config: production-shaped gates, cheap bootstrap.
CFG = ValidationConfig(min_oos_trades=300, pbo_max=0.5, bootstrap_resamples=300)

#: Twelve known monthly returns driving the synthetic equity curve.
MONTHLY = [0.02, -0.01, 0.03, 0.015, -0.005, 0.04, 0.01, 0.02, -0.02, 0.05, 0.01, 0.03]


def _equity() -> pd.Series:
    """Equity whose calendar-month returns are exactly ``MONTHLY``.

    First observation (Jan 2) is the starting equity; each month-end point
    compounds one entry of ``MONTHLY``.
    """
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-02", tz=ET)]
        + [pd.Timestamp(f"2025-{m:02d}-28", tz=ET) for m in range(1, 13)]
    )
    levels = 100_000.0 * np.cumprod([1.0] + [1.0 + r for r in MONTHLY])
    return pd.Series(levels, index=idx)


def _regime() -> pd.Series:
    """One canonical bucket per quarter of 2025, in REGIME_BUCKETS order."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2025-{m:02d}-01", tz=ET) for m in (1, 4, 7, 10)]
    )
    return pd.Series(list(REGIME_BUCKETS), index=idx)


def _ledger(n: int = 300, signal: str = "momo") -> pd.DataFrame:
    """``n`` trades, one per day from Jan 2: 20% lose -0.4R, 80% win +0.6R.

    pnl is NET dollars at $100/R; costs per trade: spread 2.00,
    slippage 1.00, commission 0.65.
    """
    entry = pd.DatetimeIndex([T0 + pd.Timedelta(days=i) for i in range(n)])
    loser = np.arange(n) % 5 == 0
    r = np.where(loser, -0.4, 0.6)
    pnl = r * 100.0
    return pd.DataFrame(
        {
            "pnl": pnl,
            "r_multiple": r,
            "entry_ts": entry,
            "exit_ts": entry + pd.Timedelta(hours=3),
            "signal_name": signal,
            "spread_cost": 2.0,
            "slippage_cost": 1.0,
            "commission": 0.65,
            "mfe": np.where(loser, 10.0, pnl * 1.5),
            "mae": -20.0,
            "exit_reason": np.where(loser, "stop", "target"),
        }
    )


def _registry(tmp_path: Path) -> TrialRegistry:
    """Registry with three recorded trials (sharpes 0.5/1.0/1.5)."""
    registry = TrialRegistry(root=tmp_path)
    for i, sharpe in enumerate([0.5, 1.0, 1.5]):
        registry.record_trial(
            f"hypothesis {i}",
            {"param": i},
            {"sharpe": sharpe},
            ts=datetime(2026, 1, 12, 10, 0, tzinfo=ET),
        )
    return registry


# ---------------------------------------------------------------------------
# THE HEADLINE BLOCK — format-exact golden (the operator's contract)
# ---------------------------------------------------------------------------


def test_headline_block_golden_character_for_character() -> None:
    headline = MonthlyHeadline(
        geometric=0.021,
        arithmetic=0.023,
        ci_lo=0.004,
        ci_hi=0.041,
        months=48,
        median=0.019,
        worst=-0.062,
        best=0.104,
        pct_positive=62.5,
        t_stat=2.31,
    )
    expected = (
        "AVG MONTHLY RETURN (out-of-sample)\n"
        "  Geometric (compounded):   2.1%   <- what the account actually does\n"
        "  Arithmetic (mean):        2.3%\n"
        "  95% bootstrap CI:         [0.4%, 4.1%]\n"
        "  Months observed:          48\n"
        "  Median month:             1.9%\n"
        "  Worst month:              -6.2%\n"
        "  Best month:               10.4%\n"
        "  % of months positive:     62%\n"
        "  t-stat vs zero:           2.31\n"
    )
    assert render_headline_block(headline) == expected


def test_headline_note_below_46_months_verbatim() -> None:
    base = dict(
        geometric=0.01,
        arithmetic=0.01,
        ci_lo=-0.002,
        ci_hi=0.02,
        median=0.01,
        worst=-0.03,
        best=0.05,
        pct_positive=60.0,
        t_stat=1.10,
    )
    with_note = render_headline_block(MonthlyHeadline(months=45, **base))
    assert with_note.endswith(
        "NOTE: 45 months < 46 needed to distinguish a Sharpe-1.0 strategy "
        "from zero.\n"
    )
    # Exactly at the threshold the note must NOT appear (45 is the last
    # underpowered count).
    assert MONTHS_TO_DISTINGUISH_SHARPE_ONE == 46
    without_note = render_headline_block(MonthlyHeadline(months=46, **base))
    assert "NOTE:" not in without_note


# ---------------------------------------------------------------------------
# Monthly headline computation
# ---------------------------------------------------------------------------


def test_monthly_returns_recovers_known_months() -> None:
    rets = monthly_returns(_equity())
    assert rets == pytest.approx(np.array(MONTHLY))


def test_compute_monthly_headline_values() -> None:
    headline = compute_monthly_headline(_equity(), CFG, seed=7)
    r = np.array(MONTHLY)
    assert headline.months == 12
    assert headline.geometric == pytest.approx(np.prod(1.0 + r) ** (1 / 12) - 1.0)
    assert headline.arithmetic == pytest.approx(r.mean())
    assert headline.median == pytest.approx(np.median(r))
    assert headline.worst == pytest.approx(r.min())
    assert headline.best == pytest.approx(r.max())
    assert headline.pct_positive == pytest.approx(100.0 * 9 / 12)
    expected_t = r.mean() / (r.std(ddof=1) / math.sqrt(12))
    assert headline.t_stat == pytest.approx(expected_t)
    # CI is exactly bootstrap_ci on the monthly returns at the same seed.
    lo, hi = bootstrap_ci(r, resamples=CFG.bootstrap_resamples, seed=7)
    assert (headline.ci_lo, headline.ci_hi) == pytest.approx((lo, hi))


def test_monthly_returns_rejects_degenerate_input() -> None:
    one_month = pd.Series(
        [100.0, 101.0],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2025-03-03", tz=ET), pd.Timestamp("2025-03-20", tz=ET)]
        ),
    )
    with pytest.raises(ValueError, match="two monthly returns"):
        monthly_returns(one_month)
    negative = pd.Series(
        [100.0, -5.0, 50.0],
        index=pd.DatetimeIndex(
            [pd.Timestamp(f"2025-{m:02d}-28", tz=ET) for m in (1, 2, 3)]
        ),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        monthly_returns(negative)


# ---------------------------------------------------------------------------
# Promotion boundaries
# ---------------------------------------------------------------------------


def _promotion(**overrides: object):
    kwargs: dict = dict(
        n_trades=300,
        expectancy_ci=(0.05, 0.60),
        pbo=0.20,
        latency_survived=True,
        positive_buckets=3,
        config=CFG,
    )
    kwargs.update(overrides)
    return evaluate_promotion(**kwargs)


def test_promotion_all_gates_pass_at_the_boundaries() -> None:
    # n == min_oos_trades, positive_buckets == 3, pbo strictly below max.
    result = _promotion()
    assert result.promoted
    assert result.failures == ()
    assert result.verdict == "PROMOTE"


def test_promotion_299_vs_300_trades() -> None:
    assert _promotion(n_trades=300).promoted
    rejected = _promotion(n_trades=299)
    assert not rejected.promoted
    assert rejected.failures == (FAIL_TRADES,)


def test_promotion_ci_touching_zero_rejects() -> None:
    # A CI whose lower bound TOUCHES zero does not exclude it.
    touching = _promotion(expectancy_ci=(0.0, 0.4))
    assert not touching.promoted
    assert touching.failures == (FAIL_CI,)
    # Strictly-negative CI also fails: promotion needs a positive edge.
    negative = _promotion(expectancy_ci=(-0.3, -0.1))
    assert negative.failures == (FAIL_CI,)
    assert _promotion(expectancy_ci=(1e-9, 0.4)).promoted


def test_promotion_pbo_latency_and_regime_gates() -> None:
    assert _promotion(pbo=0.5).failures == (FAIL_PBO_HIGH,)  # pbo_max is strict
    assert _promotion(pbo=0.499999).promoted
    assert _promotion(pbo=None).failures == (FAIL_PBO_MISSING,)
    assert _promotion(latency_survived=False).failures == (FAIL_LATENCY,)
    assert _promotion(positive_buckets=2).failures == (FAIL_REGIME,)
    assert _promotion(positive_buckets=4).promoted


def test_promotion_collects_all_failures_in_order() -> None:
    result = _promotion(
        n_trades=10,
        expectancy_ci=(-0.1, 0.2),
        pbo=0.9,
        latency_survived=False,
        positive_buckets=1,
    )
    assert result.failures == (
        FAIL_CI,
        FAIL_TRADES,
        FAIL_PBO_HIGH,
        FAIL_LATENCY,
        FAIL_REGIME,
    )


# ---------------------------------------------------------------------------
# Trade stats
# ---------------------------------------------------------------------------


def test_trade_stats_hand_checked() -> None:
    stats = compute_trade_stats(_ledger(), CFG, seed=7)
    assert stats.n == 300
    assert stats.expectancy_r == pytest.approx(0.4)  # 0.8*0.6 - 0.2*0.4
    assert stats.hit_rate == pytest.approx(0.8)
    assert stats.payoff_ratio == pytest.approx(60.0 / 40.0)
    assert stats.avg_holding_hours == pytest.approx(3.0)
    # Winners realize 60 of a 90 MFE: capture 2/3.
    assert stats.mfe_capture == pytest.approx(2.0 / 3.0)
    assert stats.ci_lo > 0.0
    assert stats.ci_excludes_zero


def test_trade_stats_requires_trades() -> None:
    with pytest.raises(ValueError, match="at least one trade"):
        compute_trade_stats(_ledger().iloc[0:0], CFG)


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------


def _cost_lines(sheet: str) -> dict[str, float]:
    block = sheet.split("COST ATTRIBUTION ($)\n", 1)[1].split("\n\n", 1)[0]
    out: dict[str, float] = {}
    for line in block.splitlines():
        parts = line.split()
        out[" ".join(parts[:-1]).rstrip(":")] = float(parts[-1])
    return out


def test_cost_attribution_sums_to_net(tmp_path: Path) -> None:
    ledger = _ledger()
    costs = compute_cost_attribution(ledger)
    # The invariant, on the computed numbers:
    assert costs.gross - costs.spread - costs.slippage - costs.fees == pytest.approx(
        costs.net
    )
    assert costs.net == pytest.approx(float(ledger["pnl"].sum()))
    assert costs.spread == pytest.approx(600.0)
    assert costs.slippage == pytest.approx(300.0)
    assert costs.fees == pytest.approx(0.65 * 300)
    # And on the RENDERED five-line table (costs display as negatives).
    sheet = render_tearsheet(
        "momo", ledger, _equity(), _registry(tmp_path), _regime(), CFG,
        pbo=0.2, latency_survived=True, seed=7,
    )
    rows = _cost_lines(sheet)
    assert set(rows) == {"Gross P&L", "Spread", "Slippage", "Fees", "Net P&L"}
    assert rows["Gross P&L"] + rows["Spread"] + rows["Slippage"] + rows[
        "Fees"
    ] == pytest.approx(rows["Net P&L"])
    assert rows["Net P&L"] == pytest.approx(12_000.0)
    assert rows["Gross P&L"] == pytest.approx(13_095.0)


def test_cost_attribution_rendered_even_with_zero_costs(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger[["spread_cost", "slippage_cost", "commission"]] = 0.0
    sheet = render_tearsheet(
        "momo", ledger, _equity(), _registry(tmp_path), _regime(), CFG,
        pbo=0.2, latency_survived=True, seed=7,
    )
    rows = _cost_lines(sheet)
    assert rows["Spread"] == rows["Slippage"] == rows["Fees"] == 0.0
    assert rows["Gross P&L"] == pytest.approx(rows["Net P&L"])
    assert "-0.00" not in sheet  # zero cost renders +0.00, never -0.00


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------


def test_regime_breakdown_buckets_and_positive_count() -> None:
    ledger = _ledger()
    stats, positive = compute_regime_breakdown(ledger, _regime())
    assert [s.bucket for s in stats] == list(REGIME_BUCKETS)
    assert sum(s.n for s in stats) == 300
    assert all(s.n > 0 for s in stats)
    assert positive == 4


def test_regime_breakdown_empty_bucket_and_negative_bucket() -> None:
    ledger = _ledger(n=200)  # ends 2025-07-20: high_vol_chop never occurs
    # Make every high_vol_trend trade a loser: its bucket goes negative.
    in_q3 = ledger["entry_ts"] >= pd.Timestamp("2025-07-01", tz=ET)
    ledger.loc[in_q3, "r_multiple"] = -0.5
    stats, positive = compute_regime_breakdown(ledger, _regime())
    by_bucket = {s.bucket: s for s in stats}
    assert by_bucket["high_vol_chop"].n == 0
    assert by_bucket["high_vol_chop"].expectancy_r is None
    assert by_bucket["high_vol_trend"].expectancy_r == pytest.approx(-0.5)
    assert positive == 2


def test_regime_breakdown_rejects_bad_inputs() -> None:
    ledger = _ledger(n=10)
    bad_label = pd.Series(
        ["sideways"], index=pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz=ET)])
    )
    with pytest.raises(ValueError, match="unknown regime buckets"):
        compute_regime_breakdown(ledger, bad_label)
    late_start = pd.Series(
        ["low_vol_trend"],
        index=pd.DatetimeIndex([pd.Timestamp("2025-06-01", tz=ET)]),
    )
    with pytest.raises(ValueError, match="before the first regime observation"):
        compute_regime_breakdown(ledger, late_start)


# ---------------------------------------------------------------------------
# Full tearsheet
# ---------------------------------------------------------------------------


def test_render_tearsheet_promotes_and_carries_every_section(tmp_path: Path) -> None:
    sheet = render_tearsheet(
        "momo", _ledger(), _equity(), _registry(tmp_path), _regime(), CFG,
        pbo=0.2, latency_survived=True, seed=7,
    )
    assert "TEARSHEET: momo" in sheet
    assert "AVG MONTHLY RETURN (out-of-sample)" in sheet
    # 12 months of equity: the power note must be present, promotion is
    # unaffected (months never gate promotion).
    assert (
        "NOTE: 12 months < 46 needed to distinguish a Sharpe-1.0 strategy "
        "from zero." in sheet
    )
    assert "  Trades (N):               300" in sheet
    assert "(excludes zero: YES)" in sheet
    assert "COST ATTRIBUTION ($)" in sheet
    assert "  Positive buckets:         4/4" in sheet
    assert "(n_trials=3)" in sheet  # DSR deflated by the registry's count
    assert "  PBO (CSCV):               0.200" in sheet
    assert "  Latency survival:         PASS" in sheet
    assert "  VERDICT: PROMOTE" in sheet
    assert "[FAIL]" not in sheet


def test_render_tearsheet_rejects_with_named_failures(tmp_path: Path) -> None:
    sheet = render_tearsheet(
        "momo", _ledger(n=299), _equity(), _registry(tmp_path), _regime(), CFG,
        pbo=None, latency_survived=False, seed=7,
    )
    assert (
        "  VERDICT: REJECT (failed: insufficient-oos-trades, pbo-missing, "
        "latency-failure)" in sheet
    )
    assert "  [FAIL] oos trades 299 >= 300" in sheet
    assert "  PBO (CSCV):               n/a" in sheet
    assert "  Latency survival:         FAIL" in sheet


def test_render_tearsheet_rejects_bad_ledger(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="missing required columns"):
        render_tearsheet(
            "momo", _ledger().drop(columns=["mfe"]), _equity(), registry,
            _regime(), CFG, latency_survived=True,
        )
    with pytest.raises(ValueError, match="no trades in ledger"):
        render_tearsheet(
            "ghost", _ledger(), _equity(), registry, _regime(), CFG,
            latency_survived=True,
        )


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def test_append_expectancy_md_ranks_all_signals_failures_included(
    tmp_path: Path,
) -> None:
    path = tmp_path / "EXPECTANCY.md"
    rows = [
        ExpectancyRow("alpha", 0.010, 24, 310, 0.15, "REJECT"),
        ExpectancyRow("bravo", 0.030, 48, 500, 0.31, "PROMOTE"),
        ExpectancyRow("charlie", -0.005, 12, 120, -0.05, "REJECT"),
    ]
    append_expectancy_md(path, rows)
    text = path.read_text(encoding="utf-8")
    assert "| 1 | bravo | 3.0% | 48 | 500 | +0.31R | PROMOTE |" in text
    assert "| 2 | alpha | 1.0% | 24 | 310 | +0.15R | REJECT |" in text
    assert "| 3 | charlie | -0.5% | 12 | 120 | -0.05R | REJECT |" in text
    assert text.index("bravo") < text.index("alpha") < text.index("charlie")
    # Appending again adds a second ranking section, header written once.
    append_expectancy_md(path, rows)
    text = path.read_text(encoding="utf-8")
    assert text.count("# EXPECTANCY\n") == 1
    assert text.count("## EXPECTANCY RANKING — 3 signals") == 2
    with pytest.raises(ValueError, match="at least one row"):
        append_expectancy_md(path, [])


def test_append_rejected_md_names_failures_and_numbers(tmp_path: Path) -> None:
    path = tmp_path / "REJECTED.md"
    append_rejected_md(
        path,
        "momo",
        (FAIL_TRADES, FAIL_PBO_HIGH),
        {"n_trades": 299, "pbo": 0.61, "expectancy_ci_lo": -0.02},
    )
    text = path.read_text(encoding="utf-8")
    assert "# REJECTED SIGNALS" in text
    assert "## REJECTED: momo" in text
    assert "- failures: insufficient-oos-trades, pbo-too-high" in text
    assert "- numbers: n_trades=299, pbo=0.61, expectancy_ci_lo=-0.02" in text
    # Second rejection appends; header stays single.
    append_rejected_md(path, "beta", (FAIL_CI,), {"expectancy_ci_lo": 0.0})
    text = path.read_text(encoding="utf-8")
    assert text.count("# REJECTED SIGNALS") == 1
    assert "## REJECTED: beta" in text
    assert "- failures: expectancy-ci-includes-zero" in text
    assert "expectancy_ci_lo=0" in text
    with pytest.raises(ValueError, match="at least one named failure"):
        append_rejected_md(path, "gamma", (), {})

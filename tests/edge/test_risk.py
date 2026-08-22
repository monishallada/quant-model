"""Risk sizing and RISK_BUDGET tests — synthetic data only, no network.

Hand-computed Kelly cases, correlation-group capping (three correlated longs
share one cap), the latching daily-loss halt, and budget math on a crafted
sample whose mean/vol are exact by construction (printed numbers asserted to
2dp). All timestamps predate the 2026-02-22 lockbox start.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from edge.core.config import RiskConfig
from edge.core.events import MARKET_TZ
from edge.risk.budget import (
    NO_EDGE_LINE,
    RiskBudget,
    build_risk_budget,
    requires_kelly_override,
)
from edge.risk.sizing import (
    DailyLossHalt,
    correlation_groups,
    daily_loss_halt,
    kelly_fraction,
    size_position,
    size_positions,
)

# Synthetic risk config for tests; mirrors the shipped shape, not production.
CFG = RiskConfig(kelly_fraction=0.25, per_trade_cap_pct=2.0, daily_loss_halt_pct=3.0)

ANN = math.sqrt(252.0)


def crafted_returns(mu: float, sigma: float, n: int, seed: int) -> npt.NDArray[np.float64]:
    """A return sample whose mean and std (ddof=1) equal mu/sigma exactly."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    out: npt.NDArray[np.float64] = mu + sigma * z
    return out


# ---------------------------------------------------------------------------
# Kelly fraction — hand-computed cases
# ---------------------------------------------------------------------------


def test_kelly_hand_computed() -> None:
    # p=0.6, b=1.5: f* = 0.6 - 0.4/1.5 = 1/3. Expectancy 0.6*1.5 - 0.4 = 0.5.
    assert kelly_fraction(0.5, 0.6, 1.5) == pytest.approx(1.0 / 3.0)
    # p=0.5, b=2: f* = 0.5 - 0.5/2 = 0.25. Expectancy 0.5.
    assert kelly_fraction(0.5, 0.5, 2.0) == pytest.approx(0.25)
    # p=1 always wins: f* = 1.
    assert kelly_fraction(2.0, 1.0, 2.0) == pytest.approx(1.0)


def test_kelly_no_edge_is_zero() -> None:
    # Negative or zero measured expectancy zeroes the bet outright.
    assert kelly_fraction(-0.1, 0.6, 1.5) == 0.0
    assert kelly_fraction(0.0, 0.6, 1.5) == 0.0
    # Positive claimed expectancy but the p/b formula says negative edge.
    assert kelly_fraction(0.1, 0.4, 1.0) == 0.0


def test_kelly_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 1.2, 1.5)  # win rate > 1
    with pytest.raises(ValueError):
        kelly_fraction(0.5, -0.1, 1.5)  # win rate < 0
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 0.6, 0.0)  # payoff must be positive
    with pytest.raises(ValueError):
        kelly_fraction(math.inf, 0.6, 1.5)  # non-finite


# ---------------------------------------------------------------------------
# size_position — Kelly multiplier and the hard cap
# ---------------------------------------------------------------------------


def test_size_position_hard_cap_beats_kelly() -> None:
    # kelly=1/3 at 0.25 mult -> 8.33% uncapped; hard cap 2% wins.
    sized = size_position(100_000.0, 1.0 / 3.0, CFG)
    assert sized.risk_pct == pytest.approx(2.0)
    assert sized.risk_dollars == pytest.approx(2_000.0)
    assert sized.capped is True


def test_size_position_uncapped_and_mult_override() -> None:
    # kelly=0.04 at 0.25 mult -> 1% of equity, under the cap.
    sized = size_position(100_000.0, 0.04, CFG)
    assert sized.risk_pct == pytest.approx(1.0)
    assert sized.risk_dollars == pytest.approx(1_000.0)
    assert sized.capped is False
    # Explicit kelly_mult overrides the config default; landing exactly on
    # the cap is not "capped".
    sized = size_position(100_000.0, 0.04, CFG, kelly_mult=0.5)
    assert sized.risk_pct == pytest.approx(2.0)
    assert sized.capped is False


def test_size_position_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        size_position(0.0, 0.1, CFG)  # non-positive equity
    with pytest.raises(ValueError):
        size_position(100_000.0, -0.1, CFG)  # negative kelly
    with pytest.raises(ValueError):
        size_position(100_000.0, 0.1, CFG, kelly_mult=0.0)  # bad multiplier


# ---------------------------------------------------------------------------
# Correlation groups — three correlated longs sized as ONE position
# ---------------------------------------------------------------------------


def _correlated_panel(seed: int = 11, n: int = 80) -> pd.DataFrame:
    """A/B/C share one driver (pairwise corr ~0.99); D is independent."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, size=n)
    frame = pd.DataFrame(
        {
            "A": base + rng.normal(0.0, 0.001, size=n),
            "B": base + rng.normal(0.0, 0.001, size=n),
            "C": base + rng.normal(0.0, 0.001, size=n),
            "D": rng.normal(0.0, 0.01, size=n),
        }
    )
    corr = frame.tail(63).corr()
    assert float(corr.loc["A", "B"]) > 0.9  # construction sanity
    assert abs(float(corr.loc["A", "D"])) < 0.5
    return frame


def test_correlation_groups_partition() -> None:
    groups = correlation_groups(_correlated_panel())
    assert groups == [("A", "B", "C"), ("D",)]


def test_correlation_groups_transitive_chain() -> None:
    # corr(A,C) ~ 0.5 (below threshold) but B bridges both at ~0.87, so the
    # connected component pulls all three into one group.
    rng = np.random.default_rng(3)
    n = 63
    g = rng.normal(size=n)
    a = g + rng.normal(size=n)
    c = g + rng.normal(size=n)
    frame = pd.DataFrame({"A": a, "B": a + c, "C": c})
    corr = frame.corr()
    assert float(corr.loc["A", "C"]) < 0.7  # the chain is genuinely indirect
    assert float(corr.loc["A", "B"]) > 0.7
    assert float(corr.loc["B", "C"]) > 0.7
    assert correlation_groups(frame) == [("A", "B", "C")]


def test_three_correlated_longs_share_one_cap() -> None:
    panel = _correlated_panel()
    kellys = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2}
    sized = size_positions(100_000.0, kellys, panel, CFG)

    # Each alone would want 0.2 * 0.25 = 5% of equity; the group of three
    # counts as ONE position against the 2% cap -> 2/3% each, 2% combined.
    for symbol in ("A", "B", "C"):
        assert sized[symbol].group == ("A", "B", "C")
        assert sized[symbol].capped is True
        assert sized[symbol].risk_pct == pytest.approx(2.0 / 3.0)
        assert sized[symbol].risk_dollars == pytest.approx(100_000.0 * (2.0 / 3.0) / 100.0)
    group_total = sum(sized[s].risk_pct for s in ("A", "B", "C"))
    assert group_total == pytest.approx(CFG.per_trade_cap_pct)

    # The uncorrelated symbol gets its own full cap.
    assert sized["D"].group == ("D",)
    assert sized["D"].capped is True
    assert sized["D"].risk_pct == pytest.approx(2.0)


def test_size_positions_proportional_split_and_uncapped_group() -> None:
    panel = _correlated_panel()
    # Unequal Kelly within the group: scaling is proportional, total == cap.
    sized = size_positions(100_000.0, {"A": 0.3, "B": 0.1, "C": 0.0, "D": 0.02}, panel, CFG)
    assert sized["A"].risk_pct == pytest.approx(1.5)
    assert sized["B"].risk_pct == pytest.approx(0.5)
    assert sized["C"].risk_pct == 0.0
    assert sized["A"].risk_pct + sized["B"].risk_pct == pytest.approx(CFG.per_trade_cap_pct)
    # D wants 0.02 * 0.25 = 0.5% -> under the cap, untouched.
    assert sized["D"].risk_pct == pytest.approx(0.5)
    assert sized["D"].capped is False


def test_size_positions_rejects_missing_returns_column() -> None:
    panel = _correlated_panel()
    with pytest.raises(ValueError, match="lacks columns"):
        size_positions(100_000.0, {"A": 0.1, "ZZZ": 0.1}, panel, CFG)


# ---------------------------------------------------------------------------
# Daily loss halt
# ---------------------------------------------------------------------------


def test_daily_loss_halt_predicate_boundary() -> None:
    assert daily_loss_halt(-3.0, CFG) is True  # at the limit -> halt
    assert daily_loss_halt(-2.99, CFG) is False
    assert daily_loss_halt(-5.0, CFG) is True
    assert daily_loss_halt(0.5, CFG) is False
    with pytest.raises(ValueError):
        daily_loss_halt(math.nan, CFG)


def test_daily_loss_halt_latches_for_session_and_resets() -> None:
    halt = DailyLossHalt(CFG)
    d1_morning = datetime(2025, 6, 2, 10, 0, tzinfo=MARKET_TZ)
    d1_noon = datetime(2025, 6, 2, 12, 0, tzinfo=MARKET_TZ)
    d1_close = datetime(2025, 6, 2, 15, 55, tzinfo=MARKET_TZ)
    d2 = datetime(2025, 6, 3, 9, 31, tzinfo=MARKET_TZ)

    assert halt.observe(d1_morning, -1.0) is True  # small loss: entries allowed
    assert halt.observe(d1_noon, -3.2) is False  # limit breached: halted
    # Recovery within the SAME session does not un-halt.
    assert halt.observe(d1_close, -0.5) is False
    assert halt.entries_allowed(d1_close) is False
    # A new session starts clean.
    assert halt.entries_allowed(d2) is True
    assert halt.observe(d2, -1.0) is True


def test_daily_loss_halt_accepts_dates_rejects_naive() -> None:
    halt = DailyLossHalt(CFG)
    assert halt.observe(date(2025, 6, 2), -4.0) is False
    assert halt.entries_allowed(date(2025, 6, 2)) is False
    assert halt.entries_allowed(date(2025, 6, 3)) is True
    with pytest.raises(ValueError):
        halt.entries_allowed(datetime(2025, 6, 2, 10, 0))  # noqa: DTZ001 — naive on purpose


# ---------------------------------------------------------------------------
# RISK_BUDGET — crafted sample with known mean/vol
# ---------------------------------------------------------------------------

# mu 0.04%/day, vol 1%/day -> annualized Sharpe 0.04*sqrt(252) = 0.63498.
MU_D, SIGMA_D = 0.0004, 0.01


def _budget(target_pct: float, mu: float = MU_D, resamples: int = 400) -> RiskBudget:
    returns = crafted_returns(mu, SIGMA_D, 252, seed=42)
    return build_risk_budget(returns, target_pct, resamples=resamples, seed=7)


def test_budget_prints_hand_computed_numbers_to_2dp() -> None:
    budget = _budget(2.0)
    report = budget.report_md

    # Annualization: 2%/day * sqrt(252) = 31.75%/yr, and the fixed anchors.
    assert "target daily vol: 2.00%/day" in report
    assert "annualized (x sqrt(252)): 31.75%/yr" in report
    assert "5%/day ~ 79%/yr" in report
    assert "10%/day ~ 159%/yr" in report
    assert "SPY runs ~16%/yr" in report

    # Measured sample: exact-by-construction mean/vol and Sharpe.
    assert "daily mean: 0.0400%/day | daily vol: 1.0000%/day" in report
    assert "annualized Sharpe: 0.63" in report
    assert "leverage to reach target: 2.00x" in report

    # Kelly geometry: implied multiple = 0.3175 / 0.63498 = 0.50.
    assert budget.implied_kelly_multiple == pytest.approx(0.5, abs=1e-9)
    assert "implied Kelly multiple at target: 0.50" in report
    assert budget.requires_override is False
    assert "OVERRIDE REQUIRED" not in report
    assert NO_EDGE_LINE not in report

    # Arithmetic vs geometric at 2x leverage: mu = 0.08%/day -> 20.16%/yr;
    # mu - sigma^2/2 = 0.08% - 0.02% = 0.06%/day -> 15.12%/yr; growth hits
    # zero at 2x full Kelly ~ 2 * 63.50 = 127.00%/yr vol.
    assert "arithmetic drift mu: 20.16%/yr" in report
    assert "geometric drift mu - sigma^2/2: 15.12%/yr" in report
    assert "2.00x full Kelly: ~127.00%/yr vol" in report


def test_budget_override_beyond_full_kelly() -> None:
    # 8%/day target: implied multiple = (0.08*sqrt(252)) / 0.63498 = 2.00 > 1.
    budget = _budget(8.0)
    assert budget.implied_kelly_multiple == pytest.approx(2.0, abs=1e-9)
    assert budget.requires_override is True
    assert "OVERRIDE REQUIRED" in budget.report_md
    assert "--override-kelly" in budget.report_md


def test_budget_no_measured_edge_branch() -> None:
    budget = _budget(2.0, mu=-MU_D)
    assert math.isinf(budget.implied_kelly_multiple)
    assert budget.requires_override is True
    assert NO_EDGE_LINE in budget.report_md
    assert "implied Kelly multiple at target: inf" in budget.report_md


def test_requires_override_boundary_exactly_one() -> None:
    assert requires_kelly_override(1.0) is False  # exactly full Kelly: no override
    assert requires_kelly_override(math.nextafter(1.0, 2.0)) is True
    assert requires_kelly_override(0.99) is False
    assert requires_kelly_override(math.inf) is True


def test_budget_ruin_table_geometry_and_determinism() -> None:
    budget = _budget(10.0)  # 10x leverage: ruin risk must be visible
    table = budget.ruin_table
    assert set(table) == {"6m", "12m"}
    for row in table.values():
        assert set(row) == {"p_dd_50", "p_dd_80", "p_below_20pct", "median_terminal", "p5_terminal"}
        for key in ("p_dd_50", "p_dd_80", "p_below_20pct"):
            assert 0.0 <= row[key] <= 1.0
        assert row["p_dd_80"] <= row["p_dd_50"]  # 80% DD implies 50% DD
        assert 0.0 <= row["p5_terminal"] <= row["median_terminal"]

    # 6m paths are prefixes of the 12m paths, so risk grows with horizon.
    assert table["6m"]["p_dd_50"] <= table["12m"]["p_dd_50"]
    assert table["6m"]["p_below_20pct"] <= table["12m"]["p_below_20pct"]
    # At 10x leverage on a 1%-vol sample, deep drawdowns are not rare events.
    assert table["12m"]["p_dd_50"] > 0.0

    # The markdown table prints the same numbers, to 2dp.
    row = table["6m"]
    assert f"| 6m (126d) | {row['p_dd_50'] * 100.0:.2f}% " in budget.report_md
    assert f"| {row['median_terminal']:.2f}x | {row['p5_terminal']:.2f}x |" in budget.report_md

    # Same inputs + seed -> identical budget.
    again = _budget(10.0)
    assert again.ruin_table == table
    assert again.report_md == budget.report_md


def test_budget_short_sample_tiles_paths_to_horizon() -> None:
    # 100 OOS days < 252-day horizon: paths are tiled, nothing breaks.
    returns = crafted_returns(MU_D, SIGMA_D, 100, seed=5)
    budget = build_risk_budget(returns, 2.0, resamples=200, seed=3)
    for row in budget.ruin_table.values():
        assert all(math.isfinite(v) for v in row.values())
    # Block length shrinks to the sample when shorter than the default 21d.
    tiny = crafted_returns(MU_D, SIGMA_D, 10, seed=6)
    tiny_budget = build_risk_budget(tiny, 2.0, resamples=50, seed=3)
    assert "block 10d" in tiny_budget.report_md


def test_budget_rejects_bad_inputs() -> None:
    good = crafted_returns(MU_D, SIGMA_D, 252, seed=1)
    with pytest.raises(ValueError):
        build_risk_budget([0.01], 2.0, resamples=10, seed=1)  # one observation
    with pytest.raises(ValueError):
        build_risk_budget(good, 0.0, resamples=10, seed=1)  # non-positive target
    with pytest.raises(ValueError):
        build_risk_budget([0.01, math.nan, 0.02], 2.0, resamples=10, seed=1)
    with pytest.raises(ValueError):
        build_risk_budget([0.01, 0.01, 0.01], 2.0, resamples=10, seed=1)  # zero variance

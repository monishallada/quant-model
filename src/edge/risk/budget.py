"""RISK_BUDGET.md — the operator's mandated pre-run report for a vol target.

Given the strategy's OOS daily-return sample and a ``target_daily_vol_pct``,
:func:`build_risk_budget` computes what that target actually costs:

- **Annualization**: daily vol scales by sqrt(252); the report anchors the
  target against 5%/day (~79%/yr), 10%/day (~159%/yr) and SPY (~16%/yr).
- **Kelly geometry**: under the Gaussian approximation, running FULL Kelly
  produces an annualized vol numerically equal to the annualized Sharpe
  (SR 0.6 => ~60%/yr at full Kelly). The Kelly multiple a target implies is
  therefore ``target_ann_vol / measured_oos_sharpe``; a measured Sharpe <= 0
  makes the multiple infinite — no measured edge supports ANY vol target.
- **Ruin geometry**: circular block-bootstrap paths
  (:func:`edge.validation.stats.block_bootstrap`) of the OOS returns levered
  to the target vol, over 6-month and 12-month horizons; probabilities of
  50% and 80% max drawdown, of equity dipping below 20% of start, and the
  median / 5th-percentile terminal equity.
- **Arithmetic vs geometric**: the levered sample's arithmetic drift ``mu``
  vs its geometric growth ``mu - sigma^2/2``; geometric growth crosses zero
  at exactly 2x full Kelly, and the report prints the annualized vol where
  that happens for THIS sample.

The guardrail: ``requires_override`` is True whenever the implied Kelly
multiple exceeds 1.0 (beyond full Kelly). This module only computes, formats,
and returns; the runner enforces ``--override-kelly``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from edge.validation.stats import block_bootstrap, sharpe

#: US equity trading days per year (annualization constant).
TRADING_DAYS_PER_YEAR: int = 252

#: Ruin-geometry horizons: (label, trading days). 6 months and 12 months.
HORIZONS: tuple[tuple[str, int], ...] = (("6m", 126), ("12m", 252))

#: Max-drawdown thresholds reported in the ruin table (fractions of peak).
RUIN_DD_THRESHOLDS: tuple[float, float] = (0.50, 0.80)

#: Equity floor for the ruin-barrier probability: equity < 20% of start.
RUIN_EQUITY_FLOOR: float = 0.20

#: Default block length for the bootstrap: one trading month, preserving
#: autocorrelation up to ~21 days (shrunk to the sample length when shorter).
BLOCK_LEN_DAYS: int = 21

#: Reference daily-vol anchors printed in every report (percent per day).
REFERENCE_DAILY_VOLS_PCT: tuple[float, float] = (5.0, 10.0)

#: SPY's rough long-run annualized vol, printed as the sanity anchor.
SPY_REFERENCE_ANN_VOL_PCT: float = 16.0

#: Kelly multiple above which the runner must demand --override-kelly.
FULL_KELLY_MULTIPLE: float = 1.0

#: The guard line printed verbatim when measured Sharpe <= 0.
NO_EDGE_LINE: str = "no measured edge supports ANY vol target"


@dataclass(frozen=True)
class RiskBudget:
    """Computed RISK_BUDGET: the report plus the machine-readable guardrail."""

    report_md: str
    requires_override: bool
    implied_kelly_multiple: float
    #: {"6m": {...}, "12m": {...}} with keys p_dd_50, p_dd_80, p_below_20pct,
    #: median_terminal, p5_terminal (probabilities as fractions, terminals as
    #: multiples of starting equity).
    ruin_table: dict[str, dict[str, float]]


def requires_kelly_override(implied_kelly_multiple: float) -> bool:
    """True when the implied Kelly multiple STRICTLY exceeds full Kelly (1.0).

    Exactly 1.0 — running precisely at full Kelly — does not require the
    override; anything beyond it (including an infinite multiple from a
    non-positive measured Sharpe) does.
    """
    return implied_kelly_multiple > FULL_KELLY_MULTIPLE


def _horizon_paths(
    scaled_returns: npt.NDArray[np.float64],
    resamples: int,
    seed: int,
    block_len: int,
) -> npt.NDArray[np.float64]:
    """``(resamples, max_horizon)`` bootstrap paths of the levered returns.

    The sample is block-bootstrapped (circular, so autocorrelation up to the
    block length survives); when the sample is shorter than the longest
    horizon, several independent resampled paths are concatenated per row
    before truncation.
    """
    t = scaled_returns.size
    max_days = max(days for _, days in HORIZONS)
    chunks = math.ceil(max_days / t)
    paths = block_bootstrap(scaled_returns, min(block_len, t), resamples * chunks, seed)
    return paths.reshape(resamples, chunks * t)[:, :max_days]


def _ruin_stats(equity: npt.NDArray[np.float64]) -> dict[str, float]:
    """Ruin metrics for one horizon's ``(n_paths, days)`` equity matrix."""
    peak = np.maximum(np.maximum.accumulate(equity, axis=1), 1.0)
    max_dd = (1.0 - equity / peak).max(axis=1)
    lo, hi = RUIN_DD_THRESHOLDS
    terminal = equity[:, -1]
    return {
        "p_dd_50": float(np.mean(max_dd >= lo)),
        "p_dd_80": float(np.mean(max_dd >= hi)),
        "p_below_20pct": float(np.mean(equity.min(axis=1) < RUIN_EQUITY_FLOOR)),
        "median_terminal": float(np.median(terminal)),
        "p5_terminal": float(np.percentile(terminal, 5.0)),
    }


def build_risk_budget(
    oos_daily_returns: npt.ArrayLike,
    target_daily_vol_pct: float,
    *,
    resamples: int,
    seed: int,
    block_len: int = BLOCK_LEN_DAYS,
) -> RiskBudget:
    """Compute and format the RISK_BUDGET report for one vol target.

    Args:
        oos_daily_returns: the strategy's out-of-sample daily returns
            (fractions, e.g. 0.01 = +1%), oldest first.
        target_daily_vol_pct: target daily volatility in percent (5.0 = 5%/day).
        resamples: bootstrap path count — pass
            ``ValidationConfig.bootstrap_resamples`` at the call site.
        seed: RNG seed for the bootstrap (reports are reproducible).
        block_len: bootstrap block length in days (default one trading
            month; shrunk to the sample length when the sample is shorter).

    Returns:
        :class:`RiskBudget` with the markdown report, the override flag
        (implied Kelly multiple > 1.0), the implied multiple itself
        (``inf`` when measured Sharpe <= 0), and the ruin table.

    Raises:
        ValueError: fewer than two return observations, non-finite returns,
            zero return variance, or a non-positive vol target.
    """
    arr = np.asarray(oos_daily_returns, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("oos_daily_returns must be a 1-D sample of at least two days")
    if not np.all(np.isfinite(arr)):
        raise ValueError("oos_daily_returns contains non-finite values")
    if not np.isfinite(target_daily_vol_pct) or target_daily_vol_pct <= 0.0:
        raise ValueError(f"target_daily_vol_pct must be positive, got {target_daily_vol_pct}")

    ann = math.sqrt(TRADING_DAYS_PER_YEAR)
    mu_d = float(np.mean(arr))
    sigma_d = float(np.std(arr, ddof=1))
    sr_ann = sharpe(arr, TRADING_DAYS_PER_YEAR)  # raises on zero variance

    target_daily = target_daily_vol_pct / 100.0
    target_ann = target_daily * ann
    leverage = target_daily / sigma_d
    implied = target_ann / sr_ann if sr_ann > 0.0 else math.inf
    override = requires_kelly_override(implied)

    # Ruin geometry on the sample levered to the target vol. Daily losses are
    # floored at -100%: leverage past ruin absorbs at zero equity rather than
    # producing negative wealth.
    scaled = np.clip(arr * leverage, -1.0, None)
    paths = _horizon_paths(scaled, resamples, seed, block_len)
    equity = np.cumprod(1.0 + paths, axis=1)
    ruin_table = {label: _ruin_stats(equity[:, :days]) for label, days in HORIZONS}

    # Arithmetic vs geometric drift of the levered sample, annualized.
    mu_target_d = mu_d * leverage
    arith_ann = mu_target_d * TRADING_DAYS_PER_YEAR
    geo_ann = (mu_target_d - target_daily**2 / 2.0) * TRADING_DAYS_PER_YEAR

    lines: list[str] = [
        "# RISK_BUDGET",
        "",
        f"Pre-run risk budget for a target daily volatility of {target_daily_vol_pct:.2f}%/day.",
        "",
        "## Volatility target",
        f"- target daily vol: {target_daily_vol_pct:.2f}%/day",
        f"- annualized (x sqrt(252)): {target_ann * 100.0:.2f}%/yr",
        "- reference: "
        + ", ".join(
            f"{v:.0f}%/day ~ {v / 100.0 * ann * 100.0:.0f}%/yr" for v in REFERENCE_DAILY_VOLS_PCT
        )
        + f"; SPY runs ~{SPY_REFERENCE_ANN_VOL_PCT:.0f}%/yr",
        "",
        "## Measured OOS sample",
        f"- observations: {arr.size} days",
        f"- daily mean: {mu_d * 100.0:.4f}%/day | daily vol: {sigma_d * 100.0:.4f}%/day",
        f"- annualized Sharpe: {sr_ann:.2f}",
        f"- leverage to reach target: {leverage:.2f}x the OOS sample as traded",
        "",
        "## Kelly geometry",
        (
            f"- full-Kelly annualized vol equals the annualized Sharpe: "
            f"{sr_ann:.2f} (~{sr_ann * 100.0:.2f}%/yr)"
        ),
        f"- required Sharpe to run {target_ann * 100.0:.2f}%/yr at full Kelly: {target_ann:.2f}",
        "- implied Kelly multiple at target: "
        + ("inf" if math.isinf(implied) else f"{implied:.2f}"),
    ]
    if sr_ann <= 0.0:
        lines.append(f"- **{NO_EDGE_LINE}** (measured Sharpe {sr_ann:.2f} <= 0)")
    if override:
        shown = "inf" if math.isinf(implied) else f"{implied:.2f}"
        lines.append(
            f"- **OVERRIDE REQUIRED**: implied Kelly multiple {shown} exceeds "
            f"{FULL_KELLY_MULTIPLE:.2f} (full Kelly); the runner demands --override-kelly"
        )

    lines += [
        "",
        (
            f"## Ruin geometry ({resamples} block-bootstrap paths, "
            f"block {min(block_len, arr.size)}d, seed {seed})"
        ),
        "",
        (
            "| horizon | P(maxDD >= 50%) | P(maxDD >= 80%) | P(equity < 20% of start) "
            "| median terminal | 5th-pct terminal |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for label, days in HORIZONS:
        row = ruin_table[label]
        lines.append(
            f"| {label} ({days}d) "
            f"| {row['p_dd_50'] * 100.0:.2f}% "
            f"| {row['p_dd_80'] * 100.0:.2f}% "
            f"| {row['p_below_20pct'] * 100.0:.2f}% "
            f"| {row['median_terminal']:.2f}x "
            f"| {row['p5_terminal']:.2f}x |"
        )

    lines += [
        "",
        "## Arithmetic vs geometric at target",
        f"- arithmetic drift mu: {arith_ann * 100.0:.2f}%/yr",
        f"- geometric drift mu - sigma^2/2: {geo_ann * 100.0:.2f}%/yr",
    ]
    if sr_ann > 0.0:
        zero_growth_vol_ann = 2.0 * sr_ann
        lines.append(
            f"- geometric growth crosses zero at 2.00x full Kelly: "
            f"~{zero_growth_vol_ann * 100.0:.2f}%/yr vol for this sample"
        )
    else:
        lines.append(
            "- geometric growth crosses zero at 2.00x full Kelly: "
            "undefined here (non-positive measured drift)"
        )
    lines.append("")

    return RiskBudget(
        report_md="\n".join(lines),
        requires_override=override,
        implied_kelly_multiple=implied,
        ruin_table=ruin_table,
    )

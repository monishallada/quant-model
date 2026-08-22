"""Statistical validation: Sharpe inference (PSR/DSR), bootstrap CIs,
block-bootstrap paths, and the CSCV probability of backtest overfitting.

Every tunable threshold comes from :class:`edge.core.config.ValidationConfig`
at the call site; the only literals in this module are mathematical constants
fixed by the cited papers.

References:
    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2) — the Probabilistic Sharpe Ratio.
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
    Portfolio Management 40(5).
    Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of
    Backtest Overfitting", Journal of Computational Finance 20(4) — CSCV.
    Politis & Romano (1992) — the circular block bootstrap.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from edge.core.config import ValidationConfig

#: Canonical CSCV partition count (Bailey et al. 2015 use S = 16).
CANONICAL_PBO_BLOCKS: int = 16
#: Hard cap on evaluated IS/OOS splits: C(16, 8) = 12870.
PBO_SPLIT_CAP: int = math.comb(CANONICAL_PBO_BLOCKS, CANONICAL_PBO_BLOCKS // 2)


def _as_1d(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Coerce to a 1-D float array; reject empty or multi-dimensional input."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    return arr


def sharpe(returns: npt.ArrayLike, periods_per_year: float) -> float:
    """Annualized Sharpe ratio: ``mean(returns)/std(returns, ddof=1) * sqrt(periods_per_year)``.

    Pass ``periods_per_year=1`` for the per-period (non-annualized) ratio
    that :func:`probabilistic_sharpe` and :func:`deflated_sharpe` expect.

    Raises:
        ValueError: fewer than two observations, zero variance, or
            non-positive ``periods_per_year``.
    """
    arr = _as_1d(returns, "returns")
    if arr.size < 2:
        raise ValueError("sharpe requires at least two observations")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        raise ValueError("sharpe undefined for zero-variance returns")
    return float(np.mean(arr)) / sd * math.sqrt(periods_per_year)


def probabilistic_sharpe(sr_hat: float, sr0: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: confidence that the true SR exceeds ``sr0``.

    PSR = Phi( (sr_hat - sr0) * sqrt(n - 1)
               / sqrt(1 - skew*sr_hat + ((kurt - 1)/4) * sr_hat**2) )

    per Bailey & Lopez de Prado (2012). ``sr_hat`` and ``sr0`` are
    per-period (non-annualized) Sharpe ratios estimated over ``n``
    observations; ``kurt`` is NON-excess kurtosis (normal returns => 3).

    Raises:
        ValueError: ``n < 2`` or a non-positive variance-adjustment term
            (extreme skew/kurtosis relative to ``sr_hat``).
    """
    if n < 2:
        raise ValueError("probabilistic_sharpe requires n >= 2 observations")
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denom_sq <= 0.0:
        raise ValueError(f"non-positive SR variance adjustment ({denom_sq}); check skew/kurt")
    stat = (sr_hat - sr0) * math.sqrt(n - 1.0) / math.sqrt(denom_sq)
    return float(norm.cdf(stat))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` zero-skill trials.

    E[max SR] = sqrt(var_trials) * ((1 - g) * z(1 - 1/n_trials)
                                    + g * z(1 - 1/(n_trials * e)))

    where ``g`` is the Euler-Mascheroni constant and ``z`` the inverse
    standard normal CDF (Bailey & Lopez de Prado 2014, eq. for E[max_n]).
    ``var_trials`` is the cross-trial variance of estimated Sharpe ratios.
    ``n_trials = 1`` returns 0.0 (no selection took place).

    Raises:
        ValueError: ``n_trials < 1`` or ``var_trials < 0``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_trials < 0.0:
        raise ValueError("var_trials must be non-negative")
    if n_trials == 1:
        return 0.0
    g = float(np.euler_gamma)
    z_hi = float(norm.ppf(1.0 - 1.0 / n_trials))
    z_lo = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(var_trials) * ((1.0 - g) * z_hi + g * z_lo)


def deflated_sharpe(
    sr_hat: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    var_trials: float,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected best of ``n_trials`` noise trials.

    DSR = PSR(sr_hat, sr0) with ``sr0 = expected_max_sharpe(n_trials,
    var_trials)`` (Bailey & Lopez de Prado 2014, "The Deflated Sharpe
    Ratio"). ``sr_hat`` is the selected strategy's per-period SR over
    ``n_obs`` observations; ``skew``/``kurt`` (non-excess) describe its
    returns. ``n_trials`` MUST be the caller's full trial count from the
    TRIALS registry — it is never re-estimated here — and ``var_trials``
    the variance of SR estimates across those trials.

    Raises:
        ValueError: propagated from :func:`expected_max_sharpe` and
            :func:`probabilistic_sharpe` on invalid inputs.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return probabilistic_sharpe(sr_hat, sr0, n_obs, skew, kurt)


def _mean_stat(values: npt.NDArray[np.float64]) -> float:
    """Default bootstrap statistic: the sample mean."""
    return float(np.mean(values))


def bootstrap_ci(
    values: npt.ArrayLike,
    stat: Callable[[npt.NDArray[np.float64]], float] = _mean_stat,
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``stat`` over ``values``.

    Draws ``resamples`` iid resamples (with replacement, same length as
    ``values``), evaluates ``stat`` on each, and returns the percentile
    interval — (2.5th, 97.5th) at the default 95% confidence. ``resamples``
    is config-driven: pass ``ValidationConfig.bootstrap_resamples``.

    Returns:
        ``(lo, hi)`` bounds of the interval.

    Raises:
        ValueError: empty ``values``, ``resamples < 1``, or ``confidence``
            outside (0, 1).
    """
    arr = _as_1d(values, "values")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    stats = np.array([stat(arr[row]) for row in idx], dtype=np.float64)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(stats, [tail * 100.0, (1.0 - tail) * 100.0])
    return float(lo), float(hi)


def block_bootstrap(
    daily_returns: npt.ArrayLike,
    block_len: int,
    n_paths: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Circular block bootstrap of a daily return series (Politis & Romano 1992).

    Concatenates blocks of ``block_len`` consecutive observations whose start
    indices are drawn uniformly with wrap-around, preserving autocorrelation
    up to the block length. Suitable for resampling equity paths in ruin
    analysis.

    Returns:
        ``(n_paths, T)`` array; each row is one resampled path of the same
        length ``T`` as the input.

    Raises:
        ValueError: fewer than two observations, ``block_len`` outside
            ``[1, T]``, or ``n_paths < 1``.
    """
    arr = _as_1d(daily_returns, "daily_returns")
    t = arr.size
    if t < 2:
        raise ValueError("block_bootstrap requires at least two observations")
    if not 1 <= block_len <= t:
        raise ValueError(f"block_len must be in [1, {t}], got {block_len}")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(t / block_len)
    starts = rng.integers(0, t, size=(n_paths, n_blocks))
    # (n_paths, n_blocks, block_len) circular indices, flattened and truncated to T.
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % t
    paths = arr[idx.reshape(n_paths, n_blocks * block_len)[:, :t]]
    return cast(npt.NDArray[np.float64], paths)


def min_trade_gate(n_oos: int, config: ValidationConfig) -> bool:
    """True iff the out-of-sample trade count clears ``config.min_oos_trades``.

    A strategy with fewer OOS trades than the configured floor has not
    earned statistical standing, whatever its point estimates say.
    """
    return n_oos >= config.min_oos_trades


def _column_sharpes(
    sums: npt.NDArray[np.float64],
    sumsqs: npt.NDArray[np.float64],
    count: int,
) -> npt.NDArray[np.float64]:
    """Per-column per-period Sharpe (mean/std, ddof=1) from running sums."""
    mean = sums / count
    var = (sumsqs - count * mean**2) / (count - 1)
    if np.any(var <= 0.0):
        raise ValueError("pbo_cscv requires positive return variance in every subsample")
    return mean / np.sqrt(var)


def pbo_cscv(returns_matrix: npt.ArrayLike, n_blocks: int = CANONICAL_PBO_BLOCKS) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2015).

    The ``(T, M)`` per-period return matrix (one column per strategy trial)
    is partitioned row-wise into ``n_blocks`` contiguous blocks (trailing
    remainder rows dropped). For each of the C(n_blocks, n_blocks/2)
    combinatorially symmetric IS/OOS splits — capped at C(16, 8) = 12870 —
    the in-sample-best strategy by Sharpe is ranked out-of-sample among the
    M strategies; with relative rank ``r_rel = r / (M + 1)`` in (0, 1), the
    logit is ``lambda = ln(r_rel / (1 - r_rel))``. PBO is the fraction of
    splits whose selected strategy ranks below the OOS median
    (``lambda < 0``).

    Args:
        returns_matrix: ``(T, M)`` returns, ``M >= 2`` strategies.
        n_blocks: even partition count >= 2; 16 is canonical, 8 is the
            cheap setting for tests.

    Raises:
        ValueError: non-2-D input, ``M < 2``, odd or out-of-range
            ``n_blocks``, or a zero-variance subsample.
    """
    x = np.asarray(returns_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D (T, M), got shape {x.shape}")
    t, m = x.shape
    if m < 2:
        raise ValueError("pbo_cscv requires at least two strategies")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be an even integer >= 2, got {n_blocks}")
    block_size = t // n_blocks
    if block_size < 2:
        raise ValueError(f"need at least {2 * n_blocks} rows for n_blocks={n_blocks}, got {t}")

    blocks = x[: block_size * n_blocks].reshape(n_blocks, block_size, m)
    block_sums = blocks.sum(axis=1)
    block_sumsqs = (blocks**2).sum(axis=1)
    total_sums = block_sums.sum(axis=0)
    total_sumsqs = block_sumsqs.sum(axis=0)
    half = n_blocks // 2
    n_half = half * block_size

    below_median = 0
    n_splits = 0
    combos = itertools.islice(itertools.combinations(range(n_blocks), half), PBO_SPLIT_CAP)
    for combo in combos:
        idx = list(combo)
        is_sums = block_sums[idx].sum(axis=0)
        is_sumsqs = block_sumsqs[idx].sum(axis=0)
        is_sr = _column_sharpes(is_sums, is_sumsqs, n_half)
        oos_sr = _column_sharpes(total_sums - is_sums, total_sumsqs - is_sumsqs, n_half)
        best = int(np.argmax(is_sr))
        rank = int(np.sum(oos_sr <= oos_sr[best]))  # ascending rank in 1..M
        r_rel = rank / (m + 1)
        if math.log(r_rel / (1.0 - r_rel)) < 0.0:
            below_median += 1
        n_splits += 1
    return below_median / n_splits

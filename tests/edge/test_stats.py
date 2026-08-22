"""Validation stats on synthetic data only: Sharpe/PSR/DSR (with calibration
against pure noise), bootstrap CIs, circular block bootstrap, CSCV PBO.
No timestamps, no market data — nothing here can touch the lockbox window."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import kurtosis, skew

from edge.core.config import ValidationConfig
from edge.validation.stats import (
    block_bootstrap,
    bootstrap_ci,
    deflated_sharpe,
    expected_max_sharpe,
    min_trade_gate,
    pbo_cscv,
    probabilistic_sharpe,
    sharpe,
)

# Synthetic config for the gates; values chosen for the tests, not production.
CFG = ValidationConfig(min_oos_trades=300, pbo_max=0.5, bootstrap_resamples=2000)


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------


def test_sharpe_known_value_and_annualization() -> None:
    # mean 0.02, std(ddof=1) 0.02 -> per-period SR exactly 1.
    returns = [0.02, 0.0, 0.04]
    assert sharpe(returns, periods_per_year=1) == pytest.approx(1.0)
    assert sharpe(returns, periods_per_year=252) == pytest.approx(math.sqrt(252))
    zero_mean = [0.01, -0.01, 0.01, -0.01]
    assert sharpe(zero_mean, periods_per_year=252) == pytest.approx(0.0)


def test_sharpe_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        sharpe([0.01], periods_per_year=252)  # one observation
    with pytest.raises(ValueError):
        sharpe([0.01, 0.01, 0.01], periods_per_year=252)  # zero variance
    with pytest.raises(ValueError):
        sharpe([0.01, 0.02], periods_per_year=0)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe (Bailey & LdP 2012)
# ---------------------------------------------------------------------------


def test_psr_matches_hand_computed() -> None:
    # (d) sr_hat=0.1, sr0=0, n=101, skew=0, kurt=3 (non-excess normal):
    #   stat = 0.1*sqrt(100)/sqrt(1 + (2/4)*0.01) = 1/sqrt(1.005) = 0.9975093361
    #   PSR  = Phi(0.9975093361) = 0.8407413278
    assert probabilistic_sharpe(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(
        0.8407413278013518, abs=1e-6
    )
    # sr_hat=0.2, sr0=0.05, n=253, skew=-0.5, kurt=4:
    #   denom^2 = 1 - (-0.5)(0.2) + (3/4)(0.04) = 1.13
    #   stat = 0.15*sqrt(252)/sqrt(1.13) = 2.2400221238; PSR = Phi(stat)
    assert probabilistic_sharpe(0.2, 0.05, 253, -0.5, 4.0) == pytest.approx(
        0.9874552566901826, abs=1e-6
    )


def test_psr_at_benchmark_is_half() -> None:
    assert probabilistic_sharpe(0.0, 0.0, 100, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe(0.1, 0.0, 1, 0.0, 3.0)  # n < 2
    with pytest.raises(ValueError):
        probabilistic_sharpe(0.5, 0.0, 100, 30.0, 3.0)  # negative variance term


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & LdP 2014)
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_hand_value_and_limits() -> None:
    # n_trials=100, var_trials=0.01, g=Euler-Mascheroni:
    #   sr0 = 0.1*((1-g)*z(0.99) + g*z(1 - 1/(100e))) = 0.2530602893
    assert expected_max_sharpe(100, 0.01) == pytest.approx(0.2530602893201685, abs=1e-6)
    assert expected_max_sharpe(1, 0.01) == 0.0  # no selection took place
    assert expected_max_sharpe(1000, 0.0) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.01)
    with pytest.raises(ValueError):
        expected_max_sharpe(100, -0.01)


def test_dsr_single_trial_reduces_to_psr() -> None:
    dsr = deflated_sharpe(0.15, 252, 0.0, 3.0, n_trials=1, var_trials=0.01)
    assert dsr == pytest.approx(probabilistic_sharpe(0.15, 0.0, 252, 0.0, 3.0))


def test_dsr_decreases_with_trials() -> None:
    dsrs = [
        deflated_sharpe(0.15, 252, 0.0, 3.0, n_trials=n, var_trials=0.01)
        for n in (2, 10, 100, 1000)
    ]
    assert dsrs == sorted(dsrs, reverse=True)
    assert dsrs[0] > dsrs[-1]


def test_dsr_best_of_noise_not_significant() -> None:
    # (c) Best of 1000 pure-noise strategies over T=252: the DSR must NOT be
    # significant (< 0.95) in at least 8 of 10 seeds — selection bias is
    # exactly what the deflation corrects for.
    n_trials, t = 1000, 252
    not_significant = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        rets = rng.standard_normal((t, n_trials)) * 0.01
        srs = rets.mean(axis=0) / rets.std(axis=0, ddof=1)
        best = int(np.argmax(srs))
        dsr = deflated_sharpe(
            sr_hat=float(srs[best]),
            n_obs=t,
            skew=float(skew(rets[:, best])),
            kurt=float(kurtosis(rets[:, best], fisher=False)),
            n_trials=n_trials,
            var_trials=float(np.var(srs, ddof=1)),
        )
        not_significant += dsr < 0.95
    assert not_significant >= 8


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_covers_true_mean() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(0.5, 0.1, size=400)
    lo, hi = bootstrap_ci(values, resamples=CFG.bootstrap_resamples, seed=7)
    assert lo < 0.5 < hi
    assert hi - lo < 0.05  # se ~ 0.005 -> 95% CI width ~ 0.02


def test_bootstrap_ci_deterministic_per_seed() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 1.0, size=100)
    first = bootstrap_ci(values, resamples=CFG.bootstrap_resamples, seed=7)
    again = bootstrap_ci(values, resamples=CFG.bootstrap_resamples, seed=7)
    other = bootstrap_ci(values, resamples=CFG.bootstrap_resamples, seed=8)
    assert first == again
    assert first != other


def test_bootstrap_ci_custom_stat() -> None:
    def med(a: npt.NDArray[np.float64]) -> float:
        return float(np.median(a))

    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 1.0, size=200)
    lo, hi = bootstrap_ci(values, med, resamples=CFG.bootstrap_resamples, seed=7)
    assert values.min() <= lo < hi <= values.max()


def test_bootstrap_ci_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([], resamples=CFG.bootstrap_resamples, seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], resamples=0, seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], resamples=10, seed=0, confidence=1.0)


# ---------------------------------------------------------------------------
# Circular block bootstrap
# ---------------------------------------------------------------------------


def _ar1(t: int, phi: float, seed: int) -> npt.NDArray[np.float64]:
    """Synthetic AR(1) series with persistence phi."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(t)
    x = np.empty(t)
    x[0] = eps[0]
    for i in range(1, t):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _mean_lag1_autocorr(paths: npt.NDArray[np.float64]) -> float:
    return float(np.mean([np.corrcoef(p[:-1], p[1:])[0, 1] for p in paths]))


def test_block_bootstrap_shape_values_determinism() -> None:
    x = _ar1(t=500, phi=0.5, seed=3)
    paths = block_bootstrap(x, block_len=20, n_paths=50, seed=11)
    assert paths.shape == (50, 500)
    assert np.isin(paths, x).all()  # circular blocks only recombine originals
    assert np.array_equal(paths, block_bootstrap(x, block_len=20, n_paths=50, seed=11))


def test_block_bootstrap_preserves_autocorrelation() -> None:
    x = _ar1(t=2000, phi=0.9, seed=3)
    blocky = block_bootstrap(x, block_len=50, n_paths=100, seed=11)
    iid = block_bootstrap(x, block_len=1, n_paths=100, seed=11)
    assert _mean_lag1_autocorr(blocky) > 0.6  # AR(1) persistence survives
    assert abs(_mean_lag1_autocorr(iid)) < 0.15  # single-point blocks destroy it


def test_block_bootstrap_rejects_invalid() -> None:
    x = _ar1(t=100, phi=0.5, seed=3)
    with pytest.raises(ValueError):
        block_bootstrap(x, block_len=0, n_paths=10, seed=0)
    with pytest.raises(ValueError):
        block_bootstrap(x, block_len=101, n_paths=10, seed=0)
    with pytest.raises(ValueError):
        block_bootstrap(x, block_len=10, n_paths=0, seed=0)


# ---------------------------------------------------------------------------
# Minimum-trade gate
# ---------------------------------------------------------------------------


def test_min_trade_gate_config_driven() -> None:
    assert not min_trade_gate(CFG.min_oos_trades - 1, CFG)
    assert min_trade_gate(CFG.min_oos_trades, CFG)
    assert min_trade_gate(CFG.min_oos_trades + 1, CFG)
    looser = ValidationConfig(min_oos_trades=5, pbo_max=0.5, bootstrap_resamples=100)
    assert min_trade_gate(5, looser)  # threshold follows the config, not a constant


# ---------------------------------------------------------------------------
# PBO via CSCV (Bailey, Borwein, LdP & Zhu 2015)
# ---------------------------------------------------------------------------


def test_pbo_noise_calibration() -> None:
    # (a) 200 iid N(0,1) strategies, T=500: no strategy has skill, so the
    # IS-best should rank below the OOS median about half the time.
    rng = np.random.default_rng(0)
    pbo = pbo_cscv(rng.standard_normal((500, 200)), n_blocks=16)
    assert 0.35 <= pbo <= 0.65


def test_pbo_planted_skill() -> None:
    # (b) One strategy with true annualized SR 2.0 among 50 pure-noise
    # strategies: selection consistently finds real skill -> low PBO.
    rng = np.random.default_rng(0)
    t, sigma = 2000, 0.01
    noise = rng.standard_normal((t, 50)) * sigma
    planted = rng.standard_normal(t) * sigma + (2.0 / math.sqrt(252)) * sigma
    pbo = pbo_cscv(np.column_stack([planted, noise]), n_blocks=8)
    assert 0.0 <= pbo < 0.2


def test_pbo_rejects_invalid() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pbo_cscv(rng.standard_normal(100), n_blocks=8)  # 1-D input
    with pytest.raises(ValueError):
        pbo_cscv(rng.standard_normal((100, 1)), n_blocks=8)  # single strategy
    with pytest.raises(ValueError):
        pbo_cscv(rng.standard_normal((100, 5)), n_blocks=7)  # odd block count
    with pytest.raises(ValueError):
        pbo_cscv(rng.standard_normal((8, 5)), n_blocks=8)  # too few rows per block
    constant = np.column_stack([np.ones(160), rng.standard_normal(160)])
    with pytest.raises(ValueError):
        pbo_cscv(constant, n_blocks=8)  # zero-variance subsample

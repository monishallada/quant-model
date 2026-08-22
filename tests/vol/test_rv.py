"""RV engine vs hand-computed / analytical references."""

import math

import numpy as np
import pandas as pd
import pytest

from catalyst.vol import rv


def _gbm_minutes(sigma_annual: float, n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    per_min = sigma_annual / math.sqrt(rv.ANNUALIZER)
    return pd.Series(rng.normal(0.0, per_min, n))


class TestEstimators:
    def test_rv_recovers_known_sigma(self):
        r = _gbm_minutes(0.20, 390 * 250)
        ann = rv.annualized_vol(rv.realized_variance(r), len(r))
        assert ann == pytest.approx(0.20, rel=0.02)

    def test_bipower_matches_rv_without_jumps(self):
        r = _gbm_minutes(0.30, 390 * 250)
        bv_ann = rv.annualized_vol(rv.bipower_variation(r), len(r))
        assert bv_ann == pytest.approx(0.30, rel=0.02)

    def test_bipower_robust_to_a_jump_rv_is_not(self):
        r = _gbm_minutes(0.20, 390).to_numpy().copy()
        r[200] += 0.02                      # a 2% one-minute jump
        s = pd.Series(r)
        rv_ann = rv.annualized_vol(rv.realized_variance(s), len(s))
        bv_ann = rv.annualized_vol(rv.bipower_variation(s), len(s))
        # hand math: jump adds 4e-4 to a ~1.6e-4 session variance ->
        # RV_ann ~ sqrt(5.6e-4*252) ~ 0.37; BV barely moves
        assert rv_ann > 0.30
        assert bv_ann < 0.25
        assert rv.jump_ratio(s) > 0.5

    def test_empty_and_nan_safety(self):
        assert rv.realized_variance(pd.Series(dtype=float)) != rv.realized_variance(pd.Series(dtype=float))
        assert rv.annualized_vol(float("nan"), 100) != rv.annualized_vol(float("nan"), 100)


class TestDiurnal:
    def test_flat_days_give_flat_curve(self):
        days = [_gbm_minutes(0.2, 390, seed=i) for i in range(30)]
        curve = rv.fit_diurnal_curve(days)
        assert curve.mean() == pytest.approx(1.0)
        assert curve.std() < 0.35           # roughly flat for iid days

    def test_u_shape_is_recovered(self):
        rng = np.random.default_rng(3)
        days = []
        scale = np.concatenate([np.full(60, 2.0), np.full(270, 0.7), np.full(60, 2.0)])
        per_min = 0.2 / math.sqrt(rv.ANNUALIZER)
        for _ in range(40):
            days.append(pd.Series(rng.normal(0, per_min, 390) * np.sqrt(scale)))
        curve = rv.fit_diurnal_curve(days)
        assert curve[0] > 1.3 and curve[-1] > 1.3     # open/close elevated
        assert curve[len(curve) // 2] < 0.8            # lunch compressed

    def test_remaining_share_monotone(self):
        curve = np.ones(26)
        shares = [rv.remaining_variance_share(curve, m) for m in (0, 100, 200, 389)]
        assert shares[0] == pytest.approx(1.0, abs=0.01)
        assert all(a > b for a, b in zip(shares, shares[1:]))
        assert shares[-1] < 0.02


class TestForecast:
    def test_forecast_recovers_sigma_on_calm_day(self):
        r = _gbm_minutes(0.20, 200)
        curve = np.ones(26)
        per_session_var = (0.20 ** 2) / 252
        f = rv.forecast_horizon_vol(r, per_session_var, curve, 200, 60)
        assert f == pytest.approx(0.20, rel=0.30)

    def test_forecast_nan_without_any_information(self):
        f = rv.forecast_horizon_vol(pd.Series(dtype=float), float("nan"),
                                    np.ones(26), 100, 30)
        assert f != f

"""Conditional-distribution engine: recovery, fallback, and calibration."""

import numpy as np
import pytest

from catalyst.vol.dist import (
    ConditionalQuantiles,
    StateSpec,
    classify_trend,
    classify_vol,
    coverage_test,
    fit,
    tod_bucket,
    QUANTILES,
)


def _synthetic(n=200_000, seed=1, skew_state=None):
    """Scaled returns ~ N(0,1) except one state cell shifted +0.5."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, n)
    tod = rng.integers(0, 6, n)
    vol = rng.integers(0, 3, n)
    trend = rng.integers(0, 3, n)
    if skew_state is not None:
        sel = (tod == skew_state[0]) & (vol == skew_state[1]) & (trend == skew_state[2])
        r[sel] += 0.5
    return r, tod, vol, trend


class TestFit:
    def test_recovers_standard_normal_quantiles(self):
        r, tod, vol, trend = _synthetic()
        f = fit(r, tod, vol, trend)
        q = f.lookup(2, 1, 1)
        # N(0,1): q05=-1.645, q50=0, q95=+1.645
        assert q[0] == pytest.approx(-1.645, abs=0.08)
        assert q[3] == pytest.approx(0.0, abs=0.05)
        assert q[6] == pytest.approx(1.645, abs=0.08)

    def test_conditional_shift_is_detected(self):
        r, tod, vol, trend = _synthetic(skew_state=(3, 2, 0))
        f = fit(r, tod, vol, trend)
        shifted = f.lookup(3, 2, 0)
        normal = f.lookup(3, 1, 1)
        assert shifted[3] - normal[3] == pytest.approx(0.5, abs=0.08)

    def test_sparse_cell_falls_back_to_marginal(self):
        r, tod, vol, trend = _synthetic(n=5_000)   # cells ~90 obs < 300
        f = fit(r, tod, vol, trend)
        assert (2, 0, 0) not in f.table
        assert f.lookup(2, 0, 0) is not None       # marginal serves

    def test_no_data_returns_none_never_fabricates(self):
        f = fit(np.array([]), np.array([]), np.array([]), np.array([]))
        assert f.lookup(0, 0, 0) is None


class TestClassifiers:
    def test_vol_states(self):
        spec = StateSpec()
        assert classify_vol(0.10, 0.20, spec) == 0
        assert classify_vol(0.20, 0.20, spec) == 1
        assert classify_vol(0.40, 0.20, spec) == 2
        assert classify_vol(float("nan"), 0.2, spec) == 1   # unknown -> middle

    def test_trend_states(self):
        spec = StateSpec()
        assert classify_trend(-0.01, 0.01, spec) == 0
        assert classify_trend(0.0, 0.01, spec) == 1
        assert classify_trend(0.01, 0.01, spec) == 2

    def test_tod_buckets_span_session(self):
        spec = StateSpec()
        assert tod_bucket(0, spec) == 0
        assert tod_bucket(389, spec) == 5


class TestCalibration:
    def test_in_sample_coverage_is_tight(self):
        r, tod, vol, trend = _synthetic()
        f = fit(r, tod, vol, trend)
        r2, tod2, vol2, trend2 = _synthetic(seed=2)
        cov = coverage_test(f, r2, tod2, vol2, trend2)
        assert cov["max_miscoverage"] < 0.02

    def test_miscalibration_is_visible(self):
        r, tod, vol, trend = _synthetic()
        f = fit(r, tod, vol, trend)
        # score against a WIDER world (sigma 1.5): coverage must break
        r2, tod2, vol2, trend2 = _synthetic(seed=3)
        cov = coverage_test(f, r2 * 1.5, tod2, vol2, trend2)
        assert cov["max_miscoverage"] > 0.05

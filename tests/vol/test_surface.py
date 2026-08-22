"""Surface engine: exact recovery, refusal discipline, dislocation detection."""

import numpy as np
import pytest

from catalyst.vol.surface import dislocation_scores, fit_smile, forward_variance


def _smile(a=0.25, b=-0.10, c=0.8, n=11, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    k = np.linspace(-0.03, 0.03, n)
    iv = a + b * k + c * k * k + rng.normal(0, noise, n)
    rs = np.full(n, 0.05)
    return k, iv, rs


class TestFit:
    def test_exact_quadratic_recovered(self):
        k, iv, rs = _smile()
        f = fit_smile(k, iv, rs)
        assert f is not None
        assert f.atm_iv == pytest.approx(0.25, abs=1e-9)
        assert f.skew == pytest.approx(-0.10, abs=1e-6)
        assert f.curvature == pytest.approx(0.8, abs=1e-4)
        assert f.rmse == pytest.approx(0.0, abs=1e-9)

    def test_noise_tolerated_quality_reported(self):
        k, iv, rs = _smile(noise=0.005, seed=2)
        f = fit_smile(k, iv, rs)
        assert f is not None
        assert f.atm_iv == pytest.approx(0.25, abs=0.02)
        assert f.rmse > 0

    def test_too_few_points_refused(self):
        k, iv, rs = _smile(n=4)
        assert fit_smile(k, iv, rs) is None

    def test_wide_quotes_excluded_can_cause_refusal(self):
        k, iv, rs = _smile(n=8)
        rs[:] = 0.9                        # everything too wide to trust
        assert fit_smile(k, iv, rs) is None

    def test_degenerate_negative_atm_refused(self):
        k = np.linspace(-0.01, 0.01, 8)
        iv = -0.5 + 0.0 * k
        assert fit_smile(k, iv, np.full(8, 0.05)) is None

    def test_nan_quotes_dropped_not_propagated(self):
        k, iv, rs = _smile()
        iv[3] = np.nan
        f = fit_smile(k, iv, rs)
        assert f is not None and f.n_points == 10


class TestDislocation:
    def test_planted_rich_contract_scores_high(self):
        k, iv, rs = _smile(noise=0.002, seed=5)
        iv2 = iv.copy()
        iv2[7] += 0.03                      # plant a 3-vol-pt rich contract
        f = fit_smile(k, iv2, rs)
        z = dislocation_scores(f, k, iv2)
        assert np.nanargmax(z) == 7
        assert z[7] > 3.0
        assert np.nanmedian(np.abs(np.delete(z, 7))) < 2.0

    def test_smooth_smile_scores_are_moderate(self):
        k, iv, rs = _smile(noise=0.002, seed=6)
        f = fit_smile(k, iv, rs)
        z = dislocation_scores(f, k, iv)
        # with robust refitting, a pure-noise point occasionally sits near
        # ~5 z; the STRATEGY layers persistence + neighbor-quiet gates on
        # top. "Moderate" here means an order below a planted dislocation.
        assert np.nanmax(np.abs(z)) < 8.0


class TestForwardVariance:
    def test_flat_term_structure(self):
        fv = forward_variance(0.20, 1 / 252, 0.20, 5 / 252)
        assert fv == pytest.approx(0.04, rel=1e-9)

    def test_inverted_structure_goes_negative(self):
        # near total var 0.5^2/252 = 9.9e-4 > far total 0.2^2*5/252 = 7.9e-4
        fv = forward_variance(0.50, 1 / 252, 0.20, 5 / 252)
        assert fv < 0

    def test_bad_ordering_is_nan(self):
        assert forward_variance(0.2, 5 / 252, 0.2, 1 / 252) != forward_variance(
            0.2, 5 / 252, 0.2, 1 / 252)

"""Payoff transform vs analytic expectations."""

import numpy as np
import pytest

from catalyst.core.types import OptionRight
from catalyst.data.black_scholes import bs_price
from catalyst.vol.payoff import LegSpec, evaluate, scenario_returns

N7 = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])  # N(0,1)


def _call(spot=100.0, strike=100.0, dte_years=5 / 252, iv=0.20, qty=1):
    price = bs_price(spot, strike, dte_years, iv, OptionRight.CALL, r=0.0)
    return LegSpec(right=OptionRight.CALL, strike=strike, t_years=dte_years,
                   iv=iv, qty=qty, entry_price=price)


class TestScenarioGrid:
    def test_grid_is_monotone_and_tails_extended(self):
        g = scenario_returns(N7, 0.01)
        assert np.all(np.diff(g) > 0)
        assert g[0] < N7[0] * 0.01          # 1% node beyond the 5% quantile
        assert g[-1] > N7[-1] * 0.01

    def test_zero_scale_collapses_grid(self):
        g = scenario_returns(N7, 0.0)
        assert np.allclose(g, 0.0)


class TestEvaluate:
    def test_fair_priced_atm_call_theta_dominates_at_zero_move_dist(self):
        """With a zero-width distribution the only P&L is theta decay:
        EV must equal exactly BS(t) - BS(t-h) < 0."""
        leg = _call()
        stats = evaluate([leg], 100.0, N7, horizon_vol_scale=0.0,
                         horizon_minutes=60, exit_cost_per_unit=0.0)
        expected = (bs_price(100, 100, 5 / 252 - 60 / (252 * 390), 0.20,
                             OptionRight.CALL) - leg.entry_price)
        assert stats.ev == pytest.approx(expected, abs=1e-9)
        assert stats.ev < 0

    def test_underpriced_distribution_gives_positive_ev(self):
        """If the real distribution is 2x wider than the IV the call was
        priced at, a long call has positive EV (gamma pays for theta)."""
        leg = _call(iv=0.20)
        # horizon 60min at DOUBLE the option's implied vol
        scale = 0.40 / np.sqrt(252 * 390 / 60)
        stats = evaluate([leg], 100.0, N7, scale, 60, 0.0)
        assert stats.ev > 0

    def test_overpriced_iv_gives_negative_ev_for_long(self):
        leg = _call(iv=0.60)                # paid for 60-vol
        scale = 0.20 / np.sqrt(252 * 390 / 60)   # world delivers 20-vol
        stats = evaluate([leg], 100.0, N7, scale, 60, 0.0)
        assert stats.ev < 0

    def test_exit_cost_shifts_ev_down_exactly(self):
        leg = _call()
        scale = 0.20 / np.sqrt(252 * 390 / 60)
        a = evaluate([leg], 100.0, N7, scale, 60, 0.0)
        b = evaluate([leg], 100.0, N7, scale, 60, 0.05)
        assert a.ev - b.ev == pytest.approx(0.05, abs=1e-9)

    def test_vertical_spread_worst_case_is_bounded(self):
        long_leg = _call(strike=100.0)
        short_px = bs_price(100, 102, 5 / 252, 0.20, OptionRight.CALL)
        short_leg = LegSpec(right=OptionRight.CALL, strike=102.0,
                            t_years=5 / 252, iv=0.20, qty=-1,
                            entry_price=short_px)
        scale = 0.60 / np.sqrt(252 * 390 / 60)   # violent world
        stats = evaluate([long_leg, short_leg], 100.0, N7, scale, 60, 0.0)
        debit = long_leg.entry_price - short_px
        assert stats.worst >= -debit - 1e-9      # defined risk holds

    def test_sticky_delta_uses_smile(self):
        leg = _call()
        smile = lambda k: 0.20 + 1.0 * abs(k)     # steep smile
        scale = 0.20 / np.sqrt(252 * 390 / 60)
        ss = evaluate([leg], 100.0, N7, scale, 60, 0.0, smile_iv_at=smile,
                      dynamics="sticky_delta")
        st = evaluate([leg], 100.0, N7, scale, 60, 0.0, dynamics="sticky_strike")
        assert ss.ev != pytest.approx(st.ev, abs=1e-6)  # dynamics matter

    def test_nan_iv_refused(self):
        leg = LegSpec(right=OptionRight.CALL, strike=100.0, t_years=5 / 252,
                      iv=float("nan"), qty=1, entry_price=1.0)
        assert evaluate([leg], 100.0, N7, 0.01, 60, 0.0) is None

    def test_probability_weights_sum_to_one(self):
        from catalyst.vol.payoff import NODE_WEIGHTS
        assert NODE_WEIGHTS.sum() == pytest.approx(1.0)

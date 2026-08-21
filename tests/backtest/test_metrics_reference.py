"""Metrics verified against hand-computed reference values (audit D-024/D-094/
D-095/D-096/D-097/D-098/D-190/D-025). Every function that feeds a verdict has
an asserted output here."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from catalyst.backtest import metrics as m
from catalyst.backtest.montecarlo import probability_of_ruin
from catalyst.core.types import Direction, TradeRecord


def _trade(pnl: float) -> TradeRecord:
    t = datetime(2024, 6, 3, 15, 45)
    return TradeRecord(position_id="p", engine="e", catalyst_ref="r",
                       underlying="SPY", direction=Direction.LONG,
                       entry_time=t, exit_time=t, entry_price=1.0,
                       exit_price=1.0, qty=1, pnl=pnl, exit_reason="test",
                       max_qty=1)


def _curve(values, freq="D"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


class TestProfitFactorAndFriends:
    def test_profit_factor_hand_computed(self):
        trades = [_trade(100), _trade(50), _trade(-75)]
        assert m.profit_factor(trades) == pytest.approx(150 / 75)

    def test_profit_factor_no_losses_is_inf(self):
        assert m.profit_factor([_trade(10)]) == float("inf")

    def test_expected_value(self):
        trades = [_trade(100), _trade(-40)]
        assert m.expected_value(trades) == pytest.approx(30.0)

    def test_concentration_positive_book(self):
        trades = [_trade(x) for x in (50, 30, 20, 10, -10)]
        c = m.concentration(trades, 3)
        assert c["share"] == pytest.approx(100 / 100)

    def test_concentration_negative_total_is_none(self):
        trades = [_trade(-50), _trade(10)]
        assert m.concentration(trades, 3)["share"] is None


class TestTradeStats:
    def test_scratch_trades_are_neither_wins_nor_losses(self):
        """Audit D-190: zero-pnl trades diluted avg_loss."""
        stats = m.trade_stats([_trade(100), _trade(0), _trade(-50)])
        assert stats["avg_loss"] == pytest.approx(-50.0)   # NOT -25
        assert stats["win_rate"] == pytest.approx(1 / 3)
        assert stats["win_loss_ratio"] == pytest.approx(2.0)


class TestSortinoSharpe:
    def test_sortino_standard_denominator(self):
        """Audit D-094: r = [+1%, +1%, -2%]; downside dev over ALL N:
        sqrt((0 + 0 + 4e-4)/3)."""
        r = pd.Series([0.01, 0.01, -0.02])
        expect_dd = np.sqrt(4e-4 / 3)
        expect = r.mean() / expect_dd * np.sqrt(252)
        assert m.sortino(r) == pytest.approx(float(expect))

    def test_sharpe_rf_reduces_ratio(self):
        r = pd.Series([0.001] * 100)
        r.iloc[0] = 0.002   # non-zero std
        assert m.sharpe(r, rf_annual=0.05) < m.sharpe(r, rf_annual=0.0)


class TestBankruptcyIsNotFlat:
    def test_avg_monthly_return_of_wipeout_is_minus_one(self):
        """Audit D-096: 100k -> -20k reported 0.0%/mo (flat)."""
        eq = _curve(np.linspace(100_000, -20_000, 300))
        assert m.avg_monthly_return(eq) == -1.0

    def test_headline_cagr_of_wipeout(self):
        eq = _curve(np.linspace(100_000, -20_000, 300))
        assert m.headline(eq)["cagr"] == -1.0            # D-097: not NaN

    def test_calmar_of_wipeout_is_finite(self):
        eq = _curve(np.linspace(100_000, -20_000, 300))
        assert np.isfinite(m.calmar(eq))                  # D-095


class TestRuinRefusesCorruptInput:
    def test_nan_returns_raise_not_zero(self):
        r = pd.Series([0.01, np.nan, -0.02])
        with pytest.raises(ValueError, match="non-finite"):
            probability_of_ruin(r, _mc_cfg())

    def test_clean_input_still_works(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.01, 250))
        p = probability_of_ruin(r, _mc_cfg())
        assert 0.0 <= p <= 1.0


def _mc_cfg():
    from catalyst.core.config import load_config
    return load_config("backtest").backtest.monte_carlo

"""short_vrp evaluate() must be exercised END TO END through its passing
branch — the wiring smoke tests never reach it, which let two classes of bug
ship: silent no-op gates (scale/tenor/date-convention mismatches) and a
NameError in the proposal construction that only fired on the first passing
event of a multi-hour backtest.

The synthetic fixture is built so every gate PASSES at the spec defaults; the
gate tests then flip one input at a time and assert the specific counter.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from catalyst.core.config import load_config
from catalyst.core.types import (
    Catalyst,
    CatalystType,
    Direction,
    Greeks,
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    SignalResult,
)
from catalyst.core.tradingcal import next_trading_day, previous_trading_day
from catalyst.strategies.archive.short_vrp.strategy import ShortVRPStrategy

SPOT = 100.0
EVENT = date(2024, 5, 15)          # a Wednesday; reaction session = same day
ENTRY = previous_trading_day(EVENT)
EXPIRY = next_trading_day(next_trading_day(EVENT))


class _StubIVRank:
    """Provider double: rank on the PROVIDER'S 0..100 scale, iv30 as level."""

    def __init__(self, rank: float = 95.0, iv30: float = 0.60):
        self.rank, self.iv30 = rank, iv30

    def iv_rank(self, symbol, day):
        return self.rank

    def atm_iv30(self, symbol, day):
        return self.iv30


def _contract(strike: float, right: OptionRight, delta: float,
              bid: float, ask: float) -> OptionContract:
    return OptionContract(
        key=OptionKey(underlying="XYZ", expiry=EXPIRY, right=right, strike=strike),
        bid=bid, ask=ask, volume=500, open_interest=1000,
        greeks=Greeks(delta=delta, theta=-0.05, vega=0.10, iv=0.60),
    )


def _chain() -> OptionChain:
    # Deltas chosen so nearest to 0.16 sits at 110C / 90P with wings listed
    # one strike further out; tight quotes so the 8% liquidity gate passes.
    contracts = [
        _contract(105, OptionRight.CALL, 0.35, 2.96, 3.04),
        _contract(110, OptionRight.CALL, 0.16, 1.47, 1.53),   # short call
        _contract(115, OptionRight.CALL, 0.07, 0.68, 0.72),   # long call wing
        _contract(100, OptionRight.CALL, 0.50, 4.95, 5.05),
        _contract(95, OptionRight.PUT, -0.35, 2.96, 3.04),
        _contract(90, OptionRight.PUT, -0.16, 1.47, 1.53),    # short put
        _contract(85, OptionRight.PUT, -0.07, 0.68, 0.72),    # long put wing
        _contract(100, OptionRight.PUT, -0.50, 4.95, 5.05),
    ]
    return OptionChain(underlying="XYZ", underlying_price=SPOT,
                       timestamp=datetime.combine(ENTRY, datetime.min.time()),
                       contracts=contracts)


def _closes() -> pd.Series:
    """~2y of closes ending before ENTRY, low realized vol (~10%)."""
    days = pd.bdate_range(end=pd.Timestamp(ENTRY) - pd.Timedelta(days=1),
                          periods=500)
    rng = np.random.default_rng(7)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.10 / 16, len(days))))
    return pd.Series(px, index=days)


def _strategy(**over) -> ShortVRPStrategy:
    cfg = load_config("backtest")
    if over:
        cfg = cfg.model_copy(deep=True,
                             update={"short_vrp": cfg.short_vrp.model_copy(update=over)})
    closes = _closes()
    # 8 prior "reaction dates" with |move| ~3%: implied (straddle/spot) must
    # comfortably beat 1.1x this for the default gate to pass.
    react = list(pd.bdate_range(end=pd.Timestamp(ENTRY) - pd.Timedelta(days=20),
                                periods=8).date)
    px = closes.copy()
    for i, r in enumerate(react):
        loc = px.index.searchsorted(pd.Timestamp(r))
        if loc < len(px):
            px.iloc[loc:] = px.iloc[loc:] * (1 + 0.03 * (-1) ** i)
    return ShortVRPStrategy(cfg, _StubIVRank(), {"XYZ": px}, {"XYZ": react})


def _catalyst() -> Catalyst:
    return Catalyst(symbol="XYZ", type=CatalystType.EARNINGS,
                    when=datetime.combine(EVENT, datetime.min.time().replace(hour=8)),
                    source="test")


def _asof() -> datetime:
    return datetime.combine(ENTRY, datetime.min.time().replace(hour=15, minute=45))


class TestEvaluatePassingPath:
    def test_full_pass_produces_condor(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # sidecar writes under cwd
        s = _strategy()
        pt = s.evaluate(_catalyst(), _chain(), SignalResult(direction=Direction.NEUTRAL, confidence=0.5),
                        _asof())
        assert pt is not None, f"gates blocked the passing fixture: {s.gates}"
        assert len(pt.legs) == 4
        # worst-case credit: shorts at bid (1.47*2) minus longs at ask (0.72*2)
        assert pt.unit_cost == pytest.approx(-1.50)
        assert pt.unit_max_loss == pytest.approx(5.0 - 1.50)
        assert pt.exit_rules.hard_exit_date == next_trading_day(EVENT)
        assert s.gates["passed"] == 1

    def test_provider_scale_is_0_to_100(self, tmp_path, monkeypatch):
        """iv_rank_min is 0..1 in config; the provider speaks 0..100. A rank
        of 55 (55th percentile) must FAIL the 0.80 gate — the bug this guards
        against is 55 > 0.80 passing on raw comparison."""
        monkeypatch.chdir(tmp_path)
        s = _strategy()
        s._iv_rank = _StubIVRank(rank=55.0)
        assert s.evaluate(_catalyst(), _chain(),
                          SignalResult(direction=Direction.NEUTRAL, confidence=0.5), _asof()) is None
        assert s.gates["iv_rank"] == 1

    def test_iv_rv_gate_uses_30d_tenor(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = _strategy()
        s._iv_rank = _StubIVRank(rank=95.0, iv30=0.05)  # 5% IV vs ~10% RV
        assert s.evaluate(_catalyst(), _chain(),
                          SignalResult(direction=Direction.NEUTRAL, confidence=0.5), _asof()) is None
        assert s.gates["iv_rv"] == 1

    def test_hist_move_gate_blocks_cheap_premium(self, tmp_path, monkeypatch):
        """Historical reaction moves ~3%/event; a chain whose straddle implies
        far less than 1.1x that must be rejected by gate 3."""
        monkeypatch.chdir(tmp_path)
        s = _strategy()
        cheap = _chain().model_copy(deep=True)
        for c in cheap.contracts:
            object.__setattr__(c, "bid", c.bid * 0.15)
            object.__setattr__(c, "ask", c.ask * 0.15)
        assert s.evaluate(_catalyst(), cheap,
                          SignalResult(direction=Direction.NEUTRAL, confidence=0.5), _asof()) is None
        assert s.gates["implied_vs_hist"] == 1

    def test_wrong_entry_day_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = _strategy()
        early = _asof() - timedelta(days=7)
        assert s.evaluate(_catalyst(), _chain(),
                          SignalResult(direction=Direction.NEUTRAL, confidence=0.5), early) is None
        assert s.gates["not_entry_day"] == 1


class TestIVProviderPointInTime:
    """The IV series is EOD (~15:59) data; decisions happen at 15:45. The
    entry day's own value is therefore FUTURE data and must be invisible by
    default (v14 verification found this leak in production)."""

    def _provider(self):
        from catalyst.data.iv_history import IVRankProvider
        p = IVRankProvider.__new__(IVRankProvider)
        idx = list(pd.bdate_range("2024-01-01", periods=300).date)
        vals = np.linspace(0.10, 0.40, 300)
        vals[-1] = 9.99                      # entry-day spike = the future
        p._series = {"XYZ": pd.Series(vals, index=idx)}
        p._lookback = 252
        return p, idx[-1]

    def test_entry_day_value_is_invisible_by_default(self):
        p, day = self._provider()
        assert p.atm_iv30("XYZ", day) != pytest.approx(9.99)
        assert p.iv_rank("XYZ", day) < 100.0

    def test_include_day_opt_in_sees_it(self):
        p, day = self._provider()
        assert p.atm_iv30("XYZ", day, include_day=True) == pytest.approx(9.99)
        # the spike outranks everything else in the window (the window holds
        # the current value itself, so the ceiling is (n-1)/n, not 100)
        assert p.iv_rank("XYZ", day, include_day=True) > 99.0


def test_archived_strategy_stays_runnable():
    """The repo promises archived campaigns can be re-run. Archived metadata is
    discovered from disk without importing the module, so load_strategy must
    import it on demand — otherwise every archived strategy is 'unknown'."""
    from catalyst.strategies.registry import load_strategy, registry
    meta = registry().get("short_vrp")
    assert meta is not None and meta.status == "archived"
    s = load_strategy("short_vrp", load_config("backtest"))
    assert s.name == "short_vrp"

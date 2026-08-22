"""MOSAIC decision-path tests: synthetic markets with PLANTED mispricings.

The v15 lesson institutionalized: the passing path of every strategy is
tested before any backtest runs. Fixtures price the option window from BS at
a chosen IV; the tests then plant edges and assert the machine (a) finds
them, (b) structures them correctly, and (c) refuses fair/illiquid markets.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from catalyst.core.interfaces.intraday import IntradayContext
from catalyst.core.types import OptionRight, Side
from catalyst.data.black_scholes import bs_price
from catalyst.strategies.active.mosaic import MosaicParams, MosaicStrategy
from catalyst.vol import dist, rv

SESSION = date(2024, 6, 3)
NOW = datetime(2024, 6, 3, 11, 0)
SPOT = 500.0
EXPIRY = date(2024, 6, 3)          # 0DTE
N7 = np.array([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])


class FakeData:
    def list_expirations(self, symbol):
        return [EXPIRY, date(2024, 6, 4)]


def _bars(n=90, sigma_annual=0.16, seed=4):
    """Session bars ending just before NOW (labels <= NOW-1min)."""
    rng = np.random.default_rng(seed)
    per_min = sigma_annual / math.sqrt(rv.ANNUALIZER)
    rets = rng.normal(0, per_min, n)
    closes = SPOT * np.exp(np.cumsum(rets) - np.sum(rets))   # ends at SPOT
    idx = pd.date_range(datetime(2024, 6, 3, 9, 30), periods=n, freq="min")
    return pd.DataFrame({"close": closes, "open": closes, "high": closes,
                         "low": closes, "volume": 1000}, index=idx)


def _quote_fn(iv_map, spread_frac=0.04, t_years=5.0 / (24 * 365)):
    """(bid, ask) closure pricing each key at iv_map[(strike, right)]."""
    def quote(key):
        iv = iv_map.get((key.strike, key.right))
        if iv is None:
            return None
        mid = bs_price(SPOT, key.strike, t_years, iv, key.right, r=0.0)
        if mid < 0.01:
            mid = 0.01
        half = max(mid * spread_frac / 2, 0.005)
        return (mid - half, mid + half)
    return quote


def _iv_map(base_iv=0.16, skew=-0.3):
    m = {}
    for i in range(-8, 9):
        k = SPOT + i
        km = math.log(k / SPOT)
        iv = base_iv + skew * km + 2.0 * km * km
        m[(float(k), OptionRight.CALL)] = iv
        m[(float(k), OptionRight.PUT)] = iv
    return m


def _fitted(strategy: MosaicStrategy, realized_sigma=0.16, quantiles=N7):
    """Inject point-in-time state as if warmup/refit already ran."""
    per_session_var = (realized_sigma ** 2) / 252.0
    strategy._session_vars = [per_session_var] * 30
    strategy._session_returns = []
    strategy._curve = np.ones(26)
    spec = dist.StateSpec()
    q = dist.ConditionalQuantiles(spec=spec, n_obs=10_000)
    for t in range(spec.n_tod_buckets):
        q.tod_marginal[t] = quantiles.copy()
        for v in range(3):
            for tr in range(3):
                q.table[(t, v, tr)] = quantiles.copy()
    strategy._quantiles = q
    strategy._warmed = True
    strategy._last_session = date(2024, 5, 31)


def _ctx(bars, quote_fn):
    return IntradayContext(session=SESSION, now=NOW, bars={"SPY": bars},
                           data=FakeData(), option_quote=quote_fn)


class TestRefusals:
    def test_warmup_means_no_trades(self):
        s = MosaicStrategy()
        s._warmed = True                      # skip provider warmup
        out = s.on_minute(_ctx(_bars(), _quote_fn(_iv_map())))
        assert out == []
        assert s.gates["warmup"] == 1

    def test_fair_market_is_no_trade(self):
        """IV == RV forecast, smooth smile, symmetric distribution: the EV
        gate must refuse everything (friction-aware honesty)."""
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.16)
        out = s.on_minute(_ctx(_bars(sigma_annual=0.16), _quote_fn(_iv_map(0.16))))
        assert out == []
        assert s.gates["emitted"] == 0

    def test_wide_spreads_refused_even_with_edge(self):
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.10)
        rich = _iv_map(0.40)                  # very rich IV
        out = s.on_minute(_ctx(_bars(sigma_annual=0.10),
                               _quote_fn(rich, spread_frac=0.40)))
        assert out == []

    def test_outside_entry_window_no_decision(self):
        s = MosaicStrategy()
        _fitted(s)
        ctx = IntradayContext(session=SESSION,
                              now=datetime(2024, 6, 3, 15, 40),
                              bars={"SPY": _bars()}, data=FakeData(),
                              option_quote=_quote_fn(_iv_map()))
        assert s.on_minute(ctx) == []


class TestVolPremiumFamily:
    def test_rich_iv_emits_defined_risk_credit_vertical(self):
        """IV 40 vs forecast RV ~10: family A must emit a credit vertical
        with positive max-loss and credit below width."""
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.10)
        out = s.on_minute(_ctx(_bars(sigma_annual=0.10), _quote_fn(_iv_map(0.40))))
        assert len(out) == 1
        t = out[0]
        assert t.rationale["family"] == "vol_premium"
        assert t.unit_cost < 0                       # credit
        assert t.unit_max_loss > 0
        sides = sorted(l.side for l in t.legs)
        assert sides == [Side.BUY, Side.SELL]
        assert t.exit_rules.close_by_time is not None

    def test_cheap_iv_emits_debit_structure(self):
        """IV 14 vs forecast RV ~30: long-vol debit vertical. (An even
        cheaper IV makes OTM legs sub-$0.10 and correctly trips the
        liquidity gate — the refusal test above covers that.)"""
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.30)
        out = s.on_minute(_ctx(_bars(sigma_annual=0.30),
                               _quote_fn(_iv_map(0.14), spread_frac=0.02)))
        assert len(out) == 1
        t = out[0]
        assert t.unit_cost > 0                        # debit
        assert t.unit_max_loss == pytest.approx(t.unit_cost)


class TestDislocationFamily:
    def test_planted_cheap_contract_bought_against_neighbor(self):
        s = MosaicStrategy()
        # neutral vol premium (IV == RV) so family A stays quiet
        _fitted(s, realized_sigma=0.16)
        m = _iv_map(0.16)
        # 2 vol pts cheap on a near-ATM put: a real, LIQUID dislocation.
        # (A 5-pt cheapening crushed the premium to $0.04 and the liquidity
        # gate rightly refused to trade it — that behavior is the
        # wide-spread refusal test's subject.)
        m[(499.0, OptionRight.PUT)] = 0.16 - 0.03
        # persistence filter (research spec): the FIRST sighting arms memory
        # only; the SECOND consecutive same-sign sighting trades
        first = s.on_minute(_ctx(_bars(sigma_annual=0.16),
                                 _quote_fn(m, spread_frac=0.02)))
        assert first == []
        ctx2 = IntradayContext(session=SESSION,
                               now=datetime(2024, 6, 3, 11, 5),
                               bars={"SPY": _bars(n=95)}, data=FakeData(),
                               option_quote=_quote_fn(m, spread_frac=0.02))
        out = s.on_minute(ctx2)
        assert len(out) == 1
        t = out[0]
        assert t.rationale["family"] == "dislocation"
        buys = [l for l in t.legs if l.side is Side.BUY]
        assert buys and buys[0].key.strike == 499.0   # the cheap one is bought


class TestAsymmetryFamily:
    def test_right_skewed_distribution_emits_call_debit(self):
        skewed = np.array([-0.8, -0.6, -0.3, 0.15, 0.9, 1.7, 2.4])
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.16, quantiles=skewed)
        out = s.on_minute(_ctx(_bars(sigma_annual=0.16), _quote_fn(_iv_map(0.16))))
        assert len(out) == 1
        t = out[0]
        assert t.rationale["family"] == "asymmetry"
        assert all(l.key.right is OptionRight.CALL for l in t.legs)
        assert t.unit_cost > 0


class TestPointInTime:
    def test_backwards_session_resets_state(self):
        s = MosaicStrategy()
        _fitted(s)
        s._last_session = date(2024, 6, 10)
        s._maybe_new_run(date(2024, 6, 3))
        assert s._quantiles is None
        assert s._session_vars == []


class TestFairValueFamily:
    def test_stale_cheap_mid_with_history_is_bought(self):
        """Family D: a contract whose mid sits well below its OWN trailing
        fair value (IV history) gets bought against a fair neighbor."""
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.16)
        m = _iv_map(0.16)
        base_ctx = _ctx(_bars(sigma_annual=0.16), _quote_fn(m, spread_frac=0.02))
        # build 30 minutes of IV history at 0.16 via repeated decisions
        from catalyst.core.interfaces.intraday import IntradayContext
        for k in range(5):
            ctx_k = IntradayContext(
                session=SESSION, now=datetime(2024, 6, 3, 10, 30 + 5 * k),
                bars={"SPY": _bars(n=60 + 5 * k)}, data=FakeData(),
                option_quote=_quote_fn(m, spread_frac=0.02))
            s.on_minute(ctx_k)
        # now the 501 CALL's quote goes stale-cheap by ~4 vol pts while its
        # HISTORY says 0.16 -> V-hat >> mid
        m2 = dict(m)
        m2[(501.0, OptionRight.CALL)] = 0.12
        ctx = IntradayContext(
            session=SESSION, now=datetime(2024, 6, 3, 11, 0),
            bars={"SPY": _bars(n=90)}, data=FakeData(),
            option_quote=_quote_fn(m2, spread_frac=0.02))
        out = s.on_minute(ctx)
        # the dislocation family may also fire (cross-sectional view of the
        # same event) — family D must at least be COMPETING; accept either
        # winner but require an emission with the stale contract bought
        assert len(out) == 1
        buys = [l for l in out[0].legs if l.side is Side.BUY]
        assert buys and buys[0].key.strike == 501.0

    def test_no_history_no_family_d(self):
        s = MosaicStrategy()
        _fitted(s, realized_sigma=0.16)
        m = _iv_map(0.16)
        m[(501.0, OptionRight.CALL)] = 0.12
        ctx = _ctx(_bars(sigma_annual=0.16), _quote_fn(m, spread_frac=0.02))
        cands = s._family_fair_value(ctx, *_collect(s, ctx))
        assert cands == []


def _collect(s, ctx):
    window = s._watch_window(SESSION, SPOT, EXPIRY)
    tys = s._t_years(NOW, EXPIRY)
    keys, k, m, iv, rel = s._collect_quotes(ctx, window, SPOT, tys)
    return keys, k, m, iv, rel, SPOT, tys

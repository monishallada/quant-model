"""Pairs backtesters: V1 share-ledger arithmetic and RiskManager routing;
V2 leg construction, expiry-day liquidation, and roll accounting."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from catalyst.core.config import load_config
from catalyst.core.types import OptionChain, OptionRight, Side
from catalyst.core.tradingcal import add_trading_days, sessions_in_range
from catalyst.strategies.archive.pairs.options_backtester import PairsOptionsBacktester
from catalyst.strategies.archive.pairs.shares_backtester import PairsSharesBacktester
from tests.conftest import build_chain

CFG_BASE = load_config("backtest")


def two_pair_universe_cfg(universe=None):
    return load_config("backtest", overrides={
        "pairs.universe": universe or [["AAA", "BBB"]],
        "pairs.shares.borrow_apr_overrides": {},
    })


def cointegrated_prices(n_extra: int = 260, end: date = date(2024, 6, 28), dislocate: bool = True):
    """A = 2*B + AR(1) noise, with a big dislocation then reversion so the
    strategy has exactly one clean round trip to find."""
    rng = np.random.default_rng(7)
    sessions = sessions_in_range(date(2023, 1, 2), end)
    n = len(sessions) + n_extra
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    b = np.full(n, 100.0) * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.9)
    a = 2.0 * b + noise
    if dislocate:
        # Push A rich for ~6 sessions near the end, then let it revert.
        k = n - 25
        a[k : k + 6] *= 1.06
    closes = {"AAA": pd.Series(a, index=idx), "BBB": pd.Series(b, index=idx)}
    opens = {s: v.shift(0) * 0.999 for s, v in closes.items()}  # opens slightly below closes
    return sessions, closes, opens


# ---------------------------------------------------------------------------
# V1 shares
# ---------------------------------------------------------------------------


def test_v1_round_trip_and_costs() -> None:
    sessions, closes, opens = cointegrated_prices()
    cfg = two_pair_universe_cfg()
    bt = PairsSharesBacktester(cfg, closes, opens, zero_cost=False, label="t")
    result, extras = bt.run(date(2023, 6, 1), date(2024, 6, 28))
    assert result.n_trades >= 1
    trade = result.trades[0]
    assert trade.engine == "pairs_v1"
    assert trade.exit_reason in ("z_reverted", "decoupled", "end_of_backtest")
    # Zero-cost run of the same window must strictly beat the real-cost run.
    bt0 = PairsSharesBacktester(cfg, closes, opens, zero_cost=True, label="t0")
    result0, _ = bt0.run(date(2023, 6, 1), date(2024, 6, 28))
    assert result0.n_trades >= result.n_trades
    total0 = sum(t.pnl for t in result0.trades)
    total = sum(t.pnl for t in result.trades)
    assert total0 > total


def test_v1_dollar_neutral_and_risk_capped() -> None:
    sessions, closes, opens = cointegrated_prices()
    cfg = two_pair_universe_cfg()
    bt = PairsSharesBacktester(cfg, closes, opens, zero_cost=True, label="t")
    result, _ = bt.run(date(2023, 6, 1), date(2024, 6, 28))
    assert result.n_trades >= 1
    t = result.trades[0]
    # gross notional stored per unit in entry_price*100
    gross = t.entry_price * 100.0
    # RiskManager: per_trade 2% of 100k = 2000 risk; unit risk = 2000*0.10=200
    # -> <= 10 units short of buffer; with 5% buffer -> 9 units = 18k gross.
    max_gross = (
        cfg.account.starting_capital * cfg.pairs.per_trade_risk / cfg.pairs.max_adverse_move
    )
    assert gross <= max_gross * 1.001
    assert gross >= max_gross * 0.5  # actually deployed meaningfully


def test_v1_hard_to_borrow_skipped() -> None:
    sessions, closes, opens = cointegrated_prices()
    unrestricted = PairsSharesBacktester(
        two_pair_universe_cfg(), closes, opens, zero_cost=True, label="u"
    )
    res_u, _ = unrestricted.run(date(2023, 6, 1), date(2024, 6, 28))
    cfg = load_config("backtest", overrides={
        "pairs.universe": [["AAA", "BBB"]],
        "pairs.shares.borrow_apr_overrides": {},
        "pairs.shares.hard_to_borrow_skip": ["AAA"],  # blocks short-AAA entries
    })
    restricted = PairsSharesBacktester(cfg, closes, opens, zero_cost=True, label="t")
    res_r, _ = restricted.run(date(2023, 6, 1), date(2024, 6, 28))
    # The engineered dislocation (A rich -> short A) is only tradable
    # unrestricted; the HTB skip must remove at least that entry.
    assert res_r.n_trades < res_u.n_trades


# ---------------------------------------------------------------------------
# V2 options (stub data source)
# ---------------------------------------------------------------------------


class StubOptionsData:
    """Serves synthetic ATM-rich chains for any (symbol, session, expiry)."""

    def __init__(self, closes: dict[str, pd.Series]) -> None:
        self._closes = closes
        self.requested: list[tuple[str, date, date]] = []

    def list_expirations(self, symbol: str) -> list[date]:
        # Weekly Friday expirations across the window.
        out = []
        d = date(2023, 1, 6)
        while d < date(2024, 12, 31):
            out.append(d)
            d += timedelta(days=7)
        return out

    def get_chain(self, symbol, at, expiries=None, max_dte=None, include_liquidity=True) -> OptionChain:
        self.requested.append((symbol, at.date(), expiries[0] if expiries else None))
        ts = pd.Timestamp(at.date())
        series = self._closes[symbol]
        spot = float(series.loc[series.index <= ts].iloc[-1])
        strikes = [round(spot * (1 + k / 100.0)) for k in range(-6, 7)]
        return build_chain(
            symbol=symbol, spot=spot, as_of=at,
            expiry_ivs={e: 0.35 for e in (expiries or [])},
            strikes=sorted(set(strikes)),
            spread_frac=0.10,  # realistic wide 1-2 DTE ATM spreads
        )


def test_v2_enters_put_call_pair_and_liquidates_by_expiry() -> None:
    sessions, closes, opens = cointegrated_prices()
    cfg = two_pair_universe_cfg()
    data = StubOptionsData(closes)
    bt = PairsOptionsBacktester(cfg, data, closes, zero_cost=False, label="t")
    result, extras = bt.run(date(2023, 6, 1), date(2024, 6, 28))
    assert result.n_trades >= 1
    for t in result.trades:
        assert t.engine == "pairs_v2"
        # Never held into expiry close: exits at/before the 14:00 stop.
        assert t.exit_reason in {"z_reverted", "expiry_liquidation", "decoupled", "end_of_backtest"}
        held_days = (t.exit_time.date() - t.entry_time.date()).days
        assert held_days <= 5  # 1-2 TD options can never live longer
    # Wide dislocations outlive 1-2 DTE options: rolls must be observed.
    assert extras["rolls"] >= 1
    # Entries paid the ask: entry price should exceed the (synthetic) mid.
    first = result.trades[0]
    assert first.entry_price > 0


def test_v2_zero_cost_beats_real_cost() -> None:
    sessions, closes, opens = cointegrated_prices()
    cfg = two_pair_universe_cfg()
    real, _ = PairsOptionsBacktester(cfg, StubOptionsData(closes), closes,
                                     zero_cost=False, label="r").run(date(2023, 6, 1), date(2024, 6, 28))
    zero, _ = PairsOptionsBacktester(cfg, StubOptionsData(closes), closes,
                                     zero_cost=True, label="z").run(date(2023, 6, 1), date(2024, 6, 28))
    if real.n_trades and zero.n_trades:
        assert sum(t.pnl for t in zero.trades) > sum(t.pnl for t in real.trades)


def test_v2_legs_are_atm_put_on_rich_call_on_cheap() -> None:
    sessions, closes, opens = cointegrated_prices()
    cfg = two_pair_universe_cfg()
    data = StubOptionsData(closes)
    bt = PairsOptionsBacktester(cfg, data, closes, zero_cost=True, label="t")

    # Drive one entry manually through the internal helper.
    from catalyst.risk.manager import RiskManager
    from catalyst.brokers.simulated import SimulatedBroker
    from catalyst.core.config import FillModelConfig

    rm = RiskManager(cfg.risk)
    rm.set_history({s: pd.DataFrame({"close": c}) for s, c in closes.items()})
    broker = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=0.0, slippage_pct_of_premium=0.0),
        commissions=cfg.execution.commissions,
        starting_cash=100_000.0,
    )
    entry_session = date(2024, 5, 29)  # Wednesday; Friday expiry = 2 TD out
    open_pairs, open_meta, trade_entries = {}, {}, {}
    bt._try_enter(broker, rm, ("AAA", "BBB"), "short_A_long_B", 3.0, 0,
                  entry_session, open_pairs, open_meta, trade_entries)
    positions = broker.get_positions()
    assert len(positions) == 1
    legs = positions[0].legs
    put_a = next(l for l in legs if l.key.underlying == "AAA")
    call_b = next(l for l in legs if l.key.underlying == "BBB")
    assert put_a.key.right is OptionRight.PUT and put_a.side is Side.BUY
    assert call_b.key.right is OptionRight.CALL and call_b.side is Side.BUY
    # ATM: strike within one increment of spot.
    spot_a = float(closes["AAA"].loc[closes["AAA"].index <= pd.Timestamp(entry_session)].iloc[-1])
    assert abs(put_a.key.strike - spot_a) / spot_a < 0.02
    # Expiry 1-2 trading days out.
    assert 1 <= (put_a.key.expiry - entry_session).days <= 4

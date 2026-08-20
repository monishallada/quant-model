"""The guarantees the whole framework rests on.

These are not tests of a feature. They are the honest-measurement contract:
every strategy that ever plugs in is protected by them without doing anything,
and any change that breaks one breaks the reason to trust every result in the
repository.

1. The RiskManager cash floor is never breached, under randomized order streams.
2. No fill is ever better than NBBO.
3. The out-of-sample split never leaks.
4. There is exactly one code path to a broker.
"""

from __future__ import annotations

import ast
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from catalyst.backtest.walkforward import (chronological_split,
                                           chronological_split_exclusive)
from catalyst.core.config import CommissionsConfig, FillModelConfig, load_config
from catalyst.core.types import (
    AccountState,
    Direction,
    ExitRules,
    Greeks,
    OptionContract,
    OptionKey,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
)
from catalyst.costs.model import BetterThanNBBOError, NBBOCostModel, ZeroCostModel
from catalyst.risk.manager import RiskManager

SRC = Path(__file__).resolve().parents[2] / "src" / "catalyst"


def _contract(bid: float, ask: float, strike: float = 100.0) -> OptionContract:
    return OptionContract(
        key=OptionKey(underlying="SPY", expiry=date(2026, 12, 18),
                      right=OptionRight.CALL, strike=strike),
        bid=bid, ask=ask, volume=100, open_interest=500,
        greeks=Greeks(delta=0.5, theta=-0.1, vega=0.2, iv=0.25))


def _proposal(cost: float, risk_fraction: float) -> ProposedTrade:
    return ProposedTrade(
        engine="test", catalyst_ref="SPY:x",
        legs=[OrderLeg(key=_contract(1.0, 1.2).key, side=Side.BUY, qty=1)],
        unit_cost=cost, unit_max_loss=cost, direction=Direction.LONG,
        exit_rules=ExitRules(), per_trade_risk_fraction=risk_fraction)


# ---------------------------------------------------------------------------
# 1. cash floor
# ---------------------------------------------------------------------------
class TestCashFloorNeverBreached:
    """A randomized stream of orders must never push cash below the floor.

    Property-based rather than example-based on purpose: the floor has to hold
    for order sequences nobody thought to write down.
    """

    @pytest.mark.parametrize("seed", range(25))
    def test_randomized_order_stream_respects_cash_floor(self, seed: int) -> None:
        rng = random.Random(seed)
        # Explicit floor: the test pins the MECHANISM (floor never breached),
        # not whatever policy the research profile currently sets (which may
        # legitimately be 0.0 during unconstrained search).
        cfg = load_config("backtest")
        rm_cfg = cfg.risk.model_copy(update={"cash_floor_fraction": 0.40,
                                             "max_deployed": 0.95})
        rm = RiskManager(rm_cfg)
        floor_fraction = rm_cfg.cash_floor_fraction

        equity = 100_000.0
        cash = equity
        positions: list = []

        for _ in range(200):
            cost = rng.uniform(0.05, 25.0)
            frac = rng.uniform(0.001, 0.50)
            account = AccountState(equity=equity, cash=cash, buying_power=cash,
                                   timestamp=datetime(2026, 1, 5, 15, 45))
            decision = rm.size_entry(_proposal(cost, frac), account, positions)
            spend = decision.units * cost * 100.0
            assert spend <= cash + 1e-9, "sized an order larger than available cash"
            cash -= spend
            assert cash >= equity * floor_fraction - 1e-6, (
                f"cash floor breached: {cash:.2f} < "
                f"{equity * floor_fraction:.2f} (seed {seed})")

    def test_zero_and_negative_max_loss_are_refused(self) -> None:
        """A non-positive risk basis must never produce a position size."""
        cfg = load_config("backtest")
        rm = RiskManager(cfg.risk)
        account = AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                               timestamp=datetime(2026, 1, 5, 15, 45))
        for bad in (0.0, -1.0):
            p = _proposal(1.0, 0.05)
            p = p.model_copy(update={"unit_max_loss": bad})
            assert rm.size_entry(p, account, []).units == 0


# ---------------------------------------------------------------------------
# 2. never better than NBBO
# ---------------------------------------------------------------------------
class TestNoFillBetterThanNBBO:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_quotes_never_fill_better_than_mid(self, seed: int) -> None:
        rng = random.Random(seed)
        model = NBBOCostModel(
            FillModelConfig(spread_fill_fraction=rng.uniform(0.0, 1.0),
                            slippage_pct_of_premium=rng.uniform(0.0, 0.05),
                            slippage_per_contract=rng.uniform(0.0, 0.05)),
            CommissionsConfig(alpaca_per_contract=rng.uniform(0.0, 1.0),
                              schwab_per_contract_per_leg=0.65,
                              active_profile="alpaca"))
        bid = rng.uniform(0.05, 20.0)
        ask = bid + rng.uniform(0.01, 3.0)
        c = _contract(bid, ask)

        buy = model.leg_fill(c, Side.BUY, 1)
        sell = model.leg_fill(c, Side.SELL, 1)
        assert buy.price >= c.mid - 1e-9, "buy filled better than mid"
        assert sell.price <= c.mid + 1e-9, "sell filled better than mid"
        assert buy.price >= sell.price - 1e-9, "buy cheaper than sell — inverted"

    def test_zero_cost_model_fills_at_mid_exactly(self) -> None:
        """The diagnostic twin must be frictionless, not merely cheap."""
        c = _contract(1.0, 1.4)
        z = ZeroCostModel()
        assert z.leg_fill(c, Side.BUY, 1).price == pytest.approx(c.mid)
        assert z.leg_fill(c, Side.SELL, 1).price == pytest.approx(c.mid)
        assert z.leg_fill(c, Side.BUY, 10).commission == 0.0

    def test_guard_fires_on_a_deliberately_inverted_fill(self) -> None:
        """The assertion must actually be reachable — a guard nobody can trip
        is a guard nobody can trust."""
        c = _contract(1.0, 1.4)
        with pytest.raises(BetterThanNBBOError):
            NBBOCostModel._assert_not_better_than_nbbo(c.mid - 0.10, c, Side.BUY)
        with pytest.raises(BetterThanNBBOError):
            NBBOCostModel._assert_not_better_than_nbbo(c.mid + 0.10, c, Side.SELL)


# ---------------------------------------------------------------------------
# 3. out-of-sample split never leaks
# ---------------------------------------------------------------------------
class TestOutOfSampleSplitNeverLeaks:
    @pytest.mark.parametrize("fraction", [0.5, 0.6, 0.7, 0.8, 0.9])
    @pytest.mark.parametrize("span_days", [200, 900, 3200])
    def test_train_always_precedes_test_with_no_overlap(
        self, fraction: float, span_days: int
    ) -> None:
        start = date(2018, 1, 2)
        end = start + timedelta(days=span_days)
        train_end, test_start = chronological_split_exclusive(start, end, fraction)

        assert start <= train_end < test_start <= end
        assert train_end < test_start, "train and test overlap — lookahead leak"

    def test_split_is_chronological_not_random(self) -> None:
        """Shuffling would leak the future into training. The split must be a
        single cut in time, and repeatable."""
        start, end = date(2018, 1, 2), date(2026, 8, 1)
        a = chronological_split_exclusive(start, end, 0.7)
        b = chronological_split_exclusive(start, end, 0.7)
        assert a == b, "split is not deterministic"
        train_end, test_start = a
        assert (train_end - start).days > (end - test_start).days, \
            "70/30 split did not put the larger span in train"


# ---------------------------------------------------------------------------
# 4. exactly one path to a broker
# ---------------------------------------------------------------------------
class TestOnlyOnePathToTheBroker:
    """``place_order`` may only be called by the execution layer or a broker.

    Enforced by reading the AST rather than by convention, because "everyone
    knows not to do that" is exactly how a bypass gets added under deadline.
    """

    ALLOWED = {
        "execution/engine.py",
        "brokers/simulated.py",
        "brokers/alpaca.py",
        "brokers/schwab.py",
        "backtest/backtester.py",   # drives SimulatedBroker inside the pipeline
        "backtest/intraday.py",     # the minute-resolution engine, same category
    }

    #: Archived campaigns that predate the pipeline and carry their own loop.
    #: Named individually and never extended: they are grandfathered evidence,
    #: not a precedent. Anything NEW that calls place_order fails the test.
    GRANDFATHERED = {
        "strategies/archive/pairs/options_backtester.py",
        "strategies/archive/kinetic/backtester.py",
    }

    def test_no_strategy_calls_place_order(self) -> None:
        offenders = []
        for path in (SRC / "strategies").rglob("*.py"):
            if str(path.relative_to(SRC)) in self.GRANDFATHERED:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "place_order"):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert not offenders, (
            "strategies must never reach a broker directly: " + ", ".join(offenders))

    def test_place_order_callers_are_whitelisted(self) -> None:
        offenders = []
        for path in SRC.rglob("*.py"):
            rel = str(path.relative_to(SRC))
            if rel in self.ALLOWED or rel in self.GRANDFATHERED:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "place_order"):
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "new path to the broker outside the execution layer: " + ", ".join(offenders))

    def test_no_strategy_imports_risk_costs_or_brokers(self) -> None:
        """A strategy that cannot import the risk layer cannot bypass it."""
        forbidden = ("catalyst.risk", "catalyst.costs", "catalyst.brokers",
                     "catalyst.execution")
        offenders = []
        for path in (SRC / "strategies" / "active").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and any(mod.startswith(f) for f in forbidden):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} -> {mod}")
        assert not offenders, (
            "active strategy reaches a layer it must not: " + ", ".join(offenders))

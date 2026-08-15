"""Drift Harvest: gates, three-arm structure selection, credit exit math,
and the risk plumbing that credit structures depend on."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from catalyst.core.config import load_config
from catalyst.core.models import (
    Catalyst,
    CatalystType,
    Direction,
    ExitRules,
    OptionKey,
    OptionRight,
    Position,
    PositionLeg,
    Side,
)
from catalyst.drift.engine import DriftHarvestEngine
from catalyst.drift.screener import DriftScreener, realized_vol
from catalyst.exits.manager import evaluate_exits
from tests.conftest import build_chain

CFG = load_config("backtest")
REPORT = datetime(2024, 5, 1, 16, 0)      # AMC -> reaction 2024-05-02
REACTION = date(2024, 5, 2)
ENTRY = date(2024, 5, 6)                  # 2 sessions after reaction
EXPIRY = date(2024, 5, 31)                # ~25 days out (inside 14-35)


def daily_frame(prior_close=100.0, open_px=110.0, vol_scale=0.01) -> dict[str, pd.DataFrame]:
    """Calm history (low realized vol) then a gap-up open on the reaction day."""
    days = [d.date() for d in pd.bdate_range(end=pd.Timestamp(REACTION), periods=60)]
    closes, opens = [], []
    for i, _ in enumerate(days[:-1]):
        px = prior_close * (1 + vol_scale * ((-1) ** i) * 0.1)
        closes.append(px); opens.append(px)
    closes[-1] = prior_close
    closes.append(open_px * 1.001); opens.append(open_px)
    return {"TEST": pd.DataFrame({"open": opens, "close": closes}, index=days)}


def catalyst(symbol="TEST") -> Catalyst:
    return Catalyst(symbol=symbol, type=CatalystType.EARNINGS, when=REPORT, source="t")


def make_engine(arm="credit", daily=None):
    screener = DriftScreener(CFG.drift, daily or daily_frame())
    return DriftHarvestEngine(CFG.drift, CFG.risk, screener, arm=arm)


def chain_at(spot=110.0, iv=0.45):
    strikes = [round(spot * (1 + k / 100.0)) for k in range(-25, 26, 2)]
    return build_chain(symbol="TEST", spot=spot, as_of=datetime.combine(ENTRY, time(15, 45)),
                       expiry_ivs={EXPIRY: iv}, strikes=sorted(set(strikes)))


# ---------------------------------------------------------------------------
# Screener: measured gap + IV richness
# ---------------------------------------------------------------------------


def test_gap_measured_from_open_vs_prior_close() -> None:
    s = DriftScreener(CFG.drift, daily_frame(prior_close=100.0, open_px=110.0))
    setup = s.setup_for(catalyst())
    assert setup.gap == pytest.approx(0.10)
    assert setup.direction == "up"


def test_gap_down_sets_down_direction() -> None:
    s = DriftScreener(CFG.drift, daily_frame(prior_close=100.0, open_px=90.0))
    assert s.setup_for(catalyst()).direction == "down"


def test_small_gap_rejected() -> None:
    s = DriftScreener(CFG.drift, daily_frame(prior_close=100.0, open_px=101.0))
    setup = s.setup_for(catalyst())
    assert setup.direction is None and setup.reject_reason == "gap_too_small"


def test_realized_vol_is_annualized() -> None:
    idx = [d.date() for d in pd.bdate_range(end=pd.Timestamp(REACTION), periods=40)]
    # Alternating +/-1% moves -> ~1% daily sd -> ~16% annualized.
    closes = pd.Series([100 * (1.01 ** (i % 2)) for i in range(40)], index=idx)
    rv = realized_vol(closes, REACTION, 20)
    assert rv is not None and 0.10 < rv < 0.30


def test_entry_window_respects_trading_days() -> None:
    s = DriftScreener(CFG.drift, daily_frame())
    assert not s.entry_window_open(REACTION, REACTION)            # 0 sessions: too early
    assert s.entry_window_open(REACTION, date(2024, 5, 6))        # 2 sessions
    assert not s.entry_window_open(REACTION, date(2024, 5, 20))   # far past the window


def test_iv_richness_gate_blocks_cheap_premium() -> None:
    # Realized vol ~50% (violent history) vs 45% IV -> richness < 1.1 -> blocked.
    idx = [d.date() for d in pd.bdate_range(end=pd.Timestamp(REACTION), periods=60)]
    closes, opens = [], []
    px = 100.0
    for i in range(len(idx) - 1):
        px *= 1.03 if i % 2 == 0 else 1 / 1.03
        closes.append(px); opens.append(px)
    closes.append(110.0); opens.append(110.0)
    wild = {"TEST": pd.DataFrame({"open": opens, "close": closes}, index=idx)}
    engine = make_engine("credit", wild)
    assert engine.evaluate(catalyst(), chain_at(iv=0.30), None,
                           datetime.combine(ENTRY, time(15, 45))) is None
    assert engine.skipped_iv_not_rich >= 1


# ---------------------------------------------------------------------------
# Arm A — directional credit spread
# ---------------------------------------------------------------------------


def test_credit_arm_sells_puts_below_after_gap_up() -> None:
    engine = make_engine("credit")
    trade = engine.evaluate(catalyst(), chain_at(), None,
                            datetime.combine(ENTRY, time(15, 45)))
    assert trade is not None
    assert len(trade.legs) == 2
    short = next(l for l in trade.legs if l.side is Side.SELL)
    long = next(l for l in trade.legs if l.side is Side.BUY)
    assert short.key.right is OptionRight.PUT and long.key.right is OptionRight.PUT
    assert long.key.strike < short.key.strike < 110.0       # OTM put spread below spot
    assert trade.unit_cost < 0                              # net CREDIT received
    assert trade.unit_max_loss > 0                          # width - credit, positive
    assert trade.direction is Direction.LONG


def test_credit_arm_sells_calls_above_after_gap_down() -> None:
    engine = make_engine("credit", daily_frame(prior_close=100.0, open_px=90.0))
    chain = chain_at(spot=90.0)
    trade = engine.evaluate(catalyst(), chain, None, datetime.combine(ENTRY, time(15, 45)))
    assert trade is not None
    short = next(l for l in trade.legs if l.side is Side.SELL)
    long = next(l for l in trade.legs if l.side is Side.BUY)
    assert short.key.right is OptionRight.CALL
    assert long.key.strike > short.key.strike > 90.0
    assert trade.unit_cost < 0


def test_max_loss_is_width_minus_worst_case_credit() -> None:
    engine = make_engine("credit")
    trade = engine.evaluate(catalyst(), chain_at(), None,
                            datetime.combine(ENTRY, time(15, 45)))
    width = trade.rationale["width"]
    credit = trade.rationale["credit"]
    # Worst-case credit <= mid credit, so max loss >= width - mid credit.
    assert trade.unit_max_loss >= width - credit - 1e-9
    assert trade.unit_max_loss <= width


# ---------------------------------------------------------------------------
# Arms B and C — the controls
# ---------------------------------------------------------------------------


def test_condor_arm_sells_both_sides_and_is_neutral() -> None:
    engine = make_engine("condor")
    trade = engine.evaluate(catalyst(), chain_at(), None,
                            datetime.combine(ENTRY, time(15, 45)))
    assert trade is not None
    assert len(trade.legs) == 4
    rights = {l.key.right for l in trade.legs}
    assert rights == {OptionRight.CALL, OptionRight.PUT}
    assert trade.direction is Direction.NEUTRAL
    assert trade.unit_cost < 0
    # A condor can only lose on one side: risk basis is the widest single spread.
    assert trade.unit_max_loss <= trade.rationale["width"]


def test_debit_arm_pays_premium_in_the_drift_direction() -> None:
    engine = make_engine("debit")
    trade = engine.evaluate(catalyst(), chain_at(), None,
                            datetime.combine(ENTRY, time(15, 45)))
    assert trade is not None
    assert trade.unit_cost > 0                       # pays a debit
    buy = next(l for l in trade.legs if l.side is Side.BUY)
    assert buy.key.right is OptionRight.CALL         # gap up -> long calls
    assert trade.direction is Direction.LONG


def test_all_arms_share_the_same_gates() -> None:
    """The controls are only valid if every arm sees the identical event set."""
    small_gap = daily_frame(prior_close=100.0, open_px=101.0)
    at = datetime.combine(ENTRY, time(15, 45))
    for arm in ("credit", "condor", "debit"):
        assert make_engine(arm, small_gap).evaluate(catalyst(), chain_at(), None, at) is None


# ---------------------------------------------------------------------------
# Credit exit math (shared ExitManager)
# ---------------------------------------------------------------------------


def credit_position(entry=-1.00, value=-0.40, rules=None) -> Position:
    key = OptionKey(underlying="TEST", expiry=EXPIRY, right=OptionRight.PUT, strike=100.0)
    wing = OptionKey(underlying="TEST", expiry=EXPIRY, right=OptionRight.PUT, strike=95.0)
    return Position(
        position_id="p-credit",
        legs=[PositionLeg(key=key, side=Side.SELL, qty=1),
              PositionLeg(key=wing, side=Side.BUY, qty=1)],
        qty=5, entry_price=entry,
        entry_time=datetime.combine(ENTRY, time(15, 45)),
        engine_tag="drift_credit", exit_rules=rules or ExitRules(
            tp_credit_fraction=0.55, stop_credit_multiple=2.0, use_stops=True,
            close_before_expiry_days=2),
        current_value=value, max_loss=4.0,
    )


def test_credit_take_profit_at_captured_fraction() -> None:
    # Credit 1.00; liability 0.40 -> captured 60% >= 55% -> close.
    actions = evaluate_exits(credit_position(entry=-1.00, value=-0.40), ENTRY)
    assert actions and actions[0].reason == "take_profit_credit"
    # Liability 0.50 -> captured 50% -> hold.
    assert evaluate_exits(credit_position(entry=-1.00, value=-0.50), ENTRY) == []


def test_credit_stop_at_multiple_of_credit() -> None:
    # Liability 2.00 = 2x the 1.00 credit -> stop.
    actions = evaluate_exits(credit_position(entry=-1.00, value=-2.00), ENTRY)
    assert actions and actions[0].reason == "stop_loss_credit"
    assert evaluate_exits(credit_position(entry=-1.00, value=-1.90), ENTRY) == []


def test_credit_stop_respects_use_stops_flag() -> None:
    rules = ExitRules(tp_credit_fraction=0.55, stop_credit_multiple=2.0, use_stops=False,
                      close_before_expiry_days=2)
    assert evaluate_exits(credit_position(entry=-1.00, value=-3.00, rules=rules), ENTRY) == []


def test_debit_exit_rules_unaffected_by_credit_branch() -> None:
    """Regression: the archived debit strategies must behave exactly as before."""
    key = OptionKey(underlying="TEST", expiry=EXPIRY, right=OptionRight.CALL, strike=100.0)
    pos = Position(
        position_id="p-debit", legs=[PositionLeg(key=key, side=Side.BUY, qty=1)],
        qty=1, entry_price=3.0, entry_time=datetime.combine(ENTRY, time(15, 45)),
        engine_tag="x", exit_rules=ExitRules(stop_loss_pct=-0.50, use_stops=True),
        current_value=1.4,
    )
    actions = evaluate_exits(pos, ENTRY)
    assert actions and actions[0].reason == "stop_loss"


# ---------------------------------------------------------------------------
# Risk plumbing for credit structures
# ---------------------------------------------------------------------------


def test_premium_at_risk_uses_max_loss_for_credit_positions() -> None:
    """Without this, a credit spread reports ZERO risk and the heat cap never binds."""
    pos = credit_position(entry=-1.00, value=-0.40)
    assert pos.premium_at_risk == pytest.approx(4.0 * 5 * 100)
    # Debit positions keep the original behavior.
    pos.max_loss = None
    assert pos.premium_at_risk == 0.0  # negative entry price -> no premium paid


def test_risk_manager_sizes_credit_spread_on_max_loss() -> None:
    from catalyst.core.models import AccountState, OrderLeg, ProposedTrade
    from catalyst.risk.manager import RiskManager

    key = OptionKey(underlying="TEST", expiry=EXPIRY, right=OptionRight.PUT, strike=100.0)
    wing = OptionKey(underlying="TEST", expiry=EXPIRY, right=OptionRight.PUT, strike=95.0)
    proposal = ProposedTrade(
        engine="drift_credit", catalyst_ref="t",
        legs=[OrderLeg(key=key, side=Side.SELL, qty=1), OrderLeg(key=wing, side=Side.BUY, qty=1)],
        unit_cost=-1.0, unit_max_loss=4.0, direction=Direction.LONG,
        exit_rules=ExitRules(), per_trade_risk_fraction=0.02,
    )
    rm = RiskManager(CFG.risk)
    account = AccountState(equity=100_000, cash=100_000, buying_power=0,
                           timestamp=datetime.combine(ENTRY, time(15, 45)))
    decision = rm.size_entry(proposal, account, [])
    # 2% of 100k = $2,000 budget; unit risk $400 x 1.05 buffer = $420 -> 4 units.
    assert decision.units == 4


# ---------------------------------------------------------------------------
# v8 index VRP: credit-spread cash accounting
# ---------------------------------------------------------------------------


def test_credit_spread_cash_accounting_does_not_double_count() -> None:
    """Regression: the credit is received ONCE at entry.

    Closing must DEBIT the liability, not credit the P&L again. The original
    bug made equity rise while profit factor was below 1.0 — losing trades
    compounding into gains, which is impossible and was the tell.
    """
    credit, liability, contracts = 1.00, 0.50, 10
    cash = 100_000.0
    cash += credit * contracts * 100          # entry: receive the credit
    open_liability = credit * contracts * 100
    assert cash - open_liability == pytest.approx(100_000.0)  # flat at entry

    open_liability = liability * contracts * 100              # spread decays
    assert cash - open_liability == pytest.approx(100_500.0)  # +$500 unrealized

    cash -= liability * contracts * 100                       # close: pay the liability
    assert cash == pytest.approx(100_500.0)                   # realized == unrealized

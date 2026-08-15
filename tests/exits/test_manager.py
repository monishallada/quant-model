"""Exit manager: each mechanical rule on hand-built positions."""

from datetime import date, datetime

from catalyst.core.types import ExitRules, OptionKey, OptionRight, Position, PositionLeg, Side
from catalyst.exits.manager import evaluate_exits

KEY = OptionKey(underlying="SPY", expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=533.0)
KEY_SHORT = OptionKey(underlying="SPY", expiry=date(2024, 6, 21), right=OptionRight.CALL, strike=540.0)


def make_position(
    entry: float = 3.0,
    value: float = 3.0,
    hw: float | None = None,
    rules: ExitRules | None = None,
    tp1_done: bool = False,
    legs: list[PositionLeg] | None = None,
    entry_day: date = date(2024, 6, 3),
) -> Position:
    return Position(
        position_id="pos-1",
        legs=legs or [PositionLeg(key=KEY, side=Side.BUY, qty=1)],
        qty=2,
        entry_price=entry,
        entry_time=datetime.combine(entry_day, datetime.min.time()),
        engine_tag="test",
        exit_rules=rules or ExitRules(),
        current_value=value,
        high_water_value=hw if hw is not None else max(entry, value),
        tp1_done=tp1_done,
    )


def test_expiry_buffer_forces_close() -> None:
    pos = make_position(rules=ExitRules(close_before_expiry_days=1))
    actions = evaluate_exits(pos, date(2024, 6, 20))  # day before expiry
    assert actions and actions[0].reason == "expiry_buffer" and actions[0].close_fraction == 1.0


def test_hard_exit_date_time_stop() -> None:
    pos = make_position(rules=ExitRules(hard_exit_date=date(2024, 6, 10)))
    assert evaluate_exits(pos, date(2024, 6, 9)) == []
    actions = evaluate_exits(pos, date(2024, 6, 10))
    assert actions and actions[0].reason == "time_stop"


def test_max_hold_days() -> None:
    pos = make_position(rules=ExitRules(max_hold_trading_days=5), entry_day=date(2024, 6, 3))
    assert evaluate_exits(pos, date(2024, 6, 7)) == []  # 4 sessions later
    actions = evaluate_exits(pos, date(2024, 6, 10))  # 5 sessions later
    assert actions and actions[0].reason == "time_stop"


def test_stop_loss_only_when_stops_on() -> None:
    rules_on = ExitRules(stop_loss_pct=-0.50, use_stops=True)
    rules_off = ExitRules(stop_loss_pct=-0.50, use_stops=False)
    losing = make_position(entry=3.0, value=1.4)
    losing_off = make_position(entry=3.0, value=1.4, rules=rules_off)
    assert evaluate_exits(make_position(entry=3.0, value=1.6, rules=rules_on), date(2024, 6, 5)) == []
    actions = evaluate_exits(make_position(entry=3.0, value=1.4, rules=rules_on), date(2024, 6, 5))
    assert actions and actions[0].reason == "stop_loss"
    assert evaluate_exits(losing_off, date(2024, 6, 5)) == []


def test_tp1_scale_out_then_trailing() -> None:
    rules = ExitRules(tp1_gain=1.0, tp1_fraction=0.5, trail_stop_pct=0.30)
    pos = make_position(entry=3.0, value=6.1, rules=rules)
    actions = evaluate_exits(pos, date(2024, 6, 5))
    assert actions and actions[0].is_tp1 and actions[0].close_fraction == 0.5

    # After tp1: trail from high water 8.0; 30% giveback triggers at <= 5.6.
    pos2 = make_position(entry=3.0, value=5.5, hw=8.0, rules=rules, tp1_done=True)
    actions2 = evaluate_exits(pos2, date(2024, 6, 5))
    assert actions2 and actions2[0].reason == "trail_stop"


def test_spread_take_profit_fraction_of_max() -> None:
    rules = ExitRules(tp_fraction_of_max=0.60)
    legs = [
        PositionLeg(key=KEY, side=Side.BUY, qty=1),
        PositionLeg(key=KEY_SHORT, side=Side.SELL, qty=1),
    ]
    # Width 7.0; trigger at value >= 4.2.
    below = make_position(entry=1.85, value=4.0, rules=rules, legs=legs)
    at = make_position(entry=1.85, value=4.3, rules=rules, legs=legs)
    assert evaluate_exits(below, date(2024, 6, 5)) == []
    actions = evaluate_exits(at, date(2024, 6, 5))
    assert actions and actions[0].reason == "take_profit"


def test_no_exit_when_nothing_triggers() -> None:
    rules = ExitRules(tp1_gain=1.0, tp1_fraction=0.5, stop_loss_pct=-0.5)
    pos = make_position(entry=3.0, value=3.3, rules=rules)
    assert evaluate_exits(pos, date(2024, 6, 5)) == []

"""Mechanical exit manager.

The sole interpreter of the ``ExitRules`` attached to every position at entry.
Returns exit *actions*; the caller (backtester / live runner) constructs and
routes the close orders. Rule precedence, most defensive first:

1. expiry buffer — never carry into assignment/pin territory
2. hard exit date — mandatory time stop (the edge is the event; don't hold past it)
3. max hold days — engine-defined holding window
4. stop loss — soft stop on net premium, only when the engine runs stops
5. spread take-profit — close at a fraction of max spread value
6. tp1 scale-out — sell a fraction at the first gain target
7. trailing stop — protects the remainder after tp1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from catalyst.core.models import OptionRight, Position
from catalyst.core.tradingcal import add_trading_days, trading_days_between


@dataclass(frozen=True)
class ExitAction:
    position_id: str
    close_fraction: float  # fraction of CURRENT open qty to close (1.0 = all)
    reason: str
    is_tp1: bool = False  # caller marks position.tp1_done on successful fill


def _spread_width(position: Position) -> float | None:
    """Max value per unit for a 2-leg same-expiry same-right vertical."""
    if len(position.legs) != 2:
        return None
    a, b = position.legs
    if a.key.expiry != b.key.expiry or a.key.right != b.key.right:
        return None
    if a.qty != 1 or b.qty != 1 or a.side == b.side:
        return None
    return abs(a.key.strike - b.key.strike)


def evaluate_exits(position: Position, today: date) -> list[ExitAction]:
    """Exit actions for one position given today's marks (already applied to
    ``position.current_value`` / ``high_water_value`` by the broker)."""
    rules = position.exit_rules
    pid = position.position_id

    # 1. Expiry buffer (trading days).
    nearest_expiry = min(leg.key.expiry for leg in position.legs)
    if today >= add_trading_days(nearest_expiry, -rules.close_before_expiry_days):
        return [ExitAction(pid, 1.0, "expiry_buffer")]

    # 2. Hard exit date (catalyst time stop / pre-catalyst exit).
    if rules.hard_exit_date is not None and today >= rules.hard_exit_date:
        return [ExitAction(pid, 1.0, "time_stop")]

    # 3. Max holding window.
    if rules.max_hold_trading_days is not None:
        held = trading_days_between(position.entry_time.date(), today)
        if held >= rules.max_hold_trading_days:
            return [ExitAction(pid, 1.0, "time_stop")]

    entry = position.entry_price
    value = position.current_value
    if entry <= 0:
        return []  # credit/zero-cost structures handled by engine-specific rules only
    gain = (value - entry) / entry

    # 4. Stop loss (soft stop; per-engine stops-on/off is a config decision).
    if rules.use_stops and rules.stop_loss_pct is not None and gain <= rules.stop_loss_pct:
        return [ExitAction(pid, 1.0, "stop_loss")]

    # 5. Spread take-profit at fraction of max value.
    if rules.tp_fraction_of_max is not None:
        width = _spread_width(position)
        if width is not None and value >= rules.tp_fraction_of_max * width:
            return [ExitAction(pid, 1.0, "take_profit")]

    # 6. tp1 scale-out.
    if (
        rules.tp1_gain is not None
        and rules.tp1_fraction is not None
        and not position.tp1_done
        and gain >= rules.tp1_gain
    ):
        return [ExitAction(pid, rules.tp1_fraction, "take_profit_1", is_tp1=True)]

    # 7. Trailing stop (armed for the remainder once tp1 is done, or whenever
    #    configured without a tp1 stage).
    if rules.trail_stop_pct is not None and (position.tp1_done or rules.tp1_gain is None):
        hw = position.high_water_value
        if hw > 0 and value <= hw * (1.0 - rules.trail_stop_pct):
            return [ExitAction(pid, 1.0, "trail_stop")]

    return []

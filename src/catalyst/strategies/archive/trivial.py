"""Trivial test strategy for Gate 1 plumbing validation ONLY.

Buys a ~0.30-delta call on the first expiry after each synthetic catalyst,
holds behind mechanical exits. It has NO hypothesized edge — it exists to
exercise the full chain (data → engine → gate → fills → exits → metrics)
on real market data. Not part of the production engine set.
"""

from __future__ import annotations

from datetime import date, datetime

from catalyst.core.interfaces import Strategy
from catalyst.core.models import (
    Catalyst,
    Direction,
    ExitRules,
    OptionChain,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)


class TrivialTestStrategy(Strategy):
    name = "trivial"

    def __init__(
        self,
        target_delta: float,
        per_trade_risk: float,
        tp1_gain: float,
        tp1_fraction: float,
        stop_loss_pct: float,
        max_hold_trading_days: int,
        close_before_expiry_days: int,
        entry_days_before_catalyst: tuple[int, int] = (3, 10),
    ) -> None:
        self._target_delta = target_delta
        self._per_trade_risk = per_trade_risk
        self._tp1_gain = tp1_gain
        self._tp1_fraction = tp1_fraction
        self._stop_loss_pct = stop_loss_pct
        self._max_hold = max_hold_trading_days
        self._expiry_buffer = close_before_expiry_days
        self._entry_window = entry_days_before_catalyst

    def required_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        after = [e for e in available if e >= catalyst.when.date()]
        return after[:1] if after else []

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,
        as_of: datetime,
    ) -> ProposedTrade | None:
        days_out = (catalyst.when.date() - as_of.date()).days
        lo, hi = self._entry_window
        if not (lo <= days_out <= hi):
            return None

        expiries = [e for e in chain.expirations() if e >= catalyst.when.date()]
        if not expiries:
            return None
        expiry = expiries[0]

        calls = [
            c
            for c in chain.slice(expiry=expiry, right=OptionRight.CALL)
            if c.greeks is not None and c.bid > 0 and c.ask > 0
        ]
        if not calls:
            return None
        pick = min(calls, key=lambda c: abs((c.greeks.delta or 0.0) - self._target_delta))
        if pick.mid <= 0:
            return None

        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[OrderLeg(key=pick.key, side=Side.BUY, qty=1)],
            unit_cost=pick.mid,
            unit_max_loss=pick.mid,
            direction=Direction.LONG,
            exit_rules=ExitRules(
                tp1_gain=self._tp1_gain,
                tp1_fraction=self._tp1_fraction,
                stop_loss_pct=self._stop_loss_pct,
                use_stops=True,
                max_hold_trading_days=self._max_hold,
                close_before_expiry_days=self._expiry_buffer,
            ),
            per_trade_risk_fraction=self._per_trade_risk,
            rationale={"picked_delta": pick.greeks.delta, "picked_strike": pick.key.strike},
        )

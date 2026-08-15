"""Engine B — Crush-resistant debit spread (STABILIZER).

Higher win rate, capped payoff. 1-3 trading days before the catalyst, buys a
~0.40Δ / sells a ~0.20Δ same-right vertical expiring just after the event.
The short leg neutralizes most of the IV crush; max loss is the net debit.
Held through the event by design, time-stopped right after it resolves.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from catalyst.core.config import EngineBConfig, RiskConfig
from catalyst.core.interfaces import CatalystStrategy
from catalyst.core.types import (
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
from catalyst.core.tradingcal import add_trading_days, trading_days_between
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.core.chains import first_expiry_on_or_after, nearest_delta

logger = logging.getLogger(__name__)


class EngineBCrushSpread(CatalystStrategy):
    name = "engine_b"

    def __init__(self, cfg: EngineBConfig, risk: RiskConfig) -> None:
        self._cfg = cfg
        self._risk = risk

    def catalyst_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        days_out = trading_days_between(as_of, catalyst.when.date())
        lo, hi = self._cfg.dte_window
        if not (lo <= days_out <= hi):
            return []
        first = first_expiry_on_or_after(available, resolve_reaction_session(catalyst))
        return [first] if first else []

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        if not cfg.enabled:
            return None
        today = as_of.date()

        days_out = trading_days_between(today, catalyst.when.date())
        lo, hi = cfg.dte_window
        if not (lo <= days_out <= hi):
            return None
        if signal.direction is Direction.NEUTRAL:
            return None

        reaction = resolve_reaction_session(catalyst)
        expiry = first_expiry_on_or_after(chain.expirations(), reaction)
        if expiry is None:
            return None

        right = OptionRight.CALL if signal.direction is Direction.LONG else OptionRight.PUT
        long_leg = nearest_delta(chain, expiry, right, cfg.long_delta)
        short_leg = nearest_delta(chain, expiry, right, cfg.short_delta)
        if long_leg is None or short_leg is None:
            return None
        if long_leg.key.strike == short_leg.key.strike:
            return None

        net_debit = long_leg.mid - short_leg.mid
        worst_debit = long_leg.ask - short_leg.bid  # worst-case NBBO entry
        width = abs(long_leg.key.strike - short_leg.key.strike)
        # Degenerate spreads: no debit means quotes are crossed/absurd; a
        # worst-case debit at/above width can never profit after costs.
        if net_debit <= 0 or worst_debit >= width:
            return None

        hard_exit = add_trading_days(reaction, self._risk.time_stop_after_catalyst_days)
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[
                OrderLeg(key=long_leg.key, side=Side.BUY, qty=1),
                OrderLeg(key=short_leg.key, side=Side.SELL, qty=1),
            ],
            unit_cost=net_debit,
            unit_max_loss=worst_debit,
            direction=signal.direction,
            exit_rules=ExitRules(
                tp_fraction_of_max=cfg.tp_fraction_of_max,
                stop_loss_pct=self._risk.stop_loss_pct,
                use_stops=cfg.use_stops,
                hard_exit_date=hard_exit,
                close_before_expiry_days=self._risk.close_before_expiry_days,
            ),
            per_trade_risk_fraction=cfg.per_trade_risk,
            rationale={
                "signal_confidence": signal.confidence,
                "long_delta": long_leg.greeks.delta if long_leg.greeks else None,
                "short_delta": short_leg.greeks.delta if short_leg.greeks else None,
                "width": width,
                "net_debit": net_debit,
                "days_out_trading": days_out,
            },
        )

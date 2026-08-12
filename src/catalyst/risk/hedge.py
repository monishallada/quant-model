"""Hedge sleeve manager — drawdown reducer, NOT a profit source.

Mechanical rule: when the driver book is net long, hold index puts whose
premium at risk approaches ``hedge_fraction`` of equity, scaled by how much
of the deployable heat is net-long:

    target = hedge_fraction × equity × clamp(net_long_premium / (max_deployed × equity), 0..1)

Puts are ~25Δ, 45-100 DTE, on the configured index. The sleeve only ADDS via
this manager; positions leave through the expiry buffer (and the kill path).
Hedge proposals pass through the same RiskManager as everything else —
breakers and the cash floor bind hedges too.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from catalyst.core.config import RiskConfig
from catalyst.core.models import (
    Direction,
    ExitRules,
    OptionChain,
    OptionRight,
    OrderLeg,
    Position,
    ProposedTrade,
    Side,
)
from catalyst.engines.util import nearest_delta

logger = logging.getLogger(__name__)

HEDGE_ENGINE_TAG = "hedge"


class HedgeManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    @property
    def symbol(self) -> str:
        return self._cfg.hedge.symbol

    def wants_check(self, session: date, positions: list[Position]) -> bool:
        """Limit hedge evaluation (and its chain pulls) to the configured
        weekday, plus any day the sleeve is empty while drivers are long."""
        if session.weekday() == self._cfg.hedge.check_weekday:
            return True
        has_hedge = any(p.engine_tag == HEDGE_ENGINE_TAG for p in positions)
        return not has_hedge and self._net_long_premium(positions) > 0

    def required_expiries(self, available: list[date], session: date) -> list[date]:
        lo, hi = self._cfg.hedge.dte_window_days
        return [e for e in available if lo <= (e - session).days <= hi]

    def _net_long_premium(self, positions: list[Position]) -> float:
        net = 0.0
        for p in positions:
            if p.engine_tag == HEDGE_ENGINE_TAG:
                continue
            if p.direction is Direction.LONG:
                net += p.premium_at_risk
            elif p.direction is Direction.SHORT:
                net -= p.premium_at_risk
        return net

    def evaluate(
        self,
        chain: OptionChain,
        equity: float,
        positions: list[Position],
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        net_long = self._net_long_premium(positions)
        if net_long <= 0 or equity <= 0:
            return None
        exposure_ratio = min(net_long / (cfg.max_deployed * equity), 1.0)
        target = cfg.hedge_fraction * equity * exposure_ratio
        current = sum(
            p.premium_at_risk for p in positions if p.engine_tag == HEDGE_ENGINE_TAG
        )
        gap = target - current
        if target <= 0 or gap < cfg.hedge.min_add_fraction * target:
            return None

        lo, hi = cfg.hedge.dte_window_days
        session = as_of.date()
        expiries = [e for e in chain.expirations() if lo <= (e - session).days <= hi]
        if not expiries:
            return None
        target_days = (lo + hi) / 2.0
        expiry = min(expiries, key=lambda e: abs((e - session).days - target_days))
        put = nearest_delta(chain, expiry, OptionRight.PUT, cfg.hedge.put_delta)
        if put is None or put.mid <= 0:
            return None

        # Size to the GAP, not the engine risk fraction: per_trade_risk here is
        # the remaining hedge budget as an equity fraction.
        gap_fraction = gap / equity
        return ProposedTrade(
            engine=HEDGE_ENGINE_TAG,
            catalyst_ref="hedge-sleeve",
            legs=[OrderLeg(key=put.key, side=Side.BUY, qty=1)],
            unit_cost=put.mid,
            unit_max_loss=put.ask,
            direction=Direction.SHORT,  # offsets net-long driver exposure
            exit_rules=ExitRules(
                use_stops=False,  # hedges ride; they exist for the tail
                close_before_expiry_days=self._cfg.close_before_expiry_days,
            ),
            per_trade_risk_fraction=gap_fraction,
            rationale={
                "net_long_premium": net_long,
                "target": target,
                "current": current,
                "picked_delta": put.greeks.delta if put.greeks else None,
            },
        )

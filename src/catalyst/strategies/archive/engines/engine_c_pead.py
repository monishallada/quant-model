"""Engine C — Post-earnings drift (ANOMALY).

Modest, documented edge: prices drift in the direction of the earnings
surprise for days after the report. Direction is the MEASURED surprise sign
(never predicted); entry 0-2 trading days after the report, once IV has
normalized post-crush. Structure is a debit spread (default) or long option,
2-4 weeks out, held over the configured drift window.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from catalyst.core.config import EngineCConfig, RiskConfig
from catalyst.core.interfaces import CatalystStrategy
from catalyst.core.types import (
    Catalyst,
    CatalystType,
    Direction,
    ExitRules,
    OptionChain,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)
from catalyst.core.tradingcal import trading_days_between
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.data.iv_history import IVRankProvider
from catalyst.core.chains import nearest_delta

logger = logging.getLogger(__name__)


class EngineCPead(CatalystStrategy):
    name = "engine_c"

    def __init__(self, cfg: EngineCConfig, risk: RiskConfig, iv_rank: IVRankProvider) -> None:
        self._cfg = cfg
        self._risk = risk
        self._iv = iv_rank

    def _expiry_window_days(self) -> tuple[int, int]:
        lo_w, hi_w = self._cfg.expiry_weeks
        return lo_w * 7, hi_w * 7

    def catalyst_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        if catalyst.type is not CatalystType.EARNINGS:
            return []
        reaction = resolve_reaction_session(catalyst)
        if as_of < reaction:
            return []
        days_after = trading_days_between(reaction, as_of)
        lo, hi = self._cfg.entry_window_days
        if not (lo <= days_after <= hi):
            return []
        lo_d, hi_d = self._expiry_window_days()
        return [e for e in available if lo_d <= (e - as_of).days <= hi_d]

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,  # unused by design: direction is the measured surprise
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        if not cfg.enabled:
            return None
        if catalyst.type is not CatalystType.EARNINGS:
            return None
        today = as_of.date()

        # Entry window: N trading days AFTER the first post-report session.
        reaction = resolve_reaction_session(catalyst)
        if today < reaction:
            return None
        days_after = trading_days_between(reaction, today)
        lo, hi = cfg.entry_window_days
        if not (lo <= days_after <= hi):
            return None

        surprise = catalyst.surprise_pct
        if surprise is None or abs(surprise) < cfg.min_surprise_pct:
            return None
        direction = Direction.LONG if surprise > 0 else Direction.SHORT

        # IV normalized post-crush.
        rank = self._iv.iv_rank(catalyst.symbol, today)
        if rank is None or rank >= cfg.iv_rank_max:
            return None

        lo_d, hi_d = self._expiry_window_days()
        expiries = [e for e in chain.expirations() if lo_d <= (e - today).days <= hi_d]
        if not expiries:
            return None
        target_days = (lo_d + hi_d) / 2.0
        expiry = min(expiries, key=lambda e: abs((e - today).days - target_days))

        right = OptionRight.CALL if direction is Direction.LONG else OptionRight.PUT

        if cfg.structure == "long_option":
            pick = nearest_delta(chain, expiry, right, cfg.single_leg_delta)
            if pick is None or pick.mid <= 0:
                return None
            legs = [OrderLeg(key=pick.key, side=Side.BUY, qty=1)]
            unit_cost, unit_max_loss = pick.mid, pick.ask
            rationale_legs = {"picked_delta": pick.greeks.delta if pick.greeks else None}
        else:
            long_leg = nearest_delta(chain, expiry, right, cfg.long_delta)
            short_leg = nearest_delta(chain, expiry, right, cfg.short_delta)
            if long_leg is None or short_leg is None:
                return None
            if long_leg.key.strike == short_leg.key.strike:
                return None
            net_debit = long_leg.mid - short_leg.mid
            worst_debit = long_leg.ask - short_leg.bid
            width = abs(long_leg.key.strike - short_leg.key.strike)
            if net_debit <= 0 or worst_debit >= width:
                return None
            legs = [
                OrderLeg(key=long_leg.key, side=Side.BUY, qty=1),
                OrderLeg(key=short_leg.key, side=Side.SELL, qty=1),
            ]
            unit_cost, unit_max_loss = net_debit, worst_debit
            rationale_legs = {
                "long_delta": long_leg.greeks.delta if long_leg.greeks else None,
                "short_delta": short_leg.greeks.delta if short_leg.greeks else None,
                "width": width,
            }

        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=legs,
            unit_cost=unit_cost,
            unit_max_loss=unit_max_loss,
            direction=direction,
            exit_rules=ExitRules(
                trail_stop_pct=cfg.trail_stop_pct,
                stop_loss_pct=self._risk.stop_loss_pct,
                use_stops=cfg.use_stops,
                max_hold_trading_days=cfg.hold_days,
                close_before_expiry_days=self._risk.close_before_expiry_days,
            ),
            per_trade_risk_fraction=cfg.per_trade_risk,
            rationale={
                "surprise_pct": surprise,
                "iv_rank": rank,
                "days_after_report": days_after,
                **rationale_legs,
            },
        )

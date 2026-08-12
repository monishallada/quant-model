"""Engine A — Pre-catalyst convexity (DRIVER).

Explosive, low-win-rate, tiny size. Buys a single slightly-OTM option 5-10
trading days before a known catalyst, ONLY when optionality is cheap
(IV rank below threshold) and the directional signal has an opinion.
Default exit is the session before the event (pre-move play, avoid crush);
holding through is a config switch.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from catalyst.core.config import EngineAConfig, RiskConfig
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
from catalyst.core.tradingcal import add_trading_days, trading_days_between
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.data.iv_history import IVRankProvider
from catalyst.engines.util import nearest_delta

logger = logging.getLogger(__name__)


class EngineAConvexity(Strategy):
    name = "engine_a"

    def __init__(self, cfg: EngineAConfig, risk: RiskConfig, iv_rank: IVRankProvider) -> None:
        self._cfg = cfg
        self._risk = risk
        self._iv = iv_rank

    def _life_band(self, catalyst: Catalyst, as_of: date) -> tuple[float, float] | None:
        """Calendar-day band for option life so the catalyst falls within the
        configured early fraction of it."""
        t_cat = (catalyst.when.date() - as_of).days
        if t_cat <= 0:
            return None
        lo_frac, hi_frac = self._cfg.catalyst_life_fraction
        return t_cat / hi_frac, t_cat / lo_frac

    def required_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        # Only request chain data on days inside the entry window — historical
        # chain pulls are the backtest's dominant cost.
        days_out = trading_days_between(as_of, catalyst.when.date())
        lo, hi = self._cfg.dte_window
        if not (lo <= days_out <= hi):
            return []
        band = self._life_band(catalyst, as_of)
        if band is None:
            return []
        lo_days, hi_days = band
        return [e for e in available if lo_days <= (e - as_of).days <= hi_days]

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

        # Trigger window: catalyst N trading days out.
        days_out = trading_days_between(today, catalyst.when.date())
        lo, hi = cfg.dte_window
        if not (lo <= days_out <= hi):
            return None

        # Direction from the signal; neutral = no trade.
        if signal.direction is Direction.NEUTRAL:
            return None

        # Cheap-optionality gate.
        rank = self._iv.iv_rank(catalyst.symbol, today)
        if rank is None or rank >= cfg.iv_rank_max:
            return None

        # Expiration: catalyst inside the first lo..hi fraction of option life.
        band = self._life_band(catalyst, today)
        if band is None:
            return None
        lo_days, hi_days = band
        expiries = [e for e in chain.expirations() if lo_days <= (e - today).days <= hi_days]
        if not expiries:
            return None
        t_cat = (catalyst.when.date() - today).days
        mid_frac = (cfg.catalyst_life_fraction[0] + cfg.catalyst_life_fraction[1]) / 2.0
        ideal_life = t_cat / mid_frac
        expiry = min(expiries, key=lambda e: abs((e - today).days - ideal_life))

        right = OptionRight.CALL if signal.direction is Direction.LONG else OptionRight.PUT
        pick = nearest_delta(chain, expiry, right, cfg.target_delta, cfg.delta_range)
        if pick is None or pick.mid <= 0:
            return None

        reaction = resolve_reaction_session(catalyst)
        if cfg.hold_through_catalyst:
            hard_exit = add_trading_days(reaction, self._risk.time_stop_after_catalyst_days)
        else:
            hard_exit = add_trading_days(reaction, -1)  # out the session before the event

        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[OrderLeg(key=pick.key, side=Side.BUY, qty=1)],
            unit_cost=pick.mid,
            unit_max_loss=pick.mid,
            direction=signal.direction,
            exit_rules=ExitRules(
                tp1_gain=cfg.tp1_gain,
                tp1_fraction=cfg.tp1_fraction,
                trail_stop_pct=cfg.trail_stop_pct,
                stop_loss_pct=self._risk.stop_loss_pct,
                use_stops=cfg.use_stops,
                hard_exit_date=hard_exit,
                close_before_expiry_days=self._risk.close_before_expiry_days,
            ),
            per_trade_risk_fraction=cfg.per_trade_risk,
            rationale={
                "iv_rank": rank,
                "signal_confidence": signal.confidence,
                "picked_delta": pick.greeks.delta if pick.greeks else None,
                "days_out_trading": days_out,
                "catalyst_life_fraction": t_cat / (expiry - today).days,
            },
        )

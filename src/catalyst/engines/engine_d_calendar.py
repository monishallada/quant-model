"""Engine D — Event term-structure calendar (ADVANCED, direction-neutral).

When a near catalyst inflates front-month IV well above back-month IV, sells
the inflated near-term ATM option and buys a cheaper longer-dated one at the
same strike. Profits when the front leg crushes post-event; the max-loss
guard closes the structure if the underlying runs away from the strike.

Disabled by default in config: validated only after Engines A-C prove out.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from catalyst.core.config import EngineDConfig, RiskConfig
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
from catalyst.engines.util import atm_contract, first_expiry_on_or_after

logger = logging.getLogger(__name__)

_BACK_MONTH_TARGET_DAYS = 30  # back leg ~1 month behind the front leg


class EngineDCalendar(Strategy):
    name = "engine_d"

    def __init__(self, cfg: EngineDConfig, risk: RiskConfig) -> None:
        self._cfg = cfg
        self._risk = risk

    def _pick_expiries(self, available: list[date], catalyst: Catalyst) -> tuple[date, date] | None:
        front = first_expiry_on_or_after(available, resolve_reaction_session(catalyst))
        if front is None:
            return None
        back_target = front + timedelta(days=_BACK_MONTH_TARGET_DAYS)
        backs = [e for e in available if e > front]
        if not backs:
            return None
        back = min(backs, key=lambda e: abs((e - back_target).days))
        if back <= front:
            return None
        return front, back

    def required_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        days_out = trading_days_between(as_of, catalyst.when.date())
        lo, hi = self._cfg.dte_window
        if not (lo <= days_out <= hi):
            return []
        pair = self._pick_expiries(available, catalyst)
        return list(pair) if pair else []

    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,  # unused: engine is direction-neutral
        as_of: datetime,
    ) -> ProposedTrade | None:
        cfg = self._cfg
        if not cfg.enabled:
            return None
        today = as_of.date()
        if today >= catalyst.when.date():
            return None

        pair = self._pick_expiries(chain.expirations(), catalyst)
        if pair is None:
            return None
        front_exp, back_exp = pair

        # Evaluate both rights; take the one with usable quotes on both legs
        # and the tighter combined spread.
        best: tuple[float, OrderLeg, OrderLeg, float, float, float] | None = None
        for right in (OptionRight.CALL, OptionRight.PUT):
            front = atm_contract(chain, front_exp, right)
            back = atm_contract(chain, back_exp, right)
            if front is None or back is None:
                continue
            if front.key.strike != back.key.strike:
                continue  # need a true calendar at one strike
            if front.greeks is None or back.greeks is None:
                continue
            front_iv, back_iv = front.greeks.iv, back.greeks.iv
            if front_iv <= 0 or back_iv <= 0:
                continue
            ratio = front_iv / back_iv
            if ratio < cfg.term_structure_ratio_min:
                continue
            net_debit = back.mid - front.mid
            if net_debit <= 0:
                continue
            combined_spread = front.spread_pct_of_mid + back.spread_pct_of_mid
            entry = (
                combined_spread,
                OrderLeg(key=front.key, side=Side.SELL, qty=1),
                OrderLeg(key=back.key, side=Side.BUY, qty=1),
                net_debit,
                ratio,
                front.key.strike,
            )
            if best is None or combined_spread < best[0]:
                best = entry
        if best is None:
            return None
        _, short_leg, long_leg, net_debit, ratio, strike = best

        reaction = resolve_reaction_session(catalyst)
        hard_exit = add_trading_days(reaction, self._risk.time_stop_after_catalyst_days)
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=catalyst.ref,
            legs=[short_leg, long_leg],
            unit_cost=net_debit,
            unit_max_loss=net_debit,
            direction=Direction.NEUTRAL,
            exit_rules=ExitRules(
                # Max-loss guard on large moves: always armed for calendars.
                stop_loss_pct=-cfg.max_loss_guard_pct,
                use_stops=True,
                hard_exit_date=hard_exit,
                close_before_expiry_days=self._risk.close_before_expiry_days,
            ),
            per_trade_risk_fraction=cfg.per_trade_risk,
            rationale={
                "term_structure_ratio": ratio,
                "strike": strike,
                "front_expiry": front_exp.isoformat(),
                "back_expiry": back_exp.isoformat(),
                "net_debit": net_debit,
            },
        )

"""RiskManager — the authoritative risk layer. Overridable by nothing.

Every entry passes through ``size_entry`` (the EntryGate protocol the
backtester and live runner both use). Checks, in order:

1. circuit breakers — daily/weekly loss halts block ALL new entries
   (exits are unaffected: they never pass through this gate)
2. per-trade sizing — fixed-fractional on the engine's configured risk
3. cash floor — cash after entry may never dip below the configured
   fraction of equity (a structural survival guarantee)
4. portfolio heat — total premium at risk capped
5. hedge budget — hedge-sleeve entries capped at hedge_fraction of equity
6. correlation cap — max concurrent positions per correlated group

Sizing can only shrink to satisfy a cap, never grow. The API takes NO
performance or return-target inputs — there is structurally nowhere to feed
"we're behind plan" into this layer, and ``RiskConfig`` is a frozen model.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from catalyst.core.config import RiskConfig
from catalyst.core.models import AccountState, Position, ProposedTrade
from catalyst.risk.gate import GateDecision
from catalyst.risk.sizing import fixed_fractional_units

logger = logging.getLogger(__name__)

HEDGE_ENGINE_TAG = "hedge"


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg
        self._marks: dict[date, float] = {}  # session -> end-of-session equity
        self._history: dict[str, pd.DataFrame] = {}
        self._corr_cache: dict[tuple[str, str, str], float] = {}

    # ------------------------------------------------------------------
    # State fed by the runner (marks + return history), never by strategies
    # ------------------------------------------------------------------

    def record_mark(self, day: date, equity: float) -> None:
        """Record end-of-session equity. Breakers compare intraday equity to
        these marks; the runner must call this every session/cycle."""
        self._marks[day] = equity

    def set_history(self, history: dict[str, pd.DataFrame]) -> None:
        """Daily OHLCV per symbol, for correlation grouping."""
        self._history = history
        self._corr_cache.clear()

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def _breaker_state(self, today: date, current_equity: float) -> str | None:
        """Returns a reason string when a breaker is tripped, else None."""
        prior_days = sorted(d for d in self._marks if d < today)
        if prior_days:
            day_ref = self._marks[prior_days[-1]]
            if day_ref > 0 and current_equity / day_ref - 1.0 <= self._cfg.daily_loss_halt:
                return (
                    f"daily breaker: {current_equity / day_ref - 1.0:.2%}"
                    f" <= {self._cfg.daily_loss_halt:.0%}"
                )
        week_anchor = today - timedelta(days=today.weekday())  # Monday
        pre_week = [d for d in prior_days if d < week_anchor]
        if pre_week:
            week_ref = self._marks[pre_week[-1]]
            if week_ref > 0 and current_equity / week_ref - 1.0 <= self._cfg.weekly_loss_halt:
                return (
                    f"weekly breaker: {current_equity / week_ref - 1.0:.2%}"
                    f" <= {self._cfg.weekly_loss_halt:.0%}"
                )
        return None

    # ------------------------------------------------------------------
    # Correlation grouping
    # ------------------------------------------------------------------

    def _correlation(self, a: str, b: str, upto: date) -> float:
        if a == b:
            return 1.0
        key = (min(a, b), max(a, b), f"{upto:%Y-%m}")
        if key in self._corr_cache:
            return self._corr_cache[key]
        ha, hb = self._history.get(a), self._history.get(b)
        corr = 1.0  # unknown history => assume correlated (conservative)
        if (
            ha is not None and hb is not None and not ha.empty and not hb.empty
            and isinstance(ha.index, pd.DatetimeIndex) and isinstance(hb.index, pd.DatetimeIndex)
        ):
            cutoff = pd.Timestamp(upto)
            ra = ha.loc[:cutoff, "close"].pct_change().dropna().tail(self._cfg.correlation_lookback_days)
            rb = hb.loc[:cutoff, "close"].pct_change().dropna().tail(self._cfg.correlation_lookback_days)
            joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(joined) >= self._cfg.correlation_lookback_days // 2:
                corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        self._corr_cache[key] = corr
        return corr

    def _correlated_count(self, symbol: str, positions: list[Position], upto: date) -> int:
        count = 0
        for pos in positions:
            if pos.engine_tag == HEDGE_ENGINE_TAG:
                continue  # hedges don't crowd the driver book
            if self._correlation(symbol, pos.underlying, upto) >= self._cfg.correlation_threshold:
                count += 1
        return count

    # ------------------------------------------------------------------
    # EntryGate protocol
    # ------------------------------------------------------------------

    def size_entry(
        self,
        proposal: ProposedTrade,
        account: AccountState,
        positions: list[Position],
    ) -> GateDecision:
        cfg = self._cfg
        today = account.timestamp.date()
        equity = account.equity
        is_hedge = proposal.engine == HEDGE_ENGINE_TAG

        # 1. Circuit breakers gate every new entry, hedges included.
        breaker = self._breaker_state(today, equity)
        if breaker is not None:
            return GateDecision(0, breaker)

        # Max loss is the risk basis for EVERY structure. A credit structure
        # has a negative unit_cost (premium received) but a real, positive max
        # loss (width - credit) that must be reserved exactly like a debit —
        # which is also how a broker margins it.
        if proposal.unit_max_loss <= 0:
            return GateDecision(0, "non-positive max loss")
        if proposal.unit_cost != proposal.unit_cost:  # NaN guard
            return GateDecision(0, "non-finite unit cost")

        # Size against worst-case entry cost plus the configured slippage
        # buffer so the risk budget is a HARD bound: actual fills can be worse
        # than mid, and sizing must never let that breach the budget.
        buffered_risk = (
            max(proposal.unit_max_loss, proposal.unit_cost) * (1.0 + cfg.sizing_cost_buffer)
        )

        # 2. Fixed-fractional base size.
        units = fixed_fractional_units(equity, proposal.per_trade_risk_fraction, buffered_risk)
        if units <= 0:
            return GateDecision(0, "risk budget below one unit")

        unit_cost_dollars = buffered_risk * 100.0
        unit_risk_dollars = buffered_risk * 100.0

        # 3. Cash floor: cash after paying the debit stays above the floor.
        floor = cfg.cash_floor_fraction * equity
        cash_headroom = account.cash - floor
        if cash_headroom < unit_cost_dollars:
            return GateDecision(0, f"cash floor: headroom {cash_headroom:,.0f} < one unit")
        units = min(units, int(cash_headroom // unit_cost_dollars))

        # 4. Portfolio heat: total premium at risk <= max_deployed.
        deployed = sum(p.premium_at_risk for p in positions)
        heat_headroom = cfg.max_deployed * equity - deployed
        if heat_headroom < unit_risk_dollars:
            return GateDecision(0, f"heat cap: deployed {deployed:,.0f} of {cfg.max_deployed * equity:,.0f}")
        units = min(units, int(heat_headroom // unit_risk_dollars))

        # 5. Hedge sleeve budget (hedge entries only).
        if is_hedge:
            hedge_deployed = sum(
                p.premium_at_risk for p in positions if p.engine_tag == HEDGE_ENGINE_TAG
            )
            hedge_headroom = cfg.hedge_fraction * equity - hedge_deployed
            if hedge_headroom < unit_risk_dollars:
                return GateDecision(0, "hedge budget exhausted")
            units = min(units, int(hedge_headroom // unit_risk_dollars))

        # 6. Correlation cap (driver entries only).
        if not is_hedge:
            symbol = proposal.legs[0].key.underlying
            correlated = self._correlated_count(symbol, positions, today)
            if correlated >= cfg.max_correlated_positions:
                return GateDecision(
                    0, f"correlation cap: {correlated} correlated positions open"
                )

        if units <= 0:
            return GateDecision(0, "caps reduced size to zero")
        return GateDecision(units, "ok")

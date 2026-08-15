"""Catalyst screener: implied move, liquidity gates, ranked scoring.

For each upcoming catalyst: measure the ATM straddle implied move on the
first expiration that captures the event, gate on liquidity and minimum
implied move, and rank survivors by a config-weighted geometric mean of
(liquidity, implied move, days-to-catalyst fit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

from catalyst.core.config import ScreenerConfig
from catalyst.core.types import Catalyst, OptionChain, OptionContract, OptionRight
from catalyst.data.catalysts import resolve_reaction_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenedCatalyst:
    catalyst: Catalyst
    implied_move: float  # ATM straddle mid / spot (through first post-event expiry)
    event_expiry: date  # first expiration capturing the event
    atm_strike: float
    spread_pct: float  # worse of call/put ATM spreads
    atm_volume: int  # combined ATM call+put volume
    atm_open_interest: int  # combined ATM call+put OI
    days_to_catalyst: int
    score: float
    rejected: str | None = None  # None = passed all gates


def _atm_pair(
    chain: OptionChain, expiry: date
) -> tuple[OptionContract, OptionContract] | None:
    calls = {c.key.strike: c for c in chain.slice(expiry=expiry, right=OptionRight.CALL)}
    puts = {c.key.strike: c for c in chain.slice(expiry=expiry, right=OptionRight.PUT)}
    shared = [k for k in calls.keys() & puts.keys() if calls[k].ask > 0 and puts[k].ask > 0]
    if not shared:
        return None
    strike = min(shared, key=lambda k: abs(k - chain.underlying_price))
    return calls[strike], puts[strike]


class CatalystScreener:
    def __init__(self, cfg: ScreenerConfig) -> None:
        self._cfg = cfg

    def screen(
        self, catalyst: Catalyst, chain: OptionChain, as_of: datetime
    ) -> ScreenedCatalyst | None:
        """Metrics + gates for one catalyst. Returns None only when the chain
        can't even price it; gate failures return a row with ``rejected`` set
        (so reports can show WHY candidates dropped)."""
        cfg = self._cfg
        reaction = resolve_reaction_session(catalyst)
        expiries = [e for e in chain.expirations() if e >= reaction]
        if not expiries:
            return None
        expiry = expiries[0]
        pair = _atm_pair(chain, expiry)
        if pair is None or chain.underlying_price <= 0:
            return None
        call, put = pair

        straddle_mid = call.mid + put.mid
        implied_move = straddle_mid / chain.underlying_price
        spread_pct = max(call.spread_pct_of_mid, put.spread_pct_of_mid)
        volume = call.volume + put.volume
        oi = call.open_interest + put.open_interest
        days_to = (catalyst.when.date() - as_of.date()).days

        rejected: str | None = None
        if spread_pct > cfg.liquidity.max_spread_pct_of_mid:
            rejected = f"spread {spread_pct:.1%} > {cfg.liquidity.max_spread_pct_of_mid:.1%}"
        elif volume < cfg.liquidity.min_option_volume:
            rejected = f"volume {volume} < {cfg.liquidity.min_option_volume}"
        elif oi < cfg.liquidity.min_open_interest:
            rejected = f"OI {oi} < {cfg.liquidity.min_open_interest}"
        elif implied_move < cfg.min_implied_move:
            rejected = f"implied move {implied_move:.1%} < {cfg.min_implied_move:.1%}"

        # Normalized 0..1 factors -> weighted geometric mean.
        liq_score = max(0.0, 1.0 - spread_pct / cfg.liquidity.max_spread_pct_of_mid)
        move_score = min(implied_move / (cfg.min_implied_move * cfg.move_score_saturation), 1.0)
        fit_score = max(0.0, 1.0 - abs(days_to - cfg.fit_ideal_days) / cfg.fit_tolerance_days)
        w = cfg.score_weights
        total_w = w.liquidity + w.implied_move + w.days_to_catalyst_fit
        score = (
            (max(liq_score, 1e-9) ** w.liquidity)
            * (max(move_score, 1e-9) ** w.implied_move)
            * (max(fit_score, 1e-9) ** w.days_to_catalyst_fit)
        ) ** (1.0 / total_w)

        return ScreenedCatalyst(
            catalyst=catalyst,
            implied_move=implied_move,
            event_expiry=expiry,
            atm_strike=call.key.strike,
            spread_pct=spread_pct,
            atm_volume=volume,
            atm_open_interest=oi,
            days_to_catalyst=days_to,
            score=score,
            rejected=rejected,
        )

    def rank(
        self,
        candidates: list[tuple[Catalyst, OptionChain]],
        as_of: datetime,
    ) -> list[ScreenedCatalyst]:
        """Screen all candidates; return PASSING ones ranked by score desc."""
        screened: list[ScreenedCatalyst] = []
        for catalyst, chain in candidates:
            row = self.screen(catalyst, chain, as_of)
            if row is None:
                continue
            if row.rejected:
                logger.debug("Screened out %s: %s", catalyst.ref, row.rejected)
                continue
            screened.append(row)
        return sorted(screened, key=lambda s: s.score, reverse=True)

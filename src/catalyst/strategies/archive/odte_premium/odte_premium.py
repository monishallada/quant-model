"""O1 — 0DTE index defined-risk premium selling (SPXW put credit spreads).

THE ONE OPTIONS CANDIDATE WITH PEER-QUALITY NET-OF-COST DOCUMENTATION
(Vilkov, SSRN 4641356; Beckmeyer et al., SSRN 4404704): systematically selling
same-day index put spreads after the opening volatility settles harvests the
0DTE variance premium. The seller's edge is documented to survive realistic
spread crossing on the index complex specifically — the cheapest venue in the
options market (this project measured the same 3x cost ratio at EOD in v8).

Why this design does NOT repeat this repo's dead ends:
- No direction call (intraday direction: coin flip, measured three ways now).
- SHORT the theta that killed every long-intraday-options campaign.
- Defined risk: max loss = (width - credit) is the sizing basis.
- ONE entry per day, hold to the mandatory 15:55 buy-back: a single spread
  crossing plus a cheap late-day exit, not a cycling toll.

Selection is fully quote-driven and point-in-time — no chains, no greeks:
1. Proxy spot from SPY bars x10, refined to a SYNTHETIC SPX spot via put-call
   parity at one probe strike (0DTE, r~0 intraday): S = K + C_mid - P_mid.
   The index level comes from the option market itself; no index feed needed.
2. Implied remaining-day move from the ATM straddle mid at the same strike.
3. Short strike = synthetic spot - k_sigma x implied move (rounded to the
   5-point SPXW grid); wing `width` points lower. Both legs must have live
   visible quotes and clear the min-credit floor or the day is skipped.

All quotes seen at decision time ts are <= ts-1min (engine-enforced); fills
happen at ts from each contract's own NBBO via the shared cost model. The EOD
chain path was measured at ~5.5 min/chain for SPXW and REJECTED — this design
costs ~6 targeted contract-day requests per session instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

from catalyst.core.interfaces.intraday import IntradayContext, IntradayStrategy
from catalyst.core.types import (
    Direction,
    ExitRules,
    OptionKey,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
)
from catalyst.strategies.registry import StrategyMeta, register

logger = logging.getLogger(__name__)

STRIKE_GRID = 5.0                      # SPXW listed strike spacing


@dataclass(frozen=True)
class OdtePremiumParams:
    index_symbol: str = "SPXW"        # SPX weeklies: cash-settled, European
    proxy_symbol: str = "SPY"         # minute-bar proxy; refined via parity
    entry_time: time = time(10, 30)   # after opening vol settles (documented)
    k_sigma: float = 1.25             # short-strike distance in implied moves
    width_points: float = 25.0        # wing distance in SPX points
    min_credit: float = 0.15          # skip all-toll-no-premium entries
    max_credit_fraction: float = 0.45 # credit > 45% of width = near-ATM; skip
    stop_credit_multiple: float = 3.0 # buy back if liability reaches 3x credit
    flatten_time: time = time(15, 55)
    per_trade_risk_fraction: float = 0.03   # one 25-wide SPXW ~ $2.5k max loss


class OdtePremiumStrategy(IntradayStrategy):
    """One defined-risk short put spread per qualifying session."""

    name = "odte_premium"

    def __init__(self, params: OdtePremiumParams | None = None) -> None:
        self._p = params or OdtePremiumParams()
        self._expiries: set[date] | None = None

    def session_universe(self, session: date) -> list[str]:
        return [self._p.proxy_symbol]

    # ------------------------------------------------------------------
    def on_minute(self, ctx: IntradayContext) -> list[ProposedTrade]:
        p = self._p
        if ctx.now.time() != p.entry_time or ctx.option_quote is None:
            return []
        if not self._has_same_day_expiry(ctx):
            return []
        proxy = ctx.spot(p.proxy_symbol)
        if proxy is None:
            return []

        # --- synthetic spot + implied move from ONE probe strike ----------
        probe_k = self._grid(proxy * 10.0)
        pc = self._pair(ctx, probe_k)
        if pc is None:
            return []
        c_mid, p_mid = pc
        spot = probe_k + c_mid - p_mid          # put-call parity, r~0 intraday
        implied_move = max(c_mid + p_mid, STRIKE_GRID)   # ATM-ish straddle

        # --- strike selection --------------------------------------------
        short_k = self._grid(spot - p.k_sigma * implied_move)
        wing_k = short_k - p.width_points
        short_key = self._put(ctx.session, short_k)
        wing_key = self._put(ctx.session, wing_k)
        sq = ctx.option_quote(short_key)
        wq = ctx.option_quote(wing_key)
        if sq is None or wq is None:
            return []
        credit = (sq[0] + sq[1]) / 2.0 - (wq[0] + wq[1]) / 2.0   # mid estimate
        width = short_k - wing_k
        if credit < p.min_credit or credit > p.max_credit_fraction * width:
            return []

        return [ProposedTrade(
            engine=self.name,
            catalyst_ref=f"{p.index_symbol}:{ctx.session}",
            legs=[OrderLeg(key=short_key, side=Side.SELL, qty=1),
                  OrderLeg(key=wing_key, side=Side.BUY, qty=1)],
            unit_cost=-credit,
            unit_max_loss=width - credit,
            direction=Direction.NEUTRAL,
            exit_rules=ExitRules(
                close_by_time=p.flatten_time,
                use_stops=True,
                stop_loss_pct=-p.stop_credit_multiple,   # credit-stop semantics
            ),
            per_trade_risk_fraction=p.per_trade_risk_fraction,
            rationale={"short_strike": short_k, "wing": wing_k, "width": width,
                       "synthetic_spot": round(spot, 2),
                       "implied_move": round(implied_move, 2),
                       "est_credit": round(credit, 2),
                       "ref": f"{p.index_symbol}:{ctx.session}"},
        )]

    # ------------------------------------------------------------------
    def _pair(self, ctx: IntradayContext, strike: float):
        cq = ctx.option_quote(self._call(ctx.session, strike))
        pq = ctx.option_quote(self._put(ctx.session, strike))
        if cq is None or pq is None:
            return None
        return (cq[0] + cq[1]) / 2.0, (pq[0] + pq[1]) / 2.0

    def _call(self, expiry: date, strike: float) -> OptionKey:
        return OptionKey(underlying=self._p.index_symbol, expiry=expiry,
                         right=OptionRight.CALL, strike=strike)

    def _put(self, expiry: date, strike: float) -> OptionKey:
        return OptionKey(underlying=self._p.index_symbol, expiry=expiry,
                         right=OptionRight.PUT, strike=strike)

    @staticmethod
    def _grid(x: float) -> float:
        return round(x / STRIKE_GRID) * STRIKE_GRID

    def _has_same_day_expiry(self, ctx: IntradayContext) -> bool:
        if self._expiries is None:
            try:
                self._expiries = set(ctx.data.list_expirations(self._p.index_symbol))
            except Exception:                       # noqa: BLE001
                return False
        return ctx.session in self._expiries


def build(cfg) -> OdtePremiumStrategy:
    section = getattr(cfg, "odte_premium", None)
    if section is None:
        return OdtePremiumStrategy()
    return OdtePremiumStrategy(OdtePremiumParams(
        k_sigma=getattr(section, "k_sigma", OdtePremiumParams.k_sigma),
        width_points=getattr(section, "width_points", OdtePremiumParams.width_points),
        stop_credit_multiple=getattr(section, "stop_credit_multiple",
                                     OdtePremiumParams.stop_credit_multiple),
        per_trade_risk_fraction=getattr(section, "per_trade_risk_fraction",
                                        OdtePremiumParams.per_trade_risk_fraction),
    ))


register(
    StrategyMeta(
        name="odte_premium",
        module="catalyst.strategies.active.odte_premium",
        status="active",
        notes=("v12/O1. 0DTE SPXW defined-risk put-spread premium selling. "
               "Quote-driven point-in-time selection (parity spot + straddle "
               "implied move); no chains, no greeks, ~6 requests/session. "
               "Short structural variance premium; no direction call."),
    ),
    build,
)

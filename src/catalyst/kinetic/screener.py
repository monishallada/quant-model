"""State 1 + State 2 — catalyst identification and signal generation.

State 1 (pre-market scan). For a symbol reacting to earnings on ``day``:

    G      = (P_pm_last - P_close) / P_close
    P_close    = prior regular-session close (last RTH minute bar before day)
    P_pm_last  = last pre-market trade price, window [04:00, 09:30)
    V_pm       = cumulative pre-market volume over that same window

    Watchlist requires |G| >= min_abs_gap AND V_pm >= min_premarket_volume.

State 2 (volatility bypass). Trading is halted 09:30–09:35 to let the
earnings IV premium collapse. At exactly 09:35:00 the first 5-minute candle
is evaluated:

    P_open  = open of the 09:30 minute bar
    P_0935  = close of the 09:34 minute bar  (i.e. last print < 09:35:00)
    dM      = P_0935 - P_open

    FIRE_LONG  : G >= +min_abs_gap AND dM > 0
    FIRE_SHORT : G <= -min_abs_gap AND dM < 0
    ABORT      : momentum contradicts the gap

Lookahead discipline (audited): every input above is drawn from bars whose
timestamps are strictly earlier than the 09:35:00 decision point. The scan
window ends at 09:29:59 and the momentum window ends at 09:34:59; nothing
from 09:35 onward touches the decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from catalyst.core.config import KineticConfig
from catalyst.data.intraday import AlpacaMinuteBars

logger = logging.getLogger(__name__)


def _parse_time(text: str) -> time:
    hh, mm = text.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class KineticSignal:
    """Full State-1/State-2 evaluation for one symbol-day."""

    symbol: str
    session: date
    prior_close: float | None
    premarket_last: float | None
    premarket_volume: int
    gap: float | None
    price_open: float | None
    price_0935: float | None
    momentum: float | None
    fires: bool
    direction: str | None  # "long" | "short" | None
    reject_reason: str | None

    @property
    def watchlisted(self) -> bool:
        """Passed State 1 (gap + liquidity), regardless of State 2 outcome."""
        return self.reject_reason not in {"no_data", "no_prior_close", "gap_too_small",
                                          "premarket_volume_too_low"}


class KineticScreener:
    """Evaluates State 1 + State 2. Results are memoized per symbol-day."""

    def __init__(self, cfg: KineticConfig, bars: AlpacaMinuteBars,
                 daily_closes: dict[str, "pd.Series"] | None = None) -> None:
        self._cfg = cfg
        self._bars = bars
        # Prior closes come from one cached daily series per symbol (same feed
        # and split adjustment as the minute bars), not a per-day minute pull.
        self._daily_closes = daily_closes if daily_closes is not None else {}
        self._pm_start = _parse_time(cfg.scan.premarket_start)
        self._pm_end = _parse_time(cfg.scan.premarket_end)
        self._session_open = _parse_time(cfg.signal.session_open)
        self._signal_time = _parse_time(cfg.signal.signal_time)
        self._memo: dict[tuple[str, date], KineticSignal] = {}

    def _empty(self, symbol: str, session: date, reason: str) -> KineticSignal:
        return KineticSignal(symbol=symbol, session=session, prior_close=None,
                             premarket_last=None, premarket_volume=0, gap=None,
                             price_open=None, price_0935=None, momentum=None,
                             fires=False, direction=None, reject_reason=reason)

    def evaluate(self, symbol: str, session: date) -> KineticSignal:
        memo_key = (symbol, session)
        if memo_key in self._memo:
            return self._memo[memo_key]
        result = self._evaluate_uncached(symbol, session)
        self._memo[memo_key] = result
        return result

    def _evaluate_uncached(self, symbol: str, session: date) -> KineticSignal:
        cfg = self._cfg
        bars = self._bars.get_day(symbol, session)
        if bars.empty:
            return self._empty(symbol, session, "no_data")

        prior_close = self._bars.prior_close(symbol, session,
                                             self._daily_closes.get(symbol))
        if prior_close is None or prior_close <= 0:
            return self._empty(symbol, session, "no_prior_close")

        # --- State 1: pre-market window [04:00, 09:30) ---
        pm = bars[(bars.index.time >= self._pm_start) & (bars.index.time < self._pm_end)]
        if pm.empty:
            return self._empty(symbol, session, "no_premarket")
        premarket_last = float(pm["close"].iloc[-1])
        premarket_volume = int(pm["volume"].sum())
        gap = (premarket_last - prior_close) / prior_close

        base = dict(symbol=symbol, session=session, prior_close=prior_close,
                    premarket_last=premarket_last, premarket_volume=premarket_volume,
                    gap=gap)
        if abs(gap) < cfg.scan.min_abs_gap:
            return KineticSignal(**base, price_open=None, price_0935=None, momentum=None,
                                 fires=False, direction=None, reject_reason="gap_too_small")
        if premarket_volume < cfg.scan.min_premarket_volume:
            return KineticSignal(**base, price_open=None, price_0935=None, momentum=None,
                                 fires=False, direction=None,
                                 reject_reason="premarket_volume_too_low")

        # --- State 2: first 5-minute candle [09:30:00, 09:35:00) ---
        window = bars[(bars.index.time >= self._session_open)
                      & (bars.index.time < self._signal_time)]
        if window.empty:
            return KineticSignal(**base, price_open=None, price_0935=None, momentum=None,
                                 fires=False, direction=None, reject_reason="no_opening_candle")
        price_open = float(window["open"].iloc[0])
        price_0935 = float(window["close"].iloc[-1])
        momentum = price_0935 - price_open

        if gap >= cfg.scan.min_abs_gap and momentum > 0:
            direction, reason = "long", None
        elif gap <= -cfg.scan.min_abs_gap and momentum < 0:
            direction, reason = "short", None
        else:
            direction, reason = None, "momentum_contradicts_gap"

        return KineticSignal(**base, price_open=price_open, price_0935=price_0935,
                             momentum=momentum, fires=direction is not None,
                             direction=direction, reject_reason=reason)

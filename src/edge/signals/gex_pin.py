"""GexPinning: the OI-only GEX regime bet — run as a test, not assumed.

THE SIGN ASSUMPTION IS THE HYPOTHESIS UNDER TEST. Open interest carries no
sign: it counts contracts outstanding, not who is short them. The classic
dealer-pinning story requires dealers to be NET LONG the gamma concentrated
near spot; OI-weighted gamma cannot verify that. This signal deliberately
bets the classic sign anyway, because the v16 research said to RUN the
regime bet rather than assume it either way: if the OOS band shows nothing,
that kills OI-only GEX as alpha for this repo — and that finding is the
point of the trial.

Mechanism traded (stated honestly, caveat included, in :data:`HYPOTHESIS`):
IF dealers are net long gamma where OI-weighted gamma is high near spot,
their delta-hedging (sell rallies, buy dips) damps moves toward the
max-gamma strike into the expiry afternoon. The trade fades distance to the
pin on expiry days: SELL above the pin, BUY below it, only when
``|spot - max_gamma_strike| < 0.5%`` of spot, in the afternoon (12:00 ET
onward) with 90+ minutes to the 16:00 close, horizon to 15:45 ET. The other
side is the directional afternoon breakout bettor paying for continuation
that dealer re-hedging statistically absorbs. All thresholds are FIXED
economic choices written into the hypothesis — never swept; a variant is
its own trial.

Data (through the engine-provided ctx ONLY — this module never touches a
loader or backend): the CatalystBridge kinds

* ``greeks_eod`` — per-expiry EOD greeks; symbol grammar
  ``"UNDERLYING:YYYY-MM-DD"`` (the expiry). ``available_at`` = asof 18:00
  ET, so intraday on day D only D-1-and-earlier greeks are visible.
* ``open_interest`` — per-expiry per-day OI, same grammar.
  ``available_at`` = NEXT business day 09:00 ET (OCC overnight cycle):
  day-D OI is NEVER usable on day D. On expiry day the freshest visible
  snapshot is D-1's OI, published that morning.

The ctx pre-filters frames to ``available_at <= decision_ts``; this signal
re-filters anyway (defense in depth — a leaked day-D OI row must never move
the pin). Expiry-day detection is structural: the signal requests both
kinds for expiry == the current ET session date; a non-expiry day has no
such listed expiration, the frames come back empty, and the signal cannot
fire — non-expiry days never fire by construction.

Regimes: ``allowed_regimes = {"low", "mid"}`` — the regime classifier's
``vol_state`` axis (:data:`edge.regime.classifier.VOL_LOW` /
:data:`~edge.regime.classifier.VOL_MID`). Pinning is a calm-tape mechanism;
in high/stressed vol the directional flow overwhelms dealer hedge flow.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Final

import pandas as pd

from edge.core.events import MARKET_TZ, Side, SignalEvent
from edge.regime.classifier import VOL_LOW, VOL_MID
from edge.signals.base import Signal, SignalConfig, SignalContext
from edge.signals.registry import register

__all__ = ["GexPinning", "GexPinConfig", "HYPOTHESIS"]

# ---------------------------------------------------------------------------
# Fixed economic choices (written into the hypothesis; never swept)
# ---------------------------------------------------------------------------

#: Regular-session close (ET); the "minutes to close" gate measures to this.
SESSION_CLOSE: Final[time] = time(16, 0)

#: The fade's horizon target: hold to 15:45 ET, before the closing auction
#: unwind can blow the pin apart.
EXIT_TIME: Final[time] = time(15, 45)

#: "Expiry afternoons": the mechanism concentrates as dealer hedge
#: rebalancing accelerates into expiry; entries start at noon ET.
AFTERNOON_START: Final[time] = time(12, 0)

#: The exact hypothesis string registered for this signal (>= 50 chars, and
#: the full honest claim: mechanism, sign caveat, counterparty, thresholds).
HYPOTHESIS: Final[str] = (
    "GEX pinning regime bet — the dealer-long-gamma SIGN ASSUMPTION is the "
    "hypothesis under test: open interest carries no sign, so OI-weighted "
    "gamma cannot verify who is short. IF dealers are net long the gamma "
    "concentrated near spot on expiry day, their delta-hedging (sell "
    "rallies, buy dips) damps moves toward the max-gamma strike into the "
    "expiry afternoon; the trade fades distance to that pin — short above "
    "it, long below it — only on expiry days when |spot - max_gamma_strike| "
    "< 0.5% of spot, in the afternoon (12:00 ET onward) with 90+ minutes to "
    "the 16:00 close, exiting by 15:45 ET, in low/mid vol regimes only "
    "(stress overwhelms hedge flow). The other side is the directional "
    "afternoon breakout bettor who keeps paying for continuation that "
    "dealer re-hedging statistically absorbs. Uses only prior-session EOD "
    "greeks and prior-session OI first published next-morning 09:00 ET. If "
    "OOS shows nothing, OI-only GEX is dead as alpha for this repo — that "
    "finding is the point."
)


class GexPinConfig(SignalConfig):
    """GexPinning's fixed thresholds. Defaults ARE the economic choice
    (written into :data:`HYPOTHESIS`); a different value is a different
    trial, never a sweep."""

    #: Fire only when |spot - pin| is under this percent of spot.
    max_pin_distance_pct: float = 0.5
    #: Fire only with at least this many minutes to the 16:00 ET close.
    min_minutes_to_close: int = 90


# ---------------------------------------------------------------------------
# ctx compatibility + point-in-time hygiene helpers
# ---------------------------------------------------------------------------


def _decision_ts(ctx: SignalContext) -> datetime:
    """The decision instant: ``ctx.decision_ts`` (base protocol) or
    ``ctx.now`` (the engine's MarketContext spelling)."""
    ts = getattr(ctx, "decision_ts", None)
    if ts is None:
        ts = getattr(ctx, "now", None)
    if not isinstance(ts, datetime):
        raise TypeError("ctx exposes neither decision_ts nor now as a datetime")
    return ts


def _load_frame(ctx: SignalContext, kind: str, symbol: str) -> pd.DataFrame:
    """Point-in-time frame via ``ctx.frame`` (base protocol) or ``ctx.pit``."""
    frame_fn = getattr(ctx, "frame", None)
    if callable(frame_fn):
        return frame_fn(kind, symbol)
    pit_fn = getattr(ctx, "pit", None)
    if callable(pit_fn):
        return pit_fn(kind, symbol)
    raise TypeError("ctx exposes neither frame() nor pit() for point-in-time data")


def _et_instant(day: date, at: time) -> datetime:
    """Tz-aware ET wall-clock instant on ``day`` (DST-safe)."""
    return datetime.combine(day, at, tzinfo=MARKET_TZ)


def _latest_visible_snapshot(
    frame: pd.DataFrame, now: datetime, kind: str
) -> pd.DataFrame | None:
    """Rows of the freshest ``asof_date`` whose ``available_at <= now``.

    Defense in depth: the ctx already filters on ``available_at``, but a
    leaked same-day OI row must never move the pin, so the visibility join
    is re-applied here. Empty in, or nothing visible -> ``None`` (no
    verified surface, no trade). A non-empty frame missing the
    point-in-time columns raises — an unstampable frame cannot be trusted.
    """
    if frame is None or frame.empty:
        return None
    for column in ("asof_date", "available_at"):
        if column not in frame.columns:
            raise ValueError(
                f"{kind} frame is missing point-in-time column {column!r}; "
                "cannot verify visibility"
            )
    available = pd.to_datetime(frame["available_at"])
    if available.dt.tz is None:
        raise ValueError(
            f"{kind} frame carries naive available_at stamps; the visibility "
            "join needs tz-aware (America/New_York) instants"
        )
    visible = frame[available <= pd.Timestamp(now)]
    if visible.empty:
        return None
    asof = pd.to_datetime(visible["asof_date"])
    snapshot = visible[asof == asof.max()]
    return snapshot.reset_index(drop=True)


def _normalize_right(values: pd.Series) -> pd.Series:
    """Map C/CALL/call -> 'C', P/PUT/put -> 'P' so frames join cleanly."""
    return values.astype(str).str.strip().str.upper().str[0]


def _max_gamma_strike(greeks: pd.DataFrame, oi: pd.DataFrame) -> float | None:
    """The pin: strike maximizing sum over rights of |gamma| * open_interest.

    Both inputs are latest-visible snapshots. Joins on (strike, right) with
    normalized right spellings; duplicate (strike, right) rows keep the last
    greek and sum the OI. Returns ``None`` when the join is empty or the
    total weight is not positive (an unweighted surface pins nothing).
    Non-empty frames missing their value columns raise — silence would kill
    the signal invisibly.
    """
    for column in ("strike", "right", "gamma"):
        if column not in greeks.columns:
            raise ValueError(f"greeks_eod snapshot is missing column {column!r}")
    for column in ("strike", "right", "open_interest"):
        if column not in oi.columns:
            raise ValueError(f"open_interest snapshot is missing column {column!r}")
    g = pd.DataFrame(
        {
            "strike": greeks["strike"].astype(float),
            "right": _normalize_right(greeks["right"]),
            "gamma": greeks["gamma"].astype(float).abs(),
        }
    ).drop_duplicates(["strike", "right"], keep="last")
    o = (
        pd.DataFrame(
            {
                "strike": oi["strike"].astype(float),
                "right": _normalize_right(oi["right"]),
                "open_interest": oi["open_interest"].astype(float),
            }
        )
        .groupby(["strike", "right"], as_index=False)["open_interest"]
        .sum()
    )
    joined = g.merge(o, on=["strike", "right"], how="inner")
    if joined.empty:
        return None
    joined["weight"] = joined["gamma"] * joined["open_interest"]
    by_strike = joined.groupby("strike")["weight"].sum().dropna()
    if by_strike.empty or float(by_strike.max()) <= 0.0:
        return None
    return float(by_strike.idxmax())


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------


@register
class GexPinning(Signal):
    """Fade distance to the OI-weighted max-gamma strike on expiry afternoons.

    See the module docstring and :data:`HYPOTHESIS` for the full (caveated)
    claim. Emits at most one event per (symbol, ET session): the pin fade is
    one afternoon thesis, not a stream — re-arguing it every minute would
    only inflate emitted-signal counts into the drop counters.
    """

    name = "gex_pin"
    hypothesis = HYPOTHESIS
    #: Regime classifier vol_state axis: calm tape only.
    allowed_regimes = frozenset({VOL_LOW, VOL_MID})
    warmup_bars = 0
    config_type = GexPinConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        super().__init__(config)
        #: (symbol, ET session date) pairs already traded this run.
        self._fired: set[tuple[str, date]] = set()

    def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
        now = _decision_ts(ctx)
        et_now = now.astimezone(MARKET_TZ)
        session = et_now.date()
        bar = ctx.bar
        symbol = bar.symbol

        if (symbol, session) in self._fired:
            return []
        # Expiry AFTERNOON only, with 90+ minutes to the close.
        if et_now.time() < AFTERNOON_START:
            return []
        cfg: GexPinConfig = self.config  # type: ignore[assignment]
        minutes_to_close = (
            _et_instant(session, SESSION_CLOSE) - et_now
        ).total_seconds() / 60.0
        if minutes_to_close < cfg.min_minutes_to_close:
            return []

        # Expiry == the current session date: on a non-expiry day no such
        # expiration is listed, both frames are empty, and nothing can fire.
        expiry_symbol = f"{symbol}:{session.isoformat()}"
        oi = _latest_visible_snapshot(
            _load_frame(ctx, "open_interest", expiry_symbol), now, "open_interest"
        )
        if oi is None:
            return []
        greeks = _latest_visible_snapshot(
            _load_frame(ctx, "greeks_eod", expiry_symbol), now, "greeks_eod"
        )
        if greeks is None:
            return []

        pin = _max_gamma_strike(greeks, oi)
        if pin is None:
            return []
        spot = float(bar.close)
        distance_pct = (spot - pin) / spot * 100.0
        # At the pin there is nothing to fade; beyond the band the pin is
        # not binding. Fire only strictly inside (0, max_pin_distance_pct).
        if distance_pct == 0.0 or abs(distance_pct) >= cfg.max_pin_distance_pct:
            return []
        side = Side.SELL if distance_pct > 0.0 else Side.BUY

        horizon_minutes = int(
            (_et_instant(session, EXIT_TIME) - et_now).total_seconds() // 60.0
        )
        if horizon_minutes <= 0:  # unreachable under the 90-minute gate; belt.
            return []

        self._fired.add((symbol, session))
        return [
            SignalEvent(
                ts=now,
                symbol=symbol,
                side=side,
                conviction=1.0,
                horizon_minutes=horizon_minutes,
                signal_name=self.name,
            )
        ]

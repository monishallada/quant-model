"""Session regime classification: fixed economic definitions, ZERO fitted parameters.

DESIGN PRINCIPLE — nothing in this module is fitted, tuned, or estimated
from returns. A measured lesson from this repo: fitted regime thresholds
overfit the sample they were tuned on; fixed economic definitions transfer.
Every number below is a *definition* (terciles, the median, a 21-session
window, 1x ATR), declared as a module constant and deliberately NOT exposed
through config — putting them in ``config/edge.yaml`` would invite sweeping
them, which is exactly the failure mode this module exists to avoid.

Inputs come ONLY through :class:`edge.data.loader.EdgeDataLoader` (the
sanctioned gateway; this module is inside the AST gateway scan), via two
kinds:

* ``vol_indices`` — the CBOE point-in-time frame (``asof_date``,
  ``available_at``, ``vix``, ``vix9d``, ...). Market-wide: the loader's
  ``symbol`` argument is a placeholder (:data:`MARKET_WIDE_SYMBOL`).
* ``bars`` — price history for the trend index (default SPY), normalized
  here to one bar per ET session (minute bars are aggregated; daily bars
  pass through). ``high``/``low`` are optional — absent, the true range
  degrades to close-to-close moves.

POINT-IN-TIME CONTRACT (non-negotiable)
---------------------------------------
Each session ``s`` is classified using ONLY rows whose ``available_at``
strictly precedes the session open (09:30 ET on ``s``). Both inputs are
evening-published — CBOE settles after the close and daily bars complete at
the close — so a session's classification uses data through the PRIOR
session, and the output row's ``available_at`` is stamped 18:00 ET on the
prior usable session (:data:`EVENING_PUBLISH_HOUR`; formally: the latest
``available_at`` among the input rows actually used, floored at that prior-
session evening stamp — never earlier than an input it depends on, and
always before the session open). Bars frames without an ``available_at``
column receive the same evening convention: a session's bar is first usable
the NEXT session. The expanding VIX percentile is prefix-invariant by
construction: truncating the future never changes a past session's state.

STATE DEFINITIONS (fixed, economic)
-----------------------------------
``vol_state``:

* ``'stressed_backwardation'`` — VIX9D/VIX ratio > 1: the 9-day expected
  vol exceeds the 30-day, i.e. the short end of the vol curve is inverted,
  which happens only under acute stress. Checked FIRST; a missing VIX9D
  never triggers it (falls through to the percentile bands).
* otherwise, the midrank percentile of the latest VIX close within its own
  FULL history to date (expanding window — every usable observation since
  the start of the loaded history, never a fitted lookback):
  ``< 33`` -> ``'low'``, ``< 67`` -> ``'mid'``, else ``'high'``.
  Midrank percentile of the last value ``x`` in history ``H`` (which
  includes ``x``): ``100 * (#{H < x} + #{H <= x}) / (2 * |H|)`` —
  scipy's ``kind='mean'`` convention; an all-constant history sits at the
  50th percentile rather than the 0th or 100th.

``trend_state``: the 21-session price change of the index vs +/- 1x its
ATR(21): ``change > ATR`` -> ``'up'``, ``change < -ATR`` -> ``'down'``,
else ``'chop'``. ATR is the simple mean of the true range over the last 21
sessions (``TR = max(high-low, |high-prev_close|, |low-prev_close|)``).
Fewer than 22 usable sessions of history -> ``'chop'`` (the neutral
default; warmup is never silently classified as trending).

PROMOTION BUCKETS (the 2x2 the promotion gates key on)
------------------------------------------------------
``bucket`` crosses a vol split with a trend split:

* vol: expanding VIX percentile >= 50 (the median of history to date)
  -> ``high_vol``, else ``low_vol``. Note this uses the percentile even
  when ``vol_state`` is ``'stressed_backwardation'``.
* trend: ``trend_state != 'chop'`` -> ``trending``, else ``chopping``.

giving :data:`BUCKETS` = ``{high_vol,low_vol} x {trending,chopping}``.

Sessions with no usable VIX observation or no usable bar (nothing published
before their open) are omitted from the output — an unclassifiable session
is dropped loudly-by-absence, never defaulted.

Engine hook: :func:`gate` answers "may this signal trade this session?"
from a signal's declared allowed buckets; unknown bucket names raise —
a typo in a regime list must never silently gate a signal off forever.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time
from typing import Final

import numpy as np
import pandas as pd

from edge.core.events import MARKET_TZ
from edge.data.loader import EdgeDataLoader

# ---------------------------------------------------------------------------
# Fixed definitions (constants, NOT config: nothing here may ever be swept)
# ---------------------------------------------------------------------------

#: Regular-session open (ET); the point-in-time cutoff for a session's inputs.
SESSION_OPEN: Final[time] = time(9, 30)

#: Evening-publication hour (ET) stamped on daily inputs and on the output
#: rows — the same conservative 18:00 convention the CBOE feed uses.
EVENING_PUBLISH_HOUR: Final[int] = 18

#: Sessions in the trend window: 21-session change vs ATR over 21 sessions.
TREND_WINDOW: Final[int] = 21

#: The trend test is +/- exactly one ATR — a definition, not a tunable.
ATR_MULTIPLE: Final[float] = 1.0

#: Expanding-percentile tercile bands for vol_state.
LOW_VOL_PCTILE: Final[float] = 33.0
HIGH_VOL_PCTILE: Final[float] = 67.0

#: Median split for the promotion buckets' vol axis.
BUCKET_VOL_PCTILE: Final[float] = 50.0

#: VIX history begins 1990; the default expanding-percentile history start.
DEFAULT_HISTORY_START: Final[date] = date(1990, 1, 2)

#: Placeholder symbol for the market-wide ``vol_indices`` kind (ignored by
#: the loader for market-wide kinds, but the call signature requires one).
MARKET_WIDE_SYMBOL: Final[str] = "MKT"

# Vocabulary.
VOL_STRESSED: Final[str] = "stressed_backwardation"
VOL_LOW: Final[str] = "low"
VOL_MID: Final[str] = "mid"
VOL_HIGH: Final[str] = "high"
VOL_STATES: Final[tuple[str, ...]] = (VOL_STRESSED, VOL_LOW, VOL_MID, VOL_HIGH)

TREND_UP: Final[str] = "up"
TREND_DOWN: Final[str] = "down"
TREND_CHOP: Final[str] = "chop"
TREND_STATES: Final[tuple[str, ...]] = (TREND_UP, TREND_DOWN, TREND_CHOP)

#: The four promotion buckets: {high_vol,low_vol} x {trending,chopping}.
BUCKETS: Final[frozenset[str]] = frozenset(
    {
        "high_vol_trending",
        "high_vol_chopping",
        "low_vol_trending",
        "low_vol_chopping",
    }
)

#: Output columns of :func:`classify`, in order.
REGIME_COLUMNS: Final[tuple[str, ...]] = (
    "session",
    "vol_state",
    "trend_state",
    "bucket",
    "vix_pctile",
    "available_at",
)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _et_instant(day: date, at: time) -> pd.Timestamp:
    """Tz-aware ET wall-clock instant on ``day`` (DST-safe: localize last)."""
    return pd.Timestamp(datetime.combine(day, at)).tz_localize(MARKET_TZ)


def _expanding_percentile(values: np.ndarray) -> float:
    """Midrank percentile of the LAST element within the whole array.

    ``100 * (#{v < x} + #{v <= x}) / (2n)`` where ``x = values[-1]`` and the
    array (history to date) includes ``x`` itself. Depends only on data up
    to and including ``x`` — prefix-invariant by construction.
    """
    current = values[-1]
    less = int(np.count_nonzero(values < current))
    less_or_equal = int(np.count_nonzero(values <= current))
    return 100.0 * (less + less_or_equal) / (2.0 * values.size)


def _vol_state_from(pctile: float, vix: float, vix9d: float) -> str:
    """Apply the fixed vol_state definition (see module docstring)."""
    if np.isfinite(vix9d) and vix > 0.0 and vix9d / vix > 1.0:
        return VOL_STRESSED
    if pctile < LOW_VOL_PCTILE:
        return VOL_LOW
    if pctile < HIGH_VOL_PCTILE:
        return VOL_MID
    return VOL_HIGH


def _trend_state_from(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> str:
    """21-session change vs +/- 1x ATR(21); warmup (< 22 sessions) is chop."""
    if closes.size < TREND_WINDOW + 1:
        return TREND_CHOP
    c = closes[-(TREND_WINDOW + 1):]
    h = highs[-(TREND_WINDOW + 1):]
    lo = lows[-(TREND_WINDOW + 1):]
    prev_close = c[:-1]
    true_range = np.maximum(
        h[1:] - lo[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(lo[1:] - prev_close)),
    )
    atr = float(true_range.mean())
    change = float(c[-1] - c[0])
    if change > ATR_MULTIPLE * atr:
        return TREND_UP
    if change < -ATR_MULTIPLE * atr:
        return TREND_DOWN
    return TREND_CHOP


def _bucket_from(pctile: float, trend_state: str) -> str:
    """Cross the median vol split with the trending/chopping split."""
    vol_half = "high_vol" if pctile >= BUCKET_VOL_PCTILE else "low_vol"
    trend_half = "trending" if trend_state != TREND_CHOP else "chopping"
    return f"{vol_half}_{trend_half}"


def _as_session_date(value: date | datetime, name: str) -> date:
    """Normalize a range bound to an ET calendar date (naive datetimes raise)."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{name} must be a plain date or a tz-aware datetime; "
                "a naive datetime has no defensible ET session date"
            )
        return value.astimezone(MARKET_TZ).date()
    return value


# ---------------------------------------------------------------------------
# Input normalization (loader frames -> tidy per-session inputs)
# ---------------------------------------------------------------------------


def _aware_et(series: pd.Series, what: str) -> pd.Series:
    """Coerce an ``available_at`` series to tz-aware ET; naive stamps raise."""
    stamps = pd.to_datetime(series)
    if stamps.dt.tz is None:
        if len(stamps):
            raise ValueError(f"{what} 'available_at' must be tz-aware (America/New_York)")
        return stamps.dt.tz_localize(MARKET_TZ)
    return stamps.dt.tz_convert(MARKET_TZ)


def _vol_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Tidy the ``vol_indices`` frame: one row per asof_date, ascending.

    Requires ``asof_date``, ``available_at``, ``vix``; ``vix9d`` is optional
    (absent -> NaN, so the stressed-backwardation branch can never fire).
    Rows without a VIX value carry no information and are dropped.
    """
    missing = [c for c in ("asof_date", "available_at", "vix") if c not in frame.columns]
    if missing:
        raise ValueError(f"vol_indices frame is missing required columns {missing}")
    work = frame.reset_index(drop=True)
    out = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(work["asof_date"]),
            "vix": work["vix"].astype(float),
            "vix9d": work["vix9d"].astype(float) if "vix9d" in work.columns else np.nan,
            "available_at": _aware_et(work["available_at"], "vol_indices"),
        }
    )
    out = out[out["vix"].notna()]
    out = out.sort_values("asof_date", kind="stable").drop_duplicates("asof_date", keep="last")
    return out.reset_index(drop=True)


def _bar_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Locate the per-row timestamps of a bars frame (column or index)."""
    if "asof_date" in frame.columns:
        return pd.to_datetime(frame["asof_date"]).reset_index(drop=True)
    if "ts" in frame.columns:
        return pd.to_datetime(frame["ts"]).reset_index(drop=True)
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index)
    raise ValueError(
        "bars frame needs an 'asof_date' or 'ts' column, or a DatetimeIndex, "
        "to place each bar on an ET session"
    )


def _daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a bars frame to one row per ET session.

    Output columns: ``day`` (python date), ``high``, ``low``, ``close``,
    ``available_at`` (tz-aware ET). Minute bars aggregate (high=max,
    low=min, close=last); daily bars pass through. Missing ``high``/``low``
    degrade to the close. ``available_at``: the LATER of 18:00 ET on the
    bar's own session (the evening-published convention) and the frame's
    own per-day max ``available_at`` when it carries one — the stamp never
    claims availability earlier than the evening convention, and never
    earlier than the source's own stamp.
    """
    if "close" not in frame.columns:
        raise ValueError("bars frame is missing required column 'close'")
    ts = _bar_timestamps(frame)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(MARKET_TZ)
    work = frame.reset_index(drop=True)
    close = work["close"].astype(float)
    high = work["high"].astype(float).fillna(close) if "high" in work.columns else close
    low = work["low"].astype(float).fillna(close) if "low" in work.columns else close
    has_avail = "available_at" in work.columns
    tidy = pd.DataFrame(
        {"_ts": ts, "day": ts.dt.date, "high": high, "low": low, "close": close}
    )
    if has_avail:
        tidy["avail"] = _aware_et(work["available_at"], "bars")
    tidy = tidy[tidy["close"].notna()].sort_values("_ts", kind="stable")
    named = {
        "high": pd.NamedAgg("high", "max"),
        "low": pd.NamedAgg("low", "min"),
        "close": pd.NamedAgg("close", "last"),
    }
    if has_avail:
        named["src_avail"] = pd.NamedAgg("avail", "max")
    daily = tidy.groupby("day", sort=True).agg(**named).reset_index()
    evening = (
        pd.to_datetime(daily["day"]) + pd.Timedelta(hours=EVENING_PUBLISH_HOUR)
    ).dt.tz_localize(MARKET_TZ)
    if has_avail:
        src = daily["src_avail"]
        daily["available_at"] = src.where(src.notna() & (src > evening), evening)
        daily = daily.drop(columns="src_avail")
    else:
        daily["available_at"] = evening
    return daily[["day", "high", "low", "close", "available_at"]]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": pd.Series(dtype="datetime64[ns]"),
            "vol_state": pd.Series(dtype=object),
            "trend_state": pd.Series(dtype=object),
            "bucket": pd.Series(dtype=object),
            "vix_pctile": pd.Series(dtype=float),
            "available_at": pd.Series(pd.DatetimeIndex([], tz=MARKET_TZ)),
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(
    loader: EdgeDataLoader,
    start: date | datetime,
    end: date | datetime,
    symbol_index: str = "SPY",
    *,
    history_start: date = DEFAULT_HISTORY_START,
) -> pd.DataFrame:
    """Classify every ``symbol_index`` session in ``[start, end]``, point-in-time.

    Data flows only through ``loader`` (kinds ``vol_indices`` + ``bars``),
    requested from ``history_start`` so the expanding VIX percentile sees
    its FULL history to date — an expanding window, never a fitted lookback.
    Each session uses only input rows with ``available_at`` strictly before
    its 09:30 ET open (see module docstring for every definition).

    Returns a frame with :data:`REGIME_COLUMNS`:

    * ``session`` — naive midnight Timestamp of the classified session,
    * ``vol_state`` — one of :data:`VOL_STATES`,
    * ``trend_state`` — one of :data:`TREND_STATES`,
    * ``bucket`` — one of :data:`BUCKETS` (the promotion 2x2),
    * ``vix_pctile`` — the expanding midrank percentile behind both vol
      classifications (diagnostic),
    * ``available_at`` — tz-aware ET instant the row became knowable:
      18:00 ET on the prior usable session under the evening-published
      conventions, always strictly before the session open.

    Sessions with no usable VIX row or no usable bar are omitted.
    """
    start_d = _as_session_date(start, "start")
    end_d = _as_session_date(end, "end")
    if start_d > end_d:
        raise ValueError(f"empty range: start {start_d} is after end {end_d}")

    vol = _vol_history(loader.load(MARKET_WIDE_SYMBOL, history_start, end_d, "vol_indices"))
    daily = _daily_bars(loader.load(symbol_index, history_start, end_d, "bars"))

    records: list[dict[str, object]] = []
    for session_day in daily["day"]:
        if not start_d <= session_day <= end_d:
            continue
        session_open = _et_instant(session_day, SESSION_OPEN)
        v_use = vol[vol["available_at"] < session_open]
        b_use = daily[daily["available_at"] < session_open]
        if v_use.empty or b_use.empty:
            continue

        pctile = _expanding_percentile(v_use["vix"].to_numpy(dtype=float))
        latest = v_use.iloc[-1]
        vol_state = _vol_state_from(pctile, float(latest["vix"]), float(latest["vix9d"]))
        trend_state = _trend_state_from(
            b_use["close"].to_numpy(dtype=float),
            b_use["high"].to_numpy(dtype=float),
            b_use["low"].to_numpy(dtype=float),
        )

        prior_session: date = b_use["day"].iloc[-1]
        stamp = max(
            _et_instant(prior_session, time(EVENING_PUBLISH_HOUR, 0)),
            v_use["available_at"].max(),
            b_use["available_at"].max(),
        )
        records.append(
            {
                "session": pd.Timestamp(session_day),
                "vol_state": vol_state,
                "trend_state": trend_state,
                "bucket": _bucket_from(pctile, trend_state),
                "vix_pctile": pctile,
                "available_at": stamp,
            }
        )
    if not records:
        return _empty_result()
    return pd.DataFrame(records)[list(REGIME_COLUMNS)]


def gate(signal_allowed_regimes: str | Iterable[str] | None, session_bucket: str) -> bool:
    """May a signal trade this session? The engine's regime hook.

    ``signal_allowed_regimes`` is the signal's declared set of allowed
    promotion buckets (a single bucket name is accepted as shorthand):

    * ``None`` — the signal opted out of regime gating: always allowed.
    * an empty collection — the signal allows NO regime: always blocked.
    * otherwise — allowed iff ``session_bucket`` is in the set.

    Every name (the session's bucket and each declared bucket) must be one
    of :data:`BUCKETS`; anything else raises ``ValueError`` — a typo in a
    regime list must fail loudly, never silently gate a signal off forever.
    """
    if session_bucket not in BUCKETS:
        raise ValueError(
            f"unknown session bucket {session_bucket!r}; expected one of {sorted(BUCKETS)}"
        )
    if signal_allowed_regimes is None:
        return True
    if isinstance(signal_allowed_regimes, str):
        allowed = {signal_allowed_regimes}
    else:
        allowed = set(signal_allowed_regimes)
    unknown = allowed - BUCKETS
    if unknown:
        raise ValueError(
            f"unknown regime bucket(s) {sorted(unknown)} in allowed set; "
            f"expected only {sorted(BUCKETS)}"
        )
    return session_bucket in allowed

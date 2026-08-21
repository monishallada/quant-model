"""NYSE trading calendar helpers (tz-aware, America/New_York).

All engine timing logic ("catalyst N trading days out", time stops, expiry
buffers) goes through these helpers so backtest and live agree exactly on
what a trading day is.
"""

from __future__ import annotations

import bisect
from datetime import date
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

_CAL_START = "2010-01-01"
_CAL_END = "2030-12-31"


@lru_cache(maxsize=1)
def _sessions() -> list[date]:
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=_CAL_START, end_date=_CAL_END)
    return [d.date() for d in pd.DatetimeIndex(sched.index)]


def is_trading_day(d: date) -> bool:
    days = _sessions()
    i = bisect.bisect_left(days, d)
    return i < len(days) and days[i] == d


def next_trading_day(d: date) -> date:
    """First trading day strictly after ``d``."""
    days = _sessions()
    i = bisect.bisect_right(days, d)
    return days[i]


def previous_trading_day(d: date) -> date:
    """Last trading day strictly before ``d``."""
    days = _sessions()
    i = bisect.bisect_left(days, d)
    if i == 0:
        # days[-1] would silently WRAP to the calendar's last session
        # (audit D-122: previous_trading_day(2010-01-04) returned 2026-xx)
        raise ValueError(f"{d} precedes the trading calendar's first session")
    return days[i - 1]


def add_trading_days(d: date, n: int) -> date:
    """The trading day ``n`` sessions after (or before, n<0) ``d``.

    If ``d`` is not itself a session, counting starts from the nearest session
    in the direction of travel.
    """
    days = _sessions()
    if n >= 0:
        i = bisect.bisect_left(days, d)
    else:
        i = bisect.bisect_right(days, d) - 1
    return days[i + n]


def trading_days_between(start: date, end: date) -> int:
    """Number of sessions strictly after ``start`` up to and including ``end``."""
    days = _sessions()
    return bisect.bisect_right(days, end) - bisect.bisect_right(days, start)


def sessions_in_range(start: date, end: date) -> list[date]:
    """All trading sessions in [start, end]."""
    days = _sessions()
    lo = bisect.bisect_left(days, start)
    hi = bisect.bisect_right(days, end)
    return days[lo:hi]


def session_close_time(day: date):
    """ET close time for a session (13:00 on early-close days, else 16:00).

    AUDIT HIGH: engines that assume 16:00 on half days trade phantom
    liquidity — ThetaData forward-fills the frozen 13:00 NBBO to 16:00, so
    stale-quote guards can never fire, and Alpaca post-market bars keep
    filling equity orders hours after the floor closed.
    """
    from datetime import time as _time
    import pandas as _pd

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=day, end_date=day)
    if sched.empty:
        # a holiday/weekend has NO close; a plausible 16:00 sent callers
        # confidently marking a non-session (audit D-207)
        raise ValueError(f"{day} is not a trading session; it has no close time")
    close_et = sched["market_close"].iloc[0].tz_convert("America/New_York")
    return close_et.time()

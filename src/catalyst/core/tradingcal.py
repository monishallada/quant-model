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

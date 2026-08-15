"""Synthetic catalysts for plumbing tests (Gate 1).

Real catalysts (FMP earnings/economic calendars) arrive in M2; the backtester
itself is catalyst-agnostic, so a deterministic weekly schedule exercises the
identical code path.
"""

from __future__ import annotations

from datetime import date, datetime, time

from catalyst.core.types import Catalyst, CatalystType
from catalyst.core.tradingcal import sessions_in_range


def weekly_synthetic_catalysts(
    symbol: str, start: date, end: date, weekday: int = 4, event_time: time = time(16, 0)
) -> list[Catalyst]:
    """One synthetic 'other' catalyst per week on ``weekday`` (Mon=0..Fri=4)."""
    out: list[Catalyst] = []
    for session in sessions_in_range(start, end):
        if session.weekday() == weekday:
            out.append(
                Catalyst(
                    symbol=symbol,
                    type=CatalystType.OTHER,
                    when=datetime.combine(session, event_time),
                    source="synthetic-weekly",
                )
            )
    return out

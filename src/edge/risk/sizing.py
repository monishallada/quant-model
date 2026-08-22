"""Position sizing: Kelly fractions, hard caps, correlation groups, loss halts.

This module is the only interpreter of :class:`edge.core.config.RiskConfig`.
Sizing is deliberately two-staged:

1. :func:`kelly_fraction` turns measured OOS edge (expectancy in R, win rate,
   payoff ratio) into a FULL-Kelly fraction of equity. A non-positive measured
   expectancy zeroes the bet regardless of what win rate and payoff imply —
   two independent estimates of edge must both be positive before any risk
   is taken.
2. :func:`size_position` / :func:`size_positions` apply the configured Kelly
   multiplier (``RiskConfig.kelly_fraction``, 0.25 by default) and then the
   per-trade HARD cap (``RiskConfig.per_trade_cap_pct``) — the cap always
   wins, regardless of how attractive Kelly says the bet is.

Correlation groups: symbols whose trailing 63-day daily-return correlation
exceeds 0.7 are transitively merged (connected components) into ONE group,
and the group's COMBINED risk counts as a single position against the
per-trade cap. Three highly correlated longs are, for risk purposes, one
position.

:func:`daily_loss_halt` / :class:`DailyLossHalt` implement the session halt:
once the day's PnL reaches ``-RiskConfig.daily_loss_halt_pct`` percent, NEW
entries are refused for the remainder of that session (the halt latches even
if PnL later recovers); a new session date resets it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from edge.core.config import RiskConfig
from edge.core.events import MARKET_TZ

#: Trailing window (trading days) for the pairwise return correlation used to
#: form correlation groups — one calendar quarter. Fixed by the risk mandate,
#: not a tunable; override per call only in tests.
CORRELATION_WINDOW_DAYS: int = 63

#: Pairwise correlation above which two symbols are merged into one
#: correlation group. Fixed by the risk mandate (strictly greater than).
CORRELATION_GROUP_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------


def kelly_fraction(oos_expectancy_R: float, oos_win_rate: float, payoff_ratio: float) -> float:
    """Full-Kelly fraction of equity to risk, from measured OOS statistics.

    For a bet that loses 1R with probability ``q = 1 - p`` and wins
    ``payoff_ratio`` R with probability ``p``, the growth-optimal (full
    Kelly) fraction is ``f* = p - q / b`` with ``b = payoff_ratio``.

    ``oos_expectancy_R`` (mean R-multiple per trade, measured out of sample)
    acts as an independent edge gate: if it is not strictly positive the
    function returns 0.0 even when ``p`` and ``b`` imply a positive Kelly —
    inconsistent measurements mean the edge is not trusted. Likewise a
    non-positive Kelly formula result returns 0.0 (never a negative size).

    Returns:
        Full-Kelly fraction in ``[0, 1]``.

    Raises:
        ValueError: ``oos_win_rate`` outside [0, 1], non-positive
            ``payoff_ratio``, or non-finite inputs.
    """
    for name, value in (
        ("oos_expectancy_R", oos_expectancy_R),
        ("oos_win_rate", oos_win_rate),
        ("payoff_ratio", payoff_ratio),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if not 0.0 <= oos_win_rate <= 1.0:
        raise ValueError(f"oos_win_rate must be in [0, 1], got {oos_win_rate}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    if oos_expectancy_R <= 0.0:
        return 0.0
    kelly = oos_win_rate - (1.0 - oos_win_rate) / payoff_ratio
    return max(kelly, 0.0)


# ---------------------------------------------------------------------------
# Correlation groups
# ---------------------------------------------------------------------------


def correlation_groups(
    daily_returns: pd.DataFrame,
    *,
    threshold: float = CORRELATION_GROUP_THRESHOLD,
    window: int = CORRELATION_WINDOW_DAYS,
) -> list[tuple[str, ...]]:
    """Partition symbols into correlation groups over the trailing window.

    Pairwise Pearson correlation is computed on the last ``window`` rows of
    ``daily_returns`` (columns = symbols, rows = days, oldest first). Two
    symbols are linked when their correlation is STRICTLY greater than
    ``threshold``; groups are the connected components of that graph, so
    linkage is transitive (A~B and B~C puts A, B, C in one group). A NaN
    correlation (e.g. a zero-variance column) links nothing.

    Returns:
        Groups as sorted tuples of symbols, ordered by first member; every
        column appears in exactly one group (singletons included).

    Raises:
        ValueError: fewer than two rows of returns, empty frame, threshold
            outside [-1, 1], or ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
    if daily_returns.shape[1] == 0:
        raise ValueError("daily_returns must have at least one symbol column")
    if len(daily_returns) < 2:
        raise ValueError("correlation_groups requires at least two rows of returns")

    tail = daily_returns.tail(window)
    corr = tail.corr()
    symbols = [str(c) for c in tail.columns]

    parent: dict[str, str] = {s: s for s in symbols}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and float(rho) > threshold:
                parent[find(a)] = find(b)

    components: dict[str, list[str]] = {}
    for s in symbols:
        components.setdefault(find(s), []).append(s)
    return sorted(tuple(sorted(members)) for members in components.values())


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSize:
    """One position's risk allocation after Kelly scaling and hard caps."""

    risk_pct: float  # percent of equity at risk, after all caps
    risk_dollars: float  # equity * risk_pct / 100
    capped: bool  # True when the per-trade/group hard cap bound
    group: tuple[str, ...]  # correlation-group members ((): standalone sizing)


def _validate_common(equity: float, config: RiskConfig, kelly_mult: float | None) -> float:
    """Shared validation; returns the effective Kelly multiplier."""
    if not np.isfinite(equity) or equity <= 0.0:
        raise ValueError(f"equity must be a positive finite number, got {equity}")
    mult = config.kelly_fraction if kelly_mult is None else kelly_mult
    if not 0.0 < mult <= 1.0:
        raise ValueError(f"kelly_mult must be in (0, 1], got {mult}")
    return mult


def size_position(
    equity: float,
    kelly: float,
    config: RiskConfig,
    *,
    kelly_mult: float | None = None,
) -> PositionSize:
    """Size one standalone position: fractional Kelly under the hard cap.

    Risk before caps is ``kelly * kelly_mult`` of equity, with ``kelly_mult``
    defaulting to ``config.kelly_fraction`` (0.25 in the shipped config).
    ``config.per_trade_cap_pct`` is a HARD cap: it binds regardless of how
    large the Kelly fraction is. For positions that trade alongside
    correlated ones, use :func:`size_positions`, which charges each
    correlation group's combined risk against the cap as one position.

    Raises:
        ValueError: non-positive equity, ``kelly`` outside [0, 1], or
            ``kelly_mult`` outside (0, 1].
    """
    mult = _validate_common(equity, config, kelly_mult)
    if not 0.0 <= kelly <= 1.0:
        raise ValueError(f"kelly must be in [0, 1], got {kelly}")
    uncapped_pct = kelly * mult * 100.0
    risk_pct = min(uncapped_pct, config.per_trade_cap_pct)
    return PositionSize(
        risk_pct=risk_pct,
        risk_dollars=equity * risk_pct / 100.0,
        capped=uncapped_pct > config.per_trade_cap_pct,
        group=(),
    )


def size_positions(
    equity: float,
    kelly_by_symbol: Mapping[str, float],
    daily_returns: pd.DataFrame,
    config: RiskConfig,
    *,
    kelly_mult: float | None = None,
) -> dict[str, PositionSize]:
    """Size a book of proposed positions with correlation-group capping.

    Each symbol's pre-cap risk is ``kelly * kelly_mult`` of equity (as in
    :func:`size_position`). Symbols are then partitioned by
    :func:`correlation_groups` over ``daily_returns`` (which must contain a
    column for every symbol), and each group's COMBINED risk counts as one
    position against ``config.per_trade_cap_pct``: when a group's total
    pre-cap risk exceeds the cap, every member is scaled proportionally so
    the group total lands exactly on the cap. Three fully correlated longs
    therefore share one cap, ~cap/3 each when equally sized.

    Returns:
        ``{symbol: PositionSize}``; each entry's ``group`` lists its
        correlation-group members.

    Raises:
        ValueError: validation failures from :func:`size_position` /
            :func:`correlation_groups`, an empty book, or a symbol missing
            from ``daily_returns``.
    """
    mult = _validate_common(equity, config, kelly_mult)
    if not kelly_by_symbol:
        raise ValueError("kelly_by_symbol must contain at least one symbol")
    for symbol, kelly in kelly_by_symbol.items():
        if not 0.0 <= kelly <= 1.0:
            raise ValueError(f"kelly for {symbol} must be in [0, 1], got {kelly}")
    missing = sorted(set(kelly_by_symbol) - {str(c) for c in daily_returns.columns})
    if missing:
        raise ValueError(f"daily_returns lacks columns for symbols: {missing}")

    groups = correlation_groups(daily_returns[list(kelly_by_symbol)])
    cap = config.per_trade_cap_pct
    sized: dict[str, PositionSize] = {}
    for group in groups:
        uncapped = {s: kelly_by_symbol[s] * mult * 100.0 for s in group}
        total = sum(uncapped.values())
        capped = total > cap
        scale = cap / total if capped else 1.0
        for symbol, pct in uncapped.items():
            risk_pct = pct * scale
            sized[symbol] = PositionSize(
                risk_pct=risk_pct,
                risk_dollars=equity * risk_pct / 100.0,
                capped=capped,
                group=group,
            )
    return sized


# ---------------------------------------------------------------------------
# Daily loss halt
# ---------------------------------------------------------------------------


def daily_loss_halt(day_pnl_pct: float, config: RiskConfig) -> bool:
    """True when the day's PnL mandates halting NEW entries for the session.

    The halt triggers at or beyond a loss of ``config.daily_loss_halt_pct``
    percent of start-of-day equity: ``day_pnl_pct <= -daily_loss_halt_pct``.
    Exits are never blocked by this predicate — only new entries.

    Raises:
        ValueError: non-finite ``day_pnl_pct``.
    """
    if not np.isfinite(day_pnl_pct):
        raise ValueError(f"day_pnl_pct must be finite, got {day_pnl_pct}")
    return day_pnl_pct <= -config.daily_loss_halt_pct


def _session_date(ts: datetime | date) -> date:
    """Session identity of a timestamp: its America/New_York calendar date."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("session timestamps must be tz-aware (America/New_York)")
        return ts.astimezone(MARKET_TZ).date()
    return ts


class DailyLossHalt:
    """Latching per-session halt on new entries after the daily loss limit.

    Feed it the running day PnL via :meth:`observe`; once
    :func:`daily_loss_halt` trips for a session, :meth:`entries_allowed`
    stays False for every timestamp in that session — recovery does not
    un-halt. A new session date starts clean.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._halted_session: date | None = None

    def observe(self, ts: datetime | date, day_pnl_pct: float) -> bool:
        """Record the session's running PnL; returns ``entries_allowed(ts)``.

        ``day_pnl_pct`` is the day's PnL as a percent of start-of-day equity
        (losses negative). ``ts`` must be a date or tz-aware datetime.
        """
        session = _session_date(ts)
        if daily_loss_halt(day_pnl_pct, self._config):
            self._halted_session = session
        return self.entries_allowed(ts)

    def entries_allowed(self, ts: datetime | date) -> bool:
        """True unless this timestamp's session has tripped the loss halt."""
        return _session_date(ts) != self._halted_session

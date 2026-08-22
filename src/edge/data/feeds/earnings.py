"""Earnings-calendar adapter: event table + proximity features from the
yfinance snapshot archive.

BACKEND-side module (research code never imports this — it goes through
:class:`edge.data.loader.EdgeDataLoader`; the AST gateway scan enforces the
wall, ``edge/data/`` is exempt).

Source
------
Per-symbol parquet snapshots under ``data_cache/earnings_yf/`` — the cache
written by the archived catalyst generation's ``YFinanceEarnings`` provider
(see ``src/catalyst/data/catalysts.py``). Schema per file: ``when`` (naive
America/New_York report datetime), ``eps_estimate``, ``eps_actual``; the
symbol is the file stem. The archive is parsed directly as the PRIMARY
source; :meth:`EarningsFeed.refresh` wraps yfinance for live incremental
updates and is never exercised by tests (tests inject a fake fetcher — no
network, ever, in adapter code paths used by tests).

=====================================================================
LOOK-AHEAD LIMITATION — READ BEFORE USING THESE FEATURES IN A BACKTEST
=====================================================================
Earnings DATES are forward-looking calendar data: a company announces its
report date days-to-weeks ahead, so a live trader on day D genuinely knows
"days_to_next_earnings" at the start of day D. The archive, however, is a
SNAPSHOT: whatever yfinance reported on the day the file was fetched
(observed fetch dates: 2026-08). Two consequences:

1. The event table stamps the snapshot itself as available at the
   snapshot's own fetch date 18:00 ET (``available_at`` ==
   ``calendar_snapshot_ts``, derived from the archive file's mtime unless a
   ``snapshot_ts`` override is passed). That is the conservative, literal
   truth about when THIS FILE existed.
2. The features frame is BY DEFAULT stamped strictly at
   ``max(asof 00:00 ET, calendar_snapshot_ts)`` — fully honest joins: no
   feature row can ever claim availability before the snapshot it was
   computed from existed. The cost is historical coverage (every pre-snapshot
   day joins only at the snapshot stamp). Opting out — stamping every row at
   ``asof_date`` 00:00 ET so historical backtests can join the full range —
   embeds the assumption that the calendar as fetched was already known on
   every historical day. Dates occasionally MOVE (postponements,
   reschedules), and a snapshot fetched AFTER the fact records where the
   date ENDED UP, not what was scheduled at the time. Historical
   ``days_to_next_earnings`` under the opt-out therefore carries look-ahead
   risk. That opt-out must be requested by its self-documenting name —
   ``EarningsConfig(allow_snapshot_lookahead=True)`` — and the feed logs a
   WARNING at construction naming the leak. Every feature row emits
   ``calendar_snapshot_ts``; any row with ``available_at <
   calendar_snapshot_ts`` (possible only under the opt-out) is subject to
   this caveat.

Feature semantics (day granularity)
-----------------------------------
* ``days_to_next_earnings``: calendar days to the next event with
  ``event_date >= asof_date`` (0 on the event day; NaN when the snapshot
  holds no future event).
* ``days_since_last_earnings``: calendar days since the most recent event
  with ``event_date <= asof_date`` (0 on the event day; NaN before the
  first known event).
* ``is_earnings_week``: an event falls in the Monday-anchored calendar week
  (Mon..Sun) containing ``asof_date``.
Day-level features cannot see BMO/AMC intraday timing — the event table's
``event_ts``/``session`` columns carry it for consumers that need it.

Cache: parquet under ``data_cache/edge/earnings/`` (``cache_root``),
cache-first; the combined events parquet is rebuilt whenever any archive
file is newer. Writes are atomic (tmp + replace).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: Directory (under the repo's ``data_cache/``) holding the per-symbol
#: yfinance snapshot parquets — the catalyst generation's cache layout.
ARCHIVE_DIRNAME = "earnings_yf"

#: Report-time session buckets. Exactly-midnight timestamps mean "date only,
#: time unknown" in the yfinance data and are labeled ``unknown``.
SESSION_BMO = "bmo"
SESSION_AMC = "amc"
SESSION_DMT = "dmt"  # during market trading (rare, usually data noise)
SESSION_UNKNOWN = "unknown"

_BMO_END = time(9, 30)
_AMC_START = time(16, 0)

#: Filenames are the symbol — refuse anything that would not round-trip.
_SAFE_SYMBOL = re.compile(r"[A-Za-z0-9._-]+")

#: A per-symbol fetcher for :meth:`EarningsFeed.refresh`: returns a frame
#: with columns ``when`` (naive ET), ``eps_estimate``, ``eps_actual`` — or
#: None on failure/empty, which must NEVER be persisted (a scrape failure
#: cached as "no earnings" silently deletes the symbol's events from every
#: future run — catalyst audit D-047).
Fetcher = Callable[[str], "pd.DataFrame | None"]


class EarningsConfig(BaseModel):
    """Adapter knobs; defaults encode the documented availability stamps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hour (ET) at which a day's calendar snapshot is stamped available.
    publish_hour_et: int = Field(default=18, ge=0, le=23)
    #: DEFAULT False = strict: feature rows are stamped ``max(asof 00:00 ET,
    #: calendar_snapshot_ts)`` — no row is ever available before the snapshot
    #: it was computed from existed. True opts INTO the documented
    #: snapshot look-ahead (rows stamped asof 00:00 ET even before the
    #: snapshot's fetch date, for historical coverage) and makes the feed
    #: log a WARNING at construction. See the module docstring.
    allow_snapshot_lookahead: bool = False
    #: Events per symbol requested from yfinance in :meth:`refresh`.
    refresh_limit: int = Field(default=100, gt=0)


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via tmp + replace so a crash can never leave a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _session_of(when: pd.Series) -> pd.Series:
    """Naive-ET report datetimes -> bmo/amc/dmt/unknown labels."""
    t = when.dt.time
    labels = np.select(
        [
            t == time(0, 0),  # date-only rows, time unknown
            t < _BMO_END,
            t >= _AMC_START,
        ],
        [SESSION_UNKNOWN, SESSION_BMO, SESSION_AMC],
        default=SESSION_DMT,
    )
    return pd.Series(labels, index=when.index)


class EarningsFeed:
    """Cache-first earnings-calendar adapter over the yfinance snapshot archive.

    Contract: every public frame carries both point-in-time columns
    (``asof_date`` tz-naive midnight, ``available_at`` tz-aware ET) plus
    ``calendar_snapshot_ts`` (tz-aware ET — the conservative stamp of when
    the underlying calendar snapshot existed: its fetch date at
    ``publish_hour_et``:00 ET). No method used by tests touches the network;
    :meth:`refresh` is the live path and takes an injectable fetcher.
    """

    _EVENTS_CACHE = "events.parquet"

    def __init__(
        self,
        archive_root: str | Path,
        cache_root: str | Path,
        config: EarningsConfig | None = None,
        snapshot_ts: datetime | pd.Timestamp | None = None,
    ) -> None:
        self._archive_root = Path(archive_root)
        self._cache_root = Path(cache_root)
        self._config = config or EarningsConfig()
        if self._config.allow_snapshot_lookahead:
            logger.warning(
                "EarningsFeed(allow_snapshot_lookahead=True): feature rows will "
                "be stamped available at asof 00:00 ET even BEFORE the calendar "
                "snapshot they were computed from existed (available_at < "
                "calendar_snapshot_ts) — historical days_to_next_earnings may "
                "embed post-hoc calendar knowledge (moved/rescheduled report "
                "dates). Do not use these rows for point-in-time evaluation."
            )
        #: Optional override for the snapshot timestamp (tests / pinned
        #: provenance). When set, the parquet events cache is BYPASSED so a
        #: stale stamp can never be served from disk.
        self._snapshot_override: pd.Timestamp | None = None
        if snapshot_ts is not None:
            ts = pd.Timestamp(snapshot_ts)
            if ts.tzinfo is None:
                raise ValueError("snapshot_ts must be tz-aware")
            self._snapshot_override = ts.tz_convert(MARKET_TZ)

    @classmethod
    def from_repo(
        cls,
        repo_root: str | Path | None = None,
        config: EarningsConfig | None = None,
        snapshot_ts: datetime | pd.Timestamp | None = None,
    ) -> EarningsFeed:
        """Build against the repo's standard archive and cache locations."""
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
        return cls(
            archive_root=root / "data_cache" / ARCHIVE_DIRNAME,
            cache_root=root / "data_cache" / "edge" / "earnings",
            config=config,
            snapshot_ts=snapshot_ts,
        )

    # ------------------------------------------------------------------
    # Archive parsing / event table
    # ------------------------------------------------------------------

    def archive_symbols(self) -> list[str]:
        """Symbols present in the snapshot archive (file stems, sorted)."""
        return sorted(p.stem for p in self._archive_root.glob("*.parquet"))

    def _snapshot_stamp(self, path: Path) -> pd.Timestamp:
        """The conservative snapshot timestamp for one archive file: its
        fetch date (mtime, unless overridden) at ``publish_hour_et``:00 ET."""
        if self._snapshot_override is not None:
            fetched = self._snapshot_override
        else:
            fetched = pd.Timestamp(
                datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TZ)
            )
        return fetched.normalize() + pd.Timedelta(hours=self._config.publish_hour_et)

    def load_events(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """The full event table from the archive, cache-first.

        Columns: ``asof_date`` (== ``event_date``, tz-naive midnight),
        ``available_at`` (== ``calendar_snapshot_ts`` — the SNAPSHOT's
        conservative availability, NOT the event's; see the module
        docstring's look-ahead section), ``symbol``, ``event_date``,
        ``event_ts`` (tz-aware ET report datetime), ``session``
        (bmo/amc/dmt/unknown), ``eps_estimate``, ``eps_actual``,
        ``calendar_snapshot_ts``. Sorted by (symbol, event_ts), one row per
        (symbol, report datetime). ``symbols`` filters after loading; a
        requested symbol absent from the archive raises (silent absence
        would read as "never reports earnings").
        """
        files = sorted(self._archive_root.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no *.parquet under {self._archive_root}")
        table = self._events_cache_read(files)
        if table is None:
            table = pd.concat(
                [self._parse_symbol_file(p) for p in files], ignore_index=True
            )
            table = (
                table.drop_duplicates(subset=["symbol", "event_ts"])
                .sort_values(["symbol", "event_ts"], kind="stable")
                .reset_index(drop=True)
            )
            if self._snapshot_override is None:  # never cache overridden stamps
                _write_atomic(self._cache_root / self._EVENTS_CACHE, table)
        if symbols is not None:
            missing = sorted(set(symbols) - set(table["symbol"].unique()))
            if missing:
                raise ValueError(
                    f"symbols not in the earnings archive: {missing} — refusing "
                    "to emit rows that would read as 'never reports earnings'"
                )
            table = table[table["symbol"].isin(set(symbols))].reset_index(drop=True)
        return table

    def _events_cache_read(self, files: list[Path]) -> pd.DataFrame | None:
        """Serve the combined parquet only while it is at least as new as
        every archive file; a snapshot override always bypasses it."""
        if self._snapshot_override is not None:
            return None
        cache = self._cache_root / self._EVENTS_CACHE
        newest_src = max(p.stat().st_mtime for p in files)
        if cache.exists() and cache.stat().st_mtime >= newest_src:
            try:
                return pd.read_parquet(cache)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable events cache %s: %s — rebuilding", cache, exc)
        return None

    def _parse_symbol_file(self, path: Path) -> pd.DataFrame:
        """One archive parquet -> tidy per-symbol event rows."""
        raw = pd.read_parquet(path)
        needed = {"when", "eps_estimate", "eps_actual"}
        if not needed.issubset(raw.columns):
            raise ValueError(
                f"{path.name}: expected columns {sorted(needed)}, got {list(raw.columns)}"
            )
        when = pd.to_datetime(raw["when"]).astype("datetime64[ns]")
        snap = self._snapshot_stamp(path)
        out = pd.DataFrame(
            {
                "asof_date": when.dt.normalize(),
                "available_at": snap,
                "symbol": path.stem,
                "event_date": when.dt.normalize(),
                # yfinance times are wall-clock ET; the 02:00–02:59 spring-
                # forward gap / fall-back hour cannot legitimately host an
                # earnings call, so shift/NaT instead of crashing on noise.
                "event_ts": when.dt.tz_localize(
                    MARKET_TZ, nonexistent="shift_forward", ambiguous="NaT"
                ),
                "session": _session_of(when),
                "eps_estimate": pd.to_numeric(raw["eps_estimate"], errors="raise"),
                "eps_actual": pd.to_numeric(raw["eps_actual"], errors="raise"),
                "calendar_snapshot_ts": snap,
            }
        )
        return out

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def features(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-symbol/per-business-day earnings-proximity features.

        One row per (business day Mon–Fri, no holiday calendar) × symbol.
        Columns: ``asof_date``, ``available_at``, ``symbol``,
        ``days_to_next_earnings``, ``days_since_last_earnings``,
        ``is_earnings_week``, ``calendar_snapshot_ts``.

        ``available_at`` is ``max(asof 00:00 ET, calendar_snapshot_ts)`` by
        default (strict): a row can never claim availability before the
        snapshot it was computed from existed. On days at/after the
        snapshot's fetch date the stamp is plain ``asof`` 00:00 ET — the
        calendar for day D is announced ahead of D, so a live trader knows
        these values before the session opens; a later stamp (e.g. same-day
        18:00) would be anti-conservative for risk gating (a decision on day
        D would see day D−1's row and believe the event is one day further
        away than it is). With ``allow_snapshot_lookahead=True`` every row
        is stamped ``asof`` 00:00 ET, accepting the LOOK-AHEAD caveat in the
        module docstring for rows where ``available_at <
        calendar_snapshot_ts``.
        """
        events = self.load_events(symbols=symbols)
        grid = pd.bdate_range(
            pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        )
        if grid.empty:
            raise ValueError(f"empty business-day grid for [{start!r}, {end!r}]")
        asof_d = grid.to_numpy().astype("datetime64[D]")
        week_start = asof_d - grid.weekday.to_numpy().astype("timedelta64[D]")
        week_end = week_start + np.timedelta64(6, "D")
        midnight_et = pd.Series(grid).dt.tz_localize(MARKET_TZ)  # never ambiguous

        frames: list[pd.DataFrame] = []
        for symbol, group in events.groupby("symbol", sort=True):
            ev = np.unique(group["event_date"].to_numpy().astype("datetime64[D]"))
            i_next = np.searchsorted(ev, asof_d, side="left")
            has_next = i_next < len(ev)
            d_next = (ev[np.minimum(i_next, len(ev) - 1)] - asof_d) / np.timedelta64(1, "D")
            i_last = np.searchsorted(ev, asof_d, side="right") - 1
            has_last = i_last >= 0
            d_last = (asof_d - ev[np.maximum(i_last, 0)]) / np.timedelta64(1, "D")
            in_week = np.searchsorted(ev, week_end, side="right") > np.searchsorted(
                ev, week_start, side="left"
            )
            snap = group["calendar_snapshot_ts"].iloc[0]
            avail = midnight_et
            if not self._config.allow_snapshot_lookahead:
                # STRICT default: no row may predate the snapshot it came from.
                avail = midnight_et.where(midnight_et >= snap, snap)
            frames.append(
                pd.DataFrame(
                    {
                        "asof_date": grid,
                        "available_at": avail,
                        "symbol": symbol,
                        "days_to_next_earnings": np.where(has_next, d_next, np.nan),
                        "days_since_last_earnings": np.where(has_last, d_last, np.nan),
                        "is_earnings_week": in_week,
                        "calendar_snapshot_ts": snap,
                    }
                )
            )
        out = pd.concat(frames, ignore_index=True)
        return (
            out.sort_values(["asof_date", "symbol"], kind="stable")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Live refresh (never used in tests except via an injected fake fetcher)
    # ------------------------------------------------------------------

    def refresh(
        self, symbols: list[str] | None = None, fetcher: Fetcher | None = None
    ) -> dict[str, int]:
        """Incrementally re-snapshot symbols into the archive (LIVE path).

        For each symbol the fetcher returns a normalized frame or None;
        None (failure OR empty — yfinance's most common failure shape) is
        logged and NEVER persisted, so a bad scrape can never overwrite a
        good snapshot or read as "no earnings" (catalyst audit D-047).
        Successful writes are atomic; the combined events cache is
        invalidated so the next :meth:`load_events` rebuilds with fresh
        snapshot stamps. Returns {symbol: rows written} for the successes.
        """
        targets = list(symbols) if symbols is not None else self.archive_symbols()
        fetch = fetcher or self._yfinance_fetch
        written: dict[str, int] = {}
        for symbol in targets:
            if not _SAFE_SYMBOL.fullmatch(symbol):
                raise ValueError(
                    f"symbol {symbol!r} is not filename-safe ([A-Za-z0-9._-] only) "
                    "— it would not round-trip through the archive layout"
                )
            frame = fetch(symbol)
            if frame is None or frame.empty:
                logger.warning(
                    "earnings refresh for %s returned nothing — NOT persisted "
                    "(existing snapshot, if any, is kept)", symbol,
                )
                continue
            frame = (
                frame[["when", "eps_estimate", "eps_actual"]]
                .drop_duplicates(subset=["when"])
                .sort_values("when", kind="stable")
                .reset_index(drop=True)
            )
            _write_atomic(self._archive_root / f"{symbol}.parquet", frame)
            written[symbol] = len(frame)
        if written:
            (self._cache_root / self._EVENTS_CACHE).unlink(missing_ok=True)
        return written

    def _yfinance_fetch(self, symbol: str) -> pd.DataFrame | None:
        """Default live fetcher (lazy yfinance import — never in test paths)."""
        import yfinance as yf  # imported lazily: live-refresh dependency only

        try:
            raw = yf.Ticker(symbol).get_earnings_dates(limit=self._config.refresh_limit)
        except Exception as exc:  # noqa: BLE001 — vendor scrape
            logger.warning("yfinance earnings fetch failed for %s: %s", symbol, exc)
            return None
        if raw is None or raw.empty:
            return None
        idx = raw.index.tz_convert(MARKET_TZ).tz_localize(None)
        return pd.DataFrame(
            {
                "when": idx,
                "eps_estimate": raw.get("EPS Estimate", pd.Series(index=raw.index)).values,
                "eps_actual": raw.get("Reported EPS", pd.Series(index=raw.index)).values,
            }
        )

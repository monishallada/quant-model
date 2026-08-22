"""FINRA Reg SHO daily short-sale volume: consolidated (CNMS) file adapter.

Parses ``CNMSshvol{YYYYMMDD}.txt`` from the raw archive at
``data_cache/edge/raw/finra_short_daily/`` — pipe-delimited with header
``Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market``, CRLF line
endings, and a trailing bare record-count line that must be skipped. Volumes
may be FRACTIONAL floats in recent files and are always parsed as float.
A missing date is a non-trading day or a file eroded off FINRA's CDN (which
answers 403 for absent dates) — both are skipped, never an error.

INTERPRETATION CAVEAT (non-negotiable context for every consumer): the
"short volume" in this feed is dominated by market-maker and liquidity-
provider shorting — a market maker filling a customer BUY prints a short
sale — so the LEVEL of ``short_ratio`` (routinely ~40-60%% of total volume)
says nothing about directional bearish positioning. LEVELS ARE MEANINGLESS;
DEVIATIONS ARE THE SIGNAL. The features are therefore z-scores only: a
rolling z of each symbol's ratio against its OWN trailing history (default
63 published sessions) and a cross-sectional z against all symbols that day.

POINT-IN-TIME: every output frame carries ``asof_date`` (the data's own
date) and ``available_at`` (tz-aware America/New_York datetime a live trader
could FIRST have seen it). FINRA publishes the daily file ~6-8pm ET the same
evening; the exact minute is not guaranteed, so per the conservative-later
rule ``available_at`` is the NEXT NYSE trading day at 09:00 ET (pre-open).
Downstream joins key on ``available_at <= decision_ts``.

Cache-first: parsed days live under ``<cache_root>/date=YYYY-MM-DD.parquet``
(atomic tmp+replace writes) and are served without re-reading the text
archive. No adapter code path used by tests touches the network; the
incremental :meth:`FinraShortDaily.update` uses an injected HTTP client and
exists for live daily top-ups only.

This is a BACKEND-side module: research code never imports it — data flows
to research only through :class:`edge.data.loader.EdgeDataLoader` (the AST
gateway scan enforces the wall; ``data/`` itself is exempt).
"""

from __future__ import annotations

import logging
import os
import re
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import pandas_market_calendars as mcal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: Archive/CDN filename pattern for the consolidated NMS daily file.
_FILENAME = "CNMSshvol{yyyymmdd}.txt"
_FILENAME_RE = re.compile(r"^CNMSshvol(\d{8})\.txt$")

#: Expected header, case-insensitive (FINRA has been stable on this).
_HEADER = ("date", "symbol", "shortvolume", "shortexemptvolume", "totalvolume", "market")

#: Columns of the per-day parsed cache frame (raw fields only — derived
#: columns are recomputed at assembly so their definitions can evolve
#: without invalidating the cache).
_RAW_COLUMNS = [
    "asof_date",
    "symbol",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
    "market",
]

#: Columns of every frame returned by :meth:`FinraShortDaily.daily`.
DAILY_COLUMNS = [*_RAW_COLUMNS, "short_ratio", "available_at"]

#: Extra columns added by :meth:`FinraShortDaily.features`.
FEATURE_COLUMNS = ["short_ratio_z", "short_ratio_xs_z"]


class FinraShortConfig(BaseModel):
    """Adapter knobs; defaults encode the task spec and are overridable —
    nothing below is baked into code paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: FINRA CDN root for incremental daily top-ups (absent dates -> 403).
    base_url: str = "https://cdn.finra.org/equity/regsho/daily"
    #: Trailing own-history window for the rolling z-score, in PUBLISHED
    #: sessions for that symbol (a symbol absent on some days is compared
    #: to its own most recent 63 published rows, not 63 calendar days).
    rolling_window: int = Field(default=63, gt=1)
    #: Minimum prior observations before a rolling z is emitted (else NaN).
    rolling_min_periods: int = Field(default=21, gt=1)
    #: ``available_at`` clock time on the NEXT NYSE trading day. The file is
    #: actually published ~6-8pm ET the same evening; next-day pre-open is
    #: the deliberate conservative-later choice.
    available_hour_et: int = Field(default=9, ge=0, le=23)
    available_minute_et: int = Field(default=0, ge=0, le=59)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def _min_periods_within_window(self) -> FinraShortConfig:
        if self.rolling_min_periods > self.rolling_window:
            raise ValueError(
                f"rolling_min_periods ({self.rolling_min_periods}) must not exceed "
                f"rolling_window ({self.rolling_window})"
            )
        return self


class _Response(Protocol):
    status_code: int
    text: str


@runtime_checkable
class HttpGetter(Protocol):
    """The one slice of ``httpx.Client`` :meth:`FinraShortDaily.update` uses —
    tests inject a fake with canned bodies; production injects a real client."""

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _Response: ...


class FinraShortDaily:
    """Cache-first adapter over the FINRA consolidated daily short-volume archive.

    Contract: :meth:`daily` returns one row per (asof_date, symbol) over an
    inclusive date range with ``short_ratio`` and point-in-time
    ``available_at``; :meth:`features` adds the rolling own-history z and the
    per-day cross-sectional z (loading its own warmup history before
    ``start`` so z-scores at the range head are real, then trimming).
    Missing dates are skipped. No test-path network: history is parsed from
    the on-disk archive only.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        cache_root: str | Path,
        config: FinraShortConfig | None = None,
        client: HttpGetter | None = None,
    ) -> None:
        self._raw_dir = Path(raw_dir)
        self._cache_root = Path(cache_root)
        self._config = config or FinraShortConfig()
        self._http = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_dates(self) -> list[date]:
        """Sorted union of dates present in the raw archive and the cache."""
        dates: set[date] = set()
        if self._raw_dir.is_dir():
            for path in self._raw_dir.iterdir():
                match = _FILENAME_RE.match(path.name)
                if match:
                    dates.add(_yyyymmdd(match.group(1)))
        if self._cache_root.is_dir():
            for path in self._cache_root.glob("date=*.parquet"):
                try:
                    dates.add(date.fromisoformat(path.stem.removeprefix("date=")))
                except ValueError:
                    logger.warning("unrecognized cache file name skipped: %s", path.name)
        return sorted(dates)

    def daily(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-symbol short-volume rows for inclusive ``[start, end]``.

        Columns: ``asof_date`` (python date), ``symbol``, ``short_volume``,
        ``short_exempt_volume``, ``total_volume`` (floats — recent files carry
        fractional volumes), ``market``, ``short_ratio``
        (ShortVolume/TotalVolume; NaN when TotalVolume is 0), and
        ``available_at`` (next NYSE trading day at the configured pre-open
        time, tz-aware America/New_York). Sorted by (asof_date, symbol).
        Dates in range with no file are skipped (non-trading or eroded).

        Raises:
            ValueError: ``start`` is after ``end``.
        """
        if start > end:
            raise ValueError(f"empty range: start {start} is after end {end}")
        frames = [
            frame
            for asof in self.available_dates()
            if start <= asof <= end and (frame := self._load_day(asof)) is not None
        ]
        if not frames:
            return _empty_daily()
        out = pd.concat(frames, ignore_index=True)
        # Defensive: asof_date comes from each ROW's Date field, so clamp to
        # the requested range even if a file carried an off-date row.
        out = out[(out["asof_date"] >= start) & (out["asof_date"] <= end)]
        if symbols is not None:
            out = out[out["symbol"].isin(set(symbols))]
        if out.empty:
            return _empty_daily()
        out = out.sort_values(["asof_date", "symbol"], kind="stable").reset_index(drop=True)
        total = out["total_volume"]
        out["short_ratio"] = (out["short_volume"] / total).where(total > 0)
        out["available_at"] = out["asof_date"].map(
            self._available_at_map(set(out["asof_date"]))
        )
        return out[DAILY_COLUMNS]

    def features(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """:meth:`daily` plus the two z-score features, trimmed to the range.

        - ``short_ratio_z``: rolling z of ``short_ratio`` against the
          symbol's OWN trailing history — mean/std over the prior
          ``rolling_window`` published rows for that symbol, EXCLUDING the
          current day (the baseline is history, the day is the observation).
          NaN until ``rolling_min_periods`` prior observations exist, and
          NaN when the trailing std is 0 (never +/-inf).
        - ``short_ratio_xs_z``: cross-sectional z of ``short_ratio`` against
          all symbols published the same day (ddof=1); NaN when the day has
          fewer than 2 symbols or zero dispersion. NOTE: when ``symbols`` is
          passed, the cross-section is that filtered universe — pass
          ``symbols=None`` for a market-wide cross-section.

        Warmup is loaded automatically: up to ``rolling_window`` archive days
        BEFORE ``start`` feed the rolling baseline, then rows before
        ``start`` are dropped — so z-scores at the head of the range are as
        real as in the middle. Remember the module-level caveat: these
        deviations are the signal; the raw ``short_ratio`` level is not.
        """
        if start > end:
            raise ValueError(f"empty range: start {start} is after end {end}")
        cfg = self._config
        warmup = [d for d in self.available_dates() if d < start][-cfg.rolling_window :]
        load_start = warmup[0] if warmup else start
        frame = self.daily(load_start, end, symbols)
        if frame.empty:
            out = _empty_daily()
            out[FEATURE_COLUMNS] = None
            return out

        frame = frame.sort_values(["symbol", "asof_date"], kind="stable").reset_index(drop=True)

        def _own_history_z(ratios: pd.Series) -> pd.Series:
            prior = ratios.shift(1)
            mu = prior.rolling(cfg.rolling_window, min_periods=cfg.rolling_min_periods).mean()
            sd = prior.rolling(cfg.rolling_window, min_periods=cfg.rolling_min_periods).std()
            return ((ratios - mu) / sd).where(sd > 0)

        by_symbol = frame.groupby("symbol", sort=False)["short_ratio"]
        frame["short_ratio_z"] = by_symbol.transform(_own_history_z)

        by_day = frame.groupby("asof_date", sort=False)["short_ratio"]
        day_mu = by_day.transform("mean")
        day_sd = by_day.transform("std")
        frame["short_ratio_xs_z"] = ((frame["short_ratio"] - day_mu) / day_sd).where(day_sd > 0)

        frame = frame[frame["asof_date"] >= start]
        frame = frame.sort_values(["asof_date", "symbol"], kind="stable").reset_index(drop=True)
        return frame[[*DAILY_COLUMNS, *FEATURE_COLUMNS]]

    def update(self, through: date) -> list[date]:
        """Top up the raw archive with NYSE trading days after the last
        archived date, through ``through`` inclusive. LIVE-ONLY path: never
        called by tests with a real client and never used to fetch history.

        A 403 or 404 means the date is a non-trading day or has eroded off
        FINRA's CDN — logged and skipped, never an error. Returns the dates
        actually written.

        Raises:
            RuntimeError: no HTTP client was injected at construction.
            ValueError: the raw archive is empty (seed it out-of-band first —
                this method refuses to become a history downloader).
        """
        if self._http is None:
            raise RuntimeError(
                "FinraShortDaily.update requires an injected HTTP client; "
                "construct with client=... (tests use canned fakes, no network)."
            )
        known = self.available_dates()
        if not known:
            raise ValueError(
                "raw archive is empty — update() only tops up an existing archive; "
                "seed history out-of-band."
            )
        first_missing = known[-1] + timedelta(days=1)
        if first_missing > through:
            return []
        calendar = mcal.get_calendar("NYSE")
        written: list[date] = []
        for stamp in calendar.valid_days(start_date=first_missing, end_date=through):
            day = stamp.date()
            url = f"{self._config.base_url}/{_FILENAME.format(yyyymmdd=day.strftime('%Y%m%d'))}"
            try:
                resp = self._http.get(url)
            except Exception as exc:  # noqa: BLE001 — one bad day must not kill the top-up
                logger.warning("FINRA fetch error %s: %s", url, exc)
                continue
            if resp.status_code in (403, 404):
                logger.info("FINRA %s -> HTTP %d (non-trading or eroded); skipped", url,
                            resp.status_code)
                continue
            if resp.status_code != 200:
                logger.warning("FINRA %s -> HTTP %d; skipped", url, resp.status_code)
                continue
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            target = self._raw_dir / _FILENAME.format(yyyymmdd=day.strftime("%Y%m%d"))
            tmp = target.with_suffix(f".tmp-{os.getpid()}")
            try:
                tmp.write_text(resp.text)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
            written.append(day)
        return written

    # ------------------------------------------------------------------
    # Cache + parse
    # ------------------------------------------------------------------

    def _cache_path(self, asof: date) -> Path:
        return self._cache_root / f"date={asof.isoformat()}.parquet"

    def _load_day(self, asof: date) -> pd.DataFrame | None:
        """Cached parsed frame if present, else parse-and-cache from the raw
        archive; None when the date has no file anywhere (skipped upstream)."""
        cache = self._cache_path(asof)
        if cache.exists():
            try:
                return pd.read_parquet(cache)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable cache %s: %s — reparsing", cache, exc)
        raw = self._raw_dir / _FILENAME.format(yyyymmdd=asof.strftime("%Y%m%d"))
        if not raw.exists():
            return None
        frame = _parse_file(raw)
        _write_atomic(cache, frame)
        return frame

    def _available_at_map(self, dates: set[date]) -> dict[date, pd.Timestamp]:
        """asof_date -> next NYSE trading day at the configured pre-open time.

        Conservative point-in-time stamp: the file lands ~6-8pm ET the same
        evening, but the exact minute is unguaranteed, so availability is
        pushed to the LATER, certain boundary — next session's pre-open.
        """
        cfg = self._config
        lo, hi = min(dates), max(dates) + timedelta(days=21)
        valid = [stamp.date() for stamp in mcal.get_calendar("NYSE").valid_days(lo, hi)]
        out: dict[date, pd.Timestamp] = {}
        for asof in dates:
            idx = bisect_right(valid, asof)
            if idx >= len(valid):  # pragma: no cover — 21-day pad spans any holiday run
                raise RuntimeError(f"no NYSE trading day found within 21 days after {asof}")
            out[asof] = pd.Timestamp(
                datetime.combine(
                    valid[idx],
                    time(cfg.available_hour_et, cfg.available_minute_et),
                    tzinfo=MARKET_TZ,
                )
            )
        return out


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def _yyyymmdd(raw: str) -> date:
    """``'20240716'`` -> ``date(2024, 7, 16)`` (raises ValueError on junk)."""
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"expected YYYYMMDD date, got {raw!r}")
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def _parse_file(path: Path) -> pd.DataFrame:
    """Parse one ``CNMSshvol`` text file into the raw per-day frame.

    Handles CRLF endings and an optional BOM; validates the header; skips
    blank lines and the trailing bare record-count line (warning when the
    count disagrees with the rows actually parsed); parses all three volume
    fields as float because recent files carry fractional volumes.

    Raises:
        ValueError: unexpected header, or a malformed data line.
    """
    lines = path.read_text().lstrip("\ufeff").splitlines()
    if not lines:
        raise ValueError(f"{path.name}: empty file")
    header = tuple(field.strip().lower() for field in lines[0].split("|"))
    if header != _HEADER:
        raise ValueError(f"{path.name}: unexpected header {lines[0]!r}")

    rows: list[tuple[Any, ...]] = []
    declared_count: float | None = None
    for lineno, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 1:
            try:
                declared_count = float(parts[0])
                continue  # trailing bare record-count line
            except ValueError:
                raise ValueError(f"{path.name}:{lineno}: malformed line {line!r}") from None
        if len(parts) != 6:
            raise ValueError(f"{path.name}:{lineno}: expected 6 fields, got {len(parts)}")
        raw_date, symbol, short_vol, exempt_vol, total_vol, market = (p.strip() for p in parts)
        try:
            rows.append(
                (
                    _yyyymmdd(raw_date),
                    symbol,
                    float(short_vol),
                    float(exempt_vol),
                    float(total_vol),
                    market,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{lineno}: malformed line {line!r}: {exc}") from None
    if declared_count is not None and declared_count != len(rows):
        logger.warning(
            "%s: trailing record count %s != %d parsed rows (kept all parsed rows)",
            path.name, declared_count, len(rows),
        )
    return pd.DataFrame(rows, columns=_RAW_COLUMNS)


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_COLUMNS)


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via tmp + replace so a crash or concurrent writer can
    never leave a torn file at the final path (same discipline as the other
    edge feed adapters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

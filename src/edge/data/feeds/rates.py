"""Rates & credit adapter: Treasury par-yield archive + FRED fredgraph fallback.

BACKEND-side module (research code never imports this — it goes through
:class:`edge.data.loader.EdgeDataLoader`; the AST gateway scan enforces the
wall, ``edge/data/`` is exempt).

Sources
-------
1. **Treasury daily par yield curve** — parsed from the on-disk archive
   ``data_cache/edge/raw/treasury_yields/yields_{YYYY}.csv``. Rows are
   newest-first, ``Date`` is ``MM/DD/YYYY``, and the tenor columns VARY BY
   YEAR (1990 has no "1 Mo"; 2026 adds "1.5 Month"). Parsing is strictly by
   header name: ``"2 Yr"`` and ``"10 Yr"`` are REQUIRED in every file, all
   other tenors are optional and carried through when present. Never fetched
   from the network in code paths used by tests.
2. **FRED fredgraph fallback** (incremental daily updates only, NOT used in
   tests): keyless ``https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES``
   for DGS2, DGS10, VIXCLS, BAMLH0A0HYM2, DFF. Missing observations are ``"."``
   and are dropped (they carry no data).

POINT-IN-TIME CONVENTION (non-negotiable)
-----------------------------------------
Every output frame carries ``asof_date`` (the data's own date, tz-naive
midnight ``datetime64[ns]``) AND ``available_at`` (tz-aware America/New_York
datetime when a live trader could FIRST have seen the value). Downstream
feature joins key on ``available_at <= decision_ts``.

* Treasury archive rows: ``available_at = asof_date 18:00 ET``. Treasury
  posts the daily curve ~3:00–4:00 PM ET; 18:00 is the deliberately LATER,
  conservative stamp.
* FRED-fetched rows: ``available_at = next BUSINESS day 18:00 ET``. FRED
  refreshes these daily series with a lag (H.15 yields and the ICE BofA OAS
  appear on FRED the following business day; DFF is published the next
  business day). Weekend as-of dates (DFF prints Sat/Sun) resolve to the
  following Monday. Conservative by construction; documented here.

Cache: parquet under ``data_cache/edge/rates/`` (``cache_root``), cache-first;
the yields cache is rebuilt automatically when any archive CSV is newer than
the parquet. Writes are atomic (tmp + replace).
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: Tenor headers that MUST be present in every yields_{YYYY}.csv.
REQUIRED_TENORS: tuple[str, ...] = ("2 Yr", "10 Yr")

#: FRED series ids served by the fredgraph fallback fetcher.
FRED_SERIES: tuple[str, ...] = ("DGS2", "DGS10", "VIXCLS", "BAMLH0A0HYM2", "DFF")

#: The FRED series used for the high-yield OAS feature.
HY_OAS_SERIES = "BAMLH0A0HYM2"

#: header unit token -> canonical one-letter suffix ("3 Mo" -> y_3m).
_UNIT_SUFFIX: dict[str, str] = {
    "mo": "m",
    "month": "m",
    "months": "m",
    "yr": "y",
    "year": "y",
    "years": "y",
}

_TENOR_NUMBER = re.compile(r"\d+(\.\d+)?")


class RatesConfig(BaseModel):
    """Adapter knobs; defaults are the documented publication schedule and the
    spec'd feature windows — overridable per environment, never baked into
    code paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hour (ET) at which a day's value is stamped available. 18:00 is
    #: deliberately later than Treasury's ~3 PM posting (conservative).
    publish_hour_et: int = Field(default=18, ge=0, le=23)
    #: FRED refreshes these daily series the following business day.
    fred_lag_business_days: int = Field(default=1, ge=1)
    #: Trailing z-score window (observations) for the 10y-2y slope.
    slope_z_window: int = Field(default=252, gt=1)
    #: Trailing z-score window (observations) for HY OAS.
    hy_oas_z_window: int = Field(default=63, gt=1)
    fred_base_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)


class _Response(Protocol):
    status_code: int
    text: str


@runtime_checkable
class HttpGetter(Protocol):
    """The one slice of ``httpx.Client`` the FRED fetcher uses — tests inject
    a fake with canned CSV text; production injects a real client."""

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _Response: ...


def _tenor_column(header: str) -> str:
    """Canonical column name for a Treasury tenor header, strictly validated.

    ``"2 Yr" -> "y_2y"``, ``"3 Mo" -> "y_3m"``, ``"1.5 Month" -> "y_1_5m"``.
    An unrecognized header raises — headers vary by year and silently
    guessing one would corrupt every downstream join.
    """
    parts = header.strip().split()
    if len(parts) == 2:
        number, unit = parts
        suffix = _UNIT_SUFFIX.get(unit.lower())
        if suffix is not None and _TENOR_NUMBER.fullmatch(number):
            return f"y_{number.replace('.', '_')}{suffix}"
    raise ValueError(
        f"unrecognized tenor column {header!r} in Treasury yields CSV — "
        "tenor headers are parsed strictly by name"
    )


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via tmp + replace so a crash can never leave a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _same_day_available(asof: pd.Series, hour: int) -> pd.Series:
    """asof (midnight datetime64) -> same-day ``hour``:00 America/New_York."""
    return (asof + pd.Timedelta(hours=hour)).dt.tz_localize(MARKET_TZ)


def _next_bday_available(asof: pd.Series, hour: int, lag_days: int) -> pd.Series:
    """asof -> ``lag_days`` BUSINESS days later at ``hour``:00 ET.

    ``roll="backward"`` first snaps a weekend as-of date to the PRECEDING
    business day, so the offset always lands on the strictly-next business
    day: Fri -> Mon, Sat -> Mon, Sun -> Mon, Wed -> Thu. (``roll="forward"``
    would push Saturday data to Tuesday — later than reality with no
    conservatism gained.)
    """
    days = np.busday_offset(
        asof.to_numpy().astype("datetime64[D]"), lag_days, roll="backward"
    )
    stamped = pd.Series(pd.to_datetime(days), index=asof.index) + pd.Timedelta(hours=hour)
    return stamped.dt.tz_localize(MARKET_TZ)


def _trailing_z(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score over the last ``window`` observations INCLUDING the
    current one — each row sees only its own past, so the statistic is
    point-in-time by construction. NaN until a full window exists
    (``min_periods=window``); std uses ddof=1 (pandas default)."""
    rolling = series.rolling(window=window, min_periods=window)
    return (series - rolling.mean()) / rolling.std()


class RatesFeed:
    """Cache-first rates/credit adapter over the Treasury archive + FRED.

    Contract: every public method returns a frame sorted ascending by
    ``asof_date`` with both point-in-time columns (``asof_date``,
    ``available_at``) present. No method used by tests touches the network;
    :meth:`fetch_fred` / :meth:`update_fred` are the incremental daily
    fallback and take an injected HTTP client.
    """

    _YIELDS_CACHE = "treasury_yields.parquet"

    def __init__(
        self,
        archive_root: str | Path,
        cache_root: str | Path,
        config: RatesConfig | None = None,
        client: HttpGetter | None = None,
    ) -> None:
        self._archive_root = Path(archive_root)
        self._cache_root = Path(cache_root)
        self._config = config or RatesConfig()
        self._client = client

    @classmethod
    def from_repo(
        cls,
        repo_root: str | Path | None = None,
        config: RatesConfig | None = None,
        client: HttpGetter | None = None,
    ) -> RatesFeed:
        """Build against the repo's standard archive and cache locations."""
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
        return cls(
            archive_root=root / "data_cache" / "edge" / "raw" / "treasury_yields",
            cache_root=root / "data_cache" / "edge" / "rates",
            config=config,
            client=client,
        )

    # ------------------------------------------------------------------
    # Treasury par-yield archive
    # ------------------------------------------------------------------

    def load_yields(self) -> pd.DataFrame:
        """Full par-yield history from the archive, cache-first.

        Columns: ``asof_date``, ``available_at``, ``y_2y``, ``y_10y``, then
        every other tenor seen in any year (alphabetical), NaN where a year's
        file lacks that tenor or the cell is empty. One row per date,
        ascending; a date appearing in two files keeps the later file's row.
        The parquet cache is served only while it is at least as new as every
        archive CSV — a refreshed archive forces a rebuild.
        """
        csvs = sorted(self._archive_root.glob("yields_*.csv"))
        if not csvs:
            raise FileNotFoundError(f"no yields_*.csv under {self._archive_root}")
        cache = self._cache_root / self._YIELDS_CACHE
        newest_src = max(p.stat().st_mtime for p in csvs)
        if cache.exists() and cache.stat().st_mtime >= newest_src:
            try:
                return pd.read_parquet(cache)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable yields cache %s: %s — rebuilding", cache, exc)
        merged = pd.concat(
            [self._parse_year_csv(path) for path in csvs], ignore_index=True
        )
        merged = (
            merged.sort_values("asof_date", kind="stable")
            .drop_duplicates("asof_date", keep="last")
            .reset_index(drop=True)
        )
        merged["available_at"] = _same_day_available(
            merged["asof_date"], self._config.publish_hour_et
        )
        required = [_tenor_column(t) for t in REQUIRED_TENORS]
        optional = sorted(
            c for c in merged.columns if c.startswith("y_") and c not in required
        )
        merged = merged[["asof_date", "available_at", *required, *optional]]
        _write_atomic(cache, merged)
        return merged

    def _parse_year_csv(self, path: Path) -> pd.DataFrame:
        """One yields_{YYYY}.csv -> tidy frame, strictly by header name.

        ``Date`` and every REQUIRED tenor must be present; any tenor header
        the normalizer does not recognize raises. Empty cells / "N/A" become
        NaN; anything else non-numeric raises rather than being coerced.
        """
        raw = pd.read_csv(path, dtype=str)
        if "Date" not in raw.columns:
            raise ValueError(f"{path.name}: no 'Date' column")
        missing = [t for t in REQUIRED_TENORS if t not in raw.columns]
        if missing:
            raise ValueError(
                f"{path.name}: missing required tenor column(s) {missing}; "
                "tenor headers vary by year and are parsed strictly by name"
            )
        out = pd.DataFrame(
            {"asof_date": pd.to_datetime(raw["Date"], format="%m/%d/%Y")}
        )
        for header in raw.columns:
            if header == "Date":
                continue
            out[_tenor_column(header)] = pd.to_numeric(raw[header], errors="raise")
        return out

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def features(self, hy_oas: pd.DataFrame | None = None) -> pd.DataFrame:
        """Point-in-time rates features, one row per Treasury as-of date.

        Columns: ``asof_date``, ``available_at`` (same-day 18:00 ET — the
        Treasury stamp), ``level_2y``, ``slope_10y_2y`` (10y − 2y),
        ``slope_z_{W}d`` (trailing W-observation z, default W=252). When HY
        OAS data is present — passed explicitly or found in this adapter's
        FRED cache for BAMLH0A0HYM2 — the frame also carries ``hy_oas`` and
        ``hy_oas_z_{W}d`` (default W=63), attached via a backward as-of merge
        on ``available_at``: each feature row sees the LATEST OAS print whose
        own ``available_at`` (next business day 18:00 ET) is <= the row's.
        The OAS therefore lags the yield columns by one business day — that
        is the real information flow, not a bug. Absent HY data, the two
        columns are omitted entirely.
        """
        cfg = self._config
        yields_frame = self.load_yields()
        out = yields_frame[["asof_date", "available_at"]].copy()
        out["level_2y"] = yields_frame["y_2y"]
        out["slope_10y_2y"] = yields_frame["y_10y"] - yields_frame["y_2y"]
        out[f"slope_z_{cfg.slope_z_window}d"] = _trailing_z(
            out["slope_10y_2y"], cfg.slope_z_window
        )
        hy = self._hy_frame(hy_oas)
        if hy is not None and not hy.empty:
            z_col = f"hy_oas_z_{cfg.hy_oas_z_window}d"
            hy = (
                hy.sort_values("available_at", kind="stable")
                .drop_duplicates("available_at", keep="last")
                .reset_index(drop=True)
            )
            hy[z_col] = _trailing_z(hy["hy_oas"], cfg.hy_oas_z_window)
            out = pd.merge_asof(
                out.sort_values("available_at", kind="stable"),
                hy[["available_at", "hy_oas", z_col]],
                on="available_at",
                direction="backward",
            )
        return out.reset_index(drop=True)

    def _hy_frame(self, hy_oas: pd.DataFrame | None) -> pd.DataFrame | None:
        """Normalize a caller-supplied HY OAS frame, else read the FRED cache.

        Accepts the value column as either ``hy_oas`` or the raw series id
        ``BAMLH0A0HYM2``; requires both point-in-time columns. Returns None
        when no HY data exists anywhere — features then omit the columns.
        """
        if hy_oas is None:
            cache = self._fred_cache_path(HY_OAS_SERIES)
            if not cache.exists():
                return None
            try:
                hy_oas = pd.read_parquet(cache)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable HY OAS cache %s: %s — skipping", cache, exc)
                return None
        frame = hy_oas.rename(columns={HY_OAS_SERIES: "hy_oas"})
        needed = {"asof_date", "available_at", "hy_oas"}
        if not needed.issubset(frame.columns):
            raise ValueError(
                f"HY OAS frame must carry columns {sorted(needed)} "
                f"(value column may be named {HY_OAS_SERIES!r}); got {list(frame.columns)}"
            )
        return frame[["asof_date", "available_at", "hy_oas"]]

    # ------------------------------------------------------------------
    # FRED fredgraph fallback (incremental daily fetch — never used in tests)
    # ------------------------------------------------------------------

    def _fred_cache_path(self, series: str) -> Path:
        return self._cache_root / f"fred_{series}.parquet"

    def fetch_fred(self, series: str, client: HttpGetter | None = None) -> pd.DataFrame:
        """Incrementally refresh one FRED series via keyless fredgraph.csv.

        Reads the existing parquet cache, requests only observations from the
        last cached as-of date onward (``cosd``), merges (a re-fetched date's
        NEW value wins — FRED revises), writes the cache atomically, and
        returns the full merged history with both point-in-time columns
        (``available_at`` = next business day 18:00 ET, conservative — see
        module docstring). Raises on a non-200 response; the cache is never
        touched by a failed fetch.
        """
        if series not in FRED_SERIES:
            raise ValueError(f"unknown FRED series {series!r}; expected one of {FRED_SERIES}")
        cache = self._fred_cache_path(series)
        existing: pd.DataFrame | None = None
        if cache.exists():
            try:
                existing = pd.read_parquet(cache)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable FRED cache %s: %s — full refetch", cache, exc)
        params: dict[str, str] = {"id": series}
        if existing is not None and not existing.empty:
            params["cosd"] = existing["asof_date"].max().date().isoformat()
        http = client or self._client or self._default_client()
        resp = http.get(self._config.fred_base_url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(
                f"fredgraph {series} -> HTTP {resp.status_code}: {resp.text[:160]}"
            )
        fresh = self._parse_fredgraph(resp.text, series)
        merged = fresh if existing is None else pd.concat([existing, fresh], ignore_index=True)
        merged = (
            merged.sort_values("asof_date", kind="stable")
            .drop_duplicates("asof_date", keep="last")
            .reset_index(drop=True)
        )
        _write_atomic(cache, merged)
        return merged

    def update_fred(self, client: HttpGetter | None = None) -> dict[str, pd.DataFrame]:
        """Incrementally refresh every series in :data:`FRED_SERIES`."""
        return {series: self.fetch_fred(series, client=client) for series in FRED_SERIES}

    def _parse_fredgraph(self, text: str, series: str) -> pd.DataFrame:
        """fredgraph.csv payload -> point-in-time frame.

        The first column is the observation date regardless of its header
        (FRED has shipped both ``DATE`` and ``observation_date``); the series
        column must match the requested id exactly. ``"."`` marks a missing
        observation (holiday/not-yet-published) and the row is dropped — it
        carries no data. Columns: ``asof_date``, ``available_at``, ``<series>``.
        """
        raw = pd.read_csv(io.StringIO(text), dtype=str)
        if raw.shape[1] < 2 or series not in raw.columns:
            raise ValueError(
                f"unexpected fredgraph payload for {series!r}: columns {list(raw.columns)}"
            )
        date_col = raw.columns[0]
        out = pd.DataFrame(
            {
                "asof_date": pd.to_datetime(raw[date_col], format="%Y-%m-%d"),
                series: pd.to_numeric(raw[series].mask(raw[series] == "."), errors="raise"),
            }
        )
        out = out.dropna(subset=[series]).reset_index(drop=True)
        out["available_at"] = _next_bday_available(
            out["asof_date"],
            self._config.publish_hour_et,
            self._config.fred_lag_business_days,
        )
        return out[["asof_date", "available_at", series]]

    def _default_client(self) -> HttpGetter:
        """Real HTTP client for production incremental fetches (never tests)."""
        client: Any = httpx.Client(timeout=self._config.request_timeout_seconds)
        return client

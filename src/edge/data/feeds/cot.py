"""CFTC Commitments of Traders — Traders in Financial Futures (TFF), futures only.

Parses the CFTC's yearly TFF zip archives (``fut_fin_txt_YYYY.zip``, each
containing a member literally named ``FinFutYY.txt`` regardless of year) from
the local raw archive — NEVER the network — and emits weekly positioning
features for the two pinned equity-index futures markets.

PINNED MARKETS — matched on ``CFTC_Contract_Market_Code``, never on
``Market_and_Exchange_Names`` (display names drift across years: the ES row
has appeared as both "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE
EXCHANGE" and "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"):

* ``13874A`` -> ``ES``  (E-mini S&P 500, CME)
* ``209742`` -> ``NQ``  (E-mini NASDAQ-100, CME)

Code ``13874A`` is deliberately distinct from ``138741`` (S&P 500
Consolidated) — matching is exact-string after whitespace strip.

FEATURES per trader class (dealer / asset_mgr / lev_money / other_rept):

* ``{cls}_net_oi``  = (Long - Short) / Open_Interest_All
* ``{cls}_net_oi_z`` = rolling z-score of ``{cls}_net_oi`` over the trailing
  ``z_window_weeks`` (default 52) weekly reports PER MARKET, window ending at
  the current row — the value at week *t* uses only weeks ``t-51..t``, so
  there is no lookahead. Rows with fewer than ``z_min_periods`` (default 52,
  i.e. a full window) trailing reports are NaN; a zero-variance window is NaN
  rather than +/-inf.

POINT-IN-TIME CONVENTION — THE TRAP THIS SOURCE EXISTS TO TEST. Each TFF
report is **as of Tuesday's close but only released Friday ~15:30 ET** (CFTC
publishes "3:30 p.m. Eastern time, Friday"). A live trader could not have
seen Tuesday's positioning on Wednesday or Thursday. Therefore::

    available_at = Friday of the SAME week as asof_date, 16:00 ET

(16:00, not 15:30 — a conservative +30 minutes for dissemination lag). All
downstream feature joins must key on ``available_at <= decision_ts``; joining
on ``asof_date`` is exactly the lookahead bug this convention exists to
prevent. Holiday-shortened weeks (~3 per year, e.g. Good Friday): the CFTC
delays the release past Friday to the next business day, so when the
computed Friday is NOT an NYSE trading day the stamp rolls FORWARD to the
next NYSE trading day at 16:00 ET (repro that motivated the rule: asof
2024-03-26 -> Good Friday 2024-03-29 is an exchange holiday -> available
Monday 2024-04-01 16:00 ET, matching the CFTC's actual ~15:30 ET Monday
release). A weekend ``asof_date`` (never observed in practice) is stamped
to the NEXT Friday — later, hence conservative.

CACHE — parquet under ``data_cache/edge/cot/``, cache-first: once
``tff_features.parquet`` exists it is served without re-reading the raw zips;
``features(refresh=True)`` rebuilds. Writes are atomic (tmp + replace).

This is a BACKEND-side module: research code must never import it directly —
all research access goes through ``edge.data.loader.EdgeDataLoader`` (the AST
gateway scan enforces the wall; ``edge/data/`` itself is exempt).

Raw archives are provisioned out-of-band (ops download the yearly zips from
cftc.gov into ``data_cache/edge/raw/cot/``); this module has NO network code
path at all.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from functools import cache
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: CFTC contract market code -> internal market symbol. Codes verified against
#: the TFF futures-only file for report date 2024-12-31 (see module docstring);
#: display names are NOT used for matching because they drift across years.
PINNED_MARKETS: Mapping[str, str] = {
    "13874A": "ES",  # E-MINI S&P 500 (CME)
    "209742": "NQ",  # E-MINI NASDAQ-100 (CME)
}

#: Output column prefix -> TFF source column prefix, one per trader class.
TRADER_CLASSES: Mapping[str, str] = {
    "dealer": "Dealer_Positions",
    "asset_mgr": "Asset_Mgr_Positions",
    "lev_money": "Lev_Money_Positions",
    "other_rept": "Other_Rept_Positions",
}

_NAME_COL = "Market_and_Exchange_Names"
_DATE_COL = "Report_Date_as_YYYY-MM-DD"
#: Header the REAL 2010-2012 yearly archives use for the same column. Despite
#: the MM_DD_YYYY name, every value in those files is ISO ``YYYY-MM-DD``
#: (verified across all rows of the real 2010-2012 archives), so the strict
#: ``%Y-%m-%d`` parse below applies unchanged — a file whose values really
#: were MM/DD/YYYY would fail loudly rather than be misread.
_LEGACY_DATE_COL = "Report_Date_as_MM_DD_YYYY"
_CODE_COL = "CFTC_Contract_Market_Code"
_OI_COL = "Open_Interest_All"

#: Source columns the parser requires; selection is by NAME so extra columns
#: and column-order drift in the raw file are harmless. The report-date
#: column is handled separately (``_DATE_COL`` with ``_LEGACY_DATE_COL`` as
#: the 2010-2012 alias) in :meth:`CotTff._parse_member`.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    _NAME_COL,
    _CODE_COL,
    _OI_COL,
    *(
        f"{prefix}_{side}_All"
        for prefix in TRADER_CLASSES.values()
        for side in ("Long", "Short", "Spread")
    ),
)


class CotTffConfig(BaseModel):
    """Adapter knobs. Defaults encode the CFTC TFF publication facts and the
    task-standard feature window; overridable per environment — nothing below
    is baked into code paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: CFTC_Contract_Market_Code -> internal symbol (exact match after strip).
    markets: dict[str, str] = Field(default_factory=lambda: dict(PINNED_MARKETS))
    #: Rolling z-score window, in weekly reports.
    z_window_weeks: int = Field(default=52, gt=1)
    #: Minimum trailing reports before a z-score is emitted (default: full
    #: window — a z on fewer weeks is a different statistic).
    z_min_periods: int = Field(default=52, gt=1)
    #: Release stamp: Friday of the report week at this ET wall-clock time.
    #: 16:00 = actual ~15:30 release + conservative 30 minutes.
    release_hour_et: int = Field(default=16, ge=0, le=23)
    release_minute_et: int = Field(default=0, ge=0, le=59)


@cache
def _nyse_trading_day_on_or_after(day: date) -> date:
    """``day`` itself when NYSE trades that day, else the next trading day.

    Cached per date — the weekly release Fridays repeat heavily across the
    per-row ``available_at`` mapping, and the NYSE schedule is immutable.
    """
    valid = mcal.get_calendar("NYSE").valid_days(day, day + timedelta(days=21))
    if len(valid) == 0:  # pragma: no cover — 21-day pad spans any holiday run
        raise RuntimeError(f"no NYSE trading day found within 21 days of {day}")
    return valid[0].date()


def tff_available_at(asof: date, *, hour: int = 16, minute: int = 0) -> pd.Timestamp:
    """First moment a live trader could have seen the report for ``asof``.

    Friday of the same Mon-Sun week as ``asof`` at ``hour:minute`` ET
    (default 16:00 — the ~15:30 Friday release plus a conservative 30
    minutes). When that Friday is NOT an NYSE trading day (holiday-shortened
    week, e.g. Good Friday) the CFTC delays the release, so the stamp rolls
    forward to the NEXT NYSE trading day at the same wall-clock time. A
    Sat/Sun ``asof`` (never observed) maps to the NEXT Friday, which is
    later and therefore conservative.
    """
    days_to_friday = (4 - asof.weekday()) % 7  # Mon=0 .. Fri=4; Sat/Sun -> next Friday
    release_day = _nyse_trading_day_on_or_after(asof + timedelta(days=days_to_friday))
    return pd.Timestamp(datetime.combine(release_day, time(hour, minute)), tz=MARKET_TZ)


class CotTff:
    """Cache-first TFF positioning adapter over the local yearly-zip archive.

    Contract: :meth:`features` returns one row per (report week, pinned
    market) with ``asof_date`` (datetime64[ns], the report's own Tuesday) and
    ``available_at`` (tz-aware America/New_York, Friday 16:00 ET of the same
    week — rolled forward to the next NYSE trading day when that Friday is
    an exchange holiday) plus per-class net/OI and rolling-z features,
    sorted by (``asof_date``, ``market_code``), deduplicated per
    (market, week).
    """

    _CACHE_FILE = "tff_features.parquet"

    def __init__(
        self,
        raw_dir: str | Path,
        cache_root: str | Path,
        config: CotTffConfig | None = None,
    ) -> None:
        self._raw_dir = Path(raw_dir)
        self._cache_root = Path(cache_root)
        self._config = config or CotTffConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def features(self, *, refresh: bool = False) -> pd.DataFrame:
        """Weekly positioning features for the pinned markets, cache-first.

        Served from ``<cache_root>/tff_features.parquet`` when present (the
        raw archive is not touched at all); ``refresh=True`` — or an
        unreadable cache file — rebuilds from the raw zips and rewrites the
        cache atomically.
        """
        cache_path = self._cache_root / self._CACHE_FILE
        if cache_path.exists() and not refresh:
            try:
                return pd.read_parquet(cache_path)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable cache %s: %s — rebuilding", cache_path, exc)
        frame = self._build_features(self.positions())
        _write_atomic(cache_path, frame)
        return frame

    def positions(self) -> pd.DataFrame:
        """Parsed raw positions for the pinned markets (uncached, no features).

        One row per (report week, market): ``asof_date``, ``available_at``,
        ``market``, ``market_code``, ``market_name``, ``open_interest``, and
        ``{cls}_{long,short,spread}`` ints per trader class. Raises
        ``FileNotFoundError`` when the raw archive holds no TFF files, and
        ``ValueError`` on a file missing required columns — loud, never
        silently partial.
        """
        frames = [self._parse_member(name, buf) for name, buf in self._iter_members()]
        if not frames:
            raise FileNotFoundError(
                f"no TFF archives (fut_fin_txt_*.zip / FinFut*.txt) under {self._raw_dir}"
            )
        merged = pd.concat(frames, ignore_index=True)
        # Yearly files should not overlap; dedupe is cheap insurance. keep="last"
        # prefers the later archive in sorted filename order.
        merged = (
            merged.sort_values(["market_code", "asof_date"], kind="stable")
            .drop_duplicates(["market_code", "asof_date"], keep="last")
            .reset_index(drop=True)
        )
        return merged

    # ------------------------------------------------------------------
    # Archive iteration + parsing
    # ------------------------------------------------------------------

    def _iter_members(self) -> list[tuple[str, io.BytesIO]]:
        """(label, buffer) for every FinFut*.txt in the raw dir, zipped or loose.

        Zips are scanned for members whose basename matches ``finfut*.txt``
        (the CFTC yearly zip names its member ``FinFutYY.txt`` for EVERY
        year); loose ``FinFut*.txt`` files are accepted too. Deterministic:
        sorted by path then member name.
        """
        if not self._raw_dir.is_dir():
            return []
        members: list[tuple[str, io.BytesIO]] = []
        for path in sorted(self._raw_dir.iterdir()):
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    for member in sorted(zf.namelist()):
                        base = Path(member).name.lower()
                        if base.startswith("finfut") and base.endswith(".txt"):
                            members.append((f"{path.name}!{member}", io.BytesIO(zf.read(member))))
            elif path.name.lower().startswith("finfut") and path.suffix.lower() == ".txt":
                members.append((path.name, io.BytesIO(path.read_bytes())))
        return members

    def _parse_member(self, label: str, buf: io.BytesIO) -> pd.DataFrame:
        """One FinFutYY.txt -> normalized rows for the pinned markets only."""
        cfg = self._config
        raw = pd.read_csv(buf, dtype=str, skipinitialspace=True)
        missing = [c for c in _REQUIRED_COLUMNS if c not in raw.columns]
        date_col = next((c for c in (_DATE_COL, _LEGACY_DATE_COL) if c in raw.columns), None)
        if date_col is None:
            missing.append(f"{_DATE_COL} (or legacy {_LEGACY_DATE_COL})")
        if missing:
            raise ValueError(f"TFF file {label} is missing required columns: {missing}")

        code = raw[_CODE_COL].fillna("").str.strip()
        sub = raw.loc[code.isin(cfg.markets)].copy()
        sub[_CODE_COL] = code.loc[sub.index]

        out = pd.DataFrame(index=sub.index)
        # Strict ISO parse for BOTH header spellings: the legacy-named column
        # in the real 2010-2012 files also holds YYYY-MM-DD values (see
        # _LEGACY_DATE_COL note); anything else raises rather than misparse.
        out["asof_date"] = pd.to_datetime(sub[date_col].str.strip(), format="%Y-%m-%d").astype(
            "datetime64[ns]"
        )
        out["available_at"] = pd.to_datetime(
            out["asof_date"].map(
                lambda ts: tff_available_at(
                    ts.date(), hour=cfg.release_hour_et, minute=cfg.release_minute_et
                )
            )
        ).astype(f"datetime64[ns, {MARKET_TZ.key}]")
        out["market_code"] = sub[_CODE_COL]
        out["market"] = out["market_code"].map(cfg.markets)
        out["market_name"] = sub[_NAME_COL].str.strip()
        out["open_interest"] = _to_int(sub[_OI_COL], _OI_COL, label)
        for cls, prefix in TRADER_CLASSES.items():
            for side in ("Long", "Short", "Spread"):
                src = f"{prefix}_{side}_All"
                out[f"{cls}_{side.lower()}"] = _to_int(sub[src], src, label)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _build_features(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Net-position-per-OI and trailing z-score per trader class per market.

        Rolling stats are computed per ``market_code`` on rows sorted
        ascending by ``asof_date`` with the window ENDING at each row — the
        z at week *t* sees only weeks ``<= t`` (no lookahead). Zero-variance
        windows and non-positive open interest yield NaN, never inf.
        """
        cfg = self._config
        frame = positions.sort_values(["market_code", "asof_date"], kind="stable").reset_index(
            drop=True
        )
        oi = frame["open_interest"].where(frame["open_interest"] > 0)
        for cls in TRADER_CLASSES:
            net_col = f"{cls}_net_oi"
            frame[net_col] = (frame[f"{cls}_long"] - frame[f"{cls}_short"]) / oi
            roll = frame.groupby("market_code", sort=False)[net_col].rolling(
                window=cfg.z_window_weeks, min_periods=cfg.z_min_periods
            )
            mean = roll.mean().reset_index(level=0, drop=True)
            std = roll.std().reset_index(level=0, drop=True)  # ddof=1
            std = std.where(std > 0)
            frame[f"{net_col}_z"] = (frame[net_col] - mean) / std
        return frame.sort_values(["asof_date", "market_code"], kind="stable").reset_index(drop=True)


def _to_int(series: pd.Series, column: str, label: str) -> pd.Series:
    """Strict str -> int64: whitespace stripped, anything non-integral raises.

    A silently-NaN'd open interest or position would corrupt every downstream
    feature, so garbage fails loudly with the file and column named. That
    includes fractional values ('100000.5'): position counts are integers,
    so a fractional cell is corrupt data — it raises instead of being
    silently truncated by the int64 cast.
    """
    try:
        numeric = pd.to_numeric(series.str.strip(), errors="raise")
        as_int = numeric.astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"TFF file {label}: column {column} has non-integer data") from exc
    if not (as_int == numeric).all():
        raise ValueError(
            f"TFF file {label}: column {column} has non-integral values "
            "— refusing to truncate silently"
        )
    return as_int


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via tmp + replace so a crash or concurrent writer can
    never leave a torn file at the final path (same pattern as alpaca_ticks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

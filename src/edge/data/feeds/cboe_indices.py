"""CBOE volatility-index closes and put/call ratios, point-in-time.

BACKEND-side adapter (research code never imports this — the AST gateway
scan enforces that; ``edge/data/`` is the exempt side of the wall). Parses
the raw archives already on disk; **no code path here touches the network**.

Sources parsed
--------------
(a) ``{VIX,VIX9D,VIX3M,VIX6M}_History.csv`` — header ``DATE,OPEN,HIGH,LOW,
    CLOSE``, MM/DD/YYYY dates — and close-only ``VVIX_History.csv`` /
    ``SKEW_History.csv`` (header ``DATE,<NAME>``).
(b) Put/call volume: the frozen ``totalpc_2006_2019.csv`` (disclaimer +
    product preamble rows before the ``DATE,CALLS,PUTS,TOTAL,P/C Ratio``
    header) merged with per-day ``daily/{YYYY-MM-DD}.json`` files (a
    ``ratios`` array of TOTAL / INDEX / ETP / EQUITY ... put/call ratios
    plus per-segment volume blocks) into one continuous 2006->present
    series, de-duplicated on the overlap (the richer daily JSON wins).

Data hygiene (flagged, never silently dropped)
----------------------------------------------
* Known bad prints — early VVIX contains impossible values (e.g. 15.71 on
  2006-03-15; VVIX has never traded near that). A close that strays outside
  ``[winsor_low_ratio, winsor_high_ratio]`` times the TRAILING rolling
  median of the prior 5 closes (min 3 observations) is winsorized to that
  median, the row keeps ``winsorized=True``, and each replacement is logged
  at WARNING. Trailing, never centered: the cleaned value at *t* must
  depend only on closes ``<= t-1`` (a centered window would leak t+1/t+2
  into the cleaning step) and the newest row must be judgeable live.
  Applied by default only to the close-only series (VVIX, SKEW) — the
  bounds are wide enough that no historical genuine VIX-family spike would
  trip them, but OHLC series have no known bad prints so they are left
  untouched by default.
* Early VIX-family rows are backfilled closes with OHLC all equal — those
  rows carry ``backfilled=True`` so downstream code can exclude them from
  anything that needs a real intraday range.

Point-in-time convention (non-negotiable)
-----------------------------------------
Every output frame carries ``asof_date`` (the data's own ET trading date,
naive midnight timestamp) and ``available_at`` (tz-aware America/New_York:
the first moment a live trader could have seen the value). CBOE publishes
final index settlements shortly after the 16:15 ET close and posts the daily
market-statistics page (put/call ratios) the same evening; the exact minute
is not archived, so we conservatively stamp **18:00 ET on the same trading
day** for both index closes and put/call — later than any plausible actual
publication, which makes each value first usable at the NEXT session. All
downstream feature joins key on ``available_at <= decision_ts``.

Caching: parquet under ``data_cache/edge/cboe_indices/``, cache-first —
pass ``refresh=True`` to force a re-parse of the raw archive. Writes are
atomic (tmp + replace).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: Index name -> raw archive filename.
INDEX_FILES: Mapping[str, str] = {
    "VIX": "VIX_History.csv",
    "VIX9D": "VIX9D_History.csv",
    "VIX3M": "VIX3M_History.csv",
    "VIX6M": "VIX6M_History.csv",
    "VVIX": "VVIX_History.csv",
    "SKEW": "SKEW_History.csv",
}

#: Series shipped with full OHLC columns; the rest are close-only.
OHLC_INDEXES = frozenset({"VIX", "VIX9D", "VIX3M", "VIX6M"})

#: ``ratios[].name`` in the daily JSON -> output column.
_RATIO_COLUMNS: Mapping[str, str] = {
    "TOTAL PUT/CALL RATIO": "total_pc",
    "INDEX PUT/CALL RATIO": "index_pc",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp_pc",
    "EQUITY PUT/CALL RATIO": "equity_pc",
}

_PC_RATIO_COLS = ("total_pc", "index_pc", "etp_pc", "equity_pc")
_PC_COUNT_COLS = ("calls", "puts", "total")

_FROZEN_PC_FILENAME = "totalpc_2006_2019.csv"
_DAILY_PC_DIRNAME = "daily"


class CboeConfig(BaseModel):
    """Adapter knobs; every number below is overridable — nothing numeric is
    baked into code paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Same-trading-day hour (ET) stamped as ``available_at``. 18:00 is
    #: deliberately LATER than CBOE's actual evening publication — the
    #: conservative choice under the point-in-time convention.
    publish_hour_et: int = Field(default=18, ge=0, le=23)
    #: Trailing rolling-median window for bad-print detection: the median of
    #: the PRIOR this-many closes (the suspect print itself is excluded).
    winsor_window: int = Field(default=5, gt=1)
    #: Minimum prior observations in the trailing window before a print can
    #: be judged.
    winsor_min_periods: int = Field(default=3, gt=1)
    #: Flag a close below ``low_ratio * median`` ...
    winsor_low_ratio: float = Field(default=0.6, gt=0.0, lt=1.0)
    #: ... or above ``high_ratio * median``. Bounds are wide enough that no
    #: genuine historical VIX-family spike trips them (largest one-day close
    #: moves sit well inside 2.5x the trailing 5-day median).
    winsor_high_ratio: float = Field(default=2.5, gt=1.0)
    #: Series the bad-print winsorization applies to (close-only series;
    #: the OHLC series have no known bad prints).
    winsorize_series: tuple[str, ...] = ("VVIX", "SKEW")
    #: Rolling window (trading days) for the put/call z-scores.
    pc_z_window: int = Field(default=21, gt=1)


class CboeIndices:
    """Cache-first adapter over the CBOE raw archives on disk.

    Contract: every public method returns a frame sorted ascending by
    ``asof_date`` (naive midnight, one row per trading date) with a tz-aware
    America/New_York ``available_at`` column stamped per the module
    docstring. Parquet cache under ``cache_dir`` is authoritative once
    written; ``refresh=True`` re-parses the raw archive and rewrites it.
    """

    def __init__(
        self,
        raw_indices_dir: str | Path,
        raw_pc_dir: str | Path,
        cache_dir: str | Path,
        config: CboeConfig | None = None,
    ) -> None:
        self._raw_indices_dir = Path(raw_indices_dir)
        self._raw_pc_dir = Path(raw_pc_dir)
        self._cache_dir = Path(cache_dir)
        self._config = config or CboeConfig()

    @classmethod
    def from_repo(cls, repo_root: str | Path | None = None, config: CboeConfig | None = None) -> CboeIndices:
        """Wire the standard repo layout: raw archives under
        ``data_cache/edge/raw/{cboe_indices,cboe_pc}``, parquet cache under
        ``data_cache/edge/cboe_indices``."""
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
        return cls(
            raw_indices_dir=root / "data_cache" / "edge" / "raw" / "cboe_indices",
            raw_pc_dir=root / "data_cache" / "edge" / "raw" / "cboe_pc",
            cache_dir=root / "data_cache" / "edge" / "cboe_indices",
            config=config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_history(self, name: str, refresh: bool = False) -> pd.DataFrame:
        """Daily history for one index.

        Columns: ``asof_date, open, high, low, close, backfilled,
        winsorized, available_at``. Close-only series (VVIX, SKEW) carry NaN
        OHLC and ``backfilled=False``; ``backfilled`` is True on OHLC rows
        where open==high==low==close (CBOE's backfilled early closes).
        ``winsorized`` marks bad prints replaced by the trailing rolling
        median of prior closes (see module docstring) — never silently
        dropped.
        """
        key = name.upper()
        if key not in INDEX_FILES:
            raise KeyError(f"unknown CBOE index {name!r}; expected one of {sorted(INDEX_FILES)}")
        cache = self._cache_dir / f"index_{key}.parquet"
        if not refresh:
            cached = _read_cache(cache)
            if cached is not None:
                return cached
        frame = self._parse_index(key)
        _write_atomic(cache, frame)
        return frame

    def put_call(self, refresh: bool = False) -> pd.DataFrame:
        """Merged continuous CBOE put/call series, 2006 -> present.

        Columns: ``asof_date, calls, puts, total, total_pc, index_pc,
        etp_pc, equity_pc, source, available_at``. The frozen CSV covers
        2006-11-01..2019-10-04 (total exchange volume only — the per-segment
        ratios are NaN there); per-day JSON files continue from 2019-10-07.
        On overlapping dates the daily JSON row wins (``source`` records
        which archive served each row). Counts are nullable Int64.
        """
        cache = self._cache_dir / "put_call.parquet"
        if not refresh:
            cached = _read_cache(cache)
            if cached is not None:
                return cached
        frame = self._parse_put_call()
        _write_atomic(cache, frame)
        return frame

    def features(self, refresh: bool = False) -> pd.DataFrame:
        """Daily volatility-structure and positioning features.

        Outer-joined on ``asof_date`` across all six indices and the
        put/call series (a missing leg yields NaN for the ratios needing
        it). Columns beyond the levels (``vix, vix9d, vix3m, vix6m, vvix,
        skew, total_pc, equity_pc``):

        * ``vix9d_vix``   = VIX9D / VIX   (short-end term-structure ratio)
        * ``vix_vix3m``   = VIX / VIX3M   (spot vs 3-month; >1 = backwardation)
        * ``vix3m_vix6m`` = VIX3M / VIX6M (belly of the curve)
        * ``vvix_vix``    = VVIX / VIX    (vol-of-vol richness)
        * ``skew``        = SKEW level (already a normalized index)
        * ``total_pc_z21`` / ``equity_pc_z21`` — rolling z-scores of the
          put/call ratios over ``pc_z_window`` trading days (default 21;
          the canonical column names keep the 21 suffix), mean/std (ddof=1)
          over a full window of valid observations, NaN until the window
          fills. The window includes the current day's value, which is fine
          point-in-time because the whole row shares its ``available_at``.

        ``available_at`` is 18:00 ET on ``asof_date`` — every input to a row
        publishes the same evening, so the row is first usable the NEXT
        session.
        """
        cache = self._cache_dir / "features.parquet"
        if not refresh:
            cached = _read_cache(cache)
            if cached is not None:
                return cached
        frame = self._build_features(refresh=refresh)
        _write_atomic(cache, frame)
        return frame

    # ------------------------------------------------------------------
    # Index parsing
    # ------------------------------------------------------------------

    def _parse_index(self, name: str) -> pd.DataFrame:
        path = self._raw_indices_dir / INDEX_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"CBOE archive missing: {path}")
        raw = pd.read_csv(path, skipinitialspace=True)
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        frame = pd.DataFrame({"asof_date": pd.to_datetime(raw["DATE"], format="%m/%d/%Y")})
        if name in OHLC_INDEXES:
            for col in ("open", "high", "low", "close"):
                frame[col] = raw[col.upper()].astype(float)
            frame["backfilled"] = (
                (frame["open"] == frame["high"])
                & (frame["high"] == frame["low"])
                & (frame["low"] == frame["close"])
            )
        else:
            frame["open"] = float("nan")
            frame["high"] = float("nan")
            frame["low"] = float("nan")
            frame["close"] = raw[name].astype(float)
            frame["backfilled"] = False
        frame = (
            frame.sort_values("asof_date", kind="stable")
            .drop_duplicates("asof_date", keep="last")
            .reset_index(drop=True)
        )
        if name in self._config.winsorize_series:
            cleaned, flag = self._winsorize(frame["asof_date"], frame["close"], name)
            frame["close"] = cleaned
            frame["winsorized"] = flag
        else:
            frame["winsorized"] = False
        frame["available_at"] = self._available_at(frame["asof_date"])
        return frame

    def _winsorize(
        self, dates: pd.Series, close: pd.Series, name: str
    ) -> tuple[pd.Series, pd.Series]:
        """Replace bad prints with the TRAILING rolling median, flag + log.

        The median at row *t* is over the PRIOR ``winsor_window`` raw closes
        (``t-window .. t-1``) — the suspect print itself is excluded, and no
        future close can influence the cleaned value at *t* (a centered
        window would be a lookahead in the cleaning step and could never
        judge the newest row live). Rows with fewer than
        ``winsor_min_periods`` prior observations are never judged.
        """
        cfg = self._config
        med = (
            close.shift(1)
            .rolling(cfg.winsor_window, min_periods=cfg.winsor_min_periods)
            .median()
        )
        bad = med.notna() & (
            (close < cfg.winsor_low_ratio * med) | (close > cfg.winsor_high_ratio * med)
        )
        for ts, raw_value, replacement in zip(dates[bad], close[bad], med[bad]):
            logger.warning(
                "CBOE %s bad print on %s: %.4f winsorized to trailing %d-day median %.4f",
                name,
                ts.date(),
                raw_value,
                cfg.winsor_window,
                replacement,
            )
        return close.where(~bad, med), bad

    # ------------------------------------------------------------------
    # Put/call parsing
    # ------------------------------------------------------------------

    def _parse_put_call(self) -> pd.DataFrame:
        frozen = self._read_frozen_pc(self._raw_pc_dir / _FROZEN_PC_FILENAME)
        frozen["source"] = "frozen"
        parts = [frozen]
        daily = self._read_daily_pc(self._raw_pc_dir / _DAILY_PC_DIRNAME)
        if not daily.empty:
            daily["source"] = "daily"
            parts.append(daily)
        pc = pd.concat(parts, ignore_index=True)
        # Dedup on overlap: daily JSON wins (richer — per-segment ratios).
        pc["_prio"] = (pc["source"] == "daily").astype(int)
        pc = (
            pc.sort_values(["asof_date", "_prio"], kind="stable")
            .drop_duplicates("asof_date", keep="last")
            .drop(columns="_prio")
            .reset_index(drop=True)
        )
        for col in _PC_COUNT_COLS:
            pc[col] = pc[col].astype("Int64")
        for col in _PC_RATIO_COLS:
            pc[col] = pc[col].astype(float)
        pc["available_at"] = self._available_at(pc["asof_date"])
        columns = ["asof_date", *_PC_COUNT_COLS, *_PC_RATIO_COLS, "source", "available_at"]
        return pc[columns]

    def _read_frozen_pc(self, path: Path) -> pd.DataFrame:
        """Frozen 2006-2019 CSV: disclaimer/product preamble rows precede the
        ``DATE,CALLS,PUTS,TOTAL,P/C Ratio`` header — located by scanning for
        the line whose first cell is DATE (robust to preamble length)."""
        if not path.exists():
            raise FileNotFoundError(f"CBOE put/call archive missing: {path}")
        lines = path.read_text().splitlines()
        header_idx = next(
            (i for i, ln in enumerate(lines) if ln.split(",")[0].strip().upper() == "DATE"),
            None,
        )
        if header_idx is None:
            raise ValueError(f"no DATE header line found in {path}")
        raw = pd.read_csv(path, skiprows=header_idx, skipinitialspace=True)
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        frame = pd.DataFrame(
            {
                "asof_date": pd.to_datetime(raw["DATE"], format="%m/%d/%Y"),
                "calls": raw["CALLS"],
                "puts": raw["PUTS"],
                "total": raw["TOTAL"],
                "total_pc": raw["P/C RATIO"].astype(float),
            }
        )
        for col in ("index_pc", "etp_pc", "equity_pc"):
            frame[col] = float("nan")  # frozen file is total-exchange only
        return frame

    def _read_daily_pc(self, daily_dir: Path) -> pd.DataFrame:
        """One row per parseable ``daily/{YYYY-MM-DD}.json``; a corrupt or
        misnamed file is logged and skipped, never crashes the merge (the
        archive may still be mid-download)."""
        columns = ["asof_date", *_PC_COUNT_COLS, *_PC_RATIO_COLS]
        if not daily_dir.is_dir():
            logger.warning("CBOE daily put/call dir missing: %s — frozen series only", daily_dir)
            return pd.DataFrame(columns=columns)
        rows: list[dict[str, Any]] = []
        for path in sorted(daily_dir.glob("*.json")):
            try:
                rows.append(self._parse_daily_pc_file(path))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("skipping unparseable CBOE daily P/C file %s: %s", path, exc)
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _parse_daily_pc_file(path: Path) -> dict[str, Any]:
        asof = date.fromisoformat(path.stem)  # ValueError on a misnamed file
        payload = json.loads(path.read_text())
        ratios = {
            str(entry["name"]).strip().upper(): float(entry["value"])
            for entry in payload.get("ratios", [])
        }
        row: dict[str, Any] = {"asof_date": pd.Timestamp(asof)}
        for source_name, column in _RATIO_COLUMNS.items():
            row[column] = ratios.get(source_name, float("nan"))
        volume = next(
            (
                entry
                for entry in payload.get("SUM OF ALL PRODUCTS", [])
                if str(entry.get("name", "")).strip().upper() == "VOLUME"
            ),
            None,
        )
        # float, NOT int: int() would silently truncate a fractional volume
        # (and reject numeric strings like "398.0"); the Int64 cast downstream
        # fails loudly on anything genuinely non-integral.
        row["calls"] = None if volume is None else float(volume["call"])
        row["puts"] = None if volume is None else float(volume["put"])
        row["total"] = None if volume is None else float(volume["total"])
        return row

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _build_features(self, refresh: bool) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for name in INDEX_FILES:
            leg = self.index_history(name, refresh=refresh)[["asof_date", "close"]].rename(
                columns={"close": name.lower()}
            )
            merged = leg if merged is None else merged.merge(leg, on="asof_date", how="outer")
        assert merged is not None
        pc = self.put_call(refresh=refresh)[["asof_date", "total_pc", "equity_pc"]]
        merged = merged.merge(pc, on="asof_date", how="outer")
        merged = merged.sort_values("asof_date", kind="stable").reset_index(drop=True)

        merged["vix9d_vix"] = merged["vix9d"] / merged["vix"]
        merged["vix_vix3m"] = merged["vix"] / merged["vix3m"]
        merged["vix3m_vix6m"] = merged["vix3m"] / merged["vix6m"]
        merged["vvix_vix"] = merged["vvix"] / merged["vix"]

        window = self._config.pc_z_window
        for col in ("total_pc", "equity_pc"):
            rolling = merged[col].rolling(window, min_periods=window)
            std = rolling.std()  # ddof=1
            merged[f"{col}_z21"] = (merged[col] - rolling.mean()) / std

        merged["available_at"] = self._available_at(merged["asof_date"])
        columns = [
            "asof_date",
            "vix",
            "vix9d",
            "vix3m",
            "vix6m",
            "vvix",
            "skew",
            "vix9d_vix",
            "vix_vix3m",
            "vix3m_vix6m",
            "vvix_vix",
            "total_pc",
            "equity_pc",
            "total_pc_z21",
            "equity_pc_z21",
            "available_at",
        ]
        return merged[columns]

    # ------------------------------------------------------------------
    # Point-in-time stamp
    # ------------------------------------------------------------------

    def _available_at(self, asof_date: pd.Series) -> pd.Series:
        """18:00 ET (wall clock) on the data's own trading date.

        Built as naive date + hours, THEN localized — so the stamp is the
        exact ET wall time on both EST and EDT dates (US DST transitions
        fall on Sundays, never trading days, and 18:00 is never ambiguous
        or nonexistent).
        """
        naive = pd.to_datetime(asof_date) + pd.Timedelta(hours=self._config.publish_hour_et)
        return naive.dt.tz_localize(MARKET_TZ)


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Cached frame if readable, else None (unreadable cache is a miss)."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
        logger.warning("unreadable cache %s: %s — re-parsing raw archive", path, exc)
        return None


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Parquet via tmp + replace: a crash or concurrent writer can never
    leave a torn file at the final path (same pattern as alpaca_ticks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

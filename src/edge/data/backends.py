"""CatalystBridge: the EdgeDataLoader backend over the catalyst parquet archive.

BACKEND-side module (research code never imports this — it goes through
:class:`edge.data.loader.EdgeDataLoader`; the AST gateway scan enforces the
wall, ``edge/data/`` is exempt).

THE ONLY CATALYST IMPORT POINT IN EDGE
--------------------------------------
This module is the single place in the edge package that imports ``catalyst``
modules. Every such import is lazy — inside :func:`build_catalyst_bridge` and
the :class:`TerminalCatalystFetcher` method bodies — so importing this module
touches nothing of catalyst, and unit tests exercise the bridge entirely
through injected doubles. The gateway scan
(:mod:`edge.validation.gateway`) bans ``catalyst`` imports everywhere
research-side; ``edge/data/`` is exempt precisely so this one bridge can
exist. Do not add catalyst imports anywhere else in edge.

CACHE-ONLY BY DEFAULT (``allow_fetch=False``)
---------------------------------------------
The catalyst providers are cache-first-with-terminal-fallback: a cache miss
inside them silently issues a ThetaData terminal (or Alpaca) request. Research
code sweeping parameters over years of data must NEVER trigger that fallback
by accident — an uncached day in a loop becomes thousands of terminal
requests, degrading the terminal for every concurrent job and masking the
real problem (an unwarmed archive). The bridge therefore reads ONLY the
parquet cache by default; a miss raises :class:`CatalystCacheMiss` (or is
skipped with a warning when ``missing_ok=True``) and the fetcher double is
never consulted. Passing ``allow_fetch=True`` — together with a fetcher — is
a deliberate operator act for warm-up scripts, not something research code
should ever do implicitly.

WHAT IT SERVES (the ``kind`` vocabulary)
----------------------------------------
The loader's typed ``DataKind`` names ``bars`` / ``quotes`` / ``trades``; the
loader forwards ``kind`` as an opaque token after its date gate, so the bridge
extends the vocabulary with the EOD option datasets. Every kind rides through
the same lockbox wall.

===============  ==========================  ===============================
kind             catalyst cache category     symbol grammar
===============  ==========================  ===============================
bars             alpaca_minute               plain ticker: ``"AAPL"``
quotes           theta_minute_quote          ``"AAPL:2026-01-16:190:C"``
greeks_eod       iv_eod                      ``"AAPL:2026-01-16"`` (expiry)
open_interest    open_interest               ``"AAPL:2026-01-16"`` (expiry)
option_eod       option_eod                  ``"AAPL:2026-01-16"`` (expiry)
trades           (not served — raises)       —
===============  ==========================  ===============================

Aliases: ``iv_eod`` -> ``greeks_eod`` (catalyst caches its EOD greeks pulls
under the ``iv_eod`` category — one dataset, both names), ``oi`` ->
``open_interest``.

Cache keys and raw-frame shapes are mirrored from the catalyst source of
truth (``catalyst/data/intraday.py``, ``iv_history.py``,
``thetadata_historical.py``); the key-builder functions below are that
mirror, and tests pin the exact strings. NOTE the ``theta_minute_quote`` key
embeds the requesting session window: cache-only mode reads the configured
canonical window (default 09:30–16:00 ET); archive entries pulled under a
different window are invisible to it — ``allow_fetch=True`` re-pulls the
canonical window and caches it under the canonical key.

POINT-IN-TIME CONVENTION (non-negotiable)
-----------------------------------------
Every output frame carries ``asof_date`` (the data's own date, tz-naive
midnight ``datetime64[ns]``) AND ``available_at`` (tz-aware America/New_York
datetime when a live trader could FIRST have seen the value). Downstream
feature joins key on ``available_at <= decision_ts``. Stamps, conservative by
construction:

* minute bars / minute NBBO: ``available_at = row ts + 1 minute`` (config
  ``minute_lag_minutes``). A bar is complete only at its close, and vendors
  stamp minute rows at interval boundaries — by interval end the row is
  certainly observable, whichever boundary the vendor meant.
* EOD greeks (``greeks_eod``) and EOD option bars (``option_eod``):
  ``available_at = asof_date 18:00 ET`` (config ``eod_publish_hour_et``).
  The terminal finalizes EOD frames shortly after the close (the archive's
  rows are stamped ~15:59:5x); 18:00 is the deliberately LATER stamp.
* open interest: ``available_at = next business day 09:00 ET`` (config
  ``oi_lag_business_days`` / ``oi_publish_hour_et``). OI is computed by OCC
  overnight and first published the following morning; day D's OI is never
  knowable on day D.

The bridge does no embargo logic — the loader is the only gate — but it DOES
honor the requested range: the per-expiry ``iv_eod`` frames span an
expiration's whole DTE window, so rows are filtered to the (already gated)
``[start, end]`` before being returned. That filter is what makes the
loader's lockbox clamp bite for range-spanning cached frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Callable, Final, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

# Catalyst cache categories (mirrored — see module docstring).
CATEGORY_STOCK_MINUTE: Final[str] = "alpaca_minute"
CATEGORY_OPTION_QUOTE: Final[str] = "theta_minute_quote"
CATEGORY_GREEKS_EOD: Final[str] = "iv_eod"
CATEGORY_OPEN_INTEREST: Final[str] = "open_interest"
CATEGORY_OPTION_EOD: Final[str] = "option_eod"

#: Canonical kinds the bridge serves (aliases in :data:`KIND_ALIASES`).
BRIDGE_KINDS: Final[tuple[str, ...]] = (
    "bars",
    "quotes",
    "greeks_eod",
    "open_interest",
    "option_eod",
)

#: alias -> canonical kind. ``iv_eod`` names the same dataset as
#: ``greeks_eod`` because catalyst caches its EOD greeks under that category.
KIND_ALIASES: Final[dict[str, str]] = {"iv_eod": "greeks_eod", "oi": "open_interest"}

_RIGHT_NORMALIZE: Final[dict[str, str]] = {"C": "C", "CALL": "C", "P": "P", "PUT": "P"}

#: Guaranteed columns of an empty ``bars`` result (identity + PIT + OHLCV).
BAR_COLUMNS: Final[tuple[str, ...]] = (
    "asof_date", "available_at", "symbol", "ts",
    "open", "high", "low", "close", "volume",
)

#: Guaranteed columns of an empty ``quotes`` result.
QUOTE_COLUMNS: Final[tuple[str, ...]] = (
    "asof_date", "available_at", "symbol", "underlying", "expiry",
    "strike", "right", "ts", "bid", "ask",
)

#: Identity/PIT columns of the EOD kinds; archive value columns (greeks, OI,
#: volume, …) ride through after them, so an EMPTY result carries only these.
EOD_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "asof_date", "available_at", "symbol", "underlying", "expiry",
)


class CatalystCacheMiss(RuntimeError):
    """Cache-only mode found no cached frame; the terminal was NOT contacted."""


class CatalystBridgeConfig(BaseModel):
    """Bridge knobs: publication stamps and the canonical NBBO session window.

    Defaults are the documented conservative publication schedule (module
    docstring) — overridable per environment, never baked into code paths.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Canonical option-NBBO session window; part of the cache-key identity.
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    #: Minutes after a minute row's stamp before it is deemed visible.
    minute_lag_minutes: int = Field(default=1, ge=0)
    #: Hour (ET) EOD greeks / EOD option bars become visible, same day.
    eod_publish_hour_et: int = Field(default=18, ge=0, le=23)
    #: Open interest lags by this many business days …
    oi_lag_business_days: int = Field(default=1, ge=1)
    #: … and becomes visible at this hour (ET) on that later day.
    oi_publish_hour_et: int = Field(default=9, ge=0, le=23)


# ---------------------------------------------------------------------------
# Symbol grammar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionSpec:
    """Parsed ``"UNDERLYING:YYYY-MM-DD:STRIKE:RIGHT"`` option-contract symbol."""

    underlying: str
    expiry: date
    strike: float
    right: str  # "C" | "P"


def parse_option_symbol(symbol: str) -> OptionSpec:
    """Parse a ``quotes`` symbol, e.g. ``"AAPL:2026-01-16:190:C"``.

    Right accepts C/P/CALL/PUT (any case). Raises ``ValueError`` on any
    malformed part — a guessed contract would silently read the wrong data.
    """
    parts = symbol.split(":")
    if len(parts) != 4 or not parts[0]:
        raise ValueError(
            f"option symbol {symbol!r} must be 'UNDERLYING:YYYY-MM-DD:STRIKE:RIGHT' "
            "(e.g. 'AAPL:2026-01-16:190:C')"
        )
    underlying, expiry_s, strike_s, right_s = parts
    try:
        expiry = date.fromisoformat(expiry_s)
    except ValueError as exc:
        raise ValueError(f"option symbol {symbol!r}: bad expiry {expiry_s!r}") from exc
    try:
        strike = float(strike_s)
    except ValueError as exc:
        raise ValueError(f"option symbol {symbol!r}: bad strike {strike_s!r}") from exc
    if strike <= 0:
        raise ValueError(f"option symbol {symbol!r}: strike must be > 0")
    right = _RIGHT_NORMALIZE.get(right_s.strip().upper())
    if right is None:
        raise ValueError(f"option symbol {symbol!r}: right must be C/P/CALL/PUT")
    return OptionSpec(underlying=underlying, expiry=expiry, strike=strike, right=right)


def parse_expiry_symbol(symbol: str) -> tuple[str, date]:
    """Parse an EOD-kind symbol ``"UNDERLYING:YYYY-MM-DD"`` -> (underlying, expiry)."""
    parts = symbol.split(":")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(
            f"symbol {symbol!r} must be 'UNDERLYING:YYYY-MM-DD' (the expiration) "
            "for the EOD option kinds"
        )
    try:
        expiry = date.fromisoformat(parts[1])
    except ValueError as exc:
        raise ValueError(f"symbol {symbol!r}: bad expiry {parts[1]!r}") from exc
    return parts[0], expiry


# ---------------------------------------------------------------------------
# Cache keys — MIRRORED from catalyst; tests pin the exact strings.
# ---------------------------------------------------------------------------


def minute_bar_key(symbol: str, day: date) -> str:
    """``alpaca_minute`` key (catalyst ``AlpacaMinuteBars.get_day``)."""
    return f"{symbol}_{day:%Y%m%d}"


def option_quote_key(
    underlying: str,
    expiry: date,
    strike: float,
    right: str,
    day: date,
    session_start: time,
    session_end: time,
) -> str:
    """``theta_minute_quote`` key (catalyst ``ThetaMinuteQuotes.contract_day``)."""
    word = "call" if right == "C" else "put"
    return (
        f"{underlying}_{expiry:%Y%m%d}_{strike:.3f}_{word}"
        f"_{day:%Y%m%d}_{session_start:%H%M}_{session_end:%H%M}"
    )


def greeks_eod_key(symbol: str, expiry: date) -> str:
    """``iv_eod`` key (catalyst ``IVRankProvider._expiry_frame``)."""
    return f"{symbol}_{expiry:%Y%m%d}"


def expiry_day_key(symbol: str, expiry: date, day: date) -> str:
    """``open_interest`` / ``option_eod`` key (catalyst ``ThetaDataHistorical``)."""
    return f"{symbol}_{expiry:%Y%m%d}_{day:%Y%m%d}"


# ---------------------------------------------------------------------------
# Protocols the bridge consumes (tests inject fakes; production the real ones)
# ---------------------------------------------------------------------------


@runtime_checkable
class RawFrameCache(Protocol):
    """The slice of catalyst's ``ParquetCache`` the bridge consumes.

    ``get`` returning ``None`` means never cached; an EMPTY frame means
    "cached empty" — a day the market genuinely had no data for. The bridge
    only reads; ``put`` exists for the fetch-side adapter.
    """

    def get(self, category: str, key: str) -> pd.DataFrame | None: ...

    def put(self, category: str, key: str, df: pd.DataFrame) -> None: ...


class CatalystFetcher(Protocol):
    """Delegate for cache misses, consulted ONLY when ``allow_fetch=True``.

    Each method may contact the ThetaData terminal or Alpaca (that is the
    point) and returns the raw catalyst-shaped frame for exactly one cache
    entry. Cache-only mode never calls any of these — tests assert it with a
    recording double.
    """

    def stock_minute_day(self, symbol: str, day: date) -> pd.DataFrame: ...

    def option_quote_day(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        right: str,
        day: date,
        session_start: time,
        session_end: time,
    ) -> pd.DataFrame: ...

    def greeks_eod_frame(self, symbol: str, expiry: date) -> pd.DataFrame: ...

    def open_interest_day(self, symbol: str, expiry: date, day: date) -> pd.DataFrame: ...

    def option_eod_day(self, symbol: str, expiry: date, day: date) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Frame normalization helpers
# ---------------------------------------------------------------------------


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    """Typed empty frame honoring the output contract (audit D-045 style)."""
    return pd.DataFrame(columns=list(columns))


def _business_days(start: date, end: date) -> list[date]:
    """Weekdays in inclusive ``[start, end]``. Market holidays remain in the
    list — a warmed catalyst archive holds cached-EMPTY frames for them, which
    contribute zero rows; a never-pulled holiday surfaces as a miss (strict by
    default) because a hole is indistinguishable from absent data."""
    return [d.date() for d in pd.bdate_range(start, end)]


def _et_minutes(raw: pd.DataFrame, what: str) -> pd.Series:
    """Minute timestamps of a raw catalyst frame as a tz-aware ET series.

    Accepts both on-disk raw shapes (a ``ts``/``timestamp`` column, tz-naive
    ET per catalyst convention) and the provider-returned shape (a naive-ET
    ``DatetimeIndex``); anything else raises.
    """
    if "ts" in raw.columns:
        ts = pd.to_datetime(raw["ts"]).reset_index(drop=True)
    elif "timestamp" in raw.columns:
        ts = pd.to_datetime(raw["timestamp"]).reset_index(drop=True)
    elif isinstance(raw.index, pd.DatetimeIndex):
        ts = pd.Series(raw.index)
    else:
        raise ValueError(
            f"{what}: raw frame carries no 'ts'/'timestamp' column and no "
            "DatetimeIndex — cannot recover minute timestamps"
        )
    if ts.dt.tz is None:
        return ts.dt.tz_localize(MARKET_TZ)
    return ts.dt.tz_convert(MARKET_TZ)


def _require_columns(raw: pd.DataFrame, needed: tuple[str, ...], what: str) -> None:
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{what}: raw frame lacks column(s) {missing}")


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


class CatalystBridge:
    """EdgeDataLoader backend over the catalyst parquet archive; cache-only
    by default.

    ``allow_fetch`` defaults to ``False`` and that default is the contract:
    research code must not silently hammer the ThetaData terminal. The
    catalyst providers fall back to a terminal request on any cache miss, so
    handing research a bridge that delegated misses by default would turn an
    unwarmed archive into thousands of accidental terminal requests per
    sweep. In cache-only mode the ``fetcher`` is NEVER called — a miss raises
    :class:`CatalystCacheMiss` (or, with ``missing_ok=True``, is skipped with
    a warning). ``allow_fetch=True`` requires a fetcher and is a deliberate
    operator choice for warm-up jobs.

    Conforms structurally to :class:`edge.data.loader.DataBackend`: research
    reaches it only through ``EdgeDataLoader.load``, which gates every range
    against the lockbox wall before this class sees it.
    """

    def __init__(
        self,
        cache: RawFrameCache,
        fetcher: CatalystFetcher | None = None,
        *,
        allow_fetch: bool = False,
        missing_ok: bool = False,
        config: CatalystBridgeConfig | None = None,
    ) -> None:
        if allow_fetch and fetcher is None:
            raise ValueError(
                "allow_fetch=True requires a fetcher; cache-only mode "
                "(the default) needs none"
            )
        self._cache = cache
        self._fetcher = fetcher
        self._allow_fetch = bool(allow_fetch)
        self._missing_ok = bool(missing_ok)
        self._config = config or CatalystBridgeConfig()

    # -- DataBackend protocol ------------------------------------------------

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        """One row per available observation for ``symbol`` over ``[start, end]``.

        ``kind`` is one of :data:`BRIDGE_KINDS` (or an alias); the symbol
        grammar per kind is in the module docstring. Called by the loader
        AFTER the lockbox gate — the bridge never widens the range it is
        handed, and range-spanning cached frames are row-filtered to it.
        """
        canonical = KIND_ALIASES.get(kind, kind)
        if canonical == "bars":
            return self._fetch_bars(symbol, start, end)
        if canonical == "quotes":
            return self._fetch_quotes(symbol, start, end)
        if canonical == "greeks_eod":
            return self._fetch_greeks_eod(symbol, start, end)
        if canonical == "open_interest":
            return self._fetch_open_interest(symbol, start, end)
        if canonical == "option_eod":
            return self._fetch_option_eod(symbol, start, end)
        if canonical == "trades":
            raise ValueError(
                "kind 'trades' is not served: the catalyst archive holds no "
                "tick trade data (minute bars and minute NBBO are the finest "
                "granularities cached)"
            )
        raise ValueError(
            f"unknown kind {kind!r}; CatalystBridge serves {BRIDGE_KINDS} "
            f"(aliases: {KIND_ALIASES})"
        )

    # -- cache-first read (the allow_fetch seam) -----------------------------

    def _read(
        self,
        category: str,
        key: str,
        fetch_call: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame | None:
        """Cached raw frame, else fetch (allow_fetch only), skip, or raise.

        Returns ``None`` only in ``missing_ok`` skip mode; an empty frame is
        a real cached answer ("no data that day") and passes through.
        """
        raw = self._cache.get(category, key)
        if raw is not None:
            return raw
        if self._allow_fetch:
            return fetch_call()
        if self._missing_ok:
            logger.warning(
                "catalyst cache miss skipped (cache-only mode): %s/%s", category, key
            )
            return None
        raise CatalystCacheMiss(
            f"no cached frame for {category}/{key} and the bridge is cache-only "
            "(the default: research must not silently hammer the ThetaData "
            "terminal). Warm the archive via the catalyst pipeline, or build "
            "the bridge with allow_fetch=True as a deliberate operator act."
        )

    # -- stock minute bars ---------------------------------------------------

    def _fetch_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if ":" in symbol or not symbol:
            raise ValueError(
                f"kind 'bars' takes a plain equity ticker, got {symbol!r}; "
                "option datasets use the ':'-delimited grammars"
            )
        frames: list[pd.DataFrame] = []
        for day in _business_days(start, end):
            raw = self._read(
                CATEGORY_STOCK_MINUTE,
                minute_bar_key(symbol, day),
                lambda d=day: self._fetcher.stock_minute_day(symbol, d),  # type: ignore[union-attr]
            )
            if raw is None or raw.empty:
                continue
            frames.append(self._minute_bar_frame(symbol, day, raw))
        if not frames:
            return _empty(BAR_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _minute_bar_frame(self, symbol: str, day: date, raw: pd.DataFrame) -> pd.DataFrame:
        what = f"minute bars {symbol} {day}"
        _require_columns(raw, ("open", "high", "low", "close", "volume"), what)
        ts = _et_minutes(raw, what)
        out = pd.DataFrame(
            {
                "asof_date": pd.Timestamp(day),
                "available_at": ts + pd.Timedelta(minutes=self._config.minute_lag_minutes),
                "symbol": symbol,
                "ts": ts,
                "open": raw["open"].to_numpy(dtype=float),
                "high": raw["high"].to_numpy(dtype=float),
                "low": raw["low"].to_numpy(dtype=float),
                "close": raw["close"].to_numpy(dtype=float),
                "volume": raw["volume"].to_numpy(dtype="int64"),
            }
        )
        # Vendor duplicates / overlapping pages must not reach research
        # (mirrors catalyst's own hygiene, audit D-214).
        return (
            out.sort_values("ts", kind="stable")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

    # -- option 1-minute NBBO ------------------------------------------------

    def _fetch_quotes(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        spec = parse_option_symbol(symbol)
        cfg = self._config
        frames: list[pd.DataFrame] = []
        for day in _business_days(start, end):
            raw = self._read(
                CATEGORY_OPTION_QUOTE,
                option_quote_key(
                    spec.underlying, spec.expiry, spec.strike, spec.right,
                    day, cfg.session_start, cfg.session_end,
                ),
                lambda d=day: self._fetcher.option_quote_day(  # type: ignore[union-attr]
                    spec.underlying, spec.expiry, spec.strike, spec.right,
                    d, cfg.session_start, cfg.session_end,
                ),
            )
            if raw is None or raw.empty:
                continue
            frames.append(self._quote_frame(symbol, spec, day, raw))
        if not frames:
            return _empty(QUOTE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _quote_frame(
        self, symbol: str, spec: OptionSpec, day: date, raw: pd.DataFrame
    ) -> pd.DataFrame:
        what = f"option NBBO {symbol} {day}"
        _require_columns(raw, ("bid", "ask"), what)
        ts = _et_minutes(raw, what)
        out = pd.DataFrame(
            {
                "asof_date": pd.Timestamp(day),
                "available_at": ts + pd.Timedelta(minutes=self._config.minute_lag_minutes),
                "symbol": symbol,
                "underlying": spec.underlying,
                "expiry": pd.Timestamp(spec.expiry),
                "strike": spec.strike,
                "right": spec.right,
                "ts": ts,
                "bid": raw["bid"].to_numpy(dtype=float),
                "ask": raw["ask"].to_numpy(dtype=float),
            }
        )
        return (
            out.sort_values("ts", kind="stable")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

    # -- EOD greeks (catalyst category "iv_eod") -----------------------------

    def _fetch_greeks_eod(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        underlying, expiry = parse_expiry_symbol(symbol)
        raw = self._read(
            CATEGORY_GREEKS_EOD,
            greeks_eod_key(underlying, expiry),
            lambda: self._fetcher.greeks_eod_frame(underlying, expiry),  # type: ignore[union-attr]
        )
        if raw is None or raw.empty:
            return _empty(EOD_IDENTITY_COLUMNS)
        what = f"EOD greeks {symbol}"
        _require_columns(raw, ("timestamp",), what)
        raw = raw.reset_index(drop=True)
        ts = pd.to_datetime(raw["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
        asof = ts.dt.normalize()
        # The per-expiry frame spans the expiration's DTE window; honoring the
        # (gated) request range HERE is what makes the lockbox clamp bite.
        mask = (asof >= pd.Timestamp(start)) & (asof <= pd.Timestamp(end))
        raw = raw.loc[mask].reset_index(drop=True)
        asof = asof.loc[mask].reset_index(drop=True)
        if raw.empty:
            return _empty(EOD_IDENTITY_COLUMNS)
        available = (
            asof + pd.Timedelta(hours=self._config.eod_publish_hour_et)
        ).dt.tz_localize(MARKET_TZ)
        out = pd.DataFrame(
            {
                "asof_date": asof,
                "available_at": available,
                "symbol": symbol,
                "underlying": underlying,
                "expiry": pd.Timestamp(expiry),
            }
        )
        for col in raw.columns:  # archive value columns ride through untouched
            out[col] = raw[col]
        return out.sort_values("asof_date", kind="stable").reset_index(drop=True)

    # -- open interest / EOD option bars -------------------------------------

    def _fetch_open_interest(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self._fetch_expiry_day_kind(
            symbol, start, end,
            category=CATEGORY_OPEN_INTEREST,
            fetch_name="open_interest_day",
            available_at=self._oi_available_at,
        )

    def _fetch_option_eod(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self._fetch_expiry_day_kind(
            symbol, start, end,
            category=CATEGORY_OPTION_EOD,
            fetch_name="option_eod_day",
            available_at=self._eod_available_at,
        )

    def _fetch_expiry_day_kind(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        category: str,
        fetch_name: str,
        available_at: Callable[[date], pd.Timestamp],
    ) -> pd.DataFrame:
        underlying, expiry = parse_expiry_symbol(symbol)
        frames: list[pd.DataFrame] = []
        for day in _business_days(start, end):
            raw = self._read(
                category,
                expiry_day_key(underlying, expiry, day),
                lambda d=day: getattr(self._fetcher, fetch_name)(underlying, expiry, d),
            )
            if raw is None or raw.empty:
                continue
            raw = raw.reset_index(drop=True)
            out = pd.DataFrame(
                {
                    "asof_date": pd.Timestamp(day),
                    "available_at": available_at(day),
                    "symbol": symbol,
                    "underlying": underlying,
                    "expiry": pd.Timestamp(expiry),
                },
                index=raw.index,
            )
            for col in raw.columns:
                out[col] = raw[col]
            frames.append(out.reset_index(drop=True))
        if not frames:
            return _empty(EOD_IDENTITY_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _eod_available_at(self, day: date) -> pd.Timestamp:
        """Same-day ``eod_publish_hour_et``:00 ET (conservative EOD stamp)."""
        return (
            pd.Timestamp(day) + pd.Timedelta(hours=self._config.eod_publish_hour_et)
        ).tz_localize(MARKET_TZ)

    def _oi_available_at(self, day: date) -> pd.Timestamp:
        """``oi_lag_business_days`` business days later, ``oi_publish_hour_et``:00 ET.

        ``roll="backward"`` snaps a (theoretical) weekend as-of to the
        preceding business day first, so the offset always lands on the
        strictly-next business day: Fri -> Mon, Wed -> Thu.
        """
        cfg = self._config
        shifted = np.busday_offset(
            np.datetime64(day.isoformat(), "D"), cfg.oi_lag_business_days, roll="backward"
        )
        return (
            pd.Timestamp(shifted) + pd.Timedelta(hours=cfg.oi_publish_hour_et)
        ).tz_localize(MARKET_TZ)


# ---------------------------------------------------------------------------
# Real catalyst wiring (the ONLY catalyst imports in edge — all lazy)
# ---------------------------------------------------------------------------


class TerminalCatalystFetcher:
    """The real :class:`CatalystFetcher` over catalyst's providers.

    Constructed only by :func:`build_catalyst_bridge` with
    ``allow_fetch=True`` (a deliberate operator act) — never by research
    code. Every method may contact the ThetaData terminal or Alpaca; fetched
    raw frames land in the shared parquet cache under the exact catalyst
    (category, key), so subsequent cache-only runs are pure hits. Request
    shapes are mirrored from the catalyst providers so the cached frames are
    indistinguishable from ones catalyst wrote itself (in particular the EOD
    greeks pull reuses ``iv_history``'s DTE window and strike band —
    diverging params under the same key would poison ``IVRankProvider``).
    """

    def __init__(
        self,
        minute_bars: object,
        minute_quotes: object,
        client: object,
        cache: RawFrameCache,
    ) -> None:
        self._minute_bars = minute_bars
        self._minute_quotes = minute_quotes
        self._client = client
        self._cache = cache

    def stock_minute_day(self, symbol: str, day: date) -> pd.DataFrame:
        # AlpacaMinuteBars.get_day caches internally (and correctly refuses to
        # cache an in-progress session, audit D-050).
        return self._minute_bars.get_day(symbol, day)  # type: ignore[attr-defined]

    def option_quote_day(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        right: str,
        day: date,
        session_start: time,
        session_end: time,
    ) -> pd.DataFrame:
        from catalyst.core.types import OptionKey, OptionRight  # sanctioned: see module docstring

        key = OptionKey(
            underlying=underlying,
            expiry=expiry,
            right=OptionRight.CALL if right == "C" else OptionRight.PUT,
            strike=strike,
        )
        # ThetaMinuteQuotes.contract_day caches internally under the identical
        # key format (and never caches transient failures).
        return self._minute_quotes.contract_day(  # type: ignore[attr-defined]
            key, day, session_start, session_end
        )

    def greeks_eod_frame(self, symbol: str, expiry: date) -> pd.DataFrame:
        # Mirror IVRankProvider._expiry_frame EXACTLY (DTE window + strike
        # band from the catalyst module, single source of truth).
        from catalyst.data import iv_history as _iv  # sanctioned: see module docstring

        df = self._client.get_dataframe(  # type: ignore[attr-defined]
            "/v3/option/history/greeks/eod",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "start_date": f"{expiry - timedelta(days=_iv._DTE_MAX):%Y%m%d}",
                "end_date": f"{expiry - timedelta(days=_iv._DTE_MIN):%Y%m%d}",
                "strike_range": _iv._STRIKE_RANGE,
            },
        )
        self._cache.put(CATEGORY_GREEKS_EOD, greeks_eod_key(symbol, expiry), df)
        return df

    def open_interest_day(self, symbol: str, expiry: date, day: date) -> pd.DataFrame:
        df = self._client.get_dataframe(  # type: ignore[attr-defined]
            "/v3/option/history/open_interest",
            {"symbol": symbol, "expiration": f"{expiry:%Y%m%d}", "date": f"{day:%Y%m%d}"},
        )
        self._cache.put(CATEGORY_OPEN_INTEREST, expiry_day_key(symbol, expiry, day), df)
        return df

    def option_eod_day(self, symbol: str, expiry: date, day: date) -> pd.DataFrame:
        df = self._client.get_dataframe(  # type: ignore[attr-defined]
            "/v3/option/history/eod",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "start_date": f"{day:%Y%m%d}",
                "end_date": f"{day:%Y%m%d}",
            },
        )
        self._cache.put(CATEGORY_OPTION_EOD, expiry_day_key(symbol, expiry, day), df)
        return df


def build_catalyst_bridge(
    repo_root: str | None = None,
    environment: str = "backtest",
    *,
    allow_fetch: bool = False,
    missing_ok: bool = False,
    config: CatalystBridgeConfig | None = None,
) -> CatalystBridge:
    """Wire a bridge to the real catalyst parquet cache (production path).

    Cache-only by default: only the parquet cache object is constructed —
    no HTTP client, no terminal session. ``allow_fetch=True`` additionally
    builds the real providers behind a :class:`TerminalCatalystFetcher`
    (still no network at construction; requests happen per cache miss).
    NEVER used in tests — tests inject in-memory doubles.
    """
    from catalyst.core.config import load_config as _load_catalyst_config  # sanctioned
    from catalyst.data.cache import ParquetCache  # sanctioned: see module docstring

    cfg = _load_catalyst_config(environment, repo_root=repo_root)  # type: ignore[arg-type]
    cache = ParquetCache(cfg.data.cache_dir)
    fetcher: CatalystFetcher | None = None
    if allow_fetch:
        from catalyst.data.intraday import AlpacaMinuteBars, ThetaMinuteQuotes  # sanctioned
        from catalyst.data.thetadata_client import ThetaDataClient  # sanctioned

        client = ThetaDataClient(cfg.data.thetadata)
        fetcher = TerminalCatalystFetcher(
            minute_bars=AlpacaMinuteBars(cfg.data.alpaca, cache),
            minute_quotes=ThetaMinuteQuotes(cfg.data.thetadata, cache, client=client),
            client=client,
            cache=cache,
        )
    return CatalystBridge(
        cache, fetcher, allow_fetch=allow_fetch, missing_ok=missing_ok, config=config
    )

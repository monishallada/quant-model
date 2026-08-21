"""Catalyst calendar providers.

Sources (chosen after empirically probing entitlements at build time):
- Historical earnings (backtest): yfinance — ~25 years of report datetimes with
  EPS estimate/actual (the FMP plan on this account is forward-only).
- Upcoming earnings (live): FMP stable earnings-calendar (verified working).
- CPI / FOMC: static CSVs in config/calendars/, built from the BLS release
  schedule (via Wayback snapshots) and the Federal Reserve's published meeting
  calendar. Scheduled meetings only — the strategy trades KNOWN catalysts, so
  unscheduled/emergency actions are deliberately excluded.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from catalyst.core.config import FMPConfig
from catalyst.core.types import Catalyst, CatalystType
from catalyst.data.cache import ParquetCache

logger = logging.getLogger(__name__)

_CPI_RELEASE_TIME = time(8, 30)
_FOMC_DECISION_TIME = time(14, 0)


class StaticEconomicCalendar:
    """CPI + FOMC catalysts from the repo's static schedule CSVs, applied to
    the configured index symbols (macro events are traded via SPY/QQQ/IWM)."""

    def __init__(self, calendars_dir: str | Path, economic_symbols: list[str]) -> None:
        self._dir = Path(calendars_dir)
        self._symbols = list(economic_symbols)

    def _warn_if_uncovered(self, start: date) -> None:
        """A request predating the shipped calendars silently returned zero
        macro events for those years (audit D-170: FOMC starts 2018-01-31)."""
        for name, first in _static_calendar_first_dates(self._dir).items():
            if start < first:
                logger.warning("%s coverage starts %s but the run starts %s — "
                               "earlier macro events are MISSING, not absent",
                               name, first, start)

    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        out: list[Catalyst] = []
        self._warn_if_uncovered(start)
        with open(self._dir / "cpi.csv") as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row["release_date"])
                if start <= d <= end:
                    for sym in self._symbols:
                        out.append(
                            Catalyst(
                                symbol=sym,
                                type=CatalystType.CPI,
                                when=datetime.combine(d, _CPI_RELEASE_TIME),
                                source="bls-schedule",
                            )
                        )
        with open(self._dir / "fomc.csv") as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row["decision_date"])
                if start <= d <= end:
                    for sym in self._symbols:
                        out.append(
                            Catalyst(
                                symbol=sym,
                                type=CatalystType.FOMC,
                                when=datetime.combine(d, _FOMC_DECISION_TIME),
                                source="fed-calendar",
                            )
                        )
        # Same-day CPI + FOMC produce two catalysts per symbol for one
        # reaction session (4 dates in the shipped calendars, audit D-128):
        # keep the EARLIER event (CPI 08:30) — it opens the reaction window.
        dedup: dict[tuple[str, date], Catalyst] = {}
        for c in sorted(out, key=lambda c: c.when):
            dedup.setdefault((c.symbol, c.when.date()), c)
        return sorted(dedup.values(), key=lambda c: c.when)


class YFinanceEarnings:
    """Historical (and near-future) earnings for a symbol universe.

    Report datetimes come tz-aware from Yahoo; they are converted to naive ET.
    Rows are cached to parquet per symbol so the 8-year backtest never
    re-scrapes.
    """

    def __init__(self, symbols: list[str], cache: ParquetCache, refresh: bool = False) -> None:
        self._symbols = list(symbols)
        self._cache = cache
        self._refresh = refresh

    def _symbol_frame(self, symbol: str) -> pd.DataFrame:
        if not self._refresh:
            cached = self._cache.get("earnings_yf", symbol)
            if cached is not None:
                return cached
        import yfinance as yf  # imported lazily: backtest-prep dependency only

        try:
            raw = yf.Ticker(symbol).get_earnings_dates(limit=100)
        except Exception as exc:  # noqa: BLE001 — vendor scrape
            # AUDIT CRITICAL class: a scrape failure cached as "no earnings"
            # would silently delete this symbol's catalysts from every future
            # backtest. Degrade for THIS call only; never cache the failure.
            logger.warning("yfinance earnings fetch failed for %s: %s (NOT cached)",
                           symbol, exc)
            return pd.DataFrame(columns=["when", "eps_estimate", "eps_actual"])
        if raw is None or raw.empty:
            # Empty is yfinance's most common FAILURE shape (rate limits,
            # layout drift). Caching it as "no earnings" deletes the symbol's
            # catalysts from every future backtest (audit D-047) — the exact
            # class the comment above claims fixed. Serve empty, cache nothing.
            logger.warning("yfinance returned EMPTY earnings for %s (NOT cached)",
                           symbol)
            return pd.DataFrame(columns=["when", "eps_estimate", "eps_actual"])
        else:
            idx = raw.index.tz_convert("America/New_York").tz_localize(None)
            df = pd.DataFrame(
                {
                    "when": idx,
                    "eps_estimate": raw.get("EPS Estimate", pd.Series(index=raw.index)).values,
                    "eps_actual": raw.get("Reported EPS", pd.Series(index=raw.index)).values,
                }
            )
            df = df.drop_duplicates(subset=["when"]).sort_values("when").reset_index(drop=True)
        self._cache.put("earnings_yf", symbol, df)
        return df

    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        out: list[Catalyst] = []
        for symbol in self._symbols:
            df = self._symbol_frame(symbol)
            for _, row in df.iterrows():
                when: datetime = pd.Timestamp(row["when"]).to_pydatetime()
                if start <= when.date() <= end:
                    est = row["eps_estimate"]
                    act = row["eps_actual"]
                    out.append(
                        Catalyst(
                            symbol=symbol,
                            type=CatalystType.EARNINGS,
                            when=when,
                            source="yfinance",
                            eps_estimate=None if pd.isna(est) else float(est),
                            eps_actual=None if pd.isna(act) else float(act),
                        )
                    )
        return sorted(out, key=lambda c: c.when)


class FMPEarningsCalendar:
    """Forward-looking earnings from FMP (live screening path).

    The current FMP plan only allows near-term 'from' dates — fine live, where
    the screener looks days ahead, never back.
    """

    def __init__(self, cfg: FMPConfig, symbols: list[str]) -> None:
        self._cfg = cfg
        self._symbols = set(symbols)

    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        import httpx

        params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": self._cfg.api_key}
        resp = httpx.get(
            f"{self._cfg.base_url}/stable/earnings-calendar",
            params=params,
            timeout=self._cfg.request_timeout_seconds,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            logger.warning("FMP earnings-calendar unexpected payload: %s", str(rows)[:200])
            return []
        out: list[Catalyst] = []
        for row in rows:
            symbol = row.get("symbol", "")
            if symbol not in self._symbols:
                continue
            d = date.fromisoformat(row["date"])
            # FMP's stable calendar lacks a BMO/AMC flag. Stamping 16:00 makes
            # resolve_reaction_session return D+1 — for a BEFORE-OPEN reporter
            # that is one day LATE: the "pre-event" entry happens after the
            # move (audit D-008, lookahead-adjacent). This provider is
            # therefore DEPRECATED for entry timing; YFinanceEarnings carries
            # real timestamps and is the wired default. Kept for EPS surprise
            # data only.
            out.append(
                Catalyst(
                    symbol=symbol,
                    type=CatalystType.EARNINGS,
                    when=datetime.combine(d, time(16, 0)),
                    source="fmp",
                    eps_estimate=row.get("epsEstimated"),
                    eps_actual=row.get("epsActual"),
                )
            )
        return sorted(out, key=lambda c: c.when)


def _static_calendar_first_dates(cal_dir: Path) -> dict[str, date]:
    firsts: dict[str, date] = {}
    for name, col in (("cpi.csv", "release_date"), ("fomc.csv", "decision_date")):
        try:
            with open(cal_dir / name) as f:
                rows = list(csv.DictReader(f))
            if rows:
                firsts[name] = min(date.fromisoformat(r[col]) for r in rows)
        except OSError:
            pass
    return firsts


class CompositeCatalystProvider:
    def __init__(self, providers: list[object]) -> None:
        self._providers = providers

    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        out: list[Catalyst] = []
        for p in self._providers:
            out.extend(p.get_catalyst_calendar(start, end))  # type: ignore[attr-defined]
        return sorted(out, key=lambda c: c.when)


def resolve_reaction_session(catalyst: Catalyst) -> date:
    """First trading session whose prices can react to the catalyst.

    Before-open events (CPI 08:30, BMO earnings) react the same day; at/after
    close events (FOMC 14:00 reacts same day; AMC earnings react next day).
    """
    from catalyst.core.tradingcal import is_trading_day, next_trading_day

    d = catalyst.when.date()
    if catalyst.when.time() >= time(16, 0):
        return next_trading_day(d)
    if not is_trading_day(d):
        return next_trading_day(d)
    return d

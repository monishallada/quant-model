"""Daily ATM implied-volatility history + IV rank.

Engine A's entry gate ("buy cheap optionality": IV rank < threshold) needs a
daily ~30-DTE ATM IV series per underlying. Build it from EOD greeks pulled
once per monthly expiration (ATM ± a few strikes via ``strike_range``), then:

    ATM IV(day) = mean(call IV, put IV) at the strike nearest spot, on the
    monthly expiration whose DTE is closest to 30 (within [20, 45]).

    IV rank(day) = percentile of ATM IV(day) among the trailing lookback
    window (default 252 sessions), 0..100.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

from catalyst.core.config import DataConfig
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_client import ThetaDataClient, ThetaDataError

logger = logging.getLogger(__name__)

_DTE_MIN, _DTE_TARGET, _DTE_MAX = 20, 30, 45
_STRIKE_RANGE = "4"


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


class IVRankProvider:
    def __init__(
        self,
        cfg: DataConfig,
        client: ThetaDataClient,
        cache: ParquetCache,
        lookback_days: int,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._cache = cache
        self._lookback = lookback_days
        self._series: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------

    def _monthly_expirations(self, symbol: str, start: date, end: date) -> list[date]:
        # year-scoped like ThetaDataHistorical (audit D-136): listings grow
        key = f"{symbol}_{date.today():%Y}"
        df = self._cache.get("expirations", key)
        if df is None:
            df = self._client.get_dataframe(
                "/v3/option/list/expirations", {"symbol": symbol}
            )
            self._cache.put("expirations", key, df)
        if df.empty:
            return []
        all_exp = sorted(pd.to_datetime(df["expiration"]).dt.date.unique())
        out: list[date] = []
        cursor = date(start.year, start.month, 1)
        while cursor <= end:
            target = _third_friday(cursor.year, cursor.month)
            near = [e for e in all_exp if abs((e - target).days) <= 3]
            if near:
                out.append(min(near, key=lambda e: abs((e - target).days)))
            cursor = (cursor.replace(day=28) + timedelta(days=5)).replace(day=1)
        return out

    def _expiry_frame(self, symbol: str, expiry: date) -> pd.DataFrame:
        """EOD greeks (ATM band) for one expiration's life. Cached once."""
        key = f"{symbol}_{expiry:%Y%m%d}"
        cached = self._cache.get("iv_eod", key)
        if cached is not None:
            return cached
        # Request exactly the DTE window the series consumes — these bulk
        # pulls are the heaviest queries the terminal serves; payload size
        # directly drives timeout risk on long warmup jobs.
        df = self._client.get_dataframe(
            "/v3/option/history/greeks/eod",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "start_date": f"{expiry - timedelta(days=_DTE_MAX):%Y%m%d}",
                "end_date": f"{expiry - timedelta(days=_DTE_MIN):%Y%m%d}",
                "strike_range": _STRIKE_RANGE,
            },
        )
        self._cache.put("iv_eod", key, df)
        return df

    def atm_iv_series(self, symbol: str, start: date, end: date) -> pd.Series:
        """Daily ~30-DTE ATM IV series for [start, end] (assembled once, cached).

        NOT constant-maturity (audit D-133): each day takes the monthly expiry
        with DTE in [20, 45] closest to 30, so tenor saw-tooths across the
        cycle. Ranks computed on it compare like-for-like (the same sawtooth),
        but the LEVEL feeding IV/RV carries tenor noise of a few vol points.
        """
        cache_key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}"
        cached = self._cache.get("iv_series", cache_key)
        if cached is not None:
            s = cached.set_index(pd.to_datetime(cached["date"]).dt.date)["atm_iv"]
            return s

        monthlies = self._monthly_expirations(symbol, start, end + timedelta(days=45))
        # Warm uncached expiry frames with LOW concurrency: these are the
        # heaviest queries the terminal serves, and sustained full-width
        # parallelism degrades its latency until requests time out.
        missing = [e for e in monthlies if not self._cache.exists("iv_eod", f"{symbol}_{e:%Y%m%d}")]
        if len(missing) > 1:
            workers = max(1, self._cfg.thetadata.max_concurrent_requests // 2)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._expiry_frame, symbol, e) for e in missing]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001 — serial path handles it
                        logger.debug("IV prefetch failed: %s", exc)

        rows: dict[date, tuple[int, float]] = {}  # day -> (|dte-30|, iv)
        complete = True
        for expiry in monthlies:
            try:
                df = self._expiry_frame(symbol, expiry)
            except ThetaDataError as exc:
                # One unfetchable month costs the IV series ~1 sample window;
                # aborting a multi-hour warmup job costs everything. But the
                # RESULT is partial and must never be cached as the truth
                # (audit D-010: gate inputs froze on holes forever).
                logger.warning("Skipping IV month %s %s: %s", symbol, expiry, exc)
                complete = False
                continue
            if df.empty:
                continue
            df = df.assign(day=pd.to_datetime(df["timestamp"]).dt.date)
            for day, day_rows in df.groupby("day"):
                if not (start <= day <= end):
                    continue
                dte = (expiry - day).days
                if not (_DTE_MIN <= dte <= _DTE_MAX):
                    continue
                spot = float(day_rows["underlying_price"].iloc[-1])
                if spot <= 0:
                    continue
                day_rows = day_rows[day_rows["implied_vol"] > 0]
                if day_rows.empty:
                    continue
                strikes = day_rows["strike"].astype(float)
                atm_strike = strikes.iloc[(strikes - spot).abs().argsort().iloc[0]]
                # The fixed strike band can strand the nearest captured strike
                # far from a moved spot; its IV is a different moneyness, not
                # "ATM" (audit D-053). A hole beats a fabricated sample.
                if abs(atm_strike - spot) / spot > 0.10:
                    logger.warning("IV %s %s: nearest strike %.1f is %.0f%% from "
                                   "spot %.2f — day dropped", symbol, day,
                                   atm_strike, 100 * abs(atm_strike - spot) / spot,
                                   spot)
                    continue
                atm = day_rows[day_rows["strike"].astype(float) == atm_strike]
                iv = float(atm["implied_vol"].mean())
                fit = abs(dte - _DTE_TARGET)
                if day not in rows or fit < rows[day][0]:
                    rows[day] = (fit, iv)

        series = pd.Series(
            {d: iv for d, (_, iv) in sorted(rows.items())}, dtype=float, name="atm_iv"
        )
        out_df = pd.DataFrame(
            {"date": [d.isoformat() for d in series.index], "atm_iv": series.values}
        )
        if complete:
            self._cache.put("iv_series", cache_key, out_df)
        else:
            logger.warning("IV series for %s assembled from PARTIAL months — "
                           "served fresh, NOT cached (audit D-010)", symbol)
        return series

    # ------------------------------------------------------------------

    def prepare(self, symbol: str, start: date, end: date) -> None:
        """Preload the series (with a full lookback warmup before ``start``)."""
        warmup_start = start - timedelta(days=int(self._lookback * 1.6))
        self._series[symbol] = self.atm_iv_series(symbol, warmup_start, end)

    def _visible(self, symbol: str, day: date, include_day: bool) -> pd.Series:
        """History visible to a decision made ON ``day``.

        AUDIT (v14 verification): the series is EOD greeks, timestamped
        ~15:59:5x. Every decision in this system is taken at the 15:45
        snapshot, so the entry day's own value is ~15 minutes in the FUTURE.
        Default is therefore strictly-before; a caller that genuinely acts
        after the close must ask for ``include_day=True`` explicitly.
        """
        series = self._series.get(symbol)
        if series is None:
            raise RuntimeError(f"IVRankProvider.prepare() not called for {symbol}")
        return series[series.index <= day] if include_day else series[series.index < day]

    def iv_rank(self, symbol: str, day: date, *,
                include_day: bool = False) -> float | None:
        """IV rank 0..100 as of ``day``, or None without enough history."""
        upto = self._visible(symbol, day, include_day)
        if len(upto) < self._lookback // 2:
            return None
        window = upto.tail(self._lookback)
        current = window.iloc[-1]
        return float((window < current).mean() * 100.0)

    def atm_iv30(self, symbol: str, day: date, *,
                 include_day: bool = False) -> float | None:
        """The ~30-DTE ATM IV level as of ``day`` (last value before it).

        The IV/RV gate needs a tenor-matched numerator: event-expiry IV is
        mechanically inflated by the event and made that gate a no-op."""
        upto = self._visible(symbol, day, include_day)
        if not len(upto):
            return None
        v = float(upto.iloc[-1])
        return v if v > 0 else None

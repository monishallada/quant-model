"""Minute-level data providers for intraday strategies.

Two sources, both verified empirically against 2018→2026 before use:

- ``AlpacaMinuteBars`` — 1-minute equity OHLCV on the **SIP** (consolidated
  tape) feed, which is the only feed here that carries pre-market sessions.
  The IEX feed returns ZERO bars before 09:30 ET, so any pre-market volume
  filter built on it would silently match nothing; SIP is mandatory for this
  class of strategy.
- ``ThetaMinuteQuotes`` — 1-minute option NBBO (bid/ask) from the local Theta
  Terminal, so entries can fill at the real ask and exits at the real bid
  rather than at a synthetic haircut.

Both cache to parquet keyed per (symbol, day), so a re-run costs no requests.
All timestamps are tz-naive US/Eastern (converted at the boundary), matching
the spec's requirement that the engine reason in exchange-local time.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

import httpx
import pandas as pd

from catalyst.core.config import AlpacaConfig, ThetaDataConfig
from catalyst.core.models import OptionKey, OptionRight
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_client import ThetaDataClient, ThetaDataError

logger = logging.getLogger(__name__)

_ET = "America/New_York"
_DATA_URL = "https://data.alpaca.markets"
_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


class AlpacaMinuteBars:
    """1-minute consolidated-tape bars, 04:00–20:00 ET, cached per symbol-day."""

    def __init__(self, cfg: AlpacaConfig, cache: ParquetCache, feed: str = "sip") -> None:
        self._cache = cache
        self._feed = feed
        self._http = httpx.Client(
            base_url=_DATA_URL,
            headers={
                "APCA-API-KEY-ID": cfg.api_key,
                "APCA-API-SECRET-KEY": cfg.secret_key,
            },
            timeout=60.0,
        )

    def get_day(self, symbol: str, day: date) -> pd.DataFrame:
        """Minute bars for ``day``, ET-indexed. Empty frame when unavailable."""
        key = f"{symbol}_{day:%Y%m%d}"
        cached = self._cache.get("alpaca_minute", key)
        if cached is not None:
            if cached.empty:
                return pd.DataFrame(columns=_BAR_COLUMNS)
            return cached.set_index(pd.to_datetime(cached["ts"])).drop(columns=["ts"])

        # 04:00 ET → 20:00 ET expressed in UTC (handles DST via tz conversion).
        start = pd.Timestamp(datetime.combine(day, time(4, 0)), tz=_ET).tz_convert("UTC")
        end = pd.Timestamp(datetime.combine(day, time(20, 0)), tz=_ET).tz_convert("UTC")
        rows: list[dict[str, object]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "timeframe": "1Min",
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": self._feed,
                "adjustment": "split",
                "limit": "10000",
            }
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self._http.get(f"/v2/stocks/{symbol}/bars", params=params)
            except httpx.HTTPError as exc:
                logger.warning("Alpaca minute bars failed %s %s: %s", symbol, day, exc)
                break
            if resp.status_code != 200:
                logger.warning("Alpaca minute bars %s %s -> HTTP %d: %s",
                               symbol, day, resp.status_code, resp.text[:160])
                break
            payload = resp.json()
            for bar in payload.get("bars") or []:
                rows.append({
                    "ts": pd.Timestamp(bar["t"]).tz_convert(_ET).tz_localize(None),
                    "open": float(bar["o"]), "high": float(bar["h"]),
                    "low": float(bar["l"]), "close": float(bar["c"]),
                    "volume": int(bar["v"]),
                })
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        df = pd.DataFrame(rows, columns=["ts", *_BAR_COLUMNS])
        self._cache.put("alpaca_minute", key, df)
        if df.empty:
            return pd.DataFrame(columns=_BAR_COLUMNS)
        return df.set_index(pd.to_datetime(df["ts"])).drop(columns=["ts"])

    def daily_closes(self, symbol: str, start: date, end: date) -> pd.Series:
        """Full daily close series for a symbol, fetched once and cached.

        Same feed and split adjustment as the minute bars, so the overnight gap
        is never corrupted by an adjustment mismatch across sources. One request
        per symbol instead of one per symbol-day.
        """
        key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_sipdaily"
        cached = self._cache.get("alpaca_daily_sip", key)
        if cached is None:
            rows: list[dict[str, object]] = []
            page_token: str | None = None
            while True:
                params: dict[str, str] = {
                    "timeframe": "1Day", "start": start.isoformat(), "end": end.isoformat(),
                    "feed": self._feed, "adjustment": "split", "limit": "10000",
                }
                if page_token:
                    params["page_token"] = page_token
                try:
                    resp = self._http.get(f"/v2/stocks/{symbol}/bars", params=params)
                except httpx.HTTPError as exc:
                    logger.warning("Alpaca daily bars failed %s: %s", symbol, exc)
                    break
                if resp.status_code != 200:
                    logger.warning("Alpaca daily %s -> HTTP %d", symbol, resp.status_code)
                    break
                payload = resp.json()
                for bar in payload.get("bars") or []:
                    rows.append({"day": bar["t"][:10], "close": float(bar["c"])})
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
            cached = pd.DataFrame(rows, columns=["day", "close"])
            self._cache.put("alpaca_daily_sip", key, cached)
        if cached.empty:
            return pd.Series(dtype=float)
        return pd.Series(cached["close"].to_numpy(),
                         index=pd.to_datetime(cached["day"]).dt.date, dtype=float)

    def prior_close(self, symbol: str, day: date, closes: pd.Series | None = None) -> float | None:
        """Prior regular-session close. Pass ``closes`` from ``daily_closes``."""
        if closes is None or closes.empty:
            return None
        prior = closes[closes.index < day]
        return float(prior.iloc[-1]) if len(prior) else None


class ThetaMinuteQuotes:
    """1-minute option NBBO for a single contract-day, cached."""

    def __init__(self, cfg: ThetaDataConfig, cache: ParquetCache,
                 client: ThetaDataClient | None = None) -> None:
        self._cache = cache
        self._client = client or ThetaDataClient(cfg)

    def contract_day(
        self, key: OptionKey, day: date, start: time, end: time
    ) -> pd.DataFrame:
        """Minute bid/ask for one contract on one day, ET-indexed.

        Columns: bid, ask. Empty frame when the terminal has no quotes.
        """
        right = "call" if key.right is OptionRight.CALL else "put"
        cache_key = (f"{key.underlying}_{key.expiry:%Y%m%d}_{key.strike:.3f}_{right}"
                     f"_{day:%Y%m%d}_{start:%H%M}_{end:%H%M}")
        cached = self._cache.get("theta_minute_quote", cache_key)
        if cached is None:
            try:
                cached = self._client.get_dataframe(
                    "/v3/option/history/quote",
                    {
                        "symbol": key.underlying,
                        "expiration": f"{key.expiry:%Y%m%d}",
                        "strike": f"{key.strike:.3f}",
                        "right": right,
                        "date": f"{day:%Y%m%d}",
                        "start_time": start.strftime("%H:%M:%S") + ".000",
                        "end_time": end.strftime("%H:%M:%S") + ".000",
                        "interval": "1m",
                    },
                )
            except ThetaDataError as exc:
                logger.warning("Minute quotes failed for %s on %s: %s", key, day, exc)
                cached = pd.DataFrame()
            self._cache.put("theta_minute_quote", cache_key, cached)
        if cached.empty:
            return pd.DataFrame(columns=["bid", "ask"])
        # .to_numpy() is load-bearing: passing Series (RangeIndex) alongside a
        # DatetimeIndex makes pandas ALIGN on labels, which matches nothing and
        # silently yields an all-NaN frame — i.e. every quote would look
        # missing and every fill would fall back to the synthetic haircut.
        out = pd.DataFrame({
            "bid": cached["bid"].to_numpy(dtype=float),
            "ask": cached["ask"].to_numpy(dtype=float),
        }, index=pd.to_datetime(cached["timestamp"]))
        return out[~out.index.duplicated(keep="first")].sort_index()


def five_minute_candles(minute_bars: pd.DataFrame, session_open: time,
                        candle_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute bars into N-minute candles anchored at the open.

    Anchoring matters: candle 0 must be exactly [09:30:00, 09:35:00) so that
    ``P_open`` and ``P_0935`` come from the window the spec names, and so no
    candle straddles the signal timestamp.
    """
    if minute_bars.empty:
        return pd.DataFrame(columns=[*_BAR_COLUMNS])
    rth = minute_bars[minute_bars.index.time >= session_open]
    if rth.empty:
        return pd.DataFrame(columns=[*_BAR_COLUMNS])
    origin = pd.Timestamp(datetime.combine(rth.index[0].date(), session_open))
    grouped = rth.resample(f"{candle_minutes}min", origin=origin)
    candles = grouped.agg(open=("open", "first"), high=("high", "max"),
                          low=("low", "min"), close=("close", "last"),
                          volume=("volume", "sum")).dropna(subset=["close"])
    return candles

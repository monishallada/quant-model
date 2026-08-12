"""Alpaca Market Data: daily underlying bars (backtest signals + live).

ThetaData's stock tier on this subscription is FREE (options are STANDARD),
which cannot serve multi-year stock history — Alpaca's data API can, with the
same keys the paper broker uses. Split-adjusted so indicator signals are not
poisoned by splits (TSLA/NVDA/AMZN in-window).
"""

from __future__ import annotations

import logging
from datetime import date

import httpx
import pandas as pd

from catalyst.core.config import AlpacaConfig
from catalyst.data.cache import ParquetCache

logger = logging.getLogger(__name__)

_DATA_URL = "https://data.alpaca.markets"


class AlpacaDailyBars:
    def __init__(self, cfg: AlpacaConfig, cache: ParquetCache) -> None:
        self._cache = cache
        self._http = httpx.Client(
            base_url=_DATA_URL,
            headers={
                "APCA-API-KEY-ID": cfg.api_key,
                "APCA-API-SECRET-KEY": cfg.secret_key,
            },
            timeout=30.0,
        )

    def get_history(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        if interval != "1d":
            raise NotImplementedError("Only daily bars are supported")
        key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_1d"
        cached = self._cache.get("alpaca_bars", key)
        if cached is not None:
            return cached.set_index(pd.to_datetime(cached["date"])).drop(columns=["date"])

        rows: list[dict[str, object]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": "split",
                "feed": "iex",
                "limit": "10000",
            }
            if page_token:
                params["page_token"] = page_token
            resp = self._http.get(f"/v2/stocks/{symbol}/bars", params=params)
            resp.raise_for_status()
            payload = resp.json()
            for bar in payload.get("bars") or []:
                rows.append(
                    {
                        "date": bar["t"][:10],
                        "open": float(bar["o"]),
                        "high": float(bar["h"]),
                        "low": float(bar["l"]),
                        "close": float(bar["c"]),
                        "volume": int(bar["v"]),
                    }
                )
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self._cache.put("alpaca_bars", key, df)
        out = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
        out.index.name = "date"
        return out

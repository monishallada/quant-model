"""ThetaDataHistorical: the backtest DataSource.

Chain snapshots are assembled from three bulk terminal requests per
(symbol, expiration, date) — first-order greeks (quotes + greeks + IV +
underlying price in one response), EOD bars (volume), and open interest —
each cached to parquet so 8-year sweeps only hit the terminal once.

Greeks fallback: when the terminal serves no usable IV for a contract
(missing/NaN/zero), IV is solved from the mid via Black-Scholes and greeks
recomputed — the served values win whenever present (verified at M0 that
first-order greeks are served on the Standard tier).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta

import pandas as pd

from catalyst.core.config import DataConfig
from catalyst.core.interfaces import DataSource
from catalyst.core.models import (
    Catalyst,
    Greeks,
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    Quote,
)
from catalyst.data.black_scholes import bs_greeks, implied_vol
from catalyst.data.cache import ParquetCache
from catalyst.data.thetadata_client import ThetaDataClient

logger = logging.getLogger(__name__)

_RIGHT_MAP = {"CALL": OptionRight.CALL, "PUT": OptionRight.PUT}


class DataUnavailableError(Exception):
    """No usable data for the requested symbol/timestamp — caller should skip."""


class ThetaDataHistorical(DataSource):
    def __init__(
        self,
        cfg: DataConfig,
        client: ThetaDataClient | None = None,
        cache: ParquetCache | None = None,
        catalyst_provider: object | None = None,
        history_provider: object | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client or ThetaDataClient(cfg.thetadata)
        self._cache = cache or ParquetCache(cfg.cache_dir)
        # Catalyst calendars come from FMP (M2); injected to keep this class vendor-pure.
        self._catalyst_provider = catalyst_provider
        # Underlying history: ThetaData's stock tier is FREE on this account and
        # cannot serve multi-year history — Alpaca daily bars are injected instead.
        self._history_provider = history_provider
        # Assembled chains are immutable; LRU-memoized so parameter sweeps that
        # replay the same days skip parquet parsing entirely.
        self._chain_memo: OrderedDict[tuple, OptionChain] = OrderedDict()

    # ------------------------------------------------------------------
    # Raw cached pulls
    # ------------------------------------------------------------------

    def _cached(self, category: str, key: str, path: str, params: dict[str, str]) -> pd.DataFrame:
        df = self._cache.get(category, key)
        if df is None:
            df = self._client.get_dataframe(path, params)
            self._cache.put(category, key, df)
        return df

    def _prefetch(self, jobs: list[tuple[str, str, str, dict[str, str]]]) -> None:
        """Warm the cache for many pulls concurrently (terminal allows a small
        number of parallel requests; sequential pulls dominate backtest wall
        time otherwise). Failures surface on the later serial read."""
        missing = [j for j in jobs if not self._cache.exists(j[0], j[1])]
        if len(missing) <= 1:
            return
        workers = max(1, self._cfg.thetadata.max_concurrent_requests)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._cached, *job) for job in missing]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:  # noqa: BLE001 — logged; serial path re-raises
                    logger.debug("Prefetch job failed (will retry serially): %s", exc)

    def _expirations(self, symbol: str) -> list[date]:
        df = self._cached(
            "expirations", symbol, "/v3/option/list/expirations", {"symbol": symbol}
        )
        if df.empty:
            return []
        return sorted(pd.to_datetime(df["expiration"]).dt.date.unique())

    def _greeks_job(
        self, symbol: str, expiry: date, day: date, snap: time
    ) -> tuple[str, str, str, dict[str, str]]:
        tstr = snap.strftime("%H:%M:%S") + ".000"
        return (
            "greeks_fo",
            f"{symbol}_{expiry:%Y%m%d}_{day:%Y%m%d}_{snap:%H%M}",
            "/v3/option/history/greeks/first_order",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "date": f"{day:%Y%m%d}",
                "start_time": tstr,
                "end_time": tstr,
                "interval": "1m",
            },
        )

    def _eod_job(self, symbol: str, expiry: date, day: date) -> tuple[str, str, str, dict[str, str]]:
        return (
            "option_eod",
            f"{symbol}_{expiry:%Y%m%d}_{day:%Y%m%d}",
            "/v3/option/history/eod",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "start_date": f"{day:%Y%m%d}",
                "end_date": f"{day:%Y%m%d}",
            },
        )

    def _oi_job(self, symbol: str, expiry: date, day: date) -> tuple[str, str, str, dict[str, str]]:
        return (
            "open_interest",
            f"{symbol}_{expiry:%Y%m%d}_{day:%Y%m%d}",
            "/v3/option/history/open_interest",
            {
                "symbol": symbol,
                "expiration": f"{expiry:%Y%m%d}",
                "date": f"{day:%Y%m%d}",
            },
        )

    def _greeks_frame(self, symbol: str, expiry: date, day: date, snap: time) -> pd.DataFrame:
        return self._cached(*self._greeks_job(symbol, expiry, day, snap))

    def _eod_frame(self, symbol: str, expiry: date, day: date) -> pd.DataFrame:
        return self._cached(*self._eod_job(symbol, expiry, day))

    def _oi_frame(self, symbol: str, expiry: date, day: date) -> pd.DataFrame:
        return self._cached(*self._oi_job(symbol, expiry, day))

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def list_expirations(self, symbol: str) -> list[date]:
        return self._expirations(symbol)

    def get_chain(
        self,
        symbol: str,
        at: datetime,
        expiries: list[date] | None = None,
        max_dte: int | None = None,
    ) -> OptionChain:
        day = at.date()
        snap = at.time()
        if expiries is not None:
            expiries = sorted(e for e in expiries if e > day)
        else:
            horizon = day + timedelta(days=max_dte if max_dte is not None else self._cfg.chain_max_dte)
            expiries = [e for e in self._expirations(symbol) if day < e <= horizon]
        if not expiries:
            raise DataUnavailableError(f"No expirations for {symbol} within horizon of {day}")

        memo_key = (symbol, day, snap, tuple(expiries))
        memoized = self._chain_memo.get(memo_key)
        if memoized is not None:
            self._chain_memo.move_to_end(memo_key)
            return memoized

        # Warm all needed pulls concurrently before serial assembly.
        jobs: list[tuple[str, str, str, dict[str, str]]] = []
        for expiry in expiries:
            jobs.append(self._greeks_job(symbol, expiry, day, snap))
            jobs.append(self._eod_job(symbol, expiry, day))
            jobs.append(self._oi_job(symbol, expiry, day))
        self._prefetch(jobs)

        contracts: list[OptionContract] = []
        underlying_prices: list[float] = []

        for expiry in expiries:
            g = self._greeks_frame(symbol, expiry, day, snap)
            if g.empty:
                continue
            # One row per contract at the snapshot (dedupe defensively).
            g = g.drop_duplicates(subset=["strike", "right"], keep="first")

            # Column-array assembly: this path runs thousands of times per
            # backtest and row-wise iteration dominated cache-hot run time.
            eod = self._eod_frame(symbol, expiry, day)
            vol_map: dict[tuple[float, str], int] = {}
            if not eod.empty:
                vol_map = {
                    (float(k), str(r)): int(v)
                    for k, r, v in zip(eod["strike"], eod["right"], eod["volume"])
                }

            oi = self._oi_frame(symbol, expiry, day)
            oi_map: dict[tuple[float, str], int] = {}
            if not oi.empty:
                oi_map = {
                    (float(k), str(r)): int(v)
                    for k, r, v in zip(oi["strike"], oi["right"], oi["open_interest"])
                }

            timestamps = pd.to_datetime(g["timestamp"]).dt.to_pydatetime()
            rows = zip(
                g["strike"].to_numpy(dtype=float),
                g["right"].astype(str).to_numpy(),
                g["bid"].to_numpy(dtype=float),
                g["ask"].to_numpy(dtype=float),
                g["underlying_price"].to_numpy(dtype=float),
                g["delta"].to_numpy(dtype=float),
                g["theta"].to_numpy(dtype=float),
                g["vega"].to_numpy(dtype=float),
                g["rho"].to_numpy(dtype=float),
                g["implied_vol"].to_numpy(dtype=float),
                timestamps,
            )
            for strike, right_s, bid, ask, upx, delta, theta, vega, rho, iv, ts in rows:
                bid = bid if bid == bid else 0.0  # NaN-safe without pd.isna calls
                ask = ask if ask == ask else 0.0
                if upx > 0:
                    underlying_prices.append(upx)
                greeks = self._scalar_greeks(
                    delta, theta, vega, rho, iv, strike, right_s, upx, day, expiry, bid, ask
                )
                contracts.append(
                    OptionContract(
                        key=OptionKey(
                            underlying=symbol,
                            expiry=expiry,
                            right=_RIGHT_MAP[right_s],
                            strike=strike,
                        ),
                        bid=bid,
                        ask=ask,
                        volume=vol_map.get((strike, right_s), 0),
                        open_interest=oi_map.get((strike, right_s), 0),
                        greeks=greeks,
                        quote_timestamp=ts,
                    )
                )

        if not contracts or not underlying_prices:
            raise DataUnavailableError(f"No chain data for {symbol} at {at}")

        underlying_price = float(pd.Series(underlying_prices).median())
        chain = OptionChain(
            underlying=symbol,
            underlying_price=underlying_price,
            timestamp=datetime.combine(day, snap),
            contracts=contracts,
        )
        self._chain_memo[memo_key] = chain
        if len(self._chain_memo) > self._cfg.chain_memory_cache_size:
            self._chain_memo.popitem(last=False)
        return chain

    def _scalar_greeks(
        self,
        delta: float,
        theta: float,
        vega: float,
        rho: float,
        iv: float,
        strike: float,
        right_s: str,
        underlying_price: float,
        day: date,
        expiry: date,
        bid: float,
        ask: float,
    ) -> Greeks | None:
        if iv == iv and iv > 0:  # NaN-safe "served IV present"
            return Greeks(
                delta=delta,
                theta=theta,
                vega=vega,
                rho=rho if rho == rho else None,
                iv=iv,
            )
        # Fallback: solve IV from mid via Black-Scholes, recompute greeks.
        mid = (bid + ask) / 2.0
        t = (expiry - day).days / 365.0
        if mid <= 0 or t <= 0 or underlying_price <= 0:
            return None
        solved = implied_vol(
            mid, underlying_price, strike, t, _RIGHT_MAP[right_s], r=self._cfg.risk_free_rate
        )
        if solved is None or not math.isfinite(solved):
            return None
        return bs_greeks(
            underlying_price, strike, t, solved, _RIGHT_MAP[right_s], r=self._cfg.risk_free_rate
        )

    def get_quote(self, symbol: str, at: datetime | None = None) -> Quote:
        if at is None:
            raise ValueError("ThetaDataHistorical requires an explicit 'at' timestamp")
        day = at.date()
        tstr = at.time().strftime("%H:%M:%S") + ".000"
        key = f"{symbol}_{day:%Y%m%d}_{at.time():%H%M}"
        df = self._cached(
            "stock_quote",
            key,
            "/v3/stock/at_time/quote",
            {
                "symbol": symbol,
                "start_date": f"{day:%Y%m%d}",
                "end_date": f"{day:%Y%m%d}",
                "time_of_day": tstr,
            },
        )
        if df.empty:
            raise DataUnavailableError(f"No quote for {symbol} at {at}")
        row = df.iloc[0]
        return Quote(
            symbol=symbol,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
        )

    def get_greeks(self, key: OptionKey, at: datetime) -> Greeks:
        chain_row = self._greeks_frame(key.underlying, key.expiry, at.date(), at.time())
        if not chain_row.empty:
            match = chain_row[
                (chain_row["strike"].astype(float) == key.strike)
                & (chain_row["right"] == ("CALL" if key.right is OptionRight.CALL else "PUT"))
            ]
            if not match.empty:
                row = match.iloc[0]

                def _f(col: str) -> float:
                    v = row.get(col)
                    return float(v) if v is not None else float("nan")

                g = self._scalar_greeks(
                    _f("delta"), _f("theta"), _f("vega"), _f("rho"), _f("implied_vol"),
                    key.strike,
                    str(row["right"]),
                    float(row["underlying_price"]),
                    at.date(),
                    key.expiry,
                    float(row["bid"]),
                    float(row["ask"]),
                )
                if g is not None:
                    return g
        raise DataUnavailableError(f"No greeks for {key} at {at}")

    def get_history(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        if self._history_provider is not None:
            return self._history_provider.get_history(symbol, start, end, interval)  # type: ignore[attr-defined]
        if interval != "1d":
            raise NotImplementedError("Backtest history is daily; intraday not needed yet")
        key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_1d"
        df = self._cached(
            "stock_eod",
            key,
            "/v3/stock/history/eod",
            {
                "symbol": symbol,
                "start_date": f"{start:%Y%m%d}",
                "end_date": f"{end:%Y%m%d}",
            },
        )
        if df.empty:
            raise DataUnavailableError(f"No history for {symbol} {start}..{end}")
        out = pd.DataFrame(
            {
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(int),
            },
            index=pd.to_datetime(df["created"]).dt.normalize(),
        )
        out.index.name = "date"
        return out

    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        if self._catalyst_provider is None:
            raise NotImplementedError("Inject an FMP catalyst provider (built in M2)")
        return self._catalyst_provider.get_catalyst_calendar(start, end)  # type: ignore[attr-defined]

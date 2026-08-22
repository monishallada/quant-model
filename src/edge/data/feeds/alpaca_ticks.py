"""Alpaca historical SIP tick data: NBBO quotes and trade prints.

Pulls ``GET /v2/stocks/{symbol}/quotes`` and ``/trades`` from
data.alpaca.markets (feed=sip), paginated via ``next_page_token``, and
normalizes to :class:`~edge.core.events.QuoteEvent` /
:class:`~edge.core.events.TradeEvent`. Auth, retry, and pagination are ported
from ``catalyst.data.alpaca_history``: APCA header auth, backoff on 429/5xx,
and a mid-pagination failure is NEVER cached (a truncated day silently served
from cache would poison every bar built on it).

Cache-first: ``<cache_root>/{quotes|trades}/symbol=<SYM>/date=<YYYY-MM-DD>.parquet``
is served without touching HTTP; writes are atomic (tmp + replace). The HTTP
client is injected so tests drive the adapter with canned pages — no network.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edge.core.events import QuoteEvent, TradeEvent

logger = logging.getLogger(__name__)

#: payload-field specs: bar column -> (Alpaca JSON key, cast).
_FieldSpec = Mapping[str, tuple[str, Callable[[Any], Any]]]
_QUOTE_FIELDS: _FieldSpec = {
    "bid": ("bp", float),
    "ask": ("ap", float),
    "bid_size": ("bs", int),
    "ask_size": ("as", int),
}
_TRADE_FIELDS: _FieldSpec = {
    "price": ("p", float),
    "size": ("s", int),
}


class AlpacaTicksConfig(BaseModel):
    """Adapter knobs; defaults mirror ``catalyst.data.alpaca_history`` and are
    overridable per environment — nothing below is baked into code paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = "https://data.alpaca.markets"
    #: SIP for parity with catalyst's intraday layer (audit D-126: IEX is a
    #: thinner venue with occasionally different prints).
    feed: Literal["sip", "iex"] = "sip"
    page_limit: int = Field(default=10_000, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_attempts: int = Field(default=6, gt=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0)


class _Response(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


@runtime_checkable
class HttpGetter(Protocol):
    """The one slice of ``httpx.Client`` the adapter uses — tests inject a
    fake with canned pages; production injects a real authenticated client."""

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _Response: ...


class AlpacaTicks:
    """Cache-first Alpaca historical tick adapter.

    Contract: :meth:`quotes` / :meth:`trades` return one symbol's events for
    one session date, sorted ascending by ``ts``, normalized to core event
    types (tz-aware America/New_York timestamps). A completed pull — even an
    empty one — is cached; a pull that fails mid-pagination returns ``[]``
    and is never cached, so the next call retries the network.
    """

    def __init__(
        self,
        client: HttpGetter,
        cache_root: str | Path,
        config: AlpacaTicksConfig | None = None,
    ) -> None:
        self._http = client
        self._cache_root = Path(cache_root)
        self._config = config or AlpacaTicksConfig()

    @classmethod
    def from_env(
        cls, cache_root: str | Path, config: AlpacaTicksConfig | None = None
    ) -> AlpacaTicks:
        """Build with a real httpx client authenticated from the environment.

        Keys come from ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` — never
        hardcoded, never committed. Raises ``RuntimeError`` when unset.
        """
        config = config or AlpacaTicksConfig()
        key = os.environ.get("ALPACA_API_KEY", "")
        # The repo's canonical .env name is ALPACA_SECRET_KEY (catalyst
        # convention); ALPACA_API_SECRET is accepted as an alias.
        secret = (os.environ.get("ALPACA_SECRET_KEY", "")
                  or os.environ.get("ALPACA_API_SECRET", ""))
        if not key or not secret:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Put them in .env — "
                "they are never hardcoded and never committed."
            )
        client = httpx.Client(
            base_url=config.base_url,
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=config.request_timeout_seconds,
        )
        return cls(client, cache_root, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quotes(self, symbol: str, day: date) -> list[QuoteEvent]:
        """All SIP NBBO quotes for ``symbol`` on ``day``, ascending by ts."""
        frame = self._frame("quotes", symbol, day, _QUOTE_FIELDS)
        return [
            QuoteEvent(
                ts=row.ts.to_pydatetime(),
                symbol=symbol,
                bid=row.bid,
                ask=row.ask,
                bid_size=row.bid_size,
                ask_size=row.ask_size,
            )
            for row in frame.itertuples(index=False)
        ]

    def trades(self, symbol: str, day: date) -> list[TradeEvent]:
        """All SIP trade prints for ``symbol`` on ``day``, ascending by ts.

        Real SIP tapes carry zero-size prints (corrections/cancels and some
        condition codes). They hold no volume information, so the event layer
        drops them here — loudly counted — while the raw parquet cache keeps
        every print verbatim as the archival truth.
        """
        frame = self._frame("trades", symbol, day, _TRADE_FIELDS)
        n_zero = int((frame["size"] <= 0).sum()) if not frame.empty else 0
        if n_zero:
            logger.info("dropping %d zero-size prints %s %s", n_zero, symbol, day)
            frame = frame[frame["size"] > 0]
        return [
            TradeEvent(ts=row.ts.to_pydatetime(), symbol=symbol, price=row.price, size=row.size)
            for row in frame.itertuples(index=False)
        ]

    # ------------------------------------------------------------------
    # Cache + fetch
    # ------------------------------------------------------------------

    def _cache_path(self, kind: str, symbol: str, day: date) -> Path:
        return self._cache_root / kind / f"symbol={symbol}" / f"date={day.isoformat()}.parquet"

    def _frame(self, kind: str, symbol: str, day: date, fields: _FieldSpec) -> pd.DataFrame:
        """Cached frame if present, else fetch-normalize-cache. A failed fetch
        yields an empty frame that is NOT cached."""
        columns = ["ts", *fields]
        path = self._cache_path(kind, symbol, day)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
                logger.warning("unreadable cache %s: %s — refetching", path, exc)
        rows = self._fetch_rows(kind, symbol, day, fields)
        if rows is None:
            logger.warning("Alpaca %s %s %s failed mid-pagination (NOT cached)", kind, symbol, day)
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
        _write_atomic(path, frame)
        return frame

    def _fetch_rows(
        self, kind: str, symbol: str, day: date, fields: _FieldSpec
    ) -> list[dict[str, Any]] | None:
        """All pages for one (kind, symbol, day); None on any failed page."""
        cfg = self._config
        rows: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            params: dict[str, str] = {
                "start": day.isoformat(),
                "end": day.isoformat(),
                "feed": cfg.feed,
                "limit": str(cfg.page_limit),
            }
            if token:
                params["page_token"] = token
            payload = self._get_with_retry(f"/v2/stocks/{symbol}/{kind}", params)
            if payload is None:
                return None
            for raw in payload.get(kind) or []:
                row: dict[str, Any] = {"ts": _parse_ts(raw["t"])}
                for column, (key, cast) in fields.items():
                    row[column] = cast(raw[key])
                rows.append(row)
            token = payload.get("next_page_token")
            if not token:
                return rows

    def _get_with_retry(self, path: str, params: dict[str, str]) -> dict[str, Any] | None:
        """GET with backoff. Alpaca rate-limits at 200 req/min and answers 429;
        a bulk tick pull hits that routinely, so it must be handled rather than
        crash a multi-hour run (ported from catalyst.data.alpaca_history)."""
        cfg = self._config
        for attempt in range(cfg.max_attempts):
            try:
                resp = self._http.get(path, params=params)
            except httpx.HTTPError as exc:
                logger.warning("Alpaca request error %s: %s", path, exc)
                time.sleep(cfg.retry_backoff_seconds * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = cfg.retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "Alpaca %s -> HTTP %d; retrying in %.0fs", path, resp.status_code, wait
                )
                time.sleep(wait)
                continue
            logger.warning("Alpaca %s -> HTTP %d: %s", path, resp.status_code, resp.text[:160])
            return None
        logger.warning("Alpaca %s exhausted retries", path)
        return None


def _parse_ts(value: str) -> datetime:
    """RFC-3339 (nanosecond precision, ``Z`` suffix) -> tz-aware datetime.

    Alpaca stamps ticks to the nanosecond; python datetimes carry microseconds,
    so the fraction is truncated to 6 digits — deterministic, and identical to
    what the parquet round-trip preserves.
    """
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    if "." in value:
        head, rest = value.split(".", 1)
        digits = len(rest)
        for i, ch in enumerate(rest):
            if not ch.isdigit():
                digits = i
                break
        frac, offset = rest[:digits], rest[digits:]
        value = f"{head}.{frac[:6].ljust(6, '0')}{offset}"
    return datetime.fromisoformat(value)


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via tmp + replace so a crash or concurrent writer can
    never leave a torn file at the final path (ported from catalyst cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

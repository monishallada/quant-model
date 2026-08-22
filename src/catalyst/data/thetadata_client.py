"""HTTP client for the local ThetaData Terminal v3 gateway.

Owns connection discipline: bounded retries with exponential backoff for
transient failures, and a reconnect wait when the terminal itself is down
(connection refused) so a terminal restart doesn't kill a long backtest.
"""

from __future__ import annotations

import io
import logging
import time

import httpx
import pandas as pd

from catalyst.core.config import ThetaDataConfig

logger = logging.getLogger(__name__)

# ThetaData "no data for this request" status (inherited from v2 conventions).


class ThetaDataDeadSession(RuntimeError):
    """HTTP 478: the terminal needs a restart. Not retryable in-process, and
    deliberately NOT a ThetaDataError so no caller cache-and-degrades it."""


class ThetaDataError(Exception):
    """Terminal returned an unrecoverable error for a request."""


class ThetaDataClient:
    def __init__(self, cfg: ThetaDataConfig) -> None:
        self._cfg = cfg
        self._http = httpx.Client(base_url=cfg.base_url, timeout=cfg.request_timeout_seconds)

    def close(self) -> None:
        self._http.close()

    def get_dataframe(self, path: str, params: dict[str, str]) -> pd.DataFrame:
        """GET a CSV endpoint into a DataFrame. Empty frame = no data.

        Retries transient failures (connection errors, 5xx, timeouts) up to
        ``max_retries`` with exponential backoff; waits ``reconnect_wait_seconds``
        extra when the terminal looks down entirely.
        """
        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp = self._http.get(path, params=params)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Terminal down/restarting — give it time to come back.
                last_error = exc
                logger.warning(
                    "ThetaData terminal unreachable (%s); waiting %.0fs (attempt %d/%d)",
                    exc, self._cfg.reconnect_wait_seconds, attempt + 1, self._cfg.max_retries,
                )
                time.sleep(self._cfg.reconnect_wait_seconds)
                continue
            except (httpx.ReadTimeout, httpx.ReadError, httpx.WriteError,
                    httpx.WriteTimeout, httpx.PoolTimeout,
                    httpx.RemoteProtocolError) as exc:
                # ReadError is what a terminal RESTART actually raises
                # mid-response (audit D-135) — it must retry, not crash.
                last_error = exc
                backoff = self._cfg.retry_backoff_seconds * (2**attempt)
                logger.warning("ThetaData request failed (%s); retrying in %.1fs", exc, backoff)
                time.sleep(backoff)
                continue

            if resp.status_code == 200:
                text = resp.text
                if not text.strip():
                    # 200-with-empty-body from a warming terminal is transient
                    # (audit D-054): retry before believing "no data".
                    if attempt < self._cfg.max_retries:
                        last_error = ThetaDataError("200 with empty body")
                        backoff = self._cfg.retry_backoff_seconds * (2**attempt)
                        logger.warning("ThetaData %s -> empty body; retrying "
                                       "in %.1fs", path, backoff)
                        time.sleep(backoff)
                        continue
                    return pd.DataFrame()
                return pd.read_csv(io.StringIO(text))
            if resp.status_code == 472:
                # "no data" from a terminal that may still be connecting —
                # observed to flip to real data on ONE retry (audit D-054).
                # Persistent 472 is the legitimate no-data answer; a full
                # backoff ladder here cost ~60s per empty contract-day and
                # throttled minute-data backtests to a crawl (v16 pilot).
                if attempt == 0:
                    last_error = ThetaDataError("HTTP 472")
                    logger.debug("ThetaData %s -> 472; one quick retry", path)
                    time.sleep(self._cfg.retry_backoff_seconds)
                    continue
                return pd.DataFrame()
            if resp.status_code == 478:
                # Dead terminal session: EVERY request will fail until the
                # terminal is restarted. Abort loudly rather than letting a
                # whole backtest degrade request-by-request (and, before the
                # cache fix, poison the cache with empties).
                raise ThetaDataDeadSession(
                    "ThetaData terminal session is dead (HTTP 478). Restart "
                    "scripts/run_thetadata.sh; ensure no second terminal uses "
                    "this API key.")
            if 500 <= resp.status_code < 600:
                last_error = ThetaDataError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                backoff = self._cfg.retry_backoff_seconds * (2**attempt)
                logger.warning("ThetaData %s -> HTTP %d; retrying in %.1fs",
                               path, resp.status_code, backoff)
                time.sleep(backoff)
                continue
            # 4xx other than no-data: request/entitlement bug — do not retry.
            raise ThetaDataError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")

        raise ThetaDataError(f"{path} failed after {self._cfg.max_retries + 1} attempts: {last_error}")

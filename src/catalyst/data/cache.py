"""Local parquet cache for historical data pulls.

8-year backtests × parameter sweeps would otherwise re-request the same
history from the ThetaData terminal for hours per run. Every raw endpoint
response is cached once on disk; sweeps and re-runs are then disk-speed and
byte-for-byte reproducible.

Keys are (category, key) → ``<cache_dir>/<category>/<key>.parquet``. An empty
response is cached too (as an empty frame) so known-empty days aren't
re-requested — ``get`` distinguishes "cached empty" from "not cached".

Durability contract (audit v15):
- writes are ATOMIC (tmp file + os.replace) — a crashed or concurrent run can
  never leave a torn parquet at the final path (D-046);
- an unreadable cache file is QUARANTINED (renamed ``.corrupt-*``) and treated
  as a miss, instead of crashing every future run that touches it;
- distinct logical keys can never collide onto one file: characters outside
  the safe set are replaced, and any key that needed replacement gets a short
  digest suffix so ``BRK/B`` and ``BRK B`` stay distinct (D-211).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_KEY_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class ParquetCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)

    def _path(self, category: str, key: str) -> Path:
        safe = _KEY_SANITIZE.sub("_", key)
        if safe != key:
            # sanitization is lossy; the digest keeps distinct keys distinct
            safe = f"{safe}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
        return self.root / category / f"{safe}.parquet"

    def get(self, category: str, key: str) -> pd.DataFrame | None:
        """Cached frame, or None when never cached (empty frame = cached empty)."""
        path = self._path(category, key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:                          # noqa: BLE001
            quarantine = path.with_suffix(
                f".corrupt-{datetime.now(UTC):%Y%m%dT%H%M%S}")
            try:
                path.rename(quarantine)
                logger.error("corrupt cache file %s: %s — quarantined as %s, "
                             "treating as a miss", path, e, quarantine.name)
            except OSError:
                logger.error("corrupt cache file %s: %s (quarantine failed)",
                             path, e)
            return None

    def exists(self, category: str, key: str) -> bool:
        return self._path(category, key).exists()

    def put(self, category: str, key: str, df: pd.DataFrame) -> None:
        path = self._path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic: write to a unique tmp then rename over the final path, so a
        # crash or a concurrent writer can never tear the file (D-046)
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        try:
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

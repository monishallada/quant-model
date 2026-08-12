"""Local parquet cache for historical data pulls.

8-year backtests × parameter sweeps would otherwise re-request the same
history from the ThetaData terminal for hours per run. Every raw endpoint
response is cached once on disk; sweeps and re-runs are then disk-speed and
byte-for-byte reproducible.

Keys are (category, key) → ``<cache_dir>/<category>/<key>.parquet``. An empty
response is cached too (as an empty frame) so known-empty days aren't
re-requested — ``get`` distinguishes "cached empty" from "not cached".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_KEY_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


class ParquetCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)

    def _path(self, category: str, key: str) -> Path:
        safe = _KEY_SANITIZE.sub("_", key)
        return self.root / category / f"{safe}.parquet"

    def get(self, category: str, key: str) -> pd.DataFrame | None:
        """Cached frame, or None when never cached (empty frame = cached empty)."""
        path = self._path(category, key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, category: str, key: str) -> bool:
        return self._path(category, key).exists()

    def put(self, category: str, key: str, df: pd.DataFrame) -> None:
        path = self._path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

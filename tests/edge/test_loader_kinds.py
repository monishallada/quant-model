"""Loader-level tests for the point-in-time feed kinds.

One test per kind (``short_ratio``, ``vol_indices``, ``rates``, ``cot``,
``insider``, ``earnings``) proves the lockbox wall clamps its rows: a
locked :class:`EdgeDataLoader` drops every row whose ``asof_date`` OR
``available_at`` fails to predate the wall. ``available_at`` GOVERNS — it
embargoes rows published late even when their data date predates the wall —
and ``asof_date`` is enforced as defense-in-depth against
anti-conservative stamps.

ALL data here is synthetic. Timestamps inside the lockbox window appear
ONLY because these are embargo-machinery tests — they exercise the wall
itself, never real market data. Every write happens under a pytest tmp
repo root, never the real one.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.loader import (
    ALL_SYMBOLS,
    EdgeDataLoader,
    FeedsBackend,
    FetchCall,
    PER_SYMBOL_PIT_KINDS,
    POINT_IN_TIME_KINDS,
)
from edge.validation.lockbox import LockboxKey, LockboxViolation

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

# Synthetic anchors around the wall (guard tests only).
WALL = date(2026, 2, 22)  # Sunday
LAST_ALLOWED = date(2026, 2, 21)
PRE_START = date(2026, 2, 2)
PRE_END = LAST_ALLOWED
POST_END = date(2026, 3, 31)

KINDS = sorted(POINT_IN_TIME_KINDS)

#: One representative value column per kind, so each canned frame is shaped
#: like the real adapter's output (values are arbitrary synthetic numbers).
VALUE_COLUMNS = {
    "short_ratio": "short_ratio",
    "vol_indices": "vix",
    "rates": "y10",
    "cot": "lev_net_oi",
    "insider": "net_insider_dollars_21d",
    "earnings": "days_to_next_earnings",
    "micro": "imbalance",
}

#: The KEEP row's available_at per kind — each mirrors that kind's real
#: publication rule for an asof of Tue 2026-02-17 (all predate the wall).
KEEP_AVAILABLE = {
    "short_ratio": pd.Timestamp(2026, 2, 18, 9, 0, tz=ET),  # next preopen
    "vol_indices": pd.Timestamp(2026, 2, 17, 18, 0, tz=ET),  # same-day 18:00
    "rates": pd.Timestamp(2026, 2, 17, 18, 0, tz=ET),  # same-day 18:00
    "cot": pd.Timestamp(2026, 2, 20, 16, 0, tz=ET),  # Friday 16:00
    "insider": pd.Timestamp(2026, 2, 17, 20, 35, tz=ET),  # acceptance + 5min
    "earnings": pd.Timestamp(2026, 2, 17, 0, 0, tz=ET),  # asof 00:00 ET
    # the minute's own CLOSE: features are computed from prints fully
    # observed by that instant, and consumed one bar later
    "micro": pd.Timestamp(2026, 2, 17, 15, 59, tz=ET),
}

#: The LATE-PUBLICATION row's available_at per kind: asof Fri 2026-02-20
#: (pre-wall) but first publicly visible at/after the wall — e.g. FINRA's
#: Friday file at Monday preopen, FRED's next-business-day refresh, a
#: late-filed Form 4, a post-wall earnings-calendar snapshot.
LATE_AVAILABLE = {
    "short_ratio": pd.Timestamp(2026, 2, 23, 9, 0, tz=ET),
    "vol_indices": pd.Timestamp(2026, 2, 22, 18, 0, tz=ET),
    "rates": pd.Timestamp(2026, 2, 23, 18, 0, tz=ET),
    "cot": pd.Timestamp(2026, 2, 27, 16, 0, tz=ET),
    "insider": pd.Timestamp(2026, 2, 23, 10, 10, tz=ET),
    "earnings": pd.Timestamp(2026, 2, 23, 18, 0, tz=ET),
    "micro": pd.Timestamp(2026, 2, 23, 9, 31, tz=ET),
}

SYMBOL = "AAPL"


class PitStubBackend:
    """Canned point-in-time frame; records calls like SyntheticBackend."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.calls: list[FetchCall] = []

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        self.calls.append(FetchCall(symbol=symbol, start=start, end=end, kind=kind))
        return self._frame


def make_repo_root(tmp_path: Path, lockbox_start: str) -> Path:
    """Copy the real edge.yaml into a tmp repo root with a rewritten wall."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    text = (REPO_ROOT / "config" / "edge.yaml").read_text()
    new_text = re.sub(r'lockbox_start:\s*"[^"]*"', f'lockbox_start: "{lockbox_start}"', text)
    assert new_text != text or lockbox_start in text
    (cfg_dir / "edge.yaml").write_text(new_text)
    return tmp_path


def build_frame(kind: str) -> pd.DataFrame:
    """Three rows straddling the wall, stamped per ``kind``'s conventions.

    - ``keep``:         asof 2026-02-17, published pre-wall  -> must survive
    - ``late_pub``:     asof 2026-02-20 (pre-wall) but first PUBLISHED
                        at/after the wall -> dropped (available_at governs)
    - ``lockbox_asof``: asof 2026-02-23 (in the lockbox) under an
                        anti-conservative pre-wall stamp -> dropped
                        (asof_date enforced as defense-in-depth)
    """
    frame = pd.DataFrame(
        {
            "row": ["keep", "late_pub", "lockbox_asof"],
            "symbol": SYMBOL,
            "asof_date": [
                pd.Timestamp(2026, 2, 17),
                pd.Timestamp(2026, 2, 20),
                pd.Timestamp(2026, 2, 23),
            ],
            "available_at": [
                KEEP_AVAILABLE[kind],
                LATE_AVAILABLE[kind],
                pd.Timestamp(2026, 2, 20, 18, 0, tz=ET),  # anti-conservative
            ],
            VALUE_COLUMNS[kind]: [1.0, 2.0, 3.0],
        }
    )
    if kind == "short_ratio":
        # FINRA's convention: asof_date holds python date objects.
        frame["asof_date"] = frame["asof_date"].dt.date
    return frame


# ---------------------------------------------------------------------------
# The wall clamps every point-in-time kind, row by row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_wall_clamps_point_in_time_rows(
    kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_repo_root(tmp_path, WALL.isoformat())
    backend = PitStubBackend(build_frame(kind))
    loader = EdgeDataLoader(backend, repo_root=root)

    with caplog.at_level(logging.WARNING, logger="edge.data.loader"):
        out = loader.load(SYMBOL, PRE_START, POST_END, kind)

    # Range clamp reached the backend first (end truncated to wall - 1)...
    assert backend.calls == [
        FetchCall(symbol=SYMBOL, start=PRE_START, end=LAST_ALLOWED, kind=kind)
    ]
    # ...then the row clamp kept ONLY the pre-wall-published pre-wall row.
    assert list(out["row"]) == ["keep"]
    assert "point-in-time clamp" in caplog.text
    assert "available_at governs" in caplog.text


def test_available_at_governs_even_without_range_clamp(tmp_path: Path) -> None:
    """A row whose asof_date predates the wall is STILL embargoed when its
    publication lands at/after the wall — with a request range entirely
    pre-wall, so only the row-level available_at clamp can drop it."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    frame = pd.DataFrame(
        {
            "row": ["early_pub", "late_pub"],
            "symbol": SYMBOL,
            "asof_date": [date(2026, 2, 20), date(2026, 2, 20)],
            "available_at": [
                pd.Timestamp(2026, 2, 20, 18, 0, tz=ET),
                pd.Timestamp(2026, 2, 23, 9, 0, tz=ET),  # Monday preopen
            ],
            "short_ratio": [0.4, 0.6],
        }
    )
    backend = PitStubBackend(frame)
    loader = EdgeDataLoader(backend, repo_root=root)

    out = loader.load(SYMBOL, PRE_START, PRE_END, "short_ratio")

    assert backend.calls == [
        FetchCall(symbol=SYMBOL, start=PRE_START, end=PRE_END, kind="short_ratio")
    ]
    assert list(out["row"]) == ["early_pub"]


def test_start_inside_lockbox_raises_for_pit_kind(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, WALL.isoformat())
    backend = PitStubBackend(build_frame("earnings"))
    loader = EdgeDataLoader(backend, repo_root=root)
    with pytest.raises(LockboxViolation):
        loader.load(SYMBOL, WALL, POST_END, "earnings")
    assert backend.calls == []  # the backend was never reached


def test_unlocked_key_passes_pit_rows_through(tmp_path: Path) -> None:
    """Once the lockbox is legitimately opened, no row filtering applies."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    key = LockboxKey.issue("pit passthrough check", repo_root=root)
    backend = PitStubBackend(build_frame("rates"))
    loader = EdgeDataLoader(backend, key=key, repo_root=root)

    out = loader.load("MKT", PRE_START, POST_END, "rates")

    assert backend.calls == [
        FetchCall(symbol="MKT", start=PRE_START, end=POST_END, kind="rates")
    ]
    assert list(out["row"]) == ["keep", "late_pub", "lockbox_asof"]


# ---------------------------------------------------------------------------
# The point-in-time frame contract is enforced loudly
# ---------------------------------------------------------------------------


def test_pit_frame_missing_available_at_rejected(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, WALL.isoformat())
    frame = pd.DataFrame({"asof_date": [pd.Timestamp(2026, 2, 17)], "vix": [15.0]})
    loader = EdgeDataLoader(PitStubBackend(frame), repo_root=root)
    with pytest.raises(ValueError, match="available_at"):
        loader.load("MKT", PRE_START, PRE_END, "vol_indices")


def test_pit_frame_missing_asof_date_rejected(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, WALL.isoformat())
    frame = pd.DataFrame(
        {"available_at": [pd.Timestamp(2026, 2, 17, 18, 0, tz=ET)], "y10": [4.2]}
    )
    loader = EdgeDataLoader(PitStubBackend(frame), repo_root=root)
    with pytest.raises(ValueError, match="asof_date"):
        loader.load("MKT", PRE_START, PRE_END, "rates")


def test_pit_naive_available_at_rejected(tmp_path: Path) -> None:
    """A naive available_at cannot be gated against an ET wall — loud error."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    frame = pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(2026, 2, 17)],
            "available_at": [pd.Timestamp(2026, 2, 17, 18, 0)],  # naive
            "vix": [15.0],
        }
    )
    loader = EdgeDataLoader(PitStubBackend(frame), repo_root=root)
    with pytest.raises(ValueError, match="tz-aware"):
        loader.load("MKT", PRE_START, PRE_END, "vol_indices")


def test_pit_nat_available_at_dropped_conservatively(tmp_path: Path) -> None:
    """Unknowable availability is treated as unavailable, never let through."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    frame = pd.DataFrame(
        {
            "row": ["keep", "unknown_pub"],
            "asof_date": [pd.Timestamp(2026, 2, 17), pd.Timestamp(2026, 2, 18)],
            "available_at": [pd.Timestamp(2026, 2, 17, 18, 0, tz=ET), pd.NaT],
            "vix": [15.0, 16.0],
        }
    )
    loader = EdgeDataLoader(PitStubBackend(frame), repo_root=root)
    out = loader.load("MKT", PRE_START, PRE_END, "vol_indices")
    assert list(out["row"]) == ["keep"]


# ---------------------------------------------------------------------------
# FeedsBackend routing (archives absent: empty frames, real adapters, no net)
# ---------------------------------------------------------------------------


def test_feeds_backend_serves_only_pit_kinds(tmp_path: Path) -> None:
    backend = FeedsBackend(repo_root=tmp_path)
    with pytest.raises(ValueError, match="point-in-time"):
        backend.fetch("SPY", PRE_START, PRE_END, "bars")
    with pytest.raises(ValueError, match="point-in-time"):
        backend.fetch("SPY", PRE_START, PRE_END, "nonsense")  # type: ignore[arg-type]


def test_with_feeds_routes_short_ratio(tmp_path: Path) -> None:
    """with_feeds reaches the real FINRA adapter; an empty archive yields an
    empty frame that still honors the point-in-time column contract."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    loader = EdgeDataLoader.with_feeds(repo_root=root)
    out = loader.load(SYMBOL, PRE_START, PRE_END, "short_ratio")
    assert out.empty
    assert {"asof_date", "available_at"} <= set(out.columns)


def test_with_feeds_routes_insider(tmp_path: Path) -> None:
    """with_feeds reaches the real EDGAR feature path; an empty filings cache
    yields an empty frame with the point-in-time columns."""
    root = make_repo_root(tmp_path, WALL.isoformat())
    loader = EdgeDataLoader.with_feeds(repo_root=root)
    out = loader.load(ALL_SYMBOLS, PRE_START, PRE_END, "insider")
    assert out.empty
    assert {"asof_date", "available_at"} <= set(out.columns)


def test_per_symbol_kinds_are_a_subset_of_pit_kinds() -> None:
    assert PER_SYMBOL_PIT_KINDS <= POINT_IN_TIME_KINDS

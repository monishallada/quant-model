"""Earnings adapter tests: canned parquet fixtures in tmp_path, zero network.

All fixture event dates predate the 2026-02-22 lockbox start. The live
refresh path is exercised only through an injected fake fetcher — yfinance
is never imported, the network is never touched.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge.data.feeds.earnings import (
    SESSION_AMC,
    SESSION_BMO,
    SESSION_DMT,
    SESSION_UNKNOWN,
    EarningsConfig,
    EarningsFeed,
)

ET = "America/New_York"

#: Deterministic snapshot override (tz-aware, pre-lockbox): the archive
#: "was fetched" mid-day 2025-06-15 -> conservative stamp 18:00 ET that day.
SNAPSHOT = pd.Timestamp("2025-06-15 12:34", tz=ET)
SNAPSHOT_STAMP = pd.Timestamp("2025-06-15 18:00", tz=ET)


def aaa_frame() -> pd.DataFrame:
    """One event per session bucket, spanning 2024-12 .. 2025-08."""
    return pd.DataFrame(
        {
            "when": pd.to_datetime(
                [
                    "2024-12-04 16:30",  # amc
                    "2025-03-05 07:00",  # bmo (Wednesday)
                    "2025-05-07 13:00",  # dmt
                    "2025-08-06 00:00",  # exact midnight -> unknown
                ]
            ),
            "eps_estimate": [1.0, 1.1, np.nan, 1.3],
            "eps_actual": [1.05, 1.15, 1.25, np.nan],
        }
    )


def bbb_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": pd.to_datetime(["2025-03-12 06:00", "2025-06-11 16:00"]),
            "eps_estimate": [2.0, 2.1],
            "eps_actual": [2.05, np.nan],
        }
    )


def make_archive(root: Path, frames: dict[str, pd.DataFrame]) -> Path:
    archive = root / "earnings_yf"
    archive.mkdir(parents=True)
    for symbol, frame in frames.items():
        frame.to_parquet(archive / f"{symbol}.parquet", index=False)
    return archive


def make_feed(
    tmp_path: Path,
    frames: dict[str, pd.DataFrame] | None = None,
    snapshot_ts: pd.Timestamp | None = SNAPSHOT,
    **cfg: object,
) -> EarningsFeed:
    if frames is None:
        frames = {"AAA": aaa_frame(), "BBB": bbb_frame()}
    archive = make_archive(tmp_path, frames)
    return EarningsFeed(
        archive, tmp_path / "cache", config=EarningsConfig(**cfg), snapshot_ts=snapshot_ts
    )


# ----------------------------------------------------------------------
# Event table
# ----------------------------------------------------------------------


def test_load_events_schema_sessions_and_order(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    events = feed.load_events()

    assert list(events.columns) == [
        "asof_date", "available_at", "symbol", "event_date", "event_ts",
        "session", "eps_estimate", "eps_actual", "calendar_snapshot_ts",
    ]
    assert len(events) == 6  # 4 AAA + 2 BBB, sorted by (symbol, event_ts)
    assert list(events["symbol"]) == ["AAA"] * 4 + ["BBB"] * 2

    aaa = events[events["symbol"] == "AAA"]
    assert list(aaa["session"]) == [SESSION_AMC, SESSION_BMO, SESSION_DMT, SESSION_UNKNOWN]
    # event_ts is the tz-aware ET report datetime; event_date its midnight
    assert aaa["event_ts"].iloc[0] == pd.Timestamp("2024-12-04 16:30", tz=ET)
    assert aaa["event_date"].iloc[0] == pd.Timestamp("2024-12-04")
    assert aaa["asof_date"].equals(aaa["event_date"])
    assert aaa["eps_estimate"].iloc[1] == pytest.approx(1.1)
    assert np.isnan(aaa["eps_actual"].iloc[3])


def test_events_available_at_is_snapshot_fetch_date_1800_et(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    events = feed.load_events()

    # The SNAPSHOT's conservative availability — not the event's own date.
    assert (events["available_at"] == SNAPSHOT_STAMP).all()
    assert (events["calendar_snapshot_ts"] == SNAPSHOT_STAMP).all()
    assert events["available_at"].iloc[0].utcoffset() == pd.Timedelta(hours=-4)  # EDT


def test_snapshot_ts_defaults_to_archive_file_mtime(tmp_path: Path) -> None:
    import os

    feed = make_feed(tmp_path, snapshot_ts=None)
    fetched = datetime.fromisoformat("2025-01-10 09:00-05:00").timestamp()
    for p in (tmp_path / "earnings_yf").glob("*.parquet"):
        os.utime(p, (fetched, fetched))

    events = feed.load_events()
    stamp = pd.Timestamp("2025-01-10 18:00", tz=ET)  # fetch DATE at 18:00 ET
    assert (events["calendar_snapshot_ts"] == stamp).all()
    assert (events["available_at"] == stamp).all()
    assert events["available_at"].iloc[0].utcoffset() == pd.Timedelta(hours=-5)  # EST


def test_naive_snapshot_ts_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        make_feed(tmp_path, snapshot_ts=pd.Timestamp("2025-06-15 12:34"))


def test_load_events_symbol_filter_and_unknown_symbol_raises(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    only_bbb = feed.load_events(symbols=["BBB"])
    assert set(only_bbb["symbol"]) == {"BBB"}
    with pytest.raises(ValueError, match="ZZZ"):
        feed.load_events(symbols=["AAA", "ZZZ"])


def test_empty_archive_raises(tmp_path: Path) -> None:
    (tmp_path / "earnings_yf").mkdir()
    feed = EarningsFeed(tmp_path / "earnings_yf", tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        feed.load_events()


def test_cache_first_then_rebuild_on_newer_archive(tmp_path: Path) -> None:
    import os

    # No snapshot override: the parquet cache is only used on this path.
    feed = make_feed(tmp_path, snapshot_ts=None)
    first = feed.load_events()
    cache = tmp_path / "cache" / "events.parquet"
    assert cache.exists()

    # Corrupt one archive file but keep its mtime OLDER than the cache: the
    # combined parquet must be served without re-parsing (parsing raises).
    bbb_path = tmp_path / "earnings_yf" / "BBB.parquet"
    bbb_path.write_text("garbage, not a parquet file")
    stale = cache.stat().st_mtime - 100
    os.utime(bbb_path, (stale, stale))
    pd.testing.assert_frame_equal(feed.load_events(), first)

    # A refreshed archive file (newer mtime) forces a rebuild with new rows.
    extra = pd.concat(
        [bbb_frame(), pd.DataFrame({"when": [pd.Timestamp("2025-09-10 16:00")],
                                    "eps_estimate": [2.2], "eps_actual": [np.nan]})],
        ignore_index=True,
    )
    extra.to_parquet(bbb_path, index=False)
    fresh = cache.stat().st_mtime + 100
    os.utime(bbb_path, (fresh, fresh))
    rebuilt = feed.load_events()
    assert len(rebuilt) == len(first) + 1
    assert pd.Timestamp("2025-09-10 16:00", tz=ET) in set(rebuilt["event_ts"])


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def test_features_days_to_next_since_last_and_week_flag(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    feats = feed.features("2025-03-03", "2025-03-10")  # Mon..Mon, 6 bdays

    assert list(feats.columns) == [
        "asof_date", "available_at", "symbol", "days_to_next_earnings",
        "days_since_last_earnings", "is_earnings_week", "calendar_snapshot_ts",
    ]
    assert len(feats) == 12  # 6 business days x 2 symbols

    aaa = feats[feats["symbol"] == "AAA"].reset_index(drop=True)
    # AAA reports Wed 2025-03-05 (bmo); next after that is 2025-05-07.
    assert list(aaa["days_to_next_earnings"]) == pytest.approx([2, 1, 0, 62, 61, 58])
    # Previous event 2024-12-04; 0 on the event day itself.
    assert list(aaa["days_since_last_earnings"]) == pytest.approx([89, 90, 0, 1, 2, 5])
    # Monday-anchored week of 03-03 contains the event; week of 03-10 does not.
    assert list(aaa["is_earnings_week"]) == [True] * 5 + [False]

    bbb = feats[feats["symbol"] == "BBB"].reset_index(drop=True)
    # BBB's first known event is 2025-03-12: no prior history -> NaN.
    assert list(bbb["days_to_next_earnings"]) == pytest.approx([9, 8, 7, 6, 5, 2])
    assert bbb["days_since_last_earnings"].isna().all()
    # 03-12 falls in the week of Mon 03-10 only.
    assert list(bbb["is_earnings_week"]) == [False] * 5 + [True]


def test_features_nan_after_last_known_event(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    feats = feed.features("2025-09-01", "2025-09-05", symbols=["AAA"])
    # AAA's last snapshot event is 2025-08-06: nothing ahead -> NaN.
    assert feats["days_to_next_earnings"].isna().all()
    assert feats["days_since_last_earnings"].iloc[0] == pytest.approx(26)
    assert not feats["is_earnings_week"].any()


def test_default_available_at_never_precedes_snapshot(tmp_path: Path) -> None:
    """THE strict-default guarantee: on the DEFAULT path no feature row can
    carry an ``available_at`` that precedes the snapshot it was computed
    from — across a grid straddling the snapshot date, for every symbol."""
    feed = make_feed(tmp_path)  # default config: strict
    feats = feed.features("2025-03-03", "2025-06-20")  # spans SNAPSHOT (06-15)

    assert (feats["calendar_snapshot_ts"] == SNAPSHOT_STAMP).all()
    assert (feats["available_at"] >= feats["calendar_snapshot_ts"]).all()
    # Pre-snapshot days exist in the grid and are clamped exactly TO the stamp.
    pre = feats[feats["asof_date"] < pd.Timestamp("2025-06-15")]
    assert not pre.empty
    assert (pre["available_at"] == SNAPSHOT_STAMP).all()
    # Post-snapshot days keep the conservative start-of-day stamp.
    post = feats[feats["asof_date"] >= pd.Timestamp("2025-06-16")]
    assert not post.empty
    assert (post["available_at"] == post["asof_date"].dt.tz_localize(ET)).all()


def test_lookahead_optin_stamps_midnight_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Opting OUT of strict stamping requires the self-documenting
    ``allow_snapshot_lookahead=True`` and logs a WARNING naming the leak at
    construction; rows then carry the documented look-ahead audit trail."""
    with caplog.at_level(logging.WARNING, logger="edge.data.feeds.earnings"):
        feed = make_feed(tmp_path, allow_snapshot_lookahead=True)
    assert any(
        "allow_snapshot_lookahead" in r.message and "look" in r.message.lower()
        for r in caplog.records
    )

    feats = feed.features("2025-03-03", "2025-03-04", symbols=["AAA"])
    assert feats["available_at"].iloc[0] == pd.Timestamp("2025-03-03 00:00", tz=ET)
    assert feats["available_at"].iloc[0].utcoffset() == pd.Timedelta(hours=-5)  # EST
    # The documented look-ahead flag: these historical rows were computed
    # from a snapshot fetched later, and the audit column shows it.
    assert (feats["calendar_snapshot_ts"] == SNAPSHOT_STAMP).all()
    assert (feats["available_at"] < feats["calendar_snapshot_ts"]).all()


def test_old_flag_name_is_gone(tmp_path: Path) -> None:
    """The renamed flag cannot be set by its old name — extra='forbid' makes
    a stale call site fail loudly instead of silently getting the default."""
    with pytest.raises(Exception, match="strict_availability"):
        EarningsConfig(strict_availability=True)  # type: ignore[call-arg]


def test_features_default_clamps_to_snapshot(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)  # strict is the DEFAULT — no flag needed
    feats = feed.features("2025-06-13", "2025-06-17", symbols=["BBB"])  # Fri..Tue

    by_day = feats.set_index(feats["asof_date"].dt.strftime("%Y-%m-%d"))
    # Before the snapshot existed (Fri 00:00 < Sun 18:00): clamped up to it.
    assert by_day.loc["2025-06-13", "available_at"] == SNAPSHOT_STAMP
    # From the first midnight AFTER the snapshot stamp: normal start-of-day.
    assert by_day.loc["2025-06-16", "available_at"] == pd.Timestamp(
        "2025-06-16 00:00", tz=ET
    )
    assert by_day.loc["2025-06-17", "available_at"] == pd.Timestamp(
        "2025-06-17 00:00", tz=ET
    )


def test_features_unknown_symbol_and_empty_grid_raise(tmp_path: Path) -> None:
    feed = make_feed(tmp_path)
    with pytest.raises(ValueError, match="ZZZ"):
        feed.features("2025-03-03", "2025-03-10", symbols=["ZZZ"])
    with pytest.raises(ValueError, match="grid"):
        feed.features("2025-03-08", "2025-03-09")  # Sat..Sun: no business days


# ----------------------------------------------------------------------
# Live refresh (injected fake fetcher only — no yfinance, no network)
# ----------------------------------------------------------------------


def test_refresh_writes_success_and_never_persists_failure(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, snapshot_ts=None)
    feed.load_events()  # build the combined cache so invalidation is visible
    cache = tmp_path / "cache" / "events.parquet"
    assert cache.exists()
    original_aaa = pd.read_parquet(tmp_path / "earnings_yf" / "AAA.parquet")

    fresh = pd.DataFrame(
        {
            # unsorted + duplicate row: refresh must sort and dedupe on write
            "when": pd.to_datetime(
                ["2025-09-10 16:00", "2025-06-11 16:00", "2025-06-11 16:00"]
            ),
            "eps_estimate": [2.2, 2.1, 2.1],
            "eps_actual": [np.nan, 2.15, 2.15],
        }
    )

    def fake_fetch(symbol: str) -> pd.DataFrame | None:
        return fresh if symbol == "BBB" else None  # AAA: scrape failure

    written = feed.refresh(["AAA", "BBB"], fetcher=fake_fetch)

    assert written == {"BBB": 2}
    # Failure is NEVER persisted: AAA's good snapshot is byte-identical.
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "earnings_yf" / "AAA.parquet"), original_aaa
    )
    stored = pd.read_parquet(tmp_path / "earnings_yf" / "BBB.parquet")
    assert list(stored["when"]) == [
        pd.Timestamp("2025-06-11 16:00"), pd.Timestamp("2025-09-10 16:00")
    ]
    assert not cache.exists()  # combined cache invalidated for rebuild


def test_refresh_empty_result_not_persisted(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, snapshot_ts=None)
    empty = pd.DataFrame({"when": pd.to_datetime([]), "eps_estimate": [], "eps_actual": []})
    written = feed.refresh(["BBB"], fetcher=lambda s: empty)
    assert written == {}
    # The existing snapshot survives an empty (= failed) scrape untouched.
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "earnings_yf" / "BBB.parquet"), bbb_frame()
    )


def test_refresh_rejects_unsafe_symbol(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, snapshot_ts=None)
    with pytest.raises(ValueError, match="BRK/B"):
        feed.refresh(["BRK/B"], fetcher=lambda s: None)

"""Rates adapter tests: canned CSV fixtures in tmp_path, zero network.

All fixture dates predate the 2026-02-22 lockbox start. The FRED fetcher is
exercised only through an injected fake client returning canned fredgraph
text — the real fredgraph URL is never contacted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edge.data.feeds.rates import (
    FRED_SERIES,
    HY_OAS_SERIES,
    RatesConfig,
    RatesFeed,
)

ET = ZoneInfo("America/New_York")

# 2024 file: no "30 Yr"; 2y pinned at 4.00 so slope is hand-computable;
# one empty "3 Mo" cell. Rows newest-first, as the Treasury archive ships.
YIELDS_2024 = """\
Date,"3 Mo","2 Yr","10 Yr"
01/05/2024,5.40,4.00,8.00
01/04/2024,,4.00,7.00
01/03/2024,5.44,4.00,6.00
01/02/2024,5.46,4.00,5.00
"""

# 2025 file: different tenor set ("1.5 Month", "30 Yr"; no "3 Mo") plus a
# July row to pin down the DST offset of available_at.
YIELDS_2025 = """\
Date,"1.5 Month","2 Yr","10 Yr","30 Yr"
07/03/2025,4.46,4.26,4.61,4.81
01/03/2025,4.45,4.25,4.60,4.80
01/02/2025,4.44,4.24,4.59,4.79
"""


class FakeClient:
    """Canned-response HttpGetter; records every call for assertions."""

    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self._responses = list(responses)

    def get(self, url: str, *, params: dict[str, str] | None = None) -> SimpleNamespace:
        self.calls.append((url, dict(params or {})))
        status, text = self._responses.pop(0)
        return SimpleNamespace(status_code=status, text=text)


def make_archive(root: Path, files: dict[str, str]) -> Path:
    archive = root / "raw" / "treasury_yields"
    archive.mkdir(parents=True)
    for name, text in files.items():
        (archive / name).write_text(text)
    return archive


def make_feed(tmp_path: Path, files: dict[str, str], **cfg: object) -> RatesFeed:
    archive = make_archive(tmp_path, files)
    return RatesFeed(archive, tmp_path / "cache", config=RatesConfig(**cfg))


# ----------------------------------------------------------------------
# Treasury archive parsing
# ----------------------------------------------------------------------


def test_load_yields_parses_varying_headers_ascending(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024, "yields_2025.csv": YIELDS_2025})
    frame = feed.load_yields()

    assert list(frame.columns[:4]) == ["asof_date", "available_at", "y_2y", "y_10y"]
    # union of optional tenors across years, deterministic order
    assert list(frame.columns[4:]) == ["y_1_5m", "y_30y", "y_3m"]
    assert list(frame["asof_date"].dt.date.astype(str)) == [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2025-01-02", "2025-01-03", "2025-07-03",
    ]
    assert frame["y_2y"].iloc[0] == pytest.approx(4.00)
    assert frame["y_10y"].iloc[3] == pytest.approx(8.00)
    # empty cell -> NaN; tenor absent from a year's file -> NaN
    assert np.isnan(frame["y_3m"].iloc[2])  # 01/04/2024 empty cell
    assert frame["y_3m"].iloc[4:].isna().all()  # 2025 file has no "3 Mo"
    assert frame["y_30y"].iloc[:4].isna().all()  # 2024 file has no "30 Yr"
    assert frame["y_1_5m"].iloc[5] == pytest.approx(4.45)


def test_available_at_is_same_day_1800_et(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024, "yields_2025.csv": YIELDS_2025})
    frame = feed.load_yields()

    winter = frame["available_at"].iloc[0]
    assert winter == pd.Timestamp("2024-01-02 18:00", tz="America/New_York")
    assert winter.utcoffset() == pd.Timedelta(hours=-5)
    summer = frame["available_at"].iloc[-1]  # 2025-07-03, EDT
    assert summer == pd.Timestamp("2025-07-03 18:00", tz="America/New_York")
    assert summer.utcoffset() == pd.Timedelta(hours=-4)


def test_missing_required_tenor_raises(tmp_path: Path) -> None:
    bad = 'Date,"3 Mo","2 Yr"\n01/02/2024,5.46,4.10\n'
    feed = make_feed(tmp_path, {"yields_2024.csv": bad})
    with pytest.raises(ValueError, match="10 Yr"):
        feed.load_yields()


def test_unrecognized_tenor_header_raises(tmp_path: Path) -> None:
    bad = 'Date,"2 Yr","10 Yr","Foo Bar"\n01/02/2024,4.10,4.40,1.00\n'
    feed = make_feed(tmp_path, {"yields_2024.csv": bad})
    with pytest.raises(ValueError, match="Foo Bar"):
        feed.load_yields()


def test_cache_first_then_rebuild_on_newer_archive(tmp_path: Path) -> None:
    import os

    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    first = feed.load_yields()
    cache = tmp_path / "cache" / "treasury_yields.parquet"
    assert cache.exists()

    # Corrupt the CSV but keep its mtime OLDER than the cache: the parquet
    # must be served without re-parsing (parsing the garbage would raise).
    csv_path = tmp_path / "raw" / "treasury_yields" / "yields_2024.csv"
    csv_path.write_text("garbage,not,a,yields,file\n")
    stale = cache.stat().st_mtime - 100
    os.utime(csv_path, (stale, stale))
    pd.testing.assert_frame_equal(feed.load_yields(), first)

    # A refreshed archive (newer mtime) forces a rebuild that sees new data.
    csv_path.write_text(YIELDS_2024 + "01/08/2024,5.38,4.00,9.00\n")
    fresh = cache.stat().st_mtime + 100
    os.utime(csv_path, (fresh, fresh))
    rebuilt = feed.load_yields()
    assert len(rebuilt) == len(first) + 1
    assert rebuilt["y_10y"].iloc[-1] == pytest.approx(9.00)


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def hy_fixture() -> pd.DataFrame:
    """HY OAS with the real one-business-day availability lag."""
    return pd.DataFrame(
        {
            "asof_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "available_at": [
                pd.Timestamp("2024-01-03 18:00", tz="America/New_York"),
                pd.Timestamp("2024-01-04 18:00", tz="America/New_York"),
                pd.Timestamp("2024-01-05 18:00", tz="America/New_York"),
            ],
            "hy_oas": [4.0, 5.0, 7.0],
        }
    )


def test_features_level_slope_and_z(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024}, slope_z_window=3)
    feats = feed.features()

    assert list(feats["level_2y"]) == pytest.approx([4.0] * 4)
    assert list(feats["slope_10y_2y"]) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    z = feats["slope_z_3d"]
    assert z.iloc[:2].isna().all()  # no full window yet
    # window [1,2,3]: mean 2, std 1 -> z=1; window [2,3,4]: mean 3, std 1 -> z=1
    assert z.iloc[2] == pytest.approx(1.0)
    assert z.iloc[3] == pytest.approx(1.0)
    # no HY data anywhere -> columns omitted entirely
    assert not any(c.startswith("hy_oas") for c in feats.columns)
    # point-in-time columns always present
    assert {"asof_date", "available_at"} <= set(feats.columns)


def test_features_hy_oas_merges_point_in_time(tmp_path: Path) -> None:
    feed = make_feed(
        tmp_path, {"yields_2024.csv": YIELDS_2024}, slope_z_window=3, hy_oas_z_window=2
    )
    feats = feed.features(hy_oas=hy_fixture())

    hy = feats["hy_oas"]
    # 01/02 18:00: no OAS print available yet (first arrives 01/03 18:00)
    assert np.isnan(hy.iloc[0])
    # each row sees the PREVIOUS day's OAS — the real publication lag
    assert hy.iloc[1] == pytest.approx(4.0)
    assert hy.iloc[2] == pytest.approx(5.0)
    assert hy.iloc[3] == pytest.approx(7.0)
    z = feats["hy_oas_z_2d"]
    assert np.isnan(z.iloc[1])  # z of the first OAS print: no full window
    # [4,5]: mean 4.5, std .7071 -> .7071 ; [5,7]: mean 6, std 1.4142 -> .7071
    assert z.iloc[2] == pytest.approx(0.70710678)
    assert z.iloc[3] == pytest.approx(0.70710678)


def test_features_hy_oas_from_fred_cache(tmp_path: Path) -> None:
    feed = make_feed(
        tmp_path, {"yields_2024.csv": YIELDS_2024}, slope_z_window=3, hy_oas_z_window=2
    )
    text = (
        f"observation_date,{HY_OAS_SERIES}\n"
        "2024-01-02,4.0\n2024-01-03,5.0\n2024-01-04,7.0\n"
    )
    feed.fetch_fred(HY_OAS_SERIES, client=FakeClient([(200, text)]))

    feats = feed.features()  # no frame passed: found in the FRED cache
    assert feats["hy_oas"].iloc[3] == pytest.approx(7.0)
    assert feats["hy_oas_z_2d"].iloc[3] == pytest.approx(0.70710678)


def test_features_rejects_malformed_hy_frame(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    with pytest.raises(ValueError, match="available_at"):
        feed.features(hy_oas=pd.DataFrame({"asof_date": [], "hy_oas": []}))


# ----------------------------------------------------------------------
# FRED fredgraph fallback (fake client only — no network)
# ----------------------------------------------------------------------


def test_fetch_fred_parses_and_stamps_next_business_day(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    text = "DATE,DGS2\n2024-01-04,4.5\n2024-01-05,4.6\n2024-01-06,.\n"
    client = FakeClient([(200, text)])
    frame = feed.fetch_fred("DGS2", client=client)

    url, params = client.calls[0]
    assert url == "https://fred.stlouisfed.org/graph/fredgraph.csv"
    assert params == {"id": "DGS2"}  # first fetch: full history, no cosd
    assert len(frame) == 2  # the "." row carries no data and is dropped
    assert list(frame.columns) == ["asof_date", "available_at", "DGS2"]
    # Thu 01/04 -> Fri 01/05 18:00 ET; Fri 01/05 -> Mon 01/08 18:00 ET
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-05 18:00", tz="America/New_York"
    )
    assert frame["available_at"].iloc[1] == pd.Timestamp(
        "2024-01-08 18:00", tz="America/New_York"
    )
    assert (tmp_path / "cache" / "fred_DGS2.parquet").exists()


def test_fetch_fred_incremental_merge_and_revision(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    first = "DATE,DGS2\n2024-01-04,4.5\n2024-01-05,4.6\n"
    second = "observation_date,DGS2\n2024-01-05,4.7\n2024-01-08,4.8\n"
    client = FakeClient([(200, first), (200, second)])

    feed.fetch_fred("DGS2", client=client)
    merged = feed.fetch_fred("DGS2", client=client)

    _, params = client.calls[1]
    assert params["cosd"] == "2024-01-05"  # incremental: from last cached date
    assert len(merged) == 3
    revised = merged.loc[merged["asof_date"] == pd.Timestamp("2024-01-05"), "DGS2"]
    assert revised.iloc[0] == pytest.approx(4.7)  # re-fetched value wins


def test_fetch_fred_weekend_asof_available_monday(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    text = "DATE,DFF\n2024-01-06,5.33\n"  # Saturday print (fed funds)
    frame = feed.fetch_fred("DFF", client=FakeClient([(200, text)]))
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-08 18:00", tz="America/New_York"
    )


def test_fetch_fred_failure_raises_and_never_caches(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    with pytest.raises(RuntimeError, match="HTTP 503"):
        feed.fetch_fred("VIXCLS", client=FakeClient([(503, "down")]))
    assert not (tmp_path / "cache" / "fred_VIXCLS.parquet").exists()


def test_fetch_fred_unknown_series_rejected(tmp_path: Path) -> None:
    feed = make_feed(tmp_path, {"yields_2024.csv": YIELDS_2024})
    with pytest.raises(ValueError, match="SPX"):
        feed.fetch_fred("SPX")
    assert "BAMLH0A0HYM2" in FRED_SERIES  # the vetted set carries the OAS series

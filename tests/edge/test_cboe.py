"""Tests for the CBOE indices + put/call point-in-time adapter.

All fixtures are canned files written into tmp_path — no network, no reads
from the real data_cache archive. Every frame is checked against the
point-in-time convention: asof_date + tz-aware America/New_York
available_at stamped 18:00 ET on the data's own trading date.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.feeds.cboe_indices import CboeConfig, CboeIndices

ET = ZoneInfo("America/New_York")

# Mon..Fri, well before the lockbox wall and before the 2020-03-08 DST jump.
WEEK = ["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06"]

FROZEN_PREAMBLE = (
    "Volume and Put/Call Ratio data is compiled for the convenience of site "
    "visitors and is furnished without responsibility for accuracy.,,,,\n"
    ", PRODUCT: TOTAL,,EXCHANGE: Cboe,\n"
)


def _mdY(iso: str) -> str:
    ts = pd.Timestamp(iso)
    return f"{ts.month:02d}/{ts.day:02d}/{ts.year}"


def write_ohlc_csv(dir_: Path, name: str, rows: list[tuple[str, float, float, float, float]]) -> None:
    lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    for iso, o, h, low, c in rows:
        lines.append(f"{_mdY(iso)},{o:.6f},{h:.6f},{low:.6f},{c:.6f}")
    (dir_ / f"{name}_History.csv").write_text("\n".join(lines) + "\n")


def write_close_csv(dir_: Path, name: str, rows: list[tuple[str, float]]) -> None:
    lines = [f"DATE,{name}"]
    for iso, c in rows:
        lines.append(f"{_mdY(iso)},{c:.6f}")
    (dir_ / f"{name}_History.csv").write_text("\n".join(lines) + "\n")


def write_frozen_pc(dir_: Path, rows: list[tuple[str, int, int, int, float]]) -> None:
    lines = [FROZEN_PREAMBLE + "DATE,CALLS,PUTS,TOTAL,P/C Ratio"]
    for iso, calls, puts, total, ratio in rows:
        ts = pd.Timestamp(iso)
        # real file: non-zero-padded dates and leading spaces before numbers
        lines.append(f"{ts.month}/{ts.day}/{ts.year}, {calls}, {puts}, {total}, {ratio}")
    (dir_ / "totalpc_2006_2019.csv").write_text("\n".join(lines) + "\n")


def write_daily_json(
    daily_dir: Path,
    iso: str,
    total: float,
    equity: float,
    calls: int,
    puts: int,
    index: float = 1.05,
    etp: float = 1.89,
) -> None:
    daily_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ratios": [
            {"name": "TOTAL PUT/CALL RATIO", "value": f"{total}"},
            {"name": "INDEX PUT/CALL RATIO", "value": f"{index}"},
            {"name": "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO", "value": f"{etp}"},
            {"name": "EQUITY PUT/CALL RATIO", "value": f"{equity}"},
            {"name": "OEX PUT/CALL RATIO", "value": "3.49"},
        ],
        "SUM OF ALL PRODUCTS": [
            {"name": "VOLUME", "call": calls, "put": puts, "total": calls + puts},
            {"name": "OPEN INTEREST", "call": 1, "put": 1, "total": 2},
        ],
    }
    import json

    (daily_dir / f"{iso}.json").write_text(json.dumps(payload))


def make_adapter(tmp_path: Path, config: CboeConfig | None = None) -> CboeIndices:
    """Standard 5-day scenario: all six indices + frozen/daily put/call with
    a one-day overlap (2020-03-04) where the two archives disagree."""
    raw_idx = tmp_path / "raw_idx"
    raw_pc = tmp_path / "raw_pc"
    raw_idx.mkdir(exist_ok=True)
    raw_pc.mkdir(exist_ok=True)

    write_ohlc_csv(raw_idx, "VIX", [(d, 19.0, 21.0, 18.5, 20.0) for d in WEEK])
    write_ohlc_csv(raw_idx, "VIX9D", [(d, 17.5, 18.5, 17.0, 18.0) for d in WEEK])
    write_ohlc_csv(raw_idx, "VIX3M", [(d, 24.5, 25.5, 24.0, 25.0) for d in WEEK])
    write_ohlc_csv(raw_idx, "VIX6M", [(d, 29.5, 30.5, 29.0, 30.0) for d in WEEK])
    write_close_csv(raw_idx, "VVIX", [(d, 100.0) for d in WEEK])
    write_close_csv(raw_idx, "SKEW", [(d, 130.0) for d in WEEK])

    write_frozen_pc(
        raw_pc,
        [
            ("2020-03-02", 1000, 910, 1910, 0.91),
            ("2020-03-03", 1000, 920, 1920, 0.92),
            ("2020-03-04", 1000, 950, 1950, 0.95),  # loses to daily on overlap
        ],
    )
    daily = raw_pc / "daily"
    write_daily_json(daily, "2020-03-04", total=0.99, equity=0.5, calls=200, puts=198)
    write_daily_json(daily, "2020-03-05", total=1.00, equity=0.6, calls=210, puts=210)
    write_daily_json(daily, "2020-03-06", total=1.10, equity=0.7, calls=220, puts=242)

    return CboeIndices(raw_idx, raw_pc, tmp_path / "cache", config=config)


def row(frame: pd.DataFrame, iso: str) -> pd.Series:
    hits = frame.loc[frame["asof_date"] == pd.Timestamp(iso)]
    assert len(hits) == 1, f"expected exactly one row for {iso}, got {len(hits)}"
    return hits.iloc[0]


# ----------------------------------------------------------------------
# Index history parsing
# ----------------------------------------------------------------------


def test_ohlc_parse_and_backfilled_flag(tmp_path: Path) -> None:
    raw_idx = tmp_path / "raw_idx"
    raw_idx.mkdir()
    write_ohlc_csv(
        raw_idx,
        "VIX",
        [
            ("1990-01-02", 17.24, 17.24, 17.24, 17.24),  # backfilled close
            ("1990-01-03", 18.19, 18.19, 18.19, 18.19),  # backfilled close
            ("2020-03-02", 19.0, 21.0, 18.5, 20.0),
        ],
    )
    adapter = CboeIndices(raw_idx, tmp_path / "raw_pc", tmp_path / "cache")
    frame = adapter.index_history("VIX")

    assert list(frame["asof_date"]) == [
        pd.Timestamp("1990-01-02"),
        pd.Timestamp("1990-01-03"),
        pd.Timestamp("2020-03-02"),
    ]
    assert list(frame["backfilled"]) == [True, True, False]
    assert not frame["winsorized"].any()  # VIX is not winsorized by default
    r = row(frame, "2020-03-02")
    assert (r["open"], r["high"], r["low"], r["close"]) == (19.0, 21.0, 18.5, 20.0)


def test_close_only_bad_print_winsorized_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 10 consecutive weekdays; index 3 is an impossible print (real-archive
    # analogue: VVIX 15.71 on 2006-03-15).
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-06", periods=10)]
    values = [90.0, 91.0, 92.0, 15.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0]
    raw_idx = tmp_path / "raw_idx"
    raw_idx.mkdir()
    write_close_csv(raw_idx, "VVIX", list(zip(dates, values)))
    adapter = CboeIndices(raw_idx, tmp_path / "raw_pc", tmp_path / "cache")

    with caplog.at_level(logging.WARNING):
        frame = adapter.index_history("VVIX")

    bad = row(frame, dates[3])
    assert bad["winsorized"]
    # trailing median of the PRIOR closes [90, 91, 92] (3 obs >= min_periods=3)
    # is 91 — the bad print is replaced using only data from before it
    assert bad["close"] == pytest.approx(91.0)
    # everything else untouched and unflagged (rows 0-2 have fewer than 3
    # prior closes so they are never judged; kept as-is, never dropped)
    good = frame.loc[frame["asof_date"] != pd.Timestamp(dates[3])]
    assert not good["winsorized"].any()
    assert list(good["close"]) == values[:3] + values[4:]
    assert len(frame) == 10  # nothing silently dropped
    assert any("VVIX" in r.message and "winsorized" in r.message for r in caplog.records)
    # close-only series: no OHLC, never marked backfilled
    assert frame["open"].isna().all()
    assert not frame["backfilled"].any()


def test_newest_row_bad_print_judged_live(tmp_path: Path) -> None:
    """The trailing window judges the NEWEST row (a centered window never
    could — its window needs t+1/t+2), and the cleaned value at t depends
    only on closes <= t-1: truncating the file after t leaves it unchanged."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-06", periods=6)]
    values = [90.0, 91.0, 92.0, 93.0, 94.0, 15.0]  # bad print is the LAST row
    raw_idx = tmp_path / "raw_idx"
    raw_idx.mkdir()
    write_close_csv(raw_idx, "VVIX", list(zip(dates, values)))
    adapter = CboeIndices(raw_idx, tmp_path / "raw_pc", tmp_path / "cache")
    frame = adapter.index_history("VVIX")

    last = row(frame, dates[5])
    assert last["winsorized"]
    # trailing 5-day median of the prior closes [90, 91, 92, 93, 94] is 92
    assert last["close"] == pytest.approx(92.0)
    assert not frame.loc[frame["asof_date"] != pd.Timestamp(dates[5]), "winsorized"].any()

    # No-lookahead property, directly: drop the last row from the raw file
    # and every earlier cleaned close is bit-identical.
    raw_idx2 = tmp_path / "raw_idx2"
    raw_idx2.mkdir()
    write_close_csv(raw_idx2, "VVIX", list(zip(dates[:5], values[:5])))
    truncated = CboeIndices(raw_idx2, tmp_path / "raw_pc", tmp_path / "cache2").index_history(
        "VVIX"
    )
    pd.testing.assert_frame_equal(truncated, frame.iloc[:5].reset_index(drop=True))


def test_unknown_index_raises(tmp_path: Path) -> None:
    adapter = CboeIndices(tmp_path, tmp_path, tmp_path / "cache")
    with pytest.raises(KeyError):
        adapter.index_history("VIX1Y")


# ----------------------------------------------------------------------
# Put/call parsing + merge
# ----------------------------------------------------------------------


def test_frozen_pc_parse(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    pc = adapter.put_call()
    r = row(pc, "2020-03-02")
    assert r["source"] == "frozen"
    assert (r["calls"], r["puts"], r["total"]) == (1000, 910, 1910)
    assert r["total_pc"] == pytest.approx(0.91)
    # frozen archive is total-exchange only: segment ratios are NaN
    assert pd.isna(r["equity_pc"]) and pd.isna(r["index_pc"]) and pd.isna(r["etp_pc"])


def test_daily_json_parse_and_overlap_dedup(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    pc = adapter.put_call()

    # continuous series, one row per date, ascending
    assert [d.strftime("%Y-%m-%d") for d in pc["asof_date"]] == WEEK
    assert pc["asof_date"].is_monotonic_increasing

    # overlap date exists in both archives -> daily JSON wins
    overlap = row(pc, "2020-03-04")
    assert overlap["source"] == "daily"
    assert overlap["total_pc"] == pytest.approx(0.99)
    assert overlap["equity_pc"] == pytest.approx(0.5)
    assert (overlap["calls"], overlap["puts"], overlap["total"]) == (200, 198, 398)
    assert overlap["index_pc"] == pytest.approx(1.05)
    assert overlap["etp_pc"] == pytest.approx(1.89)


def test_daily_json_float_volumes_parse_exactly(tmp_path: Path) -> None:
    """Volumes arriving as floats or numeric strings ('398.0') must parse to
    the exact integer counts — int() would truncate fractional numbers
    silently and reject decimal strings outright (dropping the whole day)."""
    import json

    raw_pc = tmp_path / "raw_pc"
    raw_pc.mkdir()
    write_frozen_pc(raw_pc, [("2020-03-02", 1000, 910, 1910, 0.91)])
    daily = raw_pc / "daily"
    daily.mkdir()
    payload = {
        "ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": "0.99"}],
        "SUM OF ALL PRODUCTS": [
            {"name": "VOLUME", "call": 200.0, "put": "198.0", "total": 398.0},
        ],
    }
    (daily / "2020-03-04.json").write_text(json.dumps(payload))
    adapter = CboeIndices(tmp_path / "raw_idx", raw_pc, tmp_path / "cache")
    pc = adapter.put_call()

    r = row(pc, "2020-03-04")
    assert r["source"] == "daily"  # NOT skipped as unparseable
    assert (r["calls"], r["puts"], r["total"]) == (200, 198, 398)
    assert str(pc["calls"].dtype) == "Int64"


def test_corrupt_or_misnamed_daily_files_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = make_adapter(tmp_path)
    daily = tmp_path / "raw_pc" / "daily"
    (daily / "2020-03-09.json").write_text("{not json at all")
    (daily / "not-a-date.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        pc = adapter.put_call(refresh=True)
    assert [d.strftime("%Y-%m-%d") for d in pc["asof_date"]] == WEEK  # bad files skipped
    skipped = [r for r in caplog.records if "skipping unparseable" in r.message]
    assert len(skipped) == 2


def test_missing_daily_dir_serves_frozen_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_pc = tmp_path / "raw_pc"
    raw_pc.mkdir()
    write_frozen_pc(raw_pc, [("2006-11-01", 1401036, 1271445, 2672481, 0.91)])
    adapter = CboeIndices(tmp_path / "raw_idx", raw_pc, tmp_path / "cache")
    with caplog.at_level(logging.WARNING):
        pc = adapter.put_call()
    assert len(pc) == 1
    assert row(pc, "2006-11-01")["total_pc"] == pytest.approx(0.91)
    assert any("daily put/call dir missing" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# Point-in-time stamps
# ----------------------------------------------------------------------


def test_available_at_is_same_day_1800_et(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    for frame in (adapter.index_history("VIX"), adapter.put_call(), adapter.features()):
        assert str(frame["available_at"].dt.tz) == "America/New_York"
        # exactly 18:00 ET wall clock on the data's own date
        assert (frame["available_at"].dt.hour == 18).all()
        assert (frame["available_at"].dt.normalize().dt.tz_localize(None) == frame["asof_date"]).all()

    r = row(adapter.put_call(), "2020-03-02")  # EST date
    assert r["available_at"].utcoffset() == timedelta(hours=-5)
    # after the same-day close, before the next session's open: the value is
    # first usable at the NEXT session under available_at <= decision_ts
    assert r["available_at"] > pd.Timestamp("2020-03-02 16:15", tz=ET)
    assert r["available_at"] < pd.Timestamp("2020-03-03 09:30", tz=ET)


def test_available_at_correct_across_dst(tmp_path: Path) -> None:
    raw_pc = tmp_path / "raw_pc"
    raw_pc.mkdir()
    write_frozen_pc(raw_pc, [("2006-11-01", 100, 91, 191, 0.91)])  # EST
    write_daily_json(raw_pc / "daily", "2020-07-06", total=0.8, equity=0.5, calls=10, puts=8)  # EDT
    adapter = CboeIndices(tmp_path / "raw_idx", raw_pc, tmp_path / "cache")
    pc = adapter.put_call()
    assert row(pc, "2006-11-01")["available_at"].utcoffset() == timedelta(hours=-5)
    summer = row(pc, "2020-07-06")["available_at"]
    assert summer.utcoffset() == timedelta(hours=-4)
    assert summer.hour == 18  # 18:00 WALL time in both DST regimes


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def test_feature_term_structure_ratios(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    feats = adapter.features()
    r = row(feats, "2020-03-02")
    assert r["vix9d_vix"] == pytest.approx(18.0 / 20.0)  # 0.9
    assert r["vix_vix3m"] == pytest.approx(20.0 / 25.0)  # 0.8
    assert r["vix3m_vix6m"] == pytest.approx(25.0 / 30.0)
    assert r["vvix_vix"] == pytest.approx(100.0 / 20.0)  # 5.0
    assert r["skew"] == pytest.approx(130.0)
    assert r["total_pc"] == pytest.approx(0.91)


def test_feature_pc_zscores(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path, config=CboeConfig(pc_z_window=3))
    feats = adapter.features()

    # total_pc merged series: 0.91, 0.92, 0.99, 1.00, 1.10
    # window-3 at 03-06: mean(0.99, 1.00, 1.10) = 1.03, std(ddof=1) = sqrt(0.0037)
    assert row(feats, "2020-03-06")["total_pc_z21"] == pytest.approx(0.07 / math.sqrt(0.0037))
    # equity_pc: NaN, NaN, 0.5, 0.6, 0.7 -> window only fills on the last day
    assert row(feats, "2020-03-06")["equity_pc_z21"] == pytest.approx(1.0)
    assert pd.isna(row(feats, "2020-03-05")["equity_pc_z21"])
    # z is NaN until the window is full of valid observations
    assert pd.isna(row(feats, "2020-03-02")["total_pc_z21"])
    assert pd.isna(row(feats, "2020-03-03")["total_pc_z21"])
    assert not pd.isna(row(feats, "2020-03-04")["total_pc_z21"])


def test_default_config_values() -> None:
    cfg = CboeConfig()
    assert cfg.pc_z_window == 21
    assert cfg.publish_hour_et == 18
    assert cfg.winsorize_series == ("VVIX", "SKEW")
    assert cfg.winsor_window == 5
    assert cfg.winsor_min_periods == 3  # prior observations, trailing window


# ----------------------------------------------------------------------
# Cache behavior
# ----------------------------------------------------------------------


def test_cache_first_then_refresh_reparses(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    first = adapter.put_call()
    assert (tmp_path / "cache" / "put_call.parquet").exists()
    assert row(first, "2020-03-02")["total_pc"] == pytest.approx(0.91)

    # raw archive changes AFTER the cache was written
    write_frozen_pc(tmp_path / "raw_pc", [("2020-03-02", 1000, 500, 1500, 0.50)])

    cached = adapter.put_call()  # cache-first: still the original parse
    assert row(cached, "2020-03-02")["total_pc"] == pytest.approx(0.91)
    # available_at survives the parquet round-trip tz-aware
    assert str(cached["available_at"].dt.tz) == "America/New_York"

    refreshed = adapter.put_call(refresh=True)  # forced re-parse sees the change
    assert row(refreshed, "2020-03-02")["total_pc"] == pytest.approx(0.50)
    assert len(refreshed) == 4  # 03-03/03-04 frozen rows gone; daily rows remain


def test_features_cached(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    first = adapter.features()
    assert (tmp_path / "cache" / "features.parquet").exists()
    again = adapter.features()
    pd.testing.assert_frame_equal(first, again)

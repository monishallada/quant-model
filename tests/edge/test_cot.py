"""CFTC TFF (Traders in Financial Futures) adapter: parsing, code-pinning,
feature math, and — the trap this source exists to test — the Tuesday-asof /
Friday-release availability rule.

Fixtures are hand-crafted yearly zips in tmp_path (no COT archive is on disk
yet and NO network is ever touched). The final 2024-12-31 E-mini S&P row uses
the probe-verified values: Dealer L146816/S957802, AssetMgr L1157524/S173461,
LevMoney L151407/S476014 (Other_Rept, spreads, and OI are plausible filler —
the probe verified only those three classes). All dates predate the
2026-02-22 lockbox start.
"""

from __future__ import annotations

import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.feeds.cot import (
    PINNED_MARKETS,
    TRADER_CLASSES,
    CotTff,
    CotTffConfig,
    tff_available_at,
)

ET = ZoneInfo("America/New_York")

# Last report week in the fixture: 2024-12-31 is a TUESDAY; the Friday of that
# same week is 2025-01-03 (crosses the year boundary on purpose).
LAST_ASOF = date(2024, 12, 31)
N_WEEKS = 60  # weekly Tuesdays 2023-11-14 .. 2024-12-31, spanning two yearly files

ES_CODE = "13874A"
NQ_CODE = "209742"


class _WeekValues(TypedDict):
    """One market's fixture row: OI plus (long, short, spread) per class."""

    oi: int
    dealer: tuple[int, int, int]
    asset_mgr: tuple[int, int, int]
    lev_money: tuple[int, int, int]
    other_rept: tuple[int, int, int]


# Probe-verified 2024-12-31 E-mini S&P 500 row (longs/shorts); OI/other filler.
ES_FINAL: _WeekValues = {
    "oi": 2_236_745,
    "dealer": (146_816, 957_802, 120_000),
    "asset_mgr": (1_157_524, 173_461, 90_000),
    "lev_money": (151_407, 476_014, 160_000),
    "other_rept": (231_122, 98_255, 40_000),
}

# Real files quote the market name; codes are sometimes whitespace-padded.
HEADER = (
    '"Market_and_Exchange_Names",As_of_Date_In_Form_YYMMDD,Report_Date_as_YYYY-MM-DD,'
    "CFTC_Contract_Market_Code,CFTC_Market_Code,Open_Interest_All,"
    "Dealer_Positions_Long_All,Dealer_Positions_Short_All,Dealer_Positions_Spread_All,"
    "Asset_Mgr_Positions_Long_All,Asset_Mgr_Positions_Short_All,Asset_Mgr_Positions_Spread_All,"
    "Lev_Money_Positions_Long_All,Lev_Money_Positions_Short_All,Lev_Money_Positions_Spread_All,"
    "Other_Rept_Positions_Long_All,Other_Rept_Positions_Short_All,Other_Rept_Positions_Spread_All,"
    "Tot_Rept_Positions_Long_All,Tot_Rept_Positions_Short_All"
)


def _asof(i: int) -> date:
    """Report Tuesday for week index i in [0, N_WEEKS)."""
    return LAST_ASOF - timedelta(weeks=N_WEEKS - 1 - i)


def _es_values(i: int) -> _WeekValues:
    if i == N_WEEKS - 1:
        return ES_FINAL
    return {
        "oi": 2_000_000 + 997 * i,
        "dealer": (140_000 + 137 * i + 251 * (i % 7), 900_000 + 911 * i, 118_000 + 13 * i),
        "asset_mgr": (1_100_000 + 953 * i, 170_000 + 71 * i + 173 * (i % 5), 88_000 + 17 * i),
        "lev_money": (150_000 + 89 * i + 307 * (i % 3), 470_000 + 101 * i, 155_000 + 11 * i),
        "other_rept": (220_000 + 61 * i, 95_000 + 43 * i + 97 * (i % 4), 39_000 + 7 * i),
    }


def _nq_values(i: int) -> _WeekValues:
    return {
        "oi": 250_000 + 211 * i,
        "dealer": (30_000 + 53 * i + 89 * (i % 6), 80_000 + 67 * i, 12_000 + 3 * i),
        "asset_mgr": (110_000 + 97 * i, 25_000 + 31 * i + 41 * (i % 5), 9_000 + 2 * i),
        "lev_money": (28_000 + 47 * i, 60_000 + 59 * i + 73 * (i % 3), 15_000 + 5 * i),
        "other_rept": (22_000 + 29 * i, 11_000 + 19 * i, 4_000 + i),
    }


def _row(name: str, asof: date, code: str, values: _WeekValues) -> str:
    cells = [f'"{name}"', asof.strftime("%y%m%d"), asof.isoformat(), code, "CME", str(values["oi"])]
    groups = (values["dealer"], values["asset_mgr"], values["lev_money"], values["other_rept"])
    for group in groups:
        cells.extend(str(v) for v in group)
    cells.extend(["1500000", "1500000"])  # Tot_Rept filler (unused columns)
    return ",".join(cells)


def _finfut_text(year: int) -> str:
    """One yearly FinFutYY.txt: pinned markets + decoys, with name drift and
    whitespace-padded codes in the 2023 file."""
    pad = "  " if year == 2023 else ""
    es_name = (
        "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"
        if year == 2023
        else "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"
    )
    nq_name = (
        "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE"
        if year == 2023
        else "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE"
    )
    lines = [HEADER]
    for i in range(N_WEEKS):
        asof = _asof(i)
        if asof.year != year:
            continue
        lines.append(_row(es_name, asof, ES_CODE + pad, _es_values(i)))
        lines.append(_row(nq_name, asof, NQ_CODE + pad, _nq_values(i)))
        # Decoy 1: code differs from 13874A by one char (S&P 500 Consolidated).
        lines.append(
            _row(
                "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE", asof, "138741", _nq_values(i)
            )
        )
        # Decoy 2: display name IDENTICAL to the pinned ES name, wrong code —
        # a name-based matcher would wrongly include this row.
        lines.append(_row(es_name, asof, "099999", _nq_values(i)))
    return "\n".join(lines) + "\n"


def _write_archive(raw_dir: Path, years: tuple[int, ...] = (2023, 2024)) -> None:
    """Yearly zips named like the CFTC's, each containing 'FinFutYY.txt'
    (the member is literally named that for every year, as in the real zips)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        with zipfile.ZipFile(raw_dir / f"fut_fin_txt_{year}.zip", "w") as zf:
            zf.writestr("FinFutYY.txt", _finfut_text(year))


@pytest.fixture()
def adapter(tmp_path: Path) -> CotTff:
    raw = tmp_path / "raw" / "cot"
    _write_archive(raw)
    return CotTff(raw, tmp_path / "cache" / "cot")


# ---------------------------------------------------------------------------
# Parsing + code pinning
# ---------------------------------------------------------------------------


def test_parses_verified_2024_row(adapter: CotTff) -> None:
    """The probe-verified 2024-12-31 ES row round-trips exactly."""
    frame = adapter.features()
    row = frame[(frame.market == "ES") & (frame.asof_date == pd.Timestamp(LAST_ASOF))]
    assert len(row) == 1
    row = row.iloc[0]
    assert row.market_code == ES_CODE
    assert row.open_interest == ES_FINAL["oi"]
    assert (row.dealer_long, row.dealer_short) == (146_816, 957_802)
    assert (row.asset_mgr_long, row.asset_mgr_short) == (1_157_524, 173_461)
    assert (row.lev_money_long, row.lev_money_short) == (151_407, 476_014)
    assert row.dealer_spread == ES_FINAL["dealer"][2]


def test_filters_on_code_not_display_name(adapter: CotTff) -> None:
    """Only the two pinned codes survive: the near-miss code 138741 and the
    decoy whose NAME equals the pinned ES name are both excluded, while the
    drifted 2023 vs 2024 display names (and padded codes) are both included."""
    frame = adapter.features()
    assert set(frame.market_code.unique()) == set(PINNED_MARKETS)
    assert set(frame.market.unique()) == {"ES", "NQ"}
    es_names = set(frame.loc[frame.market == "ES", "market_name"].unique())
    assert es_names == {
        "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
        "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
    }
    # 60 weeks per market, across both yearly files, no duplicates.
    assert len(frame) == 2 * N_WEEKS
    counts = frame.groupby("market_code").size()
    assert (counts == N_WEEKS).all()


def test_multi_year_concat_sorted(adapter: CotTff) -> None:
    frame = adapter.features()
    assert frame.asof_date.is_monotonic_increasing
    assert frame.asof_date.min() == pd.Timestamp(_asof(0))
    assert frame.asof_date.max() == pd.Timestamp(LAST_ASOF)


def test_missing_archive_raises(tmp_path: Path) -> None:
    empty = CotTff(tmp_path / "nowhere", tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        empty.features()


def test_fractional_count_value_raises_never_truncates(tmp_path: Path) -> None:
    """A fractional cell in a count column ('2236745.5') is corrupt data —
    it must raise with the column named, never silently truncate to int."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    text = _finfut_text(2024).replace(str(ES_FINAL["oi"]), f"{ES_FINAL['oi']}.5")
    assert ".5" in text  # the corruption actually landed
    with zipfile.ZipFile(raw / "fut_fin_txt_2024.zip", "w") as zf:
        zf.writestr("FinFutYY.txt", text)
    with pytest.raises(ValueError, match="Open_Interest_All.*non-integral"):
        CotTff(raw, tmp_path / "cache").features()


def test_legacy_2010_header_name_parses_identically(tmp_path: Path) -> None:
    """The REAL 2010-2012 yearly archives name the report-date column
    ``Report_Date_as_MM_DD_YYYY`` while its values are ISO YYYY-MM-DD; the
    parser must accept the legacy header and produce the identical frame."""
    modern_raw = tmp_path / "modern"
    _write_archive(modern_raw)
    modern = CotTff(modern_raw, tmp_path / "cache_modern").features()

    legacy_raw = tmp_path / "legacy"
    legacy_raw.mkdir()
    for year in (2023, 2024):
        text = _finfut_text(year).replace(
            "Report_Date_as_YYYY-MM-DD", "Report_Date_as_MM_DD_YYYY", 1
        )
        with zipfile.ZipFile(legacy_raw / f"fut_fin_txt_{year}.zip", "w") as zf:
            zf.writestr("FinFutYY.txt", text)
    legacy = CotTff(legacy_raw, tmp_path / "cache_legacy").features()
    pd.testing.assert_frame_equal(legacy, modern)


def test_legacy_header_with_actual_mm_dd_yyyy_values_raises(tmp_path: Path) -> None:
    """The legacy header is only an alias — the strict %Y-%m-%d value parse
    stays, so a file whose date VALUES really are MM/DD/YYYY fails loudly
    instead of being misread."""
    raw = tmp_path / "raw"
    raw.mkdir()
    text = _finfut_text(2024).replace("Report_Date_as_YYYY-MM-DD", "Report_Date_as_MM_DD_YYYY", 1)
    header, *rows = text.strip().split("\n")

    def flip(row: str) -> str:
        cells = row.split(",")
        y, m, d = cells[2].split("-")
        cells[2] = f"{m}/{d}/{y}"
        return ",".join(cells)

    with zipfile.ZipFile(raw / "fut_fin_txt_2024.zip", "w") as zf:
        zf.writestr("FinFutYY.txt", "\n".join([header, *[flip(r) for r in rows]]) + "\n")
    with pytest.raises(ValueError):
        CotTff(raw, tmp_path / "cache").features()


# ---------------------------------------------------------------------------
# available_at: THE TRAP — Tuesday report, Friday 16:00 ET release
# ---------------------------------------------------------------------------


def test_available_at_is_friday_1600_et(adapter: CotTff) -> None:
    """asof Tuesday 2024-12-31 -> available Friday 2025-01-03 16:00 ET
    (crossing the year boundary), and the column is tz-aware ET."""
    frame = adapter.features()
    assert str(frame.available_at.dtype) == "datetime64[ns, America/New_York]"
    row = frame[(frame.market == "ES") & (frame.asof_date == pd.Timestamp(LAST_ASOF))].iloc[0]
    assert row.available_at == pd.Timestamp("2025-01-03 16:00", tz=ET)
    # Every report in the frame: available on the Friday >= asof at 16:00 ET —
    # except the one holiday-shortened week in the fixture span (Good Friday
    # 2024-03-29), which rolls to the next NYSE trading day, Monday 2024-04-01.
    ats = frame.available_at
    good_friday_week = frame.asof_date == pd.Timestamp("2024-03-26")
    assert good_friday_week.any()
    assert (ats[~good_friday_week].dt.dayofweek == 4).all()
    assert (ats[good_friday_week] == pd.Timestamp("2024-04-01 16:00", tz=ET)).all()
    assert (ats.dt.hour == 16).all() and (ats.dt.minute == 0).all()
    assert ((ats.dt.tz_localize(None) - frame.asof_date) >= pd.Timedelta(0)).all()


def test_tuesday_report_not_available_on_wednesday(adapter: CotTff) -> None:
    """THE trap: the 2024-12-31 (Tuesday) report must NOT be visible on
    Wednesday — or Thursday, or Friday before 16:00 ET. A live trader first
    sees it Friday 16:00 ET."""
    frame = adapter.features()
    report_asof = pd.Timestamp(LAST_ASOF)

    def visible(decision_ts: pd.Timestamp) -> bool:
        joined = frame[frame.available_at <= decision_ts]
        return bool((joined.asof_date == report_asof).any())

    wednesday_open = pd.Timestamp("2025-01-01 09:30", tz=ET)
    wednesday_midnight = pd.Timestamp("2025-01-01 23:59", tz=ET)
    thursday_close = pd.Timestamp("2025-01-02 16:00", tz=ET)
    friday_1559 = pd.Timestamp("2025-01-03 15:59", tz=ET)
    friday_1600 = pd.Timestamp("2025-01-03 16:00", tz=ET)
    monday = pd.Timestamp("2025-01-06 09:30", tz=ET)

    assert not visible(wednesday_open)
    assert not visible(wednesday_midnight)
    assert not visible(thursday_close)
    assert not visible(friday_1559)
    assert visible(friday_1600)  # inclusive boundary: <= keys the join
    assert visible(monday)


def test_available_at_helper_schedule_rules() -> None:
    # Ordinary mid-year Tuesday -> same-week Friday.
    assert tff_available_at(date(2024, 6, 4)) == pd.Timestamp("2024-06-07 16:00", tz=ET)
    # Friday asof (hypothetical) -> that same day.
    assert tff_available_at(date(2024, 6, 7)) == pd.Timestamp("2024-06-07 16:00", tz=ET)
    # Weekend asof (never observed) -> NEXT Friday: later, hence conservative.
    assert tff_available_at(date(2024, 12, 28)) == pd.Timestamp("2025-01-03 16:00", tz=ET)
    # Configurable release wall-clock.
    assert tff_available_at(date(2024, 6, 4), hour=17, minute=30) == pd.Timestamp(
        "2024-06-07 17:30", tz=ET
    )


def test_available_at_rolls_past_good_friday_2024() -> None:
    """asof Tue 2024-03-26 -> same-week Friday 2024-03-29 is Good Friday (NYSE
    holiday). The CFTC actually released Monday 2024-04-01 ~15:30 ET, so a
    Friday stamp would be ~1 day of lookahead — the stamp must roll forward
    to the next NYSE trading day at 16:00 ET."""
    assert tff_available_at(date(2024, 3, 26)) == pd.Timestamp("2024-04-01 16:00", tz=ET)
    # The roll keeps the configured wall-clock time.
    assert tff_available_at(date(2024, 3, 26), hour=17, minute=30) == pd.Timestamp(
        "2024-04-01 17:30", tz=ET
    )
    # The surrounding ordinary weeks are untouched.
    assert tff_available_at(date(2024, 3, 19)) == pd.Timestamp("2024-03-22 16:00", tz=ET)
    assert tff_available_at(date(2024, 4, 2)) == pd.Timestamp("2024-04-05 16:00", tz=ET)


# ---------------------------------------------------------------------------
# Features: net/OI and trailing 52-week z
# ---------------------------------------------------------------------------


def test_net_position_per_oi(adapter: CotTff) -> None:
    frame = adapter.features()
    row = frame[(frame.market == "ES") & (frame.asof_date == pd.Timestamp(LAST_ASOF))].iloc[0]
    oi = ES_FINAL["oi"]
    assert row.dealer_net_oi == pytest.approx((146_816 - 957_802) / oi)
    assert row.asset_mgr_net_oi == pytest.approx((1_157_524 - 173_461) / oi)
    assert row.lev_money_net_oi == pytest.approx((151_407 - 476_014) / oi)
    lo, sh, _ = ES_FINAL["other_rept"]
    assert row.other_rept_net_oi == pytest.approx((lo - sh) / oi)


def test_z_matches_trailing_52_week_window(adapter: CotTff) -> None:
    """z at the last week == (x - mean) / std(ddof=1) over exactly the
    trailing 52 net values, per market — verified with numpy, independently
    of the adapter's pandas rolling path."""
    frame = adapter.features()
    for market in ("ES", "NQ"):
        sub = frame[frame.market == market].sort_values("asof_date")
        for cls in TRADER_CLASSES:
            net = sub[f"{cls}_net_oi"].to_numpy()
            window = net[-52:]
            expected = (net[-1] - window.mean()) / window.std(ddof=1)
            assert sub[f"{cls}_net_oi_z"].iloc[-1] == pytest.approx(expected), (market, cls)


def test_z_nan_before_full_window(adapter: CotTff) -> None:
    """Default config requires a FULL 52-week window: the first 51 weeks per
    market are NaN, so exactly N_WEEKS - 51 rows carry a z."""
    frame = adapter.features()
    for market in ("ES", "NQ"):
        z = frame.loc[frame.market == market].sort_values("asof_date")["dealer_net_oi_z"]
        assert z.iloc[:51].isna().all()
        assert z.iloc[51:].notna().all()
        assert z.notna().sum() == N_WEEKS - 51


def test_z_has_no_lookahead(tmp_path: Path) -> None:
    """Truncating the archive after week 55 leaves every remaining z
    identical — the z at week t never saw weeks > t."""
    cutoff = _asof(54)  # keep weeks 0..54 (55 weeks)

    full_raw = tmp_path / "full"
    _write_archive(full_raw)
    full = CotTff(full_raw, tmp_path / "cache_full").features()

    trunc_raw = tmp_path / "trunc"
    trunc_raw.mkdir()
    for year in (2023, 2024):
        text = _finfut_text(year)
        header, *rows = text.strip().split("\n")
        kept = [r for r in rows if r.split(",")[2] <= cutoff.isoformat()]
        with zipfile.ZipFile(trunc_raw / f"fut_fin_txt_{year}.zip", "w") as zf:
            zf.writestr("FinFutYY.txt", "\n".join([header, *kept]) + "\n")
    trunc = CotTff(trunc_raw, tmp_path / "cache_trunc").features()

    assert trunc.asof_date.max() == pd.Timestamp(cutoff)
    full_head = full[full.asof_date <= pd.Timestamp(cutoff)].reset_index(drop=True)
    z_cols = [f"{cls}_net_oi_z" for cls in TRADER_CLASSES]
    pd.testing.assert_frame_equal(trunc[z_cols], full_head[z_cols])
    # And there IS signal to compare: weeks 52..55 have non-NaN z.
    assert trunc["dealer_net_oi_z"].notna().sum() == 2 * (55 - 51)


def test_config_min_periods_overridable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_archive(raw)
    cfg = CotTffConfig(z_window_weeks=10, z_min_periods=5)
    frame = CotTff(raw, tmp_path / "cache", cfg).features()
    z = frame.loc[frame.market == "ES"].sort_values("asof_date")["lev_money_net_oi_z"]
    assert z.iloc[:4].isna().all()
    assert z.iloc[4:].notna().all()


# ---------------------------------------------------------------------------
# Cache: parquet under cache_root, cache-first
# ---------------------------------------------------------------------------


def test_cache_first_never_reparses_raw(tmp_path: Path) -> None:
    """After one build the raw zips can vanish entirely — the second call is
    served from tff_features.parquet, byte-identical."""
    raw = tmp_path / "raw"
    cache = tmp_path / "cache" / "cot"
    _write_archive(raw)
    first = CotTff(raw, cache).features()
    assert (cache / "tff_features.parquet").exists()

    for zip_path in raw.glob("*.zip"):
        zip_path.unlink()
    again = CotTff(raw, cache).features()  # would raise FileNotFoundError if it reparsed
    pd.testing.assert_frame_equal(first, again)

    # refresh=True DOES reparse — and now fails loudly on the empty archive.
    with pytest.raises(FileNotFoundError):
        CotTff(raw, cache).features(refresh=True)

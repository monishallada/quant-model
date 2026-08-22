"""FINRA Reg SHO daily short-volume adapter: canned fixtures in tmp_path, no network.

All fixture dates are NYSE trading days in January 2026 — well before the
2026-02-22 lockbox start. Files are written in the real wire format: CRLF
line endings, pipe-delimited, trailing bare record-count line, and (as in
recent FINRA files) fractional float volumes.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.feeds.finra_short import (
    DAILY_COLUMNS,
    FEATURE_COLUMNS,
    FinraShortConfig,
    FinraShortDaily,
)

ET = ZoneInfo("America/New_York")

# NYSE trading days, January 2026 (Jan 19 is the MLK holiday).
WEEK1 = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
WEEK2 = [date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14), date(2026, 1, 15),
         date(2026, 1, 16)]
MLK = date(2026, 1, 19)
WEEK3 = [date(2026, 1, 20), date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23)]

HEADER = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"

# Small windows so fixtures stay small; production default is 63/21.
CFG = FinraShortConfig(rolling_window=5, rolling_min_periods=3)


def _fmt(value: float) -> str:
    """Format a volume the way FINRA prints it: bare int, or decimal float."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _shvol_text(day: date, rows: list[tuple[str, float, float, float, str]],
                count: int | None = None) -> str:
    """One CNMSshvol file body: CRLF, header, rows, trailing bare count."""
    lines = [HEADER]
    for symbol, short_vol, exempt_vol, total_vol, market in rows:
        lines.append(
            f"{day:%Y%m%d}|{symbol}|{_fmt(short_vol)}|{_fmt(exempt_vol)}"
            f"|{_fmt(total_vol)}|{market}"
        )
    lines.append(str(count if count is not None else len(rows)))
    return "\r\n".join(lines) + "\r\n"


def _write_day(raw_dir: Path, day: date, rows: list[tuple[str, float, float, float, str]],
               count: int | None = None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"CNMSshvol{day:%Y%m%d}.txt"
    path.write_text(_shvol_text(day, rows, count))
    return path


def _adapter(tmp_path: Path, config: FinraShortConfig = CFG, client=None) -> FinraShortDaily:
    return FinraShortDaily(
        raw_dir=tmp_path / "raw",
        cache_root=tmp_path / "finra_short",
        config=config,
        client=client,
    )


def _ratio_rows(symbol_ratios: dict[str, float], total: float = 1000.0):
    """Rows whose short_ratio equals the given per-symbol values."""
    return [(sym, ratio * total, 0.0, total, "Q,N") for sym, ratio in symbol_ratios.items()]


# ----------------------------------------------------------------------
# Parsing + daily frame
# ----------------------------------------------------------------------


def test_daily_parses_ratio_fractional_volumes_and_zero_total(tmp_path: Path) -> None:
    day = WEEK1[0]
    _write_day(tmp_path / "raw", day, [
        ("AAA", 125.5, 10.25, 502.0, "B,Q,N"),   # fractional floats, recent-file style
        ("BBB", 600, 0, 1200, "Q"),
        ("ZERO", 0, 0, 0, "Q"),                  # zero TotalVolume -> NaN ratio, not inf
    ])
    frame = _adapter(tmp_path).daily(day, day)

    assert list(frame.columns) == DAILY_COLUMNS
    assert list(frame["symbol"]) == ["AAA", "BBB", "ZERO"]
    assert list(frame["asof_date"]) == [day] * 3
    aaa = frame.iloc[0]
    assert aaa["short_volume"] == pytest.approx(125.5)
    assert aaa["short_exempt_volume"] == pytest.approx(10.25)
    assert aaa["total_volume"] == pytest.approx(502.0)
    assert aaa["market"] == "B,Q,N"
    assert aaa["short_ratio"] == pytest.approx(125.5 / 502.0)
    assert frame.iloc[1]["short_ratio"] == pytest.approx(0.5)
    assert pd.isna(frame.iloc[2]["short_ratio"])


def test_trailing_count_mismatch_warns_but_parses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    day = WEEK1[0]
    _write_day(tmp_path / "raw", day, _ratio_rows({"AAA": 0.4, "BBB": 0.5}), count=99)
    with caplog.at_level("WARNING", logger="edge.data.feeds.finra_short"):
        frame = _adapter(tmp_path).daily(day, day)
    assert len(frame) == 2
    assert any("record count" in rec.message for rec in caplog.records)


def test_malformed_line_and_bad_header_raise(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    day = WEEK1[0]
    bad_line = raw / f"CNMSshvol{day:%Y%m%d}.txt"
    bad_line.write_text(f"{HEADER}\r\n{day:%Y%m%d}|AAA|100\r\n1\r\n")
    with pytest.raises(ValueError, match="expected 6 fields"):
        _adapter(tmp_path).daily(day, day)

    day2 = WEEK1[1]
    bad_header = raw / f"CNMSshvol{day2:%Y%m%d}.txt"
    bad_header.write_text("Nope|Header\r\n")
    with pytest.raises(ValueError, match="unexpected header"):
        _adapter(tmp_path).daily(day2, day2)


def test_missing_dates_are_skipped_and_symbols_filter(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    present = [WEEK1[0], WEEK1[1], WEEK1[3]]  # Jan 7 and 9 absent (eroded/non-trading)
    for day in present:
        _write_day(raw, day, _ratio_rows({"AAA": 0.4, "BBB": 0.5}))
    adapter = _adapter(tmp_path)

    frame = adapter.daily(WEEK1[0], WEEK1[4])
    assert sorted(set(frame["asof_date"])) == present

    only_bbb = adapter.daily(WEEK1[0], WEEK1[4], symbols=["BBB"])
    assert set(only_bbb["symbol"]) == {"BBB"}
    assert len(only_bbb) == len(present)


def test_empty_range_and_empty_result(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="empty range"):
        adapter.daily(WEEK1[1], WEEK1[0])
    frame = adapter.daily(WEEK1[0], WEEK1[4])  # no archive at all
    assert frame.empty
    assert list(frame.columns) == DAILY_COLUMNS


# ----------------------------------------------------------------------
# Point-in-time: available_at
# ----------------------------------------------------------------------


def test_available_at_is_next_trading_day_preopen(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    friday, mlk_friday = WEEK1[4], WEEK2[4]  # Jan 9 and Jan 16 (Jan 19 = MLK)
    for day in (friday, mlk_friday):
        _write_day(raw, day, _ratio_rows({"AAA": 0.4}))
    frame = _adapter(tmp_path).daily(friday, mlk_friday)

    by_date = frame.set_index("asof_date")["available_at"]
    # Friday's file -> Monday 09:00 ET (published ~6-8pm ET Friday evening;
    # conservative next-preopen stamp).
    assert by_date[friday] == datetime(2026, 1, 12, 9, 0, tzinfo=ET)
    # Friday before MLK Monday -> TUESDAY 09:00 ET, skipping the holiday.
    assert by_date[mlk_friday] == datetime(2026, 1, 20, 9, 0, tzinfo=ET)
    assert str(frame["available_at"].dt.tz) == "America/New_York"


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------


def test_cache_first_serves_after_raw_file_deleted(tmp_path: Path) -> None:
    day = WEEK1[0]
    raw_file = _write_day(tmp_path / "raw", day, [("AAA", 125.5, 0, 502.0, "Q,N")])
    adapter = _adapter(tmp_path)

    first = adapter.daily(day, day)
    cache_file = tmp_path / "finra_short" / f"date={day.isoformat()}.parquet"
    assert cache_file.exists()

    raw_file.unlink()  # cache must now be the only source
    second = adapter.daily(day, day)
    pd.testing.assert_frame_equal(first, second)


# ----------------------------------------------------------------------
# Features: rolling own-history z and cross-sectional z
# ----------------------------------------------------------------------

# 7 sessions of AAA ratios; window=5/min_periods=3 (CFG above).
RATIOS = [0.30, 0.50, 0.40, 0.60, 0.20, 0.45, 0.80]
DAYS7 = WEEK1 + WEEK2[:2]


def _write_ratio_history(raw_dir: Path, extra: dict[date, dict[str, float]] | None = None) -> None:
    for day, ratio in zip(DAYS7, RATIOS):
        symbol_ratios = {"AAA": ratio, **(extra or {}).get(day, {})}
        _write_day(raw_dir, day, _ratio_rows(symbol_ratios))


def test_rolling_z_matches_manual_computation(tmp_path: Path) -> None:
    _write_ratio_history(tmp_path / "raw")
    frame = _adapter(tmp_path).features(DAYS7[0], DAYS7[-1])
    assert list(frame.columns) == [*DAILY_COLUMNS, *FEATURE_COLUMNS]
    z = frame.set_index("asof_date")["short_ratio_z"]

    # Fewer than min_periods=3 PRIOR observations -> NaN.
    assert pd.isna(z[DAYS7[0]]) and pd.isna(z[DAYS7[1]]) and pd.isna(z[DAYS7[2]])

    # Day 4: baseline is the 3 prior ratios (current day excluded).
    prior3 = RATIOS[:3]
    expected3 = (RATIOS[3] - statistics.mean(prior3)) / statistics.stdev(prior3)
    assert z[DAYS7[3]] == pytest.approx(expected3)

    # Day 7: baseline is the trailing 5 prior ratios (window=5).
    prior5 = RATIOS[1:6]
    expected7 = (RATIOS[6] - statistics.mean(prior5)) / statistics.stdev(prior5)
    assert z[DAYS7[6]] == pytest.approx(expected7)

    # Only one symbol per day -> cross-sectional z undefined everywhere.
    assert frame["short_ratio_xs_z"].isna().all()


def test_features_load_warmup_history_before_start(tmp_path: Path) -> None:
    _write_ratio_history(tmp_path / "raw")
    adapter = _adapter(tmp_path)

    full = adapter.features(DAYS7[0], DAYS7[-1])
    last_only = adapter.features(DAYS7[-1], DAYS7[-1])

    assert list(last_only["asof_date"]) == [DAYS7[-1]]  # trimmed to the range...
    assert last_only.iloc[0]["short_ratio_z"] == pytest.approx(  # ...but z uses warmup
        full.set_index("asof_date")["short_ratio_z"][DAYS7[-1]]
    )


def test_constant_history_gives_nan_z_never_inf(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for day in DAYS7:
        _write_day(raw, day, _ratio_rows({"FLAT": 0.5}))
    frame = _adapter(tmp_path).features(DAYS7[0], DAYS7[-1])
    assert frame["short_ratio_z"].isna().all()  # zero trailing std -> NaN, not +/-inf


def test_cross_sectional_z_per_day(tmp_path: Path) -> None:
    day = WEEK1[0]
    _write_day(tmp_path / "raw", day, _ratio_rows({"AAA": 0.2, "BBB": 0.4, "CCC": 0.6}))
    frame = _adapter(tmp_path).features(day, day)
    xs = frame.set_index("symbol")["short_ratio_xs_z"]
    # mean 0.4, sample std 0.2 -> z = (-1, 0, +1)
    assert xs["AAA"] == pytest.approx(-1.0)
    assert xs["BBB"] == pytest.approx(0.0)
    assert xs["CCC"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Incremental update (canned client — no network, ever)
# ----------------------------------------------------------------------


class _CannedResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _CannedClient:
    """Fake HttpGetter: URL suffix -> canned response; records every call."""

    def __init__(self, responses: dict[str, _CannedResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _CannedResponse:
        self.calls.append(url)
        name = url.rsplit("/", 1)[-1]
        return self._responses[name]


def test_update_writes_new_days_and_skips_403(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_day(raw, WEEK1[0], _ratio_rows({"AAA": 0.4}))  # seed: Mon Jan 5
    client = _CannedClient({
        f"CNMSshvol{WEEK1[1]:%Y%m%d}.txt": _CannedResponse(
            200, _shvol_text(WEEK1[1], _ratio_rows({"AAA": 0.45}))
        ),
        f"CNMSshvol{WEEK1[2]:%Y%m%d}.txt": _CannedResponse(403),  # eroded/non-trading
        f"CNMSshvol{WEEK1[3]:%Y%m%d}.txt": _CannedResponse(
            200, _shvol_text(WEEK1[3], _ratio_rows({"AAA": 0.55}))
        ),
    })
    adapter = _adapter(tmp_path, client=client)

    written = adapter.update(through=WEEK1[3])
    assert written == [WEEK1[1], WEEK1[3]]
    # Only NYSE trading days after the last archived date were requested.
    assert len(client.calls) == 3

    frame = adapter.daily(WEEK1[0], WEEK1[4])
    assert sorted(set(frame["asof_date"])) == [WEEK1[0], WEEK1[1], WEEK1[3]]

    assert adapter.update(through=WEEK1[3]) == []  # already up to date


def test_update_requires_client_and_seeded_archive(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="HTTP client"):
        _adapter(tmp_path).update(through=WEEK1[0])
    with pytest.raises(ValueError, match="archive is empty"):
        _adapter(tmp_path, client=_CannedClient({})).update(through=WEEK1[0])


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


def test_config_rejects_min_periods_above_window() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        FinraShortConfig(rolling_window=5, rolling_min_periods=6)

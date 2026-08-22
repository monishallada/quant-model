"""Market internals: breadth, tick-rule signed volume, TICK proxy, z-scores.

All data is synthetic; every timestamp predates the 2026-02-22 lockbox start.
Expected values are hand-computed from the trade prints in each fixture.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.core.events import TradeEvent
from edge.features.internals import (
    INTERNALS_COLUMNS,
    INTERNALS_METRICS,
    SYMBOL_COLUMNS,
    add_zscores,
    compute_internals,
    minute_symbol_frame,
    trades_to_frame,
)

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
DAY = date(2026, 1, 5)
T0 = datetime(2026, 1, 5, 9, 30, tzinfo=ET)


def _minute(k: int) -> pd.Timestamp:
    """Close timestamp of the k-th minute bar after the open."""
    return pd.Timestamp(T0 + timedelta(minutes=k))


def _frame(rows: list[tuple[float, float, int]]) -> pd.DataFrame:
    """Trades frame from (offset_seconds, price, size) rows."""
    return pd.DataFrame(
        {
            "ts": [T0 + timedelta(seconds=o) for o, _, _ in rows],
            "price": [p for _, p, _ in rows],
            "size": [s for _, _, s in rows],
        }
    )


# AAA: seed uptick, uptick, downtick, unchanged (inherits down), gap minute,
# then an uptick. Exercises seeding, inheritance, and gap forward-fill.
AAA = _frame([(0, 100.0, 100), (30, 101.0, 50), (70, 100.0, 200), (100, 100.0, 100), (200, 102.0, 100)])
# BBB: starts one minute late; final trade lands EXACTLY on the 09:33:00
# boundary, so it must open the bar closing 09:34 (builder convention).
BBB = _frame([(65, 50.0, 200), (110, 49.5, 100), (130, 49.0, 100), (180, 49.5, 300)])
PRIOR = {"AAA": 100.5, "BBB": 49.25}


# ---------------------------------------------------------------------------
# minute_symbol_frame
# ---------------------------------------------------------------------------


def test_minute_symbol_frame_hand_computed() -> None:
    out = minute_symbol_frame(AAA)
    assert list(out.columns) == list(SYMBOL_COLUMNS)
    assert list(out.index) == [_minute(1), _minute(2), _minute(3), _minute(4)]
    assert out.index.name == "ts"

    # 09:31 — trades at 100x100 (seed +1) and 101x50 (uptick)
    row = out.loc[_minute(1)]
    assert row["close"] == 101.0
    assert row["volume"] == 150
    assert row["vwap"] == pytest.approx(15050.0 / 150.0)
    assert (row["up_volume"], row["down_volume"]) == (150, 0)
    assert row["last_tick"] == 1.0

    # 09:32 — downtick 100x200 then unchanged 100x100 (inherits the downtick)
    row = out.loc[_minute(2)]
    assert row["close"] == 100.0
    assert row["volume"] == 300
    assert row["vwap"] == pytest.approx(45050.0 / 450.0)
    assert (row["up_volume"], row["down_volume"]) == (0, 300)
    assert row["last_tick"] == -1.0

    # 09:34 — uptick 102x100; session VWAP is cumulative, not per-minute
    row = out.loc[_minute(4)]
    assert row["close"] == 102.0
    assert row["vwap"] == pytest.approx(55250.0 / 550.0)
    assert (row["up_volume"], row["down_volume"]) == (100, 0)
    assert row["last_tick"] == 1.0


def test_minute_symbol_frame_gap_minute_carries_state_with_zero_flows() -> None:
    out = minute_symbol_frame(AAA)
    gap = out.loc[_minute(3)]  # no AAA trades in [09:32, 09:33)
    assert gap["close"] == 100.0
    assert gap["vwap"] == pytest.approx(45050.0 / 450.0)
    assert gap["last_tick"] == -1.0
    assert (gap["volume"], gap["up_volume"], gap["down_volume"]) == (0, 0, 0)


def test_minute_symbol_frame_boundary_trade_opens_next_bar() -> None:
    out = minute_symbol_frame(BBB)
    assert list(out.index) == [_minute(2), _minute(3), _minute(4)]
    # the 09:33:00.000000 print belongs to the bar CLOSING 09:34, not 09:33
    assert out.loc[_minute(3), "volume"] == 100
    assert out.loc[_minute(4), "volume"] == 300
    assert out.loc[_minute(4), "close"] == 49.5
    assert out.loc[_minute(4), "vwap"] == pytest.approx(34700.0 / 700.0)
    assert out.loc[_minute(4), "last_tick"] == 1.0


def test_minute_symbol_frame_empty_input() -> None:
    out = minute_symbol_frame(_frame([]))
    assert out.empty
    assert list(out.columns) == list(SYMBOL_COLUMNS)


def test_minute_symbol_frame_rejects_malformed_input() -> None:
    naive = _frame([(0, 100.0, 100)])
    naive["ts"] = [datetime(2026, 1, 5, 9, 30)]  # noqa: DTZ001 — naive on purpose
    with pytest.raises(ValueError, match="tz-aware"):
        minute_symbol_frame(naive)

    out_of_order = _frame([(30, 100.0, 100), (0, 100.0, 100)])
    with pytest.raises(ValueError, match="out-of-order"):
        minute_symbol_frame(out_of_order)

    two_days = _frame([(0, 100.0, 100), (24 * 3600, 100.0, 100)])
    with pytest.raises(ValueError, match="sessions"):
        minute_symbol_frame(two_days)

    with pytest.raises(ValueError, match="missing columns"):
        minute_symbol_frame(pd.DataFrame({"ts": [], "price": []}))

    with pytest.raises(ValueError, match="freq"):
        minute_symbol_frame(AAA, freq=pd.Timedelta(0))


# ---------------------------------------------------------------------------
# compute_internals
# ---------------------------------------------------------------------------


def test_internals_hand_computed() -> None:
    out = compute_internals({"AAA": AAA, "BBB": BBB}, PRIOR)
    assert list(out.columns) == list(INTERNALS_COLUMNS)
    assert list(out.index) == [_minute(1), _minute(2), _minute(3), _minute(4)]

    # 09:31 — BBB has not traded yet: universe of one
    row = out.loc[_minute(1)]
    assert row["n_symbols"] == 1
    assert row["breadth_above_vwap"] == 1.0  # 101 > 100.333
    assert row["breadth_above_prior_close"] == 1.0  # 101 > 100.5
    assert (row["up_volume"], row["down_volume"]) == (150, 0)
    assert math.isnan(row["updown_volume_ratio"])  # zero down-volume -> NaN
    assert row["tick_proxy"] == 1

    # 09:32 — both below their session VWAPs; BBB above prior close
    row = out.loc[_minute(2)]
    assert row["n_symbols"] == 2
    assert row["breadth_above_vwap"] == 0.0
    assert row["breadth_above_prior_close"] == 0.5
    assert (row["up_volume"], row["down_volume"]) == (200, 400)
    assert row["updown_volume_ratio"] == pytest.approx(0.5)
    assert row["tick_proxy"] == -2

    # 09:33 — AAA is tradeless (carried state still counts in breadth)
    row = out.loc[_minute(3)]
    assert row["n_symbols"] == 2
    assert row["breadth_above_vwap"] == 0.0
    assert row["breadth_above_prior_close"] == 0.0  # 49.0 < 49.25
    assert (row["up_volume"], row["down_volume"]) == (0, 100)
    assert row["updown_volume_ratio"] == 0.0
    assert row["tick_proxy"] == -2

    # 09:34 — AAA above VWAP, BBB still below; both above prior close
    row = out.loc[_minute(4)]
    assert row["breadth_above_vwap"] == 0.5
    assert row["breadth_above_prior_close"] == 1.0
    assert (row["up_volume"], row["down_volume"]) == (400, 0)
    assert math.isnan(row["updown_volume_ratio"])
    assert row["tick_proxy"] == 2


def test_internals_point_in_time_columns() -> None:
    out = compute_internals({"AAA": AAA, "BBB": BBB}, PRIOR)
    # asof_date is the minute's own ET trading date
    assert set(out["asof_date"]) == {DAY}
    # available_at is the minute CLOSE instant, tz-aware ET, zero added lag;
    # the engine's BarEvent visibility supplies the one-bar consumer lag.
    assert str(out["available_at"].dt.tz) == "America/New_York"
    assert list(out["available_at"]) == list(out.index)
    assert out["available_at"].is_monotonic_increasing
    # first trade printed 09:30:00 -> earliest internals row closes 09:31
    assert out["available_at"].iloc[0] == _minute(1)


def test_internals_symbol_with_no_trades_never_enters_denominator() -> None:
    out = compute_internals(
        {"AAA": AAA, "BBB": BBB, "CCC": _frame([])}, {**PRIOR, "CCC": 10.0}
    )
    assert list(out["n_symbols"]) == [1, 2, 2, 2]


def test_internals_all_empty_universe_yields_empty_frame() -> None:
    out = compute_internals({"AAA": _frame([])}, {"AAA": 100.0})
    assert out.empty
    assert list(out.columns) == list(INTERNALS_COLUMNS)


def test_internals_rejects_bad_universe() -> None:
    with pytest.raises(ValueError, match="empty universe"):
        compute_internals({}, {})
    with pytest.raises(ValueError, match="prior_close missing"):
        compute_internals({"AAA": AAA, "BBB": BBB}, {"AAA": 100.5})
    with pytest.raises(ValueError, match="positive"):
        compute_internals({"AAA": AAA}, {"AAA": 0.0})


def test_internals_rejects_mixed_sessions() -> None:
    next_day = _frame([(0, 100.0, 100)])
    next_day["ts"] = [T0 + timedelta(days=1)]
    with pytest.raises(ValueError, match="different ET sessions"):
        compute_internals({"AAA": AAA, "DDD": next_day}, {**PRIOR, "DDD": 100.0})


# ---------------------------------------------------------------------------
# z-scores
# ---------------------------------------------------------------------------


def test_add_zscores_hand_computed() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
    out = add_zscores(frame, ("a", "b"), window=3)
    # trailing window of 3 ending at each row; sample std (ddof=1)
    assert out["a_z"].iloc[:2].isna().all()  # fewer than min_periods rows
    assert out["a_z"].iloc[2] == pytest.approx(1.0)  # (3 - 2) / 1
    assert out["a_z"].iloc[3] == pytest.approx(1.0)  # (4 - 3) / 1
    assert out["b_z"].isna().all()  # zero-variance window -> NaN, never inf
    # input is not mutated and raw columns survive
    assert "a_z" not in frame.columns
    assert list(out["a"]) == [1.0, 2.0, 3.0, 4.0]


def test_add_zscores_min_periods_override() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = add_zscores(frame, ("a",), window=3, min_periods=2)
    # row 1: window [1, 2] -> mean 1.5, std sqrt(0.5)
    assert out["a_z"].iloc[1] == pytest.approx(0.5 / math.sqrt(0.5))


def test_add_zscores_on_internals_output() -> None:
    internals = compute_internals({"AAA": AAA, "BBB": BBB}, PRIOR)
    out = add_zscores(internals, window=2)
    for col in INTERNALS_METRICS:
        assert f"{col}_z" in out.columns
    # breadth_above_vwap over [1.0, 0.0]: mean 0.5, std sqrt(0.5)
    assert out["breadth_above_vwap_z"].loc[_minute(2)] == pytest.approx(-0.5 / math.sqrt(0.5))
    # ratio window [NaN, 0.5] has one valid value < min_periods -> NaN;
    # window [0.5, 0.0] -> mean 0.25, std sqrt(0.125)
    assert math.isnan(out["updown_volume_ratio_z"].loc[_minute(2)])
    assert out["updown_volume_ratio_z"].loc[_minute(3)] == pytest.approx(-0.25 / math.sqrt(0.125))


def test_add_zscores_rejects_degenerate_windows() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="window"):
        add_zscores(frame, ("a",), window=1)
    with pytest.raises(ValueError, match="min_periods"):
        add_zscores(frame, ("a",), window=3, min_periods=1)


# ---------------------------------------------------------------------------
# trades_to_frame
# ---------------------------------------------------------------------------


def test_trades_to_frame_round_trip() -> None:
    events = [
        TradeEvent(ts=T0, symbol="AAA", price=100.0, size=100),
        TradeEvent(ts=T0 + timedelta(seconds=30), symbol="AAA", price=101.0, size=50),
    ]
    frame = trades_to_frame(events)
    assert list(frame.columns) == ["ts", "price", "size"]
    assert list(frame["price"]) == [100.0, 101.0]
    assert list(frame["size"]) == [100, 50]
    # feeds internals directly
    out = minute_symbol_frame(frame)
    assert out.loc[_minute(1), "volume"] == 150


def test_trades_to_frame_empty() -> None:
    frame = trades_to_frame([])
    assert frame.empty
    assert list(frame.columns) == ["ts", "price", "size"]

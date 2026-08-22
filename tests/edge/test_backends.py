"""CatalystBridge tests: in-memory canned frames, zero network, zero terminal.

The fake fetcher double records every delegation, so cache-only assertions
are literal: ``fetcher.calls == []``. Fixture dates predate the 2026-02-22
lockbox wall EXCEPT in the lockbox tests, which exercise the embargo
machinery itself — those rows are synthetic and exist only to prove the
loader never lets the bridge see them.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.backends import (
    BAR_COLUMNS,
    BRIDGE_KINDS,
    CATEGORY_GREEKS_EOD,
    CATEGORY_OPEN_INTEREST,
    CATEGORY_OPTION_EOD,
    CATEGORY_OPTION_QUOTE,
    CATEGORY_STOCK_MINUTE,
    CatalystBridge,
    CatalystBridgeConfig,
    CatalystCacheMiss,
    expiry_day_key,
    greeks_eod_key,
    minute_bar_key,
    option_quote_key,
)
from edge.data.loader import DataBackend, EdgeDataLoader
from edge.validation.lockbox import LockboxViolation

ET = ZoneInfo("America/New_York")

# Pre-wall fixture anchors (wall: 2026-02-22, from config/edge.yaml).
TUE = date(2024, 1, 2)
WED = date(2024, 1, 3)
THU = date(2024, 1, 4)
FRI = date(2024, 1, 5)
MON = date(2024, 1, 8)

EXPIRY = date(2024, 1, 19)
SESSION_START, SESSION_END = time(9, 30), time(16, 0)


# ----------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------


class FakeCache:
    """In-memory (category, key) -> frame store; records every get/put."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame] | None = None) -> None:
        self.frames = dict(frames or {})
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []

    def get(self, category: str, key: str) -> pd.DataFrame | None:
        self.get_calls.append((category, key))
        frame = self.frames.get((category, key))
        return None if frame is None else frame.copy()

    def put(self, category: str, key: str, df: pd.DataFrame) -> None:
        self.put_calls.append((category, key))
        self.frames[(category, key)] = df.copy()


class FakeFetcher:
    """Recording CatalystFetcher double; cache-only tests assert calls == []."""

    def __init__(self, frames: dict[tuple, pd.DataFrame] | None = None) -> None:
        self.calls: list[tuple] = []
        self._frames = dict(frames or {})

    def _serve(self, call: tuple) -> pd.DataFrame:
        self.calls.append(call)
        return self._frames.get(call, pd.DataFrame()).copy()

    def stock_minute_day(self, symbol: str, day: date) -> pd.DataFrame:
        return self._serve(("stock_minute_day", symbol, day))

    def option_quote_day(self, underlying, expiry, strike, right, day, s, e):
        return self._serve(("option_quote_day", underlying, expiry, strike, right, day, s, e))

    def greeks_eod_frame(self, symbol: str, expiry: date) -> pd.DataFrame:
        return self._serve(("greeks_eod_frame", symbol, expiry))

    def open_interest_day(self, symbol, expiry, day):
        return self._serve(("open_interest_day", symbol, expiry, day))

    def option_eod_day(self, symbol, expiry, day):
        return self._serve(("option_eod_day", symbol, expiry, day))


# ----------------------------------------------------------------------
# Canned raw frames (catalyst on-disk shapes: tz-naive ET timestamps)
# ----------------------------------------------------------------------


def minute_raw(day: date, minutes: tuple[str, ...] = ("09:30", "09:31"),
               base: float = 100.0) -> pd.DataFrame:
    stamps = [datetime.combine(day, time.fromisoformat(m)) for m in minutes]
    n = len(stamps)
    return pd.DataFrame({
        "ts": pd.to_datetime(stamps),
        "open": [base + i for i in range(n)],
        "high": [base + i + 0.5 for i in range(n)],
        "low": [base + i - 0.5 for i in range(n)],
        "close": [base + i + 0.25 for i in range(n)],
        "volume": [1000 + i for i in range(n)],
    })


def quote_raw(day: date, minutes: tuple[str, ...] = ("09:30", "09:31"),
              bid: float = 1.20, ask: float = 1.30) -> pd.DataFrame:
    stamps = [datetime.combine(day, time.fromisoformat(m)) for m in minutes]
    n = len(stamps)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(stamps),
        "bid": [bid + 0.01 * i for i in range(n)],
        "ask": [ask + 0.01 * i for i in range(n)],
    })


def greeks_raw(days: list[date]) -> pd.DataFrame:
    """EOD greeks per iv_eod shape; rows stamped ~15:59:55 like the archive."""
    stamps = [datetime.combine(d, time(15, 59, 55)) for d in days]
    n = len(stamps)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(stamps),
        "strike": [190.0] * n,
        "right": ["CALL"] * n,
        "implied_vol": [0.20 + 0.01 * i for i in range(n)],
        "underlying_price": [189.5 + i for i in range(n)],
    })


def oi_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [190.0, 195.0],
        "right": ["CALL", "PUT"],
        "open_interest": [1500, 800],
    })


def eod_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [190.0],
        "right": ["CALL"],
        "close": [1.25],
        "volume": [4200],
    })


def make_loader(cache: FakeCache, fetcher: FakeFetcher | None = None,
                **bridge_kwargs: object) -> EdgeDataLoader:
    return EdgeDataLoader(CatalystBridge(cache, fetcher, **bridge_kwargs))


# ----------------------------------------------------------------------
# Protocol conformance and construction
# ----------------------------------------------------------------------


def test_bridge_satisfies_loader_backend_protocol() -> None:
    assert isinstance(CatalystBridge(FakeCache()), DataBackend)


def test_allow_fetch_requires_a_fetcher() -> None:
    with pytest.raises(ValueError, match="allow_fetch"):
        CatalystBridge(FakeCache(), allow_fetch=True)


def test_unknown_and_trades_kinds_rejected() -> None:
    bridge = CatalystBridge(FakeCache())
    with pytest.raises(ValueError, match="trades"):
        bridge.fetch("AAPL", TUE, TUE, "trades")
    with pytest.raises(ValueError, match="sonar"):
        bridge.fetch("AAPL", TUE, TUE, "sonar")
    assert "bars" in BRIDGE_KINDS and "quotes" in BRIDGE_KINDS


# ----------------------------------------------------------------------
# Stock minute bars (kind "bars")
# ----------------------------------------------------------------------


def test_stock_minute_bars_cache_only_end_to_end() -> None:
    cache = FakeCache({
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", TUE)): minute_raw(TUE),
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", WED)): minute_raw(WED, base=110.0),
    })
    fetcher = FakeFetcher()
    frame = make_loader(cache, fetcher).load("AAPL", TUE, WED, "bars")

    assert fetcher.calls == []  # cache-only: the double is never consulted
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert len(frame) == 4
    assert list(frame["asof_date"].dt.date) == [TUE, TUE, WED, WED]
    # ts is tz-aware ET; the bar is visible one minute after its stamp
    assert frame["ts"].iloc[0] == pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-02 09:31", tz="America/New_York"
    )
    assert (frame["available_at"] > frame["ts"]).all()
    assert frame["close"].iloc[2] == pytest.approx(110.25)
    assert frame["volume"].iloc[1] == 1001


def test_bars_skip_weekends_and_serve_cached_empty_days() -> None:
    """Weekend days are never looked up; a cached-EMPTY day (holiday) is a
    real answer contributing zero rows — no miss, no fetch."""
    cache = FakeCache({
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", FRI)): minute_raw(FRI),
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", MON)): pd.DataFrame(),
    })
    fetcher = FakeFetcher()
    frame = make_loader(cache, fetcher).load("AAPL", FRI, MON, "bars")

    assert fetcher.calls == []
    assert [k for _, k in cache.get_calls] == [
        minute_bar_key("AAPL", FRI), minute_bar_key("AAPL", MON)  # no Sat/Sun
    ]
    assert list(frame["asof_date"].dt.date.unique()) == [FRI]


def test_cache_only_miss_raises_and_never_fetches() -> None:
    fetcher = FakeFetcher()
    loader = make_loader(FakeCache(), fetcher)
    with pytest.raises(CatalystCacheMiss, match=CATEGORY_STOCK_MINUTE):
        loader.load("AAPL", TUE, TUE, "bars")
    assert fetcher.calls == []


def test_missing_ok_skips_with_warning_and_never_fetches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = FakeFetcher()
    loader = make_loader(FakeCache(), fetcher, missing_ok=True)
    with caplog.at_level(logging.WARNING, logger="edge.data.backends"):
        frame = loader.load("AAPL", TUE, TUE, "bars")
    assert fetcher.calls == []
    assert minute_bar_key("AAPL", TUE) in caplog.text
    assert frame.empty
    assert list(frame.columns) == list(BAR_COLUMNS)


def test_allow_fetch_delegates_on_miss_only() -> None:
    cache = FakeCache({
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", TUE)): minute_raw(TUE),
    })
    fetcher = FakeFetcher({
        ("stock_minute_day", "AAPL", WED): minute_raw(WED, base=120.0),
    })
    frame = make_loader(cache, fetcher, allow_fetch=True).load("AAPL", TUE, WED, "bars")

    assert fetcher.calls == [("stock_minute_day", "AAPL", WED)]  # hit day untouched
    assert len(frame) == 4
    assert frame["open"].iloc[2] == pytest.approx(120.0)


def test_bars_reject_option_style_symbol() -> None:
    with pytest.raises(ValueError, match="plain equity ticker"):
        CatalystBridge(FakeCache()).fetch("AAPL:2024-01-19:190:C", TUE, TUE, "bars")


# ----------------------------------------------------------------------
# Option 1-minute NBBO (kind "quotes")
# ----------------------------------------------------------------------


def test_option_nbbo_cache_only_end_to_end() -> None:
    key = option_quote_key("AAPL", EXPIRY, 190.0, "C", TUE, SESSION_START, SESSION_END)
    # The key format is catalyst's on-disk contract — pin it exactly.
    assert key == "AAPL_20240119_190.000_call_20240102_0930_1600"

    cache = FakeCache({(CATEGORY_OPTION_QUOTE, key): quote_raw(TUE)})
    fetcher = FakeFetcher()
    spec = "AAPL:2024-01-19:190:C"
    frame = make_loader(cache, fetcher).load(spec, TUE, TUE, "quotes")

    assert fetcher.calls == []
    assert len(frame) == 2
    assert frame["symbol"].iloc[0] == spec
    assert frame["underlying"].iloc[0] == "AAPL"
    assert frame["expiry"].iloc[0] == pd.Timestamp("2024-01-19")
    assert frame["strike"].iloc[0] == pytest.approx(190.0)
    assert frame["right"].iloc[0] == "C"
    assert frame["bid"].iloc[1] == pytest.approx(1.21)
    assert frame["ask"].iloc[0] == pytest.approx(1.30)
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-02 09:31", tz="America/New_York"
    )


def test_option_nbbo_allow_fetch_passes_contract_and_session_window() -> None:
    fetcher = FakeFetcher({
        ("option_quote_day", "AAPL", EXPIRY, 190.0, "P", TUE, SESSION_START, SESSION_END):
            quote_raw(TUE),
    })
    loader = make_loader(FakeCache(), fetcher, allow_fetch=True)
    frame = loader.load("AAPL:2024-01-19:190:put", TUE, TUE, "quotes")

    assert fetcher.calls == [
        ("option_quote_day", "AAPL", EXPIRY, 190.0, "P", TUE, SESSION_START, SESSION_END)
    ]
    assert frame["right"].iloc[0] == "P"


@pytest.mark.parametrize("bad", [
    "AAPL",                        # no contract parts
    "AAPL:2024-01-19:190",         # missing right
    "AAPL:2024-01-19:190:X",       # bad right
    "AAPL:2024-01-19:zero:C",      # bad strike
    "AAPL:2024-01-19:-5:C",        # non-positive strike
    "AAPL:someday:190:C",          # bad expiry
    ":2024-01-19:190:C",           # empty underlying
])
def test_quotes_reject_malformed_symbols(bad: str) -> None:
    with pytest.raises(ValueError):
        CatalystBridge(FakeCache()).fetch(bad, TUE, TUE, "quotes")


# ----------------------------------------------------------------------
# EOD greeks (kind "greeks_eod", alias "iv_eod")
# ----------------------------------------------------------------------


def test_greeks_eod_filters_to_requested_range_and_stamps_evening() -> None:
    cache = FakeCache({
        (CATEGORY_GREEKS_EOD, greeks_eod_key("AAPL", EXPIRY)):
            greeks_raw([TUE, WED, THU, FRI]),
    })
    fetcher = FakeFetcher()
    frame = make_loader(cache, fetcher).load("AAPL:2024-01-19", WED, THU, "greeks_eod")

    assert fetcher.calls == []
    assert list(frame["asof_date"].dt.date) == [WED, THU]
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-03 18:00", tz="America/New_York"
    )
    # archive value columns ride through untouched
    assert frame["implied_vol"].iloc[0] == pytest.approx(0.21)
    assert frame["underlying_price"].iloc[1] == pytest.approx(191.5)
    assert frame["underlying"].iloc[0] == "AAPL"
    assert frame["expiry"].iloc[0] == pd.Timestamp("2024-01-19")


def test_iv_eod_alias_serves_the_same_frames() -> None:
    frames = {
        (CATEGORY_GREEKS_EOD, greeks_eod_key("AAPL", EXPIRY)): greeks_raw([TUE, WED]),
    }
    via_alias = make_loader(FakeCache(frames)).load("AAPL:2024-01-19", TUE, WED, "iv_eod")
    canonical = make_loader(FakeCache(frames)).load("AAPL:2024-01-19", TUE, WED, "greeks_eod")
    pd.testing.assert_frame_equal(via_alias, canonical)


def test_greeks_eod_requires_expiry_symbol() -> None:
    with pytest.raises(ValueError, match="UNDERLYING:YYYY-MM-DD"):
        CatalystBridge(FakeCache()).fetch("AAPL", TUE, TUE, "greeks_eod")


def test_greeks_eod_allow_fetch_delegates_once_per_expiry() -> None:
    fetcher = FakeFetcher({
        ("greeks_eod_frame", "AAPL", EXPIRY): greeks_raw([TUE, WED]),
    })
    loader = make_loader(FakeCache(), fetcher, allow_fetch=True)
    frame = loader.load("AAPL:2024-01-19", TUE, WED, "greeks_eod")
    assert fetcher.calls == [("greeks_eod_frame", "AAPL", EXPIRY)]
    assert len(frame) == 2


# ----------------------------------------------------------------------
# Open interest and EOD option bars
# ----------------------------------------------------------------------


def test_open_interest_stamped_next_business_day_morning() -> None:
    cache = FakeCache({
        (CATEGORY_OPEN_INTEREST, expiry_day_key("AAPL", EXPIRY, FRI)): oi_raw(),
    })
    frame = make_loader(cache, FakeFetcher()).load("AAPL:2024-01-19", FRI, FRI, "oi")

    assert len(frame) == 2  # one row per strike/right
    assert list(frame["asof_date"].dt.date) == [FRI, FRI]
    # Friday's OI is first knowable Monday morning (OCC overnight cycle)
    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-08 09:00", tz="America/New_York"
    )
    assert list(frame["open_interest"]) == [1500, 800]
    assert list(frame["right"]) == ["CALL", "PUT"]


def test_option_eod_stamped_same_evening() -> None:
    cache = FakeCache({
        (CATEGORY_OPTION_EOD, expiry_day_key("AAPL", EXPIRY, THU)): eod_raw(),
    })
    frame = make_loader(cache, FakeFetcher()).load("AAPL:2024-01-19", THU, THU, "option_eod")

    assert frame["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-04 18:00", tz="America/New_York"
    )
    assert frame["volume"].iloc[0] == 4200
    assert frame["symbol"].iloc[0] == "AAPL:2024-01-19"


def test_custom_config_changes_stamps_and_quote_key() -> None:
    cfg = CatalystBridgeConfig(
        session_start=time(9, 35), session_end=time(15, 45),
        minute_lag_minutes=2, eod_publish_hour_et=20,
    )
    key = option_quote_key("AAPL", EXPIRY, 190.0, "C", TUE, cfg.session_start, cfg.session_end)
    assert key.endswith("_0935_1545")
    cache = FakeCache({
        (CATEGORY_OPTION_QUOTE, key): quote_raw(TUE),
        (CATEGORY_OPTION_EOD, expiry_day_key("AAPL", EXPIRY, TUE)): eod_raw(),
    })
    loader = make_loader(cache, FakeFetcher(), config=cfg)
    quotes = loader.load("AAPL:2024-01-19:190:C", TUE, TUE, "quotes")
    assert quotes["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-02 09:32", tz="America/New_York"
    )
    eod = loader.load("AAPL:2024-01-19", TUE, TUE, "option_eod")
    assert eod["available_at"].iloc[0] == pd.Timestamp(
        "2024-01-02 20:00", tz="America/New_York"
    )


# ----------------------------------------------------------------------
# Lockbox wall, end-to-end through EdgeDataLoader
# ----------------------------------------------------------------------

# Synthetic guard-test anchors around the real wall (2026-02-22, a Sunday).
PRE_THU = date(2026, 2, 19)
PRE_FRI = date(2026, 2, 20)
IN_MON = date(2026, 2, 23)


def test_lockbox_clamp_stops_bar_days_before_the_wall(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A range spilling past the wall is clamped by the loader; the bridge
    never even looks up the in-lockbox day's cache key."""
    cache = FakeCache({
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", PRE_THU)): minute_raw(PRE_THU),
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", PRE_FRI)): minute_raw(PRE_FRI),
        (CATEGORY_STOCK_MINUTE, minute_bar_key("AAPL", IN_MON)): minute_raw(IN_MON),
    })
    fetcher = FakeFetcher()
    loader = make_loader(cache, fetcher)

    with caplog.at_level(logging.WARNING, logger="edge.data.loader"):
        frame = loader.load("AAPL", PRE_THU, date(2026, 3, 6), "bars")

    assert "lockbox clamp" in caplog.text
    assert [k for _, k in cache.get_calls] == [
        minute_bar_key("AAPL", PRE_THU), minute_bar_key("AAPL", PRE_FRI)
    ]  # the 2026-02-23 key is never requested
    assert frame["asof_date"].max().date() == PRE_FRI
    assert fetcher.calls == []


def test_lockbox_clamp_filters_greeks_eod_rows_at_the_wall() -> None:
    """Per-expiry iv_eod frames span the wall; the bridge's range filter must
    drop every row at/after it once the loader clamps the range."""
    expiry = date(2026, 3, 20)
    cache = FakeCache({
        (CATEGORY_GREEKS_EOD, greeks_eod_key("AAPL", expiry)): greeks_raw(
            [date(2026, 2, 18), PRE_THU, PRE_FRI, IN_MON, date(2026, 2, 24)]
        ),
    })
    frame = make_loader(cache, FakeFetcher()).load(
        "AAPL:2026-03-20", date(2026, 2, 16), date(2026, 3, 6), "greeks_eod"
    )

    assert list(frame["asof_date"].dt.date) == [date(2026, 2, 18), PRE_THU, PRE_FRI]
    assert frame["available_at"].max() < pd.Timestamp("2026-02-22", tz="America/New_York")


def test_lockbox_read_inside_the_wall_raises_before_the_bridge() -> None:
    cache = FakeCache()
    fetcher = FakeFetcher()
    loader = make_loader(cache, fetcher)
    with pytest.raises(LockboxViolation):
        loader.load("AAPL", date(2026, 3, 2), date(2026, 6, 30), "bars")
    assert cache.get_calls == []  # the backend was never touched
    assert fetcher.calls == []

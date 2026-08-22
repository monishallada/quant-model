"""Alpaca tick adapter (canned pages, no network) and bar builders.

All data is synthetic; every timestamp predates the 2026-02-22 lockbox start.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pytest

from edge.core.events import BarEvent, QuoteEvent, TradeEvent
from edge.data.bars.builders import (
    bars_to_frame,
    dollar_bars,
    imbalance_bars,
    time_bars,
    volume_bars,
)
from edge.data.feeds.alpaca_ticks import AlpacaTicks, AlpacaTicksConfig

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
DAY = date(2026, 1, 5)
T0 = datetime(2026, 1, 5, 9, 30, tzinfo=ET)

# EWMA parameters shared by the imbalance-bar tests (would come from config
# in production; pinned here so thresholds are hand-computable).
IMB = dict(
    expected_ticks_init=20.0,
    expected_imbalance_init=0.0,
    alpha_ticks=0.5,
    alpha_imbalance=0.05,
    min_expected_imbalance=0.25,
)


def _trade(offset_s: float, price: float, size: int, symbol: str = "SPY") -> TradeEvent:
    return TradeEvent(ts=T0 + timedelta(seconds=offset_s), symbol=symbol, price=price, size=size)


# ---------------------------------------------------------------------------
# Fake HTTP client (canned pages, records every request)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = f"fake-{status_code}"
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Serves canned responses per URL, in order; records (url, params)."""

    def __init__(self, pages: dict[str, list[_FakeResponse]]) -> None:
        self._pages = {url: list(seq) for url, seq in pages.items()}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _FakeResponse:
        self.calls.append((url, dict(params or {})))
        return self._pages[url].pop(0)


class _RaisingClient:
    """Any HTTP call is a test failure — used to prove cache-first behavior."""

    def get(self, url: str, *, params: dict[str, str] | None = None) -> _FakeResponse:
        raise AssertionError(f"unexpected network call: {url}")


# No-sleep config so retry tests do not stall.
FAST = AlpacaTicksConfig(retry_backoff_seconds=0.0, max_attempts=2)


# ---------------------------------------------------------------------------
# Adapter: pagination, normalization, caching
# ---------------------------------------------------------------------------


def test_quotes_paginated_fetch_and_normalization(tmp_path: Path) -> None:
    q = {"t": "2026-01-05T14:30:00.000000001Z", "bp": 99.98, "ap": 100.02, "bs": 2, "as": 3}
    q2 = {"t": "2026-01-05T14:30:01.5Z", "bp": 99.99, "ap": 100.03, "bs": 1, "as": 4}
    q3 = {"t": "2026-01-05T14:30:02Z", "bp": 100.0, "ap": 100.04, "bs": 5, "as": 6}
    client = _FakeClient(
        {
            "/v2/stocks/SPY/quotes": [
                _FakeResponse(200, {"quotes": [q, q2], "next_page_token": "tok-1"}),
                _FakeResponse(200, {"quotes": [q3], "next_page_token": None}),
            ]
        }
    )
    adapter = AlpacaTicks(client, tmp_path)
    events = adapter.quotes("SPY", DAY)

    assert len(events) == 3
    # pagination: second request carries the token from the first page
    assert [c[1].get("page_token") for c in client.calls] == [None, "tok-1"]
    first_params = client.calls[0][1]
    assert first_params["feed"] == "sip"
    assert first_params["start"] == "2026-01-05"
    assert first_params["limit"] == str(AlpacaTicksConfig().page_limit)
    # normalization: 14:30 UTC == 09:30 ET; nanoseconds truncate to microseconds
    assert events[0] == QuoteEvent(ts=T0, symbol="SPY", bid=99.98, ask=100.02, bid_size=2, ask_size=3)
    assert events[1].ts == T0 + timedelta(seconds=1, microseconds=500_000)
    assert [e.ts for e in events] == sorted(e.ts for e in events)


def test_trades_fetch_then_cache_round_trip(tmp_path: Path) -> None:
    t1 = {"t": "2026-01-05T14:30:00.25Z", "p": 100.5, "s": 100}
    t2 = {"t": "2026-01-05T14:30:03Z", "p": 100.75, "s": 250}
    client = _FakeClient(
        {"/v2/stocks/SPY/trades": [_FakeResponse(200, {"trades": [t1, t2], "next_page_token": None})]}
    )
    fetched = AlpacaTicks(client, tmp_path).trades("SPY", DAY)
    assert fetched == [
        TradeEvent(ts=T0 + timedelta(seconds=0.25), symbol="SPY", price=100.5, size=100),
        TradeEvent(ts=T0 + timedelta(seconds=3), symbol="SPY", price=100.75, size=250),
    ]
    # cache landed at the documented layout
    path = tmp_path / "trades" / "symbol=SPY" / "date=2026-01-05.parquet"
    assert path.exists()
    # second adapter with a network-forbidding client serves identical events
    cached = AlpacaTicks(_RaisingClient(), tmp_path).trades("SPY", DAY)
    assert cached == fetched


def test_retry_on_429_then_success(tmp_path: Path) -> None:
    t1 = {"t": "2026-01-05T14:30:00Z", "p": 100.0, "s": 10}
    client = _FakeClient(
        {
            "/v2/stocks/SPY/trades": [
                _FakeResponse(429),
                _FakeResponse(200, {"trades": [t1], "next_page_token": None}),
            ]
        }
    )
    events = AlpacaTicks(client, tmp_path, FAST).trades("SPY", DAY)
    assert len(events) == 1
    assert len(client.calls) == 2


def test_failed_pagination_is_not_cached(tmp_path: Path) -> None:
    t1 = {"t": "2026-01-05T14:30:00Z", "p": 100.0, "s": 10}
    client = _FakeClient(
        {
            "/v2/stocks/SPY/trades": [
                _FakeResponse(200, {"trades": [t1], "next_page_token": "tok-1"}),
                _FakeResponse(500),
                _FakeResponse(500),  # exhausts FAST.max_attempts == 2
            ]
        }
    )
    events = AlpacaTicks(client, tmp_path, FAST).trades("SPY", DAY)
    assert events == []
    assert not (tmp_path / "trades" / "symbol=SPY" / "date=2026-01-05.parquet").exists()


def test_empty_day_is_cached(tmp_path: Path) -> None:
    client = _FakeClient(
        {"/v2/stocks/SPY/quotes": [_FakeResponse(200, {"quotes": None, "next_page_token": None})]}
    )
    assert AlpacaTicks(client, tmp_path).quotes("SPY", DAY) == []
    assert (tmp_path / "quotes" / "symbol=SPY" / "date=2026-01-05.parquet").exists()
    # known-empty day never re-requested
    assert AlpacaTicks(_RaisingClient(), tmp_path).quotes("SPY", DAY) == []


def test_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        AlpacaTicks.from_env(tmp_path)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    assert isinstance(AlpacaTicks.from_env(tmp_path), AlpacaTicks)


# ---------------------------------------------------------------------------
# Time bars
# ---------------------------------------------------------------------------


def test_time_bars_hand_computed() -> None:
    trades = [
        _trade(10, 100.0, 100),
        _trade(40, 101.0, 50),
        _trade(65, 102.0, 200),   # rolls the 09:30 bar
        _trade(150, 103.0, 10),   # rolls the 09:31 bar; its own bar never closes
    ]
    bars = list(time_bars(trades, timedelta(minutes=1)))
    assert bars == [
        BarEvent(ts=T0 + timedelta(minutes=1), symbol="SPY",
                 open=100.0, high=101.0, low=100.0, close=101.0, volume=150),
        BarEvent(ts=T0 + timedelta(minutes=2), symbol="SPY",
                 open=102.0, high=102.0, low=102.0, close=102.0, volume=200),
    ]


def test_time_bars_gap_produces_no_empty_bars() -> None:
    trades = [_trade(10, 100.0, 100), _trade(15 * 60, 101.0, 100)]
    bars = list(time_bars(trades, timedelta(minutes=1)))
    # one bar, closed at its own boundary — not at the emitting trade's time,
    # and no zero-volume bars for the empty minutes in between
    assert [b.ts for b in bars] == [T0 + timedelta(minutes=1)]
    assert bars[0].volume == 100


def test_time_bars_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        list(time_bars([_trade(0, 100.0, 1)], timedelta(0)))


# ---------------------------------------------------------------------------
# Volume / dollar bars
# ---------------------------------------------------------------------------


def test_volume_bars_hand_computed() -> None:
    prices = [100.0, 101.0, 99.0, 100.5, 101.5, 100.8]
    sizes = [100, 200, 300, 150, 250, 100]
    trades = [_trade(i, p, s) for i, (p, s) in enumerate(zip(prices, sizes))]
    bars = list(volume_bars(trades, volume_per_bar=500))
    # cum vol: 100, 300, 600* | 150, 400, 500*
    assert bars == [
        BarEvent(ts=trades[2].ts, symbol="SPY",
                 open=100.0, high=101.0, low=99.0, close=99.0, volume=600),
        BarEvent(ts=trades[5].ts, symbol="SPY",
                 open=100.5, high=101.5, low=100.5, close=100.8, volume=500),
    ]


def test_volume_bars_drops_trailing_partial() -> None:
    trades = [_trade(0, 100.0, 100), _trade(1, 100.0, 100)]
    assert list(volume_bars(trades, volume_per_bar=500)) == []


def test_dollar_bars_hand_computed() -> None:
    prices = [10.0, 10.0, 10.0, 20.0, 20.0]
    sizes = [100, 100, 100, 100, 100]
    trades = [_trade(i, p, s) for i, (p, s) in enumerate(zip(prices, sizes))]
    bars = list(dollar_bars(trades, dollar_per_bar=2500.0))
    # notional: 1000, 2000, 3000* | 2000, 4000*
    assert [(b.ts, b.volume) for b in bars] == [(trades[2].ts, 300), (trades[4].ts, 200)]
    assert (bars[1].open, bars[1].close) == (20.0, 20.0)


# ---------------------------------------------------------------------------
# Imbalance bars (AFML tick-rule EWMA recursion)
# ---------------------------------------------------------------------------


def _balanced(n: int, start_offset: float = 0.0) -> list[TradeEvent]:
    """Alternating up/down prints: theta oscillates within +/-2, below any
    floored threshold."""
    return [
        _trade(start_offset + i, 100.0 if i % 2 == 0 else 100.1, 100) for i in range(n)
    ]


def test_imbalance_balanced_stream_produces_few_bars() -> None:
    bars = list(imbalance_bars(_balanced(200), **IMB))
    assert bars == []  # threshold floor = 20 * 0.25 = 5; |theta| never exceeds 2


def test_imbalance_one_sided_burst_closes_early() -> None:
    # 100 strictly rising prints: b = +1 every tick, theta grows one per trade.
    trades = [_trade(i, 100.0 + 0.1 * i, 100) for i in range(100)]
    bars = list(imbalance_bars(trades, **IMB))
    # First close at tick 5: theta = 5 >= 20 * max(1 - 0.95**5, 0.25) = 5.0
    assert bars[0].ts == trades[4].ts
    assert bars[0].volume == 500
    # E[T] shrinks toward realized bar sizes, so closes keep coming early
    assert len(bars) >= 5
    assert all(b0.ts <= b1.ts for b0, b1 in zip(bars, bars[1:]))


def test_imbalance_burst_after_balanced_prefix_triggers_close() -> None:
    balanced = _balanced(60)
    # continue upward from the last balanced print (100.1): one-sided burst
    burst = [_trade(60 + i, 100.2 + 0.1 * i, 100) for i in range(40)]
    bars = list(imbalance_bars(balanced + burst, **IMB))
    # no close during the balanced prefix; theta enters the burst at 2 and the
    # floored threshold is 5, so the 3rd burst print closes the first bar
    assert bars[0].ts == burst[2].ts
    assert bars[0].volume == (60 + 3) * 100
    assert len(bars) >= 2


def test_imbalance_rejects_bad_params() -> None:
    trades = [_trade(0, 100.0, 1)]
    bad = dict(IMB, alpha_ticks=1.5)
    with pytest.raises(ValueError):
        list(imbalance_bars(trades, **bad))
    with pytest.raises(ValueError):
        list(imbalance_bars(trades, **dict(IMB, expected_ticks_init=0.0)))


# ---------------------------------------------------------------------------
# Cross-builder invariants
# ---------------------------------------------------------------------------


def _last_constituent_ts(trades: Sequence[TradeEvent], bars: Sequence[BarEvent]) -> list[datetime]:
    """Recover each bar's last constituent trade by volume accounting: every
    builder consumes consecutive whole trades, so bar volumes partition a
    prefix of the stream."""
    out: list[datetime] = []
    i = 0
    for bar in bars:
        vol = 0
        while vol < bar.volume:
            vol += trades[i].size
            i += 1
        assert vol == bar.volume, "bar volume must be a whole-trade prefix sum"
        out.append(trades[i - 1].ts)
    return out


def test_bar_ts_never_precedes_last_constituent_trade() -> None:
    prices = [100.0, 100.5, 100.2, 101.0, 100.7, 101.5, 101.2, 102.0, 101.8, 102.5]
    sizes = [120, 80, 200, 60, 340, 90, 210, 150, 50, 400]
    trades = [_trade(7 * i, p, s) for i, (p, s) in enumerate(zip(prices, sizes))]
    built: dict[str, list[BarEvent]] = {
        "time": list(time_bars(trades, timedelta(seconds=21))),
        "volume": list(volume_bars(trades, volume_per_bar=400)),
        "dollar": list(dollar_bars(trades, dollar_per_bar=40_000.0)),
        "imbalance": list(imbalance_bars(trades, **dict(IMB, expected_ticks_init=4.0))),
    }
    for name, bars in built.items():
        assert bars, f"{name} bars: canned stream should emit at least one bar"
        for bar, last_ts in zip(bars, _last_constituent_ts(trades, bars)):
            assert bar.ts >= last_ts, f"{name} bar at {bar.ts} precedes its last trade {last_ts}"


def test_builders_reject_malformed_streams() -> None:
    out_of_order = [_trade(10, 100.0, 100), _trade(5, 100.0, 100)]
    with pytest.raises(ValueError, match="out-of-order"):
        list(volume_bars(out_of_order, volume_per_bar=1000))
    mixed = [_trade(0, 100.0, 100, symbol="SPY"), _trade(1, 100.0, 100, symbol="QQQ")]
    with pytest.raises(ValueError, match="mixed symbols"):
        list(time_bars(mixed, timedelta(minutes=1)))


def test_bars_to_frame() -> None:
    trades = [_trade(i, 100.0 + i, 100) for i in range(4)]
    frame = bars_to_frame(volume_bars(trades, volume_per_bar=200))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame.index.name == "ts"
    empty = bars_to_frame([])
    assert empty.empty and list(empty.columns) == ["open", "high", "low", "close", "volume"]

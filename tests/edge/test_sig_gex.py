"""GexPinning (gex_pin): crafted OI/gamma surfaces + spot paths near/far
from the pin; the OI availability lag (day-D OI never usable day D) enforced
inside the signal itself; non-expiry days never fire (structurally: no
expiry listed for the session date -> empty frames).

All data is synthetic and in-memory (no network, no loader); every
timestamp predates the 2026-02-22 lockbox and sits inside the development
span (2021 expiry cycle). The expiry day is Friday 2021-06-18; the
non-expiry day is Wednesday 2021-06-16.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import edge.signals.registry as sigreg
from edge.core.events import BarEvent, Side
from edge.regime.classifier import VOL_LOW, VOL_MID
from edge.research.registry import TrialRegistry
from edge.signals.base import MIN_HYPOTHESIS_CHARS, require_hypothesis
from edge.signals.gex_pin import GexPinConfig, GexPinning, HYPOTHESIS
from edge.signals.registry import run_signal

ET = ZoneInfo("America/New_York")

SYMBOL = "TST"
EXPIRY = date(2021, 6, 18)  # Friday: June 2021 monthly expiration
PRIOR = date(2021, 6, 17)  # Thursday: the last session before expiry
NON_EXPIRY = date(2021, 6, 16)  # Wednesday: nothing expires
EXPIRY_SYMBOL = f"{SYMBOL}:{EXPIRY.isoformat()}"

PIN = 100.0  # OI-weighted max-gamma strike in every crafted surface
DECOY = 105.0  # the strike day-D (leaked) data would move the pin to


def _ts(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)


def _bar(day: date, hh: int, mm: int, close: float) -> BarEvent:
    return BarEvent(
        ts=_ts(day, hh, mm),
        symbol=SYMBOL,
        open=close,
        high=close + 0.05,
        low=close - 0.05,
        close=close,
        volume=10_000,
    )


# ---------------------------------------------------------------------------
# Crafted point-in-time frames (loader/bridge shapes: asof_date naive
# midnight, available_at tz-aware ET; archive value columns ride along)
# ---------------------------------------------------------------------------


def greeks_frame(
    asof: date, available: datetime, spec: list[tuple[float, str, float]]
) -> pd.DataFrame:
    """EOD-greeks rows for one asof day: (strike, right, gamma) triples."""
    return pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(asof)] * len(spec),
            "available_at": [pd.Timestamp(available)] * len(spec),
            "symbol": EXPIRY_SYMBOL,
            "underlying": SYMBOL,
            "expiry": pd.Timestamp(EXPIRY),
            "strike": [s for s, _, _ in spec],
            "right": [r for _, r, _ in spec],
            "gamma": [g for _, _, g in spec],
            "implied_vol": 0.2,
        }
    )


def oi_frame(
    asof: date, available: datetime, spec: list[tuple[float, str, int]]
) -> pd.DataFrame:
    """Open-interest rows for one asof day: (strike, right, oi) triples."""
    return pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(asof)] * len(spec),
            "available_at": [pd.Timestamp(available)] * len(spec),
            "symbol": EXPIRY_SYMBOL,
            "underlying": SYMBOL,
            "expiry": pd.Timestamp(EXPIRY),
            "strike": [s for s, _, _ in spec],
            "right": [r for _, r, _ in spec],
            "open_interest": [n for _, _, n in spec],
        }
    )


def surface_frames() -> dict[tuple[str, str], pd.DataFrame]:
    """The honest expiry-morning view: prior-session greeks (published
    17th 18:00 ET) and prior-session OI (published 18th 09:00 ET — the OCC
    overnight cycle). OI-weighted gamma peaks at strike 100:

        95: 0.03*500 + 0.03*500   =    30
       100: 0.08*10000 + 0.08*9000 = 1520   <- the pin
       105: 0.03*800 + 0.03*700   =    45

    Right spellings deliberately differ across kinds (CALL/PUT vs C/P) to
    pin the join's normalization.
    """
    greeks = greeks_frame(
        PRIOR,
        _ts(PRIOR, 18, 0),
        [
            (95.0, "CALL", 0.03),
            (95.0, "PUT", 0.03),
            (100.0, "CALL", 0.08),
            (100.0, "PUT", 0.08),
            (105.0, "CALL", 0.03),
            (105.0, "PUT", 0.03),
        ],
    )
    oi = oi_frame(
        PRIOR,
        _ts(EXPIRY, 9, 0),
        [
            (95.0, "C", 500),
            (95.0, "P", 500),
            (100.0, "C", 10_000),
            (100.0, "P", 9_000),
            (105.0, "C", 800),
            (105.0, "P", 700),
        ],
    )
    return {("greeks_eod", EXPIRY_SYMBOL): greeks, ("open_interest", EXPIRY_SYMBOL): oi}


def leaked_day_d_frames() -> dict[tuple[str, str], pd.DataFrame]:
    """The same surface PLUS day-D rows that would move the pin to 105 —
    but which a live trader could not yet see: day-D OI is stamped the next
    business day (Monday 21st 09:00 ET), day-D greeks 18:00 ET after the
    close. If either leaks into the pin, it lands at 105 and a 100.3 spot
    is ~4.5% away -> no signal. The correct behavior is to ignore them and
    fade the 100 pin."""
    frames = surface_frames()
    leak_greeks = greeks_frame(
        EXPIRY, _ts(EXPIRY, 18, 0), [(105.0, "CALL", 5.0), (105.0, "PUT", 5.0)]
    )
    leak_oi = oi_frame(
        EXPIRY, _ts(date(2021, 6, 21), 9, 0), [(105.0, "C", 500_000), (105.0, "P", 500_000)]
    )
    frames[("greeks_eod", EXPIRY_SYMBOL)] = pd.concat(
        [frames[("greeks_eod", EXPIRY_SYMBOL)], leak_greeks], ignore_index=True
    )
    frames[("open_interest", EXPIRY_SYMBOL)] = pd.concat(
        [frames[("open_interest", EXPIRY_SYMBOL)], leak_oi], ignore_index=True
    )
    return frames


class Ctx:
    """Minimal SignalContext: decision_ts, bar, history, frame.

    ``filter_available=True`` mimics the engine (rows pre-filtered to
    ``available_at <= decision_ts``); ``False`` hands the signal the RAW
    frames — future rows included — to prove the signal re-filters itself.
    """

    def __init__(
        self,
        bar: BarEvent,
        frames: dict[tuple[str, str], pd.DataFrame] | None = None,
        *,
        filter_available: bool = True,
    ) -> None:
        self._bar = bar
        self._frames = frames or {}
        self._filter = filter_available
        self.requests: list[tuple[str, str]] = []

    @property
    def decision_ts(self) -> datetime:
        return self._bar.ts

    @property
    def bar(self) -> BarEvent:
        return self._bar

    def history(self, symbol: str, n_bars: int) -> pd.DataFrame:
        return pd.DataFrame()

    def frame(self, kind: str, symbol: str) -> pd.DataFrame:
        self.requests.append((kind, symbol))
        frame = self._frames.get(
            (kind, symbol),
            pd.DataFrame(
                columns=["asof_date", "available_at", "symbol", "underlying", "expiry"]
            ),
        )
        if self._filter and len(frame):
            visible = pd.to_datetime(frame["available_at"]) <= pd.Timestamp(
                self.decision_ts
            )
            frame = frame[visible]
        return frame.reset_index(drop=True).copy()


@pytest.fixture(autouse=True)
def _gex_pin_registered():
    """Re-register gex_pin if another test cleared the shared catalog."""
    if "gex_pin" not in sigreg.list_signals():
        sigreg.register(GexPinning)
    yield


def make_signal() -> GexPinning:
    return GexPinning()


# ---------------------------------------------------------------------------
# Registration + declarations (the hypothesis contract)
# ---------------------------------------------------------------------------


def test_registered_with_admissible_hypothesis() -> None:
    assert sigreg.get("gex_pin") is GexPinning
    assert require_hypothesis(GexPinning.hypothesis, owner="gex_pin") == HYPOTHESIS
    assert len(HYPOTHESIS.strip()) >= MIN_HYPOTHESIS_CHARS
    # The sign caveat and the fixed thresholds are written into the claim.
    assert "SIGN ASSUMPTION" in HYPOTHESIS
    assert "0.5%" in HYPOTHESIS
    assert "90+" in HYPOTHESIS
    assert "15:45" in HYPOTHESIS


def test_low_mid_vol_regimes_only() -> None:
    assert GexPinning.allowed_regimes == frozenset({VOL_LOW, VOL_MID})


def test_config_defaults_are_the_declared_economic_choices() -> None:
    cfg = GexPinConfig()
    assert cfg.max_pin_distance_pct == 0.5
    assert cfg.min_minutes_to_close == 90
    with pytest.raises(Exception):  # extra='forbid': typos never pass silently
        GexPinConfig(max_pin_distance=1.0)


# ---------------------------------------------------------------------------
# The fade: spot near/far from the crafted pin
# ---------------------------------------------------------------------------


def test_fades_short_above_pin_with_horizon_to_1545() -> None:
    ctx = Ctx(_bar(EXPIRY, 13, 30, 100.3), surface_frames())
    events = make_signal().on_bar(ctx)
    assert len(events) == 1
    event = events[0]
    assert event.side is Side.SELL  # above the pin: fade down toward it
    assert event.symbol == SYMBOL
    assert event.signal_name == "gex_pin"
    assert event.ts == ctx.decision_ts
    assert event.conviction == 1.0
    assert event.horizon_minutes == 135  # 13:30 -> 15:45 ET


def test_fades_long_below_pin() -> None:
    events = make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 99.8), surface_frames()))
    assert len(events) == 1
    assert events[0].side is Side.BUY  # below the pin: fade up toward it


def test_far_from_pin_never_fires() -> None:
    # 101.0 is ~0.99% above the 100 pin: outside the 0.5% band.
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 101.0), surface_frames())) == []


def test_exactly_at_pin_has_nothing_to_fade() -> None:
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, PIN), surface_frames())) == []


def test_fires_at_most_once_per_session() -> None:
    signal = make_signal()
    assert len(signal.on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), surface_frames()))) == 1
    assert signal.on_bar(Ctx(_bar(EXPIRY, 13, 31, 100.3), surface_frames())) == []


def test_zero_oi_weight_pins_nothing() -> None:
    frames = surface_frames()
    zeroed = frames[("open_interest", EXPIRY_SYMBOL)].copy()
    zeroed["open_interest"] = 0
    frames[("open_interest", EXPIRY_SYMBOL)] = zeroed
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), frames)) == []


# ---------------------------------------------------------------------------
# Session-time gates: expiry afternoons with 90+ minutes to the close
# ---------------------------------------------------------------------------


def test_under_90_minutes_to_close_never_fires() -> None:
    # 14:45 ET leaves 75 minutes to the close: inside the no-fire zone.
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 14, 45, 100.3), surface_frames())) == []


def test_exactly_90_minutes_to_close_fires() -> None:
    events = make_signal().on_bar(Ctx(_bar(EXPIRY, 14, 30, 100.3), surface_frames()))
    assert len(events) == 1
    assert events[0].horizon_minutes == 75  # 14:30 -> 15:45 ET


def test_morning_never_fires_even_near_pin() -> None:
    # The mechanism is an expiry-AFTERNOON one: 10:30 ET is out of window.
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 10, 30, 100.3), surface_frames())) == []


# ---------------------------------------------------------------------------
# OI availability lag: day-D OI is NEVER usable on day D
# ---------------------------------------------------------------------------


def test_day_d_rows_ignored_even_when_ctx_leaks_them() -> None:
    """The signal re-filters available_at itself (defense in depth): raw
    frames containing day-D OI/greeks that would move the pin to 105 must
    not change the trade — it still fades the visible 100 pin."""
    ctx = Ctx(_bar(EXPIRY, 13, 30, 100.3), leaked_day_d_frames(), filter_available=False)
    events = make_signal().on_bar(ctx)
    assert len(events) == 1
    assert events[0].side is Side.SELL  # 100.3 vs the 100 pin, not the 105 decoy


def test_day_d_rows_absent_under_engine_style_filtering_same_answer() -> None:
    filtered = make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), leaked_day_d_frames()))
    honest = make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), surface_frames()))
    assert [e.side for e in filtered] == [e.side for e in honest] == [Side.SELL]


def test_oi_published_only_after_expiry_is_unusable_that_day() -> None:
    """If the ONLY OI on file is day-D's own (stamped next-business-day
    09:00 ET), there is no visible snapshot on day D and nothing fires."""
    frames = surface_frames()
    frames[("open_interest", EXPIRY_SYMBOL)] = oi_frame(
        EXPIRY,
        _ts(date(2021, 6, 21), 9, 0),  # Fri OI -> Monday 09:00 ET
        [(100.0, "C", 10_000), (100.0, "P", 9_000)],
    )
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), frames)) == []


def test_prior_day_oi_not_yet_published_at_0859_is_invisible() -> None:
    """Before 09:00 ET even D-1's OI is unknown — and the afternoon gate
    aside, the pin computation itself must find no visible OI snapshot."""
    frames = surface_frames()
    bar = _bar(EXPIRY, 8, 59, 100.3)
    ctx = Ctx(bar, frames)
    # Bypass the time gates by checking the snapshot helper's verdict via
    # on_bar at an afternoon decision built on a frame whose OI publishes
    # later the same afternoon: shift the OI stamp to 14:00 ET...
    late = frames[("open_interest", EXPIRY_SYMBOL)].copy()
    late["available_at"] = pd.Timestamp(_ts(EXPIRY, 14, 0))
    frames[("open_interest", EXPIRY_SYMBOL)] = late
    # ...then decide at 13:30: the OI row exists but is not yet published.
    assert make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), frames)) == []
    # And at 08:59 nothing fires regardless (morning gate + no OI).
    assert make_signal().on_bar(ctx) == []


# ---------------------------------------------------------------------------
# Non-expiry days never fire
# ---------------------------------------------------------------------------


def test_non_expiry_day_never_fires() -> None:
    """On a non-expiry day no expiration equals the session date, both
    frames come back empty, and the signal cannot fire — even with spot
    parked exactly where an expiry-day pin would be."""
    ctx = Ctx(_bar(NON_EXPIRY, 13, 30, 100.3), surface_frames())
    assert make_signal().on_bar(ctx) == []
    # It asked for expiry == the SESSION date (the structural expiry test),
    # not for some other day's listed expiration.
    assert (
        "open_interest",
        f"{SYMBOL}:{NON_EXPIRY.isoformat()}",
    ) in ctx.requests
    assert all(sym.endswith(NON_EXPIRY.isoformat()) for _, sym in ctx.requests)


def test_expiry_day_requests_use_session_date_expiry_grammar() -> None:
    ctx = Ctx(_bar(EXPIRY, 13, 30, 100.3), surface_frames())
    make_signal().on_bar(ctx)
    assert ("open_interest", EXPIRY_SYMBOL) in ctx.requests
    assert ("greeks_eod", EXPIRY_SYMBOL) in ctx.requests


# ---------------------------------------------------------------------------
# Loud failures: an unstampable or column-less surface never fails silently
# ---------------------------------------------------------------------------


def test_missing_gamma_column_raises() -> None:
    frames = surface_frames()
    frames[("greeks_eod", EXPIRY_SYMBOL)] = frames[("greeks_eod", EXPIRY_SYMBOL)].drop(
        columns=["gamma"]
    )
    with pytest.raises(ValueError, match="gamma"):
        make_signal().on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), frames))


def test_naive_available_at_raises() -> None:
    frames = surface_frames()
    naive = frames[("open_interest", EXPIRY_SYMBOL)].copy()
    naive["available_at"] = pd.Timestamp(2021, 6, 18, 9, 0)  # tz-naive
    frames[("open_interest", EXPIRY_SYMBOL)] = naive
    with pytest.raises(ValueError, match="naive"):
        make_signal().on_bar(
            Ctx(_bar(EXPIRY, 13, 30, 100.3), frames, filter_available=False)
        )


# ---------------------------------------------------------------------------
# run_signal: the trial line is recorded BEFORE the backtest runs
# ---------------------------------------------------------------------------


def test_run_signal_records_trial_then_backtests(tmp_path) -> None:
    trials = TrialRegistry(root=tmp_path)
    signal = make_signal()

    def backtest(sig: GexPinning):
        return sig.on_bar(Ctx(_bar(EXPIRY, 13, 30, 100.3), surface_frames()))

    record, events = run_signal(
        signal,
        backtest,
        trials,
        run_config={"symbols": [SYMBOL], "span": "2021-06-14..2021-06-18"},
        ts=_ts(EXPIRY, 16, 0),  # pre-lockbox stamp
    )
    assert record.hypothesis == HYPOTHESIS
    assert record.result == {"status": "pending"}  # recorded BEFORE the run
    assert record.config["signal"] == "gex_pin"
    assert record.config["signal_config"] == {
        "max_pin_distance_pct": 0.5,
        "min_minutes_to_close": 90,
    }
    assert sorted(record.config["allowed_regimes"]) == sorted({VOL_LOW, VOL_MID})
    assert len(events) == 1 and events[0].side is Side.SELL

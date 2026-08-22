"""Per-expiry options point-in-time routing through the engine.

Covers the capability the six preloaded feed kinds could not provide: the
CatalystBridge EOD kinds (``greeks_eod`` / ``open_interest``, symbol grammar
``"UNDERLYING:YYYY-MM-DD"``) declared via ``EngineParams.options_pit`` and
served lazily per (symbol, ET session) through ``ctx.options_pit``. Proves:

* declaration validation (kinds, plain-underlying symbols, config hash);
* lazy load-once-per-(symbol, session) via the loader, reload on a new
  session with the extended range, and NO load without a request;
* visibility enforced ENGINE-side: only ``available_at <= now`` rows come
  back — in particular day-D open interest (stamped next-business-day
  09:00 ET by the bridge) is invisible for all of day D, proven both with
  a synthetic backend double and through the REAL CatalystBridge over an
  in-memory cache;
* the gex_pin signal's data path works end-to-end on a crafted surface
  (real bridge, cache-only, terminal-fetcher double never consulted).

ALL data is synthetic and in-memory, no network, every timestamp predates
the 2026-02-22 lockbox wall, and every write lands under pytest tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

from edge.core.config import (
    DataConfig,
    EdgeConfig,
    ExecutionConfig,
    RiskConfig,
    ValidationConfig,
)
from edge.core.events import SignalEvent
from edge.data.backends import (
    CATEGORY_GREEKS_EOD,
    CATEGORY_OPEN_INTEREST,
    CatalystBridge,
    expiry_day_key,
    greeks_eod_key,
)
from edge.data.loader import ALL_SYMBOLS, EdgeDataLoader
from edge.runners.engine import (
    OPTIONS_PIT_KINDS,
    BacktestEngine,
    EngineParams,
    OptionsPitSpec,
    SizeDecision,
)
from edge.signals.gex_pin import GexPinning

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

SYMBOL = "TST"
THU = date(2021, 6, 17)  # the session before expiry
FRI = date(2021, 6, 18)  # Friday: June 2021 monthly expiration
MON_AFTER = date(2021, 6, 21)  # day-FRI OI first becomes visible
EXPIRY_SYMBOL = f"{SYMBOL}:{FRI.isoformat()}"
LATENCY_MS = 100


def TS(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)


# ---------------------------------------------------------------------------
# Config / repo-root plumbing (same conventions as test_engine)
# ---------------------------------------------------------------------------


def make_repo_root(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "edge.yaml").write_text((REPO_ROOT / "config" / "edge.yaml").read_text())
    return tmp_path


def make_config() -> EdgeConfig:
    return EdgeConfig(
        data=DataConfig(lockbox_start=date(2026, 2, 22)),
        validation=ValidationConfig(min_oos_trades=1, pbo_max=0.5, bootstrap_resamples=10),
        execution=ExecutionConfig(
            spread_fill_fraction=1.0,
            slippage_pct=0.0,
            commission_per_contract=0.01,
            latency_ms=[LATENCY_MS],
        ),
        risk=RiskConfig(
            kelly_fraction=0.25,
            per_trade_cap_pct=2.0,
            daily_loss_halt_pct=50.0,
            target_daily_vol_pct=None,
        ),
    )


def base_params(**overrides) -> EngineParams:
    fields = {
        "symbols": (SYMBOL,),
        "start": THU,
        "end": FRI,
        "latency_ms": LATENCY_MS,
        "initial_equity": 100_000.0,
        "seed": 0,
        "risk_r_pct": 1.0,
    }
    fields.update(overrides)
    return EngineParams(**fields)


def declare(*kinds: str) -> tuple[OptionsPitSpec, ...]:
    return tuple(OptionsPitSpec(kind=kind, symbols=(SYMBOL,)) for kind in kinds)


# ---------------------------------------------------------------------------
# Tapes: minute bars + quotes for the underlying
# ---------------------------------------------------------------------------


def bars_frame(stamps: list[datetime], close: float = 100.3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": stamps,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 10_000,
        }
    )


def quotes_from_bars(bars: pd.DataFrame, half_spread: float = 0.05) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": bars["ts"],
            "bid": bars["close"] - half_spread,
            "ask": bars["close"] + half_spread,
            "bid_size": 50,
            "ask_size": 50,
        }
    )


FRI_STAMPS = [TS(FRI, 13, 30), TS(FRI, 13, 31), TS(FRI, 13, 32)]
THU_STAMPS = [TS(THU, 13, 30), TS(THU, 13, 31), TS(THU, 13, 32)]


def tape_frames(stamps: list[datetime]) -> dict[tuple[str, str], pd.DataFrame]:
    bars = bars_frame(stamps)
    return {(SYMBOL, "bars"): bars, (SYMBOL, "quotes"): quotes_from_bars(bars)}


# ---------------------------------------------------------------------------
# Crafted per-expiry frames (bridge output shapes: asof naive midnight,
# available_at tz-aware ET; value columns ride along)
# ---------------------------------------------------------------------------


def greeks_frame(
    asof: date, available: datetime, spec: list[tuple[float, str, float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(asof)] * len(spec),
            "available_at": [pd.Timestamp(available)] * len(spec),
            "symbol": EXPIRY_SYMBOL,
            "underlying": SYMBOL,
            "expiry": pd.Timestamp(FRI),
            "strike": [s for s, _, _ in spec],
            "right": [r for _, r, _ in spec],
            "gamma": [g for _, _, g in spec],
        }
    )


def oi_frame(
    asof: date, available: datetime, spec: list[tuple[float, str, int]]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(asof)] * len(spec),
            "available_at": [pd.Timestamp(available)] * len(spec),
            "symbol": EXPIRY_SYMBOL,
            "underlying": SYMBOL,
            "expiry": pd.Timestamp(FRI),
            "strike": [s for s, _, _ in spec],
            "right": [r for _, r, _ in spec],
            "open_interest": [n for _, _, n in spec],
        }
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class OptionsTapeBackend:
    """Synthetic bridge double: records fetches; serves per-(symbol, kind)
    frames, row-filtering the options kinds to asof within the (gated)
    requested range exactly as the real bridge does."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, date, date, str]] = []

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        self.calls.append((symbol, start, end, kind))
        frame = self.frames.get((symbol, kind))
        if frame is None:
            return pd.DataFrame()
        if kind in OPTIONS_PIT_KINDS and len(frame):
            asof = pd.to_datetime(frame["asof_date"])
            mask = (asof >= pd.Timestamp(start)) & (asof <= pd.Timestamp(end))
            frame = frame[mask]
        return frame.reset_index(drop=True).copy()

    def options_calls(self) -> list[tuple[str, date, date, str]]:
        return [c for c in self.calls if c[3] in OPTIONS_PIT_KINDS]


class FakeCache:
    """In-memory (category, key) -> raw frame store for the real bridge."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = dict(frames)
        self.get_calls: list[tuple[str, str]] = []

    def get(self, category: str, key: str) -> pd.DataFrame | None:
        self.get_calls.append((category, key))
        frame = self.frames.get((category, key))
        return None if frame is None else frame.copy()

    def put(self, category: str, key: str, df: pd.DataFrame) -> None:
        self.frames[(category, key)] = df.copy()


class TerminalTripwireFetcher:
    """A CatalystFetcher double whose every method records and fails loudly:
    the cache-only contract means NONE of these may ever run."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _trip(self, name: str, *args) -> pd.DataFrame:
        self.calls.append((name, *args))
        raise AssertionError(f"terminal fetch attempted: {name}{args}")

    def stock_minute_day(self, *args):
        return self._trip("stock_minute_day", *args)

    def option_quote_day(self, *args):
        return self._trip("option_quote_day", *args)

    def greeks_eod_frame(self, *args):
        return self._trip("greeks_eod_frame", *args)

    def open_interest_day(self, *args):
        return self._trip("open_interest_day", *args)

    def option_eod_day(self, *args):
        return self._trip("option_eod_day", *args)


class BridgeRoutingBackend:
    """bars/quotes from the in-memory tape; options kinds through a REAL
    CatalystBridge (in-memory cache, cache-only)."""

    def __init__(
        self, tape: dict[tuple[str, str], pd.DataFrame], bridge: CatalystBridge
    ) -> None:
        self._tape = tape
        self._bridge = bridge

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        if kind in OPTIONS_PIT_KINDS:
            return self._bridge.fetch(symbol, start, end, kind)
        return self._tape.get((symbol, kind), pd.DataFrame())


# ---------------------------------------------------------------------------
# Probe signal + fixed sizer
# ---------------------------------------------------------------------------


class OptionsProbe:
    """Requests one options pit frame per bar and records what came back."""

    name = "options_probe"

    def __init__(self, kind: str, symbol: str, *, request: bool = True) -> None:
        self.kind = kind
        self.symbol = symbol
        self.request = request
        self.observations: list[tuple[datetime, pd.DataFrame]] = []

    def on_bar(self, ctx) -> None:
        if not self.request:
            return
        self.observations.append((ctx.now, ctx.options_pit(self.kind, self.symbol)))
        return


class FixedSizer:
    def __init__(self, qty: int, risk_per_share: float) -> None:
        self.qty = qty
        self.risk_per_share = risk_per_share

    def size(self, signal: SignalEvent, ref_price: float, equity: float) -> SizeDecision:
        return SizeDecision(qty=self.qty, risk_per_share=self.risk_per_share)


def make_engine(
    tmp_path: Path,
    backend,
    params: EngineParams,
    signals: list,
    **hooks,
) -> BacktestEngine:
    root = make_repo_root(tmp_path)
    loader = EdgeDataLoader(backend, repo_root=root)
    return BacktestEngine(loader, make_config(), params, signals, **hooks)


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------


def test_spec_rejects_unknown_kind_and_empty_symbols() -> None:
    with pytest.raises(ValidationError):  # Literal: option_eod is not served
        OptionsPitSpec(kind="option_eod", symbols=(SYMBOL,))
    with pytest.raises(ValidationError):
        OptionsPitSpec(kind="open_interest", symbols=())
    # dicts coerce: config-driven construction needs no imports
    params = base_params(options_pit=({"kind": "greeks_eod", "symbols": [SYMBOL]},))
    assert params.options_pit == (OptionsPitSpec(kind="greeks_eod", symbols=(SYMBOL,)),)


def test_declared_symbols_must_be_plain_underlyings(tmp_path: Path) -> None:
    params = base_params(
        options_pit=(OptionsPitSpec(kind="open_interest", symbols=(EXPIRY_SYMBOL,)),)
    )
    with pytest.raises(ValueError, match="plain underlyings"):
        make_engine(tmp_path, OptionsTapeBackend({}), params, [])


def test_declaration_shapes_the_config_hash(tmp_path: Path) -> None:
    backend = OptionsTapeBackend(tape_frames(FRI_STAMPS))
    plain = make_engine(tmp_path, backend, base_params(), []).run()
    declared = make_engine(
        tmp_path, backend, base_params(options_pit=declare("open_interest")), []
    ).run()
    assert plain.config_hash != declared.config_hash


# ---------------------------------------------------------------------------
# Request gating: only declared (kind, underlying) pairs are served
# ---------------------------------------------------------------------------


def test_undeclared_kind_raises_with_guidance(tmp_path: Path) -> None:
    probe = OptionsProbe("open_interest", EXPIRY_SYMBOL)
    engine = make_engine(
        tmp_path, OptionsTapeBackend(tape_frames(FRI_STAMPS)), base_params(), [probe]
    )
    with pytest.raises(KeyError, match="options_pit"):
        engine.run()


def test_undeclared_underlying_raises(tmp_path: Path) -> None:
    probe = OptionsProbe("open_interest", f"OTHER:{FRI.isoformat()}")
    engine = make_engine(
        tmp_path,
        OptionsTapeBackend(tape_frames(FRI_STAMPS)),
        base_params(options_pit=declare("open_interest")),
        [probe],
    )
    with pytest.raises(KeyError, match="OTHER"):
        engine.run()


# ---------------------------------------------------------------------------
# Lazy loading: once per (symbol, session), nothing without a request
# ---------------------------------------------------------------------------


def test_no_request_means_no_load(tmp_path: Path) -> None:
    backend = OptionsTapeBackend(tape_frames(FRI_STAMPS))
    probe = OptionsProbe("greeks_eod", EXPIRY_SYMBOL, request=False)
    make_engine(
        tmp_path, backend, base_params(options_pit=declare("greeks_eod")), [probe]
    ).run()
    assert backend.options_calls() == []  # declared but never requested: no load


def test_loads_once_per_symbol_and_session(tmp_path: Path) -> None:
    frames = tape_frames(FRI_STAMPS)
    frames[(EXPIRY_SYMBOL, "greeks_eod")] = greeks_frame(
        THU, TS(THU, 18, 0), [(100.0, "CALL", 0.08)]
    )
    backend = OptionsTapeBackend(frames)
    probe = OptionsProbe("greeks_eod", EXPIRY_SYMBOL)
    make_engine(
        tmp_path, backend, base_params(options_pit=declare("greeks_eod")), [probe]
    ).run()
    # Three bars requested; the engine loaded exactly once, range clamped to
    # the session (never past the current ET day, never past params.end).
    assert backend.options_calls() == [(EXPIRY_SYMBOL, THU, FRI, "greeks_eod")]
    assert len(probe.observations) == 3
    for _now, frame in probe.observations:
        assert list(frame["asof_date"].dt.date) == [THU]


def test_new_session_reloads_with_extended_range(tmp_path: Path) -> None:
    frames = tape_frames(THU_STAMPS + FRI_STAMPS)
    frames[(EXPIRY_SYMBOL, "greeks_eod")] = pd.concat(
        [
            greeks_frame(THU, TS(THU, 18, 0), [(100.0, "CALL", 0.08)]),
            greeks_frame(FRI, TS(FRI, 18, 0), [(105.0, "CALL", 5.0)]),
        ],
        ignore_index=True,
    )
    backend = OptionsTapeBackend(frames)
    probe = OptionsProbe("greeks_eod", EXPIRY_SYMBOL)
    make_engine(
        tmp_path, backend, base_params(options_pit=declare("greeks_eod")), [probe]
    ).run()
    # One load per ET session; the second extends the range to the new day.
    assert backend.options_calls() == [
        (EXPIRY_SYMBOL, THU, THU, "greeks_eod"),
        (EXPIRY_SYMBOL, THU, FRI, "greeks_eod"),
    ]
    by_session: dict[date, list[pd.DataFrame]] = {}
    for now, frame in probe.observations:
        by_session.setdefault(now.astimezone(ET).date(), []).append(frame)
    # THU intraday: THU's own greeks (stamped 18:00) are not yet visible.
    assert all(frame.empty for frame in by_session[THU])
    # FRI intraday: THU rows visible, FRI rows (18:00 that evening) not.
    for frame in by_session[FRI]:
        assert list(frame["asof_date"].dt.date) == [THU]


# ---------------------------------------------------------------------------
# Visibility is enforced engine-side: day-D OI invisible on day D
# ---------------------------------------------------------------------------


def test_day_d_oi_invisible_through_ctx_synthetic(tmp_path: Path) -> None:
    """The backend SERVES the day-D OI row (its stamp: next-bday 09:00 ET);
    the ctx must still never show it on day D."""
    frames = tape_frames(FRI_STAMPS)
    frames[(EXPIRY_SYMBOL, "open_interest")] = pd.concat(
        [
            oi_frame(THU, TS(FRI, 9, 0), [(100.0, "C", 10_000)]),
            oi_frame(FRI, TS(MON_AFTER, 9, 0), [(105.0, "C", 500_000)]),
        ],
        ignore_index=True,
    )
    backend = OptionsTapeBackend(frames)
    probe = OptionsProbe("open_interest", EXPIRY_SYMBOL)
    make_engine(
        tmp_path, backend, base_params(options_pit=declare("open_interest")), [probe]
    ).run()
    # The load range includes FRI, so the raw frame the engine holds DOES
    # contain the day-FRI row — the ctx filter is what hides it.
    assert backend.options_calls() == [(EXPIRY_SYMBOL, THU, FRI, "open_interest")]
    assert len(probe.observations) == 3
    for now, frame in probe.observations:
        assert now.astimezone(ET).date() == FRI
        assert list(frame["asof_date"].dt.date) == [THU]  # D-1 only, never day D
        assert 500_000 not in set(frame["open_interest"])


def test_day_d_oi_invisible_through_real_bridge(tmp_path: Path) -> None:
    """End to end through the REAL CatalystBridge: raw archive frames in an
    in-memory cache, the bridge stamps day-D OI next-business-day 09:00 ET,
    and ctx.options_pit never shows it on day D. Cache-only: the terminal
    fetcher double is never consulted."""
    raw_oi_thu = pd.DataFrame(
        {"strike": [100.0], "right": ["CALL"], "open_interest": [10_000]}
    )
    raw_oi_fri = pd.DataFrame(
        {"strike": [105.0], "right": ["CALL"], "open_interest": [500_000]}
    )
    cache = FakeCache(
        {
            (CATEGORY_OPEN_INTEREST, expiry_day_key(SYMBOL, FRI, THU)): raw_oi_thu,
            (CATEGORY_OPEN_INTEREST, expiry_day_key(SYMBOL, FRI, FRI)): raw_oi_fri,
        }
    )
    fetcher = TerminalTripwireFetcher()
    bridge = CatalystBridge(cache, fetcher, allow_fetch=False)
    backend = BridgeRoutingBackend(tape_frames(FRI_STAMPS), bridge)
    probe = OptionsProbe("open_interest", EXPIRY_SYMBOL)
    make_engine(
        tmp_path, backend, base_params(options_pit=declare("open_interest")), [probe]
    ).run()
    assert fetcher.calls == []  # cache-only: no terminal fetch, ever
    assert len(probe.observations) == 3
    for _now, frame in probe.observations:
        # Bridge stamped THU's OI available FRI 09:00 — visible intraday FRI.
        assert list(pd.to_datetime(frame["asof_date"]).dt.date) == [THU]
        assert frame["available_at"].iloc[0] == pd.Timestamp(
            f"{FRI} 09:00", tz="America/New_York"
        )
        # Day-FRI OI (stamped MON 09:00) is in the cache but never in the ctx.
        assert 500_000 not in set(frame["open_interest"])
    assert (CATEGORY_OPEN_INTEREST, expiry_day_key(SYMBOL, FRI, FRI)) in cache.frames


# ---------------------------------------------------------------------------
# ctx.frame routing: options kinds -> options_pit, feed kinds -> pit
# ---------------------------------------------------------------------------


def test_frame_routes_options_and_feed_kinds(tmp_path: Path) -> None:
    frames = tape_frames(FRI_STAMPS)
    frames[(EXPIRY_SYMBOL, "greeks_eod")] = greeks_frame(
        THU, TS(THU, 18, 0), [(100.0, "CALL", 0.08)]
    )
    frames[(ALL_SYMBOLS, "vol_indices")] = pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(THU)],
            "available_at": [TS(THU, 18, 0)],
            "vix": [15.0],
        }
    )

    class RoutingProbe:
        name = "routing_probe"

        def __init__(self) -> None:
            self.frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

        def on_bar(self, ctx) -> None:
            self.frames.append(
                (
                    ctx.frame("greeks_eod", EXPIRY_SYMBOL),
                    ctx.frame("vol_indices"),
                    ctx.pit("vol_indices"),
                )
            )

    probe = RoutingProbe()
    make_engine(
        tmp_path,
        OptionsTapeBackend(frames),
        base_params(
            options_pit=declare("greeks_eod"), pit_kinds=("vol_indices",)
        ),
        [probe],
    ).run()
    options, via_frame, via_pit = probe.frames[0]
    assert list(options["gamma"]) == [0.08]
    pd.testing.assert_frame_equal(via_frame, via_pit)  # feed kinds: same path


def test_frame_requires_per_expiry_symbol_for_options_kinds(tmp_path: Path) -> None:
    class NoSymbol:
        name = "no_symbol"

        def on_bar(self, ctx) -> None:
            ctx.frame("greeks_eod")

    engine = make_engine(
        tmp_path,
        OptionsTapeBackend(tape_frames(FRI_STAMPS)),
        base_params(options_pit=declare("greeks_eod")),
        [NoSymbol()],
    )
    with pytest.raises(ValueError, match="per-expiry symbol"):
        engine.run()


# ---------------------------------------------------------------------------
# gex_pin end to end: crafted surface, real bridge, real engine pipeline
# ---------------------------------------------------------------------------


def test_gex_pin_data_path_end_to_end(tmp_path: Path) -> None:
    """The full chain: GexPinning -> ctx.frame -> engine options_pit ->
    loader -> REAL CatalystBridge -> in-memory raw archive frames.

    Surface: OI-weighted gamma pins strike 100 on the honest (visible)
    THU-asof data; leaked day-FRI rows would move the pin to 105 (spot
    100.3 would then be ~4.5% away and nothing could fire). A SELL fade at
    13:30 FRI therefore proves the leak-free pin AND the visibility chain.
    THU's own session requests TST:2021-06-17 — no such expiry is cached,
    the miss is skipped (missing_ok), the frames come back empty, and the
    signal structurally cannot fire on a non-expiry day.
    """
    raw_greeks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [datetime.combine(THU, time(15, 59, 55))] * 6
                + [datetime.combine(FRI, time(15, 59, 55))] * 2
            ),
            "strike": [95.0, 95.0, 100.0, 100.0, 105.0, 105.0, 105.0, 105.0],
            "right": ["CALL", "PUT"] * 4,
            "gamma": [0.03, 0.03, 0.08, 0.08, 0.03, 0.03, 5.0, 5.0],
        }
    )
    raw_oi_thu = pd.DataFrame(
        {
            "strike": [95.0, 95.0, 100.0, 100.0, 105.0, 105.0],
            "right": ["C", "P"] * 3,
            "open_interest": [500, 500, 10_000, 9_000, 800, 700],
        }
    )
    raw_oi_fri = pd.DataFrame(  # day-D leak decoy: would move the pin to 105
        {"strike": [105.0, 105.0], "right": ["C", "P"], "open_interest": [500_000] * 2}
    )
    cache = FakeCache(
        {
            (CATEGORY_GREEKS_EOD, greeks_eod_key(SYMBOL, FRI)): raw_greeks,
            (CATEGORY_OPEN_INTEREST, expiry_day_key(SYMBOL, FRI, THU)): raw_oi_thu,
            (CATEGORY_OPEN_INTEREST, expiry_day_key(SYMBOL, FRI, FRI)): raw_oi_fri,
        }
    )
    fetcher = TerminalTripwireFetcher()
    bridge = CatalystBridge(cache, fetcher, allow_fetch=False, missing_ok=True)
    backend = BridgeRoutingBackend(tape_frames(THU_STAMPS + FRI_STAMPS), bridge)
    engine = make_engine(
        tmp_path,
        backend,
        base_params(options_pit=declare("greeks_eod", "open_interest")),
        [GexPinning()],
        sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()

    assert fetcher.calls == []  # cache-only end to end: terminal never touched
    # One thesis, one trade: fired FRI 13:30 only (THU is a non-expiry day —
    # its expiry symbol has no cached surface, the frames were empty).
    assert result.emitted == 1 and result.executed == 1
    assert sum(result.drop_counters.values()) == 0
    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    assert row.symbol == SYMBOL and row.signal_name == "gex_pin"
    assert row.side == "sell"  # spot 100.3 above the 100 pin: fade down
    assert row.entry_ts.astimezone(ET).date() == FRI
    assert row.exit_reason == "end_of_data"
    # emitted == executed + drops held (the engine asserts it; restate here)
    assert result.emitted == result.executed + sum(result.drop_counters.values())

"""Event-driven backtest engine tests: a crafted 3-symbol synthetic minute
tape with planted signal timings. Proves: fills price off the NEXT bar's
quote (never same-bar), MFE/MAE match hand computation on the bar path,
per-fill cost attribution sums into the ledger, the drop-counter identity
holds, runs are byte-deterministic, EOD flatten fires, and stops trigger on
the bar path (low/high), not the close.

ALL data is synthetic, no network, every timestamp strictly predates the
2026-02-22 lockbox wall, and every write lands under pytest's tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.core.config import (
    DataConfig,
    EdgeConfig,
    ExecutionConfig,
    RiskConfig,
    ValidationConfig,
)
from edge.core.events import Side, SignalEvent
from edge.data.loader import ALL_SYMBOLS, EdgeDataLoader
from edge.execution.costmodel import CostModel, QuoteEvent
from edge.runners import backtest as backtest_cli
from edge.runners.engine import (
    DROP_STAGES,
    LEDGER_COLUMNS,
    BacktestEngine,
    EngineInvariantError,
    EngineParams,
    RiskBudgetSizer,
    RunResult,
    SizeDecision,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
D0 = date(2026, 1, 6)  # Tuesday, well before the 2026-02-22 lockbox wall
LATENCY_MS = 100
LATENCY = timedelta(milliseconds=LATENCY_MS)


def T(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 1, 6, hh, mm, ss, tzinfo=ET)


# ---------------------------------------------------------------------------
# Synthetic tape plumbing
# ---------------------------------------------------------------------------


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def make_repo_root(tmp_path: Path) -> Path:
    """Tmp repo root with the real edge.yaml copied in (wall 2026-02-22)."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "edge.yaml").write_text((REPO_ROOT / "config" / "edge.yaml").read_text())
    return tmp_path


def make_config(
    *,
    spread_fill_fraction: float = 1.0,
    slippage_pct: float = 0.0,
    commission: float = 0.01,
    daily_loss_halt_pct: float = 50.0,
) -> EdgeConfig:
    """Hand-computable frictions: full crossing, no slip, 1c/share unless overridden."""
    return EdgeConfig(
        data=DataConfig(lockbox_start=date(2026, 2, 22)),
        validation=ValidationConfig(min_oos_trades=1, pbo_max=0.5, bootstrap_resamples=10),
        execution=ExecutionConfig(
            spread_fill_fraction=spread_fill_fraction,
            slippage_pct=slippage_pct,
            commission_per_contract=commission,
            latency_ms=[LATENCY_MS],
        ),
        risk=RiskConfig(
            kelly_fraction=0.25,
            per_trade_cap_pct=2.0,
            daily_loss_halt_pct=daily_loss_halt_pct,
            target_daily_vol_pct=None,
        ),
    )


def bars_frame(rows: list[tuple[datetime, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": 100,
        }
    )


def quotes_from_bars(bars: pd.DataFrame, half_spread: float = 0.10) -> pd.DataFrame:
    """One quote per bar close ts: bid/ask straddling that bar's close."""
    return pd.DataFrame(
        {
            "ts": bars["ts"],
            "bid": bars["close"] - half_spread,
            "ask": bars["close"] + half_spread,
            "bid_size": 50,
            "ask_size": 50,
        }
    )


def flat_bars(price: float, stamps: list[datetime]) -> pd.DataFrame:
    return bars_frame([(ts, price, price, price, price) for ts in stamps])


#: Canonical AAA minute bars. Decoys prove path exclusion: the DECISION bar
#: (09:33) carries a fake low of 90 and the EXIT-FILL bar (09:38) fake
#: extremes 105/95 — neither may leak into MFE/MAE.
AAA_ROWS: list[tuple[datetime, float, float, float, float]] = [
    (T(9, 31), 100.0, 100.2, 99.8, 100.0),
    (T(9, 32), 100.0, 100.3, 99.9, 100.1),
    (T(9, 33), 100.1, 100.5, 90.0, 100.2),
    (T(9, 34), 100.2, 101.0, 100.1, 100.8),
    (T(9, 35), 100.8, 101.5, 100.4, 101.0),
    (T(9, 36), 101.0, 101.2, 99.9, 100.5),
    (T(9, 37), 100.5, 100.9, 100.3, 100.6),
    (T(9, 38), 100.6, 105.0, 95.0, 100.4),
]
STAMPS = [row[0] for row in AAA_ROWS]


def canonical_frames(
    *,
    bbb_price: float = 50.0,
    ccc_quotes: bool = True,
    aaa_rows: list[tuple[datetime, float, float, float, float]] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    aaa = bars_frame(aaa_rows if aaa_rows is not None else AAA_ROWS)
    stamps = list(aaa["ts"])
    bbb = flat_bars(bbb_price, stamps)
    ccc = flat_bars(200.0, stamps)
    frames = {
        ("AAA", "bars"): aaa,
        ("AAA", "quotes"): quotes_from_bars(aaa),
        ("BBB", "bars"): bbb,
        ("BBB", "quotes"): quotes_from_bars(bbb),
        ("CCC", "bars"): ccc,
    }
    if ccc_quotes:
        frames[("CCC", "quotes")] = quotes_from_bars(ccc)
    return frames


class PlantedSignal:
    """Emits exactly the planted (symbol, bar ts) -> (side, conviction) events."""

    name = "planted"

    def __init__(self, plan: dict[tuple[str, datetime], tuple[Side, float]]) -> None:
        self.plan = plan

    def on_bar(self, ctx) -> SignalEvent | None:
        planted = self.plan.get((ctx.bar.symbol, ctx.bar.ts))
        if planted is None:
            return None
        side, conviction = planted
        return SignalEvent(
            ts=ctx.now,
            symbol=ctx.bar.symbol,
            side=side,
            conviction=conviction,
            horizon_minutes=30,
            signal_name=self.name,
        )


class FixedSizer:
    """Deterministic test sizer: fixed qty and risk unit; conviction 0 -> 0."""

    def __init__(self, qty: int, risk_per_share: float) -> None:
        self.qty = qty
        self.risk_per_share = risk_per_share

    def size(self, signal: SignalEvent, ref_price: float, equity: float) -> SizeDecision:
        qty = 0 if signal.conviction == 0.0 else self.qty
        return SizeDecision(qty=qty, risk_per_share=self.risk_per_share)


def make_engine(
    tmp_path: Path,
    frames: dict[tuple[str, str], pd.DataFrame],
    params: EngineParams,
    signals: list,
    config: EdgeConfig | None = None,
    **hooks,
) -> BacktestEngine:
    root = make_repo_root(tmp_path)
    loader = EdgeDataLoader(TapeBackend(frames), repo_root=root)
    return BacktestEngine(loader, config or make_config(), params, signals, **hooks)


def base_params(**overrides) -> EngineParams:
    fields = {
        "symbols": ("AAA", "BBB", "CCC"),
        "start": D0,
        "end": D0,
        "latency_ms": LATENCY_MS,
        "initial_equity": 100_000.0,
        "seed": 0,
        "risk_r_pct": 1.0,
    }
    fields.update(overrides)
    return EngineParams(**fields)


def run_long_round_trip(tmp_path: Path) -> RunResult:
    """The canonical long trade: BUY AAA at 09:33 close, max-hold 3 min exit."""
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}
    engine = make_engine(
        tmp_path,
        canonical_frames(),
        base_params(max_hold_minutes=3),
        [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
    )
    return engine.run()


# ---------------------------------------------------------------------------
# Fills price off the NEXT bar's quote — never the decision bar's
# ---------------------------------------------------------------------------


def test_entry_fill_uses_next_bar_quote_never_same_bar(tmp_path: Path) -> None:
    result = run_long_round_trip(tmp_path)
    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    # Ask at the 09:34 (next) bar is 100.9; ask at the 09:33 decision bar is 100.3.
    assert row.entry_px == pytest.approx(100.9)
    assert row.entry_px != pytest.approx(100.3)
    assert row.entry_ts == T(9, 33) + LATENCY  # arrival instant, not bar close


def test_exit_fill_uses_next_bar_quote(tmp_path: Path) -> None:
    result = run_long_round_trip(tmp_path)
    row = result.ledger.iloc[0]
    # Max-hold(3) fires at the 09:37 close; the SELL fills at the 09:38 bid
    # (100.3), not the 09:37 bid (100.5).
    assert row.exit_reason == "max_hold"
    assert row.exit_ts == T(9, 37) + LATENCY
    assert row.exit_px == pytest.approx(100.3)
    assert row.exit_px != pytest.approx(100.5)


def test_signal_on_last_bar_never_fills(tmp_path: Path) -> None:
    plan = {("AAA", T(9, 38)): (Side.BUY, 1.0)}  # last bar: no next bar exists
    engine = make_engine(
        tmp_path, canonical_frames(), base_params(), [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()
    assert len(result.ledger) == 0
    assert result.drop_counters["latency-no-fill"] == 1
    assert result.emitted == 1 and result.executed == 0


# ---------------------------------------------------------------------------
# Ledger economics, hand-computed
# ---------------------------------------------------------------------------


def test_long_round_trip_ledger_hand_computed(tmp_path: Path) -> None:
    result = run_long_round_trip(tmp_path)
    assert list(result.ledger.columns) == list(LEDGER_COLUMNS)
    row = result.ledger.iloc[0]
    assert row.symbol == "AAA" and row.side == "buy" and row.qty == 10
    assert row.signal_name == "planted" and row.conviction == 1.0
    # entry 100.9 (ask@09:34), exit 100.3 (bid@09:38), 1c/share each way.
    assert row.pnl == pytest.approx((100.3 - 100.9) * 10 - 0.2)
    assert row.r_multiple == pytest.approx(-6.2 / 10.0)  # risk $1/share * 10 shares
    assert row.spread_cost == pytest.approx(2.0)  # half-spread 0.10 * 10, both ways
    assert row.slippage_cost == pytest.approx(0.0)
    assert row.commission == pytest.approx(0.2)
    assert row.holding_minutes == pytest.approx(4.0)
    # Equity closes the books: final equity == initial + sum of ledger pnl.
    assert result.equity.iloc[-1] == pytest.approx(100_000.0 + result.ledger.pnl.sum())
    assert len(result.equity) == len(STAMPS)


def test_mfe_mae_hand_computed_on_held_path(tmp_path: Path) -> None:
    result = run_long_round_trip(tmp_path)
    row = result.ledger.iloc[0]
    # Held path = bars closing in [entry_ts, exit_ts] = 09:34..09:37.
    # highs 101.0/101.5/101.2/100.9 -> 101.5; lows 100.1/100.4/99.9/100.3 -> 99.9.
    # The 09:33 decoy low (90) and 09:38 decoy extremes (105/95) are excluded.
    assert row.mfe == pytest.approx(101.5 - 100.9)
    assert row.mae == pytest.approx(100.9 - 99.9)


def test_short_round_trip_hand_computed(tmp_path: Path) -> None:
    plan = {("AAA", T(9, 33)): (Side.SELL, 1.0)}
    engine = make_engine(
        tmp_path, canonical_frames(), base_params(max_hold_minutes=2),
        [PlantedSignal(plan)], sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()
    row = result.ledger.iloc[0]
    assert row.side == "sell"
    assert row.entry_px == pytest.approx(100.7)  # bid @ 09:34
    assert row.exit_px == pytest.approx(100.7)   # ask @ 09:37 (close 100.6 + 0.10)
    assert row.exit_ts == T(9, 36) + LATENCY
    assert row.pnl == pytest.approx(0.0 * 10 - 0.2)
    # Short MFE/MAE mirror: path 09:34..09:36, min low 99.9, max high 101.5.
    assert row.mfe == pytest.approx(100.7 - 99.9)
    assert row.mae == pytest.approx(101.5 - 100.7)
    assert row.holding_minutes == pytest.approx(3.0)


def test_attribution_sums_per_fill_costs(tmp_path: Path) -> None:
    """With real frictions on, ledger costs equal the cost model's own sums."""
    config = make_config(spread_fill_fraction=0.6, slippage_pct=0.02, commission=0.65)
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}
    engine = make_engine(
        tmp_path, canonical_frames(), base_params(max_hold_minutes=3),
        [PlantedSignal(plan)], config=config, sizer=FixedSizer(2, 1.0),
    )
    result = engine.run()
    row = result.ledger.iloc[0]
    model = CostModel(config.execution, contract_multiplier=1)
    entry = model.marketable_fill(
        QuoteEvent(ts=T(9, 34), symbol="AAA", bid=100.7, ask=100.9), Side.BUY, 2
    )
    exit_ = model.marketable_fill(
        QuoteEvent(ts=T(9, 38), symbol="AAA", bid=100.3, ask=100.5), Side.SELL, 2
    )
    assert row.entry_px == pytest.approx(entry.price)
    assert row.exit_px == pytest.approx(exit_.price)
    assert row.spread_cost == pytest.approx(entry.costs.spread_cost + exit_.costs.spread_cost)
    assert row.slippage_cost == pytest.approx(
        entry.costs.slippage_cost + exit_.costs.slippage_cost
    )
    assert row.commission == pytest.approx(entry.costs.commission + exit_.costs.commission)
    assert row.pnl == pytest.approx((exit_.price - entry.price) * 2 - row.commission)


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------


def test_stop_triggers_on_bar_path_not_close(tmp_path: Path) -> None:
    rows = [
        (T(9, 31), 100.0, 100.2, 99.8, 100.0),
        (T(9, 32), 100.0, 100.3, 99.9, 100.1),
        (T(9, 33), 100.1, 100.5, 100.0, 100.2),  # decision
        (T(9, 34), 100.2, 101.0, 100.1, 100.8),  # entry fill @ ask 100.9
        (T(9, 35), 100.8, 100.9, 99.5, 100.0),   # low 99.5 > stop 98.9: no trigger
        (T(9, 36), 100.0, 100.5, 98.5, 100.0),   # low 98.5 <= 98.9, close ABOVE stop
        (T(9, 37), 100.0, 100.2, 99.0, 99.5),    # exit fill @ bid 99.4
    ]
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}
    engine = make_engine(
        tmp_path,
        canonical_frames(aaa_rows=rows),
        base_params(stop_r_multiple=2.0),  # stop = entry 100.9 - 2 * $1 = 98.9
        [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()
    assert len(result.ledger) == 1
    row = result.ledger.iloc[0]
    assert row.exit_reason == "stop"
    assert row.exit_ts == T(9, 36) + LATENCY  # bar-path breach at 09:36, not 09:35
    assert row.exit_px == pytest.approx(99.4)  # next-bar bid, not the stop level
    assert row.mae == pytest.approx(100.9 - 98.5)


def test_eod_flatten(tmp_path: Path) -> None:
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}
    engine = make_engine(
        tmp_path,
        canonical_frames(),
        base_params(eod_flatten_time=time(9, 36)),
        [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()
    row = result.ledger.iloc[0]
    assert row.exit_reason == "eod_flatten"
    assert row.exit_ts == T(9, 36) + LATENCY  # first bar at/after 09:36 ET
    assert row.exit_px == pytest.approx(100.5)  # bid @ 09:37


def test_end_of_data_flatten_closes_stranded_position(tmp_path: Path) -> None:
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}  # no exit policy configured
    engine = make_engine(
        tmp_path, canonical_frames(), base_params(), [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
    )
    result = engine.run()
    row = result.ledger.iloc[0]
    assert row.exit_reason == "end_of_data"
    assert row.exit_ts == T(9, 38)
    assert row.exit_px == pytest.approx(100.4)  # final close: a mark-out, no costs
    assert result.equity.iloc[-1] == pytest.approx(100_000.0 + result.ledger.pnl.sum())


# ---------------------------------------------------------------------------
# Drop counters and the accounting identity
# ---------------------------------------------------------------------------


def test_every_drop_stage_counts_and_identity_holds(tmp_path: Path) -> None:
    frames = canonical_frames(bbb_price=250_000.0, ccc_quotes=False)
    plan = {
        ("AAA", T(9, 31)): (Side.BUY, 0.0),   # sizing-zero (conviction 0)
        ("AAA", T(9, 33)): (Side.BUY, 1.0),   # executes
        ("AAA", T(9, 34)): (Side.BUY, 1.0),   # position-already-open
        ("AAA", T(9, 38)): (Side.BUY, 1.0),   # last bar -> latency-no-fill
        ("BBB", T(9, 33)): (Side.BUY, 1.0),   # regime-gated (see gate below)
        ("BBB", T(9, 35)): (Side.BUY, 1.0),   # risk-cap (price 250k > equity)
        ("CCC", T(9, 33)): (Side.BUY, 1.0),   # no-quote (CCC has no quote tape)
    }

    def gate(event: SignalEvent, ctx) -> bool:
        return not (event.symbol == "BBB" and ctx.now == T(9, 33))

    engine = make_engine(
        tmp_path,
        frames,
        base_params(max_hold_minutes=2),  # AAA closes at 09:36 -> 09:38 re-entry legal
        [PlantedSignal(plan)],
        sizer=FixedSizer(10, 1.0),
        regime_gate=gate,
    )
    result = engine.run()
    assert result.drop_counters == {
        "regime-gated": 1,
        "sizing-zero": 1,
        "no-quote": 1,
        "latency-no-fill": 1,
        "position-already-open": 1,
        "risk-cap": 1,
        "daily-halt": 0,
    }
    assert result.emitted == 7 and result.executed == 1
    assert result.emitted == result.executed + sum(result.drop_counters.values())
    assert result.summary["drop_counters"] == result.drop_counters
    assert len(result.ledger) == 1


def test_daily_halt_blocks_new_entries(tmp_path: Path) -> None:
    stamps = [T(9, 31), T(9, 32), T(9, 33), T(9, 34), T(9, 35)]
    aaa = bars_frame(
        [
            (T(9, 31), 100.0, 100.2, 99.8, 100.0),  # decision: BUY 100 shares
            (T(9, 32), 100.0, 100.7, 99.9, 100.5),  # entry fill @ ask 100.6
            (T(9, 33), 100.5, 100.5, 79.0, 80.0),   # crash: -$2061 > 1% of 100k
            (T(9, 34), 80.0, 80.5, 79.5, 80.0),     # halted: CCC entry must drop
            (T(9, 35), 80.0, 80.5, 79.5, 80.0),
        ]
    )
    frames = {
        ("AAA", "bars"): aaa,
        ("AAA", "quotes"): quotes_from_bars(aaa),
        ("BBB", "bars"): flat_bars(50.0, stamps),
        ("BBB", "quotes"): quotes_from_bars(flat_bars(50.0, stamps)),
        ("CCC", "bars"): flat_bars(200.0, stamps),
        ("CCC", "quotes"): quotes_from_bars(flat_bars(200.0, stamps)),
    }
    plan = {
        ("AAA", T(9, 31)): (Side.BUY, 1.0),
        ("CCC", T(9, 34)): (Side.BUY, 1.0),  # after the halt tripped
    }
    engine = make_engine(
        tmp_path,
        frames,
        base_params(),
        [PlantedSignal(plan)],
        config=make_config(daily_loss_halt_pct=1.0),
        sizer=FixedSizer(100, 1.0),
    )
    result = engine.run()
    assert result.drop_counters["daily-halt"] == 1
    assert result.emitted == 2 and result.executed == 1
    assert len(result.ledger) == 1  # only AAA; closed by the terminal flatten
    row = result.ledger.iloc[0]
    assert row.symbol == "AAA" and row.exit_reason == "end_of_data"
    assert row.pnl == pytest.approx((80.0 - 100.6) * 100 - 1.0)
    assert result.equity.iloc[-1] == pytest.approx(100_000.0 + result.ledger.pnl.sum())


# ---------------------------------------------------------------------------
# Visibility: the engine, not the signal, enforces available_at <= t
# ---------------------------------------------------------------------------


class ProbeSignal:
    """Records what the context lets a signal see at each AAA bar close."""

    name = "probe"

    def __init__(self) -> None:
        self.observations: list[tuple[datetime, int, datetime, datetime]] = []

    def on_bar(self, ctx) -> None:
        if ctx.bar.symbol != "AAA":
            return
        visible = ctx.pit("vol_indices")
        own = ctx.history()
        cross = ctx.history("CCC")
        self.observations.append((ctx.now, len(visible), own[-1].ts, cross[-1].ts))
        return


def test_context_enforces_available_at_visibility(tmp_path: Path) -> None:
    vol = pd.DataFrame(
        {
            "asof_date": [pd.Timestamp(2026, 1, 6), pd.Timestamp(2026, 1, 6)],
            "available_at": [
                pd.Timestamp(2026, 1, 6, 9, 32, 30, tz=ET),
                pd.Timestamp(2026, 1, 6, 9, 35, 30, tz=ET),
            ],
            "vix": [15.0, 16.0],
        }
    )
    frames = canonical_frames()
    frames[(ALL_SYMBOLS, "vol_indices")] = vol
    probe = ProbeSignal()
    engine = make_engine(
        tmp_path, frames, base_params(pit_kinds=("vol_indices",)), [probe]
    )
    result = engine.run()
    assert len(result.ledger) == 0 and result.emitted == 0
    by_ts = {ts: n for ts, n, _own, _cross in probe.observations}
    # A row stamped available_at 09:32:30 is INVISIBLE at the 09:32 close and
    # visible from 09:33; the 09:35:30 row appears only from 09:36.
    assert by_ts[T(9, 31)] == 0
    assert by_ts[T(9, 32)] == 0
    assert by_ts[T(9, 33)] == 1
    assert by_ts[T(9, 35)] == 1
    assert by_ts[T(9, 36)] == 2
    assert by_ts[T(9, 38)] == 2
    for now, _n, own_last, cross_last in probe.observations:
        assert own_last == now    # history never runs ahead of the clock
        assert cross_last == now  # same-close bars of OTHER symbols are visible


def test_signal_stamping_wrong_ts_is_an_invariant_error(tmp_path: Path) -> None:
    class BadClockSignal:
        name = "bad"

        def on_bar(self, ctx) -> SignalEvent:
            return SignalEvent(
                ts=ctx.now - timedelta(minutes=1),  # lies about decision time
                symbol="AAA",
                side=Side.BUY,
                conviction=1.0,
                horizon_minutes=5,
                signal_name=self.name,
            )

    engine = make_engine(tmp_path, canonical_frames(), base_params(), [BadClockSignal()])
    with pytest.raises(EngineInvariantError):
        engine.run()


# ---------------------------------------------------------------------------
# Determinism: same inputs + seed -> byte-identical ledger
# ---------------------------------------------------------------------------


def test_determinism_byte_identical_ledger(tmp_path: Path) -> None:
    first = run_long_round_trip(tmp_path)
    second = run_long_round_trip(tmp_path)
    assert first.ledger.to_csv().encode() == second.ledger.to_csv().encode()
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert first.config_hash == second.config_hash
    assert first.drop_counters == second.drop_counters


def test_seed_is_part_of_the_config_hash(tmp_path: Path) -> None:
    plan = {("AAA", T(9, 33)): (Side.BUY, 1.0)}
    frames = canonical_frames()
    runs = {}
    for seed in (0, 1):
        engine = make_engine(
            tmp_path, frames, base_params(max_hold_minutes=3, seed=seed),
            [PlantedSignal(plan)], sizer=FixedSizer(10, 1.0),
        )
        runs[seed] = engine.run()
    assert runs[0].config_hash != runs[1].config_hash
    # No hook draws randomness here, so the ledgers themselves match.
    assert runs[0].ledger.to_csv() == runs[1].ledger.to_csv()


# ---------------------------------------------------------------------------
# RunResult contract: shape, counters always present, no auto-persistence
# ---------------------------------------------------------------------------


def test_run_result_contract_and_no_auto_write(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path)
    loader = EdgeDataLoader(TapeBackend(canonical_frames()), repo_root=root)
    engine = BacktestEngine(
        loader,
        make_config(),
        base_params(max_hold_minutes=3),
        [PlantedSignal({("AAA", T(9, 33)): (Side.BUY, 1.0)})],
        sizer=FixedSizer(10, 1.0),
    )
    before = {p for p in root.rglob("*")}
    result = engine.run()
    assert {p for p in root.rglob("*")} == before  # run() persisted NOTHING
    assert not (root / "results").exists()
    assert isinstance(result.ledger, pd.DataFrame)
    assert isinstance(result.equity, pd.Series)
    assert set(result.drop_counters) == set(DROP_STAGES)
    assert len(result.config_hash) == 64 and int(result.config_hash, 16) >= 0
    summary = result.summary
    assert summary["signals_emitted"] == 1 and summary["trades_executed"] == 1
    assert summary["final_equity"] == pytest.approx(result.equity.iloc[-1])


def test_default_sizer_math() -> None:
    sizer = RiskBudgetSizer(per_trade_cap_pct=2.0, risk_r_pct=1.0)
    signal = SignalEvent(
        ts=T(9, 33), symbol="AAA", side=Side.BUY, conviction=1.0,
        horizon_minutes=5, signal_name="s",
    )
    decision = sizer.size(signal, ref_price=100.0, equity=100_000.0)
    assert decision.risk_per_share == pytest.approx(1.0)
    assert decision.qty == 2000  # $2000 budget / $1 risk per share
    half = signal.model_copy(update={"conviction": 0.25})
    assert sizer.size(half, 100.0, 100_000.0).qty == 500
    zero = signal.model_copy(update={"conviction": 0.0})
    assert sizer.size(zero, 100.0, 100_000.0).qty == 0


# ---------------------------------------------------------------------------
# The CLI stays a thin front
# ---------------------------------------------------------------------------


def test_cli_requires_symbols_start_end() -> None:
    assert backtest_cli.main([]) == 0  # bare invocation prints help
    assert backtest_cli.main(["--symbols", "AAA"]) == 2  # partial args refuse


def test_cli_unknown_strategy(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path)
    code = backtest_cli.main(
        ["--repo-root", str(root), "--strategy", "nope",
         "--symbols", "AAA", "--start", "2026-01-06", "--end", "2026-01-06"]
    )
    assert code == 2


def test_cli_reports_missing_bars_backend(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path)
    code = backtest_cli.main(
        ["--repo-root", str(root),
         "--symbols", "AAA", "--start", "2026-01-06", "--end", "2026-01-06"]
    )
    assert code == 2  # the feeds loader serves no bars yet; refuse, don't pretend

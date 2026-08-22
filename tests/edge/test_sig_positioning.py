"""Positioning signals: COT-extreme fade and insider-cluster long.

Per signal, three proofs on synthetic frames (no network, all timestamps in
the 2019-2023 development span, writes only under tmp_path):

* AVAILABILITY DISCIPLINE — the fake ctx serves frames UNFILTERED, so any
  restraint observed is the SIGNAL's own ``available_at <= decision_ts``
  re-filter: a Tuesday COT report can never trigger before its Friday
  16:00 ET release, and a late-filed Form 4 (old ``asof_date``, late
  ``available_at``) cannot backdate an entry.
* PLANTED PATTERN FIRES — a 104-week z-spike in ES lev-money positioning
  emits the fade; a 2-officer positive-dollar buy cluster emits the long.
* INVERSE PATTERN DOES NOT — in-line positioning stays silent; clustered
  selling, negative net dollars, and single-officer buying stay silent.

Plus: the distinct-officer-buyer feature the cluster definition rides on
(``officer_buyers_21d``, computed in the EDGAR feed and served through the
loader's ``insider`` kind), regime declarations, registration, and one
engine round-trip proving the COT entry lands at the open AFTER the Friday
release through the real next-bar fill path.
"""

from __future__ import annotations

from datetime import date, datetime
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
from edge.core.events import BarEvent, Side
from edge.data.feeds.edgar_form4 import insider_features
from edge.data.loader import ALL_SYMBOLS, EdgeDataLoader
from edge.regime.classifier import VOL_STATES, VOL_STRESSED
from edge.runners.engine import BacktestEngine, EngineParams
from edge.signals import registry as sigreg
from edge.signals.base import (
    ALL_REGIMES,
    MIN_HYPOTHESIS_CHARS,
    regime_allowed,
    require_hypothesis,
)
from edge.signals.positioning import (
    MINUTES_PER_SESSION,
    CotExtreme,
    CotExtremeConfig,
    InsiderCluster,
    InsiderClusterConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

# The planted COT week: report as of Tuesday 2023-06-06, released Friday
# 2023-06-09 16:00 ET. All test timestamps sit inside the development span.
LAST_ASOF = date(2023, 6, 6)  # a Tuesday
RELEASE = datetime(2023, 6, 9, 16, 0, tzinfo=ET)  # that week's Friday 16:00
MONDAY_OPEN_BAR = datetime(2023, 6, 12, 9, 31, tzinfo=ET)

Z_WINDOW = CotExtremeConfig().z_window_weeks  # 104 — the 2y fixed choice


@pytest.fixture(autouse=True)
def _registered() -> None:
    """Re-register the module's signals if another test cleared the catalog."""
    for cls in (CotExtreme, InsiderCluster):
        try:
            sigreg.get(cls.name)
        except sigreg.UnknownSignal:
            sigreg.register(cls)


# ---------------------------------------------------------------------------
# Fake ctx: engine-shaped, but serves frames UNFILTERED on purpose
# ---------------------------------------------------------------------------


class FakeCtx:
    """Engine-shaped context (``now``/``bar``/``pit``) that deliberately does
    NOT filter on ``available_at`` or symbol — whatever discipline the tests
    observe is enforced by the signal itself."""

    def __init__(self, now: datetime, bar: BarEvent, frames: dict[str, pd.DataFrame]) -> None:
        self.now = now
        self.bar = bar
        self._frames = frames

    def pit(self, kind: str, symbol: str | None = None) -> pd.DataFrame:
        return self._frames.get(kind, pd.DataFrame()).copy()

    def history(self, symbol: str | None = None, n: int | None = None) -> tuple:
        return ()


def ctx_at(now: datetime, symbol: str, frames: dict[str, pd.DataFrame]) -> FakeCtx:
    price = 100.0
    bar = BarEvent(
        ts=now, symbol=symbol, open=price, high=price, low=price, close=price, volume=100
    )
    return FakeCtx(now, bar, frames)


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------


def cot_frame(values: list[float], market: str = "ES") -> pd.DataFrame:
    """Weekly TFF-shaped rows ending at LAST_ASOF: asof Tuesdays, each
    available the SAME week's Friday 16:00 ET (asof + 3 days, 16:00)."""
    n = len(values)
    asofs = [pd.Timestamp(LAST_ASOF) - pd.Timedelta(weeks=n - 1 - i) for i in range(n)]
    return pd.DataFrame(
        {
            "asof_date": asofs,
            "available_at": [
                (a + pd.Timedelta(days=3)).tz_localize(ET) + pd.Timedelta(hours=16)
                for a in asofs
            ],
            "market": market,
            "market_code": "13874A" if market == "ES" else "209742",
            "lev_money_net_oi": values,
        }
    )


def flat_history(n: int) -> list[float]:
    """Alternating small values: nonzero variance, every z well inside +/-2."""
    return [0.0 if i % 2 == 0 else 0.02 for i in range(n)]


def spike_up_frame() -> pd.DataFrame:
    """110 weeks of flat positioning, then a crowded-LONG spike this week."""
    return cot_frame(flat_history(109) + [0.30])


def spike_down_frame() -> pd.DataFrame:
    return cot_frame(flat_history(109) + [-0.30])


def insider_row(
    symbol: str,
    asof: date,
    available: datetime,
    net_dollars: float,
    buyers: int,
) -> dict:
    return {
        "symbol": symbol,
        "asof_date": pd.Timestamp(asof),
        "available_at": pd.Timestamp(available),
        "net_insider_dollars_21d": net_dollars,
        "officer_buy": buyers > 0,
        "officer_buyers_21d": buyers,
    }


def insider_frames(*rows: dict) -> dict[str, pd.DataFrame]:
    return {"insider": pd.DataFrame(list(rows))}


# ---------------------------------------------------------------------------
# CotExtreme: availability discipline
# ---------------------------------------------------------------------------

#: Decision instants strictly between the report's Tuesday asof and its
#: Friday 16:00 release — the row exists in the served frame at every one
#: of them, and the signal must stay silent at every one of them.
BEFORE_RELEASE = [
    datetime(2023, 6, 6, 17, 0, tzinfo=ET),  # Tuesday, after the asof close
    datetime(2023, 6, 7, 9, 31, tzinfo=ET),  # Wednesday open
    datetime(2023, 6, 7, 15, 59, tzinfo=ET),
    datetime(2023, 6, 8, 12, 0, tzinfo=ET),  # Thursday
    datetime(2023, 6, 9, 9, 31, tzinfo=ET),  # Friday morning
    datetime(2023, 6, 9, 15, 59, tzinfo=ET),  # one minute before the release
]


@pytest.mark.parametrize("now", BEFORE_RELEASE, ids=lambda d: d.isoformat())
def test_tuesday_report_never_triggers_before_friday(now: datetime) -> None:
    """The spike row is IN the frame all week; the Friday 16:00 stamp alone
    must keep it invisible — asof_date grants nothing."""
    signal = CotExtreme()
    events = signal.on_bar(ctx_at(now, "SPY", {"cot": spike_up_frame()}))
    assert events == []


def test_fires_at_release_and_at_following_open() -> None:
    # At the release instant itself (Friday 16:00 close): visible, emits; the
    # engine's next-bar fill then makes the ENTRY the following open.
    at_release = CotExtreme().on_bar(ctx_at(RELEASE, "SPY", {"cot": spike_up_frame()}))
    assert len(at_release) == 1
    # A fresh run first seeing the report at Monday's open also emits.
    at_open = CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": spike_up_frame()}))
    assert len(at_open) == 1


def test_stale_report_is_history_not_news() -> None:
    """First sight of the report 6 days after release: no entry — entries
    happen only at the release open following the stamp."""
    late = datetime(2023, 6, 15, 10, 0, tzinfo=ET)  # Thursday after
    assert CotExtreme().on_bar(ctx_at(late, "SPY", {"cot": spike_up_frame()})) == []


def test_one_emission_per_report() -> None:
    signal = CotExtreme()
    frames = {"cot": spike_up_frame()}
    assert len(signal.on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", frames))) == 1
    next_bar = datetime(2023, 6, 12, 9, 32, tzinfo=ET)
    assert signal.on_bar(ctx_at(next_bar, "SPY", frames)) == []


# ---------------------------------------------------------------------------
# CotExtreme: planted pattern fires, inverse does not
# ---------------------------------------------------------------------------


def test_crowded_long_extreme_fades_short() -> None:
    frame = spike_up_frame()
    # The planted spike really is a >2z extreme of its own trailing window.
    window = frame["lev_money_net_oi"].tail(Z_WINDOW)
    z = (window.iloc[-1] - window.mean()) / window.std(ddof=1)
    assert z > 2.0

    events = CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": frame}))
    assert len(events) == 1
    event = events[0]
    assert event.side is Side.SELL  # fade the crowded long
    assert event.symbol == "SPY"
    assert event.signal_name == "cot_extreme"
    assert event.conviction == 1.0
    assert event.ts == MONDAY_OPEN_BAR
    assert event.horizon_minutes == CotExtremeConfig().horizon_sessions * MINUTES_PER_SESSION


def test_crowded_short_extreme_fades_long() -> None:
    events = CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": spike_down_frame()}))
    assert len(events) == 1
    assert events[0].side is Side.BUY


def test_inverse_no_extreme_no_signal() -> None:
    """Fresh report with positioning in line with its own 2y history."""
    frame = cot_frame(flat_history(110))  # final value 0.02: |z| ~ 1
    assert CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": frame})) == []


def test_short_history_never_fires() -> None:
    """A z on fewer than 104 weeks is a different statistic: warmup is silent."""
    frame = cot_frame(flat_history(59) + [0.30])  # 60 weeks incl. the spike
    assert CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": frame})) == []


def test_other_market_rows_are_ignored() -> None:
    frame = spike_up_frame().assign(market="NQ")  # extreme, but not ES
    assert CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "SPY", {"cot": frame})) == []


def test_emits_only_on_trade_symbol_bars() -> None:
    events = CotExtreme().on_bar(ctx_at(MONDAY_OPEN_BAR, "AAPL", {"cot": spike_up_frame()}))
    assert events == []


# ---------------------------------------------------------------------------
# InsiderCluster: availability discipline (late filings cannot backdate)
# ---------------------------------------------------------------------------

#: A cluster whose LAST Form 4 was filed late: transactions dated early
#: March, the window's availability dragged to March 20 10:10 ET.
LATE_ASOF = date(2023, 3, 1)
LATE_AVAILABLE = datetime(2023, 3, 20, 10, 10, tzinfo=ET)
LATE_CLUSTER = insider_row("AAA", LATE_ASOF, LATE_AVAILABLE, 150_000.0, buyers=2)

AFTER_ASOF_BEFORE_FILING = [
    datetime(2023, 3, 2, 10, 0, tzinfo=ET),
    datetime(2023, 3, 10, 9, 31, tzinfo=ET),
    datetime(2023, 3, 20, 10, 9, tzinfo=ET),  # one minute before availability
]


@pytest.mark.parametrize("now", AFTER_ASOF_BEFORE_FILING, ids=lambda d: d.isoformat())
def test_late_filed_form4_never_backdates(now: datetime) -> None:
    """The transactions are DATED weeks earlier, but until the late filing's
    acceptance + 5min the cluster does not exist for a live trader."""
    events = InsiderCluster().on_bar(ctx_at(now, "AAA", insider_frames(LATE_CLUSTER)))
    assert events == []


def test_fires_once_available() -> None:
    now = datetime(2023, 3, 20, 10, 31, tzinfo=ET)  # 21 min after availability
    events = InsiderCluster().on_bar(ctx_at(now, "AAA", insider_frames(LATE_CLUSTER)))
    assert len(events) == 1
    assert events[0].side is Side.BUY
    assert events[0].symbol == "AAA"
    assert events[0].ts == now


# ---------------------------------------------------------------------------
# InsiderCluster: planted pattern fires, inverse does not
# ---------------------------------------------------------------------------

CLUSTER_AVAILABLE = datetime(2023, 3, 10, 18, 35, tzinfo=ET)  # evening filing
CLUSTER_DECISION = datetime(2023, 3, 13, 9, 31, tzinfo=ET)  # Monday open bar


def cluster(net: float, buyers: int, symbol: str = "AAA") -> dict:
    return insider_row(symbol, date(2023, 3, 10), CLUSTER_AVAILABLE, net, buyers)


def test_two_officer_positive_cluster_goes_long() -> None:
    events = InsiderCluster().on_bar(
        ctx_at(CLUSTER_DECISION, "AAA", insider_frames(cluster(150_000.0, buyers=2)))
    )
    assert len(events) == 1
    event = events[0]
    assert event.side is Side.BUY
    assert event.symbol == "AAA"
    assert event.signal_name == "insider_cluster"
    assert event.conviction == 1.0
    assert (
        event.horizon_minutes
        == InsiderClusterConfig().horizon_sessions * MINUTES_PER_SESSION
    )


@pytest.mark.parametrize(
    ("net", "buyers", "why"),
    [
        (-50_000.0, 2, "two officers bought but a bigger sale nets negative"),
        (150_000.0, 1, "one officer buying repeatedly is not a cluster"),
        (-200_000.0, 0, "clustered selling is the inverse pattern"),
        (0.0, 2, "zero net dollars is not net buying"),
    ],
)
def test_inverse_patterns_stay_silent(net: float, buyers: int, why: str) -> None:
    events = InsiderCluster().on_bar(
        ctx_at(CLUSTER_DECISION, "AAA", insider_frames(cluster(net, buyers)))
    )
    assert events == [], why


def test_cluster_in_another_symbol_does_not_fire_here() -> None:
    """The frame is served unfiltered; the signal must key on the bar's own
    symbol (BBB's cluster is not AAA's information)."""
    events = InsiderCluster().on_bar(
        ctx_at(CLUSTER_DECISION, "AAA", insider_frames(cluster(150_000.0, 2, symbol="BBB")))
    )
    assert events == []


def test_stale_cluster_is_not_entered() -> None:
    late = datetime(2023, 3, 20, 9, 31, tzinfo=ET)  # 10 days after availability
    events = InsiderCluster().on_bar(
        ctx_at(late, "AAA", insider_frames(cluster(150_000.0, 2)))
    )
    assert events == []


def test_one_emission_per_cluster_row() -> None:
    signal = InsiderCluster()
    frames = insider_frames(cluster(150_000.0, 2))
    assert len(signal.on_bar(ctx_at(CLUSTER_DECISION, "AAA", frames))) == 1
    next_bar = datetime(2023, 3, 13, 9, 32, tzinfo=ET)
    assert signal.on_bar(ctx_at(next_bar, "AAA", frames)) == []


def test_missing_cluster_column_is_loud() -> None:
    row = {k: v for k, v in cluster(150_000.0, 2).items() if k != "officer_buyers_21d"}
    with pytest.raises(ValueError, match="officer_buyers_21d"):
        InsiderCluster().on_bar(ctx_at(CLUSTER_DECISION, "AAA", insider_frames(row)))


# ---------------------------------------------------------------------------
# The distinct-officer count the cluster rides on (EDGAR feature)
# ---------------------------------------------------------------------------


def _txn(
    symbol: str,
    asof: date,
    code: str,
    shares: float,
    price: float,
    officer: bool,
    available: datetime,
    cik: str | None,
    name: str,
) -> dict:
    return {
        "symbol": symbol,
        "transaction_code": code,
        "shares": shares,
        "price": price,
        "is_officer": officer,
        "asof_date": asof,
        "available_at": available,
        "insider_cik": cik,
        "insider_name": name,
    }


def test_officer_buyers_counts_distinct_officers_not_events() -> None:
    avail = datetime(2023, 3, 8, 18, 35, tzinfo=ET)
    rows = [
        # CFO buys twice (two events, ONE officer) ...
        _txn("AAA", date(2023, 3, 1), "P", 1_000, 10.0, True, avail, "0001", "CFO JANE"),
        _txn("AAA", date(2023, 3, 6), "P", 1_000, 10.0, True, avail, "0001", "CFO JANE"),
        # ... the CEO joins within the window (a second distinct officer) ...
        _txn("AAA", date(2023, 3, 8), "P", 2_000, 10.0, True, avail, "0002", "CEO JOHN"),
        # ... a DIRECTOR buy and an officer SALE must not count as buyers.
        _txn("AAA", date(2023, 3, 8), "P", 500, 10.0, False, avail, "0003", "DIR DAVE"),
        _txn("AAA", date(2023, 3, 7), "S", 100, 10.0, True, avail, "0004", "COO SAM"),
    ]
    feats = insider_features(pd.DataFrame(rows), window_days=21)
    aaa = feats[feats["symbol"] == "AAA"].set_index("asof_date")
    assert int(aaa.loc[date(2023, 3, 6), "officer_buyers_21d"]) == 1  # CFO twice
    assert int(aaa.loc[date(2023, 3, 8), "officer_buyers_21d"]) == 2  # CFO + CEO


def test_officer_buyers_window_expires() -> None:
    avail = datetime(2023, 3, 8, 18, 35, tzinfo=ET)
    rows = [
        _txn("AAA", date(2023, 1, 5), "P", 1_000, 10.0, True, avail, "0001", "CFO JANE"),
        _txn("AAA", date(2023, 3, 8), "P", 2_000, 10.0, True, avail, "0002", "CEO JOHN"),
    ]
    feats = insider_features(pd.DataFrame(rows), window_days=21)
    aaa = feats[feats["symbol"] == "AAA"].set_index("asof_date")
    # January's buy fell out of the (t-21d, t] window: only the CEO remains.
    assert int(aaa.loc[date(2023, 3, 8), "officer_buyers_21d"]) == 1


def test_identityless_frames_undercount_never_overcount() -> None:
    """Without insider_cik/insider_name every officer buy collapses onto one
    placeholder identity: the count is 0 or 1, never inflated."""
    avail = datetime(2023, 3, 8, 18, 35, tzinfo=ET)
    rows = [
        {k: v for k, v in _txn("AAA", d, "P", 1_000, 10.0, True, avail, None, "x").items()
         if k not in ("insider_cik", "insider_name")}
        for d in (date(2023, 3, 1), date(2023, 3, 8))
    ]
    feats = insider_features(pd.DataFrame(rows), window_days=21)
    assert list(feats["officer_buyers_21d"]) == [1, 1]


# ---------------------------------------------------------------------------
# Declarations: hypotheses, regimes, registration
# ---------------------------------------------------------------------------


def test_hypotheses_are_admissible_and_state_the_fixed_choices() -> None:
    for cls in (CotExtreme, InsiderCluster):
        assert require_hypothesis(cls.hypothesis, owner=cls.name) == cls.hypothesis
        assert len(cls.hypothesis.strip()) >= MIN_HYPOTHESIS_CHARS
    # The fixed economic choices are written into the hypotheses themselves.
    assert "|z| > 2" in CotExtreme.hypothesis
    assert "104 weekly" in CotExtreme.hypothesis
    assert "2 distinct officers" in InsiderCluster.hypothesis
    assert "21 calendar days" in InsiderCluster.hypothesis


def test_regime_declarations() -> None:
    # CotExtreme trades through every regime — positioning unwinds are
    # triggered BY regime shocks (stated in the class docstring).
    assert CotExtreme.allowed_regimes == ALL_REGIMES
    assert regime_allowed(CotExtreme, VOL_STRESSED) is True
    # InsiderCluster: every vol state EXCEPT stressed backwardation.
    assert InsiderCluster.allowed_regimes == frozenset(VOL_STATES) - {VOL_STRESSED}
    assert regime_allowed(InsiderCluster, VOL_STRESSED) is False
    for state in frozenset(VOL_STATES) - {VOL_STRESSED}:
        assert regime_allowed(InsiderCluster, state) is True


def test_both_signals_are_registered_and_instantiable() -> None:
    assert sigreg.get("cot_extreme") is CotExtreme
    assert sigreg.get("insider_cluster") is InsiderCluster
    for name in ("cot_extreme", "insider_cluster"):
        signal = sigreg.create(name)
        assert signal.config == signal.config_type()


def test_configs_forbid_stray_parameters() -> None:
    with pytest.raises(ValidationError):
        CotExtremeConfig(z_treshold=3.0)  # typo'd parameter must not pass
    with pytest.raises(ValidationError):
        InsiderClusterConfig(min_officer=3)


# ---------------------------------------------------------------------------
# Engine round-trip: the COT entry lands at the open AFTER the release
# ---------------------------------------------------------------------------


class TapeBackend:
    """In-memory backend serving pre-built frames keyed by (symbol, kind)."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        return self.frames.get((symbol, kind), pd.DataFrame())


def make_repo_root(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "edge.yaml").write_text((REPO_ROOT / "config" / "edge.yaml").read_text())
    return tmp_path


ENGINE_CONFIG = EdgeConfig(
    data=DataConfig(lockbox_start=date(2026, 2, 22)),
    validation=ValidationConfig(min_oos_trades=1, pbo_max=0.5, bootstrap_resamples=10),
    execution=ExecutionConfig(
        spread_fill_fraction=1.0,
        slippage_pct=0.0,
        commission_per_contract=0.01,
        latency_ms=[100],
    ),
    risk=RiskConfig(
        kelly_fraction=0.25,
        per_trade_cap_pct=2.0,
        daily_loss_halt_pct=50.0,
        target_daily_vol_pct=None,
    ),
)


FRIDAY_CLOSE = 430.0
MONDAY_CLOSE = 425.0
HALF_SPREAD = 0.05


def spy_tape() -> dict[tuple[str, str], pd.DataFrame]:
    """SPY minute bars: Friday afternoon (release day) at 430, Monday morning
    at 425 — the price GAP is what proves which session's book filled us."""
    rows = [
        (datetime(2023, 6, 9, 15, 59, tzinfo=ET), FRIDAY_CLOSE),
        (datetime(2023, 6, 9, 16, 0, tzinfo=ET), FRIDAY_CLOSE),  # release close
        (datetime(2023, 6, 12, 9, 31, tzinfo=ET), MONDAY_CLOSE),  # following open
        (datetime(2023, 6, 12, 9, 32, tzinfo=ET), MONDAY_CLOSE),
    ]
    bars = pd.DataFrame(
        {
            "ts": [ts for ts, _ in rows],
            "open": [px for _, px in rows],
            "high": [px + 0.5 for _, px in rows],
            "low": [px - 0.5 for _, px in rows],
            "close": [px for _, px in rows],
            "volume": 10_000,
        }
    )
    quotes = pd.DataFrame(
        {
            "ts": bars["ts"],
            "bid": bars["close"] - HALF_SPREAD,
            "ask": bars["close"] + HALF_SPREAD,
            "bid_size": 500,
            "ask_size": 500,
        }
    )
    return {
        ("SPY", "bars"): bars,
        ("SPY", "quotes"): quotes,
        (ALL_SYMBOLS, "cot"): spike_up_frame(),
    }


def test_engine_entry_is_the_open_following_the_friday_release(tmp_path: Path) -> None:
    loader = EdgeDataLoader(TapeBackend(spy_tape()), repo_root=make_repo_root(tmp_path))
    params = EngineParams(
        symbols=("SPY",),
        start=date(2023, 6, 9),
        end=date(2023, 6, 12),
        latency_ms=100,
        initial_equity=100_000.0,
        risk_r_pct=1.0,
        pit_kinds=("cot",),
    )
    engine = BacktestEngine(loader, ENGINE_CONFIG, params, [CotExtreme()])
    result = engine.run()

    # Exactly one emission: the 15:59 Friday close saw nothing (release is
    # 16:00), the 16:00 close emitted, Monday bars were silenced by the
    # once-per-report state.
    assert result.emitted == 1
    assert result.executed == 1
    assert len(result.ledger) == 1
    trade = result.ledger.iloc[0]
    assert trade["signal_name"] == "cot_extreme"
    assert trade["side"] == Side.SELL.value
    # The decision was Friday 16:00 (stamped as arrival = decision + 100ms),
    # but the fill met MONDAY's book: the sell crossed to Monday's 424.95
    # bid, not Friday's 429.95 — the entry IS the open following the release.
    assert trade["entry_px"] == pytest.approx(MONDAY_CLOSE - HALF_SPREAD)
    entry_ts = pd.Timestamp(trade["entry_ts"])
    assert entry_ts == pd.Timestamp(
        datetime(2023, 6, 9, 16, 0, tzinfo=ET)
    ) + pd.Timedelta(milliseconds=100)
    assert result.emitted == result.executed + sum(result.drop_counters.values())

"""The flow pair: opposite predictions, one discriminator, no guessing."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from edge.core.events import MARKET_TZ, BarEvent, Side
from edge.signals.flow import FlowConfig, FlowContinuation, FlowReversion

ET = MARKET_TZ


def _ctx(*, imbalance: float, vpin: float, intensity: float = 2.0,
         hhmm: tuple[int, int] = (10, 30), row_age_s: int = 0,
         symbol: str = "SPY") -> SimpleNamespace:
    now = datetime(2025, 6, 2, *hhmm, tzinfo=ET)
    row = pd.Series({
        "ts": pd.Timestamp(now) - pd.Timedelta(seconds=row_age_s),
        "imbalance": imbalance, "vpin": vpin, "intensity_z": intensity,
    })
    bar = BarEvent(ts=now, symbol=symbol, open=1.0, high=1.0, low=1.0,
                   close=1.0, volume=1)
    return SimpleNamespace(now=now, bar=bar, pit_latest=lambda k, s: row)


class TestContinuation:
    def test_follows_toxic_buying(self) -> None:
        events = FlowContinuation().on_bar(_ctx(imbalance=0.6, vpin=0.8))
        assert [e.side for e in events] == [Side.BUY]

    def test_follows_toxic_selling(self) -> None:
        events = FlowContinuation().on_bar(_ctx(imbalance=-0.6, vpin=0.8))
        assert [e.side for e in events] == [Side.SELL]

    def test_silent_on_benign_flow(self) -> None:
        """Low toxicity belongs to the OTHER hypothesis."""
        assert FlowContinuation().on_bar(_ctx(imbalance=0.6, vpin=0.2)) == []

    def test_silent_on_balanced_tape(self) -> None:
        assert FlowContinuation().on_bar(_ctx(imbalance=0.05, vpin=0.9)) == []


class TestReversion:
    def test_fades_benign_buying(self) -> None:
        events = FlowReversion().on_bar(_ctx(imbalance=0.6, vpin=0.2))
        assert [e.side for e in events] == [Side.SELL]

    def test_fades_benign_selling(self) -> None:
        events = FlowReversion().on_bar(_ctx(imbalance=-0.6, vpin=0.2))
        assert [e.side for e in events] == [Side.BUY]

    def test_silent_on_toxic_flow(self) -> None:
        assert FlowReversion().on_bar(_ctx(imbalance=0.6, vpin=0.9)) == []


class TestPairIsMutuallyExclusive:
    @pytest.mark.parametrize("vpin", [0.1, 0.3, 0.49, 0.51, 0.7, 0.95])
    def test_exactly_one_of_the_pair_fires(self, vpin: float) -> None:
        """The discriminator partitions: never both, never neither."""
        ctx = _ctx(imbalance=0.6, vpin=vpin)
        n = len(FlowContinuation().on_bar(ctx)) + len(FlowReversion().on_bar(ctx))
        assert n == 1

    def test_opposite_sides_on_the_same_imbalance(self) -> None:
        cont = FlowContinuation().on_bar(_ctx(imbalance=0.6, vpin=0.9))[0]
        rev = FlowReversion().on_bar(_ctx(imbalance=0.6, vpin=0.1))[0]
        assert cont.side != rev.side


class TestGates:
    def test_unformed_vpin_never_trades(self) -> None:
        assert FlowContinuation().on_bar(_ctx(imbalance=0.9, vpin=float("nan"))) == []
        assert FlowReversion().on_bar(_ctx(imbalance=0.9, vpin=float("nan"))) == []

    def test_quiet_tape_is_not_a_signal(self) -> None:
        """Low trade intensity means the imbalance is noise on thin volume."""
        assert FlowContinuation().on_bar(_ctx(imbalance=0.9, vpin=0.9, intensity=0.1)) == []

    def test_stale_row_never_trades(self) -> None:
        """A row from minutes ago means the tape went quiet — no flow signal."""
        assert FlowContinuation().on_bar(
            _ctx(imbalance=0.9, vpin=0.9, row_age_s=300)) == []

    def test_outside_entry_window_is_silent(self) -> None:
        assert FlowContinuation().on_bar(_ctx(imbalance=0.9, vpin=0.9, hhmm=(9, 31))) == []
        assert FlowContinuation().on_bar(_ctx(imbalance=0.9, vpin=0.9, hhmm=(15, 45))) == []

    def test_conviction_scales_with_imbalance(self) -> None:
        weak = FlowContinuation().on_bar(_ctx(imbalance=0.4, vpin=0.9))[0]
        strong = FlowContinuation().on_bar(_ctx(imbalance=0.9, vpin=0.9))[0]
        assert strong.conviction > weak.conviction

    def test_thresholds_are_config_not_hardcoded(self) -> None:
        cfg = FlowConfig(imbalance_threshold=0.8)
        assert FlowContinuation(cfg).on_bar(_ctx(imbalance=0.5, vpin=0.9)) == []
        assert FlowContinuation().on_bar(_ctx(imbalance=0.5, vpin=0.9)) != []

    def test_config_rejects_unknown_keys(self) -> None:
        with pytest.raises(Exception):
            FlowConfig(imbalance_thresh=0.5)  # type: ignore[call-arg]

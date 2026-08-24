"""Microstructure features: hand-computed cases, causality, exact volume clock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge.features.microstructure import (
    bucket_vpin,
    minute_microstructure,
    tick_rule_signs,
    trade_sign_imbalance,
)

ET = "America/New_York"


def _trades(prices, sizes, start="2025-06-02 09:30:00", freq="10s"):
    ts = pd.date_range(start, periods=len(prices), freq=freq, tz=ET).tz_convert("UTC")
    return pd.DataFrame({"ts": ts, "price": prices, "size": sizes})


class TestTickRule:
    def test_up_and_down_ticks_sign_by_direction(self) -> None:
        signs = tick_rule_signs(np.array([10.0, 10.1, 10.0, 9.9]))
        assert list(signs) == [0, 1, -1, -1]

    def test_zero_ticks_inherit_the_last_nonzero_sign(self) -> None:
        signs = tick_rule_signs(np.array([10.0, 10.1, 10.1, 10.1, 10.0]))
        assert list(signs) == [0, 1, 1, 1, -1]

    def test_leading_flat_run_is_never_guessed(self) -> None:
        """No predecessor means no information — 0, not an invented sign."""
        signs = tick_rule_signs(np.array([10.0, 10.0, 10.0, 10.5]))
        assert list(signs) == [0, 0, 0, 1]

    def test_empty_input(self) -> None:
        assert tick_rule_signs(np.array([])).size == 0


class TestSignImbalance:
    def test_all_buys_is_plus_one(self) -> None:
        assert trade_sign_imbalance(np.array([5, 5]), np.array([1, 1])) == 1.0

    def test_all_sells_is_minus_one(self) -> None:
        assert trade_sign_imbalance(np.array([5, 5]), np.array([-1, -1])) == -1.0

    def test_size_weighted_not_count_weighted(self) -> None:
        # one big sell outweighs two small buys
        got = trade_sign_imbalance(np.array([1, 1, 10]), np.array([1, 1, -1]))
        assert got == pytest.approx((1 + 1 - 10) / 12)

    def test_zero_volume_is_zero_not_nan(self) -> None:
        assert trade_sign_imbalance(np.array([0, 0]), np.array([1, -1])) == 0.0


class TestVpin:
    def test_perfectly_balanced_flow_is_zero(self) -> None:
        size = np.array([10.0] * 20)
        signs = np.array([1, -1] * 10, dtype=np.int8)
        assert bucket_vpin(size, signs, bucket_volume=20.0, n_buckets=5) == pytest.approx(0.0)

    def test_one_sided_flow_is_one(self) -> None:
        size = np.array([10.0] * 20)
        signs = np.ones(20, dtype=np.int8)
        assert bucket_vpin(size, signs, bucket_volume=20.0, n_buckets=5) == pytest.approx(1.0)

    def test_unformed_statistic_is_nan_never_a_number(self) -> None:
        size = np.array([10.0] * 4)
        signs = np.ones(4, dtype=np.int8)
        assert np.isnan(bucket_vpin(size, signs, bucket_volume=20.0, n_buckets=5))

    def test_boundary_prints_are_split_not_assigned_wholly(self) -> None:
        """A print straddling a bucket edge splits — the volume clock is exact."""
        # one 100-lot buy then one 100-lot sell, buckets of 50: each bucket is
        # wholly one-sided, so VPIN is 1.0 across all four buckets.
        size = np.array([100.0, 100.0])
        signs = np.array([1, -1], dtype=np.int8)
        assert bucket_vpin(size, signs, bucket_volume=50.0, n_buckets=4) == pytest.approx(1.0)
        # halve the bucket boundary alignment: 150-share buckets straddle the
        # buy/sell switch, so the middle bucket is mixed and VPIN drops.
        got = bucket_vpin(size, signs, bucket_volume=150.0, n_buckets=1)
        assert got < 1.0

    def test_unsigned_prints_inform_neither_side(self) -> None:
        size = np.array([10.0] * 10)
        signs = np.zeros(10, dtype=np.int8)
        assert bucket_vpin(size, signs, bucket_volume=20.0, n_buckets=3) == pytest.approx(0.0)


class TestMinuteFrame:
    def test_minute_rows_are_close_labelled(self) -> None:
        """Minutes are left-open/right-closed: (t-1min, t] closes at t.

        So a print landing exactly ON 09:30:00.000 belongs to the minute
        ENDING 09:30 (the 09:29-09:30 interval), and the prints after it
        belong to the 09:31 close. Boundary prints are not swept forward.
        """
        frame = minute_microstructure(_trades([10.0, 10.1, 10.2], [1, 1, 1]))
        assert str(frame["ts"].iloc[0].time()) == "09:30:00"
        assert frame["n_prints"].iloc[0] == 1
        assert str(frame["ts"].iloc[1].time()) == "09:31:00"
        assert frame["n_prints"].iloc[1] == 2

    def test_is_causal_truncation_does_not_change_earlier_rows(self) -> None:
        """Prefix invariance: the future cannot alter a past row."""
        prices = [10.0 + 0.1 * i for i in range(90)]
        sizes = [10] * 90
        full = minute_microstructure(_trades(prices, sizes))
        prefix = minute_microstructure(_trades(prices[:45], sizes[:45]))
        # The prefix's LAST minute is incomplete (its remaining prints were
        # truncated away), so compare only the minutes the prefix saw whole.
        n = len(prefix) - 1
        cols = ["volume", "signed_volume", "imbalance", "n_prints"]
        pd.testing.assert_frame_equal(
            full.iloc[:n][["ts", *cols]].reset_index(drop=True),
            prefix.iloc[:n][["ts", *cols]].reset_index(drop=True),
        )

    def test_one_sided_buying_shows_positive_imbalance(self) -> None:
        prices = [10.0 + 0.01 * i for i in range(30)]
        frame = minute_microstructure(_trades(prices, [5] * 30))
        assert frame["imbalance"].iloc[-1] > 0.9

    def test_empty_input_returns_typed_empty_frame(self) -> None:
        frame = minute_microstructure(pd.DataFrame(columns=["ts", "price", "size"]))
        assert frame.empty
        assert "vpin" in frame.columns

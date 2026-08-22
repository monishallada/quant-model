"""Contract selector: delta targeting (with fat-tailed OTM shift and
deterministic tie-breaking), DTE-rung restriction, hand-computed breakeven
gate, each liquidity gate binding by name, ask-side-crossing fill estimates
(never mid), and the delta-equivalent share position. All chains are crafted
synthetic frames; every timestamp predates the 2026-02-22 lockbox window."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.core.config import ExecutionConfig
from edge.core.events import SignalEvent, Side
from edge.contracts.selector import (
    CHAIN_COLUMNS,
    ContractChoice,
    ContractSelector,
    RejectionReason,
    SelectorConfig,
    shares_equivalent,
)

ET = ZoneInfo("America/New_York")
#: Decision instant, well BEFORE the 2026-02-22 lockbox start.
T0 = datetime(2026, 1, 6, 10, 0, tzinfo=ET)

# Expiries by days-to-expiry from T0's date (2026-01-06).
E_DTE0 = date(2026, 1, 6)
E_DTE1 = date(2026, 1, 7)
E_DTE3 = date(2026, 1, 9)
E_DTE6 = date(2026, 1, 12)
E_DTE10 = date(2026, 1, 16)

SPOT = 100.0
#: Comfortably above every accepted contract's breakeven in these tests.
EXCURSION = 0.05


@pytest.fixture
def selector() -> ContractSelector:
    """Hand-computable frictions (60% crossing, 2% slip, $0.65/contract)
    with the spec-default selection parameters (0.40 delta, 5%/3% spread,
    min_size 10, min_oi 100)."""
    execution = ExecutionConfig(
        spread_fill_fraction=0.6,
        slippage_pct=0.02,
        commission_per_contract=0.65,
        latency_ms=[100],
    )
    return ContractSelector(execution, SelectorConfig())


def sig(side: Side = Side.BUY) -> SignalEvent:
    return SignalEvent(
        ts=T0, symbol="SPY", side=side, conviction=0.8,
        horizon_minutes=120, signal_name="test_signal",
    )


def row(
    strike: float,
    expiry: date = E_DTE3,
    right: str = "C",
    bid: float = 1.00,
    ask: float = 1.04,
    delta: float = 0.40,
    oi: int = 500,
    quoted_size: int = 50,
) -> dict:
    return dict(strike=strike, expiry=expiry, right=right, bid=bid, ask=ask,
                delta=delta, oi=oi, quoted_size=quoted_size)


def chain(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(CHAIN_COLUMNS))


def pick(selector: ContractSelector, frame: pd.DataFrame, *,
         side: Side = Side.BUY, dte_rung: int = 2,
         excursion: float = EXCURSION, fat_tailed: bool = False) -> ContractChoice:
    return selector.select(
        sig(side), frame, spot=SPOT, dte_rung=dte_rung,
        median_favorable_excursion=excursion, fat_tailed=fat_tailed,
    )


# ---------------------------------------------------------------------------
# Delta targeting
# ---------------------------------------------------------------------------


def test_picks_delta_nearest_default_target(selector: ContractSelector) -> None:
    frame = chain(
        row(strike=106.0, delta=0.25),
        row(strike=101.0, delta=0.38),   # |0.38 - 0.40| = 0.02: nearest
        row(strike=97.0, delta=0.55),
    )
    choice = pick(selector, frame)
    assert choice.accepted and choice.rejection_reason is None
    assert len(choice.legs) == 1
    assert choice.legs[0].strike == 101.0
    assert choice.legs[0].delta == 0.38


def test_fat_tailed_flag_shifts_target_further_otm(selector: ContractSelector) -> None:
    frame = chain(
        row(strike=100.0, delta=0.40),
        row(strike=106.0, delta=0.25),
    )
    assert pick(selector, frame, fat_tailed=False).legs[0].strike == 100.0
    # Further OTM ONLY when the caller passes fat_tailed=True.
    assert pick(selector, frame, fat_tailed=True).legs[0].strike == 106.0


def test_delta_tie_breaks_to_higher_open_interest(selector: ContractSelector) -> None:
    # 0.35 and 0.45 are equidistant from the 0.40 target: liquidity decides.
    frame = chain(
        row(strike=105.0, delta=0.35, oi=200),
        row(strike=103.0, delta=0.45, oi=900),
    )
    assert pick(selector, frame).legs[0].strike == 103.0


def test_delta_tie_equal_oi_breaks_to_cheaper_mid(selector: ContractSelector) -> None:
    frame = chain(
        row(strike=105.0, delta=0.35, oi=500, bid=1.00, ask=1.04),  # mid 1.02
        row(strike=103.0, delta=0.45, oi=500, bid=1.10, ask=1.14),  # mid 1.12
    )
    assert pick(selector, frame).legs[0].strike == 105.0


def test_sell_signal_selects_put_by_abs_delta(selector: ContractSelector) -> None:
    frame = chain(
        row(strike=100.0, right="C", delta=0.40),    # wrong right: ignored
        row(strike=94.0, right="P", delta=-0.20),
        row(strike=99.0, right="P", delta=-0.41),    # |−0.41| nearest 0.40
    )
    choice = pick(selector, frame, side=Side.SELL)
    assert choice.accepted
    leg = choice.legs[0]
    assert leg.right == "P"
    assert leg.strike == 99.0
    assert leg.delta == -0.41          # signed delta preserved on the leg
    assert leg.side is Side.BUY        # the selector BUYS the put


# ---------------------------------------------------------------------------
# DTE ladder rungs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dte_rung, expected_strike",
    [
        (1, 101.0),   # exactly dte 1
        (2, 103.0),   # band [2, 5): the dte-3 contract, NOT dte-6
        (5, 106.0),   # band [5, 7): the dte-6 contract
        (7, 110.0),   # 7+: the dte-10 contract
    ],
)
def test_selection_restricted_to_rung(
    selector: ContractSelector, dte_rung: int, expected_strike: float
) -> None:
    frame = chain(
        row(strike=100.0, expiry=E_DTE0),
        row(strike=101.0, expiry=E_DTE1),
        row(strike=103.0, expiry=E_DTE3),
        row(strike=106.0, expiry=E_DTE6),
        row(strike=110.0, expiry=E_DTE10),
    )
    choice = pick(selector, frame, dte_rung=dte_rung)
    assert choice.accepted
    assert choice.legs[0].strike == expected_strike


def test_empty_rung_is_a_named_rejection(selector: ContractSelector) -> None:
    frame = chain(row(strike=110.0, expiry=E_DTE10))
    choice = pick(selector, frame, dte_rung=0)
    assert not choice.accepted
    assert choice.rejection_reason is RejectionReason.EMPTY_RUNG
    assert choice.legs == ()
    assert choice.rejection_detail  # never a silent nothing


def test_wrong_right_only_is_empty_rung(selector: ContractSelector) -> None:
    frame = chain(row(strike=100.0, right="C", delta=0.40))
    choice = pick(selector, frame, side=Side.SELL)  # needs puts, has none
    assert choice.rejection_reason is RejectionReason.EMPTY_RUNG


def test_invalid_rung_raises(selector: ContractSelector) -> None:
    with pytest.raises(ValueError, match="dte_rung"):
        pick(selector, chain(row(strike=100.0)), dte_rung=3)


# ---------------------------------------------------------------------------
# Breakeven gate, hand-computed
# ---------------------------------------------------------------------------
# Quote 1.00/1.04: mid = 1.02, crossed = 1.02 + 0.6*(1.04-1.02) = 1.032,
# premium = 1.032 * 1.02 = 1.05264 (ask-side crossing plus slippage).
# breakeven_move = (1.05264 / 0.40) / 100 = 0.0263160.

HAND_MID = (1.00 + 1.04) / 2.0
HAND_CROSSED = HAND_MID + 0.6 * (1.04 - HAND_MID)
HAND_PREMIUM = HAND_CROSSED * (1.0 + 0.02)
HAND_BREAKEVEN = (HAND_PREMIUM / 0.40) / SPOT


def test_breakeven_move_hand_computed(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0)), excursion=0.03)
    assert choice.accepted
    assert choice.breakeven_move == pytest.approx(0.0263160, abs=1e-7)
    assert choice.breakeven_move == pytest.approx(HAND_BREAKEVEN, abs=1e-12)


def test_breakeven_rejects_when_excursion_below(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0)), excursion=0.02)
    assert not choice.accepted
    assert choice.rejection_reason is RejectionReason.BREAKEVEN_UNREACHABLE
    assert choice.legs == ()
    # The rejection still reports the breakeven it could not clear.
    assert choice.breakeven_move == pytest.approx(HAND_BREAKEVEN, abs=1e-12)


def test_breakeven_boundary_equal_excursion_rejects(selector: ContractSelector) -> None:
    # excursion <= breakeven_move REJECTS: equality is not enough edge.
    choice = pick(selector, chain(row(strike=100.0)), excursion=HAND_BREAKEVEN)
    assert choice.rejection_reason is RejectionReason.BREAKEVEN_UNREACHABLE


def test_zero_delta_breakeven_is_infinite_and_rejects(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=140.0, delta=0.0)), excursion=0.99)
    assert choice.rejection_reason is RejectionReason.BREAKEVEN_UNREACHABLE


# ---------------------------------------------------------------------------
# Fill estimate: ask-side crossing, never mid
# ---------------------------------------------------------------------------


def test_est_fill_debit_uses_ask_side_crossing_not_mid(
    selector: ContractSelector,
) -> None:
    choice = pick(selector, chain(row(strike=100.0)))
    assert choice.accepted
    assert choice.est_fill_debit == pytest.approx(HAND_PREMIUM * 100, abs=1e-9)
    assert choice.est_fill_debit == pytest.approx(105.264, abs=1e-9)
    # Strictly worse than a mid fill: the crossing is real, not assumed away.
    assert choice.est_fill_debit > HAND_MID * 100


def test_max_loss_is_debit_plus_commission(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0)))
    assert choice.max_loss == pytest.approx(105.264 + 0.65, abs=1e-9)


# ---------------------------------------------------------------------------
# Liquidity hard gates, each the binding one
# ---------------------------------------------------------------------------


def test_wide_spread_gate(selector: ContractSelector) -> None:
    # rel_spread = 0.12 / 1.06 = 0.1132 > 0.05
    choice = pick(selector, chain(row(strike=100.0, bid=1.00, ask=1.12)))
    assert choice.rejection_reason is RejectionReason.WIDE_SPREAD
    assert "rel_spread" in (choice.rejection_detail or "")


def test_0dte_rung_uses_tighter_spread_cap(selector: ContractSelector) -> None:
    # rel_spread = 0.04 / 1.02 = 0.0392: inside 0.05, outside the 0DTE 0.03.
    quote = dict(bid=1.00, ask=1.04)
    ok = pick(selector, chain(row(strike=101.0, expiry=E_DTE1, **quote)), dte_rung=1)
    assert ok.accepted
    tight = pick(selector, chain(row(strike=100.0, expiry=E_DTE0, **quote)), dte_rung=0)
    assert tight.rejection_reason is RejectionReason.WIDE_SPREAD


def test_quoted_size_gate(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0, quoted_size=5)))
    assert choice.rejection_reason is RejectionReason.SIZE_BELOW_MIN


def test_open_interest_gate(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0, oi=50)))
    assert choice.rejection_reason is RejectionReason.OI_BELOW_MIN


def test_unquotable_zero_mid_is_no_quote(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0, bid=0.0, ask=0.0)))
    assert choice.rejection_reason is RejectionReason.NO_QUOTE


def test_gates_judge_the_delta_choice_without_fallback(
    selector: ContractSelector,
) -> None:
    # The delta-nearest contract fails a hard gate; a worse-delta contract
    # would pass. The selector must REJECT, not silently swap exposure.
    frame = chain(
        row(strike=100.0, delta=0.40, oi=50),     # nearest, fails min_oi
        row(strike=106.0, delta=0.30, oi=5000),   # passable, wrong delta
    )
    choice = pick(selector, frame)
    assert choice.rejection_reason is RejectionReason.OI_BELOW_MIN


def test_every_rejection_is_a_contract_choice_never_none(
    selector: ContractSelector,
) -> None:
    rejections = [
        pick(selector, chain(row(strike=110.0, expiry=E_DTE10)), dte_rung=0),
        pick(selector, chain(row(strike=100.0, bid=1.00, ask=1.12))),
        pick(selector, chain(row(strike=100.0, quoted_size=0))),
        pick(selector, chain(row(strike=100.0, oi=0))),
        pick(selector, chain(row(strike=100.0)), excursion=0.0),
    ]
    for choice in rejections:
        assert isinstance(choice, ContractChoice)
        assert not choice.accepted
        assert choice.rejection_reason is not None
        assert choice.rejection_detail
        assert choice.legs == ()


# ---------------------------------------------------------------------------
# shares_equivalent
# ---------------------------------------------------------------------------


def test_shares_equivalent_long_call(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=100.0, delta=0.40)))
    eq = shares_equivalent(choice, spot=SPOT)
    assert eq.shares == pytest.approx(40.0)          # 0.40 * 1 * 100
    assert eq.notional == pytest.approx(4000.0)      # 40 shares * $100


def test_shares_equivalent_long_put_is_short_shares(
    selector: ContractSelector,
) -> None:
    choice = pick(
        selector, chain(row(strike=99.0, right="P", delta=-0.40)), side=Side.SELL
    )
    eq = shares_equivalent(choice, spot=SPOT)
    assert eq.shares == pytest.approx(-40.0)
    assert eq.notional == pytest.approx(-4000.0)


def test_shares_equivalent_rejected_choice_raises(
    selector: ContractSelector,
) -> None:
    rejected = pick(selector, chain(row(strike=100.0, oi=0)))
    with pytest.raises(ValueError, match="rejected"):
        shares_equivalent(rejected, spot=SPOT)


# ---------------------------------------------------------------------------
# Input and config validation
# ---------------------------------------------------------------------------


def test_nonpositive_spot_raises(selector: ContractSelector) -> None:
    with pytest.raises(ValueError, match="spot"):
        selector.select(
            sig(), chain(row(strike=100.0)), spot=0.0, dte_rung=2,
            median_favorable_excursion=EXCURSION,
        )


def test_missing_chain_column_raises(selector: ContractSelector) -> None:
    frame = chain(row(strike=100.0)).drop(columns=["oi"])
    with pytest.raises(ValueError, match="oi"):
        pick(selector, frame)


def test_config_rejects_fat_tailed_target_less_otm() -> None:
    with pytest.raises(ValueError, match="fat_tailed_target_delta"):
        SelectorConfig(target_delta=0.40, fat_tailed_target_delta=0.50)


def test_config_rejects_looser_0dte_spread_cap() -> None:
    with pytest.raises(ValueError, match="max_rel_spread_0dte"):
        SelectorConfig(max_rel_spread=0.05, max_rel_spread_0dte=0.08)


def test_accepted_leg_metadata(selector: ContractSelector) -> None:
    choice = pick(selector, chain(row(strike=101.0, expiry=E_DTE3)))
    leg = choice.legs[0]
    assert leg.underlying == "SPY"
    assert leg.occ_symbol == "SPY260109C00101000"
    assert leg.expiry == E_DTE3
    assert leg.qty == 1
    assert leg.side is Side.BUY
    assert choice.contract_multiplier == 100

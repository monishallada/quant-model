"""Fixed-fractional sizing: budget arithmetic and skip-not-upsize behavior."""

from catalyst.risk.sizing import fixed_fractional_units


def test_basic_budget() -> None:
    # 3% of 100k = 3000; unit max loss 3.00/share = 300/unit -> 10 units.
    assert fixed_fractional_units(100_000, 0.03, 3.00) == 10


def test_rounds_down_never_up() -> None:
    assert fixed_fractional_units(100_000, 0.03, 4.10) == 7  # 3000/410 = 7.3


def test_zero_when_unit_exceeds_budget() -> None:
    # A 40.00 option = 4000/unit > 3000 budget -> skip entirely.
    assert fixed_fractional_units(100_000, 0.03, 40.0) == 0


def test_degenerate_inputs() -> None:
    assert fixed_fractional_units(0.0, 0.03, 3.0) == 0
    assert fixed_fractional_units(100_000, 0.03, 0.0) == 0
    assert fixed_fractional_units(-5.0, 0.03, 3.0) == 0

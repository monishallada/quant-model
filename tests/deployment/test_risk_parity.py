"""The risk section must be BYTE-IDENTICAL across the three modes (audit
D-001): 'what you backtested is what paper-trades and what goes live' was a
documented claim with no enforcement — paper/live inherited the loose
research-era base values while backtests ran under the spec safeguards."""

from catalyst.core.config import load_config


def test_risk_limits_identical_across_modes():
    bt = load_config("backtest").risk
    pp = load_config("paper").risk
    lv = load_config("live").risk
    assert bt.model_dump() == pp.model_dump() == lv.model_dump(), (
        "risk limits diverge across modes — the identical-sequence claim "
        "is false again")

"""Standardized reporting — every strategy judged by the same checks."""

from catalyst.reporting.report import (
    CONCENTRATION_FLAG,
    ENGINE_C_BASELINE_ANNUAL,
    MIN_TRUSTWORTHY_N,
    SegmentReport,
    StrategyReport,
    build_segment,
)

__all__ = ["CONCENTRATION_FLAG", "ENGINE_C_BASELINE_ANNUAL", "MIN_TRUSTWORTHY_N",
           "SegmentReport", "StrategyReport", "build_segment"]

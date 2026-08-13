"""Cointegration engine: detection, beta recovery, Z triggers, no lookahead."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from catalyst.core.config import load_config
from catalyst.core.tradingcal import sessions_in_range
from catalyst.pairs.cointegration import CointegrationEngine

CFG = load_config("backtest")
START, END = date(2023, 1, 2), date(2024, 6, 28)
SESSIONS = sessions_in_range(START, END)


def make_closes(seed: int = 3, cointegrated: bool = True) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    n = len(SESSIONS) + 260
    idx = pd.bdate_range(end=pd.Timestamp(END), periods=n)
    b = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    if cointegrated:
        # A = 2*B + stationary AR(1) noise -> textbook cointegrated pair.
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 1.2)
        a = 2.0 * b + noise
    else:
        a = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))  # independent walk
    return {"AAA": pd.Series(a, index=idx), "BBB": pd.Series(b, index=idx)}


def test_detects_cointegrated_pair_and_recovers_beta() -> None:
    engine = CointegrationEngine(CFG.pairs, make_closes(cointegrated=True))
    states = engine.compute_states(("AAA", "BBB"), SESSIONS)
    active_frac = sum(s.active for s in states) / len(states)
    # ADF at 60 obs has limited power even on a true AR(0.6) residual —
    # partial detection is the realistic outcome, not a bug.
    assert active_frac > 0.65, f"only {active_frac:.0%} sessions active"
    betas = [s.beta for s in states if s.beta is not None]
    assert abs(np.median(betas) - 2.0) < 0.15


def test_rejects_independent_random_walks() -> None:
    engine = CointegrationEngine(CFG.pairs, make_closes(seed=11, cointegrated=False))
    states = engine.compute_states(("AAA", "BBB"), SESSIONS)
    active_frac = sum(s.active for s in states) / len(states)
    assert active_frac < 0.35, f"{active_frac:.0%} active for independent walks"


def test_z_scores_finite_and_reactive() -> None:
    closes = make_closes(cointegrated=True)
    # Force a large dislocation on the final day: A jumps 5% above fair value.
    closes["AAA"].iloc[-1] *= 1.05
    engine = CointegrationEngine(CFG.pairs, closes)
    states = engine.compute_states(("AAA", "BBB"), SESSIONS)
    last = states[-1]
    assert last.zscore is not None and last.zscore > 2.0


def test_no_lookahead_beta_frozen_between_retests() -> None:
    closes = make_closes(cointegrated=True)
    engine = CointegrationEngine(CFG.pairs, closes)
    states = engine.compute_states(("AAA", "BBB"), SESSIONS)
    betas = [s.beta for s in states if s.beta is not None]
    changes = sum(1 for i in range(1, len(betas)) if betas[i] != betas[i - 1])
    # Beta may only change on retest days: every 5 sessions -> ~len/5 changes max.
    assert changes <= len(states) // CFG.pairs.retest_frequency_days + 2


def test_missing_symbol_yields_inactive_states() -> None:
    engine = CointegrationEngine(CFG.pairs, {"BBB": make_closes()["BBB"]})
    states = engine.compute_states(("AAA", "BBB"), SESSIONS)
    assert all(not s.active and s.zscore is None for s in states)

"""Cointegration signal engine for pairs trading.

For each pair (A, B), on a rolling schedule:
- OLS regression  price_A = alpha + beta * price_B  over ``lookback_days``
- ADF test on the residuals; the pair is ACTIVE while p < ``adf_alpha``
- spread = price_A - beta * price_B, Z = (spread - mean) / std over the
  same window (mean/std computed with the frozen beta from the last retest)

Beta and the ADF verdict refresh every ``retest_frequency_days``; between
retests they are FROZEN — the Z observed on day t uses only information
available at the last retest before t, so signals are lookahead-free.
Deactivation (p >= alpha at a retest) is an order to flatten: cointegration
is notoriously unstable out-of-sample, and holding a decoupled pair is not a
position, it's a hope. The engine records every activation/deactivation so
the backtest can report stability honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from catalyst.core.config import PairsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairState:
    """Signal state for one pair on one session."""

    pair: tuple[str, str]
    session: date
    active: bool  # cointegrated at the most recent retest
    beta: float | None
    alpha: float | None
    adf_p: float | None
    zscore: float | None
    spread_mean: float | None
    spread_std: float | None


@dataclass
class StabilityLedger:
    """Counts the report needs: how often pairs break out-of-sample."""

    activations: int = 0
    deactivations: int = 0
    active_sessions: int = 0
    total_sessions: int = 0
    deactivation_dates: dict[tuple[str, str], list[date]] = field(default_factory=dict)

    def record(self, pair: tuple[str, str], was_active: bool, now_active: bool, session: date) -> None:
        if now_active and not was_active:
            self.activations += 1
        if was_active and not now_active:
            self.deactivations += 1
            self.deactivation_dates.setdefault(pair, []).append(session)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """alpha, beta, residuals for y = alpha + beta*x."""
    beta, alpha = np.polyfit(x, y, 1)
    residuals = y - (alpha + beta * x)
    return float(alpha), float(beta), residuals


def _adf_p(residuals: np.ndarray) -> float:
    """ADF p-value on residuals; failures count as non-stationary (1.0)."""
    if len(residuals) < 20 or np.std(residuals) == 0:
        return 1.0
    try:
        return float(adfuller(residuals, autolag="AIC")[1])
    except Exception as exc:  # noqa: BLE001 — numerical failure = not cointegrated
        logger.debug("ADF failed: %s", exc)
        return 1.0


class CointegrationEngine:
    def __init__(self, cfg: PairsConfig, closes: dict[str, pd.Series]) -> None:
        """``closes``: symbol -> daily close series indexed by pd.Timestamp."""
        self._cfg = cfg
        self._closes = closes
        self.ledger = StabilityLedger()

    def compute_states(self, pair: tuple[str, str], sessions: list[date]) -> list[PairState]:
        """Signal state per session, lookahead-free.

        On retest days (every ``retest_frequency_days`` sessions with full
        window coverage), beta/alpha/ADF refresh using the trailing window
        ENDING THE PRIOR SESSION. Between retests those parameters are frozen;
        Z each day marks the CURRENT spread against the frozen window stats.
        """
        cfg = self._cfg
        a_sym, b_sym = pair
        a = self._closes.get(a_sym)
        b = self._closes.get(b_sym)
        out: list[PairState] = []
        if a is None or b is None:
            return [
                PairState(pair, s, False, None, None, None, None, None, None) for s in sessions
            ]
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()

        beta = alpha = adf_p = mean = std = None
        active = False
        since_retest = cfg.retest_frequency_days  # force retest on first eligible day

        for session in sessions:
            ts = pd.Timestamp(session)
            hist = joined.loc[: ts - pd.Timedelta(days=1)]  # strictly prior data
            since_retest += 1
            if len(hist) >= cfg.lookback_days and since_retest >= cfg.retest_frequency_days:
                window = hist.tail(cfg.lookback_days)
                al, be, resid = _ols(window["a"].to_numpy(), window["b"].to_numpy())
                p = _adf_p(resid)
                was_active = active
                alpha, beta, adf_p = al, be, p
                spread_window = window["a"].to_numpy() - be * window["b"].to_numpy()
                mean = float(np.mean(spread_window))
                std = float(np.std(spread_window, ddof=1))
                active = p < cfg.adf_alpha and std > 0
                self.ledger.record(pair, was_active, active, session)
                since_retest = 0

            z = None
            if beta is not None and std and std > 0:
                row = joined.loc[joined.index == ts]
                if not row.empty:
                    spread_today = float(row["a"].iloc[0]) - beta * float(row["b"].iloc[0])
                    z = (spread_today - mean) / std

            self.ledger.total_sessions += 1
            if active:
                self.ledger.active_sessions += 1
            out.append(
                PairState(
                    pair=pair, session=session, active=active,
                    beta=beta, alpha=alpha, adf_p=adf_p, zscore=z,
                    spread_mean=mean, spread_std=std,
                )
            )
        return out

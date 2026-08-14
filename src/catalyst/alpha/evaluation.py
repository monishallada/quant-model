"""Signal evaluation: information coefficient and decile spreads.

This is the diagnostic that should have come first in this project. Building a
portfolio around a signal that has no predictive power is expensive theatre;
the information coefficient answers "does this signal know anything?" in
seconds, before any structure, sizing, or cost model is involved.

IC = cross-sectional Spearman rank correlation between the signal on date t
and the forward return over (t, t+h]. It is computed per date and then
summarized, so the t-statistic reflects consistency through time rather than
one lucky period.

Interpretation benchmarks used by practitioners: |IC| ~0.02-0.03 is a weak but
real signal, ~0.05 is good, >0.10 in liquid equities is implausible and almost
always a bug or lookahead. The t-statistic matters more than the level: an IC
of 0.02 that is consistently positive beats an IC of 0.05 that is noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ICResult:
    signal: str
    horizon: int
    mean_ic: float
    ic_std: float
    t_stat: float
    hit_rate: float  # fraction of dates with positive IC
    n_dates: int
    decile_spread_ann: float  # annualized top-minus-bottom decile return
    top_decile_ann: float
    bottom_decile_ann: float

    @property
    def verdict(self) -> str:
        if abs(self.t_stat) < 2.0:
            return "no signal"
        if self.mean_ic > 0:
            return "predictive"
        return "predictive (inverted)"


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return over (t, t+horizon], aligned to date t. Strictly forward-looking
    by construction, which is what makes it a valid target for a signal at t."""
    return prices.shift(-horizon) / prices - 1.0


def information_coefficient(
    signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 20
) -> pd.Series:
    """Per-date cross-sectional rank correlation between signal and forward return."""
    common_dates = signal.index.intersection(fwd.index)
    out: dict[pd.Timestamp, float] = {}
    sig = signal.loc[common_dates]
    ret = fwd.loc[common_dates]
    for date in common_dates:
        s, r = sig.loc[date], ret.loc[date]
        pair = pd.concat([s, r], axis=1).dropna()
        if len(pair) < min_names:
            continue
        ic = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        if pd.notna(ic):
            out[date] = float(ic)
    return pd.Series(out).sort_index()


def decile_returns(
    signal: pd.DataFrame, fwd: pd.DataFrame, n_buckets: int = 10, min_names: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Per-date mean forward return of the top and bottom signal buckets."""
    top: dict[pd.Timestamp, float] = {}
    bottom: dict[pd.Timestamp, float] = {}
    for date in signal.index.intersection(fwd.index):
        pair = pd.concat([signal.loc[date], fwd.loc[date]], axis=1).dropna()
        if len(pair) < min_names:
            continue
        pair.columns = ["score", "ret"]
        ranked = pair.sort_values("score")
        k = max(len(ranked) // n_buckets, 1)
        bottom[date] = float(ranked["ret"].iloc[:k].mean())
        top[date] = float(ranked["ret"].iloc[-k:].mean())
    return pd.Series(top).sort_index(), pd.Series(bottom).sort_index()


def evaluate_signal(
    name: str, signal: pd.DataFrame, prices: pd.DataFrame, horizon: int,
    sample_every: int | None = None,
) -> ICResult:
    """Full evaluation of one signal at one horizon.

    ``sample_every`` defaults to the horizon so forward windows are strictly
    NON-OVERLAPPING. This matters enormously and is easy to get wrong: with
    daily sampling and a 63-day horizon, consecutive observations share 62 of
    63 days, the observations are almost perfectly autocorrelated, and the
    t-statistic is inflated by roughly sqrt(horizon) — turning noise into an
    apparently overwhelming result. Non-overlapping sampling costs sample size
    but is the only honest way to compute significance here.
    """
    if sample_every is None:
        sample_every = horizon
    fwd = forward_returns(prices, horizon)
    sig = signal.iloc[::sample_every]
    ic = information_coefficient(sig, fwd)
    if ic.empty:
        return ICResult(name, horizon, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    mean_ic = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    t_stat = mean_ic / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else 0.0
    hit = float((ic > 0).mean())

    top, bottom = decile_returns(sig, fwd)
    periods_per_year = 252 / horizon
    top_ann = float(top.mean()) * periods_per_year if len(top) else 0.0
    bot_ann = float(bottom.mean()) * periods_per_year if len(bottom) else 0.0

    return ICResult(
        signal=name, horizon=horizon, mean_ic=mean_ic, ic_std=ic_std, t_stat=float(t_stat),
        hit_rate=hit, n_dates=len(ic), decile_spread_ann=top_ann - bot_ann,
        top_decile_ann=top_ann, bottom_decile_ann=bot_ann,
    )


def evaluate_library(
    signals: dict[str, pd.DataFrame], prices: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        for name, panel in signals.items():
            r = evaluate_signal(name, panel, prices, horizon)
            rows.append({
                "signal": r.signal, "horizon_d": r.horizon, "mean_ic": r.mean_ic,
                "t_stat": r.t_stat, "hit_rate": r.hit_rate, "n_obs": r.n_dates,
                "decile_spread_ann": r.decile_spread_ann,
                "top_ann": r.top_decile_ann, "bottom_ann": r.bottom_decile_ann,
                "verdict": r.verdict,
            })
    return pd.DataFrame(rows)

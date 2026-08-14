"""Cross-sectional signal library.

Every signal maps a price panel (dates x symbols) to a score panel of the same
shape, where the score is a *cross-sectional* z-score: on each date, symbols
are ranked against each other, not against their own history. That framing is
deliberate — the four failed campaigns all used time-series signals ("is THIS
stock trending?"), which is the weakest and most competed-away form. Ranking
names against each other is the form with decades of out-of-sample support.

Every signal is strictly causal: the score on date t uses only closes at or
before t, and is evaluated against returns AFTER t. The random control exists
to prove the harness: if it shows information coefficient materially different
from zero, the evaluation is broken and no other number here can be trusted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def cross_sectional_zscore(panel: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Z-score each row (date) across symbols, winsorized to limit outliers."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0)
    z = panel.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return z.clip(-winsor, winsor)


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


# ---------------------------------------------------------------------------
# Signal definitions. Each returns a raw score panel; z-scoring happens once,
# centrally, so every signal enters the combiner on the same scale.
# ---------------------------------------------------------------------------


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic cross-sectional momentum: return over [t-lookback, t-skip].

    The skip window matters: the most recent month is dominated by short-term
    reversal, and including it is the standard way to blunt the signal.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def short_term_reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Contrarian: recent losers outperform. Sign is inverted so that a HIGH
    score always means "expected to outperform", consistent across the library."""
    return -(prices / prices.shift(lookback) - 1.0)


def low_volatility(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """The low-beta/low-vol anomaly: inverted trailing realized volatility."""
    return -_returns(prices).rolling(lookback).std()


def trend_following(prices: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """Time-series trend expressed cross-sectionally: distance above the MA."""
    return prices / prices.rolling(lookback).mean() - 1.0


def volatility_adjusted_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21, vol_window: int = 60
) -> pd.DataFrame:
    """Momentum per unit of risk — favours steady climbers over violent ones."""
    mom = momentum(prices, lookback, skip)
    vol = _returns(prices).rolling(vol_window).std()
    return mom / vol.replace(0.0, np.nan)


def residual_momentum(
    prices: pd.DataFrame, market: pd.Series, lookback: int = 252, skip: int = 21,
    beta_window: int = 120,
) -> pd.DataFrame:
    """Momentum with market beta stripped out.

    Plain momentum is partly a bet on high-beta names in a rising market. This
    variant regresses each name on the market and keeps only the idiosyncratic
    part, so the signal is not smuggling in market exposure.
    """
    rets = _returns(prices)
    mkt = market.pct_change().reindex(rets.index)
    mkt_var = mkt.rolling(beta_window).var()
    resid_cum = {}
    for sym in rets.columns:
        cov = rets[sym].rolling(beta_window).cov(mkt)
        beta = (cov / mkt_var.replace(0.0, np.nan)).clip(-3, 3)
        resid = rets[sym] - beta * mkt
        # Cumulative residual return over the momentum window, skipping the
        # most recent month for the same reason plain momentum does.
        resid_cum[sym] = resid.shift(skip).rolling(lookback - skip).sum()
    return pd.DataFrame(resid_cum, index=rets.index)


def random_control(prices: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Pure noise. MUST show ~zero information coefficient; if it does not,
    the evaluation harness has a lookahead or alignment bug."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    )


def build_signal_panels(
    prices: pd.DataFrame, market: pd.Series
) -> dict[str, pd.DataFrame]:
    """The full library, each z-scored cross-sectionally."""
    raw = {
        "mom_12_1": momentum(prices, 252, 21),
        "mom_6_1": momentum(prices, 126, 21),
        "reversal_1m": short_term_reversal(prices, 21),
        "low_vol": low_volatility(prices, 60),
        "trend_200": trend_following(prices, 200),
        "vol_adj_mom": volatility_adjusted_momentum(prices),
        "residual_mom": residual_momentum(prices, market),
        "random_control": random_control(prices),
    }
    return {name: cross_sectional_zscore(panel) for name, panel in raw.items()}

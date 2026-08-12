"""Monte Carlo path resampling: probability of ruin / survival rate.

Block-bootstraps the realized daily return series (preserving short-range
autocorrelation) into ``n_paths`` alternate histories of the same length and
measures how often equity ever breaches the ruin threshold. This answers the
survival question the point-estimate equity curve cannot: how bad could the
same trade distribution have been in a different order?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst.core.config import MonteCarloConfig


def probability_of_ruin(
    daily_returns: pd.Series,
    cfg: MonteCarloConfig,
    seed: int = 7,
) -> float:
    """Fraction of resampled paths whose equity ever drops below
    ``ruin_threshold_fraction`` of starting equity."""
    values = daily_returns.to_numpy()
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        return 0.0
    block = max(1, min(cfg.resample_block_size, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))

    ruined = 0
    for _ in range(cfg.n_paths):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        path = np.concatenate([values[s : s + block] for s in starts])[:n]
        equity = np.cumprod(1.0 + path)
        if equity.min() < cfg.ruin_threshold_fraction:
            ruined += 1
    return ruined / cfg.n_paths

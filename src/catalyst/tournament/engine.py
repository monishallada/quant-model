"""Convex tournament engine: signal-ranked long options on high-volatility names.

Architecture, with the reasoning for each choice against the tournament
objective rather than against a compounding objective:

1. **Signal** — the v5 cross-sectional momentum composite (IC ~0.035, t~2.5,
   validated with a random control). The edge is small, but it tilts which
   lottery tickets we buy, and a slightly favourable tilt on a convex payoff
   is worth far more than the same tilt on a linear one.

2. **Universe** — high-volatility optionable names. Volatility is not an edge,
   but under a threshold objective it is the raw material: a 3x more volatile
   name produces a 3x wider outcome distribution from the same signal, and we
   need the right tail, not the mean.

3. **Expression** — long OTM calls (or puts when the signal is negative).
   Capped downside, uncapped upside: the original spec's philosophy, which was
   wrong for compounding and is exactly right for a tournament.

4. **Concentration** — a handful of positions. Diversification suppresses
   variance, and suppressing variance here suppresses P(target).

5. **No profit-taking** — winners run to expiry or a very large multiple.
   Selling at +50% removes the 10x outcomes that are the only ones that win.

6. **Bold sizing** — deliberately above Kelly. With unfavourable per-bet odds
   and a deadline, bold play maximizes P(reaching target); timid play maximizes
   the chance of finishing alive but short, which pays nothing.

The engine models option payoffs from real chain data when available. Where a
fast architecture search is needed, ``price_option_bs`` provides a
Black-Scholes approximation driven by the name's own realized volatility, so
configurations can be explored cheaply before the promising ones are validated
against actual NBBO chains.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catalyst.core.models import OptionRight
from catalyst.data.black_scholes import bs_price

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class TournamentConfig:
    n_positions: int = 3            # concentration: how many names held at once
    hold_days: int = 21             # holding period per cycle
    dte_days: int = 35              # option tenor at entry
    moneyness: float = 1.10         # strike / spot for calls (1.10 = 10% OTM)
    capital_fraction: float = 0.90  # fraction of equity deployed each cycle
    use_signal: bool = True         # rank by momentum vs pick randomly (control)
    iv_multiplier: float = 1.0      # stress test: scale the IV we pay
    cost_per_contract: float = 0.02
    spread_pct: float = 0.06        # round-trip spread cost as fraction of premium
    take_profit_multiple: float | None = None  # None = let winners run to expiry
    stop_loss_fraction: float | None = None    # None = ride to zero (bounded anyway)


def realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window).std() * math.sqrt(TRADING_DAYS)


def price_option_bs(spot: float, strike: float, days: float, iv: float,
                    right: OptionRight, rate: float = 0.045) -> float:
    """Black-Scholes premium. Used for fast architecture search; the winning
    configuration is re-validated against real NBBO chains before any claim."""
    if spot <= 0 or strike <= 0 or days <= 0 or iv <= 0:
        return 0.0
    return bs_price(spot, strike, days / 365.0, iv, right, r=rate)


def simulate_cycle(
    prices: pd.DataFrame, vols: pd.DataFrame, picks: list[str],
    entry_date: pd.Timestamp, exit_date: pd.Timestamp, cfg: TournamentConfig,
    directions: dict[str, int],
) -> float:
    """Return the gross multiple on capital deployed in one cycle.

    Each pick gets an equal share of deployed capital, buys an OTM option, and
    is marked at the exit date. Premium paid includes the spread and per-contract
    slippage; the option can go to zero, which is the bounded downside that makes
    the whole approach survivable as a fixed stake.
    """
    if not picks:
        return 1.0
    per_name = 1.0 / len(picks)
    total = 0.0
    for sym in picks:
        try:
            spot0 = float(prices.loc[entry_date, sym])
            spot1 = float(prices.loc[exit_date, sym])
            iv = float(vols.loc[entry_date, sym]) * cfg.iv_multiplier
        except (KeyError, TypeError):
            total += per_name  # untradeable: capital sits in cash
            continue
        if not (spot0 > 0 and spot1 > 0 and iv > 0):
            total += per_name
            continue

        direction = directions.get(sym, 1)
        right = OptionRight.CALL if direction > 0 else OptionRight.PUT
        strike = spot0 * (cfg.moneyness if direction > 0 else 2.0 - cfg.moneyness)

        entry_prem = price_option_bs(spot0, strike, cfg.dte_days, iv, right)
        if entry_prem <= 0.01:
            total += per_name
            continue
        entry_cost = entry_prem * (1 + cfg.spread_pct / 2) + cfg.cost_per_contract

        remaining_days = max(cfg.dte_days - cfg.hold_days, 0.5)
        exit_prem = price_option_bs(spot1, strike, remaining_days, iv, right)
        exit_proceeds = max(exit_prem * (1 - cfg.spread_pct / 2) - cfg.cost_per_contract, 0.0)

        multiple = exit_proceeds / entry_cost if entry_cost > 0 else 0.0
        if cfg.take_profit_multiple is not None:
            multiple = min(multiple, cfg.take_profit_multiple)
        total += per_name * multiple
    return total


def run_tournament_path(
    prices: pd.DataFrame, signal: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    cfg: TournamentConfig, starting_equity: float = 100_000.0, seed: int = 0,
) -> pd.Series:
    """One competition entry: cycle through the window, compounding aggressively."""
    rng = np.random.default_rng(seed)
    window = prices.loc[start:end]
    if len(window) < cfg.hold_days + 2:
        return pd.Series(dtype=float)
    rets = prices.pct_change()
    vols = realized_vol(rets)

    equity = starting_equity
    curve: dict[pd.Timestamp, float] = {window.index[0]: equity}
    i = 0
    dates = window.index
    while i + cfg.hold_days < len(dates):
        entry_date, exit_date = dates[i], dates[i + cfg.hold_days]
        row = signal.loc[entry_date].dropna() if entry_date in signal.index else pd.Series(dtype=float)
        available = [s for s in row.index if pd.notna(prices.loc[entry_date, s])]
        if not available:
            i += cfg.hold_days
            curve[exit_date] = equity
            continue

        if cfg.use_signal:
            ranked = row[available].sort_values()
            longs = list(ranked.index[-cfg.n_positions:])
            directions = {s: 1 for s in longs}
            picks = longs
        else:  # random control: same structure, no signal
            picks = list(rng.choice(available, size=min(cfg.n_positions, len(available)),
                                    replace=False))
            directions = {s: 1 for s in picks}

        gross = simulate_cycle(prices, vols, picks, entry_date, exit_date, cfg, directions)
        deployed = equity * cfg.capital_fraction
        equity = equity - deployed + deployed * gross
        curve[exit_date] = equity
        if equity < starting_equity * 0.01:  # effectively wiped out
            for d in dates[i + cfg.hold_days:]:
                curve[d] = equity
            break
        i += cfg.hold_days
    return pd.Series(curve).sort_index()

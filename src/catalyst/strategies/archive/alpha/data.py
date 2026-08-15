"""Price panel assembly for the alpha research platform.

Builds a (dates x symbols) close-price panel plus a market series from the
cached SIP daily bars. Symbols with sparse history are dropped rather than
forward-filled: a name that did not trade for a stretch produces fake returns
when filled, and those fake returns land straight in the information
coefficient.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from catalyst.core.config import Config
from catalyst.data.cache import ParquetCache
from catalyst.data.intraday import AlpacaMinuteBars

logger = logging.getLogger(__name__)

MARKET_SYMBOL = "SPY"


def load_price_panel(
    cfg: Config, symbols: list[str], start: date, end: date,
    min_coverage: float = 0.60,
) -> tuple[pd.DataFrame, pd.Series]:
    """(close panel, market close series), both date-indexed and aligned."""
    cache = ParquetCache(cfg.data.cache_dir)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")

    series: dict[str, pd.Series] = {}
    for i, sym in enumerate(sorted(set(symbols) | {MARKET_SYMBOL}), 1):
        df = bars.daily_bars(sym, start, end)
        if df.empty:
            logger.warning("No daily bars for %s", sym)
            continue
        series[sym] = pd.Series(df["close"].to_numpy(dtype=float),
                                index=pd.to_datetime(pd.Index(df.index)))
        if i % 25 == 0:
            logger.info("loaded %d/%d symbols", i, len(symbols) + 1)

    if MARKET_SYMBOL not in series:
        raise RuntimeError(f"{MARKET_SYMBOL} is required as the market benchmark")

    panel = pd.DataFrame(series).sort_index()
    market = panel[MARKET_SYMBOL]

    # Drop thin names: coverage measured against the market's own calendar.
    coverage = panel.notna().sum() / max(market.notna().sum(), 1)
    keep = [c for c in panel.columns if coverage[c] >= min_coverage and c != MARKET_SYMBOL]
    dropped = [c for c in panel.columns if c not in keep and c != MARKET_SYMBOL]
    if dropped:
        logger.info("dropped %d thin symbols: %s", len(dropped), ", ".join(sorted(dropped)))
    return panel[keep], market

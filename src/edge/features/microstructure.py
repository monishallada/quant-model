"""Trade-tape microstructure features: who is transacting, how aggressively.

Every wave-1 signal read PRICE and derived a view from price. These features
read the TAPE — the sequence of prints, their signs, and their volume clock —
which carries information no OHLCV transform contains: whether a move was
bought or sold into, and whether volume is arriving in informed-looking
bursts.

Pure functions over an already-loaded trade frame (``ts``, ``price``,
``size``); this module never fetches. The caller supplies data through the
gateway loader, and the point-in-time contract is the caller's:

    a feature stamped at minute close ``t`` is computed from prints with
    ``ts <= t`` and is consumed at ``t + 1`` bar, exactly like a BarEvent.

Implemented here:

* **tick-rule signs** (Lee-Ready's degenerate case when no quote is
  available): a print above the previous distinct price is buyer-initiated,
  below is seller-initiated, equal inherits the last non-zero sign.
* **trade-sign imbalance** — signed volume over a rolling window,
  normalised by total volume: +1 all lifted offers, -1 all hit bids.
* **VPIN** (Easley/Lopez de Prado/O'Hara) on a VOLUME clock: buckets of
  equal volume, order-imbalance per bucket, averaged over the trailing
  ``n_buckets``. Bucket boundaries fall mid-print, so a print is split
  across buckets rather than being assigned wholly to one — the volume
  clock is exact, not approximated by print counts.
* **trade-intensity z** — prints per minute against the session's own
  trailing distribution, the "something is happening" gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "bucket_vpin",
    "minute_microstructure",
    "tick_rule_signs",
    "trade_sign_imbalance",
]


def tick_rule_signs(price: np.ndarray) -> np.ndarray:
    """+1 buyer-initiated, -1 seller-initiated, per the tick rule.

    A print at an unchanged price inherits the previous non-zero sign; a
    leading run of unchanged prints has no predecessor and stays 0 (never
    guessed).
    """
    if price.size == 0:
        return np.zeros(0, dtype=np.int8)
    diff = np.diff(price, prepend=price[0])
    signs = np.sign(diff).astype(np.int8)
    # carry the last non-zero sign forward across zero-tick runs
    nonzero = signs != 0
    idx = np.where(nonzero, np.arange(signs.size), 0)
    np.maximum.accumulate(idx, out=idx)
    carried = signs[idx]
    # positions before the first non-zero sign stay unsigned
    if nonzero.any():
        first = int(np.argmax(nonzero))
        carried[:first] = 0
    else:
        carried[:] = 0
    return carried.astype(np.int8)


def trade_sign_imbalance(size: np.ndarray, signs: np.ndarray) -> float:
    """Signed volume share in [-1, 1]; 0.0 when no volume is present."""
    total = float(size.sum())
    if total <= 0.0:
        return 0.0
    return float((size * signs).sum() / total)


def bucket_vpin(
    size: np.ndarray, signs: np.ndarray, bucket_volume: float, n_buckets: int
) -> float:
    """VPIN over the trailing ``n_buckets`` equal-volume buckets.

    Volume is allocated exactly: a print straddling a bucket boundary is
    split proportionally, so buckets hold identical volume by construction.
    Returns NaN when fewer than ``n_buckets`` complete buckets exist — an
    unformed statistic is never reported as a number.
    """
    if bucket_volume <= 0 or n_buckets <= 0 or size.size == 0:
        return float("nan")
    buy = np.where(signs > 0, size, 0.0).astype(float)
    sell = np.where(signs < 0, size, 0.0).astype(float)
    # unsigned prints split evenly: they inform neither side
    unsigned = np.where(signs == 0, size, 0.0).astype(float) / 2.0
    buy, sell = buy + unsigned, sell + unsigned

    cum = np.cumsum(buy + sell)
    total = float(cum[-1])
    n_complete = int(total // bucket_volume)
    if n_complete < n_buckets:
        return float("nan")
    edges = np.arange(1, n_complete + 1, dtype=float) * bucket_volume
    # interpolate cumulative buy/sell at each bucket edge (exact split)
    cum_buy = np.cumsum(buy)
    cum_sell = np.cumsum(sell)
    grid = np.concatenate([[0.0], cum])
    buy_at = np.interp(edges, grid, np.concatenate([[0.0], cum_buy]))
    sell_at = np.interp(edges, grid, np.concatenate([[0.0], cum_sell]))
    per_buy = np.diff(np.concatenate([[0.0], buy_at]))
    per_sell = np.diff(np.concatenate([[0.0], sell_at]))
    imbalance = np.abs(per_buy - per_sell) / bucket_volume
    return float(imbalance[-n_buckets:].mean())


def minute_microstructure(
    trades: pd.DataFrame,
    *,
    imbalance_window_minutes: int = 5,
    vpin_bucket_volume: float | None = None,
    vpin_buckets: int = 50,
    session_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Per-minute microstructure frame for one symbol-session.

    Returns one row per minute that had at least one print, with columns:
    ``ts`` (minute CLOSE, tz-aware), ``signed_volume``, ``volume``,
    ``imbalance`` (rolling ``imbalance_window_minutes``), ``vpin``
    (trailing, NaN until formed), ``n_prints``, ``intensity_z``.

    The frame is causal by construction: every row uses prints at or before
    its own minute close, so a consumer keying on ``ts`` is point-in-time
    exactly like the bar tape.
    """
    if trades.empty:
        return pd.DataFrame(columns=[
            "ts", "volume", "signed_volume", "imbalance", "vpin",
            "n_prints", "intensity_z",
        ])
    df = trades.loc[:, ["ts", "price", "size"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(session_tz)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    signs = tick_rule_signs(df["price"].to_numpy(dtype=float))
    df["sign"] = signs
    df["signed"] = df["size"].to_numpy(dtype=float) * signs
    # minute CLOSE labelling: a print at 09:30:00.4 belongs to the 09:31 close
    minute_close = df["ts"].dt.ceil("1min")
    grouped = df.groupby(minute_close, sort=True)
    out = pd.DataFrame({
        "volume": grouped["size"].sum().astype(float),
        "signed_volume": grouped["signed"].sum(),
        "n_prints": grouped.size().astype(int),
    })
    out.index.name = "ts"
    win = max(int(imbalance_window_minutes), 1)
    roll_signed = out["signed_volume"].rolling(win, min_periods=1).sum()
    roll_volume = out["volume"].rolling(win, min_periods=1).sum()
    out["imbalance"] = np.where(roll_volume > 0, roll_signed / roll_volume, 0.0)
    mean = out["n_prints"].expanding(min_periods=20).mean()
    std = out["n_prints"].expanding(min_periods=20).std(ddof=0)
    out["intensity_z"] = np.where(std > 0, (out["n_prints"] - mean) / std, np.nan)

    bucket = vpin_bucket_volume
    if bucket is None:
        # 1/50th of the session's median minute volume x the window: a
        # volume clock scaled to the symbol, not a hardcoded share count.
        bucket = float(out["volume"].median()) or 1.0
    sizes = df["size"].to_numpy(dtype=float)
    ends = np.searchsorted(df["ts"].to_numpy(), out.index.to_numpy(), side="right")
    vpins = [
        bucket_vpin(sizes[:e], signs[:e], bucket, vpin_buckets) if e > 0 else float("nan")
        for e in ends
    ]
    out["vpin"] = vpins
    return out.reset_index()

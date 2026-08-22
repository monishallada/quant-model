"""Market internals over an equity universe: breadth, signed volume, TICK proxy.

Pure functions over ALREADY-LOADED per-symbol trade frames — the exact shape
the data layer's tick cache serves (columns ``ts``/``price``/``size``,
ascending, tz-aware). This module never fetches: research code obtains the
input frames through :class:`edge.data.loader.EdgeDataLoader` (the only
sanctioned gateway) and hands them here. Living in ``features/``, this file
is inside the AST gateway scan, so it imports nothing from ``edge.data``
beyond what the loader delivers as plain DataFrames.

Computed per minute, across a symbol universe:

- **advance/decline breadth** — share of the universe whose last price is
  strictly above (a) its own cumulative session VWAP and (b) its prior
  session close. Ties count as NOT above (conservative).
- **up-volume / down-volume** — per-trade tick-rule signing (``+1`` if the
  price ticked up, ``-1`` if down, else the previous sign; the first trade
  of the session is seeded ``+1``, the same arbitrary-seed convention as the
  imbalance-bar builder), aggregated to per-minute up/down share volume and
  their ratio (``NaN`` when down-volume is zero — never ``inf``).
- **TICK proxy** — count of symbols whose LAST trade was an uptick minus
  those whose last trade was a downtick (tick-rule sign, carried forward
  through tradeless minutes).

Nothing is ATR-normalized here: outputs are raw internals plus optional
trailing z-scores (:func:`add_zscores`).

Minute labeling matches the bar builders: a minute covering ``[t0, t0+1min)``
is labeled by its CLOSE time ``t0+1min``, so a row's values summarize trades
strictly before its label.

POINT-IN-TIME CONVENTION — every output frame of :func:`compute_internals`
carries ``asof_date`` (the minute's own America/New_York trading date) and
``available_at`` (tz-aware America/New_York datetime when a live trader
could FIRST have seen the row). ``available_at`` equals the minute's close
instant with zero added lag, because the inputs are real-time trade prints
fully observed by the close. Consumers, however, receive completed bars
under the engine's BarEvent-visibility semantics: a bar closing at ``t`` is
delivered strictly after ``t``, so the FIRST decision that can key on the
minute-``t`` internals row is the one taken on the ``t+1min`` bar. Feature
joins keyed ``available_at <= decision_ts`` therefore reproduce exactly the
engine's one-bar visibility lag — do not subtract or add another lag on top.

All computation is single-session: session VWAP, tick-sign seeding, and the
scalar ``prior_close`` are per-session concepts, so a frame spanning more
than one ET trading date is rejected rather than silently mixed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd

from edge.core.events import MARKET_TZ, TradeEvent

#: Required columns of an input per-symbol trades frame (the tick-cache shape).
TRADE_COLUMNS: Final[tuple[str, ...]] = ("ts", "price", "size")

#: Columns of a per-symbol minute frame produced by :func:`minute_symbol_frame`.
SYMBOL_COLUMNS: Final[tuple[str, ...]] = (
    "close",
    "volume",
    "vwap",
    "up_volume",
    "down_volume",
    "last_tick",
)

#: The internals metric columns (z-score candidates), in output order.
INTERNALS_METRICS: Final[tuple[str, ...]] = (
    "breadth_above_vwap",
    "breadth_above_prior_close",
    "updown_volume_ratio",
    "tick_proxy",
)

#: Full column set of a :func:`compute_internals` output frame.
INTERNALS_COLUMNS: Final[tuple[str, ...]] = (
    "asof_date",
    "available_at",
    "n_symbols",
    "breadth_above_vwap",
    "breadth_above_prior_close",
    "up_volume",
    "down_volume",
    "updown_volume_ratio",
    "tick_proxy",
)


def trades_to_frame(trades: Iterable[TradeEvent]) -> pd.DataFrame:
    """Convert core TradeEvents to the ``ts``/``price``/``size`` frame shape.

    Convenience for callers holding event lists; the timestamps are already
    tz-aware America/New_York by the event contract.
    """
    rows = [{"ts": t.ts, "price": t.price, "size": t.size} for t in trades]
    if not rows:
        return pd.DataFrame(
            {
                "ts": pd.Series(pd.DatetimeIndex([], tz=MARKET_TZ)),
                "price": pd.Series(dtype=float),
                "size": pd.Series(dtype="int64"),
            }
        )
    return pd.DataFrame(rows)


def _validated_ts(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Validate an input trades frame; return its ``ts`` as an ET index.

    Raises ``ValueError`` on missing columns, naive timestamps, out-of-order
    timestamps, or trades spanning more than one ET trading date — a
    malformed feed is a data bug, not something to paper over.
    """
    missing = [c for c in TRADE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"trades frame missing columns {missing}; need {list(TRADE_COLUMNS)}")
    if len(frame) == 0:
        return pd.DatetimeIndex([], tz=MARKET_TZ)
    ts = pd.DatetimeIndex(frame["ts"])
    if ts.tz is None:
        raise ValueError("trades ts must be tz-aware (America/New_York); got naive timestamps")
    ts = ts.tz_convert(MARKET_TZ)
    if not ts.is_monotonic_increasing:
        raise ValueError("out-of-order trades: ts must be non-decreasing")
    sessions = set(ts.date)
    if len(sessions) > 1:
        raise ValueError(
            f"trades span multiple ET sessions {sorted(sessions)}; "
            "internals are single-session (session VWAP and prior close are per-session)"
        )
    return ts


def _tick_signs(price: np.ndarray) -> np.ndarray:
    """Tick-rule sign per trade: +1 up, -1 down, unchanged inherits the
    previous sign; the first trade is seeded +1 (imbalance-bar convention)."""
    signs = np.concatenate(([1.0], np.sign(np.diff(price))))
    signs[signs == 0.0] = np.nan
    return pd.Series(signs).ffill().to_numpy()


def minute_symbol_frame(trades: pd.DataFrame, *, freq: str | pd.Timedelta = "1min") -> pd.DataFrame:
    """One symbol's per-minute state and flows from its session trade prints.

    Input: a ``ts``/``price``/``size`` frame (single ET session, ascending,
    tz-aware). Output: a frame indexed by minute CLOSE time (tz-aware ET,
    index name ``ts``) from the symbol's first traded minute through its
    last, with columns:

    - ``close`` — last trade price at or before the minute close (state,
      carried forward through tradeless minutes);
    - ``volume`` — shares traded within the minute (0 when tradeless);
    - ``vwap`` — cumulative session VWAP through the minute close (state,
      carried forward);
    - ``up_volume`` / ``down_volume`` — tick-rule signed share volume within
      the minute (0 when tradeless);
    - ``last_tick`` — tick-rule sign (+1.0/-1.0) of the most recent trade at
      or before the minute close (state, carried forward).

    An empty input yields an empty frame with the same columns.
    """
    step = pd.Timedelta(freq)
    if step <= pd.Timedelta(0):
        raise ValueError(f"freq must be a positive interval, got {freq!r}")
    ts = _validated_ts(trades)
    if len(ts) == 0:
        idx = pd.DatetimeIndex([], tz=MARKET_TZ, name="ts")
        return pd.DataFrame({c: pd.Series(dtype=float) for c in SYMBOL_COLUMNS}, index=idx)

    price = trades["price"].to_numpy(dtype=float)
    size = trades["size"].to_numpy(dtype=float)
    signs = _tick_signs(price)

    # Bar covering [t0, t0+step) is labeled by its close t0+step, matching the
    # time-bar builder; a trade exactly on a boundary opens the NEXT bar.
    close_label = ts.floor(step) + step

    per_trade = pd.DataFrame(
        {
            "close": price,
            "volume": size,
            "vwap": np.cumsum(price * size) / np.cumsum(size),
            "up_volume": np.where(signs > 0, size, 0.0),
            "down_volume": np.where(signs < 0, size, 0.0),
            "last_tick": signs,
        },
        index=close_label,
    )
    grouped = per_trade.groupby(level=0)
    out = pd.DataFrame(
        {
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "vwap": grouped["vwap"].last(),
            "up_volume": grouped["up_volume"].sum(),
            "down_volume": grouped["down_volume"].sum(),
            "last_tick": grouped["last_tick"].last(),
        }
    )
    # Fill interior tradeless minutes: state persists, flows are zero.
    grid = pd.date_range(out.index[0], out.index[-1], freq=step)
    out = out.reindex(grid)
    flow_cols = ["volume", "up_volume", "down_volume"]
    out[flow_cols] = out[flow_cols].fillna(0.0).astype("int64")
    out[["close", "vwap", "last_tick"]] = out[["close", "vwap", "last_tick"]].ffill()
    out.index.name = "ts"
    return out


def compute_internals(
    trades_by_symbol: Mapping[str, pd.DataFrame],
    prior_close: Mapping[str, float],
    *,
    freq: str | pd.Timedelta = "1min",
) -> pd.DataFrame:
    """Per-minute market internals across a universe of trade streams.

    Args:
        trades_by_symbol: per-symbol ``ts``/``price``/``size`` frames, all
            covering the SAME single ET session (symbols with no trades are
            allowed and simply never enter any denominator).
        prior_close: prior-session close per symbol; required (positive) for
            EVERY symbol in ``trades_by_symbol``.
        freq: minute-bar interval (default one minute).

    Returns a frame indexed by minute close (tz-aware ET, name ``ts``) over
    the union grid from the earliest traded minute to the latest, with
    columns :data:`INTERNALS_COLUMNS`:

    - ``asof_date`` — the minute's ET trading date;
    - ``available_at`` — the minute close instant itself (zero added lag;
      see the module docstring: BarEvent semantics mean the first decision
      that can consume a minute-``t`` row is the ``t+1min`` one);
    - ``n_symbols`` — denominator: symbols that have traded at least once so
      far this session (a symbol enters at its first traded minute and its
      state then persists through session end);
    - ``breadth_above_vwap`` — share of ``n_symbols`` with last price
      strictly above their own cumulative session VWAP;
    - ``breadth_above_prior_close`` — share strictly above their prior
      session close;
    - ``up_volume`` / ``down_volume`` — universe tick-rule signed share
      volume within the minute;
    - ``updown_volume_ratio`` — ``up_volume / down_volume``; ``NaN`` when
      ``down_volume`` is zero (never ``inf``);
    - ``tick_proxy`` — count of symbols whose last trade was an uptick minus
      those on a downtick.

    A universe whose symbols all lack trades yields an empty typed frame.
    Raises ``ValueError`` on an empty universe mapping, a symbol missing
    from ``prior_close`` (or with a non-positive value), or symbols covering
    different ET sessions.
    """
    if not trades_by_symbol:
        raise ValueError("empty universe: trades_by_symbol has no symbols")
    for symbol in trades_by_symbol:
        if symbol not in prior_close:
            raise ValueError(f"prior_close missing symbol {symbol!r}")
        if not prior_close[symbol] > 0.0:
            raise ValueError(
                f"prior_close for {symbol!r} must be positive, got {prior_close[symbol]!r}"
            )

    frames = {
        symbol: mf
        for symbol, tf in trades_by_symbol.items()
        if not (mf := minute_symbol_frame(tf, freq=freq)).empty
    }
    if not frames:
        idx = pd.DatetimeIndex([], tz=MARKET_TZ, name="ts")
        return pd.DataFrame({c: pd.Series(dtype=object) for c in INTERNALS_COLUMNS}, index=idx)

    sessions = {f.index[0].date() for f in frames.values()}
    sessions |= {f.index[-1].date() for f in frames.values()}
    if len(sessions) > 1:
        raise ValueError(
            f"symbols cover different ET sessions {sorted(sessions)}; "
            "internals are computed one session at a time"
        )

    step = pd.Timedelta(freq)
    grid = pd.date_range(
        min(f.index[0] for f in frames.values()),
        max(f.index[-1] for f in frames.values()),
        freq=step,
    )
    # State columns forward-fill from each symbol's first traded minute
    # (leading NaN = not yet in the universe denominator); flows are zero
    # outside a symbol's own traded span.
    close = pd.DataFrame({s: f["close"].reindex(grid).ffill() for s, f in frames.items()})
    vwap = pd.DataFrame({s: f["vwap"].reindex(grid).ffill() for s, f in frames.items()})
    tick = pd.DataFrame({s: f["last_tick"].reindex(grid).ffill() for s, f in frames.items()})
    up = pd.DataFrame({s: f["up_volume"].reindex(grid) for s, f in frames.items()}).fillna(0.0)
    down = pd.DataFrame({s: f["down_volume"].reindex(grid) for s, f in frames.items()}).fillna(0.0)

    n_symbols = close.notna().sum(axis=1)
    prior = pd.Series({s: float(prior_close[s]) for s in frames})
    up_total = up.sum(axis=1)
    down_total = down.sum(axis=1)

    out = pd.DataFrame(index=grid)
    out.index.name = "ts"
    out["asof_date"] = [t.date() for t in grid]
    out["available_at"] = grid
    out["n_symbols"] = n_symbols.astype("int64")
    out["breadth_above_vwap"] = (close > vwap).sum(axis=1) / n_symbols
    out["breadth_above_prior_close"] = close.gt(prior).sum(axis=1) / n_symbols
    out["up_volume"] = up_total.astype("int64")
    out["down_volume"] = down_total.astype("int64")
    up_arr = up_total.to_numpy(dtype=float)
    down_arr = down_total.to_numpy(dtype=float)
    out["updown_volume_ratio"] = np.divide(
        up_arr, down_arr, out=np.full_like(up_arr, np.nan), where=down_arr > 0.0
    )
    out["tick_proxy"] = tick.sum(axis=1).astype("int64")
    return out


def add_zscores(
    frame: pd.DataFrame,
    columns: Sequence[str] = INTERNALS_METRICS,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Append trailing z-score columns (``<col>_z``) to an internals frame.

    The z-score at row ``t`` uses the trailing ``window`` rows ENDING AT
    ``t`` (rolling mean and sample std, ``ddof=1``) — only information
    already available at that row, so the point-in-time contract of the
    input carries through unchanged. Rows without ``min_periods`` (default:
    the full window) non-NaN observations, and windows with zero variance,
    yield ``NaN`` rather than a fabricated or infinite score. Returns a new
    frame; the input is not mutated.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 for a sample std, got {window}")
    periods = window if min_periods is None else min_periods
    if periods < 2:
        raise ValueError(f"min_periods must be >= 2 for a sample std, got {periods}")
    out = frame.copy()
    for col in columns:
        series = frame[col].astype(float)
        mean = series.rolling(window, min_periods=periods).mean()
        std = series.rolling(window, min_periods=periods).std()
        z = (series - mean) / std
        z[std == 0.0] = np.nan
        out[f"{col}_z"] = z
    return out

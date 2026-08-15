"""State 4 — the 9-EMA exit matrix. First trigger to fire liquidates.

Three mechanical triggers, checked in this order each minute:

1. **Hard stop** — option value <= (1 + hard_stop_loss) x entry value.
   The trigger is measured on the option MID (fair value) and the fill lands
   at the BID. Measuring the trigger on the bid instead would fire it the
   instant the position opens: entry is at the ask by spec, and a 0-DTE ATM
   bid/ask can be 10-25% wide, so bid-vs-ask alone would look like an
   immediate -10-25% "loss". Mid-trigger / bid-fill is the non-degenerate
   reading; stop-fire frequency is reported.
2. **Trailing momentum stop** — when a 5-minute candle closes on the wrong
   side of the 9-EMA (below for longs, above for shorts), exit at the bid at
   the open of the next minute.
3. **Time stop** — liquidate at the bid at ``time_stop``. No overnight holds.

EMA seeding: at 09:35 exactly one candle exists (09:30–09:34:59), so it seeds
the EMA and the recursive spec formula runs from there. A useful identity
falls out of that formula: with ``ema_new = k*C + (1-k)*ema_prev``, the test
``C < ema_new`` is algebraically equivalent to ``C < ema_prev`` — so
"update then compare" and "compare then update" give identical signals, and
the ordering ambiguity in the spec is moot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass
class EmaTracker:
    """Recursive EMA over completed candles, seeded by the first candle."""

    period: int
    value: float | None = None
    n_updates: int = 0

    @property
    def multiplier(self) -> float:
        return 2.0 / (self.period + 1.0)

    def update(self, close: float) -> float:
        if self.value is None:
            self.value = close  # seed with the first completed candle
        else:
            k = self.multiplier
            self.value = close * k + self.value * (1.0 - k)
        self.n_updates += 1
        return self.value


@dataclass(frozen=True)
class KineticExitSignal:
    should_exit: bool
    reason: str = ""


@dataclass
class KineticExitState:
    """Per-position exit state, evaluated minute by minute."""

    direction: str  # "long" | "short"
    entry_price: float  # per-contract premium actually paid (at the ask)
    hard_stop_loss: float  # e.g. -0.50
    time_stop: time
    ema: EmaTracker
    pending_trailing_exit: bool = False  # candle closed against the EMA; exit next minute
    candles_seen: int = 0
    _last_candle_key: object = field(default=None, repr=False)

    def on_candle_close(self, candle_close: float) -> None:
        """Feed a COMPLETED N-minute candle; may arm the trailing exit."""
        ema_value = self.ema.update(candle_close)
        self.candles_seen += 1
        if self.candles_seen < 2:
            return  # the seed candle cannot trigger against itself
        if self.direction == "long" and candle_close < ema_value:
            self.pending_trailing_exit = True
        elif self.direction == "short" and candle_close > ema_value:
            self.pending_trailing_exit = True

    def evaluate_minute(self, now: time, option_mid: float | None) -> KineticExitSignal:
        """Exit decision for the current minute. First trigger wins."""
        if option_mid is not None and self.entry_price > 0:
            change = (option_mid - self.entry_price) / self.entry_price
            if change <= self.hard_stop_loss:
                return KineticExitSignal(True, "hard_stop")
        if self.pending_trailing_exit:
            return KineticExitSignal(True, "trailing_ema_stop")
        if now >= self.time_stop:
            return KineticExitSignal(True, "time_stop")
        return KineticExitSignal(False)

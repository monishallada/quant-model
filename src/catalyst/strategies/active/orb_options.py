"""ORB-OPTIONS — opening-range breakout expressed in long options.

The equity ORB was measured in the v17 wave-1 campaign and lost: -1.5%/mo
OOS, expectancy -0.08R, CI strictly negative. This is a DIFFERENT hypothesis,
not a retune of that one, because the payoff shape is different in a way that
matters:

    A long option truncates the loss at the premium and keeps the right tail
    open. If the breakout distribution is fat-tailed to the right — a minority
    of sessions running far past the range while the majority chop back — a
    convex wrapper can be positive where the linear wrapper is negative, even
    with the SAME entry signal and a worse hit rate.

That is the whole thesis, and it is falsifiable: it requires the breakout's
favorable excursion to clear the option's breakeven often enough to pay for
every premium that expires worthless. So the machine's central gate is not a
price threshold — it is an explicit breakeven test:

    breakeven_move = (premium / |delta|) / spot        (move needed to break even)
    required       = breakeven_move + friction_share
    trade only if   expected_favorable_move > required

where the expected favorable move is measured point-in-time from the
session's own realised volatility, never assumed.

Architecture (each stage is a gate that can refuse, and every refusal is
counted so the funnel is auditable):

    1. OPENING RANGE   first N minutes' high/low from VISIBLE bars only
    2. COILED-RANGE    range width / ATR must be TIGHT — a range that already
                       spans a day's move has nothing left to break out into
    3. BREAKOUT        close beyond the range by an ATR-scaled buffer, so the
                       threshold transfers across symbols and volatility eras
    4. VOLUME CONFIRM  relative volume vs the same minute-of-day across prior
                       sessions (point-in-time median, expanding)
    5. CONTRACT        nearest expiry, delta-targeted, ATM-anchored; hard
                       liquidity gates on relative spread and absolute premium
    6. BREAKEVEN       the gate above: refuse when the option cannot pay for
                       itself on the move the session's own vol supports
    7. EXITS           ATR-scaled underlying stop, premium stop, time stop,
                       mandatory flatten — declared, never discretionary

Every threshold below is a FIXED economic choice, recorded in the params and
swept only as a separate registered trial. Nothing here is fitted to a
backtest result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from catalyst.core.interfaces.intraday import IntradayContext, IntradayStrategy
from catalyst.core.types import (
    Direction,
    ExitRules,
    OptionKey,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
)
from catalyst.data.black_scholes import bs_greeks, implied_vol
from catalyst.strategies.registry import StrategyMeta, register

logger = logging.getLogger(__name__)

# Friction constants — identical to the audited stack (mosaic.py) so this
# strategy's internal economics match what the engine will actually charge.
_CROSS = 0.6                  # fraction of half-spread paid on entry
_SLIPPAGE_PCT = 0.02          # of premium, against the trader
_COMMISSION_PER_SHARE = 0.0065  # $0.65/contract/leg/side

_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


def _leg_entry_cost(mid: float, rel_spread: float) -> float:
    """What one long leg really costs to open, per share."""
    half = 0.5 * rel_spread * mid
    px = mid + _CROSS * half
    return px * (1.0 + _SLIPPAGE_PCT) + _COMMISSION_PER_SHARE


def _leg_exit_cost(mid: float, rel_spread: float) -> float:
    """What closing one long leg gives up, per share."""
    half = 0.5 * rel_spread * mid
    return _CROSS * half + mid * _SLIPPAGE_PCT + _COMMISSION_PER_SHARE


@dataclass(frozen=True)
class OrbParams:
    """Every gate, pre-registered. A different value is a different trial."""

    symbol: str = "QQQ"
    # -- 1/2: the opening range -----------------------------------------
    range_minutes: int = 15
    #: The range must be COILED: width <= this fraction of the 14-day ATR.
    #: A range already as wide as a normal day has spent the move.
    max_range_atr_frac: float = 0.55
    #: ...and not degenerate: a range narrower than this is a data artifact.
    min_range_atr_frac: float = 0.08
    # -- 3: breakout ------------------------------------------------------
    #: Break must exceed the range edge by this fraction of ATR (transfers
    #: across symbols; never a fixed cent amount).
    breakout_atr_buffer: float = 0.05
    entry_start: time = time(9, 45)
    entry_end: time = time(11, 30)     # ORB is a morning effect or it is nothing
    # -- 4: volume confirmation ------------------------------------------
    min_rel_volume: float = 1.30
    rel_volume_lookback_sessions: int = 20
    # -- 5: contract ------------------------------------------------------
    target_delta: float = 0.45
    max_dte: int = 1                   # 0-1 DTE: maximum gamma per premium
    max_rel_spread: float = 0.06
    min_premium: float = 0.20          # below this the spread dominates
    strike_step: float = 1.0
    n_strikes_scan: int = 8
    #: Minimum time-to-expiry (years) before greeks are trusted. Inside the
    #: last few minutes an option's delta is a step function and the IV solve
    #: is meaningless, so those contracts are refused rather than modelled.
    min_t_years: float = 1.0 / (252.0 * 390.0) * 15.0   # 15 trading minutes
    # -- 6: breakeven ------------------------------------------------------
    #: The convexity gate. A long option's value is E[max(move - breakeven,
    #: 0)], NOT median(move) vs breakeven: testing the median discards the
    #: right tail the trade exists to buy, and refuses whenever most
    #: breakouts fizzle — which is always true of breakouts. So the gate
    #: requires the EXPECTED payoff beyond breakeven, measured over the
    #: conditional excursion distribution, to be at least this fraction of
    #: the breakeven move itself. The margin covers the two things the
    #: expectation does not: the strategy exits on a stop or a clock rather
    #: than at the excursion's high, and closing costs a second half-spread.
    min_expected_edge_ratio: float = 0.25
    #: Minimum prior breakout observations before the excursion statistic is
    #: usable. An unformed statistic never trades.
    min_excursion_obs: int = 10
    # -- 7: exits ----------------------------------------------------------
    stop_loss_pct: float = -0.50       # premium stop
    max_hold_minutes: int = 120
    close_by_time: time = time(15, 45)
    per_trade_risk_fraction: float = 0.02
    warmup_sessions: int = 20


class OrbOptionsStrategy(IntradayStrategy):
    """Long calls on upside breaks, long puts on downside breaks."""

    name = "orb_options"

    def __init__(self, params: OrbParams | None = None, warmup_bars_provider=None) -> None:
        self._p = params or OrbParams()
        self._bars_provider = warmup_bars_provider
        self._reset_state()

    # -- state ------------------------------------------------------------

    def _reset_state(self) -> None:
        self._session: date | None = None
        self._range_hi: float | None = None
        self._range_lo: float | None = None
        self._atr: float | None = None
        self._fired: set[str] = set()
        #: minute-of-day -> list of prior-session volumes (point-in-time)
        self._tod_volume: dict[int, list[float]] = {}
        self._sessions_seen: int = 0
        # NOTE: _gates is deliberately NOT cleared here — the pipeline deltas
        # it across six segment runs, so resetting mid-run would corrupt it.
        self._daily: list[tuple[float, float, float]] = []   # (high, low, close)
        #: Favorable excursions (fraction of spot) measured on PRIOR sessions'
        #: breakouts — the conditional distribution the breakeven gate tests.
        self._excursions: list[float] = []
        #: The most complete view of the CURRENT session seen so far. The
        #: engine's ctx.bars holds ONLY the session in progress, so a session
        #: must be accumulated while it runs and consumed when it rolls —
        #: reading ctx.bars at roll time yields the NEW session's first
        #: minutes, which silently corrupts ATR, the volume baseline and the
        #: excursion sample (this defect produced a zero-trade first run).
        self._session_frame: pd.DataFrame | None = None
        if not hasattr(self, "_gates"):
            self._gates: dict[str, int] = {}

    def _gate(self, name: str) -> None:
        self._gates[name] = self._gates.get(name, 0) + 1

    @property
    def gates(self) -> dict[str, int]:
        """Funnel counters the pipeline persists per segment. Every refusal
        is counted, so a run that trades nothing says WHY it traded nothing —
        the instrumentation gap the v17 diagnosis called out."""
        return dict(self._gates)

    def session_universe(self, session: date) -> list[str]:
        return [self._p.symbol]

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _rth(df: pd.DataFrame) -> pd.DataFrame:
        """RTH-only view. Engine frames span 04:00-20:00; premarket bars
        would corrupt the opening range and every volume statistic (this
        exact defect cost the v16 campaign a full pilot run)."""
        if df is None or df.empty:
            return df
        t = df.index.time if isinstance(df.index, pd.DatetimeIndex) else None
        if t is None:
            return df
        mask = (t >= _RTH_OPEN) & (t < _RTH_CLOSE)
        return df.loc[mask]

    def _roll_session(self, ctx: IntradayContext) -> None:
        """New session: close the previous one into the point-in-time stats."""
        if self._session == ctx.session:
            return
        if self._session is not None and self._session > ctx.session:
            self._reset_state()          # backwards move: a new run, not a gap
        prev = self._session_frame
        if self._session is not None and prev is not None:
            day = self._rth(prev)
            if not day.empty:
                self._daily.append((float(day["high"].max()),
                                    float(day["low"].min()),
                                    float(day["close"].iloc[-1])))
                for ts, vol in zip(day.index, day["volume"]):
                    self._tod_volume.setdefault(ts.hour * 60 + ts.minute,
                                                []).append(float(vol))
                self._record_excursion(day)
                self._sessions_seen += 1
        self._session = ctx.session
        self._session_frame = None
        self._range_hi = self._range_lo = self._atr = None
        self._fired.clear()

    def _record_excursion(self, day: pd.DataFrame) -> None:
        """Measure what a breakout on this COMPLETED session would have paid.

        This is the conditional distribution the breakeven gate needs: not
        "how far does this symbol move on an average day" (which an option is
        priced to match, making that comparison tautological) but "how far
        does it run AFTER a confirmed opening-range break". Computed only
        from a session that is already over, so it is point-in-time for every
        later decision.
        """
        p = self._p
        if len(day) < p.range_minutes + 5:
            return
        atr = self._true_range_atr()
        if atr is None or atr <= 0:
            return
        opening = day.iloc[:p.range_minutes]
        hi, lo = float(opening["high"].max()), float(opening["low"].min())
        width = hi - lo
        if not (p.min_range_atr_frac <= width / atr <= p.max_range_atr_frac):
            return                       # not a session this strategy would trade
        buffer = p.breakout_atr_buffer * atr
        rest = day.iloc[p.range_minutes:]
        for i, (ts, row) in enumerate(rest.iterrows()):
            if ts.time() < p.entry_start or ts.time() > p.entry_end:
                continue
            close = float(row["close"])
            if close > hi + buffer:
                window = rest.iloc[i:i + p.max_hold_minutes]
                excursion = (float(window["high"].max()) - close) / close
            elif close < lo - buffer:
                window = rest.iloc[i:i + p.max_hold_minutes]
                excursion = (close - float(window["low"].min())) / close
            else:
                continue
            self._excursions.append(max(excursion, 0.0))
            return                       # one observation per session

    def _true_range_atr(self, period: int = 14) -> float | None:
        """ATR from COMPLETED prior sessions only."""
        if len(self._daily) < period + 1:
            return None
        trs = []
        for i in range(len(self._daily) - period, len(self._daily)):
            hi, lo, _ = self._daily[i]
            prev_close = self._daily[i - 1][2]
            trs.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
        return float(np.mean(trs)) if trs else None

    def _rel_volume(self, today: pd.DataFrame, now: datetime) -> float | None:
        """Cumulative volume so far vs the prior-session median at this minute."""
        if today.empty:
            return None
        mod_now = now.hour * 60 + now.minute
        hist = []
        for mod, vols in self._tod_volume.items():
            if mod <= mod_now and vols:
                tail = vols[-self._p.rel_volume_lookback_sessions:]
                hist.append(np.median(tail))
        if not hist:
            return None
        baseline = float(np.sum(hist))
        if baseline <= 0:
            return None
        return float(today["volume"].sum()) / baseline

    # -- the decision -----------------------------------------------------

    def on_minute(self, ctx: IntradayContext) -> list[ProposedTrade]:
        p = self._p
        self._roll_session(ctx)
        # Accumulate the session BEFORE any gate can return: every early
        # return still has to leave the session record complete, or the
        # warmup counter never advances and the strategy deadlocks at zero
        # trades forever (it did — this is that bug).
        raw_all = ctx.bars.get(p.symbol)
        if raw_all is not None and not raw_all.empty:
            same_day = self._rth(raw_all)
            if not same_day.empty:
                same_day = same_day[same_day.index.date == ctx.session]
            if not same_day.empty:
                self._session_frame = same_day
        if self._sessions_seen < p.warmup_sessions:
            self._gate("warmup")
            return []
        now = ctx.now
        if not (p.entry_start <= now.time() <= p.entry_end):
            self._gate("outside_window")
            return []
        if p.symbol in self._fired:
            self._gate("already_fired")     # one breakout per session per side
            return []

        today = self._session_frame
        if today is None or today.empty:
            self._gate("no_bars")
            return []

        # 1. opening range from VISIBLE bars of the first N minutes
        range_end = datetime.combine(ctx.session, _RTH_OPEN) + timedelta(minutes=p.range_minutes)
        opening = today[today.index < range_end]
        if len(opening) < max(p.range_minutes // 2, 3):
            self._gate("range_not_formed")
            return []
        self._range_hi = float(opening["high"].max())
        self._range_lo = float(opening["low"].min())

        atr = self._atr if self._atr is not None else self._true_range_atr()
        self._atr = atr
        if atr is None or atr <= 0:
            self._gate("no_atr")
            return []

        # 2. the range must be COILED, and not degenerate
        width = self._range_hi - self._range_lo
        frac = width / atr
        if frac > p.max_range_atr_frac:
            self._gate("range_too_wide")
            return []
        if frac < p.min_range_atr_frac:
            self._gate("range_degenerate")
            return []

        # 3. breakout with an ATR-scaled buffer
        spot = float(today["close"].iloc[-1])
        buffer = p.breakout_atr_buffer * atr
        if spot > self._range_hi + buffer:
            direction, right = Direction.LONG, OptionRight.CALL
        elif spot < self._range_lo - buffer:
            direction, right = Direction.SHORT, OptionRight.PUT
        else:
            self._gate("no_breakout")
            return []

        # 4. volume confirmation
        rel_vol = self._rel_volume(today, now)
        if rel_vol is None or rel_vol < p.min_rel_volume:
            self._gate("volume_unconfirmed")
            return []

        # 5-6. contract selection + the breakeven gate
        trade = self._select_contract(ctx, spot, right, direction, atr, rel_vol)
        if trade is None:
            return []
        self._fired.add(p.symbol)
        return [trade]

    def _select_contract(self, ctx: IntradayContext, spot: float,
                         right: OptionRight, direction: Direction,
                         atr: float, rel_vol: float) -> ProposedTrade | None:
        """Delta-targeted ATM-anchored long option that clears breakeven."""
        p = self._p
        if ctx.option_quote is None:
            self._gate("no_quotes")
            return None
        expiry = self._nearest_expiry(ctx.session)
        t_years = self._t_years(ctx.session, ctx.now, expiry)
        if t_years < p.min_t_years:
            self._gate("expiry_too_close")
            return None
        step = p.strike_step
        atm = round(spot / step) * step
        best = None
        for i in range(-p.n_strikes_scan, p.n_strikes_scan + 1):
            strike = atm + i * step
            if strike <= 0:
                continue
            key = OptionKey(underlying=p.symbol, expiry=expiry,
                            right=right, strike=strike)
            q = ctx.option_quote(key)
            if q is None:
                continue
            bid, ask = q
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            mid = 0.5 * (bid + ask)
            if mid < p.min_premium:
                continue
            rel = (ask - bid) / mid
            if rel > p.max_rel_spread:
                continue
            greeks = self._delta_of(mid, spot, strike, right, t_years)
            if greeks is None:
                continue
            delta, iv = greeks
            score = abs(abs(delta) - p.target_delta)
            if best is None or score < best[0]:
                best = (score, key, mid, rel, delta, iv)
        if best is None:
            self._gate("no_contract")
            return None
        _score, key, mid, rel, delta, iv = best

        entry_px = _leg_entry_cost(mid, rel)
        exit_px = _leg_exit_cost(mid, rel)
        # Real deltas are SIGNED: a put's is negative. The magnitude is what
        # the breakeven divides by (the moneyness proxy this replaced was
        # unsigned, so an unguarded comparison silently refused every put).
        if abs(delta) <= 0.01:
            self._gate("delta_degenerate")
            return None

        # 6. THE BREAKEVEN GATE — the trade must be able to pay for itself.
        # Theta-aware: on a 0DTE contract the premium bleeds while the move
        # develops, so the move required is not (premium/delta) at entry but
        # (premium + decay over the hold)/delta. Ignoring that understates
        # what later entries need — badly, since theta accelerates into the
        # close, which is exactly when this strategy holds.
        g = bs_greeks(spot, key.strike, t_years, iv, right)
        hold_years = min(p.max_hold_minutes, 390) / (390.0 * 252.0)
        decay = abs(float(g.theta or 0.0)) * hold_years
        breakeven_move = ((entry_px + decay) / abs(delta)) / spot
        friction_share = (entry_px - mid + exit_px) / max(mid, 1e-9)
        # THE GATE: the option must be payable by the MEASURED post-breakout
        # excursion, not by the unconditional daily move. (An ATM option is
        # priced so that premium/delta ~ the unconditional expected move, so
        # comparing against that is comparing fair value to fair value and
        # refuses forever no matter how good the signal is. The conditional
        # distribution is the only thing that can beat it — and if it cannot,
        # the honest answer is that this trade should not exist.)
        if len(self._excursions) < p.min_excursion_obs:
            self._gate("excursion_unformed")
            return None
        exc = np.asarray(self._excursions, dtype=float)
        # Convex expectation over the conditional distribution: what a long
        # option is actually worth after this signal.
        expected_payoff = float(np.maximum(exc - breakeven_move, 0.0).mean())
        edge_ratio = expected_payoff / max(breakeven_move, 1e-12)
        clear_rate = float((exc > breakeven_move).mean())
        expected_move = float(np.median(exc))      # reported, never the test
        if edge_ratio <= p.min_expected_edge_ratio:
            self._gate("breakeven_unmet")
            return None

        leg = OrderLeg(key=key, side=Side.BUY, qty=1)
        exits = ExitRules(
            stop_loss_pct=p.stop_loss_pct,
            max_hold_minutes=p.max_hold_minutes,
            close_by_time=p.close_by_time,
            use_stops=True,
        )
        return ProposedTrade(
            engine=self.name,
            catalyst_ref=f"orb:{ctx.session}:{p.symbol}:{right.value}",
            legs=[leg],
            unit_cost=entry_px,
            unit_max_loss=entry_px,          # long option: loss capped at premium
            direction=direction,
            exit_rules=exits,
            per_trade_risk_fraction=p.per_trade_risk_fraction,
            rationale={
                "spot": spot, "atr": atr, "rel_volume": rel_vol,
                "range_hi": self._range_hi, "range_lo": self._range_lo,
                "strike": key.strike, "delta": delta, "iv": iv,
                "theta_decay": decay, "t_years": t_years, "mid": mid,
                "rel_spread": rel, "breakeven_move": breakeven_move,
                "expected_move": expected_move, "friction_share": friction_share,
                "excursion_obs": len(self._excursions),
                "edge_ratio": edge_ratio, "clear_rate": clear_rate,
                "expected_payoff": expected_payoff,
            },
        )

    @staticmethod
    def _t_years(session: date, now: datetime, expiry: date) -> float:
        """Time to expiry in years, counted in TRADING minutes to the close.

        A 0DTE option at 10:00 has ~6 hours of life, not "zero days" and not
        "one day" — both of those mis-price it badly. Trading-time is what
        the option decays on, so that is what the greeks get.
        """
        if expiry < session:
            return 0.0
        minutes_left = max(
            (datetime.combine(session, _RTH_CLOSE) - now).total_seconds() / 60.0, 0.0)
        whole_days = max((expiry - session).days, 0)
        return (minutes_left + whole_days * 390.0) / (390.0 * 252.0)

    def _delta_of(self, mid: float, spot: float, strike: float,
                  right: OptionRight, t_years: float) -> tuple[float, float] | None:
        """TRUE Black-Scholes delta from the contract's OWN implied vol.

        The breakeven gate divides by delta, so a crude delta corrupts the
        strategy's central economic test directly. The previous moneyness
        proxy was monotone but had no relationship to the real curvature —
        it systematically overstated delta on wings and understated it near
        the money, which made breakevens look easier to clear than they are.
        Returns (delta, iv), or None when the IV solve fails (a contract we
        cannot price is a contract we do not trade).
        """
        if t_years < self._p.min_t_years or mid <= 0.0:
            return None
        iv = implied_vol(mid, spot, strike, t_years, right)
        if iv is None or not (0.0 < iv < 5.0):
            return None
        g = bs_greeks(spot, strike, t_years, iv, right)
        if g.delta is None or abs(g.delta) < 1e-6:
            return None
        return float(g.delta), float(iv)

    @staticmethod
    def _nearest_expiry(session: date) -> date:
        """0DTE when the session itself is an expiry; else the next weekday."""
        return session


def build(cfg) -> OrbOptionsStrategy:
    """Registry entry point; config maps onto OrbParams so --set reaches every gate."""
    o = getattr(cfg, "orb_options", None)
    params = OrbParams() if o is None else OrbParams(
        symbol=o.symbol,
        range_minutes=o.range_minutes,
        max_range_atr_frac=o.max_range_atr_frac,
        breakout_atr_buffer=o.breakout_atr_buffer,
        min_rel_volume=o.min_rel_volume,
        target_delta=o.target_delta,
        max_rel_spread=o.max_rel_spread,
        min_premium=o.min_premium,
        min_expected_edge_ratio=o.min_expected_edge_ratio,
        stop_loss_pct=o.stop_loss_pct,
        max_hold_minutes=o.max_hold_minutes,
        per_trade_risk_fraction=o.per_trade_risk_fraction,
    )
    return OrbOptionsStrategy(params=params)


register(
    StrategyMeta(
        name="orb_options",
        module="catalyst.strategies.active.orb_options",
        status="active",
        notes=("v17. Opening-range breakout expressed in LONG options: the "
               "convex-wrapper hypothesis on the same entry signal whose "
               "linear form was rejected in wave 1. Central gate is an "
               "explicit breakeven test (premium/delta vs the move the "
               "session's own vol supports). Research/simulation only."),
    ),
    build,
)

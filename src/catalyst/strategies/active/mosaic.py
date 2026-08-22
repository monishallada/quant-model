"""v16 MOSAIC — the unified intraday options algorithm.

ONE master decision engine. Every minute it can act, it:

1. reads the visible market (minute bars; the watched option window's NBBO),
2. fits the local smile (quadratic-in-log-moneyness; refuses bad data),
3. forecasts remaining-horizon realized vol (bipower + diurnal curve),
4. looks up the conditional return distribution for the CURRENT state
   (time-of-day × vol-state × trend-state; fitted point-in-time),
5. generates candidates from three measured effect families:
     A. VOL PREMIUM — fitted ATM IV vs forecast RV, conditioned on state
        → defined-risk credit vertical (short the rich vol)
        → or debit vertical / long option when IV is CHEAP vs forecast
     B. SMILE DISLOCATION — one contract rich/cheap vs its own fitted smile
        → relative-value vertical (buy cheap leg, sell rich/fair neighbor)
     C. DISTRIBUTION ASYMMETRY — conditional quantiles skewed vs the
        smile-implied symmetric world → directional debit vertical in the
        favored tail,
6. transforms the conditional distribution through every candidate's payoff
   (scenario grid × sticky-strike/delta band), charges modeled exit friction,
7. ranks by EV / max-loss and emits the best candidate IFF its worst-case
   model-risk-banded EV clears the pre-registered threshold. Otherwise:
   NO TRADE.

Point-in-time discipline (enforced in code, not convention):
- All fitted state (diurnal curve, conditional quantile tables, vol baseline)
  is built ONLY from sessions strictly before the current one. Warmup history
  comes from the injected bars provider at run start (data before run start);
  in-run sessions are appended AFTER they complete.
- A new pipeline segment (session moving backwards) RESETS all fitted state,
  so train/test segments cannot leak into each other through the instance.

Walls: no broker, no risk, no cost model, no position book — the strategy
emits intent; the audited engine prices, sizes, and exits. Exits are
mechanical ExitRules (optimized hold + stop + hard flatten); continuous
EV-based rotation is approximated by short optimized holds, and that
approximation is named in the report rather than hidden.
"""

from __future__ import annotations

import logging
import math
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
from catalyst.data.black_scholes import implied_vol
from catalyst.strategies.registry import StrategyMeta, register
from catalyst.vol import dist, payoff, rv, surface

logger = logging.getLogger(__name__)

MINUTES_PER_YEAR = 252.0 * 390.0

# audited execution-cost constants (must mirror config/backtest.yaml's fill
# model; the pilot proved that omitting slippage+commissions here lets the EV
# gate approve trades the real cost model then rejects by -$59/trade)
_CROSS = 0.6                 # fraction of half-spread paid beyond mid
_SLIPPAGE_PCT = 0.02         # of premium, adverse, per side
_COMMISSION_PER_SHARE = 0.0065   # $0.65/contract/leg/side / 100 shares


def _leg_entry_cost(mid: float, rel: float, side_is_buy: bool) -> float:
    """Signed per-share premium at the FULL modeled friction (crossing +
    slippage + commission): what the audited broker will actually charge."""
    half = mid * rel / 2.0
    if side_is_buy:
        return (mid + _CROSS * half) * (1 + _SLIPPAGE_PCT) + _COMMISSION_PER_SHARE
    return -((mid - _CROSS * half) * (1 - _SLIPPAGE_PCT) - _COMMISSION_PER_SHARE)


def _leg_exit_cost(mid: float, rel: float) -> float:
    """Per-share friction to CLOSE one leg: crossing + slippage + commission."""
    half = mid * rel / 2.0
    return _CROSS * half + _SLIPPAGE_PCT * mid + _COMMISSION_PER_SHARE


_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


def _rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Regular-session bars only. The engine's frames span 04:00-20:00
    (premarket is a feature for other strategies); every MOSAIC estimator is
    a REGULAR-SESSION quantity — premarket bars polluted the RV/diurnal math
    and made minute_of_session exceed 390 by early afternoon, silently
    NaN-ing every later forecast (caught in the integration smoke)."""
    if bars.empty:
        return bars
    t = bars.index.time
    return bars[(t >= _RTH_OPEN) & (t < _RTH_CLOSE)]




@dataclass(frozen=True)
class MosaicParams:
    """Pre-registered defaults. Every number here is swept in robustness
    testing; none was chosen by peeking at test-segment results."""

    symbol: str = "SPY"
    strike_step: float = 1.0
    n_strikes_wing: int = 7              # ATM ± this many steps watched
    decision_every_min: int = 5
    entry_start: time = time(9, 45)      # skip opening rotation
    entry_end: time = time(15, 15)       # nothing new inside the last 30min
    horizon_minutes: int = 60            # candidate evaluation horizon
    # --- candidate gates (pre-registered) ---
    vol_premium_min: float = 0.04        # |ATM IV − fRV| to arm family A
    dislocation_z_min: float = 3.0       # |z| to arm family B
    asym_min: float = 0.25               # |q05+q95|/(q95−q05) to arm family C
    ev_min_frac_of_maxloss: float = 0.05 # EV must exceed 5% of max loss...
    ev_min_dollars: float = 0.02         # ...and $2/contract-unit (per share)
    max_rel_spread: float = 0.12         # per-leg liquidity gate
    smile_rmse_max: float = 0.02         # refuse decisions on bad fits
    # --- fitting / warmup ---
    warmup_sessions: int = 60
    refit_every_sessions: int = 20
    trailing_vol_sessions: int = 20
    # --- exits (interpreted by the audited engine) ---
    max_hold_minutes: int = 90
    stop_loss_pct: float = -0.45         # debit structures: % of premium
    credit_stop_multiple: float = 1.8    # credit structures
    per_trade_risk_fraction: float = 0.01
    use_stops: bool = True
    vertical_width_strikes: int = 2      # outer leg this many strikes out
    # --- family D: fair-value convergence (Muravyev-Pearson staleness) ---
    fv_g_min: float = 1.0                # mid displaced >= this many half-spreads
    fv_neighbor_g_max: float = 0.3       # hedge leg must be fairly priced
    fv_capture: float = 0.5              # conservative fraction of gap captured


class MosaicStrategy(IntradayStrategy):
    """The master algorithm. One instance persists across sessions; fitted
    state is strictly point-in-time (see module docstring)."""

    name = "mosaic"

    def __init__(self, params: MosaicParams | None = None,
                 warmup_bars_provider=None) -> None:
        self._p = params or MosaicParams()
        self._warmup_provider = warmup_bars_provider
        self._reset_state()
        #: decision diagnostics, exposed like short_vrp's gate counters
        self.gates = {"decisions": 0, "no_smile": 0, "no_dist": 0, "no_rv": 0,
                      "no_candidates": 0, "ev_rejected": 0, "emitted": 0,
                      "warmup": 0}

    # ------------------------------------------------------------------
    # point-in-time state
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self._last_session: date | None = None
        self._session_returns: list[pd.Series] = []     # completed sessions
        self._session_vars: list[float] = []            # per-session variance
        self._curve: np.ndarray = np.ones(26)
        self._quantiles: dist.ConditionalQuantiles | None = None
        self._sessions_seen = 0
        self._prev_session_bars: pd.DataFrame | None = None
        #: research-spec persistence filter: family B arms only when the SAME
        #: contract shows the SAME-sign dislocation on consecutive decisions
        #: (kills one-minute NBBO flicker, the dominant quote-only false
        #: positive)
        self._z_memory: dict = {}
        #: per-contract trailing IV samples (minute_of_session, iv) for the
        #: fair-value engine (Muravyev-Pearson: IV-bar over ~30 minutes)
        self._iv_hist: dict = {}
        self._warmed = False

    def _maybe_new_run(self, session: date) -> None:
        # STRICTLY backwards only: equal session = the normal intra-session
        # decision cadence (the <= form reset all fitted state on every
        # decision after the first — caught by the persistence unit test)
        if self._last_session is not None and session < self._last_session:
            logger.info("mosaic: session moved backwards (%s <= %s) — new "
                        "pipeline segment, resetting fitted state",
                        session, self._last_session)
            self._reset_state()

    def _warmup(self, session: date) -> None:
        """Load pre-run history (strictly before ``session``) once per run."""
        self._warmed = True
        if self._warmup_provider is None:
            return
        from catalyst.core.tradingcal import previous_trading_day
        day = session
        loaded = 0
        try:
            for _ in range(self._p.warmup_sessions):
                day = previous_trading_day(day)
                bars = self._warmup_provider.get_day(self._p.symbol, day)
                if bars is None or bars.empty:
                    continue
                bars = _rth(bars)
                r = rv.minute_log_returns(bars["close"])
                if len(r) < 100:
                    continue
                self._session_returns.append(r)
                self._session_vars.append(rv.bipower_variation(r))
                loaded += 1
        except Exception as e:                          # noqa: BLE001
            logger.warning("mosaic warmup stopped early (%s sessions): %s",
                           loaded, e)
        self._session_returns.reverse()
        self._session_vars.reverse()
        self._sessions_seen = loaded
        if loaded >= 20:
            self._refit()
        logger.info("mosaic warmup: %d prior sessions loaded", loaded)

    def _close_session(self, bars: pd.DataFrame) -> None:
        bars = _rth(bars)
        r = rv.minute_log_returns(bars["close"])
        if len(r) >= 100:
            self._session_returns.append(r)
            self._session_vars.append(rv.bipower_variation(r))
        self._sessions_seen += 1
        if (self._sessions_seen % self._p.refit_every_sessions == 0
                and len(self._session_returns) >= 20):
            self._refit()

    def _refit(self) -> None:
        """Refit diurnal curve + conditional quantile tables on ALL completed
        (strictly historical) sessions."""
        self._curve = rv.fit_diurnal_curve(self._session_returns)
        spec = dist.StateSpec()
        h = self._p.horizon_minutes
        scaled, tods, vols, trends = [], [], [], []
        recent_vars = self._session_vars
        for si, r in enumerate(self._session_returns):
            arr = r.to_numpy(dtype=float)
            n = len(arr)
            if n < h + 40:
                continue
            base_var = (float(np.median(recent_vars[max(0, si - 20):si]))
                        if si >= 5 else float("nan"))
            sess_vol = rv.annualized_vol(recent_vars[si], n)
            med_vol = (rv.annualized_vol(base_var, 390)
                       if base_var == base_var else float("nan"))
            vstate = dist.classify_vol(sess_vol, med_vol, spec)
            csum = np.concatenate([[0.0], np.cumsum(arr)])
            step = 3
            for m in range(30, n - h, step):
                fwd = csum[m + h] - csum[m]
                scale = math.sqrt(max(recent_vars[si], 1e-12) / n * h)
                trailing = csum[m] - csum[max(m - 30, 0)]
                tstate = dist.classify_trend(trailing, scale * math.sqrt(30.0 / h),
                                             spec)
                scaled.append(fwd / scale)
                tods.append(dist.tod_bucket(m, spec, n))
                vols.append(vstate)
                trends.append(tstate)
        # 60 RTH warmup sessions yield ~5-6k obs at step=3; the tod
        # marginals (min_obs 300 each) are solid from the start and the full
        # cells densify as the run accumulates sessions
        if len(scaled) >= 2000:
            self._quantiles = dist.fit(np.array(scaled), np.array(tods),
                                       np.array(vols), np.array(trends), spec)
            logger.info("mosaic refit: %d obs, %d cells",
                        self._quantiles.n_obs, len(self._quantiles.table))

    # ------------------------------------------------------------------
    # engine contract
    # ------------------------------------------------------------------
    def session_universe(self, session: date) -> list[str]:
        return [self._p.symbol]

    def on_minute(self, ctx: IntradayContext) -> list[ProposedTrade]:
        p = self._p
        self._maybe_new_run(ctx.session)
        if not self._warmed:
            self._warmup(ctx.session)
        if self._last_session is not None and ctx.session > self._last_session:
            prior = ctx.bars.get(p.symbol)
            # close out the PREVIOUS session using bars visible now (they are
            # all strictly historical once the session turned over)
            if self._prev_session_bars is not None:
                self._close_session(self._prev_session_bars)
        self._last_session = ctx.session

        bars = ctx.bars.get(p.symbol)
        if bars is None or bars.empty:
            return []
        self._prev_session_bars = bars
        bars = _rth(bars)
        if bars.empty:
            return []

        now_t = ctx.now.time()
        if not (p.entry_start <= now_t <= p.entry_end):
            return []
        if ctx.now.minute % p.decision_every_min != 0:
            return []
        if ctx.option_quote is None:
            return []
        self.gates["decisions"] += 1

        if self._quantiles is None:
            self.gates["warmup"] += 1
            return []

        spot = ctx.spot(p.symbol)
        if spot is None or spot <= 0:
            return []

        # ---- market state -------------------------------------------------
        minute_of_session = len(bars)
        today_r = rv.minute_log_returns(bars["close"])
        prior_var = (float(np.median(self._session_vars[-p.trailing_vol_sessions:]))
                     if self._session_vars else float("nan"))
        f_vol = rv.forecast_horizon_vol(today_r, prior_var, self._curve,
                                        minute_of_session, p.horizon_minutes)
        if not f_vol == f_vol:
            self.gates["no_rv"] += 1
            return []

        # ---- smile --------------------------------------------------------
        expiry = self._session_expiry(ctx)
        if expiry is None:
            return []
        t_years = self._t_years(ctx.now, expiry)
        window = self._watch_window(ctx.session, spot, expiry)
        quotes = self._collect_quotes(ctx, window, spot, t_years)
        if quotes is None:
            self.gates["no_smile"] += 1
            return []
        keys, kmny, mids, ivs, rels = quotes
        fit = surface.fit_smile(kmny, ivs, rels)
        if fit is None or fit.rmse > p.smile_rmse_max:
            self.gates["no_smile"] += 1
            return []
        zscores = surface.dislocation_scores(fit, kmny, ivs)

        # ---- conditional distribution ------------------------------------
        spec = self._quantiles.spec
        sess_vol_so_far = rv.annualized_vol(rv.bipower_variation(today_r),
                                            max(len(today_r), 1))
        med_vol = (rv.annualized_vol(prior_var, 390)
                   if prior_var == prior_var else float("nan"))
        vstate = dist.classify_vol(sess_vol_so_far, med_vol, spec)
        h_scale = f_vol * math.sqrt(p.horizon_minutes / MINUTES_PER_YEAR)
        trailing_ret = (float(np.log(bars["close"].iloc[-1]
                                     / bars["close"].iloc[-30]))
                        if len(bars) > 30 else 0.0)
        tstate = dist.classify_trend(trailing_ret,
                                     h_scale * math.sqrt(30.0 / p.horizon_minutes),
                                     spec)
        tod = dist.tod_bucket(minute_of_session, spec)
        q7 = self._quantiles.lookup(tod, vstate, tstate)
        if q7 is None:
            self.gates["no_dist"] += 1
            return []

        # ---- candidates ---------------------------------------------------
        candidates = []
        candidates += self._family_vol_premium(fit, f_vol, keys, kmny, mids,
                                               rels, spot, t_years)
        candidates += self._family_dislocation(zscores, keys, kmny, mids, ivs,
                                               rels, fit, spot, t_years)
        candidates += self._family_asymmetry(q7, fit, keys, kmny, mids, ivs,
                                             rels, spot, t_years)
        candidates += self._family_fair_value(ctx, keys, kmny, mids, ivs,
                                              rels, spot, t_years)
        if not candidates:
            self.gates["no_candidates"] += 1
            return []

        # ---- payoff transform + EV gate ----------------------------------
        best, best_score = None, -1e18
        for cand in candidates:
            stats_ss = payoff.evaluate(cand["legs_spec"], spot, q7, h_scale,
                                       p.horizon_minutes, cand["exit_cost"],
                                       smile_iv_at=fit.iv_at,
                                       dynamics="sticky_strike")
            stats_sd = payoff.evaluate(cand["legs_spec"], spot, q7, h_scale,
                                       p.horizon_minutes, cand["exit_cost"],
                                       smile_iv_at=fit.iv_at,
                                       dynamics="sticky_delta")
            if stats_ss is None or stats_sd is None:
                continue
            # model-risk band: the edge must survive the WORSE of the two
            # surface dynamics — the gap between them is not free alpha.
            # Family D adds its convergence gap: expected drift of the stale
            # mid toward fair value, which the distribution pricing cannot see.
            ev = min(stats_ss.ev, stats_sd.ev) + cand.get("ev_bonus", 0.0)
            max_loss = cand["unit_max_loss"]
            if max_loss <= 0:
                continue
            if ev < p.ev_min_dollars or ev < p.ev_min_frac_of_maxloss * max_loss:
                continue
            score = ev / max_loss
            if score > best_score:
                best, best_score = (cand, min(stats_ss.p_profit,
                                              stats_sd.p_profit)), score
        if best is None:
            self.gates["ev_rejected"] += 1
            return []

        cand, p_profit = best
        self.gates["emitted"] += 1
        self._sidecar(ctx, cand, best_score, p_profit, fit, f_vol)
        is_credit = cand["unit_cost"] < 0
        return [ProposedTrade(
            engine=self.name,
            catalyst_ref=f"{p.symbol}:{ctx.session}:{ctx.now:%H%M}",
            legs=cand["legs"],
            unit_cost=cand["unit_cost"],
            unit_max_loss=cand["unit_max_loss"],
            direction=cand["direction"],
            exit_rules=ExitRules(
                close_by_time=time(15, 45),
                max_hold_minutes=p.max_hold_minutes,
                use_stops=p.use_stops,
                stop_loss_pct=(-p.credit_stop_multiple if is_credit
                               else p.stop_loss_pct),
                close_before_expiry_days=0,
            ),
            per_trade_risk_fraction=p.per_trade_risk_fraction,
            rationale={"family": cand["family"], "ev_score": round(best_score, 4),
                       "p_profit": round(p_profit, 3),
                       "atm_iv": round(fit.atm_iv, 4), "f_rv": round(f_vol, 4),
                       "state": f"{tod}/{vstate}/{tstate}",
                       "ref": f"{p.symbol}:{ctx.session}:{ctx.now:%H%M}"},
        )]

    def _sidecar(self, ctx, cand, ev_score, p_profit, fit, f_vol) -> None:
        """Per-emission decision log (observational; failures never trade).
        Joined to the trade ledger by catalyst_ref for EV-calibration — the
        check that predicted EV and realized P&L actually correlate, which is
        the single most important honesty test of an EV machine."""
        try:
            from pathlib import Path
            path = Path("results/active/mosaic/decisions.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a") as f:
                if new:
                    f.write("catalyst_ref,family,ev_score,ev_dollars,p_profit,"
                            "unit_cost,unit_max_loss,atm_iv,f_rv,skew,rmse\n")
                f.write(f"{self._p.symbol}:{ctx.session}:{ctx.now:%H%M},"
                        f"{cand['family']},{ev_score:.5f},"
                        f"{ev_score * cand['unit_max_loss']:.5f},{p_profit:.4f},"
                        f"{cand['unit_cost']:.4f},{cand['unit_max_loss']:.4f},"
                        f"{fit.atm_iv:.4f},{f_vol:.4f},{fit.skew:.4f},"
                        f"{fit.rmse:.5f}\n")
        except Exception:                               # noqa: BLE001
            logger.debug("mosaic sidecar write failed", exc_info=True)

    # ------------------------------------------------------------------
    # candidate families
    # ------------------------------------------------------------------
    def _family_vol_premium(self, fit, f_vol, keys, kmny, mids, rels,
                            spot, t_years):
        p = self._p
        gap = fit.atm_iv - f_vol
        if abs(gap) < p.vol_premium_min:
            return []
        # rich IV → short an ATM-adjacent credit vertical on the side the
        # smile is richest; cheap IV → long the ATM debit vertical
        out = []
        side = OptionRight.PUT if fit.skew < 0 else OptionRight.CALL
        atm_i = int(np.nanargmin(np.abs(kmny)))
        step = 1 if side is OptionRight.CALL else -1
        legs = self._vertical(keys, mids, rels, atm_i, atm_i + 2 * step,
                              side, spot, t_years,
                              short_inner=(gap > 0))
        if legs:
            legs["family"] = "vol_premium"
            legs["direction"] = Direction.NEUTRAL
            out.append(legs)
        return out

    def _family_dislocation(self, z, keys, kmny, mids, ivs, rels, fit,
                            spot, t_years):
        p = self._p
        out = []
        if np.all(~np.isfinite(z)):
            return out
        # cross-section-quiet gate (research spec): if the whole smile is
        # repricing (macro print), the "outlier" is the fit lagging reality
        finite = z[np.isfinite(z)]
        if len(finite) and float(np.median(np.abs(finite))) > 1.0:
            self._z_memory.clear()
            return out
        i = int(np.nanargmax(np.abs(z)))
        if abs(z[i]) < p.dislocation_z_min:
            self._z_memory.pop(keys[i], None) if i < len(keys) else None
            return out
        # persistence: same contract, same sign, consecutive decision
        sign = 1 if z[i] > 0 else -1
        prev = self._z_memory.get(keys[i])
        self._z_memory = {keys[i]: sign}
        if prev != sign:
            return out
        # cheap contract (z<0): buy it, sell the fair neighbor (same right)
        # rich contract (z>0): sell it, buy the neighbor — defined risk both
        right = keys[i].right
        neighbors = [j for j in range(len(keys))
                     if keys[j].right is right and j != i
                     and abs(z[j]) < 1.0 and np.isfinite(z[j])]
        if not neighbors:
            return out
        j = min(neighbors, key=lambda j_: abs(kmny[j_] - kmny[i]))
        legs = self._rv_pair(keys, mids, rels, buy=i if z[i] < 0 else j,
                             sell=j if z[i] < 0 else i, spot=spot,
                             t_years=t_years)
        if legs:
            legs["family"] = "dislocation"
            legs["direction"] = Direction.NEUTRAL
            out.append(legs)
        return out

    def _family_asymmetry(self, q7, fit, keys, kmny, mids, ivs, rels,
                          spot, t_years):
        p = self._p
        lo, hi = q7[0], q7[6]
        width = hi - lo
        if width <= 0:
            return []
        asym = (hi + lo) / width
        if abs(asym) < p.asym_min:
            return []
        right = OptionRight.CALL if asym > 0 else OptionRight.PUT
        atm_i = int(np.nanargmin(np.abs(kmny)))
        step = 1 if right is OptionRight.CALL else -1
        legs = self._vertical(keys, mids, rels, atm_i, atm_i + 2 * step,
                              right, spot, t_years, short_inner=False)
        if legs:
            legs["family"] = "asymmetry"
            legs["direction"] = (Direction.LONG if right is OptionRight.CALL
                                 else Direction.SHORT)
            return [legs]
        return []


    def _family_fair_value(self, ctx, keys, kmny, mids, ivs, rels, spot,
                           t_years):
        """Family D — Muravyev-Pearson staleness convergence.

        Fair value V-hat = BS(S_now, K, T, IV-bar) with IV-bar the trailing
        ~30-minute mean of the CONTRACT'S OWN implied vols. When the quoted
        mid is displaced from V-hat by >= fv_g_min half-spreads, the mid
        empirically converges toward fair value (~54%/min, MP RFS 2020).
        Structure: buy the stale-cheap contract, sell a fairly-priced
        same-right neighbor — the vertical cancels most delta/vega so the
        position isolates the convergence. The convergence gap (taken at the
        conservative fv_capture fraction, net of the hedge leg's own gap)
        enters the ranker as an EV BONUS on top of the standard payoff EV.
        """
        p = self._p
        minute = len(_rth(ctx.bars.get(p.symbol, pd.DataFrame())))
        gaps = np.full(len(keys), np.nan)
        vhats = np.full(len(keys), np.nan)
        for i, key in enumerate(keys):
            hist = [(m_, v_) for m_, v_ in self._iv_hist.get(key, [])
                    if 0 < minute - m_ <= 30]        # strictly PRIOR samples
            if len(hist) < 4:
                continue
            iv_bar = float(np.mean([v_ for _, v_ in hist]))
            med = float(np.median([v_ for _, v_ in hist]))
            if not (0.5 * med <= iv_bar <= 2.0 * med):
                continue
            from catalyst.data.black_scholes import bs_price
            vhat = bs_price(spot, key.strike, t_years, iv_bar, key.right, r=0.0)
            half = mids[i] * rels[i] / 2.0
            if half <= 0:
                continue
            vhats[i] = vhat
            gaps[i] = (vhat - mids[i]) / half
        if np.all(~np.isfinite(gaps)):
            return []
        i = int(np.nanargmax(gaps))          # most stale-CHEAP contract
        if not np.isfinite(gaps[i]) or gaps[i] < p.fv_g_min:
            return []
        right = keys[i].right
        neighbors = [j for j in range(len(keys))
                     if keys[j].right is right and j != i
                     and np.isfinite(gaps[j]) and abs(gaps[j]) <= p.fv_neighbor_g_max]
        if not neighbors:
            return []
        j = min(neighbors, key=lambda j_: abs(kmny[j_] - kmny[i]))
        legs = self._rv_pair(keys, mids, rels, buy=i, sell=j, spot=spot,
                             t_years=t_years)
        if not legs:
            return []
        half_i = mids[i] * rels[i] / 2.0
        # conservative convergence capture, net of what the hedge leg gives up
        conv = p.fv_capture * (vhats[i] - mids[i])
        if np.isfinite(gaps[j]):
            conv -= p.fv_capture * abs(vhats[j] - mids[j])
        legs["family"] = "fair_value"
        legs["direction"] = Direction.NEUTRAL
        legs["ev_bonus"] = max(conv, 0.0)
        return [legs]

    # ------------------------------------------------------------------
    # structure builders (worst-case priced, liquidity gated)
    # ------------------------------------------------------------------
    def _vertical(self, keys, mids, rels, i_inner_hint, _unused, right, spot,
                  t_years, *, short_inner):
        """ATM-anchored two-strike vertical in ``right``. Inner leg at the
        same-right ATM strike, outer leg two strikes away from the money
        (calls: up, puts: down). Worst-case priced at 60% crossing."""
        p = self._p
        same = sorted((n for n, k in enumerate(keys) if k.right is right),
                      key=lambda n: keys[n].strike)
        if len(same) < 3:
            return None
        atm_pos = min(range(len(same)),
                      key=lambda j: abs(keys[same[j]].strike - spot))
        w = self._p.vertical_width_strikes
        outer_pos = atm_pos + (w if right is OptionRight.CALL else -w)
        if outer_pos < 0 or outer_pos >= len(same):
            return None
        i_inner, i_outer = same[atm_pos], same[outer_pos]
        ki, ko = keys[i_inner], keys[i_outer]
        if rels[i_inner] > p.max_rel_spread or rels[i_outer] > p.max_rel_spread:
            return None
        inner_side = Side.SELL if short_inner else Side.BUY
        outer_side = Side.BUY if short_inner else Side.SELL
        m_i, m_o = mids[i_inner], mids[i_outer]
        # FULL modeled friction on entry (crossing + slippage + commissions)
        pay_i = _leg_entry_cost(m_i, rels[i_inner], inner_side is Side.BUY)
        pay_o = _leg_entry_cost(m_o, rels[i_outer], outer_side is Side.BUY)
        unit_cost = pay_i + pay_o
        width = abs(ko.strike - ki.strike)
        if unit_cost >= 0:                       # debit structure
            unit_max_loss = unit_cost
            if unit_cost < 0.03:
                return None
        else:                                    # credit structure
            credit = -unit_cost
            if credit < 0.03 or credit >= width:
                return None
            unit_max_loss = width - credit
        exit_cost = (_leg_exit_cost(m_i, rels[i_inner])
                     + _leg_exit_cost(m_o, rels[i_outer]))
        legs_spec = [
            payoff.LegSpec(right=ki.right, strike=ki.strike, t_years=t_years,
                           iv=self._iv_of(m_i, spot, ki, t_years),
                           qty=(-1 if inner_side is Side.SELL else 1),
                           entry_price=abs(pay_i)),
            payoff.LegSpec(right=ko.right, strike=ko.strike, t_years=t_years,
                           iv=self._iv_of(m_o, spot, ko, t_years),
                           qty=(-1 if outer_side is Side.SELL else 1),
                           entry_price=abs(pay_o)),
        ]
        return {"legs": [OrderLeg(key=ki, side=inner_side, qty=1),
                         OrderLeg(key=ko, side=outer_side, qty=1)],
                "legs_spec": legs_spec, "unit_cost": unit_cost,
                "unit_max_loss": unit_max_loss, "exit_cost": exit_cost}

    def _rv_pair(self, keys, mids, rels, *, buy, sell, spot, t_years):
        p = self._p
        if rels[buy] > p.max_rel_spread or rels[sell] > p.max_rel_spread:
            return None
        pay_b = _leg_entry_cost(mids[buy], rels[buy], True)
        pay_s = _leg_entry_cost(mids[sell], rels[sell], False)
        unit_cost = pay_b + pay_s
        width = abs(keys[buy].strike - keys[sell].strike)
        if width <= 0:
            return None
        if unit_cost >= 0:
            unit_max_loss = unit_cost
            if unit_cost < 0.03:
                return None
        else:
            credit = -unit_cost
            if credit >= width or credit < 0.03:
                return None
            unit_max_loss = width - credit
        legs_spec = [
            payoff.LegSpec(right=keys[buy].right, strike=keys[buy].strike,
                           t_years=t_years,
                           iv=self._iv_of(mids[buy], spot, keys[buy], t_years),
                           qty=1, entry_price=pay_b),
            payoff.LegSpec(right=keys[sell].right, strike=keys[sell].strike,
                           t_years=t_years,
                           iv=self._iv_of(mids[sell], spot, keys[sell], t_years),
                           qty=-1, entry_price=abs(pay_s)),
        ]
        return {"legs": [OrderLeg(key=keys[buy], side=Side.BUY, qty=1),
                         OrderLeg(key=keys[sell], side=Side.SELL, qty=1)],
                "legs_spec": legs_spec, "unit_cost": unit_cost,
                "unit_max_loss": unit_max_loss,
                "exit_cost": (_leg_exit_cost(mids[buy], rels[buy])
                              + _leg_exit_cost(mids[sell], rels[sell]))}

    # ------------------------------------------------------------------
    # market plumbing
    # ------------------------------------------------------------------
    def _session_expiry(self, ctx: IntradayContext) -> date | None:
        """Same-session expiry when listed, else the nearest after."""
        if ctx.data is None:
            return None
        try:
            exps = ctx.data.list_expirations(self._p.symbol)
        except Exception:                               # noqa: BLE001
            return None
        future = [e for e in exps if e >= ctx.session]
        if not future:
            return None
        exp = min(future)
        return exp if (exp - ctx.session).days <= 2 else None

    def _t_years(self, now: datetime, expiry: date) -> float:
        close_dt = datetime.combine(expiry, time(16, 0))
        mins = max((close_dt - now).total_seconds() / 60.0, 1.0)
        return mins / (60.0 * 24.0 * 365.0)

    def _watch_window(self, session: date, spot: float, expiry: date
                      ) -> list[OptionKey]:
        p = self._p
        atm = round(spot / p.strike_step) * p.strike_step
        keys = []
        for i in range(-p.n_strikes_wing, p.n_strikes_wing + 1):
            k = atm + i * p.strike_step
            if k <= 0:
                continue
            for right in (OptionRight.CALL, OptionRight.PUT):
                keys.append(OptionKey(underlying=p.symbol, expiry=expiry,
                                      right=right, strike=k))
        return keys

    def _collect_quotes(self, ctx, window, spot, t_years):
        minute = len(_rth(ctx.bars.get(self._p.symbol, pd.DataFrame())))
        kmny, mids, ivs, rels, keys = [], [], [], [], []
        for key in window:
            q = ctx.option_quote(key)
            if q is None:
                continue
            bid, ask = q
            if bid <= 0 or ask <= 0 or bid > ask:
                continue
            mid = (bid + ask) / 2.0
            if mid <= 0.02:
                continue
            iv = self._iv_of(mid, spot, key, t_years)
            if iv is None or not iv == iv:
                continue
            keys.append(key)
            kmny.append(math.log(key.strike / spot))
            mids.append(mid)
            ivs.append(iv)
            rels.append((ask - bid) / mid)
            hist = self._iv_hist.setdefault(key, [])
            hist.append((minute, iv))
            # keep ~35 minutes of samples
            self._iv_hist[key] = [(m_, v_) for m_, v_ in hist if minute - m_ <= 35]
        if len(keys) < 6:
            return None
        return (keys, np.array(kmny), np.array(mids), np.array(ivs),
                np.array(rels))

    def _iv_of(self, mid: float, spot: float, key: OptionKey,
               t_years: float) -> float:
        iv = implied_vol(mid, spot, key.strike, t_years, key.right, r=0.0)
        return iv if iv is not None else float("nan")


def build(cfg) -> MosaicStrategy:
    """Registry entry point. Warmup bars come from the cached Alpaca minute
    store — strictly historical injection, same precedent as short_vrp.
    Config maps onto MosaicParams so --set sweeps reach every gate."""
    from catalyst.data.cache import ParquetCache
    from catalyst.data.intraday import AlpacaMinuteBars

    cache = ParquetCache(cfg.data.cache_dir)
    bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
    m = getattr(cfg, "mosaic", None)
    params = MosaicParams() if m is None else MosaicParams(
        symbol=m.symbol,
        horizon_minutes=m.horizon_minutes,
        vol_premium_min=m.vol_premium_min,
        dislocation_z_min=m.dislocation_z_min,
        asym_min=m.asym_min,
        ev_min_frac_of_maxloss=m.ev_min_frac_of_maxloss,
        ev_min_dollars=m.ev_min_dollars,
        max_rel_spread=m.max_rel_spread,
        max_hold_minutes=m.max_hold_minutes,
        stop_loss_pct=m.stop_loss_pct,
        credit_stop_multiple=m.credit_stop_multiple,
        per_trade_risk_fraction=m.per_trade_risk_fraction,
        warmup_sessions=m.warmup_sessions,
        use_stops=m.use_stops,
        vertical_width_strikes=m.vertical_width_strikes,
        fv_g_min=m.fv_g_min,
    )
    return MosaicStrategy(params=params, warmup_bars_provider=bars)


register(
    StrategyMeta(
        name="mosaic",
        module="catalyst.strategies.active.mosaic",
        status="active",
        notes=("v16. Unified intraday options algorithm: conditional return "
               "distributions x fitted smile x RV forecast -> payoff-"
               "transformed EV over defined-risk verticals; three candidate "
               "families (vol premium, smile dislocation, distribution "
               "asymmetry) through one EV gate. Research/simulation only."),
    ),
    build,
)

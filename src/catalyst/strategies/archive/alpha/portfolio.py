"""Market-neutral portfolio construction and backtest.

Each component here answers a specific failure measured earlier in this
project, rather than adding sophistication for its own sake:

- **Walk-forward signal weighting** — weights come from a TRAILING information
  coefficient window and are applied forward only. Engine C's +$125/trade at
  N=85 evaporated at N=926; choosing signals on the full sample is the same
  mistake in a different costume.
- **Market neutrality** — equal dollar long and short. Long-only momentum over
  2018-2026 mostly measures the S&P, and a green number below buy-and-hold is
  not an edge. Every result is also reported against SPY.
- **Inverse-volatility sizing** — the Gate 3 dial showed identical trades
  compounding at one size and self-destructing at another. Equal risk, not
  equal dollars, per name.
- **Turnover control** — costs consumed 53% of collected credit in the last
  campaign. Positions only move when the target differs enough to be worth the
  spread, and turnover is reported as a first-class number.
- **Regime scaling** — the drift campaign's edge was regime-dependent, so
  gross exposure scales down when market volatility spikes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class PortfolioConfig:
    rebalance_days: int = 5
    n_long: int = 20
    n_short: int = 20
    gross_exposure: float = 1.0  # 1.0 = 50% long + 50% short of equity
    inverse_vol_sizing: bool = True
    vol_window: int = 60
    turnover_threshold: float = 0.20  # skip trades smaller than this share of target
    half_spread_bps: float = 1.5
    slippage_bps: float = 2.0
    borrow_apr: float = 0.005
    ic_window_days: int = 504  # trailing window for walk-forward signal weights
    min_ic_weight: float = 0.0  # signals with trailing IC below this get zero weight
    regime_vol_window: int = 20
    regime_vol_cap: float = 0.30  # scale down gross above this annualized market vol
    market_neutral: bool = True
    # --- v2 construction (each fixes a diagnosed loss) ---
    score_weighted: bool = False   # weight the FULL cross-section by score instead of
    #   trading only the top/bottom buckets. Decile portfolios discard the information
    #   in the middle of the distribution and concentrate risk in ~40 names; the
    #   fundamental law says breadth is what converts a small IC into a Sharpe.
    signal_smoothing_days: int = 1  # average the composite over N days before ranking;
    #   raw weekly scores whipsaw and pay the spread for noise
    beta_neutralize: bool = False   # subtract each name's market beta so the book is
    #   beta-neutral by construction, not just dollar-neutral


@dataclass
class BacktestOutput:
    equity: pd.Series
    benchmark: pd.Series
    turnover: pd.Series
    gross_exposure: pd.Series
    weights_history: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    signal_weights: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    costs_total: float = 0.0

    def stats(self) -> dict[str, float]:
        rets = self.equity.pct_change().dropna()
        bench = self.benchmark.pct_change().dropna()
        years = len(self.equity) / TRADING_DAYS
        cagr = (self.equity.iloc[-1] / self.equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
        bench_cagr = (self.benchmark.iloc[-1] / self.benchmark.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
        vol = float(rets.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(rets) > 1 else 0.0
        sharpe = cagr / vol if vol > 0 else 0.0
        dd = float((self.equity / self.equity.cummax() - 1.0).min())
        # Alpha/beta vs the benchmark on overlapping daily returns.
        joined = pd.concat([rets.rename("p"), bench.rename("b")], axis=1).dropna()
        if len(joined) > 30 and joined["b"].var() > 0:
            beta = float(joined["p"].cov(joined["b"]) / joined["b"].var())
            alpha_ann = float(joined["p"].mean() - beta * joined["b"].mean()) * TRADING_DAYS
        else:
            beta, alpha_ann = 0.0, 0.0
        return {
            "cagr": cagr, "benchmark_cagr": bench_cagr, "excess_cagr": cagr - bench_cagr,
            "vol": vol, "sharpe": sharpe, "max_drawdown": dd, "beta": beta,
            "alpha_ann": alpha_ann, "avg_turnover": float(self.turnover.mean()),
            "total_costs": self.costs_total,
        }


def trailing_ic_weights(
    signals: dict[str, pd.DataFrame], prices: pd.DataFrame, asof: pd.Timestamp,
    window_days: int, horizon: int, min_weight: float,
) -> dict[str, float]:
    """Signal weights from trailing IC only — never from data at or after ``asof``.

    Uses non-overlapping samples inside the window for the same reason the
    diagnostic does: overlapping windows make noise look like skill.
    """
    from catalyst.alpha.evaluation import forward_returns, information_coefficient

    start = asof - pd.Timedelta(days=window_days)
    # The forward return for date d needs prices through d+horizon, so the last
    # usable signal date is horizon trading days before asof.
    usable = prices.index[(prices.index >= start) & (prices.index < asof)]
    if len(usable) < horizon * 6:
        return {name: 1.0 / len(signals) for name in signals}  # too little history: equal weight
    cutoff = usable[-horizon] if len(usable) > horizon else usable[-1]

    fwd = forward_returns(prices.loc[usable], horizon)
    raw: dict[str, float] = {}
    for name, panel in signals.items():
        sub = panel.loc[usable]
        sub = sub[sub.index <= cutoff].iloc[::horizon]  # non-overlapping
        if sub.empty:
            raw[name] = 0.0
            continue
        ic = information_coefficient(sub, fwd)
        raw[name] = float(ic.mean()) if len(ic) else 0.0

    positive = {k: v for k, v in raw.items() if v > min_weight}
    if not positive:
        return {k: 0.0 for k in signals}
    total = sum(positive.values())
    return {k: (positive.get(k, 0.0) / total) for k in signals}


def _score_weighted(composite: pd.Series, vols: pd.Series, betas: pd.Series | None,
                    cfg: PortfolioConfig) -> dict[str, float]:
    """Weight every name by its demeaned score — full-breadth construction.

    Demeaning enforces dollar neutrality; dividing by volatility enforces equal
    risk contribution; the optional beta adjustment removes residual market
    exposure that dollar-neutrality alone does not eliminate.
    """
    scores = composite.dropna()
    if len(scores) < 20:
        return {}
    centered = scores - scores.mean()
    if cfg.inverse_vol_sizing:
        v = vols.reindex(centered.index)
        centered = centered / v.replace(0.0, np.nan)
        centered = centered.dropna()
    if centered.empty or centered.abs().sum() == 0:
        return {}
    weights = centered / centered.abs().sum() * cfg.gross_exposure

    if cfg.beta_neutralize and betas is not None:
        b = betas.reindex(weights.index).fillna(1.0)
        net_beta = float((weights * b).sum())
        if abs(b.pow(2).sum()) > 1e-9:
            weights = weights - b * (net_beta / float(b.pow(2).sum()))
            scale = weights.abs().sum()
            if scale > 0:
                weights = weights / scale * cfg.gross_exposure
    return {k: float(v) for k, v in weights.items() if abs(v) > 1e-6}


def _target_weights(
    composite: pd.Series, vols: pd.Series, cfg: PortfolioConfig,
    betas: pd.Series | None = None,
) -> dict[str, float]:
    """Target portfolio weights: long the best scores, short the worst."""
    if cfg.score_weighted:
        return _score_weighted(composite, vols, betas, cfg)
    scores = composite.dropna()
    if len(scores) < cfg.n_long + cfg.n_short:
        return {}
    ranked = scores.sort_values()
    shorts = list(ranked.index[: cfg.n_short])
    longs = list(ranked.index[-cfg.n_long:])

    def leg(names: list[str], sign: float, budget: float) -> dict[str, float]:
        if cfg.inverse_vol_sizing:
            inv = {n: 1.0 / vols[n] for n in names if n in vols and vols[n] > 0}
            if not inv:
                return {}
            total = sum(inv.values())
            return {n: sign * budget * w / total for n, w in inv.items()}
        return {n: sign * budget / len(names) for n in names}

    half = cfg.gross_exposure / 2.0
    weights = leg(longs, 1.0, half)
    if cfg.market_neutral:
        weights.update(leg(shorts, -1.0, half))
    return weights


def run_portfolio_backtest(
    prices: pd.DataFrame, market: pd.Series, signals: dict[str, pd.DataFrame],
    cfg: PortfolioConfig, starting_equity: float = 100_000.0,
) -> BacktestOutput:
    rets = prices.pct_change()
    vols = rets.rolling(cfg.vol_window).std() * np.sqrt(TRADING_DAYS)
    mkt_rets = market.pct_change().reindex(rets.index)
    mkt_vol = mkt_rets.rolling(cfg.regime_vol_window).std() * np.sqrt(TRADING_DAYS)
    mkt_var = mkt_rets.rolling(120).var()
    betas = rets.apply(lambda col: col.rolling(120).cov(mkt_rets)) .div(
        mkt_var.replace(0.0, np.nan), axis=0).clip(-3, 3) if cfg.beta_neutralize else None

    dates = prices.index
    equity = starting_equity
    holdings: dict[str, float] = {}  # symbol -> signed weight of equity
    equity_curve: dict[pd.Timestamp, float] = {}
    turnover_series: dict[pd.Timestamp, float] = {}
    gross_series: dict[pd.Timestamp, float] = {}
    out = BacktestOutput(pd.Series(dtype=float), pd.Series(dtype=float),
                         pd.Series(dtype=float), pd.Series(dtype=float))
    cost_bps = (cfg.half_spread_bps + cfg.slippage_bps) / 1e4
    total_costs = 0.0

    for i, date in enumerate(dates):
        # ---- accrue one day of P&L on existing holdings ----
        if i > 0 and holdings:
            day_ret = rets.loc[date]
            pnl = 0.0
            for sym, w in holdings.items():
                r = day_ret.get(sym, np.nan)
                if pd.notna(r):
                    pnl += w * r
                if w < 0:  # borrow cost on the short leg
                    pnl -= abs(w) * cfg.borrow_apr / TRADING_DAYS
            equity *= (1.0 + pnl)

        # ---- rebalance ----
        if i % cfg.rebalance_days == 0 and i >= cfg.ic_window_days // 4:
            weights = trailing_ic_weights(signals, prices, date, cfg.ic_window_days,
                                          cfg.rebalance_days, cfg.min_ic_weight)
            out.signal_weights[date] = weights
            if sum(weights.values()) > 0:
                if cfg.signal_smoothing_days > 1:
                    window = prices.index[max(0, i - cfg.signal_smoothing_days + 1): i + 1]
                    composite = sum(signals[n].loc[window].mean() * w
                                    for n, w in weights.items() if w > 0)
                else:
                    composite = sum(signals[n].loc[date].fillna(0.0) * w
                                    for n, w in weights.items() if w > 0)
                # Regime scaling: cut gross exposure when market vol is elevated.
                scale = 1.0
                mv = mkt_vol.get(date, np.nan)
                if pd.notna(mv) and mv > cfg.regime_vol_cap:
                    scale = float(cfg.regime_vol_cap / mv)
                scaled = PortfolioConfig(**{**cfg.__dict__,
                                            "gross_exposure": cfg.gross_exposure * scale})
                target = _target_weights(composite, vols.loc[date], scaled,
                                         betas.loc[date] if betas is not None else None)

                # Turnover control: only move a position when the change is
                # material relative to the target, so small drifts do not pay
                # the spread every week.
                traded = 0.0
                new_holdings = dict(holdings)
                for sym in set(target) | set(holdings):
                    tgt = target.get(sym, 0.0)
                    cur = holdings.get(sym, 0.0)
                    delta = tgt - cur
                    if abs(delta) < cfg.turnover_threshold * max(abs(tgt), 1e-9) and tgt != 0:
                        continue
                    if abs(delta) < 1e-6:
                        continue
                    traded += abs(delta)
                    if tgt == 0.0:
                        new_holdings.pop(sym, None)
                    else:
                        new_holdings[sym] = tgt
                holdings = new_holdings
                cost = traded * cost_bps
                equity *= (1.0 - cost)
                total_costs += cost * equity
                turnover_series[date] = traded
                gross_series[date] = sum(abs(w) for w in holdings.values())

        equity_curve[date] = equity

    idx = pd.DatetimeIndex(list(equity_curve.keys()))
    bench = market.reindex(idx).ffill()
    bench = bench / bench.iloc[0] * starting_equity
    out.equity = pd.Series(list(equity_curve.values()), index=idx)
    out.benchmark = bench
    out.turnover = pd.Series(turnover_series) if turnover_series else pd.Series([0.0])
    out.gross_exposure = pd.Series(gross_series) if gross_series else pd.Series([0.0])
    out.costs_total = total_costs
    return out

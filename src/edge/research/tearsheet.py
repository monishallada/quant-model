"""Per-signal tearsheet rendering and the EXPECTANCY/REJECTED markdown writers.

Consumes a RunResult ledger (one row per closed trade, columns listed in
:data:`REQUIRED_LEDGER_COLUMNS`), the run's equity series, the append-only
:class:`~edge.research.registry.TrialRegistry` (for the cumulative trial
count that deflates the Sharpe ratio), and a regime bucket series. Produces:

* THE HEADLINE BLOCK — the operator's specified contract, rendered verbatim
  by :func:`render_headline_block` (geometric monthly return leads; monthly
  returns come from the equity series resampled by calendar month; the 95%
  CI comes from :func:`edge.validation.stats.bootstrap_ci` on those monthly
  returns; below :data:`MONTHS_TO_DISTINGUISH_SHARPE_ONE` observed months a
  statistical-power warning line is appended).
* Trade stats, the always-present five-line cost attribution, the
  four-bucket regime breakdown, deflated-Sharpe/PBO diagnostics, and the
  PROMOTION CHECK with named failures.
* ``append_expectancy_md`` — ranks EVERY signal ever tested (failures
  included) by headline geometric monthly return.
* ``append_rejected_md`` — records a rejected signal with its named gate
  failures and the numbers that failed them.

All thresholds come from :class:`edge.core.config.ValidationConfig`; the
only literals here are formatting and two documented contract constants
(:data:`MONTHS_TO_DISTINGUISH_SHARPE_ONE`,
:data:`MIN_POSITIVE_REGIME_BUCKETS`). This module reads no market data and
never touches the data layer — everything arrives pre-computed from the
engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from edge.core.config import ValidationConfig
from edge.research.registry import TrialRegistry
from edge.validation.stats import bootstrap_ci, deflated_sharpe

__all__ = [
    "FAIL_CI",
    "FAIL_LATENCY",
    "FAIL_PBO_HIGH",
    "FAIL_PBO_MISSING",
    "FAIL_REGIME",
    "FAIL_TRADES",
    "MIN_POSITIVE_REGIME_BUCKETS",
    "MONTHS_TO_DISTINGUISH_SHARPE_ONE",
    "REGIME_BUCKETS",
    "REQUIRED_LEDGER_COLUMNS",
    "VERDICT_PROMOTE",
    "VERDICT_REJECT",
    "CostAttribution",
    "ExpectancyRow",
    "MonthlyHeadline",
    "PromotionResult",
    "RegimeBucketStat",
    "TradeStats",
    "append_expectancy_md",
    "append_rejected_md",
    "compute_cost_attribution",
    "compute_monthly_headline",
    "compute_regime_breakdown",
    "compute_trade_stats",
    "evaluate_promotion",
    "monthly_returns",
    "render_headline_block",
    "render_tearsheet",
]

#: Columns every RunResult ledger row must carry (one row per closed trade).
#: ``pnl`` is NET dollars for the trade — costs already deducted; the three
#: cost columns are the per-trade friction that was deducted, so
#: ``gross = pnl + spread_cost + slippage_cost + commission`` exactly.
REQUIRED_LEDGER_COLUMNS: tuple[str, ...] = (
    "pnl",
    "r_multiple",
    "entry_ts",
    "exit_ts",
    "signal_name",
    "spread_cost",
    "slippage_cost",
    "commission",
    "mfe",
    "mae",
    "exit_reason",
)

#: The four canonical regime buckets (volatility level x price behavior).
#: The regime bucket series handed to the tearsheet must take values from
#: this set; every tearsheet displays all four, zero-trade buckets included.
REGIME_BUCKETS: tuple[str, ...] = (
    "low_vol_trending",
    "low_vol_chopping",
    "high_vol_trending",
    "high_vol_chopping",
)

#: Months of monthly returns below which the headline block appends its
#: statistical-power warning. Derivation: an annual Sharpe of 1.0 is a
#: monthly Sharpe of 1/sqrt(12) ~= 0.289, whose t-stat vs zero reaches the
#: two-sided 95% critical value ~1.96 near n = (1.96*sqrt(12))^2 ~= 46
#: months. The value 46 is fixed by the operator's headline contract.
MONTHS_TO_DISTINGUISH_SHARPE_ONE: int = 46

#: Regime robustness gate: at least this many of the four buckets must show
#: positive expectancy for promotion (per the operator's promotion contract).
MIN_POSITIVE_REGIME_BUCKETS: int = 3

VERDICT_PROMOTE = "PROMOTE"
VERDICT_REJECT = "REJECT"

# Named promotion failures — stable identifiers, quoted verbatim in the
# PROMOTION CHECK verdict line and in REJECTED.md.
FAIL_CI = "expectancy-ci-includes-zero"
FAIL_TRADES = "insufficient-oos-trades"
FAIL_PBO_HIGH = "pbo-too-high"
FAIL_PBO_MISSING = "pbo-missing"
FAIL_LATENCY = "latency-failure"
FAIL_REGIME = "regime-instability"

#: Label column width inside indented key/value lines: two-space indent plus
#: a 26-char label field puts every value at the same column, matching the
#: operator's headline block character-for-character.
_LABEL_WIDTH = 26

_RULE = "=" * 66


def _kv(label: str, value: str, suffix: str = "") -> str:
    """One indented ``label value`` line at the contract's fixed alignment."""
    return f"  {label:<{_LABEL_WIDTH}}{value}{suffix}"


def _pct1(x: float) -> str:
    """Fraction as a one-decimal percent: 0.021 -> '2.1%'."""
    return f"{x * 100.0:.1f}%"


def _rfmt(x: float) -> str:
    """Signed R-multiple with two decimals: 0.4 -> '+0.40R'."""
    return f"{x:+.2f}R"


def _money(x: float) -> str:
    """Signed dollars with two decimals (no thousands separators)."""
    return f"{x:+.2f}"


# ---------------------------------------------------------------------------
# Headline block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthlyHeadline:
    """Inputs to the operator's headline block, all pre-computed.

    Return fields (``geometric`` ... ``best``) are fractions (0.021 = 2.1%);
    ``pct_positive`` is already in percent (0-100).
    """

    geometric: float
    arithmetic: float
    ci_lo: float
    ci_hi: float
    months: int
    median: float
    worst: float
    best: float
    pct_positive: float
    t_stat: float


def monthly_returns(equity: pd.Series) -> npt.NDArray[np.float64]:
    """Calendar-month returns of an equity series.

    The equity series (tz-aware DatetimeIndex, strictly positive values,
    non-decreasing timestamps) is resampled to calendar month-ends taking the
    last observation of each month; months with no observations are skipped,
    so the next observed month's return spans the gap and geometric
    compounding over the whole series stays exact. The first month's return
    is measured from the series' first observation (its starting equity).

    Raises:
        ValueError: non-DatetimeIndex, unsorted index, non-positive equity,
            or fewer than two monthly returns.
    """
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("equity must be indexed by a DatetimeIndex")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity index must be non-decreasing")
    values = equity.astype(np.float64)
    if bool((values <= 0.0).any()):
        raise ValueError("equity must be strictly positive")
    ends = values.resample("ME").last().dropna()
    prev = ends.shift(1)
    prev.iloc[0] = float(values.iloc[0])
    rets = (ends / prev - 1.0).to_numpy(dtype=np.float64)
    if rets.size < 2:
        raise ValueError(
            f"tearsheet requires at least two monthly returns, got {rets.size}"
        )
    return rets


def compute_monthly_headline(
    equity: pd.Series, config: ValidationConfig, *, seed: int = 0
) -> MonthlyHeadline:
    """Headline numbers from the equity series (see :func:`monthly_returns`).

    Geometric leads: ``prod(1+r)^(1/n) - 1`` over the monthly returns. The
    95% CI is a percentile bootstrap of the monthly-return mean with
    ``config.bootstrap_resamples`` resamples at the given ``seed``.

    Raises:
        ValueError: propagated from :func:`monthly_returns`; any month at or
            below -100%; zero-variance monthly returns (t-stat undefined).
    """
    rets = monthly_returns(equity)
    if float(rets.min()) <= -1.0:
        raise ValueError("geometric mean undefined: a month lost 100% or more")
    n = rets.size
    geometric = float(np.exp(np.mean(np.log1p(rets))) - 1.0)
    arithmetic = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        raise ValueError("t-stat undefined for zero-variance monthly returns")
    lo, hi = bootstrap_ci(rets, resamples=config.bootstrap_resamples, seed=seed)
    return MonthlyHeadline(
        geometric=geometric,
        arithmetic=arithmetic,
        ci_lo=lo,
        ci_hi=hi,
        months=n,
        median=float(np.median(rets)),
        worst=float(rets.min()),
        best=float(rets.max()),
        pct_positive=float(100.0 * np.mean(rets > 0.0)),
        t_stat=float(np.mean(rets) / (sd / math.sqrt(n))),
    )


def render_headline_block(headline: MonthlyHeadline) -> str:
    """THE HEADLINE BLOCK — the operator's contract, character for character.

    Below :data:`MONTHS_TO_DISTINGUISH_SHARPE_ONE` observed months, the
    statistical-power NOTE line is appended verbatim. Returns a string
    ending in a newline.
    """
    lines = [
        "AVG MONTHLY RETURN (out-of-sample)",
        _kv(
            "Geometric (compounded):",
            _pct1(headline.geometric),
            "   <- what the account actually does",
        ),
        _kv("Arithmetic (mean):", _pct1(headline.arithmetic)),
        _kv(
            "95% bootstrap CI:",
            f"[{_pct1(headline.ci_lo)}, {_pct1(headline.ci_hi)}]",
        ),
        _kv("Months observed:", str(headline.months)),
        _kv("Median month:", _pct1(headline.median)),
        _kv("Worst month:", _pct1(headline.worst)),
        _kv("Best month:", _pct1(headline.best)),
        _kv("% of months positive:", f"{headline.pct_positive:.0f}%"),
        _kv("t-stat vs zero:", f"{headline.t_stat:.2f}"),
    ]
    if headline.months < MONTHS_TO_DISTINGUISH_SHARPE_ONE:
        lines.append(
            f"NOTE: {headline.months} months < {MONTHS_TO_DISTINGUISH_SHARPE_ONE} "
            "needed to distinguish a Sharpe-1.0 strategy from zero."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Trade stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeStats:
    """Per-signal trade statistics; ratios are ``None`` when undefined."""

    n: int
    expectancy_r: float
    ci_lo: float
    ci_hi: float
    hit_rate: float          # fraction of trades with pnl > 0
    payoff_ratio: float | None   # avg win / |avg loss|; None without both sides
    avg_holding_hours: float
    mfe_capture: float | None    # sum(pnl of winners) / sum(mfe of winners)

    @property
    def ci_excludes_zero(self) -> bool:
        """True iff the expectancy CI strictly excludes zero (either side)."""
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


def compute_trade_stats(
    trades: pd.DataFrame, config: ValidationConfig, *, seed: int = 0
) -> TradeStats:
    """Trade stats over a (already signal-filtered) ledger slice.

    Expectancy is the mean ``r_multiple``; its 95% CI is a percentile
    bootstrap with ``config.bootstrap_resamples`` resamples. MFE capture is
    realized pnl over MFE, summed across winners (pnl > 0).

    Raises:
        ValueError: empty ``trades``.
    """
    if len(trades) == 0:
        raise ValueError("compute_trade_stats requires at least one trade")
    r = trades["r_multiple"].to_numpy(dtype=np.float64)
    pnl = trades["pnl"].to_numpy(dtype=np.float64)
    lo, hi = bootstrap_ci(r, resamples=config.bootstrap_resamples, seed=seed)
    winners = pnl > 0.0
    losers = pnl < 0.0
    payoff: float | None = None
    if bool(winners.any()) and bool(losers.any()):
        payoff = float(pnl[winners].mean() / abs(pnl[losers].mean()))
    mfe_capture: float | None = None
    if bool(winners.any()):
        mfe_sum = float(trades.loc[winners, "mfe"].sum())
        if mfe_sum > 0.0:
            mfe_capture = float(pnl[winners].sum() / mfe_sum)
    holding = (trades["exit_ts"] - trades["entry_ts"]).mean()
    return TradeStats(
        n=int(len(trades)),
        expectancy_r=float(r.mean()),
        ci_lo=lo,
        ci_hi=hi,
        hit_rate=float(winners.mean()),
        payoff_ratio=payoff,
        avg_holding_hours=float(holding.total_seconds() / 3600.0),
        mfe_capture=mfe_capture,
    )


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostAttribution:
    """Whole-run friction decomposition in dollars.

    Invariant (by construction): ``gross - spread - slippage - fees == net``,
    where ``net`` is the ledger's summed (net) pnl.
    """

    gross: float
    spread: float
    slippage: float
    fees: float
    net: float


def compute_cost_attribution(trades: pd.DataFrame) -> CostAttribution:
    """Sum the ledger's cost columns into the five attribution numbers."""
    net = float(trades["pnl"].sum())
    spread = float(trades["spread_cost"].sum())
    slippage = float(trades["slippage_cost"].sum())
    fees = float(trades["commission"].sum())
    return CostAttribution(
        gross=net + spread + slippage + fees,
        spread=spread,
        slippage=slippage,
        fees=fees,
        net=net,
    )


def _render_cost_block(costs: CostAttribution) -> list[str]:
    """The five-line cost table — always rendered, zero costs included."""

    def _neg(cost: float) -> float:
        return -cost if cost != 0.0 else 0.0

    return [
        "COST ATTRIBUTION ($)",
        _kv("Gross P&L:", _money(costs.gross)),
        _kv("Spread:", _money(_neg(costs.spread))),
        _kv("Slippage:", _money(_neg(costs.slippage))),
        _kv("Fees:", _money(_neg(costs.fees))),
        _kv("Net P&L:", _money(costs.net)),
    ]


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBucketStat:
    """One regime bucket's trade count and expectancy (None when empty)."""

    bucket: str
    n: int
    expectancy_r: float | None


def compute_regime_breakdown(
    trades: pd.DataFrame, regime: pd.Series
) -> tuple[list[RegimeBucketStat], int]:
    """Assign each trade the regime prevailing at entry; aggregate per bucket.

    ``regime`` is a tz-aware, time-sorted series of :data:`REGIME_BUCKETS`
    labels; each trade takes the last label at or before its ``entry_ts``.
    Returns the four bucket stats in canonical order plus the count of
    positive buckets (n > 0 and expectancy > 0).

    Raises:
        ValueError: unsorted regime index, labels outside
            :data:`REGIME_BUCKETS`, or a trade entering before the first
            regime observation.
    """
    if not isinstance(regime.index, pd.DatetimeIndex):
        raise ValueError("regime must be indexed by a DatetimeIndex")
    if not regime.index.is_monotonic_increasing:
        raise ValueError("regime index must be non-decreasing")
    unknown = set(regime.unique()) - set(REGIME_BUCKETS)
    if unknown:
        raise ValueError(f"unknown regime buckets: {sorted(unknown)}")
    assigned = regime.reindex(
        pd.DatetimeIndex(trades["entry_ts"]), method="ffill"
    )
    if bool(assigned.isna().any()):
        raise ValueError("trade entered before the first regime observation")
    labels = assigned.to_numpy()
    r = trades["r_multiple"].to_numpy(dtype=np.float64)
    stats: list[RegimeBucketStat] = []
    positive = 0
    for bucket in REGIME_BUCKETS:
        mask = labels == bucket
        n = int(mask.sum())
        expectancy = float(r[mask].mean()) if n else None
        if expectancy is not None and expectancy > 0.0:
            positive += 1
        stats.append(RegimeBucketStat(bucket=bucket, n=n, expectancy_r=expectancy))
    return stats, positive


def _render_regime_block(
    stats: Sequence[RegimeBucketStat], positive: int
) -> list[str]:
    lines = ["REGIME BREAKDOWN"]
    for s in stats:
        expectancy = _rfmt(s.expectancy_r) if s.expectancy_r is not None else "n/a"
        lines.append(f"  {s.bucket:<18}N={s.n:<6}expectancy={expectancy}")
    lines.append(_kv("Positive buckets:", f"{positive}/{len(REGIME_BUCKETS)}"))
    return lines


# ---------------------------------------------------------------------------
# Promotion check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionResult:
    """Verdict of the promotion gate with its named failures (empty if promoted)."""

    promoted: bool
    failures: tuple[str, ...]

    @property
    def verdict(self) -> str:
        return VERDICT_PROMOTE if self.promoted else VERDICT_REJECT


def evaluate_promotion(
    *,
    n_trades: int,
    expectancy_ci: tuple[float, float],
    pbo: float | None,
    latency_survived: bool,
    positive_buckets: int,
    config: ValidationConfig,
) -> PromotionResult:
    """Apply the five promotion gates; failures are named and ordered.

    Gates (all must pass):
      1. expectancy CI strictly above zero (a CI touching zero fails) —
         :data:`FAIL_CI`;
      2. ``n_trades >= config.min_oos_trades`` — :data:`FAIL_TRADES`;
      3. ``pbo < config.pbo_max`` (a missing pbo fails as
         :data:`FAIL_PBO_MISSING`, an equal-or-higher one as
         :data:`FAIL_PBO_HIGH`);
      4. the latency-survival flag — :data:`FAIL_LATENCY`;
      5. ``positive_buckets >= MIN_POSITIVE_REGIME_BUCKETS`` —
         :data:`FAIL_REGIME`.
    """
    failures: list[str] = []
    ci_lo, _ = expectancy_ci
    if not ci_lo > 0.0:
        failures.append(FAIL_CI)
    if n_trades < config.min_oos_trades:
        failures.append(FAIL_TRADES)
    if pbo is None:
        failures.append(FAIL_PBO_MISSING)
    elif not pbo < config.pbo_max:
        failures.append(FAIL_PBO_HIGH)
    if not latency_survived:
        failures.append(FAIL_LATENCY)
    if positive_buckets < MIN_POSITIVE_REGIME_BUCKETS:
        failures.append(FAIL_REGIME)
    return PromotionResult(promoted=not failures, failures=tuple(failures))


def _render_promotion_block(
    result: PromotionResult,
    *,
    n_trades: int,
    ci_lo: float,
    pbo: float | None,
    latency_survived: bool,
    positive_buckets: int,
    config: ValidationConfig,
) -> list[str]:
    def _mark(ok: bool) -> str:
        return "[PASS]" if ok else "[FAIL]"

    pbo_str = f"{pbo:.3f}" if pbo is not None else "n/a"
    lines = [
        "PROMOTION CHECK",
        f"  {_mark(FAIL_CI not in result.failures)} "
        f"expectancy CI excludes zero (lo={_rfmt(ci_lo)})",
        f"  {_mark(FAIL_TRADES not in result.failures)} "
        f"oos trades {n_trades} >= {config.min_oos_trades}",
        f"  {_mark(FAIL_PBO_HIGH not in result.failures and FAIL_PBO_MISSING not in result.failures)} "
        f"pbo {pbo_str} < {config.pbo_max:.3f}",
        f"  {_mark(FAIL_LATENCY not in result.failures)} latency survival",
        f"  {_mark(FAIL_REGIME not in result.failures)} "
        f"positive regime buckets {positive_buckets} >= {MIN_POSITIVE_REGIME_BUCKETS}",
    ]
    if result.promoted:
        lines.append(f"  VERDICT: {VERDICT_PROMOTE}")
    else:
        lines.append(
            f"  VERDICT: {VERDICT_REJECT} (failed: {', '.join(result.failures)})"
        )
    return lines


# ---------------------------------------------------------------------------
# Deflated Sharpe / PBO diagnostics
# ---------------------------------------------------------------------------


def _deflated_sharpe_of_monthly(
    rets: npt.NDArray[np.float64], registry: TrialRegistry
) -> tuple[float | None, int]:
    """DSR of the monthly returns against the registry's full trial history.

    ``n_trials`` is the registry's cumulative count (floored at 1 so an
    empty registry deflates by nothing rather than erroring); ``var_trials``
    is the variance of recorded trial Sharpes. Returns ``(None, n_trials)``
    when the DSR is undefined for these inputs (tiny samples, degenerate
    moments) — diagnostics never abort a tearsheet.
    """
    n_trials = max(1, registry.cumulative_trial_count())
    var_trials = registry.var_of_trial_sharpes()
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return None, n_trials
    sr_hat = float(np.mean(rets)) / sd
    sk = float(_skew(rets))
    ku = float(_kurtosis(rets, fisher=False))
    if not (math.isfinite(sk) and math.isfinite(ku)):
        return None, n_trials
    try:
        dsr = deflated_sharpe(sr_hat, rets.size, sk, ku, n_trials, var_trials)
    except ValueError:
        return None, n_trials
    return (dsr, n_trials) if math.isfinite(dsr) else (None, n_trials)


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------


def render_tearsheet(
    signal: str,
    ledger: pd.DataFrame,
    equity: pd.Series,
    registry: TrialRegistry,
    regime: pd.Series,
    config: ValidationConfig,
    *,
    pbo: float | None = None,
    latency_survived: bool,
    seed: int = 0,
) -> str:
    """Render the full per-signal tearsheet as plain text.

    Filters ``ledger`` to rows whose ``signal_name`` equals ``signal``;
    ``equity`` is the signal's out-of-sample equity curve. ``pbo`` is the
    hook for a CSCV probability of backtest overfitting computed upstream
    (``edge.validation.stats.pbo_cscv`` over the sweep's trial-return
    matrix); ``None`` means "not computed", which fails promotion as
    :data:`FAIL_PBO_MISSING` rather than passing silently.
    ``latency_survived`` is the flag from running the fill model under every
    ``ExecutionConfig.latency_ms`` scenario — it must be supplied
    explicitly.

    Raises:
        ValueError: missing ledger columns, no trades for ``signal``, or
            invalid equity/regime inputs (propagated from the computations).
    """
    missing = [c for c in REQUIRED_LEDGER_COLUMNS if c not in ledger.columns]
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")
    trades = ledger.loc[ledger["signal_name"] == signal]
    if len(trades) == 0:
        raise ValueError(f"no trades in ledger for signal {signal!r}")

    headline = compute_monthly_headline(equity, config, seed=seed)
    trade_stats = compute_trade_stats(trades, config, seed=seed)
    costs = compute_cost_attribution(trades)
    regime_stats, positive_buckets = compute_regime_breakdown(trades, regime)
    rets = monthly_returns(equity)
    dsr, n_trials = _deflated_sharpe_of_monthly(rets, registry)
    promotion = evaluate_promotion(
        n_trades=trade_stats.n,
        expectancy_ci=(trade_stats.ci_lo, trade_stats.ci_hi),
        pbo=pbo,
        latency_survived=latency_survived,
        positive_buckets=positive_buckets,
        config=config,
    )

    payoff = (
        f"{trade_stats.payoff_ratio:.2f}"
        if trade_stats.payoff_ratio is not None
        else "n/a"
    )
    mfe = (
        f"{trade_stats.mfe_capture:.2f}"
        if trade_stats.mfe_capture is not None
        else "n/a"
    )
    excludes = "YES" if trade_stats.ci_excludes_zero else "NO"
    trade_block = [
        "TRADE STATS",
        _kv("Trades (N):", str(trade_stats.n)),
        _kv("Expectancy:", _rfmt(trade_stats.expectancy_r)),
        _kv(
            "Expectancy 95% CI:",
            f"[{_rfmt(trade_stats.ci_lo)}, {_rfmt(trade_stats.ci_hi)}]"
            f" (excludes zero: {excludes})",
        ),
        _kv("Hit rate:", _pct1(trade_stats.hit_rate)),
        _kv("Payoff ratio:", payoff),
        _kv("Avg holding:", f"{trade_stats.avg_holding_hours:.1f} h"),
        _kv("MFE capture:", mfe),
    ]

    validation_block = [
        "VALIDATION",
        _kv(
            "Deflated Sharpe:",
            (f"{dsr:.3f}" if dsr is not None else "n/a") + f" (n_trials={n_trials})",
        ),
        _kv("PBO (CSCV):", f"{pbo:.3f}" if pbo is not None else "n/a"),
        _kv("Latency survival:", "PASS" if latency_survived else "FAIL"),
    ]

    sections = [
        [_RULE, f"TEARSHEET: {signal}", _RULE],
        render_headline_block(headline).rstrip("\n").split("\n"),
        trade_block,
        _render_cost_block(costs),
        _render_regime_block(regime_stats, positive_buckets),
        validation_block,
        _render_promotion_block(
            promotion,
            n_trades=trade_stats.n,
            ci_lo=trade_stats.ci_lo,
            pbo=pbo,
            latency_survived=latency_survived,
            positive_buckets=positive_buckets,
            config=config,
        ),
    ]
    return "\n\n".join("\n".join(block) for block in sections) + "\n"


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectancyRow:
    """One signal's leaderboard row (rejected signals included)."""

    signal: str
    geometric_monthly: float  # fraction: 0.021 = 2.1%
    months: int
    n_trades: int
    expectancy_r: float
    verdict: str  # VERDICT_PROMOTE or VERDICT_REJECT


def _num(value: float) -> str:
    """Compact number for markdown: ints exact, floats to 6 significant digits."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"numbers must be int or float, got {type(value).__name__}")
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _append_md(path: Path, header: str, body: str) -> None:
    """Append ``body`` to ``path``, writing ``header`` first on a new file."""
    prefix = "" if path.exists() else header
    with open(path, "a", encoding="utf-8") as f:
        f.write(prefix + body)


def append_expectancy_md(path: str | Path, rows: Sequence[ExpectancyRow]) -> None:
    """Append a full ranking of EVERY signal ever tested to EXPECTANCY.md.

    ``rows`` must contain one row per signal ever tested — rejected signals
    included; hiding failures is survivorship bias. Rows are ranked by
    headline geometric monthly return, descending (ties broken by signal
    name for determinism). Each call appends a complete new ranking section;
    the file is a history, never rewritten.

    Raises:
        ValueError: empty ``rows``.
    """
    if not rows:
        raise ValueError("append_expectancy_md requires at least one row")
    ranked = sorted(rows, key=lambda r: (-r.geometric_monthly, r.signal))
    lines = [
        f"## EXPECTANCY RANKING — {len(ranked)} signals",
        "",
        "| Rank | Signal | Geo monthly | Months | Trades | Expectancy (R) | Verdict |",
        "|-----:|:-------|------------:|-------:|-------:|---------------:|:--------|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row.signal} | {_pct1(row.geometric_monthly)} "
            f"| {row.months} | {row.n_trades} | {_rfmt(row.expectancy_r)} "
            f"| {row.verdict} |"
        )
    header = (
        "# EXPECTANCY\n\n"
        "Every signal ever tested, ranked by out-of-sample geometric monthly "
        "return. Failures included — hiding them is survivorship bias.\n\n"
    )
    _append_md(Path(path), header, "\n".join(lines) + "\n\n")


def append_rejected_md(
    path: str | Path,
    signal: str,
    named_failures: Sequence[str],
    numbers: Mapping[str, float],
) -> None:
    """Append one rejected signal to REJECTED.md with failures and numbers.

    ``named_failures`` are the promotion gate identifiers from
    :class:`PromotionResult.failures`; ``numbers`` maps metric names to the
    values that failed them (e.g. ``{"n_trades": 299, "pbo": 0.61}``).

    Raises:
        ValueError: empty ``named_failures`` (an unrejected signal does not
            belong in REJECTED.md), or a non-numeric value in ``numbers``.
    """
    if not named_failures:
        raise ValueError("append_rejected_md requires at least one named failure")
    lines = [
        f"## REJECTED: {signal}",
        "",
        "- failures: " + ", ".join(named_failures),
    ]
    if numbers:
        lines.append(
            "- numbers: " + ", ".join(f"{k}={_num(v)}" for k, v in numbers.items())
        )
    header = (
        "# REJECTED SIGNALS\n\n"
        "Signals that failed promotion, with the named gates they failed and "
        "the numbers that failed them.\n\n"
    )
    _append_md(Path(path), header, "\n".join(lines) + "\n\n")

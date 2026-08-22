"""Contract selection: a signal plus an option-chain snapshot -> one concrete
instrument, or a NAMED rejection. The selector never silently returns nothing.

Pipeline (order is part of the contract, so each test can make one gate the
binding one):

1. **Universe filter** — keep only rows of the required right (BUY signal ->
   calls, SELL signal -> puts) whose days-to-expiry fall inside the requested
   DTE ladder rung. Rows with non-finite quote/greek fields are not
   candidates. Empty universe -> ``EMPTY_RUNG``.
2. **Delta targeting** — pick the candidate whose ``|delta|`` is nearest the
   target (default 0.40 from :class:`SelectorConfig`; the further-OTM
   ``fat_tailed_target_delta`` applies ONLY when the caller passes
   ``fat_tailed=True``, i.e. the signal's measured payoff distribution earned
   it). Ties break toward liquidity: higher open interest, then cheaper mid,
   then lower strike (fully deterministic).
3. **Liquidity HARD gates** on the selected contract, each a named rejection:
   ``NO_QUOTE`` (mid <= 0), ``WIDE_SPREAD`` (``(ask-bid)/mid`` above
   ``max_rel_spread`` — tightened to ``max_rel_spread_0dte`` for rung 0),
   ``SIZE_BELOW_MIN`` (``quoted_size < min_size``), ``OI_BELOW_MIN``
   (``oi < min_oi``). Gates judge the delta-chosen contract; they never fall
   back to a worse-delta contract, because that would silently trade a
   different exposure than the signal asked for.
4. **Fill estimate** — the estimated debit comes from the cost model's
   marketable fill (ask-side crossing plus slippage), NEVER from mid. The
   cost model remains the single source of fill truth; a crossed book raises
   there, not here.
5. **Breakeven gate** — ``breakeven_move = (premium / |delta|) / spot`` using
   the estimated fill price as premium. The caller supplies the signal's
   historical median favorable excursion (fraction of spot); if
   ``excursion <= breakeven_move`` the trade cannot clear its own premium and
   is rejected as ``BREAKEVEN_UNREACHABLE``.

DTE ladder rungs partition days-to-expiry: 0 -> exactly 0, 1 -> exactly 1,
2 -> [2, 5), 5 -> [5, 7), 7 -> 7 or more ("7+").

:func:`shares_equivalent` converts an accepted choice into the
delta-equivalent share position for the wrapper-vs-shares comparison the
tearsheet runs.

Numeric parameters live in :class:`SelectorConfig` (defaults per spec),
injected at construction — never hardcoded in the selection logic.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edge.core.config import ExecutionConfig
from edge.core.events import SignalEvent, QuoteEvent, Side
from edge.execution.costmodel import (
    OPTION_CONTRACT_MULTIPLIER,
    CostModel,
)

#: Columns every option-chain snapshot frame must carry (CatalystBridge shape).
CHAIN_COLUMNS: Final[tuple[str, ...]] = (
    "strike", "expiry", "right", "bid", "ask", "delta", "oi", "quoted_size",
)

#: The DTE ladder: each rung restricts selection to one band of days-to-expiry.
#: 0 -> dte == 0, 1 -> dte == 1, 2 -> 2 <= dte < 5, 5 -> 5 <= dte < 7,
#: 7 -> dte >= 7 ("7+"). Together the rungs partition dte >= 0.
DTE_RUNGS: Final[tuple[int, ...]] = (0, 1, 2, 5, 7)

#: Delta distances are compared at this resolution (decimal places): chain
#: deltas carry at most a few decimals, so float representation noise (e.g.
#: |0.35-0.40| != |0.45-0.40| in the 17th digit) must not manufacture a
#: spurious "nearest" — equidistant contracts tie and the documented
#: tie-break decides.
_DELTA_DISTANCE_DECIMALS: Final[int] = 12

#: rung -> (inclusive lower bound, exclusive upper bound; None = unbounded).
_RUNG_BOUNDS: Final[dict[int, tuple[int, int | None]]] = {
    0: (0, 1),
    1: (1, 2),
    2: (2, 5),
    5: (5, 7),
    7: (7, None),
}


class RejectionReason(str, enum.Enum):
    """Named reasons a selection can fail. Every rejection carries one."""

    EMPTY_RUNG = "empty_rung"                        # no candidate in rung/right
    NO_QUOTE = "no_quote"                            # mid <= 0: unquotable
    WIDE_SPREAD = "wide_spread"                      # rel spread above cap
    SIZE_BELOW_MIN = "size_below_min"                # quoted_size < min_size
    OI_BELOW_MIN = "oi_below_min"                    # open interest < min_oi
    BREAKEVEN_UNREACHABLE = "breakeven_unreachable"  # excursion <= breakeven


class SelectorConfig(BaseModel):
    """Selection parameters. Frozen; defaults are the spec's defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: |delta| the selector targets by default.
    target_delta: float = Field(default=0.40, gt=0.0, le=1.0)
    #: Further-OTM target used ONLY when the caller passes ``fat_tailed=True``
    #: from the signal's measured payoff distribution.
    fat_tailed_target_delta: float = Field(default=0.25, gt=0.0, le=1.0)
    #: Hard cap on (ask - bid) / mid.
    max_rel_spread: float = Field(default=0.05, gt=0.0)
    #: Tighter cap applied when selecting in DTE rung 0.
    max_rel_spread_0dte: float = Field(default=0.03, gt=0.0)
    #: Minimum displayed quoted size (contracts).
    min_size: int = Field(default=10, ge=0)
    #: Minimum open interest (contracts).
    min_oi: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _check_relationships(self) -> "SelectorConfig":
        if self.fat_tailed_target_delta > self.target_delta:
            raise ValueError(
                "fat_tailed_target_delta must be <= target_delta "
                "(fat-tailed means FURTHER out of the money): "
                f"{self.fat_tailed_target_delta} > {self.target_delta}")
        if self.max_rel_spread_0dte > self.max_rel_spread:
            raise ValueError(
                "max_rel_spread_0dte must be <= max_rel_spread (0DTE is the "
                f"TIGHTER gate): {self.max_rel_spread_0dte} > {self.max_rel_spread}")
        return self


@dataclass(frozen=True)
class OptionLeg:
    """One option leg of a choice. ``delta`` is SIGNED (puts negative)."""

    underlying: str
    occ_symbol: str
    right: Literal["C", "P"]
    strike: float
    expiry: date
    side: Side          # the selector always BUYS premium (defined risk)
    qty: int            # contracts; sizing beyond the 1-lot is risk's job
    delta: float        # signed per-share delta from the chain snapshot
    bid: float
    ask: float


@dataclass(frozen=True)
class ContractChoice:
    """Outcome of one selection: a concrete instrument OR a named rejection.

    Exactly one of the two shapes:

    - accepted: ``legs`` non-empty, analytics populated,
      ``rejection_reason is None``;
    - rejected: ``legs`` empty, analytics ``None``, ``rejection_reason`` set
      and ``rejection_detail`` saying which value tripped which gate.

    ``est_fill_debit`` is the dollars of premium a 1-lot pays at the cost
    model's marketable (ask-side crossing) fill — commission excluded.
    ``max_loss`` is the defined-risk worst case for the long-premium 1-lot:
    the debit plus commission. ``breakeven_move`` is the fraction of spot the
    underlying must move favorably for the option to recoup its premium.
    """

    legs: tuple[OptionLeg, ...]
    est_fill_debit: float | None
    max_loss: float | None
    breakeven_move: float | None
    contract_multiplier: int
    rejection_reason: RejectionReason | None = None
    rejection_detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None

    @classmethod
    def reject(
        cls,
        reason: RejectionReason,
        detail: str,
        *,
        contract_multiplier: int,
        breakeven_move: float | None = None,
    ) -> "ContractChoice":
        """A rejection carrying its named reason — never a bare None."""
        return cls(
            legs=(),
            est_fill_debit=None,
            max_loss=None,
            breakeven_move=breakeven_move,
            contract_multiplier=contract_multiplier,
            rejection_reason=reason,
            rejection_detail=detail,
        )


@dataclass(frozen=True)
class SharesEquivalent:
    """Delta-equivalent share position; signed (negative = short shares)."""

    shares: float
    notional: float


@dataclass(frozen=True)
class _Candidate:
    """One chain row that survived the universe filter."""

    strike: float
    expiry: date
    dte: int
    bid: float
    ask: float
    delta: float
    oi: int
    quoted_size: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _as_date(value: object) -> date:
    """Normalize an expiry cell (date / datetime / Timestamp / str) to date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _normalize_right(value: object) -> str:
    """'C'/'P' from 'C', 'call', 'P', 'put' (first letter, upper-cased)."""
    text = str(value).strip().upper()
    if not text or text[0] not in ("C", "P"):
        raise ValueError(f"unrecognized option right {value!r}: expected C/P")
    return text[0]


def _occ_symbol(underlying: str, expiry: date, right: str, strike: float) -> str:
    """OCC-style option symbol, e.g. ``SPY260106C00500000``."""
    return f"{underlying}{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


class ContractSelector:
    """Turns (signal, chain snapshot) into a :class:`ContractChoice`.

    Fill economics are delegated to :class:`~edge.execution.costmodel.CostModel`
    built from the injected :class:`~edge.core.config.ExecutionConfig` —
    the estimated debit is an honest ask-side-crossing fill, never mid.
    """

    def __init__(
        self,
        execution: ExecutionConfig,
        config: SelectorConfig | None = None,
        *,
        contract_multiplier: int = OPTION_CONTRACT_MULTIPLIER,
    ) -> None:
        self._cfg = config if config is not None else SelectorConfig()
        self._mult = contract_multiplier
        self._cost = CostModel(execution, contract_multiplier=contract_multiplier)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        signal: SignalEvent,
        chain: pd.DataFrame,
        *,
        spot: float,
        dte_rung: int,
        median_favorable_excursion: float,
        fat_tailed: bool = False,
    ) -> ContractChoice:
        """Select the contract for ``signal`` from ``chain``, or reject by name.

        Args:
            signal: the strategy's directional opinion; BUY selects calls,
                SELL selects puts (the selector always BUYS the premium).
            chain: option-chain snapshot with :data:`CHAIN_COLUMNS`.
            spot: underlying spot at decision time (> 0).
            dte_rung: one of :data:`DTE_RUNGS`; selection is restricted to
                that rung of the ladder.
            median_favorable_excursion: the signal's historical median
                favorable excursion, as a FRACTION of spot; the breakeven
                gate rejects when it does not exceed the breakeven move.
            fat_tailed: True only when the signal's measured payoff
                distribution justifies going further OTM; switches the delta
                target to ``fat_tailed_target_delta``.
        """
        if spot <= 0 or not math.isfinite(spot):
            raise ValueError(f"spot must be a finite positive number, got {spot}")
        if dte_rung not in _RUNG_BOUNDS:
            raise ValueError(f"dte_rung must be one of {DTE_RUNGS}, got {dte_rung}")
        if median_favorable_excursion < 0 or not math.isfinite(median_favorable_excursion):
            raise ValueError(
                "median_favorable_excursion must be a finite fraction >= 0, "
                f"got {median_favorable_excursion}")
        missing = [c for c in CHAIN_COLUMNS if c not in chain.columns]
        if missing:
            raise ValueError(f"chain snapshot missing columns: {missing}")

        required_right = "C" if signal.side is Side.BUY else "P"
        decision_date = signal.ts.date()  # signal.ts is ET-normalized
        lo, hi = _RUNG_BOUNDS[dte_rung]

        candidates = self._universe(chain, required_right, decision_date, lo, hi)
        if not candidates:
            return ContractChoice.reject(
                RejectionReason.EMPTY_RUNG,
                f"no {required_right} contracts with dte in rung {dte_rung} "
                f"([{lo}, {'inf' if hi is None else hi})) as of {decision_date}",
                contract_multiplier=self._mult,
            )

        target = (
            self._cfg.fat_tailed_target_delta if fat_tailed else self._cfg.target_delta
        )
        best = min(
            candidates,
            key=lambda c: (
                round(abs(abs(c.delta) - target), _DELTA_DISTANCE_DECIMALS),
                -c.oi,
                c.mid,
                c.strike,
            ),
        )

        rejected = self._liquidity_gates(best, dte_rung)
        if rejected is not None:
            return rejected

        # Honest premium: the cost model's marketable fill for a 1-lot BUY —
        # mid + spread_fill_fraction * (ask - mid), plus slippage. Never mid.
        occ = _occ_symbol(signal.symbol, best.expiry, required_right, best.strike)
        quote = QuoteEvent(
            ts=signal.ts, symbol=occ, bid=best.bid, ask=best.ask,
            bid_size=best.quoted_size, ask_size=best.quoted_size,
        )
        fill = self._cost.marketable_fill(quote, Side.BUY, 1)
        premium = fill.price
        assert premium is not None  # marketable_fill FILLED always prices

        abs_delta = abs(best.delta)
        breakeven_move = (
            math.inf if abs_delta == 0.0 else (premium / abs_delta) / spot
        )
        if median_favorable_excursion <= breakeven_move:
            return ContractChoice.reject(
                RejectionReason.BREAKEVEN_UNREACHABLE,
                f"median favorable excursion {median_favorable_excursion:.6f} "
                f"<= breakeven move {breakeven_move:.6f} "
                f"(premium {premium:.4f} / |delta| {abs_delta:.4f} / spot {spot:.4f})",
                contract_multiplier=self._mult,
                breakeven_move=breakeven_move,
            )

        est_fill_debit = premium * self._mult
        leg = OptionLeg(
            underlying=signal.symbol,
            occ_symbol=occ,
            right=required_right,  # type: ignore[arg-type]
            strike=best.strike,
            expiry=best.expiry,
            side=Side.BUY,
            qty=1,
            delta=best.delta,
            bid=best.bid,
            ask=best.ask,
        )
        return ContractChoice(
            legs=(leg,),
            est_fill_debit=est_fill_debit,
            max_loss=est_fill_debit + fill.costs.commission,
            breakeven_move=breakeven_move,
            contract_multiplier=self._mult,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _universe(
        self,
        chain: pd.DataFrame,
        required_right: str,
        decision_date: date,
        lo: int,
        hi: int | None,
    ) -> list[_Candidate]:
        """Rows of the required right inside the rung, with finite fields."""
        out: list[_Candidate] = []
        for row in chain.itertuples(index=False):
            if _normalize_right(row.right) != required_right:
                continue
            expiry = _as_date(row.expiry)
            dte = (expiry - decision_date).days
            if dte < lo or (hi is not None and dte >= hi):
                continue
            bid, ask, delta = float(row.bid), float(row.ask), float(row.delta)
            strike = float(row.strike)
            oi, size = float(row.oi), float(row.quoted_size)
            if not all(map(math.isfinite, (bid, ask, delta, strike, oi, size))):
                continue  # an unquotable row is not a candidate
            out.append(_Candidate(
                strike=strike, expiry=expiry, dte=dte, bid=bid, ask=ask,
                delta=delta, oi=int(oi), quoted_size=int(size),
            ))
        return out

    def _liquidity_gates(
        self, best: _Candidate, dte_rung: int
    ) -> ContractChoice | None:
        """Apply the HARD liquidity gates; a rejection or None (passed)."""
        mid = best.mid
        if mid <= 0.0:
            return ContractChoice.reject(
                RejectionReason.NO_QUOTE,
                f"mid {mid:.4f} <= 0 (bid {best.bid:.4f} / ask {best.ask:.4f})",
                contract_multiplier=self._mult,
            )
        cap = (
            self._cfg.max_rel_spread_0dte if dte_rung == 0
            else self._cfg.max_rel_spread
        )
        rel_spread = (best.ask - best.bid) / mid
        if rel_spread > cap:
            return ContractChoice.reject(
                RejectionReason.WIDE_SPREAD,
                f"rel_spread {rel_spread:.4f} > max {cap:.4f} "
                f"(rung {dte_rung}{', 0DTE cap' if dte_rung == 0 else ''})",
                contract_multiplier=self._mult,
            )
        if best.quoted_size < self._cfg.min_size:
            return ContractChoice.reject(
                RejectionReason.SIZE_BELOW_MIN,
                f"quoted_size {best.quoted_size} < min_size {self._cfg.min_size}",
                contract_multiplier=self._mult,
            )
        if best.oi < self._cfg.min_oi:
            return ContractChoice.reject(
                RejectionReason.OI_BELOW_MIN,
                f"oi {best.oi} < min_oi {self._cfg.min_oi}",
                contract_multiplier=self._mult,
            )
        return None


def shares_equivalent(choice: ContractChoice, spot: float) -> SharesEquivalent:
    """Delta-equivalent share position for the wrapper-vs-shares comparison.

    ``shares = sum(sign(leg.side) * leg.delta * leg.qty * multiplier)`` —
    signed, so a long put maps to a SHORT share position. ``notional`` is
    ``shares * spot``, likewise signed. Rounding to whole shares (and any
    sizing beyond the choice's 1-lot legs) is the risk module's job.

    Raises ``ValueError`` on a rejected choice: a rejection has no share
    equivalent, and pretending otherwise would let a non-trade into the
    comparison.
    """
    if not choice.accepted:
        raise ValueError(
            "cannot compute shares_equivalent of a rejected choice "
            f"({choice.rejection_reason.value if choice.rejection_reason else '?'}: "
            f"{choice.rejection_detail})")
    if spot <= 0 or not math.isfinite(spot):
        raise ValueError(f"spot must be a finite positive number, got {spot}")
    shares = sum(
        (1.0 if leg.side is Side.BUY else -1.0)
        * leg.delta * leg.qty * choice.contract_multiplier
        for leg in choice.legs
    )
    return SharesEquivalent(shares=shares, notional=shares * spot)

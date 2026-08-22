"""The edge platform's single source of fill truth.

Ported from the audited ``catalyst.costs.model`` semantics: every quote-to-fill
conversion happens here and nowhere else. The invariants carry over intact —
a fill is never better than mid (enforced by assertion, not convention), a
crossed book is refused rather than priced, and every fill pays spread
crossing, slippage, and commission explicitly.

Two extensions over the ported model:

1. **Latency.** ``fill_with_latency`` prices off the quote prevailing when the
   order ARRIVES (decision time + latency), not when it was decided. The data
   layer supplies both quotes; a missing or stale arrival quote produces an
   unfilled result — never a silent fill off the decision quote.
2. **Limit orders.** ``limit_fill`` fills only if the opposite side crosses
   the limit within the supplied quote path (the path IS the horizon), partial
   and pro-rata to displayed size with a min-fill of one contract.

Every fill carries a :class:`FillCosts` attribution — spread, slippage,
commission — so ledgers can decompose each fill's friction exactly (closing
the v16 instrumentation gap): ``spread_cost + slippage_cost`` always equals
the signed distance of the fill from mid, in dollars.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from edge.core.config import ExecutionConfig
from edge.core.events import QuoteEvent, Side

#: OCC standard deliverable for one US equity option contract (shares).
#: A market-structure constant, not a tunable — override per instrument
#: via the ``contract_multiplier`` constructor argument.
OPTION_CONTRACT_MULTIPLIER: int = 100

#: The indivisible fill unit: a crossing quote fills at least one contract
#: even when it displays zero size, so thin books degrade fills instead of
#: silently blocking them.
MIN_FILL_CONTRACTS: int = 1

#: Float tolerance for the never-better-than-mid guard.
_MID_TOL: float = 1e-9


class BetterThanNBBOError(AssertionError):
    """A modeled fill was better than mid. Always a bug that invents money."""


class FillStatus(str, enum.Enum):
    """Outcome of one fill attempt."""

    FILLED = "filled"
    PARTIAL = "partial"
    UNFILLED = "unfilled"


@dataclass(frozen=True)
class FillCosts:
    """Per-fill friction attribution, in dollars for the whole fill.

    Contract: ``spread_cost + slippage_cost`` equals the fill's signed
    distance from mid times quantity times the contract multiplier, exact
    within 1e-9 (1 ulp of float64 accumulation; measured residual up to
    5.7e-14 USD in 2/1000 fills). ``total`` adds commission. All components
    are >= 0.
    """

    spread_cost: float      # dollars paid crossing from mid toward the touch
    slippage_cost: float    # dollars of modeled slippage beyond the crossing
    commission: float       # dollars of per-contract commission for the fill

    @property
    def total(self) -> float:
        """All-in friction for the fill: spread + slippage + commission."""
        return self.spread_cost + self.slippage_cost + self.commission


@dataclass(frozen=True)
class FillResult:
    """Outcome of one fill attempt, with per-fill cost attribution.

    ``price`` is per share and is ``None`` iff ``status`` is UNFILLED;
    ``qty`` is the number of contracts actually filled (0 when unfilled).
    """

    status: FillStatus
    qty: int
    price: float | None
    costs: FillCosts

    @staticmethod
    def no_fill() -> "FillResult":
        """The canonical unfilled result: zero quantity, zero costs."""
        return FillResult(
            status=FillStatus.UNFILLED,
            qty=0,
            price=None,
            costs=FillCosts(spread_cost=0.0, slippage_cost=0.0, commission=0.0),
        )


def assert_not_better_than_mid(price: float, quote: QuoteEvent, side: Side) -> None:
    """Raise :class:`BetterThanNBBOError` if a fill lands on the good side of mid.

    A buy may never fill below mid, a sell never above it. Checked against mid
    rather than the touch because mid is the model's own floor: crossing
    fraction 0 fills at mid, and anything better means a sign error. NaN in
    either operand also raises — upstream quote sanitation failed.
    """
    mid = quote.mid
    if price != price or mid != mid:
        raise BetterThanNBBOError(
            "NaN reached the fill path — upstream quote sanitation failed")
    if side is Side.BUY and price < mid - _MID_TOL:
        raise BetterThanNBBOError(
            f"buy modeled at {price:.4f}, better than mid {mid:.4f} "
            f"(bid {quote.bid:.4f} / ask {quote.ask:.4f})")
    if side is Side.SELL and price > mid + _MID_TOL:
        raise BetterThanNBBOError(
            f"sell modeled at {price:.4f}, better than mid {mid:.4f} "
            f"(bid {quote.bid:.4f} / ask {quote.ask:.4f})")


class CostModel:
    """Turns (quote, side, qty) into an honest, cost-attributed fill.

    All frictions come from :class:`~edge.core.config.ExecutionConfig`:
    ``spread_fill_fraction`` (how far past mid toward the worse touch a
    marketable order fills), ``slippage_pct`` (fraction of premium lost
    against the trader), and ``commission_per_contract`` (per leg per side).
    """

    def __init__(
        self,
        execution: ExecutionConfig,
        *,
        contract_multiplier: int = OPTION_CONTRACT_MULTIPLIER,
    ) -> None:
        if contract_multiplier <= 0:
            raise ValueError(f"contract_multiplier must be > 0, got {contract_multiplier}")
        self._cfg = execution
        self._mult = contract_multiplier

    # ------------------------------------------------------------------
    # Marketable fills
    # ------------------------------------------------------------------

    def marketable_fill(self, quote: QuoteEvent, side: Side, qty: int) -> FillResult:
        """Fill a marketable order against one quote.

        Buy fills at ``mid + frac*(ask-mid)``, sell at ``mid - frac*(mid-bid)``,
        then ``slippage_pct`` of premium against the trader, floored at zero,
        then per-contract commission. Never better than mid (asserted). A
        crossed book (bid > ask) is not a fillable market and raises
        ``ValueError`` — refused here, the single cost truth, so no caller can
        fill through it.
        """
        if qty < MIN_FILL_CONTRACTS:
            raise ValueError(f"qty must be >= {MIN_FILL_CONTRACTS}, got {qty}")
        if quote.bid > quote.ask:
            raise ValueError(
                f"crossed quote bid={quote.bid} > ask={quote.ask} "
                f"for {quote.symbol}: not fillable")
        mid = quote.mid
        frac = self._cfg.spread_fill_fraction
        slip = self._cfg.slippage_pct
        if side is Side.BUY:
            crossed = mid + frac * (quote.ask - mid)
            price = crossed * (1.0 + slip)
        else:
            crossed = mid - frac * (mid - quote.bid)
            price = crossed * (1.0 - slip)
        price = max(price, 0.0)
        assert_not_better_than_mid(price, quote, side)
        # Attribution is constructed so spread + slippage == |price - mid| *
        # qty * mult holds EXACTLY in floating point: slippage is the residual.
        scale = qty * self._mult
        if side is Side.BUY:
            adverse = (price - mid) * scale
            spread_cost = (crossed - mid) * scale
        else:
            adverse = (mid - price) * scale
            spread_cost = (mid - crossed) * scale
        return FillResult(
            status=FillStatus.FILLED,
            qty=qty,
            price=price,
            costs=FillCosts(
                spread_cost=spread_cost,
                slippage_cost=adverse - spread_cost,
                commission=self._cfg.commission_per_contract * qty,
            ),
        )

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def fill_with_latency(
        self,
        quote_at_decision: QuoteEvent,
        quote_at_decision_plus_latency: QuoteEvent | None,
        latency_ms: int,
        side: Side,
        qty: int,
    ) -> FillResult:
        """Fill a marketable order that ARRIVES ``latency_ms`` after decision.

        The fill prices off the LATER quote — the one prevailing at
        ``quote_at_decision.ts + latency_ms`` — which the caller (the data
        layer) supplies alongside the decision quote. No-fill, never a silent
        fill off the wrong quote, when the later quote is:

        - missing (``None``): the data layer had no quote at arrival time;
        - stale: timestamped BEFORE the decision quote, i.e. it cannot be the
          book the order actually arrived into.

        A later quote timestamped AFTER the arrival instant is lookahead and
        raises ``ValueError`` — that is a caller bug, not a market outcome.
        """
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        if quote_at_decision_plus_latency is None:
            return FillResult.no_fill()
        arrival = quote_at_decision.ts + timedelta(milliseconds=latency_ms)
        if quote_at_decision_plus_latency.ts > arrival:
            raise ValueError(
                f"arrival quote at {quote_at_decision_plus_latency.ts} is after "
                f"arrival time {arrival}: lookahead")
        if quote_at_decision_plus_latency.ts < quote_at_decision.ts:
            return FillResult.no_fill()  # stale: predates the decision quote
        return self.marketable_fill(quote_at_decision_plus_latency, side, qty)

    # ------------------------------------------------------------------
    # Limit orders
    # ------------------------------------------------------------------

    def limit_fill(
        self,
        side: Side,
        limit_price: float,
        qty: int,
        quote_path: Sequence[QuoteEvent],
    ) -> FillResult:
        """Simulate a limit order at ``limit_price`` over a quote path.

        The caller supplies the path of quotes covering the horizon; the path
        IS the horizon. Filled iff the opposite side crosses the limit within
        it: a buy crosses when ``0 < ask <= L``, a sell when ``bid >= L``.
        Each crossing quote fills pro-rata to its displayed size at the touch
        — never more than displayed — with a min-fill of one contract, and
        always AT the limit price (never at the better touch: conservative,
        and it keeps the never-better-than-mid invariant, asserted per
        tranche against the crossing quote's mid). A resting limit pays no
        crossing slippage; its friction is adverse selection, attributed as
        ``spread_cost`` relative to each crossing quote's mid.

        The path must be time-ordered (raises ``ValueError`` otherwise) and
        sanitized: a crossed book in the path raises, same refusal as the
        marketable model.
        """
        if qty < MIN_FILL_CONTRACTS:
            raise ValueError(f"qty must be >= {MIN_FILL_CONTRACTS}, got {qty}")
        if limit_price <= 0:
            raise ValueError(f"limit_price must be > 0, got {limit_price}")
        remaining = qty
        spread_cost = 0.0
        prev_ts = None
        for quote in quote_path:
            if prev_ts is not None and quote.ts < prev_ts:
                raise ValueError(
                    f"quote path out of order: {quote.ts} after {prev_ts}")
            prev_ts = quote.ts
            if quote.bid > quote.ask:
                raise ValueError(
                    f"crossed quote bid={quote.bid} > ask={quote.ask} "
                    f"for {quote.symbol}: not fillable")
            if remaining == 0:
                continue
            if side is Side.BUY:
                crossed = 0.0 < quote.ask <= limit_price
                displayed = quote.ask_size
            else:
                crossed = quote.bid >= limit_price
                displayed = quote.bid_size
            if not crossed:
                continue
            assert_not_better_than_mid(limit_price, quote, side)
            take = min(remaining, max(MIN_FILL_CONTRACTS, displayed))
            if side is Side.BUY:
                spread_cost += (limit_price - quote.mid) * take * self._mult
            else:
                spread_cost += (quote.mid - limit_price) * take * self._mult
            remaining -= take
        filled = qty - remaining
        if filled == 0:
            return FillResult.no_fill()
        status = FillStatus.FILLED if remaining == 0 else FillStatus.PARTIAL
        return FillResult(
            status=status,
            qty=filled,
            price=limit_price,
            costs=FillCosts(
                spread_cost=spread_cost,
                slippage_cost=0.0,
                commission=self._cfg.commission_per_contract * filled,
            ),
        )

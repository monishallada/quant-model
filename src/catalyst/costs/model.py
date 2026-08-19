"""The single source of cost truth.

Every fill in every mode is priced here. Nothing else in the codebase is
allowed to turn a quote into a fill price — not a strategy, not a runner, not
a bespoke backtest loop. This module exists because the audit found four of
nine campaigns pricing their own fills by reading ``.bid``/``.ask`` directly,
which silently omitted commissions and slippage and made their results
incomparable with the campaigns that paid them.

The model is conservative by construction, in three layers:

1. **Never better than NBBO.** A buy fills at or above the mid, a sell at or
   below it — enforced by an assertion, not by convention. A fill better than
   the touch is a bug that flatters results, and it is the single easiest way
   for a backtest to invent money.
2. **Spread crossing.** The mid is moved ``spread_fill_fraction`` toward the
   worse side (default 60% toward the ask on buys, the bid on sells). This is
   the stand-in for limit-order reality.
3. **Slippage and commission.** A percentage of premium plus a flat per-
   contract charge, then the per-contract commission for the active broker
   profile.

``ZeroCostModel`` is the diagnostic twin: identical routing, all frictions set
to zero. Running both is mandatory in the standard report — the gap between
them is what tells you whether a strategy has an edge or merely a cost
problem, and it is the single most informative number this project produces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from catalyst.core.config import CommissionsConfig, FillModelConfig
from catalyst.core.types import OptionContract, Side


class BetterThanNBBOError(AssertionError):
    """A modeled fill was better than the quoted touch. Always a bug."""


@dataclass(frozen=True)
class Fill:
    price: float          # per share, after spread crossing + slippage
    commission: float     # dollars, for the whole leg


class CostModel(ABC):
    """Turns (contract, side, contracts) into an honest fill."""

    name: str = "unnamed"

    @abstractmethod
    def leg_fill(self, contract: OptionContract, side: Side, contracts: int) -> Fill: ...

    def equity_fill(self, bid: float, ask: float, side: Side, shares: int) -> Fill:
        """Equity fill under the SAME honesty rules: worse-side crossing plus
        bps slippage, never better than mid. Default provided here so both the
        real and zero-cost models share one code path shape."""
        raise NotImplementedError

    @staticmethod
    def _assert_not_better_than_nbbo(price: float, contract: OptionContract, side: Side) -> None:
        """A buy may never fill below the mid, a sell never above it.

        Checked against the mid rather than the touch because the model's own
        floor is the mid: crossing fraction 0 fills at mid, and anything that
        lands on the *good* side of mid means a sign error somewhere.
        """
        mid = contract.mid
        tol = 1e-9
        if side is Side.BUY and price < mid - tol:
            raise BetterThanNBBOError(
                f"buy modeled at {price:.4f}, better than mid {mid:.4f} "
                f"(bid {contract.bid:.4f} / ask {contract.ask:.4f})")
        if side is Side.SELL and price > mid + tol:
            raise BetterThanNBBOError(
                f"sell modeled at {price:.4f}, better than mid {mid:.4f} "
                f"(bid {contract.bid:.4f} / ask {contract.ask:.4f})")


class NBBOCostModel(CostModel):
    """The production cost model: worse-side crossing + slippage + commission."""

    name = "real"

    def __init__(self, fill: FillModelConfig, commissions: CommissionsConfig) -> None:
        self._fill = fill
        self._commissions = commissions

    def leg_fill(self, contract: OptionContract, side: Side, contracts: int) -> Fill:
        mid = contract.mid
        frac = self._fill.spread_fill_fraction
        slip = self._fill.slippage_pct_of_premium
        flat = self._fill.slippage_per_contract
        if side is Side.BUY:
            price = mid + frac * (contract.ask - mid)
            price = price * (1.0 + slip) + flat
        else:
            price = mid - frac * (mid - contract.bid)
            price = price * (1.0 - slip) - flat
        price = max(price, 0.0)
        self._assert_not_better_than_nbbo(price, contract, side)
        return Fill(price=price, commission=self._commissions.per_contract * abs(contracts))

    #: Equity all-in slippage in bps per side on top of worse-side crossing —
    #: liquid US large caps; the measured figure this project has always used.
    EQUITY_SLIPPAGE_BPS = 3.5

    def equity_fill(self, bid: float, ask: float, side: Side, shares: int) -> Fill:
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError(f"unusable equity quote bid={bid} ask={ask}")
        mid = (bid + ask) / 2.0
        frac = self._fill.spread_fill_fraction
        slip = self.EQUITY_SLIPPAGE_BPS / 1e4
        if side is Side.BUY:
            price = (mid + frac * (ask - mid)) * (1.0 + slip)
            if price < mid:
                raise BetterThanNBBOError(f"equity buy at {price} better than mid {mid}")
        else:
            price = (mid - frac * (mid - bid)) * (1.0 - slip)
            if price > mid:
                raise BetterThanNBBOError(f"equity sell at {price} better than mid {mid}")
        return Fill(price=max(price, 0.0), commission=0.0)  # US equities: $0


class ZeroCostModel(CostModel):
    """Frictionless diagnostic twin — mid fills, no slippage, no commission.

    Not a trading model. Its only job is to answer "is this a signal problem or
    a cost problem?", which several campaigns needed and only some measured.
    """

    name = "zero"

    def leg_fill(self, contract: OptionContract, side: Side, contracts: int) -> Fill:
        return Fill(price=max(contract.mid, 0.0), commission=0.0)

    def equity_fill(self, bid: float, ask: float, side: Side, shares: int) -> Fill:
        return Fill(price=max((bid + ask) / 2.0, 0.0), commission=0.0)


def build(fill: FillModelConfig, commissions: CommissionsConfig, *, zero: bool = False) -> CostModel:
    """Factory used by the pipeline; the `zero` arm powers the diagnostic run."""
    return ZeroCostModel() if zero else NBBOCostModel(fill, commissions)

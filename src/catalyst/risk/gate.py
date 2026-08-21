"""Entry gate protocol: the single seam through which every entry is sized.

M1 ships ``FixedFractionalGate`` (sizing + cash sanity only) so the backtester
runs end-to-end. M3 replaces it with the authoritative ``RiskManager``
(three-tier allocation, portfolio caps, circuit breakers) behind the SAME
protocol — the backtester and live runner never change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from catalyst.core.types import AccountState, Position, ProposedTrade
from catalyst.risk.sizing import fixed_fractional_units


@dataclass(frozen=True)
class GateDecision:
    units: int  # 0 = rejected/skip
    reason: str


class EntryGate(Protocol):
    def size_entry(
        self,
        proposal: ProposedTrade,
        account: AccountState,
        positions: list[Position],
    ) -> GateDecision: ...


class FixedFractionalGate:
    """Minimal M1 gate: fixed-fractional sizing + cash sanity. No caps."""

    def size_entry(
        self,
        proposal: ProposedTrade,
        account: AccountState,
        positions: list[Position],
    ) -> GateDecision:
        mult = proposal.multiplier
        units = fixed_fractional_units(
            account.equity, proposal.per_trade_risk_fraction,
            proposal.unit_max_loss, multiplier=mult,
        )
        if units <= 0:
            return GateDecision(0, "risk budget below one unit")
        # Cash basis by structure (audit D-147/D-224): a DEBIT consumes its
        # cost; a CREDIT consumes COLLATERAL (max loss) — negative unit_cost
        # made every credit trivially "affordable" and the hardcoded 100x
        # mis-sized equity proposals.
        per_unit_cash = (proposal.unit_cost if proposal.unit_cost > 0
                         else proposal.unit_max_loss)
        if per_unit_cash is None or per_unit_cash != per_unit_cash:
            return GateDecision(0, "credit structure without max_loss")
        cost = per_unit_cash * units * mult
        if cost > account.cash:
            units = int(account.cash // (per_unit_cash * mult))
            if units <= 0:
                return GateDecision(0, "insufficient cash")
        return GateDecision(units, "ok")

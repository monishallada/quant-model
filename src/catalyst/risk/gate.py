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
        units = fixed_fractional_units(
            account.equity, proposal.per_trade_risk_fraction, proposal.unit_max_loss
        )
        if units <= 0:
            return GateDecision(0, "risk budget below one unit")
        cost = proposal.unit_cost * units * 100.0
        if cost > account.cash:
            units = int(account.cash // (proposal.unit_cost * 100.0))
            if units <= 0:
                return GateDecision(0, "insufficient cash")
        return GateDecision(units, "ok")

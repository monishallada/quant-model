"""Shared selection helpers for strategy engines."""

from __future__ import annotations

from datetime import date

from catalyst.core.types import OptionChain, OptionContract, OptionRight


def quotable(contracts: list[OptionContract]) -> list[OptionContract]:
    return [c for c in contracts if c.bid > 0 and c.ask > 0 and c.greeks is not None]


def nearest_delta(
    chain: OptionChain,
    expiry: date,
    right: OptionRight,
    target_abs_delta: float,
    delta_range: tuple[float, float] | None = None,
) -> OptionContract | None:
    """Contract of ``right``/``expiry`` with |delta| nearest the target,
    optionally required to fall inside ``delta_range`` (inclusive)."""
    candidates = quotable(chain.slice(expiry=expiry, right=right))
    if delta_range is not None:
        lo, hi = delta_range
        candidates = [c for c in candidates if lo <= abs(c.greeks.delta) <= hi]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(abs(c.greeks.delta) - target_abs_delta))


def atm_contract(
    chain: OptionChain, expiry: date, right: OptionRight
) -> OptionContract | None:
    candidates = quotable(chain.slice(expiry=expiry, right=right))
    if not candidates or chain.underlying_price <= 0:
        return None
    return min(candidates, key=lambda c: abs(c.key.strike - chain.underlying_price))


def first_expiry_on_or_after(available: list[date], d: date) -> date | None:
    after = sorted(e for e in available if e >= d)
    return after[0] if after else None

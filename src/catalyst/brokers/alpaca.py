"""AlpacaBroker — paper-ready implementation of the Broker interface.

Nothing above this file knows Alpaca exists. Strategy, risk, cost and exit code
see the same ``Broker`` methods they see in backtest, so moving a validated
strategy to paper is an injection change, not a code change.

Alpaca specifics that stay contained here:

- **Symbology.** Alpaca uses compact OSI (``AAPL240419C00150000``). The
  canonical internal identity is the structured ``OptionKey``; translation
  happens at this boundary and nowhere else.
- **Multi-leg.** Spreads go as a single ``mleg`` order with per-leg ratios, so
  the legs fill together or not at all. Legging into a spread is how a
  defined-risk position silently becomes an undefined-risk one.
- **Endpoint.** Paper and live differ only by base URL; the paper endpoint is
  the default and live must be requested explicitly.

Options level 3 is required for multi-leg spreads. ``preflight()`` checks the
account's actual options level and says so plainly rather than letting orders
fail one at a time later.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from catalyst.core.interfaces import Broker
from catalyst.core.symbology import parse_osi, to_alpaca_symbol
from catalyst.core.types import (
    EquityKey,
    InstrumentKey,
    AccountState,
    Direction,
    ExitRules,
    Order,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Position,
    PositionLeg,
    Side,
)

logger = logging.getLogger(__name__)

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


@dataclass
class AlpacaCredentials:
    key_id: str
    secret_key: str
    endpoint: str = PAPER_ENDPOINT

    @property
    def is_paper(self) -> bool:
        return self.endpoint.rstrip("/") == PAPER_ENDPOINT

    @classmethod
    def from_env(cls, *, paper: bool = True) -> AlpacaCredentials:
        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise AlpacaError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Put them in .env — "
                "they are never hardcoded and never committed.")
        return cls(key_id=key, secret_key=secret,
                   endpoint=PAPER_ENDPOINT if paper else LIVE_ENDPOINT)


class AlpacaBroker(Broker):
    """Live/paper order routing and account state."""

    def __init__(self, creds: AlpacaCredentials, timeout: float = 15.0) -> None:
        self._creds = creds
        self._client = httpx.Client(
            base_url=creds.endpoint.rstrip("/"),
            headers={"APCA-API-KEY-ID": creds.key_id,
                     "APCA-API-SECRET-KEY": creds.secret_key},
            timeout=timeout)

    @property
    def endpoint(self) -> str:
        return self._creds.endpoint

    @property
    def is_paper(self) -> bool:
        return self._creds.is_paper

    # -- plumbing ---------------------------------------------------------
    def _req(self, method: str, path: str, **kw: Any) -> Any:
        resp = self._client.request(method, path, **kw)
        if resp.status_code >= 400:
            raise AlpacaError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    # -- preflight --------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        """Account facts worth knowing BEFORE the confirmation prompt."""
        acct = self._req("GET", "/v2/account")
        level = int(acct.get("options_trading_level", 0) or 0)
        return {
            "account_number": acct.get("account_number"),
            "status": acct.get("status"),
            "equity": float(acct.get("equity", 0) or 0),
            "cash": float(acct.get("cash", 0) or 0),
            "buying_power": float(acct.get("buying_power", 0) or 0),
            "options_level": level,
            "multileg_ok": level >= 3,
            "endpoint": self._creds.endpoint,
            "is_paper": self.is_paper,
            "blocked": bool(acct.get("trading_blocked") or acct.get("account_blocked")),
        }

    # -- Broker interface -------------------------------------------------
    def place_order(self, order: Order) -> OrderResult:
        if not order.legs:
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message="no legs")
        # Limit-only discipline is enforced HERE too, not just upstream: this
        # adapter must never be able to construct a market order (audit D-197).
        if order.limit_price is None:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message="refused: no limit price (market orders are banned)")

        side_of = {Side.BUY: "buy", Side.SELL: "sell"}
        # OPEN vs CLOSE must reach the exchange: transmitting a close as an
        # open DOUBLES exposure at the exact moment the system decided to
        # reduce it (audit D-001 — the worst defect in the census).
        closing = order.intent is OrderIntent.CLOSE
        intent_of = {Side.BUY: ("buy_to_close" if closing else "buy_to_open"),
                     Side.SELL: ("sell_to_close" if closing else "sell_to_open")}

        def _symbol(key) -> str:
            return key.underlying if isinstance(key, EquityKey) else to_alpaca_symbol(key)

        if len(order.legs) == 1:
            leg = order.legs[0]
            payload: dict[str, Any] = {
                "symbol": _symbol(leg.key),
                "qty": str(leg.qty),
                "side": side_of[leg.side],
                "type": "limit",
                "time_in_force": "day",
            }
            if not isinstance(leg.key, EquityKey):
                payload["position_intent"] = intent_of[leg.side]
        else:
            # Unit decomposition by GCD — min()//truncation silently converted
            # ratio structures into different structures (audit D-003).
            from functools import reduce
            units = reduce(math.gcd, (l.qty for l in order.legs))
            if units <= 0 or any(l.qty % units for l in order.legs):
                return OrderResult(order_id="", status=OrderStatus.REJECTED,
                                   message="refused: leg quantities share no integral unit")
            payload = {
                "order_class": "mleg",
                "qty": str(units),
                "type": "limit",
                "time_in_force": "day",
                "legs": [{"symbol": _symbol(l.key),
                          "ratio_qty": str(l.qty // units),
                          "side": side_of[l.side],
                          "position_intent": intent_of[l.side]}
                         for l in order.legs],
            }
        # The SIGN of the net limit price is the credit/debit carrier in
        # Alpaca's mleg API (negative = credit). abs() flipped every credit
        # order into a debit order (audit D-004).
        payload["limit_price"] = f"{order.limit_price:.2f}"

        try:
            data = self._req("POST", "/v2/orders", json=payload)
        except AlpacaError as e:
            logger.error("order rejected: %s", e)
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message=str(e))

        raw_fill = data.get("filled_avg_price")
        return OrderResult(
            order_id=str(data.get("id", "")),
            status=_map_status(data.get("status", "")),
            filled_qty=int(float(data.get("filled_qty", 0) or 0)),
            # None means "not filled yet" and must SURVIVE — coalescing to 0.0
            # fabricated a $0 fill for every resting order (audit D-030).
            avg_fill_price=float(raw_fill) if raw_fill is not None else None,
            message=str(data.get("status", "")))

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult:
        data = self._req("PATCH", f"/v2/orders/{order_id}", json=changes)
        return OrderResult(order_id=str(data.get("id", order_id)),
                           status=_map_status(data.get("status", "")))

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancelation is ASYNC at Alpaca (202 = requested, not done). The
        order can still fill after this returns; callers must re-poll before
        treating the qty as free (audit D-104)."""
        self._req("DELETE", f"/v2/orders/{order_id}")
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED,
                           message="cancel requested (async — re-poll for terminal state)")

    def get_positions(self) -> list[Position]:
        """Broker-truth positions. Reconciliation reads this, never memory.

        Every row is represented: options as OptionKey legs, shares as
        EquityKey legs, and an unparseable option-shaped symbol is a loud
        error — silently dropping rows made adjusted positions invisible to
        the risk layer (audit D-031).
        """
        out: list[Position] = []
        for p in self._req("GET", "/v2/positions"):
            symbol = str(p.get("symbol", ""))
            asset_class = str(p.get("asset_class", "")).lower()
            if asset_class == "us_option" or _looks_like_occ(symbol):
                try:
                    key: InstrumentKey = parse_osi(symbol)
                except ValueError as e:
                    raise AlpacaError(
                        f"unparseable option position symbol {symbol!r}: {e} — "
                        "refusing to reconcile with an incomplete book") from e
                mult = 100.0
            else:
                key = EquityKey(underlying=symbol)
                mult = 1.0
            qty = int(float(p.get("qty", 0) or 0))
            if qty == 0:
                continue
            side = Side.BUY if qty > 0 else Side.SELL
            market_value = float(p.get("market_value", 0) or 0)
            # Position.current_value is PER-UNIT net per share; Alpaca's
            # market_value is TOTAL dollars (audit D-032).
            per_unit = market_value / (abs(qty) * mult) if qty else 0.0
            out.append(Position(
                position_id=symbol,
                legs=[PositionLeg(key=key, side=side, qty=1)],
                qty=abs(qty),
                entry_price=float(p.get("avg_entry_price", 0) or 0),
                entry_time=datetime.now(UTC),
                engine_tag="broker",
                direction=Direction.NEUTRAL,
                exit_rules=ExitRules(),
                current_value=per_unit))
        return out

    def get_account(self) -> AccountState:
        a = self._req("GET", "/v2/account")
        return AccountState(
            equity=float(a.get("equity", 0) or 0),
            cash=float(a.get("cash", 0) or 0),
            buying_power=float(a.get("buying_power", 0) or 0),
            timestamp=datetime.now(UTC))

    def close(self) -> None:
        self._client.close()


_STATUS_MAP = {
    "new": OrderStatus.ACCEPTED, "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.ACCEPTED, "accepted_for_bidding": OrderStatus.ACCEPTED,
    "held": OrderStatus.ACCEPTED, "suspended": OrderStatus.ACCEPTED,
    "stopped": OrderStatus.ACCEPTED, "calculated": OrderStatus.ACCEPTED,
    "pending_cancel": OrderStatus.ACCEPTED, "pending_replace": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED, "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED, "done_for_day": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


def _map_status(raw: str) -> OrderStatus:
    """Every documented Alpaca status mapped explicitly; an UNKNOWN status is
    logged loudly and treated as still-live (ACCEPTED) — assuming a live order
    is dead is the more dangerous direction (audit D-105)."""
    status = _STATUS_MAP.get(raw.lower())
    if status is None:
        logger.error("UNKNOWN Alpaca order status %r — treating as ACCEPTED; "
                     "verify manually", raw)
        return OrderStatus.ACCEPTED
    return status


def _looks_like_occ(symbol: str) -> bool:
    """Option-shaped: >=15 chars ending in C/P + 8 digits."""
    return (len(symbol) >= 15 and symbol[-9] in "CP" and symbol[-8:].isdigit())




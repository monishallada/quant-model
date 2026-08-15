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
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from catalyst.core.interfaces import Broker
from catalyst.core.symbology import parse_osi, to_alpaca_symbol
from catalyst.core.types import (
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

        side_of = {Side.BUY: "buy", Side.SELL: "sell"}
        if len(order.legs) == 1:
            leg = order.legs[0]
            payload: dict[str, Any] = {
                "symbol": to_alpaca_symbol(leg.key),
                "qty": str(leg.qty),
                "side": side_of[leg.side],
                "type": "limit" if order.limit_price is not None else "market",
                "time_in_force": "day",
            }
        else:
            payload = {
                "order_class": "mleg",
                "qty": str(min(l.qty for l in order.legs)),
                "type": "limit" if order.limit_price is not None else "market",
                "time_in_force": "day",
                "legs": [{"symbol": to_alpaca_symbol(l.key),
                          "ratio_qty": str(l.qty // max(min(x.qty for x in order.legs), 1)),
                          "side": side_of[l.side],
                          "position_intent": ("buy_to_open" if l.side is Side.BUY
                                              else "sell_to_open")}
                         for l in order.legs],
            }
        # Limit-only discipline: a market order is never constructed by the
        # execution layer, so this branch exists only for completeness.
        if order.limit_price is not None:
            payload["limit_price"] = f"{abs(order.limit_price):.2f}"

        try:
            data = self._req("POST", "/v2/orders", json=payload)
        except AlpacaError as e:
            logger.error("order rejected: %s", e)
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message=str(e))

        return OrderResult(
            order_id=str(data.get("id", "")),
            status=_map_status(data.get("status", "")),
            filled_qty=int(float(data.get("filled_qty", 0) or 0)),
            avg_fill_price=float(data.get("filled_avg_price") or 0.0),
            message=str(data.get("status", "")))

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult:
        data = self._req("PATCH", f"/v2/orders/{order_id}", json=changes)
        return OrderResult(order_id=str(data.get("id", order_id)),
                           status=_map_status(data.get("status", "")))

    def cancel_order(self, order_id: str) -> OrderResult:
        self._req("DELETE", f"/v2/orders/{order_id}")
        return OrderResult(order_id=order_id, status=OrderStatus.CANCELED)

    def get_positions(self) -> list[Position]:
        """Broker-truth positions. Reconciliation reads this, never memory."""
        out: list[Position] = []
        for p in self._req("GET", "/v2/positions"):
            symbol = p.get("symbol", "")
            key = _key_from_occ(symbol)
            if key is None:
                continue          # equity leg; option strategies ignore it
            qty = int(float(p.get("qty", 0) or 0))
            side = Side.BUY if qty > 0 else Side.SELL
            out.append(Position(
                position_id=symbol,
                legs=[PositionLeg(key=key, side=side, qty=1)],
                qty=abs(qty),
                entry_price=float(p.get("avg_entry_price", 0) or 0),
                entry_time=datetime.now(UTC),
                engine_tag="broker",
                direction=Direction.NEUTRAL,
                exit_rules=ExitRules(),
                current_value=float(p.get("market_value", 0) or 0)))
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


def _map_status(raw: str) -> OrderStatus:
    return {
        "new": OrderStatus.ACCEPTED, "accepted": OrderStatus.ACCEPTED,
        "pending_new": OrderStatus.ACCEPTED, "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED,
        "cancelled": OrderStatus.CANCELED, "expired": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
    }.get(raw.lower(), OrderStatus.ACCEPTED)


def _key_from_occ(symbol: str):
    """Compact OSI -> canonical OptionKey; None when it is not an option."""
    try:
        return parse_osi(symbol)
    except Exception:
        return None

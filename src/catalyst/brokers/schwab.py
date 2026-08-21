"""SchwabBroker — Charles Schwab / thinkorswim adapter.

The transition requirement for this project is that a strategy validated on
Alpaca paper moves to Schwab live as a **config change, not a code change**.
That holds because this class implements the same ``Broker`` interface:
strategy, risk, cost and execution code never learn which broker they are
talking to.

Everything Schwab-specific is contained here:

- **OAuth 2.0 with refresh.** Unlike Alpaca's static key/secret, Schwab issues
  a short-lived access token (~30 min) against a longer-lived refresh token
  (~7 days). ``_ensure_token`` refreshes ahead of expiry so a token does not
  die mid-session between the risk check and the order.
- **21-character space-padded OSI.** Schwab pads the root to 6 characters
  (``AAPL  240419C00150000``) where Alpaca uses the compact form. Both come
  from the same canonical ``OptionKey`` via ``core.symbology``.
- **Order payload.** Schwab nests legs under ``orderLegCollection`` with
  ``instruction`` verbs (BUY_TO_OPEN / SELL_TO_CLOSE...) rather than a flat
  side, and multi-leg spreads require ``orderStrategyType: SINGLE`` with
  ``complexOrderStrategyType`` set.

**Credentials are the only thing left to supply.** Set SCHWAB_CLIENT_ID,
SCHWAB_CLIENT_SECRET, SCHWAB_REFRESH_TOKEN and SCHWAB_ACCOUNT_HASH in .env and
this adapter is live-capable. Nothing else changes — see CONTRIBUTING.md.

No live call is made anywhere during construction; a Schwab account is only
contacted when a method is invoked, and live mode additionally requires the
deploy runner's typed confirmation gate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from catalyst.core.interfaces import Broker
from catalyst.core.symbology import parse_osi, to_schwab_symbol
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

TRADER_BASE = "https://api.schwabapi.com/trader/v1"
OAUTH_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
_REFRESH_MARGIN = timedelta(minutes=5)


class SchwabError(RuntimeError):
    pass


class SchwabCredentialsMissing(SchwabError):
    """Raised at connect time, never at import time.

    Keeping this out of construction is deliberate: the adapter must be
    importable, testable and inspectable with no credentials present, so the
    architecture can be verified long before real keys exist.
    """


@dataclass
class SchwabCredentials:
    """<<< INJECTION POINT — supply these when moving to live. >>>

    None of it is hardcoded; all four come from the environment via .env.
    """

    client_id: str
    client_secret: str
    refresh_token: str
    account_hash: str

    @classmethod
    def from_env(cls) -> SchwabCredentials:
        missing = [k for k in ("SCHWAB_CLIENT_ID", "SCHWAB_CLIENT_SECRET",
                               "SCHWAB_REFRESH_TOKEN", "SCHWAB_ACCOUNT_HASH")
                   if not os.environ.get(k)]
        if missing:
            raise SchwabCredentialsMissing(
                "Schwab credentials not configured. Add to .env: "
                + ", ".join(missing)
                + ". Until then, Schwab live is unreachable by construction.")
        return cls(
            client_id=os.environ["SCHWAB_CLIENT_ID"],
            client_secret=os.environ["SCHWAB_CLIENT_SECRET"],
            refresh_token=os.environ["SCHWAB_REFRESH_TOKEN"],
            account_hash=os.environ["SCHWAB_ACCOUNT_HASH"])


@dataclass
class _Token:
    access_token: str = ""
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def stale(self) -> bool:
        return (not self.access_token) or datetime.now(UTC) >= self.expires_at - _REFRESH_MARGIN


class SchwabBroker(Broker):
    """Real-money routing. Identical interface, different wire format."""

    def __init__(self, creds: SchwabCredentials, timeout: float = 15.0) -> None:
        self._creds = creds
        self._token = _Token()
        self._client = httpx.Client(base_url=TRADER_BASE, timeout=timeout)

    @property
    def is_paper(self) -> bool:
        return False        # Schwab has no paper endpoint in this integration

    # -- OAuth ------------------------------------------------------------
    def _ensure_token(self) -> str:
        """Refresh ahead of expiry so a token cannot die mid-order."""
        if not self._token.stale:
            return self._token.access_token
        resp = httpx.post(
            OAUTH_TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": self._creds.refresh_token},
            auth=(self._creds.client_id, self._creds.client_secret),
            timeout=15.0)
        if resp.status_code >= 400:
            raise SchwabError(
                f"OAuth refresh failed ({resp.status_code}). Schwab refresh tokens "
                f"expire every ~7 days and must be re-issued: {resp.text[:200]}")
        data = resp.json()
        self._token = _Token(
            access_token=data["access_token"],
            expires_at=datetime.now(UTC) + timedelta(seconds=int(data.get("expires_in", 1800))))
        logger.info("schwab token refreshed, expires %s", self._token.expires_at)
        return self._token.access_token

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = self._client.request(method, path, headers=headers, **kw)
        if resp.status_code >= 400:
            raise SchwabError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    # -- payload construction ---------------------------------------------
    @staticmethod
    def _instruction(side: Side, intent: OrderIntent) -> str:
        opening = intent is OrderIntent.OPEN
        if side is Side.BUY:
            return "BUY_TO_OPEN" if opening else "BUY_TO_CLOSE"
        return "SELL_TO_OPEN" if opening else "SELL_TO_CLOSE"

    def build_payload(self, order: Order) -> dict[str, Any]:
        """Schwab order JSON. Public so it can be unit-tested with no network."""
        def _instrument(key) -> dict[str, str]:
            # EquityKey legs are legal orders (SimulatedBroker supports them);
            # hardcoding OPTION crashed on them via to_schwab_symbol (D-033).
            if isinstance(key, EquityKey):
                return {"symbol": key.underlying, "assetType": "EQUITY"}
            return {"symbol": to_schwab_symbol(key), "assetType": "OPTION"}

        legs = [
            {
                "instruction": self._instruction(leg.side, order.intent),
                "quantity": leg.qty,
                "instrument": _instrument(leg.key),
            }
            for leg in order.legs
        ]
        payload: dict[str, Any] = {
            "orderType": "NET_DEBIT" if order.limit_price >= 0 else "NET_CREDIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "price": f"{abs(order.limit_price):.2f}",
            "orderLegCollection": legs,
        }
        if len(legs) > 1:
            # Legs must fill together; legging in turns a defined-risk spread
            # into naked exposure between fills.
            payload["complexOrderStrategyType"] = "CUSTOM"
        return payload

    # -- Broker interface -------------------------------------------------
    def place_order(self, order: Order) -> OrderResult:
        if not order.legs:
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message="no legs")
        acct = self._creds.account_hash
        # Token acquisition failures are AUTH problems, not order rejections;
        # conflating them hid expiring refresh tokens as "rejected" (D-107).
        token = self._ensure_token()
        try:
            resp = self._client.request(
                "POST", f"/accounts/{acct}/orders",
                headers={"Authorization": f"Bearer {token}"},
                json=self.build_payload(order))
        except SchwabError as e:
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message=str(e))
        if resp.status_code >= 400:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message=resp.text[:300])
        # Schwab returns the new order id in the Location header, not the body.
        order_id = resp.headers.get("Location", "").rstrip("/").rsplit("/", 1)[-1]
        if not order_id:
            # An accepted order we cannot address is unmanageable: no cancel,
            # no status poll. Surface it loudly instead of ACCEPTED-with-empty-id
            # (audit D-108).
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message="order POST accepted but no Location header; "
                                       "order may be live — verify at broker NOW")
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED)

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult:
        """Schwab replace requires a COMPLETE replacement order and returns the
        NEW order id in the Location header; a raw delta is invalid and the old
        id is dead after success (audit D-034). ``changes`` must therefore be a
        full order payload (e.g. from build_payload)."""
        required = {"orderType", "orderLegCollection"}
        if not required.issubset(changes):
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message="replace requires a complete order payload "
                                       f"(missing {sorted(required - set(changes))})")
        acct = self._creds.account_hash
        try:
            resp = self._client.request(
                "PUT", f"/accounts/{acct}/orders/{order_id}",
                headers={"Authorization": f"Bearer {self._ensure_token()}"},
                json=changes)
        except SchwabError as e:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=str(e))
        if resp.status_code >= 400:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message=resp.text[:300])
        new_id = resp.headers.get("Location", "").rstrip("/").rsplit("/", 1)[-1]
        if not new_id:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED,
                               message="replace accepted but no new order id in "
                                       "Location header — verify manually")
        return OrderResult(order_id=new_id, status=OrderStatus.ACCEPTED,
                           message=f"replaced {order_id} -> {new_id}")

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel is asynchronous: report requested, not done."""
        acct = self._creds.account_hash
        self._req("DELETE", f"/accounts/{acct}/orders/{order_id}")
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED,
                           message="cancel requested (async — re-poll for terminal state)")

    def get_positions(self) -> list[Position]:
        acct = self._creds.account_hash
        data = self._req("GET", f"/accounts/{acct}", params={"fields": "positions"})
        out: list[Position] = []
        for p in data.get("securitiesAccount", {}).get("positions", []):
            inst = p.get("instrument", {})
            asset_type = inst.get("assetType", "")
            symbol = str(inst.get("symbol", ""))
            if asset_type == "OPTION":
                try:
                    key: InstrumentKey = parse_osi(symbol)
                except ValueError as e:
                    # Silently dropping a position hides real exposure from the
                    # risk layer (audit D-035): refuse to reconcile partially.
                    raise SchwabError(
                        f"unparseable option position symbol {symbol!r}: {e} — "
                        "refusing to reconcile with an incomplete book") from e
                mult = 100.0
            elif asset_type in ("EQUITY", "COLLECTIVE_INVESTMENT"):
                key = EquityKey(underlying=symbol)
                mult = 1.0
            else:
                continue  # cash sweeps / fixed income are not tradeable state here
            long_qty = float(p.get("longQuantity", 0) or 0)
            short_qty = float(p.get("shortQuantity", 0) or 0)
            qty = int(long_qty - short_qty)
            if qty == 0:
                continue
            market_value = float(p.get("marketValue", 0) or 0)
            out.append(Position(
                position_id=symbol,
                legs=[PositionLeg(key=key,
                                  side=Side.BUY if qty > 0 else Side.SELL, qty=1)],
                qty=abs(qty),
                entry_price=float(p.get("averagePrice", 0) or 0),
                entry_time=datetime.now(UTC),
                engine_tag="broker",
                direction=Direction.NEUTRAL,
                exit_rules=ExitRules(),
                # marketValue is TOTAL dollars; current_value is per-unit
                # per-share (audit D-036).
                current_value=market_value / (abs(qty) * mult)))
        return out

    def get_account(self) -> AccountState:
        acct = self._creds.account_hash
        data = self._req("GET", f"/accounts/{acct}")
        bal = data.get("securitiesAccount", {}).get("currentBalances", {})

        def _first(*names: str) -> float:
            # Margin and cash accounts expose different field names; missing
            # ones must not silently coalesce to $0 (audit D-110).
            for n in names:
                v = bal.get(n)
                if v is not None:
                    return float(v)
            raise SchwabError(
                f"none of {names} present in currentBalances {sorted(bal)} — "
                "refusing to report a fabricated $0 balance")

        return AccountState(
            equity=_first("liquidationValue"),
            cash=_first("cashBalance", "cashAvailableForTrading", "totalCash"),
            buying_power=_first("buyingPower", "cashAvailableForTrading",
                                "availableFunds"),
            timestamp=datetime.now(UTC))

    def preflight(self) -> dict[str, Any]:
        a = self.get_account()
        return {"account_number": f"***{self._creds.account_hash[-4:]}",
                "equity": a.equity, "cash": a.cash, "buying_power": a.buying_power,
                "endpoint": TRADER_BASE, "is_paper": False}

    def close(self) -> None:
        self._client.close()

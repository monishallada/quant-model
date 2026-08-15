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
        legs = [
            {
                "instruction": self._instruction(leg.side, order.intent),
                "quantity": leg.qty,
                "instrument": {"symbol": to_schwab_symbol(leg.key), "assetType": "OPTION"},
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
        try:
            resp = self._client.request(
                "POST", f"/accounts/{acct}/orders",
                headers={"Authorization": f"Bearer {self._ensure_token()}"},
                json=self.build_payload(order))
        except SchwabError as e:
            return OrderResult(order_id="", status=OrderStatus.REJECTED, message=str(e))
        if resp.status_code >= 400:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message=resp.text[:300])
        # Schwab returns the new order id in the Location header, not the body.
        order_id = resp.headers.get("Location", "").rstrip("/").rsplit("/", 1)[-1]
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED)

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult:
        acct = self._creds.account_hash
        self._req("PUT", f"/accounts/{acct}/orders/{order_id}", json=changes)
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED)

    def cancel_order(self, order_id: str) -> OrderResult:
        acct = self._creds.account_hash
        self._req("DELETE", f"/accounts/{acct}/orders/{order_id}")
        return OrderResult(order_id=order_id, status=OrderStatus.CANCELED)

    def get_positions(self) -> list[Position]:
        acct = self._creds.account_hash
        data = self._req("GET", f"/accounts/{acct}", params={"fields": "positions"})
        out: list[Position] = []
        for p in data.get("securitiesAccount", {}).get("positions", []):
            inst = p.get("instrument", {})
            if inst.get("assetType") != "OPTION":
                continue
            try:
                key = parse_osi(inst.get("symbol", ""))
            except Exception:
                continue
            long_qty = float(p.get("longQuantity", 0) or 0)
            short_qty = float(p.get("shortQuantity", 0) or 0)
            qty = int(long_qty - short_qty)
            out.append(Position(
                position_id=inst.get("symbol", ""),
                legs=[PositionLeg(key=key,
                                  side=Side.BUY if qty > 0 else Side.SELL, qty=1)],
                qty=abs(qty),
                entry_price=float(p.get("averagePrice", 0) or 0),
                entry_time=datetime.now(UTC),
                engine_tag="broker",
                direction=Direction.NEUTRAL,
                exit_rules=ExitRules(),
                current_value=float(p.get("marketValue", 0) or 0)))
        return out

    def get_account(self) -> AccountState:
        acct = self._creds.account_hash
        data = self._req("GET", f"/accounts/{acct}")
        bal = data.get("securitiesAccount", {}).get("currentBalances", {})
        return AccountState(
            equity=float(bal.get("liquidationValue", 0) or 0),
            cash=float(bal.get("cashBalance", 0) or 0),
            buying_power=float(bal.get("buyingPower", 0) or 0),
            timestamp=datetime.now(UTC))

    def preflight(self) -> dict[str, Any]:
        a = self.get_account()
        return {"account_number": f"***{self._creds.account_hash[-4:]}",
                "equity": a.equity, "cash": a.cash, "buying_power": a.buying_power,
                "endpoint": TRADER_BASE, "is_paper": False}

    def close(self) -> None:
        self._client.close()

"""The Nautilus-side strategy that submits and manages orders.

It receives a plan of *intent* — instrument, side, entry timestamp, hold length
— and nothing else. Every price-forming decision belongs to Nautilus: the
matching engine picks the fill, the fee model charges commission, the portfolio
tracks the position and the account computes equity.

That split is the whole point. Sharing intent is unavoidable (otherwise the two
engines would be testing different strategies); sharing anything downstream of
intent would make the second backtest a restatement of the first.
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


class MirrorConfig(StrategyConfig, frozen=True):
    plan: list          # [{instrument_id, entry_ts, side, hold_days, ...}]
    contracts: int = 1  # fixed lot; Nautilus applies its own account checks


class MirrorStrategy(Strategy):
    """Submits each planned entry when its timestamp arrives, then flattens
    after the intended hold."""

    def __init__(self, config: MirrorConfig) -> None:
        super().__init__(config)
        self._plan = sorted(config.plan, key=lambda p: p["entry_ts"])
        self._contracts = config.contracts
        self._pending = list(self._plan)
        self._open: dict[str, dict] = {}

    def on_start(self) -> None:
        for iid in {p["instrument_id"] for p in self._plan}:
            self.subscribe_quote_ticks(InstrumentId.from_str(iid))

    def on_quote_tick(self, tick: QuoteTick) -> None:
        now = tick.ts_event
        iid = str(tick.instrument_id)

        # --- entries whose time has come ---------------------------------
        still_pending = []
        for p in self._pending:
            if p["entry_ts"] <= now and p["instrument_id"] == iid:
                self._enter(p, now)
            elif p["entry_ts"] > now:
                still_pending.append(p)
            else:
                still_pending.append(p)   # different instrument; keep waiting
        self._pending = still_pending

        # --- exits: flatten once the intended hold has elapsed ------------
        state = self._open.get(iid)
        if state and now >= state["exit_ts"]:
            self._flatten(iid)

    def _enter(self, plan: dict, now: int) -> None:
        iid = plan["instrument_id"]
        if iid in self._open:
            return
        instrument_id = InstrumentId.from_str(iid)
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY if plan["side"] == "BUY" else OrderSide.SELL,
            quantity=Quantity.from_int(max(int(plan.get("units", self._contracts)), 1)),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        # Exit on the real session timestamp supplied by the caller — a
        # calendar-day approximation drifts across weekends and holidays and
        # would land the exit on a date with no quote to trigger it.
        self._open[iid] = {"exit_ts": plan.get(
            "exit_ts", now + plan["hold_days"] * 86_400 * 1_000_000_000)}

    def _flatten(self, iid: str) -> None:
        instrument_id = InstrumentId.from_str(iid)
        for position in self.cache.positions_open(instrument_id=instrument_id):
            self.close_position(position)
        self._open.pop(iid, None)

    def on_stop(self) -> None:
        for position in self.cache.positions_open():
            self.close_position(position)

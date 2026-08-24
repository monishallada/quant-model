"""EdgeDataLoader: the ONLY sanctioned data-access gateway for research code.

DESIGN PRINCIPLE — in-process Python cannot be sealed against code with
filesystem write access; the loader therefore enforces exactly what is
enforceable. (1) ACCIDENTAL lockbox violations are impossible through this
public API: every :meth:`EdgeDataLoader.load` re-reads the wall from
``config/edge.yaml`` at call time (no cached date, no env var, no kwarg
bypass), clamps a range that spills past the wall to the day before it
(with a logged warning), and raises
:class:`~edge.validation.lockbox.LockboxViolation` on a request that starts
inside the lockbox. (2) DELIBERATE bypass is burn-on-forge: the loader
unlocks on CONTENT, never class identity — it compares ``key.nonce`` to the
nonce inside ``LOCKBOX_SPENT.json`` at the repo root on every call. There
is no ``isinstance`` gate to spoof; minting a working key without
:meth:`~edge.validation.lockbox.LockboxKey.issue` requires writing that
sentinel yourself, which permanently spends the lockbox and leaves a hole
in the append-only TRIALS registry (see
:func:`~edge.validation.lockbox.detect_tamper`). (3) Residual risks are
documented in :mod:`edge.validation.lockbox`, not hidden — chiefly: a
filesystem-writing adversary CAN unlock by forging the sentinel; the design
guarantees the forge is conspicuous, permanent, and evidenced in two
places, not that it is impossible.

RESIDUAL — the ``repo_root``/``config_path`` arguments are TRUST INPUTS.
They exist as test seams; the wall, sentinel, and registry are all resolved
relative to whatever root the caller passes. Research code that constructs
``EdgeDataLoader(backend, repo_root="/elsewhere")`` (or ``config_path=...``)
pointed at an attacker-planted config + forged ``LOCKBOX_SPENT.json`` unlocks
against THAT root without burning the real repo's lockbox or leaving a record
in the real TRIALS registry — the same authority any in-process code holds by
supplying its own ``config/edge.yaml``. The act is not silent: a foreign root
in research code is diff-visible and glaring in review, and the composed
audit still bites — a legitimate OOS evaluation is required to carry a
``LOCKBOX ISSUANCE:`` record in the REAL registry, so results claiming an
opened lockbox with no matching real-registry issuance are self-convicting.
Default construction (real repo root) has no such seam; this is the
in-process/monkeypatch residual class, owned by code review.

The wall is an **America/New_York boundary** [fixes F6]: every requested
``start``/``end`` — plain date or datetime — is normalized to an ET trading
date via :func:`~edge.validation.lockbox.et_trading_date` BEFORE the gate.
Naive datetimes are rejected (never guessed); plain dates are ET calendar
dates by definition, so no westward local clock can slide a request into
the first ET lockbox session.

Signals, features, and research NEVER touch a backend, feed, or catalyst
provider directly — they construct an :class:`EdgeDataLoader` and call
:meth:`~EdgeDataLoader.load`. Fetches delegate to a pluggable
:class:`DataBackend` so catalyst providers can be adapted behind the same
wall later; :class:`SyntheticBackend` serves tests deterministic frames
with no network and no terminal.

POINT-IN-TIME KINDS — the feed adapters under ``edge/data/feeds/`` are
served through the same gate under the kind labels in
:data:`POINT_IN_TIME_KINDS` (``short_ratio``, ``vol_indices``, ``rates``,
``cot``, ``insider``, ``earnings``), routed by :class:`FeedsBackend` (or
the sanctioned research-side constructor
:meth:`EdgeDataLoader.with_feeds`). Every such frame carries ``asof_date``
(the data's own date) and ``available_at`` (tz-aware America/New_York
instant a live trader could FIRST have seen the row). For lockbox purposes
a locked loader enforces the wall on BOTH columns row-by-row — a row
survives only when its ``asof_date`` AND its ET ``available_at`` both
predate the wall. **``available_at`` governs**: it is the publication-time
truth the lockbox exists to protect, and it is the column that embargoes
rows whose asof_date predates the wall but whose publication does not
(e.g. a Friday FINRA file first available the next Tuesday pre-open).
``asof_date`` is enforced as well, defense-in-depth, so a lockbox-day data
row can never leak through an anti-conservative stamp (e.g. the earnings
features' deliberate ``asof 00:00 ET`` stamp).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple, Protocol, runtime_checkable

import pandas as pd

from edge.core.events import MARKET_TZ
from edge.validation.lockbox import LockboxWall, et_trading_date, read_sentinel

_log = logging.getLogger(__name__)

#: Kinds of market data a backend can serve. The last six are the
#: point-in-time feed kinds (see module docstring): their frames carry
#: ``asof_date`` + ``available_at`` and are row-clamped at the wall.
DataKind = Literal[
    "bars",
    "quotes",
    "trades",
    "short_ratio",
    "vol_indices",
    "rates",
    "cot",
    "insider",
    "earnings",
]

#: Kinds whose frames follow the point-in-time convention and receive the
#: row-level (asof_date AND available_at) lockbox clamp when locked.
POINT_IN_TIME_KINDS: Final[frozenset[str]] = frozenset(
    {"short_ratio", "vol_indices", "rates", "cot", "insider", "earnings", "micro"}
)

#: Columns every point-in-time frame must carry.
PIT_COLUMNS: Final[tuple[str, str]] = ("asof_date", "available_at")

#: Point-in-time kinds keyed per symbol; the rest are market-wide and the
#: ``symbol`` argument is ignored for them (pass a placeholder like "MKT").
PER_SYMBOL_PIT_KINDS: Final[frozenset[str]] = frozenset(
    {"short_ratio", "insider", "earnings", "micro"}
)

#: Pass as ``symbol`` for a per-symbol point-in-time kind to request the
#: whole cross-section (no symbol filter).
ALL_SYMBOLS: Final[str] = "*"


def _require_pit_columns(frame: pd.DataFrame, kind: str) -> None:
    """Loudly reject a point-in-time frame missing its convention columns."""
    missing = [column for column in PIT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"point-in-time kind {kind!r} requires columns {list(PIT_COLUMNS)}; "
            f"backend frame is missing {missing}"
        )


@runtime_checkable
class DataBackend(Protocol):
    """Raw fetcher the loader delegates to AFTER the lockbox gate has run.

    Contract: return one row per available observation for ``symbol`` over
    the inclusive ``[start, end]`` date range. Backends do no embargo logic —
    the loader is the only gate — and must never be imported by research
    code directly.
    """

    def fetch(self, symbol: str, start: date, end: date, kind: DataKind) -> pd.DataFrame:
        """Fetch ``kind`` data for ``symbol`` over inclusive ``[start, end]``."""
        ...


class FetchCall(NamedTuple):
    """One recorded backend fetch, for test assertions on gated ranges."""

    symbol: str
    start: date
    end: date
    kind: DataKind


class SyntheticBackend:
    """Deterministic in-memory backend for tests: no network, no terminal.

    Emits one row per calendar day in the requested range with a linearly
    drifting close (``base_price + day_offset``) and constant volume, and
    records every fetch in :attr:`calls` so tests can assert exactly what
    range survived the lockbox gate. All parameters are caller-supplied —
    nothing numeric is hardcoded here.
    """

    def __init__(self, base_price: float, daily_volume: int) -> None:
        self._base_price = base_price
        self._daily_volume = daily_volume
        self.calls: list[FetchCall] = []

    def fetch(self, symbol: str, start: date, end: date, kind: DataKind) -> pd.DataFrame:
        """Return synthetic daily rows for inclusive ``[start, end]``."""
        self.calls.append(FetchCall(symbol=symbol, start=start, end=end, kind=kind))
        days = pd.date_range(start=start, end=end, freq="D")
        return pd.DataFrame(
            {
                "symbol": symbol,
                "kind": kind,
                "close": [self._base_price + float(i) for i in range(len(days))],
                "volume": self._daily_volume,
            },
            index=days,
        )


def _trim_asof(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Trim a point-in-time frame to ``asof_date`` within inclusive [start, end]."""
    if frame.empty:
        return frame.reset_index(drop=True)
    asof = pd.to_datetime(frame["asof_date"])
    mask = (asof >= pd.Timestamp(start)) & (asof <= pd.Timestamp(end))
    return frame[mask].reset_index(drop=True)


def _filter_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Restrict a per-symbol frame to ``symbol`` (:data:`ALL_SYMBOLS` = keep all)."""
    if symbol == ALL_SYMBOLS or "symbol" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame[frame["symbol"] == symbol].reset_index(drop=True)


class FeedsBackend:
    """Routes the point-in-time feed kinds to their backend adapters.

    Backend machinery — research code must never import this name (the AST
    gateway bans backend names from the loader module); research gets a
    loader over this backend via :meth:`EdgeDataLoader.with_feeds`. Routing:

    ==============  ======================================================
    kind            adapter
    ==============  ======================================================
    short_ratio     ``feeds.finra_short.FinraShortDaily.features``
    vol_indices     ``feeds.cboe_indices.CboeIndices.features``
    rates           ``feeds.rates.RatesFeed.features``
    cot             ``feeds.cot.CotTff.features``
    insider         ``feeds.edgar_form4.insider_features`` over the cached
                    filing parquets under ``data_cache/edge/edgar_form4/``
    earnings        ``feeds.earnings.EarningsFeed.features``
    ==============  ======================================================

    ``symbol`` filters the per-symbol kinds (:data:`PER_SYMBOL_PIT_KINDS`;
    :data:`ALL_SYMBOLS` requests the whole cross-section — for
    ``short_ratio`` the cross-sectional z is always computed market-wide
    BEFORE the symbol filter); the market-wide kinds ignore it. Feed
    modules import lazily so ``import edge.data.loader`` stays light for
    research code. This backend does NO embargo logic — the loader is the
    only gate — and it never touches the network: adapters parse the
    on-disk archives, cache-first.
    """

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self._root = (
            Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        )

    def fetch(self, symbol: str, start: date, end: date, kind: DataKind) -> pd.DataFrame:
        """Serve one point-in-time kind for inclusive ``[start, end]``."""
        if kind == "short_ratio":
            return self._short_ratio(symbol, start, end)
        if kind == "vol_indices":
            return self._vol_indices(start, end)
        if kind == "rates":
            return self._rates(start, end)
        if kind == "cot":
            return self._cot(start, end)
        if kind == "insider":
            return self._insider(symbol, start, end)
        if kind == "earnings":
            return self._earnings(symbol, start, end)
        if kind == "micro":
            return self._micro(symbol, start, end)
        raise ValueError(
            f"FeedsBackend serves only the point-in-time kinds "
            f"{sorted(POINT_IN_TIME_KINDS)}, got {kind!r}"
        )

    # -- per-kind routes (lazy feed imports; data/ is gateway-exempt) -------

    def _edge_cache(self, source: str) -> Path:
        return self._root / "data_cache" / "edge" / source

    def _micro(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Per-minute trade-tape microstructure from the derived cache.

        The cache is precomputed per (symbol, session) from staged SIP trade
        prints; each row's ``available_at`` is its own minute CLOSE, because
        the features are computed from prints fully observed by that instant.
        Consumers receive it under bar-visibility semantics — a minute-t row
        is actionable on the t+1 decision, exactly like a BarEvent.
        """
        root = self._edge_cache("micro")
        if not root.exists():
            return pd.DataFrame(columns=["symbol", "ts", "asof_date", "available_at"])
        symbols = (
            [d.name.split("=", 1)[1] for d in sorted(root.glob("symbol=*"))]
            if symbol == ALL_SYMBOLS
            else [symbol]
        )
        frames = []
        for sym in symbols:
            for path in sorted((root / f"symbol={sym}").glob("date=*.parquet")):
                day = date.fromisoformat(path.name.split("=", 1)[1].removesuffix(".parquet"))
                if day < start or day > end:
                    continue
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=["symbol", "ts", "asof_date", "available_at"])
        out = pd.concat(frames, ignore_index=True)
        out["available_at"] = pd.to_datetime(out["ts"])
        out["asof_date"] = out["available_at"].dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_localize(None)
        return out.sort_values("available_at", kind="stable").reset_index(drop=True)

    def _short_ratio(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.finra_short import FinraShortDaily

        feed = FinraShortDaily(
            raw_dir=self._root / "data_cache" / "edge" / "raw" / "finra_short_daily",
            cache_root=self._edge_cache("finra_short"),
        )
        return _filter_symbol(feed.features(start, end, symbols=None), symbol)

    def _vol_indices(self, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.cboe_indices import CboeIndices

        return _trim_asof(CboeIndices.from_repo(self._root).features(), start, end)

    def _rates(self, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.rates import RatesFeed

        return _trim_asof(RatesFeed.from_repo(self._root).features(), start, end)

    def _cot(self, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.cot import CotTff

        feed = CotTff(
            raw_dir=self._root / "data_cache" / "edge" / "raw" / "cot",
            cache_root=self._edge_cache("cot"),
        )
        return _trim_asof(feed.features(), start, end)

    def _insider(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.edgar_form4 import TRANSACTION_COLUMNS, insider_features

        filings_dir = self._edge_cache("edgar_form4") / "filings"
        frames = (
            [pd.read_parquet(path) for path in sorted(filings_dir.glob("accession=*.parquet"))]
            if filings_dir.is_dir()
            else []
        )
        transactions = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=TRANSACTION_COLUMNS)
        )
        transactions = _filter_symbol(transactions, symbol)
        # Features over the FULL history first (trailing windows need their
        # warmup), trimmed to the requested range afterwards.
        return _trim_asof(insider_features(transactions), start, end)

    def _earnings(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from edge.data.feeds.earnings import EarningsFeed

        feed = EarningsFeed.from_repo(self._root)
        symbols = None if symbol == ALL_SYMBOLS else [symbol]
        return feed.features(pd.Timestamp(start), pd.Timestamp(end), symbols=symbols)


class ResearchBackend:
    """Routes every research kind to the backend that owns it.

    Market data (``bars``/``quotes`` and the per-expiry options EOD kinds)
    goes to the CatalystBridge over the catalyst parquet archive; the slow
    point-in-time feed kinds go to :class:`FeedsBackend`. Backend machinery:
    research code never names it — it gets a loader over this backend from
    :meth:`EdgeDataLoader.with_research`.

    The bridge is CACHE-ONLY by construction here: a research replay must
    never trigger a ThetaData terminal fetch as a side effect.
    """

    def __init__(self, repo_root: str | Path | None = None) -> None:
        from edge.data.backends import build_catalyst_bridge

        self._feeds = FeedsBackend(repo_root=repo_root)
        self._bridge = build_catalyst_bridge(
            repo_root=str(repo_root) if repo_root is not None else None,
            allow_fetch=False,
            missing_ok=True,
        )

    def fetch(self, symbol: str, start: date, end: date, kind: DataKind) -> pd.DataFrame:
        """Feed kinds to the feeds backend; everything else to the bridge."""
        if kind in POINT_IN_TIME_KINDS:
            return self._feeds.fetch(symbol, start, end, kind)
        return self._bridge.fetch(symbol, start, end, kind)


class EdgeDataLoader:
    """The sole data gateway; locked by default, unlocked only by content.

    Contract: :meth:`load` normalizes both range bounds to ET trading dates,
    then runs the lockbox gate, reading the wall from ``config/edge.yaml``
    at call time. There is no unlock kwarg or env var. The ONLY unlock is a
    key whose ``nonce`` equals the nonce inside ``LOCKBOX_SPENT.json`` at
    the repo root — checked by CONTENT on every call, so a key whose
    sentinel has vanished (or never matched) leaves the loader locked. No
    ``isinstance`` check exists: any nonce holder with the wrong nonce is
    simply a locked loader, and obtaining the right nonce outside
    ``LockboxKey.issue()`` means forging the sentinel — a permanent,
    registry-visible act (see :mod:`edge.validation.lockbox`).
    """

    def __init__(
        self,
        backend: DataBackend,
        *,
        key: object | None = None,
        config_path: str | Path | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        if key is not None and not isinstance(getattr(key, "nonce", None), str):
            raise TypeError(
                "key must be a nonce holder from LockboxKey.issue() "
                f"(an object with a str 'nonce'), got {type(key).__name__}"
            )
        self._backend = backend
        self._key = key
        self._repo_root = repo_root
        self._wall = LockboxWall(config_path=config_path, repo_root=repo_root)

    @classmethod
    def with_research(
        cls,
        repo_root: str | Path | None = None,
        *,
        key: object | None = None,
        config_path: str | Path | None = None,
    ) -> EdgeDataLoader:
        """Sanctioned research-side constructor for FULL campaign runs.

        Serves the market-data kinds (``bars``, ``quotes``, and the
        per-expiry options EOD kinds) alongside the point-in-time feeds,
        through one lockbox-walled gateway. The options bridge behind it is
        cache-only: a replay can never trigger a terminal fetch.
        """
        return cls(
            ResearchBackend(repo_root=repo_root),
            key=key,
            config_path=config_path,
            repo_root=repo_root,
        )

    @classmethod
    def with_feeds(
        cls,
        repo_root: str | Path | None = None,
        *,
        key: object | None = None,
        config_path: str | Path | None = None,
    ) -> EdgeDataLoader:
        """Sanctioned research-side constructor for the point-in-time kinds.

        Returns a loader over a :class:`FeedsBackend` rooted at
        ``repo_root`` (default: this repo), without the caller ever
        importing a backend name — the AST gateway bans those from
        research code. The wall, sentinel, and registry resolve against
        the same root.
        """
        return cls(
            FeedsBackend(repo_root=repo_root),
            key=key,
            config_path=config_path,
            repo_root=repo_root,
        )

    def _unlocked(self) -> bool:
        """True only when the key's nonce matches the spend sentinel RIGHT NOW.

        Re-reads ``LOCKBOX_SPENT.json`` on every call: a deleted sentinel
        re-locks immediately, and a corrupt sentinel raises
        :class:`~edge.validation.lockbox.LockboxTampered` rather than being
        quietly ignored. The comparison invokes ``str.__eq__`` on the
        sentinel's (always plain-str, JSON-decoded) nonce so a key whose
        ``nonce`` is a str subclass with an overridden ``__eq__`` cannot
        answer the check — content decides, not object behavior.
        """
        if self._key is None:
            return False
        nonce = getattr(self._key, "nonce", None)
        if not isinstance(nonce, str):
            return False
        sentinel = read_sentinel(self._repo_root)
        return sentinel is not None and str.__eq__(sentinel["nonce"], nonce) is True

    def load(
        self,
        symbol: str,
        start: date | datetime,
        end: date | datetime,
        kind: DataKind,
    ) -> pd.DataFrame:
        """Fetch ``kind`` data for ``symbol`` over inclusive ``[start, end]``.

        Both bounds are normalized to America/New_York trading dates first:
        tz-aware datetimes are converted to ET, plain dates are ET calendar
        dates, and naive datetimes raise ``ValueError`` — the wall never
        guesses a timezone. Locked (the default): raises
        :class:`~edge.validation.lockbox.LockboxViolation` if the ET start
        is at/after the lockbox wall, else clamps the ET end to the day
        before the wall (logging a warning when it truncates). The wall is
        re-read from config on every call. Unlocked (nonce matches the
        sentinel): passes the normalized range through.

        Point-in-time kinds (:data:`POINT_IN_TIME_KINDS`) additionally
        receive a row-level wall when locked: only rows whose ``asof_date``
        AND ET ``available_at`` both predate the wall survive.
        ``available_at`` governs (publication time is what the lockbox
        protects); ``asof_date`` is enforced too as defense-in-depth. See
        the module docstring. Their frames must carry both columns
        (``ValueError`` otherwise), locked or not.

        Raises:
            ValueError: ``start`` is after ``end`` (in ET), a bound is a
                naive datetime, or a point-in-time frame violates its
                column/timezone contract.
        """
        start_et = et_trading_date(start)
        end_et = et_trading_date(end)
        if start_et > end_et:
            raise ValueError(
                f"empty range: start {start_et} is after end {end_et} (ET trading dates)"
            )
        if self._unlocked():
            frame = self._backend.fetch(symbol, start_et, end_et, kind)
            if kind in POINT_IN_TIME_KINDS:
                _require_pit_columns(frame, kind)
            return frame
        gated_end = self._wall.gate(start_et, end_et)
        if gated_end != end_et:
            _log.warning(
                "lockbox clamp: %s %s request %s..%s truncated to end %s (wall %s)",
                symbol, kind, start_et, end_et, gated_end, self._wall.lockbox_start(),
            )
        frame = self._backend.fetch(symbol, start_et, gated_end, kind)
        if kind in POINT_IN_TIME_KINDS:
            frame = self._clamp_point_in_time(frame, kind, symbol)
        return frame

    def _clamp_point_in_time(self, frame: pd.DataFrame, kind: str, symbol: str) -> pd.DataFrame:
        """Row-level lockbox wall for point-in-time kinds (locked readers).

        A row survives only when BOTH stamps predate the wall (an ET
        midnight boundary):

        * ``available_at`` — the GOVERNING column: the instant a live
          trader could first have seen the row. It embargoes rows whose
          data date predates the wall but whose publication does not
          (e.g. a Friday FINRA file stamped available the next Tuesday
          pre-open is invisible to a Saturday-walled reader).
        * ``asof_date`` — enforced as well, defense-in-depth: a row whose
          own data date is at/after the wall stays embargoed even under an
          anti-conservative stamp (e.g. the earnings features' deliberate
          ``asof 00:00 ET`` stamp), and even if a backend ignored the
          range clamp.

        Rows with NaT in either column are dropped — unknowable is treated
        as unavailable (conservative). Raises ``ValueError`` on a frame
        missing the point-in-time columns or carrying a naive
        ``available_at``: an unstampable frame cannot be gated.
        """
        _require_pit_columns(frame, kind)
        if frame.empty:
            return frame.reset_index(drop=True)
        wall_ts = pd.Timestamp(self._wall.lockbox_start())  # ET midnight boundary
        available = pd.to_datetime(frame["available_at"])
        if available.dt.tz is None:
            raise ValueError(
                f"point-in-time kind {kind!r} requires a tz-aware 'available_at' "
                "(America/New_York); got naive timestamps"
            )
        asof_ok = pd.to_datetime(frame["asof_date"]) < wall_ts  # NaT compares False
        avail_ok = available.dt.tz_convert(MARKET_TZ).dt.tz_localize(None) < wall_ts
        keep = asof_ok & avail_ok
        dropped = int(len(frame) - keep.sum())
        if dropped:
            _log.warning(
                "lockbox point-in-time clamp: %s %s dropped %d/%d rows whose "
                "asof_date or available_at is at/after the wall %s "
                "(available_at governs)",
                symbol, kind, dropped, len(frame), wall_ts.date(),
            )
        return frame[keep].reset_index(drop=True)

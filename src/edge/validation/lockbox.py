"""Lockbox wall: the out-of-sample data embargo and its one-time unlock key.

DESIGN PRINCIPLE — in-process Python cannot be sealed against code with
filesystem write access. No class marker, private sentinel object, or
``isinstance`` gate survives an adversary who can write files or
monkeypatch modules, so this layer does not pretend to. The enforceable
invariants are:

(1) ACCIDENTAL lockbox violations are impossible through any public API:
    the locked loader clamps or raises on every wall-crossing request, the
    wall is re-read from ``config/edge.yaml`` on every check, there is no
    unlock kwarg or env var, and directly constructing a
    :class:`LockboxKey` yields an inert nonce holder that unlocks nothing.
(2) DELIBERATE bypass requires a conspicuous act that itself permanently
    marks the violation (burn-on-forge) and leaves evidence in more than
    one place. Unlocking requires content, not class identity: the key's
    nonce must match the nonce inside ``LOCKBOX_SPENT.json`` at the REPO
    ROOT. Forging a key therefore requires WRITING that sentinel yourself,
    which (a) permanently spends the lockbox — :meth:`LockboxKey.issue`
    refuses forever once the file exists — and (b) shows up in the
    append-only TRIALS registry diff, because a legitimate issuance
    records a matching ``LOCKBOX ISSUANCE:`` trial and a forged sentinel
    has none. Forge = burn.
(3) Residual risks are DOCUMENTED here, not hidden:
    * An adversary can write ``LOCKBOX_SPENT.json`` by hand and mint a
      key with its nonce; the loader will unlock. The act is loud: the
      sentinel sits at the repo root (visible, greppable), the lockbox is
      spent forever, and :func:`detect_tamper` flags the missing registry
      issuance record.
    * Deleting ``LOCKBOX_SPENT.json`` re-locks every key but leaves the
      orphaned issuance record in the append-only registry;
      :func:`detect_tamper` raises on that state and :meth:`LockboxKey.issue`
      runs it first, so deletion no longer silently reopens issuance.
    * Erasing ALL local evidence requires deleting the sentinel AND the
      registry line AND its digest-chained sidecar (``TRIALS.jsonl`` edits
      raise ``RegistryTampered``; ``rebuild_index()`` re-baselines only as
      an explicit, visible act). After that, version control / backup
      diffs of the repo root are the remaining detection path.
    * In-process monkeypatching of this module's functions cannot be
      prevented in Python; it is conspicuous in code review, which is the
      layer that owns it.
    * The ``repo_root``/``config_path`` arguments threaded through
      :class:`LockboxWall`, :func:`read_sentinel`, :meth:`LockboxKey.issue`,
      and :func:`detect_tamper` are TRUST INPUTS (test seams). An adversary
      who points a loader or an ``issue`` call at a root they control — with a
      hand-planted ``config/edge.yaml`` and a forged sentinel — unlocks
      against that root, and per-root the burn-on-forge invariant still holds
      there (the forged sentinel spends that root and :func:`detect_tamper`
      flags its missing issuance). What it does NOT do is burn the REAL repo
      or trace into the REAL registry; that residual is identical in power to
      supplying an attacker ``config/edge.yaml`` and is owned by code review.
      The cross-check that survives: a genuine OOS open must leave a
      ``LOCKBOX ISSUANCE:`` record in the real registry, so real results with
      no matching real-registry issuance are self-convicting.

The lockbox window opens at ``data.lockbox_start`` in ``config/edge.yaml``
(2026-02-22): research code must never touch market data on or after that
date until the lockbox is deliberately opened. The wall is an
**America/New_York boundary** — ``lockbox_start`` is an ET calendar date,
and callers' datetimes are converted to ET before any comparison (see
:func:`et_trading_date`); naive datetimes are rejected outright.
:class:`LockboxWall` re-reads the boundary from the YAML on EVERY check —
never cached — so no stale constant, env var, or monkeypatched snapshot can
widen the window.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from edge.core.config import load_config
from edge.research.registry import TrialRegistry

#: Name of the one-time spend sentinel at the REPO ROOT (not results/):
#: visible in every directory listing, greppable, and diffed by VCS.
SENTINEL_FILENAME: Final[str] = "LOCKBOX_SPENT.json"

#: Hypothesis prefix of the issuance record appended to the TRIALS registry.
ISSUANCE_PREFIX: Final[str] = "LOCKBOX ISSUANCE: "

#: The lockbox wall's timezone anchor: ET calendar dates define the boundary.
ET_ZONE: Final[ZoneInfo] = ZoneInfo("America/New_York")


class LockboxViolation(RuntimeError):
    """A locked reader requested market data at/after the lockbox wall."""


class LockboxAlreadySpent(RuntimeError):
    """The spend sentinel exists; the lockbox can never be issued again."""


class LockboxTampered(RuntimeError):
    """Sentinel and registry disagree: the lockbox audit trail was tampered with."""


def _repo_root() -> Path:
    """Repository root: three parents above this package directory."""
    return Path(__file__).resolve().parents[3]


def sentinel_path(repo_root: str | Path | None = None) -> Path:
    """Path of the spend sentinel: ``<repo_root>/LOCKBOX_SPENT.json``."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    return root / SENTINEL_FILENAME


def read_sentinel(repo_root: str | Path | None = None) -> dict[str, Any] | None:
    """The parsed spend sentinel, or None when the lockbox is unspent.

    Raises:
        LockboxTampered: the sentinel file exists but is not a JSON object
            with a string ``nonce`` — a half-forged or corrupted spend
            marker is an alarm state, never silently ignored.
    """
    path = sentinel_path(repo_root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockboxTampered(
            f"{path} exists but is not valid JSON; the spend sentinel was "
            "corrupted or hand-edited"
        ) from exc
    if not isinstance(record, dict) or not isinstance(record.get("nonce"), str):
        raise LockboxTampered(
            f"{path} exists but carries no string 'nonce'; the spend sentinel "
            "was corrupted or hand-edited"
        )
    return record


def et_trading_date(value: date | datetime) -> date:
    """Normalize a requested boundary to an America/New_York trading date.

    The lockbox wall is an ET boundary [fixes F6: a bare date compared in a
    westward local timezone served the first ET lockbox session]. Rules:

    * tz-aware ``datetime`` — converted to ET, then its calendar date;
    * naive ``datetime`` — REJECTED (``ValueError``): guessing a timezone
      is exactly the bug this anchor exists to kill;
    * plain ``date`` — taken as an ET calendar date as-is.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"naive datetime {value!r} rejected: the lockbox wall is an "
                "America/New_York boundary; pass a tz-aware datetime or a "
                "plain date (interpreted as an ET calendar date)"
            )
        return value.astimezone(ET_ZONE).date()
    if isinstance(value, date):
        return value
    raise TypeError(
        f"expected date or tz-aware datetime, got {type(value).__name__}"
    )


class LockboxWall:
    """Call-time reader and enforcer of the lockbox boundary.

    Contract: :meth:`lockbox_start` performs a fresh ``config/edge.yaml``
    read on every call — the wall is never cached on an instance, so a check
    always reflects the config on disk at the moment it runs. The wall is an
    ET calendar date; callers normalize datetimes via :func:`et_trading_date`
    BEFORE gating.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self._config_path = config_path
        self._repo_root = repo_root

    def lockbox_start(self) -> date:
        """First embargoed ET day, read from ``config/edge.yaml`` right now."""
        return load_config(
            config_path=self._config_path, repo_root=self._repo_root
        ).data.lockbox_start

    def last_allowed_day(self) -> date:
        """Latest ET date a locked reader may see: the day before the wall."""
        return self.lockbox_start() - timedelta(days=1)

    def gate(self, start: date, end: date) -> date:
        """Admit ET range ``[start, end]`` for a locked reader; return the usable end.

        Raises :class:`LockboxViolation` if ``start`` is at/after the wall
        (an explicit lockbox read); otherwise returns ``end`` clamped to
        :meth:`last_allowed_day`. Reads the wall from config at call time.
        """
        wall = self.lockbox_start()
        if start >= wall:
            raise LockboxViolation(
                f"requested {start}..{end} starts inside the lockbox (wall {wall}, "
                "an America/New_York boundary); locked research code may not "
                "read at/after the wall"
            )
        return min(end, wall - timedelta(days=1))


@dataclass(frozen=True)
class LockboxKey:
    """Bearer nonce for the one-time unlock; content is checked, never class.

    A key is JUST a nonce holder [fixes F1]. There is no construction guard,
    no private marker, and no ``isinstance`` check anywhere — all three were
    forgeable (``__new__`` skipped ``__init__``, the marker was importable,
    a subclass overrode ``__init__``) while granting authority by identity.
    Authority now lives ONLY in content: the loader unlocks when
    ``key.nonce`` equals the nonce inside ``LOCKBOX_SPENT.json`` at the repo
    root. Constructing a key directly is legal and harmless — without the
    spent-file's 128-bit nonce it unlocks nothing, and obtaining that nonce
    means either calling :meth:`issue` (the audited path) or writing the
    sentinel yourself, which permanently spends the lockbox and leaves a
    registry-diff hole (forge = burn; see the module docstring).
    """

    nonce: str

    @classmethod
    def issue(cls, purpose: str, *, repo_root: str | Path | None = None) -> LockboxKey:
        """Issue THE key: cross-anchor in the registry, then write the sentinel.

        Effects, in order:

        1. Refuse if ``LOCKBOX_SPENT.json`` already exists (spent forever)
           or if :func:`detect_tamper` finds an orphaned issuance record —
           deleting the sentinel does NOT reopen issuance [fixes F4], it
           trips the tamper check instead.
        2. Generate a random 128-bit nonce.
        3. Append a ``LOCKBOX ISSUANCE: <purpose>`` trial (nonce included)
           to the append-only TRIALS registry, so the digest-chained
           registry cross-anchors the spend.
        4. Exclusively create ``LOCKBOX_SPENT.json`` at the repo root with
           ``{nonce, purpose, issued_ts}``; a concurrent second issue loses
           the ``"x"`` race and raises, leaving its registry record as
           honest evidence of the attempt.

        Raises:
            LockboxAlreadySpent: the sentinel exists — the lockbox was
                opened (or forged, which spends it just the same) by this
                or any earlier process.
            LockboxTampered: the sentinel is missing but the registry holds
                an issuance record — someone deleted the spend marker.
        """
        root = Path(repo_root) if repo_root is not None else _repo_root()
        path = root / SENTINEL_FILENAME
        if path.exists():
            raise LockboxAlreadySpent(
                f"lockbox already spent: {path} exists; the lockbox opens exactly once"
            )
        detect_tamper(repo_root=root)
        nonce = secrets.token_hex(16)  # 128 bits
        issued_ts = datetime.now(UTC).isoformat()
        TrialRegistry(root=root).record_trial(
            hypothesis=f"{ISSUANCE_PREFIX}{purpose}",
            config={"nonce": nonce, "purpose": purpose, "sentinel": SENTINEL_FILENAME},
            result={"issued_ts": issued_ts},
        )
        try:
            # "x" = exclusive create: a concurrent second issue loses the race.
            with open(path, "x", encoding="utf-8") as f:
                json.dump(
                    {"nonce": nonce, "purpose": purpose, "issued_ts": issued_ts},
                    f,
                    indent=2,
                )
        except FileExistsError as exc:
            raise LockboxAlreadySpent(
                f"lockbox already spent: {path} exists; the lockbox opens exactly once"
            ) from exc
        return cls(nonce=nonce)


def detect_tamper(repo_root: str | Path | None = None) -> None:
    """Cross-check the spend sentinel against the TRIALS registry; raise on lies.

    The two anchors must agree. Tamper states raised as
    :class:`LockboxTampered`:

    * registry holds ``LOCKBOX ISSUANCE:`` record(s) but the sentinel is
      missing — someone deleted ``LOCKBOX_SPENT.json`` after a real issue
      [the F4 detection path];
    * the sentinel exists but the registry holds no issuance record — the
      sentinel was forged rather than issued;
    * the sentinel's nonce matches no recorded issuance nonce — the
      sentinel was replaced after issuance;
    * the sentinel exists but is corrupt (via :func:`read_sentinel`).

    A full registry scan runs underneath, so ``RegistryTampered`` from a
    doctored ``TRIALS.jsonl`` propagates too. Passing states: never spent
    (no sentinel, no issuance) or legitimately spent (sentinel nonce matches
    a recorded issuance).
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    sentinel = read_sentinel(root)  # raises LockboxTampered if corrupt
    issuances = list(
        TrialRegistry(root=root).query(
            lambda record: record.hypothesis.startswith(ISSUANCE_PREFIX)
        )
    )
    if sentinel is None:
        if issuances:
            ids = [record.trial_id for record in issuances]
            raise LockboxTampered(
                f"{sentinel_path(root)} is missing but the append-only TRIALS "
                f"registry holds issuance record(s) {ids}: the spend sentinel "
                "was deleted after the lockbox was opened"
            )
        return
    if not issuances:
        raise LockboxTampered(
            f"{sentinel_path(root)} exists but the TRIALS registry holds no "
            f"'{ISSUANCE_PREFIX}' record: the sentinel was forged, not issued"
        )
    recorded_nonces = {record.config.get("nonce") for record in issuances}
    if sentinel["nonce"] not in recorded_nonces:
        raise LockboxTampered(
            f"{sentinel_path(root)} carries a nonce matching no recorded "
            "issuance: the sentinel was replaced after the lockbox was opened"
        )


#: Import-time snapshot of the wall for reference/docs (2026-02-22, from
#: config/edge.yaml). Enforcement NEVER uses it: every check re-reads config
#: through :class:`LockboxWall` at call time.
LOCKBOX_START: Final[date] = LockboxWall().lockbox_start()

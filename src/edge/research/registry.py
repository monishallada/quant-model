"""Append-only, tamper-evident trial registry (``TRIALS.jsonl``).

Every experiment run is a *trial*: one JSON line in ``TRIALS.jsonl`` at the
repo root, carrying a tz-aware ``ts``, a sha256 ``config_hash`` of the
canonical-JSON config, and a ``trial_id`` that must equal its line number.
The cumulative trial count feeds deflated-Sharpe ``n_trials`` and the variance
of recorded Sharpes feeds its ``sr0`` — so deleting or rewriting trials is an
integrity violation, never housekeeping. Re-testing an identical config is
legitimate and recorded, but flagged ``duplicate_of=<first trial_id with that
hash>`` — silent replacement is not. Single writer at a time; readers may run
concurrently.

Design principle
----------------
In-process Python cannot be sealed against code with filesystem write access.
The enforceable invariants are:

1. ACCIDENTAL lockbox/registry violations are impossible through any public
   API: every public read path (:meth:`TrialRegistry.cumulative_trial_count`,
   :meth:`TrialRegistry.var_of_trial_sharpes`, a completed
   :meth:`TrialRegistry.query`) re-verifies the full in-file hash chain and
   the ``trial_id == line number`` law before returning data, and every append
   re-verifies before writing. No public call can silently accept a mutated
   history.
2. DELIBERATE bypass requires a conspicuous act that itself permanently marks
   the violation (burn-on-forge), and leaves evidence in more than one place:
   each line stores ``prev_sha``, the sha256 of the previous line's exact
   bytes, so editing line *k* burns the linkage of every line after *k*. To
   go undetected the forger must rewrite the whole suffix from *k*, AND forge
   the sidecar cache to match, AND the whole-file digest still diverges from
   any :meth:`TrialRegistry.rolling_digest` value the research loop mirrored
   into ``RESEARCH_LOG.md`` (see R1).
3. Residual risks are DOCUMENTED below, not hidden.

Integrity mechanics
-------------------
* **In-file hash chain.** Line 1 carries ``prev_sha == GENESIS_PREV_SHA`` (a
  fixed public constant, not the hash of any line); line *n* carries
  ``prev_sha ==`` sha256 of line *n-1*'s exact bytes (UTF-8, excluding the
  single trailing newline). The chain lives in the data, not the sidecar: any
  interior edit breaks every subsequent line's linkage.
* **The sidecar (``TRIALS.index``) is a pure cache and is NEVER trusted.** It
  stores the line count, byte size, and tail-line sha/offset purely as a
  cross-check tripwire: reads verify the full chain from genesis and then
  require the verified file to agree with the cache; disagreement in either
  direction raises :class:`RegistryTampered`. Counts are computed from the
  verified file, never served from the cache.
* **Every public read path verifies everything**, streaming once, O(n) (fine
  at 10k lines): per line it checks trailing newline, UTF-8, schema,
  ``prev_sha`` linkage, and ``trial_id == line number`` (which forces ids to
  be sequential from 1 with no repeats), then cross-checks the cache.
* **Writer preflight.** The first append per :class:`TrialRegistry` instance
  (hence at least once per process) runs the same full verification.
  Subsequent (warm) appends re-verify, before appending: the sidecar's
  agreement with the writer's committed state, the byte size, the newline
  count against the filesystem, and the tail line's sha and chain link.
* **Recovery is verification-gated.** :meth:`TrialRegistry.rebuild_index`
  recomputes the cache only from a file whose chain fully VERIFIES — it
  cannot adopt (launder) a broken chain. A legacy chainless file is refused
  everywhere except the explicit :meth:`TrialRegistry.migrate_legacy`, which
  rewrites it with chains and logs loudly at WARNING.

Residual risks (documented, not hidden)
---------------------------------------
* **R1 — self-consistent rewrite.** An attacker who rewrites the file with a
  chain that verifies AND forges the sidecar to match is undetectable
  in-process. The minimal cases: rewriting only the last line (nothing later
  references its hash), or truncating to a prefix — both require forging the
  sidecar too (evidence in a second place). Mitigation hook:
  :meth:`TrialRegistry.rolling_digest` (sha256 of the whole file) exists for
  the research loop to mirror into ``RESEARCH_LOG.md`` daily; a forged
  history then diverges from that operator-visible anchor.
* **R2 — warm-writer gap.** The warm preflight is O(bytes), not a chain walk:
  an interior edit that preserves byte size, newline count, and the tail line
  slips past ONE warm append. It is caught by every full read and by the next
  instance's first append — the burned chain link is permanent.
* **R3 — abandoned iterators.** A :meth:`TrialRegistry.query` generator
  abandoned before exhaustion has verified only the consumed prefix; the
  count/tail-vs-cache cross-checks run at completion only.
* **R4 — rebuild after sidecar loss.** :meth:`TrialRegistry.rebuild_index` on
  a chain-valid file cannot distinguish honest sidecar loss from R1. It is an
  explicit operator act, and R1's external anchor is the mitigation.
* **R5 — same-process attacker.** Code running in this interpreter can
  monkeypatch this module or hash away; no in-process design prevents that.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "GENESIS_PREV_SHA",
    "INDEX_FILENAME",
    "TRIALS_FILENAME",
    "RegistryTampered",
    "TrialRecord",
    "TrialRegistry",
    "canonical_config_hash",
]

#: Registry file at the repo root; one JSON line per trial, append-only.
TRIALS_FILENAME = "TRIALS.jsonl"
#: Sidecar cache: line count, byte size, tail sha/offset. Never trusted.
INDEX_FILENAME = "TRIALS.index"

#: ``prev_sha`` of line 1 — a fixed public constant, not the hash of any line.
GENESIS_PREV_SHA = hashlib.sha256(b"edge.research.registry/TRIALS.jsonl/genesis/v2").hexdigest()

_SHA256_HEX = r"^[0-9a-f]{64}$"

_log = logging.getLogger(__name__)


class RegistryTampered(RuntimeError):
    """The trials file or its sidecar contradicts the append-only history."""


def canonical_config_hash(config: dict[str, Any]) -> str:
    """sha256 hex digest of ``config`` serialized as canonical JSON.

    Canonical form: recursively sorted keys, compact separators, UTF-8. Two
    configs that differ only in key order hash identically. Raises
    ``TypeError`` if ``config`` is not JSON-serializable.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_hex(data: bytes) -> str:
    """sha256 hex digest of one registry line's exact bytes (no trailing newline)."""
    return hashlib.sha256(data).hexdigest()


class TrialRecord(BaseModel):
    """One recorded trial; immutable once written, extras forbidden.

    ``prev_sha`` is the in-file hash chain: sha256 of the PREVIOUS line's
    exact bytes (:data:`GENESIS_PREV_SHA` on line 1). It is part of the
    record's identity — rewriting any line burns the link stored in every
    later line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Must equal the record's line number: sequential from 1, no repeats.
    trial_id: int = Field(gt=0)
    #: When the trial ran (caller-supplied or wall clock); must be tz-aware.
    ts: datetime
    hypothesis: str
    config: dict[str, Any]
    #: sha256 of the canonical-JSON config; see :func:`canonical_config_hash`.
    config_hash: str = Field(pattern=_SHA256_HEX)
    result: dict[str, Any]
    #: trial_id of the FIRST earlier trial with the same config_hash, if any.
    duplicate_of: int | None = None
    #: sha256 of the previous line's exact bytes; GENESIS_PREV_SHA on line 1.
    prev_sha: str = Field(pattern=_SHA256_HEX)

    @field_validator("ts")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        """Reject naive timestamps: a trial's ts must name an exact instant."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must be tz-aware; got a naive datetime")
        return value


class _IndexCache(BaseModel):
    """Sidecar contents — a pure cache/tripwire, never trusted as truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_count: int = Field(ge=0)
    byte_size: int = Field(ge=0)
    #: sha256 of the last line's exact bytes (None when the registry is empty).
    tail_sha: str | None
    #: Byte offset where the last line starts (0 when the registry is empty).
    tail_offset: int = Field(ge=0)


@dataclass(frozen=True)
class _Link:
    """One verified line during a chain walk."""

    lineno: int
    offset: int
    raw_len: int  # including the trailing newline
    sha: str  # sha256 of the line's exact bytes (no trailing newline)
    record: TrialRecord


@dataclass(frozen=True)
class _WriterState:
    """A warm writer's committed view of the file, from its last full verify."""

    line_count: int
    byte_size: int
    tail_sha: str | None
    tail_offset: int
    tail_prev_sha: str | None  # the tail line's own prev_sha field


class TrialRegistry:
    """Reader/writer for the append-only trial registry rooted at ``root``.

    Contract: :meth:`record_trial` appends exactly one chained line and never
    mutates existing ones; any observed rewrite, truncation, or foreign append
    raises :class:`RegistryTampered`. All public read paths fully verify the
    in-file hash chain and the ``trial_id == line number`` law on every call —
    the sidecar is a cross-checked cache, never an answer. Single writer at a
    time: to a warm writer, another writer's append is indistinguishable from
    a foreign append and raises. See the module docstring for the design
    principle and residual risks R1-R5.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        trials_filename: str = TRIALS_FILENAME,
        index_filename: str = INDEX_FILENAME,
    ) -> None:
        """``root`` defaults to the repo root (three parents above this package)."""
        base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self._trials_path = base / trials_filename
        self._index_path = base / index_filename
        #: Committed state from this writer's last verify; None until first append.
        self._writer: _WriterState | None = None
        #: config_hash -> first trial_id; maintained alongside ``_writer``.
        self._dup_map: dict[str, int] | None = None

    # ------------------------------------------------------------------ write

    def record_trial(
        self,
        hypothesis: str,
        config: dict[str, Any],
        result: dict[str, Any],
        ts: datetime | None = None,
    ) -> TrialRecord:
        """Append one chained trial line and return the record as written.

        ``ts`` defaults to the UTC wall clock; a supplied ``ts`` must be
        tz-aware. The first append on this instance runs a FULL chain
        verification (so at least once per process); later appends re-verify
        the sidecar vs the writer's committed state, the byte size, the
        newline count, and the tail line's sha and chain link —
        :class:`RegistryTampered` on any mismatch (residual R2 documents the
        one edit shape this warm preflight cannot see). If an earlier trial
        shares this config's canonical hash, the new record carries
        ``duplicate_of`` pointing at the first such trial — it is still
        recorded.
        """
        if self._writer is None or self._dup_map is None:
            self._writer, self._dup_map = self._full_verify_for_write()
        else:
            self._verify_warm(self._writer)
        state = self._writer
        config_hash = canonical_config_hash(config)
        record = TrialRecord(
            trial_id=state.line_count + 1,
            ts=ts if ts is not None else datetime.now(UTC),
            hypothesis=hypothesis,
            config=config,
            config_hash=config_hash,
            result=result,
            duplicate_of=self._dup_map.get(config_hash),
            prev_sha=state.tail_sha if state.tail_sha is not None else GENESIS_PREV_SHA,
        )
        body = record.model_dump_json().encode("utf-8")
        with open(self._trials_path, "ab") as f:
            f.write(body + b"\n")
        new_state = _WriterState(
            line_count=state.line_count + 1,
            byte_size=state.byte_size + len(body) + 1,
            tail_sha=_sha256_hex(body),
            tail_offset=state.byte_size,
            tail_prev_sha=record.prev_sha,
        )
        self._write_cache(
            _IndexCache(
                line_count=new_state.line_count,
                byte_size=new_state.byte_size,
                tail_sha=new_state.tail_sha,
                tail_offset=new_state.tail_offset,
            )
        )
        self._writer = new_state
        self._dup_map.setdefault(config_hash, record.trial_id)
        return record

    def rebuild_index(self) -> int:
        """Recompute the sidecar cache from a trials file whose chain VERIFIES.

        Explicit recovery for honest sidecar loss (e.g. a legitimately
        imported registry). The file must fully verify — UTF-8 JSON lines
        with trailing newlines, an unbroken ``prev_sha`` chain from
        :data:`GENESIS_PREV_SHA`, and ``trial_id`` equal to the line number
        (sequential from 1, enforced, not just promised) — else
        :class:`RegistryTampered`: a broken chain can NOT be re-adopted
        (laundered) here. Residuals R1/R4: a self-consistent rewrite,
        including truncation to a prefix, does verify — anchor
        :meth:`rolling_digest` externally. Returns the trial count.
        """
        last: _Link | None = None
        count = 0
        for link in self._walk_chain():
            count += 1
            last = link
        self._write_cache(
            _IndexCache(
                line_count=count,
                byte_size=(last.offset + last.raw_len) if last is not None else 0,
                tail_sha=last.sha if last is not None else None,
                tail_offset=last.offset if last is not None else 0,
            )
        )
        self._writer = None
        self._dup_map = None
        return count

    def migrate_legacy(self) -> int:
        """Adopt a legacy CHAINLESS trials file by rewriting it with chains.

        The only path that will touch a pre-chain file: every line must be a
        well-formed legacy record with NO ``prev_sha`` and ``trial_id`` equal
        to its line number; any already-chained line means this is not a
        legacy file and raises ``ValueError``; malformed lines raise
        :class:`RegistryTampered`. The file is rewritten atomically with the
        in-file chain, the sidecar cache is rebuilt, and the migration is
        logged loudly at WARNING with the new :meth:`rolling_digest` so the
        operator can anchor it. Returns the trial count.
        """
        if self._trials_size() == 0:
            raise ValueError(f"{self._trials_path}: no legacy registry to migrate")
        prev_sha = GENESIS_PREV_SHA
        out: list[bytes] = []
        lineno = 0
        with open(self._trials_path, "rb") as f:
            for raw in f:
                lineno += 1
                if not raw.endswith(b"\n"):
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: no trailing newline"
                    )
                try:
                    obj = json.loads(raw[:-1].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno} is not JSON: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: not a JSON object"
                    )
                if "prev_sha" in obj:
                    raise ValueError(
                        f"{self._trials_path} line {lineno} already carries prev_sha — "
                        "not a legacy chainless file; refusing to migrate"
                    )
                try:
                    record = TrialRecord.model_validate({**obj, "prev_sha": prev_sha})
                except ValidationError as exc:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno} is not a valid legacy "
                        f"trial record: {exc}"
                    ) from exc
                if record.trial_id != lineno:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: trial_id={record.trial_id}; "
                        "legacy ids must be sequential from 1"
                    )
                body = record.model_dump_json().encode("utf-8")
                out.append(body + b"\n")
                prev_sha = _sha256_hex(body)
        tmp = self._trials_path.with_name(self._trials_path.name + ".migrate.tmp")
        tmp.write_bytes(b"".join(out))
        os.replace(tmp, self._trials_path)
        count = self.rebuild_index()
        _log.warning(
            "migrate_legacy: rewrote %s in place — %d legacy chainless trial lines now "
            "carry the in-file prev_sha chain; sidecar cache rebuilt; mirror "
            "rolling_digest()=%s into RESEARCH_LOG.md to anchor it",
            self._trials_path,
            count,
            self.rolling_digest(),
        )
        return count

    # ------------------------------------------------------------------- read

    def cumulative_trial_count(self) -> int:
        """Total trials ever recorded (deflated Sharpe's ``n_trials``).

        Computed by FULLY verifying the file — chain, ids, schema — and
        cross-checking the sidecar cache; the cached count is never the
        answer. O(n) by design (F7: a size-only check missed interior edits);
        <2s at 10k trials. :class:`RegistryTampered` on any violation.
        """
        return sum(1 for _ in self.query())

    def var_of_trial_sharpes(self) -> float:
        """Sample variance of recorded ``result['sharpe']`` values (DSR ``sr0``).

        Streams the registry (Welford), skipping trials whose result has no
        numeric ``sharpe``. Returns 0.0 with fewer than two Sharpe values.
        Runs the full chain verification as part of the complete scan.
        """
        n = 0
        mean = 0.0
        m2 = 0.0
        for record in self.query():
            sharpe = record.result.get("sharpe")
            if isinstance(sharpe, bool) or not isinstance(sharpe, (int, float)):
                continue
            n += 1
            delta = float(sharpe) - mean
            mean += delta / n
            m2 += delta * (float(sharpe) - mean)
        return m2 / (n - 1) if n > 1 else 0.0

    def query(
        self, predicate: Callable[[TrialRecord], bool] | None = None
    ) -> Iterator[TrialRecord]:
        """Stream trials in recorded order without loading the file into memory.

        Yields records matching ``predicate`` (all, when None). A generator:
        checks run on first ``next()`` and per line — each yielded record has
        already passed schema, ``prev_sha`` linkage, and ``trial_id == line
        number`` checks; iterating to completion additionally cross-checks the
        sidecar cache (count and tail sha), catching truncation-to-prefix and
        a rewritten final line. An early-abandoned iterator has verified only
        the consumed prefix (residual R3).
        """
        for link in self._verified_links():
            if predicate is None or predicate(link.record):
                yield link.record

    def rolling_digest(self) -> str:
        """sha256 hex of the trials file's exact bytes (of ``b""`` if absent).

        The operator-visible anchor mitigating residual R1: the research loop
        should mirror this value into ``RESEARCH_LOG.md`` daily (after a
        verified read), so a self-consistent whole-file rewrite — invisible to
        every in-process check — diverges from the logged history. A pure
        digest; performs no verification itself.
        """
        digest = hashlib.sha256()
        try:
            with open(self._trials_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
        except FileNotFoundError:
            pass
        return digest.hexdigest()

    # -------------------------------------------------------------- internals

    def _trials_size(self) -> int:
        """Current byte size of the trials file (0 if absent)."""
        try:
            return self._trials_path.stat().st_size
        except FileNotFoundError:
            return 0

    def _check_size(self, expected: int) -> None:
        """Raise unless the trials file is exactly ``expected`` bytes long."""
        size = self._trials_size()
        if size < expected:
            raise RegistryTampered(
                f"{self._trials_path} shrank: {size} bytes, expected {expected}"
            )
        if size > expected:
            raise RegistryTampered(
                f"{self._trials_path} grew outside the writer: {size} bytes, "
                f"expected {expected}"
            )

    def _count_newlines(self) -> int:
        """Number of newline bytes in the trials file (0 if absent)."""
        buf = bytearray(1 << 22)
        count = 0
        try:
            with open(self._trials_path, "rb", buffering=0) as f:
                while True:
                    read = f.readinto(buf)
                    if not read:
                        break
                    chunk = buf if read == len(buf) else buf[:read]
                    count += chunk.count(b"\n")
        except FileNotFoundError:
            return 0
        return count

    def _walk_chain(self) -> Iterator[_Link]:
        """Walk the file verifying the in-file chain and the id law per line.

        Yields a :class:`_Link` per verified line; raises
        :class:`RegistryTampered` at the first broken invariant: missing
        trailing newline, non-UTF-8, malformed record, ``prev_sha`` not
        matching the previous line's bytes, or ``trial_id != line number``.
        Yields nothing for a missing file. Does NOT consult the sidecar.
        """
        try:
            f = open(self._trials_path, "rb")
        except FileNotFoundError:
            return
        prev_sha = GENESIS_PREV_SHA
        lineno = 0
        offset = 0
        with f:
            for raw in f:
                lineno += 1
                if not raw.endswith(b"\n"):
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: no trailing newline"
                    )
                body = raw[:-1]
                try:
                    line = body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno} is not UTF-8: {exc}"
                    ) from exc
                record = self._parse_line(line, lineno)
                if record.prev_sha != prev_sha:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: prev_sha does not match "
                        "the previous line's bytes — the in-file chain is broken (an "
                        "interior rewrite burns every later link)"
                    )
                if record.trial_id != lineno:
                    raise RegistryTampered(
                        f"{self._trials_path} line {lineno}: trial_id={record.trial_id}; "
                        "trial ids must equal their line number (sequential from 1, "
                        "no repeats)"
                    )
                sha = _sha256_hex(body)
                yield _Link(lineno=lineno, offset=offset, raw_len=len(raw), sha=sha, record=record)
                prev_sha = sha
                offset += len(raw)

    def _verified_links(self) -> Iterator[_Link]:
        """Full verified walk plus sidecar cross-checks (the one read path).

        The cache gates nothing and answers nothing: after the chain walk the
        verified file must AGREE with it (size up front; count and tail sha at
        completion) — any divergence raises :class:`RegistryTampered`. A
        missing sidecar next to a non-empty file is refused: adopt explicitly
        via :meth:`rebuild_index`.
        """
        cache = self._read_cache()
        if cache is None:
            if self._trials_size() > 0:
                raise RegistryTampered(
                    f"{self._index_path} missing for non-empty {self._trials_path}; "
                    "use rebuild_index() to re-adopt the file explicitly"
                )
            return
        self._check_size(cache.byte_size)
        last: _Link | None = None
        count = 0
        for link in self._walk_chain():
            count += 1
            last = link
            yield link
        if count != cache.line_count:
            raise RegistryTampered(
                f"{self._trials_path}: {count} verified lines, sidecar cache "
                f"records {cache.line_count}"
            )
        tail_sha = last.sha if last is not None else None
        if tail_sha != cache.tail_sha:
            raise RegistryTampered(
                f"{self._trials_path}: last line does not match the sidecar's tail "
                "sha — the newest line was rewritten"
            )

    def _full_verify_for_write(self) -> tuple[_WriterState, dict[str, int]]:
        """Cold-start verify: full chain walk building writer state + dup map."""
        dup_map: dict[str, int] = {}
        last: _Link | None = None
        for link in self._verified_links():
            dup_map.setdefault(link.record.config_hash, link.record.trial_id)
            last = link
        if last is None:
            return _WriterState(0, 0, None, 0, None), dup_map
        state = _WriterState(
            line_count=last.lineno,
            byte_size=last.offset + last.raw_len,
            tail_sha=last.sha,
            tail_offset=last.offset,
            tail_prev_sha=last.record.prev_sha,
        )
        return state, dup_map

    def _verify_warm(self, state: _WriterState) -> None:
        """Warm-append preflight: sidecar agreement, size, line count, tail.

        O(bytes), not a chain walk — see residual R2 for the one edit shape
        this cannot see (same size, same newline count, tail intact); every
        full read still catches it.
        """
        cache = self._read_cache()
        if cache is None:
            raise RegistryTampered(
                f"{self._index_path} disappeared under a live writer"
            )
        committed = (state.line_count, state.byte_size, state.tail_sha, state.tail_offset)
        if (cache.line_count, cache.byte_size, cache.tail_sha, cache.tail_offset) != committed:
            raise RegistryTampered(
                f"{self._index_path} diverged from the writer's committed state"
            )
        self._check_size(state.byte_size)
        newlines = self._count_newlines()
        if newlines != state.line_count:
            raise RegistryTampered(
                f"{self._trials_path}: {newlines} lines on disk, writer committed "
                f"{state.line_count} — interior edit at constant byte size"
            )
        if state.line_count == 0:
            return
        with open(self._trials_path, "rb") as f:
            f.seek(state.tail_offset)
            chunk = f.read()
        if not chunk.endswith(b"\n"):
            raise RegistryTampered(
                f"{self._trials_path}: last line lost its trailing newline"
            )
        body = chunk[:-1]
        if _sha256_hex(body) != state.tail_sha:
            raise RegistryTampered(
                f"{self._trials_path}: last line (trial_id={state.line_count}) no "
                "longer matches its recorded hash"
            )
        tail = self._parse_line(body.decode("utf-8"), state.line_count)
        if tail.prev_sha != state.tail_prev_sha or tail.trial_id != state.line_count:
            raise RegistryTampered(
                f"{self._trials_path}: last line's chain link no longer matches the "
                "writer's committed state"
            )

    def _read_cache(self) -> _IndexCache | None:
        """Parse the sidecar; None if absent, RegistryTampered if unreadable."""
        if not self._index_path.exists():
            return None
        try:
            return _IndexCache.model_validate_json(
                self._index_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise RegistryTampered(f"{self._index_path} is unreadable: {exc}") from exc

    def _write_cache(self, cache: _IndexCache) -> None:
        """Atomically replace the sidecar with ``cache``."""
        tmp = self._index_path.with_name(self._index_path.name + ".tmp")
        tmp.write_text(cache.model_dump_json(), encoding="utf-8")
        os.replace(tmp, self._index_path)

    def _parse_line(self, line: str, lineno: int) -> TrialRecord:
        """Parse one registry line; a malformed line is tampering, not data."""
        try:
            return TrialRecord.model_validate_json(line)
        except ValidationError as exc:
            hint = ""
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and "prev_sha" not in obj:
                try:
                    TrialRecord.model_validate({**obj, "prev_sha": GENESIS_PREV_SHA})
                except ValidationError:
                    pass
                else:
                    hint = (
                        " (a legacy chainless registry must be adopted explicitly "
                        "via migrate_legacy())"
                    )
            raise RegistryTampered(
                f"{self._trials_path} line {lineno} is not a valid chained trial "
                f"record{hint}: {exc}"
            ) from exc

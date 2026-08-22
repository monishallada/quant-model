"""Trial registry: appends count, tampering is detected, duplicates are
flagged, the in-file prev_sha chain burns on forgery, the sidecar is a
never-trusted cache, and reads enforce trial_id == line number. All data is
synthetic; explicit timestamps predate the 2026-02-22 lockbox start."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from edge.research.registry import (
    GENESIS_PREV_SHA,
    INDEX_FILENAME,
    TRIALS_FILENAME,
    RegistryTampered,
    TrialRecord,
    TrialRegistry,
    canonical_config_hash,
)

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
T0 = datetime(2026, 1, 12, 10, 0, tzinfo=ET)


def _registry(tmp_path: Path) -> TrialRegistry:
    return TrialRegistry(root=tmp_path)


def _record(
    registry: TrialRegistry,
    i: int,
    sharpe: float = 1.0,
    config: dict | None = None,
) -> TrialRecord:
    return registry.record_trial(
        hypothesis=f"h{i}",
        config=config if config is not None else {"param": i},
        result={"sharpe": sharpe},
        ts=T0 + timedelta(minutes=i),
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _forge_sidecar(tmp_path: Path) -> None:
    """Forge a sidecar perfectly consistent with the trials file AS IT IS.

    The attacker's best in-process move: recompute line count, byte size, and
    tail sha/offset from the (possibly tampered) file so every cache
    cross-check passes. Detection must then come from the in-file chain.
    """
    data = (tmp_path / TRIALS_FILENAME).read_bytes()
    lines = data.split(b"\n")[:-1]
    tail = lines[-1] if lines else None
    cache = {
        "line_count": len(lines),
        "byte_size": len(data),
        "tail_sha": _sha(tail) if tail is not None else None,
        "tail_offset": len(data) - len(tail) - 1 if tail is not None else 0,
    }
    (tmp_path / INDEX_FILENAME).write_text(json.dumps(cache))


def _forge_chained_file(
    tmp_path: Path, trial_ids: list[int], write_sidecar: bool = True
) -> None:
    """Write a fully self-consistent chained file (and optionally sidecar).

    Chain linkage is always valid (independent re-implementation of the chain
    math); ``trial_ids`` controls whether the id law holds. Used to prove that
    a whole-file forger who honors the chain is still caught by structural
    checks — or, with sequential ids, only by the external anchor (R1).
    """
    prev = GENESIS_PREV_SHA
    blob = b""
    for lineno, trial_id in enumerate(trial_ids, start=1):
        config = {"param": lineno}
        line = json.dumps(
            {
                "trial_id": trial_id,
                "ts": "2026-01-12T10:00:00-05:00",
                "hypothesis": f"h{lineno}",
                "config": config,
                "config_hash": canonical_config_hash(config),
                "result": {"sharpe": 1.0},
                "duplicate_of": None,
                "prev_sha": prev,
            },
            separators=(",", ":"),
        ).encode()
        prev = _sha(line)
        blob += line + b"\n"
    (tmp_path / TRIALS_FILENAME).write_bytes(blob)
    if write_sidecar:
        _forge_sidecar(tmp_path)


def _write_legacy_file(tmp_path: Path, n: int) -> None:
    """A pre-chain registry: valid records, sequential ids, no prev_sha."""
    lines = []
    for i in range(1, n + 1):
        config = {"param": i}
        lines.append(
            json.dumps(
                {
                    "trial_id": i,
                    "ts": f"2026-01-12T10:0{i}:00-05:00",
                    "hypothesis": f"h{i}",
                    "config": config,
                    "config_hash": canonical_config_hash(config),
                    "result": {"sharpe": float(i)},
                    "duplicate_of": None,
                },
                separators=(",", ":"),
            )
        )
    (tmp_path / TRIALS_FILENAME).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Append + count
# ---------------------------------------------------------------------------


def test_append_assigns_monotonic_ids_and_counts(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.cumulative_trial_count() == 0

    records = [_record(registry, i) for i in range(1, 4)]
    assert [r.trial_id for r in records] == [1, 2, 3]
    assert registry.cumulative_trial_count() == 3
    assert len((tmp_path / TRIALS_FILENAME).read_text().splitlines()) == 3

    # A fresh instance reads the same history and continues the sequence.
    fresh = _registry(tmp_path)
    assert fresh.cumulative_trial_count() == 3
    assert _record(fresh, 4).trial_id == 4


def test_record_fields_and_config_hash(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    config = {"b": 2, "a": {"y": 1, "x": [1, 2]}}
    record = registry.record_trial(
        hypothesis="h1", config=config, result={"sharpe": 0.5}, ts=T0
    )
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert record.config_hash == expected == canonical_config_hash(config)
    assert record.ts == T0
    assert record.duplicate_of is None

    # Round-trips through the file identically.
    (loaded,) = list(registry.query())
    assert loaded == record


def test_ts_defaults_to_wall_clock_and_rejects_naive(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    record = registry.record_trial(hypothesis="h", config={"p": 1}, result={})
    assert record.ts.tzinfo is not None  # wall clock is tz-aware
    with pytest.raises(ValidationError):
        registry.record_trial(
            hypothesis="h",
            config={"p": 2},
            result={},
            ts=datetime(2026, 1, 12, 10, 0),  # noqa: DTZ001 — naive on purpose
        )


def test_query_streams_lazily_and_filters(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 6):
        _record(registry, i, sharpe=float(i))

    stream = registry.query()
    assert iter(stream) is stream  # a generator, not a loaded list
    assert next(stream).trial_id == 1

    high = list(registry.query(lambda r: r.result["sharpe"] >= 4.0))
    assert [r.trial_id for r in high] == [4, 5]
    assert [r.trial_id for r in registry.query()] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# In-file hash chain
# ---------------------------------------------------------------------------


def test_lines_carry_in_file_chain_from_genesis(tmp_path: Path) -> None:
    """Every line embeds prev_sha = sha256 of the previous line's exact bytes."""
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    raw = (tmp_path / TRIALS_FILENAME).read_bytes().split(b"\n")[:-1]
    parsed = [json.loads(line) for line in raw]
    assert parsed[0]["prev_sha"] == GENESIS_PREV_SHA
    assert parsed[1]["prev_sha"] == _sha(raw[0])
    assert parsed[2]["prev_sha"] == _sha(raw[1])
    # The chain is in the records themselves, not only on disk.
    assert [r.prev_sha for r in registry.query()] == [p["prev_sha"] for p in parsed]


# ---------------------------------------------------------------------------
# Duplicate flagging
# ---------------------------------------------------------------------------


def test_duplicate_config_flagged_not_replaced(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = _record(registry, 1, config={"a": 1, "b": {"x": 2}})
    assert first.duplicate_of is None

    # Same config, different key order: same canonical hash, flagged.
    dup = _record(registry, 2, config={"b": {"x": 2}, "a": 1})
    assert dup.duplicate_of == first.trial_id
    # A duplicate of a duplicate still points at the FIRST occurrence.
    again = _record(registry, 3, config={"a": 1, "b": {"x": 2}})
    assert again.duplicate_of == first.trial_id
    # Distinct config is not flagged.
    assert _record(registry, 4, config={"a": 999}).duplicate_of is None

    # All four are recorded — re-tests are legitimate, replacement is not.
    assert registry.cumulative_trial_count() == 4
    # A fresh instance rebuilds the hash map from the file.
    assert _record(_registry(tmp_path), 5, config={"a": 1, "b": {"x": 2}}).duplicate_of == 1


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_truncated_file_raises_on_next_append(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    lines = trials.read_text().splitlines(keepends=True)
    trials.write_text("".join(lines[:2]))  # drop the last trial

    with pytest.raises(RegistryTampered):
        _record(registry, 4)
    with pytest.raises(RegistryTampered):
        registry.cumulative_trial_count()
    with pytest.raises(RegistryTampered):
        list(_registry(tmp_path).query())


def test_edited_last_line_raises_on_next_append(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    original = trials.read_text()
    tampered = original.replace('"h3"', '"hX"')  # same byte length
    assert tampered != original and len(tampered) == len(original)
    trials.write_text(tampered)

    with pytest.raises(RegistryTampered):
        _record(registry, 4)


def test_edited_middle_line_detected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    original = trials.read_text()
    tampered = original.replace('"h2"', '"hZ"')  # same byte length, mid-file
    assert tampered != original and len(tampered) == len(original)
    trials.write_text(tampered)

    # Size and last line still match, so the full chain scan catches it...
    with pytest.raises(RegistryTampered):
        list(registry.query())
    # ...and so does the next append from a fresh instance (verifying rescan).
    with pytest.raises(RegistryTampered):
        _record(_registry(tmp_path), 4)


def test_length_changing_middle_edit_raises_on_next_append(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    trials.write_text(trials.read_text().replace('"h2"', '"h2-edited"'))

    with pytest.raises(RegistryTampered):
        _record(registry, 4)


def test_foreign_append_raises(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _record(registry, 1)

    with open(tmp_path / TRIALS_FILENAME, "a") as f:
        f.write('{"rogue": true}\n')  # bypasses the writer and its chain

    with pytest.raises(RegistryTampered):
        _record(registry, 2)
    with pytest.raises(RegistryTampered):
        registry.cumulative_trial_count()


def test_missing_sidecar_refused_until_explicit_rebuild(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    (tmp_path / INDEX_FILENAME).unlink()
    fresh = _registry(tmp_path)
    with pytest.raises(RegistryTampered):
        fresh.cumulative_trial_count()
    with pytest.raises(RegistryTampered):
        _record(fresh, 4)

    assert fresh.rebuild_index() == 3
    assert fresh.cumulative_trial_count() == 3
    record = _record(fresh, 4, config={"param": 1})  # same config as trial 1
    assert record.trial_id == 4
    assert record.duplicate_of == 1


# ---------------------------------------------------------------------------
# Burn-on-forge: the chain lives in the data, not the sidecar
# ---------------------------------------------------------------------------


def test_forged_sidecar_cannot_hide_interior_edit(tmp_path: Path) -> None:
    """F3: forging BOTH files no longer passes — the in-file chain burns.

    The attacker edits an interior line (same byte size) and then forges a
    sidecar perfectly consistent with the tampered file. Every cache
    cross-check passes; line 3's embedded prev_sha still convicts line 2.
    """
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    trials.write_text(trials.read_text().replace('"h2"', '"hZ"'))  # same size
    _forge_sidecar(tmp_path)

    fresh = _registry(tmp_path)
    with pytest.raises(RegistryTampered):
        fresh.cumulative_trial_count()
    with pytest.raises(RegistryTampered):
        list(fresh.query())
    with pytest.raises(RegistryTampered):
        fresh.var_of_trial_sharpes()
    with pytest.raises(RegistryTampered):
        _record(fresh, 4)


def test_trial_id_must_equal_line_number(tmp_path: Path) -> None:
    """F9: ids [1, 2, 1] with a VALID chain and matching sidecar now error."""
    _forge_chained_file(tmp_path, [1, 2, 1])

    fresh = _registry(tmp_path)
    with pytest.raises(RegistryTampered, match="line number"):
        fresh.cumulative_trial_count()

    # A streaming query convicts at the offending line, not silently after.
    stream = _registry(tmp_path).query()
    assert next(stream).trial_id == 1
    assert next(stream).trial_id == 2
    with pytest.raises(RegistryTampered, match="line number"):
        next(stream)


def test_interior_edit_caught_by_cumulative_count(tmp_path: Path) -> None:
    """F7 (read half): count is full verification now, never size-only."""
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)
    assert registry.cumulative_trial_count() == 3

    trials = tmp_path / TRIALS_FILENAME
    original = trials.read_text()
    tampered = original.replace('"h2"', '"hZ"')  # size-preserving interior edit
    assert len(tampered) == len(original)
    trials.write_text(tampered)

    with pytest.raises(RegistryTampered):
        registry.cumulative_trial_count()


def test_warm_writer_detects_line_structure_change(tmp_path: Path) -> None:
    """F7 (write half): warm preflight counts lines against the filesystem."""
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)  # registry is now a warm writer

    trials = tmp_path / TRIALS_FILENAME
    original = trials.read_text()
    tampered = original.replace('"h2"', '"\n2"')  # same bytes, one extra line
    assert len(tampered) == len(original)
    trials.write_text(tampered)

    with pytest.raises(RegistryTampered, match="interior edit"):
        _record(registry, 4)


def test_warm_writer_detects_sidecar_divergence(tmp_path: Path) -> None:
    """Evidence in more than one place: a warm writer re-checks its sidecar."""
    registry = _registry(tmp_path)
    _record(registry, 1)

    (tmp_path / INDEX_FILENAME).unlink()
    with pytest.raises(RegistryTampered):
        _record(registry, 2)

    _forge_sidecar(tmp_path)  # restore a consistent sidecar: writer recovers
    tampered = json.loads((tmp_path / INDEX_FILENAME).read_text())
    tampered["line_count"] = 5  # now let it lie about the count
    (tmp_path / INDEX_FILENAME).write_text(json.dumps(tampered))
    with pytest.raises(RegistryTampered):
        _record(registry, 2)


def test_warm_gap_is_documented_residual_caught_on_read(tmp_path: Path) -> None:
    """R2 (documented residual): a same-size, same-line-count, non-tail edit
    slips past ONE warm append — but the burned chain link convicts on every
    full read and on the next instance's first append."""
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)  # warm writer

    trials = tmp_path / TRIALS_FILENAME
    trials.write_text(trials.read_text().replace('"h2"', '"hZ"'))  # same size

    appended = _record(registry, 4)  # slips the O(bytes) warm preflight (R2)
    assert appended.trial_id == 4
    with pytest.raises(RegistryTampered):  # ...but the chain stays burned
        list(registry.query())
    with pytest.raises(RegistryTampered):
        _record(_registry(tmp_path), 5)


# ---------------------------------------------------------------------------
# rebuild_index: verification-gated recovery, no laundering
# ---------------------------------------------------------------------------


def test_rebuild_refuses_broken_chain(tmp_path: Path) -> None:
    """F5: rebuild_index cannot re-adopt (launder) a tampered file."""
    registry = _registry(tmp_path)
    for i in range(1, 4):
        _record(registry, i)

    trials = tmp_path / TRIALS_FILENAME
    trials.write_text(trials.read_text().replace('"h2"', '"hZ"'))  # same size
    (tmp_path / INDEX_FILENAME).unlink()  # the old laundering recipe

    fresh = _registry(tmp_path)
    with pytest.raises(RegistryTampered):
        fresh.rebuild_index()
    with pytest.raises(RegistryTampered):  # still refused afterwards
        fresh.cumulative_trial_count()


def test_rebuild_enforces_sequential_ids_from_1(tmp_path: Path) -> None:
    """F8: the docstring's 'sequential ids from 1' is enforced, not promised."""
    _forge_chained_file(tmp_path, [1, 2, 4], write_sidecar=False)
    with pytest.raises(RegistryTampered, match="line number"):
        _registry(tmp_path).rebuild_index()

    _forge_chained_file(tmp_path, [2, 3], write_sidecar=False)
    with pytest.raises(RegistryTampered, match="line number"):
        _registry(tmp_path).rebuild_index()


# ---------------------------------------------------------------------------
# Legacy chainless files: refused everywhere except migrate_legacy()
# ---------------------------------------------------------------------------


def test_legacy_chainless_file_refused_then_migrated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_legacy_file(tmp_path, 3)

    fresh = _registry(tmp_path)
    with pytest.raises(RegistryTampered):  # no sidecar: refused outright
        _record(fresh, 4)
    with pytest.raises(RegistryTampered, match="migrate_legacy"):
        fresh.rebuild_index()  # even explicit rebuild refuses a chainless file
    _forge_sidecar(tmp_path)  # a sidecar alone doesn't legitimize it either
    with pytest.raises(RegistryTampered, match="migrate_legacy"):
        _registry(tmp_path).cumulative_trial_count()

    with caplog.at_level(logging.WARNING, logger="edge.research.registry"):
        assert _registry(tmp_path).migrate_legacy() == 3
    assert "migrate_legacy" in caplog.text and "rolling_digest" in caplog.text

    migrated = _registry(tmp_path)
    records = list(migrated.query())
    assert [r.trial_id for r in records] == [1, 2, 3]
    assert [r.hypothesis for r in records] == ["h1", "h2", "h3"]
    assert records[0].prev_sha == GENESIS_PREV_SHA
    assert migrated.cumulative_trial_count() == 3
    assert _record(migrated, 4).trial_id == 4  # appends resume normally


def test_migrate_legacy_refuses_chained_or_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _registry(tmp_path).migrate_legacy()  # nothing to migrate

    registry = _registry(tmp_path)
    _record(registry, 1)
    with pytest.raises(ValueError):  # already chained: not a legacy file
        registry.migrate_legacy()


# ---------------------------------------------------------------------------
# rolling_digest: the operator-visible anchor for residual R1
# ---------------------------------------------------------------------------


def test_rolling_digest_anchors_whole_file_forgery(tmp_path: Path) -> None:
    """R1 (documented residual): truncation to a prefix plus a forged sidecar
    verifies in-process — the mitigation is the externally mirrored digest."""
    registry = _registry(tmp_path)
    assert registry.rolling_digest() == hashlib.sha256(b"").hexdigest()

    for i in range(1, 4):
        _record(registry, i)
    trials = tmp_path / TRIALS_FILENAME
    anchor = registry.rolling_digest()
    assert anchor == hashlib.sha256(trials.read_bytes()).hexdigest()

    # Attacker: truncate to a chain-valid prefix and forge the sidecar.
    lines = trials.read_bytes().split(b"\n")[:-1]
    trials.write_bytes(b"".join(line + b"\n" for line in lines[:2]))
    _forge_sidecar(tmp_path)

    fresh = _registry(tmp_path)
    assert fresh.cumulative_trial_count() == 2  # undetectable in-process (R1)
    assert fresh.rolling_digest() != anchor  # ...but the anchored digest moved


# ---------------------------------------------------------------------------
# Sharpe variance
# ---------------------------------------------------------------------------


def test_var_of_trial_sharpes(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.var_of_trial_sharpes() == 0.0

    _record(registry, 1, sharpe=1.0)
    assert registry.var_of_trial_sharpes() == 0.0  # one sample: no variance

    _record(registry, 2, sharpe=2.0)
    _record(registry, 3, sharpe=3.0)
    # Sample variance of [1, 2, 3] is 1.0.
    assert registry.var_of_trial_sharpes() == pytest.approx(1.0)

    # Trials without a numeric sharpe are skipped, not counted as zero.
    registry.record_trial(
        hypothesis="no-sharpe", config={"param": 99}, result={"pnl": 5.0}, ts=T0
    )
    assert registry.var_of_trial_sharpes() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_count_of_10k_trials_full_verification_under_2s(tmp_path: Path) -> None:
    """The count is a FULL chain verification (not a cache read) yet stays fast."""
    registry = _registry(tmp_path)
    n_trials = 10_000
    for i in range(n_trials):
        registry.record_trial(
            hypothesis=f"h{i}",
            config={"param": i},
            result={"sharpe": float(i % 7)},
            ts=T0 + timedelta(seconds=i),
        )

    fresh = _registry(tmp_path)  # no warm caches
    start = time.perf_counter()
    count = fresh.cumulative_trial_count()
    elapsed = time.perf_counter() - start
    assert count == n_trials
    assert elapsed < 2.0

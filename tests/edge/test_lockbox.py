"""Lockbox-guard tests: ET wall enforcement, content-checked one-shot key,
burn-on-forge evidence, cross-anchored spend detection.

ALL data here is synthetic. Timestamps inside the 2026-02-22..2026-12-31
lockbox window appear ONLY because these are lockbox-guard tests — they
exercise the embargo machinery itself, never real market data. Every
``issue()``/sentinel/registry write happens under a pytest tmp repo root,
never the real one.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from edge.core.config import load_config
from edge.data.loader import EdgeDataLoader, FetchCall, SyntheticBackend
from edge.research.registry import TrialRegistry
from edge.validation.lockbox import (
    ISSUANCE_PREFIX,
    LOCKBOX_START,
    SENTINEL_FILENAME,
    LockboxAlreadySpent,
    LockboxKey,
    LockboxTampered,
    LockboxViolation,
    LockboxWall,
    detect_tamper,
    et_trading_date,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("US/Pacific")

# Synthetic range anchors around the wall (guard tests only).
WALL = date(2026, 2, 22)
LAST_ALLOWED = WALL - timedelta(days=1)
PRE_START = date(2026, 1, 5)
PRE_END = date(2026, 2, 10)
IN_WINDOW_END = date(2026, 6, 30)
IN_WINDOW_START = date(2026, 3, 2)

# F6 anchors: instants whose ET calendar date is the FIRST lockbox session,
# expressed in westward/UTC zones that a naive comparison would let through.
UTC_IN_FIRST_ET_SESSION = datetime(2026, 2, 22, 14, 30, tzinfo=UTC)  # 09:30 ET Feb 22
PACIFIC_IN_FIRST_ET_SESSION = datetime(2026, 2, 21, 23, 0, tzinfo=PACIFIC)  # 02:00 ET Feb 22
UTC_STILL_LAST_ALLOWED_ET = datetime(2026, 2, 22, 3, 0, tzinfo=UTC)  # 22:00 ET Feb 21

# Synthetic backend parameters (arbitrary; supplied by the test, not the code).
BASE_PRICE = 100.0
DAILY_VOLUME = 12_000


def make_backend() -> SyntheticBackend:
    return SyntheticBackend(base_price=BASE_PRICE, daily_volume=DAILY_VOLUME)


def make_repo_root(tmp_path: Path, lockbox_start: str) -> Path:
    """Copy the real edge.yaml into a tmp repo root with a rewritten wall."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    text = (REPO_ROOT / "config" / "edge.yaml").read_text()
    new_text = re.sub(r'lockbox_start:\s*"[^"]*"', f'lockbox_start: "{lockbox_start}"', text)
    assert new_text != text or lockbox_start in text  # the rewrite must have applied
    (cfg_dir / "edge.yaml").write_text(new_text)
    return tmp_path


# ---------------------------------------------------------------------------
# Wall: config-driven, read at call time  [P8]
# ---------------------------------------------------------------------------


def test_lockbox_start_read_from_config() -> None:
    assert LOCKBOX_START == WALL
    assert LOCKBOX_START == load_config().data.lockbox_start
    wall = LockboxWall()
    assert wall.lockbox_start() == WALL
    assert wall.last_allowed_day() == LAST_ALLOWED


def test_wall_rereads_config_at_call_time(tmp_path: Path) -> None:
    """The same wall instance tracks edits to edge.yaml — nothing is cached."""
    root = make_repo_root(tmp_path, "2026-02-22")
    wall = LockboxWall(repo_root=root)
    assert wall.lockbox_start() == WALL

    make_repo_root(tmp_path, "2026-02-01")  # rewrite in place
    assert wall.lockbox_start() == date(2026, 2, 1)
    assert wall.last_allowed_day() == date(2026, 1, 31)


def test_loader_reads_wall_at_call_time(tmp_path: Path) -> None:
    """An already-constructed loader obeys a config change on the next call."""
    root = make_repo_root(tmp_path, "2026-02-22")
    backend = make_backend()
    loader = EdgeDataLoader(backend, repo_root=str(root))

    loader.load("SPY", PRE_START, IN_WINDOW_END, "bars")
    assert backend.calls[-1].end == LAST_ALLOWED

    make_repo_root(tmp_path, "2026-02-01")  # tighten the wall after construction
    loader.load("SPY", PRE_START, IN_WINDOW_END, "bars")
    assert backend.calls[-1].end == date(2026, 1, 31)


# ---------------------------------------------------------------------------
# Locked loader: clamp with warning, raise on explicit lockbox reads  [P1]
# ---------------------------------------------------------------------------


def test_locked_loader_truncates_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    backend = make_backend()
    loader = EdgeDataLoader(backend)  # default: locked

    with caplog.at_level(logging.WARNING, logger="edge.data.loader"):
        frame = loader.load("SPY", PRE_START, IN_WINDOW_END, "bars")

    assert backend.calls == [
        FetchCall(symbol="SPY", start=PRE_START, end=LAST_ALLOWED, kind="bars")
    ]
    assert frame.index.max().date() == LAST_ALLOWED
    assert "lockbox clamp" in caplog.text
    assert str(LAST_ALLOWED) in caplog.text


def test_locked_loader_leaves_prewall_range_untouched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = make_backend()
    loader = EdgeDataLoader(backend)

    with caplog.at_level(logging.WARNING, logger="edge.data.loader"):
        frame = loader.load("SPY", PRE_START, PRE_END, "quotes")

    assert backend.calls == [
        FetchCall(symbol="SPY", start=PRE_START, end=PRE_END, kind="quotes")
    ]
    assert len(frame) == (PRE_END - PRE_START).days + 1
    assert caplog.text == ""


def test_locked_loader_raises_on_explicit_lockbox_read() -> None:
    backend = make_backend()
    loader = EdgeDataLoader(backend)

    with pytest.raises(LockboxViolation):
        loader.load("SPY", IN_WINDOW_START, IN_WINDOW_END, "bars")
    with pytest.raises(LockboxViolation):  # start exactly AT the wall
        loader.load("SPY", WALL, WALL, "trades")
    assert backend.calls == []  # the backend was never reached


def test_loader_rejects_inverted_range() -> None:
    loader = EdgeDataLoader(make_backend())
    with pytest.raises(ValueError):
        loader.load("SPY", PRE_END, PRE_START, "bars")


def test_no_bypass_hooks() -> None:
    """No constructor/load kwarg unlocks; a nonce-less fake key is rejected.  [P7]"""
    backend = make_backend()
    with pytest.raises(TypeError):
        EdgeDataLoader(backend, unlocked=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EdgeDataLoader(backend, key="letmein")  # a str has no .nonce
    loader = EdgeDataLoader(backend)
    with pytest.raises(TypeError):
        loader.load("SPY", PRE_START, PRE_END, "bars", unlocked=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TZ anchor: the wall is an America/New_York boundary  [F6]
# ---------------------------------------------------------------------------


def test_et_trading_date_normalization() -> None:
    assert et_trading_date(UTC_IN_FIRST_ET_SESSION) == WALL
    assert et_trading_date(PACIFIC_IN_FIRST_ET_SESSION) == WALL
    assert et_trading_date(UTC_STILL_LAST_ALLOWED_ET) == LAST_ALLOWED
    assert et_trading_date(LAST_ALLOWED) == LAST_ALLOWED  # dates are ET as-is
    with pytest.raises(ValueError):
        et_trading_date(datetime(2026, 2, 10, 10, 0))  # naive: never guess
    with pytest.raises(TypeError):
        et_trading_date("2026-02-10")  # type: ignore[arg-type]


def test_naive_datetimes_rejected_even_when_unlocked(tmp_path: Path) -> None:
    """Naive datetimes raise on BOTH bounds and on both lock states."""
    backend = make_backend()
    loader = EdgeDataLoader(backend)
    naive = datetime(2026, 2, 10, 10, 0)
    with pytest.raises(ValueError, match="naive"):
        loader.load("SPY", naive, PRE_END, "bars")
    with pytest.raises(ValueError, match="naive"):
        loader.load("SPY", PRE_START, naive, "bars")
    assert backend.calls == []

    root = make_repo_root(tmp_path, "2026-02-22")
    key = LockboxKey.issue("naive-rejection test", repo_root=root)
    unlocked = EdgeDataLoader(make_backend(), key=key, repo_root=root)
    with pytest.raises(ValueError, match="naive"):
        unlocked.load("SPY", naive, PRE_END, "bars")


def test_utc_datetime_inside_first_et_session_refused() -> None:
    """2026-02-22 14:30 UTC is 09:30 ET on the wall day — refused."""
    backend = make_backend()
    loader = EdgeDataLoader(backend)
    with pytest.raises(LockboxViolation):
        loader.load("SPY", UTC_IN_FIRST_ET_SESSION, IN_WINDOW_END, "bars")
    assert backend.calls == []


def test_pacific_datetime_inside_first_et_session_refused() -> None:
    """2026-02-21 23:00 US/Pacific IS 2026-02-22 02:00 ET — refused.

    This is the F6 leak: a westward clock still reading Feb 21 while the
    first ET lockbox session is already underway.
    """
    backend = make_backend()
    loader = EdgeDataLoader(backend)
    with pytest.raises(LockboxViolation):
        loader.load("SPY", PACIFIC_IN_FIRST_ET_SESSION, IN_WINDOW_END, "bars")
    assert backend.calls == []


def test_wall_anchors_to_et_not_utc(caplog: pytest.LogCaptureFixture) -> None:
    """A UTC instant late on ET Feb 21 (already Feb 22 in UTC) is still allowed."""
    backend = make_backend()
    loader = EdgeDataLoader(backend)
    with caplog.at_level(logging.WARNING, logger="edge.data.loader"):
        loader.load("SPY", PRE_START, UTC_STILL_LAST_ALLOWED_ET, "bars")
    assert backend.calls == [
        FetchCall(symbol="SPY", start=PRE_START, end=LAST_ALLOWED, kind="bars")
    ]
    assert caplog.text == ""  # no clamp: the ET date was already last-allowed


# ---------------------------------------------------------------------------
# Key: content-checked, forgery-inert  [F1]
# ---------------------------------------------------------------------------


def test_issue_writes_root_sentinel_and_registry_record(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, "2026-02-22")
    key = LockboxKey.issue("final OOS evaluation", repo_root=root)

    # Sentinel: at the repo ROOT (not results/), with nonce/purpose/issued_ts.
    sentinel = root / SENTINEL_FILENAME
    assert sentinel.is_file()
    record = json.loads(sentinel.read_text())
    assert record["nonce"] == key.nonce
    assert re.fullmatch(r"[0-9a-f]{32}", key.nonce)  # 128-bit hex nonce
    assert record["purpose"] == "final OOS evaluation"
    assert record["issued_ts"]

    # Cross-anchor: the append-only registry recorded the issuance.
    issuances = list(
        TrialRegistry(root=root).query(
            lambda r: r.hypothesis.startswith(ISSUANCE_PREFIX)
        )
    )
    assert len(issuances) == 1
    assert issuances[0].hypothesis == f"{ISSUANCE_PREFIX}final OOS evaluation"
    assert issuances[0].config["nonce"] == key.nonce

    # The key unlocks: the full lockbox range reaches the backend, unclamped.
    backend = make_backend()
    loader = EdgeDataLoader(backend, key=key, repo_root=root)
    loader.load("SPY", IN_WINDOW_START, IN_WINDOW_END, "bars")
    assert backend.calls == [
        FetchCall(symbol="SPY", start=IN_WINDOW_START, end=IN_WINDOW_END, kind="bars")
    ]


def test_unlock_is_content_not_class(tmp_path: Path) -> None:
    """Any nonce holder with the RIGHT nonce unlocks; LockboxKey with the
    WRONG nonce does not. Class identity carries zero authority."""
    root = make_repo_root(tmp_path, "2026-02-22")
    real = LockboxKey.issue("content check", repo_root=root)

    class DuckKey:
        def __init__(self, nonce: str) -> None:
            self.nonce = nonce

    backend = make_backend()
    EdgeDataLoader(backend, key=DuckKey(real.nonce), repo_root=root).load(
        "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
    )
    assert backend.calls[-1].end == IN_WINDOW_END  # right content: unlocked

    wrong = LockboxKey(nonce="0" * 32)  # right class, wrong content
    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), key=wrong, repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )


def test_direct_construction_is_inert(tmp_path: Path) -> None:
    """LockboxKey() is legal and harmless: without the sentinel nonce it is
    just an object. No guard to skip, no marker to steal.  [F1a/F1b]"""
    root = make_repo_root(tmp_path, "2026-02-22")  # unspent: no sentinel at all
    key = LockboxKey(nonce="deadbeef" * 4)
    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), key=key, repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )


def test_forged_key_vectors_stay_locked(tmp_path: Path) -> None:
    """Every F1 forgery vector yields a locked loader or a TypeError."""
    root = make_repo_root(tmp_path, "2026-02-22")
    LockboxKey.issue("forgery target", repo_root=root)

    # [F1a] __new__ skipping __init__: no nonce attribute -> rejected outright.
    hollow = LockboxKey.__new__(LockboxKey)
    with pytest.raises(TypeError):
        EdgeDataLoader(make_backend(), key=hollow, repo_root=root)

    # [F1c] isinstance-passing subclass with overridden __init__: wrong content.
    class EvilKey(LockboxKey):
        def __init__(self) -> None:  # skips any parent logic
            object.__setattr__(self, "nonce", "f" * 32)

    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), key=EvilKey(), repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )

    # __eq__-lying str subclass as the nonce: content is compared via
    # str.__eq__ on the sentinel side, so the lie never gets asked.
    class LyingStr(str):
        def __eq__(self, other: object) -> bool:  # noqa: D105
            return True

        __hash__ = str.__hash__

    class LiarKey:
        nonce = LyingStr("not-the-nonce")

    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), key=LiarKey(), repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )


def test_forging_sentinel_burns_the_lockbox(tmp_path: Path) -> None:
    """Forge = burn: hand-writing the sentinel unlocks (documented residual
    risk) but permanently spends the lockbox AND leaves a registry-diff
    hole that detect_tamper() flags."""
    root = make_repo_root(tmp_path, "2026-02-22")
    forged_nonce = "ab" * 16
    (root / SENTINEL_FILENAME).write_text(
        json.dumps({"nonce": forged_nonce, "purpose": "forged", "issued_ts": "x"})
    )

    # The forge works — that is the documented residual risk...
    backend = make_backend()
    EdgeDataLoader(backend, key=LockboxKey(nonce=forged_nonce), repo_root=root).load(
        "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
    )
    assert backend.calls[-1].end == IN_WINDOW_END

    # ...but the lockbox is now spent forever,
    with pytest.raises(LockboxAlreadySpent):
        LockboxKey.issue("legitimate open", repo_root=root)
    # and the missing registry issuance record exposes the forge.
    with pytest.raises(LockboxTampered, match="forged"):
        detect_tamper(repo_root=root)


def test_corrupt_sentinel_is_tamper_not_silence(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, "2026-02-22")
    (root / SENTINEL_FILENAME).write_text("{not json")
    with pytest.raises(LockboxTampered):
        detect_tamper(repo_root=root)
    with pytest.raises(LockboxTampered):  # a keyed loader alarms, never guesses
        EdgeDataLoader(make_backend(), key=LockboxKey(nonce="0" * 32), repo_root=root).load(
            "SPY", PRE_START, PRE_END, "bars"
        )


# ---------------------------------------------------------------------------
# Spend: one-time, cross-process, deletion detected  [F4]
# ---------------------------------------------------------------------------


def test_second_issue_raises_same_process(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, "2026-02-22")
    LockboxKey.issue("first", repo_root=root)
    with pytest.raises(LockboxAlreadySpent):
        LockboxKey.issue("second", repo_root=root)


def test_second_issue_raises_in_new_process(tmp_path: Path) -> None:
    """Spent state is the file, not memory: a fresh interpreter still refuses."""
    root = make_repo_root(tmp_path, "2026-02-22")
    LockboxKey.issue("first", repo_root=root)
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from edge.validation.lockbox import LockboxAlreadySpent, LockboxKey
        try:
            LockboxKey.issue("second", repo_root=Path({str(root)!r}))
        except LockboxAlreadySpent:
            raise SystemExit(0)
        raise SystemExit(1)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_spend_survives_loader_reconstruction(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, "2026-02-22")
    key = LockboxKey.issue("evaluation", repo_root=root)

    # Re-constructed loaders: the key keeps unlocking, no-key stays locked.
    for _ in range(2):
        backend = make_backend()
        EdgeDataLoader(backend, key=key, repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )
        assert backend.calls[-1].end == IN_WINDOW_END
    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )
    # And the spend is still permanent after all that churn.
    with pytest.raises(LockboxAlreadySpent):
        LockboxKey.issue("again", repo_root=root)


def test_deleted_sentinel_relocks_and_leaves_evidence(tmp_path: Path) -> None:
    """[F4] Deleting LOCKBOX_SPENT.json relocks every key, does NOT reopen
    issue(), and is flagged by detect_tamper() via the orphaned registry record."""
    root = make_repo_root(tmp_path, "2026-02-22")
    key = LockboxKey.issue("evaluation", repo_root=root)
    (root / SENTINEL_FILENAME).unlink()

    # The key stops unlocking...
    with pytest.raises(LockboxViolation):
        EdgeDataLoader(make_backend(), key=key, repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )
    # ...re-issue trips the tamper check instead of silently reopening...
    with pytest.raises(LockboxTampered):
        LockboxKey.issue("reopen attempt", repo_root=root)
    # ...and the audit function names the orphaned issuance record.
    with pytest.raises(LockboxTampered, match="deleted"):
        detect_tamper(repo_root=root)


def test_replaced_sentinel_nonce_detected(tmp_path: Path) -> None:
    """A sentinel rewritten with a fresh nonce contradicts the recorded one."""
    root = make_repo_root(tmp_path, "2026-02-22")
    key = LockboxKey.issue("evaluation", repo_root=root)
    (root / SENTINEL_FILENAME).write_text(
        json.dumps({"nonce": "c" * 32, "purpose": "swapped", "issued_ts": "x"})
    )
    with pytest.raises(LockboxTampered, match="replaced"):
        detect_tamper(repo_root=root)
    with pytest.raises(LockboxViolation):  # the original key no longer matches
        EdgeDataLoader(make_backend(), key=key, repo_root=root).load(
            "SPY", IN_WINDOW_START, IN_WINDOW_END, "bars"
        )


def test_detect_tamper_passes_clean_states(tmp_path: Path) -> None:
    root = make_repo_root(tmp_path, "2026-02-22")
    detect_tamper(repo_root=root)  # never spent: no sentinel, no issuance
    LockboxKey.issue("legit", repo_root=root)
    detect_tamper(repo_root=root)  # legitimately spent: anchors agree

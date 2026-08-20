"""Promotion ledger — how a strategy earns its way to real money.

    backtest (passes standard report)  ->  validated
    paper session (real Alpaca paper)  ->  paper_tested
    both + typed confirmation          ->  live

The two flags are **written by evidence, never typed by hand**. `validated` is
set only by the pipeline, from the verdict its own standard report produced;
`paper_tested` is set only after an actual paper session against a real broker
account. Neither can be set from a config file or a CLI flag, which is the
point: at the live gate, "has this been paper-tested?" is answered by history
rather than by the person who wants to deploy it.

The ledger lives on disk (`results/<name>/status.json`) rather than in the
strategy's source, because a status the author can edit in the same commit as
the strategy is not evidence of anything. It also records *why* each flag was
granted, so a promotion can be audited later — and it demotes automatically if
the strategy's code changes after validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_ROOT = Path("results")
STATUS_FILE = "status.json"


@dataclass
class PromotionRecord:
    """Evidence trail for one strategy."""

    name: str
    validated: bool = False
    paper_tested: bool = False
    validated_on: str | None = None
    validated_verdict: str | None = None
    validated_avg_monthly: float | None = None
    validated_code_hash: str | None = None
    paper_tested_on: str | None = None
    paper_account: str | None = None
    paper_orders_seen: int = 0
    history: list[dict] = field(default_factory=list)

    # -- persistence ------------------------------------------------------
    @classmethod
    def path_for(cls, name: str) -> Path:
        return LEDGER_ROOT / name / STATUS_FILE

    @classmethod
    def load(cls, name: str) -> PromotionRecord:
        p = cls.path_for(name)
        if not p.exists():
            return cls(name=name)
        try:
            return cls(**json.loads(p.read_text()))
        except Exception as e:                          # noqa: BLE001
            logger.warning("unreadable promotion record %s: %s", p, e)
            return cls(name=name)

    def save(self) -> Path:
        p = self.path_for(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    def _log(self, event: str, detail: str) -> None:
        self.history.append({"ts": datetime.now(UTC).isoformat(),
                             "event": event, "detail": detail})


def code_hash(module_path: Path) -> str:
    """Fingerprint of the strategy source.

    Validation attaches to a specific body of code. If the file changes after
    validation, the evidence no longer describes what would run, so the flag is
    withdrawn rather than silently carried forward.
    """
    if not module_path.exists():
        return ""
    return hashlib.sha256(module_path.read_bytes()).hexdigest()[:16]


def record_backtest(
    name: str, verdict: str, avg_monthly: float | None, module_path: Path | None = None
) -> PromotionRecord:
    """Called by the pipeline. Grants `validated` ONLY on a passing verdict."""
    rec = PromotionRecord.load(name)
    passed = verdict.startswith("CANDIDATE") and "test segment missing" not in verdict
    rec.validated = passed
    rec.validated_on = datetime.now(UTC).isoformat() if passed else None
    rec.validated_verdict = verdict
    rec.validated_avg_monthly = avg_monthly
    rec.validated_code_hash = code_hash(module_path) if module_path else None
    rec._log("backtest", f"{'PASS' if passed else 'FAIL'}: {verdict}")
    if not passed:
        # A failing re-run withdraws a previous pass. Otherwise a strategy could
        # be validated once and edited freely afterwards.
        rec.paper_tested = False
        rec._log("demote", "validation withdrawn; paper status reset")
    rec.save()
    logger.info("promotion: %s validated=%s (%s)", name, rec.validated, verdict)
    return rec


def record_paper_session(
    name: str, account: str, orders_seen: int = 0
) -> PromotionRecord:
    """Called after a real paper session. Requires prior validation."""
    rec = PromotionRecord.load(name)
    if not rec.validated:
        rec._log("paper", "refused: not validated")
        rec.save()
        raise PermissionError(
            f"'{name}' ran in paper but is not validated — paper_tested not granted")
    rec.paper_tested = True
    rec.paper_tested_on = datetime.now(UTC).isoformat()
    rec.paper_account = account
    rec.paper_orders_seen = orders_seen
    rec._log("paper", f"paper session on {account}, {orders_seen} orders")
    rec.save()
    logger.info("promotion: %s paper_tested=True on %s", name, account)
    return rec


def check_live_eligibility(name: str, module_path: Path | None = None) -> tuple[bool, str]:
    """(eligible, reason). The live gate's source of truth."""
    rec = PromotionRecord.load(name)
    if not rec.validated:
        return False, (f"'{name}' is not validated. Run the full pipeline; the "
                       "report must return a CANDIDATE verdict.")
    if not rec.paper_tested:
        return False, (f"'{name}' is validated but never paper-tested. Run "
                       "--mode paper first. This is not overridable.")
    if module_path is not None and rec.validated_code_hash:
        current = code_hash(module_path)
        if current != rec.validated_code_hash:
            return False, (
                f"'{name}' has changed since validation "
                f"(hash {rec.validated_code_hash} -> {current}). Re-run the "
                "pipeline: the evidence no longer describes this code.")
    return True, (f"validated {rec.validated_on} ({rec.validated_verdict}); "
                  f"paper-tested {rec.paper_tested_on} on {rec.paper_account}")

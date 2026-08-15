"""Kill switch, heartbeat, structured logging.

The kill switch is a **file on disk**, not a config value or an in-process
flag, and that is the whole point: stopping a running system must not require
the running system to cooperate. Anyone with shell access can `touch` the flag
and the next cycle halts new entries — no redeploy, no API call, no restart,
and it works even if the process is wedged in a retry loop.

It halts *new entries only*. Existing positions still get marked, managed and
exited, because abandoning open risk is not a safe stop — it is a different
kind of accident.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KILL_FILE = Path(os.environ.get("CATALYST_KILL_FILE", ".catalyst_kill"))
DEFAULT_HEARTBEAT_FILE = Path(os.environ.get("CATALYST_HEARTBEAT_FILE", ".catalyst_heartbeat"))


@dataclass
class KillSwitch:
    """Presence of ``path`` means: stop opening new positions, now."""

    path: Path = DEFAULT_KILL_FILE

    def engaged(self) -> bool:
        return self.path.exists()

    def reason(self) -> str:
        try:
            return self.path.read_text().strip() or "no reason recorded"
        except OSError:
            return "unreadable"

    def engage(self, reason: str = "manual") -> None:
        self.path.write_text(f"{datetime.now(UTC).isoformat()} {reason}\n")
        logger.critical("KILL SWITCH ENGAGED: %s", reason)

    def release(self) -> None:
        """Deliberately never called by the trading loop — a system must not be
        able to clear its own stop. Operator action only."""
        self.path.unlink(missing_ok=True)
        logger.warning("kill switch released")


@dataclass
class Heartbeat:
    path: Path = DEFAULT_HEARTBEAT_FILE

    def beat(self, **fields: object) -> None:
        payload = {"ts": datetime.now(UTC).isoformat(), **fields}
        self.path.write_text(json.dumps(payload, default=str))

    def age_seconds(self) -> float | None:
        if not self.path.exists():
            return None
        return (datetime.now(UTC).timestamp() - self.path.stat().st_mtime)


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line — greppable and machine-parseable after the fact."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO, path: Path | None = None) -> None:
    handler: logging.Handler = logging.FileHandler(path) if path else logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

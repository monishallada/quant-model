"""QuantConnect LEAN engine — the deepest options modelling available.

LEAN is the strongest open-source options backtester: it models option chains,
**corporate actions including splits**, early assignment and exercise,
dividends and margin. Two of those are gaps in the native engine that have
already cost this project real accuracy — the split-adjustment mismatch that
invalidated three of six names in v9, and the absence of early-assignment
modelling on the short legs of v4 Drift and v8 Index VRP.

Two things gate it, and both are stated up front rather than discovered
mid-run:

1. **Docker.** `lean backtest` runs the engine inside a container. The CLI is
   installed here; the daemon must be started (`open -a Docker` on macOS).
2. **Data.** LEAN reads its own on-disk format. ThetaData is not a native LEAN
   provider, so option quotes must be converted into LEAN's option-chain layout
   before it can run. `ThetaToLeanWriter` below does that conversion using the
   SAME ThetaData credentials already configured in this repo — no second
   subscription, no second source of truth for prices.

Until both are satisfied `available()` returns False with the specific reason,
and the comparison report shows LEAN as unavailable rather than silently
omitting it. An engine that quietly disappears from a comparison is worse than
one that is absent, because the reader cannot tell the difference between
"agreed" and "never ran".
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.core.types import OptionRight

logger = logging.getLogger(__name__)

LEAN_ROOT = Path("lean")
DATA_ROOT = LEAN_ROOT / "data"


def _docker_running() -> tuple[bool, str]:
    exe = shutil.which("docker")
    if exe is None:
        return False, "docker not installed"
    try:
        r = subprocess.run([exe, "info"], capture_output=True, timeout=15)
    except Exception as e:                              # noqa: BLE001
        return False, f"docker check failed: {e}"
    if r.returncode != 0:
        return False, "docker installed but daemon not running (start Docker Desktop)"
    return True, "docker daemon running"


def _lean_cli() -> tuple[bool, str]:
    try:
        import lean  # noqa: F401
        return True, "lean CLI installed"
    except Exception as e:                              # noqa: BLE001
        return False, f"lean CLI unavailable: {e}"


@dataclass
class ThetaToLeanWriter:
    """Convert ThetaData option quotes into LEAN's on-disk option format.

    LEAN expects, per underlying and per date, zipped CSVs of option quotes
    keyed by the OSI-style contract identity. We already hold the canonical
    `OptionKey`, so this is a mechanical projection — and critically it reuses
    the SAME ThetaData snapshots the native engine consumes, so any divergence
    between the two engines is attributable to *engine* behaviour rather than
    to two different price sources.

    Prices are written in LEAN's deci-cent convention (x10000).
    """

    data_root: Path = DATA_ROOT

    def option_dir(self, symbol: str) -> Path:
        return self.data_root / "option" / "usa" / "minute" / symbol.lower()

    def write_chain(self, symbol: str, when: datetime, chain) -> int:
        """Write one chain snapshot. Returns contracts written."""
        out = self.option_dir(symbol)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in chain.contracts:
            if c.bid <= 0 or c.ask <= 0:
                continue
            k = c.key
            rows.append({
                "time": int(pd.Timestamp(when).value // 1_000_000),
                "expiry": k.expiry.strftime("%Y%m%d"),
                "strike": int(round(k.strike * 10000)),
                "right": "C" if k.right is OptionRight.CALL else "P",
                "bid": int(round(c.bid * 10000)),
                "ask": int(round(c.ask * 10000)),
                "bidsize": max(int(c.open_interest or 0), 1),
                "asksize": max(int(c.open_interest or 0), 1),
            })
        if not rows:
            return 0
        path = out / f"{when.date():%Y%m%d}_quote.csv"
        pd.DataFrame(rows).to_csv(path, index=False, header=False)
        return len(rows)

    def coverage(self, symbol: str) -> int:
        d = self.option_dir(symbol)
        return len(list(d.glob("*_quote.csv"))) if d.exists() else 0


class LeanEngine(BacktestEngine):
    name = "lean"
    verifies = ("independent event loop with corporate actions (splits), early "
                "assignment/exercise, dividends and margin modelling")

    def __init__(self, project_dir: Path | None = None, timeout: int = 3600) -> None:
        self._project = project_dir or (LEAN_ROOT / "project")
        self._timeout = timeout
        self.writer = ThetaToLeanWriter()

    def available(self) -> tuple[bool, str]:
        ok_cli, why_cli = _lean_cli()
        if not ok_cli:
            return False, why_cli
        ok_docker, why_docker = _docker_running()
        if not ok_docker:
            return False, why_docker
        if not self._project.exists():
            return False, (f"no LEAN project at {self._project} — run "
                           "`uv run python -m catalyst.runners.lean_setup`")
        if not DATA_ROOT.exists():
            return False, (f"no converted ThetaData at {DATA_ROOT} — run "
                           "`uv run python -m catalyst.runners.lean_setup --convert`")
        # The CLI requires a root folder initialised against a QuantConnect
        # ORGANIZATION, which needs a free account (user id + API token). A
        # hand-written lean.json is rejected: "This is an old Lean CLI root
        # folder." Report that precisely rather than letting the backtest fail
        # 4 minutes into a Docker run with a confusing message.
        cfg_file = LEAN_ROOT / "lean.json"
        if not cfg_file.exists():
            return False, ("LEAN root not initialised — run `lean init` inside "
                           f"{LEAN_ROOT} (needs a free QuantConnect account)")
        try:
            import json
            if not json.loads(cfg_file.read_text()).get("organization-id"):
                return False, ("LEAN root missing organization-id — run `lean login` "
                               "then `lean init` in the lean/ folder. Requires a free "
                               "QuantConnect account (user id + API token).")
        except Exception:                               # noqa: BLE001
            return False, f"unreadable {cfg_file}"
        return True, f"{why_cli}; {why_docker}"

    def run(self, strategy, start: date, end: date, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost: bool = False) -> EngineResult:
        ok, reason = self.available()
        if not ok:
            return EngineResult(engine=self.name, error=reason)

        # cwd is LEAN_ROOT, so the project path must be RELATIVE to it —
        # passing the repo-relative path resolves to lean/lean/project.
        rel = self._project.name if self._project.parent == LEAN_ROOT else str(self._project)
        cmd = ["lean", "backtest", rel, "--data-provider-historical", "Local"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self._timeout, cwd=str(LEAN_ROOT))
        except subprocess.TimeoutExpired:
            return EngineResult(engine=self.name,
                                error=f"LEAN exceeded {self._timeout}s")
        except Exception as e:                          # noqa: BLE001
            return EngineResult(engine=self.name, error=f"{type(e).__name__}: {e}")

        if proc.returncode != 0:
            return EngineResult(engine=self.name,
                                error=f"lean exited {proc.returncode}: "
                                      f"{proc.stderr[-400:] or proc.stdout[-400:]}")
        return _parse_results(self._project, self.name)


def _parse_results(project: Path, engine: str) -> EngineResult:
    """Read LEAN's backtest output into the shared EngineResult shape."""
    import json

    results = sorted((project / "backtests").glob("*/*.json")) if project.exists() else []
    if not results:
        return EngineResult(engine=engine, error="LEAN produced no result file")
    try:
        payload = json.loads(results[-1].read_text())
    except Exception as e:                              # noqa: BLE001
        return EngineResult(engine=engine, error=f"unreadable LEAN result: {e}")

    charts = payload.get("charts", {}) or payload.get("Charts", {})
    equity_chart = charts.get("Strategy Equity", {}).get("Series", {}).get("Equity", {})
    points = equity_chart.get("Values", []) or equity_chart.get("values", [])
    curve = {}
    for p in points:
        if isinstance(p, dict) and "x" in p:
            curve[pd.Timestamp(p["x"], unit="s")] = float(p.get("y", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            curve[pd.Timestamp(p[0], unit="s")] = float(p[1])
    series = pd.Series(curve).sort_index() if curve else pd.Series(dtype=float)
    stats = payload.get("statistics", {}) or payload.get("Statistics", {})
    return EngineResult(engine=engine, equity=series, trades=[],
                        diagnostics={"lean_statistics": stats})

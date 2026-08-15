"""Run LEAN by invoking its Docker image directly — no QuantConnect account.

The `lean` CLI is a convenience wrapper around `docker run quantconnect/lean`,
and it insists on a root folder initialised against a QuantConnect
*organization*, which needs a free account (user id + API token). That is a
packaging requirement of the CLI, not of the engine: LEAN itself is Apache-2.0
and runs from a plain `config.json`.

So this bypasses the CLI and drives the container directly. Same engine, same
modelling — corporate actions including splits, early assignment and exercise,
dividends, margin — with no external account and no credential to manage.

What gets mounted:
  data/      the ThetaData chains converted into LEAN's on-disk option format
  project/   the algorithm
  output/    LEAN writes its result JSON here, which we parse

Data comes from the SAME ThetaData key and the SAME snapshots the native engine
consumes. One subscription, one price source — which is what makes a divergence
between the engines attributable to engine behaviour rather than to two
different feeds.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.backtest.engines.lean import DATA_ROOT, LEAN_ROOT, _docker_running
from catalyst.core.interfaces.engine import BacktestEngine, EngineResult

logger = logging.getLogger(__name__)

IMAGE = "quantconnect/lean:latest"
OUTPUT_ROOT = LEAN_ROOT / "output"

#: Minimal LEAN engine config. Every handler below is the local/offline variant
#: — nothing here reaches out to QuantConnect.
BASE_CONFIG: dict = {
    "environment": "backtesting",
    "algorithm-type-name": "CatalystMirror",
    "algorithm-language": "Python",
    "algorithm-location": "/Lean/Algorithm.Python/main.py",
    "data-folder": "/Data",
    "results-destination-folder": "/Results",
    "log-handler": "QuantConnect.Logging.CompositeLogHandler",
    "messaging-handler": "QuantConnect.Messaging.Messaging",
    "job-queue-handler": "QuantConnect.Queues.JobQueue",
    "api-handler": "QuantConnect.Api.Api",
    "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
    "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
    "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
    "alpha-handler": "QuantConnect.Lean.Engine.Alphas.DefaultAlphaHandler",
    "data-channel-provider": "DataChannelProvider",
    "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
    "data-aggregator": "QuantConnect.Lean.Engine.DataFeeds.AggregationManager",
    "job-user-id": "0",
    "api-access-token": "",
    "job-organization-id": "",
    "debugging": False,
    "close-automatically": True,
}


class LeanDockerEngine(BacktestEngine):
    """LEAN via its own container. The second independent engine."""

    name = "lean"
    verifies = ("fully independent event loop with corporate actions (splits), "
                "early assignment/exercise, dividends and margin modelling")

    def __init__(self, timeout: int = 2400) -> None:
        self._timeout = timeout
        self._project = LEAN_ROOT / "project"

    # -- availability ------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        ok, why = _docker_running()
        if not ok:
            return False, why
        if not (self._project / "main.py").exists():
            return False, (f"no algorithm at {self._project}/main.py — run "
                           "`uv run python -m catalyst.runners.lean_setup`")
        if not DATA_ROOT.exists() or not any(DATA_ROOT.rglob("*.csv")):
            return False, (f"no converted ThetaData under {DATA_ROOT} — run "
                           "`uv run python -m catalyst.runners.lean_setup --convert`")
        if not _image_present():
            return False, (f"{IMAGE} not pulled — run `docker pull {IMAGE}` "
                           "(one-time, several GB)")
        return True, f"{why}; {IMAGE} present (no QuantConnect account needed)"

    # -- run ---------------------------------------------------------------
    def run(self, strategy, start: date, end: date, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost: bool = False) -> EngineResult:
        ok, reason = self.available()
        if not ok:
            return EngineResult(engine=self.name, error=reason)

        out_dir = OUTPUT_ROOT
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = dict(BASE_CONFIG)
        # LEAN charges commission through its own brokerage model; the
        # zero-cost twin removes it there rather than in our cost code, so the
        # diagnostic stays an independent measurement.
        config["transaction-model"] = ("QuantConnect.Orders.Fees.ConstantFeeModel"
                                       if zero_cost else "")
        cfg_path = LEAN_ROOT / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2))

        root = Path.cwd()
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{(root / DATA_ROOT).resolve()}:/Data",
            "-v", f"{(root / self._project).resolve()}/main.py:/Lean/Algorithm.Python/main.py",
            "-v", f"{(root / out_dir).resolve()}:/Results",
            "-v", f"{(root / cfg_path).resolve()}:/Lean/Launcher/config.json",
            IMAGE,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self._timeout)
        except subprocess.TimeoutExpired:
            return EngineResult(engine=self.name, error=f"LEAN exceeded {self._timeout}s")
        except Exception as e:                          # noqa: BLE001
            return EngineResult(engine=self.name, error=f"{type(e).__name__}: {e}")

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            return EngineResult(engine=self.name,
                                error=f"lean container exited {proc.returncode}: {tail}")
        return _parse_output(out_dir, self.name)


def _image_present() -> bool:
    exe = shutil.which("docker")
    if exe is None:
        return False
    r = subprocess.run([exe, "images", "-q", IMAGE], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _parse_output(out_dir: Path, engine: str) -> EngineResult:
    """Read LEAN's own equity curve and statistics."""
    files = sorted(out_dir.glob("*.json"))
    payloads = []
    for f in files:
        try:
            payloads.append(json.loads(f.read_text()))
        except Exception:                               # noqa: BLE001
            continue
    if not payloads:
        return EngineResult(engine=engine,
                            error=f"LEAN wrote no parseable result into {out_dir}")

    curve: dict[pd.Timestamp, float] = {}
    stats: dict = {}
    for payload in payloads:
        charts = payload.get("charts") or payload.get("Charts") or {}
        equity = (charts.get("Strategy Equity") or {}).get("series") or \
                 (charts.get("Strategy Equity") or {}).get("Series") or {}
        series = equity.get("Equity") or equity.get("equity") or {}
        for pt in (series.get("values") or series.get("Values") or []):
            try:
                if isinstance(pt, dict):
                    curve[pd.Timestamp(pt["x"], unit="s")] = float(pt["y"])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    curve[pd.Timestamp(pt[0], unit="s")] = float(pt[1])
            except Exception:                           # noqa: BLE001
                continue
        stats.update(payload.get("statistics") or payload.get("Statistics") or {})

    if not curve:
        return EngineResult(engine=engine,
                            error="LEAN result contained no equity series",
                            diagnostics={"lean_statistics": stats})
    return EngineResult(engine=engine, equity=pd.Series(curve).sort_index(),
                        trades=[],
                        diagnostics={"lean_statistics": stats, "independent": True})

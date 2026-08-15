"""archive — retire the active strategy without losing anything.

    uv run python -m catalyst.runners.archive_strategy --name long_options \
        --verdict "no alpha; leveraged beta only"

One command does all of it:

1. moves ``strategies/active/<name>.py`` to ``strategies/archive/<name>/``
2. moves ``results/active/<name>/`` to ``results/archive/<name>/``
3. writes a ``strategy.json`` stub — date tested, one-line verdict, key metrics
   and the Engine C baseline it was measured against
4. drops it from active runs while leaving it importable and re-runnable

**Never deletes.** Every archived strategy stays runnable through the *current*
pipeline, so when the pipeline improves, old results can be regenerated and
compared apples-to-apples instead of being trusted from a stale artifact. That
is the difference between an archive and a graveyard.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from catalyst.reporting.report import ENGINE_C_BASELINE_ANNUAL
from catalyst.strategies.registry import META_FILENAME, StrategyMeta

ACTIVE_DIR = Path("src/catalyst/strategies/active")
ARCHIVE_DIR = Path("src/catalyst/strategies/archive")
RESULTS_ACTIVE = Path("results/active")
RESULTS_ARCHIVE = Path("results/archive")


def _move(src: Path, dst: Path) -> None:
    """git mv when tracked, plain mv otherwise — history preserved where it exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(src)],
                             capture_output=True).returncode == 0
    if tracked:
        subprocess.run(["git", "mv", str(src), str(dst)], check=True)
    else:
        shutil.move(str(src), str(dst))


def archive(name: str, verdict: str, *, avg_monthly: float | None = None,
            notes: str = "") -> Path:
    dest = ARCHIVE_DIR / name
    if dest.exists():
        raise SystemExit(f"'{name}' is already archived at {dest} — refusing to overwrite")

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "__init__.py").touch()

    module = ACTIVE_DIR / f"{name}.py"
    if module.exists():
        _move(module, dest / f"{name}.py")
    else:
        print(f"note: no active module {module} (already moved?)")

    # Results travel with the code; a verdict without its artifacts is a rumour.
    src_results = RESULTS_ACTIVE / name
    metrics: dict = {}
    if src_results.exists():
        _move(src_results, RESULTS_ARCHIVE / name)
        report = RESULTS_ARCHIVE / name / "report.json"
        if report.exists():
            try:
                data = json.loads(report.read_text())
                full = next((s for s in data.get("segments", [])
                             if s["segment"] == "full" and s["cost_profile"] == "real"), None)
                test = next((s for s in data.get("segments", [])
                             if s["segment"] == "test" and s["cost_profile"] == "real"), None)
                if full:
                    avg_monthly = avg_monthly if avg_monthly is not None else full["avg_monthly_return"]
                    metrics = {
                        "full_avg_monthly": full["avg_monthly_return"],
                        "full_max_drawdown": full["max_drawdown"],
                        "n_trades": full["n_trades"],
                        "test_avg_monthly": test["avg_monthly_return"] if test else None,
                        "cost_gap": data.get("cost_gap"),
                    }
                verdict = verdict or data.get("verdict", "")
            except Exception as e:                 # noqa: BLE001
                print(f"note: could not read {report}: {e}")

    meta = StrategyMeta(
        name=name,
        module=f"catalyst.strategies.archive.{name}.{name}",
        status="archived",
        date_tested=str(date.today()),
        verdict=verdict,
        avg_monthly_return=avg_monthly,
        baseline_annual=ENGINE_C_BASELINE_ANNUAL,
        notes=notes,
        key_metrics=metrics,
    )
    meta.save(dest)
    print(f"archived '{name}':")
    print(f"  code    -> {dest}")
    print(f"  results -> {RESULTS_ARCHIVE / name}")
    print(f"  meta    -> {dest / META_FILENAME}")
    print(f"  verdict : {verdict or '(none)'}")
    print("  still runnable through the current pipeline; nothing was deleted.")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--verdict", default="", help="one line; read from report.json if omitted")
    ap.add_argument("--avg-monthly", type=float, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    archive(args.name, args.verdict, avg_monthly=args.avg_monthly, notes=args.notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())

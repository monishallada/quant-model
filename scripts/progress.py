#!/usr/bin/env python3
"""Live progress bar for long backtest runs (Gate 2 / Gate 3).

Reads two observable signals, no cooperation from the run needed:
  1. the run's log (auto-discovered) — completed segments + current phase
  2. data_cache/greeks_fo filenames — the newest cached session date shows
     where in the historical timeline the current segment is, and sampling
     that position over time yields a live rate -> ETA

Usage:
    python3 scripts/progress.py            # live bar, refreshes every 5s
    python3 scripts/progress.py --once     # single snapshot
Stdlib only — runs outside the project venv.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data_cache" / "greeks_fo"
LOG_GLOB = "/private/tmp/claude-501/*/*/tasks/*.output"

# Gate 2 segment plan (must mirror gate2_runner defaults).
START, END = date(2018, 1, 1), date(2026, 8, 1)
BOUNDARY = START + timedelta(days=int((END - START).days * 0.70))
SEGMENTS: list[tuple[str, date, date]] = []
for engine in ("engine_a", "engine_b"):
    for signal in ("trend", "mean_reversion", "neutral"):
        SEGMENTS.append((f"{engine}+{signal}-train", START, BOUNDARY))
        SEGMENTS.append((f"{engine}+{signal}-test", BOUNDARY, END))

BAR_WIDTH = 34


def find_log() -> Path | None:
    candidates = []
    for path in glob.glob(LOG_GLOB):
        try:
            with open(path, "rb") as f:
                head = f.read(65536)
            if b"Preparing IV history" in head or b"=====" in head:
                candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    if not candidates:
        return None
    return Path(max(candidates)[1])


def read_log_state(log: Path | None) -> tuple[int, int, int]:
    """(segments_done, iv_done, iv_total).

    Segment completion ground truth = per-segment JSON artifacts (the runner
    writes one per finished segment; its stdout result blocks can lag in a
    pipe buffer). Log is used only for IV-prep phase display.
    """
    segments_done = 0
    for label, _, _ in SEGMENTS:
        if (REPO / "results" / "gate2" / f"{label}.json").exists():
            segments_done += 1
        else:
            break  # segments finish strictly in order
    if log is None:
        return segments_done, 0, 0
    try:
        text = log.read_text(errors="ignore")
    except OSError:
        return segments_done, 0, 0
    iv = re.findall(r"Preparing IV history \w+ \((\d+)/(\d+)\)", text)
    if iv:
        return segments_done, int(iv[-1][0]), int(iv[-1][1])
    return segments_done, 0, 0


def newest_session_date() -> tuple[date | None, float]:
    """(session date of newest cache write, its mtime)."""
    newest_mtime, newest_day = 0.0, None
    try:
        with os.scandir(CACHE) as entries:
            for e in entries:
                m = e.stat().st_mtime
                if m > newest_mtime:
                    parts = e.name.split("_")  # SYM_EXP_SESSION_TIME.parquet
                    if len(parts) >= 3 and len(parts[2]) == 8:
                        newest_mtime = m
                        newest_day = datetime.strptime(parts[2], "%Y%m%d").date()
    except OSError:
        pass
    return newest_day, newest_mtime


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    log = find_log()
    samples: list[tuple[float, date]] = []  # (wallclock, session_date) for rate/ETA

    while True:
        segments_done, iv_done, iv_total = read_log_state(log)
        now = time.time()
        session_day, mtime = newest_session_date()
        cache_active = session_day is not None and (now - mtime) < 600

        lines: list[str] = []
        total = len(SEGMENTS)

        if iv_total and iv_done < iv_total and segments_done == 0 and not cache_active:
            frac = iv_done / iv_total
            lines.append(f"IV-rank prep    [{bar(frac)}] {iv_done}/{iv_total} symbols")
            lines.append("Backtest segments start after IV prep completes.")
        else:
            if segments_done >= total:
                lines.append(f"ALL SEGMENTS    [{bar(1.0)}] {total}/{total} — run complete or finishing up")
            else:
                # The runner's result blocks are print()ed and can sit in a
                # stdout buffer; the cache's newest session date is ground
                # truth. If it lies beyond the nominal segment's range, later
                # segments have silently completed — advance the index.
                idx = segments_done
                inferred = False
                if cache_active:
                    while idx < total - 1 and session_day > SEGMENTS[idx][2]:
                        idx += 1
                        inferred = True
                label, s, e = SEGMENTS[idx]
                seg_frac = 0.0
                if cache_active and s <= session_day <= e:
                    seg_frac = (session_day - s).days / max((e - s).days, 1)
                    samples.append((now, session_day))
                    samples[:] = [x for x in samples if now - x[0] < 1800][-200:]
                if inferred:
                    label += " (inferred — runner output buffered)"
                segments_done = idx
                overall = (segments_done + seg_frac) / total
                lines.append(f"OVERALL         [{bar(overall)}] {overall:5.1%}  ({segments_done}/{total} segments done)")
                seg_line = f"SEGMENT {segments_done + 1:>2}/{total}  [{bar(seg_frac)}] {seg_frac:5.1%}  {label}"
                if cache_active:
                    seg_line += f"  @ {session_day}"
                    # Rate from oldest vs newest sample >= 60s apart.
                    eta_txt = "estimating…"
                    if len(samples) >= 2 and now - samples[0][0] >= 60:
                        (t0, d0), (t1, d1) = samples[0], samples[-1]
                        days_per_sec = (d1 - d0).days / max(t1 - t0, 1)
                        if days_per_sec > 0:
                            eta_txt = "~" + fmt_eta((e - session_day).days / days_per_sec)
                    seg_line += f"  ETA {eta_txt}"
                else:
                    seg_line += "  (cache-hot replay — fast)"
                lines.append(seg_line)
                lines.append(
                    "Note: the first segment per engine does the heavy data pull; "
                    "repeat segments replay from cache in minutes."
                )

        out = "\n".join(lines)
        if args.once:
            print(out)
            return
        sys.stdout.write("\x1b[2J\x1b[H")  # clear screen
        print(f"Catalyst build — backtest progress   ({datetime.now():%H:%M:%S})\n")
        print(out)
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

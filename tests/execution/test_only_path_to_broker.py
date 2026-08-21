"""The single-path-to-broker invariant, enforced instead of asserted.

The engine docstring cited this test for months while it did not exist
(audit D-057). Contract: outside the backtest engines (which own their event
loops and drive SimulatedBroker directly by design) and archived campaign
code, ExecutionEngine is the ONLY module that may call ``place_order``.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "catalyst"

#: modules allowed to call place_order, relative to src/catalyst
ALLOWED = {
    "execution/engine.py",       # the paper/live road
    "backtest/backtester.py",    # daily engine event loop (SimulatedBroker only)
    "backtest/intraday.py",      # minute engine event loop (SimulatedBroker only)
}


def _place_order_callers() -> set[str]:
    callers: set[str] = set()
    for py in SRC.rglob("*.py"):
        rel = py.relative_to(SRC).as_posix()
        if rel.startswith("strategies/archive/") or rel.startswith("brokers/"):
            continue  # archive is frozen history; brokers IMPLEMENT the method
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "place_order"):
                callers.add(rel)
    return callers


def test_only_sanctioned_modules_place_orders():
    extra = _place_order_callers() - ALLOWED
    assert not extra, (
        f"New place_order call site(s) outside the sanctioned paths: {sorted(extra)}. "
        "Paper/live orders must route through ExecutionEngine.")

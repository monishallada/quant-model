"""AST import-gateway tests: every F2 breach bypass must now be caught.

ALL fixture code here is synthetic source text written into ``tmp_path``
trees and handed to ``scan_research_side(root=...)`` — nothing is written
into ``src/edge``, nothing is imported or executed, and no network is
touched. The live-tree test scans the real package read-only.

Each bypass from the breach report gets its own fixture asserting the EXACT
Violation(s) produced: importlib.import_module, ``__import__``,
semicolon-joined imports, parenthesized multiline from-imports, and plain
imports in the previously-unscanned packages (regime/runners/validation).
Detail strings are asserted literally on purpose: a referee's messages are
part of its contract, and silent drift should fail loudly.
"""

from __future__ import annotations

from pathlib import Path

from edge.validation.gateway import (
    KIND_BACKEND_IMPORT,
    KIND_DYNAMIC_IMPORT,
    KIND_EXEC_EVAL,
    KIND_FORBIDDEN_IMPORT,
    KIND_PARSE_ERROR,
    Violation,
    scan_research_side,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Detail strings mirrored literally (NOT imported from the module under test:
# importing them back would make these assertions tautological).
_WHY = (
    "a static scanner cannot see through dynamic imports, "
    "so their mere presence in research-side code is a violation"
)


def _forbidden_data_detail(module: str) -> str:
    return (
        f"import of '{module}': only edge.data.loader is a sanctioned "
        "data path; feeds, bars, and backend modules are off limits"
    )


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Materialize a fixture tree laid out like ``src/edge`` under tmp_path."""
    root = tmp_path / "edge_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for relpath, source in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        pkg = target.parent
        while pkg != root:
            init = pkg / "__init__.py"
            if not init.exists():
                init.write_text("")
            pkg = pkg.parent
        target.write_text(source)
    return root


# ---------------------------------------------------------------------------
# (a) The live tree is clean
# ---------------------------------------------------------------------------


def test_live_tree_scans_clean() -> None:
    """The shipped research-side tree has zero gateway violations."""
    assert scan_research_side() == []


def test_default_root_is_the_live_edge_package() -> None:
    """No-arg scan and an explicit src/edge scan see the same tree."""
    explicit = scan_research_side(REPO_ROOT / "src" / "edge")
    assert scan_research_side() == explicit == []


# ---------------------------------------------------------------------------
# (b) Every bypass from the breach report is caught
# ---------------------------------------------------------------------------


def test_importlib_import_module_bypass_flagged(tmp_path: Path) -> None:
    """The exact F2 exploit: importlib.import_module of the backend module."""
    root = _tree(
        tmp_path,
        {
            "signals/sneak.py": (
                "import importlib\n"
                "\n"
                'feeds = importlib.import_module("edge.data.feeds.alpaca_ticks")\n'
            )
        },
    )
    assert scan_research_side(root) == [
        Violation(
            file="signals/sneak.py",
            line=1,
            kind=KIND_DYNAMIC_IMPORT,
            detail=f"import of 'importlib': {_WHY}",
        ),
        Violation(
            file="signals/sneak.py",
            line=3,
            kind=KIND_DYNAMIC_IMPORT,
            detail=f"use of 'importlib': {_WHY}",
        ),
    ]


def test_dunder_import_bypass_flagged(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {"features/sneak.py": 'ticks = __import__("edge.data.feeds.alpaca_ticks")\n'},
    )
    assert scan_research_side(root) == [
        Violation(
            file="features/sneak.py",
            line=1,
            kind=KIND_DYNAMIC_IMPORT,
            detail=f"use of '__import__': {_WHY}",
        ),
    ]


def test_semicolon_joined_import_flagged(tmp_path: Path) -> None:
    """Two statements on one physical line; the grep gateway saw only one."""
    root = _tree(
        tmp_path,
        {"research/sneak.py": "import os; import edge.data.feeds.alpaca_ticks\n"},
    )
    assert scan_research_side(root) == [
        Violation(
            file="research/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.feeds.alpaca_ticks"),
        ),
    ]


def test_parenthesized_multiline_from_import_flagged(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            "execution/sneak.py": (
                "from edge.data.feeds import (\n    alpaca_ticks,\n)\n"
            )
        },
    )
    assert scan_research_side(root) == [
        Violation(
            file="execution/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.feeds"),
        ),
    ]


def test_previously_unscanned_packages_are_scanned(tmp_path: Path) -> None:
    """regime/runners/validation were blind spots; plain imports now flag."""
    root = _tree(
        tmp_path,
        {
            "regime/sneak.py": "import edge.data.bars.builders\n",
            "runners/sneak.py": "from edge.data.bars import builders\n",
            "validation/sneak.py": "import edge.data.feeds.alpaca_ticks\n",
        },
    )
    assert scan_research_side(root) == [
        Violation(
            file="regime/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.bars.builders"),
        ),
        Violation(
            file="runners/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.bars"),
        ),
        Violation(
            file="validation/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.feeds.alpaca_ticks"),
        ),
    ]


def test_relative_import_resolves_and_is_flagged(tmp_path: Path) -> None:
    """``from ...data.feeds import ...`` resolves to edge.data.feeds."""
    root = _tree(
        tmp_path,
        {"signals/deep/sneak.py": "from ...data.feeds import alpaca_ticks\n"},
    )
    assert scan_research_side(root) == [
        Violation(
            file="signals/deep/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.feeds"),
        ),
    ]


def test_from_edge_data_import_submodule_flagged(tmp_path: Path) -> None:
    """``from edge.data import feeds`` imports the submodule; flagged."""
    root = _tree(tmp_path, {"risk/sneak.py": "from edge.data import feeds\n"})
    assert scan_research_side(root) == [
        Violation(
            file="risk/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=_forbidden_data_detail("edge.data.feeds"),
        ),
    ]


def test_backend_name_import_flagged(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {"signals/sneak.py": "from edge.data.loader import SyntheticBackend\n"},
    )
    assert scan_research_side(root) == [
        Violation(
            file="signals/sneak.py",
            line=1,
            kind=KIND_BACKEND_IMPORT,
            detail=(
                "import of backend name 'SyntheticBackend' from "
                "'edge.data.loader': backends must not be touched directly; "
                "construct EdgeDataLoader instead"
            ),
        ),
    ]


def test_star_import_from_loader_flagged(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"signals/sneak.py": "from edge.data.loader import *\n"})
    assert scan_research_side(root) == [
        Violation(
            file="signals/sneak.py",
            line=1,
            kind=KIND_BACKEND_IMPORT,
            detail=(
                "star-import from 'edge.data.loader' smuggles backend names; "
                "import EdgeDataLoader explicitly"
            ),
        ),
    ]


def test_catalyst_import_flagged(tmp_path: Path) -> None:
    root = _tree(
        tmp_path, {"contracts/sneak.py": "import catalyst.data.alpaca_history\n"}
    )
    assert scan_research_side(root) == [
        Violation(
            file="contracts/sneak.py",
            line=1,
            kind=KIND_FORBIDDEN_IMPORT,
            detail=(
                "import of 'catalyst.data.alpaca_history': catalyst providers "
                "must not be imported by edge research code; go through "
                "edge.data.loader.EdgeDataLoader"
            ),
        ),
    ]


def test_exec_eval_flagged(tmp_path: Path) -> None:
    """exec/eval can synthesize imports from strings; both are flagged."""
    root = _tree(
        tmp_path,
        {
            "risk/sneak.py": (
                'exec("import edge.data.feeds.alpaca_ticks")\n'
                'bars = eval("__imp" "ort__")("edge.data.bars.builders")\n'
            )
        },
    )
    assert scan_research_side(root) == [
        Violation(
            file="risk/sneak.py",
            line=1,
            kind=KIND_EXEC_EVAL,
            detail=(
                "call or reference to 'exec': exec/eval can synthesize "
                "imports a static scanner cannot see"
            ),
        ),
        Violation(
            file="risk/sneak.py",
            line=2,
            kind=KIND_EXEC_EVAL,
            detail=(
                "call or reference to 'eval': exec/eval can synthesize "
                "imports a static scanner cannot see"
            ),
        ),
    ]


def test_sys_modules_access_flagged(tmp_path: Path) -> None:
    """Reading an already-imported feed out of sys.modules is a bypass."""
    root = _tree(
        tmp_path,
        {
            "features/sneak.py": (
                "import sys\n"
                "\n"
                'feeds = sys.modules["edge.data.feeds.alpaca_ticks"]\n'
            )
        },
    )
    assert scan_research_side(root) == [
        Violation(
            file="features/sneak.py",
            line=3,
            kind=KIND_DYNAMIC_IMPORT,
            detail=f"use of 'sys.modules': {_WHY}",
        ),
    ]


def test_unparseable_file_is_a_violation(tmp_path: Path) -> None:
    """A file the scanner cannot parse cannot be certified clean."""
    root = _tree(tmp_path, {"signals/broken.py": "def broken(:\n    pass\n"})
    violations = scan_research_side(root)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.file == "signals/broken.py"
    assert violation.kind == KIND_PARSE_ERROR
    assert violation.detail.startswith("unparseable file (")


# ---------------------------------------------------------------------------
# Sanctioned paths stay open; the data layer itself is exempt
# ---------------------------------------------------------------------------


def test_sanctioned_loader_imports_scan_clean(tmp_path: Path) -> None:
    """EdgeDataLoader via edge.data.loader is the open, sanctioned path."""
    root = _tree(
        tmp_path,
        {
            "signals/ok.py": (
                "from edge.data.loader import EdgeDataLoader\n"
                "from edge.data import loader\n"
                "import edge.data.loader\n"
            )
        },
    )
    assert scan_research_side(root) == []


def test_data_package_is_exempt(tmp_path: Path) -> None:
    """data/ is the fenced-off layer, not a consumer; it is never scanned."""
    root = _tree(
        tmp_path,
        {
            "data/feeds/alpaca_ticks.py": (
                "import importlib\n"
                'httpx = importlib.import_module("httpx")\n'
            )
        },
    )
    assert scan_research_side(root) == []

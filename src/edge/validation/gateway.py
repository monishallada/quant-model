"""AST import gateway: the static referee over research-side data access.

DESIGN PRINCIPLE (the honest threat model): in-process Python cannot be
sealed against code with filesystem write access. What this module — and the
referee layer it belongs to — actually enforces is:

1. ACCIDENTAL lockbox/registry violations are impossible through any public
   API: no plain ``import``, ``from``-import (single-line, semicolon-joined,
   or parenthesized multiline), or relative import in research-side code can
   reach a raw data module without :func:`scan_research_side` reporting it.
2. DELIBERATE bypass requires a conspicuous act — using a banned dynamic
   primitive, editing this referee, or planting research code inside the
   exempt ``data/`` package — that is itself flagged here or glaring in
   review, and that leaves evidence in more than one place (this scan, the
   diff, and the lockbox layer's burn-on-forge audit files).
3. Residual risks are DOCUMENTED below, not hidden.

What is scanned
---------------
Every ``*.py`` under the edge package root EXCEPT the ``data/`` package
itself (the data layer is the party being fenced off, not a consumer), using
:mod:`ast` — never regex over source lines. Parsing the real syntax tree is
what closes the historical bypasses: ``importlib.import_module(...)``,
``__import__(...)``, semicolon-joined statements, parenthesized multiline
``from``-imports, and relative imports all produce ordinary AST nodes. The
scan is default-deny by directory: a brand-new package is scanned the moment
it exists, so there is no such thing as a "not yet scanned" package
(the F2 breach class). A file that cannot be parsed is a violation, not a
skip — an unscannable file cannot be certified clean.

What is banned in research-side code
------------------------------------
- Any import resolving into ``edge.data.*`` other than the sanctioned
  gateway module ``edge.data.loader`` (so ``edge.data.feeds.*``,
  ``edge.data.bars.*``, the loader's concrete backend module
  ``edge.data.feeds.alpaca_ticks``, and any future backend module are all
  off limits), and any import of ``catalyst`` — research reaches data only
  through :class:`edge.data.loader.EdgeDataLoader`.
- Importing the loader's backend machinery by name
  (``DataBackend`` / ``SyntheticBackend`` / ``FeedsBackend``) or
  star-importing from the loader module, which would smuggle those names
  in. Research code that needs the point-in-time feed kinds uses the
  sanctioned constructor ``EdgeDataLoader.with_feeds`` instead.
- Dynamic import primitives — ``importlib`` (imported or referenced) and
  ``__import__`` — BANNED OUTRIGHT, and ``sys.modules`` access likewise.
  Rationale: a static scanner cannot see through a dynamic import, so the
  mere presence of the primitive in research-side code is the violation;
  there is no "safe" use to whitelist.
- ``exec`` / ``eval`` (called OR referenced — aliasing ``f = eval`` would
  defeat a call-only check), which can synthesize any of the above from
  strings the scanner cannot evaluate.

Research-side packages (:data:`RESEARCH_SIDE_PACKAGES`) are the intended
audience of the ban, but the rules apply to every scanned file — including
``core/`` and the package root — so nothing rides in under a new or
"infrastructure" directory.

RESIDUAL RISKS (documented, not defended)
-----------------------------------------
- Attribute-laundered primitives: ``getattr(builtins, "__imp" "ort__")``,
  ``vars()``, or reaching ``eval`` through an object attribute are invisible
  to any static scan. This is exactly why the named primitives are banned on
  sight rather than analyzed.
- A sanctioned ``import edge.data.loader`` grants attribute access to
  ``SyntheticBackend`` at runtime; the scan flags the import of backend
  NAMES, not attribute access. The lockbox wall (re-read per call) remains
  the runtime backstop.
- Code physically placed inside ``data/`` is exempt by design; moving
  research logic there is a conspicuous, review-visible act.
- Editing this module, the tests, or ``config/edge.yaml`` defeats the
  referee; those edits are diff-visible and the lockbox layer's
  burn-on-forge sentinels record key misuse independently.
- Non-``.py`` execution vectors (``.pth`` files, conftest monkeypatching,
  compiled extensions) are outside the scanner's reach.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, NamedTuple

#: The package the scanned root directory represents. Fixture trees handed to
#: ``scan_research_side(root=...)`` are treated as this package too, so
#: relative imports resolve identically to the live tree.
ROOT_PACKAGE: Final[str] = "edge"

#: Top-level directories under the root exempt from scanning: the data layer
#: is the thing being fenced off, not a consumer of the fence.
EXEMPT_DIRS: Final[frozenset[str]] = frozenset({"data"})

#: The packages the dynamic-import ban is aimed at. Informational: the scan
#: applies every rule to EVERY non-exempt file (default-deny), so a new
#: package is covered the moment it appears on disk.
RESEARCH_SIDE_PACKAGES: Final[tuple[str, ...]] = (
    "signals",
    "features",
    "regime",
    "research",
    "runners",
    "contracts",
    "execution",
    "risk",
    "validation",
)

#: The single sanctioned data-access module for research-side code.
SANCTIONED_DATA_MODULE: Final[str] = "edge.data.loader"

#: The loader's concrete backend module (also covered by the feeds ban, but
#: named explicitly: it is the module the F2 breach imported).
LOADER_BACKEND_MODULE: Final[str] = "edge.data.feeds.alpaca_ticks"

#: Backend machinery that must never be imported by name from the loader.
BACKEND_NAMES: Final[frozenset[str]] = frozenset(
    {"DataBackend", "SyntheticBackend", "FeedsBackend"}
)

_DYNAMIC_IMPORT_NAMES: Final[frozenset[str]] = frozenset({"importlib", "__import__"})
_EXEC_EVAL_NAMES: Final[frozenset[str]] = frozenset({"exec", "eval"})

# Violation kinds.
KIND_FORBIDDEN_IMPORT: Final[str] = "forbidden-import"
KIND_BACKEND_IMPORT: Final[str] = "backend-import"
KIND_DYNAMIC_IMPORT: Final[str] = "dynamic-import"
KIND_EXEC_EVAL: Final[str] = "exec-eval"
KIND_PARSE_ERROR: Final[str] = "parse-error"

_DYNAMIC_WHY: Final[str] = (
    "a static scanner cannot see through dynamic imports, "
    "so their mere presence in research-side code is a violation"
)


class Violation(NamedTuple):
    """One gateway finding: where it is, what class of breach, and why."""

    file: str  # path relative to the scanned root, posix-style
    line: int  # 1-indexed line of the offending node
    kind: str  # one of the KIND_* constants
    detail: str  # human-readable explanation naming the offending token


def _banned_module(resolved: str) -> tuple[str, str] | None:
    """Return ``(kind, detail)`` if importing module ``resolved`` is banned."""
    if resolved == "catalyst" or resolved.startswith("catalyst."):
        return (
            KIND_FORBIDDEN_IMPORT,
            f"import of '{resolved}': catalyst providers must not be imported by "
            "edge research code; go through edge.data.loader.EdgeDataLoader",
        )
    if resolved == "importlib" or resolved.startswith("importlib."):
        return (KIND_DYNAMIC_IMPORT, f"import of '{resolved}': {_DYNAMIC_WHY}")
    if resolved == "edge.data" or resolved == SANCTIONED_DATA_MODULE:
        return None
    if resolved.startswith(SANCTIONED_DATA_MODULE + "."):
        return None  # a name imported from the loader; names vetted separately
    if resolved.startswith("edge.data."):
        return (
            KIND_FORBIDDEN_IMPORT,
            f"import of '{resolved}': only {SANCTIONED_DATA_MODULE} is a sanctioned "
            "data path; feeds, bars, and backend modules are off limits",
        )
    return None


def _package_parts(py: Path, root: Path) -> list[str]:
    """Dotted-package parts that relative imports in ``py`` resolve against."""
    rel = py.relative_to(root)
    return [ROOT_PACKAGE, *rel.parts[:-1]]


def _resolve_from_base(node: ast.ImportFrom, package: list[str]) -> str | None:
    """Resolve an ImportFrom's base module to an absolute dotted path.

    Returns ``None`` for a relative import that escapes the root package —
    that is an ``ImportError`` at runtime, not a reachable data path.
    """
    if node.level == 0:
        return node.module or ""
    strip = node.level - 1
    if strip > len(package):
        return None
    base = package[: len(package) - strip] if strip else list(package)
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def _scan_file(py: Path, root: Path) -> list[Violation]:
    rel = py.relative_to(root).as_posix()
    try:
        tree = ast.parse(py.read_bytes(), filename=str(py))
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None) or 1
        return [
            Violation(
                file=rel,
                line=line,
                kind=KIND_PARSE_ERROR,
                detail=f"unparseable file ({exc}): an unscannable file cannot be certified clean",
            )
        ]
    package = _package_parts(py, root)
    found: list[Violation] = []
    seen: set[Violation] = set()

    def emit(line: int, kind: str, detail: str) -> None:
        violation = Violation(file=rel, line=line, kind=kind, detail=detail)
        if violation not in seen:
            seen.add(violation)
            found.append(violation)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _banned_module(alias.name)
                if hit is not None:
                    emit(node.lineno, *hit)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(node, package)
            if base is None:
                continue
            hit = _banned_module(base)
            if hit is not None:
                emit(node.lineno, *hit)
                continue
            if base in (SANCTIONED_DATA_MODULE, "edge.data"):
                for alias in node.names:
                    if alias.name == "*":
                        emit(
                            node.lineno,
                            KIND_BACKEND_IMPORT,
                            f"star-import from '{base}' smuggles backend names; "
                            "import EdgeDataLoader explicitly",
                        )
                    elif alias.name in BACKEND_NAMES:
                        emit(
                            node.lineno,
                            KIND_BACKEND_IMPORT,
                            f"import of backend name '{alias.name}' from '{base}': "
                            "backends must not be touched directly; "
                            "construct EdgeDataLoader instead",
                        )
            for alias in node.names:
                if alias.name == "*":
                    continue
                hit = _banned_module(f"{base}.{alias.name}" if base else alias.name)
                if hit is not None:
                    emit(node.lineno, *hit)
        elif isinstance(node, ast.Name):
            if node.id in _DYNAMIC_IMPORT_NAMES:
                emit(
                    node.lineno,
                    KIND_DYNAMIC_IMPORT,
                    f"use of '{node.id}': {_DYNAMIC_WHY}",
                )
            elif node.id in _EXEC_EVAL_NAMES:
                emit(
                    node.lineno,
                    KIND_EXEC_EVAL,
                    f"call or reference to '{node.id}': exec/eval can synthesize "
                    "imports a static scanner cannot see",
                )
        elif isinstance(node, ast.Attribute):
            if (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                emit(
                    node.lineno,
                    KIND_DYNAMIC_IMPORT,
                    f"use of 'sys.modules': {_DYNAMIC_WHY}",
                )
    return found


def scan_research_side(root: Path | str | None = None) -> list[Violation]:
    """Scan every non-exempt ``*.py`` under ``root`` for gateway violations.

    ``root`` is a directory laid out like ``src/edge`` (it is treated as the
    ``edge`` package regardless of its on-disk name, so fixture trees resolve
    relative imports exactly like the live tree). ``None`` scans the live
    edge package this module is installed in. Returns violations sorted by
    ``(file, line, kind)``; an empty list means the tree is clean.
    """
    root_dir = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    violations: list[Violation] = []
    for py in sorted(root_dir.rglob("*.py")):
        rel_parts = py.relative_to(root_dir).parts
        if rel_parts[0] in EXEMPT_DIRS or "__pycache__" in rel_parts:
            continue
        violations.extend(_scan_file(py, root_dir))
    return sorted(violations, key=lambda v: (v.file, v.line, v.kind))

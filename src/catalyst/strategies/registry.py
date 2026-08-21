"""Strategy registry and metadata.

Adding a strategy is: drop one file in ``strategies/active/`` implementing the
``Strategy`` interface, and register it here. Nothing else in the repository
changes — not the pipeline, not the risk layer, not the report.

The ``StrategyMeta`` record is also what the deployment gate reads. ``validated``
and ``paper_tested`` are set by tooling from actual run results, never typed in
by hand at deploy time, so "has this been paper-tested?" is answered by history
rather than by the person who wants to deploy it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
import logging
from pathlib import Path
from typing import Callable

from catalyst.core.interfaces import Strategy

# anchored to the package, not the process cwd (audit D-163)
ARCHIVE_ROOT = Path(__file__).resolve().parent / "archive"
RESULTS_ROOT = Path("results")
META_FILENAME = "strategy.json"


@dataclass
class StrategyMeta:
    """What we know about a strategy, independent of its code."""

    name: str
    module: str                       # import path of the builder
    status: str = "active"            # active | archived
    date_tested: str | None = None
    verdict: str | None = None
    avg_monthly_return: float | None = None
    baseline_annual: float | None = None      # Engine C reference at time of test
    validated: bool = False
    paper_tested: bool = False
    notes: str = ""
    key_metrics: dict = field(default_factory=dict)

    def save(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        p = root / META_FILENAME
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    @classmethod
    def load(cls, path: Path) -> StrategyMeta:
        return cls(**json.loads(path.read_text()))


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
_BUILDERS: dict[str, Callable[..., Strategy]] = {}
_META: dict[str, StrategyMeta] = {}


def register(meta: StrategyMeta, builder: Callable[..., Strategy]) -> None:
    _BUILDERS[meta.name] = builder
    _META[meta.name] = meta


def registry() -> dict[str, StrategyMeta]:
    """Metadata with the promotion ledger merged over code-declared defaults.

    A strategy's own source may not claim it is validated: whatever it declares
    is overwritten by the on-disk evidence, which only the pipeline and real
    paper sessions can write.
    """
    _ensure_loaded()
    from catalyst.strategies.promotion import PromotionRecord

    out: dict[str, StrategyMeta] = {}
    for name, meta in _META.items():
        rec = PromotionRecord.load(name)
        merged = StrategyMeta(**{**asdict(meta),
                                 "validated": rec.validated,
                                 "paper_tested": rec.paper_tested,
                                 # verdict/avg were stale strategy.json values
                                 # while the ledger moved on (audit D-153)
                                 "verdict": rec.validated_verdict or meta.verdict,
                                 "avg_monthly_return": (
                                     rec.validated_avg_monthly
                                     if rec.validated_avg_monthly is not None
                                     else meta.avg_monthly_return)})
        out[name] = merged
    return out


def load_strategy(name: str, cfg) -> Strategy:
    _ensure_loaded()
    if name not in _BUILDERS:
        # Archived strategies are discovered from disk as METADATA only — their
        # modules are not imported at startup, so no builder is registered yet.
        # Import on demand so an archived campaign stays re-runnable, which is
        # the repo's standing promise about archived code.
        meta = _META.get(name)
        if meta is not None and meta.module:
            import importlib
            importlib.import_module(meta.module)
    if name not in _BUILDERS:
        raise KeyError(f"unknown strategy '{name}'; known: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg)


logger = logging.getLogger(__name__)

_loaded = False


def _ensure_loaded() -> None:
    """Import strategy modules so their register() calls run.

    Import failures are surfaced rather than swallowed: a strategy that cannot
    be imported must not silently vanish from the registry, because the deploy
    gate would then report it as 'unknown' rather than 'broken'.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib

    # Auto-discover: dropping a file in strategies/active/ is enough. A
    # strategy that fails to import must surface loudly rather than silently
    # vanish — the deploy gate would otherwise call it "unknown" when it is
    # actually broken, and the operator would go looking in the wrong place.
    active_dir = Path(__file__).parent / "active"
    for py in sorted(active_dir.glob("*.py")):
        if py.stem.startswith("_"):
            continue
        importlib.import_module(f"catalyst.strategies.active.{py.stem}")

    # Archived strategies stay runnable: their metadata is discovered from disk
    # so they can be re-run through the current pipeline and compared.
    for meta_path in ARCHIVE_ROOT.glob(f"*/{META_FILENAME}"):
        try:
            m = StrategyMeta.load(meta_path)
        except Exception as e:                   # noqa: BLE001
            # an unreadable strategy.json silently VANISHED the strategy from
            # the registry (audit D-228) — surface it
            logger.error("unreadable archived strategy metadata %s: %s "
                         "— strategy NOT registered", meta_path, e)
            continue
        _META.setdefault(m.name, m)

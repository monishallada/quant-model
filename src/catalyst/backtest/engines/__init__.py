"""Backtest engines. The native one is the reference; the others cross-check it."""

from catalyst.backtest.engines.lean import LeanEngine, ThetaToLeanWriter
from catalyst.backtest.engines.lean_docker import LeanDockerEngine
from catalyst.backtest.engines.native import NativeEngine
from catalyst.backtest.engines.nautilus import NautilusEngine

#: Order matters: `native` is the reference every other engine is compared to,
#: because it is the only one that owns risk sizing, the cost model and exits.
DEFAULT_ENGINES = ("native", "nautilus", "lean")

_REGISTRY = {"native": NativeEngine, "nautilus": NautilusEngine,
             "lean": LeanDockerEngine, "lean_cli": LeanEngine}


def build_engines(names=DEFAULT_ENGINES):
    """Instantiate engines by name. Unknown names fail loudly."""
    out = []
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"unknown engine '{n}'; known: {sorted(_REGISTRY)}")
        out.append(_REGISTRY[n]())
    return out


__all__ = ["DEFAULT_ENGINES", "LeanDockerEngine", "LeanEngine", "NativeEngine",
           "NautilusEngine",
           "ThetaToLeanWriter", "build_engines"]

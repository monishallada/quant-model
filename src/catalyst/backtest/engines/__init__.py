"""Backtest engines. The native one is the reference; the others cross-check it."""

from catalyst.backtest.engines.lean import LeanEngine, ThetaToLeanWriter
from catalyst.backtest.engines.lean_docker import LeanDockerEngine
from catalyst.backtest.engines.native import NativeEngine
from catalyst.backtest.engines.nautilus import NautilusEngine

#: The default flow is the NATIVE engine alone — the validated pipeline every
#: archived campaign was measured with, and the only engine that owns risk
#: sizing, the cost model and exits.
#:
#: `nautilus` and `lean` remain fully wired and are opt-in via
#: `build_engines((...))`. They were moved out of the default path because a
#: cross-check is only worth running when it can actually run: Nautilus does
#: not support CATALYST cadence, and LEAN needs a per-strategy mirror algorithm
#: plus converted chain data. An engine that refuses on every run adds noise to
#: the report without adding evidence.
DEFAULT_ENGINES = ("native",)

#: Opt in explicitly when a strategy is one the second engines can mirror.
ALL_ENGINES = ("native", "nautilus", "lean")

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


__all__ = ["ALL_ENGINES", "DEFAULT_ENGINES", "LeanDockerEngine", "LeanEngine", "NativeEngine",
           "NautilusEngine",
           "ThetaToLeanWriter", "build_engines"]

"""Signal registry and THE HYPOTHESIS CONTRACT's enforcement points.

Registry
--------
:func:`register` is a class decorator that admits a concrete
:class:`~edge.signals.base.Signal` subclass under its unique ``name`` after
validating every class-level declaration — including the hypothesis, via
:func:`~edge.signals.base.require_hypothesis` (empty/whitespace/short raises
:class:`~edge.signals.base.MissingHypothesis` AT REGISTRATION, not at first
run). :func:`get`, :func:`create`, :func:`list_signals`, :func:`enable`,
:func:`disable`, and :func:`is_enabled` manage the catalog; every registered
signal is independently instantiable and independently backtestable.
Duplicate names are rejected (:class:`DuplicateSignal`) — two ideas sharing
a name would silently merge their trial histories.

THE HYPOTHESIS CONTRACT (``run_signal``)
----------------------------------------
:func:`run_signal` is the engine-facing entry point: thin orchestration
here, the backtest engine does the heavy lifting via the ``backtest``
callable. Its one hard job is ordering:

1. Re-validate the hypothesis (:class:`MissingHypothesis` refusal — even
   for a signal that somehow skipped registration).
2. ``record_trial()`` to the append-only
   :class:`~edge.research.registry.TrialRegistry` — hypothesis, full config,
   config hash — BEFORE the backtest runs.
3. Only then call the backtest.

The trial line is written first, so an experiment whose backtest crashes,
is aborted, or whose result is never looked at STILL exists in the registry
— that is the point: the multiple-testing count (deflated Sharpe's
``n_trials``) counts experiments attempted, not experiments admired. The
pre-run line's ``result`` is ``{"status": "pending"}`` and — the registry
being append-only — is never rewritten; completed metrics are recorded
downstream as their own artifacts.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from edge.research.registry import TrialRecord, TrialRegistry
from edge.signals.base import (
    ALL_REGIMES,
    MissingHypothesis,
    Signal,
    SignalConfig,
    require_hypothesis,
)

__all__ = [
    "DuplicateSignal",
    "MissingHypothesis",  # re-exported: run_signal's refusal lives in base
    "UnknownSignal",
    "create",
    "disable",
    "enable",
    "get",
    "is_enabled",
    "list_signals",
    "register",
    "run_signal",
    "trial_config_for",
]


class DuplicateSignal(ValueError):
    """A second signal class claimed an already-registered name."""


class UnknownSignal(KeyError):
    """No registered signal carries the requested name."""


@dataclass
class _Entry:
    """One catalog row: the signal class and its enable/disable toggle."""

    cls: type[Signal]
    enabled: bool = True


#: The catalog: name -> entry. Module-level by design — one process, one
#: signal namespace — with test seams (:func:`clear_registry`) below.
_SIGNALS: dict[str, _Entry] = {}


# ----------------------------------------------------------------- register


def _validate_signal_cls(signal_cls: type[Signal]) -> None:
    """Enforce every class-level declaration before a class enters the catalog."""
    if not isinstance(signal_cls, type) or not issubclass(signal_cls, Signal):
        raise TypeError(f"@register expects a Signal subclass; got {signal_cls!r}")
    if inspect.isabstract(signal_cls):
        raise TypeError(
            f"{signal_cls.__qualname__} is abstract (no on_bar implementation?) "
            "and cannot be registered"
        )
    name = getattr(signal_cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{signal_cls.__qualname__} must declare a non-empty class-level `name`"
        )
    require_hypothesis(
        getattr(signal_cls, "hypothesis", None), owner=f"signal {name!r}"
    )
    warmup = getattr(signal_cls, "warmup_bars", None)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            f"signal {name!r}: warmup_bars must be an int >= 0; got {warmup!r}"
        )
    allowed = getattr(signal_cls, "allowed_regimes", None)
    if allowed != ALL_REGIMES:
        ok = (
            isinstance(allowed, (set, frozenset))
            and len(allowed) > 0
            and all(isinstance(b, str) and b.strip() for b in allowed)
        )
        if not ok:
            raise ValueError(
                f"signal {name!r}: allowed_regimes must be ALL_REGIMES or a "
                f"non-empty set of bucket names; got {allowed!r}"
            )
    config_type = getattr(signal_cls, "config_type", None)
    if not (isinstance(config_type, type) and issubclass(config_type, SignalConfig)):
        raise ValueError(
            f"signal {name!r}: config_type must be a SignalConfig subclass; "
            f"got {config_type!r}"
        )


def register(signal_cls: type[Signal]) -> type[Signal]:
    """Class decorator admitting a concrete Signal subclass to the catalog.

    Validates name, hypothesis (:class:`MissingHypothesis` if empty,
    whitespace, or under the minimum length — enforced HERE, not at first
    run), warmup, regime declaration, and config type; rejects duplicate
    names with :class:`DuplicateSignal`. Signals start enabled. Returns the
    class unchanged so plain-decorator use keeps the name bound.
    """
    _validate_signal_cls(signal_cls)
    name = signal_cls.name
    if name in _SIGNALS:
        raise DuplicateSignal(
            f"signal name {name!r} is already registered by "
            f"{_SIGNALS[name].cls.__qualname__}; names are unique — two ideas "
            "sharing one name would merge their trial histories"
        )
    _SIGNALS[name] = _Entry(cls=signal_cls)
    return signal_cls


# ------------------------------------------------------------------ catalog


def _entry(name: str) -> _Entry:
    try:
        return _SIGNALS[name]
    except KeyError:
        raise UnknownSignal(
            f"no signal registered under {name!r}; known: {sorted(_SIGNALS)}"
        ) from None


def get(name: str) -> type[Signal]:
    """The registered class for ``name``; :class:`UnknownSignal` if absent."""
    return _entry(name).cls


def create(name: str, config: SignalConfig | None = None) -> Signal:
    """Instantiate the registered signal ``name`` with ``config``.

    ``None`` builds the signal's default config — every registered signal is
    independently instantiable and backtestable from its name alone.
    """
    return get(name)(config)


def list_signals(*, enabled_only: bool = False) -> list[str]:
    """Sorted registered names; ``enabled_only=True`` filters disabled ones."""
    return sorted(
        name
        for name, entry in _SIGNALS.items()
        if entry.enabled or not enabled_only
    )


def enable(name: str) -> None:
    """Allow ``name`` to be scheduled by engines that honor the toggle."""
    _entry(name).enabled = True


def disable(name: str) -> None:
    """Bench ``name`` without deleting it — its trial history stays attached."""
    _entry(name).enabled = False


def is_enabled(name: str) -> bool:
    """Whether ``name`` is currently enabled; :class:`UnknownSignal` if absent."""
    return _entry(name).enabled


def clear_registry() -> None:
    """Empty the catalog. Test seam — production code never unregisters."""
    _SIGNALS.clear()


def _snapshot() -> dict[str, _Entry]:
    """Copy the catalog state (test seam; pair with :func:`_restore`)."""
    return {name: _Entry(cls=e.cls, enabled=e.enabled) for name, e in _SIGNALS.items()}


def _restore(snapshot: dict[str, _Entry]) -> None:
    """Replace the catalog state with ``snapshot`` (test seam)."""
    _SIGNALS.clear()
    _SIGNALS.update(snapshot)


# --------------------------------------------------------------- run_signal


def trial_config_for(
    signal: Signal, run_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The FULL config dict recorded (and hashed) for one signal trial.

    Captures everything that defines the experiment: signal name and class,
    the signal's own config (``model_dump(mode="json")`` so the dict is
    canonically hashable), regime gating, warmup, and any engine-side
    ``run_config`` (dates, universe, cost scenario). Two trials with equal
    dicts hash identically and the second is flagged ``duplicate_of``.
    """
    allowed = signal.allowed_regimes
    config: dict[str, Any] = {
        "signal": getattr(signal, "name", None) or type(signal).__name__,
        "signal_class": f"{type(signal).__module__}.{type(signal).__qualname__}",
        "signal_config": signal.config.model_dump(mode="json"),
        "allowed_regimes": ALL_REGIMES if allowed == ALL_REGIMES else sorted(allowed),
        "warmup_bars": signal.warmup_bars,
    }
    if run_config:
        config["run"] = dict(run_config)
    return config


def run_signal(
    signal: Signal,
    backtest: Callable[[Signal], Any],
    trials: TrialRegistry,
    *,
    run_config: Mapping[str, Any] | None = None,
    ts: datetime | None = None,
) -> tuple[TrialRecord, Any]:
    """Record the trial, THEN run the backtest — in that order, always.

    The engine-facing entry point. ``backtest`` is the engine's heavy
    lifting (any callable taking the signal and returning its result);
    ``trials`` is the append-only trial registry; ``run_config`` is
    engine-side experiment config folded into the recorded dict; ``ts``
    (tz-aware) stamps the trial line, defaulting to the wall clock.

    Refuses with :class:`MissingHypothesis` — before recording anything —
    when the signal's hypothesis is empty, whitespace, or under the minimum
    length. Otherwise the trial line (hypothesis, full config, config hash,
    ``result={"status": "pending"}``) is appended BEFORE ``backtest`` is
    called, so the experiment exists in the registry even if the backtest
    raises or its result is never looked at — attempted experiments are what
    the multiple-testing count counts. Exceptions from ``backtest``
    propagate untouched; on success returns ``(trial_record, result)``.
    """
    if not isinstance(signal, Signal):
        raise TypeError(f"run_signal expects a Signal instance; got {signal!r}")
    hypothesis = require_hypothesis(
        getattr(signal, "hypothesis", None),
        owner=f"signal {getattr(signal, 'name', type(signal).__name__)!r}",
    )
    config = trial_config_for(signal, run_config)
    record = trials.record_trial(
        hypothesis=hypothesis,
        config=config,
        result={"status": "pending"},
        ts=ts,
    )
    result = backtest(signal)
    return record, result

"""Signal ABC and the hypothesis contract's vocabulary.

A signal is not a pattern — it is a FALSIFIABLE CLAIM about a market
mechanism: what structural flow creates the mispricing, who is on the other
side of the trade, and why they keep paying. That claim lives in
:attr:`Signal.hypothesis` and is REQUIRED: the registry refuses to register a
signal without one (:func:`edge.signals.registry.register`), and the
engine-facing :func:`edge.signals.registry.run_signal` refuses to run one —
in both cases by raising :class:`MissingHypothesis`. The floor is
:data:`MIN_HYPOTHESIS_CHARS` characters after stripping whitespace: long
enough that "momentum works" cannot sneak through as a hypothesis.

Contract pieces defined here:

* :class:`Signal` — the ABC every signal implements. Class-level
  declarations ``name``, ``hypothesis``, ``allowed_regimes``,
  ``warmup_bars``, ``config_type``; one method ``on_bar(ctx)`` returning
  :class:`~edge.core.events.SignalEvent` lists. Signals never size positions
  (risk does) and never touch data feeds (the engine's ctx serves
  point-in-time frames sourced through
  :class:`edge.data.loader.EdgeDataLoader`).
* :class:`SignalConfig` — pydantic base for per-signal configs: frozen,
  ``extra='forbid'`` so a typo'd parameter is an error, never a silent
  default.
* :class:`SignalContext` — the protocol of the ctx the engine hands
  ``on_bar``: the just-closed bar, the decision instant, and point-in-time
  frame access (every frame the ctx serves obeys
  ``available_at <= decision_ts``; the engine builds it on the loader, so
  signals cannot peek).
* :data:`ALL_REGIMES` / :func:`regime_allowed` — regime gating vocabulary:
  a signal declares the regime buckets it may trade in, or ``ALL``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from edge.core.events import BarEvent, SignalEvent

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ALL_REGIMES",
    "MIN_HYPOTHESIS_CHARS",
    "MissingHypothesis",
    "Signal",
    "SignalConfig",
    "SignalContext",
    "regime_allowed",
    "require_hypothesis",
]

#: Sentinel value of :attr:`Signal.allowed_regimes` meaning "trade in every
#: regime bucket" — the alternative is a non-empty set of bucket names.
ALL_REGIMES: Final[str] = "ALL"

#: Minimum length of a hypothesis AFTER stripping whitespace. Short enough
#: to write in one honest sentence, long enough that a slogan cannot pass.
MIN_HYPOTHESIS_CHARS: Final[int] = 50


class MissingHypothesis(ValueError):
    """A signal has no admissible hypothesis — registration/run is refused.

    Raised when a hypothesis is absent, empty, whitespace, or shorter than
    :data:`MIN_HYPOTHESIS_CHARS` stripped characters. An admissible
    hypothesis states the market mechanism, who is on the other side of the
    trade, and why they keep paying.
    """


def require_hypothesis(text: object, owner: str = "signal") -> str:
    """Return ``text`` if it is an admissible hypothesis, else refuse.

    Admissible: a ``str`` whose whitespace-stripped length is at least
    :data:`MIN_HYPOTHESIS_CHARS`. Anything else — ``None``, empty,
    whitespace, or too short — raises :class:`MissingHypothesis` naming
    ``owner``. Both enforcement points (registration and ``run_signal``)
    funnel through this one validator, so they cannot drift apart.
    """
    if not isinstance(text, str) or len(text.strip()) < MIN_HYPOTHESIS_CHARS:
        got = f"{len(text.strip())} stripped chars" if isinstance(text, str) else repr(text)
        raise MissingHypothesis(
            f"{owner}: hypothesis is required and must state the market mechanism, "
            f"who is on the other side, and why they keep paying — at least "
            f"{MIN_HYPOTHESIS_CHARS} characters after stripping whitespace (got {got})"
        )
    return text


class SignalConfig(BaseModel):
    """Base for per-signal parameter configs.

    Frozen (no runtime mutation of a swept parameter) and ``extra='forbid'``
    (a misspelled parameter is a :class:`pydantic.ValidationError`, never a
    silently ignored key). Every signal declares its own subclass and points
    :attr:`Signal.config_type` at it; ``model_dump(mode="json")`` of the
    config is what gets hashed into the trial registry line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


@runtime_checkable
class SignalContext(Protocol):
    """What the engine hands :meth:`Signal.on_bar` each bar.

    The ctx is the signal's ONLY window onto data: the engine builds it on
    :class:`edge.data.loader.EdgeDataLoader`, so every frame it serves is
    point-in-time — rows satisfy ``available_at <= decision_ts`` — and
    signals physically cannot look ahead. Signals never construct a ctx or a
    loader themselves.
    """

    @property
    def decision_ts(self) -> datetime:
        """The decision instant (tz-aware America/New_York): the bar's close."""
        ...

    @property
    def bar(self) -> BarEvent:
        """The just-completed bar that triggered this ``on_bar`` call."""
        ...

    def history(self, symbol: str, n_bars: int) -> "pd.DataFrame":
        """Up to the last ``n_bars`` completed bars for ``symbol``, oldest first."""
        ...

    def frame(self, kind: str, symbol: str) -> "pd.DataFrame":
        """A point-in-time feed frame (``short_ratio``/``vol_indices``/...).

        Rows are pre-filtered to ``available_at <= decision_ts`` by the
        engine's loader; the signal sees only what a live trader could have.
        """
        ...


class Signal(ABC):
    """One tradeable idea: a hypothesis, a config, and an ``on_bar`` rule.

    Concrete signals make four class-level declarations and implement one
    method::

        @register                      # edge.signals.registry
        class GapContinuation(Signal):
            name = "gap_continuation"
            hypothesis = "...mechanism, counterparty, why they keep paying..."
            allowed_regimes = {"trend_up", "trend_down"}   # or ALL_REGIMES
            warmup_bars = 30
            config_type = GapContinuationConfig

            def on_bar(self, ctx): ...

    Signals emit :class:`~edge.core.events.SignalEvent` opinions only —
    sizing belongs to risk, fills to execution. Every signal is
    independently instantiable (``cls(config)`` or ``registry.create``) and
    independently backtestable (``registry.run_signal``); the class-level
    declarations are validated at registration, and the hypothesis contract
    is re-enforced at run time.
    """

    #: Unique registry key for this signal; non-empty, validated at registration.
    name: ClassVar[str]
    #: The falsifiable claim: mechanism, counterparty, why they keep paying.
    #: Required — see :func:`require_hypothesis` and :class:`MissingHypothesis`.
    hypothesis: ClassVar[str]
    #: Regime buckets this signal may trade in: a non-empty set of bucket
    #: names, or :data:`ALL_REGIMES` (the default) for no gating.
    allowed_regimes: ClassVar[frozenset[str] | set[str] | str] = ALL_REGIMES
    #: Completed bars the engine must feed before ``on_bar`` output counts.
    warmup_bars: ClassVar[int] = 0
    #: This signal's config class; instances of it parameterize the signal.
    config_type: ClassVar[type[SignalConfig]] = SignalConfig

    def __init__(self, config: SignalConfig | None = None) -> None:
        """Bind a validated config; ``None`` builds ``config_type()`` defaults.

        A config of the wrong type is a ``TypeError`` — a swept parameter
        set must pass the SAME schema (``extra='forbid'``) as a hand-written
        one, so accepting a stray model here would bypass that.
        """
        cfg = self.config_type() if config is None else config
        if not isinstance(cfg, self.config_type):
            raise TypeError(
                f"{type(self).__name__} expects a {self.config_type.__name__} "
                f"config; got {type(cfg).__name__}"
            )
        self.config: SignalConfig = cfg

    @abstractmethod
    def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
        """React to a completed bar with zero or more directional opinions.

        ``ctx`` is the engine-provided :class:`SignalContext` (point-in-time
        frames + helpers). Return an empty list for "no opinion"; conviction
        scaling and sizing happen downstream in risk, never here.
        """
        ...


def regime_allowed(signal: Signal | type[Signal], bucket: str) -> bool:
    """True if ``signal`` may trade while the regime classifier says ``bucket``.

    ``ALL_REGIMES`` allows every bucket; otherwise membership in the
    declared set decides. Works on classes and instances alike.
    """
    allowed = signal.allowed_regimes
    if allowed == ALL_REGIMES:
        return True
    return bucket in allowed

"""Shared domain types — the vocabulary every layer speaks.

Strategies express intent in these types and nothing else; the pipeline
translates them into orders, risk decisions, fills and metrics. Keeping them
in one place is what lets a strategy be written once and run in backtest,
paper and live without modification.
"""

from catalyst.core.types.models import *  # noqa: F401,F403
from catalyst.core.types.models import __all__  # noqa: F401

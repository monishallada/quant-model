"""Edge: event-driven trading-research platform.

A clean-room sibling of ``catalyst`` — same repo, separate package, no shared
mutable state. Backtest, paper, and live consumers run the SAME code path over
an ``EventClock``; only the clock and the feed differ by environment.
"""

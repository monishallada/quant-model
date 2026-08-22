"""MOSAIC volatility & distribution engines (v16).

Pure, stateless math on point-in-time inputs: every function takes explicit
history and returns a value — no I/O, no caches, no hidden state. Fitting
happens in the strategy layer on visible-only data.
"""

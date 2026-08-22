# Edge platform entrypoints. paper/live are gated stubs: they print that
# deployment requires the operator's explicit approval and exit 1.

.PHONY: backtest paper live

backtest:
	uv run python -m edge.runners.backtest --help

paper:
	uv run python -m edge.runners.paper

live:
	uv run python -m edge.runners.live

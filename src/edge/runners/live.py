"""Live-trading entrypoint stub: ``python -m edge.runners.live``.

Gated: live deployment requires the operator's explicit approval. Real money
never moves on the say-so of code alone — this runner refuses to start and
exits nonzero until the operator signs off.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge.runners.live",
        description="Run the edge platform against a live account (gated stub).",
    )
    parser.add_argument("--config", default=None, help="path to edge.yaml (default: config/edge.yaml)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Refuse to run: live deployment is gated behind operator approval."""
    build_parser().parse_args(argv)
    print(
        "edge live runner is GATED: deployment requires the operator's explicit "
        "approval, a validated strategy, and a completed paper campaign. Refusing to start.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

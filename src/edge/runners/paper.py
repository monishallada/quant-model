"""Paper-trading entrypoint stub: ``python -m edge.runners.paper``.

Gated: paper deployment requires the operator's explicit approval. Until a
strategy has passed the validation gates and the operator signs off, this
runner refuses to start and exits nonzero.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge.runners.paper",
        description="Run the edge platform against a paper account (gated stub).",
    )
    parser.add_argument("--config", default=None, help="path to edge.yaml (default: config/edge.yaml)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Refuse to run: paper deployment is gated behind operator approval."""
    build_parser().parse_args(argv)
    print(
        "edge paper runner is GATED: deployment requires the operator's explicit "
        "approval and a strategy that has passed the validation gates. Refusing to start.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

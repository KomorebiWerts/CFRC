#!/usr/bin/env python3
"""CLI wrapper for public-safe STATE-Bench true-live score aggregation."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.state_bench.true_live_score_aggregate import (
    build_report,
    main as _aggregate_main,
)  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    _aggregate_main(argv)


__all__ = ["build_report", "main"]


if __name__ == "__main__":
    main()

"""STATE-Bench dataset adapters for shared PAPC RCBE abstractions.

Dataset-specific adapters (e.g. the travel surface) live in subpackages here and
project runtime-visible STATE-Bench structure into the benchmark-independent
method types in :mod:`papc.certspec.state_write_contract`. Per the repo boundary,
adapter / projection / restoration-cost / arm logic belongs in this package, not
in experiment scripts under ``scripts/``.
"""

from __future__ import annotations

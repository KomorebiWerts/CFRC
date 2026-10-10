"""Pure ToolSandbox prefix/suffix scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSandboxScore:
    milestone_similarity: float
    minefield_similarity: float
    turn_count: int = 0

    @property
    def similarity(self) -> float:
        return 0.0 if self.minefield_similarity != 0 else self.milestone_similarity

    def to_json(self) -> dict[str, Any]:
        return {
            "milestone_similarity": self.milestone_similarity,
            "minefield_similarity": self.minefield_similarity,
            "similarity": self.similarity,
            "turn_count": self.turn_count,
        }


def suffix_delta(prefix: ToolSandboxScore, prefix_plus_suffix: ToolSandboxScore) -> dict[str, Any]:
    """Score a residual suffix using ToolSandbox's public minefield rule."""

    delta = round(prefix_plus_suffix.similarity - prefix.similarity, 6)
    return {
        "schema": "papc.toolsandbox.suffix_delta.v1",
        "prefix_only": prefix.to_json(),
        "prefix_plus_suffix": prefix_plus_suffix.to_json(),
        "delta_similarity": delta,
        "minefield_nullifies": (
            prefix_plus_suffix.minefield_similarity != 0
            and prefix_plus_suffix.similarity == 0.0
        ),
        "provider_calls": 0,
    }


__all__ = ["ToolSandboxScore", "suffix_delta"]

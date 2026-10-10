"""Current STATE-Bench Travel residual adapter package."""

from __future__ import annotations

from papc.certspec.structured_certificate import register_target_type

__all__ = [
    "residual_gates",
    "residual_obligations",
    "state_write_contract",
]

register_target_type("state_bench", "travel", "stateful_travel_write_target_set")

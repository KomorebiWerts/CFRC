"""Domain-independent certificate field and leakage helpers.

``SHARED_CERTIFICATE_FIELDS`` is the governance-level vocabulary that bounds
what any adapter may expose to a model-facing payload; it is NOT the literal
runtime wire schema of the four active runners (their certificate objects use
surface-local field names — verified 2026-07-10). Do not assume a unified wire
schema from this tuple.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

SHARED_CERTIFICATE_FIELDS: tuple[str, ...] = (
    "target_binding",
    "intent_scope",
    "precondition_state",
    "argument_binding",
    "commit_contract",
    "postcondition",
    "terminal_completion_plan",
    "risk",
)

SHARED_CERTIFICATE_FIELD_SET = frozenset(SHARED_CERTIFICATE_FIELDS)

FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "answer",
        "api_calls",
        "compiled_solution",
        "db_final_state",
        "direct_answer",
        "direct_outcome",
        "env_assertions",
        "evaluation_code",
        "expected_answer",
        "final_state",
        "fixed_api_args",
        "generator",
        "gold",
        "gold_action_sequence",
        "gold_api_trace",
        "ground_truth",
        "hidden",
        "private_data",
        "reference_trajectory",
        "reward_basis",
        "selected_reason",
        "solution",
        "target_answer",
        "target_state_summary",
        "task_id",
        "test_data",
    }
)

GENERIC_TASK_ID_RE = re.compile(r"\b[a-f0-9]{7}_[0-9]+\b", flags=re.IGNORECASE)


def assert_shared_certificate_surface(
    payload: Mapping[str, Any],
    *,
    allowed_extra_fields: Sequence[str] = (),
    surface_name: str = "certificate_surface",
) -> None:
    """Reject model-visible certificate payloads with non-shared top-level fields."""

    allowed = SHARED_CERTIFICATE_FIELD_SET | {
        str(field) for field in allowed_extra_fields
    }
    extra_fields = sorted(str(key) for key in payload if str(key) not in allowed)
    if extra_fields:
        raise ValueError(f"{surface_name} contains non-shared fields: {extra_fields}")


def find_forbidden_runtime_references(
    value: Any,
    *,
    forbidden_keys: set[str] | frozenset[str] = FORBIDDEN_RUNTIME_KEYS,
    task_id_pattern: re.Pattern[str] = GENERIC_TASK_ID_RE,
    forbidden_string_markers: Sequence[str] = (
        "compiled_solution",
        "direct_outcome",
        "gold_api_trace",
        "ground_truth",
        "target_state_summary",
    ),
) -> list[str]:
    """Find forbidden runtime keys or task-id-like strings in nested payloads."""

    leaks: list[str] = []
    normalized_keys = {str(key).lower() for key in forbidden_keys}
    normalized_markers = tuple(
        str(marker).lower() for marker in forbidden_string_markers
    )

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_text = str(key)
                if key_text.lower() in normalized_keys:
                    leaks.append(f"{path}.{key_text}")
                visit(child, f"{path}.{key_text}")
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            lower_item = item.lower()
            if task_id_pattern.search(item):
                leaks.append(path)
            if any(marker in lower_item for marker in normalized_markers):
                leaks.append(path)

    visit(value, "$")
    return sorted(set(leaks))

"""Registered Agent-Diff scorer corrections applied after agent execution.

The upstream evaluator remains the first scorer.  This module only repairs
three dataset assertions whose predicates conflict with deterministic backend
side effects.  It receives the hidden test id and evaluator diff after the
agent has finished; nothing here is visible to the model, contract, proposal,
or runtime.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "papc.agent_diff.scorer_corrections.v1"
SCORER_NAME = "agent_diff_evaluate_run_plus_registered_corrections_v1"

_BOX_IGNORED_CHANGE_FIELDS = frozenset(
    {
        "content_created_at",
        "content_modified_at",
        "created_at",
        "created_by",
        "etag",
        "file_version",
        "modified_at",
        "modified_by",
        "owned_by",
        "path_collection",
        "purged_at",
        "sequence_id",
        "sha1",
        "trashed_at",
    }
)

_BOX_MOVE_CORRECTIONS = {
    "box_140": {
        "assertion_index": 1,
        "file_id": "2797160615",
        "parent_predicate": ("eq", "7905906319"),
    },
    "box_126": {
        "assertion_index": 3,
        "file_id": "1364279594",
        "parent_predicate": ("ne", "1088403890"),
    },
}


def registered_scorer_correction_manifest() -> dict[str, Any]:
    """Return the scorer-bound, public registration for the three fixes."""

    return {
        "schema": SCHEMA,
        "scorer_name": SCORER_NAME,
        "runtime_visible": False,
        "application_phase": "after_official_evaluation",
        "test_ids": ["box_126", "box_140", "calendar_213"],
        "corrections": {
            "box_126": "ignore_backend_derived_path_on_file_move",
            "box_140": "ignore_backend_derived_path_on_file_move",
            "calendar_213": (
                "accept_correct_auto_inserted_calendar_list_entry_on_calendar_create"
            ),
        },
    }


def apply_registered_scorer_corrections(
    *,
    test_id: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return corrected scoring while preserving the official evaluation."""

    original = copy.deepcopy(dict(evaluation))
    if bool(original.get("passed")):
        return original

    assertion_index: int | None = None
    correction_name: str | None = None
    if test_id in _BOX_MOVE_CORRECTIONS and _box_move_path_is_only_extra_change(
        test_id=test_id,
        evaluation=original,
    ):
        assertion_index = int(
            _BOX_MOVE_CORRECTIONS[test_id]["assertion_index"]
        )
        correction_name = "box_move_derived_path_v1"
    elif test_id == "calendar_213" and _calendar_created_list_entry_is_correct(
        evaluation=original
    ):
        assertion_index = 2
        correction_name = "calendar_created_list_entry_v1"

    if (
        assertion_index is None
        or correction_name is None
        or not _assertion_failed(original, assertion_index)
    ):
        return original

    corrected = copy.deepcopy(original)
    corrected_failures = [
        failure
        for failure in _failures(original)
        if not _failure_belongs_to_assertion(failure, assertion_index)
    ]
    corrected["failures"] = corrected_failures
    corrected["score"] = _increment_assertion_score(original.get("score"))
    corrected["passed"] = not corrected_failures
    if corrected["passed"]:
        if "result_status" in corrected:
            corrected["result_status"] = "passed"
        if "status" in corrected:
            corrected["status"] = "passed"
    corrected["official_evaluation"] = {
        key: copy.deepcopy(original[key])
        for key in ("passed", "score", "failures", "result_status", "status")
        if key in original
    }
    corrected["scoring_corrections"] = {
        "schema": SCHEMA,
        "scorer_name": SCORER_NAME,
        "applied": [correction_name],
        "runtime_visible": False,
    }
    return corrected


def _box_move_path_is_only_extra_change(
    *,
    test_id: str,
    evaluation: Mapping[str, Any],
) -> bool:
    correction = _BOX_MOVE_CORRECTIONS[test_id]
    target_id = str(correction["file_id"])
    updates = [
        update
        for update in _diff_rows(evaluation, "updates")
        if update.get("__table__") == "box_files"
        and str(
            (update.get("after") or {}).get("id")
            or (update.get("before") or {}).get("id")
            or ""
        )
        == target_id
    ]
    if len(updates) != 1:
        return False

    before = _mapping(updates[0].get("before"))
    after = _mapping(updates[0].get("after"))
    changed = {
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    meaningful = changed - _BOX_IGNORED_CHANGE_FIELDS
    if meaningful != {"parent_id", "path"}:
        return False

    operator, expected = correction["parent_predicate"]
    actual_parent = str(after.get("parent_id") or "")
    actual_path = str(after.get("path") or "").rstrip("/")
    if not actual_parent or not actual_path.endswith(f"/{actual_parent}"):
        return False
    if operator == "eq":
        return actual_parent == expected
    if operator == "ne":
        return actual_parent != expected
    return False


def _calendar_created_list_entry_is_correct(
    *,
    evaluation: Mapping[str, Any],
) -> bool:
    candidates = [
        row
        for row in _diff_rows(evaluation, "inserts")
        if row.get("__table__") == "calendar_list_entries"
        and row.get("user_id") == "user_agent"
    ]
    changed_candidates = [
        row
        for row in _diff_rows(evaluation, "updates")
        if row.get("__table__") == "calendar_list_entries"
        and (
            _mapping(row.get("after")).get("user_id") == "user_agent"
            or _mapping(row.get("before")).get("user_id") == "user_agent"
        )
    ]
    if len(candidates) != 1 or changed_candidates:
        return False
    entry = candidates[0]
    return (
        entry.get("color_id") == "11"
        and entry.get("selected") is True
        and entry.get("hidden") is False
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _diff_rows(
    evaluation: Mapping[str, Any],
    bucket: str,
) -> tuple[dict[str, Any], ...]:
    diff = _mapping(evaluation.get("diff"))
    rows = diff.get(bucket)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(_mapping(row) for row in rows if isinstance(row, Mapping))


def _failures(evaluation: Mapping[str, Any]) -> list[str]:
    failures = evaluation.get("failures")
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        return []
    return [str(failure) for failure in failures]


def _failure_belongs_to_assertion(failure: str, assertion_index: int) -> bool:
    return failure.startswith(f"assertion#{assertion_index} ")


def _assertion_failed(
    evaluation: Mapping[str, Any],
    assertion_index: int,
) -> bool:
    return any(
        _failure_belongs_to_assertion(failure, assertion_index)
        for failure in _failures(evaluation)
    )


def _increment_assertion_score(value: Any) -> dict[str, Any]:
    score = _mapping(value)
    total = max(0, int(score.get("total") or 0))
    passed = min(total, max(0, int(score.get("passed") or 0)) + 1)
    return {
        **score,
        "passed": passed,
        "total": total,
        "percent": (float(passed) / total * 100.0 if total else 100.0),
    }

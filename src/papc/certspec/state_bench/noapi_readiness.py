"""Shared STATE-Bench no-API helpers used by current method-form gates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from papc.paths import DATA_ROOT

FORBIDDEN_PUBLIC_MARKERS = (
    f"{DATA_ROOT}",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    '"task_id"',
    '"task_ids"',
    '"gold"',
    '"messages"',
    '"raw_prompt"',
    '"raw_response"',
    '"tool_call_arguments"',
    '"arguments"',
    '"private_root"',
    '"private_run_root"',
    '"provider_payload"',
    '"active_runner_lines"',
)
SECRET_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")

# Scorer/gold/UX keys that an online runtime selector or honest aggregate must
# never surface in a public report. A runtime selector reading any of these
# would violate the observability boundary, so a report that carries one fails
# the selector observability audit closed. These are intentionally NOT folded
# into ``FORBIDDEN_PUBLIC_MARKERS`` so existing scored-shard reducers that legally
# read (but never publish) scorer fields keep their current public-leakage audit.
FORBIDDEN_SCORER_MARKERS = (
    '"task_completion_pass"',
    '"state_requirements_met"',
    '"task_requirements_met"',
    '"state_requirements_details"',
    '"task_requirements_details"',
    '"state_requirements_gt"',
    '"state_requirements_reasoning"',
    '"task_requirements_reasoning"',
    '"ux_score"',
    '"ux_communication_quality"',
    '"ux_friction"',
    '"ux_intent_alignment"',
    '"ux_situational_awareness"',
    '"ux_user_control"',
)


def load_json_object(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def write_json_append_only(path: Path | str, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"append-only output already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def nested(mapping_: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping_
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def public_leakage_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    blob = json.dumps(report, ensure_ascii=False, sort_keys=True)
    hits = sorted(marker for marker in FORBIDDEN_PUBLIC_MARKERS if marker in blob)
    if SECRET_KEY_PATTERN.search(blob):
        hits.append("secret_key_pattern")
    return {
        "pass": not hits,
        "forbidden_marker_hits": hits,
        "checked_for_private_paths": True,
        "checked_for_credentials": True,
        "checked_for_task_ids": True,
        "checked_for_raw_payloads": True,
    }


def selector_observability_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Audit that a selector / honest-aggregate report respects the online boundary.

    Passes only when (a) the standard public-leakage audit passes, (b) no
    scorer/gold/UX key is present anywhere in the serialized report, and (c) the
    report positively declares ``selector_observability.reads_scorer_fields`` is
    ``False``. The positive declaration makes the scorer-blind contract explicit
    and machine-checkable rather than merely absent.
    """

    base = public_leakage_audit(report)
    blob = json.dumps(report, ensure_ascii=False, sort_keys=True)
    scorer_hits = sorted(
        marker for marker in FORBIDDEN_SCORER_MARKERS if marker in blob
    )
    declared = nested(report, "selector_observability", "reads_scorer_fields")
    declares_scorer_blind = declared is False
    return {
        "pass": bool(base["pass"]) and not scorer_hits and declares_scorer_blind,
        "public_leakage_pass": base["pass"],
        "forbidden_marker_hits": base["forbidden_marker_hits"],
        "scorer_marker_hits": scorer_hits,
        "declares_scorer_blind": declares_scorer_blind,
        "checked_for_scorer_fields": True,
    }


__all__ = [
    "load_json_object",
    "write_json_append_only",
    "nested",
    "public_leakage_audit",
    "selector_observability_audit",
    "FORBIDDEN_SCORER_MARKERS",
]

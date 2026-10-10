"""Public-output hygiene helpers for current ToolSandbox artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from papc.paths import DATA_ROOT

FORBIDDEN_PUBLIC_MARKERS = (
    f"{DATA_ROOT}",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    '"command"',
    '"env_setup"',
    '"private_run_root"',
    '"task_id"',
    '"gold"',
    '"messages"',
    '"raw_prompt"',
    '"raw_response"',
    '"instruction"',
    '"evaluation_criteria"',
    '"milestones"',
    '"minefields"',
    '"scenario_name"',
    '"source_file"',
    '"source_line"',
    '"scenario_milestones"',
    '"scenario_minefields"',
    '"tool_call_arguments"',
    '"expected_tool_arguments"',
    '"target_dataframe"',
    '"snapshot_constraints"',
    "sk-",
)
SECRET_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")


def public_leakage_audit(
    report: Mapping[str, Any],
    *,
    extra_payload: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit public ToolSandbox material for private/runtime markers."""

    blob = json.dumps(
        {"report": report, "rows": list(extra_payload)},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    hits = sorted(
        _marker_label(marker)
        for marker in FORBIDDEN_PUBLIC_MARKERS
        if marker in blob
    )
    if SECRET_KEY_PATTERN.search(blob):
        hits.append("secret_key_pattern")
    return {
        "pass": not hits,
        "forbidden_marker_hits": hits,
        "checked_for_private_paths": True,
        "checked_for_credentials": True,
        "checked_for_raw_scenario_payload": True,
    }


def _marker_label(marker: str) -> str:
    return "secret_key_pattern" if marker == "sk-" else marker.strip('"')

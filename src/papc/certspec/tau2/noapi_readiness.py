"""Retail-only no-API readiness helpers for tau2.

The current tau2 package keeps one official baseline wrapper and one clean CFRC
method runner. This module only validates the public-safe retail launch guard
used by the official wrapper; old cross-domain approval/preflight audits are not
part of the active method path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from papc.paths import DATA_ROOT

SCHEMA = "papc.tau2_retail_official_launch_guard_noapi.v1"
GATE_ID = "tau2_retail_official_launch_guard_noapi"
SURFACE = "tau2_retail"
MODE = "tau2_retail_official_launch_guard_noapi"
READY_DECISION = "tau2_retail_official_launch_guard_ready_for_explicit_approval_noapi"
BLOCKED_DECISION = "blocked_tau2_retail_official_launch_guard_noapi"
EXPECTED_APPROVAL_SCOPE = "tau2_retail_official_baseline_or_final_papc_execution"
VALID_LAUNCH_ARMS = ("official_baseline", "final_papc")
SUPPORTED_LAUNCH_DOMAINS = ("retail",)

RETAIL_CONTRACT: dict[str, Any] = {
    "surface": SURFACE,
    "gate_id": GATE_ID,
    "mode": MODE,
    "expected_approval_scope": EXPECTED_APPROVAL_SCOPE,
    "required_base_task_count": 114,
    "required_agent_write_tool_count": 1,
    "required_user_write_tool_count": 0,
    "method_role": "primary_state_write_positive",
    "effect_family": "order_payment_address_return_exchange_state_write",
    "launch_role": "official_baseline_or_final_papc_row",
    "supports_final_papc_launch_arm": True,
}

FORBIDDEN_FILTER_FLAGS = ("--task-ids", "--num-tasks")
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
    '"provider_payload"',
    '"private_manifest"',
    '"private_run_root"',
    '"source_file"',
    '"source_line"',
)
SECRET_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")


def build_official_launch_report(
    *,
    tau2_args: Sequence[str],
    launch_arm: str | None,
    evaluator_model: str,
    evaluator_reasoning_effort: str | None,
    api_base_supplied: bool,
    fixed_manifest_json: Path | None = None,
    cost_ledger_json: Path | None = None,
    public_output_root: Path | None = None,
    approval_scope: str | None = None,
    launch_domain: str = "retail",
    tau_repo: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a public-safe no-API launch guard for retail official runs."""

    parsed = parse_tau2_run_args(tau2_args)
    inventory = build_tau2_domain_inventory(tau_repo=tau_repo, domain="retail")
    settings_checks = {
        "launch_domain_is_retail": launch_domain == "retail",
        "launch_arm_is_explicit_and_supported": launch_arm in VALID_LAUNCH_ARMS,
        "official_cli_entrypoint_is_tau2_run": parsed["entrypoint"] == "run",
        "domain_is_retail": parsed["flags"].get("--domain") == "retail",
        "task_split_is_base": parsed["flags"].get("--task-split-name") == "base",
        "task_filters_absent_for_official_row": not any(
            flag in parsed["present_flags"] for flag in FORBIDDEN_FILTER_FLAGS
        ),
        "agent_model_explicit": bool(parsed["flags"].get("--agent-llm")),
        "agent_reasoning_effort_explicit": _reasoning_effort_explicit(
            parsed["flags"].get("--agent-llm-args")
        ),
        "user_model_explicit": bool(parsed["flags"].get("--user-llm")),
        "user_reasoning_effort_explicit": _reasoning_effort_explicit(
            parsed["flags"].get("--user-llm-args")
        ),
        "num_trials_is_official_four": parsed["flags"].get("--num-trials") == "4",
        "save_to_configured": bool(parsed["flags"].get("--save-to")),
        "evaluator_model_explicit": bool(evaluator_model),
        "evaluator_reasoning_effort_explicit": bool(evaluator_reasoning_effort),
        "api_base_configured_for_future_execute": api_base_supplied,
        "fixed_manifest_declared": fixed_manifest_json is not None,
        "cost_ledger_declared": cost_ledger_json is not None,
        "public_output_root_declared": public_output_root is not None,
        "public_output_root_is_public": _is_public_output_root(public_output_root),
    }
    if inventory["available"]:
        settings_checks.update(_inventory_checks(inventory))
    settings_blockers = sorted(
        key for key, passed in settings_checks.items() if passed is not True
    )
    execution_blockers = list(settings_blockers)
    if approval_scope != EXPECTED_APPROVAL_SCOPE:
        execution_blockers.append("explicit_paid_approval_missing")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "gate_id": GATE_ID,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "surface": SURFACE,
        "mode": MODE,
        "api_spend_performed": False,
        "no_api_calls": True,
        "provider_calls": 0,
        "decision": READY_DECISION if not settings_blockers else BLOCKED_DECISION,
        "summary": {
            "launch_arm": launch_arm or "missing",
            "launch_domain": launch_domain,
            "settings_checks_passed": sum(
                1 for passed in settings_checks.values() if passed
            ),
            "settings_checks_total": len(settings_checks),
            "ready_for_explicit_approval_request": not settings_blockers,
            "online_experiment_ready": not execution_blockers,
            "official_cli_entrypoint": parsed["entrypoint"] or "missing",
            "domain": parsed["flags"].get("--domain") or "missing",
            "task_split": parsed["flags"].get("--task-split-name") or "missing",
            "num_trials": _safe_int(parsed["flags"].get("--num-trials")),
            "uses_task_filters": any(
                flag in parsed["present_flags"] for flag in FORBIDDEN_FILTER_FLAGS
            ),
        },
        "official_cli_contract": {
            "entrypoint": "tau2 run",
            "required_domain": "retail",
            "required_split": "base",
            "required_num_trials": 4,
            "must_not_use_task_filters": list(FORBIDDEN_FILTER_FLAGS),
            "must_keep_official_tau2_runner": True,
            "required_wrapper": "scripts/tau2/papc_tau2_official_run_with_evaluator_patch.py",
            "wrapper_patch_scope": "evaluator_and_interface_models_only",
        },
        "arm_contract": {
            "requested_launch_arm": launch_arm or "missing",
            "official_baseline_and_final_papc_are_distinct_rows": True,
            "is_official_baseline_row": launch_arm == "official_baseline",
            "is_final_papc_row": launch_arm == "final_papc",
            "expected_approval_scope": EXPECTED_APPROVAL_SCOPE,
        },
        "method_alignment_contract": {
            "domain": "retail",
            "method_role": RETAIL_CONTRACT["method_role"],
            "effect_family": RETAIL_CONTRACT["effect_family"],
            "launch_role": RETAIL_CONTRACT["launch_role"],
            "base_split_only": True,
            "uses_task_ids_at_runtime": False,
            "uses_gold_or_direct_outcome_at_runtime": False,
            "uses_fixed_tool_args_at_runtime": False,
            "publishes_private_paths_or_task_ids": False,
        },
        "inventory_summary": _inventory_summary(inventory),
        "public_paths": {
            "fixed_manifest_json": _safe_path_label(fixed_manifest_json),
            "cost_ledger_json": _safe_path_label(cost_ledger_json),
            "public_output_root": _safe_path_label(public_output_root),
        },
        "settings_checks": settings_checks,
        "settings_blockers": settings_blockers,
        "execution_blockers": execution_blockers,
        "authorizations": {
            "authorizes_paid_run": False,
            "authorizes_online_execute": False,
            "authorizes_request_exact_execution_approval": not settings_blockers,
            "authorizes_full_tau2": False,
            "authorizes_leaderboard_claim": False,
            "authorizes_final_papc_claim": False,
        },
        "claim_boundaries": [
            "This is a no-API retail launch guard, not an online result.",
            "It does not authorize paid execution, full tau2, leaderboard claims, or measured-effect claims.",
        ],
    }
    report["report_digest"] = stable_sha256(
        {key: value for key, value in report.items() if key != "generated_at"}
    )
    leakage = public_leakage_audit(report)
    report["public_leakage_audit"] = leakage
    if leakage["pass"] is not True:
        report["decision"] = BLOCKED_DECISION
    return report


def build_tau2_domain_inventory(
    *,
    tau_repo: Path | None,
    domain: str,
) -> dict[str, Any]:
    """Return a public-safe inventory for the retail tau2 domain."""

    if tau_repo is None or domain != "retail":
        return {"available": False, "domain": domain}
    domain_root = tau_repo / "data" / "tau2" / "domains" / "retail"
    code_root = tau_repo / "src" / "tau2" / "domains" / "retail"
    tasks_path = domain_root / "tasks.json"
    split_path = domain_root / "split_tasks.json"
    tasks = _load_json_list_or_dict(tasks_path)
    split = _load_json_object(split_path)
    agent_tools = _tool_inventory(code_root / "tools.py")
    user_tools = _tool_inventory(code_root / "user_tools.py")
    return {
        "available": bool(tasks_path.exists() and tasks),
        "domain": "retail",
        "tasks_present": tasks_path.exists(),
        "split_present": split_path.exists(),
        "task_count": len(tasks),
        "base_task_count": _split_count(split, "base"),
        "split_names": sorted(split) if isinstance(split, dict) else [],
        "agent_read_tool_count": len(agent_tools.get("READ", [])),
        "agent_write_tool_count": len(agent_tools.get("WRITE", [])),
        "agent_generic_tool_count": len(agent_tools.get("GENERIC", [])),
        "user_read_tool_count": len(user_tools.get("READ", [])),
        "user_write_tool_count": len(user_tools.get("WRITE", [])),
        "user_generic_tool_count": len(user_tools.get("GENERIC", [])),
        "publishes_tool_names": False,
        "publishes_task_ids": False,
    }


def parse_tau2_run_args(args: Sequence[str]) -> dict[str, Any]:
    """Parse enough of ``tau2 run`` args for launch guard checks."""

    tokens = list(args)
    entrypoint = tokens[0] if tokens and not tokens[0].startswith("--") else ""
    flags: dict[str, Any] = {}
    present_flags: list[str] = []
    index = 1 if entrypoint else 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        flag, sep, inline_value = token.partition("=")
        present_flags.append(flag)
        if sep:
            flags[flag] = inline_value
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            flags[flag] = tokens[index + 1]
            index += 2
        else:
            flags[flag] = True
            index += 1
    return {
        "entrypoint": entrypoint,
        "flags": flags,
        "present_flags": sorted(set(present_flags)),
    }


def stable_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
        "checked_for_raw_trajectories": True,
        "checked_for_provider_payloads": True,
    }


def write_json_append_only(path: Path | str, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"append-only output already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inventory_checks(inventory: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "retail_tasks_inventory_present": bool(inventory.get("tasks_present")),
        "retail_base_split_inventory_present": bool(inventory.get("split_present")),
        "retail_base_task_count_matches_contract": int(
            inventory.get("base_task_count") or 0
        )
        >= int(RETAIL_CONTRACT["required_base_task_count"]),
        "retail_agent_write_tools_match_contract": int(
            inventory.get("agent_write_tool_count") or 0
        )
        >= int(RETAIL_CONTRACT["required_agent_write_tool_count"]),
        "retail_user_write_tools_match_contract": int(
            inventory.get("user_write_tool_count") or 0
        )
        >= int(RETAIL_CONTRACT["required_user_write_tool_count"]),
    }


def _inventory_summary(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(inventory.get("available")),
        "domain": str(inventory.get("domain") or "retail"),
        "task_count": _safe_int(inventory.get("task_count")),
        "base_task_count": _safe_int(inventory.get("base_task_count")),
        "split_names": list(inventory.get("split_names") or []),
        "agent_read_tool_count": _safe_int(inventory.get("agent_read_tool_count")),
        "agent_write_tool_count": _safe_int(inventory.get("agent_write_tool_count")),
        "agent_generic_tool_count": _safe_int(
            inventory.get("agent_generic_tool_count")
        ),
        "user_read_tool_count": _safe_int(inventory.get("user_read_tool_count")),
        "user_write_tool_count": _safe_int(inventory.get("user_write_tool_count")),
        "user_generic_tool_count": _safe_int(inventory.get("user_generic_tool_count")),
        "publishes_tool_names": False,
        "publishes_task_ids": False,
    }


def _load_json_list_or_dict(path: Path) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.values())
    return []


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _split_count(split: Mapping[str, Any], name: str) -> int:
    value = split.get(name)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return len(value)
    return 0


def _tool_inventory(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {"READ": [], "WRITE": [], "GENERIC": []}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {"READ": [], "WRITE": [], "GENERIC": []}
    tools = {"READ": [], "WRITE": [], "GENERIC": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            tool_type = _decorator_tool_type(decorator)
            if tool_type in tools:
                tools[tool_type].append(node.name)
    return {key: sorted(value) for key, value in tools.items()}


def _decorator_tool_type(decorator: ast.AST) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Name) or func.id != "is_tool" or not decorator.args:
        return None
    arg = decorator.args[0]
    if isinstance(arg, ast.Attribute):
        return arg.attr
    if isinstance(arg, ast.Name):
        return arg.id
    return None


def _reasoning_effort_explicit(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("reasoning_effort"))
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return "reasoning_effort" in value
    return isinstance(parsed, Mapping) and bool(parsed.get("reasoning_effort"))


def _safe_path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.as_posix()
    if any(marker in text for marker in (f"{DATA_ROOT}", "sk-")):
        return "redacted_private_or_sensitive_path"
    if text.startswith("outputs/papc/") or text == "outputs/papc":
        return text
    return path.name or "declared"


def _is_public_output_root(path: Path | None) -> bool:
    if path is None:
        return False
    text = path.as_posix()
    return text == "outputs/papc" or text.startswith("outputs/papc/")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "SCHEMA",
    "GATE_ID",
    "SURFACE",
    "MODE",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "EXPECTED_APPROVAL_SCOPE",
    "VALID_LAUNCH_ARMS",
    "SUPPORTED_LAUNCH_DOMAINS",
    "build_official_launch_report",
    "build_tau2_domain_inventory",
    "parse_tau2_run_args",
    "stable_sha256",
    "public_leakage_audit",
    "write_json_append_only",
]

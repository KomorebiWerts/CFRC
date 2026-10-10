#!/usr/bin/env python3
"""Private method-arm manifest adapter and approval guard for CFRC execution.

This module is deliberately no-API. It provides the small execution seam shared by
STATE-Bench CFRC domain runners: accept private ``manifest_rows`` as runner
candidate rows and enforce the exact paid-approval scope before any runner may
create a provider client.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

APPROVAL_FLAG_DEST = "approve_paid_method_arm_run"
APPROVAL_SCOPE_DEST = "approval_scope"
APPROVAL_FLAG_BLOCKER = "approve_paid_method_arm_run_required_for_execute"
APPROVAL_SCOPE_BLOCKER = "approval_scope_must_match_method_arm_scope"
TARGET_ARM_BLOCKER = "method_arm_manifest_target_arm_must_match_papc_cfrc"
MANIFEST_APPROVAL_SCOPE_BLOCKER = (
    "method_arm_manifest_approval_scope_must_match_method_arm_scope"
)
STRONG_BUDGET_BLOCKER = "method_arm_manifest_counted_strong_call_budget_must_be_one"
TARGET_METHOD_ARM = "cheap_to_papc_cfrc"
METHOD_ARM_APPROVAL_SCOPE = "append_only_private_state_bench_cfrc_method_arm_execution"
PRIVATE_MANIFEST_SCHEMA = "papc.private_cfrc_method_arm_manifest.v1"


def add_paid_approval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approve-paid-method-arm-run", action="store_true")
    parser.add_argument("--approval-scope", default="")


def ensure_paid_approval_defaults(args: argparse.Namespace) -> None:
    if not hasattr(args, APPROVAL_FLAG_DEST):
        setattr(args, APPROVAL_FLAG_DEST, False)
    if not hasattr(args, APPROVAL_SCOPE_DEST):
        setattr(args, APPROVAL_SCOPE_DEST, "")


def paid_execution_preflight_blockers(args: argparse.Namespace) -> list[str]:
    if not bool(getattr(args, "execute", False)):
        return []
    blockers: list[str] = []
    if getattr(args, APPROVAL_FLAG_DEST, False) is not True:
        blockers.append(APPROVAL_FLAG_BLOCKER)
    if str(getattr(args, APPROVAL_SCOPE_DEST, "") or "") != METHOD_ARM_APPROVAL_SCOPE:
        blockers.append(APPROVAL_SCOPE_BLOCKER)
    return blockers


def raw_manifest_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("selected_rows")
    if not isinstance(rows, list):
        rows = manifest.get("rows")
    if not isinstance(rows, list):
        rows = manifest.get("manifest_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def manifest_contract_blockers(
    manifest: Mapping[str, Any],
    *,
    expected_domain: str,
) -> list[str]:
    blockers: list[str] = []
    if str(manifest.get("domain") or expected_domain) != expected_domain:
        blockers.append("candidate_manifest_domain_mismatch")
    if is_private_method_arm_manifest(manifest):
        if str(manifest.get("target_method_arm") or "") != TARGET_METHOD_ARM:
            blockers.append(TARGET_ARM_BLOCKER)
        if str(manifest.get("approval_scope") or "") != METHOD_ARM_APPROVAL_SCOPE:
            blockers.append(MANIFEST_APPROVAL_SCOPE_BLOCKER)
        if int(manifest.get("counted_strong_call_budget_per_row") or 0) != 1:
            blockers.append(STRONG_BUDGET_BLOCKER)
    return sorted(set(blockers))


def is_private_method_arm_manifest(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("schema") == PRIVATE_MANIFEST_SCHEMA or isinstance(
        manifest.get("manifest_rows"), list
    )


def public_row_ordinal(
    item: Mapping[str, Any],
    *,
    index: int,
    expected_domain: str,
) -> str:
    raw = item.get("row")
    if isinstance(raw, str | int) and str(raw).strip():
        return str(raw).strip()
    for key in ("public_row_ordinal", "row_ordinal", "manifest_ordinal"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return f"method-arm-{expected_domain}-{index:04d}"


def source_path(item: Mapping[str, Any]) -> Path:
    return Path(str(item.get("source_path") or ""))


def task_id(item: Mapping[str, Any]) -> str:
    return str(item.get("task_id") or "").strip()


def run_index(item: Mapping[str, Any]) -> int:
    return int(item.get("run_index") or 1)


def candidate_class_label(item: Mapping[str, Any]) -> str:
    candidate = item.get("candidate_class")
    if isinstance(candidate, Mapping):
        for key in (
            "closure_class_key",
            "selector_reason",
            "required_action",
            "terminal_work_order_status",
        ):
            value = str(candidate.get(key) or "").strip()
            if value:
                return value
        return "private_method_arm_candidate_class"
    return str(candidate or "").strip()


def manifest_row_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    terminal_work_order = item.get("private_terminal_work_order")
    if not isinstance(terminal_work_order, Mapping):
        terminal_work_order = item.get("terminal_work_order")
    if isinstance(terminal_work_order, Mapping):
        metadata["private_terminal_work_order"] = copy.deepcopy(
            dict(terminal_work_order)
        )
    return metadata


def valid_runner_row_shape(item: Mapping[str, Any]) -> bool:
    return bool(task_id(item)) and source_path(item).is_absolute()


__all__ = [
    "APPROVAL_FLAG_BLOCKER",
    "APPROVAL_SCOPE_BLOCKER",
    "MANIFEST_APPROVAL_SCOPE_BLOCKER",
    "STRONG_BUDGET_BLOCKER",
    "TARGET_ARM_BLOCKER",
    "add_paid_approval_args",
    "candidate_class_label",
    "ensure_paid_approval_defaults",
    "manifest_contract_blockers",
    "is_private_method_arm_manifest",
    "METHOD_ARM_APPROVAL_SCOPE",
    "TARGET_METHOD_ARM",
    "manifest_row_metadata",
    "paid_execution_preflight_blockers",
    "public_row_ordinal",
    "raw_manifest_rows",
    "run_index",
    "source_path",
    "task_id",
    "valid_runner_row_shape",
]

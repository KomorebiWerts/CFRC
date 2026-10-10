#!/usr/bin/env python3
"""Run ToolSandbox base129 true-online CFRC."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.costing import cache_neutral_cost_usd  # noqa: E402
from papc.certspec.toolsandbox.base129_true_online_run import (  # noqa: E402
    BLOCKED_DECISION,
    EXECUTE_READY_DECISION,
    GROUNDED_AUTHORING_APPROVAL_BASIS,
    GROUNDED_AUTHORING_APPROVAL_REVISION,
    METHOD_PROFILE,
    NOAPI_READY_DECISION,
    PRIVATE_SCHEMA,
    REQUIRED_TOOL_ENV_VARS,
    SCHEMA,
    build_base129_true_online_run,
    scenario_specs_from_lower_receipts,
)
from papc.certspec.toolsandbox.public_hygiene import (  # noqa: E402
    public_leakage_audit,
)
from papc.paths import DATA_ROOT

DEFAULT_LOWER_RECEIPT_ROOT = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_official_bounds_private_20260627_codex/lower/receipts"
)
DEFAULT_TOOLSANDBOX_CHECKOUT = Path(f"{DATA_ROOT}/benchmarks/toolsandbox/repo")
DEFAULT_PRIVATE_OUTPUT_JSON = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_true_online_private_receipts_20260628_codex_v1.json"
)
DEFAULT_PUBLIC_OUTPUT_JSON = (
    _REPO / "outputs" / "papc" / "toolsandbox_base129_true_online_public_20260628_codex_v1.json"
)
DEFAULT_EPISODE_JSONL = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_true_online_episode_receipts_20260628_codex_v1.jsonl"
)
STRICT_C8_AUTHORIZED_PARALLELISM = 8
STRICT_C8_INCUMBENT_METHOD_VARIANT = "incumbent"
STRICT_C8_METHOD_VARIANT = "terminal_with_last_call"
STRICT_C8_SUPERVISOR_PLAN_SCHEMA = "papc.toolsandbox.c8_supervisor_plan.v1"
STRICT_C8_SUPERVISOR_ARTIFACT_SCHEMA = (
    "papc.toolsandbox.c8_supervisor_provider_zero.v1"
)
STRICT_C8_SUPERVISOR_REPORT_SCHEMA = "papc.toolsandbox.c8_supervisor_report.v1"
STRICT_C8_SUPERVISOR_READY_DECISION = "toolsandbox_c8_supervisor_ready_noapi"
STRICT_C8_LOCK_PATH = Path(
    f"{DATA_ROOT}/papc/.toolsandbox_c8_supervisor.lock"
)
STRICT_C8_INHERITED_LOCK_FD_ENV = "PAPC_TOOLSANDBOX_C8_LOCK_FD"
_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class _InfrastructureFailureStop(RuntimeError):
    """Stop one shard after its first durable infrastructure-failure receipt."""

    def __init__(self, *, public_ordinal: int, reasons: Sequence[str]) -> None:
        self.public_ordinal = int(public_ordinal)
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__(
            f"infrastructure failure at public ordinal {self.public_ordinal}: "
            + ",".join(self.reasons)
        )


class _SupervisorTerminationSignal(RuntimeError):
    """Route an external supervisor signal through fail-closed cleanup."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(signal.Signals(signum).name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-receipt-json", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cheap-model", default="gpt-5.4-mini")
    parser.add_argument("--cheap-reasoning-effort", default="high")
    parser.add_argument("--strong-model", default="gpt-5.5")
    parser.add_argument("--strong-reasoning-effort", default="high")
    parser.add_argument("--user-model", default="gpt-5.4")
    parser.add_argument("--user-reasoning-effort", default="high")
    parser.add_argument("--provider-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--episode-wall-timeout-seconds",
        type=float,
        default=None,
        help="Optional per-episode wall-clock timeout; records a runtime failure on expiry.",
    )
    parser.add_argument("--cap", type=int, default=129)
    parser.add_argument("--max-strong-residual-calls", type=int, default=7)
    parser.add_argument("--max-turn-steps", type=int, default=60)
    # Receipt-grounded authoring spends one provider call per native step, and
    # one more pass when the checker refuses. Both bounds are explicit so a run
    # command states its own worst-case provider cost.
    parser.add_argument("--max-authoring-rounds", type=int, default=14)
    parser.add_argument("--max-reproposal-rounds", type=int, default=2)
    parser.add_argument(
        "--terminal-with-last-call",
        action="store_true",
        help="Allow a fail-closed terminal answer on the final successful replica call.",
    )
    parser.add_argument(
        "--stop-on-first-infrastructure-error",
        action="store_true",
        help="Flush the failing episode receipt, then stop this shard immediately.",
    )
    parser.add_argument(
        "--strict-fresh-c8-preflight",
        action="store_true",
        help="Opt in to fresh-run c8 approval, environment, endpoint, and output guards.",
    )
    parser.add_argument("--lower-receipt-root", type=Path, default=DEFAULT_LOWER_RECEIPT_ROOT)
    parser.add_argument("--toolsandbox-checkout", type=Path, default=DEFAULT_TOOLSANDBOX_CHECKOUT)
    parser.add_argument("--private-output-json", type=Path, default=DEFAULT_PRIVATE_OUTPUT_JSON)
    parser.add_argument("--public-output-json", type=Path, default=DEFAULT_PUBLIC_OUTPUT_JSON)
    parser.add_argument("--episode-jsonl", type=Path, default=DEFAULT_EPISODE_JSONL)
    parser.add_argument("--planned-execute-private-output-json", type=Path)
    parser.add_argument("--planned-execute-public-output-json", type=Path)
    parser.add_argument("--gate-artifact-json", type=Path)
    parser.add_argument("--expected-gate-artifact-sha256")
    parser.add_argument("--c8-supervisor-plan-json", type=Path)
    parser.add_argument("--c8-supervisor-artifact-json", type=Path)
    parser.add_argument("--expected-c8-supervisor-artifact-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument(
        "--public-ordinals",
        default="",
        help="Comma-separated public ordinals for targeted runs; filter before cap.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.c8_supervisor_plan_json is not None:
        return _run_c8_supervisor(args)
    lock_fd: int | None = None
    try:
        if args.execute:
            lock_fd = (
                _acquire_or_inherit_c8_execution_lock()
                if args.strict_fresh_c8_preflight
                else _acquire_c8_shared_lock()
            )
        return _run_single_shard(args)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def _run_single_shard(args: argparse.Namespace) -> int:
    approval_receipt = (
        _read_json_object(args.approval_receipt_json)
        if args.approval_receipt_json is not None
        else {}
    )
    if args.strict_fresh_c8_preflight:
        if args.resume or args.rerun_failures:
            raise SystemExit(
                "--strict-fresh-c8-preflight forbids --resume and --rerun-failures"
            )
        _validate_strict_fresh_output_paths(args)
    existing_receipts = (
        _existing_receipts_for_resume(
            args.episode_jsonl,
            rerun_failures=args.rerun_failures,
        )
        if args.execute and args.resume
        else []
    )
    scenario_specs = _filter_public_ordinals(
        scenario_specs_from_lower_receipts(args.lower_receipt_root),
        public_ordinals=args.public_ordinals,
    )
    gate_bindings = _gate_bindings(
        args=args,
        approval_receipt=approval_receipt,
        scenario_specs=scenario_specs[: max(0, args.cap)],
    )
    pending_episode_count = _pending_episode_count(
        scenario_specs[: max(0, args.cap)],
        existing_receipts,
    )
    execution_environment_blockers: list[str] = []
    if args.execute and pending_episode_count:
        execution_environment_blockers.extend(
            _required_tool_env_blockers(REQUIRED_TOOL_ENV_VARS)
        )
    if args.strict_fresh_c8_preflight:
        execution_environment_blockers.extend(
            _strict_fresh_c8_blockers(
                args=args,
                approval_receipt=approval_receipt,
            )
        )
    execution_environment_blockers = sorted(set(execution_environment_blockers))
    runtime_factory = None
    proposer_factory = None
    if args.execute:
        if args.approval_receipt_json is None:
            raise SystemExit("--approval-receipt-json is required with --execute")
        if not args.api_base:
            raise SystemExit("--api-base is required with --execute")
        if args.episode_jsonl.exists() and not args.resume:
            raise SystemExit(f"--episode-jsonl exists; pass --resume or choose a new path: {args.episode_jsonl}")
        if pending_episode_count and not execution_environment_blockers:
            _verify_gate(args=args, gate_bindings=gate_bindings)
        runtime_factory, proposer_factory = _live_factories(args)
    try:
        private_report, public_report = build_base129_true_online_run(
            scenario_specs=scenario_specs,
            approval_receipt=approval_receipt,
            execute_requested=args.execute,
            runtime_factory=runtime_factory,
            strong_proposer_factory=proposer_factory,
            cap=args.cap,
            max_strong_residual_calls=args.max_strong_residual_calls,
            max_authoring_rounds=args.max_authoring_rounds,
            max_reproposal_rounds=args.max_reproposal_rounds,
            max_turn_steps=args.max_turn_steps,
            cheap_model=args.cheap_model,
            cheap_reasoning_effort=args.cheap_reasoning_effort,
            strong_model=args.strong_model,
            strong_reasoning_effort=args.strong_reasoning_effort,
            user_model=args.user_model,
            user_reasoning_effort=args.user_reasoning_effort,
            execution_environment_blockers=execution_environment_blockers,
            existing_receipts=existing_receipts,
            episode_receipt_callback=(
                _episode_receipt_callback(
                    args.episode_jsonl,
                    stop_on_first_infrastructure_error=(
                        args.stop_on_first_infrastructure_error
                    ),
                )
                if args.execute
                else None
            ),
            episode_wall_timeout_seconds=args.episode_wall_timeout_seconds,
            terminal_with_last_call=args.terminal_with_last_call,
        )
    except _InfrastructureFailureStop as exc:
        print(
            json.dumps(
                {
                    "schema": "papc.toolsandbox.infrastructure_failfast_stop.v1",
                    "decision": "runtime_failure",
                    "diagnostic_partial_only": True,
                    "public_ordinal": exc.public_ordinal,
                    "reasons": list(exc.reasons),
                    "episode_jsonl": str(args.episode_jsonl),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3
    private_report["gate_bindings"] = gate_bindings
    public_report["gate_bindings"] = gate_bindings
    if args.execute:
        public_report["executed_from_gate_sha256"] = (
            args.expected_gate_artifact_sha256
        )
        private_report["executed_from_gate_sha256"] = (
            args.expected_gate_artifact_sha256
        )
    if args.execute and args.strict_fresh_c8_preflight:
        _mark_strict_c8_supervisor_shard_reports(
            private_report=private_report,
            public_report=public_report,
        )
    public_report["public_leakage_audit"] = public_leakage_audit(public_report)
    if public_report["public_leakage_audit"]["pass"] is not True:
        raise RuntimeError("ToolSandbox public leakage audit failed")
    _write_pair_append_only(
        private_path=args.private_output_json,
        public_path=args.public_output_json,
        private_report=private_report,
        public_report=public_report,
    )
    stdout_summary = {
        "schema": SCHEMA,
        "private_schema": PRIVATE_SCHEMA,
        "method_profile": METHOD_PROFILE,
        "decision": public_report["decision"],
        "provider_calls": public_report["provider_calls"],
        "attempted_episode_count": public_report["summary"][
            "attempted_episode_count"
        ],
        "strong_residual_call_count": public_report["summary"][
            "strong_residual_call_count"
        ],
        "official_primary_metric": public_report["summary"]["official_primary_metric"],
        "official_avg_similarity_percent": public_report["summary"][
            "official_avg_similarity_percent"
        ],
        "official_avg_similarity_score": public_report["summary"][
            "official_avg_similarity_score"
        ],
        "method_complete_count": public_report["summary"]["method_complete_count"],
        "blockers": public_report["blockers"],
        "private_output": str(args.private_output_json),
        "public_output": str(args.public_output_json),
    }
    if "approval_revision_required" in public_report:
        stdout_summary["approval_revision_required"] = public_report[
            "approval_revision_required"
        ]
    if args.terminal_with_last_call:
        stdout_summary["method_variant"] = "terminal_with_last_call"
    print(json.dumps(stdout_summary, ensure_ascii=False, sort_keys=True))
    if public_report["decision"] == BLOCKED_DECISION:
        return 2 if args.execute else 1
    return 0 if public_report["decision"] in {NOAPI_READY_DECISION, EXECUTE_READY_DECISION} else 1


def _mark_strict_c8_supervisor_shard_reports(
    *,
    private_report: dict[str, Any],
    public_report: dict[str, Any],
) -> None:
    selected_count = _int(
        public_report.get("summary", {}).get("attempted_episode_count")
    )
    shard_markers = {
        "supervisor_shard_only": True,
        "diagnostic_partial_only": True,
        "requires_successful_supervisor_report": True,
    }
    private_report.update(shard_markers)
    public_report.update(shard_markers)
    authorizations = public_report.get("authorizations")
    if isinstance(authorizations, dict):
        authorizations["authorizes_method_result_claim"] = False
    paper_gate = public_report.get("paper_acceptance_gate")
    if isinstance(paper_gate, dict):
        paper_gate["evaluated"] = False
        paper_gate["not_evaluable_on_supervisor_shard"] = True
    public_report["claim_boundary"] = [
        f"This strict c8 supervisor shard contains selected n={selected_count}.",
        "It cannot authorize a method-result or paper claim on its own.",
        "It cannot be resumed, spliced, or combined without one successful exact-union supervisor report.",
        "The exact-union supervisor report still requires aggregate quality and cost admission before a claim.",
    ]


def _run_c8_supervisor(args: argparse.Namespace) -> int:
    if args.c8_supervisor_artifact_json is None:
        raise SystemExit("--c8-supervisor-artifact-json is required in supervisor mode")
    plan = _read_json_object(args.c8_supervisor_plan_json)
    prepared = _prepare_c8_supervisor(args=args, plan=plan)
    artifact = _c8_supervisor_artifact(
        plan_path=args.c8_supervisor_plan_json,
        prepared=prepared,
    )
    if not args.execute:
        if args.expected_c8_supervisor_artifact_sha256 is not None:
            raise SystemExit(
                "--expected-c8-supervisor-artifact-sha256 is execute-only"
            )
        _write_json_append_only(args.c8_supervisor_artifact_json, artifact)
        print(
            json.dumps(
                {
                    "schema": STRICT_C8_SUPERVISOR_ARTIFACT_SCHEMA,
                    "decision": STRICT_C8_SUPERVISOR_READY_DECISION,
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "scope": prepared["scope"],
                    "shard_count": len(prepared["shards"]),
                    "artifact": str(args.c8_supervisor_artifact_json),
                    "artifact_sha256": _file_sha256(
                        args.c8_supervisor_artifact_json
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    expected_sha256 = args.expected_c8_supervisor_artifact_sha256
    if expected_sha256 is None:
        raise SystemExit(
            "--expected-c8-supervisor-artifact-sha256 is required with --execute"
        )
    if _file_sha256(args.c8_supervisor_artifact_json) != expected_sha256:
        raise RuntimeError("ToolSandbox c8 supervisor artifact hash changed")
    if _read_json_object(args.c8_supervisor_artifact_json) != artifact:
        raise RuntimeError("ToolSandbox c8 supervisor artifact does not bind this plan")
    lock_fd = _acquire_c8_exclusive_lock()
    try:
        legacy_execute_pids = _existing_exact_runner_execute_pids()
        if legacy_execute_pids:
            raise RuntimeError(
                "pre-lease ToolSandbox runner execute process already active: "
                + ",".join(str(pid) for pid in legacy_execute_pids)
            )
        return _execute_c8_supervisor(
            prepared=prepared,
            supervisor_artifact_sha256=expected_sha256,
            lock_fd=lock_fd,
        )
    finally:
        os.close(lock_fd)


def _prepare_c8_supervisor(
    *,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if plan.get("schema") != STRICT_C8_SUPERVISOR_PLAN_SCHEMA:
        raise RuntimeError("ToolSandbox c8 supervisor plan schema mismatch")
    scope = plan.get("scope")
    if scope not in {"canary8", "base129"}:
        raise RuntimeError("ToolSandbox c8 supervisor scope must be canary8 or base129")
    aggregate_acceptance = _c8_aggregate_acceptance(plan=plan, scope=str(scope))
    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 8:
        raise RuntimeError("ToolSandbox c8 supervisor requires exactly eight shards")
    report_path = _plan_path(plan, "supervisor_report_json")
    expected_source_manifest = _source_manifest()
    expected_source_manifest_sha256 = _canonical_sha256(expected_source_manifest)
    expected_runner_sha256 = _file_sha256(Path(__file__))
    all_scenario_specs = scenario_specs_from_lower_receipts(args.lower_receipt_root)
    prepared_shards: list[dict[str, Any]] = []
    all_ordinals: list[int] = []
    common_config: dict[str, Any] | None = None
    common_toolsandbox_commit: str | None = None
    common_toolsandbox_worktree: dict[str, Any] | None = None
    planned_paths: list[Path] = [report_path]
    immutable_paths: list[Path] = [
        args.c8_supervisor_plan_json,
        args.c8_supervisor_artifact_json,
    ]

    for expected_index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise RuntimeError("ToolSandbox c8 supervisor shard must be an object")
        shard_index = _strict_plan_int(raw_shard, "shard_index")
        if shard_index != expected_index:
            raise RuntimeError(
                "ToolSandbox c8 supervisor shard indices must be "
                "0..7"
            )
        raw_ordinals = raw_shard.get("public_ordinals")
        if not isinstance(raw_ordinals, list) or not raw_ordinals:
            raise RuntimeError("ToolSandbox c8 supervisor shard ordinals are required")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_ordinals
        ):
            raise RuntimeError("ToolSandbox c8 supervisor ordinals must be integers")
        public_ordinals = [int(value) for value in raw_ordinals]
        if len(public_ordinals) != len(set(public_ordinals)):
            raise RuntimeError("ToolSandbox c8 supervisor shard ordinals repeat")
        if any(ordinal < 0 or ordinal > 128 for ordinal in public_ordinals):
            raise RuntimeError("ToolSandbox c8 supervisor ordinal outside base129")
        if any(ordinal % 8 != shard_index for ordinal in public_ordinals):
            raise RuntimeError("ToolSandbox c8 supervisor modulo assignment mismatch")
        cap = _strict_plan_int(raw_shard, "cap")
        strong_budget = _strict_plan_int(
            raw_shard,
            "max_strong_residual_calls",
        )
        if cap != len(public_ordinals):
            raise RuntimeError("ToolSandbox c8 supervisor shard cap mismatch")
        if strong_budget != 16 * cap:
            raise RuntimeError("ToolSandbox c8 supervisor strong budget mismatch")

        approval_path = _plan_path(raw_shard, "approval_receipt_json")
        gate_path = _plan_path(raw_shard, "gate_artifact_json")
        gate_sha256 = _strict_sha256(
            raw_shard.get("expected_gate_artifact_sha256"),
            field="expected_gate_artifact_sha256",
        )
        private_path = _plan_path(raw_shard, "private_output_json")
        public_path = _plan_path(raw_shard, "public_output_json")
        episode_path = _plan_path(raw_shard, "episode_jsonl")
        log_path = _plan_path(raw_shard, "log_path")
        planned_paths.extend([private_path, public_path, episode_path, log_path])
        immutable_paths.extend([approval_path, gate_path])
        if not approval_path.is_file() or not gate_path.is_file():
            raise RuntimeError("ToolSandbox c8 supervisor approval or gate is missing")
        if _file_sha256(gate_path) != gate_sha256:
            raise RuntimeError("ToolSandbox c8 supervisor child gate hash changed")
        approval = _read_json_object(approval_path)
        gate = _read_json_object(gate_path)
        _validate_c8_child_gate_report(gate)
        gate_bindings = gate.get("gate_bindings")
        if not isinstance(gate_bindings, dict):
            raise RuntimeError("ToolSandbox c8 child gate bindings are missing")
        config = gate_bindings.get("config")
        if not isinstance(config, dict):
            raise RuntimeError("ToolSandbox c8 child gate config is missing")
        if config.get("cap") != cap or config.get(
            "max_strong_residual_calls"
        ) != strong_budget:
            raise RuntimeError("ToolSandbox c8 plan budget differs from child gate")

        child_argv = _c8_child_argv_from_gate(
            args=args,
            shard=raw_shard,
            gate=gate,
        )
        child_args = parse_args(child_argv[2:])
        strict_blockers = _strict_fresh_c8_blockers(
            args=child_args,
            approval_receipt=approval,
        )
        if strict_blockers:
            raise RuntimeError(
                "ToolSandbox c8 child strict blockers: " + ",".join(strict_blockers)
            )
        selected_specs = _filter_public_ordinals(
            all_scenario_specs,
            public_ordinals=child_args.public_ordinals,
        )
        if [int(spec.public_ordinal) for spec in selected_specs] != public_ordinals:
            raise RuntimeError("ToolSandbox c8 child selection order mismatch")
        expected_bindings = _gate_bindings(
            args=child_args,
            approval_receipt=approval,
            scenario_specs=selected_specs[:cap],
        )
        if gate_bindings != expected_bindings:
            raise RuntimeError("ToolSandbox c8 child gate is stale or misbound")
        if gate_bindings.get("source_manifest") != expected_source_manifest:
            raise RuntimeError("ToolSandbox c8 child gate source manifest is stale")
        if gate_bindings.get("source_manifest_sha256") != (
            expected_source_manifest_sha256
        ):
            raise RuntimeError("ToolSandbox c8 child gate source hash is stale")
        runner_entry = next(
            (
                entry
                for entry in expected_source_manifest
                if entry.get("path")
                == Path(__file__).resolve().relative_to(_REPO.resolve()).as_posix()
            ),
            None,
        )
        if runner_entry is None or runner_entry.get("sha256") != expected_runner_sha256:
            raise RuntimeError("ToolSandbox c8 current runner is not source-bound")

        normalized_config = _c8_common_gate_config(config)
        if common_config is None:
            common_config = normalized_config
            common_toolsandbox_commit = str(gate_bindings["toolsandbox_commit"])
            common_toolsandbox_worktree = dict(gate_bindings["toolsandbox_worktree"])
        elif (
            normalized_config != common_config
            or gate_bindings.get("toolsandbox_commit") != common_toolsandbox_commit
            or gate_bindings.get("toolsandbox_worktree")
            != common_toolsandbox_worktree
        ):
            raise RuntimeError(
                "ToolSandbox c8 child gates do not share checkout and config"
            )

        prepared_shards.append(
            {
                "shard_index": shard_index,
                "public_ordinals": public_ordinals,
                "cap": cap,
                "max_strong_residual_calls": strong_budget,
                "approval_path": approval_path,
                "gate_path": gate_path,
                "gate_sha256": gate_sha256,
                "private_path": private_path,
                "public_path": public_path,
                "episode_path": episode_path,
                "log_path": log_path,
                "argv": child_argv,
                "approval_receipt_sha256": _canonical_sha256(approval),
                "approval_file_sha256": _file_sha256(approval_path),
                "gate_bindings": gate_bindings,
            }
        )
        all_ordinals.extend(public_ordinals)

    _validate_c8_supervisor_scope(scope=scope, ordinals=all_ordinals)
    _validate_c8_supervisor_paths(
        planned_paths=planned_paths,
        immutable_paths=immutable_paths,
    )
    return {
        "scope": scope,
        "report_path": report_path,
        "shards": prepared_shards,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "runner_sha256": expected_runner_sha256,
        "toolsandbox_commit": common_toolsandbox_commit,
        "toolsandbox_worktree": common_toolsandbox_worktree,
        "common_config": common_config,
        "aggregate_acceptance": aggregate_acceptance,
    }


def _validate_c8_child_gate_report(gate: Mapping[str, Any]) -> None:
    if (
        gate.get("schema") != SCHEMA
        or gate.get("decision") != NOAPI_READY_DECISION
        or gate.get("provider_calls") != 0
        or gate.get("api_spend_performed") is not False
        or gate.get("blockers") != []
        or gate.get("public_leakage_audit", {}).get("pass") is not True
    ):
        raise RuntimeError("ToolSandbox c8 child gate is not provider-zero ready")


def _c8_child_argv_from_gate(
    *,
    args: argparse.Namespace,
    shard: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[str]:
    gate_bindings = gate.get("gate_bindings")
    config = gate_bindings.get("config") if isinstance(gate_bindings, Mapping) else None
    if not isinstance(config, Mapping):
        raise RuntimeError("ToolSandbox c8 child gate config is missing")
    required = {
        "method_profile",
        "cap",
        "max_strong_residual_calls",
        "max_turn_steps",
        "max_authoring_rounds",
        "max_reproposal_rounds",
        "cheap_model",
        "cheap_reasoning_effort",
        "strong_model",
        "strong_reasoning_effort",
        "user_model",
        "user_reasoning_effort",
        "provider_timeout_seconds",
        "terminal_with_last_call",
        "method_variant",
    }
    if not required.issubset(config):
        raise RuntimeError("ToolSandbox c8 child gate config is incomplete")
    terminal_with_last_call = config["terminal_with_last_call"]
    if not isinstance(terminal_with_last_call, bool):
        raise RuntimeError("ToolSandbox c8 terminal policy must be boolean")
    if config["method_variant"] != _strict_c8_method_variant(
        terminal_with_last_call=terminal_with_last_call
    ):
        raise RuntimeError("ToolSandbox c8 method variant mismatch")
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--execute",
        "--strict-fresh-c8-preflight",
        "--stop-on-first-infrastructure-error",
        "--approval-receipt-json",
        str(_plan_path(shard, "approval_receipt_json")),
        "--api-base",
        str(args.api_base or ""),
        "--api-key-env",
        str(args.api_key_env),
        "--cheap-model",
        str(config["cheap_model"]),
        "--cheap-reasoning-effort",
        str(config["cheap_reasoning_effort"]),
        "--strong-model",
        str(config["strong_model"]),
        "--strong-reasoning-effort",
        str(config["strong_reasoning_effort"]),
        "--user-model",
        str(config["user_model"]),
        "--user-reasoning-effort",
        str(config["user_reasoning_effort"]),
        "--provider-timeout-seconds",
        str(config["provider_timeout_seconds"]),
        "--cap",
        str(config["cap"]),
        "--max-strong-residual-calls",
        str(config["max_strong_residual_calls"]),
        "--max-turn-steps",
        str(config["max_turn_steps"]),
        "--max-authoring-rounds",
        str(config["max_authoring_rounds"]),
        "--max-reproposal-rounds",
        str(config["max_reproposal_rounds"]),
        "--lower-receipt-root",
        str(args.lower_receipt_root),
        "--toolsandbox-checkout",
        str(args.toolsandbox_checkout),
        "--private-output-json",
        str(_plan_path(shard, "private_output_json")),
        "--public-output-json",
        str(_plan_path(shard, "public_output_json")),
        "--episode-jsonl",
        str(_plan_path(shard, "episode_jsonl")),
        "--gate-artifact-json",
        str(_plan_path(shard, "gate_artifact_json")),
        "--expected-gate-artifact-sha256",
        str(shard["expected_gate_artifact_sha256"]),
        "--public-ordinals",
        ",".join(str(value) for value in shard["public_ordinals"]),
    ]
    if config.get("episode_wall_timeout_seconds") is not None:
        argv.extend(
            [
                "--episode-wall-timeout-seconds",
                str(config["episode_wall_timeout_seconds"]),
            ]
        )
    if terminal_with_last_call:
        argv.append("--terminal-with-last-call")
    return argv


def _c8_common_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    shard_local = {
        "cap",
        "max_strong_residual_calls",
        "execute_output_identities",
    }
    return {str(key): value for key, value in config.items() if key not in shard_local}


def _c8_aggregate_acceptance(
    *,
    plan: Mapping[str, Any],
    scope: str,
) -> dict[str, Any] | None:
    raw = plan.get("aggregate_acceptance")
    if scope == "canary8":
        if raw is not None:
            raise RuntimeError(
                "ToolSandbox c8 canary cannot declare aggregate acceptance"
            )
        return None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError("ToolSandbox c8 aggregate_acceptance must be an object")
    required = {
        "official_avg_similarity_floor",
        "strict_complete_floor",
        "agent_side_cache_neutral_cost_ceiling_usd",
    }
    if set(raw) != required:
        raise RuntimeError(
            "ToolSandbox c8 aggregate_acceptance fields mismatch"
        )
    similarity_floor = _strict_decimal_string(
        raw,
        "official_avg_similarity_floor",
    )
    strict_floor = _strict_mapping_int(raw, "strict_complete_floor")
    cost_ceiling = _strict_decimal_string(
        raw,
        "agent_side_cache_neutral_cost_ceiling_usd",
    )
    if not Decimal("0") <= Decimal(similarity_floor) <= Decimal("1"):
        raise RuntimeError("ToolSandbox c8 similarity floor must be in [0, 1]")
    if not 0 <= strict_floor <= 129:
        raise RuntimeError("ToolSandbox c8 strict floor must be in [0, 129]")
    if Decimal(cost_ceiling) < 0:
        raise RuntimeError("ToolSandbox c8 cost ceiling must be non-negative")
    return {
        "official_avg_similarity_floor": similarity_floor,
        "strict_complete_floor": strict_floor,
        "agent_side_cache_neutral_cost_ceiling_usd": cost_ceiling,
    }


def _validate_c8_supervisor_scope(*, scope: str, ordinals: list[int]) -> None:
    if len(ordinals) != len(set(ordinals)):
        raise RuntimeError("ToolSandbox c8 supervisor ordinals overlap")
    if scope == "canary8":
        if len(ordinals) != 8 or {ordinal % 8 for ordinal in ordinals} != set(range(8)):
            raise RuntimeError("ToolSandbox c8 canary must cover one unique ordinal per residue")
        return
    if sorted(ordinals) != list(range(129)):
        raise RuntimeError("ToolSandbox c8 base129 plan must cover ordinals 0..128 exactly")


def _validate_c8_supervisor_paths(
    *,
    planned_paths: Sequence[Path],
    immutable_paths: Sequence[Path],
) -> None:
    resolved_planned = [path.resolve() for path in planned_paths]
    if len(resolved_planned) != len(set(resolved_planned)):
        raise RuntimeError("ToolSandbox c8 planned output and log paths must be distinct")
    if any(path.exists() for path in planned_paths):
        raise RuntimeError("ToolSandbox c8 planned output and log paths must be fresh")
    resolved_immutable = {path.resolve() for path in immutable_paths}
    if any(path in resolved_immutable for path in resolved_planned):
        raise RuntimeError("ToolSandbox c8 planned path collides with an input artifact")


def _c8_supervisor_artifact(
    *,
    plan_path: Path,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    shard_bindings = []
    for shard in prepared["shards"]:
        shard_bindings.append(
            {
                "shard_index": shard["shard_index"],
                "public_ordinals": shard["public_ordinals"],
                "cap": shard["cap"],
                "max_strong_residual_calls": shard[
                    "max_strong_residual_calls"
                ],
                "approval_path_sha256": _path_identity(shard["approval_path"]),
                "approval_receipt_sha256": shard["approval_receipt_sha256"],
                "approval_file_sha256": shard["approval_file_sha256"],
                "gate_path_sha256": _path_identity(shard["gate_path"]),
                "gate_artifact_sha256": shard["gate_sha256"],
                "private_output_path_sha256": _path_identity(
                    shard["private_path"]
                ),
                "public_output_path_sha256": _path_identity(shard["public_path"]),
                "episode_jsonl_path_sha256": _path_identity(
                    shard["episode_path"]
                ),
                "log_path_sha256": _path_identity(shard["log_path"]),
            }
        )
    bindings = {
        "plan_sha256": _file_sha256(plan_path),
        "supervisor_source_sha256": prepared["runner_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "toolsandbox_commit": prepared["toolsandbox_commit"],
        "toolsandbox_worktree": prepared["toolsandbox_worktree"],
        "common_config_sha256": _canonical_sha256(prepared["common_config"]),
        "method_variant": prepared["common_config"]["method_variant"],
        "method_profile": prepared["common_config"]["method_profile"],
        "terminal_with_last_call": prepared["common_config"][
            "terminal_with_last_call"
        ],
        "aggregate_acceptance": prepared["aggregate_acceptance"],
        "aggregate_acceptance_sha256": _canonical_sha256(
            prepared["aggregate_acceptance"]
        ),
        "exclusive_lock_path_sha256": _path_identity(STRICT_C8_LOCK_PATH),
        "supervisor_report_path_sha256": _path_identity(prepared["report_path"]),
        "shards": shard_bindings,
    }
    return {
        "schema": STRICT_C8_SUPERVISOR_ARTIFACT_SCHEMA,
        "decision": STRICT_C8_SUPERVISOR_READY_DECISION,
        "scope": prepared["scope"],
        "authorized_parallelism": STRICT_C8_AUTHORIZED_PARALLELISM,
        "provider_calls": 0,
        "api_spend_performed": False,
        "blockers": [],
        "bindings": bindings,
        "bindings_sha256": _canonical_sha256(bindings),
    }


def _acquire_c8_exclusive_lock() -> int:
    return _acquire_c8_lock(exclusive=True)


def _acquire_c8_shared_lock() -> int:
    return _acquire_c8_lock(exclusive=False)


def _acquire_c8_lock(*, exclusive: bool) -> int:
    STRICT_C8_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(STRICT_C8_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_fd, operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError(
            "another ToolSandbox execution holds an incompatible lease"
        ) from exc
    return lock_fd


def _acquire_or_inherit_c8_execution_lock() -> int:
    inherited = os.environ.get(STRICT_C8_INHERITED_LOCK_FD_ENV)
    if inherited is None:
        return _acquire_c8_exclusive_lock()
    try:
        inherited_fd = int(inherited)
        os.fstat(inherited_fd)
        inherited_path = Path(os.readlink(f"/proc/self/fd/{inherited_fd}")).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("strict ToolSandbox c8 inherited lease is invalid") from exc
    if inherited_path != STRICT_C8_LOCK_PATH.resolve():
        raise RuntimeError("strict ToolSandbox c8 inherited lease path mismatch")
    return os.dup(inherited_fd)


def _c8_child_environment(*, lock_fd: int) -> dict[str, str]:
    environment = dict(os.environ)
    for name in _PROXY_ENV_NAMES:
        environment.pop(name, None)
    environment["NO_PROXY"] = "localhost,127.0.0.1"
    environment["no_proxy"] = "localhost,127.0.0.1"
    environment[STRICT_C8_INHERITED_LOCK_FD_ENV] = str(lock_fd)
    return environment


def _existing_exact_runner_execute_pids(
    *,
    proc_root: Path = Path("/proc"),
    own_pid: int | None = None,
) -> list[int]:
    current_pid = os.getpid() if own_pid is None else int(own_pid)
    runner_path = Path(__file__).resolve()
    matches: list[int] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == current_pid:
            continue
        try:
            argv = [
                token.decode("utf-8", errors="surrogateescape")
                for token in (process_dir / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "--execute" not in argv:
            continue
        exact_runner = False
        for token in argv:
            candidate = Path(token)
            if candidate.is_absolute() and candidate.resolve() == runner_path:
                exact_runner = True
                break
            if candidate.is_absolute():
                continue
            try:
                cwd = (process_dir / "cwd").resolve(strict=True)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if (cwd / candidate).resolve() == runner_path:
                exact_runner = True
                break
        if exact_runner:
            matches.append(pid)
    return sorted(matches)


def _execute_c8_supervisor(
    *,
    prepared: Mapping[str, Any],
    supervisor_artifact_sha256: str,
    lock_fd: int,
) -> int:
    processes: list[subprocess.Popen[Any]] = []
    log_handles: list[Any] = []
    failure_reason: str | None = None
    interrupted = False
    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = _install_c8_supervisor_signal_handlers(handled_signals)
    try:
        for shard in prepared["shards"]:
            shard["log_path"].parent.mkdir(parents=True, exist_ok=True)
            log_handle = shard["log_path"].open("x", encoding="utf-8")
            log_handles.append(log_handle)
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                handled_signals,
            )
            try:
                process = subprocess.Popen(
                    shard["argv"],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                    env=_c8_child_environment(lock_fd=lock_fd),
                    pass_fds=(lock_fd,),
                )
                processes.append(process)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        while True:
            returncodes = [process.poll() for process in processes]
            failed = next(
                (code for code in returncodes if code is not None and code != 0),
                None,
            )
            if failed is not None:
                failure_reason = f"child_nonzero_exit:{failed}"
                break
            if all(code == 0 for code in returncodes):
                break
            time.sleep(0.2)
    except _SupervisorTerminationSignal as exc:
        interrupted = True
        failure_reason = f"supervisor_signal:{signal.Signals(exc.signum).name}"
    except KeyboardInterrupt:
        interrupted = True
        failure_reason = "supervisor_interrupted"
    except Exception as exc:
        failure_reason = f"supervisor_launch_error:{type(exc).__name__}"
    finally:
        if failure_reason is not None:
            _terminate_process_groups(processes)
        for log_handle in log_handles:
            log_handle.close()
        _restore_c8_supervisor_signal_handlers(previous_handlers)

    if failure_reason is None:
        try:
            success = _validate_c8_supervisor_success(prepared)
        except Exception as exc:
            failure_reason = f"postrun_validation_error:{type(exc).__name__}"
        else:
            aggregate_gate = success.get("aggregate_pareto_gate")
            claim_authorized = bool(
                isinstance(aggregate_gate, Mapping)
                and aggregate_gate.get("evaluated") is True
                and aggregate_gate.get("passes") is True
            )
            report = {
                "schema": STRICT_C8_SUPERVISOR_REPORT_SCHEMA,
                "decision": "toolsandbox_c8_supervisor_execute_complete",
                "diagnostic_partial_only": False,
                "scope": prepared["scope"],
                "executed_from_supervisor_artifact_sha256": (
                    supervisor_artifact_sha256
                ),
                **success,
                "requires_aggregate_quality_cost_gate": not claim_authorized,
                "authorizations": {
                    "authorizes_method_result_claim": claim_authorized,
                },
                "claim_boundary": [
                    "This report validates one fresh exact-union c8 execution.",
                    "A method-result claim is authorized only when the bound aggregate Pareto gate passes.",
                ],
            }
            _write_json_append_only(prepared["report_path"], report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0

    report = {
        "schema": STRICT_C8_SUPERVISOR_REPORT_SCHEMA,
        "decision": "runtime_failure",
        "diagnostic_partial_only": True,
        "scope": prepared["scope"],
        "executed_from_supervisor_artifact_sha256": supervisor_artifact_sha256,
        "failure_reason": failure_reason,
        "interrupted": interrupted,
        "child_returncodes": [process.poll() for process in processes],
        "observed_provider_calls_lower_bound": _observed_c8_provider_calls(
            prepared
        ),
        "provider_call_accounting_complete": False,
        "eligible_for_resume_or_splice": False,
    }
    _write_json_append_only(prepared["report_path"], report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 3


def _terminate_process_groups(
    processes: Sequence[subprocess.Popen[Any]],
    *,
    grace_seconds: float = 0.25,
) -> None:
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if processes and grace_seconds > 0:
        time.sleep(grace_seconds)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass


def _install_c8_supervisor_signal_handlers(
    handled_signals: Sequence[signal.Signals],
) -> dict[signal.Signals, Any]:
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    caught: list[int] = []

    def handler(signum: int, _frame: Any) -> None:
        if caught:
            return
        caught.append(int(signum))
        raise _SupervisorTerminationSignal(signum)

    for signum in handled_signals:
        signal.signal(signum, handler)
    return previous_handlers


def _restore_c8_supervisor_signal_handlers(
    previous_handlers: Mapping[signal.Signals, Any],
) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _validate_c8_supervisor_success(prepared: Mapping[str, Any]) -> dict[str, Any]:
    all_ordinals: list[int] = []
    provider_calls = 0
    similarity_values: list[Decimal] = []
    strict_complete_count = 0
    checked_attempt_count = 0
    checked_admitted_count = 0
    checked_replay_match_count = 0
    checked_block_count = 0
    residual_runtime_error_count = 0
    method_minefield_hit_count = 0
    total_minefield_hit_count = 0
    role_tokens = {
        "cheap": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        "strong": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
    }
    common_config = prepared.get("common_config")
    if not isinstance(common_config, Mapping):
        raise RuntimeError("ToolSandbox c8 common config is missing")
    if common_config.get("method_profile") != METHOD_PROFILE:
        raise RuntimeError("ToolSandbox c8 method profile mismatch")
    terminal_with_last_call = common_config.get("terminal_with_last_call")
    if not isinstance(terminal_with_last_call, bool):
        raise RuntimeError("ToolSandbox c8 terminal policy is missing")
    method_variant = _strict_c8_method_variant(
        terminal_with_last_call=terminal_with_last_call
    )
    if common_config.get("method_variant") != method_variant:
        raise RuntimeError("ToolSandbox c8 common method variant mismatch")
    expected_report_variant = (
        STRICT_C8_METHOD_VARIANT if terminal_with_last_call else None
    )
    scope = str(prepared["scope"])
    aggregate_acceptance = prepared.get("aggregate_acceptance")
    aggregate_evaluated = scope == "base129" and isinstance(
        aggregate_acceptance,
        Mapping,
    )
    for shard in prepared["shards"]:
        public_report = _read_json_object(shard["public_path"])
        episode_rows = _read_jsonl(shard["episode_path"])
        expected_ordinals = shard["public_ordinals"]
        public_rows = public_report.get("public_rows")
        summary = public_report.get("summary")
        if not isinstance(public_rows, list) or not isinstance(summary, Mapping):
            raise RuntimeError("child public output is incomplete")
        observed_public_ordinals = [
            _strict_mapping_int(row, "public_ordinal")
            for row in public_rows
            if isinstance(row, Mapping)
        ]
        observed_episode_ordinals = [
            _strict_mapping_int(row, "public_ordinal") for row in episode_rows
        ]
        if (
            public_report.get("schema") != SCHEMA
            or public_report.get("decision") != EXECUTE_READY_DECISION
            or public_report.get("api_spend_performed") is not True
            or public_report.get("official_evaluator_used") is not True
            or public_report.get("blockers") != []
            or public_report.get("method_variant") != expected_report_variant
            or public_report.get("executed_from_gate_sha256")
            != shard["gate_sha256"]
            or public_report.get("gate_bindings") != shard["gate_bindings"]
            or public_report.get("public_leakage_audit", {}).get("pass") is not True
            or public_report.get("supervisor_shard_only") is not True
            or public_report.get("diagnostic_partial_only") is not True
            or public_report.get("requires_successful_supervisor_report") is not True
            or public_report.get("authorizations", {}).get(
                "authorizes_method_result_claim"
            )
            is not False
            or public_report.get("paper_acceptance_gate", {}).get("evaluated")
            is not False
            or summary.get("attempted_episode_count") != shard["cap"]
            or summary.get("official_scored_episode_count") != shard["cap"]
            or summary.get("runtime_failure_count") != 0
            or summary.get("residual_runtime_error_count") != 0
            or observed_public_ordinals != expected_ordinals
            or observed_episode_ordinals != expected_ordinals
            or len(episode_rows) != shard["cap"]
        ):
            raise RuntimeError("child public output or episode union failed validation")
        for receipt in episode_rows:
            row_residual_errors = _residual_runtime_error_count(receipt)
            if (
                receipt.get("official_evaluator_used") is not True
                or receipt.get("decision") == "runtime_failure"
                or receipt.get("blockers") not in (None, [])
                or row_residual_errors != 0
            ):
                raise RuntimeError("child episode contains evaluator or runtime failure")
            residual_runtime_error_count += row_residual_errors
            if aggregate_evaluated:
                similarity_values.append(_c8_episode_similarity(receipt))
        if aggregate_evaluated:
            for row in public_rows:
                executed = row.get("decision") == "execute"
                row_attempted = _strict_mapping_bool(
                    row,
                    "checked_execution_attempted",
                )
                row_admitted = _strict_mapping_bool(
                    row,
                    "checked_execution_admitted",
                )
                row_replay_match = _strict_mapping_bool(
                    row,
                    "checked_execution_replay_match",
                )
                expected_checked = executed
                if not (
                    row_attempted
                    == row_admitted
                    == row_replay_match
                    == expected_checked
                ):
                    raise RuntimeError(
                        "child checked-execution flags do not match execution"
                    )
            shard_strict = sum(
                _strict_mapping_bool(row, "milestone_complete")
                for row in public_rows
            )
            shard_checked_attempted = sum(
                _strict_mapping_bool(row, "checked_execution_attempted")
                for row in public_rows
            )
            shard_checked_admitted = sum(
                _strict_mapping_bool(row, "checked_execution_admitted")
                for row in public_rows
            )
            shard_checked_replay_match = sum(
                _strict_mapping_bool(row, "checked_execution_replay_match")
                for row in public_rows
            )
            shard_residual_execute = sum(
                row.get("decision") == "execute" for row in public_rows
            )
            shard_method_minefield = sum(
                _strict_mapping_bool(row, "method_introduced_minefield_hit")
                for row in public_rows
            )
            shard_total_minefield = sum(
                _strict_mapping_bool(row, "minefield_hit") for row in public_rows
            )
            shard_checked_blocked = _strict_mapping_int(
                summary,
                "checked_execution_block_count",
            )
            shard_residual_errors = sum(
                _residual_runtime_error_count(receipt) for receipt in episode_rows
            )
            if (
                _strict_mapping_int(summary, "strict_complete_count")
                != shard_strict
                or _strict_mapping_int(
                    summary,
                    "checked_execution_attempt_count",
                )
                != shard_checked_attempted
                or _strict_mapping_int(
                    summary,
                    "checked_execution_admitted_count",
                )
                != shard_checked_admitted
                or shard_checked_blocked
                != shard_checked_attempted - shard_checked_admitted
                or shard_checked_replay_match != shard_checked_admitted
                or _strict_mapping_int(summary, "residual_execute_count")
                != shard_residual_execute
                or _strict_mapping_int(summary, "residual_runtime_error_count")
                != shard_residual_errors
                or _strict_mapping_int(summary, "frontier_suppressed_count") != 0
                or _strict_mapping_int(
                    summary,
                    "strong_cap_exhausted_frontier_count",
                )
                != 0
                or _strict_mapping_int(
                    summary,
                    "method_introduced_minefield_hit_count",
                )
                != shard_method_minefield
                or _strict_mapping_int(summary, "total_minefield_hit_count")
                != shard_total_minefield
            ):
                raise RuntimeError(
                    "child quality or checked-execution summary is inconsistent"
                )
            strict_complete_count += shard_strict
            checked_attempt_count += shard_checked_attempted
            checked_admitted_count += shard_checked_admitted
            checked_replay_match_count += shard_checked_replay_match
            checked_block_count += shard_checked_blocked
            method_minefield_hit_count += shard_method_minefield
            total_minefield_hit_count += shard_total_minefield
            for role in role_tokens:
                input_tokens, output_tokens, reasoning_tokens = _c8_agent_role_tokens(
                    public_report,
                    role=role,
                )
                role_tokens[role]["input_tokens"] += input_tokens
                role_tokens[role]["output_tokens"] += output_tokens
                role_tokens[role]["reasoning_tokens"] += reasoning_tokens
        provider_calls += _int(public_report.get("provider_calls"))
        all_ordinals.extend(observed_episode_ordinals)
    _validate_c8_supervisor_scope(
        scope=scope,
        ordinals=all_ordinals,
    )
    aggregate_gate: dict[str, Any]
    if aggregate_evaluated:
        if len(similarity_values) != 129:
            raise RuntimeError(
                "ToolSandbox c8 base129 requires 129 raw similarity values"
            )
        if (
            checked_attempt_count <= 0
            or checked_admitted_count != checked_attempt_count
            or checked_block_count != 0
        ):
            raise RuntimeError(
                "ToolSandbox c8 checked execution must run and admit every attempt"
            )
        if method_minefield_hit_count != 0 or total_minefield_hit_count != 0:
            raise RuntimeError("ToolSandbox c8 aggregate minefield count must be zero")
        cheap_model = common_config.get("cheap_model")
        strong_model = common_config.get("strong_model")
        if (
            not isinstance(cheap_model, str)
            or not cheap_model
            or not isinstance(strong_model, str)
            or not strong_model
        ):
            raise RuntimeError("ToolSandbox c8 agent models are missing")
        similarity_score = sum(similarity_values, Decimal("0")) / Decimal(129)
        similarity_floor = Decimal(
            str(aggregate_acceptance["official_avg_similarity_floor"])
        )
        cost_ceiling = Decimal(
            str(
                aggregate_acceptance[
                    "agent_side_cache_neutral_cost_ceiling_usd"
                ]
            )
        )
        cheap_cost = Decimal(
            str(
                cache_neutral_cost_usd(
                    model=cheap_model,
                    input_tokens=role_tokens["cheap"]["input_tokens"],
                    output_tokens=role_tokens["cheap"]["output_tokens"],
                    reasoning_tokens=role_tokens["cheap"]["reasoning_tokens"],
                )
            )
        )
        strong_cost = Decimal(
            str(
                cache_neutral_cost_usd(
                    model=strong_model,
                    input_tokens=role_tokens["strong"]["input_tokens"],
                    output_tokens=role_tokens["strong"]["output_tokens"],
                    reasoning_tokens=role_tokens["strong"]["reasoning_tokens"],
                )
            )
        )
        agent_cost = cheap_cost + strong_cost
        strict_floor = _strict_mapping_int(
            aggregate_acceptance,
            "strict_complete_floor",
        )
        failed_conditions = []
        if similarity_score < similarity_floor:
            failed_conditions.append("official_avg_similarity_below_floor")
        if strict_complete_count < strict_floor:
            failed_conditions.append("strict_complete_below_floor")
        if agent_cost > cost_ceiling:
            failed_conditions.append(
                "agent_side_cache_neutral_cost_above_ceiling"
            )
        aggregate_gate = {
            "evaluated": True,
            "passes": not failed_conditions,
            "failed_conditions": failed_conditions,
            "required_episode_count": 129,
            "episode_count": len(all_ordinals),
            "official_scored_episode_count": len(similarity_values),
            "official_avg_similarity_score": float(similarity_score),
            "official_avg_similarity_score_exact": str(similarity_score),
            "official_avg_similarity_floor": str(similarity_floor),
            "strict_complete_count": strict_complete_count,
            "strict_complete_floor": strict_floor,
            "agent_side_cache_neutral_cost_usd": float(agent_cost),
            "agent_side_cache_neutral_cost_usd_exact": str(agent_cost),
            "agent_side_cache_neutral_cost_ceiling_usd": str(cost_ceiling),
            "agent_side_token_usage": role_tokens,
            "infrastructure_failure_count": 0,
            "runtime_failure_count": 0,
            "residual_runtime_error_count": residual_runtime_error_count,
            "method_introduced_minefield_hit_count": method_minefield_hit_count,
            "total_minefield_hit_count": total_minefield_hit_count,
            "checked_execution_attempt_count": checked_attempt_count,
            "checked_execution_admitted_count": checked_admitted_count,
            "checked_execution_replay_match_count": checked_replay_match_count,
            "checked_execution_block_count": checked_block_count,
        }
    else:
        aggregate_gate = {
            "evaluated": False,
            "passes": False,
            "failed_conditions": ["requires_full_base129"],
        }
    return {
        "provider_calls": provider_calls,
        "api_spend_performed": True,
        "shard_count": len(prepared["shards"]),
        "episode_count": len(all_ordinals),
        "official_evaluator_used": True,
        "runtime_failure_count": 0,
        "residual_runtime_error_count": residual_runtime_error_count,
        "method_variant": method_variant,
        "terminal_with_last_call": terminal_with_last_call,
        "method_profile": METHOD_PROFILE,
        "aggregate_pareto_gate": aggregate_gate,
        "blockers": [],
    }


def _c8_episode_similarity(receipt: Mapping[str, Any]) -> Decimal:
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise RuntimeError("child episode similarity is missing")
    raw = evaluation.get("similarity")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise RuntimeError("child episode similarity is invalid")
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise RuntimeError("child episode similarity is invalid") from exc
    if not value.is_finite() or not Decimal("0") <= value <= Decimal("1"):
        raise RuntimeError("child episode similarity is outside [0, 1]")
    return value


def _strict_mapping_bool(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise RuntimeError(f"ToolSandbox c8 child {field} must be boolean")
    return value


def _c8_agent_role_tokens(
    public_report: Mapping[str, Any],
    *,
    role: str,
) -> tuple[int, int, int]:
    resource_accounting = public_report.get("resource_accounting")
    if not isinstance(resource_accounting, Mapping):
        raise RuntimeError("child resource accounting is missing")
    method_resources = resource_accounting.get("method_resources")
    if not isinstance(method_resources, Mapping):
        raise RuntimeError("child method resources are missing")
    runtime_ledger = method_resources.get("runtime_ledger")
    if not isinstance(runtime_ledger, Mapping):
        raise RuntimeError("child runtime ledger is missing")
    per_role = runtime_ledger.get("per_role")
    if not isinstance(per_role, Mapping):
        raise RuntimeError("child runtime ledger roles are missing")
    usage = per_role.get(role)
    if not isinstance(usage, Mapping):
        raise RuntimeError(f"child runtime ledger {role} role is missing")
    calls = _strict_mapping_int(usage, "calls")
    available = _strict_mapping_int(usage, "token_usage_call_count")
    unavailable = _strict_mapping_int(usage, "token_usage_unavailable_count")
    if calls < 0 or available < 0 or unavailable < 0:
        raise RuntimeError("child runtime ledger counts must be non-negative")
    if calls == 0:
        if (
            usage.get("observed") is not False
            or available != 0
            or unavailable != 0
        ):
            raise RuntimeError(f"child runtime ledger {role} zero-call role is invalid")
        return 0, 0, 0
    if usage.get("observed") is not True or usage.get(
        "token_usage_available"
    ) is not True or available != calls or unavailable != 0:
        raise RuntimeError(f"child runtime ledger {role} token usage is incomplete")
    input_tokens = _strict_mapping_int(usage, "input_tokens")
    output_tokens = _strict_mapping_int(usage, "output_tokens")
    reasoning_tokens = _strict_mapping_int(usage, "reasoning_tokens") if "reasoning_tokens" in usage else 0
    if input_tokens < 0 or output_tokens < 0 or reasoning_tokens < 0:
        raise RuntimeError("child runtime ledger tokens must be non-negative")
    return input_tokens, output_tokens, reasoning_tokens


def _observed_c8_provider_calls(prepared: Mapping[str, Any]) -> int:
    total = 0
    for shard in prepared["shards"]:
        if not shard["public_path"].is_file():
            continue
        try:
            total += _int(_read_json_object(shard["public_path"]).get("provider_calls"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return total


def _strict_plan_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"ToolSandbox c8 supervisor {field} must be an integer")
    return int(value)


def _strict_mapping_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"ToolSandbox c8 child {field} must be an integer")
    return int(value)


def _strict_decimal_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"ToolSandbox c8 supervisor {field} must be a decimal string"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError(
            f"ToolSandbox c8 supervisor {field} must be a decimal string"
        ) from exc
    if not parsed.is_finite():
        raise RuntimeError(
            f"ToolSandbox c8 supervisor {field} must be finite"
        )
    return value


def _strict_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"ToolSandbox c8 supervisor {field} must be sha256")
    return value


def _plan_path(payload: Mapping[str, Any], field: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"ToolSandbox c8 supervisor {field} path is required")
    return Path(value).resolve()


def _path_identity(path: Path) -> str:
    return _canonical_sha256({"path": str(path.resolve())})


def _write_json_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _live_factories(args: argparse.Namespace) -> tuple[Any, Any]:
    from papc.certspec.toolsandbox.online_official_runtime import (  # noqa: WPS433
        ToolSandboxOfficialOnlineRuntime,
        ToolSandboxOfficialOnlineRuntimeConfig,
    )
    from papc.certspec.toolsandbox.online_strong_proposer import (  # noqa: WPS433
        ToolSandboxGroundedStepProposer,
        ToolSandboxOpenAIStrongProposerConfig,
    )

    def runtime_factory() -> ToolSandboxOfficialOnlineRuntime:
        return ToolSandboxOfficialOnlineRuntime(
            ToolSandboxOfficialOnlineRuntimeConfig(
                checkout=args.toolsandbox_checkout,
                api_base=args.api_base,
                api_key_env=args.api_key_env,
                cheap_model=args.cheap_model,
                cheap_reasoning_effort=args.cheap_reasoning_effort,
                user_model=args.user_model,
                user_reasoning_effort=args.user_reasoning_effort,
                provider_timeout_seconds=args.provider_timeout_seconds,
            )
        )

    def proposer_factory() -> ToolSandboxGroundedStepProposer:
        return ToolSandboxGroundedStepProposer(
            ToolSandboxOpenAIStrongProposerConfig(
                model=args.strong_model,
                reasoning_effort=args.strong_reasoning_effort,
                api_base=args.api_base,
                api_key_env=args.api_key_env,
                provider_timeout_seconds=args.provider_timeout_seconds,
            )
        )

    return runtime_factory, proposer_factory


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> list[dict[str, str]]:
    paths = (
        Path(__file__),
        _REPO / "src/papc/__init__.py",
        _REPO / "src/papc/certspec/__init__.py",
        _REPO / "src/papc/certspec/toolsandbox/__init__.py",
        _REPO / "src/papc/certspec/toolsandbox/base129_true_online_run.py",
        _REPO / "src/papc/certspec/toolsandbox/online_cfrc_controller.py",
        _REPO / "src/papc/certspec/toolsandbox/cfrc_online_adapter.py",
        _REPO / "src/papc/certspec/toolsandbox/online_official_runtime.py",
        _REPO / "src/papc/certspec/toolsandbox/online_frontier_monitor.py",
        _REPO / "src/papc/certspec/toolsandbox/online_strong_proposer.py",
        _REPO / "src/papc/certspec/toolsandbox/grounded_residual_authoring.py",
        _REPO / "src/papc/certspec/toolsandbox/public_hygiene.py",
        _REPO / "src/papc/certspec/toolsandbox/method_runtime.py",
        _REPO / "src/papc/certspec/toolsandbox/role_bridge.py",
        _REPO / "src/papc/certspec/toolsandbox/runtime_accounting.py",
        _REPO / "src/papc/certspec/toolsandbox/base129_proposer_response_bridge.py",
        _REPO / "src/papc/certspec/toolsandbox/adapter.py",
        _REPO / "src/papc/certspec/toolsandbox/residual_certificate.py",
        _REPO / "src/papc/certspec/toolsandbox/residual_projection.py",
        _REPO / "src/papc/certspec/toolsandbox/suffix_scorer.py",
        _REPO / "src/papc/certspec/toolsandbox/state_write_contract.py",
        _REPO / "src/papc/certspec/cfrc_compiler.py",
        _REPO / "src/papc/certspec/cfrc_online_runner.py",
        _REPO / "src/papc/certspec/cfrc_ir.py",
        _REPO / "src/papc/certspec/cfrc_summary.py",
        _REPO / "src/papc/certspec/costing.py",
        _REPO / "src/papc/certspec/runtime_ledger.py",
        _REPO / "src/papc/certspec/executable_closure.py",
        _REPO / "src/papc/certspec/minimum_authority_selector.py",
        _REPO / "src/papc/certspec/slot_candidate_generator.py",
        _REPO / "src/papc/certspec/state_write_contract.py",
        _REPO / "src/papc/certspec/provider.py",
        _REPO / "src/papc/certspec/reversibility.py",
        _REPO / "src/papc/certspec/shared_abstractions.py",
        _REPO / "src/papc/certspec/grounded_proposal.py",
        _REPO / "src/papc/certspec/receipt_provenance.py",
    )
    return [
        {
            "path": path.resolve().relative_to(_REPO.resolve()).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def _git_commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_worktree_binding(checkout: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    untracked_manifest = []
    for raw_path in untracked_raw.split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        path = checkout / relative_path
        untracked_manifest.append(
            {
                "path_sha256": hashlib.sha256(raw_path).hexdigest(),
                "content_sha256": _file_sha256(path),
            }
        )
    components = {
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_manifest_sha256": _canonical_sha256(untracked_manifest),
    }
    return {
        "dirty": bool(status),
        "status_entry_count": len([item for item in status.split(b"\0") if item]),
        **components,
        "fingerprint_sha256": _canonical_sha256(components),
    }


def _gate_bindings(
    *,
    args: argparse.Namespace,
    approval_receipt: dict[str, Any],
    scenario_specs: list[Any],
) -> dict[str, Any]:
    source_manifest = _source_manifest()
    selection = [
        {
            "public_ordinal": _int(getattr(spec, "public_ordinal", None)),
            "scenario_name": str(getattr(spec, "scenario_name", "")),
        }
        for spec in scenario_specs
    ]
    bindings = {
        "approval_receipt_sha256": _canonical_sha256(approval_receipt),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "toolsandbox_commit": _git_commit(args.toolsandbox_checkout),
        "toolsandbox_worktree": _git_worktree_binding(args.toolsandbox_checkout),
        "selection_count": len(selection),
        "selection_sha256": _canonical_sha256(selection),
        "config": {
            "method_profile": METHOD_PROFILE,
            "cap": int(args.cap),
            "max_strong_residual_calls": int(args.max_strong_residual_calls),
            "max_turn_steps": int(args.max_turn_steps),
            # Both bound how many provider calls one frontier may spend, so the
            # gate must bind them or it cannot bind the run's cost.
            "max_authoring_rounds": int(args.max_authoring_rounds),
            "max_reproposal_rounds": int(args.max_reproposal_rounds),
            "cheap_model": args.cheap_model,
            "cheap_reasoning_effort": args.cheap_reasoning_effort,
            "strong_model": args.strong_model,
            "strong_reasoning_effort": args.strong_reasoning_effort,
            "user_model": args.user_model,
            "user_reasoning_effort": args.user_reasoning_effort,
            "provider_timeout_seconds": float(args.provider_timeout_seconds),
            "episode_wall_timeout_seconds": args.episode_wall_timeout_seconds,
        },
    }
    if args.terminal_with_last_call:
        bindings["config"]["terminal_with_last_call"] = True
    if args.stop_on_first_infrastructure_error:
        bindings["config"]["stop_on_first_infrastructure_error"] = True
    if args.strict_fresh_c8_preflight:
        execute_output_identities = _strict_execute_output_identities(args)
        bindings["config"].update(
            {
                "strict_fresh_c8_preflight": True,
                "authorized_parallelism": STRICT_C8_AUTHORIZED_PARALLELISM,
                "method_variant": _strict_c8_method_variant(
                    terminal_with_last_call=args.terminal_with_last_call
                ),
                "terminal_with_last_call": bool(args.terminal_with_last_call),
                "api_key_env_name_sha256": _canonical_sha256(
                    {"api_key_env": args.api_key_env}
                ),
                "endpoint_identity_sha256": _canonical_sha256(
                    {"api_base": str(args.api_base or "").strip()}
                ),
                "execute_output_identities": execute_output_identities,
            }
        )
    return bindings


def _verify_gate(
    *,
    args: argparse.Namespace,
    gate_bindings: dict[str, Any],
) -> None:
    if (
        args.gate_artifact_json is None
        or args.expected_gate_artifact_sha256 is None
    ):
        raise RuntimeError("ToolSandbox source-bound provider-zero gate is required")
    if _file_sha256(args.gate_artifact_json) != args.expected_gate_artifact_sha256:
        raise RuntimeError("ToolSandbox gate artifact hash changed")
    gate = _read_json_object(args.gate_artifact_json)
    if (
        gate.get("schema") != SCHEMA
        or gate.get("decision") != NOAPI_READY_DECISION
        or gate.get("provider_calls") != 0
        or gate.get("api_spend_performed") is not False
        or gate.get("gate_bindings") != gate_bindings
        or gate.get("public_leakage_audit", {}).get("pass") is not True
    ):
        raise RuntimeError("ToolSandbox gate does not bind this execution")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _filter_public_ordinals(
    scenario_specs: list[Any],
    *,
    public_ordinals: str,
) -> list[Any]:
    selected = _parse_public_ordinals(public_ordinals)
    if not selected:
        return scenario_specs
    return [spec for spec in scenario_specs if int(spec.public_ordinal) in selected]


def _parse_public_ordinals(value: str) -> set[int]:
    selected: set[int] = set()
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        selected.add(_int(token))
    return selected


def _pending_episode_count(
    scenario_specs: list[Any],
    existing_receipts: list[dict[str, Any]],
) -> int:
    completed = {
        _int(receipt.get("public_ordinal"))
        for receipt in existing_receipts
        if "public_ordinal" in receipt
    }
    return sum(
        1
        for spec in scenario_specs
        if _int(getattr(spec, "public_ordinal", None)) not in completed
    )


def _required_tool_env_blockers(env_vars: tuple[str, ...]) -> list[str]:
    return [
        f"required_tool_env_missing:{name}"
        for name in env_vars
        if not os.environ.get(name)
    ]


def _strict_fresh_c8_blockers(
    *,
    args: argparse.Namespace,
    approval_receipt: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not args.api_base:
        blockers.append("strict_c8_api_base_missing")
    if not os.environ.get(args.api_key_env):
        blockers.append("required_provider_env_missing:selected_api_key_env")
    blockers.extend(_required_tool_env_blockers(REQUIRED_TOOL_ENV_VARS))
    if not args.stop_on_first_infrastructure_error:
        blockers.append("strict_c8_infrastructure_failfast_required")
    # cap=1 is reserved for an eight-process, one-case-per-process canary;
    # cap=16/17 are the legal eight-shard sizes for the complete base129 run.
    if int(args.cap) not in {1, 16, 17}:
        blockers.append("strict_c8_shard_cap_mismatch")
    if int(args.max_strong_residual_calls) != 16 * int(args.cap):
        blockers.append("strict_c8_strong_call_budget_mismatch")

    expected: dict[str, Any] = {
        "approval_scope": "toolsandbox_base129_true_online_cfrc_method_arm",
        "arm": METHOD_PROFILE,
        "surface": "toolsandbox",
        "scenario_scope": "base129",
        "max_cheap_episodes": int(args.cap),
        "max_strong_residual_calls": int(args.max_strong_residual_calls),
        "cheap_model": args.cheap_model,
        "cheap_reasoning_effort": args.cheap_reasoning_effort,
        "strong_model": args.strong_model,
        "strong_reasoning_effort": args.strong_reasoning_effort,
        "user_model": args.user_model,
        "user_reasoning_effort": args.user_reasoning_effort,
        "full_agent_fallback_allowed": False,
        "resource_accounting_required": True,
        "authorized_parallelism": STRICT_C8_AUTHORIZED_PARALLELISM,
        "method_variant": _strict_c8_method_variant(
            terminal_with_last_call=args.terminal_with_last_call
        ),
        "terminal_with_last_call": bool(args.terminal_with_last_call),
        "approval_revision": GROUNDED_AUTHORING_APPROVAL_REVISION,
        "approval_revision_basis": GROUNDED_AUTHORING_APPROVAL_BASIS,
    }
    for field, expected_value in expected.items():
        if approval_receipt.get(field) != expected_value:
            blockers.append(f"strict_c8_approval_{field}_mismatch")
    return sorted(set(blockers))


def _strict_c8_method_variant(*, terminal_with_last_call: bool) -> str:
    return (
        STRICT_C8_METHOD_VARIANT
        if terminal_with_last_call
        else STRICT_C8_INCUMBENT_METHOD_VARIANT
    )


def _validate_strict_fresh_output_paths(args: argparse.Namespace) -> None:
    gate_paths = {
        "private_output_json": args.private_output_json,
        "public_output_json": args.public_output_json,
        "episode_jsonl": args.episode_jsonl,
    }
    if args.execute:
        execute_paths = dict(gate_paths)
        if (
            args.planned_execute_private_output_json is not None
            and args.planned_execute_private_output_json.resolve()
            != args.private_output_json.resolve()
        ):
            raise SystemExit(
                "strict c8 planned private output does not match execute output"
            )
        if (
            args.planned_execute_public_output_json is not None
            and args.planned_execute_public_output_json.resolve()
            != args.public_output_json.resolve()
        ):
            raise SystemExit(
                "strict c8 planned public output does not match execute output"
            )
        paths = execute_paths
    else:
        if (
            args.planned_execute_private_output_json is None
            or args.planned_execute_public_output_json is None
        ):
            raise SystemExit(
                "strict c8 noapi requires planned execute private and public outputs"
            )
        execute_paths = {
            "planned_execute_private_output_json": (
                args.planned_execute_private_output_json
            ),
            "planned_execute_public_output_json": (
                args.planned_execute_public_output_json
            ),
            "episode_jsonl": args.episode_jsonl,
        }
        paths = {**gate_paths, **execute_paths}
    resolved = {name: path.resolve() for name, path in paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise SystemExit(
            "--strict-fresh-c8-preflight requires distinct gate and execute outputs"
        )
    for name, path in paths.items():
        if path.exists():
            raise SystemExit(
                f"--strict-fresh-c8-preflight requires fresh {name}: {path}"
            )


def _strict_execute_output_identities(args: argparse.Namespace) -> dict[str, str]:
    if args.execute:
        private_path = args.private_output_json
        public_path = args.public_output_json
    else:
        private_path = args.planned_execute_private_output_json
        public_path = args.planned_execute_public_output_json
    if private_path is None or public_path is None:
        return {}
    return {
        "private_output_path_sha256": _canonical_sha256(
            {"path": str(private_path.resolve())}
        ),
        "public_output_path_sha256": _canonical_sha256(
            {"path": str(public_path.resolve())}
        ),
        "episode_jsonl_path_sha256": _canonical_sha256(
            {"path": str(args.episode_jsonl.resolve())}
        ),
    }


def _episode_receipt_callback(
    path: Path,
    *,
    stop_on_first_infrastructure_error: bool,
):
    append = _append_jsonl_callback(path)

    def callback(receipt: Mapping[str, Any]) -> None:
        append(dict(receipt))
        if not stop_on_first_infrastructure_error:
            return
        reasons = _episode_infrastructure_failure_reasons(
            receipt,
            fail_on_residual_runtime_error=True,
        )
        if reasons:
            raise _InfrastructureFailureStop(
                public_ordinal=_int(receipt.get("public_ordinal")),
                reasons=reasons,
            )

    return callback


def _episode_infrastructure_failure_reasons(
    receipt: Mapping[str, Any],
    *,
    fail_on_residual_runtime_error: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if receipt.get("decision") == "runtime_failure":
        reasons.append("decision_runtime_failure")
    runtime_error_count = _residual_runtime_error_count(receipt)
    if fail_on_residual_runtime_error and runtime_error_count > 0:
        reasons.append(f"residual_runtime_error_count:{runtime_error_count}")
    if receipt.get("official_evaluator_used") is not True:
        reasons.append("official_evaluator_not_used")
    return reasons


def _residual_runtime_error_count(receipt: Mapping[str, Any]) -> int:
    private = receipt.get("private")
    if not isinstance(private, Mapping):
        return 0
    controller = private.get("controller_receipt")
    if not isinstance(controller, Mapping):
        return 0
    attempts = controller.get("cfrc_attempts")
    if not isinstance(attempts, Sequence) or isinstance(
        attempts, (str, bytes, bytearray)
    ):
        return 0
    total = 0
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        public_summary = attempt.get("public_summary")
        if not isinstance(public_summary, Mapping):
            continue
        execution = public_summary.get("execution")
        if isinstance(execution, Mapping):
            total += _int(execution.get("runtime_error_count"))
    return total


def _existing_receipts_for_resume(
    path: Path,
    *,
    rerun_failures: bool,
) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        ordinal = _int(row.get("public_ordinal"))
        latest[ordinal] = row
    rows = [latest[ordinal] for ordinal in sorted(latest)]
    if rerun_failures:
        rows = [row for row in rows if row.get("decision") != "runtime_failure"]
    return rows


def _append_jsonl_callback(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def append(receipt: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    return append


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_pair_append_only(
    *,
    private_path: Path,
    public_path: Path,
    private_report: dict[str, Any],
    public_report: dict[str, Any],
) -> None:
    if private_path.exists():
        raise FileExistsError(private_path)
    if public_path.exists():
        raise FileExistsError(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(private_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_path.write_text(
        json.dumps(public_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

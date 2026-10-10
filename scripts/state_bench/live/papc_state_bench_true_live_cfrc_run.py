#!/usr/bin/env python3
"""Run STATE-Bench with true-live cheap prefix and CFRC strong handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papc.certspec.runtime_ledger import RuntimeLedger  # noqa: E402
from papc.certspec.state_bench import runtime_accounting  # noqa: E402
from papc.certspec.state_bench import true_live_run_plan as run_plan  # noqa: E402
from papc.certspec.state_bench import true_live_scaffold as scaffold  # noqa: E402
from papc.certspec.state_bench.noapi_readiness import (  # noqa: E402
    public_leakage_audit,
)
from papc.certspec.state_bench.true_live_online import (  # noqa: E402
    DOMAIN_PROMPT_SUFFIX_MODES,
    MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
    METHOD_ARMS,
    VALIDATED_MAINLINE_PROFILE,
    StateBenchCFRCProfile,
    execute_true_live_task,
    online_execution_contract,
)
from papc.paths import DATA_ROOT
from scripts.state_bench.common import (  # noqa: E402
    papc_method_arm_execution_adapter as method_arm,
)

SCHEMA = "papc.state_bench_true_live_cfrc_run.v1"
READY_NOT_EXECUTED_DECISION = "state_bench_true_live_cfrc_ready_not_executed"
COMPLETE_DECISION = "state_bench_true_live_cfrc_complete"
BLOCKED_DECISION = "blocked_state_bench_true_live_cfrc_run"
NONCANONICAL_MODEL_PLAN_BLOCKER = run_plan.NONCANONICAL_MODEL_PLAN_BLOCKER
EXPECTED_NOAPI_REPORT_REQUIRED_BLOCKER = run_plan.EXPECTED_NOAPI_REPORT_REQUIRED_BLOCKER

DOMAINS = ("shopping_assistant", "travel", "customer_support")
DEFAULT_STATE_BENCH_ROOT = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_PUBLIC_OUTPUT = Path("outputs/papc/state_bench_true_live_cfrc_run_public.json")
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/state_bench_true_live_cfrc_run_private"
)
MAX_EXECUTE_CONCURRENCY = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="")
    parser.add_argument("--surface", default="state_bench_true_live")
    parser.add_argument(
        "--arm",
        default="mainline",
        choices=list(METHOD_ARMS),
        help=(
            "Measured arm. mainline is the method. cheap_anchor runs the cheap "
            "model over the whole system with no strong call, no method system "
            "prompt suffix and no native cheap bound, so it measures what the "
            "cheap model alone reaches and at what cost."
        ),
    )
    parser.add_argument(
        "--domain-prompt-suffix",
        default="on",
        choices=list(DOMAIN_PROMPT_SUFFIX_MODES),
        help=(
            "How much of the STATE-Bench domain prompt suffix the run carries. "
            "on is the method statement plus a domain section that names this "
            "benchmark's thresholds, tools, and tool arguments; generic keeps "
            "the same obligations with every value, tool name, and scenario "
            "recipe removed; method_only drops the domain section, off drops "
            "the suffix. Both arms read this, so an incumbent can be measured "
            "under the same prompt the method arm receives."
        ),
    )
    parser.add_argument(
        "--cheap-native-item-loop",
        choices=("on", "off"),
        default="off",
        help=(
            "Whether the cheap prefix model reads its own transcript as native "
            "Responses items (function_call and function_call_output) instead "
            "of the text-flattened rendering. on is the shape the benchmark's "
            "own loop already chains; off is the registered shape and is byte "
            "identical to every result recorded so far. No reasoning item is "
            "carried on either setting."
        ),
    )
    parser.add_argument(
        "--cheap-max-tool-calls-per-response",
        type=int,
        default=32,
        help=(
            "Native-runtime bound on how many tool calls one cheap response "
            "may carry; every write call is preserved beyond it. 0 disables."
        ),
    )
    parser.add_argument(
        "--successor-tool-descriptions",
        choices=("on", "off"),
        default="on",
        help=(
            "Whether the successor's copy of the tool catalog keeps each tool's "
            "own description. off is the contract projection: names, argument "
            "names, types, enums, defaults and required flags only. The cheap "
            "agent's catalog is untouched either way."
        ),
    )
    parser.add_argument(
        "--residual-round-budget-declared",
        choices=("on", "off"),
        default=(
            "on" if VALIDATED_MAINLINE_PROFILE.declare_remaining_responses else "off"
        ),
        help=(
            "Whether each successor response is told how many responses remain "
            "in the frozen contract tenure, including the current response and "
            "any closing rewrite. This changes no runtime budget or authority."
        ),
    )
    parser.add_argument(
        "--closing-contract-style",
        choices=("consequence", "plain"),
        default=VALIDATED_MAINLINE_PROFILE.closing_contract_style,
        help=(
            "Wording of the instruction to close in the round that acts. "
            "'plain' states the requirement and is the wording this pool "
            "measured 94 of 150 under; 'consequence' adds what the compiler "
            "does to a response that acts without text."
        ),
    )
    parser.add_argument(
        "--closure-obligation-declared",
        choices=("on", "off"),
        default=(
            "on" if VALIDATED_MAINLINE_PROFILE.declare_closure_admissibility else "off"
        ),
        help=(
            "Whether the successor is told the conditions its closing message "
            "has to satisfy before the runner checks it. The check itself is "
            "unchanged either way; on only moves the conditions from after the "
            "draft to before it, so a message that would have been rewritten at "
            "the cost of another call is written admissibly the first time."
        ),
    )
    parser.add_argument(
        "--final-repair-role",
        choices=("strong", "cheap_prefix_model"),
        default="strong",
        help=(
            "Which model writes the closing message when the successor draft "
            "does not close. The successor never returns text beside its tool "
            "calls, so this rewrite fires on almost every acting row rather "
            "than occasionally. It reports what the admitted actions "
            "established and does not choose them, and the same deterministic "
            "closure test judges it either way, so the authoring model is an "
            "operator choice. The default keeps every registered arm frozen; "
            "the cheap anchor forces strong so the denominator cannot pick up "
            "a method-side knob."
        ),
    )
    parser.add_argument(
        "--final-repair-reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help=(
            "How much deliberation the closing message is worth. It is a "
            "different kind of work from the round it follows: that round "
            "decides which actions the residual admits, this one reports "
            "receipts that are already fixed and already checked. On the old "
            "pool the phase spends 72.1 percent of its output on reasoning to "
            "produce 188 tokens of prose. Unset inherits the successor effort, "
            "which is what every registered arm ran."
        ),
    )
    parser.add_argument(
        "--refuse-unclosable-last-slot",
        choices=("on", "off"),
        default=(
            "on" if VALIDATED_MAINLINE_PROFILE.refuse_unclosable_last_slot else "off"
        ),
        help=(
            "Whether the last remaining residual slot is refused when the "
            "proposal that would spend it is empty. A compiled residual gets "
            "min(max_tool_rounds, remaining_budget) rounds, so on the last slot "
            "a round that answers with actions has nothing left to pay for the "
            "closing call and the compiler discards the residual whole. The "
            "runner then hands back to the cheap agent, which is where spending "
            "the slot was always going to end. off keeps the slot because the "
            "two pools disagree on whether refusing it is free."
        ),
    )
    parser.add_argument(
        "--max-residual-strong-calls",
        type=int,
        default=MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
        help=(
            "Residual strong-call budget for one row, covering both shadow "
            "rounds and the closing call. The cheap_anchor arm forces 0 "
            "regardless."
        ),
    )
    parser.add_argument(
        "--domain", default="shopping_assistant,travel,customer_support"
    )
    parser.add_argument(
        "--state-bench-root", type=Path, default=DEFAULT_STATE_BENCH_ROOT
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "all"],
        help="Official protocol split to run; use all only for local smoke/debug.",
    )
    parser.add_argument(
        "--cap", type=int, default=0, help="Per-domain cap; 0 means all tasks."
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=150,
        help="Balanced total row cap across domains; 0 means unlimited.",
    )
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--public-output-json", type=Path, default=DEFAULT_PUBLIC_OUTPUT
    )
    parser.add_argument(
        "--private-output-root", type=Path, default=DEFAULT_PRIVATE_ROOT
    )
    parser.add_argument(
        "--private-output-root-label", default=DEFAULT_PRIVATE_ROOT.name
    )
    parser.add_argument(
        "--expected-noapi-report-json",
        type=Path,
        default=None,
        help=(
            "No-API gate report that this paid execution must match before any "
            "provider calls are made. Execute requires this artifact."
        ),
    )
    parser.add_argument("--cheap-model", required=True)
    parser.add_argument("--cheap-reasoning-effort", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--strong-reasoning-effort", required=True)
    parser.add_argument("--user-simulator-model", default="")
    parser.add_argument("--user-simulator-reasoning-effort", default="high")
    parser.add_argument(
        "--allow-noncanonical-model-plan",
        action="store_true",
        help=(
            "Allow a model-plan variant instead of the current registered "
            "STATE-Bench plan. Default fail-closed."
        ),
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--min-cheap-assistant-turns", type=int, default=1)
    method_arm.add_paid_approval_args(parser)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.public_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.public_output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "mode": report["mode"],
                "execute": report["execute"],
                "planned_rows": report["task_plan"]["planned_rows"],
                "completed_rows": report["summary"]["completed_rows"],
                "provider_calls": report["provider_calls"],
                "frontier_found_count": report["summary"]["frontier_found_count"],
                "blockers": report["blockers"],
                "output": str(args.public_output_json),
                "private_output_root_label": args.private_output_root_label,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if report["decision"] == BLOCKED_DECISION else 0


def apply_method_mainline(args: argparse.Namespace) -> None:
    """Normalize experiment attributes programmatic callers may not carry."""

    if getattr(args, "final_repair_reasoning_effort", None) is None:
        args.final_repair_reasoning_effort = None


def build_report(
    args: argparse.Namespace, now: datetime | None = None
) -> dict[str, Any]:
    ensure_arg_defaults(args)
    apply_method_mainline(args)
    now = now or datetime.now(timezone.utc)
    domains = parse_domains(args.domain)
    task_ids = parse_task_ids(args.task_ids)
    planned_rows = planned_row_count(args, domains, task_ids)
    planned_sections = build_planned_public_sections(
        args,
        domains=domains,
        task_ids=task_ids,
        planned_rows=planned_rows,
    )
    expected_noapi_report_consistency = run_plan.expected_noapi_report_consistency(
        args,
        current_plan=planned_sections,
    )
    blockers = sorted(
        set(
            [
                *preflight_blockers(args),
                *expected_noapi_report_consistency["fail_closed_blockers"],
            ]
        )
    )
    receipts: list[dict[str, Any]] = []
    execution_blockers: list[str] = []
    execute_wall_clock_seconds: float | None = None
    if args.execute and not blockers:
        execute_started = time.perf_counter()
        try:
            receipts = execute_tasks(args, domains=domains, task_ids=task_ids)
        except Exception as exc:
            execution_blockers.append(f"execute_failed:{type(exc).__name__}")
            private_error_path = args.private_output_root / "run_error_private.txt"
            private_error_path.parent.mkdir(parents=True, exist_ok=True)
            private_error_path.write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            execute_wall_clock_seconds = round(
                max(0.0, time.perf_counter() - execute_started),
                6,
            )
    blockers.extend(execution_blockers)
    runtime_ledger = RuntimeLedger()
    for receipt in receipts:
        runtime_ledger.merge_public_summary(receipt.get("runtime_ledger"))
    runtime_summary = runtime_ledger.public_summary()
    runtime_summary["phase_ledger"] = runtime_accounting.merge_phase_ledgers(
        [
            receipt.get("runtime_ledger", {}).get("phase_ledger")
            if isinstance(receipt.get("runtime_ledger"), Mapping)
            else None
            for receipt in receipts
        ]
    )
    frontier_found_count = sum(int(bool(row.get("frontier_found"))) for row in receipts)
    runtime_breakdown = runtime_accounting.true_live_runtime_breakdown(
        runtime_summary,
        method_wall_clock_seconds=execute_wall_clock_seconds,
        max_concurrency=args.max_concurrency,
        per_row_wall_seconds=_receipt_wall_seconds(receipts),
        frontier_found_count=frontier_found_count,
        # The accounting falls back to the old pool's models when no role map
        # is supplied, so a run on any other pool priced its cheap role at
        # gpt-5.4-mini rates and understated itself. The models this run
        # actually used are right here in args; pass them.
        role_models={
            "cheap_agent": str(args.cheap_model or ""),
            "residual_strong": str(args.strong_model or ""),
            "user_simulator": str(args.user_simulator_model or ""),
        },
    )
    provider_calls = sum(
        int(receipt.get("provider_calls") or 0) for receipt in receipts
    )
    label_counts = Counter(
        str(receipt.get("final_label") or "") for receipt in receipts
    )
    if args.execute and label_counts.get("provider_failure"):
        blockers.append("provider_failure_receipts_present")
    if args.execute and any(label != "executed" for label in label_counts):
        blockers.append("non_executed_receipts_present")
    if args.execute and len(receipts) < planned_rows:
        blockers.append("completed_rows_below_planned_rows")
    task_quality_floor = 90
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "mode": str(args.mode or "").strip()
        or (
            "state_bench_true_live_cfrc_execute"
            if args.execute
            else "state_bench_true_live_cfrc_dryrun"
        ),
        "surface": str(args.surface or "state_bench_true_live"),
        "execute": bool(args.execute),
        "api_spend_performed": bool(provider_calls),
        "provider_calls": provider_calls,
        "source_bundle": planned_sections["source_bundle"],
        "execution_contract": planned_sections["execution_contract"],
        "model_plan": planned_sections["model_plan"],
        "model_plan_consistency": build_model_plan_consistency(args),
        "expected_noapi_report_consistency": expected_noapi_report_consistency,
        "frontier_policy": planned_sections["frontier_policy"],
        "parallelism": planned_sections["parallelism"],
        "acceptance_policy": {
            "floor": {
                "task_completion_pass_count": task_quality_floor,
                "reference_runs": [
                    "state_bench_true_live_cfrc_r4_convergence_full150",
                    "state_bench_true_live_cfrc_r4_repeat_full150",
                ],
            },
            "target": {
                "task_completion_pass_count": task_quality_floor,
                "reference": "r4_grounded_schema_operating_point",
                "interpretation": ("meet_task_floor_90_before_claiming_r4_level_cost"),
                "residual_strong_resource_target": (
                    "agent_side_raw_cost_usd_must_remain_near_the_r4_runs"
                ),
                "agent_side_raw_cost_usd_ceiling": 21.5,
                "agent_side_raw_cost_usd_baseline": 21.26776025,
                "agent_side_raw_cost_usd_anchor": 20.1191705,
                "anchor_reference": "r4_primary_and_repeat_agent_side_raw_cost_usd",
                "residual_strong_input_token_reduction_min": None,
                "residual_strong_input_token_baseline": 1822558,
                "provider_latency_is_gate": False,
            },
            "resource_metrics": [
                "agent_side_raw_cost_usd",
                "residual_strong_input_tokens",
                "residual_strong_provider_latency_seconds",
                "residual_strong_input_tokens_per_frontier",
                "residual_strong_total_tokens_appendix",
            ],
            "regression_diagnosis": {
                "task_drop": (
                    "inspect frontier authority, compiler result, or strong proposal fields"
                ),
                "state_requirement_and_ux": ("diagnostic_only_not_acceptance_floors"),
            },
        },
        "task_plan": planned_sections["task_plan"],
        "summary": {
            "completed_rows": len(receipts),
            "frontier_found_count": frontier_found_count,
            "trajectory_written_count": sum(
                int(bool(row.get("trajectory_written"))) for row in receipts
            ),
            "label_counts": dict(sorted(label_counts.items())),
            "controller_telemetry": _controller_telemetry(receipts),
            "runtime_ledger": runtime_summary,
            "runtime_breakdown": runtime_breakdown,
            "official_rescoring_required": bool(args.execute and receipts),
        },
        "runner_capabilities": {
            "true_live_cheap_provider_calls": True,
            "candidate_manifest_required": False,
            "saved_cheap_trajectory_required": False,
            "compiler_constrained_controller": True,
            "private_trajectories_written": bool(args.execute and receipts),
            "public_report_discloses_private_paths": False,
        },
        "private_output_summary": {
            "label": args.private_output_root_label,
            "raw_trajectories_private_only": True,
            "raw_provider_payload_private_only": True,
        },
        "claim_boundary": [
            "This is a true-live STATE-Bench method runner: cheap is called online inside the official harness.",
            "The frontier decision is made from the current visible prefix only; no saved cheap suffix or candidate manifest is used.",
            "The run writes private trajectories for official rescoring; the public report is not itself a scored result.",
            "The CFRC arm constrains residual strong through the current compiler and execution guard.",
        ],
        "blockers": sorted(set(blockers)),
    }
    report["decision"] = (
        BLOCKED_DECISION
        if report["blockers"]
        else COMPLETE_DECISION
        if args.execute and receipts
        else READY_NOT_EXECUTED_DECISION
    )
    leakage = public_leakage_audit(report)
    report["public_leakage_audit"] = leakage
    if not leakage["pass"]:
        report["blockers"] = sorted(
            {*report["blockers"], "public_leakage_marker_present"}
        )
        report["decision"] = BLOCKED_DECISION
    return report


def _receipt_wall_seconds(receipts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for receipt in receipts:
        value = receipt.get("method_wall_clock_seconds")
        if isinstance(value, int | float):
            values.append(max(0.0, float(value)))
    return values


_CONTROLLER_SCALAR_FIELDS = (
    "staged_precommit_preview_count",
    "staged_precommit_execute_count",
    "staged_precommit_cancel_count",
    "shadow_convergence_attempt_count",
    "shadow_reserved_closure_count",
    "shadow_non_progress_stop_count",
    "shadow_convergence_safety_block_count",
    "final_repair_invoked_count",
    "residual_graph_count",
    "residual_graph_node_count",
    "residual_graph_tool_node_count",
    "residual_graph_edge_count",
)
_CONTROLLER_COUNTER_FIELDS = ("shadow_convergence_skip_reasons",)


def _controller_telemetry(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_totals: Counter[str] = Counter()
    counter_totals = {field: Counter() for field in _CONTROLLER_COUNTER_FIELDS}
    for receipt in receipts:
        summary = receipt.get("public_agent_summary")
        if not isinstance(summary, Mapping):
            continue
        scalar_totals.update(
            {field: int(summary.get(field) or 0) for field in _CONTROLLER_SCALAR_FIELDS}
        )
        for field, total in counter_totals.items():
            values = summary.get(field)
            if isinstance(values, Mapping):
                total.update(
                    {
                        str(reason): int(count)
                        for reason, count in values.items()
                        if isinstance(count, int)
                    }
                )
    return {
        **{field: scalar_totals[field] for field in _CONTROLLER_SCALAR_FIELDS},
        **{
            field: dict(sorted(counter_totals[field].items()))
            for field in _CONTROLLER_COUNTER_FIELDS
        },
    }


def build_planned_public_sections(
    args: argparse.Namespace,
    *,
    domains: list[str],
    task_ids: set[str],
    planned_rows: int | None,
) -> dict[str, Any]:
    return {
        "surface": str(args.surface or "state_bench_true_live"),
        "source_bundle": _source_bundle(),
        "execution_contract": online_execution_contract(_args_profile(args)),
        "model_plan": build_model_plan_report(args),
        "frontier_policy": build_frontier_policy(args),
        "parallelism": {
            "max_concurrency": args.max_concurrency,
        },
        "task_plan": {
            "domains": domains,
            "split": str(args.split or "test"),
            "per_domain_cap": None if args.cap == 0 else args.cap,
            "max_total_rows": None if args.max_total_rows == 0 else args.max_total_rows,
            "task_id_filter_supplied": bool(task_ids),
            "task_ids_publicly_disclosed": False,
            "planned_rows": planned_rows,
        },
    }


def _args_profile(args: argparse.Namespace) -> StateBenchCFRCProfile:
    """The operating point these arguments select.

    The cheap anchor carries no successor, so it publishes a zero budget and no
    declaration rather than inheriting the method defaults.
    """

    if str(getattr(args, "arm", "mainline")) == "cheap_anchor":
        return StateBenchCFRCProfile(
            max_residual_strong_calls=0,
            declare_remaining_responses=False,
            declare_closure_admissibility=False,
            refuse_unclosable_last_slot=False,
        )
    return StateBenchCFRCProfile(
        max_residual_strong_calls=int(
            getattr(
                args,
                "max_residual_strong_calls",
                MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
            )
        ),
        declare_remaining_responses=(
            str(getattr(
                args,
                "residual_round_budget_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_remaining_responses else "off",
            )) == "on"
        ),
        declare_closure_admissibility=(
            str(getattr(
                args,
                "closure_obligation_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_closure_admissibility else "off",
            )) == "on"
        ),
        refuse_unclosable_last_slot=(
            str(getattr(
                args,
                "refuse_unclosable_last_slot",
                "on" if VALIDATED_MAINLINE_PROFILE.refuse_unclosable_last_slot else "off",
            )) == "on"
        ),
        closing_contract_style=str(
            getattr(
                args,
                "closing_contract_style",
                VALIDATED_MAINLINE_PROFILE.closing_contract_style,
            )
        ),
    )


def _source_bundle() -> dict[str, Any]:
    """Bind the provider-zero gate to the implementation it approved."""

    implementation_paths = [Path(__file__).resolve()]
    implementation_paths.extend(
        (REPO_ROOT / "src/papc/certspec/state_bench").rglob("*.py")
    )
    implementation_paths.extend(
        [
            REPO_ROOT / "src/papc/certspec/costing.py",
            REPO_ROOT / "src/papc/certspec/provider.py",
            REPO_ROOT / "src/papc/certspec/residual_closure.py",
            REPO_ROOT / "src/papc/certspec/runtime_ledger.py",
            REPO_ROOT / "src/papc/certspec/state_write_contract.py",
            REPO_ROOT
            / "src/papc/certspec/tau2/request_obligation_contract.py",
            REPO_ROOT
            / "scripts/state_bench/common/papc_method_arm_execution_adapter.py",
        ]
    )
    records = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted({path.resolve() for path in implementation_paths})
    ]
    digest = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "implementation_sha256": digest,
        "implementation_file_count": len(records),
    }


def build_frontier_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the profile the selected arm will execute."""

    arm = str(getattr(args, "arm", "mainline"))
    cheap_anchor = arm == "cheap_anchor"
    domain_prompt_suffix = str(getattr(args, "domain_prompt_suffix", "on"))
    policy: dict[str, Any] = {
        "arm": arm,
        "domain_prompt_suffix": domain_prompt_suffix,
        # The transcript shape the cheap model reads. Both arms carry the same
        # setting, so an artifact states which shape produced its numbers.
        "cheap_native_item_loop": (
            str(getattr(args, "cheap_native_item_loop", "off")) == "on"
        ),
        "method": (
            "cheap_anchor_single_model" if cheap_anchor else "r4_grounded_schema_cfrc"
        ),
        # The anchor arm carries no method-side component, so it must not be
        # published under the component list of the method it is a denominator
        # for.
        "components": (
            []
            if cheap_anchor
            else [
                "reached_state_contract_construction",
                "executable_residual_completion",
                "checked_execution",
            ]
        ),
        "frontier": "visible_prefix_only",
        "reached_state_contract": "frontier_record_and_allowed_tool_surface",
        "residual_completion": "receipt_grounded_incremental_residual_graph",
        "checked_execution": "graph_tool_surface_and_grounded_admission",
        "terminal_completion": "single_bounded_strong_closure",
        "final_repair_role": (
            "disabled"
            if cheap_anchor
            else str(getattr(args, "final_repair_role", "strong") or "strong")
        ),
        "final_repair_context": (
            "disabled"
            if cheap_anchor
            else "benchmark_policy_domain_suffix_visible_tail_and_receipts"
        ),
        "min_cheap_assistant_turns": args.min_cheap_assistant_turns,
        "cheap_max_tool_calls_per_response": (
            0
            if cheap_anchor
            else int(getattr(args, "cheap_max_tool_calls_per_response", 32))
        ),
        "max_residual_strong_calls": (
            0
            if cheap_anchor
            else int(
                getattr(
                    args,
                    "max_residual_strong_calls",
                    MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
                )
            )
        ),
        "residual_round_budget_declared": (
            False
            if cheap_anchor
            else str(
                getattr(
                args,
                "residual_round_budget_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_remaining_responses else "off",
            )
            )
            == "on"
        ),
        "closing_contract_style": str(
            getattr(
                args,
                "closing_contract_style",
                VALIDATED_MAINLINE_PROFILE.closing_contract_style,
            )
        ),
        "closure_obligation_declared": (
            False
            if cheap_anchor
            else str(getattr(
                args,
                "closure_obligation_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_closure_admissibility else "off",
            )) == "on"
        ),
        "refuse_unclosable_last_slot": (
            False
            if cheap_anchor
            else str(getattr(
                args,
                "refuse_unclosable_last_slot",
                "on" if VALIDATED_MAINLINE_PROFILE.refuse_unclosable_last_slot else "off",
            )) == "on"
        ),
        "final_repair_reasoning_effort": (
            None
            if cheap_anchor
            else getattr(args, "final_repair_reasoning_effort", None)
        ),
        "selector_reads_scorer_fields": False,
    }
    if cheap_anchor:
        policy["cheap_anchor_disables"] = sorted(
            [
                "strong_calls",
                "cheap_batch_bound",
                "cheap_no_result_read_bound",
                # The suffix is the one anchor component that is switchable, so
                # the same incumbent can be measured with and without the domain
                # prompt the method arm receives.
                *(["system_prompt_suffix"] if domain_prompt_suffix == "off" else []),
            ]
        )
    return policy


def ensure_arg_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "mode": "",
        "surface": "state_bench_true_live",
        "arm": "mainline",
        "domain_prompt_suffix": "on",
        "cheap_native_item_loop": "off",
        "cheap_max_tool_calls_per_response": 32,
        "max_residual_strong_calls": MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
        "successor_tool_descriptions": "on",
        "residual_round_budget_declared": (
            "on" if VALIDATED_MAINLINE_PROFILE.declare_remaining_responses else "off"
        ),
        "closure_obligation_declared": (
            "on" if VALIDATED_MAINLINE_PROFILE.declare_closure_admissibility else "off"
        ),
        "refuse_unclosable_last_slot": (
            "on" if VALIDATED_MAINLINE_PROFILE.refuse_unclosable_last_slot else "off"
        ),
        "closing_contract_style": VALIDATED_MAINLINE_PROFILE.closing_contract_style,
        "domain": "shopping_assistant,travel,customer_support",
        "state_bench_root": DEFAULT_STATE_BENCH_ROOT,
        "split": "test",
        "cap": 0,
        "max_total_rows": 150,
        "task_ids": "",
        "public_output_json": DEFAULT_PUBLIC_OUTPUT,
        "private_output_root": DEFAULT_PRIVATE_ROOT,
        "private_output_root_label": DEFAULT_PRIVATE_ROOT.name,
        "expected_noapi_report_json": None,
        "cheap_model": "",
        "cheap_reasoning_effort": "high",
        "strong_model": "",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "",
        "user_simulator_reasoning_effort": "high",
        "allow_noncanonical_model_plan": False,
        "max_concurrency": 4,
        "request_timeout_seconds": 180.0,
        "api_base": "",
        "api_key_env": "OPENAI_API_KEY",
        "approve_paid_method_arm_run": False,
        "approval_scope": "",
        "execute": False,
        "min_cheap_assistant_turns": 1,
        "final_repair_reasoning_effort": None,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def preflight_blockers(args: argparse.Namespace) -> list[str]:
    """Validate only inputs that can vary on the supported method path."""

    ensure_arg_defaults(args)
    apply_method_mainline(args)
    method_arm.ensure_paid_approval_defaults(args)
    blockers = [
        *method_arm.paid_execution_preflight_blockers(args),
        *_variable_input_blockers(args),
    ]
    if args.execute:
        blockers.extend(_execute_input_blockers(args))
    return sorted(set(blockers))


def _variable_input_blockers(args: argparse.Namespace) -> list[str]:
    checks = (
        (not parse_domains(args.domain), "domain_list_empty"),
        (
            str(getattr(args, "arm", "mainline")) not in METHOD_ARMS,
            "arm_must_be_a_registered_measured_arm",
        ),
        (str(args.split or "test") not in {"test", "all"}, "split_must_be_test_or_all"),
        (args.cap < 0, "cap_must_be_non_negative"),
        (args.max_total_rows < 0, "max_total_rows_must_be_non_negative"),
        (args.max_concurrency < 1, "max_concurrency_must_be_positive"),
        (
            args.min_cheap_assistant_turns < 0,
            "min_cheap_assistant_turns_must_be_non_negative",
        ),
        (not str(args.cheap_model or "").strip(), "cheap_model_required"),
        (not str(args.strong_model or "").strip(), "strong_model_required"),
        (
            bool(model_plan_mismatches(args))
            and not bool(getattr(args, "allow_noncanonical_model_plan", False)),
            NONCANONICAL_MODEL_PLAN_BLOCKER,
        ),
    )
    return [message for invalid, message in checks if invalid]


def _execute_input_blockers(args: argparse.Namespace) -> list[str]:
    private_root = Path(args.private_output_root)
    api_key_name = str(args.api_key_env or "")
    checks = (
        (
            not str(args.user_simulator_model or "").strip(),
            "user_simulator_model_required_for_execute",
        ),
        (not str(args.mode or "").strip(), "mode_required_for_execute"),
        (
            not str(private_root).startswith(f"{DATA_ROOT}/papc/"),
            "private_output_root_must_be_under_papc_data",
        ),
        (
            private_root.exists() and any(private_root.iterdir()),
            "private_output_root_must_be_empty_or_absent",
        ),
        (not str(args.api_base or "").strip(), "api_base_required_for_execute"),
        (
            not api_key_name.strip() or not os.environ.get(api_key_name),
            "api_key_env_missing_for_execute",
        ),
        (
            run_plan.expected_noapi_report_path(args) is None,
            EXPECTED_NOAPI_REPORT_REQUIRED_BLOCKER,
        ),
        (
            args.max_concurrency > MAX_EXECUTE_CONCURRENCY,
            f"max_concurrency_must_not_exceed_{MAX_EXECUTE_CONCURRENCY}_for_execute",
        ),
        (not Path(args.state_bench_root).exists(), "state_bench_root_missing"),
    )
    return [message for invalid, message in checks if invalid]


def build_model_plan_report(args: argparse.Namespace) -> dict[str, Any]:
    return run_plan.build_model_plan_report(args)


def build_model_plan_consistency(args: argparse.Namespace) -> dict[str, Any]:
    return run_plan.build_model_plan_consistency(args)


def canonical_model_plan_report() -> dict[str, str]:
    return run_plan.canonical_model_plan_report()


def model_plan_mismatches(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    return run_plan.model_plan_mismatches(args)


def _execution_jobs(
    args: argparse.Namespace,
    *,
    domains: list[str],
    task_ids: set[str],
) -> list[tuple[str, int, Any]]:
    state_bench = scaffold.load_state_bench(args.state_bench_root)
    jobs: list[tuple[str, int, Any]] = []
    for domain_index, domain_name in enumerate(domains):
        domain_limit = domain_row_limit(args, domains, domain_index)
        if args.cap <= 0 and args.max_total_rows > 0 and domain_limit == 0:
            continue
        domain_tasks = load_tasks_for_domain(
            state_bench=state_bench,
            domain_name=domain_name,
            task_ids=task_ids,
            split=str(args.split or "test"),
            cap=domain_limit,
        )
        base_index = len(jobs)
        jobs.extend(
            (domain_name, base_index + index, task)
            for index, task in enumerate(domain_tasks)
        )
    return jobs


def execute_tasks(
    args: argparse.Namespace,
    *,
    domains: list[str],
    task_ids: set[str],
) -> list[dict[str, Any]]:
    jobs = _execution_jobs(args, domains=domains, task_ids=task_ids)
    args.private_output_root.mkdir(parents=True, exist_ok=True)
    receipts_dir = args.private_output_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    stop_submitting = False
    job_iter = iter(jobs)
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_concurrency))) as executor:
        future_map = {}

        def submit_next() -> bool:
            try:
                domain_name, row_index, task = next(job_iter)
            except StopIteration:
                return False
            future_map[
                executor.submit(execute_one, args, domain_name, row_index, task)
            ] = (
                domain_name,
                row_index,
                task,
            )
            return True

        for _ in range(min(max(1, int(args.max_concurrency)), len(jobs))):
            submit_next()
        while future_map:
            done, _pending = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done:
                domain_name, row_index, task = future_map.pop(future)
                receipt = _receipt_from_future(future, domain_name, row_index, task)
                receipt_path = receipts_dir / f"receipt_{row_index:04d}.json"
                receipt_path.write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                receipts.append(public_receipt(receipt))
                if _is_provider_usage_limit_failure(receipt):
                    stop_submitting = True
            while not stop_submitting and len(future_map) < max(
                1, int(args.max_concurrency)
            ):
                if not submit_next():
                    break
    return sorted(receipts, key=lambda row: (row["domain"], row["row_index"]))


def _receipt_from_future(
    future: Any,
    domain_name: str,
    row_index: int,
    task: Any,
) -> dict[str, Any]:
    try:
        return future.result()
    except Exception as exc:
        return {
            "schema": "papc.state_bench_true_live_task_receipt.v1",
            "row_index": row_index,
            "domain": domain_name,
            "final_label": "provider_failure",
            "trajectory_written": False,
            "frontier_found": False,
            "provider_calls": 0,
            "runtime_ledger": RuntimeLedger().public_summary(),
            "private": {
                "task_id": str(getattr(task, "task_id", "")),
                "error_type": type(exc).__name__,
                "trace": traceback.format_exc(),
            },
        }


def _is_provider_usage_limit_failure(receipt: Mapping[str, Any]) -> bool:
    if str(receipt.get("final_label") or "") != "provider_failure":
        return False
    private = (
        receipt.get("private") if isinstance(receipt.get("private"), Mapping) else {}
    )
    error_type = str(private.get("error_type") or "").lower()
    trace = str(private.get("trace") or "").lower()
    return (
        "ratelimit" in error_type
        and "usage limit" in trace
        and "has been reached" in trace
    )


def execute_one(
    args: argparse.Namespace,
    domain_name: str,
    row_index: int,
    task: Any,
) -> dict[str, Any]:
    trajectory_path = (
        args.private_output_root
        / "official_trajectories"
        / domain_name
        / "run1"
        / f"row_{row_index:04d}.json"
    )
    return execute_true_live_task(
        state_bench_root=args.state_bench_root,
        domain_name=domain_name,
        task=task,
        output_path=trajectory_path,
        cheap_model=args.cheap_model,
        cheap_reasoning_effort=args.cheap_reasoning_effort,
        strong_model=args.strong_model,
        strong_reasoning_effort=args.strong_reasoning_effort,
        user_simulator_model=args.user_simulator_model,
        user_simulator_reasoning_effort=args.user_simulator_reasoning_effort,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        request_timeout_seconds=args.request_timeout_seconds,
        min_cheap_assistant_turns=args.min_cheap_assistant_turns,
        cheap_max_tool_calls_per_response=int(
            getattr(args, "cheap_max_tool_calls_per_response", 32)
        ),
        max_residual_strong_calls=int(
            getattr(
                args,
                "max_residual_strong_calls",
                MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
            )
        ),
        successor_tool_descriptions=(
            str(getattr(args, "successor_tool_descriptions", "on")) == "on"
        ),
        residual_round_budget_declared=(
            str(getattr(
                args,
                "residual_round_budget_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_remaining_responses else "off",
            )) == "on"
        ),
        closure_obligation_declared=(
            str(getattr(
                args,
                "closure_obligation_declared",
                "on" if VALIDATED_MAINLINE_PROFILE.declare_closure_admissibility else "off",
            )) == "on"
        ),
        closing_contract_style=str(
            getattr(
                args,
                "closing_contract_style",
                VALIDATED_MAINLINE_PROFILE.closing_contract_style,
            )
        ),
        refuse_unclosable_last_slot=(
            str(getattr(
                args,
                "refuse_unclosable_last_slot",
                "on" if VALIDATED_MAINLINE_PROFILE.refuse_unclosable_last_slot else "off",
            )) == "on"
        ),
        final_repair_role=str(getattr(args, "final_repair_role", "strong") or "strong"),
        final_repair_reasoning_effort=getattr(
            args, "final_repair_reasoning_effort", None
        ),
        arm=str(getattr(args, "arm", "mainline")),
        domain_prompt_suffix=str(getattr(args, "domain_prompt_suffix", "on")),
        cheap_native_item_loop=(
            str(getattr(args, "cheap_native_item_loop", "off")) == "on"
        ),
        row_index=row_index,
    )


def public_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        "row_index": int(receipt.get("row_index") or 0),
        "domain": str(receipt.get("domain") or ""),
        "final_label": str(receipt.get("final_label") or ""),
        "trajectory_written": bool(receipt.get("trajectory_written")),
        "frontier_found": bool(receipt.get("frontier_found")),
        "provider_calls": int(receipt.get("provider_calls") or 0),
        "method_wall_clock_seconds": receipt.get("method_wall_clock_seconds"),
        "runtime_ledger": receipt.get("runtime_ledger"),
    }
    summary = receipt.get("public_agent_summary")
    if isinstance(summary, Mapping):
        public["public_agent_summary"] = dict(summary)
    return public


def load_tasks_for_domain(
    *,
    state_bench: Mapping[str, Any],
    domain_name: str,
    task_ids: set[str],
    split: str,
    cap: int,
) -> list[Any]:
    task_paths = task_paths_for_domain(
        state_bench=state_bench,
        domain_name=domain_name,
        task_ids=task_ids,
        split=split,
        cap=cap,
    )
    return [state_bench["TaskDefinition"].load(path) for path in task_paths]


def task_paths_for_domain(
    *,
    state_bench: Mapping[str, Any],
    domain_name: str,
    task_ids: set[str],
    split: str,
    cap: int,
) -> list[Path]:
    task_dir = state_bench["domain_tasks_dir"](domain_name)
    paths_by_id = {path.stem: path for path in sorted(Path(task_dir).glob("*.json"))}
    if task_ids:
        task_paths = [
            path for task_id, path in sorted(paths_by_id.items()) if task_id in task_ids
        ]
    elif split != "all":
        split_ids = state_bench["load_split_task_ids"](
            domain_name,
            split,
            state_bench["protocol_split_version"],
        )
        task_paths = [
            paths_by_id[task_id] for task_id in split_ids if task_id in paths_by_id
        ]
    else:
        task_paths = list(paths_by_id.values())
    if cap > 0:
        task_paths = task_paths[:cap]
    return task_paths


def planned_row_count(
    args: argparse.Namespace, domains: list[str], task_ids: set[str]
) -> int | None:
    if args.cap > 0:
        return len(domains) * args.cap
    if Path(args.state_bench_root).exists():
        try:
            state_bench = scaffold.load_state_bench(args.state_bench_root)
            total = 0
            for domain in domains:
                total += len(
                    task_paths_for_domain(
                        state_bench=state_bench,
                        domain_name=domain,
                        task_ids=task_ids,
                        split=str(args.split or "test"),
                        cap=0,
                    )
                )
            if args.max_total_rows > 0:
                return min(total, args.max_total_rows)
            return total
        except Exception:
            return None
    if task_ids:
        return len(task_ids)
    if args.max_total_rows > 0:
        return args.max_total_rows
    return None


def domain_row_limit(
    args: argparse.Namespace, domains: list[str], domain_index: int
) -> int:
    if args.cap > 0:
        return args.cap
    if args.max_total_rows <= 0 or not domains:
        return 0
    base = args.max_total_rows // len(domains)
    remainder = args.max_total_rows % len(domains)
    return base + (1 if domain_index < remainder else 0)


def parse_domains(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    out: list[str] = []
    for value in values:
        if value not in DOMAINS:
            continue
        if value not in out:
            out.append(value)
    return out


def parse_task_ids(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


if __name__ == "__main__":
    raise SystemExit(main())

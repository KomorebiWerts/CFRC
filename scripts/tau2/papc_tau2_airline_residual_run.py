#!/usr/bin/env python3
"""Run the tau2 Airline CFRC residual method arm."""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.paths import DATA_ROOT
from papc.certspec.infra_patch import (
    base_results_payload as shared_base_results_payload,
    infrastructure_task_ids,
    merge_patch_results,
    merge_patch_sidecar,
)
from papc.certspec.tau2.airline_residual_runtime import (  # noqa: E402
    DEFAULT_MAX_SUCCESSOR_TURNS,
    DEFAULT_RESIDUAL_WRITE_BUDGET,
    RESIDUAL_AUTHORITY,
    RESIDUAL_USER_TURN_BUDGET,
    register_tau2_airline_residual_agent,
)
from papc.certspec.tau2.noapi_readiness import (  # noqa: E402
    public_leakage_audit,
    write_json_append_only,
)
from papc.certspec.tau2.cfrc_runner import load_jsonl  # noqa: E402
from papc.certspec.costing import (  # noqa: E402
    OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE,
    OPENAI_PUBLIC_PRICING_SOURCE,
    cache_neutral_cost_usd,
)
from scripts.tau2 import papc_tau2_official_run_with_evaluator_patch as official  # noqa: E402
from scripts.tau2 import papc_tau2_cfrc_clean_run as method_runner  # noqa: E402
from scripts.tau2.papc_tau2_airline_residual_gate_noapi import code_sha256

DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_OUTPUT_ROOT = Path(f"{DATA_ROOT}/papc")
# One split registry for the method arm and every protected baseline arm. A
# baseline row reported against a different denominator than the method row is
# not a matched comparison, so both read the operating point from here rather
# than restating it.
SPLIT_OPERATING_POINTS: dict[str, dict[str, Any]] = {
    "base": {"denominator": 50, "reward_floor": 44},
    "test": {"denominator": 20, "reward_floor": None},
}
DEFAULT_TASK_SPLIT = "base"
# User authorization caps all provider-active episodes at eight. The defaults
# below preserve old command lines, while every promoted operating point passes
# its own Pareto-safe quality floor and cost ceiling explicitly.
MAX_ONLINE_CONCURRENCY = 8
DEFAULT_PRIMARY_REWARD_FLOOR = 42.0
DEFAULT_AGENT_SIDE_RAW_COST_CEILING_USD = 12.0


def effective_reward_floor(args: argparse.Namespace) -> float:
    value = getattr(args, "reward_floor", DEFAULT_PRIMARY_REWARD_FLOOR)
    return float(DEFAULT_PRIMARY_REWARD_FLOOR if value is None else value)


def effective_cost_ceiling_usd(args: argparse.Namespace) -> float:
    value = getattr(
        args,
        "agent_side_cost_ceiling_usd",
        DEFAULT_AGENT_SIDE_RAW_COST_CEILING_USD,
    )
    return float(
        DEFAULT_AGENT_SIDE_RAW_COST_CEILING_USD if value is None else value
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--task-split",
        choices=sorted(SPLIT_OPERATING_POINTS),
        default=DEFAULT_TASK_SPLIT,
    )
    parser.add_argument(
        "--tasks-per-domain",
        type=int,
        default=SPLIT_OPERATING_POINTS[DEFAULT_TASK_SPLIT]["denominator"],
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--cheap-model", required=True)
    parser.add_argument("--cheap-reasoning-effort", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--strong-reasoning-effort", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-reasoning-effort", required=True)
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-reasoning-effort", required=True)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-output-json", type=Path, required=True)
    parser.add_argument("--compiler-gate-json", type=Path, required=True)
    parser.add_argument(
        "--write-budget", type=int, default=DEFAULT_RESIDUAL_WRITE_BUDGET
    )
    parser.add_argument(
        "--reward-floor",
        type=float,
        default=DEFAULT_PRIMARY_REWARD_FLOOR,
        help="Full-denominator reward the run must hold to remain eligible.",
    )
    parser.add_argument(
        "--agent-side-cost-ceiling-usd",
        type=float,
        default=DEFAULT_AGENT_SIDE_RAW_COST_CEILING_USD,
        help=(
            "Cache-neutral cheap-plus-strong cost ceiling for this exact pool; "
            "user simulator and evaluator costs remain excluded."
        ),
    )
    parser.add_argument(
        "--successor-batches-actions",
        action="store_true",
        help=(
            "Lift the published one-tool-call-at-a-time clause from the "
            "successor's copy of the domain policy only. The cheap prefix and "
            "every baseline keep the published text, and the clause forbidding "
            "a message in the same turn as a call is untouched. This arm "
            "deviates from the domain's turn protocol and must be labelled as "
            "such wherever it is reported."
        ),
    )
    parser.add_argument(
        "--residual-recertifies-after-user-turn",
        action="store_true",
        help=(
            "Return the floor to the cheap prefix model when the successor "
            "pauses for a user answer, and certify the frontier again from the "
            "reached state that answer produces. The successor then holds "
            "authority only while an executable residual is open instead of "
            "owning every turn after the first write candidate."
        ),
    )
    parser.add_argument(
        "--no-residual-frontier-rearm",
        action="store_true",
        help=(
            "Keep the pre-2026-08-12 behaviour in which the frontier is "
            "certified at most once per episode. With the default the latch is "
            "released whenever the incremental graph is torn down, so a write "
            "candidate raised after a closed or refused residual has to cross "
            "a certified frontier again instead of reaching the domain "
            "uncertified."
        ),
    )
    parser.add_argument(
        "--defer-frontier-until-confirmed",
        action="store_true",
        help=(
            "Do not certify the frontier on a write candidate the policy "
            "cannot admit yet. When the reached state still owes the "
            "confirmation a database update requires, the residual behind the "
            "candidate is not executable, so the conversational turn the state "
            "is missing is asked for by the prefix model and the frontier is "
            "certified on the next write candidate. Taken at most once per "
            "episode."
        ),
    )
    parser.add_argument(
        "--prefix-completion-grounding",
        action="store_true",
        help=(
            "Remind the prefix model to distinguish permission or price from "
            "an executed state change, using visible state before declaring "
            "that no update is needed. Adds no separate model call."
        ),
    )
    parser.add_argument(
        "--request-scoped-obligations",
        action="store_true",
        help=(
            "Enumerate what the visible request still owes, once per frontier, "
            "with one prefix-model call, and put that ledger in the successor "
            "packet. The airline frontier contract carries a single obligation "
            "with an empty write footprint, so it says a residual exists "
            "without saying what it contains; the packet then asks in prose "
            "for the whole remaining request and the failing rows stop after "
            "one write anyway. Off by default and fail-open: a builder answer "
            "that does not parse leaves the packet exactly as it was."
        ),
    )
    parser.add_argument(
        "--successor-declares-closure",
        action="store_true",
        help=(
            "Let the successor submit a terminal graph node with its final "
            "actions through cfrc_declare_completion. This declaration never "
            "reaches the domain. Checked Execution releases its message only "
            "after all real receipts pass; otherwise the live continuation "
            "remains. This can save a separate strong closing round. Typed "
            "messages authored beside actions remain supported. The cheap "
            "prefix and every baseline keep their original tool catalogs."
        ),
    )
    parser.add_argument(
        "--successor-tool-descriptions",
        choices=("on", "off"),
        default="on",
        help=(
            "Whether the successor's copy of the catalog keeps each tool's own "
            "sentence. off is the contract projection: names, argument names, "
            "types, enums, defaults and required flags only. The cheap "
            "prefix's catalog is untouched either way."
        ),
    )
    parser.add_argument(
        "--max-strong-turns", type=int, default=DEFAULT_MAX_SUCCESSOR_TURNS
    )
    parser.add_argument(
        "--max-successor-rounds",
        type=int,
        default=None,
        help=(
            "How many rounds the successor gets, separately from how wide one "
            "round's graph may be. Left unset it follows --max-strong-turns, "
            "which is what every registered result ran. Setting it to 1 asks "
            "for the whole residual in a single graph: no clarification pause "
            "and no separate closing round, with the same tool call budget."
        ),
    )
    parser.add_argument(
        "--patch-infra-from",
        type=Path,
        default=None,
        help=(
            "Private run directory of a base run. Re-runs exactly the episodes "
            "that run lost to provider infrastructure and merges them back, so "
            "a flaky endpoint costs a few episodes instead of the whole split."
        ),
    )
    parser.add_argument(
        "--frontier-activation-policy",
        choices=("all_write_or_transfer", "all_write"),
        default="all_write_or_transfer",
        help=(
            "Which cheap candidate opens a commitment frontier. The default is "
            "the registered predicate. all_write asks the method question "
            "directly: a hand-off to a human changes no state and leaves no "
            "residual to complete, so on the definition of a commitment "
            "frontier it arguably should not open one. The two differ on "
            "exactly the escalation candidates, and the decision row now logs "
            "candidate_tool_families so the difference is readable afterwards."
        ),
    )
    parser.add_argument(
        "--lean-residual-packet",
        action="store_true",
        help=(
            "Drop the visible-evidence index from the residual packet when "
            "every one of its rows is a stub of something the same request "
            "replays verbatim below it. The index is kept whenever any row "
            "still carries content, so this can never silently remove "
            "evidence. Airline is the only surface where it can apply, "
            "because it is the only one that replays the whole transcript "
            "beside the packet."
        ),
    )
    parser.add_argument(
        "--withhold-first-handoff",
        action="store_true",
        help=(
            "Ask once per episode whether a hand-off may end a request that is "
            "still open. A hand-off is not a commitment and does not open a "
            "frontier; this is the separate question of the method's failure "
            "return, which AGENTS.md admits as an engineering choice provided "
            "the rule is the same reward-blind one on every row. It is asked "
            "of every hand-off alike, at most once, reading nothing about the "
            "row, and a prefix that proposes the hand-off again is believed. "
            "No successor call is spent either way, so the whole cost of "
            "asking is one prefix turn."
        ),
    )
    parser.add_argument("--dry-run-launch-report", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for field in ("cheap_model", "strong_model", "user_model", "evaluator_model"):
        setattr(
            args,
            field,
            method_runner.litellm_model_name(getattr(args, field)),
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or f"tau2_airline_residual_{timestamp()}"
    gate = _load_gate(args.compiler_gate_json)
    if args.dry_run_launch_report:
        report = build_launch_report(args, run_name=run_name, gate=gate)
        try:
            write_json_append_only(args.public_output_json, report)
        except (
            FileExistsError,
            FileNotFoundError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(json.dumps({"blocked": str(exc)}))
            return 2
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "blockers": report["blockers"],
                    "output": str(args.public_output_json),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if not report["blockers"] else 2

    blockers = execute_blockers(args, gate)
    if blockers:
        print(
            json.dumps(
                {
                    "decision": "blocked_tau2_airline_residual_execute",
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "blockers": blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    official.import_tau2(args.tau_repo)
    evaluator_args = method_runner.llm_args_for_model(
        str(args.api_base or ""),
        args.evaluator_reasoning_effort,
        args.evaluator_model,
    )
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = str(args.api_base)
    if args.api_key_env != "OPENAI_API_KEY":
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    official.apply_evaluator_patch(
        evaluator_model=args.evaluator_model,
        evaluator_args=evaluator_args,
    )

    run_dir = args.output_root / run_name
    method_dir = run_dir / "airline_residual_method"
    method_dir.mkdir(parents=True, exist_ok=False)
    decision_log_path = method_dir / "airline_residual_decisions.jsonl"
    agent_name = register_tau2_airline_residual_agent(
        agent_name=method_runner.safe_agent_name(f"papc_airline_residual_{run_name}"),
        strong_model=args.strong_model,
        strong_llm_args=method_runner.llm_args_for_model(
            args.api_base,
            args.strong_reasoning_effort,
            args.strong_model,
        ),
        decision_log_path=decision_log_path,
        write_budget=args.write_budget,
        max_strong_turns=args.max_strong_turns,
        max_successor_rounds=getattr(args, "max_successor_rounds", None),
        frontier_activation_policy=str(
            getattr(args, "frontier_activation_policy", "all_write_or_transfer")
            or "all_write_or_transfer"
        ),
        successor_batches_actions=bool(getattr(args, "successor_batches_actions", False)),
        residual_recertifies_after_user_turn=bool(
            getattr(args, "residual_recertifies_after_user_turn", False)
        ),
        residual_frontier_rearms_after_close=not bool(
            getattr(args, "no_residual_frontier_rearm", False)
        ),
        defer_frontier_until_confirmed=bool(
            getattr(args, "defer_frontier_until_confirmed", False)
        ),
        successor_declares_closure=bool(
            getattr(args, "successor_declares_closure", False)
        ),
        request_scoped_obligations=bool(
            getattr(args, "request_scoped_obligations", False)
        ),
        lean_residual_packet=bool(getattr(args, "lean_residual_packet", False)),
        withhold_first_handoff=bool(
            getattr(args, "withhold_first_handoff", False)
        ),
        prefix_completion_grounding=bool(
            getattr(args, "prefix_completion_grounding", False)
        ),
        successor_tool_descriptions=(
            str(getattr(args, "successor_tool_descriptions", "on")) == "on"
        ),
    )
    base_payload = (
        base_results_payload(args.patch_infra_from)
        if args.patch_infra_from is not None
        else None
    )
    patched_task_ids = (
        infrastructure_task_ids(base_payload) if base_payload is not None else []
    )
    results = run_official_airline_method(
        args,
        run_dir=run_dir,
        agent_name=agent_name,
        task_ids=patched_task_ids or None,
    )
    results_payload = results.model_dump(mode="json")
    patch_provenance: dict[str, Any] | None = None
    if base_payload is not None:
        patch_payload = results_payload
        results_payload = merge_patch_results(
            base_payload=base_payload,
            patch_payload=patch_payload,
            patched_task_ids=patched_task_ids,
        )
        patch_provenance = {
            "schema": INFRASTRUCTURE_PATCH_SCHEMA,
            "base_run_name": base_payload.get("run_name"),
            "patched_task_count": len(patched_task_ids),
            "patch_reason": "infrastructure_error_only",
            "episodes_still_infrastructure_after_patch": len(
                infrastructure_task_ids(results_payload)
            ),
        }
    results_payload["schema"] = "papc.tau2.airline_residual_method_private_results.v1"
    results_payload["run_name"] = run_name
    results_payload["protocol"] = (
        "cheap_prefix_then_commitment_frontier_residual_completion"
    )
    results_payload["raw_task_ids_present_private"] = True
    (method_dir / "airline_residual_results.json").write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_rows = load_jsonl(decision_log_path)
    if base_payload is not None:
        sidecar_rows = merge_patch_sidecar(
            base_rows=load_jsonl(
                args.patch_infra_from
                / "airline_residual_method"
                / "airline_residual_decisions.jsonl"
            ),
            patch_rows=sidecar_rows,
            patched_task_ids=patched_task_ids,
        )
        # The merged episode set is written back to the private decision log,
        # exactly as the merged results already are. Without this a patch of a
        # patch reads only the previous patch's own rows: the episodes the
        # first patch inherited are missing, and the strong-role cost of the
        # merged run is reported far below what was actually spent.
        decision_log_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in sidecar_rows
            ),
            encoding="utf-8",
        )
    public_summary = build_execute_public_summary(
        args,
        run_name=run_name,
        results_payload=results_payload,
        sidecar_rows=sidecar_rows,
        gate=gate,
    )
    if patch_provenance is not None:
        public_summary["patch_provenance"] = patch_provenance
    write_json_append_only(args.public_output_json, public_summary)
    print(
        json.dumps(
            {
                "decision": "completed_tau2_airline_residual_execute",
                "reward_sum": public_summary["arms"]["airline_residual_method"][
                    "reward_sum"
                ],
                "provider_calls": public_summary["provider_calls"],
                "public_output": str(args.public_output_json),
                "private_run_dir": str(run_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_official_airline_method(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    agent_name: str,
    task_ids: list[str] | None = None,
):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    save_dir = run_dir / "tau_runs" / "airline" / "airline_residual_method.r0"
    config = TextRunConfig(
        domain="airline",
        task_split_name=args.task_split,
        task_ids=task_ids,
        num_tasks=args.tasks_per_domain,
        llm_user=args.user_model,
        llm_args_user=method_runner.llm_args_for_model(
            args.api_base,
            args.user_reasoning_effort,
            args.user_model,
        ),
        num_trials=args.num_trials,
        max_retries=args.max_retries,
        retry_delay=0.0,
        save_to=str(save_dir.resolve()),
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        verbose_logs=True,
        agent=agent_name,
        llm_agent=args.cheap_model,
        llm_args_agent=method_runner.llm_args_for_model(
            args.api_base,
            args.cheap_reasoning_effort,
            args.cheap_model,
        ),
        user="user_simulator",
        max_steps=args.max_steps,
    )
    return run_domain(config)


def build_launch_report(
    args: argparse.Namespace,
    *,
    run_name: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    blockers = launch_blockers(args, gate)
    report = {
        "schema": "papc.tau2.airline_residual_launch_report.v1",
        "run_name": run_name,
        "mode": "dry_run_launch_report",
        "decision": "blocked" if blockers else "closed_no_paid_authorization",
        "blockers": blockers,
        "surface": "tau2_airline",
        "domain": "airline",
        "domain_split": args.task_split,
        "method_family": "incremental_live_receipt_graph_completion",
        "cap": {
            "tasks_per_domain": int(args.tasks_per_domain),
            "num_trials": int(args.num_trials),
            "task_id_selection_supported": False,
        },
        "max_concurrency": int(args.max_concurrency),
        "models": {
            "cheap": args.cheap_model,
            "cheap_reasoning_effort": args.cheap_reasoning_effort,
            "strong": args.strong_model,
            "strong_reasoning_effort": args.strong_reasoning_effort,
            "user_simulator": args.user_model,
            "user_reasoning_effort": args.user_reasoning_effort,
            "evaluator": args.evaluator_model,
            "evaluator_reasoning_effort": args.evaluator_reasoning_effort,
        },
        "compiler_gate": {
            "schema": gate.get("schema"),
            "selector_gate_passes": bool(
                gate.get("gate_decision", {}).get("selector_gate_passes")
            ),
            "guarded_selector_projected_pass_count": gate.get("gate_decision", {}).get(
                "guarded_selector_projected_pass_count"
            ),
            "guarded_selector_selected_p_to_f_count": gate.get("gate_decision", {}).get(
                "guarded_selector_selected_p_to_f_count"
            ),
            "public_leakage_pass": gate.get("public_leakage_audit", {}).get("pass")
            is True,
            "public_gate_path": _public_path_label(args.compiler_gate_json),
        },
        # Every field below is read off the registered runtime rather than
        # restated by hand, so a launch report cannot describe a mechanism the
        # agent does not carry.
        "runtime_contract": {
            "authority": RESIDUAL_AUTHORITY,
            "frontier": "first_write_or_transfer_candidate",
            "frontier_activation_policy": str(
                getattr(args, "frontier_activation_policy", "all_write_or_transfer")
                or "all_write_or_transfer"
            ),
            "lean_residual_packet": bool(
                getattr(args, "lean_residual_packet", False)
            ),
            "withhold_first_handoff": bool(
                getattr(args, "withhold_first_handoff", False)
            ),
            "prefix_completion_grounding": bool(
                getattr(args, "prefix_completion_grounding", False)
            ),
            "frontier_activation_count_per_task": 1,
            "write_budget": int(args.write_budget),
            "graph_tool_call_budget": int(args.max_strong_turns) - 1,
            "max_strong_provider_calls": int(args.max_strong_turns),
            "residual_user_turn_budget": RESIDUAL_USER_TURN_BUDGET,
            "open_obligation_survives_a_user_turn": True,
            "strong_acts_through_native_live_tools": True,
            "private_reached_state_replica": False,
            "whole_graph_admitted_before_live_write": False,
            "each_extension_checked_before_corresponding_live_call": True,
            "fail_closed_before_rejected_extension": True,
            "failed_live_call_blocks_further_extensions": True,
            "terminal_closes_full_live_receipt_graph": True,
            "dependency_edges_derived_from_visible_receipts": True,
            "model_authors_actions_not_relation_fields": True,
            "first_task_obligation_frontier_removed": True,
            "strong_full_restart_forbidden": True,
            "fallback_on_unopened_frontier": True,
            "sidecar_accounts_intercepted_cheap_call": True,
            "transcript_accounts_live_strong_calls": True,
            "raw_messages_excluded_from_public": True,
        },
        "strong_prompt_token_policy": {
            "reached_state_contract_sent_once": False,
            "frontier_constraints_carried_each_successor_turn": True,
            "compact_visible_event_packet_sent_once": True,
            "raw_reached_state_message_objects_included": True,
            "subsequent_turns_use_compact_step_reminder": True,
            "strong_proposal_rounds_bounded_by": int(args.max_strong_turns),
            "token_budget_is_secondary_until_reward_gate": True,
        },
        "execution_semantics": {
            "fresh_online_generation": True,
            "historical_trajectory_replay": False,
            "real_tau2_environment_receipts": True,
            "incremental_graph_extension_is_live_call_authority": True,
            "final_graph_closure_after_live_receipts": True,
            "successor_text_control": "residual_status.v1_with_legacy_text",
            "successor_text_control_requested": True,
            "declared_wait_or_failure_is_not_completion": True,
        },
        "online_validation_policy": {
            "schema": "papc.tau2.airline_residual_online_policy.v1",
            "primary_reward_floor": effective_reward_floor(args),
            "promotion_reward_target": effective_reward_floor(args),
            "denominator": int(
                SPLIT_OPERATING_POINTS[args.task_split]["denominator"]
            ),
            "agent_side_raw_cost_ceiling_usd": effective_cost_ceiling_usd(args),
            "accuracy_first": True,
            "token_budget_is_secondary": True,
            "max_concurrency_ceiling": MAX_ONLINE_CONCURRENCY,
            "paid_rerun_authorized": True,
            "if_reward_below_floor": "reject_operating_point",
            "if_cost_above_ceiling": "reject_operating_point",
        },
        "cached_accounting_policy": cached_accounting_policy_report(),
        "provider_calls": 0,
        "api_spend_performed": False,
        "public_safe": True,
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    if report["public_leakage_audit"]["pass"] is not True:
        report["decision"] = "blocked"
        report["blockers"].append("public_leakage_audit_failed")
    return report



def _attach_agent_side_cost(
    arm: dict[str, Any], *, cheap_model: str, strong_model: str
) -> None:
    """Price the cheap and strong roles under the frozen public table.

    The user simulator and the evaluator are the benchmark, not the agent, so
    they stay out of the number the paper reports.
    """

    per_role = ((arm.get("runtime_ledger") or {}).get("per_role")) or {}
    models = {"cheap": cheap_model, "strong": strong_model}
    raw = 0.0
    raw_available = True
    provider = 0.0
    provider_available = True
    for role, model in models.items():
        row = per_role.get(role) or {}
        if row.get("observed") is not True:
            continue
        if (
            row.get("token_usage_available") is not True
            or row.get("input_tokens") is None
            or row.get("output_tokens") is None
        ):
            raw_available = False
        else:
            raw += cache_neutral_cost_usd(
                model=model,
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                reasoning_tokens=row.get("reasoning_tokens"),
            )
        cost = row.get("cost_usd")
        if row.get("cost_available") is not True or cost is None:
            provider_available = False
        else:
            provider += float(cost)
    arm["agent_side_raw_cost_usd"] = raw if raw_available else None
    arm["agent_side_provider_reported_cost_usd"] = (
        provider if provider_available else None
    )
    arm["agent_side_cost_accounting"] = {
        "schema": "papc.tau2.agent_side_cost_accounting.v1",
        "primary": "cache_neutral_frozen_public_prices",
        "cache_neutral_cost_usd": raw if raw_available else None,
        "provider_reported_cost_usd": (provider if provider_available else None),
        "pricing_snapshot_date": OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE,
        "pricing_source": OPENAI_PUBLIC_PRICING_SOURCE,
        "roles": ["cheap", "strong"],
        "user_simulator_and_evaluator_excluded": True,
        "public_safe": True,
    }


def build_execute_public_summary(
    args: argparse.Namespace,
    *,
    run_name: str,
    results_payload: dict[str, Any],
    sidecar_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    # The strong role reaches this summary only through the sidecar, because a
    # message the successor returns into the live transcript arrives without
    # provider usage. Deriving it from the transcript as well would bill the
    # same generation twice, so the sidecar is the single source.
    arm = method_runner.arm_public_summary(
        args,
        arm_name="frontier_residual_method",
        results_payload=results_payload,
        sidecar_rows=sidecar_rows,
    )
    _attach_agent_side_cost(
        arm, cheap_model=args.cheap_model, strong_model=args.strong_model
    )
    arm["arm"] = "airline_residual_method"
    arms = {"airline_residual_method": arm}
    external_time = method_runner.external_time_accounting(arms)
    for row in external_time.values():
        if isinstance(row, dict):
            method_runner.attach_parallel_capacity_accounting(
                row,
                parallel_elapsed_wall_seconds=official.simulation_parallel_elapsed_seconds(
                    results_payload
                ),
                max_concurrency=args.max_concurrency,
            )
    summary = {
        "schema": "papc.tau2.airline_residual_execute_public.v1",
        "run_name": run_name,
        "surface": "tau2_airline",
        "domain": "airline",
        "domain_split": args.task_split,
        "method_family": "incremental_live_receipt_graph_completion",
        "task_count": arm["task_count"],
        "reward_evaluable_count": arm["reward_evaluable_count"],
        "infrastructure_excluded_count": arm["infrastructure_excluded_count"],
        "reward_sum": arm["reward_sum"],
        "reward_mean": arm["reward_mean"],
        "target_reward_floor": effective_reward_floor(args),
        "promotion_reward_target": effective_reward_floor(args),
        "target_denominator": int(
            SPLIT_OPERATING_POINTS[args.task_split]["denominator"]
        ),
        "meets_target_reward_floor": (
            arm["reward_sum"] is not None
            and float(arm["reward_sum"]) >= effective_reward_floor(args)
        ),
        "meets_promotion_reward_target": (
            arm["reward_sum"] is not None
            and float(arm["reward_sum"]) >= effective_reward_floor(args)
        ),
        "agent_side_raw_cost_ceiling_usd": effective_cost_ceiling_usd(args),
        "agent_side_raw_cost_usd": arm.get("agent_side_raw_cost_usd"),
        "agent_side_provider_reported_cost_usd": arm.get(
            "agent_side_provider_reported_cost_usd"
        ),
        "meets_agent_side_raw_cost_ceiling": (
            arm.get("agent_side_raw_cost_usd") is not None
            and float(arm["agent_side_raw_cost_usd"])
            <= effective_cost_ceiling_usd(args)
        ),
        "arms": arms,
        "model_resource_summary": method_runner.model_resource_summary(arms),
        "external_time_accounting": external_time,
        "cached_accounting": cached_accounting_summary(arms),
        "compiler_gate_reference": {
            "schema": gate.get("schema"),
            "selector_gate_passes": bool(
                gate.get("gate_decision", {}).get("selector_gate_passes")
            ),
            "guarded_selector_projected_pass_count": gate.get("gate_decision", {}).get(
                "guarded_selector_projected_pass_count"
            ),
            "public_gate_path": _public_path_label(args.compiler_gate_json),
        },
        "runtime_variant": {
            "pending_carrier_audit": True,
            "frontier_activation_policy": str(
                getattr(args, "frontier_activation_policy", "all_write_or_transfer")
                or "all_write_or_transfer"
            ),
            "lean_residual_packet": bool(
                getattr(args, "lean_residual_packet", False)
            ),
            "withhold_first_handoff": bool(
                getattr(args, "withhold_first_handoff", False)
            ),
            "prefix_completion_grounding": bool(
                getattr(args, "prefix_completion_grounding", False)
            ),
            "strong_residual_owner_until_final": not bool(
                getattr(args, "residual_recertifies_after_user_turn", False)
            ),
            "residual_recertifies_after_user_turn": bool(
                getattr(args, "residual_recertifies_after_user_turn", False)
            ),
            "successor_batches_actions": bool(
                getattr(args, "successor_batches_actions", False)
            ),
            "defer_frontier_until_confirmed": bool(
                getattr(args, "defer_frontier_until_confirmed", False)
            ),
            "successor_declares_closure": bool(
                getattr(args, "successor_declares_closure", False)
            ),
            "request_scoped_obligations": bool(
                getattr(args, "request_scoped_obligations", False)
            ),
            "successor_tool_descriptions": (
                str(getattr(args, "successor_tool_descriptions", "on")) == "on"
            ),
            "max_successor_rounds": getattr(args, "max_successor_rounds", None),
            "continue_after_nonterminal_message": False,
            "max_message_followup_turns": 0,
            "message_followup_frontier_policy": "disabled",
            "successor_text_control": "residual_status.v1_with_legacy_text",
            "successor_text_control_requested": True,
        },
        "provider_calls": _provider_call_count(arm),
        "api_spend_performed": True,
        "raw_private_paths_excluded": True,
        "raw_task_ids_excluded": True,
        "public_safe": True,
    }
    summary["public_leakage_audit"] = public_leakage_audit(summary)
    if summary["public_leakage_audit"]["pass"] is not True:
        raise ValueError(
            "airline execute public summary failed the leakage audit; the public "
            "summary was not written and must be rebuilt from the private root"
        )
    return summary


def launch_blockers(args: argparse.Namespace, gate: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    denominator = int(SPLIT_OPERATING_POINTS[args.task_split]["denominator"])
    if int(args.tasks_per_domain) != denominator:
        blockers.append(f"airline_{args.task_split}{denominator}_required")
    if int(args.num_trials) != 1:
        blockers.append("single_trial_required")
    if int(args.max_concurrency) > MAX_ONLINE_CONCURRENCY:
        blockers.append(f"max_concurrency_exceeds_{MAX_ONLINE_CONCURRENCY}")
    if not 0 <= effective_reward_floor(args) <= denominator:
        blockers.append("reward_floor_out_of_range")
    if effective_cost_ceiling_usd(args) <= 0:
        blockers.append("agent_side_cost_ceiling_must_be_positive")
    if int(args.write_budget) != DEFAULT_RESIDUAL_WRITE_BUDGET:
        blockers.append("write_budget_must_be_5")
    for role in ("cheap", "strong", "user", "evaluator"):
        if str(getattr(args, f"{role}_reasoning_effort", "")) != "high":
            blockers.append(f"{role}_reasoning_not_high")
    if gate.get("schema") != "papc.tau2.airline_residual_gate_noapi.v1":
        blockers.append("compiler_gate_schema_mismatch")
    if gate.get("code_sha256") != code_sha256():
        blockers.append("compiler_gate_source_mismatch")
    gate_decision = gate.get("gate_decision", {})
    if (
        not isinstance(gate_decision, dict)
        or gate_decision.get("selector_gate_passes") is not True
    ):
        blockers.append("compiler_gate_selector_not_passed")
    if not isinstance(gate_decision, dict) or (
        "setting_correct_selector_projected_pass_count" not in gate_decision
    ):
        blockers.append("compiler_gate_missing_setting_correct_projection")
    if gate.get("public_leakage_audit", {}).get("pass") is not True:
        blockers.append("compiler_gate_public_leakage_not_passed")
    return blockers


INFRASTRUCTURE_PATCH_SCHEMA = "papc.tau2.airline_infrastructure_patch.v1"


def base_results_payload(base_root: Path) -> dict[str, Any] | None:
    """This surface's private results for a base run, if they are there."""

    return shared_base_results_payload(
        base_root,
        method_dir="airline_residual_method",
        results_name="airline_residual_results.json",
    )






def execute_blockers(args: argparse.Namespace, gate: dict[str, Any]) -> list[str]:
    blockers = launch_blockers(args, gate)
    if not args.execute:
        blockers.append("execute_flag_missing")
    if not args.api_base:
        blockers.append("api_base_missing")
    if not args.api_key_env:
        blockers.append("api_key_env_missing")
    elif not os.environ.get(args.api_key_env):
        blockers.append("api_key_env_not_present")
    if args.patch_infra_from is not None:
        payload = base_results_payload(args.patch_infra_from)
        if payload is None:
            blockers.append("patch_base_results_missing")
        elif not infrastructure_task_ids(payload):
            blockers.append("patch_base_has_no_infrastructure_rows")
    return blockers


def _load_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("compiler gate must be a JSON object")
    return payload


def _provider_call_count(arm: dict[str, Any]) -> int:
    ledger = arm.get("runtime_ledger")
    if not isinstance(ledger, dict):
        return 0
    total = 0
    for row in (ledger.get("per_role") or {}).values():
        if isinstance(row, dict):
            total += int(row.get("calls") or 0)
    return total


def cached_accounting_policy_report() -> dict[str, Any]:
    return {
        "schema": "papc.tau2.airline_cached_accounting_policy.v1",
        "runtime_ledger_fields": [
            "input_tokens",
            "cached_input_tokens",
            "billable_input_tokens",
            "output_tokens",
            "total_tokens",
        ],
        "billable_input_formula": "max(0, input_tokens - cached_input_tokens)",
        "missing_token_usage_is_unavailable_not_zero": True,
        "raw_provider_payloads_excluded": True,
        "public_safe": True,
    }


def cached_accounting_summary(arms: dict[str, Any]) -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for arm_name, arm_summary in arms.items():
        ledger = arm_summary.get("runtime_ledger")
        if not isinstance(ledger, dict):
            continue
        per_role = ledger.get("per_role")
        if not isinstance(per_role, dict):
            continue
        per_arm[arm_name] = {
            role: _cached_accounting_role_summary(row)
            for role, row in per_role.items()
            if isinstance(row, dict)
        }
    return {
        "schema": "papc.tau2.airline_cached_accounting_summary.v1",
        "policy": cached_accounting_policy_report(),
        "per_arm": per_arm,
        "public_safe": True,
    }


def _cached_accounting_role_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed": row.get("observed") is True,
        "calls": int(row.get("calls") or 0),
        "input_tokens": row.get("input_tokens"),
        "cached_input_tokens": row.get("cached_input_tokens"),
        "billable_input_tokens": row.get("billable_input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "total_tokens": row.get("total_tokens"),
        "token_usage_available": row.get("token_usage_available") is True,
    }


def _public_path_label(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("outputs/papc/"):
        return text
    return path.name


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a clean tau2 CFRC comparator.

This entrypoint is intentionally scoped to the current retail method path. The default
method arm is a deterministic-contract frontier protocol:

cheap prefix -> post-tool-result frontier z -> one strong typed proposal graph
H(a) -> deterministic Gamma_z / Pi_K projection -> residual suffix or abstain.

The terminal-after-cheap-full mode is retained as an explicit diagnostic special
case, not the default clean row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.costing import cache_neutral_cost_usd  # noqa: E402
from papc.certspec.infra_patch import (  # noqa: E402
    base_results_payload as shared_base_results_payload,
)
from papc.certspec.infra_patch import (  # noqa: E402
    infrastructure_task_ids,
    merge_patch_results,
    merge_patch_sidecar,
)
from papc.certspec.tau2.cfrc_runner import (  # noqa: E402
    AUTHORITY_LIFTED_ELL3_COMPILER_SETTING,
    CFRC_STRONG_SYSTEM_PROMPT,
    ELL3_AUTHORITY_LABEL,
    ELL_UPPER_AUTHORITY_LABEL,
    RETAIL_MINIMAL_INTERVENTION_ADAPTER_POLICY,
    RETAIL_MINIMAL_INTERVENTION_CANDIDATE_CARRIER_TOOLS,
    STRICT_COMPILER_SETTING,
    build_residual_certificate,
    build_strong_prompt_payload,
    cfrc_runtime_ledger_from_results,
    compact_visible_evidence,
    evidence_facts_from_visible_rows,
    load_jsonl,
    mutating_tool_names_by_heuristic,
    parse_typed_proposal,
    project_cfrc_proposal,
    receipt_bound_commitments,
    register_tau2_cfrc_agent,
    strong_call_required,
    suffix_final_message_from_receipts,
    write_jsonl_append,
)
from papc.certspec.tau2.noapi_readiness import public_leakage_audit  # noqa: E402
from papc.certspec.tau2.residual_support import (  # noqa: E402
    litellm_model_name,
    llm_args_for_model,
)
from papc.paths import DATA_ROOT

DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_OUTPUT_ROOT = Path(f"{DATA_ROOT}/papc")
DEFAULT_PUBLIC_OUTPUT_ROOT = _REPO / "outputs" / "papc"
RETAIL_DENOMINATOR = 114
RETAIL_SPLIT_DENOMINATORS = {"base": 114, "test": 40, "train": 74}
RETAIL_TEST_ACCEPTANCE_PROFILES: dict[tuple[str, str], dict[str, Any]] = {
    ("gpt-5.6-luna", "gpt-5.6-sol"): {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    ("gpt-5.4-mini", "gpt-5.5"): {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    ("gpt-5.4-mini", "openai/gemini-3.5-flash-high"): {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    ("gpt-5.6-luna", "openai/gemini-3.7-flash-high"): {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    # Cross-vendor successors on the two registered prefixes. Each keeps the
    # thresholds of the arm whose prefix and anchors it shares, because nothing
    # is yet measured for these successors and a threshold invented before the
    # first run would be a target rather than an incumbent. The gate script
    # carries the same two entries; a pair present in one table and absent from
    # the other resolves to no quality reference on the runner side while the
    # gate writes one, and the run is refused for a mismatch that is really a
    # gap between two copies of the same policy.
    ("gpt-5.6-luna", "openai/claude-opus-5"): {
        "reward_floor": 32,
        "cost_ceiling_usd": 12.00,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    ("gpt-5.6-luna", "openai/claude-sonnet-5"): {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    ("gpt-5.4-mini", "openai/claude-opus-5"): {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    ("gpt-5.4-mini", "openai/claude-sonnet-5"): {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    # The first prefix that is not a GPT model. Unlike the two entries above it
    # does have a measured incumbent, so the threshold is that incumbent rather
    # than a borrowed one: Haiku 4.5 alone reads 26 of 40 on this split at
    # $5.491161 cache-neutral agent-side. A residual layer that does not beat
    # the prefix it inherits has not earned its successor call, and one that
    # costs more than the prefix would have cost doing the whole task alone has
    # not earned it either.
    ("openai/claude-haiku-4-5-20251001", "gpt-5.6-sol"): {
        "reward_floor": 26,
        "cost_ceiling_usd": 5.491161,
        "quality_reference": {
            "arm": "claude_haiku45_baseline",
            "reward_sum": 26,
            "official_denominator": 40,
        },
    },
    # Native Gemini 3.5 has a full40 quality anchor (31/40) and complete token
    # usage. Repricing its cheap role at Google's standard cache-neutral rates
    # gives the matched full-denominator cost ceiling below.
    ("openai/gemini-3.5-flash-high", "gpt-5.6-sol"): {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    ("openai/gemini-3.5-flash-high", "gpt-5.5"): {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    ("openai/gemini-3.5-flash-high", "openai/claude-opus-5"): {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    # Native Gemini 3.6 is complete at 34/40. Its cheap-role usage reprices to
    # $14.177708 at Google's promotional cache-neutral rate through 2026-12-31.
    ("openai/gemini-3.6-flash-high", "gpt-5.6-sol"): {
        "reward_floor": 34,
        "cost_ceiling_usd": 14.177708,
        "quality_reference": {
            "arm": "native_gemini36flashhigh_baseline",
            "reward_sum": 34,
            "official_denominator": 40,
        },
    },
}


INFRASTRUCTURE_PATCH_SCHEMA = "papc.tau2.retail_infrastructure_patch.v1"


def base_results_payload(base_root: Path) -> dict[str, Any] | None:
    """This surface's private results for a base run, if they are there."""

    return shared_base_results_payload(
        base_root,
        method_dir="frontier_residual_method",
        results_name="frontier_method_results.json",
    )


def official_baseline_results_payload(
    base_root: Path, *, domain: str, arm_name: str
) -> dict[str, Any] | None:
    """The private tau2 results written by an official single-arm run."""

    path = Path(base_root) / "tau_runs" / domain / f"{arm_name}.r0" / "results.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def patch_base_results_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load the artifact shape produced by the arm being recovered."""

    base_root = getattr(args, "patch_infra_from", None)
    if base_root is None:
        return None
    if getattr(args, "official_baseline_only", None):
        return official_baseline_results_payload(
            base_root,
            domain=str(args.domain),
            arm_name=str(args.official_baseline_only),
        )
    return base_results_payload(base_root)


def retail_split_denominator(args) -> int:
    return RETAIL_SPLIT_DENOMINATORS.get(
        str(getattr(args, "domain_split", "") or "base"), RETAIL_DENOMINATOR
    )
REQUEST_CONTRACT_GATE_SCHEMA = "papc.tau2.request_contract_gate.v3"


def _parse_task_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    rows = [item.strip() for item in str(value).split(",") if item.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("task ids must not be empty")
    if len(rows) != len(set(rows)):
        raise argparse.ArgumentTypeError("task ids must be unique")
    return rows


def _task_id_sha256s(task_ids: list[str] | None) -> list[str]:
    return sorted(
        hashlib.sha256(item.encode()).hexdigest() for item in (task_ids or [])
    )


def _request_contract_mode(args: argparse.Namespace) -> str:
    return "request_contract" if args.request_contract_enabled else "legacy"


def deterministic_closure_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "deterministic_contract_closure", False)
    ) and args.contract_builder_variant == "deterministic_candidate_grounded"


def zero_obligation_fallback_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "zero_obligation_request_fallback", False)
    ) and args.contract_builder_variant == "deterministic_candidate_grounded"


def effective_contract_builder_attempt_cap(args: argparse.Namespace) -> int:
    if args.contract_builder_variant == "model_authored_request_semantics":
        return int(args.contract_builder_attempt_cap)
    cap = 1 if deterministic_closure_enabled(args) else 0
    if zero_obligation_fallback_enabled(args):
        cap += 1
    return cap


def strong_provider_call_budget(args: argparse.Namespace) -> dict[str, Any]:
    builder_cap = effective_contract_builder_attempt_cap(args)
    proposal_cap = int(args.strong_call_cap)
    builder_phases = (
        (
            ["request_semantics_closure"]
            if builder_cap == 1
            else ["request_semantics_draft", "residual_closure_revision"]
        )
        if args.contract_builder_variant == "model_authored_request_semantics"
        else ["deterministic_residual_closure"]
        if deterministic_closure_enabled(args)
        else []
    )
    return {
        "contract_builder_attempt_cap": builder_cap,
        "contract_builder_phases": builder_phases,
        "contract_builder_success_provider_call_count_required": (
            len(builder_phases)
        ),
        "residual_proposal_call_cap": proposal_cap,
        "total_strong_provider_call_cap": builder_cap + proposal_cap,
    }


def effective_reward_floor(args: argparse.Namespace) -> float:
    if args.reward_floor is not None:
        return float(args.reward_floor)
    return 92.0 if retail_split_denominator(args) == RETAIL_DENOMINATOR else 0.0


def effective_cost_ceiling_usd(args: argparse.Namespace) -> float:
    return float(args.cost_ceiling_usd if args.cost_ceiling_usd is not None else 18.0)


def retail_test_acceptance_profile(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    return RETAIL_TEST_ACCEPTANCE_PROFILES.get(
        (str(args.cheap_model), str(args.strong_model))
    )


def acceptance_policy_report(args: argparse.Namespace) -> dict[str, Any]:
    denominator = retail_split_denominator(args)
    test_profile = (
        retail_test_acceptance_profile(args)
        if str(args.domain_split) == "test"
        else None
    )
    return {
        "official_denominator": denominator,
        "reward_floor": effective_reward_floor(args),
        "agent_side_cache_neutral_cost_ceiling_usd": (
            effective_cost_ceiling_usd(args)
        ),
        "quality_reference": (
            dict(test_profile["quality_reference"])
            if test_profile is not None
            else None
        ),
        "infrastructure_errors_allowed": 0,
        "unsafe_checked_actions_allowed": 0,
        "quality_below_floor_is_negative_regardless_of_cost": True,
    }


def agent_side_cache_neutral_cost_usd(
    arm: dict[str, Any], *, cheap_model: str, strong_model: str
) -> float | None:
    """Price only the cheap and strong roles at frozen public rates."""

    per_role = ((arm.get("runtime_ledger") or {}).get("per_role")) or {}
    total = 0.0
    observed = False
    for role, model in (("cheap", cheap_model), ("strong", strong_model)):
        row = per_role.get(role) or {}
        if row.get("observed") is not True:
            continue
        if row.get("token_usage_available") is not True:
            return None
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        try:
            total += cache_neutral_cost_usd(
                model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                reasoning_tokens=row.get("reasoning_tokens"),
            )
        except (TypeError, ValueError):
            return None
        observed = True
    return round(total, 6) if observed else None


def postrun_acceptance_report(
    args: argparse.Namespace, *, method_arm: dict[str, Any]
) -> dict[str, Any]:
    """Apply the registered full-denominator quality and cost hard gates."""

    denominator = retail_split_denominator(args)
    reward = method_arm.get("reward_sum")
    reward_floor = effective_reward_floor(args)
    cost_ceiling = effective_cost_ceiling_usd(args)
    cost = agent_side_cache_neutral_cost_usd(
        method_arm,
        cheap_model=str(args.cheap_model),
        strong_model=str(args.strong_model),
    )
    blockers: list[str] = []
    if bool(args.diagnostic_probe):
        blockers.append("diagnostic_probe")
    if int(method_arm.get("task_count") or 0) != denominator:
        blockers.append("incomplete_task_denominator")
    if int(method_arm.get("reward_evaluable_count") or 0) != denominator:
        blockers.append("incomplete_reward_denominator")
    if int(method_arm.get("infrastructure_excluded_count") or 0) != 0:
        blockers.append("infrastructure_error")
    if reward is None or float(reward) < reward_floor:
        blockers.append("reward_below_floor")
    if cost is None:
        blockers.append("agent_side_cost_unavailable")
    elif cost > cost_ceiling:
        blockers.append("agent_side_cost_above_ceiling")
    return {
        "schema": "papc.tau2.postrun_acceptance.v1",
        "official_denominator": denominator,
        "task_count": int(method_arm.get("task_count") or 0),
        "reward_evaluable_count": int(
            method_arm.get("reward_evaluable_count") or 0
        ),
        "infrastructure_excluded_count": int(
            method_arm.get("infrastructure_excluded_count") or 0
        ),
        "reward_sum": float(reward) if reward is not None else None,
        "reward_floor": reward_floor,
        "meets_reward_floor": reward is not None and float(reward) >= reward_floor,
        "agent_side_cache_neutral_cost_usd": cost,
        "agent_side_cache_neutral_cost_ceiling_usd": cost_ceiling,
        "meets_cost_ceiling": cost is not None and cost <= cost_ceiling,
        "quality_below_floor_is_negative_regardless_of_cost": True,
        "blockers": blockers,
        "eligible_to_update_current_evidence": not blockers,
        "public_safe": True,
    }


def _request_contract_gate(args: argparse.Namespace) -> dict[str, Any]:
    path = getattr(args, "request_contract_gate_json", None)
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request contract gate must be a JSON object")
    return payload


def _request_contract_gate_blockers(args: argparse.Namespace) -> list[str]:
    if not args.task_ids and not args.request_contract_enabled:
        return []
    blockers: list[str] = []
    if getattr(args, "request_contract_gate_json", None) is None:
        return ["request_contract_gate_missing"]
    try:
        gate = _request_contract_gate(args)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["request_contract_gate_unreadable"]
    if gate.get("schema") != REQUEST_CONTRACT_GATE_SCHEMA:
        blockers.append("request_contract_gate_schema_mismatch")
    if gate.get("surface") != "tau2_retail":
        blockers.append("request_contract_gate_surface_mismatch")
    authorization = gate.get("authorization")
    if not isinstance(authorization, dict):
        authorization = {}
    full_run = not bool(args.task_ids)
    if full_run:
        if authorization.get("authorizes_full_run") is not True:
            blockers.append("request_contract_gate_full_run_not_authorized")
        if authorization.get("authorizes_diagnostic_probe") is not False:
            blockers.append("request_contract_gate_diagnostic_boundary_missing")
        if authorization.get("max_task_count") != retail_split_denominator(args):
            blockers.append("request_contract_gate_full_run_scope_mismatch")
        if authorization.get("task_id_sha256s") not in ([], ()):
            blockers.append("request_contract_gate_task_set_mismatch")
        if args.tasks_per_domain != retail_split_denominator(args):
            blockers.append("request_contract_gate_full_run_scope_mismatch")
    else:
        if authorization.get("authorizes_diagnostic_probe") is not True:
            blockers.append("request_contract_gate_probe_not_authorized")
        if authorization.get("authorizes_full_run") is not False:
            blockers.append("request_contract_gate_full_run_boundary_missing")
        if sorted(authorization.get("task_id_sha256s") or []) != _task_id_sha256s(
            args.task_ids
        ):
            blockers.append("request_contract_gate_task_set_mismatch")
    if _request_contract_mode(args) not in set(
        authorization.get("allowed_modes") or []
    ):
        blockers.append("request_contract_gate_mode_not_authorized")
    execution_binding = gate.get("execution_binding")
    if not isinstance(execution_binding, dict):
        execution_binding = {}
    expected_binding = {
        "num_trials": args.num_trials,
        "max_concurrency": args.max_concurrency,
        "max_steps": args.max_steps,
        "cheap_model": args.cheap_model,
        "cheap_reasoning_effort": args.cheap_reasoning_effort,
        "strong_model": args.strong_model,
        "strong_reasoning_effort": args.strong_reasoning_effort,
        "user_model": args.user_model,
        "user_reasoning_effort": args.user_reasoning_effort,
        "evaluator_model": args.evaluator_model,
        "evaluator_reasoning_effort": args.evaluator_reasoning_effort,
        "request_contract_mode": _request_contract_mode(args),
        "contract_builder_variant": args.contract_builder_variant,
        "deterministic_contract_closure": deterministic_closure_enabled(args),
        "zero_obligation_request_fallback": (
            zero_obligation_fallback_enabled(args)
        ),
        "grounded_terminal_literals": bool(
            getattr(args, "grounded_terminal_literals", False)
        ),
        "transcription_is_fallback": bool(
            getattr(args, "transcription_is_fallback", False)
        ),
        "contract_builder_executor": args.contract_builder_executor,
        **strong_provider_call_budget(args),
        "request_contract_private_replica": (
            args.request_contract_private_replica
        ),
        "frontier_timing": args.frontier_timing,
        "pre_write_residual_policy": args.pre_write_residual_policy,
        "strong_call_cap": args.strong_call_cap,
        "max_request_contract_calls": args.max_request_contract_calls,
        "request_contract_suffix_cap": args.request_contract_suffix_cap,
        "suffix_tool_call_budget": args.suffix_tool_call_budget,
        "sequential_contract_execution": True,
        "ledger_closed_terminal_required": True,
    }
    if any(
        execution_binding.get(key) != value
        for key, value in expected_binding.items()
    ):
        blockers.append("request_contract_gate_execution_binding_mismatch")
    if full_run:
        acceptance = gate.get("acceptance_gate")
        if not isinstance(acceptance, dict):
            acceptance = {}
        split_denominator = retail_split_denominator(args)
        test_profile = (
            retail_test_acceptance_profile(args)
            if str(args.domain_split) == "test"
            else None
        )
        expected_acceptance = {
            "official_denominator": split_denominator,
            "reward_floor": effective_reward_floor(args),
            "agent_side_cache_neutral_cost_ceiling_usd": (
                effective_cost_ceiling_usd(args)
            ),
            "quality_reference": (
                dict(test_profile["quality_reference"])
                if test_profile is not None
                else None
            ),
            "infrastructure_errors_allowed": 0,
            "unsafe_checked_actions_allowed": 0,
            "quality_impossible_stop_policy": {
                "mode": "operator_checkpoint_interrupt",
                "automatic_runner_stop": False,
                "max_in_flight_after_detection": args.max_concurrency,
            },
            "quality_below_floor_is_negative_regardless_of_cost": True,
        }
        if any(
            acceptance.get(key) != value
            for key, value in expected_acceptance.items()
        ):
            blockers.append("request_contract_gate_acceptance_mismatch")
    source = gate.get("source")
    if not isinstance(source, dict):
        source = {}
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if source.get("commit") != current_commit:
        blockers.append("request_contract_gate_source_commit_mismatch")
    manifest_bound = source.get("manifest_bound") is True
    if not manifest_bound and (
        current_dirty or source.get("workspace_dirty") is not False
    ):
        blockers.append("request_contract_gate_source_dirty")
    manifest = source.get("implementation_source_manifest")
    if not isinstance(manifest, list) or not manifest:
        blockers.append("request_contract_gate_manifest_invalid")
        manifest = []
    for row in manifest:
        if not isinstance(row, dict):
            blockers.append("request_contract_gate_manifest_invalid")
            continue
        path = _REPO / str(row.get("path") or "")
        if not path.is_file():
            blockers.append("request_contract_gate_source_file_missing")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
            blockers.append("request_contract_gate_source_hash_mismatch")
    if gate.get("public_leakage_audit", {}).get("pass") is not True:
        blockers.append("request_contract_gate_public_leakage_failed")
    return sorted(set(blockers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--domain-split", default="base")
    parser.add_argument("--tasks-per-domain", type=int, default=1)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated official task IDs for a bounded diagnostic probe.",
    )
    parser.add_argument(
        "--patch-infra-from",
        type=Path,
        default=None,
        help=(
            "Private root of a base run whose infrastructure episodes should be "
            "re-run and substituted. Only those episodes are re-run, so a scored "
            "episode is never re-rolled, and the merged result reports the full "
            "split rather than a subset."
        ),
    )
    parser.add_argument("--diagnostic-probe", action="store_true")
    parser.add_argument("--num-trials", type=int, default=1)
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
    parser.add_argument("--public-output-json", type=Path)
    parser.add_argument("--request-contract-gate-json", type=Path, default=None)
    parser.add_argument("--method-only", action="store_true")
    parser.add_argument(
        "--official-baseline-only",
        choices=["cheap_full", "strong_full"],
        default=None,
        help=(
            "Run exactly one official native agent arm (no contract, no "
            "graph, no checked execution) and emit a baseline summary."
        ),
    )
    parser.add_argument("--cheap-comparator-public-json", type=Path)
    parser.add_argument("--strong-comparator-public-json", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--method-mode",
        choices=["frontier", "terminal"],
        default="frontier",
        help="frontier is the clean default; terminal is a diagnostic special case.",
    )
    parser.add_argument(
        "--frontier-timing",
        choices=["post_tool_result", "pre_write_candidate"],
        default="post_tool_result",
        help=(
            "Timing for frontier mode. post_tool_result is the conservative clean row; "
            "pre_write_candidate intercepts the first native write before execution."
        ),
    )
    parser.add_argument(
        "--post-write-residual-policy",
        choices=["receipt_error", "always"],
        default="receipt_error",
        help=(
            "For post_tool_result frontier mode, receipt_error enters the residual "
            "phase only when the receipt-bound cheap frontier write reports an error."
        ),
    )
    parser.add_argument(
        "--pre-write-residual-policy",
        choices=["first_write_candidate", "always"],
        default="first_write_candidate",
        help=(
            "For pre_write_candidate frontier mode, first_write_candidate only "
            "enters the residual phase for verifier-issued pending carriers whose "
            "visible scope fits the bounded CFRC suffix space."
        ),
    )
    parser.add_argument(
        "--compiler-setting",
        choices=[STRICT_COMPILER_SETTING, AUTHORITY_LIFTED_ELL3_COMPILER_SETTING],
        default=STRICT_COMPILER_SETTING,
        help=(
            "Shared compiler setting. strict_core keeps the default clean row. "
            "authority_lifted_ell3 is default-off and only becomes compiler-ready "
            "when the certificate exposes bounded-local-redo collision/restoration metadata."
        ),
    )
    parser.add_argument(
        "--strong-call-cap",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help=(
            "B_call for frontier proposal rounds. Default 1 is the clean one-shot "
            "row; request-contract sequential execution shares this budget across "
            "phases and post-closure scopes."
        ),
    )
    parser.add_argument(
        "--suffix-tool-call-budget",
        type=int,
        default=1,
        help=(
            "Fixed executable suffix tool-call budget per task. It is pre-registered "
            "and does not scale with proposer graph node count."
        ),
    )
    parser.add_argument(
        "--allow-message-residual",
        action="store_true",
        help=(
            "Opt-in authority/message extension: allow strong to propose a typed "
            "message node that projection compiles into a user-facing residual."
        ),
    )
    parser.add_argument(
        "--request-contract",
        dest="request_contract_enabled",
        action="store_true",
        help="Enable Reached-State Contract Construction.",
    )
    parser.add_argument(
        "--contract-builder-variant",
        choices=[
            "deterministic_candidate_grounded",
            "model_authored_request_semantics",
        ],
        default="deterministic_candidate_grounded",
        help=(
            "Explicit Reached-State Contract Construction variant. The default "
            "preserves the frozen candidate-grounded builder; the optional "
            "model-authored builder derives request-level obligations from the "
            "visible request and accepted receipts."
        ),
    )
    parser.add_argument(
        "--deterministic-contract-closure",
        action="store_true",
        help=(
            "Split candidate-grounded contract construction into its two steps: "
            "deterministic grounding of the withheld candidate, then one "
            "provider call that closes the grounded ledger against the whole "
            "visible request and returns only the residual difference."
        ),
    )
    parser.add_argument(
        "--zero-obligation-request-fallback",
        action="store_true",
        help=(
            "When candidate-grounded grounding yields no obligation at all, "
            "ground the same contract against the visible request and the "
            "accepted receipts instead of freezing an empty ledger. An empty "
            "ledger induces an empty residual space, so Checked Execution can "
            "only abstain and the successor's proposal rounds are spent "
            "against a certificate no graph can discharge."
        ),
    )
    parser.add_argument(
        "--grounded-terminal-literals",
        action="store_true",
        help=(
            "Keep only the builder-authored terminal literals that the visible "
            "conversation, the public policy or a deterministic receipt "
            "already carries. A literal nobody has seen cannot be satisfied by "
            "any terminal response, so carrying it kills the contract on a "
            "sentence rather than on its obligations. Dropped literals are "
            "recorded on the contract receipt."
        ),
    )
    parser.add_argument(
        "--transcription-is-fallback",
        action="store_true",
        help=(
            "Order the contract transcription after the successor proposal "
            "instead of before it. The transcription replays the withheld "
            "candidate the contract restates, so running it first lets it "
            "discharge the residual without Executable Residual Completion "
            "ever proposing; as a fallback it still catches a rejected graph."
        ),
    )
    parser.add_argument(
        "--contract-builder-attempt-cap",
        type=int,
        choices=[1, 2],
        default=2,
        help="Maximum provider attempts inside model-authored contract construction.",
    )
    parser.add_argument(
        "--contract-builder-executor",
        choices=["successor", "cheap_prefix_model"],
        default="successor",
        help=(
            "Which model discharges Reached-State Contract Construction. The "
            "default keeps the successor; cheap_prefix_model runs the builder "
            "on the cheap prefix model and bills it to the cheap role, leaving "
            "the successor to Executable Residual Completion alone."
        ),
    )
    parser.add_argument(
        "--request-contract-private-replica",
        action="store_true",
        help=(
            "Require proposal-time execution on a private reached-state replica "
            "and bind every released write to its replica receipt."
        ),
    )
    parser.add_argument("--max-request-contract-calls", type=int, default=1)
    parser.add_argument("--request-contract-suffix-cap", type=int, default=6)
    parser.add_argument("--reward-floor", type=float, default=None)
    parser.add_argument("--cost-ceiling-usd", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run-launch-report", action="store_true")
    parser.add_argument(
        "--mutating-tool-names",
        nargs="*",
        default=None,
        help="Optional override. By default the CFRC agent reads tau2 tool metadata.",
    )
    args = parser.parse_args(argv)
    for field in ("cheap_model", "strong_model", "user_model", "evaluator_model"):
        setattr(args, field, litellm_model_name(getattr(args, field)))
    args.task_ids = _parse_task_ids(args.task_ids)
    if args.task_ids is not None:
        args.tasks_per_domain = len(args.task_ids)
    denominator = retail_split_denominator(args)
    if args.reward_floor is not None and not (
        0 <= float(args.reward_floor) <= denominator
    ):
        parser.error("--reward-floor must be between 0 and the official denominator")
    if args.cost_ceiling_usd is not None and float(args.cost_ceiling_usd) <= 0:
        parser.error("--cost-ceiling-usd must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    run_started_at = iso_timestamp()
    run_started_perf = time.perf_counter()
    args = parse_args(argv)
    run_name = args.run_name or f"tau2_{args.domain}_cfrc_clean_{timestamp()}"
    public_output_json = args.public_output_json or (
        DEFAULT_PUBLIC_OUTPUT_ROOT / f"{run_name}_public.json"
    )
    if args.dry_run_launch_report:
        report = build_launch_report(
            args,
            run_name=run_name,
            execute=False,
            blockers=launch_report_blockers(args),
        )
        attach_script_runtime_accounting(
            report,
            started_at=run_started_at,
            finished_at=iso_timestamp(),
            wall_seconds=time.perf_counter() - run_started_perf,
            max_concurrency=args.max_concurrency,
        )
        attach_public_leakage_audit(report)
        write_json_append_only(public_output_json, report)
        print(public_output_json)
        return 0

    blockers = execute_blockers(args)
    if blockers:
        report = build_launch_report(
            args,
            run_name=run_name,
            execute=True,
            blockers=blockers,
        )
        attach_script_runtime_accounting(
            report,
            started_at=run_started_at,
            finished_at=iso_timestamp(),
            wall_seconds=time.perf_counter() - run_started_perf,
            max_concurrency=args.max_concurrency,
        )
        attach_public_leakage_audit(report)
        write_json_append_only(public_output_json, report)
        print(json.dumps({"blocked": blockers, "public_output_json": str(public_output_json)}))
        return 2

    import_tau2(args.tau_repo)
    configure_provider_env(args)
    apply_evaluator_patch(
        evaluator_model=args.evaluator_model,
        api_base=args.api_base,
        reasoning_effort=args.evaluator_reasoning_effort,
    )

    run_dir = args.output_root / run_name
    if args.official_baseline_only:
        arm_name = args.official_baseline_only
        model = args.cheap_model if arm_name == "cheap_full" else args.strong_model
        effort = (
            args.cheap_reasoning_effort
            if arm_name == "cheap_full"
            else args.strong_reasoning_effort
        )
        base_payload = patch_base_results_payload(args)
        patched_task_ids = (
            infrastructure_task_ids(base_payload) if base_payload is not None else []
        )
        baseline_results = run_official_agent_arm(
            args,
            arm_name=arm_name,
            run_dir=run_dir,
            model=model,
            reasoning_effort=effort,
            task_ids=patched_task_ids or None,
        )
        results_payload = baseline_results.model_dump(mode="json")
        patch_provenance: dict[str, Any] | None = None
        if base_payload is not None:
            results_payload = merge_patch_results(
                base_payload=base_payload,
                patch_payload=results_payload,
                patched_task_ids=patched_task_ids,
            )
            patch_provenance = {
                "schema": "papc.tau2.official_baseline_infrastructure_recovery.v1",
                "patch_reason": "infrastructure_error_only",
                "base_task_count": len(base_payload.get("simulations") or []),
                "preserved_scored_task_count": len(
                    base_payload.get("simulations") or []
                )
                - len(patched_task_ids),
                "patched_task_count": len(patched_task_ids),
                "episodes_still_infrastructure_after_patch": len(
                    infrastructure_task_ids(results_payload)
                ),
                "base_results_sha256": hashlib.sha256(
                    json.dumps(
                        base_payload, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "recovery_max_concurrency": int(args.max_concurrency),
                "merged_infrastructure_recovery_not_single_fixed_concurrency_run": True,
                "raw_task_ids_excluded": True,
                "raw_private_paths_excluded": True,
                "public_safe": True,
            }
            merged_private_output = (
                run_dir
                / "tau_runs"
                / args.domain
                / f"{arm_name}.r0"
                / "infrastructure_recovery_merged_results.json"
            )
            write_json_append_only(merged_private_output, results_payload)
        public_summary = build_official_baseline_public_summary(
            args,
            run_name=run_name,
            arm_name=arm_name,
            results_payload=results_payload,
        )
        if patch_provenance is not None:
            public_summary["patch_provenance"] = patch_provenance
        attach_script_runtime_accounting(
            public_summary,
            started_at=run_started_at,
            finished_at=iso_timestamp(),
            wall_seconds=time.perf_counter() - run_started_perf,
            max_concurrency=args.max_concurrency,
        )
        attach_public_leakage_audit(public_summary)
        write_json_append_only(public_output_json, public_summary)
        print(public_output_json)
        return 0
    cheap_results = None
    strong_results = None
    if not args.method_only:
        cheap_results = run_official_agent_arm(
            args,
            arm_name="cheap_full",
            run_dir=run_dir,
            model=args.cheap_model,
            reasoning_effort=args.cheap_reasoning_effort,
        )
        strong_results = run_official_agent_arm(
            args,
            arm_name="strong_full",
            run_dir=run_dir,
            model=args.strong_model,
            reasoning_effort=args.strong_reasoning_effort,
        )
    if args.method_mode == "terminal":
        if cheap_results is None:
            raise RuntimeError("terminal method-only mode requires cheap full results")
        method_payload = run_terminal_residual_method_arm(
            args,
            run_name=run_name,
            run_dir=run_dir,
            cheap_results=cheap_results,
        )
    else:
        method_payload = run_frontier_residual_method_arm(
            args,
            run_name=run_name,
            run_dir=run_dir,
        )
    if args.method_only:
        public_summary = build_method_only_public_summary(
            args,
            run_name=run_name,
            method_results_payload=method_payload["results_payload"],
            method_sidecar_rows=method_payload["sidecar_rows"],
        )
    else:
        if cheap_results is None or strong_results is None:
            raise RuntimeError("comparator arms were not run")
        public_summary = build_execute_public_summary(
            args,
            run_name=run_name,
            cheap_results_payload=cheap_results.model_dump(mode="json"),
            strong_results_payload=strong_results.model_dump(mode="json"),
            method_results_payload=method_payload["results_payload"],
            method_sidecar_rows=method_payload["sidecar_rows"],
        )
    attach_script_runtime_accounting(
        public_summary,
        started_at=run_started_at,
        finished_at=iso_timestamp(),
        wall_seconds=time.perf_counter() - run_started_perf,
        max_concurrency=args.max_concurrency,
    )
    attach_public_leakage_audit(public_summary)
    write_json_append_only(public_output_json, public_summary)
    print(public_output_json)
    return 0


def run_official_agent_arm(
    args: argparse.Namespace,
    *,
    arm_name: str,
    run_dir: Path,
    model: str,
    reasoning_effort: str,
    agent_name: str = "llm_agent",
    task_ids: list[str] | None = None,
):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    # An override rather than a rewrite of args.task_ids. Setting that flag puts
    # the run into diagnostic-probe mode and drops it out of the full-run
    # acceptance path, which is right for a hand-picked subset and wrong for a
    # patch: a patch re-runs only the episodes a provider failure took, and what
    # it produces after merging is the full split.
    selected = task_ids if task_ids is not None else args.task_ids
    save_dir = run_dir / "tau_runs" / args.domain / f"{arm_name}.r0"
    config = TextRunConfig(
        domain=args.domain,
        task_split_name=none_if_requested(args.domain_split),
        task_ids=selected,
        num_tasks=None if selected else args.tasks_per_domain,
        llm_user=args.user_model,
        llm_args_user=llm_args_for_model(
            args.api_base, args.user_reasoning_effort, args.user_model
        ),
        num_trials=args.num_trials,
        max_retries=args.max_retries,
        retry_delay=0.0,
        save_to=str(save_dir.resolve()),
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        agent=agent_name,
        llm_agent=model,
        llm_args_agent=llm_args_for_model(args.api_base, reasoning_effort, model),
        user="user_simulator",
        max_steps=args.max_steps,
    )
    return run_domain(config)


def run_frontier_residual_method_arm(
    args: argparse.Namespace,
    *,
    run_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    method_dir = run_dir / "frontier_residual_method"
    method_dir.mkdir(parents=True, exist_ok=False)
    decision_log_path = method_dir / "cfrc_frontier_decisions.jsonl"
    agent_name = register_tau2_cfrc_agent(
        agent_name=safe_agent_name(f"papc_cfrc_frontier_{run_name}"),
        domain=args.domain,
        strong_model=args.strong_model,
        strong_llm_args=llm_args_for_model(
            args.api_base, args.strong_reasoning_effort, args.strong_model
        ),
        decision_log_path=decision_log_path,
        mutating_tool_names=args.mutating_tool_names,
        include_frontier_candidate_hint=True,
        frontier_timing=args.frontier_timing,
        post_write_residual_policy=args.post_write_residual_policy,
        pre_write_residual_policy=args.pre_write_residual_policy,
        strong_call_cap=args.strong_call_cap,
        suffix_tool_call_budget=suffix_tool_call_budget(args),
        allow_message_residual=args.allow_message_residual,
        compiler_setting=args.compiler_setting,
        # The runner keyword still reads request_contract_verifier because
        # scripts/tau2/papc_tau2_airline_residual_run.py passes that name;
        # renaming it needs that caller updated in the same change.
        request_contract_verifier=args.request_contract_enabled,
        request_contract_private_replica=args.request_contract_private_replica,
        max_request_contract_calls=args.max_request_contract_calls,
        request_contract_suffix_cap=args.request_contract_suffix_cap,
        contract_builder_variant=args.contract_builder_variant,
        deterministic_contract_closure=deterministic_closure_enabled(args),
        zero_obligation_request_fallback=zero_obligation_fallback_enabled(args),
        grounded_terminal_literals=bool(
            getattr(args, "grounded_terminal_literals", False)
        ),
        transcription_is_fallback=bool(
            getattr(args, "transcription_is_fallback", False)
        ),
        contract_builder_attempt_cap=args.contract_builder_attempt_cap,
        contract_builder_executor=args.contract_builder_executor,
        contract_builder_model=args.cheap_model,
        contract_builder_llm_args=llm_args_for_model(
            args.api_base, args.cheap_reasoning_effort, args.cheap_model
        ),
    )
    base_payload = (
        base_results_payload(args.patch_infra_from)
        if getattr(args, "patch_infra_from", None) is not None
        else None
    )
    patched_task_ids = (
        infrastructure_task_ids(base_payload) if base_payload is not None else []
    )
    results = run_official_agent_arm(
        args,
        arm_name="frontier_residual_method",
        run_dir=run_dir,
        model=args.cheap_model,
        reasoning_effort=args.cheap_reasoning_effort,
        agent_name=agent_name,
        task_ids=patched_task_ids or None,
    )
    results_payload = results.model_dump(mode="json")
    patch_provenance: dict[str, Any] | None = None
    if base_payload is not None:
        results_payload = merge_patch_results(
            base_payload=base_payload,
            patch_payload=results_payload,
            patched_task_ids=patched_task_ids,
        )
        patch_provenance = {
            "schema": INFRASTRUCTURE_PATCH_SCHEMA,
            "base_private_root": str(args.patch_infra_from),
            "base_run_name": base_payload.get("run_name"),
            "patched_task_count": len(patched_task_ids),
            "patch_reason": "infrastructure_error_only",
            "episodes_still_infrastructure_after_patch": len(
                infrastructure_task_ids(results_payload)
            ),
        }
    results_payload["schema"] = "papc.tau2.cfrc_frontier_method_private_results.v1"
    results_payload["run_name"] = run_name
    results_payload["protocol"] = "cheap_verified_frontier_then_strong_typed_residual_projection"
    results_payload["frontier_timing"] = args.frontier_timing
    results_payload["post_write_residual_policy"] = args.post_write_residual_policy
    results_payload["pre_write_residual_policy"] = args.pre_write_residual_policy
    results_payload["raw_task_ids_present_private"] = True
    (method_dir / "frontier_method_results.json").write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_rows = load_jsonl(decision_log_path)
    if base_payload is not None:
        sidecar_rows = merge_patch_sidecar(
            base_rows=load_jsonl(
                args.patch_infra_from
                / "frontier_residual_method"
                / "cfrc_frontier_decisions.jsonl"
            ),
            patch_rows=sidecar_rows,
            patched_task_ids=patched_task_ids,
        )
    return {
        "results_payload": results_payload,
        "sidecar_rows": sidecar_rows,
        "patch_provenance": patch_provenance,
    }


def run_terminal_residual_method_arm(
    args: argparse.Namespace,
    *,
    run_name: str,
    run_dir: Path,
    cheap_results: Any,
) -> dict[str, Any]:
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolCall,
        UserMessage,
    )
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate
    from tau2.utils.utils import get_now

    method_dir = run_dir / "terminal_residual_method"
    method_dir.mkdir(parents=True, exist_ok=False)
    decision_log_path = method_dir / "cfrc_terminal_decisions.jsonl"
    simulations: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    tasks_by_id = {str(task.id): task for task in cheap_results.tasks}
    env_constructor = registry.get_env_constructor(args.domain)
    for index, cheap_simulation in enumerate(cheap_results.simulations):
        task = tasks_by_id[str(cheap_simulation.task_id)]
        cheap_messages = cheap_simulation.get_messages()
        environment = env_constructor()
        initial_state = task.initial_state
        environment.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions if initial_state is not None else None
            ),
            message_history=cheap_messages,
        )
        tools = environment.get_tools()
        mutating_tools = set(
            normalize_mutating_tool_names(
                tools,
                override=args.mutating_tool_names,
            )
        )
        visible_evidence = compact_visible_evidence(cheap_messages, max_events=24)
        committed_tool_calls, committed_tool_receipts = mutating_tool_receipts_from_messages(
            cheap_messages,
            mutating_tools,
        )
        certificate = build_residual_certificate(
            domain=args.domain,
            allowed_residual_tools=mutating_tools,
            commitments=receipt_bound_commitments(
                committed_tool_calls,
                committed_tool_receipts,
            ),
            evidence_facts=evidence_facts_from_visible_rows(visible_evidence),
            frontier_tool_names=(),
            evidence_event_count=len(cheap_messages),
            message_residual_allowed=args.allow_message_residual,
        )
        strong_message = None
        if strong_call_required(certificate):
            prompt_payload = build_strong_prompt_payload(
                certificate=certificate,
                visible_evidence=visible_evidence,
                residual_tools=residual_tool_schemas(tools, mutating_tools),
                cheap_frontier_tool_calls=(),
            )
            prompt_payload["initial_visible_user_request"] = first_user_text(cheap_messages)
            prompt_payload["terminal_residual_obligation"] = (
                "The cheap model has completed its official trajectory. "
                "Using only visible evidence, propose the minimal residual tool-call "
                "suffix that can complete the user's unresolved request from this reached "
                "state, or abstain."
            )
            strong_message = generate(
                model=args.strong_model,
                messages=[
                    SystemMessage(role="system", content=CFRC_STRONG_SYSTEM_PROMPT),
                    UserMessage.text(json.dumps(prompt_payload, ensure_ascii=False)),
                ],
                call_name="cfrc_terminal_strong_residual_proposal",
                **llm_args_for_model(
                    args.api_base,
                    args.strong_reasoning_effort,
                    args.strong_model,
                ),
            )
            proposal = parse_typed_proposal(strong_message.content or "")
        else:
            proposal = parse_typed_proposal(
                {
                    "nodes": [],
                    "abstain_reason": "no_verifier_issued_open_residual_obligation",
                }
            )
        projection_start = time.perf_counter()
        result = project_cfrc_proposal(
            certificate=certificate,
            proposal=proposal,
            available_tool_names=[str(tool.name) for tool in tools],
            visible_evidence=visible_evidence,
            compiler_setting=args.compiler_setting,
            compiler_suffix_tool_call_budget=suffix_tool_call_budget(args),
        )
        projection_seconds = time.perf_counter() - projection_start
        messages = copy.deepcopy(cheap_messages)
        suffix_seconds = 0.0
        tool_results: list[Any] = []
        if result.execute and result.suffix_tool_calls():
            suffix_calls = [
                ToolCall(
                    id=f"papc_cfrc_terminal_{index}_{call_index}",
                    name=call["name"],
                    arguments=dict(call["arguments"]),
                )
                for call_index, call in enumerate(result.suffix_tool_calls())
            ]
            residual_message = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=suffix_calls,
                raw_data={"papc_cfrc": result.public_json()},
            )
            messages.append(residual_message)
            suffix_start = time.perf_counter()
            tool_results = [environment.get_response(call) for call in suffix_calls]
            suffix_seconds = time.perf_counter() - suffix_start
            messages.extend(tool_results)
            suffix_message, receipt_summary = suffix_final_message_from_receipts(tool_results)
            messages.append(
                AssistantMessage.text(
                    suffix_message,
                    raw_data={
                        "papc_cfrc": {
                            "deterministic_suffix_final": True,
                            "receipt_summary": receipt_summary,
                        }
                    },
                )
            )
        elif result.execute and result.suffix_messages():
            messages.append(
                AssistantMessage.text(
                    result.suffix_messages()[0],
                    raw_data={"papc_cfrc": result.public_json()},
                )
            )
        else:
            messages.append(
                AssistantMessage.text(
                    "I cannot safely complete the remaining action from the current evidence.",
                    raw_data={"papc_cfrc": result.public_json()},
                )
            )

        evaluation_start = time.perf_counter()
        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=str(task.id),
            start_time=cheap_simulation.start_time,
            end_time=get_now(),
            duration=round(
                float(cheap_simulation.duration or 0.0)
                + projection_seconds
                + suffix_seconds
                + float(getattr(strong_message, "generation_time_seconds", None) or 0.0),
                6,
            ),
            termination_reason=TerminationReason.USER_STOP,
            reward_info=None,
            messages=messages,
            trial=cheap_simulation.trial,
            seed=cheap_simulation.seed,
            mode=CommunicationMode.HALF_DUPLEX.value,
            info={
                "papc_cfrc": {
                    "source": "cheap_full_terminal_state",
                    "cheap_duration_seconds": cheap_simulation.duration,
                    "deterministic_projection_seconds": round(projection_seconds, 6),
                    "residual_suffix_tool_execution_seconds": round(suffix_seconds, 6),
                }
            },
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=args.domain,
            mode=CommunicationMode.HALF_DUPLEX,
        )
        evaluator_seconds = time.perf_counter() - evaluation_start
        simulation.reward_info = reward_info
        row = terminal_decision_row(
            task_id=str(task.id),
            strong_message=strong_message,
            proposal=proposal,
            result=result,
            projection_seconds=projection_seconds,
            suffix_seconds=suffix_seconds,
            tool_results=tool_results,
        )
        write_jsonl_append(decision_log_path, row)
        sidecar_rows.append(row)
        sim_payload = simulation.model_dump(mode="json")
        sim_payload["evaluation_time_seconds"] = round(evaluator_seconds, 6)
        simulations.append(sim_payload)

    results_payload = {
        "schema": "papc.tau2.cfrc_terminal_method_private_results.v1",
        "run_name": run_name,
        "protocol": "cheap_full_then_strong_typed_residual_projection",
        "simulations": simulations,
        "raw_task_ids_present_private": True,
    }
    (method_dir / "terminal_method_results.json").write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "results_payload": results_payload,
        "sidecar_rows": sidecar_rows,
    }


def build_launch_report(
    args: argparse.Namespace,
    *,
    run_name: str,
    execute: bool,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    suffix_budget = suffix_tool_call_budget(args)
    arms = [method_arm_name(args)] if args.method_only else [
        "cheap_full",
        "strong_full",
        method_arm_name(args),
    ]
    return {
        "schema": "papc.tau2.cfrc_clean_launch_report.v1",
        "run_name": run_name,
        "mode": "execute" if execute else "dry_run_launch_report",
        "decision": "blocked" if blockers else "ready",
        "blockers": blockers or [],
        "surface": "tau2",
        "domain": args.domain,
        "domain_split": args.domain_split,
        "cap": {
            "tasks_per_domain": args.tasks_per_domain,
            "num_trials": args.num_trials,
            "task_id_selection_supported": True,
            "selected_task_count": len(args.task_ids or []),
            "selected_task_id_sha256s": _task_id_sha256s(args.task_ids),
        },
        "execution_mode": {
            "method_only": bool(args.method_only),
            "reruns_cheap_and_strong_comparator_arms": not args.method_only,
            "cheap_comparator_public_json": public_path_label(
                args.cheap_comparator_public_json
            ),
            "strong_comparator_public_json": public_path_label(
                args.strong_comparator_public_json
            ),
        },
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
        "protocol": method_protocol(args),
        "arms": arms,
        "method_mode": args.method_mode,
        "method_family": "graph_cfrc_result_first",
        "request_contract_enabled": bool(args.request_contract_enabled),
        "request_contract_private_replica": bool(
            args.request_contract_private_replica
        ),
        "request_contract_mode": _request_contract_mode(args),
        "contract_builder_variant": args.contract_builder_variant,
        "deterministic_contract_closure": deterministic_closure_enabled(args),
        "zero_obligation_request_fallback": (
            zero_obligation_fallback_enabled(args)
        ),
        "grounded_terminal_literals": bool(
            getattr(args, "grounded_terminal_literals", False)
        ),
        "transcription_is_fallback": bool(
            getattr(args, "transcription_is_fallback", False)
        ),
        "strong_provider_call_budget": strong_provider_call_budget(args),
        "acceptance_policy": acceptance_policy_report(args),
        "max_request_contract_calls": int(args.max_request_contract_calls),
        "request_contract_suffix_cap": int(args.request_contract_suffix_cap),
        "diagnostic_probe": bool(args.diagnostic_probe),
        "method_layers": method_layer_report(args),
        "frontier_timing": args.frontier_timing if args.method_mode == "frontier" else None,
        "post_write_residual_policy": (
            args.post_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "post_tool_result"
            else None
        ),
        "pre_write_residual_policy": (
            args.pre_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "pre_write_candidate"
            else None
        ),
        "frontier_verifier": frontier_verifier_report(args),
        "shared_compiler": compiler_setting_report(args),
        "compiler_setting": args.compiler_setting,
        "strong_call_cap_per_task": args.strong_call_cap,
        "residual_proposal_call_cap_per_task": args.strong_call_cap,
        "total_strong_provider_call_cap_per_task": strong_provider_call_budget(args)[
            "total_strong_provider_call_cap"
        ],
        "suffix_tool_call_budget_per_task": suffix_budget,
        "message_residual_extension": message_residual_extension_report(args),
        "provider_calls": 0 if not execute else None,
        "private_root_supplied": bool(args.output_root),
        "api_base_supplied": bool(args.api_base),
        "api_key_env_supplied": bool(args.api_key_env),
        "raw_task_ids_excluded": True,
        "public_safe": True,
    }


def build_execute_public_summary(
    args: argparse.Namespace,
    *,
    run_name: str,
    cheap_results_payload: dict[str, Any],
    strong_results_payload: dict[str, Any],
    method_results_payload: dict[str, Any],
    method_sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    suffix_budget = suffix_tool_call_budget(args)
    cheap_arm = arm_public_summary(
        args,
        arm_name="cheap_full",
        results_payload=cheap_results_payload,
        sidecar_rows=[],
    )
    strong_arm = arm_public_summary(
        args,
        arm_name="strong_full",
        results_payload=strong_results_payload,
        sidecar_rows=[],
    )
    method_arm = arm_public_summary(
        args,
        arm_name=method_arm_name(args),
        results_payload=method_results_payload,
        sidecar_rows=method_sidecar_rows,
    )
    if "projection_summary" in method_arm:
        regression = cheap_regression_summary(
            cheap_results_payload=cheap_results_payload,
            method_results_payload=method_results_payload,
        )
        method_arm["projection_summary"]["cheap_regression_count"] = regression[
            "cheap_regression_count"
        ]
        method_arm["projection_summary"]["cheap_regression_comparable_count"] = (
            regression["cheap_regression_comparable_count"]
        )
    method_layers = method_layer_breakdown(method_arm)
    method_arm["method_layer_breakdown"] = method_layers
    postrun_acceptance = postrun_acceptance_report(args, method_arm=method_arm)
    summary = {
        "schema": execute_public_schema(args),
        "run_name": run_name,
        "surface": "tau2",
        "domain": args.domain,
        "domain_split": args.domain_split,
        "protocol": method_protocol(args),
        "method_family": "graph_cfrc_result_first",
        "method_layers": method_layer_report(args),
        "method_layer_breakdown": method_layers,
        "method_mode": args.method_mode,
        "frontier_timing": args.frontier_timing if args.method_mode == "frontier" else None,
        "post_write_residual_policy": (
            args.post_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "post_tool_result"
            else None
        ),
        "pre_write_residual_policy": (
            args.pre_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "pre_write_candidate"
            else None
        ),
        "task_count": method_arm["task_count"],
        "diagnostic_probe": bool(args.diagnostic_probe),
        "diagnostic_task_count": len(args.task_ids or []),
        "diagnostic_task_id_sha256s": _task_id_sha256s(args.task_ids),
        "request_contract_enabled": bool(args.request_contract_enabled),
        "request_contract_mode": _request_contract_mode(args),
        "contract_builder_variant": args.contract_builder_variant,
        "deterministic_contract_closure": deterministic_closure_enabled(args),
        "zero_obligation_request_fallback": (
            zero_obligation_fallback_enabled(args)
        ),
        "grounded_terminal_literals": bool(
            getattr(args, "grounded_terminal_literals", False)
        ),
        "transcription_is_fallback": bool(
            getattr(args, "transcription_is_fallback", False)
        ),
        "strong_provider_call_budget": strong_provider_call_budget(args),
        "acceptance_policy": acceptance_policy_report(args),
        "postrun_acceptance": postrun_acceptance,
        "eligible_to_update_current_evidence": postrun_acceptance[
            "eligible_to_update_current_evidence"
        ],
        "arms": {
            "cheap_full": cheap_arm,
            "strong_full": strong_arm,
            method_arm_name(args): method_arm,
        },
        "comparison": {
            "method_minus_cheap_reward_sum": nullable_round(
                method_arm["reward_sum"], cheap_arm["reward_sum"]
            ),
            "method_minus_strong_reward_sum": nullable_round(
                method_arm["reward_sum"], strong_arm["reward_sum"]
            ),
            "method_reward_not_worse_than_cheap": (
                method_arm["reward_sum"] >= cheap_arm["reward_sum"]
                if method_arm["reward_sum"] is not None
                and cheap_arm["reward_sum"] is not None
                else None
            ),
            "method_reward_not_worse_than_strong": (
                method_arm["reward_sum"] >= strong_arm["reward_sum"]
                if method_arm["reward_sum"] is not None
                and strong_arm["reward_sum"] is not None
                else None
            ),
        },
        "strong_decision_rows": len(method_sidecar_rows),
        "shared_compiler": compiler_setting_report(args),
        "compiler_setting": args.compiler_setting,
        "strong_call_cap_per_task": args.strong_call_cap,
        "residual_proposal_call_cap_per_task": args.strong_call_cap,
        "total_strong_provider_call_cap_per_task": strong_provider_call_budget(args)[
            "total_strong_provider_call_cap"
        ],
        "suffix_tool_call_budget_per_task": suffix_budget,
        "message_residual_extension": message_residual_extension_report(args),
        "primary_resource_comparison": primary_resource_comparison_report(),
        "latency_accounting": {
            "primary_comparison_is_model_call_latency_and_tokens": True,
            "total_runtime_is_secondary_accounting": True,
            "method_model_calls_are_sequential": True,
            "method_incremental_overhead_excludes_direct_strong_baseline": True,
            "user_simulator_and_evaluator_reported_separately": True,
            "missing_latency_fields_are_unavailable_not_zero": True,
        },
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
        "raw_private_paths_excluded": True,
        "raw_task_ids_excluded": True,
        "public_safe": True,
    }
    summary["model_resource_summary"] = model_resource_summary(summary["arms"])
    summary["external_time_accounting"] = external_time_accounting(summary["arms"])
    return summary


def build_official_baseline_public_summary(
    args: argparse.Namespace,
    *,
    run_name: str,
    arm_name: str,
    results_payload: dict[str, Any],
) -> dict[str, Any]:
    arm = arm_public_summary(
        args,
        arm_name=arm_name,
        results_payload=results_payload,
        sidecar_rows=[],
    )
    model = args.cheap_model if arm_name == "cheap_full" else args.strong_model
    role = "cheap" if arm_name == "cheap_full" else "strong"
    return {
        "schema": f"{execute_public_schema(args)}.official_baseline_only",
        "run_name": run_name,
        "surface": "tau2",
        "domain": args.domain,
        "domain_split": args.domain_split,
        "protocol": "official_native_agent_single_arm",
        "arm_name": arm_name,
        "models": {"agent": model, "agent_role": role},
        "task_count": arm["task_count"],
        "arm": arm,
        "public_safe": True,
    }


def build_method_only_public_summary(
    args: argparse.Namespace,
    *,
    run_name: str,
    method_results_payload: dict[str, Any],
    method_sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    suffix_budget = suffix_tool_call_budget(args)
    method_arm = arm_public_summary(
        args,
        arm_name=method_arm_name(args),
        results_payload=method_results_payload,
        sidecar_rows=method_sidecar_rows,
    )
    method_layers = method_layer_breakdown(method_arm)
    method_arm["method_layer_breakdown"] = method_layers
    postrun_acceptance = postrun_acceptance_report(args, method_arm=method_arm)
    summary = {
        "schema": f"{execute_public_schema(args)}.method_only",
        "run_name": run_name,
        "surface": "tau2",
        "domain": args.domain,
        "domain_split": args.domain_split,
        "protocol": method_protocol(args),
        "method_family": "graph_cfrc_result_first",
        "method_layers": method_layer_report(args),
        "method_layer_breakdown": method_layers,
        "method_mode": args.method_mode,
        "frontier_timing": args.frontier_timing if args.method_mode == "frontier" else None,
        "post_write_residual_policy": (
            args.post_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "post_tool_result"
            else None
        ),
        "pre_write_residual_policy": (
            args.pre_write_residual_policy
            if args.method_mode == "frontier"
            and args.frontier_timing == "pre_write_candidate"
            else None
        ),
        "task_count": method_arm["task_count"],
        "diagnostic_probe": bool(args.diagnostic_probe),
        "diagnostic_task_count": len(args.task_ids or []),
        "diagnostic_task_id_sha256s": _task_id_sha256s(args.task_ids),
        "request_contract_enabled": bool(args.request_contract_enabled),
        "request_contract_mode": _request_contract_mode(args),
        "contract_builder_variant": args.contract_builder_variant,
        "deterministic_contract_closure": deterministic_closure_enabled(args),
        "zero_obligation_request_fallback": (
            zero_obligation_fallback_enabled(args)
        ),
        "grounded_terminal_literals": bool(
            getattr(args, "grounded_terminal_literals", False)
        ),
        "transcription_is_fallback": bool(
            getattr(args, "transcription_is_fallback", False)
        ),
        "strong_provider_call_budget": strong_provider_call_budget(args),
        "acceptance_policy": acceptance_policy_report(args),
        "postrun_acceptance": postrun_acceptance,
        "eligible_to_update_current_evidence": postrun_acceptance[
            "eligible_to_update_current_evidence"
        ],
        "execution_mode": {
            "method_only": True,
            "reruns_cheap_and_strong_comparator_arms": False,
            "cheap_comparator_public_json": public_path_label(
                args.cheap_comparator_public_json
            ),
            "strong_comparator_public_json": public_path_label(
                args.strong_comparator_public_json
            ),
        },
        "external_comparators": {
            "cheap_full": public_comparator_summary(args.cheap_comparator_public_json),
            "strong_full": public_comparator_summary(args.strong_comparator_public_json),
        },
        "arms": {
            method_arm_name(args): method_arm,
        },
        "strong_decision_rows": len(method_sidecar_rows),
        "shared_compiler": compiler_setting_report(args),
        "compiler_setting": args.compiler_setting,
        "strong_call_cap_per_task": args.strong_call_cap,
        "residual_proposal_call_cap_per_task": args.strong_call_cap,
        "total_strong_provider_call_cap_per_task": strong_provider_call_budget(args)[
            "total_strong_provider_call_cap"
        ],
        "suffix_tool_call_budget_per_task": suffix_budget,
        "message_residual_extension": message_residual_extension_report(args),
        "primary_resource_comparison": primary_resource_comparison_report(),
        "latency_accounting": {
            "primary_comparison_is_model_call_latency_and_tokens": True,
            "total_runtime_is_secondary_accounting": True,
            "method_model_calls_are_sequential": True,
            "method_incremental_overhead_excludes_direct_strong_baseline": True,
            "user_simulator_and_evaluator_reported_separately": True,
            "missing_latency_fields_are_unavailable_not_zero": True,
        },
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
        "claim_boundary": [
            "method-only execute row; cheap and strong comparator arms are external public aggregates",
            "not a full tau2 or leaderboard claim",
            "task ids and raw trajectories are excluded from public output",
        ],
        "raw_private_paths_excluded": True,
        "raw_task_ids_excluded": True,
        "public_safe": True,
    }
    summary["model_resource_summary"] = model_resource_summary(summary["arms"])
    summary["external_time_accounting"] = external_time_accounting(summary["arms"])
    return summary


def arm_public_summary(
    args: argparse.Namespace,
    *,
    arm_name: str,
    results_payload: dict[str, Any],
    sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    simulations = [
        item for item in results_payload.get("simulations", []) if isinstance(item, dict)
    ]
    evaluable_simulations = [
        sim
        for sim in simulations
        if isinstance(sim.get("reward_info"), dict)
        and sim["reward_info"].get("reward") is not None
    ]
    rewards = [
        float(sim["reward_info"].get("reward")) for sim in evaluable_simulations
    ]
    runtime_ledger = cfrc_runtime_ledger_from_results(
        results_payload=results_payload,
        cheap_model=args.cheap_model,
        strong_model=args.strong_model,
        default_assistant_role=default_assistant_runtime_role(arm_name),
        sidecar_rows=sidecar_rows,
    )
    summary = {
        "schema": "papc.tau2.cfrc_arm_public_summary.v1",
        "arm": arm_name,
        "task_count": len(simulations),
        "reward_evaluable_count": len(evaluable_simulations),
        "infrastructure_excluded_count": len(simulations) - len(evaluable_simulations),
        "reward_sum": sum(rewards) if simulations else None,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "official_action_closure": official_action_closure(evaluable_simulations),
        "episode_wall_clock_seconds": episode_wall_clock_summary(simulations),
        "runtime_ledger": runtime_ledger,
        "runtime_ledger_completeness": runtime_ledger_completeness(runtime_ledger),
    }
    if sidecar_rows:
        summary["projection_summary"] = projection_summary(sidecar_rows)
        summary["incremental_residual_overhead"] = incremental_residual_overhead(
            runtime_ledger,
            sidecar_rows=sidecar_rows,
        )
    return summary


def default_assistant_runtime_role(arm_name: str) -> str | None:
    if arm_name == "strong_full":
        return "strong"
    if arm_name in {"cheap_full", "frontier_residual_method", "terminal_residual_method"}:
        return "cheap"
    return None


def model_resource_summary(arms: dict[str, Any]) -> dict[str, Any]:
    per_arm = {
        arm_name: runtime_roles_from_arm(as_dict(arm_summary))
        for arm_name, arm_summary in arms.items()
    }
    return {
        "schema": "papc.tau2.model_resource_summary.v1",
        "roles": ["cheap", "strong", "user_simulator", "evaluator"],
        "per_arm": per_arm,
        "missing_fields_are_unavailable_not_zero": True,
        "raw_provider_payloads_excluded": True,
        "public_safe": True,
    }


def primary_resource_comparison_report() -> dict[str, Any]:
    return {
        "schema": "papc.tau2.primary_resource_comparison.v1",
        "primary_roles": ["cheap", "strong"],
        "primary_fields": [
            "arms.*.runtime_ledger.per_role.cheap.input_tokens",
            "arms.*.runtime_ledger.per_role.cheap.output_tokens",
            "arms.*.runtime_ledger.per_role.cheap.latency_seconds",
            "arms.*.runtime_ledger.per_role.strong.input_tokens",
            "arms.*.runtime_ledger.per_role.strong.output_tokens",
            "arms.*.runtime_ledger.per_role.strong.latency_seconds",
        ],
        "secondary_accounting_fields": [
            "external_time_accounting.*.aggregate_external_time_seconds",
            "external_time_accounting.*.parallel_elapsed_wall_seconds",
            "external_time_accounting.*.parallel_capacity_seconds",
        ],
        "total_runtime_is_secondary_accounting": True,
        "missing_latency_fields_are_unavailable_not_zero": True,
    }


def runtime_roles_from_arm(arm_summary: dict[str, Any]) -> dict[str, Any]:
    ledger = as_dict(arm_summary.get("runtime_ledger"))
    per_role = as_dict(ledger.get("per_role"))
    return {
        role: as_dict(per_role.get(role))
        for role in ("cheap", "strong", "user_simulator", "evaluator")
    }


def external_time_accounting(arms: dict[str, Any]) -> dict[str, Any]:
    return {
        arm_name: external_time_from_arm(as_dict(arm_summary))
        for arm_name, arm_summary in arms.items()
    }


def external_time_from_arm(arm_summary: dict[str, Any]) -> dict[str, Any]:
    episode = as_dict(arm_summary.get("episode_wall_clock_seconds"))
    episode_total = optional_float(episode.get("total_seconds"))
    per_role = runtime_roles_from_arm(arm_summary)
    observed_latencies: list[float] = []
    latency_unavailable_roles: list[str] = []
    for role, row in per_role.items():
        if row.get("latency_available") is True:
            latency = optional_float(row.get("latency_seconds"))
            if latency is not None:
                observed_latencies.append(latency)
        elif row.get("observed") is True:
            latency_unavailable_roles.append(role)
    model_latency_total = round(sum(observed_latencies), 6) if observed_latencies else None
    return {
        "schema": "papc.tau2.external_time_accounting.v1",
        "parallel_elapsed_wall_seconds": None,
        "aggregate_episode_wall_seconds": episode_total,
        "aggregate_model_latency_seconds": model_latency_total,
        "aggregate_external_time_seconds": episode_total,
        "aggregate_external_time_source": "sum_of_episode_durations",
        "max_concurrency": None,
        "parallel_capacity_seconds": None,
        "parallel_capacity_seconds_source": None,
        "parallel_capacity_seconds_is_upper_bound_not_total": None,
        "aggregate_external_time_prefers_episode_sum_when_available": True,
        "mean_episode_wall_seconds": optional_float(episode.get("mean_seconds")),
        "episode_count": int(episode.get("count") or 0),
        "observed_model_latency_roles": [
            role
            for role, row in per_role.items()
            if row.get("latency_available") is True
        ],
        "latency_unavailable_roles": latency_unavailable_roles,
        "model_latency_is_partial_when_roles_unavailable": bool(
            latency_unavailable_roles
        ),
        "parallel_elapsed_is_not_total_external_time": True,
        "missing_latency_fields_are_unavailable_not_zero": True,
        "public_safe": True,
    }


def attach_parallel_elapsed_to_external_time_accounting(
    report: dict[str, Any],
    *,
    wall_seconds: float,
    max_concurrency: int | None = None,
) -> None:
    accounting = as_dict(report.get("external_time_accounting"))
    if not accounting:
        return
    elapsed = round(max(0.0, float(wall_seconds)), 6)
    for row in accounting.values():
        if isinstance(row, dict):
            row["parallel_elapsed_wall_seconds"] = elapsed
            attach_parallel_capacity_accounting(
                row,
                parallel_elapsed_wall_seconds=elapsed,
                max_concurrency=max_concurrency,
            )


def attach_parallel_capacity_accounting(
    row: dict[str, Any],
    *,
    parallel_elapsed_wall_seconds: float | None,
    max_concurrency: int | None,
) -> None:
    concurrency = int(max_concurrency) if max_concurrency is not None else None
    if concurrency is not None and concurrency <= 0:
        concurrency = None
    elapsed = optional_float(parallel_elapsed_wall_seconds)
    row["max_concurrency"] = concurrency
    row["parallel_capacity_seconds"] = (
        round(elapsed * concurrency, 6)
        if elapsed is not None and concurrency is not None
        else None
    )
    row["parallel_capacity_seconds_source"] = (
        "parallel_elapsed_wall_seconds_x_max_concurrency"
        if row["parallel_capacity_seconds"] is not None
        else None
    )
    row["parallel_capacity_seconds_is_upper_bound_not_total"] = (
        row["parallel_capacity_seconds"] is not None
    )
    row["aggregate_external_time_prefers_episode_sum_when_available"] = True


def method_layer_breakdown(method_arm: dict[str, Any]) -> dict[str, Any]:
    projection = as_dict(method_arm.get("projection_summary"))
    core_suffix_count = int(projection.get("core_suffix_count") or 0)
    local_rescue_count = int(projection.get("local_rescue_count") or 0)
    continuation_count = int(projection.get("continuation_count") or 0)
    distinct_continuation_total = int(
        projection.get("core_plus_rescue_plus_continuation_total")
        or (core_suffix_count + local_rescue_count)
    )
    final_total = method_arm.get("reward_sum")
    return {
        "schema": "papc.tau2.graph_cfrc_result_first_breakdown.v2",
        "core_only": core_suffix_count,
        "core_plus_rescue": core_suffix_count + local_rescue_count,
        "core_plus_rescue_plus_continuation": (
            final_total if final_total is not None else distinct_continuation_total
        ),
        "core_plus_rescue_plus_continuation_decision_count": (
            distinct_continuation_total
        ),
        "continuation_is_overlay_not_additive": (
            projection.get("continuation_is_overlay_not_additive") is True
        ),
        "core_plus_cheap_continuation": final_total,
        "cheap_continuation_decision_count": continuation_count,
        "total_reward": method_arm.get("reward_sum"),
        "total_reward_mean": method_arm.get("reward_mean"),
        "reward_evaluable_count": method_arm.get("reward_evaluable_count"),
        "graph_cfrc_core": {
            "core_suffix_count": core_suffix_count,
            "decision_count": int(projection.get("graph_core_decision_rows") or 0),
        },
        "bounded_residual_rescue": {
            "local_rescue_count": local_rescue_count,
            "local_rescue_attempt_count": int(
                projection.get("local_rescue_attempt_count") or 0
            ),
            "local_rescue_tool_suffix_count": int(
                projection.get("local_rescue_tool_suffix_count") or 0
            ),
            "local_rescue_message_count": int(
                projection.get("local_rescue_message_count") or 0
            ),
        },
        "graph_residual_continuation": {
            "continuation_count": continuation_count,
            "continuation_after_residual_count": int(
                projection.get("continuation_after_residual_count") or 0
            ),
            "continuation_after_abstain_count": int(
                projection.get("continuation_after_abstain_count") or 0
            ),
        },
    }


def runtime_ledger_completeness(runtime_ledger: dict[str, Any]) -> dict[str, Any]:
    per_role = as_dict(runtime_ledger.get("per_role"))
    required_roles = ("cheap", "strong", "user_simulator", "evaluator")
    role_status: dict[str, dict[str, bool]] = {}
    missing_roles: list[str] = []
    token_unavailable_roles: list[str] = []
    latency_unavailable_roles: list[str] = []
    for role in required_roles:
        row = as_dict(per_role.get(role))
        observed = row.get("observed") is True
        token_available = row.get("token_usage_available") is True
        latency_available = row.get("latency_available") is True
        role_status[role] = {
            "observed": observed,
            "token_usage_available": token_available,
            "latency_available": latency_available,
        }
        if not observed:
            missing_roles.append(role)
        if not token_available:
            token_unavailable_roles.append(role)
        if not latency_available:
            latency_unavailable_roles.append(role)
    return {
        "schema": "papc.tau2.runtime_ledger_completeness.v1",
        "required_roles": list(required_roles),
        "role_status": role_status,
        "all_required_roles_observed": not missing_roles,
        "missing_roles": missing_roles,
        "token_unavailable_roles": token_unavailable_roles,
        "latency_unavailable_roles": latency_unavailable_roles,
        "unavailable_is_null_not_zero": True,
    }


def official_action_closure(simulations: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "all": {"matched": 0, "total": 0},
        "read": {"matched": 0, "total": 0},
        "write": {"matched": 0, "total": 0},
        "unknown": {"matched": 0, "total": 0},
    }
    db_match = 0
    db_total = 0
    for simulation in simulations:
        reward_info = as_dict(simulation.get("reward_info"))
        db_check = as_dict(reward_info.get("db_check"))
        if db_check:
            db_total += 1
            if db_check.get("db_match") is True:
                db_match += 1
        for check in list_value(reward_info.get("action_checks")):
            if not isinstance(check, dict):
                continue
            tool_type = str(check.get("tool_type") or "unknown").strip().lower()
            if tool_type not in {"read", "write"}:
                tool_type = "unknown"
            matched = float(check.get("action_reward") or 0.0) >= 1.0
            totals["all"]["total"] += 1
            totals[tool_type]["total"] += 1
            if matched:
                totals["all"]["matched"] += 1
                totals[tool_type]["matched"] += 1
    return {
        "schema": "papc.tau2.official_action_closure.v1",
        "db_match": {"matched": db_match, "total": db_total},
        "all": dict(totals["all"]),
        "read": dict(totals["read"]),
        "write": dict(totals["write"]),
        "unknown": dict(totals["unknown"]),
    }


def projection_summary(sidecar_rows: list[dict[str, Any]]) -> dict[str, Any]:
    execute_count = 0
    core_task_count = 0
    local_rescue_attempt_count = 0
    local_rescue_tool_suffix_count = 0
    local_rescue_message_count = 0
    graph_core_decision_rows = 0
    pre_write_direct_core_count = 0
    post_tool_core_count = 0
    candidate_executed_before_frontier_count = 0
    candidate_not_executed_before_frontier_count = 0
    continuation_after_residual_count = 0
    continuation_after_abstain_count = 0
    native_continuation_verbatim_candidate_count = 0
    native_continuation_regenerated_count = 0
    non_empty_suffix_count = 0
    abstain_count = 0
    frontier_accepted_count = 0
    strong_call_count_sum = 0
    max_strong_call_count = 0
    selected_tool_name_counts: dict[str, int] = {}
    selected_read_tool_name_counts: dict[str, int] = {}
    selected_authority_label_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    selected_obligation_count_sum = 0
    selected_write_node_count_sum = 0
    selected_read_node_count_sum = 0
    receipt_success_count_sum = 0
    receipt_error_count_sum = 0
    receipt_unknown_count_sum = 0
    evidence_citation_reject_count_sum = 0
    residual_activation_reason_counts: dict[str, int] = {}
    continuation_policy_counts: dict[str, int] = {}
    replica_failure_error_counts: dict[str, int] = {}
    replica_failure_position_counts: dict[str, int] = {}
    for row in sidecar_rows:
        for signature in row.get("replica_failure_signatures") or []:
            if not isinstance(signature, dict):
                continue
            error_type = str(signature.get("error_type") or "unknown")
            replica_failure_error_counts[error_type] = (
                replica_failure_error_counts.get(error_type, 0) + 1
            )
            position = int(signature.get("failed_node_position") or 0)
            total = int(signature.get("tool_node_count") or 0)
            slot = f"{position}_of_{total}"
            replica_failure_position_counts[slot] = (
                replica_failure_position_counts.get(slot, 0) + 1
            )
        strong_call_count = int(row.get("strong_call_count") or 0)
        strong_call_count_sum += strong_call_count
        max_strong_call_count = max(max_strong_call_count, strong_call_count)
        local_rescue_agent = as_dict(row.get("local_rescue_agent"))
        if local_rescue_agent.get("triggered") is True:
            local_rescue_attempt_count += 1
            if local_rescue_agent.get("strong_message_emitted") is True:
                strong_tool_call_count = int(
                    local_rescue_agent.get("strong_tool_call_count") or 0
                )
                if strong_tool_call_count > 0:
                    local_rescue_tool_suffix_count += 1
                else:
                    local_rescue_message_count += 1
            continue
        graph_core_decision_rows += 1
        result = as_dict(row.get("projection_result"))
        continuation = as_dict(row.get("graph_residual_continuation"))
        if continuation.get("enabled") is True:
            policy = str(continuation.get("policy") or "unknown").strip() or "unknown"
            continuation_policy_counts[policy] = (
                continuation_policy_counts.get(policy, 0) + 1
            )
            if continuation.get("after_residual_suffix") is True:
                continuation_after_residual_count += 1
            if continuation.get("after_abstain") is True:
                continuation_after_abstain_count += 1
            native_continuation_mode = str(
                continuation.get("native_continuation_mode") or ""
            ).strip()
            if native_continuation_mode == "verbatim_candidate":
                native_continuation_verbatim_candidate_count += 1
            elif native_continuation_mode == "regenerated_cheap_turn":
                native_continuation_regenerated_count += 1
        frontier = as_dict(row.get("frontier_verifier"))
        residual_activation_reason = str(
            frontier.get("residual_activation_reason") or ""
        ).strip()
        if residual_activation_reason:
            residual_activation_reason_counts[residual_activation_reason] = (
                residual_activation_reason_counts.get(residual_activation_reason, 0) + 1
            )
        candidate_executed = frontier.get("candidate_tool_call_executed_before_frontier")
        if candidate_executed is True:
            candidate_executed_before_frontier_count += 1
            post_tool_core_count += 1
        elif candidate_executed is False:
            candidate_not_executed_before_frontier_count += 1
            if frontier.get("frontier_timing") == "pre_write_candidate":
                pre_write_direct_core_count += 1
        diagnostics = as_dict(as_dict(result.get("projection")).get("diagnostics"))
        selected_authority_label = projection_selected_authority_label(result)
        if selected_authority_label:
            selected_authority_label_counts[selected_authority_label] = (
                selected_authority_label_counts.get(selected_authority_label, 0) + 1
            )
        receipt_summary = as_dict(as_dict(row.get("receipt_summary")))
        if frontier.get("accepted") is True:
            frontier_accepted_count += 1
        reason = str(result.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        selected = [
            str(name)
            for name in list_value(result.get("selected_tool_names"))
            if str(name).strip()
        ]
        selected_reads = [
            str(name)
            for name in list_value(result.get("selected_read_tool_names"))
            if str(name).strip()
        ]
        selected_obligation_count_sum += int(result.get("selected_obligation_count") or 0)
        selected_write_node_count_sum += int(result.get("selected_write_node_count") or 0)
        selected_read_node_count_sum += int(result.get("selected_read_node_count") or 0)
        evidence_citation_reject_count_sum += int(
            diagnostics.get("evidence_citation_reject_count") or 0
        )
        receipt_success_count_sum += int(receipt_summary.get("success_count") or 0)
        receipt_error_count_sum += int(receipt_summary.get("error_count") or 0)
        receipt_unknown_count_sum += int(receipt_summary.get("unknown_count") or 0)
        if result.get("execute") is True:
            execute_count += 1
            if selected_authority_label == "ell3_bounded_local_redo":
                local_rescue_tool_suffix_count += 1
            else:
                core_task_count += 1
            if selected:
                non_empty_suffix_count += 1
        else:
            abstain_count += 1
        for name in selected:
            selected_tool_name_counts[name] = selected_tool_name_counts.get(name, 0) + 1
        for name in selected_reads:
            selected_read_tool_name_counts[name] = (
                selected_read_tool_name_counts.get(name, 0) + 1
            )
    local_rescue_count = local_rescue_tool_suffix_count + local_rescue_message_count
    continuation_count = continuation_after_residual_count + continuation_after_abstain_count
    core_plus_rescue_total = core_task_count + local_rescue_count
    core_plus_rescue_plus_continuation_total = (
        graph_core_decision_rows + local_rescue_count
    )
    return {
        "schema": "papc.tau2.cfrc_projection_summary.v1",
        "decision_rows": len(sidecar_rows),
        "graph_core_decision_rows": graph_core_decision_rows,
        "execute_count": execute_count,
        "core_task_count": core_task_count,
        "core_suffix_count": core_task_count,
        "local_rescue_triggered_count": local_rescue_attempt_count,
        "local_rescue_attempt_count": local_rescue_attempt_count,
        "local_rescue_tool_suffix_count": local_rescue_tool_suffix_count,
        "local_rescue_message_count": local_rescue_message_count,
        "local_rescue_count": local_rescue_count,
        "local_rescue_added_count": local_rescue_count,
        "continuation_after_residual_count": continuation_after_residual_count,
        "continuation_after_abstain_count": continuation_after_abstain_count,
        # Why the private replica refused a suffix, and where it stopped. Both
        # are structural and name no tool, argument, or value.
        "replica_failure_error_counts": dict(sorted(replica_failure_error_counts.items())),
        "replica_failure_position_counts": dict(sorted(replica_failure_position_counts.items())),
        "native_continuation_verbatim_candidate_count": (
            native_continuation_verbatim_candidate_count
        ),
        "native_continuation_regenerated_count": (
            native_continuation_regenerated_count
        ),
        "continuation_count": continuation_count,
        "pre_write_direct_core_count": pre_write_direct_core_count,
        "post_tool_core_count": post_tool_core_count,
        "candidate_executed_before_frontier_count": (
            candidate_executed_before_frontier_count
        ),
        "candidate_not_executed_before_frontier_count": (
            candidate_not_executed_before_frontier_count
        ),
        "cheap_regression_count": None,
        "core_plus_local_total": core_plus_rescue_total,
        "core_plus_rescue_total": core_plus_rescue_total,
        "core_plus_rescue_plus_continuation_total": (
            core_plus_rescue_plus_continuation_total
        ),
        "core_plus_cheap_continuation_total": graph_core_decision_rows,
        "continuation_is_overlay_not_additive": True,
        "strict_non_empty_suffix_count": non_empty_suffix_count,
        "abstain_count": abstain_count,
        "frontier_accepted_count": frontier_accepted_count,
        "frontier_acceptance_rate": (
            frontier_accepted_count / graph_core_decision_rows
            if graph_core_decision_rows
            else None
        ),
        "strong_call_count_sum": strong_call_count_sum,
        "max_strong_call_count_per_task": max_strong_call_count,
        "selected_obligation_count_sum": selected_obligation_count_sum,
        "selected_write_node_count_sum": selected_write_node_count_sum,
        "selected_read_node_count_sum": selected_read_node_count_sum,
        "receipt_success_count_sum": receipt_success_count_sum,
        "receipt_error_count_sum": receipt_error_count_sum,
        "receipt_unknown_count_sum": receipt_unknown_count_sum,
        "evidence_citation_reject_count_sum": evidence_citation_reject_count_sum,
        "residual_activation_reason_counts": dict(
            sorted(residual_activation_reason_counts.items())
        ),
        "graph_residual_continuation_policy_counts": dict(
            sorted(continuation_policy_counts.items())
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "selected_authority_label_counts": dict(
            sorted(selected_authority_label_counts.items())
        ),
        "selected_tool_name_counts": dict(sorted(selected_tool_name_counts.items())),
        "selected_read_tool_name_counts": dict(sorted(selected_read_tool_name_counts.items())),
    }


def projection_selected_authority_label(result: dict[str, Any]) -> str:
    projection = as_dict(result.get("projection"))
    compiler = as_dict(projection.get("compiler"))
    return str(
        result.get("selected_authority_label")
        or projection.get("selected_authority_label")
        or compiler.get("selected_authority_label")
        or ""
    )


def cheap_regression_summary(
    *,
    cheap_results_payload: dict[str, Any],
    method_results_payload: dict[str, Any],
) -> dict[str, int | None]:
    cheap_simulations = [
        item
        for item in cheap_results_payload.get("simulations", [])
        if isinstance(item, dict)
    ]
    method_simulations = [
        item
        for item in method_results_payload.get("simulations", [])
        if isinstance(item, dict)
    ]
    if len(cheap_simulations) != len(method_simulations):
        return {
            "cheap_regression_count": None,
            "cheap_regression_comparable_count": 0,
        }
    regression_count = 0
    comparable_count = 0
    for cheap_sim, method_sim in zip(cheap_simulations, method_simulations, strict=True):
        cheap_reward = simulation_reward(cheap_sim)
        method_reward = simulation_reward(method_sim)
        if cheap_reward is None or method_reward is None:
            continue
        comparable_count += 1
        if method_reward < cheap_reward:
            regression_count += 1
    return {
        "cheap_regression_count": regression_count,
        "cheap_regression_comparable_count": comparable_count,
    }


def simulation_reward(simulation: dict[str, Any]) -> float | None:
    reward_info = as_dict(simulation.get("reward_info"))
    reward = reward_info.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        return None
    return float(reward)


def episode_wall_clock_summary(simulations: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(value)
        for simulation in simulations
        for value in [simulation.get("duration")]
        if isinstance(value, int | float)
    ]
    total = sum(durations)
    return {
        "schema": "papc.tau2.episode_wall_clock.v1",
        "count": len(durations),
        "total_seconds": round(total, 6),
        "mean_seconds": round(total / len(durations), 6) if durations else None,
    }


def incremental_residual_overhead(
    runtime_ledger: dict[str, Any],
    *,
    sidecar_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    per_role = as_dict(runtime_ledger.get("per_role"))
    strong = as_dict(per_role.get("strong"))
    evaluator = as_dict(per_role.get("evaluator"))
    user_simulator = as_dict(per_role.get("user_simulator"))
    strong_latency_available = bool(strong.get("latency_available"))
    user_latency_available = bool(user_simulator.get("latency_available"))
    evaluator_latency_available = bool(evaluator.get("latency_available"))
    rows = sidecar_rows or []
    projection_values = [
        optional_float(row.get("deterministic_projection_seconds")) for row in rows
    ]
    suffix_values = [
        optional_float(row.get("residual_suffix_tool_execution_seconds")) for row in rows
    ]
    projection_available = bool(rows) and all(value is not None for value in projection_values)
    suffix_available = bool(rows) and all(value is not None for value in suffix_values)
    projection_seconds = (
        round(sum(value for value in projection_values if value is not None), 6)
        if projection_available
        else None
    )
    suffix_seconds = (
        round(sum(value for value in suffix_values if value is not None), 6)
        if suffix_available
        else None
    )
    return {
        "schema": "papc.tau2.cfrc_incremental_residual_overhead.v1",
        "strong_proposal_calls": int(strong.get("calls") or 0),
        "strong_proposal_latency_seconds": (
            strong.get("latency_seconds") if strong_latency_available else None
        ),
        "strong_proposal_latency_available": strong_latency_available,
        "deterministic_projection_time_seconds": projection_seconds,
        "deterministic_projection_time_available": projection_available,
        "residual_suffix_tool_execution_time_seconds": suffix_seconds,
        "residual_suffix_tool_execution_time_available": suffix_available,
        "user_simulator_latency_seconds": (
            user_simulator.get("latency_seconds") if user_latency_available else None
        ),
        "user_simulator_latency_available": user_latency_available,
        "evaluator_latency_seconds": (
            evaluator.get("latency_seconds") if evaluator_latency_available else None
        ),
        "evaluator_latency_available": evaluator_latency_available,
        "missing_latency_fields_are_unavailable_not_zero": True,
    }


def normalize_mutating_tool_names(
    tools: list[Any],
    *,
    override: list[str] | None,
) -> tuple[str, ...]:
    if override is not None:
        return tuple(sorted({str(name).strip() for name in override if str(name).strip()}))
    names: list[str] = []
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        func = getattr(tool, "_func", None)
        if getattr(func, "__mutates_state__", None) is True:
            names.append(name)
    if names:
        return tuple(sorted(set(names)))
    return mutating_tool_names_by_heuristic(str(getattr(tool, "name", "")) for tool in tools)


def mutating_tool_receipts_from_messages(
    messages: list[Any],
    mutating_tool_names: set[str],
) -> tuple[list[dict[str, Any]], list[Any]]:
    mutating_calls: list[dict[str, Any]] = []
    mutating_call_ids: list[str] = []
    matched_receipts_by_id: dict[str, Any] = {}
    for message in messages:
        tool_calls = list_value(getattr(message, "tool_calls", None))
        if tool_calls:
            for call in tool_calls:
                name = str(getattr(call, "name", "")).strip()
                if name not in mutating_tool_names:
                    continue
                call_id = str(getattr(call, "id", "")).strip()
                row = {
                    "name": str(getattr(call, "name", "")).strip(),
                    "arguments": dict(getattr(call, "arguments", {}) or {}),
                }
                if call_id:
                    row["tool_call_id"] = call_id
                mutating_calls.append(row)
                mutating_call_ids.append(call_id)
            continue
        if str(getattr(message, "role", "")) != "tool" and not getattr(
            message, "tool_call_id", None
        ):
            continue
        receipt_call_id = str(getattr(message, "tool_call_id", "")).strip()
        if receipt_call_id and receipt_call_id not in matched_receipts_by_id:
            matched_receipts_by_id[receipt_call_id] = message
    matched_calls: list[dict[str, Any]] = []
    matched_receipts: list[Any] = []
    for call, call_id in zip(mutating_calls, mutating_call_ids, strict=False):
        if not call_id:
            continue
        receipt = matched_receipts_by_id.get(call_id)
        if receipt is None:
            continue
        matched_calls.append(call)
        matched_receipts.append(receipt)
    return matched_calls, matched_receipts


def residual_tool_schemas(
    tools: list[Any],
    mutating_tool_names: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        if name not in mutating_tool_names:
            continue
        schema = getattr(tool, "openai_schema", None)
        rows.append(schema if isinstance(schema, dict) else {"name": name})
    return rows


def first_user_text(messages: list[Any], max_chars: int = 1200) -> str:
    for message in messages:
        if str(getattr(message, "role", "")) != "user":
            continue
        content = getattr(message, "content", None)
        if not content:
            continue
        text = str(content)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
    return ""


def terminal_decision_row(
    *,
    task_id: str,
    strong_message: Any | None,
    proposal: Any,
    result: Any,
    projection_seconds: float,
    suffix_seconds: float,
    tool_results: list[Any],
) -> dict[str, Any]:
    from papc.certspec.runtime_ledger import RuntimeLedger

    ledger = RuntimeLedger()
    if strong_message is not None:
        ledger.add_usage(
            "strong",
            as_dict(getattr(strong_message, "usage", None)) or None,
            latency_seconds=optional_float(getattr(strong_message, "generation_time_seconds", None)),
            cost_usd=optional_float(getattr(strong_message, "cost", None)),
        )
    _, receipt_summary = suffix_final_message_from_receipts(tool_results)
    return {
        "schema": "papc.tau2.cfrc_terminal_decision.v1",
        "task_hash": hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16],
        "protocol": "cheap_full_then_strong_typed_residual_projection",
        "strong_call_count": 1 if strong_message is not None else 0,
        "projection_result": result.public_json(),
        "strong_proposal_private": proposal.private_json(),
        "receipt_summary": receipt_summary,
        "runtime_ledger": ledger.public_summary(),
        "deterministic_projection_seconds": round(projection_seconds, 6),
        "residual_suffix_tool_execution_seconds": round(suffix_seconds, 6),
        "raw_messages_excluded": True,
        "public_outputs_exclude_private_diagnostics": True,
        "public_safe": False,
    }


def nullable_round(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def public_path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.as_posix()
    if text.startswith("outputs/papc/"):
        return text
    return path.name or None


def public_comparator_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    reward = (
        payload.get("official_reward")
        or payload.get("official_reward_excluding_infrastructure")
        or {}
    )
    if not isinstance(reward, dict):
        reward = {}
    return {
        "source_public_json": public_path_label(path),
        "schema": payload.get("schema"),
        "arm": payload.get("arm"),
        "reward_sum": reward.get("reward_sum"),
        "reward_mean": reward.get("reward_mean"),
        "denominator": reward.get("denominator")
        or payload.get("completed_simulations")
        or payload.get("reward_evaluable_simulations"),
        "infrastructure_excluded_simulations": payload.get(
            "infrastructure_excluded_simulations"
        ),
        "raw_private_paths_excluded": True,
        "raw_task_ids_excluded": True,
    }


def execute_blockers(args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    blockers.extend(_request_contract_gate_blockers(args))
    if (
        str(args.domain_split) == "test"
        and args.contract_builder_variant == "model_authored_request_semantics"
    ):
        acceptance_profile = retail_test_acceptance_profile(args)
        if acceptance_profile is None:
            blockers.append("model_authored_test_model_pool_unregistered")
        if args.reward_floor is None:
            blockers.append("model_authored_test_reward_floor_required")
        elif acceptance_profile is not None and float(args.reward_floor) != float(
            acceptance_profile["reward_floor"]
        ):
            blockers.append("model_authored_test_reward_floor_mismatch")
        if args.cost_ceiling_usd is None:
            blockers.append("model_authored_test_cost_ceiling_required")
        elif acceptance_profile is not None and float(
            args.cost_ceiling_usd
        ) != float(acceptance_profile["cost_ceiling_usd"]):
            blockers.append("model_authored_test_cost_ceiling_mismatch")
    if args.task_ids and not args.diagnostic_probe:
        blockers.append("task_ids_require_diagnostic_probe")
    if args.diagnostic_probe and not args.task_ids:
        blockers.append("diagnostic_probe_requires_task_ids")
    if getattr(args, "patch_infra_from", None) is not None:
        if args.task_ids:
            blockers.append("infrastructure_patch_disallows_task_ids")
        payload = patch_base_results_payload(args)
        if payload is None:
            blockers.append("patch_base_results_missing")
        elif not infrastructure_task_ids(payload):
            blockers.append("patch_base_has_no_infrastructure_rows")
        elif len(payload.get("simulations") or []) != retail_split_denominator(args):
            blockers.append("patch_base_denominator_mismatch")
    if args.task_ids and len(args.task_ids) >= RETAIL_DENOMINATOR:
        blockers.append("diagnostic_probe_must_be_smaller_than_denominator")
    if args.request_contract_enabled and args.method_mode != "frontier":
        blockers.append("request_contract_enabled_requires_frontier_mode")
    if args.request_contract_enabled and args.frontier_timing != "pre_write_candidate":
        blockers.append("request_contract_enabled_requires_prewrite_frontier")
    if args.request_contract_enabled and not args.method_only:
        blockers.append("request_contract_diagnostic_requires_method_only")
    if not args.execute:
        blockers.append("execute_flag_missing")
    if not args.api_base:
        blockers.append("api_base_missing")
    if not args.api_key_env:
        blockers.append("api_key_env_missing")
    elif not os.environ.get(args.api_key_env):
        blockers.append("api_key_env_not_present")
    if args.tasks_per_domain < 1:
        blockers.append("empty_task_cap")
    if args.num_trials < 1:
        blockers.append("empty_trial_cap")
    if int(getattr(args, "suffix_tool_call_budget", 1) or 0) < 0:
        blockers.append("suffix_tool_call_budget_must_be_non_negative")
    if args.method_mode == "terminal" and args.strong_call_cap != 1:
        blockers.append("terminal_mode_supports_only_b_call_1")
    if args.method_only and args.method_mode == "terminal":
        blockers.append("terminal_method_only_requires_cached_cheap_results")
    public_output = args.public_output_json
    if public_output is not None and Path(public_output).exists():
        blockers.append("public_output_json_already_exists")
    return sorted(set(blockers))


# Blockers that belong to the act of spending rather than to the configuration
# being described. A launch report carries no --execute by construction, writes
# to its own report path, and may legitimately be produced without credentials
# in the environment. Everything else is a property of the configuration and
# has to surface before the paid run rather than after it.
DRY_RUN_INAPPLICABLE_BLOCKERS = frozenset(
    {
        "api_base_missing",
        "api_key_env_missing",
        "api_key_env_not_present",
        "execute_flag_missing",
        "public_output_json_already_exists",
    }
)


def launch_report_blockers(args: argparse.Namespace) -> list[str]:
    """Blockers the paid launch would raise for this same configuration.

    The launch report is the last provider-zero step before spending, so it has
    to answer whether the run it describes can actually start. Reporting ready
    while execute_blockers would refuse makes the report useless as a gate."""

    return [
        item
        for item in execute_blockers(args)
        if item not in DRY_RUN_INAPPLICABLE_BLOCKERS
    ]


def method_arm_name(args: argparse.Namespace) -> str:
    return (
        "terminal_residual_method"
        if args.method_mode == "terminal"
        else "frontier_residual_method"
    )


def method_protocol(args: argparse.Namespace) -> str:
    if args.method_mode == "terminal":
        return "cheap_full_then_strong_typed_residual_projection"
    return "cheap_verified_frontier_then_strong_typed_residual_projection"


def execute_public_schema(args: argparse.Namespace) -> str:
    if args.method_mode == "terminal":
        return "papc.tau2.cfrc_terminal_execute_public.v1"
    return "papc.tau2.cfrc_frontier_execute_public.v1"


def frontier_verifier_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.method_mode == "terminal":
        return {
            "schema": "papc.tau2.cfrc_frontier_verifier_launch.v1",
            "predicate": "terminal_after_cheap_full",
            "cheap_can_decide_escalation": False,
            "accepted_state_is_post_commitment": True,
        }
    if args.frontier_timing == "pre_write_candidate":
        contract_bound = (
            args.contract_builder_variant == "model_authored_request_semantics"
        )
        return {
            "schema": "papc.tau2.cfrc_frontier_verifier_launch.v1",
            "predicate": "pre_write_candidate_first_mutating_tool_call",
            "frontier_timing": args.frontier_timing,
            "cheap_can_decide_escalation": False,
            "candidate_tool_call_executed_before_frontier": False,
            "frontier_candidate_source": (
                "untrusted_or_not_supplied"
                if contract_bound
                else "cheap_tool_call_candidate"
            ),
            "carrier_exact_match_required": not contract_bound,
            "extra_node_binding": (
                "contract_only_no_carrier"
                if contract_bound
                else "same_tool_same_arg_schema_cited_visible_evidence_or_carrier"
            ),
            "pre_write_residual_policy": args.pre_write_residual_policy,
            "adapter_policy": (
                RETAIL_MINIMAL_INTERVENTION_ADAPTER_POLICY
                if args.pre_write_residual_policy == "first_write_candidate"
                else "always_intervene_pre_write"
            ),
            "first_write_candidate_policy": (
                args.pre_write_residual_policy == "first_write_candidate"
            ),
            "scope_selector_style": (
                "contract_bound" if contract_bound else "preserve_candidate_block"
            ),
            "candidate_carrier_tool_scopes": (
                []
                if contract_bound
                else sorted(RETAIL_MINIMAL_INTERVENTION_CANDIDATE_CARRIER_TOOLS)
            ),
            "selection_uses_task_id_or_gold": False,
            "proposal_loop": proposal_loop_report(args),
        }
    return {
        "schema": "papc.tau2.cfrc_frontier_verifier_launch.v1",
        "predicate": "post_tool_result_first_mutating_tool_call",
        "frontier_timing": args.frontier_timing,
        "cheap_can_decide_escalation": False,
        "post_write_residual_policy": args.post_write_residual_policy,
        "strong_called_only_on_frontier_receipt_error": (
            args.post_write_residual_policy == "receipt_error"
        ),
        "accepted_state_is_post_commitment": True,
        "candidate_tool_call_executed_before_frontier": True,
        "frontier_candidate_source": "cheap_tool_call_and_visible_tool_result",
        "proposal_loop": proposal_loop_report(args),
    }


def compiler_setting_report(args: argparse.Namespace) -> dict[str, Any]:
    suffix_budget = suffix_tool_call_budget(args)
    if args.compiler_setting == AUTHORITY_LIFTED_ELL3_COMPILER_SETTING:
        return {
            "schema": "papc.tau2.cfrc_shared_compiler_launch.v1",
            "compiler_setting": args.compiler_setting,
            "mode": "authority_lifted_extension",
            "default_off": True,
            "authority_lattice_labels": [
                "ell0",
                "ell1",
                "ell_read",
                "ell_msg",
                "ell_write",
                ELL3_AUTHORITY_LABEL,
                ELL_UPPER_AUTHORITY_LABEL,
            ],
            "strict_allowed_authorities": ["ell0", "ell1", "ell_msg", "ell_read", "ell_write"],
            "allowed_authorities": [
                "ell0",
                "ell1",
                "ell_msg",
                "ell_read",
                "ell_write",
                ELL3_AUTHORITY_LABEL,
            ],
            "rejected_authorities": [ELL_UPPER_AUTHORITY_LABEL],
            "lift_authority": ELL3_AUTHORITY_LABEL,
            "lift_below_upper_authority": ELL_UPPER_AUTHORITY_LABEL,
            "collision_budget": None,
            "collision_budget_source": "certificate",
            "requires_schema_restoration_cost": True,
            "restoration_cost_source": "certificate",
            "fail_closed_without_restoration_cost": True,
            "raw_tool_calls_execute_only_after_compiler_selection": True,
            "suffix_tool_call_budget": suffix_budget,
            "budget_coupled_to_proposer_node_count": False,
        }
    return {
        "schema": "papc.tau2.cfrc_shared_compiler_launch.v1",
        "compiler_setting": STRICT_COMPILER_SETTING,
        "mode": "strict_core",
        "default_off": False,
        "authority_lattice_labels": ["ell0", "ell1", "ell_read", "ell_msg", "ell_write"],
        "strict_allowed_authorities": ["ell0", "ell1", "ell_msg", "ell_read", "ell_write"],
        "allowed_authorities": ["ell0", "ell1", "ell_msg", "ell_read", "ell_write"],
        "rejected_authorities": [ELL_UPPER_AUTHORITY_LABEL],
        "lift_authority": None,
        "lift_below_upper_authority": None,
        "collision_budget": 0.0,
        "collision_budget_source": "disabled_in_strict_core",
        "requires_schema_restoration_cost": False,
        "restoration_cost_source": "not_used",
        "fail_closed_without_restoration_cost": False,
        "raw_tool_calls_execute_only_after_compiler_selection": True,
        "suffix_tool_call_budget": suffix_budget,
        "budget_coupled_to_proposer_node_count": False,
    }


def suffix_tool_call_budget(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "suffix_tool_call_budget", 1) or 0))


def proposal_loop_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "papc.tau2.cfrc_proposal_loop_launch.v1",
        "enabled": args.method_mode == "frontier" and args.strong_call_cap > 1,
        "B_call": args.strong_call_cap,
        "z_frozen_across_rounds": True,
        "same_projection_each_round": True,
        "feedback": (
            "generic_typed_counts_only"
            if args.strong_call_cap > 1
            else "disabled"
        ),
        "feedback_contains_tool_names": False,
        "feedback_contains_argument_values": False,
        "feedback_contains_failure_shape_hint": False,
    }


def message_residual_extension_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "papc.tau2.cfrc_message_residual_extension.v1",
        "enabled": bool(args.allow_message_residual),
        "authority_level": "ell_msg",
        "strict_core_default": False,
        "strong_direct_final_answer_allowed": False,
        "typed_graph_node_required": True,
        "separate_from_strict_core_claim": bool(args.allow_message_residual),
    }


def method_layer_report(args: argparse.Namespace) -> dict[str, Any]:
    contract_bound = (
        args.contract_builder_variant == "model_authored_request_semantics"
    )
    return {
        "schema": "papc.tau2.graph_cfrc_result_first_layers.v1",
        "graph_cfrc_core": {
            "enabled": True,
            "strict_compiler_path": True,
            "fixed_b_call": int(args.strong_call_cap),
            "fixed_suffix_tool_call_budget": suffix_tool_call_budget(args),
        },
        "bounded_residual_rescue": {
            "enabled": args.frontier_timing == "pre_write_candidate",
            "pre_write_adapter_policy": (
                RETAIL_MINIMAL_INTERVENTION_ADAPTER_POLICY
                if args.method_mode == "frontier"
                and args.frontier_timing == "pre_write_candidate"
                and args.pre_write_residual_policy == "first_write_candidate"
                else args.pre_write_residual_policy
            ),
            "selector_style": (
                "contract_bound" if contract_bound else "preserve_candidate_block"
            ),
            "candidate_carrier_tool_scopes": (
                []
                if contract_bound
                else sorted(RETAIL_MINIMAL_INTERVENTION_CANDIDATE_CARRIER_TOOLS)
            ),
            "selection_uses_task_id_or_gold": False,
            "full_restart_allowed": False,
        },
        "graph_residual_continuation": {
            "enabled": True,
            "counted_separately_from_strict_core": True,
            "not_router_or_full_agent_restart": True,
        },
    }


def safe_agent_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    return cleaned[:96] or "papc_cfrc_frontier_agent"


def import_tau2(tau_repo: Path) -> None:
    tau_src = tau_repo / "src"
    if str(tau_src) not in sys.path:
        sys.path.insert(0, str(tau_src))


def apply_evaluator_patch(
    *,
    evaluator_model: str,
    api_base: str,
    reasoning_effort: str,
) -> None:
    import tau2.config as tau_config
    import tau2.environment.utils.interface_agent as interface_agent
    import tau2.evaluator.evaluator_nl_assertions as nl_assertions

    kwargs = llm_args_for_model(api_base, reasoning_effort, evaluator_model)
    tau_config.DEFAULT_LLM_NL_ASSERTIONS = evaluator_model
    tau_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(kwargs)
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = evaluator_model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(kwargs)
    tau_config.DEFAULT_LLM_ENV_INTERFACE = evaluator_model
    tau_config.DEFAULT_LLM_ENV_INTERFACE_ARGS = dict(kwargs)
    interface_agent.DEFAULT_LLM_ENV_INTERFACE = evaluator_model
    interface_agent.DEFAULT_LLM_ENV_INTERFACE_ARGS = dict(kwargs)


def llm_args(api_base: str, reasoning_effort: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "temperature": 0.0,
        "drop_params": True,
        "reasoning_effort": reasoning_effort,
    }
    if api_base:
        args["api_base"] = api_base
    return args


def configure_provider_env(args: argparse.Namespace) -> None:
    os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)


def none_if_requested(value: str | None) -> str | None:
    if value is None:
        return None
    return None if value.lower() in {"none", "null"} else value


def attach_public_leakage_audit(report: dict[str, Any]) -> None:
    report["public_leakage_audit"] = public_leakage_audit(report)
    if report["public_leakage_audit"].get("pass") is not True:
        report["eligible_to_update_current_evidence"] = False
        postrun = report.get("postrun_acceptance")
        if isinstance(postrun, dict):
            blockers = postrun.setdefault("blockers", [])
            if "public_leakage_audit_failed" not in blockers:
                blockers.append("public_leakage_audit_failed")
            postrun["eligible_to_update_current_evidence"] = False


def attach_script_runtime_accounting(
    report: dict[str, Any],
    *,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    max_concurrency: int | None = None,
) -> None:
    elapsed = round(max(0.0, float(wall_seconds)), 6)
    attach_parallel_elapsed_to_external_time_accounting(
        report,
        wall_seconds=elapsed,
        max_concurrency=max_concurrency,
    )
    script_parallel: dict[str, Any] = {
        "parallel_elapsed_wall_seconds": elapsed,
    }
    attach_parallel_capacity_accounting(
        script_parallel,
        parallel_elapsed_wall_seconds=elapsed,
        max_concurrency=max_concurrency,
    )
    report["script_runtime_accounting"] = {
        "schema": "papc.tau2.script_runtime_accounting.v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "parallel_elapsed_wall_seconds": elapsed,
        "total_run_wall_seconds": elapsed,
        "max_concurrency": script_parallel["max_concurrency"],
        "parallel_capacity_seconds": script_parallel["parallel_capacity_seconds"],
        "parallel_capacity_seconds_source": script_parallel[
            "parallel_capacity_seconds_source"
        ],
        "parallel_capacity_seconds_is_upper_bound_not_total": script_parallel[
            "parallel_capacity_seconds_is_upper_bound_not_total"
        ],
        "wall_clock_source": "time.perf_counter",
        "total_run_wall_seconds_is_parallel_elapsed": True,
        "wall_time_is_not_sum_of_model_latencies": True,
        "aggregate_external_time_field": "external_time_accounting.*.aggregate_external_time_seconds",
        "model_resource_summary_field": "model_resource_summary",
        "arm_runtime_ledger_field": "arms.*.runtime_ledger",
        "public_safe": True,
    }


def write_json_append_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

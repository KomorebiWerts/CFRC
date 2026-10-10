#!/usr/bin/env python3
"""Source-bound provider-zero gate for bounded tau2 request-contract pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from papc.certspec.tau2.noapi_readiness import (
    public_leakage_audit,
    write_json_append_only,
)
from papc.paths import DATA_ROOT

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "papc.tau2.request_contract_gate.v3"
MAX_DIAGNOSTIC_TASKS = 12
RETAIL_FULL_DENOMINATOR = 114
RETAIL_SPLIT_DENOMINATORS = {"base": 114, "test": 40, "train": 74}
FULL_DENOMINATORS = {"retail": RETAIL_FULL_DENOMINATOR}
DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
RETAIL_EXECUTION_BINDING = {
    "num_trials": 1,
    "max_concurrency": 5,
    "max_steps": 100,
    "cheap_model": "gpt-5.4-mini",
    "cheap_reasoning_effort": "high",
    "strong_model": "gpt-5.5",
    "strong_reasoning_effort": "high",
    "user_model": "gpt-5.5",
    "user_reasoning_effort": "high",
    "evaluator_model": "gpt-5.5",
    "evaluator_reasoning_effort": "high",
    "request_contract_mode": "request_contract",
    "contract_builder_variant": "deterministic_candidate_grounded",
    "deterministic_contract_closure": False,
    "zero_obligation_request_fallback": False,
    "grounded_terminal_literals": False,
    "transcription_is_fallback": False,
    "contract_builder_executor": "successor",
    "contract_builder_attempt_cap": 0,
    "contract_builder_phases": [],
    "contract_builder_success_provider_call_count_required": 0,
    "request_contract_private_replica": True,
    "frontier_timing": "pre_write_candidate",
    "pre_write_residual_policy": "first_write_candidate",
    "strong_call_cap": 2,
    "residual_proposal_call_cap": 2,
    "total_strong_provider_call_cap": 2,
    # One contract per episode: the setting defines a single commitment
    # frontier, and on the official test split the multi-scope path won one
    # row in six where the native agent won two.
    "max_request_contract_calls": 1,
    "request_contract_suffix_cap": 6,
    "suffix_tool_call_budget": 6,
    "sequential_contract_execution": True,
    "ledger_closed_terminal_required": True,
}
EXECUTION_BINDINGS = {"retail": RETAIL_EXECUTION_BINDING}
MODEL_POOL_ARMS = {
    "gpt54mini-gpt55-high": {
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.5",
        "strong_reasoning_effort": "high",
    },
    "gpt56-luna-sol-high": {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
    },
    "deepseek-v4-flash-pro-high": {
        "cheap_model": "openai/deepseek/deepseek-v4-flash",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/deepseek/deepseek-v4-pro",
        "strong_reasoning_effort": "high",
    },
    "gpt54mini-claude-opus5-high": {
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/claude-opus-5",
        "strong_reasoning_effort": "high",
    },
    "gpt54mini-claude-sonnet5-high": {
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/claude-sonnet-5",
        "strong_reasoning_effort": "high",
    },
    "gpt56luna-claude-sonnet5-high": {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/claude-sonnet-5",
        "strong_reasoning_effort": "high",
    },
    "gpt56luna-claude-opus5-high": {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/claude-opus-5",
        "strong_reasoning_effort": "high",
    },
    "gemini35flashhigh-gpt56sol-high": {
        "cheap_model": "openai/gemini-3.5-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
    },
    "gemini35flashhigh-gpt55-high": {
        "cheap_model": "openai/gemini-3.5-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.5",
        "strong_reasoning_effort": "high",
    },
    "gpt54mini-gemini35flashhigh-high": {
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/gemini-3.5-flash-high",
        "strong_reasoning_effort": "high",
    },
    "gpt56luna-gemini37flash-high": {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/gemini-3.7-flash-high",
        "strong_reasoning_effort": "high",
    },
    "gemini36flashhigh-gpt56sol-high": {
        "cheap_model": "openai/gemini-3.6-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
    },
    "gemini35flashhigh-claude-opus5-high": {
        "cheap_model": "openai/gemini-3.5-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "openai/claude-opus-5",
        "strong_reasoning_effort": "high",
    },
    # The first pool whose prefix is a Claude model. The ``openai/`` prefix on
    # the cheap side selects the same CLIProxy/LiteLLM route used by the other
    # tau2 arms; it is only a transport selector, so the native tools payload
    # and the CFRC runner stay unchanged.
    "claudehaiku45-gpt56sol-high": {
        "cheap_model": "openai/claude-haiku-4-5-20251001",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
    },
}
RETAIL_TEST_ACCEPTANCE_BY_ARM = {
    "gpt54mini-gpt55-high": {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    "gpt56-luna-sol-high": {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    "gpt54mini-gemini35flashhigh-high": {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    "gpt56luna-gemini37flash-high": {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    # The prefix and the anchors are those of the luna arm above, so its floor
    # and ceiling carry over unchanged. Nothing is yet measured for this
    # successor, and a threshold invented before the first run would be a
    # target rather than an incumbent.
    # The prefix and its anchors are those of the Sonnet arm, so the floor
    # carries over. Opus 5 is the dearest successor measured on this surface,
    # its own single run costing $25.198050, so the ceiling is raised to leave
    # the first shot bounded rather than refused on price alone.
    "gpt56luna-claude-opus5-high": {
        "reward_floor": 32,
        "cost_ceiling_usd": 12.00,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    "gpt56luna-claude-sonnet5-high": {
        "reward_floor": 32,
        "cost_ceiling_usd": 6.70,
        "quality_reference": {
            "arm": "sol_baseline",
            "reward_sum": 32,
            "official_denominator": 40,
        },
    },
    "gpt54mini-claude-sonnet5-high": {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    "gpt54mini-claude-opus5-high": {
        "reward_floor": 29,
        "cost_ceiling_usd": 5.50,
        "quality_reference": {
            "arm": "frozen_old_pool_cfrc",
            "reward_sum": 29,
            "official_denominator": 40,
        },
    },
    # The native Gemini parent is full40 at 31/40. Its complete cheap-role
    # usage (13,421,151 input; 2,780,087 output tokens) is repriced at Google's
    # standard cache-neutral Gemini 3.5 Flash rates, yielding $45.152510.
    "gemini35flashhigh-gpt56sol-high": {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    "gemini35flashhigh-gpt55-high": {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    "gemini35flashhigh-claude-opus5-high": {
        "reward_floor": 31,
        "cost_ceiling_usd": 45.152510,
        "quality_reference": {
            "arm": "native_gemini35flashhigh_baseline",
            "reward_sum": 31,
            "official_denominator": 40,
        },
    },
    # Native Gemini 3.6 is complete at 34/40. Its cheap-role usage
    # (10,169,095 input; 1,746,903 output tokens) costs $14.177708 at Google's
    # promotional cache-neutral rate in force through 2026-12-31.
    "gemini36flashhigh-gpt56sol-high": {
        "reward_floor": 34,
        "cost_ceiling_usd": 14.177708,
        "quality_reference": {
            "arm": "native_gemini36flashhigh_baseline",
            "reward_sum": 34,
            "official_denominator": 40,
        },
    },
    # The reference is this arm's own cheap prefix measured alone on the same
    # split, 26 of 40: a residual layer that does not beat the prefix it
    # inherits has not earned its successor call. The ceiling is that same
    # baseline's cache-neutral agent-side cost, so the arm must also not cost
    # more than the prefix would have cost doing the whole task by itself.
    "claudehaiku45-gpt56sol-high": {
        "reward_floor": 26,
        "cost_ceiling_usd": 5.491161,
        "quality_reference": {
            "arm": "claude_haiku45_baseline",
            "reward_sum": 26,
            "official_denominator": 40,
        },
    },
}
COMMON_FILES = (
    "src/papc/run.py",
    "src/papc/certspec/tau2/request_obligation_contract.py",
    "scripts/tau2/papc_tau2_request_contract_gate_noapi.py",
)
SURFACE_FILES = {
    "retail": (
        "src/papc/certspec/tau2/cfrc_runner.py",
        "scripts/tau2/papc_tau2_cfrc_clean_run.py",
    ),
}


def parse_task_ids(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not rows:
        raise ValueError("diagnostic_task_ids_empty")
    if len(rows) != len(set(rows)):
        raise ValueError("diagnostic_task_ids_duplicate")
    if len(rows) > MAX_DIAGNOSTIC_TASKS:
        raise ValueError("diagnostic_task_count_exceeds_12")
    return rows


def task_id_sha256s(task_ids: tuple[str, ...]) -> list[str]:
    return sorted(hashlib.sha256(item.encode()).hexdigest() for item in task_ids)


def source_paths(surface: str) -> tuple[str, ...]:
    return tuple(sorted({*COMMON_FILES, *SURFACE_FILES[surface]}))


def source_manifest(surface: str) -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "sha256": hashlib.sha256((REPO / path).read_bytes()).hexdigest(),
        }
        for path in source_paths(surface)
    ]


def source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def benchmark_source_state(tau_repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tau_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tau_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def build_gate(
    *,
    surface: str,
    task_ids: tuple[str, ...],
    full_run: bool = False,
    user_authorized: bool = False,
    tau_repo: Path = DEFAULT_TAU_REPO,
    split: str = "base",
    model_pool_arm: str | None = None,
    contract_builder_variant: str = "deterministic_candidate_grounded",
    deterministic_contract_closure: bool = False,
    zero_obligation_request_fallback: bool = False,
    grounded_terminal_literals: bool = False,
    transcription_is_fallback: bool = False,
    contract_builder_attempt_cap: int = 2,
    contract_builder_executor: str = "successor",
    pre_write_residual_policy: str = "first_write_candidate",
    max_concurrency: int | None = None,
    reward_floor: float | None = None,
    cost_ceiling_usd: float | None = None,
) -> dict[str, Any]:
    source = source_state()
    benchmark_source = benchmark_source_state(tau_repo)
    manifest = source_manifest(surface)
    blockers: list[str] = []
    if full_run and surface not in FULL_DENOMINATORS:
        blockers.append("full_run_scope_unsupported_surface")
    if full_run and surface == "retail" and not user_authorized:
        blockers.append("retail_full_run_requires_explicit_user_authorization")
    if full_run and task_ids:
        blockers.append("full_run_must_use_official_base_selection")
    if not full_run and not task_ids:
        blockers.append("diagnostic_task_ids_empty")
    if not manifest:
        blockers.append("source_manifest_missing")
    if model_pool_arm is not None and surface != "retail":
        blockers.append("named_model_pool_arm_requires_retail")
    if model_pool_arm is not None and model_pool_arm not in MODEL_POOL_ARMS:
        blockers.append("named_model_pool_arm_unknown")
    if contract_builder_variant not in {
        "deterministic_candidate_grounded",
        "model_authored_request_semantics",
    }:
        blockers.append("contract_builder_variant_invalid")
    if (
        surface != "retail"
        and contract_builder_variant != "deterministic_candidate_grounded"
    ):
        blockers.append("contract_builder_variant_requires_retail")
    if int(contract_builder_attempt_cap) not in {1, 2}:
        blockers.append("contract_builder_attempt_cap_invalid")
    if contract_builder_executor not in {"successor", "cheap_prefix_model"}:
        blockers.append("contract_builder_executor_invalid")
    if (
        contract_builder_executor == "cheap_prefix_model"
        and contract_builder_variant != "model_authored_request_semantics"
        and not (
            contract_builder_variant == "deterministic_candidate_grounded"
            and bool(deterministic_contract_closure)
        )
    ):
        blockers.append("contract_builder_executor_requires_model_authored")
    retail_denominator = RETAIL_SPLIT_DENOMINATORS.get(
        split, RETAIL_FULL_DENOMINATOR
    )
    if reward_floor is not None and not (
        0 <= float(reward_floor) <= retail_denominator
    ):
        blockers.append("reward_floor_invalid")
    if cost_ceiling_usd is not None and float(cost_ceiling_usd) <= 0:
        blockers.append("cost_ceiling_invalid")
    if (
        surface == "retail"
        and split == "test"
        and contract_builder_variant == "model_authored_request_semantics"
    ):
        acceptance_profile = RETAIL_TEST_ACCEPTANCE_BY_ARM.get(
            str(model_pool_arm)
        )
        if acceptance_profile is None:
            blockers.append("model_authored_test_model_pool_arm_required")
        if reward_floor is None:
            blockers.append("model_authored_test_reward_floor_required")
        elif acceptance_profile is not None and float(reward_floor) != float(
            acceptance_profile["reward_floor"]
        ):
            blockers.append("model_authored_test_reward_floor_mismatch")
        if cost_ceiling_usd is None:
            blockers.append("model_authored_test_cost_ceiling_required")
        elif acceptance_profile is not None and float(
            cost_ceiling_usd
        ) != float(acceptance_profile["cost_ceiling_usd"]):
            blockers.append("model_authored_test_cost_ceiling_mismatch")
    if max_concurrency is not None and max_concurrency < 1:
        blockers.append("max_concurrency_invalid")
    execution_binding = dict(EXECUTION_BINDINGS.get(surface, {}))
    if model_pool_arm in MODEL_POOL_ARMS:
        execution_binding.update(MODEL_POOL_ARMS[model_pool_arm])
        execution_binding["model_pool_arm"] = model_pool_arm
    if surface == "retail":
        closure_pass_enabled = bool(deterministic_contract_closure) and (
            contract_builder_variant == "deterministic_candidate_grounded"
        )
        zero_obligation_fallback_enabled = bool(
            zero_obligation_request_fallback
        ) and (contract_builder_variant == "deterministic_candidate_grounded")
        effective_builder_attempt_cap = (
            int(contract_builder_attempt_cap)
            if contract_builder_variant == "model_authored_request_semantics"
            else (1 if closure_pass_enabled else 0)
            + (1 if zero_obligation_fallback_enabled else 0)
        )
        residual_proposal_call_cap = int(execution_binding["strong_call_cap"])
        execution_binding.update(
            {
                "contract_builder_variant": contract_builder_variant,
                "deterministic_contract_closure": closure_pass_enabled,
                "zero_obligation_request_fallback": (
                    zero_obligation_fallback_enabled
                ),
                "grounded_terminal_literals": bool(
                    grounded_terminal_literals
                ),
                "transcription_is_fallback": bool(transcription_is_fallback),
                "contract_builder_executor": contract_builder_executor,
                "pre_write_residual_policy": pre_write_residual_policy,
                "contract_builder_attempt_cap": effective_builder_attempt_cap,
                "contract_builder_phases": (
                    (
                        ["request_semantics_closure"]
                        if effective_builder_attempt_cap == 1
                        else [
                            "request_semantics_draft",
                            "residual_closure_revision",
                        ]
                    )
                    if contract_builder_variant == "model_authored_request_semantics"
                    else ["deterministic_residual_closure"]
                    if closure_pass_enabled
                    else []
                ),
                # The fallback raises the cap but is not part of a successful
                # candidate-grounded construction: it fires only where
                # grounding produced nothing, so it is not a required phase.
                "contract_builder_success_provider_call_count_required": (
                    effective_builder_attempt_cap
                    - (1 if zero_obligation_fallback_enabled else 0)
                ),
                "residual_proposal_call_cap": residual_proposal_call_cap,
                "total_strong_provider_call_cap": (
                    effective_builder_attempt_cap + residual_proposal_call_cap
                ),
            }
        )
    if max_concurrency is not None:
        execution_binding["max_concurrency"] = max_concurrency
    diagnostic_authorized = bool(not full_run and task_ids and not blockers)
    full_authorized = bool(
        full_run and surface in FULL_DENOMINATORS and not blockers
    )
    gate: dict[str, Any] = {
        "schema": SCHEMA,
        "surface": f"tau2_{surface}",
        "decision": (
            "authorized_bounded_full_run"
            if full_authorized
            else (
                "authorized_bounded_diagnostic"
                if diagnostic_authorized
                else "blocked"
            )
        ),
        "blockers": blockers,
        "provider_calls": 0,
        "api_spend_performed": False,
        "authorization": {
            "authorization_basis": (
                (
                    "explicit_user_authorization_recorded_by_operator"
                    if user_authorized
                    else "explicit_user_authorization_required"
                )
                if full_run
                else "bounded_mechanism_canary"
            ),
            "authorizes_diagnostic_probe": diagnostic_authorized,
            "authorizes_full_run": full_authorized,
            "max_task_count": (
                (
                    RETAIL_SPLIT_DENOMINATORS.get(split, RETAIL_FULL_DENOMINATOR)
                    if surface == "retail"
                    else FULL_DENOMINATORS.get(surface, 0)
                )
                if full_run
                else len(task_ids)
            ),
            "task_id_sha256s": task_id_sha256s(task_ids),
            "single_trial_only": True,
            "allowed_modes": ["request_contract"],
            "diagnostic_output_ineligible_for_current_evidence": not full_run,
        },
        "execution_binding": execution_binding,
        "acceptance_gate": (
            (
                {
                    "official_denominator": (
                        RETAIL_SPLIT_DENOMINATORS.get(split, RETAIL_FULL_DENOMINATOR)
                    ),
                    "reward_floor": (
                        float(reward_floor)
                        if reward_floor is not None
                        else (92 if split == "base" else 0)
                    ),
                    "agent_side_cache_neutral_cost_ceiling_usd": (
                        float(cost_ceiling_usd)
                        if cost_ceiling_usd is not None
                        else 18.0
                    ),
                    "quality_reference": (
                        dict(
                            RETAIL_TEST_ACCEPTANCE_BY_ARM[
                                str(model_pool_arm)
                            ]["quality_reference"]
                        )
                        if split == "test"
                        and str(model_pool_arm)
                        in RETAIL_TEST_ACCEPTANCE_BY_ARM
                        else (
                            {
                                "arm": "sol_baseline",
                                "reward_sum": 32,
                                "official_denominator": 40,
                            }
                            if split == "test"
                            else None
                        )
                    ),
                    "infrastructure_errors_allowed": 0,
                    "unsafe_checked_actions_allowed": 0,
                    "quality_impossible_stop_policy": {
                        "mode": "operator_checkpoint_interrupt",
                        "automatic_runner_stop": False,
                        "max_in_flight_after_detection": int(
                            execution_binding["max_concurrency"]
                        ),
                    },
                    "quality_below_floor_is_negative_regardless_of_cost": True,
                }
            )
            if full_run
            else {
                "official_denominator": len(task_ids),
                "all_selected_tasks_must_pass": True,
                "unsafe_checked_actions_allowed": 0,
                "infrastructure_errors_allowed": 0,
                "promotion_to_full_requires_fresh_gate": True,
            }
        ),
        "source": {
            "commit": source["commit"],
            "workspace_dirty": source["dirty"],
            "manifest_bound": bool(manifest),
            "implementation_source_manifest": manifest,
        },
        "benchmark_source": {
            "commit": benchmark_source["commit"],
            "workspace_dirty": benchmark_source["dirty"],
        },
        "zero_provider_gate": True,
        "public_safe": True,
    }
    gate["public_leakage_audit"] = public_leakage_audit(gate)
    if gate["public_leakage_audit"]["pass"] is not True:
        gate["decision"] = "blocked"
        gate["blockers"].append("public_leakage_audit_failed")
        gate["authorization"]["authorizes_diagnostic_probe"] = False
        gate["authorization"]["authorizes_full_run"] = False
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=sorted(SURFACE_FILES), required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument(
        "--split",
        choices=sorted(RETAIL_SPLIT_DENOMINATORS),
        default="base",
        help="Retail task split for the full-run scope (official split_tasks.json).",
    )
    parser.add_argument(
        "--user-authorized",
        action="store_true",
        help=(
            "Record that the user explicitly authorized this full run; "
            "required before a retail full-run gate can authorize."
        ),
    )
    parser.add_argument(
        "--model-pool-arm",
        choices=sorted(MODEL_POOL_ARMS),
        default=None,
        help=(
            "Bind only a registered Retail cheap/successor pair; method "
            "configuration remains a separate explicit choice."
        ),
    )
    parser.add_argument(
        "--contract-builder-variant",
        choices=[
            "deterministic_candidate_grounded",
            "model_authored_request_semantics",
        ],
        default="deterministic_candidate_grounded",
        help=(
            "Bind the Retail contract builder explicitly. The default preserves "
            "every registered model arm's frozen behavior."
        ),
    )
    parser.add_argument(
        "--deterministic-contract-closure",
        action="store_true",
        help=(
            "Bind the two-step candidate-grounded builder: deterministic "
            "grounding, then one provider call that closes the grounded ledger "
            "against the whole visible request and returns only the difference."
        ),
    )
    parser.add_argument(
        "--zero-obligation-request-fallback",
        action="store_true",
        help=(
            "Bind the empty-grounding branch of the candidate-grounded "
            "builder: when grounding yields no obligation, re-ground the "
            "contract against the visible request and the accepted receipts "
            "instead of freezing an empty ledger."
        ),
    )
    parser.add_argument(
        "--pre-write-residual-policy",
        choices=("first_write_candidate", "always"),
        default="first_write_candidate",
        help=(
            "Which write candidates open a commitment frontier. "
            "first_write_candidate certifies only the first one in an episode "
            "and leaves every later write to the prefix; always certifies each "
            "of them, which is the same rule Airline already applies under the "
            "name all_write. The default preserves the earlier binding."
        ),
    )
    parser.add_argument(
        "--grounded-terminal-literals",
        action="store_true",
        help=(
            "Bind the grounded terminal-literal filter: a builder-authored "
            "literal survives into the contract only when the visible "
            "conversation, the public policy or a deterministic receipt "
            "already carries it."
        ),
    )
    parser.add_argument(
        "--transcription-is-fallback",
        action="store_true",
        help=(
            "Bind the transcription as the fallback after the successor "
            "proposal instead of the first residual round."
        ),
    )
    parser.add_argument(
        "--contract-builder-attempt-cap",
        type=int,
        choices=[1, 2],
        default=2,
    )
    parser.add_argument(
        "--contract-builder-executor",
        choices=["successor", "cheap_prefix_model"],
        default="successor",
        help=(
            "Which model discharges Reached-State Contract Construction. "
            "cheap_prefix_model keeps the successor on Executable Residual "
            "Completion alone."
        ),
    )
    parser.add_argument("--reward-floor", type=float, default=None)
    parser.add_argument("--cost-ceiling-usd", type=float, default=None)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override only the registered runtime concurrency in the gate.",
    )
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        task_ids = () if args.full_run else parse_task_ids(args.task_ids)
    except ValueError as exc:
        print(json.dumps({"decision": "blocked", "blockers": [str(exc)]}))
        return 2
    gate = build_gate(
        surface=args.surface,
        task_ids=task_ids,
        full_run=args.full_run,
        split=args.split,
        user_authorized=args.user_authorized,
        tau_repo=args.tau_repo,
        model_pool_arm=args.model_pool_arm,
        contract_builder_variant=args.contract_builder_variant,
        deterministic_contract_closure=args.deterministic_contract_closure,
        zero_obligation_request_fallback=(
            args.zero_obligation_request_fallback
        ),
        grounded_terminal_literals=args.grounded_terminal_literals,
        transcription_is_fallback=args.transcription_is_fallback,
        contract_builder_attempt_cap=args.contract_builder_attempt_cap,
        contract_builder_executor=args.contract_builder_executor,
        pre_write_residual_policy=args.pre_write_residual_policy,
        max_concurrency=args.max_concurrency,
        reward_floor=args.reward_floor,
        cost_ceiling_usd=args.cost_ceiling_usd,
    )
    write_json_append_only(args.output, gate)
    print(
        json.dumps(
            {
                "decision": gate["decision"],
                "provider_calls": 0,
                "surface": gate["surface"],
                "authorized_task_count": len(task_ids),
                "authorizes_full_run": gate["authorization"][
                    "authorizes_full_run"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if gate["decision"]
        in {"authorized_bounded_diagnostic", "authorized_bounded_full_run"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

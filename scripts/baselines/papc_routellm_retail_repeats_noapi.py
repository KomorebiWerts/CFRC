#!/usr/bin/env python3
"""Freeze and reduce the three-run RouteLLM-BERT Retail estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_GATE = "papc.routellm.retail.repeat_protocol.v1"
SCHEMA_RESULT = "papc.routellm.retail.repeat_aggregate.v1"
DENOMINATOR = 40
STRONG_CENTER_PASSED = 30.0
STRONG_ENVELOPE_HALF_WIDTH = 1.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"not_a_mapping:{path}")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o644), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_result(payload: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "schema": "papc.table1.result_receipt.v2",
        "result_status": "completed",
        "surface": "tau2_retail",
        "row_id": "routellm_task",
        "attempted_count": DENOMINATOR,
        "scored_count": DENOMINATOR,
        "case_receipt_count": DENOMINATOR,
        "replicates_per_task": 1,
        "infra_error_count": 0,
        "registry_order_matches_test_manifest": True,
        "aggregation_scope": "single_full_official_test",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{label}_{field}_mismatch")
    provider = payload.get("agent_side_provider_api")
    local = payload.get("local_router")
    if not isinstance(provider, Mapping) or not isinstance(local, Mapping):
        raise ValueError(f"{label}_accounting_missing")
    if (
        provider.get("usage_complete") is not True
        or provider.get("retry_count") != 0
        or local.get("decision_count") != DENOMINATOR
    ):
        raise ValueError(f"{label}_accounting_incomplete")
    score = payload.get("official_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{label}_score_invalid")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError(f"{label}_score_out_of_range")


def _binding(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: payload[key]
        for key in (
            "method_artifact_sha256",
            "calibration_sha256",
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "trial_policy_sha256",
            "runtime_sha256",
            "scorer_sha256",
            "cheap_model",
            "cheap_reasoning_effort",
            "strong_model",
            "strong_reasoning_effort",
            "official_metric",
            "official_scorer",
        )
    }


def _route_counts(path: Path, *, label: str) -> Mapping[str, int]:
    payload = _load(path)
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, Sequence) or len(outcomes) != DENOMINATOR:
        raise ValueError(f"{label}_private_outcomes_invalid")
    counts = {"cheap": 0, "strong": 0}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError(f"{label}_private_outcome_invalid")
        roles = outcome.get("selected_roles")
        if roles == ["cheap"]:
            counts["cheap"] += 1
        elif roles == ["strong"]:
            counts["strong"] += 1
        else:
            raise ValueError(f"{label}_selected_role_invalid")
    if counts["cheap"] + counts["strong"] != DENOMINATOR:
        raise ValueError(f"{label}_route_count_invalid")
    return counts


def build_gate(args: argparse.Namespace) -> Mapping[str, Any]:
    primary = _load(args.primary_result)
    _validate_result(primary, label="primary")
    repeat_paths = [path.resolve() for path in args.repeat_result]
    private_paths = [path.resolve() for path in args.private_run_output]
    if len(repeat_paths) != 2 or len(set(repeat_paths)) != 2:
        raise ValueError("exactly_two_unique_repeat_results_required")
    existing = [str(path) for path in repeat_paths if path.exists()]
    if existing:
        raise FileExistsError(f"repeat_result_already_exists:{existing}")
    if len(private_paths) != 3 or len(set(private_paths)) != 3:
        raise ValueError("exactly_three_unique_private_outputs_required")
    if not private_paths[0].is_file():
        raise FileNotFoundError("primary_private_output_missing")
    if any(path.exists() for path in private_paths[1:]):
        raise FileExistsError("repeat_private_output_already_exists")
    primary_routes = _route_counts(private_paths[0], label="primary")
    return {
        "schema": SCHEMA_GATE,
        "decision": "ready_for_two_prespecified_repeats",
        "provider_calls": 0,
        "api_spend_performed": False,
        "public_safe": True,
        "method": "RouteLLM-BERT",
        "surface": "tau2_retail",
        "test_denominator_per_run": DENOMINATOR,
        "policy_binding": _binding(primary),
        "strong_quality_protocol": {
            "center_passed": STRONG_CENTER_PASSED,
            "envelope_half_width_passed": STRONG_ENVELOPE_HALF_WIDTH,
            "lower_passed": STRONG_CENTER_PASSED - STRONG_ENVELOPE_HALF_WIDTH,
            "upper_passed": STRONG_CENTER_PASSED + STRONG_ENVELOPE_HALF_WIDTH,
            "source": "user_frozen_before_repeat_2_and_repeat_3_on_2026-08-02",
            "test_blind_role": "evaluation_only_no_threshold_or_policy_change",
        },
        "aggregation_rule": {
            "run_count": 3,
            "quality": "arithmetic_mean_of_three_raw_full_test_pass_counts",
            "agent_side_cost": "arithmetic_mean_of_three_raw_cache_neutral_costs",
            "provider_calls": "arithmetic_mean_of_three_raw_agent_side_calls",
            "route_counts": "arithmetic_mean_of_three_raw_route_counts",
            "primary_run_fixed_before_protocol": True,
            "repeat_2_and_3_frozen_before_execution": True,
            "best_run_selection_forbidden": True,
            "strong_capping_forbidden": True,
            "registered_quality_corrections_forbidden": True,
            "cross_run_row_splicing_forbidden": True,
        },
        "runs": [
            {
                "ordinal": 1,
                "result": str(args.primary_result.resolve()),
                "private_run_output": str(private_paths[0]),
                "sha256": _sha256(args.primary_result),
                "route_counts": primary_routes,
                "status": "completed_before_repeat_protocol",
            },
            *[
                {
                    "ordinal": ordinal,
                    "result": str(path),
                    "private_run_output": str(private_path),
                    "status": "prespecified_not_started",
                }
                for ordinal, (path, private_path) in enumerate(
                    zip(repeat_paths, private_paths[1:], strict=True), start=2
                )
            ],
        ],
        "unresolved_infrastructure_failures": 0,
    }


def build_aggregate(args: argparse.Namespace) -> Mapping[str, Any]:
    gate = _load(args.gate_json)
    if gate.get("schema") != SCHEMA_GATE:
        raise ValueError("repeat_gate_schema_invalid")
    run_records = gate.get("runs")
    if not isinstance(run_records, Sequence) or len(run_records) != 3:
        raise ValueError("repeat_gate_runs_invalid")
    paths = [Path(str(item["result"])) for item in run_records]
    private_paths = [Path(str(item["private_run_output"])) for item in run_records]
    payloads = [_load(path) for path in paths]
    for ordinal, payload in enumerate(payloads, start=1):
        _validate_result(payload, label=f"run_{ordinal}")
    binding = _binding(payloads[0])
    if any(_binding(payload) != binding for payload in payloads[1:]):
        raise ValueError("repeat_policy_binding_drift")
    pass_counts = [float(item["official_score"]) * DENOMINATOR for item in payloads]
    costs = [
        float(item["agent_side_provider_api"]["cache_neutral_cost_usd"])
        for item in payloads
    ]
    calls = [
        int(item["agent_side_provider_api"]["provider_calls"])
        for item in payloads
    ]
    route_counts = [
        _route_counts(path, label=f"run_{ordinal}")
        for ordinal, path in enumerate(private_paths, start=1)
    ]
    mean_passed = sum(pass_counts) / 3.0
    run_config_sha256s = [str(item["run_config_sha256"]) for item in payloads]
    if any(
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in run_config_sha256s
    ):
        raise ValueError("repeat_run_config_sha256_invalid")
    lower = STRONG_CENTER_PASSED - STRONG_ENVELOPE_HALF_WIDTH
    upper = STRONG_CENTER_PASSED + STRONG_ENVELOPE_HALF_WIDTH
    return {
        "schema": SCHEMA_RESULT,
        "decision": "completed_prespecified_three_run_center_estimate",
        "provider_calls_added_by_reducer": 0,
        "api_spend_performed_by_reducer": False,
        "public_safe": True,
        "surface": "tau2_retail",
        "method": "RouteLLM-BERT",
        "denominator_per_run": DENOMINATOR,
        "run_count": 3,
        "raw_runs": [
            {
                "ordinal": ordinal,
                "passed": passed,
                "cache_neutral_cost_usd": cost,
                "agent_side_provider_calls": call_count,
                "route_counts": routes,
                "run_config_sha256": run_config_sha256,
                "result_sha256": _sha256(path),
            }
            for ordinal, (
                path,
                passed,
                cost,
                call_count,
                routes,
                run_config_sha256,
            ) in enumerate(
                zip(
                    paths,
                    pass_counts,
                    costs,
                    calls,
                    route_counts,
                    run_config_sha256s,
                    strict=True,
                ),
                start=1,
            )
        ],
        "center_estimate": {
            "passed": mean_passed,
            "accuracy": mean_passed / DENOMINATOR,
            "cache_neutral_cost_usd": sum(costs) / 3.0,
            "agent_side_provider_calls": sum(calls) / 3.0,
            "route_counts": {
                "cheap": sum(item["cheap"] for item in route_counts) / 3.0,
                "strong": sum(item["strong"] for item in route_counts) / 3.0,
            },
        },
        "strong_envelope": {
            "center_passed": STRONG_CENTER_PASSED,
            "lower_passed": lower,
            "upper_passed": upper,
            "center_estimate_in_envelope": lower <= mean_passed <= upper,
        },
        "quality_capping_applied": False,
        "registered_quality_corrections_applied": False,
        "cross_run_row_splicing_applied": False,
        "unresolved_infrastructure_failures": 0,
        "policy_binding": binding,
        "gate_sha256": _sha256(args.gate_json),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "aggregate"), required=True)
    parser.add_argument("--primary-result", type=Path)
    parser.add_argument("--repeat-result", type=Path, action="append", default=[])
    parser.add_argument(
        "--private-run-output", type=Path, action="append", default=[]
    )
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "gate":
        if args.primary_result is None or args.gate_json is not None:
            parser.error("gate requires --primary-result and forbids --gate-json")
    elif (
        args.gate_json is None
        or args.primary_result is not None
        or args.repeat_result
        or args.private_run_output
    ):
        parser.error("aggregate requires only --gate-json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_gate(args) if args.mode == "gate" else build_aggregate(args)
    _write_new(args.output_json, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

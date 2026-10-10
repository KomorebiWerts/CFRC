#!/usr/bin/env python3
"""Freeze and replay Retail-trained FrugalGPT on paired Airline base50 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.frugalgpt_upstream import (  # noqa: E402
    FRUGALGPT_UPSTREAM_REVISION,
    FrugalGPTDistilBertScorer,
    FrugalGPTScorerArtifact,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    STRONG_MODEL,
    CheapCompletionRouterView,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256,
    validate_router_runtime_environment,
)
from papc.paths import DATA_ROOT

EXPECTED_COUNT = 50
STRONG_BUDGET_FRACTION = 0.9
STRONG_COUNT = math.floor(EXPECTED_COUNT * STRONG_BUDGET_FRACTION)
PRIVATE_POLICY_SCHEMA = "papc.frugalgpt.airline_base50.ood_policy.private.v2"
PUBLIC_POLICY_SCHEMA = "papc.frugalgpt.airline_base50.ood_policy.v2"
RESULT_SCHEMA = "papc.frugalgpt.airline_base50.ood_replay.v2"

DEFAULT_ARTIFACT = Path(f"{DATA_ROOT}/papc/frugalgpt_retail_v1/artifact.json")
DEFAULT_CHECKPOINT = Path(f"{DATA_ROOT}/papc/frugalgpt_retail_v1/checkpoint")
DEFAULT_CHEAP_RAW = Path(
    f"{DATA_ROOT}/benchmarks/tau_bench/repo/data/simulations/"
    "tau2_airline_official_cheap_base50_gpt54mini_high_mc5_"
    "20260701_codex_v1/results.json"
)
DEFAULT_STRONG_RAW = Path(
    f"{DATA_ROOT}/benchmarks/tau_bench/repo/data/simulations/"
    "tau2_airline_official_strong_base50_gpt55_high_mc5_"
    "20260701_codex_v1/results.json"
)
DEFAULT_CHEAP_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/tau2_airline_official_cheap_base50_gpt54mini_high_mc5_"
    "20260701_codex_v1_public.json"
)
DEFAULT_STRONG_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/tau2_airline_official_strong_base50_gpt55_high_mc5_"
    "20260701_codex_v1_public.json"
)
DEFAULT_JOIN_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/tau2_airline_official_baseline_join_base50_mc5_"
    "20260701_codex_v1_public.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--artifact-json", type=Path, default=DEFAULT_ARTIFACT)
    freeze.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    freeze.add_argument("--cheap-raw-json", type=Path, default=DEFAULT_CHEAP_RAW)
    freeze.add_argument("--private-output-json", type=Path, required=True)
    freeze.add_argument("--public-output-json", type=Path, required=True)

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--private-policy-json", type=Path, required=True)
    reduce.add_argument("--public-policy-json", type=Path, required=True)
    reduce.add_argument("--prior-public-policy-json", type=Path)
    reduce.add_argument("--cheap-raw-json", type=Path, default=DEFAULT_CHEAP_RAW)
    reduce.add_argument("--strong-raw-json", type=Path, default=DEFAULT_STRONG_RAW)
    reduce.add_argument("--cheap-public-json", type=Path, default=DEFAULT_CHEAP_PUBLIC)
    reduce.add_argument(
        "--strong-public-json", type=Path, default=DEFAULT_STRONG_PUBLIC
    )
    reduce.add_argument("--join-public-json", type=Path, default=DEFAULT_JOIN_PUBLIC)
    reduce.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "freeze":
        private, public = _freeze(args)
        _write_append_only(args.private_output_json, private, mode=0o600)
        _write_append_only(args.public_output_json, public, mode=0o644)
        print(
            json.dumps(
                {
                    "decision": public["decision"],
                    "cheap": public["endpoint_selection"]["cheap"],
                    "strong": public["endpoint_selection"]["strong"],
                    "policy_sha256": public["policy_sha256"],
                    "provider_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    result = _reduce(args)
    _write_append_only(args.output_json, result, mode=0o644)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "accuracy": result["official_outcome"]["accuracy"],
                "cost_usd": result["agent_side_accounting"]["cost_usd"],
                "provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _freeze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_sha256 = validate_router_runtime_environment()
    if runtime_sha256 != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("frugalgpt_runtime_environment_mismatch")
    artifact = FrugalGPTScorerArtifact.from_public_record(
        _read_object(args.artifact_json)
    )
    if (
        artifact.surface != "tau2_retail"
        or artifact.upstream_revision != FRUGALGPT_UPSTREAM_REVISION
        or artifact.cheap_model != CHEAP_MODEL
        or artifact.strong_model != STRONG_MODEL
    ):
        raise ValueError("frugalgpt_retail_artifact_binding_mismatch")
    scorer = FrugalGPTDistilBertScorer.load(
        artifact=artifact, checkpoint_dir=args.checkpoint_dir
    )
    source = _read_object(args.cheap_raw_json)
    simulations = _outcome_blind_simulations(source, expected_model=CHEAP_MODEL)
    scored: list[dict[str, Any]] = []
    projection_bindings: list[dict[str, Any]] = []
    for ordinal, simulation in enumerate(simulations):
        public_episode = _public_history(simulation["messages"])
        original_request = _first_user_message(public_episode)
        generated_answer = json.dumps(
            tuple(public_episode),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        score = scorer(
            CheapCompletionRouterView(
                original_request=original_request,
                cheap_generated_answer=generated_answer,
            )
        )
        task_key_sha256 = _text_sha256(simulation["task_id"])
        answer_sha256 = _text_sha256(generated_answer)
        scored.append(
            {
                "public_ordinal": ordinal,
                "task_id": simulation["task_id"],
                "task_key_sha256": task_key_sha256,
                "original_request_sha256": _text_sha256(original_request),
                "cheap_generated_answer_sha256": answer_sha256,
                "predicted_cheap_quality": score,
            }
        )
        projection_bindings.append(
            {
                "public_ordinal": ordinal,
                "task_key_sha256": task_key_sha256,
                "cheap_generated_answer_sha256": answer_sha256,
            }
        )
    ranked = sorted(
        scored,
        key=lambda row: (
            row["predicted_cheap_quality"],
            row["original_request_sha256"],
            row["cheap_generated_answer_sha256"],
        ),
    )
    strong_ordinals = {
        row["public_ordinal"] for row in ranked[:STRONG_COUNT]
    }
    routes = [
        {
            **row,
            "selected_endpoint": (
                "strong" if row["public_ordinal"] in strong_ordinals else "cheap"
            ),
        }
        for row in scored
    ]
    public_routes = [
        {
            key: value
            for key, value in row.items()
            if key not in {"task_id", "task_key_sha256"}
        }
        for row in routes
    ]
    common = {
        "method": "FrugalGPT (Retail scorer, 90% strong budget)",
        "surface": "tau2_airline",
        "split": "base50_all_final_test",
        "official_test_count": EXPECTED_COUNT,
        "endpoint_selection": {
            "cheap": EXPECTED_COUNT - STRONG_COUNT,
            "strong": STRONG_COUNT,
        },
        "router": {
            "upstream_revision": FRUGALGPT_UPSTREAM_REVISION,
            "source_surface": "tau2_retail",
            "artifact_sha256": artifact.sha256,
            "artifact_file_sha256": _file_sha256(args.artifact_json),
            "checkpoint_bundle_sha256": artifact.checkpoint_bundle_sha256,
            "runtime_environment_sha256": runtime_sha256,
            "input": "original request plus completed cheap public episode",
            "ranking": "ascending predicted cheap quality",
            "strong_budget_fraction": STRONG_BUDGET_FRACTION,
            "strong_count_rule": "floor(N * strong_budget_fraction)",
            "tie_breaker": (
                "original_request_sha256 then cheap_generated_answer_sha256 ascending"
            ),
            "task_id_used_by_router": False,
            "retail_accept_threshold_audit_only": (
                artifact.operating_point.accept_threshold
            ),
        },
        "source_binding": {
            "cheap_raw_file_sha256": _file_sha256(args.cheap_raw_json),
            "cheap_public_projection_bundle_sha256": _digest(
                projection_bindings
            ),
            "cheap_projection": (
                "tau2 public role/content/tool-call projection; raw_data, reward_info, reviews, costs, and task payloads excluded"
            ),
            "endpoint_outcome_fields_accessed_by_freeze": False,
            "strong_endpoint_accessed_by_freeze": False,
            "task_id_used_for_join_only": True,
            "compliance_correction": (
                "v2 removes task-id-derived routing; all 50 predicted scores are unique"
            ),
        },
        "provider_calls": 0,
        "secret_reads": 0,
    }
    private_unsigned = {
        "schema": PRIVATE_POLICY_SCHEMA,
        "decision": "frozen_without_endpoint_outcome_access",
        **common,
        "routes": routes,
    }
    private = {**private_unsigned, "policy_sha256": _digest(private_unsigned)}
    public_unsigned = {
        "schema": PUBLIC_POLICY_SCHEMA,
        "decision": "frozen_without_endpoint_outcome_access",
        **common,
        "routes": public_routes,
        "claim_boundary": [
            "Airline base50 is treated as one final-test set; no Airline outcome is used for training or calibration.",
            "The 90% strong-restart budget is frozen from the Retail scorer ranking before endpoint outcomes are reduced.",
            "All tasks incur the completed cheap episode; escalations select the paired pristine Strong endpoint from the same July 1 run family.",
        ],
    }
    public = {**public_unsigned, "policy_sha256": _digest(public_unsigned)}
    return private, public


def _outcome_blind_simulations(
    source: Mapping[str, Any], *, expected_model: str
) -> list[dict[str, Any]]:
    info = source.get("info")
    agent_info = info.get("agent_info") if isinstance(info, Mapping) else None
    simulations = source.get("simulations")
    if (
        not isinstance(agent_info, Mapping)
        or agent_info.get("llm") != expected_model
        or not isinstance(simulations, list)
        or len(simulations) != EXPECTED_COUNT
    ):
        raise ValueError("airline_cheap_source_identity_invalid")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in simulations:
        if not isinstance(raw, Mapping):
            raise ValueError("airline_cheap_simulation_invalid")
        task_id = raw.get("task_id")
        messages = raw.get("messages")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or not isinstance(messages, list)
            or not messages
        ):
            raise ValueError("airline_cheap_simulation_binding_invalid")
        seen.add(task_id)
        # Deliberately copy only policy-visible fields.  In particular, this
        # function never accesses reward_info, reviews, task payloads, or costs.
        rows.append({"task_id": task_id, "messages": messages})
    return rows


def _public_history(messages: Sequence[Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            raise ValueError("airline_message_invalid")
        role = raw.get("role")
        if role == "system":
            continue
        if role in {"user", "assistant"}:
            item: dict[str, Any] = {"role": role}
            content = raw.get("content")
            if isinstance(content, str) and content:
                item["content"] = content
            calls = raw.get("tool_calls")
            if calls:
                if not isinstance(calls, list):
                    raise ValueError("airline_tool_calls_invalid")
                item["tool_calls"] = [
                    {
                        "id": str(call.get("id", "")),
                        "name": str(call.get("name", "")),
                        "arguments": dict(call.get("arguments", {})),
                    }
                    for call in calls
                    if isinstance(call, Mapping)
                    and isinstance(call.get("arguments"), Mapping)
                ]
            projected.append(item)
        elif role == "tool":
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": str(raw.get("id", "")),
                    "output": str(raw.get("content", "") or ""),
                    "status": "error" if bool(raw.get("error")) else "completed",
                }
            )
        else:
            raise ValueError("airline_message_role_invalid")
    if not projected:
        raise ValueError("airline_public_history_empty")
    return projected


def _first_user_message(history: Sequence[Mapping[str, Any]]) -> str:
    for message in history:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("airline_initial_user_message_missing")


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    private = _read_object(args.private_policy_json)
    public = _read_object(args.public_policy_json)
    _validate_policies(private, public)
    prior_policy_sha256 = None
    if args.prior_public_policy_json is not None:
        prior = _read_object(args.prior_public_policy_json)
        _validate_decision_manifest_equal(prior, public)
        prior_policy_sha256 = _file_sha256(args.prior_public_policy_json)
    cheap_source = _read_object(args.cheap_raw_json)
    strong_source = _read_object(args.strong_raw_json)
    cheap_public = _read_object(args.cheap_public_json)
    strong_public = _read_object(args.strong_public_json)
    join_public = _read_object(args.join_public_json)
    cheap = _endpoint_rows(cheap_source, endpoint="cheap")
    strong = _endpoint_rows(strong_source, endpoint="strong")
    _validate_endpoint_pair(cheap_source, strong_source, cheap, strong)
    _validate_public_endpoint(cheap_public, endpoint="cheap", rows=cheap)
    _validate_public_endpoint(strong_public, endpoint="strong", rows=strong)
    if (
        join_public.get("domain") != "airline"
        or join_public.get("num_trials") != 1
        or join_public.get("reducer_provider_calls") != 0
    ):
        raise ValueError("airline_join_source_invalid")

    routes = private.get("routes")
    if not isinstance(routes, list) or len(routes) != EXPECTED_COUNT:
        raise ValueError("airline_policy_routes_invalid")
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    reward_sum = 0.0
    result_rows: list[dict[str, Any]] = []
    for route in routes:
        ordinal = route.get("public_ordinal")
        endpoint = route.get("selected_endpoint")
        task_id = route.get("task_id")
        if (
            not isinstance(ordinal, int)
            or endpoint not in {"cheap", "strong"}
            or task_id not in cheap
            or task_id not in strong
        ):
            raise ValueError("airline_policy_route_invalid")
        cheap_row = cheap[task_id]
        selected = cheap_row if endpoint == "cheap" else strong[task_id]
        for row in (cheap_row,) if endpoint == "cheap" else (
            cheap_row,
            strong[task_id],
        ):
            accounting = row["accounting"]
            for key in ("calls", "input_tokens", "output_tokens"):
                totals[key] += accounting[key]
            totals["cost_usd"] += accounting["cost_usd"]
        reward = selected["reward"]
        reward_sum += reward
        result_rows.append(
            {
                "public_ordinal": ordinal,
                "original_request_sha256": route["original_request_sha256"],
                "selected_endpoint": endpoint,
                "reward": reward,
            }
        )
    result = {
        "schema": RESULT_SCHEMA,
        "decision": "observed_frozen_ood_replay_complete",
        "method": private["method"],
        "surface": "tau2_airline",
        "split": "base50_all_final_test",
        "endpoint_selection": private["endpoint_selection"],
        "official_outcome": {
            "metric": "official binary reward",
            "passed": int(reward_sum),
            "denominator": EXPECTED_COUNT,
            "accuracy": reward_sum / EXPECTED_COUNT,
        },
        "agent_side_accounting": {
            **totals,
            "cost_usd": round(totals["cost_usd"], 12),
            "all_cheap_episodes_included": True,
            "selected_strong_restarts_included": True,
            "harness_and_user_simulator_excluded": True,
            "pricing": "frozen cache-neutral",
        },
        "unresolved_infrastructure_failures": 0,
        "provider_calls_added_by_replay": 0,
        "source_binding": {
            "private_policy_sha256": _file_sha256(args.private_policy_json),
            "public_policy_sha256": _file_sha256(args.public_policy_json),
            "cheap_raw_sha256": _file_sha256(args.cheap_raw_json),
            "strong_raw_sha256": _file_sha256(args.strong_raw_json),
            "cheap_public_sha256": _file_sha256(args.cheap_public_json),
            "strong_public_sha256": _file_sha256(args.strong_public_json),
            "join_public_sha256": _file_sha256(args.join_public_json),
            "same_july_1_base50_run_family": True,
            "cross_run_cost_stitching": False,
            "prior_v1_public_policy_sha256": prior_policy_sha256,
            "decision_manifest_equal_to_prior_v1": (
                prior_policy_sha256 is not None
            ),
        },
        "rows": sorted(result_rows, key=lambda row: row["public_ordinal"]),
    }
    return {**result, "result_sha256": _digest(result)}


def _endpoint_rows(
    source: Mapping[str, Any], *, endpoint: str
) -> dict[str, dict[str, Any]]:
    expected_model = CHEAP_MODEL if endpoint == "cheap" else STRONG_MODEL
    info = source.get("info")
    agent_info = info.get("agent_info") if isinstance(info, Mapping) else None
    simulations = source.get("simulations")
    if (
        not isinstance(agent_info, Mapping)
        or agent_info.get("llm") != expected_model
        or not isinstance(simulations, list)
        or len(simulations) != EXPECTED_COUNT
    ):
        raise ValueError(f"airline_{endpoint}_source_invalid")
    rows: dict[str, dict[str, Any]] = {}
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            raise ValueError(f"airline_{endpoint}_simulation_invalid")
        task_id = simulation.get("task_id")
        reward_info = simulation.get("reward_info")
        messages = simulation.get("messages")
        if (
            not isinstance(task_id, str)
            or task_id in rows
            or not isinstance(reward_info, Mapping)
            or not isinstance(messages, list)
        ):
            raise ValueError(f"airline_{endpoint}_simulation_invalid")
        reward = _unit_float(reward_info.get("reward"), "reward")
        rows[task_id] = {
            "reward": reward,
            "accounting": _agent_accounting(messages, endpoint=endpoint),
            "opening_sha256": _text_sha256(
                _first_user_message(_public_history(messages))
            ),
        }
    return rows


def _agent_accounting(
    messages: Sequence[Any], *, endpoint: str
) -> dict[str, Any]:
    calls = input_tokens = output_tokens = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if usage is None:
            continue
        if not isinstance(usage, Mapping):
            raise ValueError(f"airline_{endpoint}_usage_invalid")
        input_tokens += _nonnegative_int(usage.get("prompt_tokens"), "input_tokens")
        output_tokens += _nonnegative_int(
            usage.get("completion_tokens"), "output_tokens"
        )
        calls += 1
    model = CHEAP_MODEL if endpoint == "cheap" else STRONG_MODEL
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    cost = (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000.0
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


def _validate_endpoint_pair(
    cheap_source: Mapping[str, Any],
    strong_source: Mapping[str, Any],
    cheap: Mapping[str, Any],
    strong: Mapping[str, Any],
) -> None:
    if set(cheap) != set(strong) or len(cheap) != EXPECTED_COUNT:
        raise ValueError("airline_endpoint_membership_mismatch")
    if cheap_source.get("tasks") != strong_source.get("tasks"):
        raise ValueError("airline_endpoint_task_payload_mismatch")


def _validate_public_endpoint(
    report: Mapping[str, Any],
    *,
    endpoint: str,
    rows: Mapping[str, Mapping[str, Any]],
) -> None:
    arm = "cheap_full" if endpoint == "cheap" else "strong_full"
    payload = report.get("arms", {}).get(arm)
    if not isinstance(payload, Mapping):
        raise ValueError(f"airline_{endpoint}_public_arm_missing")
    per_role = payload.get("runtime_ledger", {}).get("per_role", {}).get(endpoint)
    aggregate = {
        "calls": sum(row["accounting"]["calls"] for row in rows.values()),
        "input_tokens": sum(
            row["accounting"]["input_tokens"] for row in rows.values()
        ),
        "output_tokens": sum(
            row["accounting"]["output_tokens"] for row in rows.values()
        ),
        "cost_usd": sum(row["accounting"]["cost_usd"] for row in rows.values()),
        "reward_sum": sum(row["reward"] for row in rows.values()),
    }
    if (
        payload.get("task_count") != EXPECTED_COUNT
        or payload.get("infrastructure_excluded_count") != 0
        or payload.get("reward_evaluable_count") != EXPECTED_COUNT
        or payload.get("reward_sum") != aggregate["reward_sum"]
        or not isinstance(per_role, Mapping)
        or per_role.get("calls") != aggregate["calls"]
        or per_role.get("billable_input_tokens") != aggregate["input_tokens"]
        or per_role.get("output_tokens") != aggregate["output_tokens"]
        or not math.isclose(
            float(per_role.get("cost_usd")),
            aggregate["cost_usd"],
            abs_tol=1e-6,
        )
    ):
        raise ValueError(f"airline_{endpoint}_public_accounting_mismatch")


def _validate_policies(
    private: Mapping[str, Any], public: Mapping[str, Any]
) -> None:
    if (
        private.get("schema") != PRIVATE_POLICY_SCHEMA
        or public.get("schema") != PUBLIC_POLICY_SCHEMA
        or private.get("decision") != "frozen_without_endpoint_outcome_access"
        or public.get("decision") != "frozen_without_endpoint_outcome_access"
        or private.get("endpoint_selection")
        != {"cheap": EXPECTED_COUNT - STRONG_COUNT, "strong": STRONG_COUNT}
        or public.get("endpoint_selection") != private.get("endpoint_selection")
    ):
        raise ValueError("airline_policy_binding_invalid")
    for record in (private, public):
        unsigned = dict(record)
        claimed = unsigned.pop("policy_sha256", None)
        if claimed != _digest(unsigned):
            raise ValueError("airline_policy_sha256_invalid")
    private_routes = private.get("routes")
    public_routes = public.get("routes")
    if not isinstance(private_routes, list) or not isinstance(public_routes, list):
        raise ValueError("airline_policy_routes_invalid")
    projected = [
        {
            key: value
            for key, value in row.items()
            if key not in {"task_id", "task_key_sha256"}
        }
        for row in private_routes
        if isinstance(row, Mapping)
    ]
    if projected != public_routes:
        raise ValueError("airline_public_private_policy_mismatch")


def _validate_decision_manifest_equal(
    prior: Mapping[str, Any], corrected: Mapping[str, Any]
) -> None:
    prior_routes = prior.get("routes")
    corrected_routes = corrected.get("routes")
    if not isinstance(prior_routes, list) or not isinstance(corrected_routes, list):
        raise ValueError("airline_policy_decision_manifest_missing")
    def projection(rows: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "public_ordinal": row.get("public_ordinal"),
                "selected_endpoint": row.get("selected_endpoint"),
            }
            for row in rows
            if isinstance(row, Mapping)
        ]
    if (
        prior.get("endpoint_selection") != corrected.get("endpoint_selection")
        or projection(prior_routes) != projection(corrected_routes)
    ):
        raise ValueError("airline_compliance_correction_changed_decisions")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"json_object_required:{path}")
    return value


def _write_append_only(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, mode), "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _unit_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"airline_{name}_invalid")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"airline_{name}_invalid")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

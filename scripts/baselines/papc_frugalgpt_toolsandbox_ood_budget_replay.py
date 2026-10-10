#!/usr/bin/env python3
"""Freeze and replay the Retail-trained FrugalGPT scorer on ToolSandbox."""

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

EXPECTED_COUNT = 129
STRONG_BUDGET_FRACTION = 0.9
STRONG_COUNT = math.floor(EXPECTED_COUNT * STRONG_BUDGET_FRACTION)
REQUEST_MANIFEST_SCHEMA = "papc.routellm.toolsandbox.request_manifest.private.v1"
PRIVATE_POLICY_SCHEMA = "papc.frugalgpt.toolsandbox.ood_policy.private.v2"
PUBLIC_POLICY_SCHEMA = "papc.frugalgpt.toolsandbox.ood_policy.v2"
RESULT_SCHEMA = "papc.frugalgpt.toolsandbox.ood_replay.v2"

DEFAULT_ARTIFACT = Path(f"{DATA_ROOT}/papc/frugalgpt_retail_v1/artifact.json")
DEFAULT_CHECKPOINT = Path(
    f"{DATA_ROOT}/papc/frugalgpt_retail_v1/checkpoint"
)
DEFAULT_REQUEST_MANIFEST = Path(
    f"{DATA_ROOT}/papc/"
    "routellm_toolsandbox_request_manifest_private_20260802_v2.json"
)
DEFAULT_CHEAP_TRAJECTORIES = Path(
    f"{DATA_ROOT}/papc/"
    "toolsandbox_base129_baseline_accounting_private_"
    "20260701_codex_gpt54user_v1/cheap_full/raw_tool_sandbox/trajectories"
)
DEFAULT_CHEAP_PRIVATE = Path(
    f"{DATA_ROOT}/papc/"
    "toolsandbox_base129_baseline_accounting_cheap_full_private_"
    "20260701_codex_gpt54user_v1.json"
)
DEFAULT_STRONG_PRIVATE = Path(
    f"{DATA_ROOT}/papc/"
    "toolsandbox_base129_baseline_accounting_strong_full_private_"
    "20260701_codex_gpt54user_v1.json"
)
DEFAULT_CHEAP_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/toolsandbox_base129_baseline_accounting_cheap_full_"
    "20260701_codex_gpt54user_v1.json"
)
DEFAULT_STRONG_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/toolsandbox_base129_baseline_accounting_strong_full_"
    "20260701_codex_gpt54user_v1.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--artifact-json", type=Path, default=DEFAULT_ARTIFACT)
    freeze.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    freeze.add_argument(
        "--request-manifest-json", type=Path, default=DEFAULT_REQUEST_MANIFEST
    )
    freeze.add_argument(
        "--cheap-trajectories", type=Path, default=DEFAULT_CHEAP_TRAJECTORIES
    )
    freeze.add_argument("--private-output-json", type=Path, required=True)
    freeze.add_argument("--public-output-json", type=Path, required=True)

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--private-policy-json", type=Path, required=True)
    reduce.add_argument("--public-policy-json", type=Path, required=True)
    reduce.add_argument("--cheap-private-json", type=Path, default=DEFAULT_CHEAP_PRIVATE)
    reduce.add_argument(
        "--strong-private-json", type=Path, default=DEFAULT_STRONG_PRIVATE
    )
    reduce.add_argument("--cheap-public-json", type=Path, default=DEFAULT_CHEAP_PUBLIC)
    reduce.add_argument(
        "--strong-public-json", type=Path, default=DEFAULT_STRONG_PUBLIC
    )
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
                "mean_similarity": result["official_outcome"]["mean_similarity"],
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
    artifact_record = _read_object(args.artifact_json)
    artifact = FrugalGPTScorerArtifact.from_public_record(artifact_record)
    if (
        artifact.surface != "tau2_retail"
        or artifact.upstream_revision != FRUGALGPT_UPSTREAM_REVISION
        or artifact.cheap_model != CHEAP_MODEL
        or artifact.strong_model != STRONG_MODEL
    ):
        raise ValueError("frugalgpt_retail_artifact_binding_mismatch")
    scorer = FrugalGPTDistilBertScorer.load(
        artifact=artifact,
        checkpoint_dir=args.checkpoint_dir,
    )
    manifest_rows = _request_manifest_rows(_read_object(args.request_manifest_json))

    scored: list[dict[str, Any]] = []
    answer_bindings: list[dict[str, Any]] = []
    for row in manifest_rows:
        conversation_path = (
            args.cheap_trajectories
            / row["scenario_name"]
            / "conversation.json"
        )
        public_episode = _outcome_blind_public_episode(conversation_path)
        opening = next(
            (
                message["content"]
                for message in public_episode
                if message["role"] == "user"
            ),
            None,
        )
        if opening != row["original_request"]:
            raise ValueError("toolsandbox_cheap_opening_mismatch")
        generated_answer = json.dumps(
            public_episode,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        score = scorer(
            CheapCompletionRouterView(
                original_request=row["original_request"],
                cheap_generated_answer=generated_answer,
            )
        )
        answer_sha256 = _text_sha256(generated_answer)
        scored.append(
            {
                "public_ordinal": row["public_ordinal"],
                "scenario_name": row["scenario_name"],
                "scenario_name_sha256": _text_sha256(row["scenario_name"]),
                "original_request_sha256": _text_sha256(row["original_request"]),
                "cheap_generated_answer_sha256": answer_sha256,
                "predicted_cheap_quality": score,
            }
        )
        answer_bindings.append(
            {
                "public_ordinal": row["public_ordinal"],
                "conversation_file_sha256": _file_sha256(conversation_path),
                "outcome_blind_projection_sha256": answer_sha256,
            }
        )

    ranked = sorted(
        scored,
        key=lambda row: (
            row["predicted_cheap_quality"],
            row["original_request_sha256"],
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
    routes.sort(key=lambda row: row["public_ordinal"])
    public_routes = [
        {key: value for key, value in row.items() if key != "scenario_name"}
        for row in routes
    ]
    common = {
        "method": "FrugalGPT (Retail scorer, 90% strong budget)",
        "surface": "toolsandbox",
        "split": "base129",
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
            "tie_breaker": "original_request_sha256 ascending",
            "retail_accept_threshold_audit_only": (
                artifact.operating_point.accept_threshold
            ),
        },
        "source_binding": {
            "request_manifest_schema": REQUEST_MANIFEST_SCHEMA,
            "request_manifest_sha256": _file_sha256(args.request_manifest_json),
            "cheap_answer_projection_bundle_sha256": _digest(answer_bindings),
            "cheap_answer_projection": (
                "non-system role/content/tool-call allowlist; tool_details excluded"
            ),
            "endpoint_outcome_fields_accessed_by_freeze": False,
            "target_endpoint_receipts_accessed_by_freeze": False,
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
            "ToolSandbox has no disjoint train/dev split, so the Retail scorer is used without target-domain calibration.",
            "The 90% strong-restart budget is frozen before endpoint outcomes are reduced.",
            "Every task first incurs its completed cheap episode; escalations restart Strong from the pristine paired endpoint.",
        ],
    }
    public = {**public_unsigned, "policy_sha256": _digest(public_unsigned)}
    return private, public


def _outcome_blind_public_episode(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("toolsandbox_cheap_conversation_invalid")
    projected: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("toolsandbox_cheap_message_invalid")
        role = item.get("role")
        if role == "system":
            continue
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("toolsandbox_cheap_message_role_invalid")
        record: dict[str, Any] = {
            "role": role,
            "content": str(item.get("content", "")),
        }
        if role == "assistant" and "tool_calls" in item:
            tool_calls = item.get("tool_calls")
            if not isinstance(tool_calls, list):
                raise ValueError("toolsandbox_cheap_tool_calls_invalid")
            record["tool_calls"] = [
                {
                    "id": str(call.get("id", "")),
                    "type": str(call.get("type", "")),
                    "function": {
                        "name": str(call.get("function", {}).get("name", "")),
                        "arguments": str(
                            call.get("function", {}).get("arguments", "")
                        ),
                    },
                }
                for call in tool_calls
                if isinstance(call, Mapping)
                and isinstance(call.get("function"), Mapping)
            ]
        if role == "tool":
            record["tool_call_id"] = str(item.get("tool_call_id", ""))
            record["name"] = str(item.get("name", ""))
        projected.append(record)
    if not any(item["role"] == "assistant" for item in projected):
        raise ValueError("toolsandbox_cheap_answer_missing")
    return projected


def _request_manifest_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        source.get("schema") != REQUEST_MANIFEST_SCHEMA
        or source.get("surface") != "toolsandbox"
        or source.get("split") != "base129"
        or source.get("official_test_count") != EXPECTED_COUNT
    ):
        raise ValueError("toolsandbox_request_manifest_identity_invalid")
    raw_rows = source.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_request_manifest_rows_invalid")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "public_ordinal",
            "scenario_name",
            "original_request",
        }:
            raise ValueError("toolsandbox_request_manifest_row_invalid")
        row = dict(raw)
        if (
            not isinstance(row["public_ordinal"], int)
            or isinstance(row["public_ordinal"], bool)
            or not isinstance(row["scenario_name"], str)
            or not row["scenario_name"]
            or not isinstance(row["original_request"], str)
            or not row["original_request"].strip()
        ):
            raise ValueError("toolsandbox_request_manifest_value_invalid")
        rows.append(row)
    rows.sort(key=lambda row: row["public_ordinal"])
    if [row["public_ordinal"] for row in rows] != list(range(EXPECTED_COUNT)):
        raise ValueError("toolsandbox_request_manifest_ordinals_invalid")
    if len({row["scenario_name"] for row in rows}) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_request_manifest_scenarios_duplicate")
    return rows


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    private = _read_object(args.private_policy_json)
    public = _read_object(args.public_policy_json)
    _validate_policies(private, public)
    _validate_endpoint_public(_read_object(args.cheap_public_json), endpoint="cheap")
    _validate_endpoint_public(
        _read_object(args.strong_public_json), endpoint="strong"
    )
    cheap = _receipts_by_ordinal(_read_object(args.cheap_private_json), "cheap")
    strong = _receipts_by_ordinal(
        _read_object(args.strong_private_json), "strong"
    )
    if set(cheap) != set(range(EXPECTED_COUNT)) or set(strong) != set(
        range(EXPECTED_COUNT)
    ):
        raise ValueError("toolsandbox_endpoint_membership_mismatch")

    routes = private.get("routes")
    if not isinstance(routes, list) or len(routes) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_policy_routes_invalid")
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "cost_usd": 0.0,
    }
    similarities: list[float] = []
    milestone_complete = 0
    rows: list[dict[str, Any]] = []
    for route in routes:
        ordinal = route.get("public_ordinal")
        endpoint = route.get("selected_endpoint")
        if not isinstance(ordinal, int) or endpoint not in {"cheap", "strong"}:
            raise ValueError("toolsandbox_policy_route_invalid")
        cheap_receipt = cheap[ordinal]
        selected_receipt = cheap_receipt if endpoint == "cheap" else strong[ordinal]
        for receipt in (cheap_receipt,) if endpoint == "cheap" else (
            cheap_receipt,
            strong[ordinal],
        ):
            accounting = _agent_accounting(
                receipt,
                "cheap" if receipt is cheap_receipt else "strong",
            )
            for key in ("calls", "input_tokens", "output_tokens"):
                totals[key] += accounting[key]
            totals["latency_seconds"] += accounting["latency_seconds"]
            totals["cost_usd"] += accounting["cost_usd"]
        private_record = selected_receipt.get("private")
        scenario = (
            private_record.get("scenario_name")
            if isinstance(private_record, Mapping)
            else None
        )
        if scenario != route.get("scenario_name"):
            raise ValueError("toolsandbox_policy_scenario_mismatch")
        evaluation = selected_receipt.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("toolsandbox_endpoint_evaluation_missing")
        similarity = _finite_float(evaluation.get("similarity"), "similarity")
        milestone = _finite_float(
            evaluation.get("milestone_similarity"), "milestone_similarity"
        )
        similarities.append(similarity)
        milestone_complete += int(milestone >= 1.0)
        rows.append(
            {
                "public_ordinal": ordinal,
                "selected_endpoint": endpoint,
                "scenario_name_sha256": route["scenario_name_sha256"],
                "similarity": similarity,
                "milestone_complete": milestone >= 1.0,
            }
        )

    result = {
        "schema": RESULT_SCHEMA,
        "decision": "observed_frozen_ood_replay_complete",
        "method": private["method"],
        "surface": "toolsandbox",
        "split": "base129",
        "provider_calls_added_by_replay": 0,
        "endpoint_selection": private["endpoint_selection"],
        "official_outcome": {
            "metric": "official mean state similarity",
            "mean_similarity": sum(similarities) / EXPECTED_COUNT,
            "milestone_complete": milestone_complete,
            "denominator": EXPECTED_COUNT,
        },
        "agent_side_accounting": {
            **totals,
            "cost_usd": round(totals["cost_usd"], 12),
            "all_cheap_episodes_included": True,
            "selected_strong_restarts_included": True,
            "harness_and_user_simulator_excluded": True,
        },
        "source_binding": {
            "private_policy_sha256": _file_sha256(args.private_policy_json),
            "public_policy_sha256": _file_sha256(args.public_policy_json),
            "cheap_private_sha256": _file_sha256(args.cheap_private_json),
            "strong_private_sha256": _file_sha256(args.strong_private_json),
            "cheap_public_sha256": _file_sha256(args.cheap_public_json),
            "strong_public_sha256": _file_sha256(args.strong_public_json),
        },
        "rows": sorted(rows, key=lambda row: row["public_ordinal"]),
    }
    return {**result, "result_sha256": _digest(result)}


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
        raise ValueError("toolsandbox_policy_binding_invalid")
    for record in (private, public):
        unsigned = dict(record)
        claimed = unsigned.pop("policy_sha256", None)
        if claimed != _digest(unsigned):
            raise ValueError("toolsandbox_policy_sha256_invalid")
    private_routes = private.get("routes")
    public_routes = public.get("routes")
    if not isinstance(private_routes, list) or not isinstance(public_routes, list):
        raise ValueError("toolsandbox_policy_routes_invalid")
    projected = [
        {key: value for key, value in row.items() if key != "scenario_name"}
        for row in private_routes
        if isinstance(row, Mapping)
    ]
    if projected != public_routes:
        raise ValueError("toolsandbox_public_private_policy_mismatch")


def _validate_endpoint_public(report: Mapping[str, Any], *, endpoint: str) -> None:
    expected_arm = "cheap_full" if endpoint == "cheap" else "strong_full"
    expected_model = CHEAP_MODEL if endpoint == "cheap" else STRONG_MODEL
    summary = report.get("summary")
    plan = report.get("model_plan")
    if (
        report.get("decision")
        != "toolsandbox_base129_baseline_accounting_ready"
        or report.get("arm") != expected_arm
        or not isinstance(summary, Mapping)
        or summary.get("attempted_episode_count") != EXPECTED_COUNT
        or summary.get("runtime_failure_count") != 0
        or not isinstance(plan, Mapping)
        or plan.get("agent_model") != expected_model
    ):
        raise ValueError(f"toolsandbox_{endpoint}_endpoint_invalid")


def _receipts_by_ordinal(
    report: Mapping[str, Any], endpoint: str
) -> dict[int, Mapping[str, Any]]:
    receipts = report.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_COUNT:
        raise ValueError(f"toolsandbox_{endpoint}_receipt_count_invalid")
    result: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError(f"toolsandbox_{endpoint}_receipt_invalid")
        ordinal = receipt.get("public_ordinal")
        private = receipt.get("private")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal in result
            or not isinstance(private, Mapping)
            or private.get("exception_type") is not None
            or receipt.get("official_evaluator_used") is not True
        ):
            raise ValueError(f"toolsandbox_{endpoint}_receipt_invalid")
        result[ordinal] = receipt
    return result


def _agent_accounting(
    receipt: Mapping[str, Any], endpoint: str
) -> dict[str, Any]:
    private = receipt.get("private")
    ledger = private.get("runtime_ledger") if isinstance(private, Mapping) else None
    per_role = ledger.get("per_role") if isinstance(ledger, Mapping) else None
    role = per_role.get(endpoint) if isinstance(per_role, Mapping) else None
    if not isinstance(role, Mapping) or role.get("token_usage_available") is not True:
        raise ValueError(f"toolsandbox_{endpoint}_usage_unavailable")
    calls = _nonnegative_int(role.get("calls"), "calls")
    input_tokens = _nonnegative_int(
        role.get("billable_input_tokens"), "input_tokens"
    )
    output_tokens = _nonnegative_int(role.get("output_tokens"), "output_tokens")
    latency = _finite_float(role.get("latency_seconds"), "latency_seconds")
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
        "latency_seconds": latency,
        "cost_usd": cost,
    }


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


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"toolsandbox_{name}_nonfinite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"toolsandbox_{name}_invalid")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

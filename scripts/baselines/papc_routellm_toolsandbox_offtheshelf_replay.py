#!/usr/bin/env python3
"""Freeze and replay RouteLLM-BERT on paired ToolSandbox endpoints."""

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

from papc.baselines.routellm_bert_upstream import (  # noqa: E402
    ROUTELLM_BERT_BASE_BUNDLE_SHA256,
    ROUTELLM_BERT_BASE_REPO,
    ROUTELLM_BERT_BASE_REVISION,
    ROUTELLM_BERT_MAX_LENGTH,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256,
    validate_router_runtime_environment,
)
from papc.paths import DATA_ROOT

REQUEST_MANIFEST_SCHEMA = "papc.routellm.toolsandbox.request_manifest.private.v1"
POLICY_SCHEMA = "papc.routellm.toolsandbox.offtheshelf_policy.v2"
PRIVATE_POLICY_SCHEMA = "papc.routellm.toolsandbox.offtheshelf_policy.private.v2"
RESULT_SCHEMA = "papc.routellm.toolsandbox.offtheshelf_replay.v2"
EXPECTED_COUNT = 129

DEFAULT_MODEL = Path(
    f"{DATA_ROOT}/papc/upstream_models/routellm-bert-gpt4-augmented-f625a9c"
)
DEFAULT_REQUEST_MANIFEST = Path(
    f"{DATA_ROOT}/papc/"
    "routellm_toolsandbox_request_manifest_private_20260802_v2.json"
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
    REPO_ROOT / "outputs/papc/toolsandbox_base129_baseline_accounting_cheap_full_"
    "20260701_codex_gpt54user_v1.json"
)
DEFAULT_STRONG_PUBLIC = (
    REPO_ROOT / "outputs/papc/toolsandbox_base129_baseline_accounting_strong_full_"
    "20260701_codex_gpt54user_v1.json"
)
DEFAULT_PRIVATE_POLICY = Path(
    f"{DATA_ROOT}/papc/"
    "routellm_toolsandbox_offtheshelf_policy_private_20260802_v2.json"
)
DEFAULT_PUBLIC_POLICY = (
    REPO_ROOT
    / "outputs/papc/routellm_toolsandbox_offtheshelf_policy_noapi_20260802_v2.json"
)
DEFAULT_RESULT = (
    REPO_ROOT
    / "outputs/papc/routellm_toolsandbox_offtheshelf_result_noapi_20260802_v2.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--base-model-dir", type=Path, default=DEFAULT_MODEL)
    freeze.add_argument(
        "--request-manifest-json", type=Path, default=DEFAULT_REQUEST_MANIFEST
    )
    freeze.add_argument("--strong-budget-fraction", type=float, default=0.5)
    freeze.add_argument(
        "--private-output-json", type=Path, default=DEFAULT_PRIVATE_POLICY
    )
    freeze.add_argument(
        "--public-output-json", type=Path, default=DEFAULT_PUBLIC_POLICY
    )

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument(
        "--private-policy-json", type=Path, default=DEFAULT_PRIVATE_POLICY
    )
    reduce.add_argument(
        "--public-policy-json", type=Path, default=DEFAULT_PUBLIC_POLICY
    )
    reduce.add_argument(
        "--cheap-private-json", type=Path, default=DEFAULT_CHEAP_PRIVATE
    )
    reduce.add_argument(
        "--strong-private-json", type=Path, default=DEFAULT_STRONG_PRIVATE
    )
    reduce.add_argument("--cheap-public-json", type=Path, default=DEFAULT_CHEAP_PUBLIC)
    reduce.add_argument(
        "--strong-public-json", type=Path, default=DEFAULT_STRONG_PUBLIC
    )
    reduce.add_argument("--output-json", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "freeze":
        private, public = _freeze(args)
        _write_append_only(args.private_output_json, private)
        _write_append_only(args.public_output_json, public)
        print(json.dumps({"decision": public["decision"], "provider_calls": 0}))
        return 0
    result = _reduce(args)
    _write_append_only(args.output_json, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "mean_similarity": result["official_outcome"]["mean_similarity"],
                "milestone_complete": result["official_outcome"]["milestone_complete"],
                "agent_side_cost_usd": result["agent_side_accounting"]["cost_usd"],
                "provider_calls": 0,
            }
        )
    )
    return 0


def _freeze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    fraction = float(args.strong_budget_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("strong_budget_fraction_out_of_range")
    runtime_sha = validate_router_runtime_environment()
    if runtime_sha != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("router_runtime_environment_sha256_mismatch")
    bundle_sha = _directory_sha256(args.base_model_dir)
    if bundle_sha != ROUTELLM_BERT_BASE_BUNDLE_SHA256:
        raise ValueError("routellm_base_model_bundle_mismatch")

    rows = _request_manifest_rows(_read_object(args.request_manifest_json))
    if len(rows) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_source_count_mismatch")
    scores = _strong_win_scores(args.base_model_dir, rows)
    routes = _routes_from_scores(rows, scores, fraction=fraction)
    strong_count = sum(row["selected_endpoint"] == "strong" for row in routes)
    public_routes = [
        {key: value for key, value in row.items() if key != "scenario_name"}
        for row in routes
    ]
    common = {
        "method": "RouteLLM-BERT (released, 50% strong budget)",
        "surface": "toolsandbox",
        "split": "base129",
        "official_test_count": EXPECTED_COUNT,
        "strong_budget_fraction": fraction,
        "strong_endpoint_count": strong_count,
        "cheap_endpoint_count": EXPECTED_COUNT - strong_count,
        "router": {
            "upstream_repo": ROUTELLM_BERT_BASE_REPO,
            "upstream_revision": ROUTELLM_BERT_BASE_REVISION,
            "bundle_sha256": bundle_sha,
            "runtime_environment_sha256": runtime_sha,
            "input": "initial user request only",
            "calibration": "none; released checkpoint, fixed 50% strong budget",
            "ranking_tie_breaker": "original_request_sha256",
        },
        "source_binding": {
            "request_manifest_schema": REQUEST_MANIFEST_SCHEMA,
            "request_manifest_sha256": _file_sha256(args.request_manifest_json),
            "endpoint_outcome_source_accessed_by_freeze": False,
        },
        "provider_calls": 0,
    }
    private = {
        "schema": PRIVATE_POLICY_SCHEMA,
        "decision": "frozen_without_endpoint_outcome_access",
        **common,
        "routes": routes,
    }
    public = {
        "schema": POLICY_SCHEMA,
        "decision": "frozen_without_endpoint_outcome_access",
        **common,
        "routes": public_routes,
        "claim_boundary": [
            "The released request-only BERT router is used off the shelf.",
            "The 50% strong budget is fixed without ToolSandbox outcome tuning.",
            "Freeze reads only the outcome-free request manifest and performs no provider calls.",
        ],
    }
    return private, public


def _request_manifest_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(source) != {
        "schema",
        "surface",
        "split",
        "official_test_count",
        "rows",
    }:
        raise ValueError("toolsandbox_request_manifest_fields_invalid")
    if (
        source.get("schema") != REQUEST_MANIFEST_SCHEMA
        or source.get("surface") != "toolsandbox"
        or source.get("split") != "base129"
        or source.get("official_test_count") != EXPECTED_COUNT
    ):
        raise ValueError("toolsandbox_request_manifest_identity_invalid")
    receipts = source.get("rows")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_request_manifest_rows_invalid")
    rows: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    seen_scenarios: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "public_ordinal",
            "scenario_name",
            "original_request",
        }:
            raise ValueError("toolsandbox_request_manifest_row_fields_invalid")
        ordinal = receipt.get("public_ordinal")
        scenario = receipt.get("scenario_name")
        original_request = receipt.get("original_request")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise ValueError("toolsandbox_source_ordinal_invalid")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("toolsandbox_source_scenario_invalid")
        if not isinstance(original_request, str) or not original_request.strip():
            raise ValueError("toolsandbox_original_request_missing")
        if ordinal in seen_ordinals or scenario in seen_scenarios:
            raise ValueError("toolsandbox_source_membership_duplicate")
        seen_ordinals.add(ordinal)
        seen_scenarios.add(scenario)
        rows.append(
            {
                "public_ordinal": ordinal,
                "scenario_name": scenario,
                "original_request": original_request,
            }
        )
    rows.sort(key=lambda row: row["public_ordinal"])
    if [row["public_ordinal"] for row in rows] != list(range(EXPECTED_COUNT)):
        raise ValueError("toolsandbox_source_ordinals_not_exact")
    return rows


def _strong_win_scores(
    base_model_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    if model.config.id2label.get(0) != "strong":
        raise ValueError("routellm_label_zero_not_strong")
    inputs = tokenizer(
        [str(row["original_request"]) for row in rows],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ROUTELLM_BERT_MAX_LENGTH,
    )
    with torch.no_grad():
        return torch.softmax(model(**inputs).logits, dim=-1)[:, 0].tolist()


def _routes_from_scores(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    fraction: float,
) -> list[dict[str, Any]]:
    scored = [
        {
            "public_ordinal": row["public_ordinal"],
            "scenario_name": row["scenario_name"],
            "scenario_name_sha256": _text_sha256(str(row["scenario_name"])),
            "original_request_sha256": _text_sha256(str(row["original_request"])),
            "strong_win_score": float(score),
        }
        for row, score in zip(rows, scores, strict=True)
    ]
    ranked = sorted(
        scored,
        key=lambda row: (
            row["strong_win_score"],
            row["original_request_sha256"],
        ),
        reverse=True,
    )
    strong_count = math.ceil(len(rows) * fraction)
    strong_ordinals = {row["public_ordinal"] for row in ranked[:strong_count]}
    return [
        {
            **row,
            "selected_endpoint": (
                "strong" if row["public_ordinal"] in strong_ordinals else "cheap"
            ),
        }
        for row in scored
    ]


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    private_policy = _read_object(args.private_policy_json)
    public_policy = _read_object(args.public_policy_json)
    if private_policy.get("schema") != PRIVATE_POLICY_SCHEMA:
        raise ValueError("toolsandbox_private_policy_schema_invalid")
    if public_policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("toolsandbox_public_policy_schema_invalid")
    if private_policy.get("decision") != "frozen_without_endpoint_outcome_access":
        raise ValueError("toolsandbox_policy_not_frozen")
    _validate_public_private_policy(private_policy, public_policy)

    cheap_public = _read_object(args.cheap_public_json)
    strong_public = _read_object(args.strong_public_json)
    _validate_endpoint_public(cheap_public, endpoint="cheap")
    _validate_endpoint_public(strong_public, endpoint="strong")
    cheap = _receipts_by_ordinal(_read_object(args.cheap_private_json), "cheap")
    strong = _receipts_by_ordinal(_read_object(args.strong_private_json), "strong")
    expected = set(range(EXPECTED_COUNT))
    if set(cheap) != expected or set(strong) != expected:
        raise ValueError("toolsandbox_endpoint_membership_mismatch")

    selected_rows: list[dict[str, Any]] = []
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "cost_usd": 0.0,
    }
    similarities: list[float] = []
    complete = 0
    routes = private_policy.get("routes")
    if not isinstance(routes, list) or len(routes) != EXPECTED_COUNT:
        raise ValueError("toolsandbox_policy_routes_invalid")
    for route in routes:
        if not isinstance(route, Mapping):
            raise ValueError("toolsandbox_policy_route_invalid")
        ordinal = route.get("public_ordinal")
        endpoint = route.get("selected_endpoint")
        if endpoint not in {"cheap", "strong"} or not isinstance(ordinal, int):
            raise ValueError("toolsandbox_route_selection_invalid")
        receipt = (cheap if endpoint == "cheap" else strong)[ordinal]
        private = receipt.get("private")
        scenario = (
            private.get("scenario_name") if isinstance(private, Mapping) else None
        )
        if scenario != route.get("scenario_name"):
            raise ValueError("toolsandbox_route_scenario_mismatch")
        evaluation = receipt.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("toolsandbox_endpoint_evaluation_missing")
        similarity = _finite_float(evaluation.get("similarity"), "similarity")
        milestone = _finite_float(
            evaluation.get("milestone_similarity"), "milestone_similarity"
        )
        similarities.append(similarity)
        complete += int(milestone >= 1.0)
        accounting = _selected_agent_accounting(receipt, endpoint)
        for key in ("calls", "input_tokens", "output_tokens"):
            totals[key] += accounting[key]
        totals["latency_seconds"] += accounting["latency_seconds"]
        totals["cost_usd"] += accounting["cost_usd"]
        selected_rows.append(
            {
                "public_ordinal": ordinal,
                "scenario_name_sha256": route["scenario_name_sha256"],
                "selected_endpoint": endpoint,
                "similarity": similarity,
                "milestone_complete": milestone >= 1.0,
                "agent_calls": accounting["calls"],
                "agent_cost_usd": accounting["cost_usd"],
            }
        )

    return {
        "schema": RESULT_SCHEMA,
        "decision": "observed_offtheshelf_replay_complete",
        "method": private_policy["method"],
        "surface": "toolsandbox",
        "split": "base129",
        "provider_calls": 0,
        "endpoint_selection": {
            "cheap": sum(row["selected_endpoint"] == "cheap" for row in selected_rows),
            "strong": sum(
                row["selected_endpoint"] == "strong" for row in selected_rows
            ),
        },
        "official_outcome": {
            "metric": "official mean state similarity",
            "mean_similarity": sum(similarities) / EXPECTED_COUNT,
            "milestone_complete": complete,
            "denominator": EXPECTED_COUNT,
        },
        "agent_side_accounting": {
            **totals,
            "cost_usd": round(totals["cost_usd"], 12),
            "pricing": {
                model: {
                    "input_per_million_usd": price.input_per_million_usd,
                    "output_per_million_usd": price.output_per_million_usd,
                }
                for model, price in FROZEN_CACHE_NEUTRAL_PRICES.items()
                if model in {"gpt-5.4-mini", "gpt-5.5"}
            },
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
        "rows": sorted(selected_rows, key=lambda row: row["public_ordinal"]),
        "claim_boundary": [
            "Request-level routing selects one complete endpoint trajectory per scenario.",
            "The router policy was frozen from an outcome-free request manifest before this reducer accessed endpoint outcomes.",
            "No provider calls are made by replay; cost is reconstructed from observed agent usage.",
        ],
    }


def _validate_public_private_policy(
    private: Mapping[str, Any], public: Mapping[str, Any]
) -> None:
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
    expected_model = "gpt-5.4-mini" if endpoint == "cheap" else "gpt-5.5"
    if report.get("decision") != "toolsandbox_base129_baseline_accounting_ready":
        raise ValueError(f"toolsandbox_{endpoint}_endpoint_not_ready")
    if report.get("arm") != expected_arm:
        raise ValueError(f"toolsandbox_{endpoint}_endpoint_arm_mismatch")
    summary = report.get("summary")
    plan = report.get("model_plan")
    if (
        not isinstance(summary, Mapping)
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
        raise ValueError(f"toolsandbox_{endpoint}_receipt_count_mismatch")
    result: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError(f"toolsandbox_{endpoint}_receipt_invalid")
        ordinal = receipt.get("public_ordinal")
        private = receipt.get("private")
        if not isinstance(ordinal, int) or ordinal in result:
            raise ValueError(f"toolsandbox_{endpoint}_ordinal_invalid")
        if (
            not isinstance(private, Mapping)
            or private.get("exception_type") is not None
            or receipt.get("official_evaluator_used") is not True
        ):
            raise ValueError(f"toolsandbox_{endpoint}_runtime_failure")
        result[ordinal] = receipt
    return result


def _selected_agent_accounting(
    receipt: Mapping[str, Any], endpoint: str
) -> dict[str, Any]:
    private = receipt.get("private")
    ledger = private.get("runtime_ledger") if isinstance(private, Mapping) else None
    per_role = ledger.get("per_role") if isinstance(ledger, Mapping) else None
    role = per_role.get(endpoint) if isinstance(per_role, Mapping) else None
    if not isinstance(role, Mapping) or role.get("token_usage_available") is not True:
        raise ValueError(f"toolsandbox_{endpoint}_agent_usage_unavailable")
    calls = role.get("calls")
    input_tokens = role.get("billable_input_tokens")
    output_tokens = role.get("output_tokens")
    latency = role.get("latency_seconds")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (calls, input_tokens, output_tokens)
    ):
        raise ValueError(f"toolsandbox_{endpoint}_agent_usage_invalid")
    latency_value = _finite_float(latency, "latency_seconds")
    model = "gpt-5.4-mini" if endpoint == "cheap" else "gpt-5.5"
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    cost = (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000.0
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_value,
        "cost_usd": cost,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"json_object_required:{path}")
    return value


def _write_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    root = Path(path)
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    if not files:
        raise ValueError("routellm_base_model_directory_empty")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = file_path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"toolsandbox_{name}_nonfinite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze and replay an off-the-shelf RouteLLM-BERT Agent-Diff cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.agent_diff_router_driver import (  # noqa: E402
    AGENT_DIFF_SCORER_NAME,
    load_agent_diff_official_registry,
)
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
from scripts.baselines.papc_frugalgpt_two_surface_test_replay_noapi import (  # noqa: E402
    _validate_agent_clean_strong_source,
)

POLICY_SCHEMA = "papc.routellm.agent_diff.offtheshelf_policy.v2"
RESULT_SCHEMA = "papc.routellm.agent_diff.offtheshelf_replay.v3"
DEFAULT_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/datasets/agent-diff-bench"
)
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_BASE_MODEL = Path(
    f"{DATA_ROOT}/papc/upstream_models/"
    "routellm-bert-gpt4-augmented-f625a9c"
)
DEFAULT_POLICY = (
    REPO_ROOT
    / "outputs/papc/"
    "routellm_agent_diff_offtheshelf_policy_noapi_20260802_v2.json"
)
PRIOR_POLICY = (
    REPO_ROOT
    / "outputs/papc/"
    "routellm_agent_diff_offtheshelf_policy_noapi_20260801_v1.json"
)
DEFAULT_RESULT = (
    REPO_ROOT
    / "outputs/papc/"
    "routellm_agent_diff_offtheshelf_result_noapi_20260802_v3.json"
)
DEFAULT_CHEAP_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/"
    "agent_diff_hf_test45_cheap_baseline_public_20260706_codex_cheap45_v1.json"
)
PRIVATE_ROOT = Path(f"{DATA_ROOT}/papc")
DEFAULT_CHEAP_PRIVATE = (
    PRIVATE_ROOT
    / "agent_diff_hf_test45_cheap_baseline_private_20260706_codex_cheap45_v1"
    / "raw_rows"
)
SCORER_PATH = REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--agent-diff-dataset", type=Path, default=DEFAULT_DATASET)
    freeze.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    freeze.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    freeze.add_argument("--strong-budget-fraction", type=float, default=0.5)
    freeze.add_argument("--prior-policy-json", type=Path, default=PRIOR_POLICY)
    freeze.add_argument("--output-json", type=Path, default=DEFAULT_POLICY)

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--agent-diff-dataset", type=Path, default=DEFAULT_DATASET)
    reduce.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    reduce.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    reduce.add_argument("--cheap-public-json", type=Path, default=DEFAULT_CHEAP_PUBLIC)
    reduce.add_argument("--strong-public-json", type=Path, required=True)
    reduce.add_argument(
        "--cheap-private-rows", type=Path, default=DEFAULT_CHEAP_PRIVATE
    )
    reduce.add_argument("--agent-clean-strong-root", type=Path, required=True)
    reduce.add_argument("--output-json", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"json_object_required:{path}")
    return value


def _write_json_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _freeze_policy(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.strong_budget_fraction < 1.0:
        raise ValueError("strong_budget_fraction_out_of_range")
    runtime_sha256 = validate_router_runtime_environment()
    if runtime_sha256 != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("router_runtime_environment_sha256_mismatch")
    bundle_sha256 = _directory_sha256(args.base_model_dir)
    if bundle_sha256 != ROUTELLM_BERT_BASE_BUNDLE_SHA256:
        raise ValueError("routellm_base_model_bundle_mismatch")

    registry = load_agent_diff_official_registry(
        args.agent_diff_dataset,
        split_gate_json=args.split_gate_json,
    )
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    if model.config.id2label.get(0) != "strong":
        raise ValueError("routellm_label_zero_not_strong")
    inputs = tokenizer(
        [task.original_request for task in registry.tasks],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ROUTELLM_BERT_MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.softmax(logits, dim=-1)[:, 0].tolist()
    scored = [
        {
            "native_ordinal": task.native_ordinal,
            "original_request_sha256": hashlib.sha256(
                task.original_request.encode("utf-8")
            ).hexdigest(),
            "service": task.service,
            "strong_win_score": float(score),
        }
        for task, score in zip(registry.tasks, scores, strict=True)
    ]
    ranked = sorted(
        scored,
        key=lambda item: (
            item["strong_win_score"],
            item["original_request_sha256"],
        ),
        reverse=True,
    )
    strong_count = math.ceil(len(ranked) * args.strong_budget_fraction)
    strong_ordinals = {
        int(item["native_ordinal"]) for item in ranked[:strong_count]
    }
    routes = [
        {
            **item,
            "selected_endpoint": (
                "strong" if int(item["native_ordinal"]) in strong_ordinals else "cheap"
            ),
        }
        for item in scored
    ]
    prior = _read_json(args.prior_policy_json)
    prior_routes = {
        int(item["native_ordinal"]): item["selected_endpoint"]
        for item in prior.get("routes", [])
    }
    current_routes = {
        int(item["native_ordinal"]): item["selected_endpoint"]
        for item in routes
    }
    if prior_routes != current_routes:
        raise ValueError("agent_diff_v2_route_assignments_changed_from_v1")
    return {
        "schema": POLICY_SCHEMA,
        "decision": "frozen_before_outcome_replay",
        "method": "RouteLLM-BERT (released, 50% strong budget)",
        "surface": "agent_diff",
        "split": "official_test",
        "official_test_count": len(routes),
        "test_registry_sha256": registry.split.test_registry_sha256,
        "split_gate_report_sha256": registry.split_gate_report_sha256,
        "router": {
            "base_repo": ROUTELLM_BERT_BASE_REPO,
            "base_revision": ROUTELLM_BERT_BASE_REVISION,
            "base_bundle_sha256": bundle_sha256,
            "input": "original_request_only",
            "score": "softmax_label_0_strong",
            "device": "cpu",
            "runtime_environment_sha256": runtime_sha256,
        },
        "selection": {
            "rule": (
                "top_ranked_by_strong_win_score_then_original_request_sha256"
            ),
            "strong_budget_fraction": args.strong_budget_fraction,
            "rounding": "ceil",
            "strong_count": strong_count,
            "cheap_count": len(routes) - strong_count,
            "strong_threshold": ranked[strong_count - 1]["strong_win_score"],
            "next_cheap_score": ranked[strong_count]["strong_win_score"],
            "cutoff_score_tie": (
                ranked[strong_count - 1]["strong_win_score"]
                == ranked[strong_count]["strong_win_score"]
            ),
            "all_scores_unique": (
                len({item["strong_win_score"] for item in ranked}) == len(ranked)
            ),
            "strong_service_counts": dict(
                Counter(
                    item["service"]
                    for item in routes
                    if item["selected_endpoint"] == "strong"
                )
            ),
        },
        "routes": routes,
        "outcome_firewall": {
            "official_test_outcomes_read": False,
            "endpoint_trajectory_artifacts_read": False,
            "test_ids_present": False,
            "hidden_answers_present": False,
            "task_identity_used_as_router_input": False,
            "task_identity_used_as_tie_break": False,
            "native_ordinal_present_for_offline_join_only": True,
        },
        "prior_policy_binding": {
            "prior_policy_file_sha256": _file_sha256(args.prior_policy_json),
            "route_assignments_unchanged": True,
            "change": "task_identity_tie_break_removed_no_cutoff_tie",
        },
        "provider_calls": 0,
        "api_spend_performed": False,
        "public_safe": True,
    }


def _validate_public_summary(
    summary: Mapping[str, Any], *, model: str, effort: str
) -> None:
    dataset = summary.get("dataset")
    model_plan = summary.get("model_plan")
    aggregate = summary.get("aggregate")
    if not isinstance(dataset, Mapping) or not isinstance(model_plan, Mapping):
        raise ValueError("endpoint_public_summary_shape_invalid")
    if not isinstance(aggregate, Mapping):
        raise ValueError("endpoint_public_aggregate_missing")
    if dataset.get("split") != "test" or dataset.get("loaded_rows") != 45:
        raise ValueError("endpoint_public_split_invalid")
    if model_plan.get("model") != model:
        raise ValueError("endpoint_public_model_mismatch")
    if model_plan.get("reasoning_effort") != effort:
        raise ValueError("endpoint_public_effort_mismatch")
    if aggregate.get("errors") != 0 or aggregate.get("evaluated") != 45:
        raise ValueError("endpoint_public_evaluation_incomplete")


def _validate_clean_strong_public_binding(
    summary: Mapping[str, Any], *, clean_root: Path
) -> None:
    private_artifacts = summary.get("private_artifacts")
    execution_contract = summary.get("execution_contract")
    if summary.get("decision") != "completed":
        raise ValueError("strong_clean_public_summary_incomplete")
    if any(
        field in summary
        for field in (
            "merge_revision",
            "retry_public_summary",
            "source_public_summary",
            "transient_error_ordinals_replaced",
        )
    ):
        raise ValueError("strong_clean_public_merge_overlay_forbidden")
    if (
        not isinstance(private_artifacts, Mapping)
        or private_artifacts.get("raw_rows") != 45
        or private_artifacts.get("raw_root_label")
        != Path(clean_root).resolve().parent.name
    ):
        raise ValueError("strong_clean_public_private_root_mismatch")
    if (
        not isinstance(execution_contract, Mapping)
        or isinstance(execution_contract.get("provider_max_attempts"), bool)
        or not isinstance(execution_contract.get("provider_max_attempts"), int)
        or not 1 <= execution_contract["provider_max_attempts"] <= 3
        or execution_contract.get("stop_on_first_infrastructure_error") is not True
    ):
        raise ValueError("strong_clean_public_execution_contract_invalid")


def _load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for file_path in sorted(Path(path).glob("row_*.json")):
        row = _read_json(file_path)
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("endpoint_row_ordinal_invalid")
        if ordinal in rows:
            raise ValueError("endpoint_row_ordinal_duplicate")
        row["_source_sha256"] = _file_sha256(file_path)
        rows[ordinal] = row
    return rows


def _validate_rows(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    registry: Any,
    model: str,
    effort: str,
) -> None:
    if set(rows) != set(range(1, 46)):
        raise ValueError("endpoint_row_membership_invalid")
    for task in registry.tasks:
        row = rows[task.native_ordinal]
        if (
            row.get("ordinal") != task.native_ordinal
            or row.get("test_id") != task.native_test_id
            or row.get("service") != task.service
            or row.get("model") != model
            or row.get("reasoning_effort") != effort
            or row.get("status") != "evaluated"
        ):
            raise ValueError("endpoint_row_binding_invalid")
        agent = row.get("agent")
        evaluation = row.get("evaluation")
        if not isinstance(agent, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("endpoint_row_receipt_missing")
        usage = agent.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError("endpoint_row_usage_missing")
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            if isinstance(usage.get(field), bool) or not isinstance(
                usage.get(field), int
            ):
                raise ValueError("endpoint_row_usage_incomplete")
        if isinstance(agent.get("provider_call_count"), bool) or not isinstance(
            agent.get("provider_call_count"), int
        ):
            raise ValueError("endpoint_row_provider_calls_invalid")


def _load_scorer() -> tuple[Callable[..., dict[str, Any]], str]:
    spec = importlib.util.spec_from_file_location("papc_agent_diff_scoring", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("agent_diff_scorer_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_registered_scorer_corrections, module.SCORER_NAME


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    apply_corrections: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    raw_passed = 0
    corrected_passed = 0
    correction_count = 0
    calls = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cost = 0.0
    model_counts: Counter[str] = Counter()
    for row in rows:
        evaluation = row["evaluation"]
        raw = bool(evaluation.get("passed"))
        corrected = apply_corrections(
            test_id=row["test_id"], evaluation=evaluation
        )
        corrected_value = bool(corrected.get("passed"))
        raw_passed += raw
        corrected_passed += corrected_value
        correction_count += int(corrected_value and not raw)
        agent = row["agent"]
        usage = agent["usage"]
        calls += int(agent["provider_call_count"])
        input_tokens += int(usage["input_tokens"])
        output_tokens += int(usage["output_tokens"])
        reasoning_tokens += int(usage["reasoning_tokens"])
        model = str(row["model"])
        model_counts[model] += 1
        price = FROZEN_CACHE_NEUTRAL_PRICES[model]
        cost += (
            int(usage["input_tokens"]) * price.input_per_million_usd
            + int(usage["output_tokens"]) * price.output_per_million_usd
        ) / 1_000_000
    return {
        "rows": len(rows),
        "raw_official_passed": raw_passed,
        "registered_corrected_passed": corrected_passed,
        "registered_correction_count": correction_count,
        "registered_corrected_accuracy": round(corrected_passed / len(rows), 9),
        "provider_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_neutral_cost_usd": round(cost, 9),
        "usage_complete": True,
        "model_counts": dict(model_counts),
    }


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    clean_strong_root = getattr(args, "agent_clean_strong_root", None)
    if clean_strong_root is None:
        raise RuntimeError(
            "agent_diff_requires_fresh_clean_full45_strong_source;"
            "legacy_merged_strong_usage_incomplete"
        )
    clean_strong_root = Path(clean_strong_root).resolve()
    policy = _read_json(args.policy_json)
    if (
        policy.get("schema") != POLICY_SCHEMA
        or policy.get("decision") != "frozen_before_outcome_replay"
        or policy.get("provider_calls") != 0
        or policy.get("outcome_firewall", {}).get("official_test_outcomes_read")
        is not False
    ):
        raise ValueError("routellm_policy_not_result_blind_frozen")
    registry = load_agent_diff_official_registry(
        args.agent_diff_dataset,
        split_gate_json=args.split_gate_json,
    )
    if policy.get("test_registry_sha256") != registry.split.test_registry_sha256:
        raise ValueError("routellm_policy_test_registry_mismatch")

    cheap_public = _read_json(args.cheap_public_json)
    strong_public = _read_json(args.strong_public_json)
    _validate_public_summary(cheap_public, model="gpt-5.4-mini", effort="high")
    _validate_public_summary(strong_public, model="gpt-5.5", effort="high")
    _validate_clean_strong_public_binding(
        strong_public,
        clean_root=clean_strong_root,
    )
    cheap = _load_rows(args.cheap_private_rows)
    strong = _load_rows(clean_strong_root)
    expected_ordinals = set(range(1, registry.denominator + 1))
    strong_provider_retry_count = _validate_agent_clean_strong_source(
        strong,
        expected_ordinals=expected_ordinals,
    )
    _validate_rows(
        cheap,
        registry=registry,
        model="gpt-5.4-mini",
        effort="high",
    )
    _validate_rows(
        strong,
        registry=registry,
        model="gpt-5.5",
        effort="high",
    )
    for ordinal in range(1, 46):
        if cheap[ordinal].get("public_key") != strong[ordinal].get("public_key"):
            raise ValueError("endpoint_row_public_key_mismatch")

    routes = policy.get("routes")
    if not isinstance(routes, list) or len(routes) != 45:
        raise ValueError("routellm_policy_routes_invalid")
    selected_rows: list[dict[str, Any]] = []
    selected_by_role: dict[str, list[dict[str, Any]]] = {
        "cheap": [],
        "strong": [],
    }
    for route, task in zip(routes, registry.tasks, strict=True):
        if (
            route.get("native_ordinal") != task.native_ordinal
            or route.get("original_request_sha256")
            != hashlib.sha256(task.original_request.encode("utf-8")).hexdigest()
        ):
            raise ValueError("routellm_policy_route_binding_invalid")
        role = route.get("selected_endpoint")
        if role not in selected_by_role:
            raise ValueError("routellm_policy_endpoint_invalid")
        row = (strong if role == "strong" else cheap)[task.native_ordinal]
        selected_rows.append(row)
        selected_by_role[role].append(row)

    apply_corrections, scorer_name = _load_scorer()
    if scorer_name != AGENT_DIFF_SCORER_NAME:
        raise ValueError("agent_diff_registered_scorer_name_mismatch")
    routed = _summarize_rows(
        selected_rows,
        apply_corrections=apply_corrections,
    )
    cheap_selected = _summarize_rows(
        selected_by_role["cheap"],
        apply_corrections=apply_corrections,
    )
    strong_selected = _summarize_rows(
        selected_by_role["strong"],
        apply_corrections=apply_corrections,
    )
    cheap_full = _summarize_rows(
        list(cheap.values()),
        apply_corrections=apply_corrections,
    )
    strong_full = _summarize_rows(
        list(strong.values()),
        apply_corrections=apply_corrections,
    )
    if (
        cheap_full["raw_official_passed"]
        != cheap_public["aggregate"]["passed"]
        or strong_full["raw_official_passed"]
        != strong_public["aggregate"]["passed"]
    ):
        raise ValueError("endpoint_public_private_score_mismatch")
    public_reasoning_legacy_zero: dict[str, bool] = {}
    for endpoint, summary, public in (
        ("cheap", cheap_full, cheap_public),
        ("strong", strong_full, strong_public),
    ):
        resource = public["agent_resource"]
        if (
            summary["provider_calls"] != resource["provider_calls"]
            or summary["input_tokens"] != resource["usage"]["input_tokens"]
            or summary["output_tokens"] != resource["usage"]["output_tokens"]
        ):
            raise ValueError("endpoint_public_private_usage_mismatch")
        public_reasoning = resource["usage"]["reasoning_tokens"]
        raw_reasoning = summary["reasoning_tokens"]
        legacy_zero = public_reasoning == 0 and raw_reasoning > 0
        if public_reasoning != raw_reasoning and not legacy_zero:
            raise ValueError("endpoint_public_private_usage_mismatch")
        public_reasoning_legacy_zero[endpoint] = legacy_zero

    return {
        "schema": RESULT_SCHEMA,
        "decision": "completed_zero_provider_replay",
        "row_id": "routellm_bert_released_50pct_budget",
        "method": "RouteLLM-BERT (released, 50% strong budget)",
        "surface": "agent_diff",
        "split": "official_test",
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.5",
        "strong_reasoning_effort": "high",
        "official_test_count": 45,
        "official_metric": "accuracy",
        "official_scorer": scorer_name,
        "result": routed,
        "selected_endpoint_breakdown": {
            "cheap": cheap_selected,
            "strong": strong_selected,
        },
        "full_endpoint_anchors": {
            "cheap": cheap_full,
            "strong": strong_full,
        },
        "local_router": {
            "provider_calls": 0,
            "provider_cost_usd": 0.0,
            "base_bundle_sha256": ROUTELLM_BERT_BASE_BUNDLE_SHA256,
        },
        "replay_contract": {
            "policy_sha256": _file_sha256(args.policy_json),
            "policy_frozen_before_endpoint_outcomes_opened": True,
            "task_level_route_precedes_episode_environment": True,
            "endpoint_trajectories_use_exact_selected_models_and_effort": True,
            "historical_retry_overlay_used": False,
            "fresh_clean_strong_source_required": True,
            "provider_calls_added_by_reducer": 0,
            "api_spend_added_by_reducer_usd": 0.0,
        },
        "source_bindings": {
            "test_registry_sha256": registry.split.test_registry_sha256,
            "split_gate_report_sha256": registry.split_gate_report_sha256,
            "registered_scorer_sha256": _file_sha256(SCORER_PATH),
            "cheap_public_sha256": _file_sha256(args.cheap_public_json),
            "fresh_clean_strong_public_sha256": _file_sha256(
                args.strong_public_json
            ),
            "fresh_clean_strong_raw_root_label": clean_strong_root.parent.name,
            "fresh_clean_strong_full45_raw_bundle_sha256": _canonical_digest(
                sorted(row["_source_sha256"] for row in strong.values())
            ),
            "fresh_clean_strong_row_sha256s": sorted(
                row["_source_sha256"] for row in strong.values()
            ),
            "fresh_clean_strong_rows": len(strong),
            "fresh_clean_strong_provider_errors": 0,
            "fresh_clean_strong_provider_retries": strong_provider_retry_count,
            "fresh_clean_strong_usage_complete": True,
            "fresh_clean_strong_public_reasoning_legacy_zero": (
                public_reasoning_legacy_zero["strong"]
            ),
            "fresh_clean_strong_raw_reasoning_tokens": strong_full[
                "reasoning_tokens"
            ],
            "historical_retry_overlay_used": False,
            "cheap_selected_row_sha256s": sorted(
                row["_source_sha256"] for row in selected_by_role["cheap"]
            ),
            "strong_selected_row_sha256s": sorted(
                row["_source_sha256"] for row in selected_by_role["strong"]
            ),
        },
        "frozen_cache_neutral_prices_per_million": {
            model: {
                "input_usd": price.input_per_million_usd,
                "output_usd": price.output_per_million_usd,
            }
            for model, price in FROZEN_CACHE_NEUTRAL_PRICES.items()
            if model in {"gpt-5.4-mini", "gpt-5.5"}
        },
        "provider_calls_added_by_reducer": 0,
        "api_spend_performed_by_reducer": False,
        "usage_complete": True,
        "public_safe": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _freeze_policy(args) if args.mode == "freeze" else _reduce(args)
    _write_json_append_only(args.output_json, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "provider_calls_added": 0,
                "output": str(args.output_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrate released RouteLLM on STATE dev16 and replay full test endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.routellm_bert_upstream import (  # noqa: E402
    ROUTELLM_BERT_BASE_BUNDLE_SHA256,
    ROUTELLM_BERT_BASE_REPO,
    ROUTELLM_BERT_BASE_REVISION,
    RouteLLMBertScorer,
)
from papc.baselines.routellm_mf_upstream import (  # noqa: E402
    RouteLLMMFDevExample,
    freeze_dev_threshold,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    STATE_BENCH,
    STRONG_MODEL,
    RequestRouterView,
    SplitManifest,
)
from papc.baselines.state_bench_native_runtime import (  # noqa: E402
    load_state_bench_native_api,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    load_state_bench_official_test_registry,
)
from papc.paths import DATA_ROOT

CALIBRATION_SCHEMA = "papc.routellm.state.dev16_calibration.v1"
POLICY_SCHEMA = "papc.routellm.state.dev16_test_policy.v1"
PRIVATE_POLICY_SCHEMA = "papc.routellm.state.dev16_test_policy_private.v1"
RESULT_SCHEMA = "papc.routellm.state.dev16_replay.v1"
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/shared_endpoint_dev16_calibration_v1"
)
DEFAULT_BASE_MODEL = Path(
    f"{DATA_ROOT}/papc/upstream_models/"
    "routellm-bert-gpt4-augmented-f625a9c"
)
CHEAP_ROOTS = {
    domain: Path(
        f"{DATA_ROOT}/papc/g8733_direct_gpt54mini_86_full_private_20260616"
    )
    / domain
    / "official_weak_only_full_agent__gpt_5_4_mini"
    / "official_full_agent_trajectories/run1"
    for domain in ("customer_support", "shopping_assistant", "travel")
}
STRONG_ROOTS = {
    "customer_support": Path(
        f"{DATA_ROOT}/papc/state_bench_full_test_gpt55_high_baseline_20260623/"
        "customer_support/strong_gpt55_high/official_full_agent_trajectories/run1"
    ),
    "shopping_assistant": Path(
        f"{DATA_ROOT}/papc/state_bench_full_test_gpt55_high_baseline_20260623/"
        "shopping_assistant/strong_gpt55_high/official_full_agent_trajectories/run1"
    ),
    "travel": Path(
        f"{DATA_ROOT}/papc/state_bench_travel_full_test_telemetry_20260623/"
        "strong_gpt55_high/official_full_agent_trajectories/run1"
    ),
}
EXPECTED_FULL = {
    "cheap": {
        "correct": 72,
        "tokens": (4_384_210, 1_596_928, 462_049, 355_090),
    },
    "strong": {
        "correct": 92,
        "tokens": (3_671_931, 2_306_560, 229_843, 130_945),
        "calls": 1_037,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "freeze", "reduce"), required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--public-policy-json", type=Path)
    parser.add_argument("--private-policy-json", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    return parser.parse_args(argv)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"json_object_required:{path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != 16 or any(not isinstance(value, Mapping) for value in values):
        raise ValueError(f"state_dev16_rows_invalid:{path}")
    return tuple(dict(value) for value in values)


def _write_new(path: Path | None, value: Mapping[str, Any], *, private: bool) -> None:
    if path is None:
        raise ValueError("required_output_path_missing")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600 if private else 0o644,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(dict(value)))


def _agent_cost(row: Mapping[str, Any]) -> float:
    raw = row.get("receipts")
    if not isinstance(raw, list):
        raise ValueError("state_dev16_receipts_missing")
    accounting = aggregate_online_accounting(
        tuple(CallReceipt(**item) for item in raw),
        prices=FROZEN_CACHE_NEUTRAL_PRICES,
    )
    value = accounting["agent_side"]["cost_usd"]
    if value is None:
        raise ValueError("state_dev16_agent_cost_incomplete")
    return float(value)


def _scorer(args: argparse.Namespace) -> RouteLLMBertScorer:
    return RouteLLMBertScorer.from_released_base_bundle(args.base_model.resolve())


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    gate_path = (
        REPO_ROOT / "outputs/papc/shared_dev16_state_bench_gate_noapi_20260802_v1.json"
    )
    gate = _read(gate_path)
    if (
        gate.get("selected_count") != 16
        or gate.get("official_test_outcomes_read") is not False
        or gate.get("runtime_routing_uses_task_or_example_id") is not False
    ):
        raise ValueError("state_dev16_gate_not_qualified")
    selected = gate.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != 16:
        raise ValueError("state_dev16_membership_invalid")
    cheap_path = args.private_root / "state_bench/cheap/dev16_rows.jsonl"
    strong_path = args.private_root / "state_bench/strong/dev16_rows.jsonl"
    cheap_rows = _read_jsonl(cheap_path)
    strong_rows = _read_jsonl(strong_path)
    selection_sha = gate["source_authority"]["selection_sha256"]
    scorer = _scorer(args)
    examples = []
    score_rows = []
    local_receipts = []
    for expected, cheap, strong in zip(
        selected, cheap_rows, strong_rows, strict=True
    ):
        for row, endpoint, model in (
            (cheap, "cheap", CHEAP_MODEL),
            (strong, "strong", STRONG_MODEL),
        ):
            if (
                row.get("ordinal") != expected["ordinal"]
                or row.get("example_sha256") != expected["example_sha256"]
                or row.get("payload_sha256") != expected["payload_sha256"]
                or row.get("selection_sha256") != selection_sha
                or row.get("endpoint_role") != endpoint
                or row.get("model") != model
            ):
                raise ValueError("state_dev16_endpoint_binding_mismatch")
        if cheap.get("original_request") != strong.get("original_request"):
            raise ValueError("state_dev16_request_pairing_mismatch")
        prediction = scorer(
            RequestRouterView(original_request=cheap["original_request"])
        )
        local_receipts.extend(prediction.receipts)
        examples.append(
            RouteLLMMFDevExample(
                example_sha256=cheap["example_sha256"],
                predicted_strong_win_rate=prediction.value,
                cheap_official_quality=float(cheap["official_score"]),
                strong_official_quality=float(strong["official_score"]),
                cheap_cost_usd=_agent_cost(cheap),
                strong_cost_usd=_agent_cost(strong),
            )
        )
        score_rows.append(
            {
                "ordinal": cheap["ordinal"],
                "request_sha256": hashlib.sha256(
                    cheap["original_request"].encode()
                ).hexdigest(),
                "predicted_strong_win_rate": prediction.value,
                "cheap_quality": cheap["official_score"],
                "strong_quality": strong["official_score"],
            }
        )
    manifest = SplitManifest(
        role="router_dev",
        registry_sha256=selection_sha,
        example_sha256s=tuple(row["example_sha256"] for row in selected),
    )
    calibration = freeze_dev_threshold(examples, dev_manifest=manifest)
    local = aggregate_online_accounting(
        local_receipts, prices=FROZEN_CACHE_NEUTRAL_PRICES
    )
    record = {
        "schema": CALIBRATION_SCHEMA,
        "decision": "frozen_before_official_test_outcomes",
        "method": "RouteLLM-BERT (released)",
        "surface": STATE_BENCH,
        "router_visible_fields": ["original_request"],
        "base_repo": ROUTELLM_BERT_BASE_REPO,
        "base_revision": ROUTELLM_BERT_BASE_REVISION,
        "base_bundle_sha256": ROUTELLM_BERT_BASE_BUNDLE_SHA256,
        "calibration": calibration.public_record(),
        "calibration_sha256": calibration.sha256,
        "dev_score_rows": score_rows,
        "dev_score_rows_sha256": _digest(score_rows),
        "dev_gate_file_sha256": _file_sha256(gate_path),
        "cheap_dev_rows_file_sha256": _file_sha256(cheap_path),
        "strong_dev_rows_file_sha256": _file_sha256(strong_path),
        "runtime_tie_break_uses_task_or_example_id": False,
        "dev_labels_use": "threshold_calibration_only",
        "official_test_rows_read": False,
        "official_test_outcomes_read": False,
        "local_router_accounting": local,
        "provider_calls": 0,
        "secret_reads": 0,
    }
    record["calibration_policy_sha256"] = _digest(record)
    _write_new(args.calibration_json, record, private=False)
    return record


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.public_policy_json is None or args.private_policy_json is None:
        raise ValueError("policy_paths_required")
    calibration = _read(args.calibration_json)
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("decision") != "frozen_before_official_test_outcomes"
        or calibration.get("runtime_tie_break_uses_task_or_example_id") is not False
    ):
        raise ValueError("routellm_state_calibration_not_qualified")
    threshold = calibration["calibration"]["threshold"]
    api = load_state_bench_native_api(args.state_repo)
    registry = load_state_bench_official_test_registry(
        args.state_repo, split_gate_json=args.split_gate_json
    )
    scorer = _scorer(args)
    decisions = []
    receipts = []
    for ordinal, task_ref in enumerate(registry.tasks, start=1):
        task = api.task_definition_class.load(task_ref.native_task_path)
        prediction = scorer(RequestRouterView(original_request=task.opening_message))
        receipts.extend(prediction.receipts)
        decisions.append(
            {
                "ordinal": ordinal,
                "request_sha256": hashlib.sha256(
                    task.opening_message.encode()
                ).hexdigest(),
                "predicted_strong_win_rate": prediction.value,
                "route": "strong" if prediction.value >= threshold else "cheap",
            }
        )
    private = {
        "schema": PRIVATE_POLICY_SCHEMA,
        "decisions": decisions,
        "decision_manifest_sha256": _digest(decisions),
        "decision_function": "released_bert(request)>=dev_threshold",
        "runtime_decision_fields": ["original_request"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "official_test_outcomes_read": False,
        "provider_calls": 0,
    }
    _write_new(args.private_policy_json, private, private=True)
    counts = Counter(item["route"] for item in decisions)
    local = aggregate_online_accounting(receipts, prices=FROZEN_CACHE_NEUTRAL_PRICES)
    public = {
        "schema": POLICY_SCHEMA,
        "decision": "frozen_before_endpoint_outcomes",
        "method": "RouteLLM-BERT (released + dev16 threshold)",
        "surface": STATE_BENCH,
        "denominator": len(decisions),
        "cheap_count": counts["cheap"],
        "strong_count": counts["strong"],
        "threshold": threshold,
        "route_rule": "predicted_strong_win_rate >= threshold",
        "runtime_decision_fields": ["original_request"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "calibration_policy_sha256": calibration["calibration_policy_sha256"],
        "calibration_policy_file_sha256": _file_sha256(args.calibration_json),
        "test_registry_sha256": registry.test_registry_sha256,
        "private_policy_sha256": _digest(private),
        "private_policy_file_sha256": _file_sha256(args.private_policy_json),
        "local_router_accounting": local,
        "official_test_outcomes_read": False,
        "provider_calls": 0,
        "secret_reads": 0,
    }
    public["policy_sha256"] = _digest(public)
    _write_new(args.public_policy_json, public, private=False)
    return public


def _tokens(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    usage = row.get("token_usage") or {}
    return tuple(
        int(usage.get(key) or 0)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    )


def _cost(model: str, tokens: tuple[int, int, int, int]) -> float:
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    return (
        tokens[0] * price.input_per_million_usd
        + tokens[2] * price.output_per_million_usd
    ) / 1_000_000


def _state_row(root: Path, locator: str) -> dict[str, Any]:
    path = Path(root).resolve() / f"{locator}.json"
    return {**_read(path), "_source_sha256": _file_sha256(path)}


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.public_policy_json is None
        or args.private_policy_json is None
        or args.result_json is None
    ):
        raise ValueError("policy_and_result_paths_required")
    public = _read(args.public_policy_json)
    private = _read(args.private_policy_json)
    if (
        public.get("schema") != POLICY_SCHEMA
        or private.get("schema") != PRIVATE_POLICY_SCHEMA
        or public.get("private_policy_sha256") != _digest(private)
        or public.get("private_policy_file_sha256")
        != _file_sha256(args.private_policy_json)
        or private.get("official_test_outcomes_read") is not False
        or private.get("runtime_tie_break_uses_task_or_example_id") is not False
    ):
        raise ValueError("routellm_state_policy_binding_mismatch")
    api = load_state_bench_native_api(args.state_repo)
    registry = load_state_bench_official_test_registry(
        args.state_repo, split_gate_json=args.split_gate_json
    )
    decisions = private.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(registry.tasks):
        raise ValueError("routellm_state_decisions_invalid")
    correct = 0
    total_cost = 0.0
    strong_calls = 0
    full_strong_calls = 0
    selected_tokens = [0, 0, 0, 0]
    full_quality = {"cheap": 0, "strong": 0}
    full_tokens = {"cheap": [0, 0, 0, 0], "strong": [0, 0, 0, 0]}
    source_hashes = []
    selected_manifest = []
    for ordinal, (task_ref, decision) in enumerate(
        zip(registry.tasks, decisions, strict=True), start=1
    ):
        task = api.task_definition_class.load(task_ref.native_task_path)
        request_sha = hashlib.sha256(task.opening_message.encode()).hexdigest()
        if (
            decision.get("ordinal") != ordinal
            or decision.get("request_sha256") != request_sha
            or decision.get("route") not in {"cheap", "strong"}
        ):
            raise ValueError("routellm_state_decision_join_mismatch")
        rows = {
            "cheap": _state_row(
                CHEAP_ROOTS[task_ref.domain], task_ref.native_task_locator
            ),
            "strong": _state_row(
                STRONG_ROOTS[task_ref.domain], task_ref.native_task_locator
            ),
        }
        for endpoint, row, model in (
            ("cheap", rows["cheap"], CHEAP_MODEL),
            ("strong", rows["strong"], STRONG_MODEL),
        ):
            conversation = row.get("conversation") or []
            if (
                row.get("task_id") != task.task_id
                or row.get("agent_model")
                != {"model_name": model, "reasoning_level": "high"}
                or not conversation
                or conversation[0].get("content") != task.opening_message
                or row.get("task_completion_pass") not in {0, 1}
            ):
                raise ValueError(f"state_{endpoint}_source_binding_mismatch")
            full_quality[endpoint] += int(row["task_completion_pass"])
            for index, value in enumerate(_tokens(row)):
                full_tokens[endpoint][index] += value
            if endpoint == "strong":
                ledger = row.get("papc_runtime_ledger") or {}
                if (
                    ledger.get("provider_error_calls") != 0
                    or ledger.get("calls") != ledger.get("provider_ok_calls")
                ):
                    raise ValueError("state_strong_provider_error")
                full_strong_calls += int(ledger["calls"])
            source_hashes.append(row["_source_sha256"])
        route = decision["route"]
        chosen = rows[route]
        tokens = _tokens(chosen)
        for index, value in enumerate(tokens):
            selected_tokens[index] += value
        total_cost += _cost(
            CHEAP_MODEL if route == "cheap" else STRONG_MODEL, tokens
        )
        if route == "strong":
            ledger = chosen.get("papc_runtime_ledger") or {}
            strong_calls += int(ledger["calls"])
        score = int(chosen["task_completion_pass"])
        correct += score
        selected_manifest.append(
            {
                "ordinal": ordinal,
                "request_sha256": request_sha,
                "route": route,
                "official_score": score,
            }
        )
    for endpoint in ("cheap", "strong"):
        expected = EXPECTED_FULL[endpoint]
        if (
            full_quality[endpoint] != expected["correct"]
            or tuple(full_tokens[endpoint]) != expected["tokens"]
        ):
            raise ValueError(f"state_{endpoint}_full_anchor_mismatch")
    if full_strong_calls != EXPECTED_FULL["strong"]["calls"]:
        raise ValueError("state_strong_full_call_anchor_mismatch")
    counts = Counter(item["route"] for item in decisions)
    result = {
        "schema": RESULT_SCHEMA,
        "decision": "completed_zero_provider_endpoint_replay",
        "method": "RouteLLM-BERT (released + dev16 threshold)",
        "surface": STATE_BENCH,
        "correct": correct,
        "denominator": len(decisions),
        "accuracy": correct / len(decisions),
        "metric": "task_completion_pass@1",
        "cheap_count": counts["cheap"],
        "strong_count": counts["strong"],
        "cache_neutral_agent_side_cost_usd": total_cost,
        "agent_side_provider_calls": None,
        "selected_strong_provider_calls": strong_calls,
        "cheap_call_telemetry_note": (
            "historical cheap rows predate per-call ledger; tokens and cost are exact"
        ),
        "selected_tokens": {
            "input": selected_tokens[0],
            "cached_input": selected_tokens[1],
            "output": selected_tokens[2],
            "reasoning": selected_tokens[3],
        },
        "full_endpoint_quality_anchors": full_quality,
        "policy_sha256": public["policy_sha256"],
        "policy_file_sha256": _file_sha256(args.public_policy_json),
        "selected_outcome_manifest_sha256": _digest(selected_manifest),
        "source_file_bundle_sha256": _digest(sorted(source_hashes)),
        "runtime_decision_fields": ["original_request"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "policy_frozen_before_endpoint_outcomes": True,
        "unresolved_infrastructure_failures": 0,
        "provider_calls_added_by_replay": 0,
        "secret_reads": 0,
    }
    result["result_sha256"] = _digest(result)
    _write_new(args.result_json, result, private=False)
    return result


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.mode == "calibrate":
        return _calibrate(args)
    if args.mode == "freeze":
        return _freeze(args)
    return _reduce(args)


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(argv)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - exact provider-zero blocker report.
        print(
            json.dumps(
                {
                    "schema": RESULT_SCHEMA,
                    "decision": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc) or type(exc).__name__,
                    "provider_calls": 0,
                    "secret_reads": 0,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

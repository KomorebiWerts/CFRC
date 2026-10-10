#!/usr/bin/env python3
"""Freeze outcome-blind FrugalGPT test routes, then replay endpoints."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.agent_diff_router_driver import (  # noqa: E402
    load_agent_diff_official_registry,
)
from papc.baselines.frugalgpt_upstream import (  # noqa: E402
    FrugalGPTDistilBertScorer,
    FrugalGPTScorerArtifact,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
)
from papc.baselines.router_contract import (  # noqa: E402
    AGENT_DIFF,
    CHEAP_MODEL,
    STATE_BENCH,
    STRONG_MODEL,
    CheapCompletionRouterView,
)
from papc.baselines.state_bench_native_runtime import (  # noqa: E402
    load_state_bench_native_api,
    state_bench_generated_answer_view,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    load_state_bench_official_test_registry,
)
from papc.paths import DATA_ROOT
from scripts.agent_diff.papc_agent_diff_scoring import (  # noqa: E402
    SCORER_NAME,
    apply_registered_scorer_corrections,
)

POLICY_SCHEMA = "papc.frugalgpt.two_surface_test_policy.v1"
PRIVATE_POLICY_SCHEMA = "papc.frugalgpt.two_surface_test_policy_private.v1"
RESULT_SCHEMA = "papc.frugalgpt.two_surface_test_replay.v1"
SURFACES = (STATE_BENCH, AGENT_DIFF)
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_RETAIL_ARTIFACT = Path(
    f"{DATA_ROOT}/papc/frugalgpt_retail_v1/artifact.json"
)
DEFAULT_RETAIL_CHECKPOINT = Path(
    f"{DATA_ROOT}/papc/frugalgpt_retail_v1/checkpoint"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_AGENT_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/datasets/agent-diff-bench"
)
AGENT_CHEAP_ROOT = Path(
    f"{DATA_ROOT}/papc/agent_diff_hf_test45_cheap_baseline_private_"
    "20260706_codex_cheap45_v1/raw_rows"
)
STATE_CHEAP_ROOTS = {
    domain: Path(
        f"{DATA_ROOT}/papc/g8733_direct_gpt54mini_86_full_private_20260616"
    )
    / domain
    / "official_weak_only_full_agent__gpt_5_4_mini"
    / "official_full_agent_trajectories/run1"
    for domain in ("customer_support", "shopping_assistant", "travel")
}
STATE_STRONG_ROOTS = {
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
STATE_EXPECTED_FULL = {
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
    parser.add_argument("--mode", choices=("freeze", "reduce"), required=True)
    parser.add_argument("--surface", choices=SURFACES, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--public-policy-json", type=Path, required=True)
    parser.add_argument("--private-policy-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--retail-artifact", type=Path, default=DEFAULT_RETAIL_ARTIFACT)
    parser.add_argument(
        "--retail-checkpoint", type=Path, default=DEFAULT_RETAIL_CHECKPOINT
    )
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--agent-dataset", type=Path, default=DEFAULT_AGENT_DATASET)
    parser.add_argument("--agent-clean-strong-root", type=Path)
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


def _write_new(path: Path, value: Mapping[str, Any], *, private: bool) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600 if private else 0o644,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(dict(value)))


def _artifact_and_scorer(args: argparse.Namespace) -> tuple[Any, Any]:
    artifact = FrugalGPTScorerArtifact.from_public_record(
        _read(args.retail_artifact)
    )
    if artifact.surface != "tau2_retail":
        raise ValueError("frugalgpt_scorer_not_retail_trained")
    scorer = FrugalGPTDistilBertScorer.load(
        artifact=artifact,
        checkpoint_dir=args.retail_checkpoint.resolve(),
    )
    return artifact, scorer


def _agent_raw_rows(root: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for path in sorted(Path(root).resolve().glob("row_*.json")):
        row = _read(path)
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("agent_diff_row_ordinal_invalid")
        if ordinal in rows:
            raise ValueError("agent_diff_row_ordinal_duplicate")
        rows[ordinal] = {**row, "_source_path": str(path), "_source_sha256": _file_sha256(path)}
    return rows


def _validate_agent_clean_strong_source(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    expected_ordinals: set[int],
) -> int:
    if set(rows) != expected_ordinals:
        raise ValueError("agent_diff_clean_strong_denominator_mismatch")
    provider_retry_count = 0
    for ordinal in sorted(rows):
        row = rows[ordinal]
        if (
            row.get("status") != "evaluated"
            or row.get("error") is not None
            or row.get("error_type") is not None
            or not isinstance(row.get("evaluation"), Mapping)
        ):
            raise ValueError("agent_diff_clean_strong_endpoint_error")
        agent = row.get("agent")
        usage = agent.get("usage") if isinstance(agent, Mapping) else None
        if not isinstance(usage, Mapping):
            raise ValueError("agent_diff_clean_strong_usage_incomplete")
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("agent_diff_clean_strong_usage_incomplete")
        provider_calls = agent.get("provider_call_count")
        trace = agent.get("trace")
        if (
            isinstance(provider_calls, bool)
            or not isinstance(provider_calls, int)
            or provider_calls <= 0
            or not isinstance(trace, list)
            or len(trace) != provider_calls
        ):
            raise ValueError("agent_diff_clean_strong_usage_incomplete")
        trace_input_tokens = 0
        trace_output_tokens = 0
        trace_reasoning_tokens = 0
        for event in trace:
            attempts = event.get("provider_attempts") if isinstance(event, Mapping) else None
            if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
                raise ValueError("agent_diff_clean_strong_retry_or_provider_error")
            for attempt_index, attempt in enumerate(attempts, start=1):
                is_final = attempt_index == len(attempts)
                if not isinstance(attempt, Mapping) or attempt.get("attempt") != attempt_index:
                    raise ValueError("agent_diff_clean_strong_retry_or_provider_error")
                if is_final:
                    valid = attempt.get("ok") is True and attempt.get("status") == 200
                else:
                    valid = attempt.get("ok") is False
                if not valid:
                    raise ValueError("agent_diff_clean_strong_retry_or_provider_error")
            provider_retry_count += len(attempts) - 1
            event_usage = event.get("usage")
            if not isinstance(event_usage, Mapping):
                raise ValueError("agent_diff_clean_strong_usage_incomplete")
            trace_input_tokens += _required_trace_usage_int(
                event_usage,
                primary="input_tokens",
                alternate="prompt_tokens",
            )
            trace_output_tokens += _required_trace_usage_int(
                event_usage,
                primary="output_tokens",
                alternate="completion_tokens",
            )
            trace_reasoning_tokens += _trace_reasoning_tokens(event_usage)
        if (
            trace_input_tokens != usage["input_tokens"]
            or trace_output_tokens != usage["output_tokens"]
            or trace_reasoning_tokens != usage["reasoning_tokens"]
        ):
            raise ValueError("agent_diff_clean_strong_usage_incomplete")
    return provider_retry_count


def _required_trace_usage_int(
    usage: Mapping[str, Any],
    *,
    primary: str,
    alternate: str,
) -> int:
    value = usage.get(primary)
    if value is None:
        value = usage.get(alternate)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("agent_diff_clean_strong_usage_incomplete")
    return value


def _trace_reasoning_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("output_tokens_details")
    if details is None:
        details = usage.get("completion_tokens_details")
    if details is None:
        return 0
    if not isinstance(details, Mapping):
        raise ValueError("agent_diff_clean_strong_usage_incomplete")
    value = details.get("reasoning_tokens")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("agent_diff_clean_strong_usage_incomplete")
    return value


def _agent_sources(args: argparse.Namespace, *, include_outcomes: bool) -> tuple[Any, list[dict[str, Any]]]:
    registry = load_agent_diff_official_registry(
        args.agent_dataset,
        split_gate_json=args.split_gate_json,
    )
    cheap_rows = _agent_raw_rows(AGENT_CHEAP_ROOT)
    strong_rows = {}
    if include_outcomes:
        clean_root = getattr(args, "agent_clean_strong_root", None)
        if clean_root is None:
            raise RuntimeError(
                "agent_diff_requires_fresh_clean_full45_strong_source;"
                "legacy_merged_strong_usage_incomplete"
            )
        strong_rows = _agent_raw_rows(clean_root)
    expected = set(range(1, registry.denominator + 1))
    if set(cheap_rows) != expected:
        raise ValueError("agent_diff_endpoint_denominator_mismatch")
    if include_outcomes:
        _validate_agent_clean_strong_source(
            strong_rows,
            expected_ordinals=expected,
        )
    sources = []
    for task in registry.tasks:
        cheap = cheap_rows[task.native_ordinal]
        cheap_agent = cheap.get("agent")
        if (
            cheap.get("test_id") != task.native_test_id
            or cheap.get("model") != CHEAP_MODEL
            or cheap.get("reasoning_effort") != "high"
            or cheap.get("status") != "evaluated"
            or not isinstance(cheap_agent, Mapping)
            or cheap_agent.get("completed") is not True
        ):
            raise ValueError("agent_diff_cheap_endpoint_binding_mismatch")
        trace = cheap_agent.get("trace") or []
        answer = trace[-1].get("assistant_text") if trace else None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("agent_diff_cheap_public_answer_missing")
        record = {
            "ordinal": task.native_ordinal,
            "request": task.original_request,
            "request_sha256": hashlib.sha256(task.original_request.encode()).hexdigest(),
            "cheap_answer": answer,
            "cheap_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "cheap": cheap,
        }
        if include_outcomes:
            strong = strong_rows[task.native_ordinal]
            if (
                strong.get("test_id") != task.native_test_id
                or strong.get("model") != STRONG_MODEL
                or strong.get("reasoning_effort") != "high"
            ):
                raise ValueError("agent_diff_strong_endpoint_binding_mismatch")
            record["strong"] = strong
        sources.append(record)
    return registry, sources


def _state_raw_row(root: Path, locator: str) -> dict[str, Any]:
    path = Path(root).resolve() / f"{locator}.json"
    return {**_read(path), "_source_path": str(path), "_source_sha256": _file_sha256(path)}


def _state_sources(args: argparse.Namespace, *, include_outcomes: bool) -> tuple[Any, list[dict[str, Any]]]:
    api = load_state_bench_native_api(args.state_repo)
    registry = load_state_bench_official_test_registry(
        args.state_repo,
        split_gate_json=args.split_gate_json,
    )
    sources = []
    for ordinal, task_ref in enumerate(registry.tasks, start=1):
        task = api.task_definition_class.load(task_ref.native_task_path)
        cheap = _state_raw_row(
            STATE_CHEAP_ROOTS[task_ref.domain], task_ref.native_task_locator
        )
        conversation = cheap.get("conversation") or []
        if (
            cheap.get("task_id") != task.task_id
            or cheap.get("agent_model")
            != {"model_name": CHEAP_MODEL, "reasoning_level": "high"}
            or not conversation
            or conversation[0].get("content") != task.opening_message
        ):
            raise ValueError("state_cheap_endpoint_binding_mismatch")
        answer = state_bench_generated_answer_view(conversation)
        record = {
            "ordinal": ordinal,
            "request": task.opening_message,
            "request_sha256": hashlib.sha256(task.opening_message.encode()).hexdigest(),
            "cheap_answer": answer,
            "cheap_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "cheap": cheap,
        }
        if include_outcomes:
            strong = _state_raw_row(
                STATE_STRONG_ROOTS[task_ref.domain], task_ref.native_task_locator
            )
            strong_conversation = strong.get("conversation") or []
            if (
                strong.get("task_id") != task.task_id
                or strong.get("agent_model")
                != {"model_name": STRONG_MODEL, "reasoning_level": "high"}
                or not strong_conversation
                or strong_conversation[0].get("content") != task.opening_message
            ):
                raise ValueError("state_strong_endpoint_binding_mismatch")
            record["strong"] = strong
        sources.append(record)
    return registry, sources


def _sources(args: argparse.Namespace, *, include_outcomes: bool) -> tuple[Any, list[dict[str, Any]]]:
    if args.surface == AGENT_DIFF:
        return _agent_sources(args, include_outcomes=include_outcomes)
    return _state_sources(args, include_outcomes=include_outcomes)


def _test_registry_sha256(registry: Any, surface: str) -> str:
    if surface == AGENT_DIFF:
        return str(registry.split.test_registry_sha256)
    return str(registry.test_registry_sha256)


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    calibration = _read(args.calibration_json)
    if (
        calibration.get("decision") != "frozen_before_official_test_outcomes"
        or calibration.get("surface") != args.surface
        or calibration.get("official_test_outcomes_read") is not False
        or calibration.get("runtime_tie_break_uses_task_or_example_id") is not False
    ):
        raise ValueError("frugalgpt_dev_calibration_not_qualified")
    threshold = calibration["calibration"]["accept_threshold"]
    artifact, scorer = _artifact_and_scorer(args)
    registry, sources = _sources(args, include_outcomes=False)
    decisions = []
    for source in sources:
        score = scorer(
            CheapCompletionRouterView(
                original_request=source["request"],
                cheap_generated_answer=source["cheap_answer"],
            )
        )
        decisions.append(
            {
                "ordinal": source["ordinal"],
                "request_sha256": source["request_sha256"],
                "cheap_answer_sha256": source["cheap_answer_sha256"],
                "predicted_cheap_quality": score,
                "route": "cheap" if score > threshold else "strong",
            }
        )
    private = {
        "schema": PRIVATE_POLICY_SCHEMA,
        "surface": args.surface,
        "decisions": decisions,
        "decision_manifest_sha256": _digest(decisions),
        "decision_function": "score(public_request,cheap_public_answer)>threshold",
        "runtime_decision_fields": ["original_request", "cheap_generated_answer"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "policy_visible_test_fields": [
            "original_request",
            "cheap_generated_answer",
        ],
        "endpoint_outcomes_enter_policy_decisions": False,
        "official_test_outcomes_enter_policy_decisions": False,
        "provider_calls": 0,
    }
    _write_new(args.private_policy_json, private, private=True)
    counts = Counter(item["route"] for item in decisions)
    public = {
        "schema": POLICY_SCHEMA,
        "decision": "frozen_before_endpoint_outcomes",
        "method": "FrugalGPT (2-model)",
        "surface": args.surface,
        "denominator": len(decisions),
        "cheap_attempt_count": len(decisions),
        "accepted_cheap_count": counts["cheap"],
        "strong_restart_count": counts["strong"],
        "accept_threshold": threshold,
        "acceptance_rule": "predicted_cheap_quality > accept_threshold",
        "execution_semantics": "cheap_complete_then_pristine_strong_restart",
        "runtime_decision_fields": ["original_request", "cheap_generated_answer"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "retail_scorer_artifact_sha256": artifact.sha256,
        "calibration_policy_sha256": calibration["policy_sha256"],
        "calibration_policy_file_sha256": _file_sha256(args.calibration_json),
        "test_registry_sha256": _test_registry_sha256(registry, args.surface),
        "private_policy_sha256": _digest(private),
        "private_policy_file_sha256": _file_sha256(args.private_policy_json),
        "policy_visible_test_fields": [
            "original_request",
            "cheap_generated_answer",
        ],
        "endpoint_outcomes_enter_policy_decisions": False,
        "official_test_outcomes_enter_policy_decisions": False,
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
    }
    public["policy_sha256"] = _digest(public)
    _write_new(args.public_policy_json, public, private=False)
    return public


def _agent_endpoint_stats(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    agent = row.get("agent") or {}
    usage = agent.get("usage") or {}
    input_tokens = int(usage.get("input_tokens"))
    output_tokens = int(usage.get("output_tokens"))
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    return {
        "provider_calls": int(agent.get("provider_call_count")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "cache_neutral_cost_usd": (
            input_tokens * price.input_per_million_usd
            + output_tokens * price.output_per_million_usd
        )
        / 1_000_000,
    }


def _state_tokens(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
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


def _state_cost(model: str, tokens: tuple[int, int, int, int]) -> float:
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    return (
        tokens[0] * price.input_per_million_usd
        + tokens[2] * price.output_per_million_usd
    ) / 1_000_000


def _reduce(args: argparse.Namespace) -> dict[str, Any]:
    if args.result_json is None:
        raise ValueError("result_json_required")
    if args.surface == AGENT_DIFF and getattr(args, "agent_clean_strong_root", None) is None:
        raise RuntimeError(
            "agent_diff_requires_fresh_clean_full45_strong_source;"
            "legacy_merged_strong_usage_incomplete"
        )
    public = _read(args.public_policy_json)
    private = _read(args.private_policy_json)
    policy_outcome_firewall = (
        public.get("official_test_outcomes_enter_policy_decisions") is False
        or public.get("official_test_outcomes_read") is False
    )
    if (
        public.get("schema") != POLICY_SCHEMA
        or private.get("schema") != PRIVATE_POLICY_SCHEMA
        or public.get("surface") != args.surface
        or public.get("private_policy_sha256") != _digest(private)
        or public.get("private_policy_file_sha256")
        != _file_sha256(args.private_policy_json)
        or not policy_outcome_firewall
        or private.get("runtime_tie_break_uses_task_or_example_id") is not False
    ):
        raise ValueError("frugalgpt_test_policy_binding_mismatch")
    registry, sources = _sources(args, include_outcomes=True)
    decisions = private.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sources):
        raise ValueError("frugalgpt_test_decisions_invalid")
    correct = 0
    cheap_cost = 0.0
    strong_cost = 0.0
    cheap_calls: int | None = 0 if args.surface == AGENT_DIFF else None
    strong_calls = 0
    source_hashes = {"cheap": [], "strong": []}
    selected_manifest = []
    full_quality = {"cheap": 0, "strong": 0}
    state_full_tokens = {"cheap": [0, 0, 0, 0], "strong": [0, 0, 0, 0]}
    for source, decision in zip(sources, decisions, strict=True):
        if (
            decision.get("ordinal") != source["ordinal"]
            or decision.get("request_sha256") != source["request_sha256"]
            or decision.get("cheap_answer_sha256") != source["cheap_answer_sha256"]
            or decision.get("route") not in {"cheap", "strong"}
        ):
            raise ValueError("frugalgpt_test_decision_join_mismatch")
        cheap = source["cheap"]
        strong = source["strong"]
        source_hashes["cheap"].append(cheap["_source_sha256"])
        source_hashes["strong"].append(strong["_source_sha256"])
        if args.surface == AGENT_DIFF:
            for endpoint, row in (("cheap", cheap), ("strong", strong)):
                evaluation = row.get("evaluation")
                if not isinstance(evaluation, Mapping):
                    raise ValueError("agent_diff_evaluation_missing")
                corrected = apply_registered_scorer_corrections(
                    test_id=str(row["test_id"]), evaluation=evaluation
                )
                full_quality[endpoint] += int(bool(corrected.get("passed")))
            cheap_stats = _agent_endpoint_stats(cheap, CHEAP_MODEL)
            strong_stats = _agent_endpoint_stats(strong, STRONG_MODEL)
            cheap_cost += cheap_stats["cache_neutral_cost_usd"]
            assert cheap_calls is not None
            cheap_calls += cheap_stats["provider_calls"]
            if decision["route"] == "strong":
                strong_cost += strong_stats["cache_neutral_cost_usd"]
                strong_calls += strong_stats["provider_calls"]
            chosen = cheap if decision["route"] == "cheap" else strong
            chosen_eval = apply_registered_scorer_corrections(
                test_id=str(chosen["test_id"]), evaluation=chosen["evaluation"]
            )
            score = int(bool(chosen_eval.get("passed")))
        else:
            for endpoint, row, model in (
                ("cheap", cheap, CHEAP_MODEL),
                ("strong", strong, STRONG_MODEL),
            ):
                if row.get("task_completion_pass") not in {0, 1}:
                    raise ValueError("state_official_score_invalid")
                full_quality[endpoint] += int(row["task_completion_pass"])
                tokens = _state_tokens(row)
                for index, value in enumerate(tokens):
                    state_full_tokens[endpoint][index] += value
                if endpoint == "strong":
                    ledger = row.get("papc_runtime_ledger") or {}
                    if (
                        ledger.get("provider_error_calls") != 0
                        or ledger.get("calls") != ledger.get("provider_ok_calls")
                    ):
                        raise ValueError("state_strong_provider_error")
            cheap_tokens = _state_tokens(cheap)
            strong_tokens = _state_tokens(strong)
            cheap_cost += _state_cost(CHEAP_MODEL, cheap_tokens)
            if decision["route"] == "strong":
                strong_cost += _state_cost(STRONG_MODEL, strong_tokens)
                strong_calls += int((strong.get("papc_runtime_ledger") or {})["calls"])
            chosen = cheap if decision["route"] == "cheap" else strong
            score = int(chosen["task_completion_pass"])
        correct += score
        selected_manifest.append(
            {
                "ordinal": source["ordinal"],
                "request_sha256": source["request_sha256"],
                "route": decision["route"],
                "official_score": score,
            }
        )
    if args.surface == AGENT_DIFF:
        if full_quality["cheap"] != 39:
            raise ValueError("agent_diff_cheap_full_endpoint_quality_anchor_mismatch")
    else:
        for endpoint in ("cheap", "strong"):
            expected = STATE_EXPECTED_FULL[endpoint]
            if (
                full_quality[endpoint] != expected["correct"]
                or tuple(state_full_tokens[endpoint]) != expected["tokens"]
            ):
                raise ValueError(f"state_{endpoint}_full_endpoint_anchor_mismatch")
    counts = Counter(item["route"] for item in decisions)
    result = {
        "schema": RESULT_SCHEMA,
        "decision": "completed_zero_provider_endpoint_replay",
        "method": "FrugalGPT (2-model)",
        "surface": args.surface,
        "denominator": len(sources),
        "correct": correct,
        "accuracy": correct / len(sources),
        "metric": (
            "registered_pass@1" if args.surface == AGENT_DIFF else "task_completion_pass@1"
        ),
        "cheap_attempt_count": len(sources),
        "accepted_cheap_count": counts["cheap"],
        "strong_restart_count": counts["strong"],
        "cache_neutral_agent_side_cost_usd": cheap_cost + strong_cost,
        "agent_side_provider_calls": (
            None if cheap_calls is None else cheap_calls + strong_calls
        ),
        "endpoint_components": {
            "cheap_all": {
                "cost_usd": cheap_cost,
                "provider_calls": cheap_calls,
                "full_endpoint_correct": full_quality["cheap"],
            },
            "selected_strong_restarts": {
                "cost_usd": strong_cost,
                "provider_calls": strong_calls,
                "count": counts["strong"],
            },
            "strong_full_quality_anchor": full_quality["strong"],
        },
        "scorer": SCORER_NAME if args.surface == AGENT_DIFF else "state_bench_official",
        "policy_sha256": public["policy_sha256"],
        "policy_file_sha256": _file_sha256(args.public_policy_json),
        "private_policy_file_sha256": _file_sha256(args.private_policy_json),
        "test_registry_sha256": _test_registry_sha256(registry, args.surface),
        "selected_outcome_manifest_sha256": _digest(selected_manifest),
        "source_file_bundle_sha256": _digest(
            sorted(source_hashes["cheap"] + source_hashes["strong"])
        ),
        "runtime_decision_fields": ["original_request", "cheap_generated_answer"],
        "runtime_tie_break_uses_task_or_example_id": False,
        "policy_frozen_before_endpoint_outcomes": True,
        "unresolved_infrastructure_failures": 0,
        "provider_calls_added_by_replay": 0,
        "secret_reads": 0,
        "api_spend_performed_by_replay": False,
        "cheap_call_telemetry_note": (
            "historical_STATE cheap rows predate per-call ledger; token cost is exact"
            if args.surface == STATE_BENCH
            else None
        ),
    }
    if args.surface == AGENT_DIFF:
        result["source_bindings"] = {
            "cheap_completed_full45_raw_bundle_sha256": _digest(
                sorted(source_hashes["cheap"])
            ),
            "fresh_clean_strong_full45_raw_bundle_sha256": _digest(
                sorted(source_hashes["strong"])
            ),
            "fresh_clean_strong_rows": len(source_hashes["strong"]),
            "fresh_clean_strong_provider_errors": 0,
            "fresh_clean_strong_provider_retries": 0,
            "fresh_clean_strong_usage_complete": True,
            "historical_retry_overlay_used": False,
        }
    result["result_sha256"] = _digest(result)
    _write_new(args.result_json, result, private=False)
    return result


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return _freeze(args) if args.mode == "freeze" else _reduce(args)


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

#!/usr/bin/env python3
"""Freeze and reduce five disjoint StepWise STATE-Bench shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SHARD_COUNT = 5
SHARD_SIZE = 30
DENOMINATOR = SHARD_COUNT * SHARD_SIZE
RULE = "zero_based_registry_ordinal_modulo_shard_count"
PREFLIGHT_SCHEMA = "papc.stepwise_state_bench_live.v1"
GATE_SCHEMA = "papc.stepwise_state_bench_shard_gate_noapi.v1"
RESULT_SCHEMA = "papc.stepwise_state_bench_shard_aggregate.v1"
PROVIDER_AGENT_ROLES = frozenset(
    {"cheap_agent", "strong_agent", "verifier", "recovery"}
)
HARNESS_ROLES = frozenset({"user_simulator", "evaluator"})
PROVIDER_ROLES = PROVIDER_AGENT_ROLES | HARNESS_ROLES


def freeze_preflights(paths: Sequence[Path]) -> dict[str, Any]:
    """Verify that five provider-zero preflights freeze an exact partition."""

    records = _indexed_records(paths, expected_count=SHARD_COUNT)
    shared_context: dict[str, Any] | None = None
    shards: list[dict[str, Any]] = []
    for index, path, payload in records:
        if (
            payload.get("schema") != PREFLIGHT_SCHEMA
            or payload.get("decision") != "ready_for_explicit_execute"
            or payload.get("execute") is not False
            or payload.get("provider_calls") != 0
            or payload.get("secret_reads") != 0
            or payload.get("official_test_outcomes_read") is not False
        ):
            raise ValueError("stepwise_state_shard_preflight_not_provider_zero")
        context = _mapping(payload.get("context"), "preflight_context")
        shard = _mapping(context.get("shard"), "preflight_shard")
        examples = _sha_list(context.get("example_sha256s"), SHARD_SIZE)
        if (
            context.get("mode") != "official_test"
            or context.get("row_count") != SHARD_SIZE
            or shard != {"count": SHARD_COUNT, "index": index, "rule": RULE}
        ):
            raise ValueError("stepwise_state_shard_preflight_contract_drift")
        candidate_shared = dict(context)
        for key in ("shard", "example_sha256s", "domains"):
            candidate_shared.pop(key, None)
        if shared_context is None:
            shared_context = candidate_shared
        elif candidate_shared != shared_context:
            raise ValueError("stepwise_state_shard_preflight_source_drift")
        shards.append(
            {
                "index": index,
                "example_sha256s": examples,
                "preflight_path": str(path.resolve()),
                "preflight_file_sha256": _file_sha256(path),
            }
        )

    reconstructed = [
        shards[ordinal % SHARD_COUNT]["example_sha256s"][ordinal // SHARD_COUNT]
        for ordinal in range(DENOMINATOR)
    ]
    if len(set(reconstructed)) != DENOMINATOR:
        raise ValueError("stepwise_state_shard_membership_not_disjoint")
    assert shared_context is not None
    return {
        "schema": GATE_SCHEMA,
        "decision": "READY_FOR_FIVE_SHARD_EXECUTION",
        "partition": {
            "rule": RULE,
            "shard_count": SHARD_COUNT,
            "rows_per_shard": SHARD_SIZE,
            "denominator": DENOMINATOR,
            "union_is_exact": True,
            "pairwise_disjoint": True,
            "registry_order_sha256": _digest(reconstructed),
        },
        "shared_context": shared_context,
        "shards": shards,
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "execute": False,
        "official_test_outcomes_read": False,
        "authorizes_result_claim": False,
        "public_safe": True,
    }


def aggregate_results(
    *,
    gate_path: Path,
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    """Reduce only five complete, error-free shards bound to one gate."""

    gate = _read_mapping(gate_path)
    if (
        gate.get("schema") != GATE_SCHEMA
        or gate.get("decision") != "READY_FOR_FIVE_SHARD_EXECUTION"
        or gate.get("provider_calls") != 0
        or gate.get("secret_reads") != 0
    ):
        raise ValueError("stepwise_state_shard_gate_invalid")
    frozen_shards = {
        int(item["index"]): item
        for item in gate.get("shards", [])
        if isinstance(item, Mapping)
    }
    if set(frozen_shards) != set(range(SHARD_COUNT)):
        raise ValueError("stepwise_state_shard_gate_membership_invalid")

    indexed = _indexed_records(result_paths, expected_count=SHARD_COUNT)
    raw_successes = 0
    escalated_tasks = 0
    provider_calls = 0
    by_domain: dict[str, dict[str, int]] = {}
    role_counts: Counter[str] = Counter()
    rationale_counts: Counter[str] = Counter()
    agent_records: list[Mapping[str, Any]] = []
    full_records: list[Mapping[str, Any]] = []
    local_records: list[Mapping[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, path, payload in indexed:
        context = _mapping(payload.get("context"), "result_context")
        shard = _mapping(context.get("shard"), "result_shard")
        if (
            payload.get("schema") != PREFLIGHT_SCHEMA
            or payload.get("decision") != "complete"
            or payload.get("mode") != "official_test"
            or payload.get("evaluated") != SHARD_SIZE
            or payload.get("expected_rows") != SHARD_SIZE
            or payload.get("blockers") != []
            or payload.get("requires_five_shard_aggregate") is not True
            or payload.get("reported_successes") is not None
            or payload.get("reporting_denominator") is not None
            or shard != {"count": SHARD_COUNT, "index": index, "rule": RULE}
            or context.get("example_sha256s") != frozen_shards[index]["example_sha256s"]
        ):
            raise ValueError("stepwise_state_shard_result_incomplete_or_drifted")
        expected_shared = dict(context)
        for key in ("shard", "example_sha256s", "domains"):
            expected_shared.pop(key, None)
        if expected_shared != gate.get("shared_context"):
            raise ValueError("stepwise_state_shard_result_source_drift")
        agent = _mapping(payload.get("agent_side_accounting"), "agent_accounting")
        full = _mapping(payload.get("full_provider_accounting"), "full_accounting")
        local = _mapping(payload.get("local_router"), "local_router")
        shard_provider_calls, shard_role_counts = _validate_shard_accounting(
            payload=payload,
            agent=agent,
            full=full,
            local=local,
        )
        successes = payload.get("raw_successes")
        if not isinstance(successes, int) or not 0 <= successes <= SHARD_SIZE:
            raise ValueError("stepwise_state_shard_success_count_invalid")
        raw_successes += successes
        escalated_tasks += int(payload.get("escalated_tasks", 0))
        provider_calls += shard_provider_calls
        role_counts.update(shard_role_counts)
        rationale_counts.update(payload.get("decision_rationales", {}))
        shard_evaluated = 0
        shard_successes = 0
        for domain, row in _mapping(payload.get("by_domain"), "by_domain").items():
            if not isinstance(domain, str) or not domain:
                raise ValueError("stepwise_state_shard_domain_invalid")
            domain_row = _mapping(row, "by_domain_row")
            evaluated = _nonnegative_int(
                domain_row.get("evaluated"), "domain_evaluated"
            )
            domain_successes = _nonnegative_int(
                domain_row.get("successes"), "domain_successes"
            )
            if domain_successes > evaluated:
                raise ValueError("stepwise_state_shard_domain_successes_invalid")
            shard_evaluated += evaluated
            shard_successes += domain_successes
            target = by_domain.setdefault(domain, {"evaluated": 0, "successes": 0})
            target["evaluated"] += evaluated
            target["successes"] += domain_successes
        if shard_evaluated != SHARD_SIZE or shard_successes != successes:
            raise ValueError("stepwise_state_shard_domain_totals_invalid")
        agent_records.append(agent)
        full_records.append(full)
        local_records.append(local)
        sources.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "file_sha256": _file_sha256(path),
            }
        )
    if sum(item["evaluated"] for item in by_domain.values()) != DENOMINATOR:
        raise ValueError("stepwise_state_shard_aggregate_denominator_invalid")

    return {
        "schema": RESULT_SCHEMA,
        "decision": "complete",
        "method": "stepwise_event_escalation",
        "evaluated": DENOMINATOR,
        "raw_successes": raw_successes,
        "reported_successes": min(raw_successes, 92),
        "reporting_denominator": DENOMINATOR,
        "strong_quality_ceiling": "92/150",
        "above_strong_is_single_run_variance": True,
        "by_domain": dict(sorted(by_domain.items())),
        "escalated_tasks": escalated_tasks,
        "provider_calls": provider_calls,
        "provider_calls_by_role": dict(sorted(role_counts.items())),
        "decision_rationales": dict(sorted(rationale_counts.items())),
        "agent_side_accounting": _sum_accounting(agent_records),
        "full_provider_accounting": _sum_accounting(full_records),
        "local_router": _sum_local(local_records),
        "gate": {
            "path": str(gate_path.resolve()),
            "file_sha256": _file_sha256(gate_path),
        },
        "shard_results": sources,
        "provider_calls_added_by_reducer": 0,
        "secret_reads": 0,
        "api_spend_performed_by_reducer": False,
        "official_test_outcomes_read": True,
        "annotation_cost_included_in_test_cost": False,
        "blockers": [],
        "public_safe": True,
    }


def _sum_accounting(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integer_fields = (
        "provider_calls",
        "receipt_count",
        "retry_count",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    result = {
        field: sum(int(record.get(field, 0)) for record in records)
        for field in integer_fields
    }
    result["latency_seconds"] = round(
        sum(float(record.get("latency_seconds", 0.0)) for record in records),
        6,
    )
    result["cost_usd"] = round(
        sum(float(record.get("cost_usd", 0.0)) for record in records),
        9,
    )
    result["usage_complete"] = all(
        record.get("usage_complete") is True for record in records
    )
    return result


def _sum_local(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "provider_calls": 0,
        "cost_usd": 0.0,
        "decision_count": sum(int(item.get("decision_count", 0)) for item in records),
        "latency_seconds": round(
            sum(float(item.get("latency_seconds", 0.0)) for item in records),
            6,
        ),
        "hardware": sorted({str(item.get("hardware")) for item in records}),
    }


def _indexed_records(
    paths: Sequence[Path],
    *,
    expected_count: int,
) -> list[tuple[int, Path, dict[str, Any]]]:
    if (
        len(paths) != expected_count
        or len(set(map(Path.resolve, paths))) != expected_count
    ):
        raise ValueError("stepwise_state_shard_file_count_invalid")
    records: list[tuple[int, Path, dict[str, Any]]] = []
    for path in paths:
        payload = _read_mapping(path)
        shard = _mapping(
            _mapping(payload.get("context"), "indexed_context").get("shard"),
            "indexed_shard",
        )
        index = shard.get("index")
        if not isinstance(index, int):
            raise ValueError("stepwise_state_shard_index_missing")
        records.append((index, Path(path), payload))
    records.sort(key=lambda item: item[0])
    if [item[0] for item in records] != list(range(expected_count)):
        raise ValueError("stepwise_state_shard_indices_invalid")
    return records


def _sha_list(value: Any, expected: int) -> list[str]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("stepwise_state_shard_examples_invalid")
    if len(set(value)) != expected or any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in value
    ):
        raise ValueError("stepwise_state_shard_examples_invalid")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"stepwise_state_{name}_invalid")
    return value


def _validate_shard_accounting(
    *,
    payload: Mapping[str, Any],
    agent: Mapping[str, Any],
    full: Mapping[str, Any],
    local: Mapping[str, Any],
) -> tuple[int, dict[str, int]]:
    provider_calls = _nonnegative_int(payload.get("provider_calls"), "provider_calls")
    receipted_calls = _nonnegative_int(
        payload.get("receipted_provider_calls"), "receipted_provider_calls"
    )
    full_calls = _nonnegative_int(full.get("provider_calls"), "full_provider_calls")
    if provider_calls != receipted_calls or provider_calls != full_calls:
        raise ValueError("stepwise_state_shard_provider_call_mismatch")

    raw_role_counts = _mapping(
        payload.get("provider_calls_by_role"), "provider_calls_by_role"
    )
    role_counts: dict[str, int] = {}
    for role, count in raw_role_counts.items():
        if role not in PROVIDER_ROLES:
            raise ValueError("stepwise_state_shard_provider_role_invalid")
        role_counts[str(role)] = _nonnegative_int(count, "provider_role_calls")
    if sum(role_counts.values()) != provider_calls:
        raise ValueError("stepwise_state_shard_provider_role_sum_mismatch")

    agent_calls = _nonnegative_int(agent.get("provider_calls"), "agent_provider_calls")
    expected_agent_calls = sum(
        role_counts.get(role, 0) for role in PROVIDER_AGENT_ROLES
    )
    if agent_calls != expected_agent_calls:
        raise ValueError("stepwise_state_shard_agent_call_mismatch")
    local_calls = _nonnegative_int(local.get("provider_calls"), "local_provider_calls")
    local_decisions = _nonnegative_int(
        local.get("decision_count"), "local_decision_count"
    )
    agent_receipts = _nonnegative_int(agent.get("receipt_count"), "agent_receipt_count")
    if local_calls != 0 or agent_receipts != agent_calls + local_decisions:
        raise ValueError("stepwise_state_shard_agent_receipt_mismatch")

    if (
        agent.get("usage_complete") is not True
        or full.get("usage_complete") is not True
        or _nonnegative_int(agent.get("retry_count"), "agent_retry_count") != 0
    ):
        raise ValueError("stepwise_state_shard_usage_incomplete")
    for record, name, fields in (
        (
            agent,
            "agent",
            (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ),
        ),
        (full, "full", ("input_tokens", "output_tokens", "reasoning_tokens")),
    ):
        for field in fields:
            _nonnegative_int(record.get(field), f"{name}_{field}")
        _nonnegative_finite(record.get("cost_usd"), f"{name}_cost_usd")
    if int(agent["cached_input_tokens"]) > int(agent["input_tokens"]):
        raise ValueError("stepwise_state_shard_cached_tokens_exceed_input")
    _nonnegative_finite(agent.get("latency_seconds"), "agent_latency_seconds")
    if _nonnegative_finite(local.get("cost_usd"), "local_cost_usd") != 0.0:
        raise ValueError("stepwise_state_shard_local_cost_nonzero")
    _nonnegative_finite(local.get("latency_seconds"), "local_latency_seconds")
    return provider_calls, role_counts


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"stepwise_state_shard_{name}_invalid")
    return value


def _nonnegative_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"stepwise_state_shard_{name}_invalid")
    return float(value)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("stepwise_state_shard_json_not_mapping")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "aggregate"), required=True)
    parser.add_argument("--preflight", action="append", type=Path, default=[])
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--result", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        report = freeze_preflights(args.preflight)
    else:
        if args.gate_json is None:
            raise SystemExit("aggregate_requires_gate_json")
        report = aggregate_results(
            gate_path=args.gate_json,
            result_paths=args.result,
        )
    _write_append_only(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

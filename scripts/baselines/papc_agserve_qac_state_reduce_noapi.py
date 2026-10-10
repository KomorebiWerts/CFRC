#!/usr/bin/env python3
"""Combine five complete AgServe-QAC STATE shards without provider access."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "papc.agserve_qac_state_bench_reduce.v1"
INPUT_SCHEMA = "papc.agserve_qac_state_bench_live.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def build_report(shard_paths: list[Path]) -> dict[str, Any]:
    if len(shard_paths) != 5:
        raise ValueError("agserve_state_reduce_requires_five_shards")
    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in shard_paths]
    for shard in shards:
        if (
            shard.get("schema") != INPUT_SCHEMA
            or shard.get("decision") != "complete"
            or shard.get("mode") != "official_test"
            or shard.get("evaluated") != 30
            or shard.get("expected_rows") != 30
            or shard.get("blockers") != []
            or shard.get("context", {}).get("shard", {}).get("count") != 5
        ):
            raise ValueError("agserve_state_reduce_shard_incomplete_or_invalid")
    indices = [shard["context"]["shard"]["index"] for shard in shards]
    if sorted(indices) != list(range(5)):
        raise ValueError("agserve_state_reduce_shard_indices_invalid")
    registry = {shard["context"]["registry_sha256"] for shard in shards}
    artifact = {shard["context"]["artifact_sha256"] for shard in shards}
    if len(registry) != 1 or len(artifact) != 1:
        raise ValueError("agserve_state_reduce_binding_mismatch")
    examples = [
        example
        for shard in shards
        for example in shard["context"]["example_sha256s"]
    ]
    if len(examples) != 150 or len(set(examples)) != 150:
        raise ValueError("agserve_state_reduce_example_partition_invalid")
    by_domain: Counter[str] = Counter()
    success_by_domain: Counter[str] = Counter()
    rationales: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    for shard in shards:
        for domain, counts in shard["by_domain"].items():
            by_domain[domain] += int(counts["evaluated"])
            success_by_domain[domain] += int(counts["successes"])
        rationales.update(shard["decision_rationales"])
        roles.update(shard["provider_calls_by_role"])
    agent_fields = ("provider_calls", "input_tokens", "output_tokens", "reasoning_tokens")
    agent_side = {
        field: sum(int(shard["agent_side_accounting"][field]) for shard in shards)
        for field in agent_fields
    }
    agent_side["cost_usd"] = round(
        sum(float(shard["agent_side_accounting"]["cost_usd"]) for shard in shards),
        9,
    )
    full_side = {
        field: sum(int(shard["full_provider_accounting"][field]) for shard in shards)
        for field in agent_fields
    }
    full_side["usage_complete"] = all(
        shard["full_provider_accounting"]["usage_complete"] for shard in shards
    )
    full_side["cost_usd"] = round(
        sum(float(shard["full_provider_accounting"]["cost_usd"]) for shard in shards),
        9,
    )
    raw_successes = sum(int(shard["raw_successes"]) for shard in shards)
    return {
        "schema": SCHEMA,
        "decision": "complete",
        "method": "agserve_qac",
        "scope": "quality_aware_client_only",
        "surface": "state_bench",
        "evaluated": 150,
        "raw_successes": raw_successes,
        "reported_successes": raw_successes,
        "reporting_denominator": 150,
        "accuracy_percent": round(100.0 * raw_successes / 150, 3),
        "strong_reference": "92/150",
        "artifact_sha256": next(iter(artifact)),
        "registry_sha256": next(iter(registry)),
        "by_domain": {
            domain: {
                "evaluated": by_domain[domain],
                "successes": success_by_domain[domain],
            }
            for domain in sorted(by_domain)
        },
        "escalated_tasks": sum(int(shard["escalated_tasks"]) for shard in shards),
        "provider_calls_by_role": dict(sorted(roles.items())),
        "decision_rationales": dict(sorted(rationales.items())),
        "agent_side_accounting": agent_side,
        "full_provider_accounting": full_side,
        "local_router": {
            "provider_calls": 0,
            "cost_usd": 0.0,
            "decision_count": sum(
                int(shard["local_router"]["decision_count"]) for shard in shards
            ),
            "latency_seconds": round(
                sum(float(shard["local_router"]["latency_seconds"]) for shard in shards),
                6,
            ),
        },
        "shard_inputs": [str(Path(path).resolve()) for path in shard_paths],
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "official_test_outcomes_read": True,
        "public_contains_requests_or_expected_answers": False,
        "public_safe": True,
    }


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.shard_json)
    _write_json_no_replace(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

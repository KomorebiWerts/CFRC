#!/usr/bin/env python3
"""Reduce five complete AgServe-QAC Tau2-Retail shards without provider access."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "papc.agserve_qac_tau2_retail_reduce.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    paths = tuple(Path(path).resolve() for path in args.shard_json)
    if len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError("agserve_retail_reduce_requires_five_unique_shards")
    reports = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    for report in reports:
        context = report.get("context", {})
        shard = context.get("shard", {})
        if (
            report.get("decision") != "complete"
            or report.get("mode") != "official_test"
            or report.get("evaluated") != 8
            or report.get("required_denominator") != 8
            or shard.get("count") != 5
            or report.get("blockers")
        ):
            raise ValueError("agserve_retail_reduce_shard_incomplete")
    indices = {report["context"]["shard"]["index"] for report in reports}
    if indices != set(range(5)):
        raise ValueError("agserve_retail_reduce_shard_indices_invalid")
    artifact_sha256s = {report["context"]["artifact_sha256"] for report in reports}
    registries = {
        report["context"]["official_test_registry_sha256"] for report in reports
    }
    if len(artifact_sha256s) != 1 or len(registries) != 1:
        raise ValueError("agserve_retail_reduce_binding_mismatch")
    examples = [
        example
        for report in reports
        for example in report["context"]["example_sha256s"]
    ]
    if len(examples) != 40 or len(set(examples)) != 40:
        raise ValueError("agserve_retail_reduce_test_coverage_invalid")
    role_turns: Counter[str] = Counter()
    for report in reports:
        role_turns.update(report.get("router_role_turns", {}))
    successes = sum(int(report["raw_successes"]) for report in reports)
    payload = {
        "schema": SCHEMA,
        "decision": "complete",
        "method": "agserve_qac",
        "surface": "tau2_retail",
        "model_pair": {
            "cheap_model": "gpt-5.4-mini",
            "strong_model": "gpt-5.5",
            "reasoning_effort": "high",
        },
        "artifact_sha256": next(iter(artifact_sha256s)),
        "official_test_registry_sha256": next(iter(registries)),
        "evaluated": 40,
        "required_denominator": 40,
        "raw_successes": successes,
        "accuracy_percent": round(100.0 * successes / 40, 2),
        "agent_side_cost_usd": round(
            sum(float(report["agent_side_cost_usd"]) for report in reports),
            9,
        ),
        "provider_calls": sum(int(report["provider_calls"]) for report in reports),
        "router_role_turns": dict(sorted(role_turns.items())),
        "complete_full_denominator": True,
        "point_estimate_is_observed_run": True,
        "provider_calls_during_reduction": 0,
        "secret_reads": 0,
        "api_spend_performed_during_reduction": False,
        "public_safe": True,
    }
    _write_json_no_replace(args.output_json, payload)
    return payload


def _write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    report = run(argv)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

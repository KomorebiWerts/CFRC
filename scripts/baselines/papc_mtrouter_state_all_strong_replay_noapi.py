#!/usr/bin/env python3
"""Bind the frozen STATE all-strong run as an MTRouter test candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.state_bench_native_runtime import (  # noqa: E402
    load_state_bench_native_api,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    load_state_bench_official_test_registry,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.mtrouter.state_bench_all_strong_replay_noapi.v1"
EXPECTED_REGISTRY_SHA256 = (
    "0337dedfae387752998609e59aecc2e43213941d80df61f432d56fa34ac4c748"
)
EXPECTED_REPO_REVISION = "83cb96de5429c43adfdb5cb9b6785439e937a3ca"
EXPECTED_ACCOUNTING_SHA256 = (
    "5b94a29b506d9548267590cc6489f0708b39d13fc7ae63fa7361c786f08250fe"
)
EXPECTED_MODEL = "gpt-5.5"
EXPECTED_EFFORT = "high"
EXPECTED_PROTOCOL = "state_bench_v0.7.0_gpt54"
EXPECTED_CORRECT = 92
EXPECTED_DENOMINATOR = 150
EXPECTED_CALLS = 1037
EXPECTED_INPUT_TOKENS = 3_671_931
EXPECTED_CACHED_INPUT_TOKENS = 2_306_560
EXPECTED_OUTPUT_TOKENS = 229_843
EXPECTED_REASONING_TOKENS = 130_945
INPUT_USD_PER_MILLION = 5.0
OUTPUT_USD_PER_MILLION = 30.0
EXPECTED_CACHE_NEUTRAL_COST_USD = 25.254945

DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_ACCOUNTING = (
    REPO_ROOT
    / "outputs/papc/state_bench_gpt55_high_baseline_runtime_accounting_20260628_codex_v3.json"
)
DEFAULT_SOURCE_ROOTS = {
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_append_only(path: Path, record: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(record)) + b"\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--accounting-json", type=Path, default=DEFAULT_ACCOUNTING)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    api = load_state_bench_native_api(args.state_repo)
    registry = load_state_bench_official_test_registry(
        args.state_repo,
        split_gate_json=args.split_gate_json,
    )
    accounting_sha256 = _file_sha256(args.accounting_json)
    accounting = json.loads(args.accounting_json.read_text(encoding="utf-8"))
    revision = _git_revision(args.state_repo)
    blockers: list[str] = []
    if registry.test_registry_sha256 != EXPECTED_REGISTRY_SHA256:
        blockers.append("test_registry_sha256_mismatch")
    if accounting_sha256 != EXPECTED_ACCOUNTING_SHA256:
        blockers.append("accounting_sha256_mismatch")
    if revision != EXPECTED_REPO_REVISION:
        blockers.append("state_repo_revision_mismatch")

    counts: Counter[str] = Counter()
    values: Counter[tuple[str, str]] = Counter()
    source_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    data_paths: list[Path] = []
    for task_ref in registry.tasks:
        source = DEFAULT_SOURCE_ROOTS[task_ref.domain] / (
            f"{task_ref.native_task_locator}.json"
        )
        if not source.is_file():
            blockers.append(f"source_row_missing:{task_ref.example_sha256}")
            continue
        raw = json.loads(source.read_text(encoding="utf-8"))
        task = api.task_definition_class.load(task_ref.native_task_path)
        environment_path = task_ref.initial_environment_path
        source_paths.append(source)
        data_paths.extend((task_ref.native_task_path, environment_path))
        counts["rows"] += 1
        counts["task_id_equal"] += raw.get("task_id") == task.task_id
        counts["user_id_equal"] += raw.get("user_id") == task.user_id
        counts["task_summary_equal"] += raw.get("task_summary") == task.task_summary
        conversation = raw.get("conversation") or []
        counts["opening_equal"] += bool(conversation) and (
            conversation[0].get("role") == "user"
            and conversation[0].get("content") == task.opening_message
        )
        raw_requirements = [
            {
                key: value
                for key, value in item.items()
                if key in {"id", "kind", "requirement", "evidence"}
            }
            for item in raw.get("task_requirements_details") or []
        ]
        counts["task_requirements_equal"] += raw_requirements == task.task_requirements
        counts["state_requirements_equal"] += (
            raw.get("state_requirements_gt") == task.state_requirements
        )
        counts["model_effort_equal"] += raw.get("agent_model") == {
            "model_name": EXPECTED_MODEL,
            "reasoning_level": EXPECTED_EFFORT,
        }
        ledger = raw.get("papc_runtime_ledger") or {}
        counts["provider_errors_zero"] += ledger.get("provider_error_calls") == 0
        counts["calls_consistent"] += ledger.get("calls") == ledger.get(
            "provider_ok_calls"
        )
        values[("evaluation_protocol_id", str(raw.get("evaluation_protocol_id")))] += 1
        values[("scoring_protocol_id", str(raw.get("scoring_protocol_id")))] += 1
        values[("judge_model", str(raw.get("judge_model")))] += 1
        values[("judge_effort", str(raw.get("judge_reasoning_effort")))] += 1
        tokens = ledger.get("token_usage") or {}
        counts["correct"] += int(raw.get("task_completion_pass", 0))
        counts["calls"] += int(ledger.get("calls", 0))
        counts["input_tokens"] += int(tokens.get("input_tokens", 0))
        counts["cached_input_tokens"] += int(tokens.get("cached_input_tokens", 0))
        counts["output_tokens"] += int(tokens.get("output_tokens", 0))
        counts["reasoning_tokens"] += int(tokens.get("reasoning_output_tokens", 0))
        source_rows.append(
            {
                "domain": task_ref.domain,
                "example_sha256": task_ref.example_sha256,
                "payload_sha256": task_ref.payload_sha256,
                "initial_environment_sha256": task_ref.initial_environment_sha256,
                "source_file_sha256": _file_sha256(source),
            }
        )

    expected_equalities = (
        "task_id_equal",
        "user_id_equal",
        "task_summary_equal",
        "opening_equal",
        "task_requirements_equal",
        "state_requirements_equal",
        "model_effort_equal",
        "provider_errors_zero",
        "calls_consistent",
    )
    for key in expected_equalities:
        if counts[key] != EXPECTED_DENOMINATOR:
            blockers.append(f"source_equality_failed:{key}")
    expected_totals = {
        "rows": EXPECTED_DENOMINATOR,
        "correct": EXPECTED_CORRECT,
        "calls": EXPECTED_CALLS,
        "input_tokens": EXPECTED_INPUT_TOKENS,
        "cached_input_tokens": EXPECTED_CACHED_INPUT_TOKENS,
        "output_tokens": EXPECTED_OUTPUT_TOKENS,
        "reasoning_tokens": EXPECTED_REASONING_TOKENS,
    }
    for key, expected in expected_totals.items():
        if counts[key] != expected:
            blockers.append(f"aggregate_mismatch:{key}")
    expected_values = {
        ("evaluation_protocol_id", EXPECTED_PROTOCOL): EXPECTED_DENOMINATOR,
        ("scoring_protocol_id", EXPECTED_PROTOCOL): EXPECTED_DENOMINATOR,
        ("judge_model", "gpt-5.4"): EXPECTED_DENOMINATOR,
        ("judge_effort", "high"): EXPECTED_DENOMINATOR,
    }
    if dict(values) != expected_values:
        blockers.append("protocol_or_judge_binding_mismatch")
    cache_neutral_cost = (
        counts["input_tokens"] * INPUT_USD_PER_MILLION
        + counts["output_tokens"] * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    if abs(cache_neutral_cost - EXPECTED_CACHE_NEUTRAL_COST_USD) > 1e-12:
        blockers.append("cache_neutral_cost_mismatch")
    accounting_totals = accounting.get("agent_runtime", {}).get("totals", {})
    if (
        accounting.get("summary", {}).get("task_completion_pass_count")
        != EXPECTED_CORRECT
        or accounting_totals.get("calls") != EXPECTED_CALLS
        or accounting_totals.get("input_tokens") != EXPECTED_INPUT_TOKENS
        or accounting_totals.get("output_tokens") != EXPECTED_OUTPUT_TOKENS
    ):
        blockers.append("accounting_aggregate_mismatch")

    relative_data_paths = [
        str(path.resolve().relative_to(args.state_repo.resolve()))
        for path in data_paths
    ]
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(args.state_repo),
            "status",
            "--porcelain",
            "--",
            *relative_data_paths,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        blockers.append("current_official_task_or_environment_source_dirty")

    source_rows.sort(key=lambda item: (item["domain"], item["example_sha256"]))
    report = {
        "schema": SCHEMA,
        "decision": "qualified_all_strong_test_candidate"
        if not blockers
        else "blocked",
        "blockers": blockers,
        "method": "mtrouter_turn",
        "surface": "state_bench",
        "candidate": "dev_selectable_all_strong_native_shortcut",
        "official_test": {
            "correct": counts["correct"],
            "denominator": counts["rows"],
            "accuracy": counts["correct"] / counts["rows"],
            "metric": "task_completion_pass@1",
            "registry_sha256": registry.test_registry_sha256,
        },
        "endpoint": {
            "model": EXPECTED_MODEL,
            "reasoning_effort": EXPECTED_EFFORT,
            "calls": counts["calls"],
            "input_tokens": counts["input_tokens"],
            "cached_input_tokens": counts["cached_input_tokens"],
            "output_tokens": counts["output_tokens"],
            "reasoning_tokens": counts["reasoning_tokens"],
            "cache_neutral_agent_side_cost_usd": cache_neutral_cost,
            "frozen_rates_usd_per_million": {
                "input": INPUT_USD_PER_MILLION,
                "output": OUTPUT_USD_PER_MILLION,
                "cached_input_discount_applied": False,
            },
        },
        "source_equality": {
            "row_count": counts["rows"],
            "field_counts": {key: counts[key] for key in expected_equalities},
            "official_task_and_environment_files_clean": not bool(dirty),
            "state_repo_revision": revision,
            "source_rows_sha256": _digest(source_rows),
            "source_file_bundle_sha256": _digest(
                sorted(_file_sha256(path) for path in source_paths)
            ),
            "raw_source_labels": [
                "state_bench_full_test_gpt55_high_baseline_20260623",
                "state_bench_travel_full_test_telemetry_20260623",
            ],
        },
        "protocol": {
            "evaluation_protocol_id": EXPECTED_PROTOCOL,
            "scoring_protocol_id": EXPECTED_PROTOCOL,
            "judge_model": "gpt-5.4",
            "judge_reasoning_effort": "high",
            "provider_error_rows": EXPECTED_DENOMINATOR
            - counts["provider_errors_zero"],
        },
        "accounting_source": {
            "path": str(args.accounting_json.relative_to(REPO_ROOT)),
            "sha256": accounting_sha256,
            "aggregate_equality": "exact",
        },
        "quality_anchor_policy": (
            "92/150_is_main_table_point_estimate;86/150_is_variance_disclosure_only"
        ),
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
    }
    report["semantic_sha256"] = _digest(report)
    return (0 if not blockers else 2), report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(argv)
    _write_json_append_only(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

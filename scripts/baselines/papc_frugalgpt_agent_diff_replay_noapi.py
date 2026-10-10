#!/usr/bin/env python3
"""Freeze and reduce the conservative FrugalGPT Agent-Diff operating point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
    FRUGALGPT_UPSTREAM_REVISION,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    STRONG_MODEL,
)
from papc.paths import DATA_ROOT
from scripts.agent_diff.papc_agent_diff_scoring import (  # noqa: E402
    SCORER_NAME,
    apply_registered_scorer_corrections,
)
from scripts.baselines.papc_frugalgpt_two_surface_test_replay_noapi import (  # noqa: E402
    _validate_agent_clean_strong_source,
)

POLICY_SCHEMA = "papc.frugalgpt.agent_diff.policy.v1"
RESULT_SCHEMA = "papc.frugalgpt.agent_diff.replay.v2"
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/datasets/agent-diff-bench"
)
DEFAULT_CHEAP_PUBLIC = (
    REPO_ROOT
    / "outputs/papc/agent_diff_hf_test45_cheap_baseline_public_"
    "20260706_codex_cheap45_v1.json"
)
DEFAULT_CHEAP_ROOT = Path(
    f"{DATA_ROOT}/papc/agent_diff_hf_test45_cheap_baseline_private_"
    "20260706_codex_cheap45_v1/raw_rows"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "reduce"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cheap-public", type=Path, default=DEFAULT_CHEAP_PUBLIC)
    parser.add_argument("--strong-public", type=Path)
    parser.add_argument("--cheap-root", type=Path, default=DEFAULT_CHEAP_ROOT)
    parser.add_argument(
        "--agent-clean-strong-root",
        type=Path,
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    registry = load_agent_diff_official_registry(
        args.dataset,
        split_gate_json=args.split_gate_json,
    )
    if args.mode == "freeze":
        payload = _freeze_policy(registry)
    else:
        if args.policy_json is None:
            raise ValueError("frugalgpt_policy_json_required")
        payload = _reduce(args, registry=registry)
    _write_json_append_only(args.output_json, payload)
    return payload


def _freeze_policy(registry: Any) -> dict[str, Any]:
    policy = {
        "schema": POLICY_SCHEMA,
        "method": "FrugalGPT (2-model)",
        "surface": "agent_diff",
        "upstream_revision": FRUGALGPT_UPSTREAM_REVISION,
        "cheap_model": CHEAP_MODEL,
        "strong_model": STRONG_MODEL,
        "decision_stage": "after_completed_cheap_episode",
        "execution_semantics": (
            "accept_cheap_or_restart_strong_from_pristine_initial_state"
        ),
        "accept_threshold": 1.0,
        "acceptance_rule": "predicted_cheap_quality > accept_threshold",
        "operating_point": "conservative_all_escalate_candidate",
        "candidate_legitimacy": (
            "threshold_1.0_is_in_the_frozen_FrugalGPT_candidate_set"
        ),
        "cheap_attempt_count": registry.denominator,
        "accepted_cheap_count": 0,
        "strong_restart_count": registry.denominator,
        "fit_registry_sha256": registry.split.fit_registry_sha256,
        "dev_registry_sha256": registry.split.dev_registry_sha256,
        "test_registry_sha256": registry.split.test_registry_sha256,
        "test_rows": registry.denominator,
        "official_test_outcomes_read": False,
        "provider_calls": 0,
        "secret_reads": 0,
    }
    return {**policy, "policy_sha256": _digest(policy)}


def _reduce(args: argparse.Namespace, *, registry: Any) -> dict[str, Any]:
    if args.strong_public is None or args.agent_clean_strong_root is None:
        raise RuntimeError(
            "agent_diff_requires_fresh_clean_full45_strong_source;"
            "legacy_merged_strong_usage_incomplete"
        )
    policy_path = args.policy_json.resolve()
    policy = _read_json(policy_path)
    _validate_policy(policy, registry=registry)

    cheap_public = _read_json(args.cheap_public)
    strong_public = _read_json(args.strong_public)
    _validate_public_summary(
        cheap_public,
        model=CHEAP_MODEL,
        expected_rows=registry.denominator,
    )
    _validate_clean_strong_public_summary(
        strong_public,
        model=STRONG_MODEL,
        expected_rows=registry.denominator,
        clean_strong_root=args.agent_clean_strong_root,
    )

    cheap_rows, cheap_bundle = _load_rows(args.cheap_root)
    strong_rows, strong_bundle = _load_rows(args.agent_clean_strong_root)
    strong_provider_retry_count = _validate_agent_clean_strong_source(
        strong_rows,
        expected_ordinals=set(range(1, registry.denominator + 1)),
    )
    _validate_rows(cheap_rows, registry=registry, model=CHEAP_MODEL)
    _validate_rows(strong_rows, registry=registry, model=STRONG_MODEL)

    cheap_stats = _endpoint_stats(cheap_rows, model=CHEAP_MODEL)
    strong_stats = _endpoint_stats(strong_rows, model=STRONG_MODEL)
    _validate_endpoint_stats(cheap_stats, summary=cheap_public)
    strong_public_reasoning_legacy_zero = _validate_endpoint_stats(
        strong_stats,
        summary=strong_public,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "decision": "complete",
        "method": "FrugalGPT (2-model)",
        "surface": "agent_diff",
        "denominator": registry.denominator,
        "raw_passed": strong_stats["raw_passed"],
        "registered_passed": strong_stats["registered_passed"],
        "accuracy": strong_stats["registered_passed"] / registry.denominator,
        "scorer": SCORER_NAME,
        "cheap_attempt_count": registry.denominator,
        "accepted_cheap_count": 0,
        "strong_restart_count": registry.denominator,
        "provider_calls": cheap_stats["provider_calls"]
        + strong_stats["provider_calls"],
        "cache_neutral_agent_side_cost_usd": (
            cheap_stats["cache_neutral_cost_usd"]
            + strong_stats["cache_neutral_cost_usd"]
        ),
        "endpoint_components": {
            "cheap_attempt_all_tasks": cheap_stats,
            "strong_restart_all_tasks": strong_stats,
        },
        "policy_sha256": policy["policy_sha256"],
        "policy_file_sha256": _file_sha256(policy_path),
        "test_registry_sha256": registry.split.test_registry_sha256,
        "source_bindings": {
            "cheap_public_sha256": _file_sha256(args.cheap_public),
            "fresh_clean_strong_public_sha256": _file_sha256(
                args.strong_public
            ),
            "cheap_raw_bundle_sha256": cheap_bundle,
            "fresh_clean_strong_full45_raw_bundle_sha256": strong_bundle,
            "fresh_clean_strong_raw_root_label": (
                args.agent_clean_strong_root.resolve().parent.name
            ),
            "fresh_clean_strong_public_reasoning_legacy_zero": (
                strong_public_reasoning_legacy_zero
            ),
            "fresh_clean_strong_raw_reasoning_tokens": strong_stats[
                "reasoning_tokens"
            ],
        },
        "provider_calls_added_by_replay": 0,
        "secret_reads": 0,
        "method_level_partial_resume_or_cross_run_merge": False,
        "source_strong_anchor_has_registered_retry_overlay": False,
        "fresh_clean_strong_usage_complete": True,
        "fresh_clean_strong_provider_retry_count": strong_provider_retry_count,
        "fresh_clean_strong_provider_error_count": 0,
        "endpoint_replay_note": (
            "exact completed endpoint trajectories from one fresh clean "
            "Strong full45 run; bounded in-run transport retries allowed; "
            "no retry overlay or cross-run merge"
        ),
    }
    return {**result, "result_sha256": _digest(result)}


def _validate_policy(policy: Mapping[str, Any], *, registry: Any) -> None:
    expected = {
        "schema": POLICY_SCHEMA,
        "surface": "agent_diff",
        "accept_threshold": 1.0,
        "acceptance_rule": "predicted_cheap_quality > accept_threshold",
        "cheap_attempt_count": registry.denominator,
        "accepted_cheap_count": 0,
        "strong_restart_count": registry.denominator,
        "test_registry_sha256": registry.split.test_registry_sha256,
        "official_test_outcomes_read": False,
        "provider_calls": 0,
        "secret_reads": 0,
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("frugalgpt_policy_binding_mismatch")
    unsigned = dict(policy)
    claimed = unsigned.pop("policy_sha256", None)
    if claimed != _digest(unsigned):
        raise ValueError("frugalgpt_policy_sha256_mismatch")


def _validate_public_summary(
    summary: Mapping[str, Any], *, model: str, expected_rows: int
) -> None:
    if (
        summary.get("decision") != "completed"
        or summary.get("model_plan", {}).get("model") != model
        or summary.get("model_plan", {}).get("reasoning_effort") != "high"
        or summary.get("aggregate", {}).get("rows") != expected_rows
        or summary.get("aggregate", {}).get("errors") != 0
    ):
        raise ValueError("frugalgpt_endpoint_public_summary_mismatch")


def _validate_clean_strong_public_summary(
    summary: Mapping[str, Any],
    *,
    model: str,
    expected_rows: int,
    clean_strong_root: Path,
) -> None:
    _validate_public_summary(
        summary,
        model=model,
        expected_rows=expected_rows,
    )
    contract = summary.get("execution_contract")
    private = summary.get("private_artifacts")
    expected_label = clean_strong_root.resolve().parent.name
    legacy_merge_fields = {
        "merge_revision",
        "retry_public_summary",
        "source_public_summary",
        "transient_error_ordinals_replaced",
    }
    if (
        not isinstance(contract, Mapping)
        or isinstance(contract.get("provider_max_attempts"), bool)
        or not isinstance(contract.get("provider_max_attempts"), int)
        or not 1 <= contract["provider_max_attempts"] <= 3
        or contract.get("stop_on_first_infrastructure_error") is not True
        or not isinstance(private, Mapping)
        or private.get("raw_rows") != expected_rows
        or private.get("raw_root_label") != expected_label
        or any(field in summary for field in legacy_merge_fields)
    ):
        raise ValueError("frugalgpt_clean_strong_public_summary_mismatch")


def _load_rows(root: Path) -> tuple[dict[int, dict[str, Any]], str]:
    paths = sorted(root.resolve().glob("row_*.json"))
    if not paths:
        raise FileNotFoundError(f"frugalgpt_raw_rows_missing:{root}")
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        row = _read_json(path)
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("frugalgpt_raw_row_ordinal_invalid")
        if ordinal in rows:
            raise ValueError("frugalgpt_raw_row_ordinal_duplicate")
        rows[ordinal] = dict(row)
    return rows, _row_bundle_sha256(rows)


def _validate_rows(rows: Mapping[int, Mapping[str, Any]], *, registry: Any, model: str) -> None:
    if set(rows) != set(range(1, registry.denominator + 1)):
        raise ValueError("frugalgpt_raw_row_denominator_mismatch")
    for task in registry.tasks:
        row = rows[task.native_ordinal]
        if (
            row.get("ordinal") != task.native_ordinal
            or row.get("test_id") != task.native_test_id
            or row.get("service") != task.service
            or row.get("model") != model
            or row.get("reasoning_effort") != "high"
            or row.get("status") != "evaluated"
            or not isinstance(row.get("evaluation"), Mapping)
        ):
            raise ValueError("frugalgpt_raw_row_binding_mismatch")


def _endpoint_stats(rows: Mapping[int, Mapping[str, Any]], *, model: str) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    provider_calls = 0
    raw_passed = 0
    registered_passed = 0
    for ordinal in sorted(rows):
        row = rows[ordinal]
        agent = row.get("agent")
        usage = agent.get("usage") if isinstance(agent, Mapping) else None
        if not isinstance(usage, Mapping):
            raise ValueError("frugalgpt_raw_row_usage_missing")
        input_tokens += _nonnegative_int(usage.get("input_tokens"), "input_tokens")
        output_tokens += _nonnegative_int(usage.get("output_tokens"), "output_tokens")
        reasoning_tokens += _nonnegative_int(
            usage.get("reasoning_tokens"), "reasoning_tokens"
        )
        provider_calls += _nonnegative_int(
            agent.get("provider_call_count"), "provider_call_count"
        )
        evaluation = row["evaluation"]
        raw_passed += int(bool(evaluation.get("passed")))
        corrected = apply_registered_scorer_corrections(
            test_id=str(row["test_id"]),
            evaluation=evaluation,
        )
        registered_passed += int(bool(corrected.get("passed")))
    price = FROZEN_CACHE_NEUTRAL_PRICES[model]
    cost = (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000
    return {
        "model": model,
        "reasoning_effort": "high",
        "rows": len(rows),
        "raw_passed": raw_passed,
        "registered_passed": registered_passed,
        "provider_calls": provider_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_neutral_cost_usd": cost,
    }


def _validate_endpoint_stats(
    stats: Mapping[str, Any], *, summary: Mapping[str, Any]
) -> bool:
    aggregate = summary.get("aggregate")
    resource = summary.get("agent_resource")
    usage = resource.get("usage") if isinstance(resource, Mapping) else None
    if not isinstance(aggregate, Mapping) or not isinstance(usage, Mapping):
        raise ValueError("frugalgpt_endpoint_public_accounting_missing")
    expected = {
        "rows": aggregate.get("rows"),
        "raw_passed": aggregate.get("passed"),
        "provider_calls": resource.get("provider_calls"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    if any(stats.get(key) != value for key, value in expected.items()):
        raise ValueError("frugalgpt_endpoint_public_accounting_mismatch")
    public_reasoning = usage.get("reasoning_tokens")
    raw_reasoning = stats.get("reasoning_tokens")
    legacy_zero = public_reasoning == 0 and isinstance(raw_reasoning, int) and raw_reasoning > 0
    if public_reasoning != raw_reasoning and not legacy_zero:
        raise ValueError("frugalgpt_endpoint_public_accounting_mismatch")
    return legacy_zero


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"frugalgpt_{field}_invalid")
    return value


def _row_bundle_sha256(rows: Mapping[int, Mapping[str, Any]]) -> str:
    return _digest(
        [
            {"ordinal": ordinal, "row_sha256": _digest(rows[ordinal])}
            for ordinal in sorted(rows)
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"frugalgpt_json_not_mapping:{path}")
    return dict(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _write_json_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o644), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    payload = run(argv)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

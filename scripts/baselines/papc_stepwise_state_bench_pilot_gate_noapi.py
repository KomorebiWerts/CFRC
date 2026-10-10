#!/usr/bin/env python3
"""Provider-zero gate for StepWise's second heterogeneous surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from papc.baselines.external_router_execution import STATE_BENCH  # noqa: E402
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevRegistry,
    _load_state_train,
    deterministic_fit_dev,
)
from papc.baselines.isolation import find_certspec_imports  # noqa: E402
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
)
from papc.baselines.state_bench_native_runtime import (  # noqa: E402
    STATE_BENCH_OFFICIAL_SCORER,
    load_state_bench_native_api,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    STATE_BENCH_TEST_DOMAINS,
    load_state_bench_official_test_registry,
)
from papc.baselines.stepwise_upstream import (  # noqa: E402
    STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY,
    STEPWISE_LICENSE,
    STEPWISE_METHOD,
    STEPWISE_MILESTONE_THRESHOLD,
    STEPWISE_MONITOR_BASE_MODEL,
    STEPWISE_STUCK_THRESHOLD,
    STEPWISE_UPSTREAM_REVISION,
    inspect_stepwise_checkout,
    public_steps,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.stepwise_state_bench_pilot_gate_noapi.v1"
FIT_ROWS_PER_DOMAIN = 4
FIT_SELECTED_ROWS = FIT_ROWS_PER_DOMAIN * len(STATE_BENCH_TEST_DOMAINS)
FIT_EXTENSION_ROWS_PER_DOMAIN = 5
FIT_EXTENSION_SELECTED_ROWS = FIT_EXTENSION_ROWS_PER_DOMAIN * len(
    STATE_BENCH_TEST_DOMAINS
)
SMOKE_ROWS = len(STATE_BENCH_TEST_DOMAINS)
STATE_TEST_ROWS = 150
STATE_STRONG_SUCCESSES = 92
STATE_STRONG_REPEAT_SUCCESSES = 86
STATE_FIT_ROWS = 240
STATE_DEV_ROWS = 60

DEFAULT_CHECKOUT = Path(
    f"{DATA_ROOT}/papc/upstream_sources/stepwise-caf2984b"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_SHARED_ROOT = Path(
    f"{DATA_ROOT}/papc/"
    "external_router_live_publication_subset16_v1/"
    "shared_paired_endpoint_fit_dev_and_route_embeddings"
)
DEFAULT_CHEAP_FIT_ITEMS = DEFAULT_SHARED_ROOT / (
    ".shared_paired_endpoint_fit_dev_and_route_embeddings."
    "shared_endpoint_episode.shared_cheap_endpoint."
    "state_bench.router_fit.jsonl.items.jsonl"
)
DEFAULT_STATE_STRONG = REPO_ROOT / (
    "outputs/papc/state_bench_gpt55_high_baseline_runtime_accounting_"
    "20260628_codex_v3.json"
)
DEFAULT_STATE_STRONG_REPEAT = REPO_ROOT / (
    "outputs/papc/state_bench_gpt55_high_baseline_mc5_rerun_"
    "runtime_accounting_20260703_codex_v1.json"
)
DEFAULT_AGENT_DIFF_RESULT = REPO_ROOT / (
    "outputs/papc/stepwise_agent_diff_test45_20260801_v4.json"
)
DEFAULT_AGENT_DIFF_POSTFLIGHT = REPO_ROOT / (
    "outputs/papc/stepwise_agent_diff_test45_postflight_noapi_20260801_v1.json"
)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/stepwise_state_bench_pilot_v1"
)


@dataclass(frozen=True)
class PilotRow:
    example_sha256: str
    domain: str
    ordinal: int


@dataclass(frozen=True)
class StatePilotRegistryBinding:
    source_binding_sha256: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_rows: tuple[PilotRow, ...]
    dev_rows: tuple[PilotRow, ...]


def build_gate(
    *,
    checkout: Path,
    state_repo: Path,
    split_gate_json: Path,
    cheap_fit_items: Path,
    state_strong: Path,
    state_strong_repeat: Path,
    agent_diff_result: Path,
    agent_diff_postflight: Path,
    private_root: Path,
) -> dict[str, Any]:
    """Freeze the bounded State adaptation without reading a secret."""

    blockers: list[str] = []
    checkout_readiness = inspect_stepwise_checkout(checkout)
    blockers.extend(checkout_readiness.blockers)

    registry: StatePilotRegistryBinding | None = None
    try:
        registry = _load_registry_binding(
            state_repo=state_repo,
            split_gate_json=split_gate_json,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        blockers.append(f"stepwise_state_registry_invalid:{exc}")

    checkpoint: dict[str, Any] | None = None
    selected_rows: tuple[PilotRow, ...] = ()
    extension_rows: tuple[PilotRow, ...] = ()
    smoke_rows: tuple[PilotRow, ...] = ()
    if registry is not None:
        selected_rows = _balanced_selection(
            registry.fit_rows,
            per_domain=FIT_ROWS_PER_DOMAIN,
        )
        extension_rows = _balanced_selection(
            registry.fit_rows,
            per_domain=FIT_EXTENSION_ROWS_PER_DOMAIN,
        )
        smoke_rows = _balanced_selection(registry.dev_rows, per_domain=1)
        try:
            checkpoint = _checkpoint_progress(
                cheap_fit_items,
                registry=registry,
                selected_rows=selected_rows,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"stepwise_state_fit_checkpoint_invalid:{exc}")

    strong_anchor = _state_strong_anchor(
        state_strong,
        expected_successes=STATE_STRONG_SUCCESSES,
    )
    if strong_anchor["status"] != "BOUND":
        blockers.append("stepwise_state_strong_anchor_not_bound")
    strong_repeat = _state_strong_anchor(
        state_strong_repeat,
        expected_successes=STATE_STRONG_REPEAT_SUCCESSES,
    )
    if strong_repeat["status"] != "BOUND":
        blockers.append("stepwise_state_strong_repeat_not_bound")

    first_surface = _agent_diff_anchor(
        result_path=agent_diff_result,
        postflight_path=agent_diff_postflight,
    )
    if first_surface["status"] != "BOUND":
        blockers.append("stepwise_agent_diff_first_surface_not_bound")

    native_runtime: dict[str, Any] | None = None
    official_test: dict[str, Any] | None = None
    try:
        native_runtime = load_state_bench_native_api(state_repo).public_record()
        test_registry = load_state_bench_official_test_registry(
            state_repo,
            split_gate_json=split_gate_json,
        )
        official_test = test_registry.public_record()
        if registry is not None and (
            test_registry.test_registry_sha256 != registry.test_registry_sha256
        ):
            raise ValueError("stepwise_state_test_registry_mismatch")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        blockers.append(f"stepwise_state_native_authority_invalid:{exc}")

    isolation_violations = find_certspec_imports(
        (
            REPO_ROOT / "src/papc/baselines",
            REPO_ROOT / "scripts/baselines",
        )
    )
    if isolation_violations:
        blockers.append("protected_baseline_certspec_import_violation")
    blockers = list(dict.fromkeys(blockers))

    selected_complete = bool(
        checkpoint and checkpoint.get("selected_complete") is True
    )
    if blockers:
        decision = "BLOCKED"
    elif selected_complete:
        decision = "READY_FOR_ANNOTATION"
    else:
        decision = "READY_FOR_EXPLICIT_BOUNDED_FIT_COLLECTION"

    collect_prefix = max((row.ordinal for row in selected_rows), default=0)
    extension_collect_prefix = max(
        (row.ordinal for row in extension_rows),
        default=0,
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "scope": "bounded_stepwise_state_bench_second_surface",
        "cross_surface_eligibility": {
            "minimum_heterogeneous_surfaces_per_method": 2,
            "selected_surfaces": ["agent_diff", "state_bench"],
            "first_surface": first_surface,
            "second_surface_status": (
                "RESULT_PENDING" if not blockers else "BLOCKED"
            ),
            "method_level_claim_eligible": False,
            "single_surface_result_is_diagnostic_only": True,
        },
        "method": STEPWISE_METHOD,
        "adaptation_boundary": {
            "purpose": "minimal_faithful_second_surface_not_maximal_tuning",
            "kept_from_upstream": [
                "gpt_labeled_stuck_and_milestone_detectors",
                "modernbert_binary_classifiers",
                "five_step_public_history_windows",
                "released_thresholds",
                "permanent_strong_takeover_from_current_state",
            ],
            "not_added": [
                "state_bench_answer_hints",
                "task_id_routing",
                "test_threshold_tuning",
                "full_240_row_fit_collection",
                "full_60_row_dev_collection",
            ],
        },
        "upstream": {
            "revision": STEPWISE_UPSTREAM_REVISION,
            "license": STEPWISE_LICENSE,
            "licensed_subtree": "WebRouter",
            "readiness": checkout_readiness.public_record(),
        },
        "model_pair": {
            "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "monitor_base_model": STEPWISE_MONITOR_BASE_MODEL,
        },
        "released_policy": {
            "stuck_threshold": STEPWISE_STUCK_THRESHOLD,
            "milestone_threshold": STEPWISE_MILESTONE_THRESHOLD,
            "threshold_policy": "released_defaults_no_dev_or_test_tuning",
        },
        "fit_plan": {
            "selection": "first_four_frozen_fit_rows_per_state_domain",
            "selected_count": len(selected_rows),
            "selected_example_sha256s": [
                row.example_sha256 for row in selected_rows
            ],
            "domain_counts": dict(
                sorted(Counter(row.domain for row in selected_rows).items())
            ),
            "sequential_collection_prefix_rows": collect_prefix,
            "checkpoint": checkpoint,
            "planned_annotation_provider_attempts": (
                FIT_SELECTED_ROWS
                * STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY
            ),
            "extension_only_if_label_degenerate": {
                "selection": "first_five_frozen_fit_rows_per_state_domain",
                "selected_count": len(extension_rows),
                "sequential_collection_prefix_rows": extension_collect_prefix,
                "maximum_annotation_provider_attempts": (
                    FIT_EXTENSION_SELECTED_ROWS
                    * STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY
                ),
                "official_outcomes_not_used": True,
            },
        },
        "artifact_gate": {
            "task_level_internal_split": "80/20_seed_42",
            "epochs": 5,
            "learning_rate": 2e-5,
            "max_length": 512,
            "requires_both_training_labels_per_detector": True,
            "does_not_tune_runtime_thresholds": True,
        },
        "heldout_smoke": {
            "purpose": "runtime_and_receipt_integrity_only_not_quality_tuning",
            "selection": "first_frozen_dev_row_per_state_domain",
            "row_count": len(smoke_rows),
            "rows": [
                {"example_sha256": row.example_sha256, "domain": row.domain}
                for row in smoke_rows
            ],
            "score_is_not_a_threshold_selection_input": True,
        },
        "official_test": {
            "registry": official_test,
            "rows": STATE_TEST_ROWS,
            "official_scorer": STATE_BENCH_OFFICIAL_SCORER,
            "quality_anchor": strong_anchor,
            "strong_repeat_variance_disclosure": strong_repeat,
            "reported_successes": (
                "min(raw_stepwise_successes,92); denominator=150"
            ),
            "above_strong_single_run_policy": (
                "variance_not_a_higher_routing_ceiling"
            ),
            "cost_rule": (
                "count_cheap_agent_strong_agent_milestone_verifier_"
                "user_simulator_and_evaluator_on_the_full_denominator"
            ),
            "annotation_cost_reported_separately": True,
        },
        "registries": (
            {
                "source_binding_sha256": registry.source_binding_sha256,
                "fit_registry_sha256": registry.fit_registry_sha256,
                "dev_registry_sha256": registry.dev_registry_sha256,
                "test_registry_sha256": registry.test_registry_sha256,
            }
            if registry
            else None
        ),
        "native_runtime_authority": native_runtime,
        "private_root": str(Path(private_root).resolve()),
        "provider_calls": 0,
        "api_spend_performed": False,
        "secret_reads": 0,
        "execute": False,
        "official_test_outcomes_read": False,
        "protected_baseline_import_isolation": not isolation_violations,
        "blockers": blockers,
        "next_legal_stage": (
            "after_explicit_authorization_collect_only_the_frozen_state_"
            f"fit_prefix_{collect_prefix}"
            if decision == "READY_FOR_EXPLICIT_BOUNDED_FIT_COLLECTION"
            else (
                "run_the_24_call_state_annotation_stage"
                if decision == "READY_FOR_ANNOTATION"
                else "resolve_provider_zero_blockers"
            )
        ),
        "authorizations": {
            "authorizes_fit_collection": False,
            "authorizes_annotation_calls": False,
            "authorizes_smoke_execution": False,
            "authorizes_test_execution": False,
            "authorizes_result_claim": False,
        },
        "public_safe": True,
    }


def _load_registry_binding(
    *,
    state_repo: Path,
    split_gate_json: Path,
) -> StatePilotRegistryBinding:
    split = _read_mapping(split_gate_json)
    if split.get("decision") != "ready_for_training_collection_plan":
        raise ValueError("stepwise_state_split_gate_not_ready")
    entry = split.get("surface_splits", {}).get(STATE_BENCH)
    if not isinstance(entry, dict):
        raise TypeError("stepwise_state_split_entry_missing")
    if (
        entry.get("router_fit_count") != STATE_FIT_ROWS
        or entry.get("router_dev_count") != STATE_DEV_ROWS
        or entry.get("official_test_count") != STATE_TEST_ROWS
    ):
        raise ValueError("stepwise_state_split_counts_mismatch")
    train = _load_state_train(Path(state_repo).resolve())
    fit, dev = deterministic_fit_dev(train)
    fit_registry = FitDevRegistry(
        surface=STATE_BENCH,
        split_role="router_fit",
        source_binding_sha256=entry["source_binding_sha256"],
        registry_sha256=entry["fit_registry_sha256"],
        rows=fit,
    )
    dev_registry = FitDevRegistry(
        surface=STATE_BENCH,
        split_role="router_dev",
        source_binding_sha256=entry["source_binding_sha256"],
        registry_sha256=entry["dev_registry_sha256"],
        rows=dev,
    )
    return StatePilotRegistryBinding(
        source_binding_sha256=entry["source_binding_sha256"],
        fit_registry_sha256=fit_registry.registry_sha256,
        dev_registry_sha256=dev_registry.registry_sha256,
        test_registry_sha256=entry["test_registry_sha256"],
        fit_rows=tuple(
            PilotRow(row.example_sha256, row.stratum, ordinal)
            for ordinal, row in enumerate(fit_registry.rows, start=1)
        ),
        dev_rows=tuple(
            PilotRow(row.example_sha256, row.stratum, ordinal)
            for ordinal, row in enumerate(dev_registry.rows, start=1)
        ),
    )


def _balanced_selection(
    rows: tuple[PilotRow, ...],
    *,
    per_domain: int,
) -> tuple[PilotRow, ...]:
    selected: list[PilotRow] = []
    for domain in STATE_BENCH_TEST_DOMAINS:
        matches = [row for row in rows if row.domain == domain][:per_domain]
        if len(matches) != per_domain:
            raise ValueError("stepwise_state_balanced_selection_incomplete")
        selected.extend(matches)
    return tuple(sorted(selected, key=lambda row: row.ordinal))


def _checkpoint_progress(
    path: Path,
    *,
    registry: StatePilotRegistryBinding,
    selected_rows: tuple[PilotRow, ...],
) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        return {
            "path": str(checkpoint_path),
            "status": "MISSING",
            "observed_prefix_rows": 0,
            "selected_observed_count": 0,
            "selected_complete": False,
            "provider_calls": 0,
        }
    raw = checkpoint_path.read_bytes()
    records = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "checkpoint_header":
        raise ValueError("stepwise_state_checkpoint_header_missing")
    header = records[0]
    if (
        header.get("item_count") != STATE_FIT_ROWS
        or header.get("registry_sha256") != registry.fit_registry_sha256
        or header.get("source_binding_sha256")
        != registry.source_binding_sha256
    ):
        raise ValueError("stepwise_state_checkpoint_header_mismatch")
    items = records[1:]
    if len(items) > len(registry.fit_rows):
        raise ValueError("stepwise_state_checkpoint_too_many_items")
    observed_hashes: list[str] = []
    selected = {row.example_sha256 for row in selected_rows}
    selected_records: list[dict[str, Any]] = []
    for expected, item in zip(
        registry.fit_rows[: len(items)],
        items,
        strict=True,
    ):
        if (
            item.get("record_type") != "item"
            or item.get("ordinal") != expected.ordinal
            or item.get("example_sha256") != expected.example_sha256
        ):
            raise ValueError("stepwise_state_checkpoint_membership_mismatch")
        output = item.get("output")
        if not isinstance(output, dict) or output.get("model_role") != "cheap":
            raise ValueError("stepwise_state_checkpoint_output_invalid")
        episode = output.get("public_episode")
        if not isinstance(episode, list) or not public_steps(episode):
            raise ValueError("stepwise_state_checkpoint_public_steps_missing")
        observed_hashes.append(expected.example_sha256)
        if expected.example_sha256 in selected:
            selected_records.append(item)
    selected_complete = len(selected_records) == len(selected_rows)
    return {
        "path": str(checkpoint_path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "SELECTED_COMPLETE" if selected_complete else "INCOMPLETE",
        "observed_prefix_rows": len(items),
        "selected_observed_count": len(selected_records),
        "selected_complete": selected_complete,
        "observed_example_sha256s": observed_hashes,
        "selected_prefix_sha256": (
            _digest(selected_records) if selected_complete else None
        ),
        "provider_calls": 0,
    }


def _state_strong_anchor(path: Path, *, expected_successes: int) -> dict[str, Any]:
    try:
        payload = _read_mapping(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(path), "status": "INVALID", "error": str(exc)}
    model = payload.get("model_plan")
    summary = payload.get("summary")
    totals = payload.get("agent_runtime", {}).get("totals")
    valid = (
        isinstance(model, dict)
        and model.get("agent_model") == STRONG_MODEL
        and model.get("agent_reasoning_effort") == REASONING_EFFORT
        and isinstance(summary, dict)
        and summary.get("rows") == STATE_TEST_ROWS
        and summary.get("task_completion_pass_count") == expected_successes
        and isinstance(totals, dict)
        and totals.get("rows") == STATE_TEST_ROWS
    )
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "status": "BOUND" if valid else "INVALID",
        "passed": summary.get("task_completion_pass_count")
        if isinstance(summary, dict)
        else None,
        "rows": summary.get("rows") if isinstance(summary, dict) else None,
        "use": "quality_ceiling_or_variance_disclosure_only",
        "current_test_cost_authority": False,
    }


def _agent_diff_anchor(*, result_path: Path, postflight_path: Path) -> dict[str, Any]:
    try:
        result = _read_mapping(result_path)
        postflight = _read_mapping(postflight_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "error": str(exc)}
    accounting = result.get("agent_side_accounting")
    result_context = result.get("context")
    post_context = postflight.get("context")
    valid = (
        result.get("decision") == "complete"
        and result.get("evaluated") == 45
        and result.get("raw_successes") == 43
        and result.get("reported_successes") == 40
        and result.get("strong_quality_ceiling") == "40/45"
        and isinstance(accounting, dict)
        and accounting.get("usage_complete") is True
        and accounting.get("cost_usd") == 21.726045
        and postflight.get("provider_calls") == 0
        and postflight.get("secret_reads") == 0
        and isinstance(result_context, dict)
        and isinstance(post_context, dict)
        and result_context.get("artifact_sha256")
        == post_context.get("artifact_sha256")
    )
    return {
        "status": "BOUND" if valid else "INVALID",
        "surface": "agent_diff",
        "reported_successes": result.get("reported_successes"),
        "raw_successes": result.get("raw_successes"),
        "rows": result.get("evaluated"),
        "cost_usd": accounting.get("cost_usd")
        if isinstance(accounting, dict)
        else None,
        "result_sha256": _file_sha256(result_path),
        "postflight_sha256": _file_sha256(postflight_path),
        "qualification": "single_surface_diagnostic_until_state_result",
    }


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("stepwise_state_json_not_mapping")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json_append_only(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument(
        "--cheap-fit-items", type=Path, default=DEFAULT_CHEAP_FIT_ITEMS
    )
    parser.add_argument("--state-strong", type=Path, default=DEFAULT_STATE_STRONG)
    parser.add_argument(
        "--state-strong-repeat", type=Path, default=DEFAULT_STATE_STRONG_REPEAT
    )
    parser.add_argument(
        "--agent-diff-result", type=Path, default=DEFAULT_AGENT_DIFF_RESULT
    )
    parser.add_argument(
        "--agent-diff-postflight",
        type=Path,
        default=DEFAULT_AGENT_DIFF_POSTFLIGHT,
    )
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        raise SystemExit("provider_zero_gate_rejects_execute")
    report = build_gate(
        checkout=args.checkout,
        state_repo=args.state_repo,
        split_gate_json=args.split_gate_json,
        cheap_fit_items=args.cheap_fit_items,
        state_strong=args.state_strong,
        state_strong_repeat=args.state_strong_repeat,
        agent_diff_result=args.agent_diff_result,
        agent_diff_postflight=args.agent_diff_postflight,
        private_root=args.private_root,
    )
    _write_json_append_only(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["decision"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

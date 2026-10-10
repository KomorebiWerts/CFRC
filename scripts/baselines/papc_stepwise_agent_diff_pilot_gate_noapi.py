#!/usr/bin/env python3
"""Provider-zero gate for a bounded StepWise Agent-Diff pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from papc.baselines.external_router_execution import AGENT_DIFF  # noqa: E402
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevRegistry,
    _load_agent_diff_train,
    deterministic_fit_dev,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    load_native_fit_dev_authority,
)
from papc.baselines.isolation import find_certspec_imports  # noqa: E402
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
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

SCHEMA = "papc.stepwise_agent_diff_pilot_gate_noapi.v1"
PILOT_FIT_ROWS = 12
PILOT_EXTENSION_ROWS = 16
PILOT_SMOKE_ROWS = 4
AGENT_DIFF_TEST_ROWS = 45
AGENT_DIFF_STRONG_SUCCESSES = 40

DEFAULT_CHECKOUT = Path(
    f"{DATA_ROOT}/papc/upstream_sources/stepwise-caf2984b"
)
DEFAULT_SHARED_ROOT = Path(
    f"{DATA_ROOT}/papc/"
    "external_router_live_publication_subset16_v1/"
    "shared_paired_endpoint_fit_dev_and_route_embeddings"
)
DEFAULT_CHEAP_FIT_ITEMS = DEFAULT_SHARED_ROOT / (
    ".shared_paired_endpoint_fit_dev_and_route_embeddings."
    "shared_endpoint_episode.shared_cheap_endpoint."
    "agent_diff.router_fit.jsonl.items.jsonl"
)
DEFAULT_SPLIT_GATE = (
    REPO_ROOT
    / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_AGENT_DIFF_REPO = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo"
)
DEFAULT_AGENT_DIFF_DATASET = (
    DEFAULT_AGENT_DIFF_REPO / "datasets/agent-diff-bench"
)
DEFAULT_AGENT_DIFF_SDK = DEFAULT_AGENT_DIFF_REPO / "sdk/agent-diff-python"
DEFAULT_AGENT_DIFF_SCORER = (
    REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"
)
# The live baseline runner is being evolved by another active experiment.  Bind
# the byte-identical frozen authority copy instead of accepting source drift.
DEFAULT_AGENT_DIFF_BASELINE_AUTHORITY = REPO_ROOT / "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py"
DEFAULT_AGENT_DIFF_STRONG_TEST = REPO_ROOT / (
    "outputs/papc/agent_diff_hf_test45_strong_baseline_public_"
    "20260706_codex_strong45_merged_v2.json"
)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v1"
)
DEFAULT_AGENT_DIFF_BASE_URL = "http://localhost:8020"


@dataclass(frozen=True)
class PilotRegistryBinding:
    """Outcome-blind fit/dev membership needed by the bounded pilot."""

    source_binding_sha256: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    fit_rows: tuple[tuple[str, str], ...]
    dev_rows: tuple[tuple[str, str], ...]


def build_gate(
    *,
    checkout: Path,
    cheap_fit_items: Path,
    split_gate_json: Path,
    agent_diff_repo: Path,
    agent_diff_dataset: Path,
    agent_diff_sdk_path: Path,
    agent_diff_scorer_path: Path,
    agent_diff_baseline_authority_path: Path,
    agent_diff_strong_test: Path,
    private_root: Path,
    agent_diff_base_url: str,
) -> dict[str, Any]:
    """Freeze a small faithful adaptation without reading a provider secret."""

    blockers: list[str] = []
    checkout_readiness = inspect_stepwise_checkout(checkout)
    blockers.extend(checkout_readiness.blockers)

    registry: PilotRegistryBinding | None = None
    try:
        registry = _load_registry_binding(
            split_gate_json=split_gate_json,
            agent_diff_dataset=agent_diff_dataset,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        blockers.append(f"stepwise_agent_diff_registry_invalid:{exc}")

    fit_progress: dict[str, Any] | None = None
    if registry is not None:
        try:
            fit_progress = _fit_checkpoint_progress(
                cheap_fit_items,
                registry=registry,
                selected_rows=PILOT_FIT_ROWS,
            )
            if fit_progress["status"] != "SELECTED_PREFIX_COMPLETE":
                blockers.append(
                    "stepwise_agent_diff_pilot_fit_not_complete:"
                    f"{fit_progress['selected_observed_count']}/"
                    f"{PILOT_FIT_ROWS}"
                )
        except FileNotFoundError:
            fit_progress = _missing_fit_progress(cheap_fit_items)
            blockers.append(
                "stepwise_agent_diff_pilot_fit_not_complete:"
                f"0/{PILOT_FIT_ROWS}"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"stepwise_agent_diff_fit_checkpoint_invalid:{exc}")

    strong_anchor = _strong_test_anchor(agent_diff_strong_test)
    if strong_anchor["status"] != "BOUND":
        blockers.append("stepwise_agent_diff_strong_test_anchor_not_bound")

    native_runtime: dict[str, Any] | None = None
    try:
        native_runtime = load_native_fit_dev_authority(
            surfaces=(AGENT_DIFF,),
            private_root=private_root,
            state_repo=Path("/unused/state"),
            tau2_data_repo=Path("/unused/tau2-data"),
            tau2_runtime_repo=Path("/unused/tau2-runtime"),
            agent_diff_repo=agent_diff_repo,
            agent_diff_dataset=agent_diff_dataset,
            agent_diff_sdk_path=agent_diff_sdk_path,
            agent_diff_scorer_path=agent_diff_scorer_path,
            agent_diff_baseline_authority_path=(
                agent_diff_baseline_authority_path
            ),
            agent_diff_base_url=agent_diff_base_url,
        ).public_record()
    except (OSError, TypeError, ValueError, ImportError) as exc:
        blockers.append(f"stepwise_agent_diff_native_authority_invalid:{exc}")

    isolation_violations = find_certspec_imports(
        (
            REPO_ROOT / "src/papc/baselines",
            REPO_ROOT / "scripts/baselines",
        )
    )
    if isolation_violations:
        blockers.append("protected_baseline_certspec_import_violation")
    blockers = list(dict.fromkeys(blockers))

    fit_rows = registry.fit_rows[:PILOT_FIT_ROWS] if registry else ()
    extension_rows = (
        registry.fit_rows[PILOT_FIT_ROWS:PILOT_EXTENSION_ROWS]
        if registry
        else ()
    )
    smoke_rows = _one_per_service(registry.dev_rows) if registry else ()
    if registry is not None and len(smoke_rows) != PILOT_SMOKE_ROWS:
        blockers.append("stepwise_agent_diff_smoke_service_coverage_invalid")
    decision = "READY_FOR_ANNOTATION" if not blockers else "BLOCKED"
    return {
        "schema": SCHEMA,
        "decision": decision,
        "scope": "bounded_stepwise_agent_diff_pilot",
        "method": STEPWISE_METHOD,
        "adaptation_boundary": {
            "purpose": (
                "credible_story_closing_baseline_not_a_maximal_stepwise_"
                "reimplementation"
            ),
            "kept_from_upstream": [
                "gpt_labeled_stuck_detector",
                "gpt_labeled_milestone_detector",
                "modernbert_binary_classifiers",
                "five_step_public_history_windows",
                "stuck_immediate_permanent_takeover",
                "milestone_strong_verification_then_permanent_takeover_on_"
                "failure",
            ],
            "deliberately_not_added": [
                "benchmark_specific_answer_hints",
                "test_threshold_tuning",
                "full_141_row_fit_collection",
                "full_38_row_strong_dev_collection",
                "general_purpose_stepwise_baseline_framework",
            ],
        },
        "upstream": {
            "repository": "https://github.com/yale-nlp/StepWise",
            "revision": STEPWISE_UPSTREAM_REVISION,
            "licensed_subtree": "WebRouter",
            "license": STEPWISE_LICENSE,
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
            "selection": "first_12_rows_of_frozen_outcome_blind_fit_registry",
            "selected_count": len(fit_rows),
            "selected_example_sha256s": [row[0] for row in fit_rows],
            "service_counts": dict(
                sorted(Counter(row[1] for row in fit_rows).items())
            ),
            "checkpoint": fit_progress,
            "annotation_calls_per_trajectory": (
                STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY
            ),
            "planned_annotation_provider_attempts": (
                PILOT_FIT_ROWS
                * STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY
            ),
            "extension_only_if_label_degenerate": {
                "rule": (
                    "append_the_next_four_frozen_fit_rows_only_if_either_"
                    "detector_has_no_positive_or_no_negative_training_label"
                ),
                "maximum_fit_rows": PILOT_EXTENSION_ROWS,
                "extension_example_sha256s": [
                    row[0] for row in extension_rows
                ],
                "maximum_annotation_provider_attempts": (
                    PILOT_EXTENSION_ROWS
                    * STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY
                ),
                "official_outcomes_not_used": True,
            },
        },
        "artifact_gate": {
            "task_level_internal_split": "80/20_seed_42",
            "train_script": "pinned_upstream_train_router.py",
            "epochs": 5,
            "learning_rate": 2e-5,
            "max_length": 512,
            "requires_both_labels_in_each_training_dataset": True,
            "does_not_tune_runtime_thresholds": True,
        },
        "heldout_smoke": {
            "purpose": "runtime_and_receipt_integrity_only_not_quality_tuning",
            "selection": "first_frozen_dev_row_per_service",
            "row_count": len(smoke_rows),
            "rows": [
                {"example_sha256": row[0], "service": row[1]}
                for row in smoke_rows
            ],
            "go_rule": (
                "all_four_rows_evaluated_without_infra_or_receipt_error"
            ),
            "score_is_not_a_threshold_selection_input": True,
        },
        "official_test": {
            "rows": AGENT_DIFF_TEST_ROWS,
            "quality_anchor": strong_anchor,
            "reported_successes": (
                "min(raw_stepwise_successes,40); denominator=45"
            ),
            "above_strong_single_run_policy": (
                "variance_not_a_higher_routing_ceiling"
            ),
            "cost_rule": (
                "count_all_cheap_agent_strong_agent_and_milestone_verifier_"
                "provider_calls_on_the_full_45_row_denominator"
            ),
            "annotation_cost_reported_separately": True,
        },
        "registries": (
            {
                "source_binding_sha256": registry.source_binding_sha256,
                "fit_registry_sha256": registry.fit_registry_sha256,
                "dev_registry_sha256": registry.dev_registry_sha256,
            }
            if registry
            else None
        ),
        "native_runtime_authority": native_runtime,
        "provider_calls": 0,
        "api_spend_performed": False,
        "secret_reads": 0,
        "execute": False,
        "official_test_outcomes_read": False,
        "protected_baseline_import_isolation": not isolation_violations,
        "blockers": blockers,
        "next_legal_stage": (
            "complete_only_the_selected_fit_prefix_then_run_the_24_call_"
            "append_only_annotation_stage"
            if decision == "BLOCKED"
            else "run_the_24_call_append_only_annotation_stage"
        ),
        "authorizations": {
            "authorizes_endpoint_fit_completion": False,
            "authorizes_annotation_calls": False,
            "authorizes_training": False,
            "authorizes_smoke_execution": False,
            "authorizes_test_execution": False,
            "authorizes_baseline_result_claim": False,
        },
        "public_safe": True,
    }


def _load_registry_binding(
    *,
    split_gate_json: Path,
    agent_diff_dataset: Path,
) -> PilotRegistryBinding:
    split = json.loads(Path(split_gate_json).read_text(encoding="utf-8"))
    entry = split["surface_splits"][AGENT_DIFF]
    source_binding = entry["source_binding_sha256"]
    fit_sha = entry["fit_registry_sha256"]
    dev_sha = entry["dev_registry_sha256"]
    train_rows = _load_agent_diff_train(Path(agent_diff_dataset).resolve())
    fit_rows, dev_rows = deterministic_fit_dev(train_rows)
    fit = FitDevRegistry(
        surface=AGENT_DIFF,
        split_role="router_fit",
        source_binding_sha256=source_binding,
        registry_sha256=fit_sha,
        rows=fit_rows,
    )
    dev = FitDevRegistry(
        surface=AGENT_DIFF,
        split_role="router_dev",
        source_binding_sha256=source_binding,
        registry_sha256=dev_sha,
        rows=dev_rows,
    )
    return PilotRegistryBinding(
        source_binding_sha256=source_binding,
        fit_registry_sha256=fit.registry_sha256,
        dev_registry_sha256=dev.registry_sha256,
        fit_rows=tuple((row.example_sha256, row.stratum) for row in fit.rows),
        dev_rows=tuple((row.example_sha256, row.stratum) for row in dev.rows),
    )


def _fit_checkpoint_progress(
    path: Path,
    *,
    registry: PilotRegistryBinding,
    selected_rows: int,
) -> dict[str, Any]:
    records = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or records[0].get("record_type") != "checkpoint_header":
        raise ValueError("endpoint_items_header_missing")
    header = records[0]
    if header.get("registry_sha256") != registry.fit_registry_sha256:
        raise ValueError("endpoint_items_registry_sha256_mismatch")
    declared = header.get("item_count")
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared != len(registry.fit_rows)
    ):
        raise ValueError("endpoint_items_declared_count_mismatch")
    items = records[1:]
    if len(items) > declared:
        raise ValueError("endpoint_items_exceed_declared_count")
    assistant_steps = 0
    for index, item in enumerate(items, start=1):
        if item.get("record_type") != "item" or item.get("ordinal") != index:
            raise ValueError("endpoint_item_ordinal_or_type_mismatch")
        expected_sha = registry.fit_rows[index - 1][0]
        if item.get("example_sha256") != expected_sha:
            raise ValueError("endpoint_item_not_frozen_registry_prefix")
        output = item.get("output")
        if not isinstance(output, dict) or output.get("model_role") != "cheap":
            raise ValueError("endpoint_item_output_or_role_invalid")
        request = output.get("original_request")
        episode = output.get("public_episode")
        if (
            not isinstance(request, str)
            or not request.strip()
            or not isinstance(episode, list)
            or not episode
        ):
            raise ValueError("endpoint_item_public_trajectory_invalid")
        score = output.get("official_score")
        if isinstance(score, bool) or score not in {0, 0.0, 1, 1.0}:
            raise ValueError("endpoint_item_official_score_not_binary")
        step_count = len(public_steps(tuple(episode)))
        if step_count < 1:
            raise ValueError("endpoint_item_has_no_public_agent_step")
        if index <= selected_rows:
            assistant_steps += step_count
    observed_selected = min(len(items), selected_rows)
    selected_prefix = records[: observed_selected + 1]
    return {
        "path": str(Path(path)),
        "file_sha256": _file_sha256(Path(path)),
        "selected_prefix_sha256": _semantic_sha256(selected_prefix),
        "declared_full_fit_count": declared,
        "observed_full_checkpoint_count": len(items),
        "selected_fit_count": selected_rows,
        "selected_observed_count": observed_selected,
        "selected_prefix_complete": len(items) >= selected_rows,
        "selected_public_assistant_steps_observed": assistant_steps,
        "status": (
            "SELECTED_PREFIX_COMPLETE"
            if len(items) >= selected_rows
            else "SELECTED_PREFIX_INCOMPLETE"
        ),
        "official_outcomes_used_for_selection": False,
    }


def _missing_fit_progress(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "declared_full_fit_count": None,
        "observed_full_checkpoint_count": 0,
        "selected_prefix_sha256": None,
        "selected_fit_count": PILOT_FIT_ROWS,
        "selected_observed_count": 0,
        "selected_prefix_complete": False,
        "status": "MISSING",
        "official_outcomes_used_for_selection": False,
    }


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _one_per_service(
    rows: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row[1] not in seen:
            selected.append(row)
            seen.add(row[1])
    return tuple(selected)


def _strong_test_anchor(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"path": str(path), "status": "MISSING"}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(path), "status": "INVALID", "error": str(exc)}
    model = payload.get("model_plan")
    dataset = payload.get("dataset")
    aggregate = payload.get("aggregate")
    valid = (
        isinstance(model, dict)
        and model.get("model") == STRONG_MODEL
        and model.get("reasoning_effort") == REASONING_EFFORT
        and isinstance(dataset, dict)
        and dataset.get("expected_rows") == AGENT_DIFF_TEST_ROWS
        and dataset.get("loaded_rows") == AGENT_DIFF_TEST_ROWS
        and isinstance(aggregate, dict)
        and aggregate.get("evaluated") == AGENT_DIFF_TEST_ROWS
        and aggregate.get("passed") == AGENT_DIFF_STRONG_SUCCESSES
        and aggregate.get("errors") == 0
    )
    return {
        "path": str(path),
        "file_sha256": _file_sha256(Path(path)),
        "status": "BOUND" if valid else "INVALID",
        "passed": aggregate.get("passed") if isinstance(aggregate, dict) else None,
        "rows": (
            aggregate.get("evaluated") if isinstance(aggregate, dict) else None
        ),
        "use": "quality_ceiling_only",
        "cost_authority": False,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_append_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument(
        "--cheap-fit-items", type=Path, default=DEFAULT_CHEAP_FIT_ITEMS
    )
    parser.add_argument(
        "--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE
    )
    parser.add_argument(
        "--agent-diff-repo", type=Path, default=DEFAULT_AGENT_DIFF_REPO
    )
    parser.add_argument(
        "--agent-diff-dataset", type=Path, default=DEFAULT_AGENT_DIFF_DATASET
    )
    parser.add_argument(
        "--agent-diff-sdk-path", type=Path, default=DEFAULT_AGENT_DIFF_SDK
    )
    parser.add_argument(
        "--agent-diff-scorer-path", type=Path, default=DEFAULT_AGENT_DIFF_SCORER
    )
    parser.add_argument(
        "--agent-diff-baseline-authority-path",
        type=Path,
        default=DEFAULT_AGENT_DIFF_BASELINE_AUTHORITY,
    )
    parser.add_argument(
        "--agent-diff-strong-test",
        type=Path,
        default=DEFAULT_AGENT_DIFF_STRONG_TEST,
    )
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument(
        "--agent-diff-base-url", default=DEFAULT_AGENT_DIFF_BASE_URL
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_gate(
        checkout=args.checkout,
        cheap_fit_items=args.cheap_fit_items,
        split_gate_json=args.split_gate_json,
        agent_diff_repo=args.agent_diff_repo,
        agent_diff_dataset=args.agent_diff_dataset,
        agent_diff_sdk_path=args.agent_diff_sdk_path,
        agent_diff_scorer_path=args.agent_diff_scorer_path,
        agent_diff_baseline_authority_path=(
            args.agent_diff_baseline_authority_path
        ),
        agent_diff_strong_test=args.agent_diff_strong_test,
        private_root=args.private_root,
        agent_diff_base_url=args.agent_diff_base_url,
    )
    write_json_append_only(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "fit_progress": report["fit_plan"]["checkpoint"],
                "planned_annotation_provider_attempts": report["fit_plan"][
                    "planned_annotation_provider_attempts"
                ],
                "provider_calls": report["provider_calls"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "READY_FOR_ANNOTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())

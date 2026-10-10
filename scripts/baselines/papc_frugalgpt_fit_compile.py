#!/usr/bin/env python3
"""Train and seal one receipt-bound FrugalGPT surface artifact locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.agent_diff_router_driver import (  # noqa: E402
    load_agent_diff_official_registry,
)
from papc.baselines.external_router_execution import (  # noqa: E402
    load_execution_plan,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevRegistry,
    load_fit_dev_registry,
)
from papc.baselines.frugalgpt_fit_compile import (  # noqa: E402
    build_frugalgpt_dev_examples,
    build_frugalgpt_training_examples,
    find_shared_endpoint_job,
    load_bound_endpoint_episodes,
)
from papc.baselines.frugalgpt_upstream import (  # noqa: E402
    FrugalGPTFitOnlyDistilBertScorer,
    FrugalGPTTrainingExport,
    bind_frugalgpt_scorer_artifact,
    frugalgpt_training_dependency_blockers,
    select_frugalgpt_operating_point,
    train_and_export_frugalgpt_scorer,
)
from papc.baselines.router_contract import (  # noqa: E402
    AGENT_DIFF,
    STATE_BENCH,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    SplitManifest,
    assert_disjoint_manifests,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    load_state_bench_official_test_registry,
)
from papc.baselines.tau2_router_driver import (  # noqa: E402
    load_tau2_official_test_registry,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.frugalgpt.fit_compile.v1"
DEFAULT_SPLIT_GATE = (
    REPO_ROOT
    / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_BUDGET_GATE = (
    REPO_ROOT
    / "outputs/papc/external_router_budget_gate_noapi_20260730_v10.json"
)
DEFAULT_AUTHORITY = (
    REPO_ROOT
    / "outputs/papc/table1_authority_manifest_noapi_20260730_v2.json"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_TAU2_REPO = Path(
    f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo"
)
DEFAULT_AGENT_DIFF_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/"
    "datasets/agent-diff-bench"
)
DEFAULT_BASE_MODEL_DIR = (
    Path(f"{DATA_ROOT}/papc/upstream_models")
    / "distilbert-base-uncased-12040acc"
)
SURFACES = (STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE, AGENT_DIFF)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "compile"), required=True)
    parser.add_argument("--surface", choices=SURFACES, required=True)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument(
        "--budget-gate-json",
        type=Path,
        default=DEFAULT_BUDGET_GATE,
    )
    parser.add_argument(
        "--authority-manifest-json",
        type=Path,
        default=DEFAULT_AUTHORITY,
    )
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_TAU2_REPO)
    parser.add_argument(
        "--agent-diff-dataset",
        type=Path,
        default=DEFAULT_AGENT_DIFF_DATASET,
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-export-json", type=Path, required=True)
    parser.add_argument("--artifact-output-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(
    argv: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    try:
        plan = load_execution_plan(
            split_gate_json=args.split_gate_json,
            budget_gate_json=args.budget_gate_json,
            authority_manifest_json=args.authority_manifest_json,
            private_root=args.private_root,
        )
        jobs = {
            "cheap_fit": find_shared_endpoint_job(
                plan,
                surface=args.surface,
                split_role="router_fit",
                model_role="cheap",
            ),
            "cheap_dev": find_shared_endpoint_job(
                plan,
                surface=args.surface,
                split_role="router_dev",
                model_role="cheap",
            ),
            "strong_dev": find_shared_endpoint_job(
                plan,
                surface=args.surface,
                split_role="router_dev",
                model_role="strong",
            ),
        }
        registries = {
            name: load_fit_dev_registry(
                job=job,
                split_gate_json=args.split_gate_json,
                state_repo=args.state_repo,
                tau2_repo=args.tau2_repo,
                agent_diff_dataset=args.agent_diff_dataset,
            )
            for name, job in jobs.items()
        }
        fit_manifest = _manifest(registries["cheap_fit"])
        dev_manifest = _manifest(registries["cheap_dev"])
        if _manifest(registries["strong_dev"]) != dev_manifest:
            raise ValueError("frugalgpt_strong_dev_registry_mismatch")
        test_manifest = _load_test_manifest(args)
        assert_disjoint_manifests(
            fit_manifest,
            dev_manifest,
            test_manifest,
        )
        report = _base_report(
            args,
            plan_sha256=plan.sha256,
            fit_manifest=fit_manifest,
            dev_manifest=dev_manifest,
            test_manifest=test_manifest,
            jobs=jobs,
        )
        blockers = _preflight_blockers(args, jobs=jobs)
        if blockers:
            return 2, {
                **report,
                "decision": "BLOCKED",
                "blockers": blockers,
            }
        if not args.execute:
            return 0, {
                **report,
                "decision": "ready_for_explicit_local_execute",
                "blockers": ["execute_flag_missing"],
            }
        if args.mode == "train":
            local_result = _run_train(
                args,
                plan=plan,
                job=jobs["cheap_fit"],
                fit_manifest=fit_manifest,
            )
        else:
            local_result = _run_compile(
                args,
                plan=plan,
                jobs=jobs,
                fit_manifest=fit_manifest,
                dev_manifest=dev_manifest,
                test_manifest=test_manifest,
            )
        return 0, {
            **report,
            "decision": "local_operation_complete",
            "local_result": local_result,
            "blockers": [],
        }
    except Exception as exc:
        return 2, {
            "schema": SCHEMA,
            "mode": args.mode,
            "surface": args.surface,
            "decision": "BLOCKED",
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "official_test_outcomes_read": False,
            "blockers": [f"{type(exc).__name__}:{exc}"],
        }


def _run_train(
    args: argparse.Namespace,
    *,
    plan: Any,
    job: Any,
    fit_manifest: SplitManifest,
) -> dict[str, Any]:
    episodes = load_bound_endpoint_episodes(
        job.output_path,
        plan=plan,
        job=job,
        manifest=fit_manifest,
        model_role="cheap",
    )
    training_examples = build_frugalgpt_training_examples(
        episodes,
        fit_manifest=fit_manifest,
    )
    export = train_and_export_frugalgpt_scorer(
        examples=training_examples,
        fit_manifest=fit_manifest,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        base_model_dir=args.base_model_dir.resolve(),
        device=args.device,
    )
    _write_json_append_only(
        args.training_export_json,
        export.public_record(),
    )
    return {
        "fit_rows": len(training_examples),
        "fit_negative_rows": export.checkpoint_manifest.fit_negative_rows,
        "fit_positive_rows": export.checkpoint_manifest.fit_positive_rows,
        "checkpoint_bundle_sha256": export.checkpoint_bundle_sha256,
        "checkpoint_manifest_sha256": export.checkpoint_manifest.sha256,
        "training_export_json": str(
            args.training_export_json.resolve()
        ),
    }


def _run_compile(
    args: argparse.Namespace,
    *,
    plan: Any,
    jobs: Mapping[str, Any],
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
) -> dict[str, Any]:
    training_record = json.loads(
        args.training_export_json.read_text(encoding="utf-8")
    )
    export = FrugalGPTTrainingExport.from_public_record(
        training_record,
        checkpoint_dir=args.checkpoint_dir.resolve(),
    )
    if (
        export.checkpoint_manifest.fit_registry_sha256
        != fit_manifest.registry_sha256
    ):
        raise ValueError("frugalgpt_training_export_fit_mismatch")
    scorer = FrugalGPTFitOnlyDistilBertScorer.load(
        training_export=export,
    )
    cheap = load_bound_endpoint_episodes(
        jobs["cheap_dev"].output_path,
        plan=plan,
        job=jobs["cheap_dev"],
        manifest=dev_manifest,
        model_role="cheap",
    )
    strong = load_bound_endpoint_episodes(
        jobs["strong_dev"].output_path,
        plan=plan,
        job=jobs["strong_dev"],
        manifest=dev_manifest,
        model_role="strong",
    )
    dev_examples = build_frugalgpt_dev_examples(
        cheap,
        strong,
        dev_manifest=dev_manifest,
        score_cheap_answer=scorer,
    )
    operating_point = select_frugalgpt_operating_point(
        dev_examples,
        dev_manifest=dev_manifest,
    )
    artifact = bind_frugalgpt_scorer_artifact(
        surface=args.surface,
        fit_manifest=fit_manifest,
        dev_manifest=dev_manifest,
        test_manifest=test_manifest,
        checkpoint_bundle_sha256=export.checkpoint_bundle_sha256,
        checkpoint_manifest=export.checkpoint_manifest,
        operating_point=operating_point,
    )
    _write_json_append_only(
        args.artifact_output_json,
        artifact.public_record(),
    )
    return {
        "artifact_sha256": artifact.sha256,
        "artifact_output_json": str(
            args.artifact_output_json.resolve()
        ),
        "accept_threshold": operating_point.accept_threshold,
        "dev_quality": operating_point.dev_quality,
        "strong_full_dev_anchor": (
            operating_point.strong_full_dev_anchor
        ),
        "dev_cost_usd": operating_point.dev_cost_usd,
        "dev_rows": operating_point.dev_rows,
    }


def _preflight_blockers(
    args: argparse.Namespace,
    *,
    jobs: Mapping[str, Any],
) -> list[str]:
    blockers = []
    required_outputs = (
        (jobs["cheap_fit"].output_path,)
        if args.mode == "train"
        else (
            jobs["cheap_dev"].output_path,
            jobs["strong_dev"].output_path,
            args.training_export_json,
            args.checkpoint_dir,
        )
    )
    for path in required_outputs:
        if not Path(path).exists():
            blockers.append(f"required_input_missing:{Path(path)}")
    if args.mode == "train":
        blockers.extend(
            frugalgpt_training_dependency_blockers(
                base_model_dir=args.base_model_dir,
                device=args.device,
            )
        )
        for path in (args.checkpoint_dir, args.training_export_json):
            if os.path.lexists(path):
                blockers.append(f"append_only_target_exists:{path}")
    else:
        if args.artifact_output_json is None:
            blockers.append("artifact_output_json_required")
        elif os.path.lexists(args.artifact_output_json):
            blockers.append(
                f"append_only_target_exists:{args.artifact_output_json}"
            )
    return blockers


def _base_report(
    args: argparse.Namespace,
    *,
    plan_sha256: str,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    jobs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": args.mode,
        "surface": args.surface,
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "official_test_outcomes_read": False,
        "plan_sha256": plan_sha256,
        "fit_registry_sha256": fit_manifest.registry_sha256,
        "dev_registry_sha256": dev_manifest.registry_sha256,
        "test_registry_sha256": test_manifest.registry_sha256,
        "fit_rows": len(fit_manifest.example_sha256s),
        "dev_rows": len(dev_manifest.example_sha256s),
        "test_rows": len(test_manifest.example_sha256s),
        "input_jobs": {
            name: {
                "job_id": job.job_id,
                "output_path": str(job.output_path),
            }
            for name, job in jobs.items()
        },
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "training_export_json": str(
            args.training_export_json.resolve()
        ),
        "artifact_output_json": (
            str(args.artifact_output_json.resolve())
            if args.artifact_output_json is not None
            else None
        ),
    }


def _manifest(registry: FitDevRegistry) -> SplitManifest:
    return SplitManifest(
        role=registry.split_role,
        registry_sha256=registry.registry_sha256,
        example_sha256s=tuple(
            row.example_sha256 for row in registry.rows
        ),
    )


def _load_test_manifest(args: argparse.Namespace) -> SplitManifest:
    if args.surface == STATE_BENCH:
        registry = load_state_bench_official_test_registry(
            args.state_repo,
            split_gate_json=args.split_gate_json,
        )
        sha256 = registry.test_registry_sha256
        examples = tuple(item.example_sha256 for item in registry.tasks)
    elif args.surface in {TAU2_RETAIL, TAU2_AIRLINE}:
        registry = load_tau2_official_test_registry(
            args.tau2_repo,
            domain=(
                "retail"
                if args.surface == TAU2_RETAIL
                else "airline"
            ),
        )
        sha256 = registry.test_registry_sha256
        examples = tuple(item.example_sha256 for item in registry.tasks)
    else:
        registry = load_agent_diff_official_registry(
            args.agent_diff_dataset,
            split_gate_json=args.split_gate_json,
        )
        sha256 = registry.split.test_registry_sha256
        examples = tuple(item.example_sha256 for item in registry.tasks)
    return SplitManifest(
        role="official_test",
        registry_sha256=sha256,
        example_sha256s=examples,
    )


def _write_json_append_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(argv)
    if args.report_json is not None:
        _write_json_append_only(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

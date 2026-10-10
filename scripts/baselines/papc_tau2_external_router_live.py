#!/usr/bin/env python3
"""Run one explicitly authorized, full Tau2 external-router Table 1 cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.external_router_execution import (  # noqa: E402
    TABLE1_AUTHORITY_FILE_SHA256,
    TABLE1_AUTHORITY_SEMANTIC_SHA256,
    TEST_STAGE,
    BatchCoordinator,
    ExecutionJob,
    ExecutionPlan,
    JobExecutionResult,
    StageExecutionAuthorization,
    load_execution_plan,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    activate_tau2_dependency_environment,
)
from papc.baselines.opensquilla_upstream import (  # noqa: E402
    OPENSQUILLA_TRAINING_SOURCE_SHA256S,
    OPENSQUILLA_UPSTREAM_REVISION,
    opensquilla_bundle_tree_sha256,
)
from papc.baselines.routellm_bert_upstream import (  # noqa: E402
    RouteLLMBertArtifact,
)
from papc.baselines.router_contract import (  # noqa: E402
    AUTOMIX,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    ROUTELLM,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    validate_router_runtime_environment,
)
from papc.baselines.table1_results import (  # noqa: E402
    SurfaceAuthority,
    load_table1_authority,
)
from papc.baselines.tau2_external_router_live import (  # noqa: E402
    Tau2CellExecution,
    Tau2PolicyResources,
    Tau2TypedArtifact,
    execute_tau2_official_cell,
    load_tau2_typed_artifact,
    validate_tau2_live_bindings,
    write_public_receipt_append_only,
)
from papc.baselines.tau2_native_runtime import (  # noqa: E402
    Tau2NativeAPI,
    Tau2NativeRunConfig,
    load_tau2_native_api,
    validate_tau2_native_source,
)
from papc.baselines.tau2_router_driver import (  # noqa: E402
    Tau2OfficialTestRegistry,
    load_tau2_official_test_registry,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.tau2.external_router_live.v1"
DEFAULT_TAU2_DATA_REPO = Path(f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo")
DEFAULT_TAU2_RUNTIME_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")


@dataclass(frozen=True)
class PreparedTau2LiveRun:
    runtime_environment_sha256: str
    tau2_dependency_environment_sha256: str
    plan: ExecutionPlan
    job: ExecutionJob
    registry: Tau2OfficialTestRegistry
    authority: SurfaceAuthority
    typed_artifact: Tau2TypedArtifact
    resources: Tau2PolicyResources
    authorization: StageExecutionAuthorization
    native_config: Tau2NativeRunConfig
    runtime_source: Mapping[str, str]
    runtime_resource_binding: Mapping[str, Any]
    public_result_json: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-gate-json", type=Path, required=True)
    parser.add_argument("--budget-gate-json", type=Path, required=True)
    parser.add_argument("--authority-json", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--typed-artifact-json", type=Path, required=True)
    parser.add_argument(
        "--tau2-repo",
        type=Path,
        default=DEFAULT_TAU2_DATA_REPO,
    )
    parser.add_argument(
        "--tau2-runtime-repo",
        type=Path,
        default=DEFAULT_TAU2_RUNTIME_REPO,
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--automix-tokenizer-root", type=Path)
    parser.add_argument("--mtrouter-device", default="cpu")
    parser.add_argument("--opensquilla-checkout-root", type=Path)
    parser.add_argument("--opensquilla-bundle-dir", type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--authorization-id")
    parser.add_argument("--plan-sha256")
    parser.add_argument(
        "--stage-provider-attempt-hard-cap",
        type=int,
    )
    parser.add_argument(
        "--stage-observed-spend-stop-usd",
        type=Decimal,
    )
    parser.add_argument("--public-result-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--local-hardware-label")
    parser.add_argument("--local-software-label")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def prepare_live_run(
    args: argparse.Namespace,
    *,
    artifact_loader: Callable[..., Tau2TypedArtifact] = (load_tau2_typed_artifact),
    runtime_source_validator: Callable[[Path], Mapping[str, str]] = (
        validate_tau2_native_source
    ),
    runtime_environment_validator: Callable[[], str] | None = None,
    tau2_dependency_environment_validator: Callable[[Path], str] = (
        activate_tau2_dependency_environment
    ),
) -> PreparedTau2LiveRun:
    environment_validator = (
        runtime_environment_validator or validate_router_runtime_environment
    )
    runtime_environment_sha256 = environment_validator()
    tau2_dependency_environment_sha256 = (
        tau2_dependency_environment_validator(
            Path(args.tau2_runtime_repo).resolve()
        )
    )
    plan = load_execution_plan(
        split_gate_json=args.split_gate_json,
        budget_gate_json=args.budget_gate_json,
        authority_manifest_json=args.authority_json,
        private_root=args.private_root,
    )
    try:
        job = plan.job(args.job_id)
    except KeyError as exc:
        raise ValueError("explicit_job_id_not_in_execution_plan") from exc
    if job.stage != TEST_STAGE or job.kind != "official_test_cell":
        raise ValueError("tau2_live_job_not_publication_test_cell")
    if job.surface not in {"tau2_retail", "tau2_airline"}:
        raise ValueError("tau2_live_job_not_tau2")
    domain = "retail" if job.surface == "tau2_retail" else "airline"
    registry = load_tau2_official_test_registry(
        args.tau2_repo,
        domain=domain,
    )
    runtime_source = runtime_source_validator(args.tau2_runtime_repo)
    runtime_registry = load_tau2_official_test_registry(
        Path(runtime_source["runtime_repo"]),
        domain=domain,
    )
    if (
        runtime_registry.source_binding_sha256 != registry.source_binding_sha256
        or runtime_registry.test_registry_sha256 != registry.test_registry_sha256
    ):
        raise ValueError("tau2_runtime_data_registry_mismatch")
    authority_manifest = load_table1_authority(args.authority_json)
    if (
        authority_manifest.sha256 != TABLE1_AUTHORITY_SEMANTIC_SHA256
        or authority_manifest.sha256
        != plan.authority_manifest_semantic_sha256
        or plan.authority_manifest_file_sha256
        != TABLE1_AUTHORITY_FILE_SHA256
    ):
        raise ValueError("tau2_table1_authority_manifest_mismatch")
    authority = authority_manifest.for_surface(job.surface)
    typed_artifact = artifact_loader(
        method=job.method,
        path=args.typed_artifact_json,
    )
    validate_tau2_live_bindings(
        job=job,
        registry=registry,
        authority=authority,
        typed_artifact=typed_artifact,
    )
    if not isinstance(args.authorization_id, str):
        raise ValueError("explicit_stage_authorization_id_missing")
    if not isinstance(args.plan_sha256, str):
        raise ValueError("explicit_plan_sha256_missing")
    if args.stage_provider_attempt_hard_cap is None:
        raise ValueError("explicit_stage_attempt_cap_missing")
    if args.stage_observed_spend_stop_usd is None:
        raise ValueError("explicit_stage_usd_stop_missing")
    authorization = StageExecutionAuthorization(
        authorization_id=args.authorization_id,
        plan_sha256=args.plan_sha256,
        stage=job.stage,
        provider_attempt_hard_cap=(args.stage_provider_attempt_hard_cap),
        observed_spend_stop_usd=(args.stage_observed_spend_stop_usd),
    )
    # Construction validates plan SHA, exact stage cap, and exact USD stop.
    BatchCoordinator(plan=plan, authorization=authorization)
    if job.output_path.exists():
        raise FileExistsError("job_output_exists_no_overwrite")
    public_result = Path(args.public_result_json).resolve()
    if public_result.exists():
        raise FileExistsError("public_result_output_exists_no_overwrite")
    plan_authority = next(
        item for item in plan.authorities if item.surface == job.surface
    )
    if (
        plan_authority.denominator != authority.denominator
        or plan_authority.replicates_per_task
        != authority.replicates_per_task
        or plan_authority.trial_policy_sha256
        != authority.trial_policy_sha256
        or plan_authority.test_registry_sha256
        != authority.test_registry_sha256
        or plan_authority.fit_registry_sha256
        != authority.fit_registry_sha256
        or plan_authority.dev_registry_sha256
        != authority.dev_registry_sha256
        or plan_authority.source_binding_sha256
        != registry.source_binding_sha256
        or plan_authority.official_scorer != authority.official_scorer
        or plan_authority.official_metric != authority.official_metric
        or plan_authority.scorer_sha256 != authority.scorer_sha256
        or plan_authority.runtime_name != authority.runtime_name
        or plan_authority.runtime_sha256 != authority.runtime_sha256
    ):
        raise ValueError("tau2_execution_surface_authority_mismatch")
    native_config = Tau2NativeRunConfig(
        api_base=args.api_base,
        private_output_root=(
            job.output_path.parent / f"{job.output_path.stem}.runs"
        ).resolve(),
        max_provider_calls=job.provider_attempt_hard_cap,
        max_observed_provider_cost_usd=float(authorization.observed_spend_stop_usd),
        max_steps=args.max_steps,
    )
    resources = Tau2PolicyResources(
        checkpoint_path=args.checkpoint_path,
        automix_tokenizer_root=args.automix_tokenizer_root,
        mtrouter_device=args.mtrouter_device,
        opensquilla_checkout_root=args.opensquilla_checkout_root,
        opensquilla_bundle_dir=args.opensquilla_bundle_dir,
    )
    runtime_resource_binding = _validate_resource_paths(
        typed_artifact,
        resources,
    )
    return PreparedTau2LiveRun(
        runtime_environment_sha256=runtime_environment_sha256,
        tau2_dependency_environment_sha256=(
            tau2_dependency_environment_sha256
        ),
        plan=plan,
        job=job,
        registry=registry,
        authority=authority,
        typed_artifact=typed_artifact,
        resources=resources,
        authorization=authorization,
        native_config=native_config,
        runtime_source=runtime_source,
        runtime_resource_binding=runtime_resource_binding,
        public_result_json=public_result,
    )


def run_live(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str] = os.environ,
    prepare: Callable[[argparse.Namespace], PreparedTau2LiveRun] = (prepare_live_run),
    provider_client_factory: Callable[..., Any] | None = None,
    native_api_loader: Callable[[Path], Tau2NativeAPI] = (load_tau2_native_api),
    cell_executor: Callable[..., Tau2CellExecution] = (execute_tau2_official_cell),
) -> dict[str, Any]:
    prepared = prepare(args)
    base = {
        "schema": SCHEMA,
        "scope": "tau2_external_router_official_test_cell",
        "job_id": prepared.job.job_id,
        "method": prepared.job.method,
        "surface": prepared.job.surface,
        "plan_sha256": prepared.plan.sha256,
        "runtime_environment_sha256": (
            prepared.runtime_environment_sha256
        ),
        "tau2_dependency_environment_sha256": (
            prepared.tau2_dependency_environment_sha256
        ),
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "execution_requested": bool(args.execute),
        "runtime_source": dict(prepared.runtime_source),
        "runtime_resource_binding": dict(
            prepared.runtime_resource_binding
        ),
        "test_registry_sha256": (prepared.registry.test_registry_sha256),
        "source_binding_sha256": (prepared.registry.source_binding_sha256),
        "method_artifact_sha256": (prepared.typed_artifact.binding.artifact_sha256),
        "full_denominator_required": prepared.job.item_count,
        "partial_resume_or_cross_run_merge_allowed": False,
    }
    if not args.execute:
        return {
            **base,
            "decision": "ready_for_explicit_execution",
            "blockers": ["execute_flag_missing"],
            "authorizes_paid_run": False,
        }
    if args.api_key_env != "OPENAI_API_KEY":
        raise ValueError("api_key_env_must_be_OPENAI_API_KEY")
    _validate_proxy_environment(env)
    try:
        api_key = env[args.api_key_env]
    except KeyError as exc:
        raise ValueError("api_key_environment_variable_missing") from exc
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key_environment_variable_empty")
    if provider_client_factory is None:
        from openai import OpenAI

        provider_client_factory = OpenAI
    provider_client = provider_client_factory(
        api_key=api_key,
        base_url=args.api_base.rstrip("/"),
        max_retries=0,
    )
    native_api = native_api_loader(args.tau2_runtime_repo)
    coordinator = BatchCoordinator(
        plan=prepared.plan,
        authorization=prepared.authorization,
    )
    completed: list[Tau2CellExecution] = []

    def execute(
        job: ExecutionJob,
        broker: Any,
    ) -> JobExecutionResult:
        cell = cell_executor(
            job=job,
            registry=prepared.registry,
            authority=prepared.authority,
            typed_artifact=prepared.typed_artifact,
            resources=prepared.resources,
            native_api=native_api,
            native_config=prepared.native_config,
            broker=broker,
            provider_client=provider_client,
            plan_sha256=prepared.plan.sha256,
            local_hardware=args.local_hardware_label,
            local_software=args.local_software_label,
        )
        completed.append(cell)
        return cell.result

    result = coordinator.execute_job(prepared.job.job_id, execute)
    if len(completed) != 1 or completed[0].result != result:
        raise RuntimeError("tau2_live_cell_result_not_retained")
    write_public_receipt_append_only(
        prepared.public_result_json,
        completed[0].public_receipt,
    )
    ledger = coordinator.public_report()["ledger"]["jobs"][prepared.job.job_id]
    return {
        **base,
        "decision": "completed",
        "provider_calls": ledger["receipts_settled"],
        "secret_reads": 1,
        "api_spend_performed": True,
        "blockers": [],
        "private_output": str(prepared.job.output_path),
        "public_result": str(prepared.public_result_json),
        "public_receipt_sha256": (completed[0].public_receipt.receipt_sha256),
        "ledger": coordinator.public_report()["ledger"],
        "authorizes_paid_run": True,
    }


def _validate_resource_paths(
    typed_artifact: Tau2TypedArtifact,
    resources: Tau2PolicyResources,
) -> dict[str, Any]:
    method = typed_artifact.method
    if method == MTROUTER:
        artifact = typed_artifact.artifact
        if artifact.uses_all_strong_shortcut:
            if resources.checkpoint_path is not None:
                raise ValueError("mtrouter_all_strong_asset_forbidden")
            return {
                "kind": "mtrouter_all_strong_native_shortcut",
                "execution_mode": artifact.execution_mode,
                "checkpoint_sha256": None,
                "dev_selection_sha256": artifact.dev_selection.sha256,
                "model_names": list(artifact.model_names),
            }
    if method == ROUTELLM and isinstance(
        typed_artifact.artifact,
        RouteLLMBertArtifact,
    ):
        checkpoint = resources.checkpoint_path
        if checkpoint is None or not checkpoint.is_dir():
            raise FileNotFoundError("typed_checkpoint_directory_missing")
        return {
            "kind": "checkpoint_directory",
            "path_label": checkpoint.name,
        }
    if method in {ROUTELLM, MTROUTER} and (
        resources.checkpoint_path is None or not resources.checkpoint_path.is_file()
    ):
        raise FileNotFoundError("typed_checkpoint_file_missing")
    if method in {ROUTELLM, MTROUTER}:
        checkpoint = resources.checkpoint_path
        assert checkpoint is not None
        return {
            "kind": "checkpoint_file",
            "path_label": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    if method == FRUGALGPT and (
        resources.checkpoint_path is None or not resources.checkpoint_path.is_dir()
    ):
        raise FileNotFoundError("typed_checkpoint_directory_missing")
    if method == FRUGALGPT:
        checkpoint = resources.checkpoint_path
        assert checkpoint is not None
        return {
            "kind": "checkpoint_directory",
            "path_label": checkpoint.name,
        }
    if method == AUTOMIX and (
        resources.automix_tokenizer_root is None
        or not resources.automix_tokenizer_root.is_dir()
    ):
        raise FileNotFoundError("automix_tokenizer_root_missing")
    if method == AUTOMIX:
        tokenizer = resources.automix_tokenizer_root
        assert tokenizer is not None
        return {
            "kind": "automix_tokenizer_directory",
            "path_label": tokenizer.name,
        }
    if method == OPENSQUILLA_G0 and (
        resources.opensquilla_checkout_root is None
        or not resources.opensquilla_checkout_root.is_dir()
    ):
        raise FileNotFoundError("opensquilla_checkout_root_missing")
    if method == OPENSQUILLA_G0 and (
        resources.opensquilla_bundle_dir is None
        or not resources.opensquilla_bundle_dir.is_dir()
    ):
        raise FileNotFoundError("opensquilla_adapted_bundle_dir_missing")
    if method == OPENSQUILLA_G0:
        checkout = resources.opensquilla_checkout_root
        bundle = resources.opensquilla_bundle_dir
        assert checkout is not None and bundle is not None
        checkout = checkout.resolve()
        bundle = bundle.resolve()
        if checkout == bundle or checkout in bundle.parents or bundle in checkout.parents:
            raise ValueError("opensquilla_execution_resources_not_independent")
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != OPENSQUILLA_UPSTREAM_REVISION:
            raise ValueError("opensquilla_checkout_revision_mismatch")
        strategy_path = (
            checkout
            / "src/opensquilla/squilla_router/v4_phase3.py"
        )
        strategy_sha256 = hashlib.sha256(
            strategy_path.read_bytes()
        ).hexdigest()
        if strategy_sha256 != OPENSQUILLA_TRAINING_SOURCE_SHA256S[
            "src/opensquilla/squilla_router/v4_phase3.py"
        ]:
            raise ValueError("opensquilla_checkout_source_mismatch")
        bundle_sha256 = opensquilla_bundle_tree_sha256(bundle)
        if bundle_sha256 != typed_artifact.artifact.bundle_sha256:
            raise ValueError("opensquilla_fit_adapted_bundle_mismatch")
        return {
            "kind": "opensquilla_fit_adapted_runtime",
            "checkout_path_label": checkout.name,
            "checkout_revision": revision,
            "checkout_strategy_sha256": strategy_sha256,
            "fit_adapted_bundle_path_label": bundle.name,
            "fit_adapted_bundle_tree_sha256": bundle_sha256,
            "artifact_bundle_sha256": (
                typed_artifact.artifact.bundle_sha256
            ),
            "checkout_and_fit_adapted_bundle_are_distinct_paths": True,
            "cold_start_checkout_bundle_used": False,
        }
    raise ValueError("typed_runtime_resource_method_invalid")


def _validate_proxy_environment(env: Mapping[str, str]) -> None:
    no_proxy = {
        item.strip() for item in env.get("NO_PROXY", "").split(",") if item.strip()
    }
    if not {"localhost", "127.0.0.1"}.issubset(no_proxy):
        raise ValueError("no_proxy_localhost_binding_missing")
    if any(
        env.get(name)
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
        )
    ):
        raise ValueError("http_proxy_must_be_unset")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = prepare_live_run(args)
    except Exception as exc:  # noqa: BLE001 - paid path must fail closed.
        report = {
            "schema": SCHEMA,
            "decision": "blocked",
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "execution_started": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        _write_optional_report(args.report_json, report)
        print(json.dumps(report, sort_keys=True))
        return 2
    try:
        report = run_live(
            args,
            prepare=lambda parsed: prepared,
        )
    except Exception as exc:  # noqa: BLE001 - post-authority state is unknown.
        report = {
            "schema": SCHEMA,
            "decision": "blocked",
            "provider_calls": None,
            "secret_reads": None,
            "api_spend_performed": None,
            "execution_started": bool(args.execute),
            "requires_private_ledger_inspection": bool(args.execute),
            "error": f"{type(exc).__name__}:{exc}",
        }
        _write_optional_report(args.report_json, report)
        print(json.dumps(report, sort_keys=True))
        return 2
    _write_optional_report(args.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def _write_optional_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

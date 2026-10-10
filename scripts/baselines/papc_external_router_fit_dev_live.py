#!/usr/bin/env python3
"""Dispatch authorized early Table 1 router stages over frozen fit/dev rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.external_router_execution import (  # noqa: E402
    AGENT_DIFF,
    AUTOMIX_STAGE,
    SHARED_STAGE,
    BatchCoordinator,
    ExecutionJob,
    ExecutionPlan,
    StageExecutionAuthorization,
    load_execution_plan,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FIT_DEV_DISPATCH_SCHEMA,
    PRODUCTION_WIRED_KINDS,
    BoundAutoMixTokenizer,
    PreparedFitDevJob,
    StageItemExecutor,
    execute_prepared_fit_dev_job,
    inspect_fit_dev_item_checkpoint,
    load_bound_automix_tokenizer,
    load_fit_dev_registry,
    openai_compatible_executor_factory,
    prepare_fit_dev_job,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    build_native_shared_endpoint_executor,
    load_native_fit_dev_authority,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    validate_router_runtime_environment,
)
from papc.paths import DATA_ROOT

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
DEFAULT_TAU2_RUNTIME_REPO = Path(
    f"{DATA_ROOT}/benchmarks/tau_bench/repo"
)
DEFAULT_AGENT_DIFF_REPO = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo"
)
DEFAULT_AGENT_DIFF_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/"
    "datasets/agent-diff-bench"
)
DEFAULT_AGENT_DIFF_SDK = (
    DEFAULT_AGENT_DIFF_REPO / "sdk" / "agent-diff-python"
)
DEFAULT_AGENT_DIFF_SCORER = (
    REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"
)
DEFAULT_AGENT_DIFF_BASELINE_AUTHORITY = (
    REPO_ROOT
    / "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py"
)
DEFAULT_AUTOMIX_TOKENIZER_ROOT = Path(
    f"{DATA_ROOT}/papc/upstream_models/llama2-tokenizer-d02ad6cb"
)
DEFAULT_CANONICAL_PUBLICATION_PRIVATE_ROOT = (
    Path(f"{DATA_ROOT}/papc")
    / "external_router_live_publication_subset16_v1"
)
SUPPORTED_STAGES = (SHARED_STAGE, AUTOMIX_STAGE)

ExecutorFactory = Callable[[str, str, int], StageItemExecutor]


def _positive_decimal(value: str) -> Decimal:
    try:
        resolved = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("invalid decimal") from exc
    if not resolved.is_finite() or resolved <= 0:
        raise argparse.ArgumentTypeError("decimal must be finite and positive")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--canonical-publication-private-root",
        type=Path,
        default=DEFAULT_CANONICAL_PUBLICATION_PRIVATE_ROOT,
    )
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, required=True)
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Repeat to dispatch a subset; default is the complete stage.",
    )
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument(
        "--stage-provider-attempt-hard-cap",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--stage-observed-spend-stop-usd",
        type=_positive_decimal,
        required=True,
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        choices=range(1, 6),
        default=1,
    )
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_TAU2_REPO)
    parser.add_argument(
        "--tau2-runtime-repo",
        type=Path,
        default=DEFAULT_TAU2_RUNTIME_REPO,
    )
    parser.add_argument(
        "--agent-diff-repo",
        type=Path,
        default=DEFAULT_AGENT_DIFF_REPO,
    )
    parser.add_argument(
        "--agent-diff-dataset",
        type=Path,
        default=DEFAULT_AGENT_DIFF_DATASET,
    )
    parser.add_argument(
        "--agent-diff-sdk-path",
        type=Path,
        default=DEFAULT_AGENT_DIFF_SDK,
    )
    parser.add_argument(
        "--agent-diff-scorer-path",
        type=Path,
        default=DEFAULT_AGENT_DIFF_SCORER,
    )
    parser.add_argument(
        "--agent-diff-baseline-authority-path",
        type=Path,
        default=DEFAULT_AGENT_DIFF_BASELINE_AUTHORITY,
    )
    parser.add_argument(
        "--agent-diff-base-url",
        default="http://localhost:8020",
    )
    parser.add_argument(
        "--automix-tokenizer-root",
        type=Path,
        default=DEFAULT_AUTOMIX_TOKENIZER_ROOT,
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] = os.environ,
    executor_factory: ExecutorFactory | None = None,
    provider_client_factory: Callable[..., Any] | None = None,
    native_authority_loader: Callable[..., Any] = (
        load_native_fit_dev_authority
    ),
    native_executor_builder: Callable[..., StageItemExecutor] = (
        build_native_shared_endpoint_executor
    ),
    tokenizer_loader: Callable[[Path], BoundAutoMixTokenizer] = (
        load_bound_automix_tokenizer
    ),
    runtime_environment_validator: Callable[[], str] = (
        validate_router_runtime_environment
    ),
) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    try:
        runtime_environment_sha256 = runtime_environment_validator()
        plan = load_execution_plan(
            split_gate_json=args.split_gate_json,
            budget_gate_json=args.budget_gate_json,
            authority_manifest_json=args.authority_manifest_json,
            private_root=args.private_root,
        )
        canonical_publication_plan = load_execution_plan(
            split_gate_json=args.split_gate_json,
            budget_gate_json=args.budget_gate_json,
            authority_manifest_json=args.authority_manifest_json,
            private_root=args.canonical_publication_private_root,
        )
        canonical_plan_binding = _canonical_plan_binding(
            plan,
            canonical_publication_plan,
        )
        authorization = StageExecutionAuthorization(
            authorization_id=args.authorization_id,
            plan_sha256=args.plan_sha256,
            stage=args.stage,
            provider_attempt_hard_cap=(
                args.stage_provider_attempt_hard_cap
            ),
            observed_spend_stop_usd=(
                args.stage_observed_spend_stop_usd
            ),
        )
        BatchCoordinator(plan=plan, authorization=authorization)
        jobs = _select_jobs(plan.jobs, args.stage, args.job_id)
        if (
            args.max_concurrency != 1
            and any(
                job.kind == "shared_endpoint_episode"
                for job in jobs
            )
        ):
            raise ValueError(
                "fit_dev_native_episode_concurrency_must_be_one"
            )
        shared_surfaces = tuple(
            sorted(
                {
                    job.surface
                    for job in jobs
                    if job.kind == "shared_endpoint_episode"
                }
            )
        )
        native_authority = (
            native_authority_loader(
                surfaces=shared_surfaces,
                private_root=args.private_root,
                state_repo=args.state_repo,
                tau2_data_repo=args.tau2_repo,
                tau2_runtime_repo=args.tau2_runtime_repo,
                agent_diff_repo=args.agent_diff_repo,
                agent_diff_dataset=args.agent_diff_dataset,
                agent_diff_sdk_path=args.agent_diff_sdk_path,
                agent_diff_scorer_path=args.agent_diff_scorer_path,
                agent_diff_baseline_authority_path=(
                    args.agent_diff_baseline_authority_path
                ),
                agent_diff_base_url=args.agent_diff_base_url,
            )
            if shared_surfaces
            else None
        )
        native_authority_record = (
            native_authority.public_record()
            if native_authority is not None
            else None
        )
        if native_authority_record is not None and not isinstance(
            native_authority_record,
            Mapping,
        ):
            raise TypeError("fit_dev_native_authority_record_invalid")
        automix_tokenizer = (
            tokenizer_loader(args.automix_tokenizer_root)
            if any(
                job.kind == "automix_n32_verifier" for job in jobs
            )
            else None
        )
        preflight = _provider_zero_preflight(
            jobs=jobs,
            plan=plan,
            split_gate_json=args.split_gate_json,
            state_repo=args.state_repo,
            tau2_repo=args.tau2_repo,
            agent_diff_dataset=args.agent_diff_dataset,
        )
        deferred_job_ids = {
            dependency["job_id"]
            for dependency in preflight["dependencies"]
            if dependency["generated_by_selected_stage"]
        }
        prepared_before_secret = {
            job.job_id: prepare_fit_dev_job(
                plan=plan,
                job_id=job.job_id,
                authorization=authorization,
                split_gate_json=args.split_gate_json,
                state_repo=args.state_repo,
                tau2_repo=args.tau2_repo,
                agent_diff_dataset=args.agent_diff_dataset,
                automix_tokenizer=automix_tokenizer,
            )
            for job in jobs
            if job.job_id not in deferred_job_ids
        }
        checkpoint_preflight = {}
        for job_id, prepared in prepared_before_secret.items():
            checkpoint = inspect_fit_dev_item_checkpoint(prepared)
            _assert_no_uncheckpointed_agent_diff_native_output(
                prepared,
                resumed_item_count=checkpoint["resumed_item_count"],
            )
            checkpoint_preflight[job_id] = checkpoint
    except Exception as exc:  # noqa: BLE001 - fail before any secret access.
        return 2, {
            "schema": FIT_DEV_DISPATCH_SCHEMA,
            "decision": "blocked_before_secret_read",
            "execution_requested": bool(args.execute),
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "blockers": [str(exc) or type(exc).__name__],
            "error_type": type(exc).__name__,
            "authorizes_paid_run": False,
        }
    base = {
        "schema": FIT_DEV_DISPATCH_SCHEMA,
        "scope": "external_router_early_paid_stage",
        "stage": args.stage,
        "plan_sha256": plan.sha256,
        "plan_run_specific_sha256": plan.run_specific_sha256,
        "plan_protocol_semantic_sha256": (
            plan.protocol_semantic_sha256
        ),
        "plan_hash_semantics": {
            "authorization_binding": "run_specific_sha256",
            "artifact_and_publication_protocol_binding": (
                "protocol_semantic_sha256"
            ),
            "run_specific_field": "private_root_label",
        },
        "canonical_execution_plan_binding": canonical_plan_binding,
        "runtime_environment_sha256": runtime_environment_sha256,
        "execution_requested": bool(args.execute),
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "job_count": len(jobs),
        "job_ids": [job.job_id for job in jobs],
        "preflight": preflight,
        "native_runtime_authority": (
            dict(native_authority_record)
            if native_authority_record is not None
            else None
        ),
        "jobs_prepared_before_secret": len(prepared_before_secret),
        "item_checkpoint_preflight": checkpoint_preflight,
        "jobs_deferred_until_bound_upstream": len(deferred_job_ids),
        "official_test_outcomes_read": False,
        "max_retries": 0,
        "max_concurrency": args.max_concurrency,
        "automix_request_artifacts_built_before_secret": (
            sum(
                job.kind == "automix_n32_verifier"
                and job.job_id in prepared_before_secret
                for job in jobs
            )
        ),
        "partial_resume_or_cross_run_merge_allowed": False,
        "same_job_bound_item_checkpoint_continuation_allowed": True,
        "cross_manifest_or_protocol_merge_allowed": False,
    }
    if not args.execute:
        return 0, {
            **base,
            "decision": "ready_for_explicit_execute",
            "blockers": ["execute_flag_missing"],
            "authorizes_paid_run": False,
        }
    blockers = _proxy_blockers(env)
    if blockers:
        return 2, {
            **base,
            "decision": "blocked_before_secret_read",
            "blockers": blockers,
            "authorizes_paid_run": False,
        }
    if args.api_key_env != "OPENAI_API_KEY":
        return 2, {
            **base,
            "decision": "blocked_before_secret_read",
            "blockers": ["api_key_env_must_be_OPENAI_API_KEY"],
            "authorizes_paid_run": False,
        }
    try:
        api_key = env[args.api_key_env]
    except KeyError:
        return 2, {
            **base,
            "decision": "blocked_secret_unavailable",
            "secret_reads": 1,
            "blockers": ["api_key_environment_variable_missing"],
            "authorizes_paid_run": False,
        }
    if not isinstance(api_key, str) or not api_key.strip():
        return 2, {
            **base,
            "decision": "blocked_secret_unavailable",
            "secret_reads": 1,
            "blockers": ["api_key_environment_variable_empty"],
            "authorizes_paid_run": False,
        }
    try:
        if executor_factory is not None:
            executor = executor_factory(
                args.api_base.rstrip("/"),
                api_key,
                0,
            )
        else:
            provider_executor = (
                openai_compatible_executor_factory(
                    args.api_base.rstrip("/"),
                    api_key,
                    0,
                    client_factory=provider_client_factory,
                )
                if any(
                    job.kind
                    in {"route_embedding", "automix_n32_verifier"}
                    for job in jobs
                )
                else None
            )
            native_executor = (
                native_executor_builder(
                    authority=native_authority,
                    api_base=args.api_base.rstrip("/"),
                    api_key=api_key,
                    max_retries=0,
                    max_observed_provider_cost_usd=float(
                        authorization.observed_spend_stop_usd
                    ),
                    api_key_env=args.api_key_env,
                    client_factory=provider_client_factory,
                )
                if native_authority is not None
                else None
            )
            executor = _combined_executor(
                provider_executor=provider_executor,
                native_executor=native_executor,
            )
    except Exception as exc:  # noqa: BLE001 - no provider call has occurred.
        return 2, {
            **base,
            "decision": "blocked_provider_client_construction",
            "provider_calls": 0,
            "secret_reads": 1,
            "api_spend_performed": False,
            "blockers": [str(exc) or type(exc).__name__],
            "authorizes_paid_run": False,
        }
    coordinator = BatchCoordinator(plan=plan, authorization=authorization)
    completed: list[dict[str, Any]] = []
    try:
        for job in jobs:
            prepared = prepared_before_secret.get(job.job_id)
            if prepared is None:
                prepared = prepare_fit_dev_job(
                    plan=plan,
                    job_id=job.job_id,
                    authorization=authorization,
                    split_gate_json=args.split_gate_json,
                    state_repo=args.state_repo,
                    tau2_repo=args.tau2_repo,
                    agent_diff_dataset=args.agent_diff_dataset,
                    automix_tokenizer=automix_tokenizer,
                )
            _, report = execute_prepared_fit_dev_job(
                prepared,
                executor,
                coordinator=coordinator,
                max_concurrency=args.max_concurrency,
            )
            completed.append(report)
    except Exception as exc:  # noqa: BLE001 - no partial job publication.
        ledger = coordinator.public_report()["ledger"]
        return 2, {
            **base,
            "decision": "authorized_execution_failed",
            "provider_calls": ledger["stages"][args.stage][
                "attempts_reserved"
            ],
            "secret_reads": 1,
            "api_spend_performed": (
                ledger["stages"][args.stage]["attempts_reserved"] > 0
            ),
            "completed_jobs": completed,
            "blockers": [str(exc) or type(exc).__name__],
            "coordinator": coordinator.public_report(),
            "authorizes_paid_run": True,
        }
    ledger = coordinator.public_report()["ledger"]
    return 0, {
        **base,
        "decision": "complete",
        "provider_calls": ledger["stages"][args.stage]["attempts_reserved"],
        "secret_reads": 1,
        "api_spend_performed": True,
        "completed_jobs": completed,
        "blockers": [],
        "coordinator": coordinator.public_report(),
        "authorizes_paid_run": True,
    }


def _combined_executor(
    *,
    provider_executor: StageItemExecutor | None,
    native_executor: StageItemExecutor | None,
) -> StageItemExecutor:
    if provider_executor is None and native_executor is None:
        raise ValueError("fit_dev_production_executor_missing")

    def execute(job, item, broker):
        if job.kind == "shared_endpoint_episode":
            if native_executor is None:
                raise ValueError("fit_dev_native_executor_missing")
            return native_executor(job, item, broker)
        if provider_executor is None:
            raise ValueError("fit_dev_provider_executor_missing")
        return provider_executor(job, item, broker)

    return execute


def _canonical_plan_binding(
    fit_dev_plan: ExecutionPlan,
    publication_plan: ExecutionPlan,
) -> dict[str, Any]:
    fit_dev_record = fit_dev_plan.public_record()
    publication_record = publication_plan.public_record()
    differing_fields = sorted(
        key
        for key in set(fit_dev_record) | set(publication_record)
        if fit_dev_record.get(key) != publication_record.get(key)
    )
    if differing_fields not in ([], ["private_root_label"]):
        raise ValueError(
            "fit_dev_publication_plan_semantics_diverge:"
            + ",".join(differing_fields)
        )
    if (
        fit_dev_plan.protocol_semantic_sha256
        != publication_plan.protocol_semantic_sha256
    ):
        raise ValueError(
            "fit_dev_publication_protocol_semantic_sha256_mismatch"
        )
    return {
        "fit_dev_run_specific_sha256": (
            fit_dev_plan.run_specific_sha256
        ),
        "publication_run_specific_sha256": (
            publication_plan.run_specific_sha256
        ),
        "protocol_semantic_sha256": (
            fit_dev_plan.protocol_semantic_sha256
        ),
        "run_specific_hashes_differ": (
            fit_dev_plan.run_specific_sha256
            != publication_plan.run_specific_sha256
        ),
        "differing_public_record_fields": differing_fields,
        "protocol_semantics_match": True,
    }


def _select_jobs(
    all_jobs: tuple[ExecutionJob, ...],
    stage: str,
    requested_ids: list[str],
) -> tuple[ExecutionJob, ...]:
    stage_jobs = tuple(job for job in all_jobs if job.stage == stage)
    if not stage_jobs:
        raise ValueError("fit_dev_stage_has_no_jobs")
    selected = stage_jobs
    if requested_ids:
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("fit_dev_requested_job_duplicate")
        by_id = {job.job_id: job for job in stage_jobs}
        if any(job_id not in by_id for job_id in requested_ids):
            raise ValueError("fit_dev_requested_job_outside_stage")
        requested = set(requested_ids)
        selected = tuple(
            job for job in stage_jobs if job.job_id in requested
        )
    unwired = tuple(
        sorted({job.kind for job in selected} - PRODUCTION_WIRED_KINDS)
    )
    if unwired:
        raise ValueError(
            "fit_dev_production_kind_not_wired:" + ",".join(unwired)
        )
    return selected


def _provider_zero_preflight(
    *,
    jobs: tuple[ExecutionJob, ...],
    plan: ExecutionPlan,
    split_gate_json: Path,
    state_repo: Path,
    tau2_repo: Path,
    agent_diff_dataset: Path,
) -> dict[str, Any]:
    selected_positions = {
        job.job_id: position for position, job in enumerate(jobs)
    }
    dependencies: list[dict[str, Any]] = []
    for position, job in enumerate(jobs):
        if job.output_path.exists():
            raise FileExistsError(f"job_output_exists_no_overwrite:{job.job_id}")
        registry = load_fit_dev_registry(
            job=job,
            split_gate_json=split_gate_json,
            state_repo=state_repo,
            tau2_repo=tau2_repo,
            agent_diff_dataset=agent_diff_dataset,
        )
        if len(registry.rows) != job.item_count:
            raise ValueError("fit_dev_preflight_denominator_mismatch")
        if _needs_shared_cheap(job):
            dependency_id = ".".join(
                (
                    SHARED_STAGE,
                    "shared_endpoint_episode",
                    "shared_cheap_endpoint",
                    job.surface,
                    job.split_role,
                )
            )
            dependency = plan.job(dependency_id)
            available = dependency.output_path.is_file()
            generated_earlier = (
                dependency_id in selected_positions
                and selected_positions[dependency_id] < position
            )
            if not available and not generated_earlier:
                raise FileNotFoundError(
                    f"fit_dev_upstream_output_missing:{dependency_id}"
                )
            dependencies.append(
                {
                    "job_id": job.job_id,
                    "upstream_job_id": dependency_id,
                    "available_now": available,
                    "generated_by_selected_stage": generated_earlier,
                }
            )
    return {
        "source_registries_reconstructed": True,
        "full_membership_bound": True,
        "test_outcomes_read": False,
        "fresh_append_only_outputs": True,
        "dependencies": dependencies,
        "provider_calls": 0,
        "secret_reads": 0,
    }


def _needs_shared_cheap(job: ExecutionJob) -> bool:
    return (
        job.kind in {"route_embedding", "automix_n32_verifier"}
        or (
            job.kind == "shared_endpoint_episode"
            and dict(job.details).get("model_role") == "strong"
        )
    )


def _assert_no_uncheckpointed_agent_diff_native_output(
    prepared: PreparedFitDevJob,
    *,
    resumed_item_count: int,
) -> None:
    job = prepared.job
    if job.kind != "shared_endpoint_episode" or job.surface != AGENT_DIFF:
        return
    role = dict(job.details).get("model_role")
    if role not in {"cheap", "strong"}:
        raise ValueError("agent_diff_native_output_role_invalid")
    native_root = job.output_path.parent / f".{job.output_path.stem}.native"
    for item in prepared.items[resumed_item_count:]:
        if item.source is None:
            raise ValueError("agent_diff_native_output_source_missing")
        raw_path = (
            native_root
            / item.source.example_sha256
            / f"{role}.json"
        ).resolve()
        if os.path.lexists(raw_path):
            raise FileExistsError(
                "uncheckpointed_agent_diff_native_output_exists:"
                f"{raw_path}"
            )


def _proxy_blockers(env: Mapping[str, str]) -> list[str]:
    no_proxy = {
        item.strip()
        for item in env.get("NO_PROXY", "").split(",")
        if item.strip()
    }
    blockers = []
    if not {"localhost", "127.0.0.1"}.issubset(no_proxy):
        blockers.append("no_proxy_localhost_binding_missing")
    if any(
        env.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    ):
        blockers.append("http_proxy_must_be_unset")
    return blockers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(argv)
    if args.report_json is not None:
        _write_report_append_only(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return code


def _write_report_append_only(path: Path, report: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".complete",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

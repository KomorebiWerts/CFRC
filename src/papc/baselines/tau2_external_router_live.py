"""Authorized, source-bound Tau2 Table 1 cell execution.

This module is provider-neutral until ``execute_tau2_official_cell`` receives
an already constructed provider client and a ``ProviderAttemptBroker``.  It
keeps the official Tau2 runtime, typed router artifacts, full-denominator
result sink, and normalized public Table 1 receipt on one fail-closed path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from papc.baselines.automix_upstream import (
    AGENT_CONTEXT_SERIALIZER_PROTOCOL,
    AutoMixDevCandidateReceipt,
    AutoMixPOMDPPolicy,
    AutoMixRuntimeArtifact,
    AutoMixVerifierRequest,
    validate_automix_verifier_tokenizer_root,
)
from papc.baselines.external_router_execution import (
    ExecutionContractError,
    ExecutionJob,
    JobExecutionResult,
    ProviderAttemptBroker,
)
from papc.baselines.frugalgpt_upstream import FrugalGPTScorerArtifact
from papc.baselines.mtrouter_upstream import (
    MTRouterArtifact,
    load_mtrouter_artifact,
)
from papc.baselines.opensquilla_upstream import load_opensquilla_runtime_artifact
from papc.baselines.routellm_bert_upstream import (
    ROUTELLM_BERT_ARTIFACT_SCHEMA,
    RouteLLMBertArtifact,
)
from papc.baselines.routellm_mf_upstream import (
    RouteLLMMFArtifact,
    RouteLLMMFDevCalibration,
    RouteLLMMFDevCandidateReceipt,
    RouteLLMMFEmbeddingRequest,
    RouteLLMMFEmbeddingResult,
)
from papc.baselines.router_accounting import (
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (
    AUTOMIX,
    CHEAP_MODEL,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    REASONING_EFFORT,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    STRONG_MODEL,
    RouterArtifactBinding,
)
from papc.baselines.router_runtime import StaticRoutingResult
from papc.baselines.sealed_router_policy import (
    AutoMixVerifierTransportResult,
    SealedRouterPolicy,
    load_automix_sealed_policy,
    load_frugalgpt_sealed_policy,
    load_mtrouter_sealed_policy,
    load_opensquilla_sealed_policy,
    load_routellm_bert_sealed_policy,
    load_routellm_sealed_policy,
)
from papc.baselines.table1_paired_stats import (
    CaseOutcomeCost,
    RegistryOrderedOutcomeCostManifest,
)
from papc.baselines.table1_results import (
    FULL_AGGREGATION_SCOPE,
    LocalRouterReport,
    ProviderAPIAccounting,
    SurfaceAuthority,
    Table1ResultReceipt,
)
from papc.baselines.tau2_native_runtime import (
    Tau2NativeAPI,
    Tau2NativeEpisode,
    Tau2NativeExecutor,
    Tau2NativeRunConfig,
)
from papc.baselines.tau2_router_driver import (
    TAU2_TASK_LEVEL_METHODS,
    TAU2_TURN_LEVEL_METHODS,
    Tau2OfficialTestRegistry,
    Tau2RouterDriver,
)

TAU2_LIVE_CELL_SCHEMA = "papc.tau2.external_router_live_cell.v1"
TAU2_METHODS = frozenset({ROUTELLM, FRUGALGPT, AUTOMIX, MTROUTER, OPENSQUILLA_G0})


@dataclass(frozen=True)
class Tau2TypedArtifact:
    method: str
    artifact: Any
    binding: RouterArtifactBinding

    def __post_init__(self) -> None:
        if self.method not in TAU2_METHODS:
            raise ValueError("tau2_typed_artifact_method_invalid")
        if not isinstance(self.binding, RouterArtifactBinding):
            raise TypeError("tau2_typed_artifact_binding_invalid")
        if self.binding.method != self.method:
            raise ValueError("tau2_typed_artifact_method_mismatch")


@dataclass(frozen=True)
class Tau2PolicyResources:
    checkpoint_path: Path | None = None
    automix_tokenizer_root: Path | None = None
    mtrouter_device: str = "cpu"
    opensquilla_checkout_root: Path | None = None
    opensquilla_bundle_dir: Path | None = None


@dataclass(frozen=True)
class Tau2CellExecution:
    result: JobExecutionResult
    public_receipt: Table1ResultReceipt
    private_payload: dict[str, Any]


def load_tau2_typed_artifact(
    *,
    method: str,
    path: Path,
) -> Tau2TypedArtifact:
    """Load one canonical method-specific artifact without provider access."""

    if method == ROUTELLM:
        artifact = _load_routellm_artifact(path)
        binding = artifact.to_router_binding()
    elif method == FRUGALGPT:
        artifact = FrugalGPTScorerArtifact.from_public_record(
            _read_mapping(path, "frugalgpt_artifact")
        )
        binding = artifact.router_binding()
    elif method == AUTOMIX:
        artifact = _load_automix_artifact(path)
        binding = artifact.router_binding()
    elif method == MTROUTER:
        artifact = load_mtrouter_artifact(path)
        binding = artifact.router_binding()
    elif method == OPENSQUILLA_G0:
        artifact = load_opensquilla_runtime_artifact(path)
        binding = artifact.router_binding()
    else:
        raise ValueError("tau2_typed_artifact_method_invalid")
    return Tau2TypedArtifact(
        method=method,
        artifact=artifact,
        binding=binding,
    )


def validate_tau2_live_bindings(
    *,
    job: ExecutionJob,
    registry: Tau2OfficialTestRegistry,
    authority: SurfaceAuthority,
    typed_artifact: Tau2TypedArtifact,
) -> None:
    if job.kind != "official_test_cell":
        raise ExecutionContractError("tau2_live_job_not_official_test_cell")
    if job.surface not in {"tau2_retail", "tau2_airline"}:
        raise ExecutionContractError("tau2_live_job_surface_not_tau2")
    if (
        job.method != typed_artifact.method
        or job.method != typed_artifact.binding.method
    ):
        raise ExecutionContractError("tau2_live_job_method_mismatch")
    if (
        registry.surface != job.surface
        or authority.surface != job.surface
        or typed_artifact.binding.surface != job.surface
    ):
        raise ExecutionContractError("tau2_live_surface_mismatch")
    if (
        registry.denominator != job.item_count
        or authority.denominator != job.item_count
    ):
        raise ExecutionContractError("tau2_live_denominator_mismatch")
    if (
        registry.test_registry_sha256 != job.registry_sha256
        or authority.test_registry_sha256 != job.registry_sha256
        or typed_artifact.binding.test_registry_sha256 != job.registry_sha256
    ):
        raise ExecutionContractError("tau2_live_test_registry_mismatch")
    if registry.source_binding_sha256 != job.source_binding_sha256:
        raise ExecutionContractError("tau2_live_source_binding_mismatch")
    if (
        authority.fit_registry_sha256 != typed_artifact.binding.fit_registry_sha256
        or authority.dev_registry_sha256 != typed_artifact.binding.dev_registry_sha256
    ):
        raise ExecutionContractError("tau2_live_fit_dev_registry_mismatch")
    if (
        authority.replicates_per_task != job.replicates_per_task
        or authority.trial_policy_sha256 != job.trial_policy_sha256
        or authority.test_registry_sha256
        != job.authority_test_registry_sha256
        or authority.official_scorer != job.official_scorer
        or authority.official_metric != job.official_metric
        or authority.scorer_sha256 != job.scorer_sha256
        or authority.runtime_name != job.runtime_name
        or authority.runtime_sha256 != job.runtime_sha256
    ):
        raise ExecutionContractError("tau2_execution_authority_mismatch")


def build_tau2_sealed_policy(
    *,
    typed_artifact: Tau2TypedArtifact,
    resources: Tau2PolicyResources,
    broker: ProviderAttemptBroker,
    provider_client: Any,
) -> SealedRouterPolicy:
    method = typed_artifact.method
    artifact = typed_artifact.artifact
    binding = typed_artifact.binding
    if method == ROUTELLM:
        if isinstance(artifact, RouteLLMBertArtifact):
            return load_routellm_bert_sealed_policy(
                binding=binding,
                artifact=artifact,
                checkpoint_dir=_required_dir(
                    resources.checkpoint_path,
                    "routellm_bert_checkpoint",
                ),
            )
        checkpoint = _required_file(resources.checkpoint_path, "routellm_checkpoint")
        return load_routellm_sealed_policy(
            binding=binding,
            artifact=artifact,
            checkpoint_payload=checkpoint.read_bytes(),
            embed_prompt=_routellm_embed_transport(
                provider_client=provider_client,
                broker=broker,
            ),
        )
    if method == FRUGALGPT:
        checkpoint_dir = _required_dir(
            resources.checkpoint_path,
            "frugalgpt_checkpoint",
        )
        return load_frugalgpt_sealed_policy(
            binding=binding,
            artifact=artifact,
            checkpoint_dir=checkpoint_dir,
        )
    if method == AUTOMIX:
        tokenizer = _load_automix_tokenizer(
            _required_dir(
                resources.automix_tokenizer_root,
                "automix_tokenizer",
            )
        )
        return load_automix_sealed_policy(
            binding=binding,
            artifact=artifact,
            tokenizer=tokenizer,
            transport=_automix_verifier_transport(
                provider_client=provider_client,
                broker=broker,
            ),
        )
    if method == MTROUTER:
        checkpoint = resources.checkpoint_path
        if (
            isinstance(artifact, MTRouterArtifact)
            and artifact.uses_all_strong_shortcut
        ):
            if checkpoint is not None:
                raise ValueError("tau2_mtrouter_all_strong_asset_forbidden")
        else:
            checkpoint = _required_file(
                checkpoint,
                "mtrouter_checkpoint",
            )
        return load_mtrouter_sealed_policy(
            binding=binding,
            artifact=artifact,
            checkpoint_path=checkpoint,
            device=resources.mtrouter_device,
        )
    if method == OPENSQUILLA_G0:
        checkout = _required_dir(
            resources.opensquilla_checkout_root,
            "opensquilla_checkout",
        )
        bundle = _required_dir(
            resources.opensquilla_bundle_dir,
            "opensquilla_adapted_bundle",
        )
        return load_opensquilla_sealed_policy(
            binding=binding,
            artifact=artifact,
            checkout_root=checkout,
            bundle_dir=bundle,
        )
    raise ValueError("tau2_typed_artifact_method_invalid")


def execute_tau2_official_cell(
    *,
    job: ExecutionJob,
    registry: Tau2OfficialTestRegistry,
    authority: SurfaceAuthority,
    typed_artifact: Tau2TypedArtifact,
    resources: Tau2PolicyResources,
    native_api: Tau2NativeAPI,
    native_config: Tau2NativeRunConfig,
    broker: ProviderAttemptBroker,
    provider_client: Any,
    plan_sha256: str,
    local_hardware: str | None = None,
    local_software: str | None = None,
) -> Tau2CellExecution:
    """Run and atomically publish one complete Retail or Airline test cell."""

    validate_tau2_live_bindings(
        job=job,
        registry=registry,
        authority=authority,
        typed_artifact=typed_artifact,
    )
    _require_sha256(plan_sha256, "tau2_live_plan_sha256")
    if job.output_path.exists():
        raise FileExistsError("tau2_live_job_output_exists")
    policy = build_tau2_sealed_policy(
        typed_artifact=typed_artifact,
        resources=resources,
        broker=broker,
        provider_client=provider_client,
    )
    driver = Tau2RouterDriver(policy=policy, registry=registry)
    outcomes: list[dict[str, Any]] = []
    raw_cases: list[dict[str, Any]] = []
    usage_cases: list[dict[str, Any]] = []
    outcome_cost_cases: list[CaseOutcomeCost] = []
    receipts: list[CallReceipt] = []
    scores: list[float] = []
    decision_count = 0
    for registry_index, task in enumerate(registry.tasks):
        executor = Tau2NativeExecutor(
            api=native_api,
            driver=driver,
            task=task,
            config=native_config,
            attempt_broker=broker,
        )
        outcome = _execute_tau2_task(executor)
        case_receipts = tuple(outcome["receipts"])
        private_simulations = _validated_private_simulations(
            outcome.get("private_simulations"),
            root=native_config.private_output_root,
        )
        case_accounting = aggregate_online_accounting(
            case_receipts,
            prices=FROZEN_CACHE_NEUTRAL_PRICES,
        )
        scores.append(outcome["official_score"])
        receipts.extend(case_receipts)
        decision_count += outcome["decision_count"]
        public_case = {
            "example_sha256": task.example_sha256,
            "official_score": outcome["official_score"],
            "selected_roles": outcome["selected_roles"],
            "decision_count": outcome["decision_count"],
            "receipt_count": len(case_receipts),
        }
        outcomes.append(public_case)
        raw_case = {
            "registry_index": registry_index,
            **public_case,
            "private_simulations": private_simulations,
            "receipts": [receipt.public_record() for receipt in case_receipts],
        }
        usage_case = {
            "registry_index": registry_index,
            "example_sha256": task.example_sha256,
            "receipts": [receipt.public_record() for receipt in case_receipts],
        }
        raw_cases.append(raw_case)
        usage_cases.append(usage_case)
        case_agent = case_accounting["agent_side"]
        outcome_cost_cases.append(
            CaseOutcomeCost(
                registry_index=registry_index,
                case_id_sha256=task.example_sha256,
                official_outcome=outcome["official_score"],
                provider_calls=case_agent["provider_calls"],
                retry_count=case_agent["retry_count"],
                input_tokens=case_agent["input_tokens"],
                cached_input_tokens=case_agent["cached_input_tokens"],
                output_tokens=case_agent["output_tokens"],
                reasoning_tokens=case_agent["reasoning_tokens"],
                cache_neutral_cost_usd=case_agent["cost_usd"],
                private_raw_case_receipt_sha256=_digest(raw_case),
                raw_usage_receipt_sha256=_digest(usage_case),
            )
        )
    if len(outcomes) != job.item_count or len(scores) != job.item_count:
        raise ExecutionContractError("tau2_live_cell_denominator_incomplete")
    if decision_count <= 0:
        raise ExecutionContractError("tau2_live_cell_decisions_missing")
    accounting = aggregate_online_accounting(
        receipts,
        prices=FROZEN_CACHE_NEUTRAL_PRICES,
    )
    agent = accounting["agent_side"]
    if agent["usage_complete"] is not True:
        raise ExecutionContractError("tau2_live_agent_usage_incomplete")
    expected_order = [task.example_sha256 for task in registry.tasks]
    observed_order = [item["example_sha256"] for item in raw_cases]
    if observed_order != expected_order:
        raise ExecutionContractError("tau2_live_registry_order_mismatch")
    implementation_bundle_sha256 = _implementation_bundle_sha256(job.method)
    run_config_sha256 = _digest(
        {
            "plan_sha256": plan_sha256,
            "job": job.public_record(),
            "native_config": native_config.public_record(),
            "native_runtime": native_api.public_record(),
            "method_artifact": typed_artifact.binding.public_record(),
            "resource_binding": _resource_binding(
                method=job.method,
                resources=resources,
                typed_artifact=typed_artifact,
            ),
        }
    )
    private_raw_case_manifest = {
        "schema": "papc.tau2.private_raw_case_manifest.v1",
        "row_id": job.method,
        "surface": job.surface,
        "test_registry_sha256": job.registry_sha256,
        "cases": raw_cases,
    }
    raw_usage_receipt_manifest = {
        "schema": "papc.tau2.raw_usage_receipt_manifest.v1",
        "row_id": job.method,
        "surface": job.surface,
        "test_registry_sha256": job.registry_sha256,
        "cases": usage_cases,
    }
    outcome_cost_manifest = RegistryOrderedOutcomeCostManifest(
        row_id=job.method,
        surface=job.surface,
        test_registry_sha256=authority.test_registry_sha256,
        trial_policy_sha256=authority.trial_policy_sha256,
        official_metric=authority.official_metric,
        cases=tuple(outcome_cost_cases),
        registry_order_matches_test_manifest=True,
    )
    private_raw_case_manifest_sha256 = _digest(private_raw_case_manifest)
    raw_usage_receipt_manifest_sha256 = _digest(raw_usage_receipt_manifest)
    registry_ordered_outcome_cost_manifest_sha256 = (
        outcome_cost_manifest.manifest_sha256
    )
    evidence_dir = (
        job.output_path.parent
        / f".{job.output_path.name}.{uuid4().hex}.private"
    )
    evidence_dir.mkdir(parents=True, exist_ok=False)
    _write_json_line_append_only(
        evidence_dir / "raw_case_manifest.json",
        private_raw_case_manifest,
    )
    _write_json_line_append_only(
        evidence_dir / "raw_usage_receipt_manifest.json",
        raw_usage_receipt_manifest,
    )
    _write_json_line_append_only(
        evidence_dir / "registry_ordered_outcome_cost_manifest.json",
        outcome_cost_manifest.private_record(),
    )
    private_payload = {
        "schema": TAU2_LIVE_CELL_SCHEMA,
        "plan_sha256": plan_sha256,
        "job": job.public_record(),
        "method_artifact": typed_artifact.binding.public_record(),
        "attempted_count": len(outcomes),
        "scored_count": len(scores),
        "infra_failure_count": 0,
        "official_score": sum(scores) / len(scores),
        "outcomes": outcomes,
        "private_raw_case_manifest": private_raw_case_manifest,
        "private_raw_case_manifest_sha256": (private_raw_case_manifest_sha256),
        "raw_usage_receipt_manifest": raw_usage_receipt_manifest,
        "raw_usage_receipt_manifest_sha256": (raw_usage_receipt_manifest_sha256),
        "registry_ordered_outcome_cost_manifest": (
            outcome_cost_manifest.private_record()
        ),
        "registry_ordered_outcome_cost_manifest_sha256": (
            registry_ordered_outcome_cost_manifest_sha256
        ),
        "private_evidence_directory": evidence_dir.name,
        "private_evidence_files": {
            "raw_case_manifest": "raw_case_manifest.json",
            "raw_usage_receipt_manifest": "raw_usage_receipt_manifest.json",
            "registry_ordered_outcome_cost_manifest": (
                "registry_ordered_outcome_cost_manifest.json"
            ),
        },
        "implementation_bundle_sha256": (implementation_bundle_sha256),
        "run_config_sha256": run_config_sha256,
        "registry_order_matches_test_manifest": True,
        "accounting": accounting,
        "merged_from_prior_runs": False,
        "baseline_results_retained_without_filtering": True,
    }
    payload_sha256 = _digest(private_payload)
    _write_json_line_append_only(job.output_path, private_payload)
    public_receipt = Table1ResultReceipt(
        row_id=job.method,
        surface=job.surface,
        result_status="completed",
        aggregation_scope=FULL_AGGREGATION_SCOPE,
        attempted_count=len(outcomes),
        scored_count=len(scores),
        infra_error_count=0,
        official_score=sum(scores) / len(scores),
        replicates_per_task=authority.replicates_per_task,
        trial_policy_sha256=authority.trial_policy_sha256,
        test_registry_sha256=authority.test_registry_sha256,
        official_scorer=authority.official_scorer,
        official_metric=authority.official_metric,
        scorer_sha256=authority.scorer_sha256,
        runtime_name=authority.runtime_name,
        runtime_sha256=authority.runtime_sha256,
        cheap_model=CHEAP_MODEL,
        cheap_reasoning_effort=REASONING_EFFORT,
        strong_model=STRONG_MODEL,
        strong_reasoning_effort=REASONING_EFFORT,
        fit_registry_sha256=typed_artifact.binding.fit_registry_sha256,
        dev_registry_sha256=typed_artifact.binding.dev_registry_sha256,
        method_artifact_sha256=typed_artifact.binding.artifact_sha256,
        calibration_sha256=typed_artifact.binding.calibration_sha256,
        implementation_bundle_sha256=implementation_bundle_sha256,
        run_config_sha256=run_config_sha256,
        private_raw_case_manifest_sha256=(private_raw_case_manifest_sha256),
        raw_usage_receipt_manifest_sha256=(raw_usage_receipt_manifest_sha256),
        registry_ordered_outcome_cost_manifest_sha256=(
            registry_ordered_outcome_cost_manifest_sha256
        ),
        case_receipt_count=len(raw_cases),
        registry_order_matches_test_manifest=True,
        agent_side_provider_api=ProviderAPIAccounting(
            usage_complete=True,
            provider_calls=agent["provider_calls"],
            retry_count=agent["retry_count"],
            input_tokens=agent["input_tokens"],
            cached_input_tokens=agent["cached_input_tokens"],
            output_tokens=agent["output_tokens"],
            reasoning_tokens=agent["reasoning_tokens"],
            cache_neutral_cost_usd=agent["cost_usd"],
        ),
        local_router=LocalRouterReport(
            decision_count=decision_count,
            latency_seconds_total=sum(
                receipt.latency_seconds
                for receipt in receipts
                if receipt.role == "router" and receipt.provider_calls == 0
            ),
            hardware=local_hardware
            or _local_hardware_label(typed_artifact.artifact),
            software=local_software
            or _local_software_label(typed_artifact.artifact),
        ),
        public_safe=True,
    )
    public_receipt.validate_against(authority)
    outcome_cost_manifest.validate_against(public_receipt)
    return Tau2CellExecution(
        result=JobExecutionResult(
            completed_item_count=len(outcomes),
            registry_sha256=registry.test_registry_sha256,
            output_payload_sha256=payload_sha256,
            infra_failure_count=0,
            replicates_per_task=authority.replicates_per_task,
            trial_policy_sha256=authority.trial_policy_sha256,
            authority_manifest_file_sha256=(
                job.authority_manifest_file_sha256
            ),
            authority_manifest_semantic_sha256=(
                job.authority_manifest_semantic_sha256
            ),
            scorer_sha256=job.scorer_sha256,
            runtime_sha256=job.runtime_sha256,
            merged_from_prior_runs=False,
        ),
        public_receipt=public_receipt,
        private_payload=private_payload,
    )


def write_public_receipt_append_only(
    path: Path,
    receipt: Table1ResultReceipt,
) -> None:
    _write_json_append_only(path, receipt.public_record())


def _execute_tau2_task(
    executor: Tau2NativeExecutor,
) -> dict[str, Any]:
    method = executor.driver.artifact.method
    if method == ROUTELLM:
        native = executor.run_request_routed()
        return _native_outcome(
            native,
            private_simulations=executor.private_simulation_manifest,
        )
    if method in TAU2_TURN_LEVEL_METHODS:
        native = executor.run_history_routed()
        return _native_outcome(
            native,
            private_simulations=executor.private_simulation_manifest,
        )
    if method not in TAU2_TASK_LEVEL_METHODS:
        raise ValueError("tau2_live_method_not_supported")
    episode = executor.bootstrap_pristine_episode(
        role="cheap",
        model=CHEAP_MODEL,
    )
    if method == FRUGALGPT:
        result = executor.driver.run_frugalgpt(
            episode=episode,
            run_episode=executor.run_episode_callback,
            score_selected=executor.score_selected_callback,
        )
    elif method == AUTOMIX:
        result = executor.driver.run_automix(
            episode=episode,
            run_episode=executor.run_episode_callback,
            score_selected=executor.score_selected_callback,
        )
    else:
        raise ValueError("tau2_live_task_method_not_supported")
    return _static_outcome(
        result,
        private_simulations=executor.private_simulation_manifest,
    )


def _native_outcome(
    native: Tau2NativeEpisode,
    *,
    private_simulations: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    return {
        "official_score": native.official_score,
        "selected_roles": tuple(
            decision.selected_role for decision in native.decisions
        ),
        "decision_count": len(native.decisions),
        "receipts": native.receipts,
        "private_simulations": private_simulations,
    }


def _static_outcome(
    result: StaticRoutingResult,
    *,
    private_simulations: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    return {
        "official_score": result.official_score,
        "selected_roles": (result.selected_role,),
        "decision_count": 1,
        "receipts": result.receipts,
        "private_simulations": private_simulations,
    }


def _routellm_embed_transport(
    *,
    provider_client: Any,
    broker: ProviderAttemptBroker,
):
    def embed(
        request: RouteLLMMFEmbeddingRequest,
    ) -> RouteLLMMFEmbeddingResult:
        returned: list[tuple[tuple[float, ...], CallReceipt]] = []

        def raw_attempt() -> CallReceipt:
            started = time.perf_counter()
            response = provider_client.embeddings.create(
                model=request.model,
                input=request.prompt,
            )
            data = tuple(getattr(response, "data", ()) or ())
            if len(data) != 1:
                raise ValueError("routellm_embedding_response_count_invalid")
            embedding = tuple(getattr(data[0], "embedding", ()) or ())
            usage = getattr(response, "usage", None)
            input_tokens = _usage_int(
                usage,
                "prompt_tokens",
                fallback="total_tokens",
            )
            receipt = CallReceipt(
                role="router",
                phase="route_embedding",
                model=ROUTELLM_EMBEDDING_MODEL,
                reasoning_effort="none",
                provider_calls=1,
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                retry_count=0,
                latency_seconds=max(0.0, time.perf_counter() - started),
                provider_reported_cost_usd=_cache_neutral_cost(
                    model=ROUTELLM_EMBEDDING_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=0,
                ),
            )
            returned.append((embedding, receipt))
            return receipt

        broker.perform(raw_attempt)
        if len(returned) != 1:
            raise RuntimeError("routellm_embedding_result_missing")
        embedding, receipt = returned[0]
        return RouteLLMMFEmbeddingResult(
            request_sha256=request.sha256,
            embedding=embedding,
            receipt=receipt,
        )

    return embed


def _automix_verifier_transport(
    *,
    provider_client: Any,
    broker: ProviderAttemptBroker,
):
    def verify(
        request: AutoMixVerifierRequest,
    ) -> AutoMixVerifierTransportResult:
        returned: list[tuple[tuple[str, ...], CallReceipt]] = []

        def raw_attempt() -> CallReceipt:
            started = time.perf_counter()
            response = provider_client.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{"role": "user", "content": request.prompt}],
                n=request.sample_count,
                temperature=request.temperature,
                stop=request.stop,
                max_tokens=request.max_output_tokens,
                reasoning_effort=REASONING_EFFORT,
            )
            choices = tuple(getattr(response, "choices", ()) or ())
            outputs = tuple(
                str(getattr(getattr(choice, "message", None), "content", ""))
                for choice in choices
            )
            if len(outputs) != request.sample_count:
                raise ValueError("automix_verifier_response_count_invalid")
            usage = getattr(response, "usage", None)
            input_tokens = _usage_int(usage, "prompt_tokens")
            output_tokens = _usage_int(usage, "completion_tokens")
            receipt = CallReceipt(
                role="verifier",
                phase="self_verification_n32",
                model=CHEAP_MODEL,
                reasoning_effort=REASONING_EFFORT,
                provider_calls=1,
                input_tokens=input_tokens,
                cached_input_tokens=_nested_usage_int(
                    usage,
                    "prompt_tokens_details",
                    "cached_tokens",
                ),
                output_tokens=output_tokens,
                reasoning_tokens=_nested_usage_int(
                    usage,
                    "completion_tokens_details",
                    "reasoning_tokens",
                ),
                retry_count=0,
                latency_seconds=max(0.0, time.perf_counter() - started),
                provider_reported_cost_usd=_cache_neutral_cost(
                    model=CHEAP_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
            returned.append((outputs, receipt))
            return receipt

        broker.perform(raw_attempt)
        if len(returned) != 1:
            raise RuntimeError("automix_verifier_result_missing")
        outputs, receipt = returned[0]
        return AutoMixVerifierTransportResult(
            request_prompt_sha256=request.prompt_sha256,
            outputs=outputs,
            receipts=(receipt,),
        )

    return verify


def _load_automix_tokenizer(root: Path) -> Any:
    validate_automix_verifier_tokenizer_root(root)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
    )


def _load_routellm_artifact(
    path: Path,
) -> RouteLLMMFArtifact | RouteLLMBertArtifact:
    raw = _read_mapping(path, "routellm_artifact")
    if raw.get("schema") == ROUTELLM_BERT_ARTIFACT_SCHEMA:
        return RouteLLMBertArtifact.from_public_record(raw)
    calibration_raw = raw.get("dev_calibration")
    if not isinstance(calibration_raw, Mapping):
        raise ValueError("routellm_artifact_calibration_invalid")
    candidates = calibration_raw.get("candidate_receipts")
    if not isinstance(candidates, list):
        raise ValueError("routellm_artifact_candidates_invalid")
    calibration = RouteLLMMFDevCalibration(
        dev_registry_sha256=calibration_raw["dev_registry_sha256"],
        dev_membership_sha256=calibration_raw["dev_membership_sha256"],
        strong_full_dev_anchor=calibration_raw["strong_full_dev_anchor"],
        threshold=calibration_raw["threshold"],
        dev_row_count=calibration_raw["dev_row_count"],
        observed_strong_count=calibration_raw["observed_strong_count"],
        dev_examples_sha256=calibration_raw["dev_examples_sha256"],
        candidate_receipts=tuple(
            RouteLLMMFDevCandidateReceipt(**dict(item)) for item in candidates
        ),
        selection_rule=calibration_raw["selection_rule"],
    )
    classes = raw.get("class_counts")
    if not isinstance(classes, Mapping):
        raise ValueError("routellm_artifact_class_counts_invalid")
    artifact = RouteLLMMFArtifact(
        surface=raw["surface"],
        fit_registry_sha256=raw["fit_registry_sha256"],
        dev_registry_sha256=raw["dev_registry_sha256"],
        test_registry_sha256=raw["test_registry_sha256"],
        fit_corpus_sha256=raw["fit_corpus_sha256"],
        fit_membership_sha256=raw["fit_membership_sha256"],
        dev_membership_sha256=raw["dev_membership_sha256"],
        split_binding_sha256=raw["split_binding_sha256"],
        fit_row_count=raw["fit_row_count"],
        discarded_tie_count=raw["discarded_tie_count"],
        class_counts=(
            ("cheap_wins", classes["cheap_wins"]),
            ("strong_wins", classes["strong_wins"]),
        ),
        class_counts_sha256=raw["class_counts_sha256"],
        checkpoint_sha256=raw["checkpoint_sha256"],
        config_sha256=raw["config_sha256"],
        serializer_sha256=raw["serializer_sha256"],
        dev_calibration=calibration,
        upstream_revision=raw["upstream_revision"],
        cheap_model=raw["cheap_model"],
        strong_model=raw["strong_model"],
        reasoning_effort=raw["reasoning_effort"],
        embedding_model=raw["embedding_model"],
    )
    if _digest(artifact.public_record()) != _digest(raw):
        raise ValueError("routellm_artifact_record_not_canonical")
    return artifact


def _load_automix_artifact(path: Path) -> AutoMixRuntimeArtifact:
    raw = _read_mapping(path, "automix_artifact")
    if raw.get("schema") != "papc.automix.runtime_artifact.v2":
        raise ValueError("automix_artifact_schema_invalid")
    policy_raw = raw.get("policy")
    if not isinstance(policy_raw, Mapping):
        raise ValueError("automix_artifact_policy_invalid")
    candidates = policy_raw.get("candidate_receipts")
    if not isinstance(candidates, list):
        raise ValueError("automix_artifact_candidates_invalid")

    def parameter(value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    policy = AutoMixPOMDPPolicy(
        family=policy_raw["family"],
        parameter=parameter(policy_raw["parameter"]),
        ibc_lift=policy_raw["ibc_lift"],
        fit_rows=policy_raw["fit_rows"],
        candidate_count=policy_raw["candidate_count"],
        cheap_cost=policy_raw["cheap_cost"],
        strong_cost=policy_raw["strong_cost"],
        verifier_cost=policy_raw["verifier_cost"],
        cheap_quality_p10=policy_raw["cheap_quality_p10"],
        strong_quality_p10=policy_raw["strong_quality_p10"],
        fit_candidate_set_sha256=policy_raw["fit_candidate_set_sha256"],
        dev_registry_sha256=policy_raw["dev_registry_sha256"],
        dev_membership_sha256=policy_raw["dev_membership_sha256"],
        dev_examples_sha256=policy_raw["dev_examples_sha256"],
        strong_full_dev_anchor=policy_raw["strong_full_dev_anchor"],
        dev_quality=policy_raw["dev_quality"],
        dev_cost_usd=policy_raw["dev_cost_usd"],
        dev_strong_count=policy_raw["dev_strong_count"],
        candidate_receipts=tuple(
            AutoMixDevCandidateReceipt(
                family=item["family"],
                parameter=parameter(item["parameter"]),
                config_sha256=item["config_sha256"],
                fit_ibc_lift=item["fit_ibc_lift"],
                dev_quality=item["dev_quality"],
                provider_api_cost_usd=item["provider_api_cost_usd"],
                strong_count=item["strong_count"],
            )
            for item in candidates
        ),
        selection_rule=policy_raw["selection_rule"],
        deterministic_tie_break=policy_raw["deterministic_tie_break"],
        upstream_revision=policy_raw["upstream_revision"],
        verifier_samples=policy_raw["verifier_samples"],
        pomdp_num_bins=policy_raw["pomdp_num_bins"],
        prompt_template_sha256=policy_raw["prompt_template_sha256"],
    )
    artifact = AutoMixRuntimeArtifact(
        surface=raw["surface"],
        policy=policy,
        fit_registry_sha256=raw["fit_registry_sha256"],
        dev_registry_sha256=raw["dev_registry_sha256"],
        test_registry_sha256=raw["test_registry_sha256"],
        fit_membership_sha256=raw["fit_membership_sha256"],
        dev_membership_sha256=raw["dev_membership_sha256"],
        split_binding_sha256=raw["split_binding_sha256"],
        cost_rule_sha256=raw["cost_rule_sha256"],
        context_serializer_sha256=raw["context_serializer_sha256"],
        cheap_model=raw["cheap_model"],
        strong_model=raw["strong_model"],
        upstream_revision=raw["upstream_revision"],
    )
    if (
        raw.get("context_serializer_protocol") != AGENT_CONTEXT_SERIALIZER_PROTOCOL
        or raw.get("policy_sha256") != policy.sha256
        or _digest(artifact.public_record()) != _digest(raw)
    ):
        raise ValueError("automix_artifact_record_not_canonical")
    return artifact


def _usage_int(
    usage: Any,
    field_name: str,
    *,
    fallback: str | None = None,
) -> int:
    value = _get_value(usage, field_name)
    if value is None and fallback is not None:
        value = _get_value(usage, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provider_usage_{field_name}_invalid")
    return value


def _nested_usage_int(
    usage: Any,
    parent: str,
    child: str,
) -> int:
    value = _get_value(_get_value(usage, parent), child)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provider_usage_{parent}_{child}_invalid")
    return value


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _cache_neutral_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    price = FROZEN_CACHE_NEUTRAL_PRICES.get(model)
    if price is None:
        raise ValueError("provider_model_price_missing")
    return (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000


def _validated_private_simulations(
    manifest: Any,
    *,
    root: Path,
) -> list[dict[str, str]]:
    if not isinstance(manifest, tuple) or not manifest:
        raise ValueError("tau2_private_simulation_manifest_missing")
    validated: list[dict[str, str]] = []
    for item in manifest:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "sha256",
        }:
            raise ValueError("tau2_private_simulation_record_invalid")
        relative = item["relative_path"]
        claimed = _require_sha256(
            item["sha256"],
            "tau2_private_simulation_sha256",
        )
        if not isinstance(relative, str) or not relative:
            raise ValueError("tau2_private_simulation_path_invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("tau2_private_simulation_path_escapes_root") from exc
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != claimed
        ):
            raise ValueError("tau2_private_simulation_hash_mismatch")
        validated.append(
            {
                "relative_path": relative,
                "sha256": claimed,
            }
        )
    return validated


def _local_hardware_label(artifact: Any) -> str:
    if isinstance(artifact, RouteLLMBertArtifact):
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    return platform.processor() or platform.machine() or "unknown-local-cpu"


def _local_software_label(artifact: Any) -> str:
    if isinstance(artifact, RouteLLMBertArtifact):
        import torch
        import transformers

        return (
            f"python-{platform.python_version()}:"
            f"torch-{torch.__version__}:transformers-{transformers.__version__}"
        )
    return f"python-{platform.python_version()}:{platform.system().lower()}"


def _implementation_bundle_sha256(method: str) -> str:
    root = Path(__file__).resolve().parent
    method_files = {
        ROUTELLM: (
            "routellm_mf_upstream.py",
            "routellm_bert_upstream.py",
        ),
        FRUGALGPT: ("frugalgpt_upstream.py",),
        AUTOMIX: ("automix_upstream.py",),
        MTROUTER: ("mtrouter_upstream.py",),
        OPENSQUILLA_G0: ("opensquilla_upstream.py",),
    }[method]
    relatives = (
        "tau2_external_router_live.py",
        "tau2_native_runtime.py",
        "tau2_router_driver.py",
        "sealed_router_policy.py",
        "router_runtime.py",
        "router_accounting.py",
        "external_router_execution.py",
        *method_files,
    )
    return _digest(
        [
            {
                "path": relative,
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            }
            for relative in relatives
        ]
    )


def _resource_binding(
    *,
    method: str,
    resources: Tau2PolicyResources,
    typed_artifact: Tau2TypedArtifact,
) -> dict[str, Any]:
    if method == MTROUTER:
        artifact = typed_artifact.artifact
        if (
            isinstance(artifact, MTRouterArtifact)
            and artifact.uses_all_strong_shortcut
        ):
            if resources.checkpoint_path is not None:
                raise ValueError("tau2_mtrouter_all_strong_asset_forbidden")
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
        return {
            "kind": "checkpoint_directory",
            "sha256": _directory_bundle_sha256(
                _required_dir(
                    resources.checkpoint_path,
                    "routellm_bert_checkpoint",
                )
            ),
        }
    if method in {ROUTELLM, MTROUTER}:
        checkpoint = _required_file(
            resources.checkpoint_path,
            "typed_checkpoint",
        )
        return {
            "kind": "checkpoint_file",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    if method == FRUGALGPT:
        return {
            "kind": "checkpoint_directory",
            "sha256": _directory_bundle_sha256(
                _required_dir(
                    resources.checkpoint_path,
                    "frugalgpt_checkpoint",
                )
            ),
        }
    if method == AUTOMIX:
        return {
            "kind": "tokenizer_directory",
            "sha256": _directory_bundle_sha256(
                _required_dir(
                    resources.automix_tokenizer_root,
                    "automix_tokenizer",
                )
            ),
        }
    if method == OPENSQUILLA_G0:
        checkout = _required_dir(
            resources.opensquilla_checkout_root,
            "opensquilla_checkout",
        )
        bundle = _required_dir(
            resources.opensquilla_bundle_dir,
            "opensquilla_adapted_bundle",
        )
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "kind": "opensquilla_adapted_runtime",
            "checkout_revision": completed.stdout.strip(),
            "checkout_source_sha256": _directory_bundle_sha256(checkout),
            "adapted_bundle_source_sha256": _directory_bundle_sha256(bundle),
            "artifact_bundle_sha256": typed_artifact.artifact.bundle_sha256,
            "bundle_digest_reverified_by_loader": True,
        }
    raise ValueError("tau2_resource_method_invalid")


def _directory_bundle_sha256(root: Path) -> str:
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    )
    if not files:
        raise ValueError("tau2_resource_directory_empty")
    return _digest(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ]
    )


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label}_not_mapping")
    return payload


def _required_file(path: Path | None, label: str) -> Path:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"{label}_missing")
    return Path(path).resolve()


def _required_dir(path: Path | None, label: str) -> Path:
    if path is None or not Path(path).is_dir():
        raise FileNotFoundError(f"{label}_missing")
    return Path(path).resolve()


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _write_json_line_append_only(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_append_only(path, raw)


def _write_json_append_only(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_append_only(path, raw)


def _atomic_write_append_only(path: Path, raw: bytes) -> None:
    """Publish complete bytes with no overwrite and no partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

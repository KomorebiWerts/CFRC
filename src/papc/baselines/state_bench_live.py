"""Authorized, source-bound STATE-Bench execution for Table 1 routers.

This module is inert until a caller supplies an exact execution-v4 plan,
Table-1 authority manifest, stage authorization, explicit official-test job,
sealed typed router artifact, and an API key obtained outside this module.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from papc.baselines.automix_upstream import (
    AutoMixDevCandidateReceipt,
    AutoMixPOMDPPolicy,
    AutoMixRuntimeArtifact,
    AutoMixVerifierRequest,
    validate_automix_verifier_tokenizer_root,
)
from papc.baselines.external_router_execution import (
    OFFICIAL_REPLICATES_PER_TASK,
    STATE_BENCH,
    TABLE1_AUTHORITY_FILE_SHA256,
    TEST_STAGE,
    BatchCoordinator,
    ExecutionJob,
    ExecutionPlan,
    JobExecutionResult,
    ProviderAttemptBroker,
    StageExecutionAuthorization,
    load_execution_plan,
)
from papc.baselines.frugalgpt_upstream import FrugalGPTScorerArtifact
from papc.baselines.mtrouter_upstream import (
    MTRouterArtifact,
    load_mtrouter_artifact,
)
from papc.baselines.opensquilla_upstream import (
    OPENSQUILLA_TRAINING_SOURCE_SHA256S,
    OPENSQUILLA_UPSTREAM_REVISION,
    load_opensquilla_runtime_artifact,
    opensquilla_bundle_tree_sha256,
)
from papc.baselines.routellm_mf_upstream import (
    RouteLLMMFArtifact,
    RouteLLMMFDevCalibration,
    RouteLLMMFDevCandidateReceipt,
    RouteLLMMFEmbeddingRequest,
    RouteLLMMFEmbeddingResult,
)
from papc.baselines.router_accounting import (
    AGENT_SIDE_ROLES,
    CallReceipt,
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
    ExternalMethod,
    RouterArtifactBinding,
)
from papc.baselines.router_runtime_environment import (
    validate_router_runtime_environment,
)
from papc.baselines.sealed_router_policy import (
    AutoMixVerifierTransportResult,
    SealedRouterPolicy,
    load_automix_sealed_policy,
    load_frugalgpt_sealed_policy,
    load_mtrouter_sealed_policy,
    load_opensquilla_sealed_policy,
    load_routellm_sealed_policy,
)
from papc.baselines.state_bench_native_runtime import (
    FROZEN_STATE_BENCH_RUNTIME_REVISION,
    STATE_BENCH_EVALUATOR_MODEL,
    STATE_BENCH_OFFICIAL_SCORER,
    STATE_BENCH_USER_MODEL,
    StateBenchNativeAPI,
    StateBenchNativeExecutor,
    StateBenchNativeRunConfig,
    StateBenchRuntimePolicy,
    brokered_provider_create,
    load_state_bench_native_api,
)
from papc.baselines.state_bench_router_driver import (
    StateBenchOfficialTaskRef,
    StateBenchOfficialTestRegistry,
    StateBenchRouterDriver,
    load_state_bench_official_test_registry,
)
from papc.baselines.table1_authority import trial_policy_sha256
from papc.baselines.table1_paired_stats import (
    CaseOutcomeCost,
    RegistryOrderedOutcomeCostManifest,
)
from papc.baselines.table1_results import (
    FULL_AGGREGATION_SCOPE,
    LocalRouterReport,
    ProviderAPIAccounting,
    SurfaceAuthority,
    Table1AuthorityManifest,
    Table1ResultReceipt,
    load_table1_authority,
)

STATE_BENCH_LIVE_SCHEMA = "papc.state_bench.external_router_live.v1"
STATE_BENCH_AUTHORITY_MANIFEST_SHA256 = (
    "a23f0614f71270ed6b9870880389817d48e6cdd82d5de82b4dfcbb7e20acb101"
)
STATE_BENCH_OFFICIAL_METRIC = "task_completion_pass"
STATE_BENCH_RUNTIME_NAME = "state_bench.native_single_task_run"
STATE_BENCH_OFFICIAL_TEST_COUNT = 150
STATE_BENCH_RUNTIME_ENVIRONMENT_PACKAGES = (
    "openai",
    "azure-identity",
    "azure-core",
    "python-dotenv",
    "pyyaml",
    "tiktoken",
    "tenacity",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "joblib",
    "torch",
    "transformers",
    "lightgbm",
    "onnxruntime",
    "sentence-transformers",
)
STATE_BENCH_ROUTER_MODULES = (
    "papc.baselines.routellm_mf_upstream",
    "papc.baselines.frugalgpt_upstream",
    "papc.baselines.automix_upstream",
    "papc.baselines.mtrouter_upstream",
    "papc.baselines.opensquilla_upstream",
)
FROZEN_STATE_BENCH_RUNTIME_ENVIRONMENT_SHA256 = (
    "737c7536f3299014144f65ca8946868e0622bd56a4964558213d834d53550787"
)
# This fingerprint covers the interpreter, locked dependencies, STATE import,
# and five router-module imports.  Real-artifact load/inference receipts remain
# a separate prerequisite in the unified upstream execution gate.


class StateBenchLiveContractError(ValueError):
    """A provider-zero execution authority or source binding is invalid."""


@dataclass(frozen=True)
class StateBenchLiveRequest:
    method: ExternalMethod
    typed_router_artifact_json: Path
    runtime_asset_path: Path | None
    state_repo: Path
    split_gate_json: Path
    budget_gate_json: Path
    authority_manifest_json: Path
    authority_manifest_sha256: str
    private_root: Path
    api_base: str
    api_key_env: str
    authorization_id: str
    plan_sha256: str
    stage: str
    stage_provider_attempt_hard_cap: int
    stage_observed_spend_stop_usd: Decimal
    job_id: str
    device: str = "cuda"
    opensquilla_checkout_root: Path | None = None


@dataclass(frozen=True)
class TypedStateBenchArtifact:
    method: ExternalMethod
    artifact: Any
    runtime_asset_path: Path | None

    @property
    def binding(self) -> RouterArtifactBinding:
        if self.method == ROUTELLM:
            return self.artifact.to_router_binding()
        if self.method in {FRUGALGPT, AUTOMIX, MTROUTER, OPENSQUILLA_G0}:
            return self.artifact.router_binding()
        raise StateBenchLiveContractError("state_bench_live_method_invalid")


@dataclass(frozen=True)
class StateBenchLiveContext:
    request: StateBenchLiveRequest
    plan: ExecutionPlan
    job: ExecutionJob
    authorization: StageExecutionAuthorization
    authority_manifest: Table1AuthorityManifest
    surface_authority: SurfaceAuthority
    registry: StateBenchOfficialTestRegistry
    native_api: StateBenchNativeAPI
    runtime_environment_sha256: str
    typed_artifact: TypedStateBenchArtifact
    native_config: StateBenchNativeRunConfig

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": STATE_BENCH_LIVE_SCHEMA,
            "method": self.request.method,
            "plan_sha256": self.plan.sha256,
            "job_id": self.job.job_id,
            "stage": self.authorization.stage,
            "stage_provider_attempt_hard_cap": (
                self.authorization.provider_attempt_hard_cap
            ),
            "stage_observed_spend_stop_usd": float(
                self.authorization.observed_spend_stop_usd
            ),
            "authority_manifest_sha256": self.authority_manifest.sha256,
            "test_registry_sha256": self.registry.test_registry_sha256,
            "source_binding_sha256": self.registry.source_binding_sha256,
            "runtime_environment_sha256": self.runtime_environment_sha256,
            "official_test_count": self.registry.denominator,
            "replicates_per_task": self.surface_authority.replicates_per_task,
            "table1_trial_policy_sha256": (self.surface_authority.trial_policy_sha256),
            "execution_trial_policy_sha256": self.job.trial_policy_sha256,
            "method_artifact_sha256": self.typed_artifact.binding.artifact_sha256,
            "runtime_asset_binding_sha256": _digest(
                state_bench_runtime_asset_binding(
                    typed=self.typed_artifact,
                    opensquilla_checkout_root=(
                        self.request.opensquilla_checkout_root
                    ),
                )
            ),
            "output_target": str(self.job.output_path),
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "ready_for_explicit_execute": True,
        }


@dataclass(frozen=True)
class StateBenchNativeClients:
    cheap: Any
    strong: Any
    embedding: Any
    simulator: Any
    evaluator: Any


@dataclass(frozen=True)
class StateBenchTaskResult:
    example_sha256: str
    official_score: float
    decision_count: int
    local_router_latency_seconds: float
    receipts: tuple[CallReceipt, ...]
    private_case_manifest_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "state_bench_task_example")
        if (
            isinstance(self.official_score, bool)
            or not isinstance(self.official_score, (int, float))
            or not 0.0 <= float(self.official_score) <= 1.0
        ):
            raise ValueError("state_bench_task_official_score_invalid")
        if (
            isinstance(self.decision_count, bool)
            or not isinstance(self.decision_count, int)
            or self.decision_count <= 0
        ):
            raise ValueError("state_bench_task_decision_count_invalid")
        if (
            isinstance(self.local_router_latency_seconds, bool)
            or not isinstance(self.local_router_latency_seconds, (int, float))
            or float(self.local_router_latency_seconds) < 0.0
        ):
            raise ValueError("state_bench_task_router_latency_invalid")
        receipts = tuple(self.receipts)
        if not receipts or any(not isinstance(item, CallReceipt) for item in receipts):
            raise TypeError("state_bench_task_receipts_invalid")
        _require_atomic_complete_provider_receipts(receipts)
        private_record = _mapping(
            self.private_case_manifest_record,
            "state_bench_private_case_manifest_record",
        )
        if private_record.get("example_sha256") != self.example_sha256:
            raise ValueError("state_bench_private_case_example_sha256_mismatch")
        private_files = _sequence(
            private_record.get("private_files"),
            "state_bench_private_case_files",
        )
        if not private_files:
            raise ValueError("state_bench_private_case_files_empty")
        for raw_file in private_files:
            file_record = _mapping(
                raw_file,
                "state_bench_private_case_file",
            )
            _require_sha256(
                file_record.get("sha256"),
                "state_bench_private_case_file_sha256",
            )
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(
            self,
            "private_case_manifest_record",
            dict(private_record),
        )


RawClientFactory = Callable[[str, str], Any]
TaskRunner = Callable[[StateBenchOfficialTaskRef], StateBenchTaskResult]


def _normalise_responses_item(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return item


class _StatelessResponsesResource:
    """Expand response ids into explicit history for compatible endpoints."""

    def __init__(self, upstream: Any) -> None:
        self._upstream = upstream
        self._histories: dict[str, list[Any]] = {}

    def create(self, *args: Any, **kwargs: Any) -> Any:
        request = dict(kwargs)
        previous_response_id = request.pop("previous_response_id", None)
        current_input = [
            _normalise_responses_item(item)
            for item in list(request.get("input") or [])
        ]
        if previous_response_id is not None:
            prior = self._histories.get(str(previous_response_id))
            if prior is None:
                raise ValueError(
                    "state_bench_previous_response_history_missing"
                )
            current_input = [*prior, *current_input]
        request["input"] = current_input
        response = self._upstream.create(*args, **request)
        response_id = str(getattr(response, "id", ""))
        if not response_id:
            raise ValueError("state_bench_provider_response_id_missing")
        output = [
            _normalise_responses_item(item)
            for item in list(getattr(response, "output", None) or [])
        ]
        self._histories[response_id] = [*current_input, *output]
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


class _StatelessOpenAICompatibleClient:
    def __init__(self, upstream: Any) -> None:
        self._upstream = upstream
        self.responses = _StatelessResponsesResource(upstream.responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


def state_bench_runtime_environment_record(
    api: StateBenchNativeAPI,
) -> dict[str, Any]:
    """Return the canonical locked-runtime record without provider access."""

    if not isinstance(api, StateBenchNativeAPI):
        raise TypeError("state_bench_live_native_api_invalid")
    modules = [
        importlib.import_module(module_name).__name__
        for module_name in STATE_BENCH_ROUTER_MODULES
    ]
    if modules != list(STATE_BENCH_ROUTER_MODULES):
        raise StateBenchLiveContractError(
            "state_bench_live_router_module_identity_mismatch"
        )
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            package_name: importlib.metadata.version(package_name)
            for package_name in STATE_BENCH_RUNTIME_ENVIRONMENT_PACKAGES
        },
        "state": api.public_record(),
        "router_modules": modules,
    }


def validate_state_bench_runtime_environment(
    api: StateBenchNativeAPI,
) -> str:
    """Fail closed unless the exact locked dependency environment is active."""

    validate_router_runtime_environment()
    observed = _digest(state_bench_runtime_environment_record(api))
    if observed != FROZEN_STATE_BENCH_RUNTIME_ENVIRONMENT_SHA256:
        raise StateBenchLiveContractError(
            "state_bench_live_runtime_environment_fingerprint_mismatch"
        )
    return observed


def prepare_state_bench_live_context(
    request: StateBenchLiveRequest,
    *,
    registry_loader: Callable[..., StateBenchOfficialTestRegistry] = (
        load_state_bench_official_test_registry
    ),
    native_api_loader: Callable[[Path], StateBenchNativeAPI] = (
        load_state_bench_native_api
    ),
    typed_artifact_loader: Callable[
        [ExternalMethod, Path, Path],
        TypedStateBenchArtifact,
    ] = None,  # type: ignore[assignment]
    source_authority_loader: Callable[
        [StateBenchNativeAPI, StateBenchOfficialTestRegistry],
        tuple[str, str],
    ] = None,  # type: ignore[assignment]
) -> StateBenchLiveContext:
    """Validate every provider-zero boundary before a caller may read a key."""

    if not isinstance(request, StateBenchLiveRequest):
        raise TypeError("state_bench_live_request_invalid")
    artifact_loader = typed_artifact_loader or load_typed_state_bench_artifact
    authority_loader = source_authority_loader or state_bench_source_authority
    plan = load_execution_plan(
        split_gate_json=request.split_gate_json,
        budget_gate_json=request.budget_gate_json,
        authority_manifest_json=request.authority_manifest_json,
        private_root=request.private_root,
    )
    if request.plan_sha256 != plan.sha256:
        raise StateBenchLiveContractError("state_bench_live_plan_sha256_mismatch")
    if request.stage != TEST_STAGE:
        raise StateBenchLiveContractError("state_bench_live_stage_not_official_test")
    authorization = StageExecutionAuthorization(
        authorization_id=request.authorization_id,
        plan_sha256=request.plan_sha256,
        stage=request.stage,
        provider_attempt_hard_cap=request.stage_provider_attempt_hard_cap,
        observed_spend_stop_usd=request.stage_observed_spend_stop_usd,
    )
    # Constructor validation binds the exact plan SHA, stage cap, and USD stop.
    BatchCoordinator(plan=plan, authorization=authorization)

    try:
        job = plan.job(request.job_id)
    except KeyError as exc:
        raise StateBenchLiveContractError(
            "state_bench_live_explicit_job_unknown"
        ) from exc
    if (
        job.stage != TEST_STAGE
        or job.kind != "official_test_cell"
        or job.surface != STATE_BENCH
        or job.method != request.method
        or job.item_count != STATE_BENCH_OFFICIAL_TEST_COUNT
        or job.replicates_per_task != OFFICIAL_REPLICATES_PER_TASK
        or not job.requires_router_artifact
    ):
        raise StateBenchLiveContractError(
            "state_bench_live_job_not_exact_official_cell"
        )
    if job.output_path.exists():
        raise FileExistsError("state_bench_live_output_exists_no_overwrite")

    authority_manifest = load_table1_authority(request.authority_manifest_json)
    if (
        request.authority_manifest_sha256 != authority_manifest.sha256
        or authority_manifest.sha256 != STATE_BENCH_AUTHORITY_MANIFEST_SHA256
        or plan.authority_manifest_semantic_sha256
        != authority_manifest.sha256
        or plan.authority_manifest_file_sha256
        != TABLE1_AUTHORITY_FILE_SHA256
    ):
        raise StateBenchLiveContractError(
            "state_bench_live_authority_manifest_sha256_mismatch"
        )
    surface_authority = authority_manifest.for_surface(STATE_BENCH)
    registry = registry_loader(
        request.state_repo,
        split_gate_json=request.split_gate_json,
    )
    if job.trial_policy_sha256 != surface_authority.trial_policy_sha256:
        raise StateBenchLiveContractError(
            "state_bench_live_execution_authority_trial_policy_mismatch"
        )
    native_api = native_api_loader(request.state_repo)
    runtime_environment_sha256 = validate_state_bench_runtime_environment(
        native_api
    )
    runtime_sha256, scorer_sha256 = authority_loader(native_api, registry)
    _validate_state_authority(
        surface_authority=surface_authority,
        registry=registry,
        runtime_sha256=runtime_sha256,
        scorer_sha256=scorer_sha256,
    )
    plan_authority = next(
        (item for item in plan.authorities if item.surface == STATE_BENCH),
        None,
    )
    if (
        plan_authority is None
        or plan_authority.denominator != STATE_BENCH_OFFICIAL_TEST_COUNT
        or plan_authority.replicates_per_task != 1
        or plan_authority.trial_policy_sha256
        != surface_authority.trial_policy_sha256
        or plan_authority.test_registry_sha256 != registry.test_registry_sha256
        or plan_authority.fit_registry_sha256
        != surface_authority.fit_registry_sha256
        or plan_authority.dev_registry_sha256
        != surface_authority.dev_registry_sha256
        or plan_authority.source_binding_sha256 != registry.source_binding_sha256
        or plan_authority.official_scorer
        != surface_authority.official_scorer
        or plan_authority.official_metric
        != surface_authority.official_metric
        or plan_authority.scorer_sha256
        != surface_authority.scorer_sha256
        or plan_authority.runtime_name != surface_authority.runtime_name
        or plan_authority.runtime_sha256
        != surface_authority.runtime_sha256
        or job.registry_sha256 != registry.test_registry_sha256
        or job.authority_test_registry_sha256
        != surface_authority.test_registry_sha256
        or job.source_binding_sha256 != registry.source_binding_sha256
        or job.trial_policy_sha256 != plan_authority.trial_policy_sha256
        or job.authority_manifest_file_sha256
        != plan.authority_manifest_file_sha256
        or job.authority_manifest_semantic_sha256
        != plan.authority_manifest_semantic_sha256
        or job.official_scorer != surface_authority.official_scorer
        or job.official_metric != surface_authority.official_metric
        or job.scorer_sha256 != surface_authority.scorer_sha256
        or job.runtime_name != surface_authority.runtime_name
        or job.runtime_sha256 != surface_authority.runtime_sha256
    ):
        raise StateBenchLiveContractError(
            "state_bench_live_execution_trial_authority_mismatch"
        )

    typed_artifact = artifact_loader(
        request.method,
        request.typed_router_artifact_json,
        request.runtime_asset_path,
    )
    binding = typed_artifact.binding
    if (
        typed_artifact.method != request.method
        or binding.surface != STATE_BENCH
        or binding.test_registry_sha256 != registry.test_registry_sha256
        or binding.fit_registry_sha256 != surface_authority.fit_registry_sha256
        or binding.dev_registry_sha256 != surface_authority.dev_registry_sha256
    ):
        raise StateBenchLiveContractError(
            "state_bench_live_typed_artifact_authority_mismatch"
        )
    state_bench_runtime_asset_binding(
        typed=typed_artifact,
        opensquilla_checkout_root=request.opensquilla_checkout_root,
    )
    native_config = StateBenchNativeRunConfig(
        api_base=request.api_base,
        private_output_root=request.private_root,
        max_provider_calls=job.provider_attempt_hard_cap,
        max_observed_provider_cost_usd=float(authorization.observed_spend_stop_usd),
    )
    if request.api_key_env != "OPENAI_API_KEY":
        raise StateBenchLiveContractError("state_bench_live_api_key_env_not_frozen")
    return StateBenchLiveContext(
        request=request,
        plan=plan,
        job=job,
        authorization=authorization,
        authority_manifest=authority_manifest,
        surface_authority=surface_authority,
        registry=registry,
        native_api=native_api,
        runtime_environment_sha256=runtime_environment_sha256,
        typed_artifact=typed_artifact,
        native_config=native_config,
    )


def execute_state_bench_live_context(
    context: StateBenchLiveContext,
    *,
    api_key: str,
    raw_client_factory: RawClientFactory | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> tuple[JobExecutionResult, dict[str, Any]]:
    """Execute one explicitly authorized whole official cell."""

    if not isinstance(context, StateBenchLiveContext):
        raise TypeError("state_bench_live_context_invalid")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("state_bench_live_api_key_missing")
    coordinator = BatchCoordinator(
        plan=context.plan,
        authorization=context.authorization,
    )
    clients = build_state_bench_native_clients(
        api=context.native_api,
        api_base=context.native_config.api_base,
        api_key=api_key,
        raw_client_factory=raw_client_factory,
    )

    def execute_job(
        job: ExecutionJob,
        broker: ProviderAttemptBroker,
    ) -> JobExecutionResult:
        sealed_policy = load_state_bench_sealed_policy(
            typed=context.typed_artifact,
            broker=broker,
            clients=clients,
            tokenizer_loader=tokenizer_loader,
            device=context.request.device,
            opensquilla_checkout_root=(
                context.request.opensquilla_checkout_root
            ),
        )
        driver = StateBenchRouterDriver(
            policy=sealed_policy,
            registry=context.registry,
        )

        def run_task(task: StateBenchOfficialTaskRef) -> StateBenchTaskResult:
            return run_state_bench_task(
                context=context,
                driver=driver,
                task=task,
                clients=clients,
                broker=broker,
            )

        return execute_state_bench_official_cell(
            job=job,
            registry=context.registry,
            surface_authority=context.surface_authority,
            binding=sealed_policy.binding,
            run_task=run_task,
            private_root=context.request.private_root,
            implementation_bundle_sha256=(
                state_bench_implementation_bundle_sha256(sealed_policy.binding.method)
            ),
            run_config_sha256=state_bench_run_config_sha256(context),
        )

    result = coordinator.execute_job(context.job.job_id, execute_job)
    return result, coordinator.public_report()


def build_state_bench_native_clients(
    *,
    api: StateBenchNativeAPI,
    api_base: str,
    api_key: str,
    raw_client_factory: RawClientFactory | None = None,
) -> StateBenchNativeClients:
    factory = raw_client_factory or _openai_raw_client

    def native(model: str) -> Any:
        raw = _StatelessOpenAICompatibleClient(factory(api_base, api_key))
        client = object.__new__(api.llm_client_class)
        client.provider = "openai"
        client.endpoint = api_base
        client.deployment = model
        client.deployments = [model]
        client._chat_response_history = {}
        client._client = raw
        return client

    return StateBenchNativeClients(
        cheap=native(CHEAP_MODEL),
        strong=native(STRONG_MODEL),
        embedding=native(ROUTELLM_EMBEDDING_MODEL),
        simulator=native(STATE_BENCH_USER_MODEL),
        evaluator=native(STATE_BENCH_EVALUATOR_MODEL),
    )


def load_state_bench_sealed_policy(
    *,
    typed: TypedStateBenchArtifact,
    broker: ProviderAttemptBroker,
    clients: StateBenchNativeClients,
    tokenizer_loader: Callable[[Path], Any] | None,
    device: str,
    opensquilla_checkout_root: Path | None = None,
) -> SealedRouterPolicy:
    binding = typed.binding
    asset = typed.runtime_asset_path
    if typed.method == ROUTELLM:

        def embed(
            request: RouteLLMMFEmbeddingRequest,
        ) -> RouteLLMMFEmbeddingResult:
            response, receipt = brokered_provider_create(
                broker=broker,
                create=clients.embedding._client.embeddings.create,
                kwargs={
                    "model": ROUTELLM_EMBEDDING_MODEL,
                    "input": request.prompt,
                    "encoding_format": "float",
                },
                role="router",
                model=ROUTELLM_EMBEDDING_MODEL,
                reasoning_effort="embedding",
                phase="route_embedding",
            )
            data = getattr(response, "data", None)
            if not isinstance(data, Sequence) or len(data) != 1:
                raise ValueError("state_bench_route_embedding_payload_invalid")
            vector = getattr(data[0], "embedding", None)
            if not isinstance(vector, Sequence):
                raise ValueError("state_bench_route_embedding_vector_invalid")
            return RouteLLMMFEmbeddingResult(
                request_sha256=request.sha256,
                embedding=tuple(float(value) for value in vector),
                receipt=receipt,
            )

        return load_routellm_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_payload=asset.read_bytes(),
            embed_prompt=embed,
        )
    if typed.method == FRUGALGPT:
        return load_frugalgpt_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_dir=asset,
        )
    if typed.method == AUTOMIX:
        loader = tokenizer_loader or _load_automix_tokenizer
        tokenizer = loader(asset)

        def verify(
            request: AutoMixVerifierRequest,
        ) -> AutoMixVerifierTransportResult:
            response, receipt = brokered_provider_create(
                broker=broker,
                create=clients.cheap._client.chat.completions.create,
                kwargs={
                    "model": CHEAP_MODEL,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "n": request.sample_count,
                    "temperature": request.temperature,
                    "stop": request.stop,
                    "max_tokens": request.max_output_tokens,
                },
                role="verifier",
                model=CHEAP_MODEL,
                reasoning_effort="published_sampling",
                phase="automix_n32_verifier",
            )
            choices = getattr(response, "choices", None)
            if not isinstance(choices, Sequence):
                raise ValueError("state_bench_automix_choices_invalid")
            outputs = tuple(
                str(getattr(getattr(choice, "message", None), "content", ""))
                for choice in choices
            )
            if len(outputs) != request.sample_count or any(
                not item for item in outputs
            ):
                raise ValueError("state_bench_automix_sample_count_invalid")
            return AutoMixVerifierTransportResult(
                request_prompt_sha256=request.prompt_sha256,
                outputs=outputs,
                receipts=(receipt,),
            )

        return load_automix_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            tokenizer=tokenizer,
            transport=verify,
        )
    if typed.method == MTROUTER:
        return load_mtrouter_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_path=asset,
            device=device,
        )
    if typed.method == OPENSQUILLA_G0:
        checkout = _required_directory(
            opensquilla_checkout_root,
            "state_bench_opensquilla_checkout",
        )
        return load_opensquilla_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkout_root=checkout,
            bundle_dir=asset,
        )
    raise StateBenchLiveContractError("state_bench_live_method_invalid")


def run_state_bench_task(
    *,
    context: StateBenchLiveContext,
    driver: StateBenchRouterDriver,
    task: StateBenchOfficialTaskRef,
    clients: StateBenchNativeClients,
    broker: ProviderAttemptBroker,
) -> StateBenchTaskResult:
    domain = context.native_api.get_domain_config(task.domain)
    runtime_policy = StateBenchRuntimePolicy(
        cheap_client=clients.cheap,
        strong_client=clients.strong,
        embedding_client=clients.embedding,
        simulator_client=clients.simulator,
        evaluator_client=clients.evaluator,
        task_requirements_judge=context.native_api.task_requirements_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
        ux_judge=context.native_api.ux_quality_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
    )
    executor = StateBenchNativeExecutor(
        api=context.native_api,
        driver=driver,
        task=task,
        config=context.native_config,
        policy=runtime_policy,
        split_gate_json=context.request.split_gate_json,
        attempt_broker=broker,
    )
    if driver.artifact.method in {ROUTELLM, FRUGALGPT, AUTOMIX}:
        episode = executor.bind_pristine_episode()
        if driver.artifact.method == ROUTELLM:
            result = driver.run_routellm(
                episode=episode,
                run_episode=executor.run_episode_callback,
                score_selected=executor.score_selected_callback,
            )
        elif driver.artifact.method == FRUGALGPT:
            result = driver.run_frugalgpt(
                episode=episode,
                run_episode=executor.run_episode_callback,
                score_selected=executor.score_selected_callback,
            )
        else:
            result = driver.run_automix(
                episode=episode,
                run_episode=executor.run_episode_callback,
                score_selected=executor.score_selected_callback,
            )
        local_receipts = tuple(
            receipt
            for receipt in result.receipts
            if receipt.provider_calls == 0 or receipt.role in {"router", "verifier"}
        )
        score = result.official_score
        decision_count = 1
    else:
        episode = executor.run_history_routed()
        score = episode.score_handle.score()
        local_receipts = tuple(
            receipt for receipt in episode.receipts if receipt.provider_calls == 0
        )
        decision_count = len(episode.decisions)
    receipts = (*tuple(executor.budget.receipts), *local_receipts)
    return StateBenchTaskResult(
        example_sha256=task.example_sha256,
        official_score=score,
        decision_count=decision_count,
        local_router_latency_seconds=sum(
            receipt.latency_seconds
            for receipt in local_receipts
            if receipt.role == "router" and receipt.provider_calls == 0
        ),
        receipts=receipts,
        private_case_manifest_record=_private_case_manifest_record(
            example_sha256=task.example_sha256,
            private_root=context.request.private_root,
            run_dirs=tuple(executor.private_run_dirs),
        ),
    )


def execute_state_bench_official_cell(
    *,
    job: ExecutionJob,
    registry: StateBenchOfficialTestRegistry,
    surface_authority: SurfaceAuthority,
    binding: RouterArtifactBinding,
    run_task: TaskRunner,
    private_root: Path,
    implementation_bundle_sha256: str,
    run_config_sha256: str,
) -> JobExecutionResult:
    """Run all 150 tasks and publish only one complete normalized receipt."""

    if (
        job.kind != "official_test_cell"
        or job.surface != STATE_BENCH
        or job.item_count != STATE_BENCH_OFFICIAL_TEST_COUNT
        or registry.denominator != STATE_BENCH_OFFICIAL_TEST_COUNT
        or job.registry_sha256 != registry.test_registry_sha256
        or binding.test_registry_sha256 != registry.test_registry_sha256
        or job.method != binding.method
    ):
        raise StateBenchLiveContractError("state_bench_live_cell_binding_mismatch")
    if job.output_path.exists():
        raise FileExistsError("state_bench_live_output_exists_no_overwrite")
    _require_sha256(
        implementation_bundle_sha256,
        "state_bench_implementation_bundle_sha256",
    )
    _require_sha256(
        run_config_sha256,
        "state_bench_run_config_sha256",
    )
    evidence_dir = (
        job.output_path.parent / f".{job.output_path.name}.{uuid4().hex}.private"
    )
    evidence_dir.mkdir(parents=True, exist_ok=False)
    results: list[StateBenchTaskResult] = []
    for task in registry.tasks:
        result = run_task(task)
        if result.example_sha256 != task.example_sha256:
            raise StateBenchLiveContractError(
                "state_bench_live_task_result_identity_mismatch"
            )
        _validate_private_case_manifest_record(
            result.private_case_manifest_record,
            private_root=private_root,
        )
        results.append(result)
    if len(results) != STATE_BENCH_OFFICIAL_TEST_COUNT:
        raise StateBenchLiveContractError(
            "state_bench_live_cell_denominator_incomplete"
        )
    private_case_manifest_sha256 = _write_private_manifest(
        evidence_dir / "raw_case_manifest.json",
        {
            "schema": "papc.state_bench.private_raw_case_manifest.v1",
            "job_id": job.job_id,
            "registry_sha256": registry.test_registry_sha256,
            "cases": [dict(item.private_case_manifest_record) for item in results],
        },
    )
    raw_usage_manifest_sha256 = _write_private_jsonl(
        evidence_dir / "raw_usage_receipt_manifest.jsonl",
        (
            {
                "case_index": case_index,
                "example_sha256": result.example_sha256,
                "receipt_index": receipt_index,
                "receipt": receipt.public_record(),
            }
            for case_index, result in enumerate(results)
            for receipt_index, receipt in enumerate(result.receipts)
        ),
    )
    outcome_cost_manifest = RegistryOrderedOutcomeCostManifest(
        row_id=binding.method,
        surface=STATE_BENCH,
        test_registry_sha256=surface_authority.test_registry_sha256,
        trial_policy_sha256=surface_authority.trial_policy_sha256,
        official_metric=surface_authority.official_metric,
        cases=tuple(
            _case_outcome_cost(case_index, result)
            for case_index, result in enumerate(results)
        ),
        registry_order_matches_test_manifest=True,
    )
    _write_private_manifest(
        evidence_dir / "registry_ordered_outcome_cost_manifest.json",
        outcome_cost_manifest.private_record(),
    )
    receipt = build_state_bench_table1_receipt(
        results=tuple(results),
        authority=surface_authority,
        binding=binding,
        implementation_bundle_sha256=implementation_bundle_sha256,
        run_config_sha256=run_config_sha256,
        private_raw_case_manifest_sha256=(private_case_manifest_sha256),
        raw_usage_receipt_manifest_sha256=(raw_usage_manifest_sha256),
        registry_ordered_outcome_cost_manifest_sha256=(
            outcome_cost_manifest.manifest_sha256
        ),
    )
    outcome_cost_manifest.validate_against(receipt)
    payload = (
        json.dumps(
            receipt.public_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_publish_no_replace(job.output_path, payload)
    return JobExecutionResult(
        completed_item_count=len(results),
        registry_sha256=job.registry_sha256,
        output_payload_sha256=hashlib.sha256(payload).hexdigest(),
        infra_failure_count=0,
        replicates_per_task=job.replicates_per_task,
        trial_policy_sha256=job.trial_policy_sha256,
        authority_manifest_file_sha256=(
            job.authority_manifest_file_sha256
        ),
        authority_manifest_semantic_sha256=(
            job.authority_manifest_semantic_sha256
        ),
        scorer_sha256=job.scorer_sha256,
        runtime_sha256=job.runtime_sha256,
        merged_from_prior_runs=False,
    )


def build_state_bench_table1_receipt(
    *,
    results: tuple[StateBenchTaskResult, ...],
    authority: SurfaceAuthority,
    binding: RouterArtifactBinding,
    implementation_bundle_sha256: str,
    run_config_sha256: str,
    private_raw_case_manifest_sha256: str,
    raw_usage_receipt_manifest_sha256: str,
    registry_ordered_outcome_cost_manifest_sha256: str,
) -> Table1ResultReceipt:
    if len(results) != STATE_BENCH_OFFICIAL_TEST_COUNT:
        raise StateBenchLiveContractError(
            "state_bench_live_cell_denominator_incomplete"
        )
    receipts = tuple(receipt for result in results for receipt in result.receipts)
    provider = tuple(receipt for receipt in receipts if receipt.provider_calls > 0)
    _require_atomic_complete_provider_receipts(provider)
    agent = tuple(receipt for receipt in provider if receipt.role in AGENT_SIDE_ROLES)
    if not agent:
        raise StateBenchLiveContractError(
            "state_bench_live_agent_provider_receipts_empty"
        )
    accounting = ProviderAPIAccounting(
        usage_complete=True,
        provider_calls=sum(item.provider_calls for item in agent),
        retry_count=sum(item.retry_count for item in agent),
        input_tokens=sum(int(item.input_tokens) for item in agent),
        cached_input_tokens=sum(int(item.cached_input_tokens) for item in agent),
        output_tokens=sum(int(item.output_tokens) for item in agent),
        reasoning_tokens=sum(int(item.reasoning_tokens) for item in agent),
        cache_neutral_cost_usd=sum(
            float(item.provider_reported_cost_usd) for item in agent
        ),
    )
    table_receipt = Table1ResultReceipt(
        row_id=binding.method,
        surface=STATE_BENCH,
        result_status="completed",
        aggregation_scope=FULL_AGGREGATION_SCOPE,
        attempted_count=len(results),
        scored_count=len(results),
        infra_error_count=0,
        official_score=sum(item.official_score for item in results) / len(results),
        replicates_per_task=authority.replicates_per_task,
        trial_policy_sha256=authority.trial_policy_sha256,
        test_registry_sha256=authority.test_registry_sha256,
        official_scorer=authority.official_scorer,
        official_metric=authority.official_metric,
        scorer_sha256=authority.scorer_sha256,
        runtime_name=authority.runtime_name,
        runtime_sha256=authority.runtime_sha256,
        cheap_model=binding.cheap_model,
        cheap_reasoning_effort=binding.cheap_reasoning_effort,
        strong_model=binding.strong_model,
        strong_reasoning_effort=binding.strong_reasoning_effort,
        fit_registry_sha256=binding.fit_registry_sha256,
        dev_registry_sha256=binding.dev_registry_sha256,
        method_artifact_sha256=binding.artifact_sha256,
        calibration_sha256=binding.calibration_sha256,
        implementation_bundle_sha256=implementation_bundle_sha256,
        run_config_sha256=run_config_sha256,
        private_raw_case_manifest_sha256=(private_raw_case_manifest_sha256),
        raw_usage_receipt_manifest_sha256=(raw_usage_receipt_manifest_sha256),
        registry_ordered_outcome_cost_manifest_sha256=(
            registry_ordered_outcome_cost_manifest_sha256
        ),
        case_receipt_count=len(results),
        registry_order_matches_test_manifest=True,
        agent_side_provider_api=accounting,
        local_router=LocalRouterReport(
            decision_count=sum(item.decision_count for item in results),
            latency_seconds_total=sum(
                item.local_router_latency_seconds for item in results
            ),
            hardware=platform.machine() or "unknown",
            software=f"papc-state-bench-live:{binding.method}",
        ),
        public_safe=True,
    )
    table_receipt.validate_against(authority)
    return table_receipt


def state_bench_implementation_bundle_sha256(
    method: ExternalMethod,
) -> str:
    baseline_root = Path(__file__).resolve().parent
    method_file = {
        ROUTELLM: "routellm_mf_upstream.py",
        FRUGALGPT: "frugalgpt_upstream.py",
        AUTOMIX: "automix_upstream.py",
        MTROUTER: "mtrouter_upstream.py",
        OPENSQUILLA_G0: "opensquilla_upstream.py",
    }[method]
    return _source_bundle_sha256(
        root=baseline_root,
        paths=tuple(
            baseline_root / name
            for name in (
                "state_bench_live.py",
                "state_bench_native_runtime.py",
                "state_bench_router_driver.py",
                "sealed_router_policy.py",
                "router_accounting.py",
                "router_contract.py",
                "router_policies.py",
                "router_runtime.py",
                method_file,
            )
        ),
    )


def state_bench_run_config_sha256(
    context: StateBenchLiveContext,
) -> str:
    return _digest(
        {
            "schema": "papc.state_bench.external_router_run_config.v1",
            "plan_sha256": context.plan.sha256,
            "job": context.job.public_record(),
            "stage_authorization": {
                "authorization_id": context.authorization.authorization_id,
                "plan_sha256": context.authorization.plan_sha256,
                "stage": context.authorization.stage,
                "provider_attempt_hard_cap": (
                    context.authorization.provider_attempt_hard_cap
                ),
                "observed_spend_stop_usd": str(
                    context.authorization.observed_spend_stop_usd
                ),
            },
            "authority_manifest_sha256": context.authority_manifest.sha256,
            "runtime_environment_sha256": (
                context.runtime_environment_sha256
            ),
            "native_run_config": context.native_config.public_record(),
            "method_artifact": context.typed_artifact.binding.public_record(),
            "runtime_asset_binding": state_bench_runtime_asset_binding(
                typed=context.typed_artifact,
                opensquilla_checkout_root=(
                    context.request.opensquilla_checkout_root
                ),
            ),
            "authority": {
                "replicates_per_task": 1,
                "table1_trial_policy_sha256": (
                    context.surface_authority.trial_policy_sha256
                ),
                "execution_trial_policy_sha256": (
                    context.job.trial_policy_sha256
                ),
                "authority_manifest_file_sha256": (
                    context.plan.authority_manifest_file_sha256
                ),
                "authority_manifest_semantic_sha256": (
                    context.plan.authority_manifest_semantic_sha256
                ),
                "scorer_sha256": context.job.scorer_sha256,
                "runtime_sha256": context.job.runtime_sha256,
            },
        }
    )


def _private_case_manifest_record(
    *,
    example_sha256: str,
    private_root: Path,
    run_dirs: tuple[Path, ...],
) -> dict[str, Any]:
    if not run_dirs:
        raise StateBenchLiveContractError("state_bench_private_case_run_dirs_empty")
    root = Path(private_root).resolve()
    files: list[dict[str, Any]] = []
    for run_index, raw_dir in enumerate(run_dirs):
        run_dir = Path(raw_dir).resolve()
        try:
            relative_dir = run_dir.relative_to(root)
        except ValueError as exc:
            raise StateBenchLiveContractError(
                "state_bench_private_case_dir_outside_root"
            ) from exc
        generated = run_dir / "trajectory_generated.json"
        if not generated.is_file():
            raise FileNotFoundError("state_bench_private_generated_trajectory_missing")
        for path in sorted(item for item in run_dir.iterdir() if item.is_file()):
            files.append(
                {
                    "run_index": run_index,
                    "path": (relative_dir.joinpath(path.name).as_posix()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "example_sha256": example_sha256,
        "run_count": len(run_dirs),
        "private_files": files,
    }


def _validate_private_case_manifest_record(
    record: Mapping[str, Any],
    *,
    private_root: Path,
) -> None:
    root = Path(private_root).resolve()
    for raw_file in _sequence(
        record.get("private_files"),
        "state_bench_private_case_files",
    ):
        file_record = _mapping(
            raw_file,
            "state_bench_private_case_file",
        )
        relative = Path(str(file_record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise StateBenchLiveContractError("state_bench_private_case_path_invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StateBenchLiveContractError(
                "state_bench_private_case_path_outside_root"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError("state_bench_private_case_file_missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != file_record.get("sha256"):
            raise StateBenchLiveContractError(
                "state_bench_private_case_file_sha256_mismatch"
            )
        if path.stat().st_size != file_record.get("size_bytes"):
            raise StateBenchLiveContractError(
                "state_bench_private_case_file_size_mismatch"
            )


def _case_outcome_cost(
    case_index: int,
    result: StateBenchTaskResult,
) -> CaseOutcomeCost:
    provider = tuple(item for item in result.receipts if item.provider_calls > 0)
    agent = tuple(item for item in provider if item.role in AGENT_SIDE_ROLES)
    return CaseOutcomeCost(
        registry_index=case_index,
        case_id_sha256=result.example_sha256,
        official_outcome=result.official_score,
        provider_calls=sum(item.provider_calls for item in agent),
        retry_count=sum(item.retry_count for item in agent),
        input_tokens=sum(int(item.input_tokens) for item in agent),
        cached_input_tokens=sum(int(item.cached_input_tokens) for item in agent),
        output_tokens=sum(int(item.output_tokens) for item in agent),
        reasoning_tokens=sum(int(item.reasoning_tokens) for item in agent),
        cache_neutral_cost_usd=sum(
            float(item.provider_reported_cost_usd) for item in agent
        ),
        private_raw_case_receipt_sha256=_digest(result.private_case_manifest_record),
        raw_usage_receipt_sha256=_digest(
            [item.public_record() for item in result.receipts]
        ),
    )


def _write_private_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes_no_replace(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_private_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]] | Any,
) -> str:
    resolved = tuple(records)
    if not resolved:
        raise StateBenchLiveContractError("state_bench_private_manifest_empty")
    encoded = b"".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in resolved
    )
    _write_bytes_no_replace(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def load_typed_state_bench_artifact(
    method: ExternalMethod,
    artifact_json: Path,
    runtime_asset_path: Path | None,
) -> TypedStateBenchArtifact:
    record = _read_mapping(artifact_json, "state_bench_typed_router_artifact")
    asset = (
        None
        if runtime_asset_path is None
        else Path(runtime_asset_path).resolve()
    )
    if method == ROUTELLM:
        artifact = _load_routellm_artifact(record)
        if asset is None or not asset.is_file():
            raise FileNotFoundError("state_bench_routellm_checkpoint_missing")
    elif method == FRUGALGPT:
        artifact = FrugalGPTScorerArtifact.from_public_record(dict(record))
        if asset is None or not asset.is_dir():
            raise FileNotFoundError("state_bench_frugalgpt_checkpoint_missing")
    elif method == AUTOMIX:
        artifact = _load_automix_artifact(record)
        if asset is None or not asset.is_dir():
            raise FileNotFoundError("state_bench_automix_tokenizer_missing")
    elif method == MTROUTER:
        artifact = load_mtrouter_artifact(Path(artifact_json))
        if artifact.uses_all_strong_shortcut:
            if asset is not None:
                raise StateBenchLiveContractError(
                    "state_bench_mtrouter_all_strong_asset_forbidden"
                )
        elif asset is None or not asset.is_file():
            raise FileNotFoundError("state_bench_mtrouter_checkpoint_missing")
    elif method == OPENSQUILLA_G0:
        artifact = load_opensquilla_runtime_artifact(artifact_json)
        if asset is None or not asset.is_dir():
            raise FileNotFoundError(
                "state_bench_opensquilla_adapted_bundle_missing"
            )
    else:
        raise StateBenchLiveContractError("state_bench_live_method_invalid")
    typed = TypedStateBenchArtifact(
        method=method,
        artifact=artifact,
        runtime_asset_path=asset,
    )
    if typed.binding.method != method or typed.binding.surface != STATE_BENCH:
        raise StateBenchLiveContractError(
            "state_bench_typed_artifact_method_or_surface_mismatch"
        )
    return typed


def state_bench_runtime_asset_binding(
    *,
    typed: TypedStateBenchArtifact,
    opensquilla_checkout_root: Path | None,
) -> dict[str, Any]:
    """Bind every local runtime asset, including both OpenSquilla roots."""

    asset = typed.runtime_asset_path
    if typed.method == MTROUTER:
        artifact = typed.artifact
        if not isinstance(artifact, MTRouterArtifact):
            raise TypeError("state_bench_mtrouter_typed_artifact_required")
        if artifact.uses_all_strong_shortcut:
            if asset is not None:
                raise StateBenchLiveContractError(
                    "state_bench_mtrouter_all_strong_asset_forbidden"
                )
            return {
                "kind": "mtrouter_all_strong_native_shortcut",
                "execution_mode": artifact.execution_mode,
                "checkpoint_sha256": None,
                "dev_selection_sha256": artifact.dev_selection.sha256,
                "model_names": list(artifact.model_names),
            }
    if typed.method in {ROUTELLM, MTROUTER}:
        if asset is None or not asset.is_file():
            raise FileNotFoundError("state_bench_runtime_checkpoint_missing")
        return {
            "kind": "checkpoint_file",
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        }
    if typed.method in {FRUGALGPT, AUTOMIX}:
        if asset is None:
            raise FileNotFoundError("state_bench_runtime_directory_missing")
        return {
            "kind": "runtime_directory",
            "source_sha256": _directory_source_sha256(asset),
        }
    if typed.method == OPENSQUILLA_G0:
        if asset is None:
            raise FileNotFoundError(
                "state_bench_opensquilla_adapted_bundle_missing"
            )
        checkout = _required_directory(
            opensquilla_checkout_root,
            "state_bench_opensquilla_checkout",
        )
        if checkout == asset or checkout in asset.parents or asset in checkout.parents:
            raise StateBenchLiveContractError(
                "state_bench_opensquilla_resources_not_independent"
            )
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        checkout_revision = completed.stdout.strip()
        if checkout_revision != OPENSQUILLA_UPSTREAM_REVISION:
            raise StateBenchLiveContractError(
                "state_bench_opensquilla_checkout_revision_mismatch"
            )
        strategy_path = (
            checkout
            / "src/opensquilla/squilla_router/v4_phase3.py"
        )
        strategy_sha256 = hashlib.sha256(
            strategy_path.read_bytes()
        ).hexdigest()
        expected_strategy_sha256 = OPENSQUILLA_TRAINING_SOURCE_SHA256S[
            "src/opensquilla/squilla_router/v4_phase3.py"
        ]
        if strategy_sha256 != expected_strategy_sha256:
            raise StateBenchLiveContractError(
                "state_bench_opensquilla_checkout_source_mismatch"
            )
        adapted_bundle_sha256 = opensquilla_bundle_tree_sha256(asset)
        if adapted_bundle_sha256 != typed.artifact.bundle_sha256:
            raise StateBenchLiveContractError(
                "state_bench_opensquilla_adapted_bundle_mismatch"
            )
        return {
            "kind": "opensquilla_adapted_runtime",
            "checkout_path_label": checkout.name,
            "checkout_revision": checkout_revision,
            "checkout_source_sha256": _directory_source_sha256(checkout),
            "checkout_strategy_sha256": strategy_sha256,
            "fit_adapted_bundle_path_label": asset.name,
            "fit_adapted_bundle_tree_sha256": adapted_bundle_sha256,
            "artifact_bundle_sha256": typed.artifact.bundle_sha256,
            "checkout_and_fit_adapted_bundle_are_distinct_paths": True,
            "cold_start_checkout_bundle_used": False,
        }
    raise StateBenchLiveContractError("state_bench_live_method_invalid")


def state_bench_source_authority(
    api: StateBenchNativeAPI,
    registry: StateBenchOfficialTestRegistry,
) -> tuple[str, str]:
    """Recompute the exact STATE source bundles used by authority v2."""

    root = api.runtime_repo.resolve()
    if api.runtime_revision != FROZEN_STATE_BENCH_RUNTIME_REVISION:
        raise StateBenchLiveContractError("state_bench_live_runtime_revision_mismatch")
    runtime_paths = {
        *root.joinpath("state_bench").glob("**/*.py"),
        *root.joinpath("state_bench", "configs").glob("**/*.yaml"),
        *root.joinpath("state_bench", "configs").glob("**/*.json"),
    }
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "agents/papc_openai_compatible.py",
        "clients/papc_openai_compatible.py",
    ):
        path = root / relative
        if path.is_file():
            runtime_paths.add(path)
    source_tree_sha256 = _source_bundle_sha256(
        root=root,
        paths=tuple(runtime_paths),
    )
    runtime_sha256 = _digest(
        {
            "schema": "papc.table1.native_runtime_source_bundle.v1",
            "surface": STATE_BENCH,
            "revision": api.runtime_revision,
            "source_binding_sha256": registry.source_binding_sha256,
            "source_tree_sha256": source_tree_sha256,
        }
    )
    scorer_sha256 = _source_bundle_sha256(
        root=root,
        paths=(
            root / "state_bench" / "scoring.py",
            root / "state_bench" / "scripts" / "score.py",
            root / "state_bench" / "protocol.py",
            root / "state_bench" / "configs" / "eval_protocols" / "gpt54.json",
        ),
    )
    return runtime_sha256, scorer_sha256


def _validate_state_authority(
    *,
    surface_authority: SurfaceAuthority,
    registry: StateBenchOfficialTestRegistry,
    runtime_sha256: str,
    scorer_sha256: str,
) -> None:
    expected_trial = trial_policy_sha256(
        surface=STATE_BENCH,
        test_registry_sha256=registry.test_registry_sha256,
    )
    if (
        surface_authority.surface != STATE_BENCH
        or surface_authority.denominator != STATE_BENCH_OFFICIAL_TEST_COUNT
        or surface_authority.replicates_per_task != 1
        or surface_authority.test_registry_sha256 != registry.test_registry_sha256
        or surface_authority.trial_policy_sha256 != expected_trial
        or surface_authority.official_scorer != STATE_BENCH_OFFICIAL_SCORER
        or surface_authority.official_metric != STATE_BENCH_OFFICIAL_METRIC
        or surface_authority.runtime_name != STATE_BENCH_RUNTIME_NAME
        or surface_authority.runtime_sha256 != runtime_sha256
        or surface_authority.scorer_sha256 != scorer_sha256
    ):
        raise StateBenchLiveContractError("state_bench_live_authority_source_mismatch")


def _load_routellm_artifact(
    record: Mapping[str, Any],
) -> RouteLLMMFArtifact:
    calibration_record = _mapping(
        record.get("dev_calibration"),
        "state_bench_routellm_dev_calibration",
    )
    calibration = RouteLLMMFDevCalibration(
        **_constructor_values(
            RouteLLMMFDevCalibration,
            calibration_record,
            replacements={
                "candidate_receipts": tuple(
                    RouteLLMMFDevCandidateReceipt(
                        **_constructor_values(
                            RouteLLMMFDevCandidateReceipt,
                            _mapping(
                                item,
                                "state_bench_routellm_dev_candidate",
                            ),
                        )
                    )
                    for item in _sequence(
                        calibration_record.get("candidate_receipts"),
                        "state_bench_routellm_dev_candidates",
                    )
                )
            },
        )
    )
    class_counts = _mapping(
        record.get("class_counts"),
        "state_bench_routellm_class_counts",
    )
    artifact = RouteLLMMFArtifact(
        **_constructor_values(
            RouteLLMMFArtifact,
            record,
            replacements={
                "class_counts": (
                    ("cheap_wins", class_counts["cheap_wins"]),
                    ("strong_wins", class_counts["strong_wins"]),
                ),
                "dev_calibration": calibration,
            },
        )
    )
    if artifact.public_record() != dict(record):
        raise ValueError("state_bench_routellm_artifact_not_canonical")
    return artifact


def _load_automix_artifact(
    record: Mapping[str, Any],
) -> AutoMixRuntimeArtifact:
    policy_record = _mapping(
        record.get("policy"),
        "state_bench_automix_policy",
    )
    candidates = tuple(
        AutoMixDevCandidateReceipt(
            **_constructor_values(
                AutoMixDevCandidateReceipt,
                _mapping(item, "state_bench_automix_candidate"),
            )
        )
        for item in _sequence(
            policy_record.get("candidate_receipts"),
            "state_bench_automix_candidates",
        )
    )
    policy = AutoMixPOMDPPolicy(
        **_constructor_values(
            AutoMixPOMDPPolicy,
            policy_record,
            replacements={"candidate_receipts": candidates},
        )
    )
    artifact = AutoMixRuntimeArtifact(
        **_constructor_values(
            AutoMixRuntimeArtifact,
            record,
            replacements={"policy": policy},
        )
    )
    if artifact.public_record() != dict(record):
        raise ValueError("state_bench_automix_artifact_not_canonical")
    return artifact


def _constructor_values(
    cls: type,
    record: Mapping[str, Any],
    *,
    replacements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    replacements = replacements or {}
    values: dict[str, Any] = {}
    for item in fields(cls):
        if not item.init:
            continue
        if item.name in replacements:
            values[item.name] = replacements[item.name]
        elif item.name in record:
            values[item.name] = record[item.name]
    return values


def _openai_raw_client(api_base: str, api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        max_retries=0,
    )


def _load_automix_tokenizer(path: Path) -> Any:
    validate_automix_verifier_tokenizer_root(path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "state_bench_automix_transformers_dependency_missing"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    configured = str(getattr(tokenizer, "name_or_path", ""))
    if not configured:
        raise ValueError("state_bench_automix_tokenizer_identity_missing")
    return tokenizer


def _require_atomic_complete_provider_receipts(
    receipts: Sequence[CallReceipt],
) -> None:
    for receipt in receipts:
        if receipt.provider_calls == 0:
            continue
        if (
            receipt.provider_calls != 1
            or receipt.retry_count != 0
            or receipt.input_tokens is None
            or receipt.cached_input_tokens is None
            or receipt.output_tokens is None
            or receipt.reasoning_tokens is None
            or receipt.provider_reported_cost_usd is None
        ):
            raise StateBenchLiveContractError(
                "state_bench_live_provider_receipt_not_atomic_complete"
            )


def _atomic_publish_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        with os.fdopen(
            os.open(temporary, flags, 0o600),
            "wb",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_bytes_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(
        os.open(target, flags, 0o600),
        "wb",
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _source_bundle_sha256(
    *,
    root: Path,
    paths: Sequence[Path],
) -> str:
    source_root = Path(root).resolve()
    manifest: list[dict[str, str]] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"state_bench_live_source_file_missing:{path}")
        try:
            label = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError("state_bench_live_source_path_outside_root") from exc
        manifest.append(
            {
                "path": label,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not manifest:
        raise ValueError("state_bench_live_source_bundle_empty")
    return _digest(manifest)


def _required_directory(
    path: Path | None,
    field_name: str,
) -> Path:
    if path is None:
        raise FileNotFoundError(f"{field_name}_missing")
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{field_name}_missing")
    return resolved


def _directory_source_sha256(root: Path) -> str:
    source_root = _required_directory(root, "state_bench_runtime_directory")
    paths = tuple(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    )
    return _source_bundle_sha256(root=source_root, paths=paths)


def _read_mapping(path: Path, field_name: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(_mapping(payload, field_name))


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name}_not_mapping")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TypeError(f"{field_name}_not_sequence")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

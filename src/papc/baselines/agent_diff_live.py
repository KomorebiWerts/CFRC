"""Authorized whole-cell Agent-Diff execution for Table 1 routers.

The module is inert until the caller supplies a source-bound execution plan,
the frozen Table 1 authority, an exact stage authorization, and a sealed
method artifact.  Provider credentials are supplied by the CLI only after
``prepare_agent_diff_live_context`` succeeds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from papc.baselines.agent_diff_router_driver import (
    AGENT_DIFF_EXECUTED_METHODS,
    AGENT_DIFF_OFFICIAL_TEST_COUNT,
    FROZEN_AGENT_DIFF_REPO_REVISION,
    AgentDiffNativeAPI,
    AgentDiffNativeRunConfig,
    AgentDiffOfficialRegistry,
    AgentDiffOfficialTaskRef,
    AgentDiffRouterDriver,
    load_agent_diff_native_api,
    load_agent_diff_official_registry,
)
from papc.baselines.external_router_execution import (
    AGENT_DIFF,
    OFFICIAL_REPLICATES_PER_TASK,
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
from papc.baselines.routellm_mf_upstream import (
    RouteLLMMFArtifact,
    RouteLLMMFDevCalibration,
    RouteLLMMFDevCandidateReceipt,
    RouteLLMMFEmbeddingRequest,
    RouteLLMMFEmbeddingResult,
)
from papc.baselines.router_accounting import (
    AGENT_SIDE_ROLES,
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
)
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    FRUGALGPT,
    MTROUTER,
    REASONING_EFFORT,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    STRONG_MODEL,
    ExternalMethod,
    FullEpisodeBinding,
    RouterArtifactBinding,
)
from papc.baselines.router_runtime import (
    GeneratedEpisode,
    HistoryTurn,
    run_history_routed_episode,
)
from papc.baselines.router_runtime_environment import (
    validate_router_runtime_environment,
)
from papc.baselines.sealed_router_policy import (
    SealedRouterPolicy,
    load_frugalgpt_sealed_policy,
    load_mtrouter_sealed_policy,
    load_routellm_sealed_policy,
)
from papc.baselines.surface_adapters import (
    build_agent_diff_full_episode_binding,
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
    Table1AuthorityManifest,
    Table1ResultReceipt,
    load_table1_authority,
)

AGENT_DIFF_LIVE_SCHEMA = "papc.agent_diff.external_router_live.v1"
AGENT_DIFF_AUTHORITY_MANIFEST_SHA256 = (
    "a23f0614f71270ed6b9870880389817d48e6cdd82d5de82b4dfcbb7e20acb101"
)
AGENT_DIFF_OFFICIAL_METRIC = "passed"
AGENT_DIFF_RUNTIME_NAME = "agent_diff.react_bash_executor_proxy"
AGENT_DIFF_MAX_ITERATIONS = 40
AGENT_DIFF_MAX_MODEL_TOKENS = 1600
AGENT_DIFF_TEMPERATURE = 0.0
AGENT_DIFF_PROVIDER_TIMEOUT_SECONDS = 120.0
AGENT_DIFF_COMMAND_TIMEOUT_SECONDS = 30
AGENT_DIFF_OBSERVATION_CHARS = 12_000
AGENT_DIFF_ENV_TTL_SECONDS = 7_200
AGENT_DIFF_PROMPT_PROFILE = "baseline"
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL | re.IGNORECASE)
_SENSITIVE_EXECUTOR_ENV = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "AGENT_DIFF_API_KEY",
    }
)

_SERVICE_CONFIG = {
    "slack": {
        "name": "Slack",
        "base_url": "https://slack.com/api",
        "extra": "",
    },
    "box": {
        "name": "Box",
        "base_url": "https://api.box.com/2.0",
        "extra": "",
    },
    "calendar": {
        "name": "Google Calendar",
        "base_url": "https://www.googleapis.com/calendar/v3",
        "extra": (
            "- **Current Date/Time**: Sunday, June 17, 2018 at 00:01 "
            "(midnight), timezone America/Los_Angeles.\n"
        ),
    },
    "linear": {
        "name": "Linear",
        "base_url": "https://api.linear.app/graphql",
        "extra": "",
    },
}

_REACT_SYSTEM_PROMPT = """You are an AI assistant that completes tasks by interacting with APIs via bash commands.

## Current Session
- **Service**: {service_name}
- **Base URL**: {base_url}
{extra_context}
## Environment
- Authentication is handled automatically via proxy. Use placeholder tokens where credentials would go.
- You execute bash commands, primarily curl, to interact with the API.
- If you are not sure how to use the API, explore the endpoint, parameters, and learn how it works.

## Response Format
Respond using XML tags:

<thinking>Your reasoning</thinking>
<action>Your bash command</action>

When done:
<thinking>Your reasoning</thinking>
<done>Brief summary</done>

## Rules
1. Execute ONE command at a time, then wait for the result.
2. Parse API responses carefully. Extract IDs and data needed for subsequent calls.
3. If a command fails, analyze the error and try a different approach.
4. Only use <done> when the task is fully completed.
"""


class AgentDiffLiveContractError(ValueError):
    """A provider-zero authority, source, or result binding is invalid."""


@dataclass(frozen=True)
class AgentDiffLiveRequest:
    method: ExternalMethod
    typed_router_artifact_json: Path
    runtime_asset_path: Path | None
    agent_diff_repo: Path
    agent_diff_dataset: Path
    agent_diff_sdk_path: Path
    scorer_path: Path
    baseline_authority_path: Path
    split_gate_json: Path
    budget_gate_json: Path
    authority_manifest_json: Path
    authority_manifest_sha256: str
    private_root: Path
    api_base: str
    api_key_env: str
    agent_diff_base_url: str
    authorization_id: str
    plan_sha256: str
    stage_provider_attempt_hard_cap: int
    stage_observed_spend_stop_usd: Decimal
    job_id: str
    mtrouter_device: str = "cuda"


@dataclass(frozen=True)
class AgentDiffTypedArtifact:
    method: ExternalMethod
    artifact: RouteLLMMFArtifact | FrugalGPTScorerArtifact | MTRouterArtifact
    binding: RouterArtifactBinding
    runtime_asset_path: Path | None

    def __post_init__(self) -> None:
        if self.method not in AGENT_DIFF_EXECUTED_METHODS:
            raise AgentDiffLiveContractError(
                "agent_diff_live_method_not_in_publication_subset"
            )
        if not isinstance(self.binding, RouterArtifactBinding):
            raise TypeError("agent_diff_live_router_binding_invalid")
        if (
            self.binding.method != self.method
            or self.binding.surface != AGENT_DIFF
        ):
            raise AgentDiffLiveContractError(
                "agent_diff_live_typed_artifact_method_or_surface_mismatch"
            )
        asset = (
            None
            if self.runtime_asset_path is None
            else Path(self.runtime_asset_path).resolve()
        )
        if self.method == MTROUTER:
            if not isinstance(self.artifact, MTRouterArtifact):
                raise TypeError(
                    "agent_diff_live_mtrouter_typed_artifact_required"
                )
            if self.artifact.uses_all_strong_shortcut:
                if asset is not None:
                    raise AgentDiffLiveContractError(
                        "agent_diff_live_mtrouter_all_strong_asset_forbidden"
                    )
            elif asset is None:
                raise FileNotFoundError(
                    "agent_diff_live_mtrouter_checkpoint_missing"
                )
        elif asset is None:
            raise FileNotFoundError(
                "agent_diff_live_runtime_asset_missing"
            )
        object.__setattr__(self, "runtime_asset_path", asset)


@dataclass(frozen=True)
class AgentDiffLiveContext:
    request: AgentDiffLiveRequest
    plan: ExecutionPlan
    job: ExecutionJob
    authorization: StageExecutionAuthorization
    authority_manifest: Table1AuthorityManifest
    authority_manifest_file_sha256: str
    surface_authority: SurfaceAuthority
    registry: AgentDiffOfficialRegistry
    native_api: AgentDiffNativeAPI
    runtime_environment_sha256: str
    typed_artifact: AgentDiffTypedArtifact
    native_config: AgentDiffNativeRunConfig

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": AGENT_DIFF_LIVE_SCHEMA,
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
            "authority_manifest_file_sha256": (
                self.authority_manifest_file_sha256
            ),
            "test_registry_sha256": self.registry.split.test_registry_sha256,
            "source_binding_sha256": self.registry.split.source_binding_sha256,
            "table1_trial_policy_sha256": (
                self.surface_authority.trial_policy_sha256
            ),
            "execution_trial_policy_sha256": self.job.trial_policy_sha256,
            "runtime_sha256": self.surface_authority.runtime_sha256,
            "scorer_sha256": self.surface_authority.scorer_sha256,
            "runtime_environment_sha256": self.runtime_environment_sha256,
            "method_artifact_sha256": self.typed_artifact.binding.artifact_sha256,
            "output_target": self.job.relative_output_path,
            "official_test_count": self.registry.denominator,
            "replicates_per_task": self.surface_authority.replicates_per_task,
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "ready_for_explicit_execute": True,
        }


@dataclass(frozen=True)
class AgentDiffPrivateTask:
    ref: AgentDiffOfficialTaskRef
    test_id: str = field(repr=False)
    test_name: str = field(repr=False)
    answer: Mapping[str, Any] = field(repr=False)
    info: Mapping[str, Any] = field(repr=False)
    metadata: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentDiffOfficialTaskRef):
            raise TypeError("agent_diff_live_private_task_ref_invalid")
        if self.test_id != self.ref.native_test_id:
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_test_id_mismatch"
            )
        for name, value in (
            ("answer", self.answer),
            ("info", self.info),
            ("metadata", self.metadata),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"agent_diff_live_private_{name}_invalid")
            object.__setattr__(self, name, dict(value))

    @property
    def initial_environment_sha256(self) -> str:
        return _digest(
            {
                "service": self.info.get("service"),
                "seed_template": self.info.get("seed_template"),
                "impersonate_user_id": self.info.get("impersonate_user_id"),
            }
        )


@dataclass(frozen=True)
class AgentDiffTaskResult:
    example_sha256: str
    official_score: float
    decision_count: int
    local_router_latency_seconds: float
    receipts: tuple[CallReceipt, ...]
    private_case_manifest_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "agent_diff_live_task_example")
        if self.official_score not in {0.0, 1.0}:
            raise ValueError("agent_diff_live_task_score_not_binary")
        if (
            isinstance(self.decision_count, bool)
            or not isinstance(self.decision_count, int)
            or self.decision_count <= 0
        ):
            raise ValueError("agent_diff_live_task_decision_count_invalid")
        if (
            isinstance(self.local_router_latency_seconds, bool)
            or not isinstance(
                self.local_router_latency_seconds,
                (int, float),
            )
            or float(self.local_router_latency_seconds) < 0.0
        ):
            raise ValueError("agent_diff_live_task_router_latency_invalid")
        receipts = tuple(self.receipts)
        if not receipts or any(not isinstance(item, CallReceipt) for item in receipts):
            raise TypeError("agent_diff_live_task_receipts_invalid")
        _require_complete_provider_receipts(receipts)
        record = _mapping(
            self.private_case_manifest_record,
            "agent_diff_live_private_case_manifest_record",
        )
        if record.get("example_sha256") != self.example_sha256:
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_identity_mismatch"
            )
        if not _sequence(
            record.get("private_files"),
            "agent_diff_live_private_case_files",
        ):
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_files_empty"
            )
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "private_case_manifest_record", dict(record))


@dataclass(frozen=True)
class AgentDiffCellExecution:
    result: JobExecutionResult
    public_receipt: Table1ResultReceipt
    evidence_dir: Path


@dataclass
class _NativeSession:
    task: AgentDiffPrivateTask
    api: AgentDiffNativeAPI
    config: AgentDiffNativeRunConfig
    provider_client: Any
    broker: ProviderAttemptBroker
    raw_path: Path
    api_key_env: str
    client: Any = field(init=False, default=None, repr=False)
    environment: Any = field(init=False, default=None, repr=False)
    run: Any = field(init=False, default=None, repr=False)
    bash: Any = field(init=False, default=None, repr=False)
    receipts: list[CallReceipt] = field(init=False, default_factory=list)
    trace: list[dict[str, Any]] = field(init=False, default_factory=list)
    evaluation: dict[str, Any] | None = field(init=False, default=None)
    completion_reason: str | None = field(init=False, default=None)
    cleanup_status: str | None = field(init=False, default=None)
    error_record: dict[str, Any] | None = field(init=False, default=None)

    def start(self) -> None:
        if self.client is not None:
            raise RuntimeError("agent_diff_native_session_already_started")
        # A truthy whitespace value prevents the upstream constructor from
        # consulting AGENT_DIFF_API_KEY while normalizing to no platform key.
        self.client = self.api.agent_diff_class(
            api_key=" ",
            base_url=self.config.agent_diff_base_url,
        )
        self.environment = self.client.init_env(
            templateService=self.task.info["service"],
            templateName=self.task.info["seed_template"],
            impersonateUserId=self.task.info["impersonate_user_id"],
            ttlSeconds=AGENT_DIFF_ENV_TTL_SECONDS,
        )
        self.run = self.client.start_run(
            envId=self.environment.environmentId,
        )
        self.bash = self.api.bash_executor_class(
            self.environment.environmentId,
            base_url=self.client.base_url,
            # Keep this truthy until the upstream proxy has performed its own
            # normalization; passing the client's normalized ``None`` would
            # make BaseExecutorProxy consult AGENT_DIFF_API_KEY again.
            api_key=" ",
            workspace_dir=str(self.raw_path.parent / "workspaces"),
        )

    @property
    def environment_instance_sha256(self) -> str:
        if self.environment is None or self.run is None:
            raise RuntimeError("agent_diff_native_session_not_started")
        return _digest(
            {
                "environment_id": str(self.environment.environmentId),
                "run_id": str(self.run.runId),
            }
        )

    def model_turn(
        self,
        *,
        model: str,
        public_history: Sequence[Mapping[str, Any]],
    ) -> tuple[str, CallReceipt]:
        if model not in {CHEAP_MODEL, STRONG_MODEL}:
            raise ValueError("agent_diff_native_turn_model_not_frozen")
        role = "cheap_agent" if model == CHEAP_MODEL else "strong_agent"
        response_box: dict[str, Any] = {}

        def raw_attempt() -> CallReceipt:
            started = time.perf_counter()
            response = self.provider_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": _system_prompt(self.task.ref.service),
                    },
                    *[dict(item) for item in public_history],
                ],
                reasoning_effort=REASONING_EFFORT,
                max_tokens=AGENT_DIFF_MAX_MODEL_TOKENS,
                temperature=AGENT_DIFF_TEMPERATURE,
            )
            receipt = _provider_response_receipt(
                response=response,
                role=role,
                phase="agent_turn",
                model=model,
                reasoning_effort=REASONING_EFFORT,
                latency_seconds=max(0.0, time.perf_counter() - started),
            )
            _require_complete_provider_receipt(receipt)
            response_box["response"] = response
            response_box["text"] = _chat_message_text(response)
            return receipt

        receipt = self.broker.perform(raw_attempt)
        if "response" not in response_box:
            raise RuntimeError("agent_diff_provider_response_missing_after_settle")
        self.receipts.append(receipt)
        text = str(response_box["text"])
        if not text.strip():
            raise ValueError("agent_diff_provider_message_empty")
        return text, receipt

    def execute_action(self, action: str) -> dict[str, Any]:
        if self.bash is None:
            raise RuntimeError("agent_diff_native_bash_not_started")
        started = time.perf_counter()
        with _sanitized_executor_environment(self.api_key_env):
            result = self.bash.execute(
                action,
                timeout=AGENT_DIFF_COMMAND_TIMEOUT_SECONDS,
            )
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        exit_code = int(result.get("exit_code") or 0)
        observation = stdout.strip() or "(empty output)"
        if exit_code != 0:
            observation = f"{stdout}\n[stderr]: {stderr}".strip()
        return {
            "action": action,
            "tool_exit_code": exit_code,
            "tool_status": result.get("status"),
            "tool_latency_seconds": round(
                max(0.0, time.perf_counter() - started),
                6,
            ),
            "observation": observation[:AGENT_DIFF_OBSERVATION_CHARS],
            "workspace_path": result.get("workspace_path"),
        }

    def evaluate(self) -> dict[str, Any]:
        if self.evaluation is not None:
            return self.evaluation
        if self.client is None or self.run is None:
            raise RuntimeError("agent_diff_native_session_not_started")
        self.client.evaluate_run(
            runId=self.run.runId,
            expectedOutput=dict(self.task.answer),
        )
        result = self.client.get_results_for_run(runId=self.run.runId)
        corrected = self.api.apply_scorer_corrections(
            test_id=self.task.test_id,
            evaluation={
                "passed": bool(result.passed),
                "score": _jsonable(result.score),
                "failures": list(result.failures),
                "diff": _jsonable(result.diff),
                "result_status": result.status,
            },
        )
        if not isinstance(corrected, dict) or not isinstance(
            corrected.get("passed"),
            bool,
        ):
            raise ValueError("agent_diff_corrected_evaluation_invalid")
        self.evaluation = corrected
        return corrected

    def finalize(self, *, error: BaseException | None = None) -> Path:
        if error is not None:
            self.error_record = _safe_exception_record(error)
        cleanup_error: BaseException | None = None
        if self.client is not None and self.environment is not None:
            try:
                self.client.delete_env(
                    envId=self.environment.environmentId,
                )
                self.cleanup_status = "deleted"
            except BaseException as exc:  # noqa: BLE001 - preserve diagnostics.
                cleanup_error = exc
                self.cleanup_status = "delete_failed"
        payload = {
            "schema": "papc.agent_diff.external_router_private_episode.v1",
            "ordinal": self.task.ref.native_ordinal,
            "test_id": self.task.test_id,
            "test_name": self.task.test_name,
            "service": self.task.ref.service,
            "example_sha256": self.task.ref.example_sha256,
            "payload_sha256": self.task.ref.payload_sha256,
            "metadata": dict(self.task.metadata),
            "completion_reason": self.completion_reason,
            "trace": self.trace,
            "evaluation": self.evaluation,
            "environment_id": (
                str(self.environment.environmentId)
                if self.environment is not None
                else None
            ),
            "run_id": str(self.run.runId) if self.run is not None else None,
            "cleanup_status": self.cleanup_status,
            "cleanup_error": (
                _safe_exception_record(cleanup_error)
                if cleanup_error is not None
                else None
            ),
            "error": self.error_record,
            "receipts": [item.public_record() for item in self.receipts],
        }
        _write_json_no_replace(self.raw_path, payload)
        if cleanup_error is not None and error is None:
            raise RuntimeError("agent_diff_native_environment_cleanup_failed") from (
                cleanup_error
            )
        return self.raw_path


def prepare_agent_diff_live_context(
    request: AgentDiffLiveRequest,
    *,
    plan_loader: Callable[..., ExecutionPlan] = load_execution_plan,
    registry_loader: Callable[..., AgentDiffOfficialRegistry] = (
        load_agent_diff_official_registry
    ),
    native_api_loader: Callable[..., AgentDiffNativeAPI] = (
        load_agent_diff_native_api
    ),
    typed_artifact_loader: Callable[
        [ExternalMethod, Path, Path],
        AgentDiffTypedArtifact,
    ] = None,  # type: ignore[assignment]
    source_authority_loader: Callable[
        [Path, AgentDiffOfficialRegistry, Path],
        tuple[str, str],
    ] = None,  # type: ignore[assignment]
    runtime_environment_validator: Callable[
        [],
        str,
    ] = validate_router_runtime_environment,
) -> AgentDiffLiveContext:
    """Validate every provider-zero boundary before any key or environment."""

    if not isinstance(request, AgentDiffLiveRequest):
        raise TypeError("agent_diff_live_request_invalid")
    artifact_loader = typed_artifact_loader or load_agent_diff_typed_artifact
    authority_loader = source_authority_loader or agent_diff_source_authority
    plan = plan_loader(
        split_gate_json=request.split_gate_json,
        budget_gate_json=request.budget_gate_json,
        authority_manifest_json=request.authority_manifest_json,
        private_root=request.private_root,
    )
    if request.plan_sha256 != plan.sha256:
        raise AgentDiffLiveContractError("agent_diff_live_plan_sha256_mismatch")
    authorization = StageExecutionAuthorization(
        authorization_id=request.authorization_id,
        plan_sha256=request.plan_sha256,
        stage=TEST_STAGE,
        provider_attempt_hard_cap=request.stage_provider_attempt_hard_cap,
        observed_spend_stop_usd=request.stage_observed_spend_stop_usd,
    )
    BatchCoordinator(plan=plan, authorization=authorization)
    try:
        job = plan.job(request.job_id)
    except KeyError as exc:
        raise AgentDiffLiveContractError(
            "agent_diff_live_explicit_job_unknown"
        ) from exc
    if (
        job.stage != TEST_STAGE
        or job.kind != "official_test_cell"
        or job.surface != AGENT_DIFF
        or job.method != request.method
        or job.item_count != AGENT_DIFF_OFFICIAL_TEST_COUNT
        or job.replicates_per_task != OFFICIAL_REPLICATES_PER_TASK
        or not job.requires_router_artifact
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_job_not_exact_official_cell"
        )
    if job.output_path.exists():
        raise FileExistsError("agent_diff_live_output_exists_no_overwrite")

    authority_path = Path(request.authority_manifest_json).resolve()
    authority_manifest = load_table1_authority(authority_path)
    authority_file_sha256 = _file_sha256(authority_path)
    if (
        request.authority_manifest_sha256 != authority_manifest.sha256
        or authority_manifest.sha256 != AGENT_DIFF_AUTHORITY_MANIFEST_SHA256
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_authority_manifest_sha256_mismatch"
        )
    surface_authority = authority_manifest.for_surface(AGENT_DIFF)
    registry = registry_loader(
        request.agent_diff_dataset,
        split_gate_json=request.split_gate_json,
    )

    # This check intentionally precedes SDK loading, artifact loading, key
    # access, provider-client construction, and benchmark environment startup.
    if job.trial_policy_sha256 != surface_authority.trial_policy_sha256:
        raise AgentDiffLiveContractError(
            "agent_diff_live_execution_authority_trial_policy_mismatch"
        )
    _validate_plan_registry_authority(
        plan=plan,
        job=job,
        registry=registry,
        authority=surface_authority,
        authority_manifest=authority_manifest,
        authority_manifest_file_sha256=authority_file_sha256,
    )
    runtime_environment_sha256 = _require_sha256(
        runtime_environment_validator(),
        "agent_diff_live_runtime_environment_sha256",
    )

    native_api = native_api_loader(
        sdk_path=request.agent_diff_sdk_path,
        scorer_path=request.scorer_path,
        baseline_authority_path=request.baseline_authority_path,
    )
    runtime_sha256, scorer_sha256 = authority_loader(
        request.agent_diff_repo,
        registry,
        request.scorer_path,
    )
    _validate_surface_authority(
        authority=surface_authority,
        registry=registry,
        runtime_sha256=runtime_sha256,
        scorer_sha256=scorer_sha256,
    )
    typed_artifact = artifact_loader(
        request.method,
        request.typed_router_artifact_json,
        request.runtime_asset_path,
    )
    binding = typed_artifact.binding
    if (
        binding.fit_registry_sha256 != surface_authority.fit_registry_sha256
        or binding.dev_registry_sha256 != surface_authority.dev_registry_sha256
        or binding.test_registry_sha256 != registry.split.test_registry_sha256
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_typed_artifact_authority_mismatch"
        )
    if request.api_key_env != "OPENAI_API_KEY":
        raise AgentDiffLiveContractError(
            "agent_diff_live_api_key_env_not_frozen"
        )
    native_config = AgentDiffNativeRunConfig(
        api_base=request.api_base,
        agent_diff_base_url=request.agent_diff_base_url,
        private_output_root=Path(request.private_root).resolve(),
        max_provider_calls=job.provider_attempt_hard_cap,
        max_observed_provider_cost_usd=float(
            authorization.observed_spend_stop_usd
        ),
        max_iterations=AGENT_DIFF_MAX_ITERATIONS,
        max_model_tokens=AGENT_DIFF_MAX_MODEL_TOKENS,
    )
    return AgentDiffLiveContext(
        request=request,
        plan=plan,
        job=job,
        authorization=authorization,
        authority_manifest=authority_manifest,
        authority_manifest_file_sha256=authority_file_sha256,
        surface_authority=surface_authority,
        registry=registry,
        native_api=native_api,
        runtime_environment_sha256=runtime_environment_sha256,
        typed_artifact=typed_artifact,
        native_config=native_config,
    )


def execute_agent_diff_live_context(
    context: AgentDiffLiveContext,
    *,
    api_key: str,
    provider_client: Any,
) -> tuple[AgentDiffCellExecution, dict[str, Any]]:
    """Execute exactly one authorized complete official-test cell."""

    if not isinstance(context, AgentDiffLiveContext):
        raise TypeError("agent_diff_live_context_invalid")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("agent_diff_live_api_key_missing")
    if provider_client is None:
        raise TypeError("agent_diff_live_provider_client_missing")
    coordinator = BatchCoordinator(
        plan=context.plan,
        authorization=context.authorization,
    )
    completed: list[AgentDiffCellExecution] = []

    def execute_job(
        job: ExecutionJob,
        broker: ProviderAttemptBroker,
    ) -> JobExecutionResult:
        sealed_policy = load_agent_diff_sealed_policy(
            typed=context.typed_artifact,
            provider_client=provider_client,
            broker=broker,
            device=context.request.mtrouter_device,
        )
        driver = AgentDiffRouterDriver(
            policy=sealed_policy,
            registry=context.registry,
        )
        cell = execute_agent_diff_official_cell(
            context=context,
            job=job,
            driver=driver,
            provider_client=provider_client,
            broker=broker,
            implementation_bundle_sha256=(
                agent_diff_implementation_bundle_sha256(
                    sealed_policy.binding.method
                )
            ),
            run_config_sha256=agent_diff_run_config_sha256(context),
        )
        completed.append(cell)
        return cell.result

    result = coordinator.execute_job(context.job.job_id, execute_job)
    if len(completed) != 1 or completed[0].result != result:
        raise RuntimeError("agent_diff_live_cell_result_not_retained")
    return completed[0], coordinator.public_report()


def execute_agent_diff_official_cell(
    *,
    context: AgentDiffLiveContext,
    job: ExecutionJob,
    driver: AgentDiffRouterDriver,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    implementation_bundle_sha256: str,
    run_config_sha256: str,
) -> AgentDiffCellExecution:
    """Run all 45 rows and publish only a complete, self-validated receipt."""

    binding = driver.artifact
    registry = context.registry
    if (
        job != context.job
        or job.kind != "official_test_cell"
        or job.surface != AGENT_DIFF
        or job.item_count != AGENT_DIFF_OFFICIAL_TEST_COUNT
        or registry.denominator != AGENT_DIFF_OFFICIAL_TEST_COUNT
        or job.registry_sha256 != registry.split.test_registry_sha256
        or binding.test_registry_sha256 != registry.split.test_registry_sha256
        or job.method != binding.method
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_cell_binding_mismatch"
        )
    if job.output_path.exists():
        raise FileExistsError("agent_diff_live_output_exists_no_overwrite")
    _require_sha256(
        implementation_bundle_sha256,
        "agent_diff_live_implementation_bundle_sha256",
    )
    _require_sha256(
        run_config_sha256,
        "agent_diff_live_run_config_sha256",
    )
    evidence_dir = (
        job.output_path.parent / f".{job.output_path.name}.{uuid4().hex}.private"
    )
    evidence_dir.mkdir(parents=True, exist_ok=False)
    private_tasks = load_agent_diff_private_tasks(registry)
    results: list[AgentDiffTaskResult] = []
    for case_index, task in enumerate(private_tasks):
        case_dir = evidence_dir / (
            f"case_{case_index:03d}_{task.ref.example_sha256[:12]}"
        )
        case_dir.mkdir(parents=False, exist_ok=False)
        result = run_agent_diff_task(
            context=context,
            driver=driver,
            task=task,
            provider_client=provider_client,
            broker=broker,
            case_dir=case_dir,
        )
        if result.example_sha256 != registry.tasks[case_index].example_sha256:
            raise AgentDiffLiveContractError(
                "agent_diff_live_task_result_identity_mismatch"
            )
        _validate_private_case_manifest_record(
            result.private_case_manifest_record,
            private_root=context.request.private_root,
        )
        results.append(result)
    if len(results) != AGENT_DIFF_OFFICIAL_TEST_COUNT:
        raise AgentDiffLiveContractError(
            "agent_diff_live_cell_denominator_incomplete"
        )

    private_raw_case_manifest_sha256 = _write_private_manifest(
        evidence_dir / "raw_case_manifest.json",
        {
            "schema": "papc.agent_diff.private_raw_case_manifest.v1",
            "job_id": job.job_id,
            "registry_sha256": registry.split.test_registry_sha256,
            "cases": [
                dict(result.private_case_manifest_record)
                for result in results
            ],
        },
    )
    raw_usage_receipt_manifest_sha256 = _write_private_jsonl(
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
        surface=AGENT_DIFF,
        test_registry_sha256=(
            context.surface_authority.test_registry_sha256
        ),
        trial_policy_sha256=(
            context.surface_authority.trial_policy_sha256
        ),
        official_metric=context.surface_authority.official_metric,
        cases=tuple(
            _case_outcome_cost(case_index, result)
            for case_index, result in enumerate(results)
        ),
        registry_order_matches_test_manifest=True,
    )
    outcome_path = (
        evidence_dir / "registry_ordered_outcome_cost_manifest.json"
    )
    _write_private_manifest(outcome_path, outcome_cost_manifest.private_record())
    loaded_outcome = RegistryOrderedOutcomeCostManifest.from_private_record(
        _read_mapping(
            outcome_path,
            "agent_diff_live_registry_ordered_outcomes",
        )
    )
    if loaded_outcome.manifest_sha256 != outcome_cost_manifest.manifest_sha256:
        raise AgentDiffLiveContractError(
            "agent_diff_live_outcome_manifest_roundtrip_mismatch"
        )

    receipt = build_agent_diff_table1_receipt(
        results=tuple(results),
        authority=context.surface_authority,
        binding=binding,
        implementation_bundle_sha256=implementation_bundle_sha256,
        run_config_sha256=run_config_sha256,
        private_raw_case_manifest_sha256=(
            private_raw_case_manifest_sha256
        ),
        raw_usage_receipt_manifest_sha256=(
            raw_usage_receipt_manifest_sha256
        ),
        registry_ordered_outcome_cost_manifest_sha256=(
            outcome_cost_manifest.manifest_sha256
        ),
    )
    loaded_outcome.validate_against(receipt)
    payload = (
        json.dumps(
            receipt.public_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    # Parse the exact public bytes before publication as a final self-check.
    roundtrip = Table1ResultReceipt.from_public_record(
        json.loads(payload.decode("utf-8"))
    )
    if roundtrip != receipt:
        raise AgentDiffLiveContractError(
            "agent_diff_live_public_receipt_roundtrip_mismatch"
        )
    _write_bytes_no_replace(job.output_path, payload)
    result = JobExecutionResult(
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
    return AgentDiffCellExecution(
        result=result,
        public_receipt=receipt,
        evidence_dir=evidence_dir,
    )


def run_agent_diff_task(
    *,
    context: AgentDiffLiveContext,
    driver: AgentDiffRouterDriver,
    task: AgentDiffPrivateTask,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    case_dir: Path,
) -> AgentDiffTaskResult:
    """Execute one registry row under the artifact's native routing level."""

    private_paths: list[Path] = []
    if driver.artifact.method in {ROUTELLM, FRUGALGPT}:
        episode = driver.bind_public_episode(
            task=task.ref,
            initial_environment_sha256=task.initial_environment_sha256,
        )
        episode_counter = 0

        def run_episode(
            binding: FullEpisodeBinding,
            role: str,
            model: str,
        ) -> GeneratedEpisode:
            nonlocal episode_counter
            path = case_dir / f"episode_{episode_counter:02d}_{role}.json"
            episode_counter += 1
            generated = _run_static_episode(
                task=task,
                binding=binding,
                role=role,
                model=model,
                api=context.native_api,
                config=context.native_config,
                provider_client=provider_client,
                broker=broker,
                raw_path=path,
                api_key_env=context.request.api_key_env,
            )
            private_paths.append(path)
            return generated

        if driver.artifact.method == ROUTELLM:
            routed = driver.run_routellm(
                episode=episode,
                run_episode=run_episode,
                score_selected=_score_handle,
            )
        else:
            routed = driver.run_frugalgpt(
                episode=episode,
                run_episode=run_episode,
                score_selected=_score_handle,
            )
        routing_path = case_dir / "routing.json"
        _write_json_no_replace(
            routing_path,
            {
                "schema": "papc.agent_diff.external_router_decisions.v1",
                "method": driver.artifact.method,
                "example_sha256": task.ref.example_sha256,
                "status": "completed",
                "decisions": [routed.decision.public_record()],
                "selected_roles": [routed.selected_role],
                "selected_models": [routed.selected_model],
                "episode_count": routed.episode_count,
                "strong_restarted_from_pristine": (
                    routed.strong_restarted_from_pristine
                ),
            },
        )
        private_paths.append(routing_path)
        receipts = routed.receipts
        official_score = float(routed.official_score)
        decision_count = 1
    else:
        path = case_dir / "history_episode.json"
        routed, official_score = _run_mtrouter_episode(
            task=task,
            driver=driver,
            api=context.native_api,
            config=context.native_config,
            provider_client=provider_client,
            broker=broker,
            raw_path=path,
            api_key_env=context.request.api_key_env,
        )
        private_paths.append(path)
        routing_path = case_dir / "routing.json"
        _write_json_no_replace(
            routing_path,
            {
                "schema": "papc.agent_diff.external_router_decisions.v1",
                "method": driver.artifact.method,
                "example_sha256": task.ref.example_sha256,
                "status": routed.status,
                "decisions": [
                    decision.public_record()
                    for decision in routed.decisions
                ],
                "selected_roles": list(routed.selected_roles),
                "selected_models": [
                    decision.selected_model
                    for decision in routed.decisions
                ],
                "turn_count": len(routed.decisions),
            },
        )
        private_paths.append(routing_path)
        receipts = routed.receipts
        decision_count = len(routed.decisions)
    local_latency = sum(
        receipt.latency_seconds
        for receipt in receipts
        if receipt.role == "router" and receipt.provider_calls == 0
    )
    return AgentDiffTaskResult(
        example_sha256=task.ref.example_sha256,
        official_score=official_score,
        decision_count=decision_count,
        local_router_latency_seconds=local_latency,
        receipts=tuple(receipts),
        private_case_manifest_record=_private_case_manifest_record(
            example_sha256=task.ref.example_sha256,
            private_root=context.request.private_root,
            paths=tuple(private_paths),
        ),
    )


def run_agent_diff_pristine_fit_dev_episode(
    *,
    task: AgentDiffPrivateTask,
    role: str,
    model: str,
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    raw_path: Path,
    api_key_env: str,
) -> tuple[GeneratedEpisode, float]:
    """Run one source-verified official-train row from a pristine environment."""

    binding = build_agent_diff_full_episode_binding(
        service=task.ref.service,
        original_request=task.ref.original_request,
        initial_environment_sha256=task.initial_environment_sha256,
    )
    generated = _run_static_episode(
        task=task,
        binding=binding,
        role=role,
        model=model,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=Path(raw_path).resolve(),
        api_key_env=api_key_env,
    )
    return generated, _score_handle(generated.opaque_score_handle)


def run_agent_diff_history_fit_dev_episode(
    *,
    task: AgentDiffPrivateTask,
    artifact: RouterArtifactBinding,
    decide_next: Callable[[Any], Any],
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    raw_path: Path,
    api_key_env: str,
) -> tuple[Any, float]:
    """Run one source-bound train row with an injected turn policy."""

    if (
        not isinstance(artifact, RouterArtifactBinding)
        or artifact.method != MTROUTER
        or artifact.surface != AGENT_DIFF
    ):
        raise ValueError("agent_diff_fit_dev_history_artifact_invalid")
    if not callable(decide_next):
        raise TypeError("agent_diff_fit_dev_history_decider_invalid")
    session = _NativeSession(
        task=task,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=Path(raw_path).resolve(),
        api_key_env=api_key_env,
    )
    terminal_evaluation: dict[str, Any] | None = None
    error: BaseException | None = None
    try:
        session.start()

        def execute_turn(decision, history):
            nonlocal terminal_evaluation
            text, receipt = session.model_turn(
                model=decision.selected_model,
                public_history=history,
            )
            parsed = _parse_react_message(text)
            event: dict[str, Any] = {
                "iteration": len(session.trace) + 1,
                "model": decision.selected_model,
                "selected_role": decision.selected_role,
                "assistant_text": text,
                "assistant_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "receipt": receipt.public_record(),
            }
            action = parsed["action"]
            done = parsed["done"]
            if action:
                tool = session.execute_action(action)
                event.update(tool)
                session.trace.append(event)
                return HistoryTurn(
                    appended_public_history=(
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "<observation>\n"
                                f"{tool['observation']}\n"
                                "</observation>"
                            ),
                        },
                    ),
                    receipts=(receipt,),
                    terminal=False,
                )
            if done:
                event["done"] = done
                session.trace.append(event)
                session.completion_reason = "done_tag"
                terminal_evaluation = session.evaluate()
                return HistoryTurn(
                    appended_public_history=(
                        {"role": "assistant", "content": text},
                    ),
                    receipts=(receipt,),
                    terminal=True,
                    opaque_score_handle={
                        "official_score": (
                            1.0
                            if terminal_evaluation["passed"] is True
                            else 0.0
                        )
                    },
                )
            event["format_error"] = "missing_action_or_done"
            session.trace.append(event)
            return HistoryTurn(
                appended_public_history=(
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Please respond using either <action>one bash "
                            "command</action> or <done>brief summary</done>."
                        ),
                    },
                ),
                receipts=(receipt,),
                terminal=False,
            )

        routed = run_history_routed_episode(
            artifact=artifact,
            original_request=task.ref.original_request,
            initial_public_history=(
                {"role": "user", "content": task.ref.original_request},
            ),
            turn_cap=config.max_iterations,
            decide_next=decide_next,
            execute_turn=execute_turn,
            score_terminal=_score_handle,
        )
        if terminal_evaluation is None:
            session.completion_reason = "max_iterations_exhausted"
            terminal_evaluation = session.evaluate()
        official_score = (
            1.0 if terminal_evaluation["passed"] is True else 0.0
        )
        return routed, official_score
    except BaseException as exc:
        error = exc
        raise
    finally:
        session.finalize(error=error)


def load_agent_diff_private_tasks(
    registry: AgentDiffOfficialRegistry,
) -> tuple[AgentDiffPrivateTask, ...]:
    """Reload hidden evaluator fields and bind them to registry order."""

    if not isinstance(registry, AgentDiffOfficialRegistry):
        raise TypeError("agent_diff_live_registry_invalid")
    rows = _read_jsonl(registry.dataset_dir / "test.jsonl")
    if len(rows) != registry.denominator:
        raise AgentDiffLiveContractError(
            "agent_diff_live_private_test_denominator_mismatch"
        )
    tasks: list[AgentDiffPrivateTask] = []
    for ref, raw in zip(registry.tasks, rows, strict=True):
        if (
            ref.payload_sha256 != _digest(raw)
            or _required_text(
                raw.get("test_id"),
                "agent_diff_live_test_id",
            )
            != ref.native_test_id
            or _required_text(
                raw.get("service"),
                "agent_diff_live_service",
            )
            != ref.service
            or _required_text(
                raw.get("question"),
                "agent_diff_live_question",
            )
            != ref.original_request
        ):
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_row_registry_mismatch"
            )
        tasks.append(
            AgentDiffPrivateTask(
                ref=ref,
                test_id=_required_text(
                    raw.get("test_id"),
                    "agent_diff_live_test_id",
                ),
                test_name=_required_text(
                    raw.get("test_name"),
                    "agent_diff_live_test_name",
                ),
                answer=_json_object(
                    raw.get("answer"),
                    "agent_diff_live_answer",
                ),
                info=_json_object(
                    raw.get("info"),
                    "agent_diff_live_info",
                ),
                metadata={
                    key: raw.get(key)
                    for key in (
                        "task_horizon",
                        "operation_type",
                        "entity_scope",
                        "information_availability",
                        "prompt_ambiguity",
                    )
                },
            )
        )
    return tuple(tasks)


def load_agent_diff_typed_artifact(
    method: ExternalMethod,
    artifact_json: Path,
    runtime_asset_path: Path | None,
) -> AgentDiffTypedArtifact:
    """Load one canonical typed artifact without provider access."""

    path = Path(artifact_json).resolve()
    asset = (
        None
        if runtime_asset_path is None
        else Path(runtime_asset_path).resolve()
    )
    if method == ROUTELLM:
        artifact = _load_routellm_artifact(path)
        binding = artifact.to_router_binding()
        if asset is None or not asset.is_file():
            raise FileNotFoundError(
                "agent_diff_live_routellm_checkpoint_missing"
            )
    elif method == FRUGALGPT:
        artifact = FrugalGPTScorerArtifact.from_public_record(
            _read_mapping(path, "agent_diff_live_frugalgpt_artifact")
        )
        binding = artifact.router_binding()
        if asset is None or not asset.is_dir():
            raise FileNotFoundError(
                "agent_diff_live_frugalgpt_checkpoint_missing"
            )
    elif method == MTROUTER:
        artifact = load_mtrouter_artifact(path)
        binding = artifact.router_binding()
        if artifact.uses_all_strong_shortcut:
            if asset is not None:
                raise AgentDiffLiveContractError(
                    "agent_diff_live_mtrouter_all_strong_asset_forbidden"
                )
        elif asset is None or not asset.is_file():
            raise FileNotFoundError(
                "agent_diff_live_mtrouter_checkpoint_missing"
            )
    else:
        raise AgentDiffLiveContractError(
            "agent_diff_live_method_not_in_publication_subset"
        )
    return AgentDiffTypedArtifact(
        method=method,
        artifact=artifact,
        binding=binding,
        runtime_asset_path=asset,
    )


def load_agent_diff_sealed_policy(
    *,
    typed: AgentDiffTypedArtifact,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    device: str,
) -> SealedRouterPolicy:
    binding = typed.binding
    asset = typed.runtime_asset_path
    if typed.method == ROUTELLM:
        return load_routellm_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_payload=asset.read_bytes(),
            embed_prompt=_routellm_embed_transport(
                provider_client=provider_client,
                broker=broker,
            ),
        )
    if typed.method == FRUGALGPT:
        return load_frugalgpt_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_dir=asset,
        )
    if typed.method == MTROUTER:
        return load_mtrouter_sealed_policy(
            binding=binding,
            artifact=typed.artifact,
            checkpoint_path=asset,
            device=device,
        )
    raise AgentDiffLiveContractError(
        "agent_diff_live_method_not_in_publication_subset"
    )


def _routellm_embed_transport(
    *,
    provider_client: Any,
    broker: ProviderAttemptBroker,
) -> Callable[[RouteLLMMFEmbeddingRequest], RouteLLMMFEmbeddingResult]:
    def embed(
        request: RouteLLMMFEmbeddingRequest,
    ) -> RouteLLMMFEmbeddingResult:
        response_box: dict[str, Any] = {}

        def raw_attempt() -> CallReceipt:
            started = time.perf_counter()
            response = provider_client.embeddings.create(
                model=ROUTELLM_EMBEDDING_MODEL,
                input=request.prompt,
                encoding_format="float",
            )
            receipt = _provider_response_receipt(
                response=response,
                role="router",
                phase="route_embedding",
                model=ROUTELLM_EMBEDDING_MODEL,
                reasoning_effort="embedding",
                latency_seconds=max(
                    0.0,
                    time.perf_counter() - started,
                ),
            )
            _require_complete_provider_receipt(receipt)
            response_box["response"] = response
            return receipt

        receipt = broker.perform(raw_attempt)
        if "response" not in response_box:
            raise RuntimeError(
                "agent_diff_route_embedding_response_missing_after_settle"
            )
        data = getattr(response_box["response"], "data", None)
        if not isinstance(data, Sequence) or len(data) != 1:
            raise ValueError("agent_diff_route_embedding_payload_invalid")
        vector = getattr(data[0], "embedding", None)
        if not isinstance(vector, Sequence):
            raise ValueError("agent_diff_route_embedding_vector_invalid")
        embedding = tuple(float(value) for value in vector)
        return RouteLLMMFEmbeddingResult(
            request_sha256=request.sha256,
            embedding=embedding,
            receipt=receipt,
        )

    return embed


def build_agent_diff_table1_receipt(
    *,
    results: tuple[AgentDiffTaskResult, ...],
    authority: SurfaceAuthority,
    binding: RouterArtifactBinding,
    implementation_bundle_sha256: str,
    run_config_sha256: str,
    private_raw_case_manifest_sha256: str,
    raw_usage_receipt_manifest_sha256: str,
    registry_ordered_outcome_cost_manifest_sha256: str,
) -> Table1ResultReceipt:
    if len(results) != AGENT_DIFF_OFFICIAL_TEST_COUNT:
        raise AgentDiffLiveContractError(
            "agent_diff_live_cell_denominator_incomplete"
        )
    receipts = tuple(receipt for result in results for receipt in result.receipts)
    _require_complete_provider_receipts(receipts)
    agent = tuple(
        receipt
        for receipt in receipts
        if receipt.provider_calls > 0 and receipt.role in AGENT_SIDE_ROLES
    )
    if not agent:
        raise AgentDiffLiveContractError(
            "agent_diff_live_agent_provider_receipts_empty"
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
    receipt = Table1ResultReceipt(
        row_id=binding.method,
        surface=AGENT_DIFF,
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
        private_raw_case_manifest_sha256=(
            private_raw_case_manifest_sha256
        ),
        raw_usage_receipt_manifest_sha256=(
            raw_usage_receipt_manifest_sha256
        ),
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
            software=f"papc-agent-diff-live:{binding.method}",
        ),
        public_safe=True,
    )
    receipt.validate_against(authority)
    return receipt


def agent_diff_source_authority(
    agent_diff_repo: Path,
    registry: AgentDiffOfficialRegistry,
    scorer_path: Path,
) -> tuple[str, str]:
    """Recompute the exact Agent-Diff source bundles used by authority v2."""

    root = Path(agent_diff_repo).resolve()
    if _git_output(root, "rev-parse", "HEAD") != FROZEN_AGENT_DIFF_REPO_REVISION:
        raise AgentDiffLiveContractError(
            "agent_diff_live_runtime_revision_mismatch"
        )
    dataset = registry.dataset_dir.resolve()
    if not dataset.is_relative_to(root):
        raise AgentDiffLiveContractError(
            "agent_diff_live_dataset_outside_runtime_repo"
        )
    runtime_paths = (
        *_required_glob(
            root / "backend" / "src",
            "**/*.py",
            "agent_diff_live_backend_sources",
        ),
        *_required_glob(
            root / "sdk" / "agent-diff-python" / "agent_diff",
            "**/*.py",
            "agent_diff_live_sdk_sources",
        ),
        *_optional_paths(
            root,
            (
                "backend/pyproject.toml",
                "backend/uv.lock",
                "sdk/agent-diff-python/pyproject.toml",
                "sdk/agent-diff-python/uv.lock",
            ),
        ),
    )
    runtime_sha256 = _digest(
        {
            "schema": "papc.table1.native_runtime_source_bundle.v1",
            "surface": AGENT_DIFF,
            "revision": FROZEN_AGENT_DIFF_REPO_REVISION,
            "source_binding_sha256": registry.split.source_binding_sha256,
            "source_tree_sha256": _source_bundle_sha256(
                root=root,
                paths=runtime_paths,
            ),
        }
    )
    upstream_scorer_sha256 = _source_bundle_sha256(
        root=root,
        paths=(
            *_required_glob(
                root
                / "backend"
                / "src"
                / "platform"
                / "evaluationEngine",
                "**/*.py",
                "agent_diff_live_evaluator_sources",
            ),
            root / "backend" / "src" / "platform" / "api" / "routes.py",
        ),
    )
    scorer_sha256 = _digest(
        {
            "schema": "papc.table1.agent_diff_scorer_bundle.v1",
            "upstream_scorer_sha256": upstream_scorer_sha256,
            "registered_correction_sha256": _file_sha256(
                Path(scorer_path).resolve()
            ),
        }
    )
    return runtime_sha256, scorer_sha256


def agent_diff_implementation_bundle_sha256(
    method: ExternalMethod,
) -> str:
    root = Path(__file__).resolve().parent
    method_file = {
        ROUTELLM: "routellm_mf_upstream.py",
        FRUGALGPT: "frugalgpt_upstream.py",
        MTROUTER: "mtrouter_upstream.py",
    }[method]
    return _source_bundle_sha256(
        root=root,
        paths=tuple(
            root / name
            for name in (
                "agent_diff_live.py",
                "agent_diff_router_driver.py",
                "external_router_execution.py",
                "sealed_router_policy.py",
                "router_accounting.py",
                "router_contract.py",
                "router_policies.py",
                "router_runtime.py",
                "table1_paired_stats.py",
                "table1_results.py",
                method_file,
            )
        ),
    )


def agent_diff_run_config_sha256(
    context: AgentDiffLiveContext,
) -> str:
    return _digest(
        {
            "schema": "papc.agent_diff.external_router_run_config.v1",
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
            "authority_manifest_file_sha256": (
                context.authority_manifest_file_sha256
            ),
            "runtime_environment_sha256": (
                context.runtime_environment_sha256
            ),
            "native_api": context.native_api.public_record(),
            "native_run_config": context.native_config.public_record(),
            "method_artifact": context.typed_artifact.binding.public_record(),
            "runtime_asset_sha256": _path_content_sha256(
                context.typed_artifact.runtime_asset_path
            )
            if context.typed_artifact.runtime_asset_path is not None
            else None,
            "trial_policy": {
                "replicates_per_task": 1,
                "table1_trial_policy_sha256": (
                    context.surface_authority.trial_policy_sha256
                ),
                "execution_trial_policy_sha256": (
                    context.job.trial_policy_sha256
                ),
            },
            "react_protocol": {
                "prompt_profile": AGENT_DIFF_PROMPT_PROFILE,
                "prompt_sha256s": {
                    service: hashlib.sha256(
                        _system_prompt(service).encode("utf-8")
                    ).hexdigest()
                    for service in sorted(_SERVICE_CONFIG)
                },
                "max_iterations": AGENT_DIFF_MAX_ITERATIONS,
                "max_model_tokens": AGENT_DIFF_MAX_MODEL_TOKENS,
                "temperature": AGENT_DIFF_TEMPERATURE,
                "provider_retries": 0,
                "command_timeout_seconds": (
                    AGENT_DIFF_COMMAND_TIMEOUT_SECONDS
                ),
                "observation_chars": AGENT_DIFF_OBSERVATION_CHARS,
                "environment_ttl_seconds": AGENT_DIFF_ENV_TTL_SECONDS,
            },
        }
    )


def _run_static_episode(
    *,
    task: AgentDiffPrivateTask,
    binding: FullEpisodeBinding,
    role: str,
    model: str,
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    raw_path: Path,
    api_key_env: str,
) -> GeneratedEpisode:
    if role not in {"cheap", "strong"}:
        raise ValueError("agent_diff_live_static_role_invalid")
    expected_model = CHEAP_MODEL if role == "cheap" else STRONG_MODEL
    if model != expected_model:
        raise ValueError("agent_diff_live_static_model_mismatch")
    session = _NativeSession(
        task=task,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=raw_path,
        api_key_env=api_key_env,
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": task.ref.original_request}
    ]
    last_text = ""
    error: BaseException | None = None
    try:
        session.start()
        for iteration in range(1, config.max_iterations + 1):
            text, receipt = session.model_turn(
                model=model,
                public_history=history,
            )
            last_text = text
            parsed = _parse_react_message(text)
            event: dict[str, Any] = {
                "iteration": iteration,
                "model": model,
                "assistant_text": text,
                "assistant_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "receipt": receipt.public_record(),
            }
            action = parsed["action"]
            done = parsed["done"]
            if action:
                tool = session.execute_action(action)
                event.update(tool)
                session.trace.append(event)
                history.append({"role": "assistant", "content": text})
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "<observation>\n"
                            f"{tool['observation']}\n"
                            "</observation>"
                        ),
                    }
                )
                continue
            if done:
                event["done"] = done
                session.trace.append(event)
                history.append({"role": "assistant", "content": text})
                session.completion_reason = "done_tag"
                break
            event["format_error"] = "missing_action_or_done"
            session.trace.append(event)
            history.append({"role": "assistant", "content": text})
            history.append(
                {
                    "role": "user",
                    "content": (
                        "Please respond using either <action>one bash "
                        "command</action> or <done>brief summary</done>."
                    ),
                }
            )
        else:
            session.completion_reason = "max_iterations_exhausted"
        evaluation = session.evaluate()
        return GeneratedEpisode(
            role=role,
            model=model,
            initial_environment_sha256=binding.initial_environment_sha256,
            environment_instance_sha256=(
                session.environment_instance_sha256
            ),
            generated_answer_view=last_text,
            receipts=tuple(session.receipts),
            opaque_score_handle={
                "official_score": (
                    1.0 if evaluation["passed"] is True else 0.0
                )
            },
            public_episode_history=tuple(history),
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        session.finalize(error=error)


def _run_mtrouter_episode(
    *,
    task: AgentDiffPrivateTask,
    driver: AgentDiffRouterDriver,
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: ProviderAttemptBroker,
    raw_path: Path,
    api_key_env: str,
) -> tuple[Any, float]:
    session = _NativeSession(
        task=task,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=raw_path,
        api_key_env=api_key_env,
    )
    terminal_evaluation: dict[str, Any] | None = None
    error: BaseException | None = None
    try:
        session.start()

        def execute_turn(decision, history):
            nonlocal terminal_evaluation
            text, receipt = session.model_turn(
                model=decision.selected_model,
                public_history=history,
            )
            parsed = _parse_react_message(text)
            event: dict[str, Any] = {
                "iteration": len(session.trace) + 1,
                "model": decision.selected_model,
                "selected_role": decision.selected_role,
                "assistant_text": text,
                "assistant_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "receipt": receipt.public_record(),
            }
            action = parsed["action"]
            done = parsed["done"]
            if action:
                tool = session.execute_action(action)
                event.update(tool)
                session.trace.append(event)
                return HistoryTurn(
                    appended_public_history=(
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "<observation>\n"
                                f"{tool['observation']}\n"
                                "</observation>"
                            ),
                        },
                    ),
                    receipts=(receipt,),
                    terminal=False,
                )
            if done:
                event["done"] = done
                session.trace.append(event)
                session.completion_reason = "done_tag"
                terminal_evaluation = session.evaluate()
                return HistoryTurn(
                    appended_public_history=(
                        {"role": "assistant", "content": text},
                    ),
                    receipts=(receipt,),
                    terminal=True,
                    opaque_score_handle={
                        "official_score": (
                            1.0
                            if terminal_evaluation["passed"] is True
                            else 0.0
                        )
                    },
                )
            event["format_error"] = "missing_action_or_done"
            session.trace.append(event)
            return HistoryTurn(
                appended_public_history=(
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Please respond using either <action>one bash "
                            "command</action> or <done>brief summary</done>."
                        ),
                    },
                ),
                receipts=(receipt,),
                terminal=False,
            )

        routed = driver.run_mtrouter(
            task=task.ref,
            turn_cap=config.max_iterations,
            execute_turn=execute_turn,
            score_terminal=_score_handle,
        )
        if terminal_evaluation is None:
            session.completion_reason = "max_iterations_exhausted"
            terminal_evaluation = session.evaluate()
        official_score = (
            1.0 if terminal_evaluation["passed"] is True else 0.0
        )
        return routed, official_score
    except BaseException as exc:
        error = exc
        raise
    finally:
        session.finalize(error=error)


def _validate_plan_registry_authority(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    registry: AgentDiffOfficialRegistry,
    authority: SurfaceAuthority,
    authority_manifest: Table1AuthorityManifest,
    authority_manifest_file_sha256: str,
) -> None:
    plan_authority = next(
        (item for item in plan.authorities if item.surface == AGENT_DIFF),
        None,
    )
    if (
        plan_authority is None
        or plan_authority.denominator != AGENT_DIFF_OFFICIAL_TEST_COUNT
        or plan_authority.replicates_per_task != 1
        or plan_authority.test_registry_sha256
        != registry.split.test_registry_sha256
        or plan_authority.source_binding_sha256
        != registry.split.source_binding_sha256
        or plan_authority.trial_policy_sha256
        != authority.trial_policy_sha256
        or plan_authority.fit_registry_sha256
        != authority.fit_registry_sha256
        or plan_authority.dev_registry_sha256
        != authority.dev_registry_sha256
        or plan_authority.official_scorer != authority.official_scorer
        or plan_authority.official_metric != authority.official_metric
        or plan_authority.scorer_sha256 != authority.scorer_sha256
        or plan_authority.runtime_name != authority.runtime_name
        or plan_authority.runtime_sha256 != authority.runtime_sha256
        or job.registry_sha256 != registry.split.test_registry_sha256
        or job.source_binding_sha256 != registry.split.source_binding_sha256
        or plan.authority_manifest_file_sha256
        != authority_manifest_file_sha256
        or plan.authority_manifest_semantic_sha256
        != authority_manifest.sha256
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_execution_trial_authority_mismatch"
        )
    expected = {
        "authority_test_registry_sha256": authority.test_registry_sha256,
        "authority_manifest_file_sha256": authority_manifest_file_sha256,
        "authority_manifest_semantic_sha256": authority_manifest.sha256,
        "official_scorer": authority.official_scorer,
        "official_metric": authority.official_metric,
        "scorer_sha256": authority.scorer_sha256,
        "runtime_name": authority.runtime_name,
        "runtime_sha256": authority.runtime_sha256,
    }
    for key, expected_value in expected.items():
        if getattr(job, key) != expected_value:
            raise AgentDiffLiveContractError(
                f"agent_diff_live_execution_authority_mismatch:{key}"
            )


def _validate_surface_authority(
    *,
    authority: SurfaceAuthority,
    registry: AgentDiffOfficialRegistry,
    runtime_sha256: str,
    scorer_sha256: str,
) -> None:
    if (
        authority.surface != AGENT_DIFF
        or authority.denominator != AGENT_DIFF_OFFICIAL_TEST_COUNT
        or authority.replicates_per_task != 1
        or authority.test_registry_sha256
        != registry.split.test_registry_sha256
        or authority.fit_registry_sha256
        != registry.split.fit_registry_sha256
        or authority.dev_registry_sha256
        != registry.split.dev_registry_sha256
        or authority.official_scorer
        != "agent_diff_evaluate_run_plus_registered_corrections_v1"
        or authority.official_metric != AGENT_DIFF_OFFICIAL_METRIC
        or authority.runtime_name != AGENT_DIFF_RUNTIME_NAME
        or authority.runtime_sha256 != runtime_sha256
        or authority.scorer_sha256 != scorer_sha256
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_authority_source_mismatch"
        )


def _provider_response_receipt(
    *,
    response: Any,
    role: str,
    phase: str,
    model: str,
    reasoning_effort: str,
    latency_seconds: float,
) -> CallReceipt:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_int(
        usage,
        "input_tokens",
        fallback="prompt_tokens",
    )
    output_tokens = _usage_int(
        usage,
        "output_tokens",
        fallback="completion_tokens",
    )
    if output_tokens is None and model == ROUTELLM_EMBEDDING_MODEL:
        output_tokens = 0
    input_details = _usage_value(
        usage,
        "input_tokens_details",
        fallback="prompt_tokens_details",
    )
    output_details = _usage_value(
        usage,
        "output_tokens_details",
        fallback="completion_tokens_details",
    )
    cached_tokens = _usage_int(input_details, "cached_tokens")
    reasoning_tokens = _usage_int(output_details, "reasoning_tokens")
    if input_tokens is not None and cached_tokens is None:
        cached_tokens = 0
    if output_tokens is not None and reasoning_tokens is None:
        reasoning_tokens = 0
    return CallReceipt(
        role=role,  # type: ignore[arg-type]
        phase=phase,
        model=model,
        reasoning_effort=reasoning_effort,
        provider_calls=1,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        retry_count=0,
        latency_seconds=latency_seconds,
        provider_reported_cost_usd=_cache_neutral_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _require_complete_provider_receipts(
    receipts: Sequence[CallReceipt],
) -> None:
    provider = tuple(
        receipt for receipt in receipts if receipt.provider_calls > 0
    )
    if not provider:
        raise AgentDiffLiveContractError(
            "agent_diff_live_provider_receipts_empty"
        )
    for receipt in provider:
        _require_complete_provider_receipt(receipt)


def _require_complete_provider_receipt(receipt: CallReceipt) -> None:
    if (
        receipt.provider_calls != 1
        or receipt.retry_count != 0
        or receipt.input_tokens is None
        or receipt.cached_input_tokens is None
        or receipt.output_tokens is None
        or receipt.reasoning_tokens is None
        or receipt.provider_reported_cost_usd is None
    ):
        raise AgentDiffLiveContractError(
            "agent_diff_live_raw_provider_receipt_incomplete"
        )


def _case_outcome_cost(
    case_index: int,
    result: AgentDiffTaskResult,
) -> CaseOutcomeCost:
    agent = tuple(
        item
        for item in result.receipts
        if item.provider_calls > 0 and item.role in AGENT_SIDE_ROLES
    )
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
        private_raw_case_receipt_sha256=_digest(
            result.private_case_manifest_record
        ),
        raw_usage_receipt_sha256=_digest(
            [item.public_record() for item in result.receipts]
        ),
    )


def _private_case_manifest_record(
    *,
    example_sha256: str,
    private_root: Path,
    paths: tuple[Path, ...],
) -> dict[str, Any]:
    if not paths:
        raise AgentDiffLiveContractError(
            "agent_diff_live_private_case_paths_empty"
        )
    root = Path(private_root).resolve()
    files: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_path_outside_root"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(
                "agent_diff_live_private_case_file_missing"
            )
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": _file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "example_sha256": example_sha256,
        "private_files": files,
    }


def _validate_private_case_manifest_record(
    record: Mapping[str, Any],
    *,
    private_root: Path,
) -> None:
    root = Path(private_root).resolve()
    for raw in _sequence(
        record.get("private_files"),
        "agent_diff_live_private_case_files",
    ):
        item = _mapping(raw, "agent_diff_live_private_case_file")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_path_invalid"
            )
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_path_outside_root"
            ) from exc
        if (
            not path.is_file()
            or _file_sha256(path) != item.get("sha256")
            or path.stat().st_size != item.get("size_bytes")
        ):
            raise AgentDiffLiveContractError(
                "agent_diff_live_private_case_file_binding_mismatch"
            )


def _load_routellm_artifact(path: Path) -> RouteLLMMFArtifact:
    raw = _read_mapping(path, "agent_diff_live_routellm_artifact")
    calibration_raw = _mapping(
        raw.get("dev_calibration"),
        "agent_diff_live_routellm_calibration",
    )
    candidates = _sequence(
        calibration_raw.get("candidate_receipts"),
        "agent_diff_live_routellm_candidates",
    )
    calibration = RouteLLMMFDevCalibration(
        **_constructor_values(
            RouteLLMMFDevCalibration,
            calibration_raw,
            replacements={
                "candidate_receipts": tuple(
                    RouteLLMMFDevCandidateReceipt(
                        **_constructor_values(
                            RouteLLMMFDevCandidateReceipt,
                            _mapping(
                                item,
                                "agent_diff_live_routellm_candidate",
                            ),
                        )
                    )
                    for item in candidates
                )
            },
        )
    )
    class_counts = _mapping(
        raw.get("class_counts"),
        "agent_diff_live_routellm_class_counts",
    )
    artifact = RouteLLMMFArtifact(
        **_constructor_values(
            RouteLLMMFArtifact,
            raw,
            replacements={
                "class_counts": (
                    ("cheap_wins", class_counts["cheap_wins"]),
                    ("strong_wins", class_counts["strong_wins"]),
                ),
                "dev_calibration": calibration,
            },
        )
    )
    if artifact.public_record() != dict(raw):
        raise AgentDiffLiveContractError(
            "agent_diff_live_routellm_artifact_not_canonical"
        )
    return artifact


def _constructor_values(
    cls: type,
    record: Mapping[str, Any],
    *,
    replacements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = replacements or {}
    values: dict[str, Any] = {}
    for item in fields(cls):
        if not item.init:
            continue
        if item.name in supplied:
            values[item.name] = supplied[item.name]
        elif item.name in record:
            values[item.name] = record[item.name]
    return values


def _score_handle(handle: Any) -> float:
    if not isinstance(handle, Mapping):
        raise TypeError("agent_diff_live_score_handle_invalid")
    score = handle.get("official_score")
    if score not in {0.0, 1.0}:
        raise ValueError("agent_diff_live_score_handle_not_binary")
    return float(score)


def _system_prompt(service: str) -> str:
    config = _SERVICE_CONFIG.get(service)
    if config is None:
        raise ValueError("agent_diff_live_service_unsupported")
    return _REACT_SYSTEM_PROMPT.format(
        service_name=config["name"],
        base_url=config["base_url"],
        extra_context=config["extra"],
    )


def _parse_react_message(text: str) -> dict[str, str | None]:
    action = _ACTION_RE.search(text)
    done = _DONE_RE.search(text)
    return {
        "action": action.group(1).strip() if action else None,
        "done": done.group(1).strip() if done else None,
    }


def _chat_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or not choices:
        return ""
    chunks: list[str] = []
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, Sequence) and not isinstance(
            content,
            (str, bytes),
        ):
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))
    return "".join(chunks)


@contextlib.contextmanager
def _sanitized_executor_environment(api_key_env: str) -> Iterable[None]:
    sensitive = {*_SENSITIVE_EXECUTOR_ENV, api_key_env}
    saved = {key: os.environ.get(key) for key in sensitive}
    try:
        for key in sensitive:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cache_neutral_cost(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    price = FROZEN_CACHE_NEUTRAL_PRICES.get(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000


def _usage_value(
    value: Any,
    name: str,
    *,
    fallback: str | None = None,
) -> Any:
    if isinstance(value, Mapping):
        observed = value.get(name)
        if observed is None and fallback is not None:
            observed = value.get(fallback)
        return observed
    observed = getattr(value, name, None)
    if observed is None and fallback is not None:
        observed = getattr(value, fallback, None)
    return observed


def _usage_int(
    value: Any,
    name: str,
    *,
    fallback: str | None = None,
) -> int | None:
    observed = _usage_value(value, name, fallback=fallback)
    if isinstance(observed, bool) or not isinstance(observed, int):
        return None
    return observed if observed >= 0 else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_exception_record(error: BaseException) -> dict[str, Any]:
    """Retain infra classification without persisting exception text/secrets."""

    record: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_message_persisted": False,
    }
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        record["http_status_code"] = status
    return record


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_invalid")
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise TypeError(f"{field_name}_invalid")
    return dict(parsed)


def _path_content_sha256(path: Path) -> str:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return _file_sha256(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError("agent_diff_live_runtime_asset_missing")
    return _source_bundle_sha256(
        root=resolved,
        paths=tuple(
            item
            for item in resolved.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts
        ),
    )


def _source_bundle_sha256(
    *,
    root: Path,
    paths: Sequence[Path],
) -> str:
    source_root = Path(root).resolve()
    resolved = tuple(sorted({Path(path).resolve() for path in paths}))
    if not resolved:
        raise AgentDiffLiveContractError(
            "agent_diff_live_source_bundle_empty"
        )
    manifest: list[dict[str, str]] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(
                f"agent_diff_live_source_bundle_file_missing:{path}"
            )
        try:
            label = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise AgentDiffLiveContractError(
                "agent_diff_live_source_bundle_path_outside_root"
            ) from exc
        manifest.append({"path": label, "sha256": _file_sha256(path)})
    return _digest(manifest)


def _required_glob(
    root: Path,
    pattern: str,
    label: str,
) -> tuple[Path, ...]:
    paths = tuple(
        sorted(path for path in Path(root).glob(pattern) if path.is_file())
    )
    if not paths:
        raise FileNotFoundError(f"{label}_missing")
    return paths


def _optional_paths(
    root: Path,
    relatives: Sequence[str],
) -> tuple[Path, ...]:
    return tuple(
        path
        for relative in relatives
        if (path := Path(root) / relative).is_file()
    )


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(repo).resolve()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("agent_diff_live_git_probe_failed")
    return completed.stdout.strip()


def _write_private_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    encoded = (
        json.dumps(
            dict(payload),
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
    records: Iterable[Mapping[str, Any]],
) -> str:
    resolved = tuple(records)
    if not resolved:
        raise AgentDiffLiveContractError(
            "agent_diff_live_private_manifest_empty"
        )
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


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes_no_replace(path, encoded)


def _write_bytes_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_mapping(path: Path, field_name: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field_name}_invalid")
    return dict(raw)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("agent_diff_live_jsonl_row_invalid")
    return rows


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name}_invalid")
    return dict(value)


def _sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name}_invalid")
    return tuple(value)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _require_sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name}_invalid")
    return text


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

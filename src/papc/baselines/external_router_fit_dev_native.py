"""Native pristine-episode execution for shared external-router fit/dev data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from papc.baselines.agent_diff_live import (
    AgentDiffPrivateTask,
    run_agent_diff_history_fit_dev_episode,
    run_agent_diff_pristine_fit_dev_episode,
)
from papc.baselines.agent_diff_router_driver import (
    AgentDiffNativeAPI,
    AgentDiffNativeRunConfig,
    AgentDiffOfficialTaskRef,
    load_agent_diff_native_api,
)
from papc.baselines.external_router_execution import (
    AGENT_DIFF,
    STATE_BENCH,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    TURN_STAGE,
    ExecutionContractError,
    ExecutionJob,
    ProviderAttemptBroker,
)
from papc.baselines.external_router_fit_dev import (
    FitDevSourceRow,
    StageItemExecutor,
    StageItemInput,
    StageItemResult,
)
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
    HistoryRouterView,
    RouterArtifactBinding,
)
from papc.baselines.router_policies import RouterDecision
from papc.baselines.state_bench_live import (
    StateBenchNativeClients,
    build_state_bench_native_clients,
    validate_state_bench_runtime_environment,
)
from papc.baselines.state_bench_native_runtime import (
    StateBenchNativeAPI,
    StateBenchNativeExecutor,
    StateBenchNativeRunConfig,
    StateBenchRuntimePolicy,
    load_state_bench_native_api,
    state_bench_generated_answer_view,
)
from papc.baselines.state_bench_router_driver import (
    StateBenchOfficialTaskRef,
)
from papc.baselines.tau2_native_runtime import (
    Tau2NativeAPI,
    Tau2NativeExecutor,
    Tau2NativeRunConfig,
    load_tau2_native_api,
    tau2_generated_answer_view,
)
from papc.baselines.tau2_router_driver import Tau2OfficialTaskRef

NATIVE_FIT_DEV_AUTHORITY_SCHEMA = (
    "papc.external_router.fit_dev_native_authority.v2"
)
NATIVE_SHARED_ENDPOINT_KINDS = frozenset({"shared_endpoint_episode"})
FROZEN_TAU2_DEPENDENCY_PACKAGES = {
    "addict": "2.4.0",
    "deepdiff": "8.6.2",
    "docstring-parser": "0.17.0",
    "fastapi": "0.121.2",
    "httpx": "0.28.1",
    "litellm": "1.81.11",
    "loguru": "0.7.3",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "psutil": "7.1.3",
    "python-dotenv": "1.2.1",
    "pyyaml": "6.0.3",
    "requests": "2.32.5",
    "rich": "14.2.0",
    "tabulate": "0.9.0",
    "tau2": "1.0.0",
    "tenacity": "9.1.2",
    "toml": "0.10.2",
    "typer": "0.20.0",
    "uvicorn": "0.38.0",
}
FROZEN_TAU2_UV_LOCK_SHA256 = (
    "62d3a8c4807b89e85703b3c03f2c21048a2da9736ca83d9af6f61adc74ac5314"
)
FROZEN_TAU2_DEPENDENCY_ENVIRONMENT_SHA256 = (
    "0b94fd74b883a2ecc52f8dcb0476637d51d29f39ae24fff5302a48359abd4730"
)

ProviderClientFactory = Callable[..., Any]
NativeHistoryDecision = Callable[[HistoryRouterView], RouterDecision]


@dataclass(frozen=True)
class NativeFitDevHistoryEpisode:
    """Official fit/dev history episode returned to TURN_STAGE adapters."""

    official_score: float
    decisions: tuple[RouterDecision, ...]
    public_episode: tuple[dict[str, Any], ...] = field(repr=False)

    def __post_init__(self) -> None:
        score = float(self.official_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("fit_dev_native_history_score_invalid")
        decisions = tuple(self.decisions)
        if not decisions or any(
            not isinstance(item, RouterDecision) for item in decisions
        ):
            raise ValueError("fit_dev_native_history_decisions_invalid")
        episode = tuple(dict(item) for item in self.public_episode)
        if not episode:
            raise ValueError("fit_dev_native_history_episode_empty")
        object.__setattr__(self, "official_score", score)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "public_episode", episode)


NativeFitDevHistoryExecutor = Callable[
    [
        ExecutionJob,
        FitDevSourceRow,
        RouterArtifactBinding,
        NativeHistoryDecision,
        ProviderAttemptBroker,
    ],
    NativeFitDevHistoryEpisode,
]


@dataclass(frozen=True)
class NativeFitDevAuthority:
    """Provider-zero binding for only the selected native train surfaces."""

    surfaces: tuple[str, ...]
    private_root: Path = field(repr=False)
    agent_diff_dataset: Path = field(repr=False)
    agent_diff_base_url: str
    state_api: StateBenchNativeAPI | None = field(default=None, repr=False)
    state_runtime_environment_sha256: str | None = None
    tau2_api: Tau2NativeAPI | None = field(default=None, repr=False)
    tau2_dependency_environment_sha256: str | None = None
    tau2_payload_sha256s: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict,
        repr=False,
    )
    agent_diff_api: AgentDiffNativeAPI | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        surfaces = tuple(sorted(set(self.surfaces)))
        allowed = {STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE, AGENT_DIFF}
        if not surfaces or set(surfaces) - allowed:
            raise ValueError("fit_dev_native_authority_surfaces_invalid")
        object.__setattr__(self, "surfaces", surfaces)
        for field_name in ("private_root", "agent_diff_dataset"):
            path = Path(getattr(self, field_name)).resolve()
            if not path.is_absolute():
                raise ValueError(f"fit_dev_native_{field_name}_not_absolute")
            object.__setattr__(self, field_name, path)
        if not self.agent_diff_base_url.startswith(("http://", "https://")):
            raise ValueError("fit_dev_native_agent_diff_base_url_invalid")
        if STATE_BENCH in surfaces and self.state_api is None:
            raise ValueError("fit_dev_native_state_authority_missing")
        if (
            {TAU2_RETAIL, TAU2_AIRLINE} & set(surfaces)
            and self.tau2_api is None
        ):
            raise ValueError("fit_dev_native_tau2_authority_missing")
        if AGENT_DIFF in surfaces and self.agent_diff_api is None:
            raise ValueError("fit_dev_native_agent_diff_authority_missing")
        object.__setattr__(
            self,
            "tau2_payload_sha256s",
            {
                domain: dict(values)
                for domain, values in self.tau2_payload_sha256s.items()
            },
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": NATIVE_FIT_DEV_AUTHORITY_SCHEMA,
            "surfaces": list(self.surfaces),
            "state": (
                _fit_dev_runtime_public_record(
                    self.state_api.public_record()
                )
                if self.state_api is not None
                else None
            ),
            "state_runtime_environment_sha256": (
                self.state_runtime_environment_sha256
            ),
            "tau2": (
                _fit_dev_runtime_public_record(
                    self.tau2_api.public_record()
                )
                if self.tau2_api is not None
                else None
            ),
            "tau2_dependency_environment_sha256": (
                self.tau2_dependency_environment_sha256
            ),
            "tau2_train_payload_counts": {
                domain: len(values)
                for domain, values in self.tau2_payload_sha256s.items()
            },
            "agent_diff": (
                self.agent_diff_api.public_record()
                if self.agent_diff_api is not None
                else None
            ),
            "official_train_only": True,
            "official_test_outcomes_read": False,
            "provider_calls": 0,
            "secret_reads": 0,
        }


def _fit_dev_runtime_public_record(
    runtime_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(runtime_record)
    protocol_split = record.pop("official_split", None)
    if protocol_split != "test":
        raise ValueError(
            "fit_dev_runtime_protocol_default_split_not_test"
        )
    record["runtime_protocol_default_split"] = protocol_split
    record["fit_dev_execution_split"] = "train"
    return record


def load_native_fit_dev_authority(
    *,
    surfaces: Sequence[str],
    private_root: Path,
    state_repo: Path,
    tau2_data_repo: Path,
    tau2_runtime_repo: Path,
    agent_diff_repo: Path,
    agent_diff_dataset: Path,
    agent_diff_sdk_path: Path,
    agent_diff_scorer_path: Path,
    agent_diff_baseline_authority_path: Path,
    agent_diff_base_url: str,
) -> NativeFitDevAuthority:
    """Bind frozen native code and train data without reading a secret."""

    selected = tuple(sorted(set(surfaces)))
    state_api: StateBenchNativeAPI | None = None
    state_environment: str | None = None
    tau2_api: Tau2NativeAPI | None = None
    tau2_environment: str | None = None
    tau2_payloads: dict[str, dict[str, str]] = {}
    agent_api: AgentDiffNativeAPI | None = None

    if STATE_BENCH in selected:
        state_api = load_state_bench_native_api(state_repo)
        state_environment = validate_state_bench_runtime_environment(
            state_api
        )

    tau_surfaces = {TAU2_RETAIL, TAU2_AIRLINE} & set(selected)
    if tau_surfaces:
        tau2_environment = activate_tau2_dependency_environment(
            Path(tau2_runtime_repo).resolve()
        )
        tau2_api = load_tau2_native_api(tau2_runtime_repo)
        for surface in sorted(tau_surfaces):
            domain = "retail" if surface == TAU2_RETAIL else "airline"
            tau2_payloads[domain] = _bind_tau2_train_payloads(
                data_repo=Path(tau2_data_repo).resolve(),
                runtime_repo=tau2_api.runtime_repo,
                domain=domain,
            )

    if AGENT_DIFF in selected:
        repo = Path(agent_diff_repo).resolve()
        dataset = Path(agent_diff_dataset).resolve()
        if not dataset.is_relative_to(repo):
            raise ValueError("fit_dev_native_agent_diff_dataset_outside_repo")
        agent_api = load_agent_diff_native_api(
            sdk_path=agent_diff_sdk_path,
            scorer_path=agent_diff_scorer_path,
            baseline_authority_path=agent_diff_baseline_authority_path,
        )

    return NativeFitDevAuthority(
        surfaces=selected,
        private_root=Path(private_root).resolve(),
        agent_diff_dataset=Path(agent_diff_dataset).resolve(),
        agent_diff_base_url=agent_diff_base_url.rstrip("/"),
        state_api=state_api,
        state_runtime_environment_sha256=state_environment,
        tau2_api=tau2_api,
        tau2_dependency_environment_sha256=tau2_environment,
        tau2_payload_sha256s=tau2_payloads,
        agent_diff_api=agent_api,
    )


def build_native_shared_endpoint_executor(
    *,
    authority: NativeFitDevAuthority,
    api_base: str,
    api_key: str,
    max_retries: int,
    max_observed_provider_cost_usd: float,
    api_key_env: str = "OPENAI_API_KEY",
    client_factory: ProviderClientFactory | None = None,
) -> StageItemExecutor:
    """Construct the native shared-endpoint executor without a provider call."""

    if not isinstance(authority, NativeFitDevAuthority):
        raise TypeError("fit_dev_native_authority_invalid")
    if not isinstance(api_base, str) or not api_base.startswith(
        ("http://", "https://")
    ):
        raise ValueError("fit_dev_native_api_base_invalid")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("fit_dev_native_api_key_invalid")
    if isinstance(max_retries, bool) or max_retries != 0:
        raise ValueError("fit_dev_native_provider_retries_must_be_zero")
    if (
        isinstance(max_observed_provider_cost_usd, bool)
        or not isinstance(max_observed_provider_cost_usd, (int, float))
        or float(max_observed_provider_cost_usd) <= 0.0
    ):
        raise ValueError("fit_dev_native_observed_cost_stop_invalid")
    if api_key_env != "OPENAI_API_KEY":
        raise ValueError("fit_dev_native_api_key_env_not_frozen")

    factory = client_factory
    if factory is None:
        from openai import OpenAI

        factory = OpenAI

    state_clients = (
        _build_state_clients(
            api=authority.state_api,
            api_base=api_base,
            api_key=api_key,
            client_factory=factory,
        )
        if authority.state_api is not None
        else None
    )
    agent_provider_client = (
        factory(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            max_retries=0,
        )
        if authority.agent_diff_api is not None
        else None
    )

    def execute(
        job: ExecutionJob,
        item: StageItemInput,
        broker: ProviderAttemptBroker,
    ) -> StageItemResult:
        if (
            job.kind != "shared_endpoint_episode"
            or item.source is None
            or job.surface not in authority.surfaces
        ):
            raise ExecutionContractError(
                "fit_dev_native_shared_job_binding_invalid"
            )
        if not isinstance(broker, ProviderAttemptBroker):
            raise TypeError("fit_dev_native_attempt_broker_invalid")
        role, model = _job_role_model(job)
        if job.surface == STATE_BENCH:
            generated, score = _run_state(
                authority=authority,
                clients=state_clients,
                job=job,
                source=item.source,
                role=role,
                model=model,
                broker=broker,
                api_base=api_base,
                max_observed_cost=max_observed_provider_cost_usd,
            )
        elif job.surface in {TAU2_RETAIL, TAU2_AIRLINE}:
            bound_initial_user_message = None
            if role == "strong":
                upstream = item.upstream_payload
                candidate = (
                    upstream.get("original_request")
                    if upstream is not None
                    else None
                )
                if (
                    not isinstance(candidate, str)
                    or not candidate.strip()
                ):
                    raise ExecutionContractError(
                        "fit_dev_native_tau2_paired_request_missing"
                    )
                bound_initial_user_message = candidate
            generated, score = _run_tau2(
                authority=authority,
                job=job,
                source=item.source,
                role=role,
                model=model,
                broker=broker,
                api_base=api_base,
                max_observed_cost=max_observed_provider_cost_usd,
                bound_initial_user_message=(
                    bound_initial_user_message
                ),
            )
        elif job.surface == AGENT_DIFF:
            generated, score = _run_agent_diff(
                authority=authority,
                provider_client=agent_provider_client,
                job=job,
                source=item.source,
                role=role,
                model=model,
                broker=broker,
                api_base=api_base,
                api_key_env=api_key_env,
                max_observed_cost=max_observed_provider_cost_usd,
            )
        else:
            raise ExecutionContractError(
                "fit_dev_native_surface_not_supported"
            )
        return StageItemResult(
            output={
                "model_role": role,
                "original_request": generated["original_request"],
                "generated_answer": generated["generated_answer"],
                "public_episode": generated["public_episode"],
                "official_score": score,
            },
            receipts=tuple(generated["receipts"]),
        )

    return execute


def build_native_fit_dev_history_executor(
    *,
    authority: NativeFitDevAuthority,
    api_base: str,
    api_key: str,
    max_retries: int,
    max_observed_provider_cost_usd: float,
    api_key_env: str = "OPENAI_API_KEY",
    client_factory: ProviderClientFactory | None = None,
) -> NativeFitDevHistoryExecutor:
    """Construct source-bound turn execution without making a provider call."""

    if not isinstance(authority, NativeFitDevAuthority):
        raise TypeError("fit_dev_native_authority_invalid")
    if not isinstance(api_base, str) or not api_base.startswith(
        ("http://", "https://")
    ):
        raise ValueError("fit_dev_native_api_base_invalid")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("fit_dev_native_api_key_invalid")
    if isinstance(max_retries, bool) or max_retries != 0:
        raise ValueError("fit_dev_native_provider_retries_must_be_zero")
    if (
        isinstance(max_observed_provider_cost_usd, bool)
        or not isinstance(max_observed_provider_cost_usd, (int, float))
        or float(max_observed_provider_cost_usd) <= 0.0
    ):
        raise ValueError("fit_dev_native_observed_cost_stop_invalid")
    if api_key_env != "OPENAI_API_KEY":
        raise ValueError("fit_dev_native_api_key_env_not_frozen")
    factory = client_factory
    if factory is None:
        from openai import OpenAI

        factory = OpenAI
    state_clients = (
        _build_state_clients(
            api=authority.state_api,
            api_base=api_base,
            api_key=api_key,
            client_factory=factory,
        )
        if authority.state_api is not None
        else None
    )
    agent_provider_client = (
        factory(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            max_retries=0,
        )
        if authority.agent_diff_api is not None
        else None
    )

    def execute(
        job: ExecutionJob,
        source: FitDevSourceRow,
        artifact: RouterArtifactBinding,
        decide_next: NativeHistoryDecision,
        broker: ProviderAttemptBroker,
    ) -> NativeFitDevHistoryEpisode:
        if (
            job.stage != TURN_STAGE
            or source.surface != job.surface
            or artifact.surface != job.surface
            or not callable(decide_next)
            or not isinstance(broker, ProviderAttemptBroker)
        ):
            raise ExecutionContractError(
                "fit_dev_native_history_job_binding_invalid"
            )
        if job.surface == STATE_BENCH:
            return _run_state_history(
                authority=authority,
                clients=state_clients,
                job=job,
                source=source,
                artifact=artifact,
                decide_next=decide_next,
                broker=broker,
                api_base=api_base,
                max_observed_cost=max_observed_provider_cost_usd,
            )
        if job.surface in {TAU2_RETAIL, TAU2_AIRLINE}:
            executor = build_tau2_fit_dev_pristine_executor(
                authority=authority,
                job=job,
                source=source,
                broker=broker,
                api_base=api_base,
                max_observed_cost=max_observed_provider_cost_usd,
            )
            native = executor.run_fit_dev_history_routed(
                artifact=artifact,
                decide_next=decide_next,
            )
            return NativeFitDevHistoryEpisode(
                official_score=native.official_score,
                decisions=native.decisions,
                public_episode=native.public_transcript,
            )
        if job.surface == AGENT_DIFF:
            api = authority.agent_diff_api
            if api is None or agent_provider_client is None:
                raise RuntimeError(
                    "fit_dev_native_agent_diff_runtime_unavailable"
                )
            task = _load_agent_diff_train_task(
                authority.agent_diff_dataset,
                source,
            )
            routed, score = run_agent_diff_history_fit_dev_episode(
                task=task,
                artifact=artifact,
                decide_next=decide_next,
                api=api,
                config=AgentDiffNativeRunConfig(
                    api_base=api_base,
                    agent_diff_base_url=authority.agent_diff_base_url,
                    private_output_root=_native_output_root(job),
                    max_provider_calls=job.attempts_per_item,
                    max_observed_provider_cost_usd=(
                        max_observed_provider_cost_usd
                    ),
                ),
                provider_client=agent_provider_client,
                broker=broker,
                raw_path=(
                    _native_output_root(job)
                    / source.example_sha256
                    / f"{artifact.method}.json"
                ),
                api_key_env=api_key_env,
            )
            return NativeFitDevHistoryEpisode(
                official_score=score,
                decisions=tuple(routed.decisions),
                public_episode=tuple(routed.final_public_history),
            )
        raise ExecutionContractError(
            "fit_dev_native_history_surface_not_supported"
        )

    return execute


def _run_state_history(
    *,
    authority: NativeFitDevAuthority,
    clients: StateBenchNativeClients | None,
    job: ExecutionJob,
    source: FitDevSourceRow,
    artifact: RouterArtifactBinding,
    decide_next: NativeHistoryDecision,
    broker: ProviderAttemptBroker,
    api_base: str,
    max_observed_cost: float,
) -> NativeFitDevHistoryEpisode:
    api = authority.state_api
    if api is None or clients is None:
        raise RuntimeError("fit_dev_native_state_runtime_unavailable")
    task = _state_fit_dev_task(api=api, source=source)
    domain = api.get_domain_config(task.domain)
    policy = StateBenchRuntimePolicy(
        cheap_client=clients.cheap,
        strong_client=clients.strong,
        simulator_client=clients.simulator,
        evaluator_client=clients.evaluator,
        task_requirements_judge=api.task_requirements_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
        ux_judge=api.ux_quality_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
    )
    executor = StateBenchNativeExecutor.for_fit_dev_pristine_episode(
        api=api,
        task=task,
        config=StateBenchNativeRunConfig(
            api_base=api_base,
            private_output_root=_native_output_root(job),
            max_provider_calls=job.attempts_per_item,
            max_observed_provider_cost_usd=max_observed_cost,
        ),
        policy=policy,
        registry_sha256=job.registry_sha256,
        attempt_broker=broker,
    )
    native = executor.run_fit_dev_history_routed(
        artifact=artifact,
        decide_next=decide_next,
    )
    score = native.score_handle.score()
    return NativeFitDevHistoryEpisode(
        official_score=score,
        decisions=native.decisions,
        public_episode=native.public_transcript,
    )


def _build_state_clients(
    *,
    api: StateBenchNativeAPI | None,
    api_base: str,
    api_key: str,
    client_factory: ProviderClientFactory,
) -> StateBenchNativeClients:
    if api is None:
        raise ValueError("fit_dev_native_state_api_missing")

    def raw_factory(base: str, key: str) -> Any:
        return client_factory(
            api_key=key,
            base_url=base.rstrip("/"),
            max_retries=0,
        )

    return build_state_bench_native_clients(
        api=api,
        api_base=api_base.rstrip("/"),
        api_key=api_key,
        raw_client_factory=raw_factory,
    )


def _run_state(
    *,
    authority: NativeFitDevAuthority,
    clients: StateBenchNativeClients | None,
    job: ExecutionJob,
    source: FitDevSourceRow,
    role: str,
    model: str,
    broker: ProviderAttemptBroker,
    api_base: str,
    max_observed_cost: float,
) -> tuple[dict[str, Any], float]:
    api = authority.state_api
    if api is None or clients is None:
        raise RuntimeError("fit_dev_native_state_runtime_unavailable")
    task = _state_fit_dev_task(api=api, source=source)
    domain = api.get_domain_config(task.domain)
    policy = StateBenchRuntimePolicy(
        cheap_client=clients.cheap,
        strong_client=clients.strong,
        simulator_client=clients.simulator,
        evaluator_client=clients.evaluator,
        task_requirements_judge=api.task_requirements_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
        ux_judge=api.ux_quality_judge_class(
            client=clients.evaluator,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
            reasoning_effort=REASONING_EFFORT,
        ),
    )
    executor = StateBenchNativeExecutor.for_fit_dev_pristine_episode(
        api=api,
        task=task,
        config=StateBenchNativeRunConfig(
            api_base=api_base,
            private_output_root=_native_output_root(job),
            max_provider_calls=job.attempts_per_item,
            max_observed_provider_cost_usd=max_observed_cost,
        ),
        policy=policy,
        registry_sha256=job.registry_sha256,
        attempt_broker=broker,
    )
    native = executor.run_single_model(role=role, model=model)
    score = native.score_handle.score()
    return (
        {
            "original_request": native.initial_public_user_message,
            "generated_answer": state_bench_generated_answer_view(
                native.public_transcript
            ),
            "public_episode": [
                dict(turn) for turn in native.public_transcript
            ],
            "receipts": tuple(executor.budget.receipts),
        },
        score,
    )


def _state_fit_dev_task(
    *,
    api: StateBenchNativeAPI,
    source: FitDevSourceRow,
) -> StateBenchOfficialTaskRef:
    locator = dict(source.native_locator)
    task_path = Path(locator["task_path"]).resolve()
    environment_path = Path(locator["initial_environment_path"]).resolve()
    task = StateBenchOfficialTaskRef(
        domain=locator["domain"],
        native_task_locator=locator["task_id"],
        native_task_path=task_path,
        initial_environment_path=environment_path,
        payload_sha256=source.payload_sha256,
        initial_environment_sha256=_digest(_read_mapping(environment_path)),
        example_sha256=source.example_sha256,
    )
    loaded = api.task_definition_class.load(task_path)
    if str(getattr(loaded, "task_id", "")) != task.native_task_locator:
        raise ExecutionContractError(
            "fit_dev_native_state_source_payload_mismatch"
        )
    return task


def _run_tau2(
    *,
    authority: NativeFitDevAuthority,
    job: ExecutionJob,
    source: FitDevSourceRow,
    role: str,
    model: str,
    broker: ProviderAttemptBroker,
    api_base: str,
    max_observed_cost: float,
    bound_initial_user_message: str | None,
) -> tuple[dict[str, Any], float]:
    executor = build_tau2_fit_dev_pristine_executor(
        authority=authority,
        job=job,
        source=source,
        broker=broker,
        api_base=api_base,
        max_observed_cost=max_observed_cost,
        bound_initial_user_message=bound_initial_user_message,
    )
    native = executor.run_single_model(role=role, model=model)
    return (
        {
            "original_request": native.initial_public_user_message,
            "generated_answer": tau2_generated_answer_view(
                native.public_transcript
            ),
            "public_episode": [
                dict(turn) for turn in native.public_transcript
            ],
            "receipts": native.receipts,
        },
        native.official_score,
    )


def build_tau2_fit_dev_pristine_executor(
    *,
    authority: NativeFitDevAuthority,
    job: ExecutionJob,
    source: FitDevSourceRow,
    broker: ProviderAttemptBroker,
    api_base: str,
    max_observed_cost: float,
    bound_initial_user_message: str | None = None,
) -> Tau2NativeExecutor:
    """Bind one frozen Tau2 train row for a native injected agent."""

    api = authority.tau2_api
    if api is None:
        raise RuntimeError("fit_dev_native_tau2_runtime_unavailable")
    domain = "retail" if job.surface == TAU2_RETAIL else "airline"
    locator = dict(source.native_locator)
    if (
        locator.get("domain") != domain
        or authority.tau2_payload_sha256s.get(domain, {}).get(
            locator.get("task_id", "")
        )
        != source.payload_sha256
    ):
        raise ExecutionContractError(
            "fit_dev_native_tau2_source_payload_mismatch"
        )
    task = Tau2OfficialTaskRef(
        native_task_locator=locator["task_id"],
        payload_sha256=source.payload_sha256,
        example_sha256=source.example_sha256,
    )
    return Tau2NativeExecutor.for_fit_dev_pristine_episode(
        api=api,
        task=task,
        domain=domain,
        config=Tau2NativeRunConfig(
            api_base=api_base,
            private_output_root=_native_output_root(job),
            max_provider_calls=job.attempts_per_item,
            max_observed_provider_cost_usd=max_observed_cost,
        ),
        source_binding_sha256=job.source_binding_sha256,
        attempt_broker=broker,
        bound_initial_user_message=bound_initial_user_message,
    )


def _run_agent_diff(
    *,
    authority: NativeFitDevAuthority,
    provider_client: Any,
    job: ExecutionJob,
    source: FitDevSourceRow,
    role: str,
    model: str,
    broker: ProviderAttemptBroker,
    api_base: str,
    api_key_env: str,
    max_observed_cost: float,
) -> tuple[dict[str, Any], float]:
    api = authority.agent_diff_api
    if api is None or provider_client is None:
        raise RuntimeError("fit_dev_native_agent_diff_runtime_unavailable")
    task = _load_agent_diff_train_task(
        authority.agent_diff_dataset,
        source,
    )
    generated, score = run_agent_diff_pristine_fit_dev_episode(
        task=task,
        role=role,
        model=model,
        api=api,
        config=AgentDiffNativeRunConfig(
            api_base=api_base,
            agent_diff_base_url=authority.agent_diff_base_url,
            private_output_root=_native_output_root(job),
            max_provider_calls=job.attempts_per_item,
            max_observed_provider_cost_usd=max_observed_cost,
        ),
        provider_client=provider_client,
        broker=broker,
        raw_path=(
            _native_output_root(job)
            / source.example_sha256
            / f"{role}.json"
        ),
        api_key_env=api_key_env,
    )
    return (
        {
            "original_request": task.ref.original_request,
            "generated_answer": generated.generated_answer_view,
            "public_episode": [
                dict(turn) for turn in generated.public_episode_history
            ],
            "receipts": generated.receipts,
        },
        score,
    )


def _load_agent_diff_train_task(
    dataset: Path,
    source: FitDevSourceRow,
) -> AgentDiffPrivateTask:
    locator = dict(source.native_locator)
    train_path = Path(locator["train_jsonl"]).resolve()
    expected_path = (Path(dataset).resolve() / "train.jsonl").resolve()
    if train_path != expected_path:
        raise ExecutionContractError(
            "fit_dev_native_agent_diff_train_path_mismatch"
        )
    ordinal = int(locator["train_ordinal"])
    rows = _read_jsonl(train_path)
    if ordinal <= 0 or ordinal > len(rows):
        raise ExecutionContractError(
            "fit_dev_native_agent_diff_train_ordinal_invalid"
        )
    raw = rows[ordinal - 1]
    if (
        _digest(raw) != source.payload_sha256
        or raw.get("test_id") != locator["test_id"]
        or raw.get("service") != locator["service"]
        or raw.get("question") != source.original_request
    ):
        raise ExecutionContractError(
            "fit_dev_native_agent_diff_source_payload_mismatch"
        )
    ref = AgentDiffOfficialTaskRef(
        native_ordinal=ordinal,
        native_test_id=_required_text(raw.get("test_id")),
        service=_required_text(raw.get("service")),
        original_request=source.original_request,
        payload_sha256=source.payload_sha256,
        example_sha256=source.example_sha256,
    )
    return AgentDiffPrivateTask(
        ref=ref,
        test_id=ref.native_test_id,
        test_name=_required_text(raw.get("test_name")),
        answer=_json_object(raw.get("answer")),
        info=_json_object(raw.get("info")),
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


def _bind_tau2_train_payloads(
    *,
    data_repo: Path,
    runtime_repo: Path,
    domain: str,
) -> dict[str, str]:
    relative = Path("data/tau2/domains") / domain
    data_tasks = data_repo / relative / "tasks.json"
    data_split = data_repo / relative / "split_tasks.json"
    runtime_tasks = runtime_repo / relative / "tasks.json"
    runtime_split = runtime_repo / relative / "split_tasks.json"
    for data_path, runtime_path in (
        (data_tasks, runtime_tasks),
        (data_split, runtime_split),
    ):
        if (
            not data_path.is_file()
            or not runtime_path.is_file()
            or hashlib.sha256(data_path.read_bytes()).digest()
            != hashlib.sha256(runtime_path.read_bytes()).digest()
        ):
            raise ValueError(
                f"fit_dev_native_tau2_{domain}_runtime_data_mismatch"
            )
    tasks = json.loads(data_tasks.read_text(encoding="utf-8"))
    split = _read_mapping(data_split)
    raw_train = split.get("train")
    if not isinstance(tasks, list) or not isinstance(raw_train, list):
        raise TypeError("fit_dev_native_tau2_train_source_invalid")
    by_id = {
        _required_text(task.get("id")): _digest(task)
        for task in tasks
        if isinstance(task, Mapping)
    }
    if any(str(task_id) not in by_id for task_id in raw_train):
        raise ValueError("fit_dev_native_tau2_train_task_missing")
    return {str(task_id): by_id[str(task_id)] for task_id in raw_train}


def activate_tau2_dependency_environment(runtime_repo: Path) -> str:
    """Validate and append the frozen tau2 dependency overlay."""

    repo = Path(runtime_repo).resolve()
    site_packages = (
        repo / ".venv/lib/python3.12/site-packages"
    ).resolve()
    uv_lock = repo / "uv.lock"
    if not site_packages.is_dir() or not uv_lock.is_file():
        raise FileNotFoundError(
            "fit_dev_native_tau2_dependency_environment_missing"
        )
    distributions = {
        _canonical_package_name(item.metadata["Name"]): item.version
        for item in importlib.metadata.distributions(
            path=[str(site_packages)]
        )
        if item.metadata["Name"]
    }
    observed_packages = {
        name: distributions.get(name)
        for name in FROZEN_TAU2_DEPENDENCY_PACKAGES
    }
    if observed_packages != FROZEN_TAU2_DEPENDENCY_PACKAGES:
        raise ValueError(
            "fit_dev_native_tau2_dependency_versions_mismatch"
        )
    record = {
        "python": platform.python_version(),
        "site_packages": str(site_packages),
        "uv_lock_sha256": hashlib.sha256(
            uv_lock.read_bytes()
        ).hexdigest(),
        "packages": observed_packages,
    }
    if record["uv_lock_sha256"] != FROZEN_TAU2_UV_LOCK_SHA256:
        raise ValueError("fit_dev_native_tau2_uv_lock_mismatch")
    environment_sha256 = _digest(record)
    if (
        environment_sha256
        != FROZEN_TAU2_DEPENDENCY_ENVIRONMENT_SHA256
    ):
        raise ValueError(
            "fit_dev_native_tau2_dependency_environment_mismatch"
        )
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))
    return environment_sha256


def _job_role_model(job: ExecutionJob) -> tuple[str, str]:
    role = dict(job.details).get("model_role")
    expected_method = {
        "cheap": "shared_cheap_endpoint",
        "strong": "shared_strong_endpoint",
    }
    if role not in expected_method or job.method != expected_method[role]:
        raise ExecutionContractError(
            "fit_dev_native_shared_role_binding_invalid"
        )
    return role, CHEAP_MODEL if role == "cheap" else STRONG_MODEL


def _native_output_root(job: ExecutionJob) -> Path:
    return (
        job.output_path.parent
        / f".{job.output_path.stem}.native"
    ).resolve()


def _json_object(value: Any) -> dict[str, Any]:
    resolved = json.loads(value) if isinstance(value, str) else value
    if not isinstance(resolved, Mapping):
        raise TypeError("fit_dev_native_agent_diff_json_object_invalid")
    return dict(resolved)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("fit_dev_native_json_mapping_invalid")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("fit_dev_native_jsonl_row_invalid")
    return rows


def _required_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fit_dev_native_required_text_invalid")
    return value.strip()


def _canonical_package_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "NATIVE_FIT_DEV_AUTHORITY_SCHEMA",
    "NATIVE_SHARED_ENDPOINT_KINDS",
    "FROZEN_TAU2_DEPENDENCY_ENVIRONMENT_SHA256",
    "activate_tau2_dependency_environment",
    "NativeFitDevAuthority",
    "build_native_shared_endpoint_executor",
    "load_native_fit_dev_authority",
]

"""Official STATE-Bench native runtime boundary for external routers.

This module binds the benchmark-owned agent, user simulator, termination rule,
and scorer.  Provider clients and fitted router policies are deliberately
injected.  Until that policy bundle is supplied, the companion preflight
fails closed without reading a secret or making a provider call.
"""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from papc.baselines.external_router_execution import ProviderAttemptBroker
from papc.baselines.router_accounting import (
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    ModelPrice,
)
from papc.baselines.router_contract import (
    AUTOMIX,
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM_EMBEDDING_MODEL,
    STRONG_MODEL,
    HistoryRouterView,
    ModelRole,
    RouterArtifactBinding,
    validate_public_history,
)
from papc.baselines.router_policies import RouterDecision
from papc.baselines.router_runtime import GeneratedEpisode
from papc.baselines.state_bench_router_driver import (
    STATE_BENCH_TURN_LEVEL_METHODS,
    StateBenchOfficialTaskRef,
    StateBenchRouterDriver,
    load_state_bench_official_test_registry,
)
from papc.baselines.surface_adapters import state_bench_native_termination

FROZEN_STATE_BENCH_RUNTIME_REVISION = "83cb96de5429c43adfdb5cb9b6785439e937a3ca"
FROZEN_STATE_BENCH_RUNTIME_SOURCE_BINDING_SHA256 = (
    "7279b30456a12e9c2c6bacae58a1a89c55740cba5b6468c1d937bb21bbc2eac3"
)
FROZEN_STATE_BENCH_EXECUTION_SOURCE_BINDING_SHA256 = (
    "11dbb9e3edf3ccb6a45bb0a1e37a5d93616fa618932c221fcb5b1e20fdfdeede"
)
STATE_BENCH_OFFICIAL_SPLIT = "test"
STATE_BENCH_OFFICIAL_SPLIT_VERSION = "train_test"
STATE_BENCH_NATIVE_AGENT = "StateBenchAgent"
STATE_BENCH_NATIVE_USER = "UserSimulator"
STATE_BENCH_OFFICIAL_SCORER = "state_bench.scripts.score.score_one"
STATE_BENCH_USER_MODEL = "gpt-5.4"
STATE_BENCH_EVALUATOR_MODEL = "gpt-5.4"
STATE_BENCH_PRICING_SNAPSHOT_DATE = "2026-07-10"
STATE_BENCH_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
STATE_BENCH_HARNESS_MODEL_PRICE = ModelPrice(
    input_per_million_usd=2.5,
    output_per_million_usd=15.0,
)

_RUNTIME_SOURCE_FILES = (
    "state_bench/agents/base.py",
    "state_bench/agents/state_bench.py",
    "state_bench/client.py",
    "state_bench/domain.py",
    "state_bench/env_loader.py",
    "state_bench/orchestrator.py",
    "state_bench/protocol.py",
    "state_bench/scoring.py",
    "state_bench/scripts/score.py",
    "state_bench/simulator.py",
)
_RUNTIME_EXECUTION_CORE_FILES = (
    "pyproject.toml",
    "state_bench/__init__.py",
    "state_bench/agents/__init__.py",
    "state_bench/agents/base.py",
    "state_bench/agents/state_bench.py",
    "state_bench/client.py",
    "state_bench/configs/eval_protocols/gpt54.json",
    "state_bench/configs/llm.yaml",
    "state_bench/domain.py",
    "state_bench/env_loader.py",
    "state_bench/environment.py",
    "state_bench/orchestrator.py",
    "state_bench/paths.py",
    "state_bench/protocol.py",
    "state_bench/schemas.py",
    "state_bench/scoring.py",
    "state_bench/scripts/__init__.py",
    "state_bench/scripts/score.py",
    "state_bench/simulator.py",
    "state_bench/version.py",
)
_FIT_DEV_UNUSED_ROOT_EXTENSIONS = (
    "agents/papc_openai_compatible.py",
    "clients/papc_openai_compatible.py",
)


@dataclass(frozen=True)
class StateBenchNativeAPI:
    runtime_repo: Path
    runtime_revision: str
    runtime_worktree_dirty: bool
    source_binding_sha256: str
    execution_source_binding_sha256: str
    execution_source_file_sha256s: tuple[tuple[str, str], ...]
    dirty_path_records: tuple[Mapping[str, Any], ...]
    symbol_origins: tuple[tuple[str, str], ...]
    protocol_id: str
    protocol_num_runs: int
    run_task: Callable[..., Any] = field(repr=False)
    get_domain_config: Callable[[str], Any] = field(repr=False)
    load_task_environment: Callable[..., Any] = field(repr=False)
    task_definition_class: type = field(repr=False)
    base_agent_class: type = field(repr=False)
    agent_runtime_context_class: type = field(repr=False)
    state_bench_agent_class: type = field(repr=False)
    llm_client_class: type = field(repr=False)
    score_one: Callable[..., dict[str, Any]] = field(repr=False)
    task_requirements_judge_class: type = field(repr=False)
    ux_quality_judge_class: type = field(repr=False)
    protocol: Any = field(repr=False)

    def public_record(self) -> dict[str, Any]:
        return {
            "runtime_repo": str(self.runtime_repo),
            "runtime_revision": self.runtime_revision,
            "runtime_worktree_dirty": self.runtime_worktree_dirty,
            "runtime_source_binding_sha256": self.source_binding_sha256,
            "runtime_execution_source_binding_sha256": (
                self.execution_source_binding_sha256
            ),
            "runtime_execution_file_sha256s": dict(
                self.execution_source_file_sha256s
            ),
            "runtime_dirty_path_records": [
                dict(item) for item in self.dirty_path_records
            ],
            "runtime_symbol_origins": dict(self.symbol_origins),
            "fit_dev_root_extension_usage": {
                relative: False
                for relative in _FIT_DEV_UNUSED_ROOT_EXTENSIONS
            },
            "agent": STATE_BENCH_NATIVE_AGENT,
            "user_simulator": STATE_BENCH_NATIVE_USER,
            "official_scorer": STATE_BENCH_OFFICIAL_SCORER,
            "official_split": STATE_BENCH_OFFICIAL_SPLIT,
            "official_split_version": STATE_BENCH_OFFICIAL_SPLIT_VERSION,
            "protocol_id": self.protocol_id,
            "protocol_num_runs": self.protocol_num_runs,
            "protocol_prompt_hashes_valid": True,
        }


def load_state_bench_native_api(runtime_repo: Path) -> StateBenchNativeAPI:
    """Import and hash the native APIs without constructing provider clients."""

    repo = Path(runtime_repo).resolve()
    if not repo.is_dir():
        raise FileNotFoundError("state_bench_runtime_repo_missing")
    revision = _git_output(repo, "rev-parse", "HEAD")
    if revision != FROZEN_STATE_BENCH_RUNTIME_REVISION:
        raise ValueError("state_bench_runtime_revision_not_frozen")
    source_paths = tuple(repo / relative for relative in _RUNTIME_SOURCE_FILES)
    if any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("state_bench_native_source_file_missing")

    loaded = sys.modules.get("state_bench")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if not loaded_path.is_relative_to(repo):
            raise RuntimeError("different_state_bench_runtime_already_imported")
    elif str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    importlib.import_module("state_bench")
    agents_base = importlib.import_module("state_bench.agents.base")
    agents_native = importlib.import_module("state_bench.agents.state_bench")
    client_module = importlib.import_module("state_bench.client")
    domain_module = importlib.import_module("state_bench.domain")
    env_module = importlib.import_module("state_bench.env_loader")
    orchestrator_module = importlib.import_module("state_bench.orchestrator")
    protocol_module = importlib.import_module("state_bench.protocol")
    schemas_module = importlib.import_module("state_bench.schemas")
    scoring_module = importlib.import_module("state_bench.scoring")
    score_module = importlib.import_module("state_bench.scripts.score")

    protocol = protocol_module.load_default_protocol()
    prompt_errors = protocol.validate_prompt_hashes()
    if prompt_errors:
        raise ValueError(
            "state_bench_protocol_prompt_hash_mismatch:" + "|".join(prompt_errors)
        )
    if protocol.split != STATE_BENCH_OFFICIAL_SPLIT:
        raise ValueError("state_bench_protocol_split_mismatch")
    if protocol.split_version != STATE_BENCH_OFFICIAL_SPLIT_VERSION:
        raise ValueError("state_bench_protocol_split_version_mismatch")
    if set(protocol.domains) != {
        "customer_support",
        "shopping_assistant",
        "travel",
    }:
        raise ValueError("state_bench_protocol_domains_mismatch")
    if protocol.data.get("simulator", {}).get("model") != STATE_BENCH_USER_MODEL:
        raise ValueError("state_bench_protocol_simulator_model_mismatch")
    if protocol.data.get("judge", {}).get("model") != STATE_BENCH_EVALUATOR_MODEL:
        raise ValueError("state_bench_protocol_judge_model_mismatch")

    source_binding_sha256 = _source_binding_sha256(source_paths)
    if source_binding_sha256 != FROZEN_STATE_BENCH_RUNTIME_SOURCE_BINDING_SHA256:
        raise ValueError("state_bench_runtime_source_binding_not_frozen")
    execution_paths = _runtime_execution_source_paths(repo)
    execution_source_file_sha256s = tuple(
        (path.relative_to(repo).as_posix(), _file_sha256(path))
        for path in execution_paths
    )
    execution_source_binding_sha256 = _digest(
        [
            {"path": relative, "sha256": sha256}
            for relative, sha256 in execution_source_file_sha256s
        ]
    )
    if (
        execution_source_binding_sha256
        != FROZEN_STATE_BENCH_EXECUTION_SOURCE_BINDING_SHA256
    ):
        raise ValueError(
            "state_bench_runtime_execution_source_binding_not_frozen"
        )
    symbol_origins = _runtime_symbol_origins(
        repo=repo,
        symbols={
            "run_task": orchestrator_module.run_task,
            "get_domain_config": domain_module.get_domain_config,
            "load_task_environment": env_module.load_task_environment,
            "TaskDefinition": schemas_module.TaskDefinition,
            "BaseAgent": agents_base.BaseAgent,
            "AgentRuntimeContext": agents_base.AgentRuntimeContext,
            "StateBenchAgent": agents_native.StateBenchAgent,
            "LLMClient": client_module.LLMClient,
            "score_one": score_module.score_one,
            "TaskRequirementsJudge": scoring_module.TaskRequirementsJudge,
            "UXQualityJudge": scoring_module.UXQualityJudge,
        },
        allowed_paths={
            relative for relative, _ in execution_source_file_sha256s
        },
    )
    dirty_path_records = _runtime_dirty_path_records(
        repo=repo,
        execution_paths={
            relative for relative, _ in execution_source_file_sha256s
        },
    )

    return StateBenchNativeAPI(
        runtime_repo=repo,
        runtime_revision=revision,
        runtime_worktree_dirty=bool(dirty_path_records),
        source_binding_sha256=source_binding_sha256,
        execution_source_binding_sha256=(
            execution_source_binding_sha256
        ),
        execution_source_file_sha256s=execution_source_file_sha256s,
        dirty_path_records=dirty_path_records,
        symbol_origins=symbol_origins,
        protocol_id=protocol.protocol_id,
        protocol_num_runs=protocol.num_runs,
        run_task=orchestrator_module.run_task,
        get_domain_config=domain_module.get_domain_config,
        load_task_environment=env_module.load_task_environment,
        task_definition_class=schemas_module.TaskDefinition,
        base_agent_class=agents_base.BaseAgent,
        agent_runtime_context_class=agents_base.AgentRuntimeContext,
        state_bench_agent_class=agents_native.StateBenchAgent,
        llm_client_class=client_module.LLMClient,
        score_one=score_module.score_one,
        task_requirements_judge_class=(scoring_module.TaskRequirementsJudge),
        ux_quality_judge_class=scoring_module.UXQualityJudge,
        protocol=protocol,
    )


@dataclass(frozen=True)
class StateBenchNativeRunConfig:
    api_base: str
    private_output_root: Path
    max_provider_calls: int
    max_observed_provider_cost_usd: float
    reasoning_effort: str = REASONING_EFFORT
    user_model: str = STATE_BENCH_USER_MODEL
    evaluator_model: str = STATE_BENCH_EVALUATOR_MODEL

    def __post_init__(self) -> None:
        api_base = _required_text(self.api_base, "state_bench_api_base")
        if not api_base.startswith(("http://", "https://")):
            raise ValueError("state_bench_api_base_invalid")
        object.__setattr__(self, "api_base", api_base.rstrip("/"))
        output_root = Path(self.private_output_root)
        if not output_root.is_absolute():
            raise ValueError("state_bench_private_output_root_not_absolute")
        object.__setattr__(self, "private_output_root", output_root)
        calls = self.max_provider_calls
        if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
            raise ValueError("state_bench_max_provider_calls_invalid")
        cap = self.max_observed_provider_cost_usd
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or float(cap) <= 0.0
        ):
            raise ValueError("state_bench_observed_cost_stop_invalid")
        object.__setattr__(
            self,
            "max_observed_provider_cost_usd",
            float(cap),
        )
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("state_bench_reasoning_effort_not_frozen")
        if self.user_model != STATE_BENCH_USER_MODEL:
            raise ValueError("state_bench_user_model_not_frozen")
        if self.evaluator_model != STATE_BENCH_EVALUATOR_MODEL:
            raise ValueError("state_bench_evaluator_model_not_frozen")

    def public_record(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "private_output_root": str(self.private_output_root),
            "max_provider_calls": self.max_provider_calls,
            "max_observed_provider_cost_usd": (self.max_observed_provider_cost_usd),
            "reasoning_effort": self.reasoning_effort,
            "user_model": self.user_model,
            "evaluator_model": self.evaluator_model,
            "provider_call_cap_is_pre_call_mechanical": True,
            "usd_limit_semantics": "post_call_observed_cost_stop_only",
            "strict_prepaid_usd_ceiling": False,
        }


class StateBenchProviderBudgetExceeded(RuntimeError):
    pass


@dataclass
class StateBenchProviderBudget:
    """Reserve every actual provider attempt before it can leave the process."""

    max_calls: int
    max_observed_cost_usd: float
    calls: int = 0
    observed_cost_usd: float = 0.0
    receipts: list[CallReceipt] = field(default_factory=list)
    provider_cost_observations: int = 0
    provider_cost_complete: bool = True
    missing_provider_cost_receipts: int = 0

    def observe_broker_receipt(self, receipt: CallReceipt) -> None:
        """Mirror one globally brokered receipt without reserving it twice."""

        if self.calls >= self.max_calls:
            raise StateBenchProviderBudgetExceeded(
                "state_bench_provider_call_cap_exhausted"
            )
        self.calls += 1
        self.settle(receipt)

    def reserve(self) -> None:
        if not self.provider_cost_complete:
            raise StateBenchProviderBudgetExceeded(
                "state_bench_provider_cost_incomplete_stop_reached"
            )
        if self.calls >= self.max_calls:
            raise StateBenchProviderBudgetExceeded(
                "state_bench_provider_call_cap_exhausted"
            )
        if self.observed_cost_usd > self.max_observed_cost_usd:
            raise StateBenchProviderBudgetExceeded(
                "state_bench_observed_cost_stop_reached"
            )
        self.calls += 1

    def settle(
        self,
        receipt: CallReceipt,
        *,
        successful: bool = True,
    ) -> None:
        if not isinstance(receipt, CallReceipt):
            raise TypeError("state_bench_provider_receipt_invalid")
        if receipt.provider_calls != 1:
            raise ValueError("state_bench_provider_receipt_call_count_invalid")
        self.receipts.append(receipt)
        observed = receipt.provider_reported_cost_usd
        if observed is None:
            self.provider_cost_complete = False
            self.missing_provider_cost_receipts += 1
            if successful:
                raise StateBenchProviderBudgetExceeded(
                    "state_bench_successful_provider_cost_unavailable"
                )
            return
        self.provider_cost_observations += 1
        self.observed_cost_usd += float(observed)
        if self.observed_cost_usd > self.max_observed_cost_usd:
            raise StateBenchProviderBudgetExceeded(
                "state_bench_observed_provider_cost_stop_exceeded"
            )

    def public_record(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls,
            "receipt_count": len(self.receipts),
            "provider_cost_observations": self.provider_cost_observations,
            "provider_cost_complete": self.provider_cost_complete,
            "missing_provider_cost_receipts": (self.missing_provider_cost_receipts),
            "observed_provider_cost_usd": round(
                self.observed_cost_usd,
                9,
            ),
            "max_provider_calls": self.max_calls,
            "max_observed_provider_cost_usd": (self.max_observed_cost_usd),
            "provider_call_cap_is_pre_call_mechanical": True,
            "usd_limit_semantics": "post_call_observed_cost_stop_only",
            "strict_prepaid_usd_ceiling": False,
        }


@dataclass(frozen=True)
class StateBenchRuntimePolicy:
    """Injected provider clients and official judge instances.

    No constructor for this bundle is provided in the protected baseline
    package.  The paid runtime remains unavailable until a separately reviewed
    loader supplies these exact native objects.
    """

    cheap_client: Any = field(repr=False)
    strong_client: Any = field(repr=False)
    embedding_client: Any | None = field(default=None, repr=False)
    simulator_client: Any = field(default=None, repr=False)
    evaluator_client: Any = field(default=None, repr=False)
    task_requirements_judge: Any = field(default=None, repr=False)
    ux_judge: Any = field(default=None, repr=False)

    def validate(
        self,
        api: StateBenchNativeAPI,
        *,
        require_embedding: bool,
        require_automix: bool,
    ) -> None:
        expected_clients = (
            (self.cheap_client, CHEAP_MODEL),
            (self.strong_client, STRONG_MODEL),
            (self.simulator_client, STATE_BENCH_USER_MODEL),
            (self.evaluator_client, STATE_BENCH_EVALUATOR_MODEL),
        )
        if require_embedding:
            expected_clients = (
                *expected_clients,
                (self.embedding_client, ROUTELLM_EMBEDDING_MODEL),
            )
        if not isinstance(require_automix, bool):
            raise TypeError("state_bench_automix_requirement_invalid")
        for client, model in expected_clients:
            if not isinstance(client, api.llm_client_class):
                raise TypeError("state_bench_runtime_policy_client_not_native")
            if str(getattr(client, "deployment", "")) != model:
                raise ValueError("state_bench_runtime_policy_model_mismatch")
        if not isinstance(
            self.task_requirements_judge,
            api.task_requirements_judge_class,
        ):
            raise TypeError("state_bench_task_requirements_judge_not_official")
        if not isinstance(self.ux_judge, api.ux_quality_judge_class):
            raise TypeError("state_bench_ux_judge_not_official")
        if (
            self.task_requirements_judge.client is not self.evaluator_client
            or self.ux_judge.client is not self.evaluator_client
        ):
            raise ValueError("state_bench_official_judges_client_mismatch")


@dataclass(frozen=True)
class StateBenchNativeEpisode:
    termination_reason: str
    initial_public_user_message: str = field(repr=False)
    public_transcript: tuple[dict[str, Any], ...] = field(repr=False)
    decisions: tuple[RouterDecision, ...]
    receipts: tuple[CallReceipt, ...]
    score_handle: "StateBenchScoreHandle" = field(repr=False)


@dataclass
class StateBenchScoreHandle:
    executor: "StateBenchNativeExecutor" = field(repr=False)
    trajectory: Any = field(repr=False)
    run_dir: Path = field(repr=False)
    _score: float | None = field(default=None, init=False, repr=False)

    def score(self) -> float:
        if self._score is None:
            self._score = self.executor._score_selected_trajectory(
                trajectory=self.trajectory,
                run_dir=self.run_dir,
            )
        return self._score


NativeHistoryDecision = Callable[[HistoryRouterView], RouterDecision]


class StateBenchNativeExecutor:
    """Run source-bound official episodes through native benchmark objects."""

    def __init__(
        self,
        *,
        api: StateBenchNativeAPI,
        driver: StateBenchRouterDriver,
        task: StateBenchOfficialTaskRef,
        config: StateBenchNativeRunConfig,
        policy: StateBenchRuntimePolicy,
        split_gate_json: Path,
        attempt_broker: ProviderAttemptBroker | None = None,
    ) -> None:
        if not isinstance(api, StateBenchNativeAPI):
            raise TypeError("state_bench_native_api_invalid")
        if not isinstance(driver, StateBenchRouterDriver):
            raise TypeError("state_bench_native_driver_invalid")
        if not driver.registry.contains(task):
            raise ValueError("state_bench_native_task_not_in_registry")
        runtime_registry = load_state_bench_official_test_registry(
            api.runtime_repo,
            split_gate_json=split_gate_json,
        )
        if (
            runtime_registry.test_registry_sha256
            != driver.registry.test_registry_sha256
        ):
            raise ValueError("state_bench_runtime_data_registry_mismatch")
        policy.validate(
            api,
            require_embedding=driver.artifact.method == "routellm_task",
            require_automix=driver.artifact.method == AUTOMIX,
        )
        self.api = api
        self.driver = driver
        self.task_ref = task
        self.config = config
        self.policy = policy
        self.attempt_broker = attempt_broker
        self._runtime_label = driver.artifact.method
        self._registry_metadata = {
            "test_registry_sha256": driver.registry.test_registry_sha256,
        }
        self._model_by_role = {
            "cheap": driver.artifact.model_for("cheap"),
            "strong": driver.artifact.model_for("strong"),
        }
        self.private_run_dirs: list[Path] = []
        self.budget = StateBenchProviderBudget(
            max_calls=config.max_provider_calls,
            max_observed_cost_usd=(config.max_observed_provider_cost_usd),
        )

    @classmethod
    def for_fit_dev_pristine_episode(
        cls,
        *,
        api: StateBenchNativeAPI,
        task: StateBenchOfficialTaskRef,
        config: StateBenchNativeRunConfig,
        policy: StateBenchRuntimePolicy,
        registry_sha256: str,
        attempt_broker: ProviderAttemptBroker,
    ) -> "StateBenchNativeExecutor":
        """Bind one source-verified train row without a router artifact."""

        if not isinstance(api, StateBenchNativeAPI):
            raise TypeError("state_bench_native_api_invalid")
        if not isinstance(task, StateBenchOfficialTaskRef):
            raise TypeError("state_bench_native_task_ref_invalid")
        if not isinstance(config, StateBenchNativeRunConfig):
            raise TypeError("state_bench_native_run_config_invalid")
        if not isinstance(policy, StateBenchRuntimePolicy):
            raise TypeError("state_bench_native_policy_invalid")
        if not isinstance(attempt_broker, ProviderAttemptBroker):
            raise TypeError("state_bench_provider_attempt_broker_required")
        if (
            not isinstance(registry_sha256, str)
            or len(registry_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in registry_sha256
            )
        ):
            raise ValueError("state_bench_fit_dev_registry_sha256_invalid")
        policy.validate(
            api,
            require_embedding=False,
            require_automix=False,
        )
        executor = object.__new__(cls)
        executor.api = api
        executor.driver = None
        executor.task_ref = task
        executor.config = config
        executor.policy = policy
        executor.attempt_broker = attempt_broker
        executor._runtime_label = "fit_dev_shared_endpoint"
        executor._registry_metadata = {
            "fit_dev_registry_sha256": registry_sha256,
        }
        executor._model_by_role = {
            "cheap": CHEAP_MODEL,
            "strong": STRONG_MODEL,
        }
        executor.private_run_dirs = []
        executor.budget = StateBenchProviderBudget(
            max_calls=config.max_provider_calls,
            max_observed_cost_usd=(config.max_observed_provider_cost_usd),
        )
        return executor

    def bind_pristine_episode(self) -> Any:
        """Bind the public opening message without launching an agent."""

        task = self._load_task()
        return self.driver.bind_public_episode(
            task=self.task_ref,
            public_initial_user_message=task.opening_message,
        )

    def run_episode_callback(
        self,
        binding: Any,
        role: ModelRole,
        model: str,
    ) -> GeneratedEpisode:
        native = self.run_single_model(role=role, model=model)
        if native.initial_public_user_message != binding.original_request:
            raise RuntimeError("state_bench_pristine_restart_initial_user_mismatch")
        return GeneratedEpisode(
            role=role,
            model=model,
            initial_environment_sha256=(self.task_ref.initial_environment_sha256),
            environment_instance_sha256=_digest(
                {
                    "initial_environment_sha256": (
                        self.task_ref.initial_environment_sha256
                    ),
                    "run_dir": str(native.score_handle.run_dir),
                }
            ),
            generated_answer_view=state_bench_generated_answer_view(
                native.public_transcript
            ),
            receipts=native.receipts,
            opaque_score_handle=native.score_handle,
            public_episode_history=native.public_transcript,
        )

    @staticmethod
    def score_selected_callback(handle: Any) -> float:
        if not isinstance(handle, StateBenchScoreHandle):
            raise TypeError("state_bench_score_handle_invalid")
        return handle.score()

    def run_single_model(
        self,
        *,
        role: ModelRole,
        model: str,
    ) -> StateBenchNativeEpisode:
        expected_model = self._model_by_role[role]
        if model != expected_model:
            raise ValueError("state_bench_selected_model_mismatch")
        task = self._load_task()
        domain = self.api.get_domain_config(self.task_ref.domain)
        env_data, environment_path = self.api.load_task_environment(
            domain,
            task,
        )
        if Path(environment_path).resolve() != (self.task_ref.initial_environment_path):
            raise RuntimeError("state_bench_native_environment_path_mismatch")
        client = (
            self.policy.cheap_client if role == "cheap" else self.policy.strong_client
        )
        run_dir = self._new_private_run_dir()
        before = len(self.budget.receipts)
        with ExitStack() as stack:
            stack.enter_context(
                _instrument_client(
                    client=client,
                    budget=self.budget,
                    role=("cheap_agent" if role == "cheap" else "strong_agent"),
                    model=model,
                    reasoning_effort=self.config.reasoning_effort,
                    phase="native_agent",
                    attempt_broker=self.attempt_broker,
                )
            )
            stack.enter_context(
                _instrument_client(
                    client=self.policy.simulator_client,
                    budget=self.budget,
                    role="user_simulator",
                    model=self.config.user_model,
                    reasoning_effort="protocol",
                    phase="native_user_simulator",
                    attempt_broker=self.attempt_broker,
                )
            )
            trajectory = self.api.run_task(
                task=task,
                env_data=env_data,
                user_id=task.user_id,
                client=client,
                domain=domain,
                simulator_client=self.policy.simulator_client,
                agent_reasoning_effort=self.config.reasoning_effort,
                trajectory_metadata={
                    "router_method": self._runtime_label,
                    "selected_role": role,
                    "selected_model": model,
                    **self._registry_metadata,
                },
            )
        return self._native_episode(
            trajectory=trajectory,
            run_dir=run_dir,
            decisions=(),
            receipts=tuple(self.budget.receipts[before:]),
        )

    def run_history_routed(self) -> StateBenchNativeEpisode:
        artifact = self.driver.artifact
        if artifact.method not in STATE_BENCH_TURN_LEVEL_METHODS:
            raise ValueError("state_bench_history_runtime_method_invalid")
        return self._run_history_routed(
            artifact=artifact,
            decide_next=self.driver.policy.history_decider(),
        )

    def run_fit_dev_history_routed(
        self,
        *,
        artifact: RouterArtifactBinding,
        decide_next: NativeHistoryDecision,
    ) -> StateBenchNativeEpisode:
        """Run one source-bound train row with an injected turn policy."""

        if self.driver is not None:
            raise ValueError("state_bench_fit_dev_executor_required")
        if (
            not isinstance(artifact, RouterArtifactBinding)
            or artifact.method not in STATE_BENCH_TURN_LEVEL_METHODS
            or artifact.surface != "state_bench"
        ):
            raise ValueError("state_bench_fit_dev_history_artifact_invalid")
        if not callable(decide_next):
            raise TypeError("state_bench_fit_dev_history_decider_invalid")
        return self._run_history_routed(
            artifact=artifact,
            decide_next=decide_next,
        )

    def _run_history_routed(
        self,
        *,
        artifact: RouterArtifactBinding,
        decide_next: NativeHistoryDecision,
    ) -> StateBenchNativeEpisode:
        task = self._load_task()
        domain = self.api.get_domain_config(self.task_ref.domain)
        env_data, environment_path = self.api.load_task_environment(
            domain,
            task,
        )
        if Path(environment_path).resolve() != (self.task_ref.initial_environment_path):
            raise RuntimeError("state_bench_native_environment_path_mismatch")
        env = domain.environment_class(
            env_data.deep_copy(),
            now=task.now,
        )
        routed_agent, decisions, local_receipts = self._build_turn_routed_agent(
            task=task,
            domain=domain,
            env=env,
            decide_next=decide_next,
            artifact=artifact,
        )
        run_dir = self._new_private_run_dir()
        before = len(self.budget.receipts)
        with ExitStack() as stack:
            for client, role, model in (
                (self.policy.cheap_client, "cheap_agent", CHEAP_MODEL),
                (
                    self.policy.strong_client,
                    "strong_agent",
                    STRONG_MODEL,
                ),
                (
                    self.policy.simulator_client,
                    "user_simulator",
                    self.config.user_model,
                ),
            ):
                stack.enter_context(
                    _instrument_client(
                        client=client,
                        budget=self.budget,
                        role=role,
                        model=model,
                        reasoning_effort=(
                            self.config.reasoning_effort
                            if role != "user_simulator"
                            else "protocol"
                        ),
                        phase=(
                            "native_agent"
                            if role != "user_simulator"
                            else "native_user_simulator"
                        ),
                        attempt_broker=self.attempt_broker,
                    )
                )
            trajectory = self.api.run_task(
                task=task,
                env_data=env_data,
                user_id=task.user_id,
                client=None,
                domain=domain,
                agent=routed_agent,
                env=env,
                simulator_client=self.policy.simulator_client,
                trajectory_metadata={
                    "router_method": artifact.method,
                    "execution_level": "turn",
                    **self._registry_metadata,
                },
            )
        provider_receipts = tuple(self.budget.receipts[before:])
        return self._native_episode(
            trajectory=trajectory,
            run_dir=run_dir,
            decisions=tuple(decisions),
            receipts=(*tuple(local_receipts), *provider_receipts),
        )

    def _build_turn_routed_agent(
        self,
        *,
        task: Any,
        domain: Any,
        env: Any,
        decide_next: NativeHistoryDecision,
        artifact: RouterArtifactBinding,
    ) -> tuple[Any, list[RouterDecision], list[CallReceipt]]:
        system_prompt = domain.agent_system_prompt.format(
            now=task.now,
            user_id=task.user_id,
        )
        context = self.api.agent_runtime_context_class(
            task_id=task.task_id,
            user_id=task.user_id,
            domain=domain.name,
            now=task.now,
            task_summary=task.task_summary,
            state_requirements=task.state_requirements,
            task_requirements=task.task_requirements,
        )
        role_agents = {
            "cheap": self.api.state_bench_agent_class(
                self.policy.cheap_client,
                system_prompt,
                domain.tool_schemas,
                env.tool_handlers,
                runtime_context=context,
                agent_reasoning_effort=self.config.reasoning_effort,
            ),
            "strong": self.api.state_bench_agent_class(
                self.policy.strong_client,
                system_prompt,
                domain.tool_schemas,
                env.tool_handlers,
                runtime_context=context,
                agent_reasoning_effort=self.config.reasoning_effort,
            ),
        }
        decisions: list[RouterDecision] = []
        local_receipts: list[CallReceipt] = []
        base_agent = self.api.base_agent_class
        turn_cap = int(domain.max_agent_turns)

        class TurnRoutedAgent(base_agent):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                super().__init__(runtime_context=context)

            def act(
                self,
                conversation: list[Any],
            ) -> tuple[str, list[dict[str, Any]], list[Any]]:
                public_history = state_bench_public_history(conversation)
                view = HistoryRouterView(
                    original_request=task.opening_message,
                    public_history=public_history,
                    turn_index=len(decisions),
                    turn_cap=turn_cap,
                )
                started = time.perf_counter()
                decision = decide_next(view)
                elapsed = max(0.0, time.perf_counter() - started)
                _validate_history_decision(
                    decision,
                    artifact=artifact,
                )
                decisions.append(decision)
                local_receipts.append(
                    CallReceipt.local_router(
                        method=artifact.method,
                        latency_seconds=elapsed,
                    )
                )
                selected = role_agents[decision.selected_role]
                before_usage = _usage_tuple(selected.token_usage)
                response = selected.act(conversation)
                after_usage = _usage_tuple(selected.token_usage)
                self.add_token_usage(
                    input_tokens=after_usage[0] - before_usage[0],
                    cached_input_tokens=after_usage[1] - before_usage[1],
                    output_tokens=after_usage[2] - before_usage[2],
                    reasoning_output_tokens=(after_usage[3] - before_usage[3]),
                )
                return response

        return TurnRoutedAgent(), decisions, local_receipts

    def _native_episode(
        self,
        *,
        trajectory: Any,
        run_dir: Path,
        decisions: tuple[RouterDecision, ...],
        receipts: tuple[CallReceipt, ...],
    ) -> StateBenchNativeEpisode:
        transcript = state_bench_public_history(trajectory.conversation)
        initial = _first_public_user_message(transcript)
        termination_reason = (
            "native_user_simulator_task_done"
            if any(
                item.get("role") == "user"
                and state_bench_native_termination(str(item.get("content", "")))
                for item in reversed(transcript)
            )
            else "native_turn_cap_or_no_terminal_marker"
        )
        handle = StateBenchScoreHandle(
            executor=self,
            trajectory=trajectory,
            run_dir=run_dir,
        )
        trajectory.save(run_dir / "trajectory_generated.json")
        return StateBenchNativeEpisode(
            termination_reason=termination_reason,
            initial_public_user_message=initial,
            public_transcript=transcript,
            decisions=decisions,
            receipts=receipts,
            score_handle=handle,
        )

    def _score_selected_trajectory(
        self,
        *,
        trajectory: Any,
        run_dir: Path,
    ) -> float:
        unscored_path = run_dir / "trajectory_unscored.json"
        scored_path = run_dir / "trajectory_scored.json"
        trajectory.save(unscored_path)
        before = len(self.budget.receipts)
        with _instrument_client(
            client=self.policy.evaluator_client,
            budget=self.budget,
            role="evaluator",
            model=self.config.evaluator_model,
            reasoning_effort=self.config.reasoning_effort,
            phase="official_score",
            attempt_broker=self.attempt_broker,
        ):
            result = self.api.score_one(
                unscored_path,
                self.task_ref.native_task_path.parent,
                self.policy.task_requirements_judge,
                self.policy.ux_judge,
                scored_path,
                self.api.protocol,
                self.task_ref.domain,
            )
        if result.get("status") != "OK":
            raise RuntimeError("state_bench_official_scorer_failed")
        scored = json.loads(scored_path.read_text(encoding="utf-8"))
        score = scored.get("task_completion_pass")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("state_bench_official_score_invalid")
        evaluator_receipts = self.budget.receipts[before:]
        if not evaluator_receipts and self.task_ref.domain:
            # Some tasks have deterministic empty requirements, but UX remains
            # judge-scored in the official scorer and must yield a receipt.
            raise RuntimeError("state_bench_official_evaluator_receipt_missing")
        return float(score)

    def _load_task(self) -> Any:
        task = self.api.task_definition_class.load(self.task_ref.native_task_path)
        if str(task.task_id) != self.task_ref.native_task_locator:
            raise RuntimeError("state_bench_native_task_locator_mismatch")
        task_payload = json.loads(
            self.task_ref.native_task_path.read_text(encoding="utf-8")
        )
        environment_payload = json.loads(
            self.task_ref.initial_environment_path.read_text(encoding="utf-8")
        )
        if _digest(environment_payload) != (self.task_ref.initial_environment_sha256):
            raise RuntimeError("state_bench_native_initial_environment_hash_mismatch")
        if (
            _digest(
                {
                    "task": task_payload,
                    "initial_environment": environment_payload,
                }
            )
            != self.task_ref.payload_sha256
        ):
            raise RuntimeError("state_bench_native_task_payload_hash_mismatch")
        return task

    def _new_private_run_dir(self) -> Path:
        run_dir = (
            self.config.private_output_root
            / "state_bench_external_router"
            / self._runtime_label
            / self.task_ref.example_sha256[:16]
            / uuid4().hex
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        self.private_run_dirs.append(run_dir)
        return run_dir


def state_bench_public_history(
    conversation: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Project native transcript items onto the public router history."""

    projected: list[dict[str, Any]] = []
    for raw in conversation:
        item = _item_mapping(raw)
        role = item.get("role")
        if role in {"user", "assistant", "tool"}:
            public: dict[str, Any] = {"role": role}
            if item.get("content") is not None:
                public["content"] = deepcopy(item.get("content"))
            if role == "assistant" and item.get("tool_calls"):
                public["tool_calls"] = deepcopy(item["tool_calls"])
            projected.append(public)
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            projected.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": str(item.get("name", "")),
                            "arguments": arguments,
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            projected.append(
                {
                    "role": "tool",
                    "content": deepcopy(item.get("output")),
                }
            )
    return validate_public_history(projected)


@contextmanager
def _instrument_client(
    *,
    client: Any,
    budget: StateBenchProviderBudget,
    role: Literal[
        "router",
        "cheap_agent",
        "strong_agent",
        "verifier",
        "user_simulator",
        "evaluator",
    ],
    model: str,
    reasoning_effort: str,
    phase: str,
    provider_api: Literal["responses", "embeddings"] = "responses",
    attempt_broker: ProviderAttemptBroker | None = None,
) -> Any:
    """Wrap one raw provider method so retries are also counted."""

    provider = getattr(getattr(client, "_client", None), provider_api, None)
    original = getattr(provider, "create", None)
    if not callable(original):
        raise TypeError(f"state_bench_native_client_{provider_api}_api_missing")

    def budgeted_create(*args: Any, **kwargs: Any) -> Any:
        requested_model = str(kwargs.get("model", ""))
        if requested_model != model:
            raise ValueError("state_bench_provider_request_model_mismatch")
        if attempt_broker is not None:
            response, receipt = brokered_provider_create(
                broker=attempt_broker,
                create=original,
                args=args,
                kwargs=kwargs,
                role=role,
                model=model,
                reasoning_effort=reasoning_effort,
                phase=phase,
            )
            budget.observe_broker_receipt(receipt)
            return response
        budget.reserve()
        started = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        except Exception:
            receipt = CallReceipt(
                role=role,
                phase=phase,
                model=model,
                reasoning_effort=reasoning_effort,
                provider_calls=1,
                input_tokens=None,
                cached_input_tokens=None,
                output_tokens=None,
                reasoning_tokens=None,
                retry_count=0,
                latency_seconds=max(
                    0.0,
                    time.perf_counter() - started,
                ),
                provider_reported_cost_usd=None,
            )
            budget.settle(receipt, successful=False)
            raise
        receipt = _response_receipt(
            response=response,
            role=role,
            phase=phase,
            model=model,
            reasoning_effort=reasoning_effort,
            latency_seconds=max(
                0.0,
                time.perf_counter() - started,
            ),
        )
        budget.settle(receipt)
        return response

    setattr(provider, "create", budgeted_create)
    try:
        yield client
    finally:
        setattr(provider, "create", original)


def brokered_provider_create(
    *,
    broker: ProviderAttemptBroker,
    create: Callable[..., Any],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any],
    role: Literal[
        "router",
        "cheap_agent",
        "strong_agent",
        "verifier",
        "user_simulator",
        "evaluator",
    ],
    model: str,
    reasoning_effort: str,
    phase: str,
) -> tuple[Any, CallReceipt]:
    """Reserve, execute, price, and settle exactly one raw provider attempt."""

    if not isinstance(broker, ProviderAttemptBroker):
        raise TypeError("state_bench_provider_attempt_broker_required")
    response_box: dict[str, Any] = {}

    def raw_attempt() -> CallReceipt:
        started = time.perf_counter()
        response = create(*args, **dict(kwargs))
        receipt = _response_receipt(
            response=response,
            role=role,
            phase=phase,
            model=model,
            reasoning_effort=reasoning_effort,
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
        raise RuntimeError("state_bench_provider_response_missing_after_settle")
    return response_box["response"], receipt


def _response_receipt(
    *,
    response: Any,
    role: str,
    phase: str,
    model: str,
    reasoning_effort: str,
    latency_seconds: float,
) -> CallReceipt:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    raw_input_tokens = getattr(usage, "input_tokens", None)
    if raw_input_tokens is None:
        raw_input_tokens = getattr(usage, "prompt_tokens", None)
    raw_output_tokens = getattr(usage, "output_tokens", None)
    if raw_output_tokens is None:
        raw_output_tokens = getattr(usage, "completion_tokens", None)
    if raw_output_tokens is None and model == ROUTELLM_EMBEDDING_MODEL:
        raw_output_tokens = 0
    raw_cached_tokens = getattr(input_details, "cached_tokens", None)
    if raw_cached_tokens is None:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        raw_cached_tokens = getattr(prompt_details, "cached_tokens", None)
    if raw_cached_tokens is None and model == ROUTELLM_EMBEDDING_MODEL:
        raw_cached_tokens = 0
    raw_reasoning_tokens = getattr(
        output_details,
        "reasoning_tokens",
        None,
    )
    if raw_reasoning_tokens is None:
        completion_details = getattr(
            usage,
            "completion_tokens_details",
            None,
        )
        raw_reasoning_tokens = getattr(
            completion_details,
            "reasoning_tokens",
            None,
        )
    if raw_reasoning_tokens is None and model == ROUTELLM_EMBEDDING_MODEL:
        raw_reasoning_tokens = 0
    input_tokens = _optional_int(raw_input_tokens)
    output_tokens = _optional_int(raw_output_tokens)
    cached_tokens = _optional_int(raw_cached_tokens)
    reasoning_tokens = _optional_int(raw_reasoning_tokens)
    if input_tokens is not None and cached_tokens is None:
        cached_tokens = 0
    if output_tokens is not None and reasoning_tokens is None:
        reasoning_tokens = 0
    cache_neutral_cost = _cache_neutral_provider_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
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
        provider_reported_cost_usd=cache_neutral_cost,
    )


def _cache_neutral_provider_cost(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    price = (
        STATE_BENCH_HARNESS_MODEL_PRICE
        if model == STATE_BENCH_USER_MODEL
        else FROZEN_CACHE_NEUTRAL_PRICES.get(model)
    )
    if price is None:
        return None
    return (
        input_tokens * price.input_per_million_usd
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000


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
        raise ValueError("state_bench_raw_provider_receipt_incomplete")


def _validate_history_decision(
    decision: RouterDecision,
    *,
    artifact: Any,
) -> None:
    if not isinstance(decision, RouterDecision):
        raise TypeError("state_bench_history_decision_invalid")
    if decision.method != artifact.method:
        raise ValueError("state_bench_history_decision_method_mismatch")
    if decision.surface != artifact.surface:
        raise ValueError("state_bench_history_decision_surface_mismatch")
    if decision.artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("state_bench_history_decision_artifact_mismatch")
    if decision.selected_model != artifact.model_for(decision.selected_role):
        raise ValueError("state_bench_history_decision_model_mismatch")


def state_bench_generated_answer_view(
    transcript: Sequence[Mapping[str, Any]],
) -> str:
    messages = [
        str(item.get("content", "")).strip()
        for item in transcript
        if item.get("role") == "assistant" and str(item.get("content", "")).strip()
    ]
    return "\n".join(messages) or "[no assistant text]"


def _first_public_user_message(
    transcript: Sequence[Mapping[str, Any]],
) -> str:
    for item in transcript:
        if item.get("role") == "user":
            return _required_text(
                item.get("content"),
                "state_bench_initial_user_message",
            )
    raise RuntimeError("state_bench_initial_user_message_missing")


def _item_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    fields = {
        name: getattr(raw, name)
        for name in (
            "role",
            "content",
            "tool_calls",
            "type",
            "name",
            "arguments",
            "output",
        )
        if hasattr(raw, name)
    }
    return fields


def _usage_tuple(usage: Any) -> tuple[int, int, int, int]:
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "cached_input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(usage, "reasoning_output_tokens", 0) or 0),
    )


def _source_binding_sha256(paths: Sequence[Path]) -> str:
    return _digest(
        [
            {
                "path": str(path.relative_to(path.parents[1])),
                "sha256": _file_sha256(path),
            }
            for path in paths
        ]
    )


def _runtime_execution_source_paths(repo: Path) -> tuple[Path, ...]:
    root = Path(repo).resolve()
    paths = {root / relative for relative in _RUNTIME_EXECUTION_CORE_FILES}
    domains = root / "state_bench" / "domains"
    for path in domains.rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".py", ".md"}
            and not {"tasks", "task_envs", "splits"} & set(path.parts)
        ):
            paths.add(path)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            "state_bench_runtime_execution_source_file_missing"
        )
    return tuple(
        sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    )


def _runtime_symbol_origins(
    *,
    repo: Path,
    symbols: Mapping[str, Any],
    allowed_paths: set[str],
) -> tuple[tuple[str, str], ...]:
    root = Path(repo).resolve()
    origins: list[tuple[str, str]] = []
    for label, symbol in symbols.items():
        raw_origin = inspect.getsourcefile(symbol)
        if raw_origin is None:
            raise RuntimeError(
                f"state_bench_runtime_symbol_origin_missing:{label}"
            )
        origin = Path(raw_origin).resolve()
        try:
            relative = origin.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"state_bench_runtime_symbol_outside_repo:{label}"
            ) from exc
        if relative not in allowed_paths:
            raise RuntimeError(
                f"state_bench_runtime_symbol_source_unbound:{label}"
            )
        origins.append((label, relative))
    return tuple(sorted(origins))


def _runtime_dirty_path_records(
    *,
    repo: Path,
    execution_paths: set[str],
) -> tuple[Mapping[str, Any], ...]:
    root = Path(repo).resolve()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    )
    chunks = tuple(
        item for item in completed.stdout.split(b"\0") if item
    )
    records: list[Mapping[str, Any]] = []
    for raw in chunks:
        decoded = raw.decode("utf-8")
        if len(decoded) < 4:
            raise RuntimeError("state_bench_runtime_git_status_invalid")
        status = decoded[:2]
        relative = decoded[3:]
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        path = root / relative
        records.append(
            {
                "path": relative,
                "git_status": status,
                "sha256": _file_sha256(path) if path.is_file() else None,
                "used_by_fit_dev_execution": relative in execution_paths,
            }
        )
    return tuple(sorted(records, key=lambda item: str(item["path"])))


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 0:
        raise ValueError("state_bench_provider_usage_negative")
    return resolved


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError("state_bench_provider_cost_negative")
    return resolved


def _digest(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

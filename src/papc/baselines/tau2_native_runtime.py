"""Official tau2 text-runtime callbacks for protected external routers.

The runtime is loaded from a clean upstream tau2 checkout.  Router policies
remain injected callables: this module owns only the public tau2 agent,
message, environment, termination, and official-scoring boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from papc.baselines.external_router_execution import ProviderAttemptBroker
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    ROUTELLM_LOCAL_MODEL,
    STRONG_MODEL,
    HistoryRouterView,
    ModelRole,
    RequestRouterView,
    RouterArtifactBinding,
    validate_public_history,
)
from papc.baselines.router_policies import RouterDecision, decide_routellm
from papc.baselines.router_runtime import GeneratedEpisode, ScalarPrediction
from papc.baselines.tau2_router_driver import (
    TAU2_TURN_LEVEL_METHODS,
    Tau2OfficialTaskRef,
    Tau2OfficialTestRegistry,
    Tau2RouterDriver,
    load_tau2_official_test_registry,
)

FROZEN_TAU2_RUNTIME_REVISION = "f1eb5ae72bdcbd4b491a196d6640f5bbed2ae480"
TAU2_NATIVE_AGENT = "llm_agent"
TAU2_NATIVE_USER = "user_simulator"
TAU2_OFFICIAL_SPLIT = "test"
TAU2_OFFICIAL_EVALUATION = "ALL_WITH_NL_ASSERTIONS"
TAU2_USER_MODEL = STRONG_MODEL
TAU2_EVALUATOR_MODEL = STRONG_MODEL
TAU2_MAX_INFRA_RETRIES = 2


@dataclass(frozen=True)
class Tau2NativeAPI:
    runtime_repo: Path
    runtime_revision: str
    text_run_config: Any = field(repr=False)
    run_single_task: Callable[..., Any] = field(repr=False)
    get_tasks: Callable[..., list[Any]] = field(repr=False)
    registry: Any = field(repr=False)
    llm_agent_class: type = field(repr=False)
    evaluation_type: Any = field(repr=False)
    agent_module: Any = field(repr=False)
    user_module: Any = field(repr=False)
    evaluator_module: Any = field(repr=False)

    def public_record(self) -> dict[str, Any]:
        return {
            "runtime_repo": str(self.runtime_repo),
            "runtime_revision": self.runtime_revision,
            "agent": TAU2_NATIVE_AGENT,
            "user": TAU2_NATIVE_USER,
            "official_split": TAU2_OFFICIAL_SPLIT,
            "official_evaluation": TAU2_OFFICIAL_EVALUATION,
        }


def load_tau2_native_api(runtime_repo: Path) -> Tau2NativeAPI:
    """Import the frozen official tau2 runtime without calling a provider."""

    source = validate_tau2_native_source(runtime_repo)
    repo = Path(source["runtime_repo"])
    revision = source["runtime_revision"]
    source_root = (repo / "src").resolve()
    loaded = sys.modules.get("tau2")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if not loaded_path.is_relative_to(source_root):
            raise RuntimeError("different_tau2_runtime_already_imported")
    elif str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    importlib.import_module("tau2")
    agent_module = importlib.import_module("tau2.agent.llm_agent")
    simulation_module = importlib.import_module("tau2.data_model.simulation")
    evaluator_module = importlib.import_module("tau2.evaluator.evaluator_nl_assertions")
    evaluator_api = importlib.import_module("tau2.evaluator.evaluator")
    registry_module = importlib.import_module("tau2.registry")
    batch_module = importlib.import_module("tau2.runner.batch")
    helpers_module = importlib.import_module("tau2.runner.helpers")
    user_module = importlib.import_module("tau2.user.user_simulator")
    evaluation_type = evaluator_api.EvaluationType
    if not hasattr(evaluation_type, TAU2_OFFICIAL_EVALUATION):
        raise RuntimeError("tau2_official_evaluation_api_missing")
    return Tau2NativeAPI(
        runtime_repo=repo,
        runtime_revision=revision,
        text_run_config=simulation_module.TextRunConfig,
        run_single_task=batch_module.run_single_task,
        get_tasks=helpers_module.get_tasks,
        registry=registry_module.registry,
        llm_agent_class=agent_module.LLMAgent,
        evaluation_type=evaluation_type,
        agent_module=agent_module,
        user_module=user_module,
        evaluator_module=evaluator_module,
    )


def validate_tau2_native_source(runtime_repo: Path) -> dict[str, str]:
    """Validate the frozen checkout without importing tau2 or reading dotenv."""

    repo = Path(runtime_repo).resolve()
    revision = _git_output(repo, "rev-parse", "HEAD")
    if revision != FROZEN_TAU2_RUNTIME_REVISION:
        raise ValueError("tau2_runtime_revision_not_frozen")
    if _git_output(repo, "status", "--porcelain"):
        raise ValueError("tau2_runtime_worktree_dirty")
    source_root = (repo / "src").resolve()
    if not source_root.is_dir():
        raise FileNotFoundError("tau2_runtime_source_root_missing")
    return {
        "runtime_repo": str(repo),
        "runtime_revision": revision,
        "source_root": str(source_root),
    }


@dataclass(frozen=True)
class Tau2NativeRunConfig:
    api_base: str
    private_output_root: Path
    max_provider_calls: int
    max_observed_provider_cost_usd: float
    seed: int = 310
    max_steps: int = 100
    user_model: str = TAU2_USER_MODEL
    evaluator_model: str = TAU2_EVALUATOR_MODEL
    reasoning_effort: str = REASONING_EFFORT
    max_infra_retries: int = TAU2_MAX_INFRA_RETRIES

    def __post_init__(self) -> None:
        api_base = _required_text(self.api_base, "tau2_api_base")
        if not api_base.startswith(("http://", "https://")):
            raise ValueError("tau2_api_base_invalid")
        object.__setattr__(self, "api_base", api_base.rstrip("/"))
        output_root = Path(self.private_output_root)
        if not output_root.is_absolute():
            raise ValueError("tau2_private_output_root_not_absolute")
        object.__setattr__(self, "private_output_root", output_root)
        for field_name in ("max_provider_calls", "max_steps"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"tau2_{field_name}_invalid")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("tau2_seed_invalid")
        cap = self.max_observed_provider_cost_usd
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or float(cap) <= 0.0
        ):
            raise ValueError("tau2_provider_cost_cap_invalid")
        object.__setattr__(
            self,
            "max_observed_provider_cost_usd",
            float(cap),
        )
        if self.user_model != TAU2_USER_MODEL:
            raise ValueError("tau2_user_model_not_frozen")
        if self.evaluator_model != TAU2_EVALUATOR_MODEL:
            raise ValueError("tau2_evaluator_model_not_frozen")
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("tau2_reasoning_effort_not_frozen")
        if self.max_infra_retries not in {0, TAU2_MAX_INFRA_RETRIES}:
            raise ValueError("tau2_max_infra_retries_not_frozen")

    def llm_args(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "reasoning_effort": self.reasoning_effort,
        }

    def public_record(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "private_output_root": str(self.private_output_root),
            "max_provider_calls": self.max_provider_calls,
            "max_observed_provider_cost_usd": (self.max_observed_provider_cost_usd),
            "seed": self.seed,
            "max_steps": self.max_steps,
            "user_model": self.user_model,
            "evaluator_model": self.evaluator_model,
            "reasoning_effort": self.reasoning_effort,
            "max_infra_retries": self.max_infra_retries,
        }


class Tau2ProviderBudgetExceeded(RuntimeError):
    pass


@dataclass
class Tau2ProviderBudget:
    max_calls: int
    max_observed_cost_usd: float
    calls: int = 0
    observed_cost_usd: float = 0.0

    def ensure_call_capacity(self, requested_calls: int) -> None:
        if requested_calls <= 0:
            raise ValueError("tau2_requested_provider_calls_invalid")
        if self.calls + requested_calls > self.max_calls:
            raise Tau2ProviderBudgetExceeded("tau2_provider_call_cap_exhausted")

    def charge_external_receipts(
        self,
        receipts: Sequence[CallReceipt],
    ) -> None:
        missing_cost = any(
            receipt.provider_calls > 0 and receipt.provider_reported_cost_usd is None
            for receipt in receipts
        )
        if missing_cost:
            raise Tau2ProviderBudgetExceeded("tau2_provider_cost_unavailable")
        provider_calls = sum(receipt.provider_calls for receipt in receipts)
        self.ensure_call_capacity(provider_calls)
        self.calls += provider_calls
        known_cost = sum(
            float(receipt.provider_reported_cost_usd)
            for receipt in receipts
            if receipt.provider_reported_cost_usd is not None
        )
        self.observed_cost_usd += known_cost
        if self.observed_cost_usd > self.max_observed_cost_usd:
            raise Tau2ProviderBudgetExceeded("tau2_observed_provider_cost_cap_exceeded")

    def wrap(self, generate: Callable[..., Any]) -> Callable[..., Any]:
        def budgeted_generate(*args: Any, **kwargs: Any) -> Any:
            self.ensure_call_capacity(1)
            message = generate(*args, **kwargs)
            self.calls += 1
            raw_cost = getattr(message, "cost", None)
            if raw_cost is None:
                raise Tau2ProviderBudgetExceeded("tau2_provider_cost_unavailable")
            self.observed_cost_usd += float(raw_cost)
            if self.observed_cost_usd > self.max_observed_cost_usd:
                raise Tau2ProviderBudgetExceeded(
                    "tau2_observed_provider_cost_cap_exceeded"
                )
            return message

        return budgeted_generate

    def public_record(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls,
            "observed_provider_cost_usd": round(
                self.observed_cost_usd,
                9,
            ),
            "max_provider_calls": self.max_calls,
            "max_observed_provider_cost_usd": self.max_observed_cost_usd,
            "stops_before_next_call_after_cap": True,
            "strict_prepaid_usd_ceiling": False,
        }


@dataclass(frozen=True)
class Tau2NativeEpisode:
    official_score: float
    termination_reason: str
    initial_public_user_message: str = field(repr=False)
    public_transcript: tuple[dict[str, Any], ...] = field(repr=False)
    decisions: tuple[RouterDecision, ...]
    receipts: tuple[CallReceipt, ...]
    simulation: Any = field(repr=False)


class Tau2NativeExecutor:
    """Execute one source-bound official tau2 task through native APIs."""

    def __init__(
        self,
        *,
        api: Tau2NativeAPI,
        driver: Tau2RouterDriver,
        task: Tau2OfficialTaskRef,
        config: Tau2NativeRunConfig,
        attempt_broker: ProviderAttemptBroker | None = None,
    ) -> None:
        if not isinstance(api, Tau2NativeAPI):
            raise TypeError("tau2_native_api_invalid")
        if not isinstance(driver, Tau2RouterDriver):
            raise TypeError("tau2_native_driver_invalid")
        if not driver.registry.contains(task):
            raise ValueError("tau2_native_task_not_in_registry")
        runtime_registry = load_tau2_official_test_registry(
            api.runtime_repo,
            domain=driver.registry.domain,
        )
        if (
            runtime_registry.test_registry_sha256
            != driver.registry.test_registry_sha256
        ):
            raise ValueError("tau2_runtime_data_registry_mismatch")
        self.api = api
        self.driver = driver
        self.task_ref = task
        self.config = config
        self._domain = driver.registry.domain
        self._task_split_name = TAU2_OFFICIAL_SPLIT
        self._runtime_label = driver.artifact.method
        self._source_binding_sha256 = (
            driver.registry.source_binding_sha256
        )
        self._model_by_role = {
            "cheap": driver.artifact.model_for("cheap"),
            "strong": driver.artifact.model_for("strong"),
        }
        if attempt_broker is not None and not isinstance(
            attempt_broker,
            ProviderAttemptBroker,
        ):
            raise TypeError("tau2_provider_attempt_broker_invalid")
        self.attempt_broker = attempt_broker
        self.budget = Tau2ProviderBudget(
            max_calls=config.max_provider_calls,
            max_observed_cost_usd=(config.max_observed_provider_cost_usd),
        )
        self._bound_initial_user_message: str | None = None
        self._cached_episode: (
            tuple[
                ModelRole,
                str,
                Tau2NativeEpisode,
            ]
            | None
        ) = None
        self._last_native_provider_receipts: tuple[CallReceipt, ...] = ()
        self._private_simulation_manifest: list[dict[str, str]] = []

    @classmethod
    def for_fit_dev_pristine_episode(
        cls,
        *,
        api: Tau2NativeAPI,
        task: Tau2OfficialTaskRef,
        domain: str,
        config: Tau2NativeRunConfig,
        source_binding_sha256: str,
        attempt_broker: ProviderAttemptBroker,
        bound_initial_user_message: str | None = None,
    ) -> "Tau2NativeExecutor":
        """Bind one source-verified official-train row without a router."""

        if not isinstance(api, Tau2NativeAPI):
            raise TypeError("tau2_native_api_invalid")
        if not isinstance(task, Tau2OfficialTaskRef):
            raise TypeError("tau2_native_task_ref_invalid")
        if domain not in {"retail", "airline"}:
            raise ValueError("tau2_fit_dev_domain_invalid")
        if not isinstance(config, Tau2NativeRunConfig):
            raise TypeError("tau2_native_run_config_invalid")
        if not isinstance(attempt_broker, ProviderAttemptBroker):
            raise TypeError("tau2_provider_attempt_broker_invalid")
        if bound_initial_user_message is not None and (
            not isinstance(bound_initial_user_message, str)
            or not bound_initial_user_message.strip()
        ):
            raise ValueError(
                "tau2_bound_initial_user_message_invalid"
            )
        if (
            not isinstance(source_binding_sha256, str)
            or len(source_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_binding_sha256
            )
        ):
            raise ValueError("tau2_fit_dev_source_binding_sha256_invalid")
        executor = object.__new__(cls)
        executor.api = api
        executor.driver = None
        executor.task_ref = task
        executor.config = config
        executor._domain = domain
        executor._task_split_name = "train"
        executor._runtime_label = "fit_dev_shared_endpoint"
        executor._source_binding_sha256 = source_binding_sha256
        executor._model_by_role = {
            "cheap": CHEAP_MODEL,
            "strong": STRONG_MODEL,
        }
        executor.attempt_broker = attempt_broker
        executor.budget = Tau2ProviderBudget(
            max_calls=config.max_provider_calls,
            max_observed_cost_usd=(config.max_observed_provider_cost_usd),
        )
        executor._bound_initial_user_message = (
            bound_initial_user_message
        )
        executor._cached_episode = None
        executor._last_native_provider_receipts = ()
        executor._private_simulation_manifest = []
        return executor

    @classmethod
    def for_custom_official_episode(
        cls,
        *,
        api: Tau2NativeAPI,
        task: Tau2OfficialTaskRef,
        registry: Tau2OfficialTestRegistry,
        config: Tau2NativeRunConfig,
        attempt_broker: ProviderAttemptBroker,
        runtime_label: str,
    ) -> "Tau2NativeExecutor":
        """Bind one official-test row to an isolated injected mixed-model agent."""

        if not isinstance(api, Tau2NativeAPI):
            raise TypeError("tau2_custom_native_api_invalid")
        if not isinstance(task, Tau2OfficialTaskRef):
            raise TypeError("tau2_custom_task_ref_invalid")
        if not isinstance(registry, Tau2OfficialTestRegistry):
            raise TypeError("tau2_custom_registry_invalid")
        if not registry.contains(task):
            raise ValueError("tau2_custom_task_not_in_registry")
        if not isinstance(config, Tau2NativeRunConfig):
            raise TypeError("tau2_custom_config_invalid")
        if not isinstance(attempt_broker, ProviderAttemptBroker):
            raise TypeError("tau2_custom_attempt_broker_invalid")
        label = _required_text(runtime_label, "tau2_custom_runtime_label")
        runtime_registry = load_tau2_official_test_registry(
            api.runtime_repo,
            domain=registry.domain,
        )
        if (
            runtime_registry.test_registry_sha256
            != registry.test_registry_sha256
            or runtime_registry.source_binding_sha256
            != registry.source_binding_sha256
        ):
            raise ValueError("tau2_custom_runtime_data_registry_mismatch")

        executor = object.__new__(cls)
        executor.api = api
        executor.driver = None
        executor.task_ref = task
        executor.config = config
        executor._domain = registry.domain
        executor._task_split_name = TAU2_OFFICIAL_SPLIT
        executor._runtime_label = label
        executor._source_binding_sha256 = registry.source_binding_sha256
        executor._model_by_role = {
            "cheap": CHEAP_MODEL,
            "strong": STRONG_MODEL,
        }
        executor.attempt_broker = attempt_broker
        executor.budget = Tau2ProviderBudget(
            max_calls=config.max_provider_calls,
            max_observed_cost_usd=(config.max_observed_provider_cost_usd),
        )
        executor._bound_initial_user_message = None
        executor._cached_episode = None
        executor._last_native_provider_receipts = ()
        executor._private_simulation_manifest = []
        return executor

    @property
    def private_simulation_manifest(self) -> tuple[dict[str, str], ...]:
        return tuple(deepcopy(self._private_simulation_manifest))

    def bootstrap_pristine_episode(
        self,
        *,
        role: ModelRole,
        model: str,
    ) -> Any:
        """Open one official episode and cache it for a cascade callback."""

        native = self.run_single_model(role=role, model=model)
        binding = self.driver.bind_public_episode(
            task=self.task_ref,
            public_initial_user_message=native.initial_public_user_message,
            initial_environment_sha256=self._pristine_binding_sha256(),
        )
        self._cached_episode = (role, model, native)
        return binding

    def run_episode_callback(
        self,
        binding: Any,
        role: ModelRole,
        model: str,
    ) -> GeneratedEpisode:
        cached = self._cached_episode
        if cached is not None and cached[0:2] == (role, model):
            native = cached[2]
            self._cached_episode = None
        else:
            native = self.run_single_model(role=role, model=model)
        if native.initial_public_user_message != binding.original_request:
            raise RuntimeError("tau2_pristine_restart_initial_user_mismatch")
        return self._generated_episode(
            binding=binding,
            role=role,
            model=model,
            native=native,
        )

    @staticmethod
    def score_selected_callback(simulation: Any) -> float:
        reward_info = getattr(simulation, "reward_info", None)
        reward = getattr(reward_info, "reward", None)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError("tau2_official_reward_missing")
        return float(reward)

    def run_single_model(
        self,
        *,
        role: ModelRole,
        model: str,
    ) -> Tau2NativeEpisode:
        self._validate_model(role, model)
        simulation = self._run_official(
            agent_name=TAU2_NATIVE_AGENT,
            agent_model=model,
        )
        roles = tuple(role for _ in _generated_assistant_messages(simulation))
        return self._native_episode(
            simulation=simulation,
            decisions=(),
            roles=roles,
            router_receipts=(),
        )

    def run_registered_mixed_model_agent(
        self,
        *,
        register_agent: Callable[[Callable[..., Any]], str],
        agent_model: str,
    ) -> Tau2NativeEpisode:
        """Run a registered agent whose brokered calls may use both models."""

        if self.attempt_broker is None:
            raise ValueError("tau2_mixed_agent_requires_attempt_broker")
        if agent_model != CHEAP_MODEL:
            raise ValueError("tau2_mixed_agent_entry_model_not_frozen")
        if not callable(register_agent):
            raise TypeError("tau2_mixed_agent_registrar_invalid")
        native_receipts: list[CallReceipt] = []
        generate = importlib.import_module("tau2.utils.llm_utils").generate
        agent_name = register_agent(
            build_brokered_tau2_agent_generate(
                generate,
                broker=self.attempt_broker,
                receipts=native_receipts,
                max_infra_retries=self.config.max_infra_retries,
            )
        )
        simulation = self._run_official(
            agent_name=_required_text(
                agent_name,
                "tau2_mixed_agent_name",
            ),
            agent_model=agent_model,
            receipt_sink=native_receipts,
        )
        public_transcript = tau2_public_history(
            tuple(getattr(simulation, "messages", ()) or ())
        )
        initial_user = _first_public_user_text(public_transcript)
        if self._bound_initial_user_message is None:
            self._bound_initial_user_message = initial_user
        elif self._bound_initial_user_message != initial_user:
            raise RuntimeError("tau2_pristine_restart_initial_user_mismatch")
        native_receipts = self._last_native_provider_receipts
        if not any(
            receipt.role in {"cheap_agent", "strong_agent"}
            for receipt in native_receipts
        ):
            raise RuntimeError("tau2_mixed_agent_receipt_missing")
        termination = getattr(
            getattr(simulation, "termination_reason", ""),
            "value",
            getattr(simulation, "termination_reason", ""),
        )
        return Tau2NativeEpisode(
            official_score=self.score_selected_callback(simulation),
            termination_reason=str(termination),
            initial_public_user_message=initial_user,
            public_transcript=public_transcript,
            decisions=(),
            receipts=native_receipts,
            simulation=simulation,
        )

    def run_request_routed(self) -> Tau2NativeEpisode:
        if self.driver.artifact.method != ROUTELLM:
            raise ValueError("tau2_native_request_method_invalid")
        decisions: list[RouterDecision] = []
        router_receipts: list[CallReceipt] = []

        def decide_request(view: RequestRouterView) -> RouterDecision:
            self.budget.ensure_call_capacity(1)
            prediction = self.driver.policy.predict_routellm(view)
            _validate_routellm_prediction(prediction)
            provider_receipts = tuple(
                receipt
                for receipt in prediction.receipts
                if receipt.provider_calls > 0
            )
            if provider_receipts:
                self.budget.charge_external_receipts(provider_receipts)
            router_receipts.extend(prediction.receipts)
            return decide_routellm(
                binding=self.driver.artifact,
                view=view,
                predicted_strong_win_rate=prediction.value,
                threshold=self.driver.policy.routellm_threshold,
            )

        agent_name = self._register_routed_agent(
            mode="request",
            decide=decide_request,
            artifact=self.driver.artifact,
            decisions=decisions,
            router_receipts=router_receipts,
            add_local_router_receipt=False,
        )
        simulation = self._run_official(
            agent_name=agent_name,
            agent_model=CHEAP_MODEL,
        )
        if len(decisions) != 1:
            raise RuntimeError("tau2_native_request_decision_count_invalid")
        return self._native_episode(
            simulation=simulation,
            decisions=tuple(decisions),
            roles=tuple(
                decisions[0].selected_role
                for _ in _generated_assistant_messages(simulation)
            ),
            router_receipts=tuple(router_receipts),
        )

    def run_history_routed(self) -> Tau2NativeEpisode:
        artifact = self.driver.artifact
        if artifact.method not in TAU2_TURN_LEVEL_METHODS:
            raise ValueError("tau2_native_history_method_invalid")
        return self._run_history_routed(
            artifact=artifact,
            decide=self.driver.policy.history_decider(),
        )

    def run_fit_dev_history_routed(
        self,
        *,
        artifact: RouterArtifactBinding,
        decide_next: Callable[[HistoryRouterView], RouterDecision],
    ) -> Tau2NativeEpisode:
        """Run one source-bound official-train row with a turn policy."""

        if self.driver is not None:
            raise ValueError("tau2_fit_dev_executor_required")
        if (
            not isinstance(artifact, RouterArtifactBinding)
            or artifact.method not in TAU2_TURN_LEVEL_METHODS
            or artifact.surface
            not in {"tau2_retail", "tau2_airline"}
        ):
            raise ValueError("tau2_fit_dev_history_artifact_invalid")
        if not callable(decide_next):
            raise TypeError("tau2_fit_dev_history_decider_invalid")
        return self._run_history_routed(
            artifact=artifact,
            decide=decide_next,
        )

    def _run_history_routed(
        self,
        *,
        artifact: RouterArtifactBinding,
        decide: Callable[[HistoryRouterView], RouterDecision],
    ) -> Tau2NativeEpisode:
        decisions: list[RouterDecision] = []
        router_receipts: list[CallReceipt] = []
        agent_name = self._register_routed_agent(
            mode="history",
            decide=decide,
            artifact=artifact,
            decisions=decisions,
            router_receipts=router_receipts,
            add_local_router_receipt=True,
        )
        simulation = self._run_official(
            agent_name=agent_name,
            agent_model=CHEAP_MODEL,
        )
        return self._native_episode(
            simulation=simulation,
            decisions=tuple(decisions),
            roles=tuple(decision.selected_role for decision in decisions),
            router_receipts=tuple(router_receipts),
        )

    def _register_routed_agent(
        self,
        *,
        mode: str,
        decide: Callable[[Any], RouterDecision],
        artifact: RouterArtifactBinding,
        decisions: list[RouterDecision],
        router_receipts: list[CallReceipt],
        add_local_router_receipt: bool,
    ) -> str:
        base = self.api.llm_agent_class
        role_args = {
            "cheap": self.config.llm_args(),
            "strong": self.config.llm_args(),
        }

        class ProtectedRoutedAgent(base):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._papc_seed: int | None = None
                self._papc_original_request: str | None = None

            def set_seed(self, seed: int) -> None:
                self._papc_seed = seed
                super().set_seed(seed)

            def generate_next_message(
                self,
                message: Any,
                state: Any,
            ) -> tuple[Any, Any]:
                public_history = tau2_public_history((*tuple(state.messages), message))
                if self._papc_original_request is None:
                    self._papc_original_request = _first_public_user_text(
                        public_history
                    )
                if mode == "request":
                    if decisions:
                        decision = decisions[0]
                    else:
                        view: Any = RequestRouterView(self._papc_original_request)
                        decision = _timed_native_decision(
                            decide,
                            view,
                            artifact.method,
                            router_receipts,
                            add_local_router_receipt=(add_local_router_receipt),
                        )
                else:
                    view = HistoryRouterView(
                        original_request=self._papc_original_request,
                        public_history=public_history,
                        turn_index=len(decisions),
                        turn_cap=self_outer.config.max_steps,
                    )
                    decision = _timed_native_decision(
                        decide,
                        view,
                        artifact.method,
                        router_receipts,
                        add_local_router_receipt=(add_local_router_receipt),
                    )
                _validate_native_decision(decision, artifact)
                if mode == "history" or not decisions:
                    decisions.append(decision)
                self.llm = decision.selected_model
                self.llm_args = deepcopy(role_args[decision.selected_role])
                if self._papc_seed is not None:
                    self.llm_args["seed"] = self._papc_seed
                return super().generate_next_message(message, state)

        self_outer = self

        def factory(
            tools: Any,
            domain_policy: str,
            llm: str,
            llm_args: Mapping[str, Any] | None = None,
            **_: Any,
        ) -> Any:
            return ProtectedRoutedAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=dict(llm_args or {}),
            )

        name = (
            "papc_external_router_"
            f"{artifact.method}_{artifact.artifact_sha256[:10]}_"
            f"{uuid4().hex[:10]}"
        )
        self.api.registry.register_agent_factory(factory, name)
        return name

    def _run_official(
        self,
        *,
        agent_name: str,
        agent_model: str,
        receipt_sink: list[CallReceipt] | None = None,
    ) -> Any:
        tasks = self.api.get_tasks(
            task_set_name=self._domain,
            task_split_name=self._task_split_name,
            task_ids=[self.task_ref.native_task_locator],
            num_tasks=1,
        )
        if (
            len(tasks) != 1
            or str(getattr(tasks[0], "id", "")) != self.task_ref.native_task_locator
        ):
            raise RuntimeError("tau2_native_official_task_lookup_mismatch")
        run_id = uuid4().hex
        run_dir = self.config.private_output_root / run_id
        config = self.api.text_run_config(
            domain=self._domain,
            task_split_name=self._task_split_name,
            task_ids=[self.task_ref.native_task_locator],
            num_tasks=1,
            num_trials=1,
            max_retries=0,
            retry_delay=0.0,
            max_concurrency=1,
            seed=self.config.seed,
            log_level="INFO",
            agent=agent_name,
            llm_agent=agent_model,
            llm_args_agent=self.config.llm_args(),
            user=TAU2_NATIVE_USER,
            llm_user=self.config.user_model,
            llm_args_user=self.config.llm_args(),
            max_steps=self.config.max_steps,
            auto_review=False,
            hallucination_retries=0,
        )
        native_receipts = [] if receipt_sink is None else receipt_sink
        if native_receipts:
            raise ValueError("tau2_native_receipt_sink_not_empty")
        with _official_evaluator_config(self.api, self.config):
            with _instrument_provider_attempts(
                self.api,
                budget=self.budget,
                broker=self.attempt_broker,
                receipts=native_receipts,
                max_infra_retries=self.config.max_infra_retries,
            ):
                with _replay_bound_initial_user_message(
                    self.api,
                    self._bound_initial_user_message,
                ):
                    simulation = self.api.run_single_task(
                        config,
                        tasks[0],
                        seed=self.config.seed,
                        evaluation_type=getattr(
                            self.api.evaluation_type,
                            TAU2_OFFICIAL_EVALUATION,
                        ),
                        save_dir=run_dir,
                        verbose_logs=False,
                        auto_review=False,
                    )
        self._last_native_provider_receipts = tuple(native_receipts)
        private_path, private_sha256 = _write_private_simulation(
            run_dir,
            simulation,
        )
        self._private_simulation_manifest.append(
            {
                "relative_path": private_path.relative_to(
                    self.config.private_output_root
                ).as_posix(),
                "sha256": private_sha256,
            }
        )
        return simulation

    def _native_episode(
        self,
        *,
        simulation: Any,
        decisions: tuple[RouterDecision, ...],
        roles: tuple[ModelRole, ...],
        router_receipts: tuple[CallReceipt, ...],
    ) -> Tau2NativeEpisode:
        public_transcript = tau2_public_history(
            tuple(getattr(simulation, "messages", ()) or ())
        )
        initial_user = _first_public_user_text(public_transcript)
        if self._bound_initial_user_message is None:
            self._bound_initial_user_message = initial_user
        elif self._bound_initial_user_message != initial_user:
            raise RuntimeError("tau2_pristine_restart_initial_user_mismatch")
        reward = self.score_selected_callback(simulation)
        generated_messages = _generated_assistant_messages(simulation)
        if len(generated_messages) != len(roles):
            raise RuntimeError("tau2_native_decision_agent_call_mismatch")
        if self.attempt_broker is None:
            native_receipts = tuple(
                _assistant_receipt(
                    message,
                    role=role,
                    model=self._model_by_role[role],
                )
                for message, role in zip(
                    generated_messages,
                    roles,
                    strict=True,
                )
            )
        else:
            native_receipts = self._last_native_provider_receipts
            agent_receipts = tuple(
                receipt
                for receipt in native_receipts
                if receipt.role in {"cheap_agent", "strong_agent"}
            )
            if len(agent_receipts) != len(roles):
                raise RuntimeError("tau2_brokered_agent_receipt_count_mismatch")
            for receipt, role in zip(agent_receipts, roles, strict=True):
                if (
                    receipt.role != f"{role}_agent"
                    or receipt.model != self._model_by_role[role]
                ):
                    raise RuntimeError("tau2_brokered_agent_receipt_role_mismatch")
        termination = getattr(
            getattr(simulation, "termination_reason", ""),
            "value",
            getattr(simulation, "termination_reason", ""),
        )
        return Tau2NativeEpisode(
            official_score=reward,
            termination_reason=str(termination),
            initial_public_user_message=initial_user,
            public_transcript=public_transcript,
            decisions=decisions,
            receipts=(*router_receipts, *native_receipts),
            simulation=simulation,
        )

    def _generated_episode(
        self,
        *,
        binding: Any,
        role: ModelRole,
        model: str,
        native: Tau2NativeEpisode,
    ) -> GeneratedEpisode:
        return GeneratedEpisode(
            role=role,
            model=model,
            initial_environment_sha256=(binding.initial_environment_sha256),
            environment_instance_sha256=_digest(
                {
                    "simulation_id": str(getattr(native.simulation, "id", "")),
                    "task_example_sha256": self.task_ref.example_sha256,
                }
            ),
            generated_answer_view=tau2_generated_answer_view(
                native.public_transcript
            ),
            receipts=native.receipts,
            opaque_score_handle=native.simulation,
            public_episode_history=native.public_transcript,
        )

    def _pristine_binding_sha256(self) -> str:
        record = {
            "protocol": "tau2.official.pristine.v1",
            "source_binding_sha256": self._source_binding_sha256,
            "task_example_sha256": self.task_ref.example_sha256,
            "seed": self.config.seed,
            "user_model": self.config.user_model,
            "max_steps": self.config.max_steps,
        }
        if self._task_split_name != TAU2_OFFICIAL_SPLIT:
            record["task_split_name"] = self._task_split_name
        return _digest(record)

    def _validate_model(self, role: ModelRole, model: str) -> None:
        if model != self._model_by_role[role]:
            raise ValueError("tau2_native_role_model_mismatch")


def tau2_public_history(
    messages: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    projected: list[dict[str, Any]] = []
    for message in messages:
        nested = getattr(message, "tool_messages", None)
        if nested is not None:
            projected.extend(tau2_public_history(tuple(nested)))
            continue
        role = str(getattr(message, "role", ""))
        if role == "system":
            continue
        if role in {"user", "assistant"}:
            item: dict[str, Any] = {"role": role}
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                item["content"] = content
            calls = getattr(message, "tool_calls", None)
            if calls:
                item["tool_calls"] = [
                    {
                        "id": str(getattr(call, "id", "")),
                        "name": str(getattr(call, "name", "")),
                        "arguments": deepcopy(getattr(call, "arguments", {})),
                    }
                    for call in calls
                ]
            projected.append(item)
        elif role == "tool":
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": str(getattr(message, "id", "")),
                    "output": str(getattr(message, "content", "") or ""),
                    "status": (
                        "error"
                        if bool(getattr(message, "error", False))
                        else "completed"
                    ),
                }
            )
        else:
            raise ValueError("tau2_native_message_role_unsupported")
    return validate_public_history(projected)


def tau2_generated_answer_view(
    public_transcript: Sequence[Mapping[str, Any]],
) -> str:
    """Return the exact canonical response view used by the native runtime."""

    return json.dumps(
        tuple(dict(item) for item in public_transcript),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _timed_native_decision(
    decide: Callable[[Any], RouterDecision],
    view: Any,
    method: str,
    receipts: list[CallReceipt],
    *,
    add_local_router_receipt: bool,
) -> RouterDecision:
    started = time.perf_counter()
    decision = decide(view)
    if add_local_router_receipt:
        receipts.append(
            CallReceipt.local_router(
                method=method,
                latency_seconds=max(0.0, time.perf_counter() - started),
            )
        )
    return decision


def _validate_routellm_prediction(
    prediction: ScalarPrediction,
) -> None:
    if not isinstance(prediction, ScalarPrediction):
        raise TypeError("tau2_routellm_scalar_prediction_required")
    embedding = tuple(
        receipt
        for receipt in prediction.receipts
        if receipt.model == ROUTELLM_EMBEDDING_MODEL
    )
    local_bert = tuple(
        receipt
        for receipt in prediction.receipts
        if receipt.role == "router"
        and receipt.model == ROUTELLM_LOCAL_MODEL
        and receipt.provider_calls == 0
        and receipt.reasoning_effort == "local"
    )
    if len(local_bert) == 1 and len(prediction.receipts) == 1:
        return
    if (
        len(embedding) != 1
        or embedding[0].role != "router"
        or embedding[0].provider_calls != 1
    ):
        raise ValueError("tau2_routellm_embedding_receipt_invalid")


def _validate_native_decision(
    decision: RouterDecision,
    artifact: Any,
) -> None:
    if not isinstance(decision, RouterDecision):
        raise TypeError("tau2_native_router_decision_invalid")
    if decision.method != artifact.method:
        raise ValueError("tau2_native_router_method_mismatch")
    if decision.surface != artifact.surface:
        raise ValueError("tau2_native_router_surface_mismatch")
    if decision.artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("tau2_native_router_artifact_mismatch")
    if decision.selected_model != artifact.model_for(decision.selected_role):
        raise ValueError("tau2_native_router_model_mismatch")


def _first_public_user_text(
    history: Sequence[Mapping[str, Any]],
) -> str:
    for item in history:
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise RuntimeError("tau2_native_initial_public_user_message_missing")


def _generated_assistant_messages(simulation: Any) -> tuple[Any, ...]:
    messages = tuple(getattr(simulation, "messages", ()) or ())
    return tuple(
        message
        for message in messages
        if getattr(message, "role", None) == "assistant"
        and (
            getattr(message, "usage", None) is not None
            or (
                isinstance(
                    (cost := getattr(message, "cost", None)),
                    (int, float),
                )
                and not isinstance(cost, bool)
                and cost > 0
            )
            or (
                isinstance(
                    (
                        generation_time := getattr(
                            message,
                            "generation_time_seconds",
                            None,
                        )
                    ),
                    (int, float),
                )
                and not isinstance(generation_time, bool)
                and generation_time > 0
            )
        )
    )


def _assistant_receipt(
    message: Any,
    *,
    role: ModelRole,
    model: str,
) -> CallReceipt:
    usage = getattr(message, "usage", None)
    resolved = usage if isinstance(usage, Mapping) else {}
    input_tokens = _optional_int(
        resolved.get("prompt_tokens", resolved.get("input_tokens"))
    )
    output_tokens = _optional_int(
        resolved.get("completion_tokens", resolved.get("output_tokens"))
    )
    cached_tokens = _optional_int(resolved.get("cached_input_tokens", 0))
    reasoning_tokens = _optional_int(resolved.get("reasoning_tokens", 0))
    return CallReceipt(
        role=f"{role}_agent",  # type: ignore[arg-type]
        phase="agent",
        model=model,
        reasoning_effort=REASONING_EFFORT,
        provider_calls=1,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        retry_count=0,
        latency_seconds=float(getattr(message, "generation_time_seconds", 0.0) or 0.0),
        provider_reported_cost_usd=_optional_float(getattr(message, "cost", None)),
    )


@contextmanager
def _instrument_provider_attempts(
    api: Tau2NativeAPI,
    *,
    budget: Tau2ProviderBudget,
    broker: ProviderAttemptBroker | None,
    receipts: list[CallReceipt],
    max_infra_retries: int,
):
    modules = (
        (api.agent_module, "agent"),
        (api.user_module, "user_simulator"),
        (api.evaluator_module, "evaluator"),
    )
    originals = tuple(module.generate for module, _ in modules)
    try:
        for (module, role), original in zip(
            modules,
            originals,
            strict=True,
        ):
            if broker is None:
                module.generate = budget.wrap(original)
            else:
                module.generate = _brokered_tau2_generate(
                    original,
                    broker=broker,
                    role=role,
                    receipts=receipts,
                    max_infra_retries=max_infra_retries,
                )
        yield
    finally:
        for (module, _), original in zip(
            modules,
            originals,
            strict=True,
        ):
            module.generate = original


@contextmanager
def _replay_bound_initial_user_message(
    api: Tau2NativeAPI,
    initial_user_message: str | None,
):
    """Replay one paired opening without claiming a provider attempt."""

    if initial_user_message is None:
        yield
        return
    message = _required_text(
        initial_user_message,
        "tau2_bound_initial_user_message",
    )
    module = api.user_module
    original = module.generate
    assistant_message = getattr(module, "AssistantMessage", None)
    text_factory = getattr(assistant_message, "text", None)
    if not callable(text_factory):
        raise RuntimeError(
            "tau2_bound_initial_user_message_factory_missing"
        )
    replayed = False

    def generate(*args: Any, **kwargs: Any) -> Any:
        nonlocal replayed
        if not replayed:
            if kwargs.get("call_name") != "user_simulator_response":
                raise RuntimeError(
                    "tau2_bound_initial_user_message_call_mismatch"
                )
            replayed = True
            return text_factory(
                message,
                cost=0.0,
                usage=None,
                raw_data=None,
                generation_time_seconds=0.0,
            )
        return original(*args, **kwargs)

    try:
        module.generate = generate
        yield
        if not replayed:
            raise RuntimeError(
                "tau2_bound_initial_user_message_not_consumed"
            )
    finally:
        module.generate = original


def _brokered_tau2_generate(
    generate: Callable[..., Any],
    *,
    broker: ProviderAttemptBroker,
    role: str,
    receipts: list[CallReceipt],
    max_infra_retries: int = 0,
) -> Callable[..., Any]:
    """Force every visible transport attempt to equal one ledger reservation."""

    if (
        isinstance(max_infra_retries, bool)
        or not isinstance(max_infra_retries, int)
        or max_infra_retries < 0
    ):
        raise ValueError("tau2_max_infra_retries_invalid")

    def brokered(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("num_retries") not in {None, 0}:
            raise ValueError("tau2_hidden_provider_retries_forbidden")
        call_kwargs = dict(kwargs)
        call_kwargs["num_retries"] = 0
        model = _provider_model(args, call_kwargs)
        retry_index = 0
        while True:
            returned: list[Any] = []
            retry_errors: list[BaseException] = []

            def raw_attempt() -> CallReceipt:
                started = time.perf_counter()
                try:
                    message = generate(*args, **call_kwargs)
                except BaseException as exc:
                    if (
                        retry_index < max_infra_retries
                        and _is_recoverable_tau2_provider_error(exc)
                    ):
                        retry_errors.append(exc)
                        return _tau2_infra_retry_receipt(
                            role=role,
                            model=model,
                            latency_seconds=time.perf_counter() - started,
                        )
                    raise
                returned.append(message)
                return _tau2_provider_receipt(
                    message,
                    role=role,
                    model=model,
                )

            receipt = broker.perform(raw_attempt)
            if retry_errors:
                if returned:
                    raise RuntimeError(
                        "tau2_infra_retry_returned_provider_message"
                    )
                receipts.append(replace(receipt, retry_count=1))
                retry_index += 1
                continue
            if len(returned) != 1:
                raise RuntimeError("tau2_provider_attempt_result_missing")
            receipts.append(receipt)
            return returned[0]

    return brokered


def build_brokered_tau2_agent_generate(
    generate: Callable[..., Any],
    *,
    broker: ProviderAttemptBroker,
    receipts: list[CallReceipt],
    max_infra_retries: int = 0,
) -> Callable[..., Any]:
    """Expose the native one-attempt accounting wrapper to injected agents."""

    if not callable(generate):
        raise TypeError("tau2_generate_callback_invalid")
    if not isinstance(broker, ProviderAttemptBroker):
        raise TypeError("tau2_provider_attempt_broker_invalid")
    if not isinstance(receipts, list):
        raise TypeError("tau2_provider_receipt_sink_invalid")
    return _brokered_tau2_generate(
        generate,
        broker=broker,
        role="agent",
        receipts=receipts,
        max_infra_retries=max_infra_retries,
    )


def _provider_model(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> str:
    raw = kwargs.get("model", args[0] if args else None)
    return _required_text(raw, "tau2_provider_model")


def _tau2_provider_receipt(
    message: Any,
    *,
    role: str,
    model: str,
) -> CallReceipt:
    usage = getattr(message, "usage", None)
    resolved = usage if isinstance(usage, Mapping) else {}
    input_tokens = _optional_int(
        resolved.get("prompt_tokens", resolved.get("input_tokens"))
    )
    output_tokens = _optional_int(
        resolved.get("completion_tokens", resolved.get("output_tokens"))
    )
    if input_tokens is None or output_tokens is None:
        raise ValueError("tau2_provider_attempt_usage_missing")
    if role == "agent":
        receipt_role = "cheap_agent" if model == CHEAP_MODEL else "strong_agent"
        if model not in {CHEAP_MODEL, STRONG_MODEL}:
            raise ValueError("tau2_provider_agent_model_not_frozen")
        phase = "agent"
    elif role in {"user_simulator", "evaluator"}:
        receipt_role = role
        phase = role
    else:
        raise ValueError("tau2_provider_receipt_role_invalid")
    return CallReceipt(
        role=receipt_role,  # type: ignore[arg-type]
        phase=phase,
        model=model,
        reasoning_effort=REASONING_EFFORT,
        provider_calls=1,
        input_tokens=input_tokens,
        cached_input_tokens=_optional_int(resolved.get("cached_input_tokens", 0)),
        output_tokens=output_tokens,
        reasoning_tokens=_optional_int(resolved.get("reasoning_tokens", 0)),
        retry_count=0,
        latency_seconds=float(getattr(message, "generation_time_seconds", 0.0) or 0.0),
        provider_reported_cost_usd=_optional_float(getattr(message, "cost", None)),
    )


def _is_recoverable_tau2_provider_error(exc: BaseException) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "apitimeouterror",
            "request timed out",
            "stream disconnected before completion",
            "stream closed before response.completed",
        )
    )


def _tau2_infra_retry_receipt(
    *,
    role: str,
    model: str,
    latency_seconds: float,
) -> CallReceipt:
    if role == "agent":
        if model not in {CHEAP_MODEL, STRONG_MODEL}:
            raise ValueError("tau2_provider_agent_model_not_frozen")
        receipt_role = "cheap_agent" if model == CHEAP_MODEL else "strong_agent"
    elif role in {"user_simulator", "evaluator"}:
        receipt_role = role
    else:
        raise ValueError("tau2_provider_receipt_role_invalid")
    return CallReceipt(
        role=receipt_role,  # type: ignore[arg-type]
        phase=f"{role}_infra_retry_no_usage_receipt",
        model=model,
        reasoning_effort=REASONING_EFFORT,
        provider_calls=1,
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        retry_count=0,
        latency_seconds=latency_seconds,
        provider_reported_cost_usd=0.0,
    )


@contextmanager
def _official_evaluator_config(
    api: Tau2NativeAPI,
    config: Tau2NativeRunConfig,
):
    module = api.evaluator_module
    original_model = module.DEFAULT_LLM_NL_ASSERTIONS
    original_args = module.DEFAULT_LLM_NL_ASSERTIONS_ARGS
    try:
        module.DEFAULT_LLM_NL_ASSERTIONS = config.evaluator_model
        module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = config.llm_args()
        yield
    finally:
        module.DEFAULT_LLM_NL_ASSERTIONS = original_model
        module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = original_args


def _write_private_simulation(
    run_dir: Path,
    simulation: Any,
) -> tuple[Path, str]:
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "official_simulation.json"
    payload = simulation.model_dump(mode="json")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("tau2_runtime_git_probe_failed")
    return completed.stdout.strip()


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()

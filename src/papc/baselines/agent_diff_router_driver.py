"""Protected Agent-Diff bindings for external task and turn routers.

The dataset locator, hidden expected state diff, environment handle, and
official scorer remain harness-only.  Router policies receive only the public
request or public ReAct history.  This module is provider-neutral and does not
import PAPC method code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    AUTOMIX,
    CHEAP_MODEL,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    REASONING_EFFORT,
    ROUTELLM,
    STRONG_MODEL,
    ExternalMethod,
    FullEpisodeBinding,
)
from papc.baselines.router_runtime import (
    ExecuteHistoryTurn,
    FullEpisodeRunner,
    HistoryRoutingResult,
    ScoreSelected,
    StaticRoutingResult,
    run_frugalgpt_episode,
    run_history_routed_episode,
    run_routellm_episode,
)
from papc.baselines.sealed_router_policy import (
    SealedRouterPolicy,
    require_sealed_router_policy,
)
from papc.baselines.surface_adapters import (
    AGENT_DIFF_FULL_DENOMINATOR,
    AGENT_DIFF_NATIVE_TERMINAL,
    AGENT_DIFF_OFFICIAL_SCORER,
    AGENT_DIFF_SERVICES,
    AGENT_DIFF_SURFACE,
    build_agent_diff_full_episode_binding,
)

AGENT_DIFF_SPLIT_GATE_SCHEMA = "papc.external_router_split_gate_noapi.v2"
AGENT_DIFF_DEV_SPLIT_SEED = "papc.external_router.dev_split.v1"
AGENT_DIFF_OFFICIAL_TRAIN_COUNT = 179
AGENT_DIFF_ROUTER_FIT_COUNT = 141
AGENT_DIFF_ROUTER_DEV_COUNT = 38
AGENT_DIFF_OFFICIAL_TEST_COUNT = AGENT_DIFF_FULL_DENOMINATOR
AGENT_DIFF_TEST_SERVICE_COUNTS = {
    "box": 10,
    "calendar": 12,
    "linear": 11,
    "slack": 12,
}

# Frozen before reading any external-router test result.  AutoMix and
# OpenSquilla remain method-eligible on Agent-Diff, but are omitted from the
# recommended result-blind compute subset rather than mislabeled N/A.
AGENT_DIFF_TASK_LEVEL_METHODS = frozenset({ROUTELLM, FRUGALGPT})
AGENT_DIFF_TURN_LEVEL_METHODS = frozenset({MTROUTER})
AGENT_DIFF_EXECUTED_METHODS = (
    ROUTELLM,
    FRUGALGPT,
    MTROUTER,
)
AGENT_DIFF_COMPUTE_SUBSET_OMISSIONS = {
    AUTOMIX: "result_blind_compute_subset_omitted",
    OPENSQUILLA_G0: "result_blind_compute_subset_omitted",
}

FROZEN_AGENT_DIFF_REPO_REVISION = "920abcedd8a5ef071892dd1177aab17fe909a36d"
FROZEN_AGENT_DIFF_SDK_SOURCE_SHA256 = (
    "cca1e878a559acea7f9b43345c5caad267ebaa67bbd05b9f3ff56462bf44e8d3"
)
FROZEN_AGENT_DIFF_SCORER_SHA256 = (
    "fbad09413c78bfbc215fc631caf873b3e2e3a1cd9081de2267a601d102caa33a"
)
FROZEN_AGENT_DIFF_BASELINE_AUTHORITY_SHA256 = (
    "3f1ffb1f018a81a01fffde89cacad69a2049a22f19f9f80a13443f9d073a804b"
)
AGENT_DIFF_SCORER_NAME = "agent_diff_evaluate_run_plus_registered_corrections_v1"
AGENT_DIFF_SCORER_CORRECTION_SCHEMA = "papc.agent_diff.scorer_corrections.v1"


@dataclass(frozen=True)
class AgentDiffOfficialTaskRef:
    """One official test row with private harness-only locators."""

    native_ordinal: int = field(repr=False)
    native_test_id: str = field(repr=False)
    service: str
    original_request: str = field(repr=False)
    payload_sha256: str
    example_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.native_ordinal, bool)
            or not isinstance(self.native_ordinal, int)
            or self.native_ordinal <= 0
        ):
            raise ValueError("agent_diff_native_ordinal_invalid")
        object.__setattr__(
            self,
            "native_test_id",
            _required_text(self.native_test_id, "agent_diff_native_test_id"),
        )
        service = _required_text(self.service, "agent_diff_service")
        if service not in AGENT_DIFF_SERVICES:
            raise ValueError("agent_diff_service_unsupported")
        object.__setattr__(self, "service", service)
        if not isinstance(self.original_request, str):
            raise TypeError("agent_diff_original_request_must_be_str")
        if not self.original_request.strip():
            raise ValueError("agent_diff_original_request_empty")
        for field_name in ("payload_sha256", "example_sha256"):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class AgentDiffSplitBinding:
    source_binding_sha256: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_example_sha256s: tuple[str, ...] = field(repr=False)
    dev_example_sha256s: tuple[str, ...] = field(repr=False)
    tasks: tuple[AgentDiffOfficialTaskRef, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "source_binding_sha256",
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        fit = tuple(
            _required_sha256(value, "agent_diff_fit_example_sha256")
            for value in self.fit_example_sha256s
        )
        dev = tuple(
            _required_sha256(value, "agent_diff_dev_example_sha256")
            for value in self.dev_example_sha256s
        )
        tasks = tuple(self.tasks)
        if len(fit) != AGENT_DIFF_ROUTER_FIT_COUNT:
            raise ValueError("agent_diff_router_fit_count_mismatch")
        if len(dev) != AGENT_DIFF_ROUTER_DEV_COUNT:
            raise ValueError("agent_diff_router_dev_count_mismatch")
        if len(tasks) != AGENT_DIFF_OFFICIAL_TEST_COUNT:
            raise ValueError("agent_diff_official_test_count_mismatch")
        if any(not isinstance(task, AgentDiffOfficialTaskRef) for task in tasks):
            raise TypeError("agent_diff_official_task_ref_invalid")
        test = tuple(task.example_sha256 for task in tasks)
        if set(fit) & set(dev) or set(fit) & set(test) or set(dev) & set(test):
            raise ValueError("agent_diff_fit_dev_test_overlap")
        if any(len(values) != len(set(values)) for values in (fit, dev, test)):
            raise ValueError("agent_diff_split_example_duplicate")
        if Counter(task.service for task in tasks) != Counter(
            AGENT_DIFF_TEST_SERVICE_COUNTS
        ):
            raise ValueError("agent_diff_test_service_counts_mismatch")
        object.__setattr__(self, "fit_example_sha256s", fit)
        object.__setattr__(self, "dev_example_sha256s", dev)
        object.__setattr__(self, "tasks", tasks)

    def public_record(self) -> dict[str, Any]:
        return {
            "surface": AGENT_DIFF_SURFACE,
            "official_train_count": AGENT_DIFF_OFFICIAL_TRAIN_COUNT,
            "router_fit_count": len(self.fit_example_sha256s),
            "router_dev_count": len(self.dev_example_sha256s),
            "official_test_count": len(self.tasks),
            "test_service_counts": dict(AGENT_DIFF_TEST_SERVICE_COUNTS),
            "source_binding_sha256": self.source_binding_sha256,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "native_test_ids_present": False,
            "raw_questions_present": False,
            "hidden_answers_present": False,
        }


@dataclass(frozen=True)
class AgentDiffOfficialRegistry:
    dataset_dir: Path = field(repr=False)
    split_gate_report_sha256: str
    split: AgentDiffSplitBinding

    def __post_init__(self) -> None:
        dataset = Path(self.dataset_dir).resolve()
        if not dataset.is_dir():
            raise FileNotFoundError("agent_diff_dataset_dir_missing")
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(
            self,
            "split_gate_report_sha256",
            _required_sha256(
                self.split_gate_report_sha256,
                "agent_diff_split_gate_report_sha256",
            ),
        )
        if not isinstance(self.split, AgentDiffSplitBinding):
            raise TypeError("agent_diff_split_binding_invalid")

    @property
    def tasks(self) -> tuple[AgentDiffOfficialTaskRef, ...]:
        return self.split.tasks

    @property
    def denominator(self) -> int:
        return AGENT_DIFF_OFFICIAL_TEST_COUNT

    def contains(self, task: AgentDiffOfficialTaskRef) -> bool:
        if not isinstance(task, AgentDiffOfficialTaskRef):
            return False
        return any(
            candidate.native_ordinal == task.native_ordinal
            and candidate.native_test_id == task.native_test_id
            and candidate.payload_sha256 == task.payload_sha256
            and candidate.example_sha256 == task.example_sha256
            for candidate in self.tasks
        )

    def public_record(self) -> dict[str, Any]:
        return {
            **self.split.public_record(),
            "split_gate_schema": AGENT_DIFF_SPLIT_GATE_SCHEMA,
            "split_gate_report_sha256": self.split_gate_report_sha256,
        }


def compute_agent_diff_split_binding(
    dataset_dir: Path,
) -> AgentDiffSplitBinding:
    """Recompute the exact 141/38/45 binding used by the split gate."""

    dataset = Path(dataset_dir).resolve()
    train_path = dataset / "train.jsonl"
    test_path = dataset / "test.jsonl"
    train_rows = _read_jsonl(train_path)
    test_rows = _read_jsonl(test_path)
    if len(train_rows) != AGENT_DIFF_OFFICIAL_TRAIN_COUNT:
        raise ValueError("agent_diff_official_train_count_mismatch")
    if len(test_rows) != AGENT_DIFF_OFFICIAL_TEST_COUNT:
        raise ValueError("agent_diff_official_test_count_mismatch")
    source_binding = _source_binding_sha256((train_path, test_path))
    train_sources = tuple(_source_row(raw) for raw in train_rows)
    test_sources = tuple(_source_row(raw) for raw in test_rows)
    fit_sources, dev_sources = _deterministic_fit_dev(train_sources)
    fit_examples = tuple(row.example_sha256 for row in fit_sources)
    dev_examples = tuple(row.example_sha256 for row in dev_sources)
    test_examples = tuple(row.example_sha256 for row in test_sources)
    tasks = tuple(
        AgentDiffOfficialTaskRef(
            native_ordinal=ordinal,
            native_test_id=_required_text(
                raw.get("test_id"),
                "agent_diff_test_id",
            ),
            service=_required_text(raw.get("service"), "agent_diff_service"),
            original_request=_required_text(
                raw.get("question"),
                "agent_diff_question",
            ),
            payload_sha256=source.payload_sha256,
            example_sha256=source.example_sha256,
        )
        for ordinal, (raw, source) in enumerate(
            zip(test_rows, test_sources, strict=True),
            start=1,
        )
    )
    return AgentDiffSplitBinding(
        source_binding_sha256=source_binding,
        fit_registry_sha256=_registry_sha256(
            role="router_fit",
            source_binding_sha256=source_binding,
            example_sha256s=fit_examples,
        ),
        dev_registry_sha256=_registry_sha256(
            role="router_dev",
            source_binding_sha256=source_binding,
            example_sha256s=dev_examples,
        ),
        test_registry_sha256=_registry_sha256(
            role="official_test",
            source_binding_sha256=source_binding,
            example_sha256s=test_examples,
        ),
        fit_example_sha256s=fit_examples,
        dev_example_sha256s=dev_examples,
        tasks=tasks,
    )


def load_agent_diff_official_registry(
    dataset_dir: Path,
    *,
    split_gate_json: Path,
) -> AgentDiffOfficialRegistry:
    gate_path = Path(split_gate_json).resolve()
    gate = _read_json(gate_path)
    if not isinstance(gate, Mapping):
        raise TypeError("agent_diff_split_gate_not_mapping")
    if gate.get("schema") != AGENT_DIFF_SPLIT_GATE_SCHEMA:
        raise ValueError("agent_diff_split_gate_schema_mismatch")
    if gate.get("decision") != "ready_for_training_collection_plan":
        raise ValueError("agent_diff_split_gate_not_ready")
    surface_splits = gate.get("surface_splits")
    if not isinstance(surface_splits, Mapping):
        raise TypeError("agent_diff_split_gate_surfaces_invalid")
    agent_gate = surface_splits.get(AGENT_DIFF_SURFACE)
    if not isinstance(agent_gate, Mapping):
        raise TypeError("agent_diff_split_gate_entry_missing")

    split = compute_agent_diff_split_binding(dataset_dir)
    expected = {
        "official_train_count": AGENT_DIFF_OFFICIAL_TRAIN_COUNT,
        "router_fit_count": AGENT_DIFF_ROUTER_FIT_COUNT,
        "router_dev_count": AGENT_DIFF_ROUTER_DEV_COUNT,
        "official_test_count": AGENT_DIFF_OFFICIAL_TEST_COUNT,
        "source_binding_sha256": split.source_binding_sha256,
        "fit_registry_sha256": split.fit_registry_sha256,
        "dev_registry_sha256": split.dev_registry_sha256,
        "test_registry_sha256": split.test_registry_sha256,
    }
    for key, expected_value in expected.items():
        if agent_gate.get(key) != expected_value:
            raise ValueError(f"agent_diff_split_gate_{key}_mismatch")
    return AgentDiffOfficialRegistry(
        dataset_dir=Path(dataset_dir),
        split_gate_report_sha256=_file_sha256(gate_path),
        split=split,
    )


@dataclass(frozen=True)
class AgentDiffRouterDriver:
    """Bind one fitted external router to the frozen Agent-Diff subset."""

    policy: SealedRouterPolicy
    registry: AgentDiffOfficialRegistry

    def __post_init__(self) -> None:
        require_sealed_router_policy(self.policy)
        if not isinstance(self.registry, AgentDiffOfficialRegistry):
            raise TypeError("agent_diff_router_registry_invalid")
        if self.artifact.method not in AGENT_DIFF_EXECUTED_METHODS:
            raise ValueError("agent_diff_method_not_in_frozen_compute_subset")
        if self.artifact.surface != AGENT_DIFF_SURFACE:
            raise ValueError("agent_diff_router_surface_mismatch")
        expected_hashes = (
            self.registry.split.fit_registry_sha256,
            self.registry.split.dev_registry_sha256,
            self.registry.split.test_registry_sha256,
        )
        artifact_hashes = (
            self.artifact.fit_registry_sha256,
            self.artifact.dev_registry_sha256,
            self.artifact.test_registry_sha256,
        )
        if artifact_hashes != expected_hashes:
            raise ValueError("agent_diff_router_split_registry_mismatch")
        if (
            self.artifact.cheap_model != CHEAP_MODEL
            or self.artifact.strong_model != STRONG_MODEL
        ):
            raise ValueError("agent_diff_router_model_plan_not_frozen")
        if (
            self.artifact.cheap_reasoning_effort != REASONING_EFFORT
            or self.artifact.strong_reasoning_effort != REASONING_EFFORT
        ):
            raise ValueError("agent_diff_router_reasoning_effort_not_frozen")

    @property
    def artifact(self):
        return self.policy.binding

    def bind_public_episode(
        self,
        *,
        task: AgentDiffOfficialTaskRef,
        initial_environment_sha256: str,
    ) -> FullEpisodeBinding:
        self._require_official_task(task)
        return build_agent_diff_full_episode_binding(
            service=task.service,
            original_request=task.original_request,
            initial_environment_sha256=initial_environment_sha256,
        )

    def run_routellm(
        self,
        *,
        episode: FullEpisodeBinding,
        run_episode: FullEpisodeRunner,
        score_selected: ScoreSelected,
    ) -> StaticRoutingResult:
        self._require_method(ROUTELLM)
        self._require_episode(episode)
        result = run_routellm_episode(
            artifact=self.artifact,
            episode=episode,
            predict_strong_win_rate=self.policy.predict_routellm,
            threshold=self.policy.routellm_threshold,
            run_episode=run_episode,
            score_selected=score_selected,
        )
        _validate_online_receipts(
            result.receipts,
            required_agent_roles=(result.selected_role,),
        )
        return result

    def run_frugalgpt(
        self,
        *,
        episode: FullEpisodeBinding,
        run_episode: FullEpisodeRunner,
        score_selected: ScoreSelected,
    ) -> StaticRoutingResult:
        self._require_method(FRUGALGPT)
        self._require_episode(episode)
        result = run_frugalgpt_episode(
            artifact=self.artifact,
            episode=episode,
            predict_cheap_quality=self.policy.predict_frugalgpt,
            accept_threshold=self.policy.frugalgpt_accept_threshold,
            run_episode=run_episode,
            score_selected=score_selected,
        )
        required_roles = (
            ("cheap", "strong") if result.selected_role == "strong" else ("cheap",)
        )
        _validate_online_receipts(
            result.receipts,
            required_agent_roles=required_roles,
        )
        return result

    def run_mtrouter(
        self,
        *,
        task: AgentDiffOfficialTaskRef,
        turn_cap: int,
        execute_turn: ExecuteHistoryTurn,
        score_terminal: ScoreSelected,
    ) -> HistoryRoutingResult:
        self._require_method(MTROUTER)
        self._require_official_task(task)
        result = run_history_routed_episode(
            artifact=self.artifact,
            original_request=task.original_request,
            initial_public_history=(
                {"role": "user", "content": task.original_request},
            ),
            turn_cap=turn_cap,
            decide_next=self.policy.history_decider(),
            execute_turn=execute_turn,
            score_terminal=score_terminal,
        )
        _validate_online_receipts(
            result.receipts,
            required_agent_roles=result.selected_roles,
        )
        return result

    def public_record(self) -> dict[str, Any]:
        return {
            "method": self.artifact.method,
            "surface": AGENT_DIFF_SURFACE,
            "execution_level": router_execution_level(self.artifact.method),
            "fit_registry_sha256": self.registry.split.fit_registry_sha256,
            "dev_registry_sha256": self.registry.split.dev_registry_sha256,
            "test_registry_sha256": self.registry.split.test_registry_sha256,
            "official_test_count": self.registry.denominator,
            "cheap_model": self.artifact.cheap_model,
            "cheap_reasoning_effort": self.artifact.cheap_reasoning_effort,
            "strong_model": self.artifact.strong_model,
            "strong_reasoning_effort": self.artifact.strong_reasoning_effort,
            "sealed_policy": True,
            "router_inputs_exclude_test_id": True,
            "router_inputs_exclude_hidden_answer": True,
            "compute_subset_omissions": dict(AGENT_DIFF_COMPUTE_SUBSET_OMISSIONS),
        }

    def _require_method(self, method: ExternalMethod) -> None:
        if self.artifact.method != method:
            raise ValueError("agent_diff_router_method_mismatch")

    def _require_episode(self, episode: FullEpisodeBinding) -> None:
        if not isinstance(episode, FullEpisodeBinding):
            raise TypeError("agent_diff_full_episode_binding_invalid")
        if episode.surface != AGENT_DIFF_SURFACE:
            raise ValueError("agent_diff_episode_surface_mismatch")
        if episode.official_denominator != self.registry.denominator:
            raise ValueError("agent_diff_episode_denominator_mismatch")

    def _require_official_task(self, task: AgentDiffOfficialTaskRef) -> None:
        if not self.registry.contains(task):
            raise ValueError("agent_diff_task_not_in_official_test_registry")


def router_execution_level(method: ExternalMethod) -> str:
    if method in AGENT_DIFF_TASK_LEVEL_METHODS:
        return "task"
    if method in AGENT_DIFF_TURN_LEVEL_METHODS:
        return "turn"
    if method in AGENT_DIFF_COMPUTE_SUBSET_OMISSIONS:
        return "compute_subset_omitted"
    raise ValueError("agent_diff_router_method_invalid")


@dataclass(frozen=True)
class AgentDiffNativeAPI:
    sdk_path: Path
    repo_revision: str
    sdk_source_sha256: str
    scorer_sha256: str
    baseline_authority_sha256: str
    agent_diff_class: type = field(repr=False)
    bash_executor_class: type = field(repr=False)
    apply_scorer_corrections: Callable[..., dict[str, Any]] = field(repr=False)

    def public_record(self) -> dict[str, Any]:
        return {
            "repo_revision": self.repo_revision,
            "sdk_source_sha256": self.sdk_source_sha256,
            "scorer_sha256": self.scorer_sha256,
            "baseline_authority_sha256": self.baseline_authority_sha256,
            "native_agent_loop": "react_xml_bash_executor_proxy",
            "native_terminal_signal": AGENT_DIFF_NATIVE_TERMINAL,
            "official_scorer": AGENT_DIFF_OFFICIAL_SCORER,
            "registered_scorer": AGENT_DIFF_SCORER_NAME,
            "scorer_correction_schema": (AGENT_DIFF_SCORER_CORRECTION_SCHEMA),
        }


def load_agent_diff_native_api(
    *,
    sdk_path: Path,
    scorer_path: Path,
    baseline_authority_path: Path,
) -> AgentDiffNativeAPI:
    """Bind SDK and scorer code without starting an environment or provider."""

    sdk = Path(sdk_path).resolve()
    scorer = Path(scorer_path).resolve()
    authority = Path(baseline_authority_path).resolve()
    if not sdk.is_dir():
        raise FileNotFoundError("agent_diff_sdk_path_missing")
    if not scorer.is_file():
        raise FileNotFoundError("agent_diff_scorer_path_missing")
    if not authority.is_file():
        raise FileNotFoundError("agent_diff_baseline_authority_path_missing")
    revision = _git_output(sdk, "rev-parse", "HEAD")
    if revision != FROZEN_AGENT_DIFF_REPO_REVISION:
        raise ValueError("agent_diff_repo_revision_not_frozen")
    sdk_sha256 = _python_tree_sha256(sdk)
    if sdk_sha256 != FROZEN_AGENT_DIFF_SDK_SOURCE_SHA256:
        raise ValueError("agent_diff_sdk_source_not_frozen")
    scorer_sha256 = _file_sha256(scorer)
    if scorer_sha256 != FROZEN_AGENT_DIFF_SCORER_SHA256:
        raise ValueError("agent_diff_scorer_source_not_frozen")
    authority_sha256 = _file_sha256(authority)
    if authority_sha256 != FROZEN_AGENT_DIFF_BASELINE_AUTHORITY_SHA256:
        raise ValueError("agent_diff_baseline_authority_not_frozen")

    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    import agent_diff  # type: ignore

    scoring = _load_module_from_path(
        "_papc_agent_diff_registered_scoring",
        scorer,
    )
    if getattr(scoring, "SCHEMA", None) != AGENT_DIFF_SCORER_CORRECTION_SCHEMA:
        raise ValueError("agent_diff_scorer_schema_mismatch")
    if getattr(scoring, "SCORER_NAME", None) != AGENT_DIFF_SCORER_NAME:
        raise ValueError("agent_diff_scorer_name_mismatch")
    apply_corrections = getattr(
        scoring,
        "apply_registered_scorer_corrections",
        None,
    )
    if not callable(apply_corrections):
        raise RuntimeError("agent_diff_registered_scorer_api_missing")
    agent_class = getattr(agent_diff, "AgentDiff", None)
    bash_class = getattr(agent_diff, "BashExecutorProxy", None)
    if not isinstance(agent_class, type) or not isinstance(bash_class, type):
        raise RuntimeError("agent_diff_native_sdk_api_missing")
    return AgentDiffNativeAPI(
        sdk_path=sdk,
        repo_revision=revision,
        sdk_source_sha256=sdk_sha256,
        scorer_sha256=scorer_sha256,
        baseline_authority_sha256=authority_sha256,
        agent_diff_class=agent_class,
        bash_executor_class=bash_class,
        apply_scorer_corrections=apply_corrections,
    )


@dataclass(frozen=True)
class AgentDiffNativeRunConfig:
    api_base: str
    agent_diff_base_url: str
    private_output_root: Path
    max_provider_calls: int
    max_observed_provider_cost_usd: float
    max_iterations: int = 40
    max_model_tokens: int = 1600
    reasoning_effort: str = REASONING_EFFORT

    def __post_init__(self) -> None:
        for field_name in ("api_base", "agent_diff_base_url"):
            value = _required_text(getattr(self, field_name), field_name)
            if not value.startswith(("http://", "https://")):
                raise ValueError(f"agent_diff_{field_name}_invalid")
            object.__setattr__(self, field_name, value.rstrip("/"))
        output = Path(self.private_output_root)
        if not output.is_absolute():
            raise ValueError("agent_diff_private_output_root_not_absolute")
        object.__setattr__(self, "private_output_root", output)
        for field_name in (
            "max_provider_calls",
            "max_iterations",
            "max_model_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"agent_diff_{field_name}_invalid")
        cap = self.max_observed_provider_cost_usd
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or not math.isfinite(float(cap))
            or float(cap) <= 0.0
        ):
            raise ValueError("agent_diff_provider_cost_cap_invalid")
        object.__setattr__(
            self,
            "max_observed_provider_cost_usd",
            float(cap),
        )
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("agent_diff_reasoning_effort_not_frozen")

    def public_record(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "agent_diff_base_url": self.agent_diff_base_url,
            "private_output_root": str(self.private_output_root),
            "max_provider_calls": self.max_provider_calls,
            "max_observed_provider_cost_usd": (self.max_observed_provider_cost_usd),
            "max_iterations": self.max_iterations,
            "max_model_tokens": self.max_model_tokens,
            "reasoning_effort": self.reasoning_effort,
        }


class AgentDiffProviderBudgetExceeded(RuntimeError):
    pass


@dataclass
class AgentDiffProviderBudget:
    """Pre-call provider cap plus post-receipt observed-spend stop."""

    max_calls: int
    max_observed_cost_usd: float
    calls: int = 0
    observed_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_calls, bool)
            or not isinstance(self.max_calls, int)
            or self.max_calls <= 0
        ):
            raise ValueError("agent_diff_provider_call_cap_invalid")
        if (
            isinstance(self.max_observed_cost_usd, bool)
            or not isinstance(self.max_observed_cost_usd, (int, float))
            or not math.isfinite(float(self.max_observed_cost_usd))
            or float(self.max_observed_cost_usd) <= 0.0
        ):
            raise ValueError("agent_diff_provider_cost_cap_invalid")
        self.max_observed_cost_usd = float(self.max_observed_cost_usd)

    def ensure_call_capacity(self, requested_calls: int = 1) -> None:
        if (
            isinstance(requested_calls, bool)
            or not isinstance(requested_calls, int)
            or requested_calls <= 0
        ):
            raise ValueError("agent_diff_requested_provider_calls_invalid")
        if self.calls + requested_calls > self.max_calls:
            raise AgentDiffProviderBudgetExceeded(
                "agent_diff_provider_call_cap_exhausted"
            )

    def charge_receipts(self, receipts: Sequence[CallReceipt]) -> None:
        resolved = tuple(receipts)
        if not resolved:
            raise ValueError("agent_diff_provider_receipts_empty")
        if any(not isinstance(receipt, CallReceipt) for receipt in resolved):
            raise TypeError("agent_diff_provider_receipt_invalid")
        provider_receipts = tuple(
            receipt for receipt in resolved if receipt.provider_calls > 0
        )
        provider_calls = sum(receipt.provider_calls for receipt in provider_receipts)
        if provider_calls:
            self.ensure_call_capacity(provider_calls)
        if any(
            receipt.provider_reported_cost_usd is None for receipt in provider_receipts
        ):
            raise AgentDiffProviderBudgetExceeded(
                "agent_diff_observed_provider_cost_unavailable"
            )
        self.calls += provider_calls
        self.observed_cost_usd += sum(
            float(receipt.provider_reported_cost_usd)
            for receipt in provider_receipts
            if receipt.provider_reported_cost_usd is not None
        )
        if self.observed_cost_usd > self.max_observed_cost_usd:
            raise AgentDiffProviderBudgetExceeded(
                "agent_diff_observed_provider_cost_cap_exceeded"
            )

    def public_record(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls,
            "observed_provider_cost_usd": round(
                self.observed_cost_usd,
                9,
            ),
            "max_provider_calls": self.max_calls,
            "max_observed_provider_cost_usd": self.max_observed_cost_usd,
            "provider_call_cap_is_pre_call_hard_stop": True,
            "observed_usd_is_post_receipt_stop": True,
            "strict_prepaid_usd_ceiling": False,
        }


def _validate_online_receipts(
    receipts: Sequence[CallReceipt],
    *,
    required_agent_roles: Sequence[str],
) -> None:
    resolved = tuple(receipts)
    if not resolved:
        raise RuntimeError("agent_diff_online_receipts_empty")
    provider_receipts = tuple(
        receipt for receipt in resolved if receipt.provider_calls > 0
    )
    if not provider_receipts:
        raise RuntimeError("agent_diff_provider_receipts_missing")
    for receipt in provider_receipts:
        if receipt.input_tokens is None or receipt.output_tokens is None:
            raise RuntimeError("agent_diff_provider_usage_incomplete")
        if receipt.provider_reported_cost_usd is None:
            raise RuntimeError("agent_diff_observed_provider_cost_missing")
    required_counts = Counter(f"{role}_agent" for role in required_agent_roles)
    observed_counts = Counter()
    for receipt in provider_receipts:
        if receipt.role in {"cheap_agent", "strong_agent"}:
            observed_counts[receipt.role] += receipt.provider_calls
            expected_model = (
                CHEAP_MODEL if receipt.role == "cheap_agent" else STRONG_MODEL
            )
            if receipt.model != expected_model:
                raise RuntimeError("agent_diff_agent_receipt_model_mismatch")
    if any(observed_counts[role] < count for role, count in required_counts.items()):
        raise RuntimeError("agent_diff_agent_turn_receipt_missing")


@dataclass(frozen=True)
class _SourceRow:
    stratum: str
    source_identity: str
    payload_sha256: str

    @property
    def example_sha256(self) -> str:
        return _digest(
            {
                "surface": AGENT_DIFF_SURFACE,
                "stratum": self.stratum,
                "source_identity": self.source_identity,
                "payload_sha256": self.payload_sha256,
            }
        )


def _source_row(raw: Mapping[str, Any]) -> _SourceRow:
    return _SourceRow(
        stratum=_required_text(raw.get("service"), "agent_diff_service"),
        source_identity=_required_text(
            raw.get("test_id"),
            "agent_diff_test_id",
        ),
        payload_sha256=_digest(raw),
    )


def _deterministic_fit_dev(
    rows: Sequence[_SourceRow],
) -> tuple[tuple[_SourceRow, ...], tuple[_SourceRow, ...]]:
    by_stratum: dict[str, list[_SourceRow]] = defaultdict(list)
    for row in rows:
        by_stratum[row.stratum].append(row)
    fit: list[_SourceRow] = []
    dev: list[_SourceRow] = []
    for stratum, stratum_rows in sorted(by_stratum.items()):
        ordered = sorted(
            stratum_rows,
            key=lambda row: _digest(
                {
                    "seed": AGENT_DIFF_DEV_SPLIT_SEED,
                    "stratum": stratum,
                    "example_sha256": row.example_sha256,
                }
            ),
        )
        dev_count = max(1, (len(ordered) + 4) // 5)
        if dev_count >= len(ordered):
            raise ValueError("agent_diff_stratum_too_small_for_fit_dev")
        dev.extend(ordered[:dev_count])
        fit.extend(ordered[dev_count:])
    return (
        tuple(sorted(fit, key=lambda row: row.example_sha256)),
        tuple(sorted(dev, key=lambda row: row.example_sha256)),
    )


def _registry_sha256(
    *,
    role: str,
    source_binding_sha256: str,
    example_sha256s: Sequence[str],
) -> str:
    return _digest(
        {
            "role": role,
            "surface": AGENT_DIFF_SURFACE,
            "source_binding_sha256": source_binding_sha256,
            "example_sha256s": tuple(example_sha256s),
        }
    )


def _source_binding_sha256(paths: Sequence[Path]) -> str:
    return _digest(
        [
            {
                "name": path.name,
                "parent": path.parent.name,
                "sha256": _file_sha256(path),
            }
            for path in paths
        ]
    )


def _python_tree_sha256(root: Path) -> str:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*.py"))
    ]
    if not rows:
        raise FileNotFoundError("agent_diff_sdk_python_sources_missing")
    return _digest(rows)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("agent_diff_scorer_module_spec_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("agent_diff_jsonl_row_not_mapping")
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name}_invalid")
    return text


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

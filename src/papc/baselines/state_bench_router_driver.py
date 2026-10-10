"""Protected STATE-Bench bindings for external task- and turn-level routers.

The native task locator and task-local environment remain harness-only.  A
router sees only the opening user message or the public conversation history.
The official test registry is recomputed from the same task-plus-environment
payloads used by the external-router split gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    run_automix_episode,
    run_frugalgpt_episode,
    run_history_routed_episode,
    run_routellm_episode,
)
from papc.baselines.sealed_router_policy import (
    SealedRouterPolicy,
    require_sealed_router_policy,
)
from papc.baselines.surface_adapters import (
    STATE_BENCH_DOMAINS,
    STATE_BENCH_FULL_DENOMINATOR,
    STATE_BENCH_SURFACE,
    build_state_bench_full_episode_binding,
)

STATE_BENCH_SPLIT_GATE_SCHEMA = "papc.external_router_split_gate_noapi.v2"
STATE_BENCH_TEST_DOMAINS = (
    "customer_support",
    "shopping_assistant",
    "travel",
)
STATE_BENCH_TASK_LEVEL_METHODS = frozenset({ROUTELLM, FRUGALGPT, AUTOMIX})
STATE_BENCH_TURN_LEVEL_METHODS = frozenset({MTROUTER, OPENSQUILLA_G0})
STATE_BENCH_TEST_PER_DOMAIN = 50
STATE_BENCH_TRAIN_PER_DOMAIN = 100


@dataclass(frozen=True)
class StateBenchOfficialTaskRef:
    """Harness-only locator for one source-bound official test task."""

    domain: str
    native_task_locator: str = field(repr=False)
    native_task_path: Path = field(repr=False)
    initial_environment_path: Path = field(repr=False)
    payload_sha256: str
    initial_environment_sha256: str
    example_sha256: str

    def __post_init__(self) -> None:
        domain = _required_domain(self.domain)
        locator = _required_text(
            self.native_task_locator,
            "state_bench_native_task_locator",
        )
        if Path(locator).name != locator:
            raise ValueError("state_bench_native_task_locator_not_atomic")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "native_task_locator", locator)
        for field_name in ("native_task_path", "initial_environment_path"):
            path = Path(getattr(self, field_name)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"state_bench_{field_name}_missing")
            object.__setattr__(self, field_name, path)
        for field_name in (
            "payload_sha256",
            "initial_environment_sha256",
            "example_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class StateBenchOfficialTestRegistry:
    state_repo: Path = field(repr=False)
    split_gate_report_sha256: str
    source_binding_sha256: str
    test_registry_sha256: str
    tasks: tuple[StateBenchOfficialTaskRef, ...] = field(repr=False)

    def __post_init__(self) -> None:
        repo = Path(self.state_repo).resolve()
        if not repo.is_dir():
            raise FileNotFoundError("state_bench_repo_missing")
        object.__setattr__(self, "state_repo", repo)
        for field_name in (
            "split_gate_report_sha256",
            "source_binding_sha256",
            "test_registry_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        tasks = tuple(self.tasks)
        if len(tasks) != STATE_BENCH_FULL_DENOMINATOR:
            raise ValueError("state_bench_official_test_denominator_mismatch")
        if any(not isinstance(task, StateBenchOfficialTaskRef) for task in tasks):
            raise TypeError("state_bench_registry_task_ref_invalid")
        domain_counts = Counter(task.domain for task in tasks)
        if domain_counts != Counter(
            {domain: STATE_BENCH_TEST_PER_DOMAIN for domain in STATE_BENCH_TEST_DOMAINS}
        ):
            raise ValueError("state_bench_official_test_domain_counts_mismatch")
        identities = {(task.domain, task.native_task_locator) for task in tasks}
        if len(identities) != len(tasks):
            raise ValueError("state_bench_registry_task_locator_duplicate")
        if len({task.example_sha256 for task in tasks}) != len(tasks):
            raise ValueError("state_bench_registry_example_sha256_duplicate")
        object.__setattr__(self, "tasks", tasks)

    @property
    def denominator(self) -> int:
        return STATE_BENCH_FULL_DENOMINATOR

    def contains(self, task: StateBenchOfficialTaskRef) -> bool:
        if not isinstance(task, StateBenchOfficialTaskRef):
            return False
        return any(
            candidate.domain == task.domain
            and candidate.native_task_locator == task.native_task_locator
            and candidate.native_task_path == task.native_task_path
            and candidate.initial_environment_path == task.initial_environment_path
            and candidate.payload_sha256 == task.payload_sha256
            and candidate.initial_environment_sha256 == task.initial_environment_sha256
            and candidate.example_sha256 == task.example_sha256
            for candidate in self.tasks
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "surface": STATE_BENCH_SURFACE,
            "official_test_count": self.denominator,
            "domain_counts": {
                domain: sum(task.domain == domain for task in self.tasks)
                for domain in STATE_BENCH_TEST_DOMAINS
            },
            "split_gate_schema": STATE_BENCH_SPLIT_GATE_SCHEMA,
            "split_gate_report_sha256": self.split_gate_report_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "example_sha256s": [task.example_sha256 for task in self.tasks],
            "task_and_initial_environment_payloads_bound": True,
            "raw_task_locators_present": False,
            "raw_environment_paths_present": False,
        }


def load_state_bench_official_test_registry(
    state_repo: Path,
    *,
    split_gate_json: Path,
) -> StateBenchOfficialTestRegistry:
    """Recompute and bind the v2 official-test registry provider-free."""

    repo = Path(state_repo).resolve()
    gate_path = Path(split_gate_json).resolve()
    gate = _read_json(gate_path)
    if not isinstance(gate, Mapping):
        raise TypeError("state_bench_split_gate_not_mapping")
    if gate.get("schema") != STATE_BENCH_SPLIT_GATE_SCHEMA:
        raise ValueError("state_bench_split_gate_schema_mismatch")
    if gate.get("decision") != "ready_for_training_collection_plan":
        raise ValueError("state_bench_split_gate_not_ready")
    surface_splits = gate.get("surface_splits")
    if not isinstance(surface_splits, Mapping):
        raise TypeError("state_bench_split_gate_surfaces_invalid")
    state_gate = surface_splits.get(STATE_BENCH_SURFACE)
    if not isinstance(state_gate, Mapping):
        raise TypeError("state_bench_split_gate_entry_missing")
    if state_gate.get("official_test_count") != STATE_BENCH_FULL_DENOMINATOR:
        raise ValueError("state_bench_split_gate_test_count_mismatch")
    if state_gate.get("official_train_count") != (
        STATE_BENCH_TRAIN_PER_DOMAIN * len(STATE_BENCH_TEST_DOMAINS)
    ):
        raise ValueError("state_bench_split_gate_train_count_mismatch")

    split_paths = tuple(
        repo / "state_bench" / "domains" / domain / "splits" / "train_test.json"
        for domain in STATE_BENCH_TEST_DOMAINS
    )
    source_paths: list[Path] = list(split_paths)
    task_refs: list[StateBenchOfficialTaskRef] = []
    for domain, split_path in zip(
        STATE_BENCH_TEST_DOMAINS,
        split_paths,
        strict=True,
    ):
        split = _read_json(split_path)
        if not isinstance(split, Mapping):
            raise TypeError("state_bench_split_payload_not_mapping")
        splits = split.get("splits")
        if not isinstance(splits, Mapping):
            raise TypeError("state_bench_split_roles_not_mapping")
        raw_test = splits.get("test")
        raw_train = splits.get("train")
        if not isinstance(raw_train, list):
            raise TypeError("state_bench_train_split_not_list")
        if len(raw_train) != STATE_BENCH_TRAIN_PER_DOMAIN:
            raise ValueError("state_bench_domain_train_count_mismatch")
        if not isinstance(raw_test, list):
            raise TypeError("state_bench_test_split_not_list")
        if len(raw_test) != STATE_BENCH_TEST_PER_DOMAIN:
            raise ValueError("state_bench_domain_test_count_mismatch")
        for split_name in ("train", "test"):
            raw_locators = splits.get(split_name)
            if not isinstance(raw_locators, list):
                raise TypeError("state_bench_split_role_not_list")
            for raw_locator in raw_locators:
                (
                    task_ref,
                    task_path,
                    environment_path,
                ) = _load_task_ref(
                    repo,
                    domain=domain,
                    raw_locator=raw_locator,
                )
                source_paths.extend((task_path, environment_path))
                if split_name == "test":
                    task_refs.append(task_ref)

    source_binding_sha256 = _source_binding_sha256(source_paths)
    expected_source_binding = _required_sha256(
        state_gate.get("source_binding_sha256"),
        "state_bench_gate_source_binding_sha256",
    )
    if source_binding_sha256 != expected_source_binding:
        raise ValueError("state_bench_split_gate_source_binding_mismatch")
    example_sha256s = tuple(task.example_sha256 for task in task_refs)
    test_registry_sha256 = _digest(
        {
            "role": "official_test",
            "surface": STATE_BENCH_SURFACE,
            "source_binding_sha256": source_binding_sha256,
            "example_sha256s": example_sha256s,
        }
    )
    expected_test_registry = _required_sha256(
        state_gate.get("test_registry_sha256"),
        "state_bench_gate_test_registry_sha256",
    )
    if test_registry_sha256 != expected_test_registry:
        raise ValueError("state_bench_split_gate_test_registry_mismatch")
    return StateBenchOfficialTestRegistry(
        state_repo=repo,
        split_gate_report_sha256=_file_sha256(gate_path),
        source_binding_sha256=source_binding_sha256,
        test_registry_sha256=test_registry_sha256,
        tasks=tuple(task_refs),
    )


@dataclass(frozen=True)
class StateBenchRouterDriver:
    """Bind one frozen router artifact to the v2 official test registry."""

    policy: SealedRouterPolicy
    registry: StateBenchOfficialTestRegistry

    def __post_init__(self) -> None:
        require_sealed_router_policy(self.policy)
        if not isinstance(self.registry, StateBenchOfficialTestRegistry):
            raise TypeError("state_bench_router_registry_invalid")
        if self.artifact.surface != STATE_BENCH_SURFACE:
            raise ValueError("state_bench_router_surface_mismatch")
        if self.artifact.test_registry_sha256 != self.registry.test_registry_sha256:
            raise ValueError("state_bench_router_test_registry_mismatch")
        if (
            self.artifact.cheap_model != CHEAP_MODEL
            or self.artifact.strong_model != STRONG_MODEL
        ):
            raise ValueError("state_bench_router_model_plan_not_frozen")
        if (
            self.artifact.cheap_reasoning_effort != REASONING_EFFORT
            or self.artifact.strong_reasoning_effort != REASONING_EFFORT
        ):
            raise ValueError("state_bench_router_reasoning_effort_not_frozen")

    @property
    def artifact(self):
        return self.policy.binding

    def bind_public_episode(
        self,
        *,
        task: StateBenchOfficialTaskRef,
        public_initial_user_message: str,
    ) -> FullEpisodeBinding:
        self._require_official_task(task)
        return build_state_bench_full_episode_binding(
            domain=task.domain,
            original_request=_required_text(
                public_initial_user_message,
                "state_bench_public_initial_user_message",
            ),
            initial_environment_sha256=(task.initial_environment_sha256),
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
        return run_routellm_episode(
            artifact=self.artifact,
            episode=episode,
            predict_strong_win_rate=self.policy.predict_routellm,
            threshold=self.policy.routellm_threshold,
            run_episode=run_episode,
            score_selected=score_selected,
        )

    def run_frugalgpt(
        self,
        *,
        episode: FullEpisodeBinding,
        run_episode: FullEpisodeRunner,
        score_selected: ScoreSelected,
    ) -> StaticRoutingResult:
        self._require_method(FRUGALGPT)
        self._require_episode(episode)
        return run_frugalgpt_episode(
            artifact=self.artifact,
            episode=episode,
            predict_cheap_quality=self.policy.predict_frugalgpt,
            accept_threshold=self.policy.frugalgpt_accept_threshold,
            run_episode=run_episode,
            score_selected=score_selected,
        )

    def run_automix(
        self,
        *,
        episode: FullEpisodeBinding,
        run_episode: FullEpisodeRunner,
        score_selected: ScoreSelected,
    ) -> StaticRoutingResult:
        self._require_method(AUTOMIX)
        self._require_episode(episode)
        return run_automix_episode(
            artifact=self.artifact,
            episode=episode,
            verify_cheap=self.policy.verify_automix,
            pomdp_retry=self.policy.automix_should_retry,
            run_episode=run_episode,
            score_selected=score_selected,
        )

    def run_history_router(
        self,
        *,
        task: StateBenchOfficialTaskRef,
        public_initial_user_message: str,
        turn_cap: int,
        execute_turn: ExecuteHistoryTurn,
        score_terminal: ScoreSelected,
    ) -> HistoryRoutingResult:
        self._require_official_task(task)
        if self.artifact.method not in STATE_BENCH_TURN_LEVEL_METHODS:
            raise ValueError("state_bench_history_router_method_invalid")
        initial_message = _required_text(
            public_initial_user_message,
            "state_bench_public_initial_user_message",
        )
        return run_history_routed_episode(
            artifact=self.artifact,
            original_request=initial_message,
            initial_public_history=({"role": "user", "content": initial_message},),
            turn_cap=turn_cap,
            decide_next=self.policy.history_decider(),
            execute_turn=execute_turn,
            score_terminal=score_terminal,
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "method": self.artifact.method,
            "surface": STATE_BENCH_SURFACE,
            "execution_level": router_execution_level(self.artifact.method),
            "test_registry_sha256": self.registry.test_registry_sha256,
            "official_test_count": self.registry.denominator,
            "cheap_model": self.artifact.cheap_model,
            "cheap_reasoning_effort": (self.artifact.cheap_reasoning_effort),
            "strong_model": self.artifact.strong_model,
            "strong_reasoning_effort": (self.artifact.strong_reasoning_effort),
            "sealed_policy": True,
            "router_inputs_exclude_native_task_locator": True,
            "router_inputs_exclude_initial_environment": True,
        }

    def _require_method(self, method: ExternalMethod) -> None:
        if self.artifact.method != method:
            raise ValueError("state_bench_task_router_method_mismatch")

    def _require_episode(self, episode: FullEpisodeBinding) -> None:
        if not isinstance(episode, FullEpisodeBinding):
            raise TypeError("state_bench_full_episode_binding_invalid")
        if episode.surface != STATE_BENCH_SURFACE:
            raise ValueError("state_bench_episode_surface_mismatch")
        if episode.official_denominator != self.registry.denominator:
            raise ValueError("state_bench_episode_denominator_mismatch")

    def _require_official_task(
        self,
        task: StateBenchOfficialTaskRef,
    ) -> None:
        if not self.registry.contains(task):
            raise ValueError("state_bench_task_not_in_official_test_registry")


def router_execution_level(method: ExternalMethod) -> str:
    if method in STATE_BENCH_TASK_LEVEL_METHODS:
        return "task"
    if method in STATE_BENCH_TURN_LEVEL_METHODS:
        return "turn"
    raise ValueError("state_bench_router_method_invalid")


def _load_task_ref(
    repo: Path,
    *,
    domain: str,
    raw_locator: Any,
) -> tuple[StateBenchOfficialTaskRef, Path, Path]:
    locator = _required_text(
        str(raw_locator),
        "state_bench_source_task_locator",
    )
    if Path(locator).name != locator:
        raise ValueError("state_bench_source_task_locator_not_atomic")
    domain_dir = repo / "state_bench" / "domains" / domain
    task_path = (domain_dir / "tasks" / f"{locator}.json").resolve()
    task = _read_json(task_path)
    if not isinstance(task, Mapping):
        raise TypeError("state_bench_task_payload_not_mapping")
    if str(task.get("task_id")) != locator:
        raise ValueError("state_bench_source_task_locator_mismatch")
    raw_environment_path = task.get("task_env_path")
    if not isinstance(raw_environment_path, str) or not raw_environment_path.strip():
        raise ValueError("state_bench_task_environment_path_invalid")
    environment_path = (repo / raw_environment_path).resolve()
    if not environment_path.is_relative_to(repo):
        raise ValueError("state_bench_task_environment_outside_repo")
    environment = _read_json(environment_path)
    initial_environment_sha256 = _digest(environment)
    payload_sha256 = _digest(
        {
            "task": task,
            "initial_environment": environment,
        }
    )
    example_sha256 = _digest(
        {
            "surface": STATE_BENCH_SURFACE,
            "stratum": domain,
            "source_identity": locator,
            "payload_sha256": payload_sha256,
        }
    )
    return (
        StateBenchOfficialTaskRef(
            domain=domain,
            native_task_locator=locator,
            native_task_path=task_path,
            initial_environment_path=environment_path,
            payload_sha256=payload_sha256,
            initial_environment_sha256=initial_environment_sha256,
            example_sha256=example_sha256,
        ),
        task_path,
        environment_path,
    )


def _required_domain(value: Any) -> str:
    domain = _required_text(value, "state_bench_domain")
    if domain not in STATE_BENCH_TEST_DOMAINS or domain not in STATE_BENCH_DOMAINS:
        raise ValueError("state_bench_domain_unsupported")
    return domain


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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

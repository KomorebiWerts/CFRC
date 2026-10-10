"""Protected tau2 bindings for external task- and turn-level routers.

This module deliberately stops at tau2's public harness boundary.  Native
episode execution, user simulation, and official scoring are injected by the
caller; no CFRC contract, graph, selector, or scorer implementation is visible
here.
"""

from __future__ import annotations

import hashlib
import json
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
    TAU2_AIRLINE_FULL_DENOMINATOR,
    TAU2_AIRLINE_SURFACE,
    TAU2_RETAIL_FULL_DENOMINATOR,
    TAU2_RETAIL_SURFACE,
    build_tau2_full_episode_binding,
)

TAU2_TEST_DOMAINS = ("retail", "airline")
TAU2_TASK_LEVEL_METHODS = frozenset({ROUTELLM, FRUGALGPT, AUTOMIX})
TAU2_TURN_LEVEL_METHODS = frozenset({MTROUTER, OPENSQUILLA_G0})
TAU2_OFFICIAL_TEST_COUNTS = {
    "retail": TAU2_RETAIL_FULL_DENOMINATOR,
    "airline": TAU2_AIRLINE_FULL_DENOMINATOR,
}


@dataclass(frozen=True)
class Tau2OfficialTaskRef:
    """Harness-only locator for one source-bound official test task.

    ``native_task_locator`` is needed by tau2 to open the task, but is never
    copied into a router view or a public report.
    """

    native_task_locator: str = field(repr=False)
    payload_sha256: str
    example_sha256: str

    def __post_init__(self) -> None:
        locator = _required_text(
            self.native_task_locator,
            "tau2_native_task_locator",
        )
        object.__setattr__(self, "native_task_locator", locator)
        for field_name in ("payload_sha256", "example_sha256"):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class Tau2OfficialTestRegistry:
    domain: str
    surface: str
    source_binding_sha256: str
    test_registry_sha256: str
    tasks: tuple[Tau2OfficialTaskRef, ...] = field(repr=False)

    def __post_init__(self) -> None:
        domain = _required_domain(self.domain)
        expected_surface = _surface_for_domain(domain)
        if self.surface != expected_surface:
            raise ValueError("tau2_registry_surface_mismatch")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "surface", expected_surface)
        for field_name in (
            "source_binding_sha256",
            "test_registry_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        tasks = tuple(self.tasks)
        expected_count = TAU2_OFFICIAL_TEST_COUNTS[domain]
        if len(tasks) != expected_count:
            raise ValueError("tau2_official_test_denominator_mismatch")
        if any(not isinstance(task, Tau2OfficialTaskRef) for task in tasks):
            raise TypeError("tau2_registry_task_ref_invalid")
        if len({task.native_task_locator for task in tasks}) != len(tasks):
            raise ValueError("tau2_registry_task_locator_duplicate")
        if len({task.example_sha256 for task in tasks}) != len(tasks):
            raise ValueError("tau2_registry_example_sha256_duplicate")
        object.__setattr__(self, "tasks", tasks)

    @property
    def denominator(self) -> int:
        return TAU2_OFFICIAL_TEST_COUNTS[self.domain]

    def contains(self, task: Tau2OfficialTaskRef) -> bool:
        if not isinstance(task, Tau2OfficialTaskRef):
            return False
        return any(
            candidate.example_sha256 == task.example_sha256
            and candidate.payload_sha256 == task.payload_sha256
            and candidate.native_task_locator == task.native_task_locator
            for candidate in self.tasks
        )

    def public_record(self) -> dict[str, Any]:
        """Return the source binding without raw task locators."""

        return {
            "domain": self.domain,
            "surface": self.surface,
            "official_test_count": self.denominator,
            "source_binding_sha256": self.source_binding_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "example_sha256s": [
                task.example_sha256 for task in self.tasks
            ],
            "raw_task_locators_present": False,
        }


def load_tau2_official_test_registry(
    tau2_repo: Path,
    *,
    domain: str,
) -> Tau2OfficialTestRegistry:
    """Load and hash tau2's official test split without exposing task content."""

    resolved_domain = _required_domain(domain)
    domain_dir = (
        Path(tau2_repo)
        / "data"
        / "tau2"
        / "domains"
        / resolved_domain
    )
    split_path = domain_dir / "split_tasks.json"
    tasks_path = domain_dir / "tasks.json"
    split_payload = _read_json(split_path)
    tasks_payload = _read_json(tasks_path)
    if not isinstance(split_payload, Mapping):
        raise TypeError("tau2_split_payload_not_mapping")
    if not isinstance(tasks_payload, list):
        raise TypeError("tau2_tasks_payload_not_list")
    tasks_by_locator: dict[str, Mapping[str, Any]] = {}
    for raw in tasks_payload:
        if not isinstance(raw, Mapping):
            raise TypeError("tau2_task_payload_not_mapping")
        locator = _required_text(raw.get("id"), "tau2_source_task_id")
        if locator in tasks_by_locator:
            raise ValueError("tau2_source_task_id_duplicate")
        tasks_by_locator[locator] = raw
    raw_test = split_payload.get("test")
    if not isinstance(raw_test, list):
        raise TypeError("tau2_test_split_not_list")
    expected_count = TAU2_OFFICIAL_TEST_COUNTS[resolved_domain]
    if len(raw_test) != expected_count:
        raise ValueError("tau2_official_test_denominator_mismatch")
    source_binding_sha256 = _source_binding_sha256(
        (split_path, tasks_path)
    )
    surface = _surface_for_domain(resolved_domain)
    task_refs: list[Tau2OfficialTaskRef] = []
    for raw_locator in raw_test:
        locator = _required_text(raw_locator, "tau2_test_task_locator")
        task_payload = tasks_by_locator.get(locator)
        if task_payload is None:
            raise ValueError("tau2_test_task_missing_from_tasks")
        payload_sha256 = _digest(task_payload)
        task_refs.append(
            Tau2OfficialTaskRef(
                native_task_locator=locator,
                payload_sha256=payload_sha256,
                example_sha256=_digest(
                    {
                        "surface": surface,
                        "stratum": resolved_domain,
                        "source_identity": locator,
                        "payload_sha256": payload_sha256,
                    }
                ),
            )
        )
    example_sha256s = tuple(task.example_sha256 for task in task_refs)
    test_registry_sha256 = _digest(
        {
            "role": "official_test",
            "surface": surface,
            "source_binding_sha256": source_binding_sha256,
            "example_sha256s": example_sha256s,
        }
    )
    return Tau2OfficialTestRegistry(
        domain=resolved_domain,
        surface=surface,
        source_binding_sha256=source_binding_sha256,
        test_registry_sha256=test_registry_sha256,
        tasks=tuple(task_refs),
    )


@dataclass(frozen=True)
class Tau2RouterDriver:
    """Bind one frozen external router artifact to one official tau2 split."""

    policy: SealedRouterPolicy
    registry: Tau2OfficialTestRegistry

    def __post_init__(self) -> None:
        require_sealed_router_policy(self.policy)
        if not isinstance(self.registry, Tau2OfficialTestRegistry):
            raise TypeError("tau2_router_registry_invalid")
        if self.artifact.surface != self.registry.surface:
            raise ValueError("tau2_router_surface_mismatch")
        if (
            self.artifact.test_registry_sha256
            != self.registry.test_registry_sha256
        ):
            raise ValueError("tau2_router_test_registry_mismatch")
        if (
            self.artifact.cheap_model != CHEAP_MODEL
            or self.artifact.strong_model != STRONG_MODEL
        ):
            raise ValueError("tau2_router_model_plan_not_frozen")
        if (
            self.artifact.cheap_reasoning_effort != REASONING_EFFORT
            or self.artifact.strong_reasoning_effort != REASONING_EFFORT
        ):
            raise ValueError("tau2_router_reasoning_effort_not_frozen")

    @property
    def artifact(self):
        return self.policy.binding

    def bind_public_episode(
        self,
        *,
        task: Tau2OfficialTaskRef,
        public_initial_user_message: str,
        initial_environment_sha256: str,
    ) -> FullEpisodeBinding:
        """Create the router-visible episode only after tau2 emits user text."""

        self._require_official_task(task)
        return build_tau2_full_episode_binding(
            domain=self.registry.domain,
            original_request=_required_text(
                public_initial_user_message,
                "tau2_public_initial_user_message",
            ),
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
        task: Tau2OfficialTaskRef,
        public_initial_user_message: str,
        turn_cap: int,
        execute_turn: ExecuteHistoryTurn,
        score_terminal: ScoreSelected,
    ) -> HistoryRoutingResult:
        """Route each native agent turn using public history only."""

        self._require_official_task(task)
        if self.artifact.method not in TAU2_TURN_LEVEL_METHODS:
            raise ValueError("tau2_history_router_method_invalid")
        initial_message = _required_text(
            public_initial_user_message,
            "tau2_public_initial_user_message",
        )
        return run_history_routed_episode(
            artifact=self.artifact,
            original_request=initial_message,
            initial_public_history=(
                {"role": "user", "content": initial_message},
            ),
            turn_cap=turn_cap,
            decide_next=self.policy.history_decider(),
            execute_turn=execute_turn,
            score_terminal=score_terminal,
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "method": self.artifact.method,
            "surface": self.registry.surface,
            "domain": self.registry.domain,
            "execution_level": (
                "task"
                if self.artifact.method in TAU2_TASK_LEVEL_METHODS
                else "turn"
            ),
            "test_registry_sha256": self.registry.test_registry_sha256,
            "official_test_count": self.registry.denominator,
            "cheap_model": self.artifact.cheap_model,
            "cheap_reasoning_effort": (
                self.artifact.cheap_reasoning_effort
            ),
            "strong_model": self.artifact.strong_model,
            "strong_reasoning_effort": (
                self.artifact.strong_reasoning_effort
            ),
            "sealed_policy": True,
            "router_inputs_exclude_native_task_locator": True,
        }

    def _require_method(self, method: ExternalMethod) -> None:
        if self.artifact.method != method:
            raise ValueError("tau2_task_router_method_mismatch")

    def _require_episode(self, episode: FullEpisodeBinding) -> None:
        if not isinstance(episode, FullEpisodeBinding):
            raise TypeError("tau2_full_episode_binding_invalid")
        if episode.surface != self.registry.surface:
            raise ValueError("tau2_episode_surface_mismatch")
        if episode.official_denominator != self.registry.denominator:
            raise ValueError("tau2_episode_denominator_mismatch")

    def _require_official_task(self, task: Tau2OfficialTaskRef) -> None:
        if not self.registry.contains(task):
            raise ValueError("tau2_task_not_in_official_test_registry")


def router_execution_level(method: ExternalMethod) -> str:
    if method in TAU2_TASK_LEVEL_METHODS:
        return "task"
    if method in TAU2_TURN_LEVEL_METHODS:
        return "turn"
    raise ValueError("tau2_router_method_invalid")


def _required_domain(domain: Any) -> str:
    value = _required_text(domain, "tau2_domain")
    if value not in TAU2_TEST_DOMAINS:
        raise ValueError("tau2_domain_unsupported")
    return value


def _surface_for_domain(domain: str) -> str:
    return (
        TAU2_RETAIL_SURFACE
        if domain == "retail"
        else TAU2_AIRLINE_SURFACE
    )


def _source_binding_sha256(paths: Sequence[Path]) -> str:
    return _digest(
        [
            {
                "name": path.name,
                "parent": path.parent.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ]
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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

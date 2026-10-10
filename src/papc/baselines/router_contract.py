"""Method-neutral contracts for protected external routing baselines."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ROUTELLM = "routellm_task"
FRUGALGPT = "frugalgpt_episode_cascade"
MTROUTER = "mtrouter_turn"
OPENSQUILLA_G0 = "opensquilla_g0_calibrated"
AUTOMIX = "automix_pomdp"

ExternalMethod = Literal[
    "routellm_task",
    "frugalgpt_episode_cascade",
    "mtrouter_turn",
    "opensquilla_g0_calibrated",
    "automix_pomdp",
]
ModelRole = Literal["cheap", "strong"]
SplitRole = Literal["router_fit", "router_dev", "official_test"]

CHEAP_MODEL = "gpt-5.4-mini"
STRONG_MODEL = "gpt-5.5"
REASONING_EFFORT = "high"
ROUTELLM_EMBEDDING_MODEL = "text-embedding-3-small"
ROUTELLM_LOCAL_MODEL = "routellm-bert-local"
AUTOMIX_MAIN_VERIFIER_SAMPLES = 32
AUTOMIX_POMDP_NUM_BINS = 8
AUTOMIX_PUBLISHED_VERIFIER_SAMPLE_COUNTS = (4, 8, 12, 16, 24, 32)

STATE_BENCH = "state_bench"
AGENT_DIFF = "agent_diff"
TAU2_RETAIL = "tau2_retail"
TAU2_AIRLINE = "tau2_airline"
TOOL_SANDBOX = "toolsandbox"

ALL_SURFACES = (
    STATE_BENCH,
    TAU2_RETAIL,
    TAU2_AIRLINE,
    TOOL_SANDBOX,
    AGENT_DIFF,
)

UPSTREAM_REVISIONS: dict[ExternalMethod, str] = {
    ROUTELLM: "0b64fdafe049e596a3f5657c219329f24af24198",
    FRUGALGPT: "2b23e6acbd4b39f71912816640dcc59e45aa4b5e",
    MTROUTER: "ee9e4573a128c3e54aa8c83800318bfb783c4dfe",
    OPENSQUILLA_G0: "7e3965e7cc64316be87dc0d535eace39d22a2c93",
    AUTOMIX: "531af3ee3c4e4f4a53da1ffbc91dec4d6bb73781",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_HISTORY_KEYS = frozenset(
    {
        "id",
        "type",
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "call_id",
        "arguments",
        "output",
        "status",
    }
)
_FORBIDDEN_ROUTER_FIELD_PARTS = frozenset(
    {
        "answer_sketch",
        "completed_action",
        "contract",
        "direct_outcome",
        "fixed_kb_query",
        "fixed_object_id",
        "fixed_tool_args",
        "frontier",
        "gold",
        "hidden",
        "official_outcome",
        "official_reward",
        "official_score",
        "reached_state",
        "state_dump",
        "task_id",
    }
)


@dataclass(frozen=True)
class ExternalMethodContract:
    method: ExternalMethod
    paper_variant: str
    decision_stage: str
    visible_inputs: tuple[str, ...]
    training_substrate: str
    execution_semantics: str
    supported_surfaces: tuple[str, ...]


METHOD_CONTRACTS: dict[ExternalMethod, ExternalMethodContract] = {
    ROUTELLM: ExternalMethodContract(
        method=ROUTELLM,
        paper_variant="RouteLLM-MF task-level two-model adaptation",
        decision_stage="before_episode",
        visible_inputs=("original_request",),
        training_substrate="paired_full_episode_preference",
        execution_semantics="run_selected_model_from_pristine_initial_state",
        supported_surfaces=(
            STATE_BENCH,
            AGENT_DIFF,
            TAU2_RETAIL,
            TAU2_AIRLINE,
        ),
    ),
    FRUGALGPT: ExternalMethodContract(
        method=FRUGALGPT,
        paper_variant="FrugalGPT cheap-first two-model episode adaptation",
        decision_stage="after_completed_cheap_episode",
        visible_inputs=("original_request", "cheap_generated_answer"),
        training_substrate="paired_full_episode_output_quality_cost",
        execution_semantics=(
            "accept_cheap_or_restart_strong_from_pristine_initial_state"
        ),
        supported_surfaces=(
            STATE_BENCH,
            TAU2_RETAIL,
            TAU2_AIRLINE,
            AGENT_DIFF,
        ),
    ),
    MTROUTER: ExternalMethodContract(
        method=MTROUTER,
        paper_variant="MTRouter history-conditioned turn routing",
        decision_stage="before_each_native_agent_turn",
        visible_inputs=("original_request", "public_native_history"),
        training_substrate="uniform_turn_exploration",
        execution_semantics="continue_current_episode_with_selected_model",
        supported_surfaces=(
            STATE_BENCH,
            TAU2_RETAIL,
            TAU2_AIRLINE,
            AGENT_DIFF,
        ),
    ),
    OPENSQUILLA_G0: ExternalMethodContract(
        method=OPENSQUILLA_G0,
        paper_variant=(
            "OpenSquilla released V4.2 initialization with fit-only "
            "in-domain LightGBM continuation and dev-calibrated "
            "two-tier projection"
        ),
        decision_stage="before_each_native_agent_turn",
        visible_inputs=(
            "original_request",
            "public_native_history",
            "public_harness_features",
            "prior_public_route_decisions",
            "previous_assistant_usage",
        ),
        training_substrate=(
            "released_v4_2_init_plus_fit_only_in_domain_paired_log_"
            "lightgbm_continuation_plus_dev_two_tier_projection"
        ),
        execution_semantics="continue_current_episode_with_selected_model",
        supported_surfaces=(
            STATE_BENCH,
            TAU2_RETAIL,
            TAU2_AIRLINE,
            AGENT_DIFF,
        ),
    ),
    AUTOMIX: ExternalMethodContract(
        method=AUTOMIX,
        paper_variant="AutoMix-POMDP agent-episode adaptation",
        decision_stage="after_completed_cheap_episode_and_self_verification",
        visible_inputs=(
            "original_request",
            "cheap_public_episode",
            "cheap_self_verification_confidence",
        ),
        training_substrate=(
            "paired_full_episode_quality_cost_plus_cheap_self_verification"
        ),
        execution_semantics=(
            "accept_cheap_or_restart_strong_from_pristine_initial_state"
        ),
        supported_surfaces=(
            STATE_BENCH,
            TAU2_RETAIL,
            TAU2_AIRLINE,
            AGENT_DIFF,
        ),
    ),
}

INELIGIBLE_SURFACE_REASONS: dict[
    ExternalMethod,
    dict[str, str],
] = {
    method: {
        surface: (
            "no_disjoint_official_train_test_for_in_domain_adaptation"
            if surface == TOOL_SANDBOX
            else "outside_frozen_published_regime"
        )
        for surface in ALL_SURFACES
        if surface not in contract.supported_surfaces
    }
    for method, contract in METHOD_CONTRACTS.items()
}


@dataclass(frozen=True)
class SplitManifest:
    role: SplitRole
    registry_sha256: str
    example_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"router_fit", "router_dev", "official_test"}:
            raise ValueError("router_split_role_invalid")
        object.__setattr__(
            self,
            "registry_sha256",
            _required_sha256(self.registry_sha256, "registry_sha256"),
        )
        examples = tuple(
            _required_sha256(value, "example_sha256")
            for value in self.example_sha256s
        )
        if not examples:
            raise ValueError("router_split_examples_empty")
        if len(set(examples)) != len(examples):
            raise ValueError("router_split_examples_duplicate")
        object.__setattr__(self, "example_sha256s", examples)


def assert_disjoint_manifests(
    fit: SplitManifest,
    dev: SplitManifest,
    test: SplitManifest,
) -> None:
    if (fit.role, dev.role, test.role) != (
        "router_fit",
        "router_dev",
        "official_test",
    ):
        raise ValueError("router_split_manifest_order_invalid")
    groups = (
        set(fit.example_sha256s),
        set(dev.example_sha256s),
        set(test.example_sha256s),
    )
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("router_split_overlap")
    if len({fit.registry_sha256, dev.registry_sha256, test.registry_sha256}) != 3:
        raise ValueError("router_split_registry_digest_reused")


@dataclass(frozen=True)
class RouterArtifactBinding:
    method: ExternalMethod
    surface: str
    upstream_revision: str
    artifact_sha256: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    calibration_sha256: str
    cheap_model: str = CHEAP_MODEL
    cheap_reasoning_effort: str = REASONING_EFFORT
    strong_model: str = STRONG_MODEL
    strong_reasoning_effort: str = REASONING_EFFORT

    def __post_init__(self) -> None:
        contract = METHOD_CONTRACTS.get(self.method)
        if contract is None:
            raise ValueError("router_method_invalid")
        surface = _required_text(self.surface, "surface")
        if surface not in contract.supported_surfaces:
            raise ValueError("router_method_surface_not_supported")
        object.__setattr__(self, "surface", surface)
        upstream_revision = _required_text(
            self.upstream_revision,
            "upstream_revision",
        )
        if upstream_revision != UPSTREAM_REVISIONS[self.method]:
            raise ValueError("router_upstream_revision_not_frozen")
        object.__setattr__(self, "upstream_revision", upstream_revision)
        for field_name in (
            "artifact_sha256",
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "calibration_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if self.cheap_model != CHEAP_MODEL:
            raise ValueError("router_cheap_model_not_frozen")
        if self.strong_model != STRONG_MODEL:
            raise ValueError("router_strong_model_not_frozen")
        if (
            self.cheap_reasoning_effort != REASONING_EFFORT
            or self.strong_reasoning_effort != REASONING_EFFORT
        ):
            raise ValueError("router_reasoning_effort_not_frozen")
        if len(
            {
                self.fit_registry_sha256,
                self.dev_registry_sha256,
                self.test_registry_sha256,
            }
        ) != 3:
            raise ValueError("router_binding_split_digest_reused")

    def model_for(self, role: ModelRole) -> str:
        if role == "cheap":
            return self.cheap_model
        if role == "strong":
            return self.strong_model
        raise ValueError("router_model_role_invalid")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullEpisodeBinding:
    surface: str
    surface_variant: str
    original_request: str = field(repr=False)
    initial_environment_sha256: str
    official_scorer: str
    official_denominator: int
    _public_input_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface",
            _required_text(self.surface, "surface"),
        )
        object.__setattr__(
            self,
            "surface_variant",
            _required_text(self.surface_variant, "surface_variant"),
        )
        request = _required_text(self.original_request, "original_request")
        object.__setattr__(self, "original_request", request)
        object.__setattr__(
            self,
            "initial_environment_sha256",
            _required_sha256(
                self.initial_environment_sha256,
                "initial_environment_sha256",
            ),
        )
        object.__setattr__(
            self,
            "official_scorer",
            _required_text(self.official_scorer, "official_scorer"),
        )
        if (
            isinstance(self.official_denominator, bool)
            or not isinstance(self.official_denominator, int)
            or self.official_denominator <= 0
        ):
            raise ValueError("official_denominator_invalid")
        object.__setattr__(
            self,
            "_public_input_sha256",
            _digest({"original_request": request}),
        )

    @property
    def public_input_sha256(self) -> str:
        return self._public_input_sha256


@dataclass(frozen=True)
class RequestRouterView:
    original_request: str = field(repr=False)
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        request = _required_text(self.original_request, "original_request")
        object.__setattr__(self, "original_request", request)
        object.__setattr__(self, "_sha256", _digest({"request": request}))

    @property
    def sha256(self) -> str:
        return self._sha256


@dataclass(frozen=True)
class CheapCompletionRouterView:
    original_request: str = field(repr=False)
    cheap_generated_answer: str = field(repr=False)
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        request = _required_text(self.original_request, "original_request")
        answer = _required_text(
            self.cheap_generated_answer,
            "cheap_generated_answer",
        )
        object.__setattr__(self, "original_request", request)
        object.__setattr__(self, "cheap_generated_answer", answer)
        object.__setattr__(
            self,
            "_sha256",
            _digest({"request": request, "cheap_generated_answer": answer}),
        )

    @property
    def sha256(self) -> str:
        return self._sha256


@dataclass(frozen=True)
class AutoMixRouterView:
    original_request: str = field(repr=False)
    cheap_public_episode: tuple[dict[str, Any], ...] = field(repr=False)
    cheap_generated_answer: str = field(repr=False)
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        request = _required_text(self.original_request, "original_request")
        public_episode = _validated_public_history(self.cheap_public_episode)
        if not public_episode:
            raise ValueError("automix_public_episode_empty")
        answer = _required_text(
            self.cheap_generated_answer,
            "cheap_generated_answer",
        )
        object.__setattr__(self, "original_request", request)
        object.__setattr__(self, "cheap_public_episode", public_episode)
        object.__setattr__(self, "cheap_generated_answer", answer)
        object.__setattr__(
            self,
            "_sha256",
            _digest(
                {
                    "verification_context": public_episode,
                    "question": request,
                    "generated_answer": answer,
                }
            ),
        )

    @property
    def sha256(self) -> str:
        return self._sha256


@dataclass(frozen=True)
class HistoryRouterView:
    original_request: str = field(repr=False)
    public_history: tuple[dict[str, Any], ...] = field(repr=False)
    turn_index: int
    turn_cap: int
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        request = _required_text(self.original_request, "original_request")
        history = _validated_public_history(self.public_history)
        turn_index = _nonnegative_int(self.turn_index, "turn_index")
        turn_cap = _positive_int(self.turn_cap, "turn_cap")
        if turn_index >= turn_cap:
            raise ValueError("router_turn_index_out_of_range")
        object.__setattr__(self, "original_request", request)
        object.__setattr__(self, "public_history", history)
        object.__setattr__(self, "turn_index", turn_index)
        object.__setattr__(self, "turn_cap", turn_cap)
        object.__setattr__(
            self,
            "_sha256",
            _digest(
                {
                    "request": request,
                    "public_history": history,
                    "turn_index": turn_index,
                    "turn_cap": turn_cap,
                }
            ),
        )

    @property
    def sha256(self) -> str:
        return self._sha256

    def history_for_runtime(self) -> tuple[dict[str, Any], ...]:
        return deepcopy(self.public_history)


@dataclass(frozen=True)
class PublicFeature:
    name: str
    value: float

    def __post_init__(self) -> None:
        name = _required_text(self.name, "public_feature_name")
        _reject_forbidden_name(name)
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("public_feature_value_must_be_numeric")
        value = float(self.value)
        if not _is_finite(value):
            raise ValueError("public_feature_value_not_finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class RankerRouterView:
    history: HistoryRouterView
    public_features: tuple[PublicFeature, ...]
    prior_route_decisions: tuple[ModelRole, ...] = ()
    previous_assistant_usage: tuple[PublicFeature, ...] = ()
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.history, HistoryRouterView):
            raise TypeError("ranker_history_view_invalid")
        features = tuple(self.public_features)
        if len({feature.name for feature in features}) != len(features):
            raise ValueError("public_feature_name_duplicate")
        prior_route_decisions = tuple(self.prior_route_decisions)
        if any(role not in {"cheap", "strong"} for role in prior_route_decisions):
            raise ValueError("prior_route_decision_invalid")
        if len(prior_route_decisions) > self.history.turn_index:
            raise ValueError("prior_route_decisions_exceed_turn_index")
        assistant_usage = tuple(self.previous_assistant_usage)
        if len({feature.name for feature in assistant_usage}) != len(
            assistant_usage
        ):
            raise ValueError("assistant_usage_feature_name_duplicate")
        object.__setattr__(self, "public_features", features)
        object.__setattr__(
            self,
            "prior_route_decisions",
            prior_route_decisions,
        )
        object.__setattr__(
            self,
            "previous_assistant_usage",
            assistant_usage,
        )
        object.__setattr__(
            self,
            "_sha256",
            _digest(
                {
                    "history_sha256": self.history.sha256,
                    "public_features": [
                        {"name": feature.name, "value": feature.value}
                        for feature in features
                    ],
                    "prior_route_decisions": prior_route_decisions,
                    "previous_assistant_usage": [
                        {"name": feature.name, "value": feature.value}
                        for feature in assistant_usage
                    ],
                }
            ),
        )

    @property
    def sha256(self) -> str:
        return self._sha256


def validate_public_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return _validated_public_history(history)


def _validated_public_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("router_public_history_must_be_sequence")
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise TypeError(f"router_public_history_{index}_must_be_mapping")
        item = dict(raw)
        for key in item:
            _reject_forbidden_name(str(key))
        if item.get("tool_calls") is not None:
            _reject_forbidden_tool_calls(item["tool_calls"])
        unknown = sorted(set(item) - _PUBLIC_HISTORY_KEYS)
        if unknown:
            raise ValueError(
                f"router_public_history_{index}_unsupported_keys:"
                + ",".join(unknown)
            )
        _canonical_bytes(item)
        items.append(deepcopy(item))
    return tuple(items)


def _reject_forbidden_tool_calls(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key in value:
            _reject_forbidden_name(str(raw_key))
        if value.get("arguments") is not None:
            _reject_forbidden_payload(value["arguments"])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden_tool_calls(item)


def _reject_forbidden_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            _reject_forbidden_name(key)
            _reject_forbidden_payload(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden_payload(item)


def _reject_forbidden_name(name: str) -> None:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if any(part in normalized for part in _FORBIDDEN_ROUTER_FIELD_PARTS):
        raise ValueError(f"router_forbidden_field:{name}")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name}_invalid")
    return text


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name}_invalid")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    resolved = _nonnegative_int(value, field_name)
    if resolved == 0:
        raise ValueError(f"{field_name}_invalid")
    return resolved


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()

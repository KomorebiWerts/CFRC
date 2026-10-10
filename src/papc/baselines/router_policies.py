"""Frozen decision rules for the five external routing baselines."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from papc.baselines.router_contract import (
    AUTOMIX,
    AUTOMIX_MAIN_VERIFIER_SAMPLES,
    AUTOMIX_POMDP_NUM_BINS,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    ROUTELLM,
    AutoMixRouterView,
    CheapCompletionRouterView,
    ExternalMethod,
    HistoryRouterView,
    ModelRole,
    RankerRouterView,
    RequestRouterView,
    RouterArtifactBinding,
)

DecisionStage = Literal[
    "before_episode",
    "after_completed_cheap_episode",
    "after_completed_cheap_episode_and_self_verification",
    "before_native_agent_turn",
]


@dataclass(frozen=True)
class RouterDecision:
    method: ExternalMethod
    surface: str
    selected_role: ModelRole
    selected_model: str
    decision_stage: DecisionStage
    execution_semantics: str
    public_input_sha256: str
    artifact_sha256: str
    scores: tuple[tuple[str, float], ...]
    threshold: float | None
    rationale: str

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


def decide_routellm(
    *,
    binding: RouterArtifactBinding,
    view: RequestRouterView,
    predicted_strong_win_rate: float | None,
    threshold: float,
) -> RouterDecision:
    _require_method(binding, ROUTELLM)
    if not isinstance(view, RequestRouterView):
        raise TypeError("routellm_request_view_required")
    boundary = _unit_interval(threshold, "routellm_threshold")
    if predicted_strong_win_rate is None:
        role: ModelRole = "strong"
        score = 1.0
        rationale = "router_failure_strong_fallback"
    else:
        score = _unit_interval(
            predicted_strong_win_rate,
            "routellm_strong_win_rate",
        )
        role = "strong" if score >= boundary else "cheap"
        rationale = "strong_win_rate_threshold"
    return _decision(
        binding=binding,
        role=role,
        stage="before_episode",
        execution_semantics="run_selected_model_from_pristine_initial_state",
        public_input_sha256=view.sha256,
        scores=(("predicted_strong_win_rate", score),),
        threshold=boundary,
        rationale=rationale,
    )


def decide_frugalgpt(
    *,
    binding: RouterArtifactBinding,
    view: CheapCompletionRouterView,
    predicted_cheap_quality: float | None,
    accept_threshold: float,
) -> RouterDecision:
    _require_method(binding, FRUGALGPT)
    if not isinstance(view, CheapCompletionRouterView):
        raise TypeError("frugalgpt_cheap_completion_view_required")
    boundary = _unit_interval(
        accept_threshold,
        "frugalgpt_accept_threshold",
    )
    if predicted_cheap_quality is None:
        role: ModelRole = "strong"
        score = 0.0
        rationale = "scorer_failure_strong_fallback"
    else:
        score = _unit_interval(
            predicted_cheap_quality,
            "frugalgpt_predicted_quality",
        )
        role = "cheap" if score > boundary else "strong"
        rationale = "cheap_quality_accept_threshold"
    semantics = (
        "accept_completed_cheap_episode"
        if role == "cheap"
        else "restart_strong_from_pristine_initial_state"
    )
    return _decision(
        binding=binding,
        role=role,
        stage="after_completed_cheap_episode",
        execution_semantics=semantics,
        public_input_sha256=view.sha256,
        scores=(("predicted_cheap_quality", score),),
        threshold=boundary,
        rationale=rationale,
    )


def decide_automix(
    *,
    binding: RouterArtifactBinding,
    view: AutoMixRouterView,
    verification_confidence: float | None,
    verifier_samples: int,
    pomdp_num_bins: int,
    pomdp_retry: bool | None,
) -> RouterDecision:
    _require_method(binding, AUTOMIX)
    if not isinstance(view, AutoMixRouterView):
        raise TypeError("automix_router_view_required")
    if verifier_samples != AUTOMIX_MAIN_VERIFIER_SAMPLES:
        raise ValueError("automix_verifier_samples_not_frozen")
    if pomdp_num_bins != AUTOMIX_POMDP_NUM_BINS:
        raise ValueError("automix_pomdp_num_bins_not_frozen")
    if verification_confidence is None or pomdp_retry is None:
        confidence = 0.0
        role: ModelRole = "strong"
        rationale = "verification_or_pomdp_failure_strong_fallback"
    else:
        confidence = _unit_interval(
            verification_confidence,
            "automix_verification_confidence",
        )
        if not isinstance(pomdp_retry, bool):
            raise TypeError("automix_pomdp_retry_must_be_bool")
        role = "strong" if pomdp_retry else "cheap"
        rationale = "official_pomdp_retry_policy"
    semantics = (
        "accept_completed_cheap_episode"
        if role == "cheap"
        else "restart_strong_from_pristine_initial_state"
    )
    return _decision(
        binding=binding,
        role=role,
        stage="after_completed_cheap_episode_and_self_verification",
        execution_semantics=semantics,
        public_input_sha256=view.sha256,
        scores=(
            ("cheap_self_verification_confidence", confidence),
            ("verifier_samples", float(verifier_samples)),
            ("pomdp_num_bins", float(pomdp_num_bins)),
        ),
        threshold=None,
        rationale=rationale,
    )


def decide_mtrouter(
    *,
    binding: RouterArtifactBinding,
    view: HistoryRouterView,
    cheap_predicted_terminal_value: float,
    strong_predicted_terminal_value: float,
) -> RouterDecision:
    _require_method(binding, MTROUTER)
    if not isinstance(view, HistoryRouterView):
        raise TypeError("mtrouter_history_view_required")
    cheap_value = _finite(
        cheap_predicted_terminal_value,
        "mtrouter_cheap_value",
    )
    strong_value = _finite(
        strong_predicted_terminal_value,
        "mtrouter_strong_value",
    )
    role: ModelRole = "cheap" if cheap_value >= strong_value else "strong"
    return _decision(
        binding=binding,
        role=role,
        stage="before_native_agent_turn",
        execution_semantics="continue_current_episode_with_selected_model",
        public_input_sha256=view.sha256,
        scores=(
            ("cheap_predicted_terminal_value", cheap_value),
            ("strong_predicted_terminal_value", strong_value),
        ),
        threshold=None,
        rationale=(
            "greedy_argmax_tie_cheap"
            if cheap_value == strong_value
            else "greedy_argmax"
        ),
    )


def decide_mtrouter_all_strong(
    *,
    binding: RouterArtifactBinding,
    view: HistoryRouterView,
) -> RouterDecision:
    """Execute the dev-selected all-strong candidate without a checkpoint."""
    _require_method(binding, MTROUTER)
    if not isinstance(view, HistoryRouterView):
        raise TypeError("mtrouter_history_view_required")
    return _decision(
        binding=binding,
        role="strong",
        stage="before_native_agent_turn",
        execution_semantics="continue_current_episode_with_selected_model",
        public_input_sha256=view.sha256,
        scores=(("dev_selected_all_strong", 1.0),),
        threshold=None,
        rationale="dev_selected_all_strong_native_shortcut",
    )


def decide_opensquilla_g0(
    *,
    binding: RouterArtifactBinding,
    view: RankerRouterView,
    route_class: int,
    confidence: float,
    strong_class_floor: int,
) -> RouterDecision:
    _require_method(binding, OPENSQUILLA_G0)
    if not isinstance(view, RankerRouterView):
        raise TypeError("opensquilla_ranker_view_required")
    if isinstance(route_class, bool) or route_class not in {0, 1, 2, 3}:
        raise ValueError("opensquilla_route_class_invalid")
    if (
        isinstance(strong_class_floor, bool)
        or strong_class_floor not in {1, 2, 3, 4}
    ):
        raise ValueError("opensquilla_strong_class_floor_invalid")
    resolved_confidence = _unit_interval(
        confidence,
        "opensquilla_confidence",
    )
    role: ModelRole = (
        "strong" if route_class >= strong_class_floor else "cheap"
    )
    return _decision(
        binding=binding,
        role=role,
        stage="before_native_agent_turn",
        execution_semantics="continue_current_episode_with_selected_model",
        public_input_sha256=view.sha256,
        scores=(
            ("route_class", float(route_class)),
            ("confidence", resolved_confidence),
            ("strong_class_floor", float(strong_class_floor)),
        ),
        threshold=None,
        rationale="dev_calibrated_monotone_tier_mapping",
    )


def _decision(
    *,
    binding: RouterArtifactBinding,
    role: ModelRole,
    stage: DecisionStage,
    execution_semantics: str,
    public_input_sha256: str,
    scores: tuple[tuple[str, float], ...],
    threshold: float | None,
    rationale: str,
) -> RouterDecision:
    return RouterDecision(
        method=binding.method,
        surface=binding.surface,
        selected_role=role,
        selected_model=binding.model_for(role),
        decision_stage=stage,
        execution_semantics=execution_semantics,
        public_input_sha256=public_input_sha256,
        artifact_sha256=binding.artifact_sha256,
        scores=scores,
        threshold=threshold,
        rationale=rationale,
    )


def _require_method(
    binding: RouterArtifactBinding,
    expected: ExternalMethod,
) -> None:
    if not isinstance(binding, RouterArtifactBinding):
        raise TypeError("router_artifact_binding_required")
    if binding.method != expected:
        raise ValueError("router_artifact_method_mismatch")


def _unit_interval(value: Any, field_name: str) -> float:
    resolved = _finite(value, field_name)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return resolved


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name}_not_finite")
    return resolved

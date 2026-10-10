"""Neutral full-episode and visible-history runtimes for external routers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    AUTOMIX,
    AUTOMIX_MAIN_VERIFIER_SAMPLES,
    AUTOMIX_POMDP_NUM_BINS,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    ROUTELLM_LOCAL_MODEL,
    AutoMixRouterView,
    CheapCompletionRouterView,
    FullEpisodeBinding,
    HistoryRouterView,
    ModelRole,
    RequestRouterView,
    RouterArtifactBinding,
    validate_public_history,
)
from papc.baselines.router_policies import (
    RouterDecision,
    decide_automix,
    decide_frugalgpt,
    decide_routellm,
)


@dataclass(frozen=True)
class ScalarPrediction:
    value: float | None
    receipts: tuple[CallReceipt, ...]

    def __post_init__(self) -> None:
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
        ):
            raise ValueError("router_scalar_prediction_invalid")
        if not self.receipts:
            raise ValueError("router_scalar_prediction_receipts_empty")
        if any(not isinstance(item, CallReceipt) for item in self.receipts):
            raise TypeError("router_scalar_prediction_receipt_invalid")


PredictScalar = Callable[[Any], float | None | ScalarPrediction]
ScoreSelected = Callable[[Any], float]


@dataclass(frozen=True)
class GeneratedEpisode:
    role: ModelRole
    model: str
    initial_environment_sha256: str
    environment_instance_sha256: str
    generated_answer_view: str
    receipts: tuple[CallReceipt, ...]
    opaque_score_handle: Any
    public_episode_history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"cheap", "strong"}:
            raise ValueError("generated_episode_role_invalid")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("generated_episode_model_invalid")
        for field_name in (
            "initial_environment_sha256",
            "environment_instance_sha256",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"generated_episode_{field_name}_invalid")
        if (
            not isinstance(self.generated_answer_view, str)
            or not self.generated_answer_view.strip()
        ):
            raise ValueError("generated_episode_answer_view_invalid")
        if any(not isinstance(item, CallReceipt) for item in self.receipts):
            raise TypeError("generated_episode_receipt_invalid")
        object.__setattr__(
            self,
            "public_episode_history",
            validate_public_history(self.public_episode_history),
        )


@dataclass(frozen=True)
class StaticRoutingResult:
    method: str
    decision: RouterDecision
    selected_role: ModelRole
    selected_model: str
    official_score: float
    episode_count: int
    strong_restarted_from_pristine: bool
    receipts: tuple[CallReceipt, ...]


FullEpisodeRunner = Callable[
    [FullEpisodeBinding, ModelRole, str],
    GeneratedEpisode,
]


@dataclass(frozen=True)
class AutoMixVerification:
    confidence: float
    sample_count: int
    receipts: tuple[CallReceipt, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("automix_verification_confidence_invalid")
        if self.sample_count != AUTOMIX_MAIN_VERIFIER_SAMPLES:
            raise ValueError("automix_verification_sample_count_not_frozen")
        if not self.receipts:
            raise ValueError("automix_verification_receipts_empty")
        if any(receipt.role != "verifier" for receipt in self.receipts):
            raise ValueError("automix_verification_receipt_role_invalid")


AutoMixVerifier = Callable[
    [AutoMixRouterView, int],
    AutoMixVerification,
]
AutoMixPOMDP = Callable[[float], bool]


def run_routellm_episode(
    *,
    artifact: RouterArtifactBinding,
    episode: FullEpisodeBinding,
    predict_strong_win_rate: PredictScalar,
    threshold: float,
    run_episode: FullEpisodeRunner,
    score_selected: ScoreSelected,
) -> StaticRoutingResult:
    _validate_static_inputs(
        artifact=artifact,
        episode=episode,
        expected_method=ROUTELLM,
        run_episode=run_episode,
        score_selected=score_selected,
    )
    view = RequestRouterView(episode.original_request)
    prediction, router_receipts = _timed_scalar_prediction(
        artifact.method,
        predict_strong_win_rate,
        view,
    )
    _validate_routellm_receipts(router_receipts)
    decision = decide_routellm(
        binding=artifact,
        view=view,
        predicted_strong_win_rate=prediction,
        threshold=threshold,
    )
    generated = run_episode(
        episode,
        decision.selected_role,
        decision.selected_model,
    )
    _validate_generated_episode(
        generated,
        episode=episode,
        expected_role=decision.selected_role,
        expected_model=decision.selected_model,
    )
    score = _score_after_selection(score_selected, generated.opaque_score_handle)
    return StaticRoutingResult(
        method=artifact.method,
        decision=decision,
        selected_role=decision.selected_role,
        selected_model=decision.selected_model,
        official_score=score,
        episode_count=1,
        strong_restarted_from_pristine=False,
        receipts=(*router_receipts, *generated.receipts),
    )


def run_frugalgpt_episode(
    *,
    artifact: RouterArtifactBinding,
    episode: FullEpisodeBinding,
    predict_cheap_quality: PredictScalar,
    accept_threshold: float,
    run_episode: FullEpisodeRunner,
    score_selected: ScoreSelected,
) -> StaticRoutingResult:
    _validate_static_inputs(
        artifact=artifact,
        episode=episode,
        expected_method=FRUGALGPT,
        run_episode=run_episode,
        score_selected=score_selected,
    )
    cheap_episode = run_episode(
        episode,
        "cheap",
        artifact.cheap_model,
    )
    _validate_generated_episode(
        cheap_episode,
        episode=episode,
        expected_role="cheap",
        expected_model=artifact.cheap_model,
    )
    view = CheapCompletionRouterView(
        original_request=episode.original_request,
        cheap_generated_answer=cheap_episode.generated_answer_view,
    )
    prediction, router_receipts = _timed_scalar_prediction(
        artifact.method,
        predict_cheap_quality,
        view,
    )
    decision = decide_frugalgpt(
        binding=artifact,
        view=view,
        predicted_cheap_quality=prediction,
        accept_threshold=accept_threshold,
    )
    generated = cheap_episode
    receipts: tuple[CallReceipt, ...] = (
        *cheap_episode.receipts,
        *router_receipts,
    )
    episode_count = 1
    restarted = False
    if decision.selected_role == "strong":
        strong_episode = run_episode(
            episode,
            "strong",
            artifact.strong_model,
        )
        _validate_generated_episode(
            strong_episode,
            episode=episode,
            expected_role="strong",
            expected_model=artifact.strong_model,
        )
        if (
            strong_episode.environment_instance_sha256
            == cheap_episode.environment_instance_sha256
        ):
            raise RuntimeError("frugalgpt_strong_environment_not_fresh")
        generated = strong_episode
        receipts = (*receipts, *strong_episode.receipts)
        episode_count = 2
        restarted = True
    score = _score_after_selection(score_selected, generated.opaque_score_handle)
    return StaticRoutingResult(
        method=artifact.method,
        decision=decision,
        selected_role=decision.selected_role,
        selected_model=decision.selected_model,
        official_score=score,
        episode_count=episode_count,
        strong_restarted_from_pristine=restarted,
        receipts=receipts,
    )


def run_automix_episode(
    *,
    artifact: RouterArtifactBinding,
    episode: FullEpisodeBinding,
    verify_cheap: AutoMixVerifier,
    pomdp_retry: AutoMixPOMDP,
    run_episode: FullEpisodeRunner,
    score_selected: ScoreSelected,
) -> StaticRoutingResult:
    _validate_static_inputs(
        artifact=artifact,
        episode=episode,
        expected_method=AUTOMIX,
        run_episode=run_episode,
        score_selected=score_selected,
    )
    if not callable(verify_cheap):
        raise TypeError("automix_verifier_not_callable")
    if not callable(pomdp_retry):
        raise TypeError("automix_pomdp_not_callable")
    cheap_episode = run_episode(
        episode,
        "cheap",
        artifact.cheap_model,
    )
    _validate_generated_episode(
        cheap_episode,
        episode=episode,
        expected_role="cheap",
        expected_model=artifact.cheap_model,
    )
    view = AutoMixRouterView(
        original_request=episode.original_request,
        cheap_public_episode=cheap_episode.public_episode_history,
        cheap_generated_answer=cheap_episode.generated_answer_view,
    )
    verification = verify_cheap(view, AUTOMIX_MAIN_VERIFIER_SAMPLES)
    if not isinstance(verification, AutoMixVerification):
        raise TypeError("automix_verification_result_invalid")
    if any(
        receipt.model != artifact.cheap_model
        for receipt in verification.receipts
    ):
        raise ValueError("automix_verifier_model_not_cheap")
    if sum(
        receipt.provider_calls for receipt in verification.receipts
    ) != 1:
        raise ValueError("automix_verifier_provider_call_count_not_one")
    started = time.perf_counter()
    try:
        retry = pomdp_retry(float(verification.confidence))
    except Exception:
        retry = None
    latency = max(0.0, time.perf_counter() - started)
    decision = decide_automix(
        binding=artifact,
        view=view,
        verification_confidence=float(verification.confidence),
        verifier_samples=verification.sample_count,
        pomdp_num_bins=AUTOMIX_POMDP_NUM_BINS,
        pomdp_retry=retry,
    )
    receipts: tuple[CallReceipt, ...] = (
        *cheap_episode.receipts,
        *verification.receipts,
        CallReceipt.local_router(
            method=artifact.method,
            latency_seconds=latency,
        ),
    )
    generated = cheap_episode
    episode_count = 1
    restarted = False
    if decision.selected_role == "strong":
        strong_episode = run_episode(
            episode,
            "strong",
            artifact.strong_model,
        )
        _validate_generated_episode(
            strong_episode,
            episode=episode,
            expected_role="strong",
            expected_model=artifact.strong_model,
        )
        if (
            strong_episode.environment_instance_sha256
            == cheap_episode.environment_instance_sha256
        ):
            raise RuntimeError("automix_strong_environment_not_fresh")
        generated = strong_episode
        receipts = (*receipts, *strong_episode.receipts)
        episode_count = 2
        restarted = True
    score = _score_after_selection(score_selected, generated.opaque_score_handle)
    return StaticRoutingResult(
        method=artifact.method,
        decision=decision,
        selected_role=decision.selected_role,
        selected_model=decision.selected_model,
        official_score=score,
        episode_count=episode_count,
        strong_restarted_from_pristine=restarted,
        receipts=receipts,
    )


@dataclass(frozen=True)
class HistoryTurn:
    appended_public_history: tuple[dict[str, Any], ...]
    receipts: tuple[CallReceipt, ...]
    terminal: bool
    opaque_score_handle: Any | None = None

    def __post_init__(self) -> None:
        history = validate_public_history(self.appended_public_history)
        if not history:
            raise ValueError("history_turn_append_empty")
        object.__setattr__(self, "appended_public_history", history)
        if any(not isinstance(item, CallReceipt) for item in self.receipts):
            raise TypeError("history_turn_receipt_invalid")
        if self.terminal and self.opaque_score_handle is None:
            raise ValueError("history_turn_terminal_score_handle_missing")
        if not self.terminal and self.opaque_score_handle is not None:
            raise ValueError("history_turn_nonterminal_score_handle_present")


@dataclass(frozen=True)
class HistoryRoutingResult:
    method: str
    status: str
    decisions: tuple[RouterDecision, ...]
    selected_roles: tuple[ModelRole, ...]
    official_score: float | None
    final_public_history: tuple[dict[str, Any], ...]
    receipts: tuple[CallReceipt, ...]


HistoryDecision = Callable[[HistoryRouterView], RouterDecision]
ExecuteHistoryTurn = Callable[
    [RouterDecision, tuple[dict[str, Any], ...]],
    HistoryTurn,
]


def run_history_routed_episode(
    *,
    artifact: RouterArtifactBinding,
    original_request: str,
    initial_public_history: Sequence[Mapping[str, Any]],
    turn_cap: int,
    decide_next: HistoryDecision,
    execute_turn: ExecuteHistoryTurn,
    score_terminal: ScoreSelected,
) -> HistoryRoutingResult:
    if artifact.method not in {MTROUTER, OPENSQUILLA_G0}:
        raise ValueError("history_router_method_invalid")
    if not callable(decide_next):
        raise TypeError("history_router_decider_not_callable")
    if not callable(execute_turn):
        raise TypeError("history_router_executor_not_callable")
    if not callable(score_terminal):
        raise TypeError("history_router_scorer_not_callable")
    if isinstance(turn_cap, bool) or not isinstance(turn_cap, int) or turn_cap <= 0:
        raise ValueError("history_router_turn_cap_invalid")
    history = validate_public_history(initial_public_history)
    decisions: list[RouterDecision] = []
    receipts: list[CallReceipt] = []
    for turn_index in range(turn_cap):
        view = HistoryRouterView(
            original_request=original_request,
            public_history=history,
            turn_index=turn_index,
            turn_cap=turn_cap,
        )
        started = time.perf_counter()
        decision = decide_next(view)
        latency = max(0.0, time.perf_counter() - started)
        _validate_history_decision(
            decision,
            artifact=artifact,
        )
        decisions.append(decision)
        receipts.append(
            CallReceipt.local_router(
                method=artifact.method,
                latency_seconds=latency,
            )
        )
        turn = execute_turn(decision, deepcopy(history))
        if not isinstance(turn, HistoryTurn):
            raise TypeError("history_router_turn_result_invalid")
        history = (*history, *turn.appended_public_history)
        receipts.extend(turn.receipts)
        if turn.terminal:
            score = _score_after_selection(
                score_terminal,
                turn.opaque_score_handle,
            )
            return HistoryRoutingResult(
                method=artifact.method,
                status="completed",
                decisions=tuple(decisions),
                selected_roles=tuple(
                    decision.selected_role for decision in decisions
                ),
                official_score=score,
                final_public_history=deepcopy(history),
                receipts=tuple(receipts),
            )
    return HistoryRoutingResult(
        method=artifact.method,
        status="turn_cap_exhausted",
        decisions=tuple(decisions),
        selected_roles=tuple(
            decision.selected_role for decision in decisions
        ),
        official_score=None,
        final_public_history=deepcopy(history),
        receipts=tuple(receipts),
    )


def _validate_static_inputs(
    *,
    artifact: RouterArtifactBinding,
    episode: FullEpisodeBinding,
    expected_method: str,
    run_episode: FullEpisodeRunner,
    score_selected: ScoreSelected,
) -> None:
    if artifact.method != expected_method:
        raise ValueError("static_router_method_mismatch")
    if artifact.surface != episode.surface:
        raise ValueError("static_router_surface_mismatch")
    if not callable(run_episode):
        raise TypeError("static_router_episode_runner_not_callable")
    if not callable(score_selected):
        raise TypeError("static_router_scorer_not_callable")


def _timed_scalar_prediction(
    method: str,
    predict: PredictScalar,
    view: Any,
) -> tuple[float | None, tuple[CallReceipt, ...]]:
    if not callable(predict):
        raise TypeError("router_predictor_not_callable")
    started = time.perf_counter()
    try:
        raw_value = predict(view)
    except Exception:
        raw_value = None
    latency = max(0.0, time.perf_counter() - started)
    if isinstance(raw_value, ScalarPrediction):
        return raw_value.value, raw_value.receipts
    return raw_value, (
        CallReceipt.local_router(
            method=method,
            latency_seconds=latency,
        ),
    )


def _validate_routellm_receipts(receipts: tuple[CallReceipt, ...]) -> None:
    embedding_receipts = tuple(
        receipt
        for receipt in receipts
        if receipt.role == "router"
        and receipt.model == ROUTELLM_EMBEDDING_MODEL
        and receipt.provider_calls > 0
    )
    embedding_calls = sum(
        receipt.provider_calls for receipt in embedding_receipts
    )
    total_calls = sum(receipt.provider_calls for receipt in receipts)
    local_bert_receipts = tuple(
        receipt
        for receipt in receipts
        if receipt.role == "router"
        and receipt.model == ROUTELLM_LOCAL_MODEL
        and receipt.provider_calls == 0
        and receipt.reasoning_effort == "local"
    )
    if (
        len(local_bert_receipts) == 1
        and len(receipts) == 1
        and total_calls == 0
    ):
        return
    if embedding_calls == 0:
        raise RuntimeError("routellm_test_embedding_receipt_missing")
    if embedding_calls != 1 or total_calls != 1:
        raise RuntimeError(
            "routellm_embedding_provider_call_count_not_one"
        )


def _validate_generated_episode(
    generated: GeneratedEpisode,
    *,
    episode: FullEpisodeBinding,
    expected_role: ModelRole,
    expected_model: str,
) -> None:
    if not isinstance(generated, GeneratedEpisode):
        raise TypeError("generated_episode_result_invalid")
    if generated.role != expected_role:
        raise ValueError("generated_episode_role_mismatch")
    if generated.model != expected_model:
        raise ValueError("generated_episode_model_mismatch")
    if generated.initial_environment_sha256 != episode.initial_environment_sha256:
        raise RuntimeError("generated_episode_initial_environment_mismatch")


def _score_after_selection(score_selected: ScoreSelected, handle: Any) -> float:
    value = score_selected(handle)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError("selected_episode_official_score_invalid")
    return float(value)


def _validate_history_decision(
    decision: RouterDecision,
    *,
    artifact: RouterArtifactBinding,
) -> None:
    if not isinstance(decision, RouterDecision):
        raise TypeError("history_router_decision_invalid")
    if decision.method != artifact.method:
        raise ValueError("history_router_decision_method_mismatch")
    if decision.surface != artifact.surface:
        raise ValueError("history_router_decision_surface_mismatch")
    if decision.artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("history_router_decision_artifact_mismatch")
    if decision.selected_model != artifact.model_for(decision.selected_role):
        raise ValueError("history_router_decision_model_mismatch")

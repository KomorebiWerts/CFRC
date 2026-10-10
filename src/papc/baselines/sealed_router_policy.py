"""Typed, artifact-bound runtime policies for official external-router runs.

The public surface drivers accept only policies produced by the method-specific
loaders in this module.  Provider callbacks remain transports: thresholds,
POMDP decisions, checkpoint selection, and verifier generation settings all
come from the validated typed artifact.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.automix_upstream import (
    AutoMixRuntimeArtifact,
    AutoMixVerifierAdapter,
    AutoMixVerifierRequest,
    VerifierTokenizer,
)
from papc.baselines.frugalgpt_upstream import (
    FrugalGPTDistilBertScorer,
    FrugalGPTScorerArtifact,
)
from papc.baselines.mtrouter_upstream import (
    MTRouterArtifact,
    load_mtrouter_runtime,
)
from papc.baselines.opensquilla_upstream import (
    OpenSquillaPriorRouteDecision,
    OpenSquillaPublicContext,
    OpenSquillaRuntimeArtifact,
    OpenSquillaTurnInput,
    OpenSquillaUpstreamAdapter,
)
from papc.baselines.routellm_bert_upstream import (
    RouteLLMBertArtifact,
    RouteLLMBertScorer,
)
from papc.baselines.routellm_mf_upstream import (
    EmbedPrompt,
    RouteLLMMFArtifact,
    load_routellm_mf_predictor,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    AUTOMIX,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    ROUTELLM_LOCAL_MODEL,
    AutoMixRouterView,
    CheapCompletionRouterView,
    ExternalMethod,
    HistoryRouterView,
    PublicFeature,
    RankerRouterView,
    RequestRouterView,
    RouterArtifactBinding,
)
from papc.baselines.router_policies import (
    RouterDecision,
    decide_opensquilla_g0,
)
from papc.baselines.router_runtime import (
    AutoMixVerification,
    HistoryDecision,
    ScalarPrediction,
)

_SEAL = object()
_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_FILE_RE = re.compile(r"(?:^|[\\/\\s])[^\\s/\\\\]+\\.[a-z0-9]{1,8}(?:$|\\s)")


@dataclass(frozen=True)
class AutoMixVerifierTransportResult:
    """Provider outputs tied to the exact typed verifier request."""

    request_prompt_sha256: str
    outputs: tuple[str, ...]
    receipts: tuple[CallReceipt, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_prompt_sha256, str)
            or len(self.request_prompt_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.request_prompt_sha256
            )
        ):
            raise ValueError("automix_transport_request_sha256_invalid")
        outputs = tuple(self.outputs)
        if not outputs or any(not isinstance(item, str) for item in outputs):
            raise ValueError("automix_transport_outputs_invalid")
        receipts = tuple(self.receipts)
        if not receipts or any(
            not isinstance(item, CallReceipt) for item in receipts
        ):
            raise TypeError("automix_transport_receipts_invalid")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "receipts", receipts)


class AutoMixVerifierTransport(Protocol):
    def __call__(
        self,
        request: AutoMixVerifierRequest,
    ) -> AutoMixVerifierTransportResult: ...


class SealedRouterPolicy:
    """Opaque policy accepted by official surface drivers."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("sealed_router_policy_is_immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        seal: object,
    ) -> None:
        if seal is not _SEAL:
            raise TypeError("sealed_router_policy_loader_required")
        if not isinstance(binding, RouterArtifactBinding):
            raise TypeError("sealed_router_binding_required")
        self._binding = binding
        self._seal = seal
        self._frozen = False

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    @property
    def binding(self) -> RouterArtifactBinding:
        return self._binding

    def predict_routellm(
        self,
        view: RequestRouterView,
    ) -> ScalarPrediction:
        raise TypeError("sealed_routellm_policy_required")

    @property
    def routellm_threshold(self) -> float:
        raise TypeError("sealed_routellm_policy_required")

    def predict_frugalgpt(
        self,
        view: CheapCompletionRouterView,
    ) -> float:
        raise TypeError("sealed_frugalgpt_policy_required")

    @property
    def frugalgpt_accept_threshold(self) -> float:
        raise TypeError("sealed_frugalgpt_policy_required")

    def verify_automix(
        self,
        view: AutoMixRouterView,
        sample_count: int,
    ) -> AutoMixVerification:
        raise TypeError("sealed_automix_policy_required")

    def automix_should_retry(self, confidence: float) -> bool:
        raise TypeError("sealed_automix_policy_required")

    def history_decider(self) -> HistoryDecision:
        raise TypeError("sealed_history_policy_required")

    def public_record(self) -> dict[str, Any]:
        return {
            "sealed": True,
            "binding": self.binding.public_record(),
        }


class _RouteLLMPolicy(SealedRouterPolicy):
    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        predictor: Callable[[RequestRouterView], ScalarPrediction],
        threshold: float,
        seal: object,
    ) -> None:
        super().__init__(binding=binding, seal=seal)
        self._predictor = predictor
        self._threshold = threshold
        self._freeze()

    def predict_routellm(
        self,
        view: RequestRouterView,
    ) -> ScalarPrediction:
        prediction = self._predictor(view)
        if not isinstance(prediction, ScalarPrediction):
            raise TypeError("sealed_routellm_prediction_invalid")
        embedding_calls = sum(
            receipt.provider_calls
            for receipt in prediction.receipts
            if receipt.role == "router"
            and receipt.model == ROUTELLM_EMBEDDING_MODEL
        )
        total_calls = sum(
            receipt.provider_calls for receipt in prediction.receipts
        )
        local_calls = sum(
            1
            for receipt in prediction.receipts
            if receipt.role == "router"
            and receipt.model == ROUTELLM_LOCAL_MODEL
            and receipt.provider_calls == 0
            and receipt.reasoning_effort == "local"
        )
        if (
            local_calls == 1
            and len(prediction.receipts) == 1
            and total_calls == 0
        ):
            return prediction
        if embedding_calls != 1 or total_calls != 1:
            raise ValueError(
                "sealed_routellm_embedding_provider_call_count_not_one"
            )
        return prediction

    @property
    def routellm_threshold(self) -> float:
        return self._threshold


class _FrugalGPTPolicy(SealedRouterPolicy):
    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        scorer: Callable[[CheapCompletionRouterView], float],
        accept_threshold: float,
        seal: object,
    ) -> None:
        super().__init__(binding=binding, seal=seal)
        self._scorer = scorer
        self._accept_threshold = accept_threshold
        self._freeze()

    def predict_frugalgpt(
        self,
        view: CheapCompletionRouterView,
    ) -> float:
        return self._scorer(view)

    @property
    def frugalgpt_accept_threshold(self) -> float:
        return self._accept_threshold


class _AutoMixPolicy(SealedRouterPolicy):
    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        adapter: AutoMixVerifierAdapter,
        transport: AutoMixVerifierTransport,
        seal: object,
    ) -> None:
        super().__init__(binding=binding, seal=seal)
        self._adapter = adapter
        self._transport = transport
        self._freeze()

    def verify_automix(
        self,
        view: AutoMixRouterView,
        sample_count: int,
    ) -> AutoMixVerification:
        request = self._adapter.build_request(view)
        if sample_count != request.sample_count:
            raise ValueError("automix_policy_sample_count_not_artifact_bound")
        result = self._transport(request)
        if not isinstance(result, AutoMixVerifierTransportResult):
            raise TypeError("automix_transport_result_invalid")
        if result.request_prompt_sha256 != request.prompt_sha256:
            raise ValueError("automix_transport_request_sha256_mismatch")
        if sum(
            receipt.provider_calls for receipt in result.receipts
        ) != 1:
            raise ValueError(
                "automix_transport_provider_call_count_not_one"
            )
        return self._adapter.consume_outputs(
            outputs=result.outputs,
            receipts=result.receipts,
        )

    def automix_should_retry(self, confidence: float) -> bool:
        return self._adapter.should_retry(confidence)


class _MTRouterPolicy(SealedRouterPolicy):
    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        runtime: Any,
        seal: object,
    ) -> None:
        super().__init__(binding=binding, seal=seal)
        self._runtime = runtime
        self._freeze()

    def history_decider(self) -> HistoryDecision:
        binding = self.binding
        runtime = self._runtime

        def decide(view: HistoryRouterView) -> RouterDecision:
            return runtime.decide(binding=binding, view=view)

        return decide


class _OpenSquillaPolicy(SealedRouterPolicy):
    def __init__(
        self,
        *,
        binding: RouterArtifactBinding,
        artifact: OpenSquillaRuntimeArtifact,
        adapter: OpenSquillaUpstreamAdapter,
        seal: object,
    ) -> None:
        super().__init__(binding=binding, seal=seal)
        self._artifact = artifact
        self._adapter = adapter
        self._freeze()

    def history_decider(self) -> HistoryDecision:
        binding = self.binding
        artifact = self._artifact
        adapter = self._adapter
        prior_predictions: list[OpenSquillaPriorRouteDecision] = []
        prior_roles: list[str] = []

        def decide(view: HistoryRouterView) -> RouterDecision:
            turn, ranker_view = _opensquilla_views(
                view,
                prior_predictions=tuple(prior_predictions[-5:]),
                prior_roles=tuple(prior_roles),
            )
            prediction = _run_opensquilla_prediction(adapter, turn)
            if prediction.public_input_sha256 != turn.sha256:
                raise ValueError("opensquilla_prediction_input_mismatch")
            if prediction.bundle_sha256 != artifact.bundle_sha256:
                raise ValueError("opensquilla_prediction_bundle_mismatch")
            if prediction.projection_sha256 != artifact.projection.sha256:
                raise ValueError("opensquilla_prediction_projection_mismatch")
            route_index = int(prediction.route_class[1:])
            decision = decide_opensquilla_g0(
                binding=binding,
                view=ranker_view,
                route_class=route_index,
                confidence=prediction.confidence,
                strong_class_floor=(
                    artifact.projection.strong_class_floor
                ),
            )
            if decision.selected_role != prediction.selected_role:
                raise ValueError("opensquilla_projection_decision_mismatch")
            prior_predictions.append(
                OpenSquillaPriorRouteDecision(
                    route_class=prediction.route_class,
                    difficulty=prediction.difficulty,
                    margin=prediction.margin,
                )
            )
            prior_roles.append(decision.selected_role)
            return decision

        return decide


def load_routellm_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: RouteLLMMFArtifact,
    checkpoint_payload: bytes,
    embed_prompt: EmbedPrompt,
) -> SealedRouterPolicy:
    if not isinstance(artifact, RouteLLMMFArtifact):
        raise TypeError("sealed_routellm_typed_artifact_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.to_router_binding(),
        method=ROUTELLM,
    )
    predictor = load_routellm_mf_predictor(
        artifact=artifact,
        checkpoint_payload=checkpoint_payload,
        embed_prompt=embed_prompt,
    )
    return _RouteLLMPolicy(
        binding=resolved,
        predictor=predictor,
        threshold=artifact.threshold,
        seal=_SEAL,
    )


def load_routellm_bert_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: RouteLLMBertArtifact,
    checkpoint_dir: Path,
) -> SealedRouterPolicy:
    if not isinstance(artifact, RouteLLMBertArtifact):
        raise TypeError("sealed_routellm_bert_typed_artifact_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.to_router_binding(),
        method=ROUTELLM,
    )
    scorer = RouteLLMBertScorer(
        artifact=artifact,
        checkpoint_dir=checkpoint_dir,
    )
    return _RouteLLMPolicy(
        binding=resolved,
        predictor=scorer,
        threshold=artifact.threshold,
        seal=_SEAL,
    )


def load_released_routellm_bert_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    base_model_dir: Path,
    threshold: float,
) -> SealedRouterPolicy:
    """Load released RouteLLM-BERT with an externally frozen OOD threshold."""

    if not isinstance(binding, RouterArtifactBinding) or binding.method != ROUTELLM:
        raise TypeError("sealed_released_routellm_bert_binding_required")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("sealed_released_routellm_bert_threshold_invalid")
    scorer = RouteLLMBertScorer.from_released_base_bundle(base_model_dir)
    return _RouteLLMPolicy(
        binding=binding,
        predictor=scorer,
        threshold=float(threshold),
        seal=_SEAL,
    )


def load_frugalgpt_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: FrugalGPTScorerArtifact,
    checkpoint_dir: Path,
) -> SealedRouterPolicy:
    if not isinstance(artifact, FrugalGPTScorerArtifact):
        raise TypeError("sealed_frugalgpt_typed_artifact_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.router_binding(),
        method=FRUGALGPT,
    )
    scorer = FrugalGPTDistilBertScorer.load(
        artifact=artifact,
        checkpoint_dir=checkpoint_dir,
    )
    return _FrugalGPTPolicy(
        binding=resolved,
        scorer=scorer,
        accept_threshold=artifact.operating_point.accept_threshold,
        seal=_SEAL,
    )


def load_automix_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: AutoMixRuntimeArtifact,
    tokenizer: VerifierTokenizer,
    transport: AutoMixVerifierTransport,
) -> SealedRouterPolicy:
    if not isinstance(artifact, AutoMixRuntimeArtifact):
        raise TypeError("sealed_automix_typed_artifact_required")
    if not callable(transport):
        raise TypeError("sealed_automix_transport_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.router_binding(),
        method=AUTOMIX,
    )
    return _AutoMixPolicy(
        binding=resolved,
        adapter=AutoMixVerifierAdapter(
            artifact=artifact,
            tokenizer=tokenizer,
        ),
        transport=transport,
        seal=_SEAL,
    )


def load_mtrouter_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: MTRouterArtifact,
    checkpoint_path: Path | None,
    device: str = "cpu",
) -> SealedRouterPolicy:
    if not isinstance(artifact, MTRouterArtifact):
        raise TypeError("sealed_mtrouter_typed_artifact_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.router_binding(),
        method=MTROUTER,
    )
    runtime = load_mtrouter_runtime(
        artifact=artifact,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return _MTRouterPolicy(
        binding=resolved,
        runtime=runtime,
        seal=_SEAL,
    )


def load_opensquilla_sealed_policy(
    *,
    binding: RouterArtifactBinding,
    artifact: OpenSquillaRuntimeArtifact,
    checkout_root: Path,
    bundle_dir: Path,
) -> SealedRouterPolicy:
    if not isinstance(artifact, OpenSquillaRuntimeArtifact):
        raise TypeError("sealed_opensquilla_typed_artifact_required")
    resolved = _match_binding(
        expected=binding,
        actual=artifact.router_binding(),
        method=OPENSQUILLA_G0,
    )
    adapter = OpenSquillaUpstreamAdapter.load(
        checkout_root=checkout_root,
        projection=artifact.projection,
        bundle_dir=bundle_dir,
        expected_bundle_sha256=artifact.bundle_sha256,
    )
    return _OpenSquillaPolicy(
        binding=resolved,
        artifact=artifact,
        adapter=adapter,
        seal=_SEAL,
    )


def require_sealed_router_policy(
    policy: Any,
) -> SealedRouterPolicy:
    if (
        not isinstance(policy, SealedRouterPolicy)
        or getattr(policy, "_seal", None) is not _SEAL
        or getattr(policy, "_frozen", False) is not True
    ):
        raise TypeError("sealed_router_policy_loader_required")
    if not isinstance(policy.binding, RouterArtifactBinding):
        raise TypeError("sealed_router_binding_required")
    return policy


def _match_binding(
    *,
    expected: RouterArtifactBinding,
    actual: RouterArtifactBinding,
    method: ExternalMethod,
) -> RouterArtifactBinding:
    if not isinstance(expected, RouterArtifactBinding):
        raise TypeError("sealed_router_expected_binding_required")
    if not isinstance(actual, RouterArtifactBinding):
        raise TypeError("sealed_router_artifact_binding_required")
    if expected.method != method or actual.method != method:
        raise ValueError("sealed_router_method_mismatch")
    for item in fields(RouterArtifactBinding):
        field_name = item.name
        if getattr(expected, field_name) != getattr(actual, field_name):
            raise ValueError(f"sealed_router_binding_{field_name}_mismatch")
    return actual


def _run_opensquilla_prediction(
    adapter: OpenSquillaUpstreamAdapter,
    turn: OpenSquillaTurnInput,
) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(adapter.predict(turn))
    raise RuntimeError("opensquilla_sync_driver_running_event_loop")


def _opensquilla_views(
    view: HistoryRouterView,
    *,
    prior_predictions: tuple[OpenSquillaPriorRouteDecision, ...],
    prior_roles: tuple[str, ...],
) -> tuple[OpenSquillaTurnInput, RankerRouterView]:
    if not isinstance(view, HistoryRouterView):
        raise TypeError("opensquilla_history_view_required")
    history_text = _history_json(view.public_history)
    user_turns = [
        _message_text(item)
        for item in view.public_history
        if item.get("role") == "user"
    ]
    current_user = user_turns[-1] if user_turns else view.original_request
    recent_users = tuple(user_turns[:-1][-4:])
    assistant_turns = [
        _message_text(item)
        for item in view.public_history
        if item.get("role") == "assistant"
    ]
    previous_assistant = assistant_turns[-1] if assistant_turns else None
    tool_outputs = [
        _message_text(item)
        for item in view.public_history
        if item.get("role") == "tool" or "output" in item
    ]
    context = OpenSquillaPublicContext(
        turn_index=view.turn_index,
        context_tokens_est=max(1, (len(history_text) + 3) // 4),
        n_tools=sum(
            1
            for item in view.public_history
            if item.get("role") == "tool" or "tool_calls" in item
        ),
        tool_result_length=sum(len(item) for item in tool_outputs),
        has_code_block="```" in history_text,
        has_file_reference=bool(_FILE_RE.search(history_text)),
        has_url=bool(_URL_RE.search(history_text)),
        has_tool_results=bool(tool_outputs),
    )
    turn = OpenSquillaTurnInput(
        current_user_text=current_user,
        recent_user_turns=recent_users,
        previous_assistant_text=previous_assistant,
        previous_assistant_usage={},
        prior_route_decisions=prior_predictions,
        public_context_metadata=context,
    )
    public_features = tuple(
        PublicFeature(name=name, value=float(value))
        for name, value in context.public_record().items()
    )
    ranker_view = RankerRouterView(
        history=view,
        public_features=public_features,
        prior_route_decisions=prior_roles,  # type: ignore[arg-type]
        previous_assistant_usage=(),
    )
    return turn, ranker_view


def opensquilla_turn_input_from_history(
    view: HistoryRouterView,
) -> OpenSquillaTurnInput:
    """Project public history exactly as the production policy does."""

    turn, _ = _opensquilla_views(
        view,
        prior_predictions=(),
        prior_roles=(),
    )
    return turn


def _message_text(message: dict[str, Any]) -> str:
    value = message.get("content")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _history_json(history: Sequence[dict[str, Any]]) -> str:
    return json.dumps(
        history,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

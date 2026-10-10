"""Minimal StepWise runtime on the frozen Agent-Diff native session."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.baselines.agent_diff_live import (
    AgentDiffNativeAPI,
    AgentDiffNativeRunConfig,
    AgentDiffPrivateTask,
    _chat_message_text,
    _NativeSession,
    _parse_react_message,
    _provider_response_receipt,
    _require_complete_provider_receipt,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
    HistoryRouterView,
)
from papc.baselines.stepwise_upstream import (
    STEPWISE_METHOD,
    StepWiseController,
    StepWiseDecision,
    StepWiseMonitor,
)

STEPWISE_VERIFIER_MAX_TOKENS = 64
STEPWISE_VERIFIER_TEMPERATURE = 0.0


@dataclass(frozen=True)
class StepWiseDetectorArtifact:
    """Source-bound local detector directory produced by the upstream trainer."""

    path: Path
    bundle_sha256: str
    file_sha256s: tuple[tuple[str, str], ...]
    max_length: int
    eval_results: Mapping[str, Any]

    def public_record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bundle_sha256": self.bundle_sha256,
            "file_sha256s": dict(self.file_sha256s),
            "max_length": self.max_length,
            "eval_results": dict(self.eval_results),
        }


@dataclass(frozen=True)
class StepWiseAgentDiffEpisode:
    """One completed native episode with all StepWise decisions and receipts."""

    official_score: float
    decisions: tuple[StepWiseDecision, ...]
    public_history: tuple[Mapping[str, Any], ...]
    provider_receipts: tuple[CallReceipt, ...]
    local_router_receipts: tuple[CallReceipt, ...]
    raw_path: Path

    def __post_init__(self) -> None:
        if self.official_score not in {0.0, 1.0}:
            raise ValueError("stepwise_agent_diff_score_not_binary")
        if not self.decisions:
            raise ValueError("stepwise_agent_diff_decisions_empty")
        if not self.provider_receipts:
            raise ValueError("stepwise_agent_diff_provider_receipts_empty")


class ModernBERTStepWiseMonitor:
    """The two released binary classifiers, loaded for local CPU inference."""

    def __init__(self, *, stuck_dir: Path, milestone_dir: Path) -> None:
        self.stuck_artifact = inspect_detector_artifact(stuck_dir)
        self.milestone_artifact = inspect_detector_artifact(milestone_dir)
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._device = torch.device("cpu")
        self._stuck_tokenizer = AutoTokenizer.from_pretrained(
            self.stuck_artifact.path,
            local_files_only=True,
        )
        self._stuck_model = AutoModelForSequenceClassification.from_pretrained(
            self.stuck_artifact.path,
            local_files_only=True,
        ).to(self._device)
        self._milestone_tokenizer = AutoTokenizer.from_pretrained(
            self.milestone_artifact.path,
            local_files_only=True,
        )
        self._milestone_model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.milestone_artifact.path,
                local_files_only=True,
            ).to(self._device)
        )
        self._stuck_model.eval()
        self._milestone_model.eval()

    def __call__(self, stuck_text: str, milestone_text: str) -> tuple[float, float]:
        return (
            self._positive_probability(
                text=stuck_text,
                tokenizer=self._stuck_tokenizer,
                model=self._stuck_model,
                max_length=self.stuck_artifact.max_length,
            ),
            self._positive_probability(
                text=milestone_text,
                tokenizer=self._milestone_tokenizer,
                model=self._milestone_model,
                max_length=self.milestone_artifact.max_length,
            ),
        )

    def _positive_probability(
        self,
        *,
        text: str,
        tokenizer: Any,
        model: Any,
        max_length: int,
    ) -> float:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = model(**encoded).logits
            probability = self._torch.softmax(logits, dim=-1)[0, 1].item()
        return float(probability)


def inspect_detector_artifact(path: Path) -> StepWiseDetectorArtifact:
    """Bind the complete local classifier bundle and released train settings."""

    root = Path(path).resolve()
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_config.json",
        "eval_results.json",
    )
    hashes: list[tuple[str, str]] = []
    for name in required:
        candidate = root / name
        if not candidate.is_file():
            raise FileNotFoundError(f"stepwise_detector_file_missing:{name}")
        hashes.append((name, _file_sha256(candidate)))
    training = _read_mapping(root / "training_config.json")
    if (
        training.get("max_length") != 512
        or training.get("num_epochs") != 5
        or float(training.get("learning_rate", -1.0)) != 2e-5
        or training.get("seed") != 42
        or training.get("use_class_weights") is not True
        or training.get("weight_strategy") != "balanced"
    ):
        raise ValueError("stepwise_detector_training_config_drifted")
    eval_results = _read_mapping(root / "eval_results.json")
    return StepWiseDetectorArtifact(
        path=root,
        bundle_sha256=_digest(hashes),
        file_sha256s=tuple(hashes),
        max_length=512,
        eval_results=eval_results,
    )


def run_stepwise_agent_diff_episode(
    *,
    task: AgentDiffPrivateTask,
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: Any,
    raw_path: Path,
    api_key_env: str,
    monitor: StepWiseMonitor,
    session_factory: Callable[..., Any] = _NativeSession,
) -> StepWiseAgentDiffEpisode:
    """Run cheap-first StepWise with permanent takeover from current state."""

    if not callable(monitor):
        raise TypeError("stepwise_agent_diff_monitor_invalid")
    controller = StepWiseController()
    session = session_factory(
        task=task,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=Path(raw_path).resolve(),
        api_key_env=api_key_env,
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": task.ref.original_request}
    ]
    decisions: list[StepWiseDecision] = []
    local_receipts: list[CallReceipt] = []
    error: BaseException | None = None
    try:
        session.start()
        for iteration in range(1, config.max_iterations + 1):
            monitor_latency = 0.0

            def timed_monitor(stuck_text: str, milestone_text: str) -> tuple[float, float]:
                nonlocal monitor_latency
                started = time.perf_counter()
                result = monitor(stuck_text, milestone_text)
                monitor_latency += max(0.0, time.perf_counter() - started)
                return result

            def verify(packet: str) -> bool:
                receipt, verified = _milestone_verification(
                    packet=packet,
                    provider_client=provider_client,
                    broker=broker,
                )
                session.receipts.append(receipt)
                return verified

            decision = controller.decide(
                view=HistoryRouterView(
                    original_request=task.ref.original_request,
                    public_history=tuple(history),
                    turn_index=iteration - 1,
                    turn_cap=config.max_iterations,
                ),
                monitor=timed_monitor,
                verifier=verify,
            )
            decisions.append(decision)
            local_receipts.append(
                CallReceipt.local_router(
                    method=STEPWISE_METHOD,
                    latency_seconds=monitor_latency,
                )
            )
            model = CHEAP_MODEL if decision.selected_role == "cheap" else STRONG_MODEL
            text, receipt = session.model_turn(
                model=model,
                public_history=history,
            )
            parsed = _parse_react_message(text)
            event: dict[str, Any] = {
                "iteration": iteration,
                "model": model,
                "selected_role": decision.selected_role,
                "stepwise_decision": decision.public_record(),
                "local_monitor_latency_seconds": monitor_latency,
                "assistant_text": text,
                "assistant_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "receipt": receipt.public_record(),
            }
            action = parsed["action"]
            done = parsed["done"]
            if action:
                tool = session.execute_action(action)
                event.update(tool)
                session.trace.append(event)
                history.extend(
                    (
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "<observation>\n"
                                f"{tool['observation']}\n"
                                "</observation>"
                            ),
                        },
                    )
                )
                continue
            if done:
                event["done"] = done
                session.trace.append(event)
                history.append({"role": "assistant", "content": text})
                session.completion_reason = "done_tag"
                break
            event["format_error"] = "missing_action_or_done"
            session.trace.append(event)
            history.extend(
                (
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Please respond using either <action>one bash "
                            "command</action> or <done>brief summary</done>."
                        ),
                    },
                )
            )
        else:
            session.completion_reason = "max_iterations_exhausted"
        evaluation = session.evaluate()
        score = 1.0 if evaluation["passed"] is True else 0.0
        return StepWiseAgentDiffEpisode(
            official_score=score,
            decisions=tuple(decisions),
            public_history=tuple(history),
            provider_receipts=tuple(session.receipts),
            local_router_receipts=tuple(local_receipts),
            raw_path=Path(raw_path).resolve(),
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        session.finalize(error=error)


def _milestone_verification(
    *,
    packet: str,
    provider_client: Any,
    broker: Any,
) -> tuple[CallReceipt, bool]:
    response_box: dict[str, Any] = {}

    def raw_attempt() -> CallReceipt:
        started = time.perf_counter()
        response = provider_client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Verify recent tool-agent progress. Return exactly "
                        "CORRECT or INCORRECT."
                    ),
                },
                {"role": "user", "content": packet},
            ],
            reasoning_effort=REASONING_EFFORT,
            max_tokens=STEPWISE_VERIFIER_MAX_TOKENS,
            temperature=STEPWISE_VERIFIER_TEMPERATURE,
        )
        receipt = _provider_response_receipt(
            response=response,
            role="verifier",
            phase="milestone_verification",
            model=STRONG_MODEL,
            reasoning_effort=REASONING_EFFORT,
            latency_seconds=max(0.0, time.perf_counter() - started),
        )
        _require_complete_provider_receipt(receipt)
        response_box["text"] = _chat_message_text(response)
        return receipt

    receipt = broker.perform(raw_attempt)
    normalized = str(response_box.get("text", "")).strip().upper().rstrip(".")
    if normalized == "CORRECT":
        return receipt, True
    if normalized == "INCORRECT":
        return receipt, False
    raise ValueError("stepwise_milestone_verifier_response_invalid")


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("stepwise_detector_json_not_mapping")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    "STEPWISE_VERIFIER_MAX_TOKENS",
    "ModernBERTStepWiseMonitor",
    "StepWiseAgentDiffEpisode",
    "StepWiseDetectorArtifact",
    "inspect_detector_artifact",
    "run_stepwise_agent_diff_episode",
]

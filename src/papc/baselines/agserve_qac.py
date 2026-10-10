"""AgServe's quality-aware client adapted to a frozen two-model agent tier.

This module implements only the paper's client-side Q-Judge/R-Judge policy.
It intentionally does not claim AgServe's server-side KV-cache migration or
resource scheduler.  Judges see the public request and public native history;
they never receive benchmark identifiers, hidden state, or official outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.router_contract import (
    CHEAP_MODEL,
    STRONG_MODEL,
    HistoryRouterView,
    ModelRole,
)

AGSERVE_QAC_METHOD = "agserve_qac"
AGSERVE_UPSTREAM_REVISION = "174c671255b21b468a514d6d357cc21f0d4f36b2"
AGSERVE_QJ_MAX_LENGTH = 128
AGSERVE_RJ_MAX_LENGTH = 512
AGSERVE_UPDATE_PROBABILITY = 0.5

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HTTP_ERROR_RE = re.compile(
    r"(?:http(?:/\d(?:\.\d)?)?\s*(?:status)?|"
    r"status(?:\s*code)?|error(?:\s*code)?)\s*[:=]?\s*[45]\d\d\b",
    re.I,
)
_EXPLICIT_FAILURE_MARKERS = (
    "missing_action_or_done",
    "please respond using either <action>",
    "invalid action",
    "invalid tool",
    "tool not found",
    "command failed",
    "request failed",
    "permission denied",
    "unauthorized",
    "forbidden",
    "traceback (most recent call last)",
)


class AgServeJudge(Protocol):
    def task_strong_probability(self, request: str) -> float: ...

    def response_bad_probability(
        self,
        *,
        selected_role: ModelRole,
        request: str,
        public_history: tuple[dict[str, Any], ...],
    ) -> float: ...


@dataclass(frozen=True)
class AgServeArtifact:
    root: Path = field(repr=False)
    qj_dir: Path = field(repr=False)
    rj_dir: Path = field(repr=False)
    bundle_sha256: str
    qj_threshold: float
    rj_threshold: float
    state_check_frequency: int
    agent_diff_check_frequency: int
    metadata: dict[str, Any] = field(repr=False)

    def public_record(self) -> dict[str, Any]:
        source_bindings = self.metadata.get("source_bindings", {})
        qj_source = source_bindings.get("q_judge", {})
        rj_source = source_bindings.get("r_judge", {})
        return {
            "bundle_sha256": self.bundle_sha256,
            "upstream_revision": AGSERVE_UPSTREAM_REVISION,
            "qj_threshold": self.qj_threshold,
            "rj_threshold": self.rj_threshold,
            "state_check_frequency": self.state_check_frequency,
            "agent_diff_check_frequency": self.agent_diff_check_frequency,
            "training": self.metadata.get("training", {}),
            "source_bindings": {
                "q_judge": {
                    key: qj_source.get(key)
                    for key in (
                        "paired_examples",
                        "positive_strong_only",
                        "train_keys_sha256",
                        "calibration_keys_sha256",
                    )
                },
                "r_judge": {
                    key: rj_source.get(key)
                    for key in (
                        "train_count",
                        "calibration_count",
                        "train_labels",
                        "calibration_labels",
                    )
                },
                "official_test_examples_used": source_bindings.get(
                    "official_test_examples_used"
                ),
            },
        }


@dataclass(frozen=True)
class AgServeDecision:
    method: str
    selected_role: ModelRole
    selected_model: str
    turn_index: int
    public_input_sha256: str
    task_strong_probability: float | None
    response_bad_probability: float | None
    quality_issue: str | None
    failure_count: int
    rationale: str
    sticky_upgrade: bool

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


class AgServeQACController:
    """Two-tier Q-Judge/R-Judge controller with paper-faithful promotion.

    The released client retries a first failed quality check and promotes with
    probability 0.5, then promotes after the second consecutive failure.  We
    replace its process-global RNG with a hash-derived Bernoulli so the same
    public history gives the same decision without using a task identifier.
    """

    def __init__(
        self,
        *,
        judge: AgServeJudge,
        qj_threshold: float,
        rj_threshold: float,
        check_frequency: int,
    ) -> None:
        if not callable(getattr(judge, "task_strong_probability", None)):
            raise TypeError("agserve_qj_not_callable")
        if not callable(getattr(judge, "response_bad_probability", None)):
            raise TypeError("agserve_rj_not_callable")
        self.judge = judge
        self.qj_threshold = _unit_interval(qj_threshold, "qj_threshold")
        self.rj_threshold = _unit_interval(rj_threshold, "rj_threshold")
        if isinstance(check_frequency, bool) or not isinstance(check_frequency, int):
            raise TypeError("agserve_check_frequency_must_be_int")
        if check_frequency <= 0:
            raise ValueError("agserve_check_frequency_must_be_positive")
        self.check_frequency = check_frequency
        self._sticky_strong = False
        self._failure_count = 0
        self._cheap_responses_since_check = 0

    def decide(self, view: HistoryRouterView) -> AgServeDecision:
        if not isinstance(view, HistoryRouterView):
            raise TypeError("agserve_history_view_required")
        if view.turn_index == 0:
            probability = _unit_interval(
                self.judge.task_strong_probability(view.original_request),
                "task_strong_probability",
            )
            role: ModelRole = (
                "strong" if probability >= self.qj_threshold else "cheap"
            )
            self._sticky_strong = role == "strong"
            return self._decision(
                view=view,
                role=role,
                task_probability=probability,
                response_probability=None,
                issue=None,
                rationale=(
                    "q_judge_strong" if role == "strong" else "q_judge_cheap"
                ),
            )

        if self._sticky_strong:
            return self._decision(
                view=view,
                role="strong",
                task_probability=None,
                response_probability=None,
                issue=None,
                rationale="sticky_strong_after_upgrade",
            )

        self._cheap_responses_since_check += 1
        explicit_issue = _explicit_quality_issue(view.public_history)
        response_probability: float | None = None
        learned_issue: str | None = None
        if self._cheap_responses_since_check >= self.check_frequency:
            response_probability = _unit_interval(
                self.judge.response_bad_probability(
                    selected_role="cheap",
                    request=view.original_request,
                    public_history=view.public_history,
                ),
                "response_bad_probability",
            )
            self._cheap_responses_since_check = 0
            if response_probability > self.rj_threshold:
                learned_issue = "r_judge_low_quality"
        issue = explicit_issue or learned_issue
        if issue is None:
            self._failure_count = 0
            return self._decision(
                view=view,
                role="cheap",
                task_probability=None,
                response_probability=response_probability,
                issue=None,
                rationale=(
                    "r_judge_pass" if response_probability is not None
                    else "between_quality_checks"
                ),
            )

        self._failure_count += 1
        promote = self._failure_count >= 2 or _promotion_coin(view.sha256)
        if promote:
            self._sticky_strong = True
            self._failure_count = 0
            role = "strong"
            rationale = "quality_failure_upgrade"
        else:
            role = "cheap"
            rationale = "quality_failure_cheap_retry"
        return self._decision(
            view=view,
            role=role,
            task_probability=None,
            response_probability=response_probability,
            issue=issue,
            rationale=rationale,
        )

    def _decision(
        self,
        *,
        view: HistoryRouterView,
        role: ModelRole,
        task_probability: float | None,
        response_probability: float | None,
        issue: str | None,
        rationale: str,
    ) -> AgServeDecision:
        return AgServeDecision(
            method=AGSERVE_QAC_METHOD,
            selected_role=role,
            selected_model=CHEAP_MODEL if role == "cheap" else STRONG_MODEL,
            turn_index=view.turn_index,
            public_input_sha256=view.sha256,
            task_strong_probability=task_probability,
            response_bad_probability=response_probability,
            quality_issue=issue,
            failure_count=self._failure_count,
            rationale=rationale,
            sticky_upgrade=self._sticky_strong,
        )


class TransformerAgServeJudge:
    """Local BERT-family Q-Judge and DistilBERT R-Judge inference."""

    def __init__(self, artifact: AgServeArtifact, *, device: str = "cpu") -> None:
        if not isinstance(artifact, AgServeArtifact):
            raise TypeError("agserve_artifact_required")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = torch.device(device)
        self._qj_tokenizer = AutoTokenizer.from_pretrained(
            artifact.qj_dir, local_files_only=True
        )
        self._qj_model = AutoModelForSequenceClassification.from_pretrained(
            artifact.qj_dir, local_files_only=True
        ).to(self._device)
        self._rj_tokenizer = AutoTokenizer.from_pretrained(
            artifact.rj_dir, local_files_only=True
        )
        self._rj_model = AutoModelForSequenceClassification.from_pretrained(
            artifact.rj_dir, local_files_only=True
        ).to(self._device)
        self._qj_model.eval()
        self._rj_model.eval()

    def task_strong_probability(self, request: str) -> float:
        return self._positive_probability(
            text=request,
            tokenizer=self._qj_tokenizer,
            model=self._qj_model,
            max_length=AGSERVE_QJ_MAX_LENGTH,
        )

    def response_bad_probability(
        self,
        *,
        selected_role: ModelRole,
        request: str,
        public_history: tuple[dict[str, Any], ...],
    ) -> float:
        response = _latest_content(public_history, role="assistant")
        observation = _latest_content(public_history, role="user", skip_first=True)
        model_size = "smaller" if selected_role == "cheap" else "bigger"
        text = (
            f"[CLS] {model_size} [SEP] {request} [SEP] "
            f"{response} [SEP] {observation} [SEP]"
        )
        return self._positive_probability(
            text=text,
            tokenizer=self._rj_tokenizer,
            model=self._rj_model,
            max_length=AGSERVE_RJ_MAX_LENGTH,
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


def inspect_agserve_artifact(root: Path) -> AgServeArtifact:
    base = Path(root).resolve()
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("agserve_metadata_missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "papc.agserve_qac_artifact.v1":
        raise ValueError("agserve_artifact_schema_mismatch")
    if metadata.get("upstream_revision") != AGSERVE_UPSTREAM_REVISION:
        raise ValueError("agserve_upstream_revision_mismatch")
    qj_dir = base / "q_judge"
    rj_dir = base / "r_judge"
    required_names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    files = [metadata_path]
    for directory in (qj_dir, rj_dir):
        for name in required_names:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"agserve_model_file_missing:{path}")
            files.append(path)
    actual_hashes = {
        str(path.relative_to(base)): _file_sha256(path) for path in files
    }
    expected_hashes = metadata.get("file_sha256s")
    if not isinstance(expected_hashes, dict) or any(
        expected_hashes.get(name) != digest
        for name, digest in actual_hashes.items()
        if name != "metadata.json"
    ):
        raise ValueError("agserve_artifact_file_hash_mismatch")
    bundle = _digest(
        {name: digest for name, digest in actual_hashes.items() if name != "metadata.json"}
    )
    if bundle != metadata.get("bundle_sha256") or not _SHA256_RE.fullmatch(bundle):
        raise ValueError("agserve_artifact_bundle_hash_mismatch")
    thresholds = metadata.get("thresholds")
    frequencies = metadata.get("check_frequencies")
    if not isinstance(thresholds, dict) or not isinstance(frequencies, dict):
        raise TypeError("agserve_artifact_policy_missing")
    return AgServeArtifact(
        root=base,
        qj_dir=qj_dir,
        rj_dir=rj_dir,
        bundle_sha256=bundle,
        qj_threshold=_unit_interval(thresholds.get("q_judge"), "qj_threshold"),
        rj_threshold=_unit_interval(thresholds.get("r_judge"), "rj_threshold"),
        state_check_frequency=_positive_int(
            frequencies.get("state_bench"), "state_check_frequency"
        ),
        agent_diff_check_frequency=_positive_int(
            frequencies.get("agent_diff"), "agent_diff_check_frequency"
        ),
        metadata=metadata,
    )


def _explicit_quality_issue(history: tuple[dict[str, Any], ...]) -> str | None:
    recent = "\n".join(
        str(item.get("content", "")) for item in history[-2:]
    ).lower()
    if any(marker in recent for marker in _EXPLICIT_FAILURE_MARKERS):
        return "explicit_runtime_failure"
    if _HTTP_ERROR_RE.search(recent):
        return "explicit_http_failure"
    return None


def _latest_content(
    history: tuple[dict[str, Any], ...],
    *,
    role: str,
    skip_first: bool = False,
) -> str:
    candidates = history[1:] if skip_first else history
    for item in reversed(candidates):
        if item.get("role") == role:
            return str(item.get("content", ""))
    return ""


def _promotion_coin(public_input_sha256: str) -> bool:
    value = int(public_input_sha256[:16], 16) / float(16**16 - 1)
    return value > (1.0 - AGSERVE_UPDATE_PROBABILITY)


def _unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"agserve_{label}_must_be_number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"agserve_{label}_outside_unit_interval")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"agserve_{label}_must_be_int")
    if value <= 0:
        raise ValueError(f"agserve_{label}_must_be_positive")
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
    "AGSERVE_QAC_METHOD",
    "AGSERVE_UPSTREAM_REVISION",
    "AgServeArtifact",
    "AgServeDecision",
    "AgServeQACController",
    "TransformerAgServeJudge",
    "inspect_agserve_artifact",
]

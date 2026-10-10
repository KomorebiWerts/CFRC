"""Provider-neutral AutoMix adapter pinned to the published implementation.

Adapted for PAPC from AutoMix revision
``531af3ee3c4e4f4a53da1ffbc91dec4d6bb73781`` (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    AUTOMIX,
    CHEAP_MODEL,
    STRONG_MODEL,
    AutoMixRouterView,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)
from papc.baselines.router_runtime import AutoMixVerification

AUTOMIX_UPSTREAM_REVISION = "531af3ee3c4e4f4a53da1ffbc91dec4d6bb73781"

VERIFIER_SAMPLES = 32
VERIFIER_TEMPERATURE = 1.0
VERIFIER_STOP = "---"
VERIFIER_MAX_OUTPUT_TOKENS = 250
VERIFIER_MAX_PROMPT_TOKENS = 3950
VERIFIER_PROVIDER_BATCH_SIZE = 32
VERIFIER_TOKENIZER_ID = "meta-llama/Llama-2-13b-hf"
VERIFIER_TOKENIZER_SOURCE_REPO = "hf-internal-testing/llama-tokenizer"
VERIFIER_TOKENIZER_SOURCE_REVISION = (
    "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
)
VERIFIER_TOKENIZER_FILE_SHA256S = {
    "special_tokens_map.json": (
        "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531"
    ),
    "tokenizer.json": (
        "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
    ),
    "tokenizer.model": (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    ),
    "tokenizer_config.json": (
        "c43a787e4eb83e0167f36aff633ebe8523f073e98d684e05f189a2a4e784293d"
    ),
}
VERIFIER_TOKENIZER_BUNDLE_SHA256 = hashlib.sha256(
    json.dumps(
        VERIFIER_TOKENIZER_FILE_SHA256S,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
POMDP_NUM_BINS = 8
AGENT_CONTEXT_SERIALIZER_PROTOCOL = (
    "papc.automix.public_episode.canonical_json.v1"
)
AGENT_CONTEXT_SERIALIZER_SHA256 = hashlib.sha256(
    AGENT_CONTEXT_SERIALIZER_PROTOCOL.encode("utf-8")
).hexdigest()

POMDPFamily = Literal[
    "pomdp_simple",
    "greedy_pomdp",
    "double_threshold",
    "threshold",
]
POMDP_FAMILIES: tuple[POMDPFamily, ...] = (
    "pomdp_simple",
    "greedy_pomdp",
    "double_threshold",
    "threshold",
)
_FAMILY_INDEX = {name: index for index, name in enumerate(POMDP_FAMILIES)}

VERIFIER_PROMPT_TEMPLATE = """Context: The manuscript, discovered in 1980 in a dusty attic, turned out to be a lost work of Shakespeare.

Question: Whose lost work was discovered in a dusty attic in 1980?

AI Generated Answer: Shakespeare

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifically mentions that a lost work of Shakespeare was discovered in 1980 in a dusty attic.

Verification Decision: The AI generated answer is Correct.

---

Context: The celestial event, known as the Pink Moon, is unique to the month of April and has cultural significance in many indigenous tribes.

Question: In which month does the celestial event, the Pink Moon, occur?

AI Generated Answer: July

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context clearly states that the Pink Moon is unique to the month of April.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The Mona Lisa, housed in the Louvre Museum, is believed to be a portrait of Lisa Gherardini, painted by Leonardo da Vinci in the early 16th century.

Question: Who is believed to have painted the Mona Lisa in the early 16th century?

AI Generated Answer: Vincent van Gogh

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifies that the Mona Lisa was painted by Leonardo da Vinci in the early 16th century.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The planet Kepler-442b, located 1,100 light-years away, is one of the most Earth-like planets ever discovered, having a similar size and orbiting within its star's habitable zone.

Question: How far away is the planet Kepler-442b?

AI Generated Answer: 1,100 light-years

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context states that Kepler-442b is located 1,100 light-years away.

Verification Decision: The AI generated answer is Correct.

---

Context: {context}

Question: {question}

AI Generated Answer: {generated_answer}

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation:"""

VERIFIER_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    VERIFIER_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


class VerifierTokenizer(Protocol):
    """Minimal local tokenizer interface needed by the upstream truncation rule."""

    def tokenize(self, text: str) -> Sequence[Any]: ...

    def convert_tokens_to_string(self, tokens: Sequence[Any]) -> str: ...


@dataclass(frozen=True)
class AutoMixVerifierRequest:
    """A generation request for the caller to execute and receipt."""

    prompt: str
    sample_count: int = VERIFIER_SAMPLES
    temperature: float = VERIFIER_TEMPERATURE
    stop: str = VERIFIER_STOP
    max_output_tokens: int = VERIFIER_MAX_OUTPUT_TOKENS
    max_provider_batch_size: int = VERIFIER_PROVIDER_BATCH_SIZE
    model_role: Literal["cheap"] = "cheap"
    tokenizer_id: str = VERIFIER_TOKENIZER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("automix_verifier_prompt_empty")
        if self.sample_count != VERIFIER_SAMPLES:
            raise ValueError("automix_verifier_sample_count_not_published")
        if self.temperature != VERIFIER_TEMPERATURE:
            raise ValueError("automix_verifier_temperature_not_published")
        if self.stop != VERIFIER_STOP:
            raise ValueError("automix_verifier_stop_not_published")
        if self.max_output_tokens != VERIFIER_MAX_OUTPUT_TOKENS:
            raise ValueError("automix_verifier_max_tokens_not_published")
        if self.max_provider_batch_size != VERIFIER_PROVIDER_BATCH_SIZE:
            raise ValueError("automix_verifier_batch_size_not_published")
        if self.model_role != "cheap":
            raise ValueError("automix_verifier_role_not_self_verification")
        if self.tokenizer_id != VERIFIER_TOKENIZER_ID:
            raise ValueError("automix_verifier_tokenizer_not_published")

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def generation_config(self) -> dict[str, int | float | str]:
        return {
            "n": self.sample_count,
            "temperature": self.temperature,
            "stop": self.stop,
            "max_tokens": self.max_output_tokens,
            "max_provider_batch_size": self.max_provider_batch_size,
        }


@dataclass(frozen=True)
class AutoMixVerificationScore:
    confidence: float
    sample_count: int
    valid_count: int
    correct_count: int

    def __post_init__(self) -> None:
        if self.sample_count != VERIFIER_SAMPLES:
            raise ValueError("automix_verification_sample_count_not_published")
        if not 0 <= self.correct_count <= self.valid_count <= self.sample_count:
            raise ValueError("automix_verification_counts_invalid")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("automix_verification_confidence_invalid")


@dataclass(frozen=True)
class AutoMixFitExample:
    example_sha256: str
    cheap_quality: float
    strong_quality: float
    verification_confidence: float

    def __post_init__(self) -> None:
        _required_sha256(self.example_sha256, "automix_fit_example")
        object.__setattr__(
            self,
            "cheap_quality",
            _unit_float(self.cheap_quality, "automix_cheap_quality"),
        )
        object.__setattr__(
            self,
            "strong_quality",
            _unit_float(self.strong_quality, "automix_strong_quality"),
        )
        object.__setattr__(
            self,
            "verification_confidence",
            _unit_float(
                self.verification_confidence,
                "automix_verification_confidence",
            ),
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "cheap_quality": self.cheap_quality,
            "strong_quality": self.strong_quality,
            "verification_confidence": self.verification_confidence,
        }


@dataclass(frozen=True)
class AutoMixDevExample:
    example_sha256: str
    cheap_quality: float
    strong_quality: float
    verification_confidence: float
    cheap_episode_cost_usd: float
    strong_episode_cost_usd: float
    verifier_cost_usd: float

    def __post_init__(self) -> None:
        _required_sha256(self.example_sha256, "automix_dev_example")
        for field_name in (
            "cheap_quality",
            "strong_quality",
            "verification_confidence",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        for field_name in (
            "cheap_episode_cost_usd",
            "strong_episode_cost_usd",
            "verifier_cost_usd",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(getattr(self, field_name), field_name),
            )
        if self.verifier_cost_usd <= 0.0:
            raise ValueError("automix_verifier_cost_must_be_positive")

    def public_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "cheap_quality": self.cheap_quality,
            "strong_quality": self.strong_quality,
            "verification_confidence": self.verification_confidence,
            "cheap_episode_cost_usd": self.cheap_episode_cost_usd,
            "strong_episode_cost_usd": self.strong_episode_cost_usd,
            "verifier_cost_usd": self.verifier_cost_usd,
        }


POMDPParameter = float | tuple[float, float] | tuple[int, ...]


@dataclass(frozen=True)
class AutoMixFitCandidate:
    family: POMDPFamily
    parameter: POMDPParameter
    fit_ibc_lift: float

    def __post_init__(self) -> None:
        if self.family not in POMDP_FAMILIES:
            raise ValueError("automix_fit_candidate_family_invalid")
        _validate_parameter(self.family, self.parameter)
        if not math.isfinite(self.fit_ibc_lift):
            raise ValueError("automix_fit_candidate_ibc_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "family_index": _FAMILY_INDEX[self.family],
            "parameter": self.parameter,
            "fit_ibc_lift": self.fit_ibc_lift,
        }


@dataclass(frozen=True)
class AutoMixPOMDPCandidateSet:
    fit_registry_sha256: str
    fit_membership_sha256: str
    fit_examples_sha256: str
    fit_rows: int
    candidates: tuple[AutoMixFitCandidate, ...]
    cheap_cost: float
    strong_cost: float
    verifier_cost: float
    cheap_quality_p10: float
    strong_quality_p10: float

    def __post_init__(self) -> None:
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_examples_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if (
            isinstance(self.fit_rows, bool)
            or not isinstance(self.fit_rows, int)
            or self.fit_rows <= 0
        ):
            raise ValueError("automix_candidate_set_fit_rows_invalid")
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(item, AutoMixFitCandidate) for item in candidates
        ):
            raise ValueError("automix_candidate_set_empty")
        if len(
            {
                json.dumps(
                    (item.family, item.parameter),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in candidates
            }
        ) != len(candidates):
            raise ValueError("automix_candidate_set_duplicate")
        if not any(
            item.family == "threshold" and item.parameter == 1.0
            for item in candidates
        ):
            raise ValueError("automix_candidate_set_all_strong_missing")
        object.__setattr__(self, "candidates", candidates)

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "fit_registry_sha256": self.fit_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "fit_examples_sha256": self.fit_examples_sha256,
            "fit_rows": self.fit_rows,
            "candidates": [item.public_record() for item in self.candidates],
            "cheap_cost": self.cheap_cost,
            "strong_cost": self.strong_cost,
            "verifier_cost": self.verifier_cost,
            "cheap_quality_p10": self.cheap_quality_p10,
            "strong_quality_p10": self.strong_quality_p10,
        }


@dataclass(frozen=True)
class AutoMixDevCandidateReceipt:
    family: POMDPFamily
    parameter: POMDPParameter
    config_sha256: str
    fit_ibc_lift: float
    dev_quality: float
    provider_api_cost_usd: float
    strong_count: int

    def __post_init__(self) -> None:
        AutoMixFitCandidate(
            family=self.family,
            parameter=self.parameter,
            fit_ibc_lift=self.fit_ibc_lift,
        )
        if self.config_sha256 != _digest(
            {"family": self.family, "parameter": self.parameter}
        ):
            raise ValueError("automix_dev_config_sha256_mismatch")
        object.__setattr__(
            self,
            "dev_quality",
            _unit_float(self.dev_quality, "automix_dev_quality"),
        )
        object.__setattr__(
            self,
            "provider_api_cost_usd",
            _nonnegative_float(
                self.provider_api_cost_usd,
                "automix_dev_provider_api_cost",
            ),
        )
        if (
            isinstance(self.strong_count, bool)
            or not isinstance(self.strong_count, int)
            or self.strong_count < 0
        ):
            raise ValueError("automix_dev_strong_count_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "family_index": _FAMILY_INDEX[self.family],
            "parameter": self.parameter,
            "config_sha256": self.config_sha256,
            "fit_ibc_lift": self.fit_ibc_lift,
            "dev_quality": self.dev_quality,
            "provider_api_cost_usd": self.provider_api_cost_usd,
            "strong_count": self.strong_count,
        }


@dataclass(frozen=True)
class AutoMixPOMDPPolicy:
    """Frozen result of the published AutoMix candidate search."""

    family: POMDPFamily
    parameter: POMDPParameter
    ibc_lift: float
    fit_rows: int
    candidate_count: int
    cheap_cost: float
    strong_cost: float
    verifier_cost: float
    cheap_quality_p10: float
    strong_quality_p10: float
    fit_candidate_set_sha256: str
    dev_registry_sha256: str
    dev_membership_sha256: str
    dev_examples_sha256: str
    strong_full_dev_anchor: float
    dev_quality: float
    dev_cost_usd: float
    dev_strong_count: int
    candidate_receipts: tuple[AutoMixDevCandidateReceipt, ...]
    selection_rule: str = (
        "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
    )
    deterministic_tie_break: str = "published_family_order_then_canonical_parameter"
    upstream_revision: str = AUTOMIX_UPSTREAM_REVISION
    verifier_samples: int = VERIFIER_SAMPLES
    pomdp_num_bins: int = POMDP_NUM_BINS
    prompt_template_sha256: str = VERIFIER_PROMPT_TEMPLATE_SHA256

    def __post_init__(self) -> None:
        if self.family not in POMDP_FAMILIES:
            raise ValueError("automix_pomdp_family_invalid")
        if not math.isfinite(self.ibc_lift):
            raise ValueError("automix_pomdp_ibc_lift_invalid")
        if self.fit_rows <= 0 or self.candidate_count <= 0:
            raise ValueError("automix_pomdp_fit_support_invalid")
        for field_name in (
            "fit_candidate_set_sha256",
            "dev_registry_sha256",
            "dev_membership_sha256",
            "dev_examples_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        for field_name in ("strong_full_dev_anchor", "dev_quality"):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "dev_cost_usd",
            _nonnegative_float(self.dev_cost_usd, "automix_dev_cost"),
        )
        if (
            isinstance(self.dev_strong_count, bool)
            or not isinstance(self.dev_strong_count, int)
            or self.dev_strong_count < 0
        ):
            raise ValueError("automix_dev_strong_count_invalid")
        receipts = tuple(self.candidate_receipts)
        if len(receipts) != self.candidate_count or any(
            not isinstance(item, AutoMixDevCandidateReceipt)
            for item in receipts
        ):
            raise ValueError("automix_dev_candidate_receipts_invalid")
        if not any(
            item.family == "threshold"
            and item.parameter == 1.0
            for item in receipts
        ):
            raise ValueError("automix_dev_all_strong_candidate_required")
        object.__setattr__(self, "candidate_receipts", receipts)
        if self.selection_rule != (
            "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
        ):
            raise ValueError("automix_dev_selection_rule_invalid")
        if self.upstream_revision != AUTOMIX_UPSTREAM_REVISION:
            raise ValueError("automix_upstream_revision_mismatch")
        if self.verifier_samples != VERIFIER_SAMPLES:
            raise ValueError("automix_policy_verifier_samples_mismatch")
        if self.pomdp_num_bins != POMDP_NUM_BINS:
            raise ValueError("automix_policy_num_bins_mismatch")
        if self.prompt_template_sha256 != VERIFIER_PROMPT_TEMPLATE_SHA256:
            raise ValueError("automix_policy_prompt_mismatch")
        _validate_parameter(self.family, self.parameter)

    @property
    def upstream_best_param(self) -> tuple[POMDPParameter, int]:
        return self.parameter, _FAMILY_INDEX[self.family]

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.public_record(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "family_index": _FAMILY_INDEX[self.family],
            "parameter": self.parameter,
            "ibc_lift": self.ibc_lift,
            "fit_rows": self.fit_rows,
            "candidate_count": self.candidate_count,
            "cheap_cost": self.cheap_cost,
            "strong_cost": self.strong_cost,
            "verifier_cost": self.verifier_cost,
            "cheap_quality_p10": self.cheap_quality_p10,
            "strong_quality_p10": self.strong_quality_p10,
            "fit_candidate_set_sha256": self.fit_candidate_set_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "dev_examples_sha256": self.dev_examples_sha256,
            "strong_full_dev_anchor": self.strong_full_dev_anchor,
            "dev_quality": self.dev_quality,
            "dev_cost_usd": self.dev_cost_usd,
            "dev_strong_count": self.dev_strong_count,
            "candidate_receipts": [
                item.public_record() for item in self.candidate_receipts
            ],
            "selection_rule": self.selection_rule,
            "deterministic_tie_break": self.deterministic_tie_break,
            "upstream_revision": self.upstream_revision,
            "verifier_samples": self.verifier_samples,
            "pomdp_num_bins": self.pomdp_num_bins,
            "prompt_template_sha256": self.prompt_template_sha256,
        }

    def should_retry(self, verification_confidence: float) -> bool:
        confidence = _unit_float(
            verification_confidence,
            "automix_verification_confidence",
        )
        return _retry_for(self.family, self.parameter, confidence)

    def __call__(self, verification_confidence: float) -> bool:
        return self.should_retry(verification_confidence)


@dataclass(frozen=True)
class AutoMixRuntimeArtifact:
    """Split- and serializer-bound policy used by the episode driver."""

    surface: str
    policy: AutoMixPOMDPPolicy
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_membership_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    cost_rule_sha256: str
    verifier_tokenizer_sha256: str = VERIFIER_TOKENIZER_BUNDLE_SHA256
    context_serializer_sha256: str = AGENT_CONTEXT_SERIALIZER_SHA256
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL
    upstream_revision: str = AUTOMIX_UPSTREAM_REVISION

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("automix_runtime_surface_invalid")
        if not isinstance(self.policy, AutoMixPOMDPPolicy):
            raise TypeError("automix_runtime_policy_invalid")
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_membership_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "cost_rule_sha256",
            "verifier_tokenizer_sha256",
            "context_serializer_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if len(
            {
                self.fit_registry_sha256,
                self.dev_registry_sha256,
                self.test_registry_sha256,
            }
        ) != 3:
            raise ValueError("automix_runtime_split_digest_reused")
        if self.context_serializer_sha256 != AGENT_CONTEXT_SERIALIZER_SHA256:
            raise ValueError("automix_runtime_serializer_not_frozen")
        if self.verifier_tokenizer_sha256 != (
            VERIFIER_TOKENIZER_BUNDLE_SHA256
        ):
            raise ValueError("automix_runtime_tokenizer_not_frozen")
        if (
            self.policy.dev_registry_sha256 != self.dev_registry_sha256
            or self.policy.dev_membership_sha256
            != self.dev_membership_sha256
        ):
            raise ValueError("automix_runtime_dev_selection_mismatch")
        if self.cheap_model != CHEAP_MODEL or self.strong_model != STRONG_MODEL:
            raise ValueError("automix_runtime_models_not_frozen")
        if self.upstream_revision != AUTOMIX_UPSTREAM_REVISION:
            raise ValueError("automix_runtime_revision_not_frozen")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.public_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": "papc.automix.runtime_artifact.v2",
            "surface": self.surface,
            "policy": self.policy.public_record(),
            "policy_sha256": self.policy.sha256,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "split_binding_sha256": self.split_binding_sha256,
            "cost_rule_sha256": self.cost_rule_sha256,
            "verifier_tokenizer_id": VERIFIER_TOKENIZER_ID,
            "verifier_tokenizer_source_repo": (
                VERIFIER_TOKENIZER_SOURCE_REPO
            ),
            "verifier_tokenizer_source_revision": (
                VERIFIER_TOKENIZER_SOURCE_REVISION
            ),
            "verifier_tokenizer_file_sha256s": dict(
                sorted(VERIFIER_TOKENIZER_FILE_SHA256S.items())
            ),
            "verifier_tokenizer_sha256": self.verifier_tokenizer_sha256,
            "context_serializer_protocol": AGENT_CONTEXT_SERIALIZER_PROTOCOL,
            "context_serializer_sha256": self.context_serializer_sha256,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "upstream_revision": self.upstream_revision,
        }

    def router_binding(self) -> RouterArtifactBinding:
        return RouterArtifactBinding(
            method=AUTOMIX,
            surface=self.surface,
            upstream_revision=self.upstream_revision,
            artifact_sha256=self.sha256,
            fit_registry_sha256=self.fit_registry_sha256,
            dev_registry_sha256=self.dev_registry_sha256,
            test_registry_sha256=self.test_registry_sha256,
            calibration_sha256=self.policy.sha256,
        )


def validate_automix_verifier_tokenizer_root(
    root: str | Path,
) -> dict[str, Any]:
    """Validate the exact local Llama-2 tokenizer bytes without network use."""

    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("automix_verifier_tokenizer_root_missing")
    observed: dict[str, str] = {}
    for relative, expected in sorted(
        VERIFIER_TOKENIZER_FILE_SHA256S.items()
    ):
        path = resolved / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"automix_verifier_tokenizer_file_missing:{relative}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"automix_verifier_tokenizer_file_mismatch:{relative}"
            )
        observed[relative] = digest
    config = json.loads(
        (resolved / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(config, Mapping)
        or config.get("tokenizer_class") != "LlamaTokenizer"
    ):
        raise ValueError("automix_verifier_tokenizer_config_invalid")
    return {
        "tokenizer_id": VERIFIER_TOKENIZER_ID,
        "source_repo": VERIFIER_TOKENIZER_SOURCE_REPO,
        "source_revision": VERIFIER_TOKENIZER_SOURCE_REVISION,
        "file_sha256s": observed,
        "bundle_sha256": VERIFIER_TOKENIZER_BUNDLE_SHA256,
        "local_files_only": True,
    }


def bind_automix_runtime_artifact(
    *,
    surface: str,
    candidate_set: AutoMixPOMDPCandidateSet,
    policy: AutoMixPOMDPPolicy,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    cost_rule_sha256: str,
) -> AutoMixRuntimeArtifact:
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    if not isinstance(candidate_set, AutoMixPOMDPCandidateSet):
        raise TypeError("automix_candidate_set_invalid")
    if not isinstance(policy, AutoMixPOMDPPolicy):
        raise TypeError("automix_policy_invalid")
    if (
        candidate_set.fit_registry_sha256 != fit_manifest.registry_sha256
        or candidate_set.fit_membership_sha256
        != _membership_sha256(fit_manifest, fit_manifest.example_sha256s)
    ):
        raise ValueError("automix_fit_candidate_registry_mismatch")
    if policy.fit_candidate_set_sha256 != candidate_set.sha256:
        raise ValueError("automix_policy_candidate_set_mismatch")
    if (
        policy.dev_registry_sha256 != dev_manifest.registry_sha256
        or policy.dev_membership_sha256
        != _membership_sha256(dev_manifest, dev_manifest.example_sha256s)
    ):
        raise ValueError("automix_policy_dev_registry_mismatch")
    return AutoMixRuntimeArtifact(
        surface=surface,
        policy=policy,
        fit_registry_sha256=fit_manifest.registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_membership_sha256=candidate_set.fit_membership_sha256,
        dev_membership_sha256=policy.dev_membership_sha256,
        split_binding_sha256=_digest(
            {
                "fit": fit_manifest.__dict__,
                "dev": dev_manifest.__dict__,
                "test": test_manifest.__dict__,
            }
        ),
        cost_rule_sha256=cost_rule_sha256,
    )


@dataclass(frozen=True)
class AutoMixVerifierAdapter:
    """Build and consume the published self-verification request."""

    artifact: AutoMixRuntimeArtifact
    tokenizer: VerifierTokenizer

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AutoMixRuntimeArtifact):
            raise TypeError("automix_verifier_artifact_invalid")
        if self.tokenizer is None:
            raise TypeError("automix_verifier_tokenizer_required")

    def build_request(self, view: AutoMixRouterView) -> AutoMixVerifierRequest:
        if not isinstance(view, AutoMixRouterView):
            raise TypeError("automix_router_view_required")
        return build_automix_verifier_request(
            context=serialize_automix_agent_context(view),
            question=view.original_request,
            generated_answer=view.cheap_generated_answer,
            tokenizer=self.tokenizer,
        )

    def consume_outputs(
        self,
        *,
        outputs: Sequence[str],
        receipts: Sequence[CallReceipt],
    ) -> AutoMixVerification:
        score = parse_automix_verifier_outputs(outputs)
        resolved_receipts = tuple(receipts)
        if not resolved_receipts:
            raise ValueError("automix_verifier_receipts_empty")
        if any(
            receipt.role != "verifier"
            or receipt.model != self.artifact.cheap_model
            for receipt in resolved_receipts
        ):
            raise ValueError("automix_verifier_receipt_binding_invalid")
        if sum(receipt.provider_calls for receipt in resolved_receipts) <= 0:
            raise ValueError("automix_verifier_provider_call_missing")
        if any(
            receipt.input_tokens is None or receipt.output_tokens is None
            for receipt in resolved_receipts
        ):
            raise ValueError("automix_verifier_usage_missing")
        return AutoMixVerification(
            confidence=score.confidence,
            sample_count=score.sample_count,
            receipts=resolved_receipts,
        )

    def should_retry(self, confidence: float) -> bool:
        return self.artifact.policy.should_retry(confidence)


def serialize_automix_agent_context(view: AutoMixRouterView) -> str:
    if not isinstance(view, AutoMixRouterView):
        raise TypeError("automix_router_view_required")
    return json.dumps(
        {
            "protocol": AGENT_CONTEXT_SERIALIZER_PROTOCOL,
            "public_episode": view.cheap_public_episode,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def format_automix_verifier_prompt(
    *,
    context: str,
    question: str,
    generated_answer: str,
) -> str:
    for field_name, value in (
        ("context", context),
        ("question", question),
        ("generated_answer", generated_answer),
    ):
        if not isinstance(value, str):
            raise TypeError(f"automix_verifier_{field_name}_must_be_text")
    return VERIFIER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        generated_answer=generated_answer,
    )


def build_automix_verifier_request(
    *,
    context: str,
    question: str,
    generated_answer: str,
    tokenizer: VerifierTokenizer,
) -> AutoMixVerifierRequest:
    """Build the prompt and apply the upstream left-truncation rule locally."""

    if tokenizer is None:
        raise TypeError("automix_verifier_tokenizer_required")
    prompt = format_automix_verifier_prompt(
        context=context,
        question=question,
        generated_answer=generated_answer,
    )
    tokens = tuple(tokenizer.tokenize(prompt))
    if len(tokens) > VERIFIER_MAX_PROMPT_TOKENS:
        prompt = tokenizer.convert_tokens_to_string(
            tokens[-VERIFIER_MAX_PROMPT_TOKENS:]
        )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("automix_verifier_truncated_prompt_invalid")
    return AutoMixVerifierRequest(prompt=prompt)


def parse_automix_verifier_outputs(
    outputs: Sequence[str],
) -> AutoMixVerificationScore:
    """Parse the 32 sampled decisions exactly as the published notebook."""

    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise TypeError("automix_verifier_outputs_must_be_sequence")
    resolved = tuple(outputs)
    if len(resolved) != VERIFIER_SAMPLES:
        raise ValueError("automix_verifier_output_count_not_published")
    if any(not isinstance(item, str) for item in resolved):
        raise TypeError("automix_verifier_output_must_be_text")

    lowered = tuple(item.lower() for item in resolved)
    valid_count = sum(
        "the ai generated answer is" in item for item in lowered
    )
    correct_count = sum(
        "the ai generated answer is correct" in item for item in lowered
    )
    confidence = correct_count / valid_count if valid_count else 0.0
    return AutoMixVerificationScore(
        confidence=confidence,
        sample_count=len(resolved),
        valid_count=valid_count,
        correct_count=correct_count,
    )


def fit_automix_pomdp(
    examples: Sequence[AutoMixFitExample],
    *,
    fit_manifest: SplitManifest,
    cheap_cost: float,
    strong_cost: float,
    verifier_cost: float,
) -> AutoMixPOMDPCandidateSet:
    """Generate the published fit-only four-family candidate union."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("automix_fit_examples_must_be_sequence")
    supplied = tuple(examples)
    if not supplied or any(not isinstance(item, AutoMixFitExample) for item in supplied):
        raise ValueError("automix_fit_examples_invalid")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("automix_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("automix_fit_manifest_role_invalid")
    example_ids = tuple(item.example_sha256 for item in supplied)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("automix_fit_example_duplicate")
    _require_exact_membership(
        fit_manifest,
        example_ids,
        field_name="automix_fit",
    )
    by_id = {item.example_sha256: item for item in supplied}
    resolved = tuple(by_id[value] for value in fit_manifest.example_sha256s)

    cheap_cost = _nonnegative_float(cheap_cost, "automix_cheap_cost")
    strong_cost = _nonnegative_float(strong_cost, "automix_strong_cost")
    verifier_cost = _nonnegative_float(verifier_cost, "automix_verifier_cost")
    if strong_cost <= cheap_cost:
        raise ValueError("automix_strong_cost_must_exceed_cheap")
    if verifier_cost <= 0.0:
        raise ValueError("automix_verifier_cost_must_be_positive")

    cheap = np.asarray([item.cheap_quality for item in resolved], dtype=float)
    strong = np.asarray([item.strong_quality for item in resolved], dtype=float)
    confidence = np.asarray(
        [item.verification_confidence for item in resolved],
        dtype=float,
    )
    cheap_mean = float(cheap.mean())
    strong_mean = float(strong.mean())
    if strong_mean <= cheap_mean:
        raise ValueError("automix_fit_strong_not_better")

    cheap_p10 = float(np.quantile(cheap, 0.10))
    strong_p10 = float(np.quantile(strong, 0.10))
    categories = _categorize(
        cheap,
        strong,
        cheap_p10=cheap_p10,
        strong_p10=strong_p10,
    )
    candidates = _published_candidates(
        cheap=cheap,
        strong=strong,
        confidence=confidence,
        categories=categories,
    )

    base_slope = (strong_mean - cheap_mean) / (strong_cost - cheap_cost)
    scored: list[AutoMixFitCandidate] = []
    for family, parameter in candidates:
        retry = np.asarray(
            [
                _retry_for(family, parameter, float(value))
                for value in confidence
            ],
            dtype=bool,
        )
        avg_performance = float(np.where(retry, strong, cheap).mean())
        avg_cost = (
            (
                int((~retry).sum()) * cheap_cost
                + int(retry.sum()) * (cheap_cost + strong_cost)
            )
            / len(resolved)
            + verifier_cost
        )
        denominator = avg_cost - cheap_cost
        if denominator == 0.0:
            continue
        automix_slope = (avg_performance - cheap_mean) / denominator
        ibc_lift = (automix_slope - base_slope) / base_slope
        if math.isfinite(ibc_lift):
            scored.append(
                AutoMixFitCandidate(
                    family=family,
                    parameter=parameter,
                    fit_ibc_lift=ibc_lift,
                )
            )

    if not scored:
        raise ValueError("automix_no_finite_candidate")
    return AutoMixPOMDPCandidateSet(
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=_membership_sha256(
            fit_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        fit_examples_sha256=_digest(
            [item.public_record() for item in resolved]
        ),
        fit_rows=len(resolved),
        candidates=tuple(scored),
        cheap_cost=cheap_cost,
        strong_cost=strong_cost,
        verifier_cost=verifier_cost,
        cheap_quality_p10=cheap_p10,
        strong_quality_p10=strong_p10,
    )


def select_automix_pomdp(
    candidate_set: AutoMixPOMDPCandidateSet,
    examples: Sequence[AutoMixDevExample],
    *,
    dev_manifest: SplitManifest,
) -> AutoMixPOMDPPolicy:
    """Select among fit-generated candidates on held-out dev only."""

    if not isinstance(candidate_set, AutoMixPOMDPCandidateSet):
        raise TypeError("automix_candidate_set_invalid")
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("automix_dev_examples_must_be_sequence")
    supplied = tuple(examples)
    if not supplied or any(
        not isinstance(item, AutoMixDevExample) for item in supplied
    ):
        raise ValueError("automix_dev_examples_invalid")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("automix_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("automix_dev_manifest_role_invalid")
    example_ids = tuple(item.example_sha256 for item in supplied)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("automix_dev_example_duplicate")
    _require_exact_membership(
        dev_manifest,
        example_ids,
        field_name="automix_dev",
    )
    by_id = {item.example_sha256: item for item in supplied}
    resolved = tuple(by_id[value] for value in dev_manifest.example_sha256s)
    quality_floor = (
        sum(item.strong_quality for item in resolved) / len(resolved)
    )
    receipts: list[AutoMixDevCandidateReceipt] = []
    for candidate in candidate_set.candidates:
        retry = tuple(
            _retry_for(
                candidate.family,
                candidate.parameter,
                item.verification_confidence,
            )
            for item in resolved
        )
        receipts.append(
            AutoMixDevCandidateReceipt(
                family=candidate.family,
                parameter=candidate.parameter,
                config_sha256=_digest(
                    {
                        "family": candidate.family,
                        "parameter": candidate.parameter,
                    }
                ),
                fit_ibc_lift=candidate.fit_ibc_lift,
                dev_quality=sum(
                    item.strong_quality if use_strong else item.cheap_quality
                    for item, use_strong in zip(resolved, retry)
                )
                / len(resolved),
                provider_api_cost_usd=sum(
                    item.cheap_episode_cost_usd
                    + item.verifier_cost_usd
                    + (item.strong_episode_cost_usd if use_strong else 0.0)
                    for item, use_strong in zip(resolved, retry)
                ),
                strong_count=sum(retry),
            )
        )
    feasible = tuple(
        item
        for item in receipts
        if item.dev_quality + 1e-12 >= quality_floor
    )
    if not feasible:
        raise ValueError("automix_no_dev_candidate_meets_quality_floor")
    selected = min(
        feasible,
        key=lambda item: (
            item.provider_api_cost_usd,
            item.strong_count,
            item.config_sha256,
        ),
    )
    return AutoMixPOMDPPolicy(
        family=selected.family,
        parameter=selected.parameter,
        ibc_lift=selected.fit_ibc_lift,
        fit_rows=candidate_set.fit_rows,
        candidate_count=len(candidate_set.candidates),
        cheap_cost=candidate_set.cheap_cost,
        strong_cost=candidate_set.strong_cost,
        verifier_cost=candidate_set.verifier_cost,
        cheap_quality_p10=candidate_set.cheap_quality_p10,
        strong_quality_p10=candidate_set.strong_quality_p10,
        fit_candidate_set_sha256=candidate_set.sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        dev_examples_sha256=_digest(
            [item.public_record() for item in resolved]
        ),
        strong_full_dev_anchor=quality_floor,
        dev_quality=selected.dev_quality,
        dev_cost_usd=selected.provider_api_cost_usd,
        dev_strong_count=selected.strong_count,
        candidate_receipts=tuple(receipts),
    )


def _published_candidates(
    *,
    cheap: np.ndarray,
    strong: np.ndarray,
    confidence: np.ndarray,
    categories: np.ndarray,
) -> tuple[tuple[POMDPFamily, POMDPParameter], ...]:
    simple = _pomdp_simple_parameters(confidence, categories)
    greedy = _greedy_pomdp_parameters(cheap, strong, confidence)
    gap = 1.0 / POMDP_NUM_BINS
    double = tuple(
        (i * gap, j * gap)
        for i in range(POMDP_NUM_BINS)
        for j in range(i + 1, POMDP_NUM_BINS)
    )
    threshold = tuple(i * gap for i in range(POMDP_NUM_BINS + 1))
    return (
        *(("pomdp_simple", parameter) for parameter in simple),
        *(("greedy_pomdp", parameter) for parameter in greedy),
        *(("double_threshold", parameter) for parameter in double),
        *(("threshold", parameter) for parameter in threshold),
    )


def _pomdp_simple_parameters(
    confidence: np.ndarray,
    categories: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    gap = 1.0 / POMDP_NUM_BINS
    category_names = ("NEEDY", "GOOD", "HOPELESS")
    observation_probabilities = np.zeros((POMDP_NUM_BINS + 1, 3))
    for index in range(POMDP_NUM_BINS + 1):
        probability = index * gap
        mask = (
            (confidence - probability < gap / 2)
            & (confidence - probability > -gap / 2)
        )
        denominator = int(mask.sum())
        if denominator:
            observation_probabilities[index] = [
                int(((categories == name) & mask).sum()) / denominator
                for name in category_names
            ]

    action_sequences: set[tuple[int, ...]] = set()
    for reward in range(5000):
        action_array = np.asarray(
            (
                (-reward, 0, -reward - 1),
                (-100, -100, -reward - 100),
            ),
            dtype=float,
        )
        actions = tuple(
            int(np.argmax((probabilities * action_array).sum(axis=1)))
            for probabilities in observation_probabilities
        )
        action_sequences.add(actions)
    return tuple(sorted(action_sequences))


def _greedy_pomdp_parameters(
    cheap: np.ndarray,
    strong: np.ndarray,
    confidence: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    gap = 1.0 / POMDP_NUM_BINS
    means: list[float] = []
    weights: list[float] = []
    for index in range(POMDP_NUM_BINS + 1):
        mask = np.floor_divide(confidence, gap) == index
        weights.append(float(mask.sum()) / len(confidence))
        means.append(float((strong[mask] - cheap[mask]).mean()) if mask.any() else 0.0)

    smoothed = _gaussian_filter_reflect(np.asarray(means), sigma=2.0)
    points = list(zip(smoothed, range(POMDP_NUM_BINS + 1), weights))
    ordered = sorted(points, key=lambda item: item[0])[::-1]
    parameters: list[tuple[int, ...]] = []
    current = [0] * (POMDP_NUM_BINS + 1)
    for _, index, _ in ordered:
        current = current.copy()
        current[index] = 1
        parameters.append(tuple(current))
    return tuple(parameters)


def _gaussian_filter_reflect(values: np.ndarray, *, sigma: float) -> np.ndarray:
    """Match scipy.ndimage.gaussian_filter's 1-D defaults without SciPy."""

    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(values.astype(float), radius, mode="symmetric")
    return np.convolve(padded, weights, mode="valid")


def _categorize(
    cheap: np.ndarray,
    strong: np.ndarray,
    *,
    cheap_p10: float,
    strong_p10: float,
) -> np.ndarray:
    categories = np.full(len(cheap), "UNDEFINED", dtype=object)
    needy = (cheap <= strong) & (cheap != strong)
    good = (cheap == strong) & (cheap != 0)
    hopeless = (cheap <= cheap_p10) & (strong <= strong_p10)
    categories[hopeless] = "HOPELESS"
    categories[good] = "GOOD"
    categories[needy] = "NEEDY"
    return categories


def _retry_for(
    family: POMDPFamily,
    parameter: POMDPParameter,
    confidence: float,
) -> bool:
    if family == "threshold":
        return confidence <= float(parameter)
    if family == "double_threshold":
        lower, upper = parameter
        return lower <= confidence <= upper
    index = _nearest_probability_index(confidence)
    if index is None:
        raise ValueError("automix_confidence_outside_pomdp_bins")
    return bool(parameter[index])


def _nearest_probability_index(confidence: float) -> int | None:
    gap = 1.0 / POMDP_NUM_BINS
    for index in range(POMDP_NUM_BINS + 1):
        probability = index * gap
        if confidence - probability < gap / 2 and confidence - probability >= -gap / 2:
            return index
    return None


def _validate_parameter(
    family: POMDPFamily,
    parameter: POMDPParameter,
) -> None:
    if family == "threshold":
        _unit_float(parameter, "automix_threshold")
        return
    if family == "double_threshold":
        if not isinstance(parameter, tuple) or len(parameter) != 2:
            raise ValueError("automix_double_threshold_invalid")
        lower = _unit_float(parameter[0], "automix_double_threshold_lower")
        upper = _unit_float(parameter[1], "automix_double_threshold_upper")
        if lower >= upper:
            raise ValueError("automix_double_threshold_order_invalid")
        return
    if (
        not isinstance(parameter, tuple)
        or len(parameter) != POMDP_NUM_BINS + 1
        or any(value not in {0, 1} for value in parameter)
    ):
        raise ValueError("automix_pomdp_action_sequence_invalid")


def _unit_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return resolved


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_membership(
    manifest: SplitManifest,
    example_sha256s: Sequence[str],
    *,
    field_name: str,
) -> None:
    supplied = tuple(example_sha256s)
    if len(supplied) != len(set(supplied)):
        raise ValueError(f"{field_name}_example_duplicate")
    if set(supplied) != set(manifest.example_sha256s):
        raise ValueError(f"{field_name}_registry_membership_mismatch")


def _membership_sha256(
    manifest: SplitManifest,
    example_sha256s: Sequence[str],
) -> str:
    _require_exact_membership(
        manifest,
        example_sha256s,
        field_name="automix_membership",
    )
    return _digest(
        {
            "role": manifest.role,
            "registry_sha256": manifest.registry_sha256,
            "example_sha256s": manifest.example_sha256s,
        }
    )


def _required_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _nonnegative_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if resolved < 0.0:
        raise ValueError(f"{field_name}_negative")
    return resolved


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name}_not_finite")
    return resolved


__all__ = [
    "AGENT_CONTEXT_SERIALIZER_PROTOCOL",
    "AGENT_CONTEXT_SERIALIZER_SHA256",
    "AUTOMIX_UPSTREAM_REVISION",
    "AutoMixFitExample",
    "AutoMixFitCandidate",
    "AutoMixPOMDPCandidateSet",
    "AutoMixDevCandidateReceipt",
    "AutoMixDevExample",
    "AutoMixPOMDPPolicy",
    "AutoMixRuntimeArtifact",
    "AutoMixVerificationScore",
    "AutoMixVerifierAdapter",
    "AutoMixVerifierRequest",
    "POMDP_FAMILIES",
    "POMDP_NUM_BINS",
    "VERIFIER_MAX_OUTPUT_TOKENS",
    "VERIFIER_MAX_PROMPT_TOKENS",
    "VERIFIER_PROMPT_TEMPLATE",
    "VERIFIER_PROMPT_TEMPLATE_SHA256",
    "VERIFIER_PROVIDER_BATCH_SIZE",
    "VERIFIER_SAMPLES",
    "VERIFIER_STOP",
    "VERIFIER_TEMPERATURE",
    "VERIFIER_TOKENIZER_BUNDLE_SHA256",
    "VERIFIER_TOKENIZER_FILE_SHA256S",
    "VERIFIER_TOKENIZER_ID",
    "VERIFIER_TOKENIZER_SOURCE_REPO",
    "VERIFIER_TOKENIZER_SOURCE_REVISION",
    "VerifierTokenizer",
    "build_automix_verifier_request",
    "bind_automix_runtime_artifact",
    "fit_automix_pomdp",
    "format_automix_verifier_prompt",
    "parse_automix_verifier_outputs",
    "select_automix_pomdp",
    "serialize_automix_agent_context",
    "validate_automix_verifier_tokenizer_root",
]

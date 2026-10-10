"""Pinned two-model FrugalGPT adapter for protected agent baselines.

The original method is a response cascade.  PAPC keeps its cheap-first
two-model specialization explicit: the quality scorer sees only the original
request and the completed cheap agent answer, and escalation restarts the
strong agent from a pristine environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.router_contract import (
    CHEAP_MODEL,
    FRUGALGPT,
    STRONG_MODEL,
    CheapCompletionRouterView,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)

FRUGALGPT_UPSTREAM_REVISION = "2b23e6acbd4b39f71912816640dcc59e45aa4b5e"
SCORER_BASE_MODEL = "distilbert-base-uncased"
SCORER_MAX_LENGTH = 512
SCORER_TRAIN_VALIDATION_FRACTION = 0.55
SCORER_EPOCHS = 8
SCORER_TRAIN_BATCH_SIZE = 8
SCORER_EVAL_BATCH_SIZE = 64
SCORER_WARMUP_STEPS = 500
SCORER_WEIGHT_DECAY = 0.01
SCORER_SEED = 2024
SCORER_LEARNING_RATE = 5e-5
SCORER_ADAM_BETA1 = 0.9
SCORER_ADAM_BETA2 = 0.999
SCORER_ADAM_EPSILON = 1e-8
SCORER_MAX_GRAD_NORM = 1.0
SCORER_LOGGING_STEPS = 10
SCORER_OPTIMIZER = "adamw_torch"
SCORER_EARLY_STOPPING = False
SCORER_CHECKPOINT_SELECTION = "best_eval_loss"
SCORER_PREFIX = ""
CASCADE_OUTER_HOLDOUT_FRACTION = 0.01
CASCADE_DEPTH = 2
SCORER_SERIALIZER_PROTOCOL = (
    "papc.frugalgpt.query_answer.q_suffix.v2:"
    "query+' '+answer;removeprefix('');'Q:'+split('Q:')[-1]"
)
SCORER_SERIALIZER_SHA256 = hashlib.sha256(
    SCORER_SERIALIZER_PROTOCOL.encode("utf-8")
).hexdigest()
CHECKPOINT_MANIFEST_FILENAME = "papc_frugalgpt_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA = "papc.frugalgpt.checkpoint_manifest.v1"
TRAINING_EXPORT_SCHEMA = "papc.frugalgpt.training_export.v1"
UPSTREAM_SOURCE_SHA256S = {
    "src/FrugalGPT/llmcascade.py": (
        "ce86403b47f0e113c36d97c3acc28f032f23ed9faeb093c70123d594966dc32b"
    ),
    "src/FrugalGPT/scoring.py": (
        "9b2aa61ad85d934220807505ed9c3190f6f4afb443859fe6ff9a734bb3d93b5d"
    ),
    "src/FrugalGPT/llmchain.py": (
        "5ff4deebc83bda89dc29471c5d56ae46ad6d328f5342429df120ef896cbd8803"
    ),
    "src/FrugalGPT/optimizer.py": (
        "13593591a8c85da404574b06103b24c228c5b9563e69b5e1eff23d4c2afd20ee"
    ),
}


def frugalgpt_scorer_text(
    *,
    original_request: str,
    generated_answer: str,
) -> str:
    """Mirror the pinned ``scorer_text(query + ' ' + answer)`` transform."""

    request = _required_text_preserve(
        original_request,
        "frugalgpt_original_request",
    )
    answer = _required_text_preserve(
        generated_answer,
        "frugalgpt_generated_answer",
    )
    joined = f"{request} {answer}".removeprefix(SCORER_PREFIX)
    return f"Q:{joined.split('Q:')[-1]}"


def scorer_training_config() -> dict[str, Any]:
    return {
        "base_model": SCORER_BASE_MODEL,
        "max_length": SCORER_MAX_LENGTH,
        "validation_fraction": SCORER_TRAIN_VALIDATION_FRACTION,
        "epochs": SCORER_EPOCHS,
        "train_batch_size": SCORER_TRAIN_BATCH_SIZE,
        "eval_batch_size": SCORER_EVAL_BATCH_SIZE,
        "warmup_steps": SCORER_WARMUP_STEPS,
        "weight_decay": SCORER_WEIGHT_DECAY,
        "learning_rate": SCORER_LEARNING_RATE,
        "adam_beta1": SCORER_ADAM_BETA1,
        "adam_beta2": SCORER_ADAM_BETA2,
        "adam_epsilon": SCORER_ADAM_EPSILON,
        "max_grad_norm": SCORER_MAX_GRAD_NORM,
        "logging_steps": SCORER_LOGGING_STEPS,
        "optimizer": SCORER_OPTIMIZER,
        "early_stopping": SCORER_EARLY_STOPPING,
        "checkpoint_selection": SCORER_CHECKPOINT_SELECTION,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "lr_scheduler_type": "linear",
        "num_labels": 2,
        "seed": SCORER_SEED,
        "prefix": SCORER_PREFIX,
        "cascade_outer_holdout_fraction": (CASCADE_OUTER_HOLDOUT_FRACTION),
        "cascade_depth": CASCADE_DEPTH,
        "cascade_model_order": [CHEAP_MODEL, STRONG_MODEL],
        "score_noise_injection": False,
        "batch_build": False,
        "training_label": "binary_official_task_success",
        "serializer_protocol": SCORER_SERIALIZER_PROTOCOL,
        "upstream_revision": FRUGALGPT_UPSTREAM_REVISION,
        "papc_adaptation": ("cheap_first_two_model_full_episode_answer_quality"),
    }


SCORER_TRAINING_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        scorer_training_config(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class FrugalGPTTrainingExample:
    example_sha256: str
    original_request: str
    cheap_generated_answer: str
    cheap_official_quality: int

    def __post_init__(self) -> None:
        _required_sha256(self.example_sha256, "frugalgpt_example_sha256")
        _required_text(self.original_request, "frugalgpt_original_request")
        _required_text(
            self.cheap_generated_answer,
            "frugalgpt_generated_answer",
        )
        if isinstance(
            self.cheap_official_quality, bool
        ) or self.cheap_official_quality not in {0, 1}:
            raise ValueError("frugalgpt_training_label_not_binary")

    @property
    def scorer_text(self) -> str:
        return frugalgpt_scorer_text(
            original_request=self.original_request,
            generated_answer=self.cheap_generated_answer,
        )

    def digest_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "scorer_text_sha256": hashlib.sha256(
                self.scorer_text.encode("utf-8")
            ).hexdigest(),
            "cheap_official_quality": self.cheap_official_quality,
        }


@dataclass(frozen=True)
class FrugalGPTDevExample:
    example_sha256: str
    predicted_cheap_quality: float
    cheap_official_quality: float
    strong_official_quality: float
    cheap_episode_cost_usd: float
    strong_episode_cost_usd: float

    def __post_init__(self) -> None:
        _required_sha256(
            self.example_sha256,
            "frugalgpt_dev_example_sha256",
        )
        for field_name in (
            "predicted_cheap_quality",
            "cheap_official_quality",
            "strong_official_quality",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        for field_name in (
            "cheap_episode_cost_usd",
            "strong_episode_cost_usd",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(getattr(self, field_name), field_name),
            )

    def digest_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "predicted_cheap_quality": self.predicted_cheap_quality,
            "cheap_official_quality": self.cheap_official_quality,
            "strong_official_quality": self.strong_official_quality,
            "cheap_episode_cost_usd": self.cheap_episode_cost_usd,
            "strong_episode_cost_usd": self.strong_episode_cost_usd,
        }


@dataclass(frozen=True)
class FrugalGPTDevCandidateReceipt:
    accept_threshold: float
    config_sha256: str
    dev_quality: float
    provider_api_cost_usd: float
    strong_count: int

    def __post_init__(self) -> None:
        for field_name in ("accept_threshold", "dev_quality"):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        if self.config_sha256 != _digest({"accept_threshold": self.accept_threshold}):
            raise ValueError("frugalgpt_dev_config_sha256_mismatch")
        object.__setattr__(
            self,
            "provider_api_cost_usd",
            _nonnegative_float(
                self.provider_api_cost_usd,
                "provider_api_cost_usd",
            ),
        )
        if (
            isinstance(self.strong_count, bool)
            or not isinstance(self.strong_count, int)
            or self.strong_count < 0
        ):
            raise ValueError("frugalgpt_dev_strong_count_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "accept_threshold": self.accept_threshold,
            "config_sha256": self.config_sha256,
            "dev_quality": self.dev_quality,
            "provider_api_cost_usd": self.provider_api_cost_usd,
            "strong_count": self.strong_count,
        }


@dataclass(frozen=True)
class FrugalGPTOperatingPoint:
    accept_threshold: float
    dev_quality: float
    dev_cost_usd: float
    dev_rows: int
    strong_full_dev_anchor: float
    dev_examples_sha256: str
    dev_registry_sha256: str
    dev_membership_sha256: str
    candidate_receipts: tuple[FrugalGPTDevCandidateReceipt, ...]
    selection_rule: str = "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"

    def __post_init__(self) -> None:
        for field_name in (
            "accept_threshold",
            "dev_quality",
            "strong_full_dev_anchor",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "dev_cost_usd",
            _nonnegative_float(self.dev_cost_usd, "dev_cost_usd"),
        )
        if (
            isinstance(self.dev_rows, bool)
            or not isinstance(self.dev_rows, int)
            or self.dev_rows <= 0
        ):
            raise ValueError("frugalgpt_dev_rows_invalid")
        _required_sha256(
            self.dev_examples_sha256,
            "frugalgpt_dev_examples_sha256",
        )
        _required_sha256(
            self.dev_registry_sha256,
            "frugalgpt_dev_registry_sha256",
        )
        _required_sha256(
            self.dev_membership_sha256,
            "frugalgpt_dev_membership_sha256",
        )
        receipts = tuple(self.candidate_receipts)
        if not receipts or any(
            not isinstance(item, FrugalGPTDevCandidateReceipt) for item in receipts
        ):
            raise ValueError("frugalgpt_dev_candidate_receipts_invalid")
        if any(item.strong_count > self.dev_rows for item in receipts):
            raise ValueError("frugalgpt_dev_candidate_strong_count_invalid")
        if not any(item.strong_count == self.dev_rows for item in receipts):
            raise ValueError("frugalgpt_dev_all_strong_candidate_required")
        object.__setattr__(self, "candidate_receipts", receipts)
        if self.selection_rule != (
            "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
        ):
            raise ValueError("frugalgpt_dev_selection_rule_invalid")

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "accept_threshold": self.accept_threshold,
            "dev_quality": self.dev_quality,
            "dev_cost_usd": self.dev_cost_usd,
            "dev_rows": self.dev_rows,
            "strong_full_dev_anchor": self.strong_full_dev_anchor,
            "dev_examples_sha256": self.dev_examples_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "candidate_receipts": [
                item.public_record() for item in self.candidate_receipts
            ],
            "selection_rule": self.selection_rule,
            "strict_acceptance": "score > accept_threshold",
            "upstream_distance_threshold": 1.0 - self.accept_threshold,
        }


def select_frugalgpt_operating_point(
    examples: Sequence[FrugalGPTDevExample],
    *,
    dev_manifest: SplitManifest,
) -> FrugalGPTOperatingPoint:
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("frugalgpt_dev_examples_must_be_sequence")
    supplied = tuple(examples)
    if not supplied or any(
        not isinstance(item, FrugalGPTDevExample) for item in supplied
    ):
        raise ValueError("frugalgpt_dev_examples_invalid")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("frugalgpt_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("frugalgpt_dev_manifest_role_invalid")
    example_ids = tuple(item.example_sha256 for item in supplied)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("frugalgpt_dev_example_duplicate")
    _require_exact_membership(
        dev_manifest,
        example_ids,
        field_name="frugalgpt_dev",
    )
    by_id = {item.example_sha256: item for item in supplied}
    resolved = tuple(by_id[value] for value in dev_manifest.example_sha256s)
    minimum = sum(item.strong_official_quality for item in resolved) / len(resolved)
    candidates = tuple(
        sorted(
            {
                0.0,
                1.0,
                *(item.predicted_cheap_quality for item in resolved),
            }
        )
    )
    receipts: list[FrugalGPTDevCandidateReceipt] = []
    for threshold in candidates:
        realized_quality = 0.0
        realized_cost = 0.0
        for item in resolved:
            accept_cheap = item.predicted_cheap_quality > threshold
            realized_quality += (
                item.cheap_official_quality
                if accept_cheap
                else item.strong_official_quality
            )
            realized_cost += item.cheap_episode_cost_usd
            if not accept_cheap:
                realized_cost += item.strong_episode_cost_usd
        mean_quality = realized_quality / len(resolved)
        receipts.append(
            FrugalGPTDevCandidateReceipt(
                accept_threshold=threshold,
                config_sha256=_digest({"accept_threshold": threshold}),
                dev_quality=mean_quality,
                provider_api_cost_usd=realized_cost,
                strong_count=sum(
                    item.predicted_cheap_quality <= threshold for item in resolved
                ),
            )
        )
    feasible = tuple(item for item in receipts if item.dev_quality + 1e-12 >= minimum)
    if not feasible:
        raise ValueError("frugalgpt_no_dev_operating_point_meets_quality")
    selected = min(
        feasible,
        key=lambda item: (
            item.provider_api_cost_usd,
            item.strong_count,
            item.config_sha256,
        ),
    )
    return FrugalGPTOperatingPoint(
        accept_threshold=selected.accept_threshold,
        dev_quality=selected.dev_quality,
        dev_cost_usd=selected.provider_api_cost_usd,
        dev_rows=len(resolved),
        strong_full_dev_anchor=minimum,
        dev_examples_sha256=_digest([item.digest_record() for item in resolved]),
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        candidate_receipts=tuple(receipts),
    )


def frugalgpt_fit_corpus_record(
    examples: Sequence[FrugalGPTTrainingExample],
    *,
    fit_manifest: SplitManifest,
) -> dict[str, Any]:
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("frugalgpt_fit_examples_must_be_sequence")
    supplied = tuple(examples)
    if not supplied or any(
        not isinstance(item, FrugalGPTTrainingExample) for item in supplied
    ):
        raise ValueError("frugalgpt_fit_examples_invalid")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("frugalgpt_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("frugalgpt_fit_manifest_role_invalid")
    example_ids = tuple(item.example_sha256 for item in supplied)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("frugalgpt_fit_example_duplicate")
    _require_exact_membership(
        fit_manifest,
        example_ids,
        field_name="frugalgpt_fit",
    )
    by_id = {item.example_sha256: item for item in supplied}
    resolved = tuple(by_id[value] for value in fit_manifest.example_sha256s)
    negative_rows = sum(item.cheap_official_quality == 0 for item in resolved)
    positive_rows = sum(item.cheap_official_quality == 1 for item in resolved)
    if not negative_rows or not positive_rows:
        raise ValueError("frugalgpt_fit_class_support_invalid")
    class_counts = {
        "negative": negative_rows,
        "positive": positive_rows,
    }
    return {
        "schema": "papc.frugalgpt.fit_corpus.v1",
        "rows": len(resolved),
        "negative_rows": negative_rows,
        "positive_rows": positive_rows,
        "class_counts_sha256": _digest(class_counts),
        "fit_registry_sha256": fit_manifest.registry_sha256,
        "fit_membership_sha256": _membership_sha256(
            fit_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        "examples": [item.digest_record() for item in resolved],
        "serializer_sha256": SCORER_SERIALIZER_SHA256,
    }


def frugalgpt_fit_corpus_sha256(
    examples: Sequence[FrugalGPTTrainingExample],
    *,
    fit_manifest: SplitManifest,
) -> str:
    return _digest(frugalgpt_fit_corpus_record(examples, fit_manifest=fit_manifest))


@dataclass(frozen=True)
class FrugalGPTCheckpointManifest:
    fit_registry_sha256: str
    fit_membership_sha256: str
    fit_corpus_sha256: str
    fit_rows: int
    fit_negative_rows: int
    fit_positive_rows: int
    fit_class_counts_sha256: str
    serializer_sha256: str = SCORER_SERIALIZER_SHA256
    training_config_sha256: str = SCORER_TRAINING_CONFIG_SHA256
    upstream_revision: str = FRUGALGPT_UPSTREAM_REVISION
    base_model: str = SCORER_BASE_MODEL
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL

    def __post_init__(self) -> None:
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_corpus_sha256",
            "fit_class_counts_sha256",
            "serializer_sha256",
            "training_config_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        counts = (
            self.fit_rows,
            self.fit_negative_rows,
            self.fit_positive_rows,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in counts
            )
            or self.fit_negative_rows + self.fit_positive_rows != self.fit_rows
        ):
            raise ValueError("frugalgpt_manifest_class_support_invalid")
        expected_class_hash = _digest(
            {
                "negative": self.fit_negative_rows,
                "positive": self.fit_positive_rows,
            }
        )
        if self.fit_class_counts_sha256 != expected_class_hash:
            raise ValueError("frugalgpt_manifest_class_hash_mismatch")
        if self.serializer_sha256 != SCORER_SERIALIZER_SHA256:
            raise ValueError("frugalgpt_manifest_serializer_mismatch")
        if self.training_config_sha256 != SCORER_TRAINING_CONFIG_SHA256:
            raise ValueError("frugalgpt_manifest_training_config_mismatch")
        if self.upstream_revision != FRUGALGPT_UPSTREAM_REVISION:
            raise ValueError("frugalgpt_manifest_revision_mismatch")
        if self.base_model != SCORER_BASE_MODEL:
            raise ValueError("frugalgpt_manifest_base_model_mismatch")
        if (self.cheap_model, self.strong_model) != (
            CHEAP_MODEL,
            STRONG_MODEL,
        ):
            raise ValueError("frugalgpt_manifest_model_pair_mismatch")

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "fit_registry_sha256": self.fit_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "fit_corpus_sha256": self.fit_corpus_sha256,
            "fit_rows": self.fit_rows,
            "fit_negative_rows": self.fit_negative_rows,
            "fit_positive_rows": self.fit_positive_rows,
            "fit_class_counts_sha256": self.fit_class_counts_sha256,
            "serializer_sha256": self.serializer_sha256,
            "training_config_sha256": self.training_config_sha256,
            "upstream_revision": self.upstream_revision,
            "upstream_source_sha256s": UPSTREAM_SOURCE_SHA256S,
            "base_model": self.base_model,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "adaptation": ("cheap_first_two_model_full_episode_answer_quality"),
        }

    @classmethod
    def from_record(
        cls,
        record: Any,
    ) -> FrugalGPTCheckpointManifest:
        if not isinstance(record, dict):
            raise TypeError("frugalgpt_manifest_record_invalid")
        if record.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
            raise ValueError("frugalgpt_manifest_schema_invalid")
        if record.get("upstream_source_sha256s") != UPSTREAM_SOURCE_SHA256S:
            raise ValueError("frugalgpt_manifest_source_hash_mismatch")
        if record.get("adaptation") != (
            "cheap_first_two_model_full_episode_answer_quality"
        ):
            raise ValueError("frugalgpt_manifest_adaptation_mismatch")
        allowed = {
            "schema",
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_corpus_sha256",
            "fit_rows",
            "fit_negative_rows",
            "fit_positive_rows",
            "fit_class_counts_sha256",
            "serializer_sha256",
            "training_config_sha256",
            "upstream_revision",
            "upstream_source_sha256s",
            "base_model",
            "cheap_model",
            "strong_model",
            "adaptation",
        }
        if set(record) != allowed:
            raise ValueError("frugalgpt_manifest_fields_invalid")
        return cls(
            fit_registry_sha256=record["fit_registry_sha256"],
            fit_membership_sha256=record["fit_membership_sha256"],
            fit_corpus_sha256=record["fit_corpus_sha256"],
            fit_rows=record["fit_rows"],
            fit_negative_rows=record["fit_negative_rows"],
            fit_positive_rows=record["fit_positive_rows"],
            fit_class_counts_sha256=record["fit_class_counts_sha256"],
            serializer_sha256=record["serializer_sha256"],
            training_config_sha256=record["training_config_sha256"],
            upstream_revision=record["upstream_revision"],
            base_model=record["base_model"],
            cheap_model=record["cheap_model"],
            strong_model=record["strong_model"],
        )


def build_frugalgpt_checkpoint_manifest(
    examples: Sequence[FrugalGPTTrainingExample],
    *,
    fit_manifest: SplitManifest,
) -> FrugalGPTCheckpointManifest:
    record = frugalgpt_fit_corpus_record(
        examples,
        fit_manifest=fit_manifest,
    )
    return FrugalGPTCheckpointManifest(
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=record["fit_membership_sha256"],
        fit_corpus_sha256=_digest(record),
        fit_rows=record["rows"],
        fit_negative_rows=record["negative_rows"],
        fit_positive_rows=record["positive_rows"],
        fit_class_counts_sha256=record["class_counts_sha256"],
    )


@dataclass(frozen=True)
class FrugalGPTTrainingExport:
    checkpoint_dir: Path
    checkpoint_bundle_sha256: str
    checkpoint_manifest: FrugalGPTCheckpointManifest

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_dir, Path):
            raise TypeError("frugalgpt_training_checkpoint_path_invalid")
        _required_sha256(
            self.checkpoint_bundle_sha256,
            "frugalgpt_training_checkpoint_bundle_sha256",
        )
        if not isinstance(
            self.checkpoint_manifest,
            FrugalGPTCheckpointManifest,
        ):
            raise TypeError("frugalgpt_training_manifest_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_EXPORT_SCHEMA,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "checkpoint_manifest": self.checkpoint_manifest.public_record(),
            "checkpoint_manifest_sha256": self.checkpoint_manifest.sha256,
        }

    @classmethod
    def from_public_record(
        cls,
        record: Any,
        *,
        checkpoint_dir: Path,
    ) -> FrugalGPTTrainingExport:
        if not isinstance(record, dict):
            raise TypeError("frugalgpt_training_export_record_invalid")
        if set(record) != {
            "schema",
            "checkpoint_bundle_sha256",
            "checkpoint_manifest",
            "checkpoint_manifest_sha256",
        }:
            raise ValueError("frugalgpt_training_export_fields_invalid")
        if record.get("schema") != TRAINING_EXPORT_SCHEMA:
            raise ValueError("frugalgpt_training_export_schema_invalid")
        manifest = FrugalGPTCheckpointManifest.from_record(
            record.get("checkpoint_manifest")
        )
        if record.get("checkpoint_manifest_sha256") != manifest.sha256:
            raise ValueError(
                "frugalgpt_training_export_manifest_sha256_mismatch"
            )
        return cls(
            checkpoint_dir=Path(checkpoint_dir),
            checkpoint_bundle_sha256=record.get(
                "checkpoint_bundle_sha256"
            ),
            checkpoint_manifest=manifest,
        )


class TrainFrugalGPTScorer(Protocol):
    def __call__(
        self,
        *,
        texts: tuple[str, ...],
        labels: tuple[int, ...],
        output_dir: Path,
        base_model_dir: Path | None,
        config: Mapping[str, Any],
        device: str,
    ) -> None: ...


def train_and_export_frugalgpt_scorer(
    *,
    examples: Sequence[FrugalGPTTrainingExample],
    fit_manifest: SplitManifest,
    checkpoint_dir: Path,
    base_model_dir: Path | None = None,
    device: str = "cuda",
    train_backend: TrainFrugalGPTScorer | None = None,
) -> FrugalGPTTrainingExport:
    """Train and atomically export the fit-only pinned DistilBERT scorer.

    The only accepted examples are the exact ``router_fit`` registry.  The
    external router-dev and official-test splits are deliberately not part of
    this API; dev remains reserved for the separate threshold selector.
    """

    if not isinstance(device, str) or not device.strip():
        raise ValueError("frugalgpt_training_device_invalid")
    target = Path(checkpoint_dir)
    if not target.name:
        raise ValueError("frugalgpt_checkpoint_directory_path_invalid")
    if os.path.lexists(target):
        raise FileExistsError("frugalgpt_checkpoint_directory_already_exists")
    manifest = build_frugalgpt_checkpoint_manifest(
        examples,
        fit_manifest=fit_manifest,
    )
    by_id = {item.example_sha256: item for item in examples}
    resolved = tuple(
        by_id[example_sha256] for example_sha256 in fit_manifest.example_sha256s
    )
    texts = tuple(item.scorer_text for item in resolved)
    labels = tuple(item.cheap_official_quality for item in resolved)

    backend = train_backend or _train_frugalgpt_with_transformers
    if train_backend is None:
        blockers = frugalgpt_training_dependency_blockers(
            base_model_dir=base_model_dir,
            device=device,
        )
        if blockers:
            raise RuntimeError("frugalgpt_training_not_ready:" + ",".join(blockers))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.training.",
            dir=target.parent,
        )
    )
    try:
        backend(
            texts=texts,
            labels=labels,
            output_dir=temporary,
            base_model_dir=base_model_dir,
            config=scorer_training_config(),
            device=device,
        )
        # ``Trainer.save_model`` writes a pickle containing its temporary
        # output path.  It is neither required for inference nor part of the
        # frozen training recipe, and retaining it makes byte-identical
        # training exports hash differently across otherwise identical runs.
        (temporary / "training_args.bin").unlink(missing_ok=True)
        _validate_frugalgpt_checkpoint_files(temporary)
        (temporary / CHECKPOINT_MANIFEST_FILENAME).write_text(
            _canonical_json_text(manifest.public_record()),
            encoding="utf-8",
        )
        bundle_sha256 = _directory_sha256(temporary)
        if os.path.lexists(target):
            raise FileExistsError("frugalgpt_checkpoint_directory_already_exists")
        os.rename(temporary, target)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return FrugalGPTTrainingExport(
        checkpoint_dir=target,
        checkpoint_bundle_sha256=bundle_sha256,
        checkpoint_manifest=manifest,
    )


def _train_frugalgpt_with_transformers(
    *,
    texts: tuple[str, ...],
    labels: tuple[int, ...],
    output_dir: Path,
    base_model_dir: Path | None,
    config: Mapping[str, Any],
    device: str,
) -> None:
    """Local-files-only reproduction of the pinned upstream Score.train."""

    if dict(config) != scorer_training_config():
        raise ValueError("frugalgpt_training_config_not_frozen")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("frugalgpt_training_device_unsupported")
    try:
        import numpy as numpy_module
        import torch
        from sklearn.model_selection import train_test_split
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("frugalgpt_training_dependency_missing") from exc

    if base_model_dir is None:
        raise ValueError("frugalgpt_base_model_directory_required")
    base_model = Path(base_model_dir)
    random.seed(SCORER_SEED)
    numpy_module.random.seed(SCORER_SEED)
    torch.manual_seed(SCORER_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SCORER_SEED)
    set_seed(SCORER_SEED)

    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        list(texts),
        list(labels),
        test_size=SCORER_TRAIN_VALIDATION_FRACTION,
        random_state=SCORER_SEED,
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        base_model,
        local_files_only=True,
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        local_files_only=True,
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(
            self,
            split_texts: Sequence[str],
            split_labels: Sequence[int],
        ) -> None:
            self.encodings = tokenizer(
                list(split_texts),
                truncation=True,
                padding=True,
                max_length=SCORER_MAX_LENGTH,
            )
            self.labels = tuple(split_labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {
                key: torch.tensor(value[index]) for key, value in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[index])
            return item

        def __len__(self) -> int:
            return len(self.labels)

    train_dataset = Dataset(train_texts, train_labels)
    validation_dataset = Dataset(
        validation_texts,
        validation_labels,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, references = prediction
        predictions = numpy_module.argmax(logits, axis=-1)
        return {"accuracy": float(numpy_module.mean(predictions == references))}

    with tempfile.TemporaryDirectory(
        prefix=".frugalgpt-trainer.",
        dir=output_dir.parent,
    ) as trainer_output:
        arguments = {
            "output_dir": trainer_output,
            "num_train_epochs": SCORER_EPOCHS,
            "per_device_train_batch_size": SCORER_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": SCORER_EVAL_BATCH_SIZE,
            "warmup_steps": SCORER_WARMUP_STEPS,
            "weight_decay": SCORER_WEIGHT_DECAY,
            "learning_rate": SCORER_LEARNING_RATE,
            "adam_beta1": SCORER_ADAM_BETA1,
            "adam_beta2": SCORER_ADAM_BETA2,
            "adam_epsilon": SCORER_ADAM_EPSILON,
            "max_grad_norm": SCORER_MAX_GRAD_NORM,
            "logging_steps": SCORER_LOGGING_STEPS,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "lr_scheduler_type": "linear",
            "optim": SCORER_OPTIMIZER,
            "seed": SCORER_SEED,
            "data_seed": SCORER_SEED,
            "report_to": [],
            "use_cpu": device == "cpu",
        }
        training_parameters = inspect.signature(TrainingArguments.__init__).parameters
        evaluation_key = (
            "eval_strategy"
            if "eval_strategy" in training_parameters
            else "evaluation_strategy"
        )
        arguments[evaluation_key] = "epoch"
        training_args = TrainingArguments(**arguments)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


@dataclass(frozen=True)
class FrugalGPTScorerArtifact:
    surface: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_membership_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    checkpoint_bundle_sha256: str
    checkpoint_manifest_sha256: str
    fit_corpus_sha256: str
    fit_class_counts_sha256: str
    fit_rows: int
    fit_negative_rows: int
    fit_positive_rows: int
    operating_point: FrugalGPTOperatingPoint
    serializer_sha256: str = SCORER_SERIALIZER_SHA256
    training_config_sha256: str = SCORER_TRAINING_CONFIG_SHA256
    upstream_revision: str = FRUGALGPT_UPSTREAM_REVISION
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL

    def __post_init__(self) -> None:
        _required_text(self.surface, "frugalgpt_surface")
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_membership_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "checkpoint_bundle_sha256",
            "checkpoint_manifest_sha256",
            "fit_corpus_sha256",
            "fit_class_counts_sha256",
            "serializer_sha256",
            "training_config_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if (
            len(
                {
                    self.fit_registry_sha256,
                    self.dev_registry_sha256,
                    self.test_registry_sha256,
                }
            )
            != 3
        ):
            raise ValueError("frugalgpt_split_digest_reused")
        counts = (
            self.fit_rows,
            self.fit_negative_rows,
            self.fit_positive_rows,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in counts
            )
            or self.fit_negative_rows + self.fit_positive_rows != self.fit_rows
        ):
            raise ValueError("frugalgpt_fit_class_support_invalid")
        if self.fit_class_counts_sha256 != _digest(
            {
                "negative": self.fit_negative_rows,
                "positive": self.fit_positive_rows,
            }
        ):
            raise ValueError("frugalgpt_fit_class_hash_mismatch")
        if not isinstance(self.operating_point, FrugalGPTOperatingPoint):
            raise TypeError("frugalgpt_operating_point_invalid")
        if (
            self.operating_point.dev_registry_sha256 != self.dev_registry_sha256
            or self.operating_point.dev_membership_sha256 != self.dev_membership_sha256
        ):
            raise ValueError("frugalgpt_operating_point_registry_mismatch")
        if self.serializer_sha256 != SCORER_SERIALIZER_SHA256:
            raise ValueError("frugalgpt_serializer_not_frozen")
        if self.training_config_sha256 != SCORER_TRAINING_CONFIG_SHA256:
            raise ValueError("frugalgpt_training_config_not_frozen")
        if self.upstream_revision != FRUGALGPT_UPSTREAM_REVISION:
            raise ValueError("frugalgpt_revision_not_frozen")
        if self.cheap_model != CHEAP_MODEL or self.strong_model != STRONG_MODEL:
            raise ValueError("frugalgpt_models_not_frozen")

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": "papc.frugalgpt.scorer_artifact.v3",
            "surface": self.surface,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "split_binding_sha256": self.split_binding_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "checkpoint_manifest_sha256": (self.checkpoint_manifest_sha256),
            "fit_corpus_sha256": self.fit_corpus_sha256,
            "fit_class_counts_sha256": self.fit_class_counts_sha256,
            "fit_rows": self.fit_rows,
            "fit_negative_rows": self.fit_negative_rows,
            "fit_positive_rows": self.fit_positive_rows,
            "operating_point": self.operating_point.public_record(),
            "operating_point_sha256": self.operating_point.sha256,
            "serializer_sha256": self.serializer_sha256,
            "training_config_sha256": self.training_config_sha256,
            "upstream_revision": self.upstream_revision,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
        }

    @classmethod
    def from_public_record(
        cls,
        record: Any,
    ) -> FrugalGPTScorerArtifact:
        if not isinstance(record, dict):
            raise TypeError("frugalgpt_artifact_record_invalid")
        if record.get("schema") != "papc.frugalgpt.scorer_artifact.v3":
            raise ValueError("frugalgpt_artifact_schema_invalid")
        operating_record = record.get("operating_point")
        if not isinstance(operating_record, dict):
            raise ValueError("frugalgpt_artifact_operating_point_invalid")
        operating_point = FrugalGPTOperatingPoint(
            accept_threshold=operating_record["accept_threshold"],
            dev_quality=operating_record["dev_quality"],
            dev_cost_usd=operating_record["dev_cost_usd"],
            dev_rows=operating_record["dev_rows"],
            strong_full_dev_anchor=operating_record["strong_full_dev_anchor"],
            dev_examples_sha256=operating_record["dev_examples_sha256"],
            dev_registry_sha256=operating_record["dev_registry_sha256"],
            dev_membership_sha256=operating_record["dev_membership_sha256"],
            candidate_receipts=tuple(
                FrugalGPTDevCandidateReceipt(**item)
                for item in operating_record["candidate_receipts"]
            ),
            selection_rule=operating_record["selection_rule"],
        )
        artifact = cls(
            surface=record["surface"],
            fit_registry_sha256=record["fit_registry_sha256"],
            dev_registry_sha256=record["dev_registry_sha256"],
            test_registry_sha256=record["test_registry_sha256"],
            fit_membership_sha256=record["fit_membership_sha256"],
            dev_membership_sha256=record["dev_membership_sha256"],
            split_binding_sha256=record["split_binding_sha256"],
            checkpoint_bundle_sha256=record["checkpoint_bundle_sha256"],
            checkpoint_manifest_sha256=record["checkpoint_manifest_sha256"],
            fit_corpus_sha256=record["fit_corpus_sha256"],
            fit_class_counts_sha256=record["fit_class_counts_sha256"],
            fit_rows=record["fit_rows"],
            fit_negative_rows=record["fit_negative_rows"],
            fit_positive_rows=record["fit_positive_rows"],
            operating_point=operating_point,
            serializer_sha256=record["serializer_sha256"],
            training_config_sha256=record["training_config_sha256"],
            upstream_revision=record["upstream_revision"],
            cheap_model=record["cheap_model"],
            strong_model=record["strong_model"],
        )
        if record.get("operating_point_sha256") != operating_point.sha256:
            raise ValueError("frugalgpt_operating_point_sha256_mismatch")
        if artifact.public_record() != record:
            raise ValueError("frugalgpt_artifact_record_not_canonical")
        return artifact

    def router_binding(self) -> RouterArtifactBinding:
        return RouterArtifactBinding(
            method=FRUGALGPT,
            surface=self.surface,
            upstream_revision=self.upstream_revision,
            artifact_sha256=self.sha256,
            fit_registry_sha256=self.fit_registry_sha256,
            dev_registry_sha256=self.dev_registry_sha256,
            test_registry_sha256=self.test_registry_sha256,
            calibration_sha256=self.operating_point.sha256,
        )


def bind_frugalgpt_scorer_artifact(
    *,
    surface: str,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    checkpoint_bundle_sha256: str,
    checkpoint_manifest: FrugalGPTCheckpointManifest,
    operating_point: FrugalGPTOperatingPoint,
) -> FrugalGPTScorerArtifact:
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    if not isinstance(checkpoint_manifest, FrugalGPTCheckpointManifest):
        raise TypeError("frugalgpt_checkpoint_manifest_invalid")
    if checkpoint_manifest.fit_registry_sha256 != fit_manifest.registry_sha256:
        raise ValueError("frugalgpt_checkpoint_fit_registry_mismatch")
    expected_fit_membership = _membership_sha256(
        fit_manifest,
        fit_manifest.example_sha256s,
    )
    if checkpoint_manifest.fit_membership_sha256 != expected_fit_membership:
        raise ValueError("frugalgpt_checkpoint_fit_membership_mismatch")
    if not isinstance(operating_point, FrugalGPTOperatingPoint):
        raise TypeError("frugalgpt_operating_point_invalid")
    if (
        operating_point.dev_registry_sha256 != dev_manifest.registry_sha256
        or operating_point.dev_membership_sha256
        != _membership_sha256(dev_manifest, dev_manifest.example_sha256s)
    ):
        raise ValueError("frugalgpt_operating_point_dev_membership_mismatch")
    return FrugalGPTScorerArtifact(
        surface=surface,
        fit_registry_sha256=fit_manifest.registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_membership_sha256=checkpoint_manifest.fit_membership_sha256,
        dev_membership_sha256=operating_point.dev_membership_sha256,
        split_binding_sha256=_digest(
            {
                "fit": fit_manifest.__dict__,
                "dev": dev_manifest.__dict__,
                "test": test_manifest.__dict__,
            }
        ),
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest.sha256,
        fit_corpus_sha256=checkpoint_manifest.fit_corpus_sha256,
        fit_class_counts_sha256=(checkpoint_manifest.fit_class_counts_sha256),
        fit_rows=checkpoint_manifest.fit_rows,
        fit_negative_rows=checkpoint_manifest.fit_negative_rows,
        fit_positive_rows=checkpoint_manifest.fit_positive_rows,
        operating_point=operating_point,
    )


@dataclass(frozen=True)
class FrugalGPTReadiness:
    decision: str
    blockers: tuple[str, ...]
    training_dependency_blockers: tuple[str, ...]
    loading_dependency_blockers: tuple[str, ...]
    checkpoint_bundle_sha256: str | None
    checkpoint_manifest_sha256: str | None

    def __post_init__(self) -> None:
        if self.decision not in {"READY", "BLOCKED"}:
            raise ValueError("frugalgpt_readiness_decision_invalid")
        if (self.decision == "READY") != (not self.blockers):
            raise ValueError("frugalgpt_readiness_decision_inconsistent")

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": "papc.frugalgpt.readiness.v1",
            "decision": self.decision,
            "blockers": list(self.blockers),
            "training_environment_ready": (not self.training_dependency_blockers),
            "training_dependency_blockers": list(self.training_dependency_blockers),
            "loading_environment_ready": (not self.loading_dependency_blockers),
            "loading_dependency_blockers": list(self.loading_dependency_blockers),
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "checkpoint_manifest_sha256": (self.checkpoint_manifest_sha256),
            "provider_calls": 0,
            "secret_reads": 0,
            "authorizes_paid_run": False,
            "authorizes_result_claim": False,
        }


def inspect_frugalgpt_readiness(
    *,
    artifact: FrugalGPTScorerArtifact,
    checkpoint_dir: Path,
) -> FrugalGPTReadiness:
    """Validate a local scorer bundle without loading a model or provider."""

    if not isinstance(artifact, FrugalGPTScorerArtifact):
        raise TypeError("frugalgpt_scorer_artifact_invalid")
    training_dependencies = _missing_modules(
        ("torch", "transformers", "sklearn", "accelerate")
    )
    loading_dependencies = _missing_modules(("torch", "transformers"))
    blockers = list(loading_dependencies)
    checkpoint = Path(checkpoint_dir)
    bundle_sha256: str | None = None
    manifest_sha256: str | None = None

    try:
        bundle_sha256 = _directory_sha256(checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        blockers.append(str(exc))
    if bundle_sha256 is not None and bundle_sha256 != artifact.checkpoint_bundle_sha256:
        blockers.append("frugalgpt_checkpoint_bundle_sha256_mismatch")

    manifest: FrugalGPTCheckpointManifest | None = None
    manifest_path = checkpoint / CHECKPOINT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        blockers.append("frugalgpt_checkpoint_manifest_missing")
    else:
        try:
            manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = FrugalGPTCheckpointManifest.from_record(manifest_record)
            if _canonical_json_text(manifest.public_record()) != (
                manifest_path.read_text(encoding="utf-8")
            ):
                blockers.append("frugalgpt_checkpoint_manifest_not_canonical")
            manifest_sha256 = manifest.sha256
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            blockers.append(f"frugalgpt_checkpoint_manifest_invalid:{exc}")
    if manifest is not None:
        if manifest.sha256 != artifact.checkpoint_manifest_sha256:
            blockers.append("frugalgpt_checkpoint_manifest_sha256_mismatch")
        expected_manifest_fields = (
            ("fit_registry_sha256", artifact.fit_registry_sha256),
            ("fit_membership_sha256", artifact.fit_membership_sha256),
            ("fit_corpus_sha256", artifact.fit_corpus_sha256),
            ("fit_rows", artifact.fit_rows),
            ("fit_negative_rows", artifact.fit_negative_rows),
            ("fit_positive_rows", artifact.fit_positive_rows),
            (
                "fit_class_counts_sha256",
                artifact.fit_class_counts_sha256,
            ),
        )
        if any(
            getattr(manifest, field_name) != expected
            for field_name, expected in expected_manifest_fields
        ):
            blockers.append("frugalgpt_checkpoint_manifest_artifact_mismatch")

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        blockers.append("frugalgpt_checkpoint_config_missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise TypeError("config_not_mapping")
            if config.get("model_type") != "distilbert":
                blockers.append("frugalgpt_checkpoint_model_type_mismatch")
            label_count = config.get("num_labels")
            if label_count is None and isinstance(config.get("id2label"), dict):
                label_count = len(config["id2label"])
            if label_count is None and config.get("architectures") == [
                "DistilBertForSequenceClassification"
            ]:
                label_count = 2
            if label_count != 2:
                blockers.append("frugalgpt_checkpoint_num_labels_mismatch")
        except (
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            blockers.append(f"frugalgpt_checkpoint_config_invalid:{exc}")

    if not any(
        (checkpoint / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    ):
        blockers.append("frugalgpt_checkpoint_weights_missing")
    if not (checkpoint / "tokenizer_config.json").is_file():
        blockers.append("frugalgpt_checkpoint_tokenizer_config_missing")
    if not any(
        (checkpoint / filename).is_file()
        for filename in ("tokenizer.json", "vocab.txt")
    ):
        blockers.append("frugalgpt_checkpoint_tokenizer_vocabulary_missing")

    deduplicated = tuple(dict.fromkeys(blockers))
    return FrugalGPTReadiness(
        decision="READY" if not deduplicated else "BLOCKED",
        blockers=deduplicated,
        training_dependency_blockers=training_dependencies,
        loading_dependency_blockers=loading_dependencies,
        checkpoint_bundle_sha256=bundle_sha256,
        checkpoint_manifest_sha256=manifest_sha256,
    )


class _FrugalGPTCheckpointScorer:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module

    def __call__(self, view: CheapCompletionRouterView) -> float:
        if not isinstance(view, CheapCompletionRouterView):
            raise TypeError("frugalgpt_cheap_completion_view_required")
        text = frugalgpt_scorer_text(
            original_request=view.original_request,
            generated_answer=view.cheap_generated_answer,
        )
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=SCORER_MAX_LENGTH,
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = self._torch.nn.functional.softmax(
                logits,
                dim=-1,
            )
        return float(probabilities[0, 1].detach().cpu().item())


class FrugalGPTFitOnlyDistilBertScorer(_FrugalGPTCheckpointScorer):
    """Load a fit-bound checkpoint solely to score the router-dev split."""

    @classmethod
    def load(
        cls,
        *,
        training_export: FrugalGPTTrainingExport,
    ) -> FrugalGPTFitOnlyDistilBertScorer:
        if not isinstance(training_export, FrugalGPTTrainingExport):
            raise TypeError("frugalgpt_training_export_invalid")
        checkpoint = training_export.checkpoint_dir
        _validate_frugalgpt_checkpoint_files(checkpoint)
        manifest_path = checkpoint / CHECKPOINT_MANIFEST_FILENAME
        try:
            manifest_record = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (
            FileNotFoundError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "frugalgpt_fit_checkpoint_manifest_invalid"
            ) from exc
        manifest = FrugalGPTCheckpointManifest.from_record(manifest_record)
        if (
            manifest != training_export.checkpoint_manifest
            or manifest_path.read_text(encoding="utf-8")
            != _canonical_json_text(manifest.public_record())
            or _directory_sha256(checkpoint)
            != training_export.checkpoint_bundle_sha256
        ):
            raise RuntimeError("frugalgpt_fit_checkpoint_binding_mismatch")
        tokenizer, model, torch_module = _load_frugalgpt_checkpoint(
            checkpoint
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
        )


class FrugalGPTDistilBertScorer(_FrugalGPTCheckpointScorer):
    """Direct final-artifact loader; no arbitrary score callback is accepted."""

    def __init__(
        self,
        *,
        artifact: FrugalGPTScorerArtifact,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
        )
        self.artifact = artifact

    @classmethod
    def load(
        cls,
        *,
        artifact: FrugalGPTScorerArtifact,
        checkpoint_dir: Path,
    ) -> FrugalGPTDistilBertScorer:
        if not isinstance(artifact, FrugalGPTScorerArtifact):
            raise TypeError("frugalgpt_scorer_artifact_invalid")
        checkpoint = Path(checkpoint_dir)
        readiness = inspect_frugalgpt_readiness(
            artifact=artifact,
            checkpoint_dir=checkpoint,
        )
        if readiness.decision != "READY":
            raise RuntimeError(
                "frugalgpt_checkpoint_not_ready:" + ",".join(readiness.blockers)
            )
        tokenizer, model, torch_module = _load_frugalgpt_checkpoint(
            checkpoint
        )
        return cls(
            artifact=artifact,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
        )


def _load_frugalgpt_checkpoint(checkpoint: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "frugalgpt_torch_transformers_dependency_missing"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, torch


def frugalgpt_training_dependency_blockers(
    *,
    base_model_dir: Path | None,
    device: str = "cuda",
) -> tuple[str, ...]:
    """Inspect everything the local-files-only trainer needs before fitting."""

    if not isinstance(device, str) or not device.strip():
        raise ValueError("frugalgpt_training_device_invalid")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("frugalgpt_training_device_unsupported")
    blockers = list(
        _missing_modules(("torch", "transformers", "sklearn", "accelerate"))
    )
    if not blockers and device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            blockers.append("training_device_unavailable:cuda")
    if base_model_dir is None:
        blockers.append("frugalgpt_base_model_directory_required")
        return tuple(blockers)
    base_model = Path(base_model_dir)
    if not base_model.is_dir():
        blockers.append("frugalgpt_base_model_directory_missing")
        return tuple(blockers)
    if any(item.is_symlink() for item in base_model.rglob("*")):
        blockers.append("frugalgpt_base_model_symlink_forbidden")
    config_path = base_model / "config.json"
    if not config_path.is_file():
        blockers.append("frugalgpt_base_model_config_missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("model_type") != "distilbert":
                blockers.append("frugalgpt_base_model_not_distilbert")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            blockers.append("frugalgpt_base_model_config_invalid")
    if not any(
        (base_model / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    ):
        blockers.append("frugalgpt_base_model_weights_missing")
    if not (base_model / "tokenizer_config.json").is_file():
        blockers.append("frugalgpt_base_model_tokenizer_config_missing")
    if not any(
        (base_model / filename).is_file()
        for filename in ("tokenizer.json", "vocab.txt")
    ):
        blockers.append("frugalgpt_base_model_tokenizer_vocabulary_missing")
    return tuple(dict.fromkeys(blockers))


def _validate_frugalgpt_checkpoint_files(checkpoint: Path) -> None:
    root = Path(checkpoint)
    if not root.is_dir():
        raise ValueError("frugalgpt_trained_checkpoint_directory_missing")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("frugalgpt_checkpoint_symlink_forbidden")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("frugalgpt_checkpoint_config_missing")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frugalgpt_checkpoint_config_invalid") from exc
    if not isinstance(config, dict) or config.get("model_type") != "distilbert":
        raise ValueError("frugalgpt_checkpoint_model_type_mismatch")
    label_count = config.get("num_labels")
    if label_count is None and isinstance(config.get("id2label"), dict):
        label_count = len(config["id2label"])
    if label_count is None and config.get("architectures") == [
        "DistilBertForSequenceClassification"
    ]:
        label_count = 2
    if label_count != 2:
        raise ValueError("frugalgpt_checkpoint_num_labels_mismatch")
    if not any(
        (root / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    ):
        raise ValueError("frugalgpt_checkpoint_weights_missing")
    if not (root / "tokenizer_config.json").is_file():
        raise ValueError("frugalgpt_checkpoint_tokenizer_config_missing")
    if not any(
        (root / filename).is_file() for filename in ("tokenizer.json", "vocab.txt")
    ):
        raise ValueError("frugalgpt_checkpoint_tokenizer_vocabulary_missing")


def _directory_sha256(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError("frugalgpt_checkpoint_directory_missing")
    digest = hashlib.sha256()
    files = tuple(
        sorted(
            item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
        )
    )
    if not files:
        raise ValueError("frugalgpt_checkpoint_directory_empty")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("frugalgpt_checkpoint_symlink_forbidden")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        raw = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
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
        field_name="frugalgpt_membership",
    )
    return _digest(
        {
            "role": manifest.role,
            "registry_sha256": manifest.registry_sha256,
            "example_sha256s": manifest.example_sha256s,
        }
    )


def _canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _missing_modules(module_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"missing_dependency:{module_name}"
        for module_name in module_names
        if importlib.util.find_spec(module_name) is None
    )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_text")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _required_text_preserve(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_text")
    if not value.strip():
        raise ValueError(f"{field_name}_empty")
    return value


def _required_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name}_not_finite")
    return resolved


def _unit_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return resolved


def _nonnegative_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if resolved < 0.0:
        raise ValueError(f"{field_name}_negative")
    return resolved


__all__ = [
    "CHECKPOINT_MANIFEST_FILENAME",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "FRUGALGPT_UPSTREAM_REVISION",
    "SCORER_BASE_MODEL",
    "SCORER_CHECKPOINT_SELECTION",
    "SCORER_EARLY_STOPPING",
    "SCORER_MAX_LENGTH",
    "SCORER_OPTIMIZER",
    "SCORER_SERIALIZER_SHA256",
    "SCORER_TRAINING_CONFIG_SHA256",
    "TRAINING_EXPORT_SCHEMA",
    "FrugalGPTDevExample",
    "FrugalGPTDevCandidateReceipt",
    "FrugalGPTDistilBertScorer",
    "FrugalGPTFitOnlyDistilBertScorer",
    "FrugalGPTCheckpointManifest",
    "FrugalGPTOperatingPoint",
    "FrugalGPTReadiness",
    "FrugalGPTScorerArtifact",
    "FrugalGPTTrainingExample",
    "FrugalGPTTrainingExport",
    "TrainFrugalGPTScorer",
    "build_frugalgpt_checkpoint_manifest",
    "bind_frugalgpt_scorer_artifact",
    "frugalgpt_fit_corpus_record",
    "frugalgpt_fit_corpus_sha256",
    "frugalgpt_scorer_text",
    "frugalgpt_training_dependency_blockers",
    "inspect_frugalgpt_readiness",
    "scorer_training_config",
    "select_frugalgpt_operating_point",
    "train_and_export_frugalgpt_scorer",
]

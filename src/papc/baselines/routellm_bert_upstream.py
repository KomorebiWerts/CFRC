"""In-domain adapter for RouteLLM's released request-only BERT router."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from papc.baselines.routellm_mf_upstream import (
    ROUTELLM_MF_UPSTREAM_REVISION,
    PairedFitOutcome,
    RouteLLMMFDevCalibration,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM,
    ROUTELLM_LOCAL_MODEL,
    STRONG_MODEL,
    UPSTREAM_REVISIONS,
    RequestRouterView,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)
from papc.baselines.router_runtime import ScalarPrediction

ROUTELLM_BERT_VARIANT = "bert"
ROUTELLM_BERT_LOCAL_MODEL = ROUTELLM_LOCAL_MODEL
ROUTELLM_BERT_BASE_REPO = "routellm/bert_gpt4_augmented"
ROUTELLM_BERT_BASE_REVISION = "f625a9c4e11ca22924e5e418c56e015a862a8928"
ROUTELLM_BERT_BASE_BUNDLE_SHA256 = (
    "ef6d810db2a670694d5cc0057b2969e60af104b43987e35b1b030f231d183bf1"
)
ROUTELLM_BERT_MAX_LENGTH = 512
ROUTELLM_BERT_BATCH_SIZE = 8
ROUTELLM_BERT_NUM_EPOCHS = 3
ROUTELLM_BERT_LEARNING_RATE = 2e-5
ROUTELLM_BERT_WEIGHT_DECAY = 0.01
ROUTELLM_BERT_RANDOM_SEED = 42
ROUTELLM_BERT_MANIFEST_FILENAME = "papc_routellm_bert_manifest.json"
ROUTELLM_BERT_MANIFEST_SCHEMA = "papc.routellm_bert.checkpoint_manifest.v1"
ROUTELLM_BERT_ARTIFACT_SCHEMA = "papc.routellm_bert.artifact.v1"
ROUTELLM_BERT_TRAINING_CONFIG = {
    "batch_size": ROUTELLM_BERT_BATCH_SIZE,
    "learning_rate": ROUTELLM_BERT_LEARNING_RATE,
    "max_length": ROUTELLM_BERT_MAX_LENGTH,
    "num_epochs": ROUTELLM_BERT_NUM_EPOCHS,
    "optimizer": "AdamW",
    "random_seed": ROUTELLM_BERT_RANDOM_SEED,
    "weight_decay": ROUTELLM_BERT_WEIGHT_DECAY,
}
ROUTELLM_BERT_TRAINING_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        ROUTELLM_BERT_TRAINING_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class RouteLLMBertFitExample:
    example_sha256: str
    original_request: str = field(repr=False)
    label: int

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "routellm_bert_fit_example")
        if not isinstance(self.original_request, str) or not self.original_request.strip():
            raise ValueError("routellm_bert_fit_request_empty")
        if self.label not in {0, 1, 2}:
            raise ValueError("routellm_bert_fit_label_invalid")

    def digest_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "original_request_sha256": hashlib.sha256(
                self.original_request.encode("utf-8")
            ).hexdigest(),
            "label": self.label,
        }


def build_routellm_bert_fit_examples(
    outcomes: Sequence[PairedFitOutcome],
    *,
    fit_manifest: SplitManifest,
) -> tuple[RouteLLMBertFitExample, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise TypeError("routellm_bert_fit_outcomes_must_be_sequence")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("routellm_bert_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("routellm_bert_fit_manifest_role_invalid")
    supplied = tuple(outcomes)
    by_id = {item.example_sha256: item for item in supplied}
    if len(by_id) != len(supplied):
        raise ValueError("routellm_bert_fit_example_duplicate")
    if set(by_id) != set(fit_manifest.example_sha256s):
        raise ValueError("routellm_bert_fit_membership_mismatch")
    examples = []
    for example_sha256 in fit_manifest.example_sha256s:
        item = by_id[example_sha256]
        if item.strong_score > item.cheap_score:
            label = 0
        elif item.strong_score == item.cheap_score:
            label = 1
        else:
            label = 2
        examples.append(
            RouteLLMBertFitExample(
                example_sha256=example_sha256,
                original_request=item.original_request,
                label=label,
            )
        )
    if len({item.label for item in examples}) < 2:
        raise ValueError("routellm_bert_fit_class_support_insufficient")
    return tuple(examples)


@dataclass(frozen=True)
class RouteLLMBertCheckpointManifest:
    fit_registry_sha256: str
    fit_membership_sha256: str
    fit_corpus_sha256: str
    fit_rows: int
    class_counts: tuple[tuple[str, int], ...]
    training_config_sha256: str = ROUTELLM_BERT_TRAINING_CONFIG_SHA256
    upstream_revision: str = ROUTELLM_MF_UPSTREAM_REVISION
    base_repo: str = ROUTELLM_BERT_BASE_REPO
    base_revision: str = ROUTELLM_BERT_BASE_REVISION
    base_bundle_sha256: str = ROUTELLM_BERT_BASE_BUNDLE_SHA256

    def __post_init__(self) -> None:
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_corpus_sha256",
            "training_config_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not isinstance(self.fit_rows, int) or isinstance(self.fit_rows, bool):
            raise ValueError("routellm_bert_fit_rows_invalid")
        expected_names = ("strong_wins", "tie", "cheap_wins")
        if tuple(name for name, _ in self.class_counts) != expected_names:
            raise ValueError("routellm_bert_class_names_invalid")
        if (
            any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for _, count in self.class_counts
            )
            or sum(count for _, count in self.class_counts) != self.fit_rows
            or sum(count > 0 for _, count in self.class_counts) < 2
        ):
            raise ValueError("routellm_bert_class_counts_invalid")
        if self.training_config_sha256 != ROUTELLM_BERT_TRAINING_CONFIG_SHA256:
            raise ValueError("routellm_bert_training_config_mismatch")
        if self.upstream_revision != ROUTELLM_MF_UPSTREAM_REVISION:
            raise ValueError("routellm_bert_upstream_revision_mismatch")
        if self.upstream_revision != UPSTREAM_REVISIONS[ROUTELLM]:
            raise ValueError("routellm_bert_contract_revision_mismatch")
        if (
            self.base_repo != ROUTELLM_BERT_BASE_REPO
            or self.base_revision != ROUTELLM_BERT_BASE_REVISION
            or self.base_bundle_sha256 != ROUTELLM_BERT_BASE_BUNDLE_SHA256
        ):
            raise ValueError("routellm_bert_base_binding_mismatch")

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": ROUTELLM_BERT_MANIFEST_SCHEMA,
            "fit_registry_sha256": self.fit_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "fit_corpus_sha256": self.fit_corpus_sha256,
            "fit_rows": self.fit_rows,
            "class_counts": dict(self.class_counts),
            "training_config": ROUTELLM_BERT_TRAINING_CONFIG,
            "training_config_sha256": self.training_config_sha256,
            "upstream_revision": self.upstream_revision,
            "base_repo": self.base_repo,
            "base_revision": self.base_revision,
            "base_bundle_sha256": self.base_bundle_sha256,
            "label_order": ["strong_wins", "tie", "cheap_wins"],
        }

    @classmethod
    def from_record(cls, record: Any) -> RouteLLMBertCheckpointManifest:
        if not isinstance(record, Mapping):
            raise TypeError("routellm_bert_manifest_record_invalid")
        if record.get("schema") != ROUTELLM_BERT_MANIFEST_SCHEMA:
            raise ValueError("routellm_bert_manifest_schema_invalid")
        if record.get("training_config") != ROUTELLM_BERT_TRAINING_CONFIG:
            raise ValueError("routellm_bert_training_record_mismatch")
        if record.get("label_order") != ["strong_wins", "tie", "cheap_wins"]:
            raise ValueError("routellm_bert_label_order_mismatch")
        counts = record.get("class_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("routellm_bert_class_counts_record_invalid")
        return cls(
            fit_registry_sha256=record["fit_registry_sha256"],
            fit_membership_sha256=record["fit_membership_sha256"],
            fit_corpus_sha256=record["fit_corpus_sha256"],
            fit_rows=record["fit_rows"],
            class_counts=tuple(
                (name, counts[name])
                for name in ("strong_wins", "tie", "cheap_wins")
            ),
            training_config_sha256=record["training_config_sha256"],
            upstream_revision=record["upstream_revision"],
            base_repo=record["base_repo"],
            base_revision=record["base_revision"],
            base_bundle_sha256=record["base_bundle_sha256"],
        )


@dataclass(frozen=True)
class RouteLLMBertTrainingExport:
    checkpoint_dir: Path
    checkpoint_bundle_sha256: str
    checkpoint_manifest: RouteLLMBertCheckpointManifest

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_dir, Path):
            raise TypeError("routellm_bert_checkpoint_path_invalid")
        _require_sha256(
            self.checkpoint_bundle_sha256,
            "routellm_bert_checkpoint_bundle",
        )
        if not isinstance(
            self.checkpoint_manifest,
            RouteLLMBertCheckpointManifest,
        ):
            raise TypeError("routellm_bert_checkpoint_manifest_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "checkpoint_manifest": self.checkpoint_manifest.public_record(),
            "checkpoint_manifest_sha256": self.checkpoint_manifest.sha256,
        }

    @classmethod
    def from_record(
        cls,
        record: Any,
        *,
        checkpoint_dir: Path,
    ) -> RouteLLMBertTrainingExport:
        if not isinstance(record, Mapping):
            raise TypeError("routellm_bert_training_export_invalid")
        manifest = RouteLLMBertCheckpointManifest.from_record(
            record.get("checkpoint_manifest")
        )
        if record.get("checkpoint_manifest_sha256") != manifest.sha256:
            raise ValueError("routellm_bert_manifest_sha256_mismatch")
        return cls(
            checkpoint_dir=checkpoint_dir,
            checkpoint_bundle_sha256=record["checkpoint_bundle_sha256"],
            checkpoint_manifest=manifest,
        )


def train_and_export_routellm_bert(
    *,
    examples: Sequence[RouteLLMBertFitExample],
    fit_manifest: SplitManifest,
    checkpoint_dir: Path,
    base_model_dir: Path,
    device: str = "cuda",
) -> RouteLLMBertTrainingExport:
    resolved = _ordered_fit_examples(examples, fit_manifest=fit_manifest)
    target = Path(checkpoint_dir)
    base = Path(base_model_dir)
    if os.path.lexists(target):
        raise FileExistsError("routellm_bert_checkpoint_already_exists")
    if not base.is_dir():
        raise FileNotFoundError("routellm_bert_base_model_missing")
    if _directory_sha256(base) != ROUTELLM_BERT_BASE_BUNDLE_SHA256:
        raise ValueError("routellm_bert_base_model_bundle_mismatch")
    manifest = _build_checkpoint_manifest(resolved, fit_manifest=fit_manifest)

    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(ROUTELLM_BERT_RANDOM_SEED)
    np.random.seed(ROUTELLM_BERT_RANDOM_SEED)
    torch.manual_seed(ROUTELLM_BERT_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ROUTELLM_BERT_RANDOM_SEED)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("routellm_bert_cuda_unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=3,
    ).to(resolved_device)
    encoded = tokenizer(
        [item.original_request for item in resolved],
        padding=True,
        truncation=True,
        max_length=ROUTELLM_BERT_MAX_LENGTH,
        return_tensors="pt",
    )
    labels = torch.tensor([item.label for item in resolved], dtype=torch.long)
    names = tuple(encoded)
    dataset = TensorDataset(*(encoded[name] for name in names), labels)
    generator = torch.Generator().manual_seed(ROUTELLM_BERT_RANDOM_SEED)
    loader = DataLoader(
        dataset,
        batch_size=ROUTELLM_BERT_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=ROUTELLM_BERT_LEARNING_RATE,
        weight_decay=ROUTELLM_BERT_WEIGHT_DECAY,
    )
    model.train()
    for _ in range(ROUTELLM_BERT_NUM_EPOCHS):
        for batch in loader:
            inputs = {
                name: value.to(resolved_device)
                for name, value in zip(names, batch[:-1], strict=True)
            }
            outputs = model(**inputs, labels=batch[-1].to(resolved_device))
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        _write_canonical_json(
            temporary / ROUTELLM_BERT_MANIFEST_FILENAME,
            manifest.public_record(),
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return RouteLLMBertTrainingExport(
        checkpoint_dir=target,
        checkpoint_bundle_sha256=_directory_sha256(target),
        checkpoint_manifest=manifest,
    )


@dataclass(frozen=True)
class RouteLLMBertArtifact:
    surface: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_membership_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    checkpoint_bundle_sha256: str
    checkpoint_manifest_sha256: str
    dev_calibration: RouteLLMMFDevCalibration
    variant: str = ROUTELLM_BERT_VARIANT
    upstream_revision: str = ROUTELLM_MF_UPSTREAM_REVISION
    base_repo: str = ROUTELLM_BERT_BASE_REPO
    base_revision: str = ROUTELLM_BERT_BASE_REVISION
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL
    reasoning_effort: str = REASONING_EFFORT

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("routellm_bert_artifact_surface_invalid")
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_membership_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "checkpoint_bundle_sha256",
            "checkpoint_manifest_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if len(
            {
                self.fit_registry_sha256,
                self.dev_registry_sha256,
                self.test_registry_sha256,
            }
        ) != 3:
            raise ValueError("routellm_bert_artifact_split_hash_reused")
        if not isinstance(self.dev_calibration, RouteLLMMFDevCalibration):
            raise TypeError("routellm_bert_dev_calibration_invalid")
        if (
            self.dev_calibration.dev_registry_sha256 != self.dev_registry_sha256
            or self.dev_calibration.dev_membership_sha256
            != self.dev_membership_sha256
        ):
            raise ValueError("routellm_bert_dev_calibration_mismatch")
        if (
            self.variant != ROUTELLM_BERT_VARIANT
            or self.upstream_revision != ROUTELLM_MF_UPSTREAM_REVISION
            or self.upstream_revision != UPSTREAM_REVISIONS[ROUTELLM]
            or self.base_repo != ROUTELLM_BERT_BASE_REPO
            or self.base_revision != ROUTELLM_BERT_BASE_REVISION
        ):
            raise ValueError("routellm_bert_artifact_source_mismatch")
        if (self.cheap_model, self.strong_model) != (CHEAP_MODEL, STRONG_MODEL):
            raise ValueError("routellm_bert_artifact_model_pair_mismatch")
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("routellm_bert_artifact_effort_mismatch")

    @property
    def threshold(self) -> float:
        return self.dev_calibration.threshold

    @property
    def artifact_sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": ROUTELLM_BERT_ARTIFACT_SCHEMA,
            "method": ROUTELLM,
            "variant": self.variant,
            "surface": self.surface,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "split_binding_sha256": self.split_binding_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "dev_calibration": self.dev_calibration.public_record(),
            "dev_calibration_sha256": self.dev_calibration.sha256,
            "upstream_revision": self.upstream_revision,
            "base_repo": self.base_repo,
            "base_revision": self.base_revision,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "reasoning_effort": self.reasoning_effort,
        }

    def to_router_binding(self) -> RouterArtifactBinding:
        return RouterArtifactBinding(
            method=ROUTELLM,
            surface=self.surface,
            upstream_revision=self.upstream_revision,
            artifact_sha256=self.artifact_sha256,
            fit_registry_sha256=self.fit_registry_sha256,
            dev_registry_sha256=self.dev_registry_sha256,
            test_registry_sha256=self.test_registry_sha256,
            calibration_sha256=self.dev_calibration.sha256,
        )

    @classmethod
    def from_public_record(cls, record: Any) -> RouteLLMBertArtifact:
        if not isinstance(record, Mapping):
            raise TypeError("routellm_bert_artifact_record_invalid")
        if (
            record.get("schema") != ROUTELLM_BERT_ARTIFACT_SCHEMA
            or record.get("method") != ROUTELLM
            or record.get("variant") != ROUTELLM_BERT_VARIANT
        ):
            raise ValueError("routellm_bert_artifact_schema_invalid")
        calibration_raw = record.get("dev_calibration")
        if not isinstance(calibration_raw, Mapping):
            raise ValueError("routellm_bert_calibration_record_invalid")
        candidates = calibration_raw.get("candidate_receipts")
        if not isinstance(candidates, list):
            raise ValueError("routellm_bert_candidates_invalid")
        from papc.baselines.routellm_mf_upstream import (
            RouteLLMMFDevCandidateReceipt,
        )

        calibration = RouteLLMMFDevCalibration(
            dev_registry_sha256=calibration_raw["dev_registry_sha256"],
            dev_membership_sha256=calibration_raw["dev_membership_sha256"],
            strong_full_dev_anchor=calibration_raw["strong_full_dev_anchor"],
            threshold=calibration_raw["threshold"],
            dev_row_count=calibration_raw["dev_row_count"],
            observed_strong_count=calibration_raw["observed_strong_count"],
            dev_examples_sha256=calibration_raw["dev_examples_sha256"],
            candidate_receipts=tuple(
                RouteLLMMFDevCandidateReceipt(**dict(item))
                for item in candidates
            ),
            selection_rule=calibration_raw["selection_rule"],
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
            dev_calibration=calibration,
            variant=record["variant"],
            upstream_revision=record["upstream_revision"],
            base_repo=record["base_repo"],
            base_revision=record["base_revision"],
            cheap_model=record["cheap_model"],
            strong_model=record["strong_model"],
            reasoning_effort=record["reasoning_effort"],
        )
        if artifact.public_record() != dict(record):
            raise ValueError("routellm_bert_artifact_not_canonical")
        return artifact


def bind_routellm_bert_artifact(
    *,
    surface: str,
    training_export: RouteLLMBertTrainingExport,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    dev_calibration: RouteLLMMFDevCalibration,
) -> RouteLLMBertArtifact:
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    manifest = training_export.checkpoint_manifest
    if manifest.fit_registry_sha256 != fit_manifest.registry_sha256:
        raise ValueError("routellm_bert_fit_registry_mismatch")
    fit_membership = _membership_sha256(fit_manifest)
    if manifest.fit_membership_sha256 != fit_membership:
        raise ValueError("routellm_bert_fit_membership_mismatch")
    return RouteLLMBertArtifact(
        surface=surface,
        fit_registry_sha256=fit_manifest.registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_membership_sha256=fit_membership,
        dev_membership_sha256=dev_calibration.dev_membership_sha256,
        split_binding_sha256=_digest(
            {
                "fit": fit_manifest.registry_sha256,
                "dev": dev_manifest.registry_sha256,
                "test": test_manifest.registry_sha256,
                "fit_examples": fit_manifest.example_sha256s,
                "dev_examples": dev_manifest.example_sha256s,
                "test_examples": test_manifest.example_sha256s,
            }
        ),
        checkpoint_bundle_sha256=training_export.checkpoint_bundle_sha256,
        checkpoint_manifest_sha256=manifest.sha256,
        dev_calibration=dev_calibration,
    )


class RouteLLMBertScorer:
    def __init__(self, *, artifact: RouteLLMBertArtifact, checkpoint_dir: Path):
        checkpoint = Path(checkpoint_dir)
        if _directory_sha256(checkpoint) != artifact.checkpoint_bundle_sha256:
            raise ValueError("routellm_bert_checkpoint_bundle_mismatch")
        manifest = RouteLLMBertCheckpointManifest.from_record(
            json.loads(
                (checkpoint / ROUTELLM_BERT_MANIFEST_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
        )
        if manifest.sha256 != artifact.checkpoint_manifest_sha256:
            raise ValueError("routellm_bert_checkpoint_manifest_mismatch")
        self._load_checkpoint(checkpoint)

    @classmethod
    def from_training_export(
        cls,
        export: RouteLLMBertTrainingExport,
    ) -> RouteLLMBertScorer:
        if not isinstance(export, RouteLLMBertTrainingExport):
            raise TypeError("routellm_bert_training_export_required")
        checkpoint = export.checkpoint_dir
        if _directory_sha256(checkpoint) != export.checkpoint_bundle_sha256:
            raise ValueError("routellm_bert_checkpoint_bundle_mismatch")
        instance = cls.__new__(cls)
        instance._load_checkpoint(checkpoint)
        return instance

    @classmethod
    def from_released_base_bundle(
        cls,
        checkpoint_dir: Path,
    ) -> RouteLLMBertScorer:
        """Load the frozen released checkpoint without claiming PAPC training."""

        checkpoint = Path(checkpoint_dir)
        if _directory_sha256(checkpoint) != ROUTELLM_BERT_BASE_BUNDLE_SHA256:
            raise ValueError("routellm_bert_released_base_bundle_mismatch")
        instance = cls.__new__(cls)
        instance._load_checkpoint(checkpoint)
        return instance

    def _load_checkpoint(self, checkpoint: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
        ).eval().to(self._device)

    def __call__(self, view: RequestRouterView) -> ScalarPrediction:
        if not isinstance(view, RequestRouterView):
            raise TypeError("routellm_bert_request_view_required")
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        inputs = self._tokenizer(
            view.original_request,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ROUTELLM_BERT_MAX_LENGTH,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
            strong_win_rate = self._torch.softmax(logits, dim=-1)[0].item()
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
        latency = max(0.0, time.perf_counter() - started)
        if not math.isfinite(strong_win_rate):
            raise ValueError("routellm_bert_prediction_invalid")
        return ScalarPrediction(
            value=float(strong_win_rate),
            receipts=(
                CallReceipt.local_router(
                    method=ROUTELLM_BERT_LOCAL_MODEL,
                    latency_seconds=latency,
                ),
            ),
        )


def _ordered_fit_examples(
    examples: Sequence[RouteLLMBertFitExample],
    *,
    fit_manifest: SplitManifest,
) -> tuple[RouteLLMBertFitExample, ...]:
    if fit_manifest.role != "router_fit":
        raise ValueError("routellm_bert_fit_manifest_role_invalid")
    supplied = tuple(examples)
    by_id = {item.example_sha256: item for item in supplied}
    if len(by_id) != len(supplied) or set(by_id) != set(
        fit_manifest.example_sha256s
    ):
        raise ValueError("routellm_bert_fit_membership_mismatch")
    return tuple(by_id[value] for value in fit_manifest.example_sha256s)


def _build_checkpoint_manifest(
    examples: Sequence[RouteLLMBertFitExample],
    *,
    fit_manifest: SplitManifest,
) -> RouteLLMBertCheckpointManifest:
    counts = Counter(item.label for item in examples)
    return RouteLLMBertCheckpointManifest(
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=_membership_sha256(fit_manifest),
        fit_corpus_sha256=_digest([item.digest_record() for item in examples]),
        fit_rows=len(examples),
        class_counts=(
            ("strong_wins", counts[0]),
            ("tie", counts[1]),
            ("cheap_wins", counts[2]),
        ),
    )


def _membership_sha256(manifest: SplitManifest) -> str:
    return _digest(
        {
            "role": manifest.role,
            "registry_sha256": manifest.registry_sha256,
            "example_sha256s": manifest.example_sha256s,
        }
    )


def _directory_sha256(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError("routellm_bert_checkpoint_directory_missing")
    digest = hashlib.sha256()
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    if not files:
        raise ValueError("routellm_bert_checkpoint_directory_empty")
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = file_path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _write_canonical_json(path: Path, record: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name}_invalid")


__all__ = [
    "ROUTELLM_BERT_ARTIFACT_SCHEMA",
    "ROUTELLM_BERT_BASE_REPO",
    "ROUTELLM_BERT_BASE_REVISION",
    "ROUTELLM_BERT_BASE_BUNDLE_SHA256",
    "ROUTELLM_BERT_LOCAL_MODEL",
    "ROUTELLM_BERT_MANIFEST_FILENAME",
    "RouteLLMBertArtifact",
    "RouteLLMBertCheckpointManifest",
    "RouteLLMBertFitExample",
    "RouteLLMBertScorer",
    "RouteLLMBertTrainingExport",
    "bind_routellm_bert_artifact",
    "build_routellm_bert_fit_examples",
    "train_and_export_routellm_bert",
]
